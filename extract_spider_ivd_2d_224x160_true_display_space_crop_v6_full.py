
# ========================= V5 DISPLAY SPACE CROP =========================
# 训练裁剪必须发生在 images_display 坐标空间。
#
# 原流程:
# normalized_volume[z] -> crop -> resize
#
# V5:
# images_display -> crop -> resize
#
# mask只用于定位bbox，不参与强度图裁剪。
# ========================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 SPIDER 腰椎 MRI 数据集生成“每个椎间盘一个样本”的 2D/2.5D 数据。

默认目录结构：
    D:/SPIDER/
    ├─ images/
    ├─ masks/
    ├─ overview.csv
    └─ radiological_gradings.csv

核心原则：
1. 默认只处理 T2 序列；
2. SPIDER 椎间盘标签值为 201~225，故 mask_value = 200 + IVD label；
3. 先按物理方向统一到 LPS，再转换为“头侧朝上、前方朝左”的矢状位；
4. 默认严格选择体数据的中间矢状位切片，不使用 mask 最大面积选层；
5. 默认使用物理空间custom_2to1：先用spacing换算毫米，再按1.1倍逐步
   扩宽，避免非等距像素病例在输出224×112时被拉伸；
6. 保存原始 Pfirrmann 1~5 和模型常用 class_id 0~4；
7. IVD label 是自尾侧向头侧的编号，不强制映射成 L5-S1 等解剖名称。
8. 完整中间切面同时保存原始强度MHA/NPY、原始矩阵16-bit PNG，以及按
   物理spacing恢复正常比例的显示PNG；主显示严格采用ITK-SNAP思路：
   对整个三维图像强度直方图取第1与第99百分位作为自动窗宽上下限。
9. 最终逐椎间盘2:1训练图与完整质控图共用同一个整卷1%~99%窗口：
   先整卷线性映射，再按物理spacing确定2:1框，最后resize到224×112。

安装：
    pip install SimpleITK pandas numpy Pillow

运行：
    python extract_spider_ivd_2d.py --root "D:\\SPIDER"

先检查环境和裁剪逻辑：
    python extract_spider_ivd_2d.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


MEDICAL_EXTENSIONS = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")
SCRIPT_VERSION = "SPIDER_IVD_EXTRACT_V17_TRUE_DISPLAY_SPACE_CROP"
MASK_SUFFIX_RE = re.compile(
    r"(?:[_\-.](?:mask|masks|seg|segmentation|label|labels))+$",
    flags=re.IGNORECASE,
)


@dataclass
class SeriesRecord:
    series_name: str
    patient: str
    subset: str
    image_path: str
    mask_path: str
    window_center: Optional[float] = None
    window_width: Optional[float] = None


@dataclass
class DiscRecord:
    series_name: str
    patient: str
    subset: str
    image_path: str
    mask_path: str
    ivd_label: int
    mask_value: int
    pfirrmann_grade_original: int
    class_id: int
    common_center_slice: int
    center_slice: int
    slice_selection: str
    center_row: float
    center_col: float
    bbox_width_mm: float
    bbox_height_mm: float
    mask_voxels_3d: int
    modic: Optional[int] = None
    up_endplate: Optional[int] = None
    low_endplate: Optional[int] = None
    spondylolisthesis: Optional[int] = None
    disc_herniation: Optional[int] = None
    disc_narrowing: Optional[int] = None
    disc_bulging: Optional[int] = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="提取 SPIDER 数据集中逐椎间盘的 T2 矢状位 2D/2.5D 样本。",
    )
    p.add_argument("--root", type=Path, default=Path(r"/"))
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--masks-dir", type=Path, default=None)
    p.add_argument("--overview-csv", type=Path, default=None)
    p.add_argument("--gradings-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--sequence-regex",
        default=r"(?i)_t2$",
        help="用于筛选序列名的正则表达式；默认只保留标准 T2。",
    )
    p.add_argument(
        "--output-size",
        type=int,
        default=None,
        help="旧版兼容参数；填写后会强制输出为该尺寸的正方形。",
    )
    p.add_argument("--output-width", type=int, default=224)
    p.add_argument("--output-height", type=int, default=160)
    p.add_argument(
        "--mode",
        choices=("2d", "2.5d"),
        default="2d",
        help="2d=单张灰度图；2.5d=相邻三张切片作为 RGB 三通道。",
    )
    p.add_argument(
        "--slice-policy",
        choices=("common_max", "disc_max", "middle"),
        default="middle",
        help=(
            "middle=严格使用体数据正中层（默认，不回退）；"
            "common_max=所有椎间盘总面积最大的共同层；"
            "disc_max=每个椎间盘分别选最大面积层。"
        ),
    )
    p.add_argument(
        "--crop-policy",
        choices=(
            "custom_2to1",
            "bbox_expand",
            "bbox_max",
            "bbox_ratio",
            "mask_width",
            "fixed_mm",
            "mask_center_expand",
        ),
        default="mask_center_expand",
        help=(
            "custom_2to1=横向较长时按倍率迭代扩宽，直到高度能容纳mask；"
            "bbox_expand=直接只扩展不缩小，将紧密bbox补成2:1；"
            "bbox_max=令M=max(mask宽,高)，裁剪宽2M、高M；"
            "bbox_ratio=从紧密mask bbox扩展为与输出相同纵横比的矩形；"
            "mask_width=按当前椎间盘mask横向宽度作正方形裁剪；"
            "fixed_mm=使用固定物理毫米视野。"
        ),
    )
    p.add_argument(
        "--mask-width-scale",
        type=float,
        default=1.1,
        help=(
            "custom_2to1/bbox_expand/bbox_max/bbox_ratio/mask_width"
            "的可选统一倍率；默认1.0。"
            "如设1.2，会在默认裁剪框基础上再增加20%%上下文。"
        ),
    )
    p.add_argument(
        "--custom-expand-ratio",
        type=float,
        default=1.1,
        help=(
            "custom_2to1中，当mask横向宽度大于高度、但其一半仍小于"
            "mask高度时，每轮横向扩大的倍率；默认1.1。"
        ),
    )
    p.add_argument(
        "--full-display-max-side",
        type=int,
        default=2048,
        help=(
            "完整切片物理比例PNG的最长边上限；默认2048。只影响质控PNG，"
            "不影响原始MHA/NPY和训练裁剪。"
        ),
    )
    p.add_argument(
        "--full-display-gamma",
        type=float,
        default=1.0,
        help=(
            "完整切片PNG的可选gamma；1.0保持窗宽线性显示，"
            "小于1会提亮中间灰度，例如0.85。只影响质控PNG。"
        ),
    )
    p.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="固定方形物理视野；不填写时仅用 training 子集自动估计。",
    )
    p.add_argument("--min-fov-mm", type=float, default=72.0)
    p.add_argument("--fov-percentile", type=float, default=95.0)
    p.add_argument("--width-context-factor", type=float, default=1.8)
    p.add_argument("--height-context-factor", type=float, default=5.0)
    p.add_argument("--ivd-mask-offset", type=int, default=200)
    p.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help=(
            "旧版兼容参数；V15主流程固定使用ITK-SNAP整卷1%%窗口，"
            "该参数不再控制训练图或完整质控图。"
        ),
    )
    p.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help=(
            "旧版兼容参数；V15主流程固定使用ITK-SNAP整卷99%%窗口，"
            "该参数不再控制训练图或完整质控图。"
        ),
    )
    p.add_argument("--save-npy", action="store_true")
    p.add_argument(
        "--no-save-full-slices",
        dest="save_full_slices",
        action="store_false",
        help="不保存用于质控的完整未裁剪选中切面。",
    )
    p.set_defaults(save_full_slices=True)
    p.add_argument("--qc-count", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"找不到列 {list(candidates)}。当前 CSV 列为：{list(df.columns)}"
        )
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="gbk")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def normalize_patient_id(value: object) -> str:
    text = clean_text(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def parse_int(value: object, field: str) -> int:
    text = clean_text(value)
    if not text:
        raise ValueError(f"{field} 为空")
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"{field} 不是整数：{value}")
    return int(number)


def parse_optional_int(value: object) -> Optional[int]:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
        return int(number) if number.is_integer() else None
    except ValueError:
        return None


def parse_optional_float(value: object) -> Optional[float]:
    """兼容数值、'[1093]'、'1093\\1094'等可能的医学元数据写法。"""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    if not match:
        return None
    try:
        number = float(match.group(0))
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def patient_from_series_name(series_name: str) -> str:
    # SPIDER 的标准名称是 1_t1、1_t2、107_t2 等。
    match = re.match(r"^(.+?)[_\-.]t(?:1|2)(?:[_\-.].*)?$", series_name, re.I)
    if match:
        return normalize_patient_id(match.group(1))
    return normalize_patient_id(series_name.split("_", 1)[0])


def strip_medical_extension(path_or_name: object) -> str:
    name = Path(str(path_or_name)).name
    lower = name.lower()
    for ext in sorted(MEDICAL_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def canonical_stem(path_or_name: object) -> str:
    stem = strip_medical_extension(path_or_name)
    return MASK_SUFFIX_RE.sub("", stem).strip().lower()


def safe_component(value: object) -> str:
    text = clean_text(value, "unknown")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    return text.strip("-") or "unknown"


def discover_medical_files(folder: Path) -> Dict[str, List[Path]]:
    if not folder.exists():
        raise FileNotFoundError(f"目录不存在：{folder}")
    result: Dict[str, List[Path]] = defaultdict(list)
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.lower().endswith(ext) for ext in MEDICAL_EXTENSIONS):
            result[canonical_stem(path.name)].append(path.resolve())
    return dict(result)


def resolve_unique_file(
    series_name: str,
    index: Dict[str, List[Path]],
    kind: str,
) -> Path:
    key = canonical_stem(series_name)
    candidates = index.get(key, [])
    if not candidates:
        raise FileNotFoundError(f"{kind} 中找不到序列 {series_name!r}")
    if len(candidates) > 1:
        exact = [
            p for p in candidates
            if strip_medical_extension(p.name).lower()
            == strip_medical_extension(series_name).lower()
        ]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(
            f"{kind} 中序列 {series_name!r} 匹配到多个文件："
            + ", ".join(str(p) for p in candidates)
        )
    return candidates[0]


def geometry_equal(image, mask, tolerance: float = 1e-5) -> bool:
    if image.GetSize() != mask.GetSize():
        return False
    for a, b in (
        (image.GetSpacing(), mask.GetSpacing()),
        (image.GetOrigin(), mask.GetOrigin()),
        (image.GetDirection(), mask.GetDirection()),
    ):
        if len(a) != len(b) or not np.allclose(a, b, atol=tolerance, rtol=0):
            return False
    return True


def canonicalize_lps_arrays(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SimpleITK 的 LPS 图像转为统一的矢状位数组。

    LPS 图像 GetArrayFromImage 后为 [S, P, L]，即 [z, y, x]。
    转成 [sagittal_slice(LR), row(SI), col(AP)]，再翻转 SI：
      - 第 0 轴：从右到左的矢状位切片；
      - 第 1 轴：图像上方为头侧、下方为尾侧；
      - 第 2 轴：图像左侧为前方、右侧为后方。

    这与用户旧代码 RAS + fix_view 的显示方向一致。
    """
    if image_zyx.ndim != 3 or mask_zyx.ndim != 3:
        raise ValueError(
            f"canonicalize_lps_arrays 只接受3D数组，"
            f"实际 image={image_zyx.shape}, mask={mask_zyx.shape}"
        )
    image_sag = np.transpose(image_zyx, (2, 0, 1))
    mask_sag = np.transpose(mask_zyx, (2, 0, 1))
    image_sag = np.flip(image_sag, axis=1).copy()
    mask_sag = np.flip(mask_sag, axis=1).copy()
    return image_sag, mask_sag


def get_orientation_code(image) -> str:
    if sitk is None:
        return "unknown"
    try:
        return str(
            sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                image.GetDirection()
            )
        )
    except Exception:
        return "unknown"


def load_image_and_mask(
    image_path: Path,
    mask_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[float, float, float],
    bool,
    Dict[str, object],
]:
    if sitk is None:
        raise ImportError(
            "缺少 SimpleITK。请运行：pip install SimpleITK pandas numpy Pillow"
        )
    image_itk = sitk.ReadImage(str(image_path))
    mask_itk = sitk.ReadImage(str(mask_path))
    if image_itk.GetDimension() != 3 or mask_itk.GetDimension() != 3:
        raise ValueError(
            f"只支持 3D 数据，实际 image={image_itk.GetDimension()}D, "
            f"mask={mask_itk.GetDimension()}D"
        )
    original_orientation = get_orientation_code(image_itk)
    original_size_xyz = tuple(int(x) for x in image_itk.GetSize())
    original_spacing_xyz = tuple(float(x) for x in image_itk.GetSpacing())
    resampled = False
    if not geometry_equal(image_itk, mask_itk):
        mask_itk = sitk.Resample(
            mask_itk,
            image_itk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt16,
        )
        resampled = True

    # 不能假设所有医院的 .mha 轴顺序相同。先用 direction cosine 将两者
    # 统一到 LPS；DICOMOrient 只置换/翻转体素，不做插值。
    image_lps = sitk.DICOMOrient(image_itk, "LPS")
    mask_lps = sitk.DICOMOrient(mask_itk, "LPS")
    if not geometry_equal(image_lps, mask_lps):
        # 正常情况下不会触发；作为保险，仍使用最近邻对齐标签。
        mask_lps = sitk.Resample(
            mask_lps,
            image_lps,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt16,
        )
        resampled = True

    image_zyx = sitk.GetArrayFromImage(image_lps).astype(
        np.float32, copy=False
    )
    mask_zyx = sitk.GetArrayFromImage(mask_lps).astype(np.int32, copy=False)
    image, mask = canonicalize_lps_arrays(image_zyx, mask_zyx)

    # LPS spacing=(LR, AP, SI)。统一数组为[slice_LR,row_SI,col_AP]，
    # 因而对后续2D裁剪返回(col_AP, row_SI, slice_LR)。
    spacing_lps = tuple(float(x) for x in image_lps.GetSpacing())
    spacing = (spacing_lps[1], spacing_lps[2], spacing_lps[0])
    if image.shape != mask.shape:
        raise RuntimeError(
            f"重采样后 image/mask 尺寸仍不一致：{image.shape} vs {mask.shape}"
        )
    info = {
        "original_orientation": original_orientation,
        "canonical_orientation": get_orientation_code(image_lps),
        "original_size_x": original_size_xyz[0],
        "original_size_y": original_size_xyz[1],
        "original_size_z": original_size_xyz[2],
        "original_spacing_x": original_spacing_xyz[0],
        "original_spacing_y": original_spacing_xyz[1],
        "original_spacing_z": original_spacing_xyz[2],
        "canonical_sagittal_slices": int(image.shape[0]),
        "canonical_rows": int(image.shape[1]),
        "canonical_cols": int(image.shape[2]),
    }
    return image, mask, spacing, resampled, info


def robust_window_values(
    array: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Tuple[float, float]:
    """只计算稳健窗宽上下界，不复制整套体数据。"""
    finite = array[np.isfinite(array)]
    foreground = finite[finite > 0]
    values = foreground if foreground.size >= 100 else finite
    if values.size == 0:
        raise ValueError("图像中没有有限数值")
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError(f"图像强度没有有效范围：low={low}, high={high}")
    return float(low), float(high)


def itksnap_auto_window_values(volume: np.ndarray) -> Tuple[float, float]:
    """
    复现ITK-SNAP自动窗宽的核心规则。

    ITK-SNAP官方说明：对整套图像的强度直方图取第1和第99百分位作为
    自动窗口上下限。这里包含所有有限体素（包括背景0），不使用CSV中的
    WindowCenter/WindowWidth，也不只统计当前二维切片。
    """
    finite = np.asarray(volume)[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("图像中没有有限强度值")
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        raise ValueError(
            f"ITK-SNAP自动窗无法得到有效范围：low={low}, high={high}"
        )
    return float(low), float(high)


def normalize_with_window(
    array: np.ndarray,
    low: float,
    high: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """按明确窗宽上下界映射到[0,1]，可选gamma仅用于显示。"""
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"非法显示窗口：low={low}, high={high}")
    if gamma <= 0:
        raise ValueError(f"gamma必须大于0，实际为{gamma}")
    normalized = np.clip(
        (array.astype(np.float32) - low) / (high - low), 0, 1
    )
    normalized[~np.isfinite(normalized)] = 0
    if not math.isclose(gamma, 1.0):
        normalized = np.power(normalized, gamma)
    return normalized.astype(np.float32)


def robust_normalize_volume(
    volume: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Tuple[np.ndarray, float, float]:
    low, high = robust_window_values(
        volume, lower_percentile, upper_percentile
    )
    normalized = normalize_with_window(volume, low, high, gamma=1.0)
    return normalized, low, high


def choose_common_sagittal_slice(
    mask: np.ndarray,
    ivd_mask_offset: int = 200,
) -> int:
    """选择所有 IVD 标签总面积最大的共同中矢状位层。"""
    ivd_binary = (mask > ivd_mask_offset) & (mask <= ivd_mask_offset + 25)
    area_per_slice = ivd_binary.reshape(ivd_binary.shape[0], -1).sum(axis=1)
    if int(area_per_slice.max()) == 0:
        raise ValueError(
            f"mask 中不存在 {ivd_mask_offset + 1}~{ivd_mask_offset + 25} "
            "范围的椎间盘标签"
        )
    return int(np.argmax(area_per_slice))


def choose_disc_slice(
    binary: np.ndarray,
    policy: str,
    common_slice: int,
) -> Tuple[int, str]:
    area_per_slice = binary.reshape(binary.shape[0], -1).sum(axis=1)
    valid_slices = np.flatnonzero(area_per_slice > 0)
    if valid_slices.size == 0:
        raise ValueError("椎间盘 mask 为空")

    if policy == "disc_max":
        return int(np.argmax(area_per_slice)), "disc_max"

    if policy == "middle":
        requested = int(binary.shape[0] // 2)
        if area_per_slice[requested] == 0:
            raise ValueError(
                f"严格中间矢状位 slice={requested} 上不存在该椎间盘mask；"
                "按照middle策略跳过，不回退到其他切片"
            )
        return requested, "middle_strict"

    requested = int(common_slice)
    selection = "common_max"
    if area_per_slice[requested] > 0:
        return requested, selection

    # 仅common_max兼容策略允许选择距离共同层最近的有效层。
    nearest = int(valid_slices[np.argmin(np.abs(valid_slices - requested))])
    return nearest, selection + "_nearest_valid"


def measure_disc(
    mask: np.ndarray,
    mask_value: int,
    spacing_xyz: Sequence[float],
    slice_policy: str = "common_max",
    common_slice: Optional[int] = None,
) -> Dict[str, object]:
    binary = mask == int(mask_value)
    voxel_count = int(binary.sum())
    if voxel_count == 0:
        raise ValueError(f"mask 中不存在标签值 {mask_value}")

    if common_slice is None:
        if slice_policy == "middle":
            common_slice = int(mask.shape[0] // 2)
        elif slice_policy == "disc_max":
            # disc_max不使用共同参考层。
            common_slice = -1
        else:
            common_slice = choose_common_sagittal_slice(mask)
    center_slice, slice_selection = choose_disc_slice(
        binary, slice_policy, common_slice
    )
    binary_2d = binary[center_slice]
    rows, cols = np.where(binary_2d)
    if rows.size == 0:
        raise RuntimeError(f"标签 {mask_value} 的最大面积切片为空")

    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    spacing_x, spacing_y = float(spacing_xyz[0]), float(spacing_xyz[1])
    return {
        "center_slice": center_slice,
        "slice_selection": slice_selection,
        "center_row": (row_min + row_max) / 2.0,
        "center_col": (col_min + col_max) / 2.0,
        "bbox_width_mm": (col_max - col_min + 1) * spacing_x,
        "bbox_height_mm": (row_max - row_min + 1) * spacing_y,
        "mask_voxels_3d": voxel_count,
    }


def compute_auto_fov_mm(
    records: Sequence[DiscRecord],
    percentile: float,
    minimum: float,
    width_factor: float,
    height_factor: float,
) -> Tuple[float, Dict[str, float]]:
    training = [r for r in records if r.subset.lower() == "training"]
    source = training if training else list(records)
    if not source:
        raise ValueError("没有可用于估计 FOV 的有效椎间盘")
    widths = np.asarray([r.bbox_width_mm for r in source], dtype=np.float64)
    heights = np.asarray([r.bbox_height_mm for r in source], dtype=np.float64)
    p_width = float(np.percentile(widths, percentile))
    p_height = float(np.percentile(heights, percentile))
    raw_fov = max(minimum, p_width * width_factor, p_height * height_factor)
    fov_mm = float(math.ceil(raw_fov / 2.0) * 2.0)
    details = {
        "source_count": len(source),
        "percentile": percentile,
        "width_percentile_mm": p_width,
        "height_percentile_mm": p_height,
        "width_context_factor": width_factor,
        "height_context_factor": height_factor,
        "minimum_fov_mm": minimum,
        "selected_fov_mm": fov_mm,
    }
    return fov_mm, details


def crop_fixed(
    array_2d: np.ndarray,
    center_row: float,
    center_col: float,
    crop_rows: int,
    crop_cols: int,
    fill_value: float = 0,
    shift_to_fit: bool = True,
) -> np.ndarray:
    if array_2d.ndim != 2:
        raise ValueError(f"crop_fixed 需要二维数组，实际 shape={array_2d.shape}")
    crop_rows, crop_cols = max(1, int(crop_rows)), max(1, int(crop_cols))
    row0 = int(round(center_row - (crop_rows - 1) / 2.0))
    col0 = int(round(center_col - (crop_cols - 1) / 2.0))

    # fixed_mm模式可以平移窗口以减少padding；mask_width模式必须关闭它，
    # 从而保证椎间盘始终严格位于正方形中心，上下左右各取一半。
    if shift_to_fit:
        if crop_rows <= array_2d.shape[0]:
            row0 = min(max(row0, 0), array_2d.shape[0] - crop_rows)
        if crop_cols <= array_2d.shape[1]:
            col0 = min(max(col0, 0), array_2d.shape[1] - crop_cols)
    row1, col1 = row0 + crop_rows, col0 + crop_cols

    src_row0, src_col0 = max(0, row0), max(0, col0)
    src_row1, src_col1 = min(array_2d.shape[0], row1), min(array_2d.shape[1], col1)
    output = np.full((crop_rows, crop_cols), fill_value, dtype=array_2d.dtype)
    if src_row1 > src_row0 and src_col1 > src_col0:
        dst_row0, dst_col0 = src_row0 - row0, src_col0 - col0
        dst_row1 = dst_row0 + (src_row1 - src_row0)
        dst_col1 = dst_col0 + (src_col1 - src_col0)
        output[dst_row0:dst_row1, dst_col0:dst_col1] = array_2d[
            src_row0:src_row1, src_col0:src_col1
        ]
    return output


def centered_padding_fraction(
    array_shape: Sequence[int],
    center_row: float,
    center_col: float,
    crop_rows: int,
    crop_cols: int,
    shift_to_fit: bool,
) -> float:
    """计算与 crop_fixed 完全一致的padding占比。"""
    height, width = int(array_shape[0]), int(array_shape[1])
    row0 = int(round(center_row - (crop_rows - 1) / 2.0))
    col0 = int(round(center_col - (crop_cols - 1) / 2.0))
    if shift_to_fit:
        if crop_rows <= height:
            row0 = min(max(row0, 0), height - crop_rows)
        if crop_cols <= width:
            col0 = min(max(col0, 0), width - crop_cols)
    row1, col1 = row0 + crop_rows, col0 + crop_cols
    valid_rows = max(0, min(height, row1) - max(0, row0))
    valid_cols = max(0, min(width, col1) - max(0, col0))
    valid_fraction = valid_rows * valid_cols / float(crop_rows * crop_cols)
    return float(np.clip(1.0 - valid_fraction, 0.0, 1.0))


def resize_float_image(
    array_2d: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    image = Image.fromarray(array_2d.astype(np.float32), mode="F")
    image = image.resize(
        (output_width, output_height),
        resample=Image.Resampling.BICUBIC,
    )
    return np.clip(np.asarray(image, dtype=np.float32), 0, 1)


def resize_binary_mask(
    array_2d: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    image = Image.fromarray((array_2d > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize(
        (output_width, output_height),
        resample=Image.Resampling.NEAREST,
    )
    return (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8)


def resize_label_mask(
    array_2d: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """最近邻缩放多类别标签，保留201~225等原始标签值。"""
    image = Image.fromarray(array_2d.astype(np.int32), mode="I")
    image = image.resize(
        (output_width, output_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(image, dtype=np.int32)


def physical_display_size(
    array_shape: Sequence[int],
    spacing_x: float,
    spacing_y: float,
    max_side: int = 2048,
) -> Tuple[int, int, Dict[str, float]]:
    """
    根据物理spacing计算方形显示像素尺寸。

    输入数组为[row_SI, col_AP]，所以：
      physical_width  = cols * spacing_x
      physical_height = rows * spacing_y

    先以较小spacing为显示像素间距重采样，再等比例限制最长边。返回
    (new_height, new_width, info)。这只用于PNG质控显示，不改原始MHA。
    """
    if len(array_shape) < 2:
        raise ValueError(f"array_shape至少需要二维，实际为{array_shape}")
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError(
            f"spacing必须大于0：spacing_x={spacing_x}, spacing_y={spacing_y}"
        )
    if max_side <= 0:
        raise ValueError(f"max_side必须大于0，实际为{max_side}")

    height, width = int(array_shape[0]), int(array_shape[1])
    isotropic_spacing = min(float(spacing_x), float(spacing_y))
    raw_new_width = max(
        1, int(round(width * float(spacing_x) / isotropic_spacing))
    )
    raw_new_height = max(
        1, int(round(height * float(spacing_y) / isotropic_spacing))
    )
    downscale = min(
        1.0, float(max_side) / float(max(raw_new_width, raw_new_height))
    )
    new_width = max(1, int(round(raw_new_width * downscale)))
    new_height = max(1, int(round(raw_new_height * downscale)))

    return new_height, new_width, {
        "physical_width_mm": float(width * spacing_x),
        "physical_height_mm": float(height * spacing_y),
        "native_pixel_aspect_ratio": float(width / height),
        "physical_aspect_ratio": float(
            width * spacing_x / (height * spacing_y)
        ),
        "display_height": int(new_height),
        "display_width": int(new_width),
        "display_pixel_aspect_ratio": float(new_width / new_height),
        "display_isotropic_spacing_reference": float(isotropic_spacing),
        "display_downscale": float(downscale),
    }


def square_geometry_from_mask_width(
    binary_mask_2d: np.ndarray,
    width_scale: float = 1.0,
) -> Dict[str, object]:
    """
    根据目标椎间盘 mask 的横向包围盒宽度确定正方形裁剪框。

    横向边长 = ceil(mask宽度 * width_scale)
    纵向边长 = 横向边长
    中心 = mask二维包围盒中心，因此上下、左右各取一半。
    """
    rows, cols = np.where(binary_mask_2d > 0)
    if rows.size == 0:
        raise ValueError("选定切片上的目标椎间盘 mask 为空")
    if width_scale <= 0:
        raise ValueError(f"mask width scale 必须大于0，实际为 {width_scale}")

    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    mask_width = col_max - col_min + 1
    mask_height = row_max - row_min + 1
    side = max(1, int(math.ceil(mask_width * width_scale)))

    # side不能小于原始横向mask宽度，避免浮点或错误倍率切掉mask。
    side = max(side, mask_width)
    return {
        "center_row": (row_min + row_max) / 2.0,
        "center_col": (col_min + col_max) / 2.0,
        "side_pixels": side,
        "mask_bbox_width_pixels": mask_width,
        "mask_bbox_height_pixels": mask_height,
        "mask_bbox_row_min": row_min,
        "mask_bbox_row_max": row_max,
        "mask_bbox_col_min": col_min,
        "mask_bbox_col_max": col_max,
    }


def rectangle_geometry_from_tight_bbox(
    binary_mask_2d: np.ndarray,
    width_scale: float,
    output_height: int,
    output_width: int,
    spacing_x: float,
    spacing_y: float,
) -> Dict[str, object]:
    """
    从椎间盘紧密bbox构造不拉伸的矩形裁剪框。

    目标输出默认是 W:H=224:112=2:1。裁剪框也保持相同的物理纵横比：
        (crop_cols * spacing_x) / (crop_rows * spacing_y)
        == output_width / output_height

    先确保矩形完整包含紧密bbox，再对宽高统一乘以width_scale增加上下文。
    """
    tight = square_geometry_from_mask_width(binary_mask_2d, width_scale=1.0)
    if output_height <= 0 or output_width <= 0:
        raise ValueError("输出宽高必须大于0")
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError(f"非法像素间距：spacing_x={spacing_x}, spacing_y={spacing_y}")

    output_aspect = output_width / float(output_height)
    # 使裁剪框的物理纵横比与输出图一致，从而避免空间拉伸。
    crop_pixel_aspect = output_aspect * spacing_y / spacing_x
    mask_width = int(tight["mask_bbox_width_pixels"])
    mask_height = int(tight["mask_bbox_height_pixels"])

    base_cols = max(mask_width, int(math.ceil(mask_height * crop_pixel_aspect)))
    crop_cols = max(mask_width, int(math.ceil(base_cols * width_scale)))
    crop_rows = max(mask_height, int(math.ceil(crop_cols / crop_pixel_aspect)))

    # 如果整数取整导致纵横比略偏，以完整包含mask为优先，再重新扩宽。
    if crop_rows < mask_height:
        crop_rows = mask_height
        crop_cols = max(crop_cols, int(math.ceil(crop_rows * crop_pixel_aspect)))

    return {
        **tight,
        "crop_rows": crop_rows,
        "crop_cols": crop_cols,
        "output_aspect_ratio": output_aspect,
        "crop_pixel_aspect_ratio": crop_pixel_aspect,
        "crop_physical_aspect_ratio": (
            crop_cols * spacing_x / float(crop_rows * spacing_y)
        ),
    }


def rectangle_geometry_from_bbox_max(
    binary_mask_2d: np.ndarray,
    scale: float = 1.0,
) -> Dict[str, object]:
    """
    使用用户指定的 max-bbox 规则构造 2:1 矩形：

        M = max(mask_bbox_width, mask_bbox_height)
        crop_height = ceil(M * scale)
        crop_width  = 2 * crop_height

    例如 bbox宽20、高30，scale=1.0：
        M=30 -> crop_width=60, crop_height=30
    """
    tight = square_geometry_from_mask_width(binary_mask_2d, width_scale=1.0)
    if scale < 1.0:
        raise ValueError("bbox_max的scale不能小于1.0，否则可能截断mask")
    mask_width = int(tight["mask_bbox_width_pixels"])
    mask_height = int(tight["mask_bbox_height_pixels"])
    maximum = max(mask_width, mask_height)
    crop_rows = max(1, int(math.ceil(maximum * scale)))
    crop_cols = crop_rows * 2
    return {
        **tight,
        "bbox_max_pixels": maximum,
        "crop_rows": crop_rows,
        "crop_cols": crop_cols,
        "crop_pixel_aspect_ratio": 2.0,
    }


def rectangle_geometry_expand_only(
    binary_mask_2d: np.ndarray,
    scale: float = 1.0,
) -> Dict[str, object]:
    """
    只扩展、不缩小地把紧密bbox补成2:1。

    设 length=横向宽度，width=纵向高度：
      1) length >= 2*width：
         保持length，令新width=length/2；
      2) length < 2*width：
         保持width，令新length=2*width。

    注意分界必须是 length >= 2*width，而不能只是 length > width。
    否则例如30×20若变成30×15，会切掉原mask。
    """
    tight = square_geometry_from_mask_width(binary_mask_2d, width_scale=1.0)
    if scale < 1.0:
        raise ValueError("bbox_expand的scale不能小于1.0，否则可能截断mask")

    length = int(tight["mask_bbox_width_pixels"])
    width = int(tight["mask_bbox_height_pixels"])
    if length >= 2 * width:
        crop_cols = max(length, int(math.ceil(length * scale)))
        crop_rows = max(width, int(math.ceil(crop_cols / 2.0)))
        branch = "keep_length_expand_width"
    else:
        crop_rows = max(width, int(math.ceil(width * scale)))
        crop_cols = crop_rows * 2
        branch = "keep_width_expand_length"

    return {
        **tight,
        "crop_rows": crop_rows,
        "crop_cols": crop_cols,
        "crop_pixel_aspect_ratio": crop_cols / float(crop_rows),
        "bbox_expand_branch": branch,
    }


def rectangle_geometry_custom_2to1(
    binary_mask_2d: np.ndarray,
    expand_ratio: float = 1.1,
    scale: float = 1.0,
    spacing_x: float = 1.0,
    spacing_y: float = 1.0,
) -> Dict[str, object]:
    """
    用户指定的物理空间迭代式2:1裁剪策略，只扩展、不截断mask。

    先把紧密bbox从像素换算成毫米：
      W_mask_mm = W_mask_pixels * spacing_x
      H_mask_mm = H_mask_pixels * spacing_y

    1. W_mask_mm > H_mask_mm：
       从 W_mm=W_mask_mm 开始，令 H_target_mm=W_mm/2；
       只要 H_target_mm < H_mask_mm，就令 W_mm *= expand_ratio。
    2. H_mask_mm >= W_mask_mm：
       保持 H_mm=H_mask_mm，令 W_mm=2H_mm。

    最后将目标毫米视野换算回原图像素。由于整数像素取整，实际物理比例
    会非常接近且不小于2:1；resize到224×112时不会发生明显空间拉伸。
    """
    tight = square_geometry_from_mask_width(
        binary_mask_2d, width_scale=1.0
    )
    if expand_ratio <= 1.0:
        raise ValueError(
            "custom_2to1的expand_ratio必须大于1.0，"
            "否则迭代可能无法结束"
        )
    if scale < 1.0:
        raise ValueError(
            "custom_2to1的scale不能小于1.0，否则可能截断mask"
        )
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError(
            f"custom_2to1 spacing必须大于0，实际为"
            f"{spacing_x}, {spacing_y}"
        )

    mask_width = int(tight["mask_bbox_width_pixels"])
    mask_height = int(tight["mask_bbox_height_pixels"])
    mask_width_mm = float(mask_width * spacing_x)
    mask_height_mm = float(mask_height * spacing_y)
    expansion_steps = 0

    if mask_width_mm > mask_height_mm:
        expanded_width_mm = mask_width_mm
        target_height_mm = expanded_width_mm / 2.0
        while target_height_mm < mask_height_mm:
            expanded_width_mm *= expand_ratio
            target_height_mm = expanded_width_mm / 2.0
            expansion_steps += 1
            if expansion_steps > 100:
                raise RuntimeError("custom_2to1扩展超过100轮，请检查参数")
        branch = "physical_width_iterative_expand"
    else:
        target_height_mm = mask_height_mm
        expanded_width_mm = 2.0 * target_height_mm
        branch = "physical_keep_height_double_width"

    target_height_mm *= scale
    target_width_mm = 2.0 * target_height_mm

    crop_rows = max(
        mask_height, int(math.ceil(target_height_mm / spacing_y))
    )
    # 先使用整数行数确定实际物理高度，再向上取整横向像素，确保视野
    # 物理宽度不小于物理高度的两倍。
    actual_height_mm = crop_rows * spacing_y
    crop_cols = max(
        mask_width,
        int(math.ceil((2.0 * actual_height_mm) / spacing_x)),
    )
    actual_width_mm = crop_cols * spacing_x

    return {
        **tight,
        "crop_rows": crop_rows,
        "crop_cols": crop_cols,
        "mask_bbox_width_mm": mask_width_mm,
        "mask_bbox_height_mm": mask_height_mm,
        "crop_width_mm": float(actual_width_mm),
        "crop_height_mm": float(actual_height_mm),
        "crop_pixel_aspect_ratio": crop_cols / float(crop_rows),
        "crop_physical_aspect_ratio": (
            actual_width_mm / float(actual_height_mm)
        ),
        "target_physical_aspect_ratio": 2.0,
        "custom_expand_ratio": float(expand_ratio),
        "custom_scale": float(scale),
        "custom_expansion_steps": int(expansion_steps),
        "custom_unscaled_width_mm": float(expanded_width_mm),
        "custom_unscaled_height_mm": float(
            expanded_width_mm / 2.0
            if branch == "physical_width_iterative_expand"
            else mask_height_mm
        ),
        "custom_target_width_mm": float(target_width_mm),
        "custom_target_height_mm": float(target_height_mm),
        "custom_branch": branch,
        "custom_rule": True,
        "custom_uses_physical_spacing": True,
    }



def rectangle_geometry_adaptive_ratio(
    binary_mask_2d: np.ndarray,
    scale: float,
    output_height: int,
    output_width: int,
) -> Dict[str, object]:
    """
    Adaptive crop rule:

    Output aspect ratio:
        output_width : output_height

    Rule:
        bbox_width >= bbox_height:
            keep width, calculate height

        bbox_height > bbox_width:
            keep height, calculate width

    The crop is centered on the mask bounding box center.
    """

    rows, cols = np.where(binary_mask_2d > 0)

    if rows.size == 0:
        raise ValueError("empty mask for adaptive_ratio")

    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())

    bbox_h = row_max - row_min + 1
    bbox_w = col_max - col_min + 1

    center_row = (row_min + row_max) / 2.0
    center_col = (col_min + col_max) / 2.0

    target_ratio = float(output_width) / float(output_height)

    if bbox_w >= bbox_h:
        crop_cols = int(math.ceil(bbox_w * scale))
        crop_rows = int(math.ceil(crop_cols / target_ratio))
        rule = "keep_width"
    else:
        crop_rows = int(math.ceil(bbox_h * scale))
        crop_cols = int(math.ceil(crop_rows * target_ratio))
        rule = "keep_height"

    return {
        "center_row": center_row,
        "center_col": center_col,
        "crop_rows": crop_rows,
        "crop_cols": crop_cols,
        "mask_bbox_width_pixels": bbox_w,
        "mask_bbox_height_pixels": bbox_h,
        "adaptive_rule": rule,
    }



def build_physical_display_volume_and_mask(
    normalized_volume,
    mask_volume,
    spacing_xyz,
    max_side=2048,
):
    """
    V6:
    创建真正的 display pixel space。
    图像和mask在同一物理显示坐标中。
    """
    spacing_x = float(spacing_xyz[0])
    spacing_y = float(spacing_xyz[1])

    h, w, _ = physical_display_size(
        normalized_volume[0].shape,
        spacing_x,
        spacing_y,
        max_side=max_side,
    )

    imgs = []
    masks = []

    for z in range(normalized_volume.shape[0]):
        imgs.append(resize_float_image(normalized_volume[z], h, w))
        masks.append(resize_label_mask(mask_volume[z], h, w))

    return np.stack(imgs), np.stack(masks)


def extract_one_crop(
    normalized_volume: np.ndarray,
    mask: np.ndarray,
    record: DiscRecord,
    spacing_xyz: Sequence[float],
    fov_mm: Optional[float],
    output_height: int,
    output_width: int,
    mode: str,
    crop_policy: str = "custom_2to1",
    mask_width_scale: float = 1.0,
    custom_expand_ratio: float = 1.1,
    display_volume: Optional[np.ndarray] = None,
    display_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    spacing_x, spacing_y = float(spacing_xyz[0]), float(spacing_xyz[1])
    z = int(record.center_slice)

    if display_volume is not None and display_mask is not None:
        working_volume = display_volume
        working_mask = display_mask
    else:
        working_volume = normalized_volume
        working_mask = mask

    mask_2d = (working_mask[z] == record.mask_value).astype(np.uint8)
    if crop_policy == "custom_2to1":
        mask_geometry = rectangle_geometry_custom_2to1(
            mask_2d,
            expand_ratio=custom_expand_ratio,
            scale=mask_width_scale,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )
        crop_rows = int(mask_geometry["crop_rows"])
        crop_cols = int(mask_geometry["crop_cols"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "bbox_expand":
        mask_geometry = rectangle_geometry_expand_only(
            mask_2d,
            scale=mask_width_scale,
        )
        crop_rows = int(mask_geometry["crop_rows"])
        crop_cols = int(mask_geometry["crop_cols"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "bbox_max":
        mask_geometry = rectangle_geometry_from_bbox_max(
            mask_2d,
            scale=mask_width_scale,
        )
        crop_rows = int(mask_geometry["crop_rows"])
        crop_cols = int(mask_geometry["crop_cols"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "bbox_ratio":
        mask_geometry = rectangle_geometry_from_tight_bbox(
            mask_2d,
            width_scale=mask_width_scale,
            output_height=output_height,
            output_width=output_width,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
        )
        crop_rows = int(mask_geometry["crop_rows"])
        crop_cols = int(mask_geometry["crop_cols"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "mask_center_expand":
        mask_geometry = rectangle_geometry_adaptive_ratio(
            mask_2d,
            scale=mask_width_scale,
            output_height=output_height,
            output_width=output_width,
        )
        crop_rows = int(mask_geometry["crop_rows"])
        crop_cols = int(mask_geometry["crop_cols"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "mask_width":
        mask_geometry = square_geometry_from_mask_width(
            mask_2d, width_scale=mask_width_scale
        )
        crop_rows = crop_cols = int(mask_geometry["side_pixels"])
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    elif crop_policy == "fixed_mm":
        mask_geometry = square_geometry_from_mask_width(
            mask_2d, width_scale=1.0
        )
        if fov_mm is None or fov_mm <= 0:
            raise ValueError("fixed_mm 裁剪必须提供有效 fov_mm")
        output_aspect = output_width / float(output_height)
        crop_cols = max(1, int(round(fov_mm / spacing_x)))
        crop_rows = max(
            1, int(round((fov_mm / output_aspect) / spacing_y))
        )
        center_row = float(mask_geometry["center_row"])
        center_col = float(mask_geometry["center_col"])
    else:
        raise ValueError(f"未知 crop_policy：{crop_policy}")

    mask_crop = crop_fixed(
        mask_2d,
        center_row,
        center_col,
        crop_rows,
        crop_cols,
        fill_value=0,
        shift_to_fit=(crop_policy == "fixed_mm"),
    )
    mask_out = resize_binary_mask(mask_crop, output_height, output_width)

    if mode == "2d":
        image_crop = crop_fixed(
            working_volume[z],
            center_row,
            center_col,
            crop_rows,
            crop_cols,
            fill_value=0,
            shift_to_fit=(crop_policy == "fixed_mm"),
        )
        image_out = resize_float_image(
            image_crop, output_height, output_width
        )
    else:
        channels = []
        for dz in (-1, 0, 1):
            zi = min(max(z + dz, 0), normalized_volume.shape[0] - 1)
            crop = crop_fixed(
                working_volume[zi],
                center_row,
                center_col,
                crop_rows,
                crop_cols,
                fill_value=0,
                shift_to_fit=(crop_policy == "fixed_mm"),
            )
            channels.append(
                resize_float_image(crop, output_height, output_width)
            )
        image_out = np.stack(channels, axis=-1)

    if int(mask_out.sum()) == 0:
        raise RuntimeError(
            f"输出 mask 为空，可能 FOV/坐标错误：{record.series_name}, "
            f"IVD={record.ivd_label}"
        )
    image_for_qc = image_out if image_out.ndim == 2 else image_out[..., 1]
    padding_fraction = centered_padding_fraction(
        working_volume.shape[1:],
        center_row,
        center_col,
        crop_rows,
        crop_cols,
        shift_to_fit=(crop_policy == "fixed_mm"),
    )
    return image_out, mask_out, {
        "crop_rows_original": crop_rows,
        "crop_cols_original": crop_cols,
        "crop_center_row": center_row,
        "crop_center_col": center_col,
        "crop_policy": crop_policy,
        "mask_width_scale": mask_width_scale,
        "custom_expand_ratio": custom_expand_ratio,
        "padding_fraction": padding_fraction,
        "foreground_ratio_gt_002": float(np.mean(image_for_qc > 0.02)),
        "mask_area_ratio": float(np.mean(mask_out > 0)),
        **mask_geometry,
    }


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(uint8).save(path)


def save_mask(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L").save(path)


def save_raw16_preview(
    array: np.ndarray,
    path: Path,
) -> Dict[str, object]:
    """
    保存16-bit PNG预览。

    若原始强度能够直接表示为 uint16，则只四舍五入，不重新拉伸；
    否则按该切片的最小/最大值映射到0~65535，并在metadata中记录映射。
    精确原始浮点强度始终以 NPY/MHA 为准。
    """
    finite = np.isfinite(array)
    if not np.any(finite):
        raw_min = raw_max = 0.0
        encoded = np.zeros(array.shape, dtype=np.uint16)
        encoding = "all_nonfinite_to_zero"
        scale = 1.0
        offset = 0.0
    else:
        raw_min = float(np.min(array[finite]))
        raw_max = float(np.max(array[finite]))
        clean = np.where(finite, array, 0.0).astype(np.float32, copy=False)
        if raw_min >= 0.0 and raw_max <= 65535.0:
            encoded = np.clip(np.rint(clean), 0, 65535).astype(np.uint16)
            encoding = "direct_round_uint16"
            scale = 1.0
            offset = 0.0
        elif raw_max > raw_min:
            scale = 65535.0 / (raw_max - raw_min)
            offset = raw_min
            encoded = np.clip(
                np.rint((clean - offset) * scale), 0, 65535
            ).astype(np.uint16)
            encoding = "slice_minmax_uint16"
        else:
            encoded = np.zeros(array.shape, dtype=np.uint16)
            encoding = "constant_to_zero"
            scale = 1.0
            offset = raw_min

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="I;16").save(path)
    return {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw16_encoding": encoding,
        "raw16_scale": float(scale),
        "raw16_offset": float(offset),
    }


def save_raw_2d_mha(
    array: np.ndarray,
    path: Path,
    spacing_x: float,
    spacing_y: float,
) -> None:
    """
    把当前已统一显示方向的二维数组原值写入 MHA。

    spacing_x 对应图像横轴(AP)，spacing_y 对应图像纵轴(SI)。
    不做归一化、窗宽调整、裁剪或resize。
    """
    if sitk is None:
        raise ImportError("保存原始MHA需要SimpleITK")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing((float(spacing_x), float(spacing_y)))
    sitk.WriteImage(image, str(path), True)


def colorize_full_label_mask(mask_2d: np.ndarray) -> np.ndarray:
    """把SPIDER完整标签图转成便于检查的彩色mask。"""
    mask_int = mask_2d.astype(np.int32, copy=False)
    rgb = np.zeros((*mask_int.shape, 3), dtype=np.uint8)

    vertebrae = (mask_int >= 1) & (mask_int <= 25)
    partial_vertebrae = (mask_int >= 101) & (mask_int <= 125)
    canal = mask_int == 100
    discs = (mask_int >= 201) & (mask_int <= 225)

    rgb[vertebrae] = (40, 210, 70)          # 绿色：完整椎体
    rgb[partial_vertebrae] = (255, 210, 30) # 黄色：部分椎体
    rgb[canal] = (40, 130, 255)             # 蓝色：椎管
    rgb[discs] = (255, 45, 45)              # 红色：椎间盘
    return rgb


def make_full_slice_overlay(
    display_image: np.ndarray,
    mask_2d: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    gray = np.clip(np.round(display_image * 255), 0, 255).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    colors = colorize_full_label_mask(mask_2d)
    foreground = np.any(colors > 0, axis=2)
    output = base.copy()
    output[foreground] = np.clip(
        (1.0 - alpha) * base[foreground] + alpha * colors[foreground],
        0,
        255,
    ).astype(np.uint8)
    return output


def save_full_selected_slice(
    image_slice_raw: np.ndarray,
    mask_slice: np.ndarray,
    output_dir: Path,
    subset: str,
    series_name: str,
    slice_index: int,
    spacing_x: float,
    spacing_y: float,
    max_display_side: int,
    display_low: float,
    display_high: float,
    display_window_source: str,
    display_gamma: float,
) -> Dict[str, object]:
    """
    保存完整、未裁剪、未resize的选中矢状位。

    raw_mha_images/raw_arrays保存精确原始强度，不进行归一化；
    images_raw16保存原始像素矩阵16-bit预览；
    images_display使用ITK-SNAP整卷1%~99%自动窗口，并按spacing恢复比例；
    images_display_native保留未校正像素比例的窗宽PNG用于排查；
    images_display_slice_auto保留旧版单切片自动窗作为亮度对照；
    masks/label_masks/overlays均与正常物理比例PNG完全对齐。
    """
    try:
        slice_auto_image, slice_auto_low, slice_auto_high = (
            robust_normalize_volume(
                image_slice_raw, 1.0, 99.0
            )
        )
    except ValueError:
        slice_auto_image = np.zeros_like(image_slice_raw, dtype=np.float32)
        slice_auto_low, slice_auto_high = 0.0, 0.0

    try:
        display_image = normalize_with_window(
            image_slice_raw,
            display_low,
            display_high,
            gamma=display_gamma,
        )
    except ValueError:
        display_image = slice_auto_image
        display_low, display_high = slice_auto_low, slice_auto_high
        display_window_source = "slice_auto_fallback"

    safe_subset = safe_component(subset)
    safe_series = safe_component(series_name)
    stem = f"{safe_series}_slice{slice_index:03d}"
    png_filename = f"{stem}.png"
    raw_mha_rel = (
        Path("full_slices") / "raw_mha_images" / safe_subset / f"{stem}.mha"
    )
    raw_mask_mha_rel = (
        Path("full_slices") / "raw_mha_masks" / safe_subset / f"{stem}.mha"
    )
    raw_npy_rel = (
        Path("full_slices") / "raw_arrays" / safe_subset / f"{stem}.npy"
    )
    raw16_rel = (
        Path("full_slices") / "images_raw16" / safe_subset / png_filename
    )
    display_rel = (
        Path("full_slices") / "images_display" / safe_subset / png_filename
    )
    display_native_rel = (
        Path("full_slices")
        / "images_display_native"
        / safe_subset
        / png_filename
    )
    slice_auto_rel = (
        Path("full_slices")
        / "images_display_slice_auto"
        / safe_subset
        / png_filename
    )
    mask_rel = Path("full_slices") / "masks" / safe_subset / png_filename
    overlay_rel = Path("full_slices") / "overlays" / safe_subset / png_filename
    label_rel = (
        Path("full_slices") / "label_masks" / safe_subset / png_filename
    )

    raw_float32 = np.asarray(image_slice_raw, dtype=np.float32)
    (output_dir / raw_npy_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / raw_npy_rel, raw_float32, allow_pickle=False)
    save_raw_2d_mha(
        raw_float32,
        output_dir / raw_mha_rel,
        spacing_x,
        spacing_y,
    )
    save_raw_2d_mha(
        np.asarray(mask_slice, dtype=np.uint16),
        output_dir / raw_mask_mha_rel,
        spacing_x,
        spacing_y,
    )
    raw16_info = save_raw16_preview(raw_float32, output_dir / raw16_rel)
    save_image(display_image, output_dir / display_native_rel)

    display_height, display_width, physical_display_info = (
        physical_display_size(
            image_slice_raw.shape,
            spacing_x,
            spacing_y,
            max_side=max_display_side,
        )
    )
    physical_display_image = resize_float_image(
        display_image,
        display_height,
        display_width,
    )
    physical_slice_auto_image = resize_float_image(
        slice_auto_image,
        display_height,
        display_width,
    )
    physical_display_labels = resize_label_mask(
        mask_slice,
        display_height,
        display_width,
    )
    save_image(physical_display_image, output_dir / display_rel)
    save_image(physical_slice_auto_image, output_dir / slice_auto_rel)
    save_mask(physical_display_labels > 0, output_dir / mask_rel)

    (output_dir / overlay_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        make_full_slice_overlay(
            physical_display_image, physical_display_labels
        ),
        mode="RGB",
    ).save(output_dir / overlay_rel)

    (output_dir / label_rel).parent.mkdir(parents=True, exist_ok=True)
    labels_uint8 = np.clip(
        physical_display_labels, 0, 255
    ).astype(np.uint8)
    Image.fromarray(labels_uint8, mode="L").save(output_dir / label_rel)

    return {
        "full_slice_index": int(slice_index),
        "full_slice_height": int(image_slice_raw.shape[0]),
        "full_slice_width": int(image_slice_raw.shape[1]),
        "full_slice_spacing_x": float(spacing_x),
        "full_slice_spacing_y": float(spacing_y),
        "full_slice_display_low": float(display_low),
        "full_slice_display_high": float(display_high),
        "full_slice_display_gamma": float(display_gamma),
        "full_slice_display_window_source": str(display_window_source),
        "full_slice_slice_auto_low": float(slice_auto_low),
        "full_slice_slice_auto_high": float(slice_auto_high),
        "full_slice_image_relative_path": display_rel.as_posix(),
        "full_slice_display_relative_path": display_rel.as_posix(),
        "full_slice_display_native_relative_path": (
            display_native_rel.as_posix()
        ),
        "full_slice_display_slice_auto_relative_path": (
            slice_auto_rel.as_posix()
        ),
        "full_slice_raw16_relative_path": raw16_rel.as_posix(),
        "full_slice_raw_npy_relative_path": raw_npy_rel.as_posix(),
        "full_slice_raw_mha_relative_path": raw_mha_rel.as_posix(),
        "full_slice_raw_mask_mha_relative_path": raw_mask_mha_rel.as_posix(),
        "full_slice_mask_relative_path": mask_rel.as_posix(),
        "full_slice_overlay_relative_path": overlay_rel.as_posix(),
        "full_slice_label_relative_path": label_rel.as_posix(),
        **physical_display_info,
        **raw16_info,
    }


def overlay_for_qc(image_path: Path, mask_path: Path, caption: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    rgb = np.asarray(image).copy()
    rgb[mask] = (
        0.55 * rgb[mask] + 0.45 * np.asarray([255, 40, 40])
    ).astype(np.uint8)
    tile = Image.fromarray(rgb)
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    box_height = 28
    draw.rectangle((0, 0, tile.width, box_height), fill=(0, 0, 0))
    draw.text((4, 3), caption[:58], fill=(255, 255, 255), font=font)
    return tile


def create_qc_montage(
    metadata: pd.DataFrame,
    output_dir: Path,
    count: int,
    seed: int,
    filename: str = "qc_montage.png",
) -> Optional[Path]:
    if metadata.empty or count <= 0:
        return None
    sampled = metadata.sample(n=min(count, len(metadata)), random_state=seed)
    tiles = []
    for _, row in sampled.iterrows():
        caption = (
            f"P{row['patient']} IVD{int(row['ivd_label']):02d} "
            f"G{int(row['pfirrmann_grade_original'])} {row['subset']}"
        )
        tiles.append(
            overlay_for_qc(
                output_dir / row["image_relative_path"],
                output_dir / row["mask_relative_path"],
                caption,
            )
        )
    cols = min(6, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color="white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    path = output_dir / filename
    canvas.save(path)
    return path


def add_qc_flags(metadata: pd.DataFrame) -> pd.DataFrame:
    """标记值得人工复查的样本；只标记，不自动删除。"""
    metadata = metadata.copy()
    reasons: List[str] = []
    for row in metadata.itertuples(index=False):
        row_reasons = []
        if float(row.foreground_ratio_gt_002) < 0.10:
            row_reasons.append("very_low_foreground")
        if float(row.mask_area_ratio) < 0.002:
            row_reasons.append("very_small_mask")
        if float(row.mask_area_ratio) > 0.25:
            row_reasons.append("very_large_mask")
        if float(row.padding_fraction) > 0.10:
            row_reasons.append("large_padding")
        if str(row.original_orientation).lower() == "unknown":
            row_reasons.append("unknown_orientation")
        if "nearest_valid" in str(row.slice_selection):
            row_reasons.append("common_slice_missing_disc")
        reasons.append(";".join(row_reasons))
    metadata["qc_flag"] = [bool(x) for x in reasons]
    metadata["qc_reasons"] = reasons
    return metadata


def prepare_paths(args: argparse.Namespace) -> None:
    root = args.root.expanduser()
    args.root = root
    args.images_dir = (args.images_dir or root / "images").expanduser()
    args.masks_dir = (args.masks_dir or root / "masks").expanduser()
    args.overview_csv = (args.overview_csv or root / "overview.csv").expanduser()
    args.gradings_csv = (
        args.gradings_csv or root / "radiological_gradings.csv"
    ).expanduser()
    args.output_dir = (
        args.output_dir or root / "derived_ivd_2d"
    ).expanduser()


def initialize_output_directories(args: argparse.Namespace) -> None:
    """在处理数据前创建所有顶层输出目录，便于用户立即确认功能已启用。"""
    directories = [
        args.output_dir / "images",
        args.output_dir / "masks",
    ]
    if args.save_npy:
        directories.append(args.output_dir / "arrays")
    if args.save_full_slices:
        directories.extend(
            [
                args.output_dir / "full_slices" / "raw_mha_images",
                args.output_dir / "full_slices" / "raw_mha_masks",
                args.output_dir / "full_slices" / "raw_arrays",
                args.output_dir / "full_slices" / "images_raw16",
                args.output_dir / "full_slices" / "images_display",
                args.output_dir / "full_slices" / "images_display_native",
                args.output_dir / "full_slices" / "images_display_slice_auto",
                args.output_dir / "full_slices" / "masks",
                args.output_dir / "full_slices" / "label_masks",
                args.output_dir / "full_slices" / "overlays",
            ]
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def build_series_records(
    args: argparse.Namespace,
    overview: pd.DataFrame,
    image_index: Dict[str, List[Path]],
    mask_index: Dict[str, List[Path]],
) -> Tuple[List[SeriesRecord], List[Dict[str, object]]]:
    name_col = find_column(
        overview,
        ["new_file_name", "new filename", "file_name", "filename", "series"],
    )
    subset_col = find_column(overview, ["subset", "split"], required=False)
    window_center_col = find_column(
        overview,
        ["WindowCenter", "window_center", "window center"],
        required=False,
    )
    window_width_col = find_column(
        overview,
        ["WindowWidth", "window_width", "window width"],
        required=False,
    )
    sequence_re = re.compile(args.sequence_regex)
    records: List[SeriesRecord] = []
    failures: List[Dict[str, object]] = []

    for row_index, row in overview.iterrows():
        series_name = strip_medical_extension(clean_text(row[name_col]))
        if not series_name or not sequence_re.search(series_name):
            continue
        patient = patient_from_series_name(series_name)
        subset = clean_text(row[subset_col], "unspecified") if subset_col else "unspecified"
        subset = subset.lower()
        window_center = (
            parse_optional_float(row[window_center_col])
            if window_center_col is not None
            else None
        )
        window_width = (
            parse_optional_float(row[window_width_col])
            if window_width_col is not None
            else None
        )
        if window_width is not None and window_width <= 0:
            window_width = None
        try:
            image_path = resolve_unique_file(series_name, image_index, "images")
            mask_path = resolve_unique_file(series_name, mask_index, "masks")
            records.append(
                SeriesRecord(
                    series_name=series_name,
                    patient=patient,
                    subset=subset,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    window_center=window_center,
                    window_width=window_width,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "stage": "series_matching",
                    "overview_row": int(row_index),
                    "series_name": series_name,
                    "patient": patient,
                    "error": str(exc),
                }
            )
    if not records:
        raise RuntimeError(
            f"没有匹配到任何序列。请检查 --sequence-regex={args.sequence_regex!r}、"
            f"overview.csv 和 images/masks 文件名。"
        )
    return records, failures


def build_grading_lookup(
    gradings: pd.DataFrame,
) -> Tuple[Dict[str, List[pd.Series]], Dict[str, Optional[str]]]:
    columns = {
        "patient": find_column(gradings, ["Patient", "patient_id", "subject"]),
        "ivd_label": find_column(
            gradings, ["IVD label", "ivd_label", "disc label", "disc_label"]
        ),
        "grade": find_column(
            gradings,
            ["Pfirrman grade", "Pfirrmann grade", "pfirrmann_grade", "grade"],
        ),
        "modic": find_column(gradings, ["Modic"], required=False),
        "up_endplate": find_column(
            gradings, ["UP endplate", "upper endplate"], required=False
        ),
        "low_endplate": find_column(
            gradings, ["LOW endplate", "lower endplate"], required=False
        ),
        "spondylolisthesis": find_column(
            gradings, ["Spondylolisthesis"], required=False
        ),
        "disc_herniation": find_column(
            gradings, ["Disc herniation"], required=False
        ),
        "disc_narrowing": find_column(
            gradings, ["Disc narrowing"], required=False
        ),
        "disc_bulging": find_column(
            gradings, ["Disc bulging"], required=False
        ),
    }
    lookup: Dict[str, List[pd.Series]] = defaultdict(list)
    for _, row in gradings.iterrows():
        lookup[normalize_patient_id(row[columns["patient"]])].append(row)
    return dict(lookup), columns


def collect_disc_records(
    args: argparse.Namespace,
    series_records: Sequence[SeriesRecord],
    grading_lookup: Dict[str, List[pd.Series]],
    grading_columns: Dict[str, Optional[str]],
    failures: List[Dict[str, object]],
) -> Tuple[List[DiscRecord], Dict[str, Dict[str, object]]]:
    records: List[DiscRecord] = []
    volume_info_by_series: Dict[str, Dict[str, object]] = {}
    total = len(series_records)
    for number, series in enumerate(series_records, start=1):
        print(
            f"\r[1/2 测量mask] {number}/{total} {series.series_name}",
            end="",
            flush=True,
        )
        try:
            image, mask, spacing, resampled, volume_info = load_image_and_mask(
                Path(series.image_path), Path(series.mask_path)
            )
            volume_info_by_series[series.series_name] = {
                **volume_info,
                "mask_resampled_to_image": bool(resampled),
            }
            if args.save_full_slices:
                # 在首次成功读取T2体数据时立即保存，不依赖后续等级匹配、
                # 中间层mask是否存在或椎间盘裁剪是否成功。
                full_slice_index = int(image.shape[0] // 2)
                try:
                    # ITK-SNAP官方自动对比度：对完整3D图像的强度直方图
                    # 取第1和第99百分位。不能使用不同厂家不一致的
                    # WindowCenter/WindowWidth，也不能只统计当前2D切片。
                    full_display_low, full_display_high = (
                        itksnap_auto_window_values(image)
                    )
                    full_window_source = (
                        "itksnap_volume_histogram_percentile_01_99"
                    )

                    volume_info_by_series[series.series_name].update(
                        {
                            "overview_window_center": (
                                series.window_center
                                if series.window_center is not None
                                else np.nan
                            ),
                            "overview_window_width": (
                                series.window_width
                                if series.window_width is not None
                                else np.nan
                            ),
                        }
                    )
                    full_slice_info = save_full_selected_slice(
                        image[full_slice_index],
                        mask[full_slice_index],
                        args.output_dir,
                        series.subset,
                        series.series_name,
                        full_slice_index,
                        spacing[0],
                        spacing[1],
                        args.full_display_max_side,
                        full_display_low,
                        full_display_high,
                        full_window_source,
                        args.full_display_gamma,
                    )
                    volume_info_by_series[series.series_name].update(
                        full_slice_info
                    )
                except Exception as full_exc:
                    failures.append(
                        {
                            "stage": "full_slice_saving",
                            **asdict(series),
                            "error": str(full_exc),
                        }
                    )
            if args.slice_policy == "middle":
                # 严格使用体数据正中层，不计算任何mask面积最大层。
                common_slice = int(mask.shape[0] // 2)
            elif args.slice_policy == "disc_max":
                # 每个椎间盘在measure_disc内部独立选择，本处不需要共同层。
                common_slice = -1
            else:
                common_slice = choose_common_sagittal_slice(
                    mask, args.ivd_mask_offset
                )
        except Exception as exc:
            failures.append(
                {
                    "stage": "volume_loading",
                    **asdict(series),
                    "error": str(exc),
                }
            )
            continue

        grading_rows = grading_lookup.get(series.patient, [])
        if not grading_rows:
            failures.append(
                {
                    "stage": "grading_matching",
                    **asdict(series),
                    "error": "radiological_gradings.csv 中找不到该 Patient",
                }
            )
            continue

        for grading_row in grading_rows:
            try:
                ivd_label = parse_int(
                    grading_row[grading_columns["ivd_label"]], "IVD label"
                )
                grade = parse_int(
                    grading_row[grading_columns["grade"]], "Pfirrmann grade"
                )
                if grade not in (1, 2, 3, 4, 5):
                    raise ValueError(f"Pfirrmann grade 应为 1~5，实际为 {grade}")
                mask_value = args.ivd_mask_offset + ivd_label
                measured = measure_disc(
                    mask,
                    mask_value,
                    spacing,
                    slice_policy=args.slice_policy,
                    common_slice=common_slice,
                )

                extras = {}
                for field in (
                    "modic",
                    "up_endplate",
                    "low_endplate",
                    "spondylolisthesis",
                    "disc_herniation",
                    "disc_narrowing",
                    "disc_bulging",
                ):
                    column = grading_columns[field]
                    extras[field] = (
                        parse_optional_int(grading_row[column])
                        if column is not None
                        else None
                    )

                records.append(
                    DiscRecord(
                        series_name=series.series_name,
                        patient=series.patient,
                        subset=series.subset,
                        image_path=series.image_path,
                        mask_path=series.mask_path,
                        ivd_label=ivd_label,
                        mask_value=mask_value,
                        pfirrmann_grade_original=grade,
                        class_id=grade - 1,
                        common_center_slice=int(common_slice),
                        center_slice=int(measured["center_slice"]),
                        slice_selection=str(measured["slice_selection"]),
                        center_row=float(measured["center_row"]),
                        center_col=float(measured["center_col"]),
                        bbox_width_mm=float(measured["bbox_width_mm"]),
                        bbox_height_mm=float(measured["bbox_height_mm"]),
                        mask_voxels_3d=int(measured["mask_voxels_3d"]),
                        **extras,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "stage": "disc_measurement",
                        **asdict(series),
                        "ivd_label": clean_text(
                            grading_row[grading_columns["ivd_label"]]
                        ),
                        "grade": clean_text(grading_row[grading_columns["grade"]]),
                        "error": str(exc),
                    }
                )
    print()
    if not records:
        raise RuntimeError(
            "没有得到任何有效椎间盘。请重点检查 mask 标签是否为 201~225，"
            "以及 radiological_gradings.csv 的 Patient/IVD label。"
        )
    return records, volume_info_by_series


def extract_and_save(
    args: argparse.Namespace,
    disc_records: Sequence[DiscRecord],
    fov_mm: Optional[float],
    volume_info_by_series: Dict[str, Dict[str, object]],
    failures: List[Dict[str, object]],
) -> pd.DataFrame:
    by_series: Dict[str, List[DiscRecord]] = defaultdict(list)
    for record in disc_records:
        by_series[record.series_name].append(record)

    output_rows: List[Dict[str, object]] = []
    series_items = sorted(by_series.items())
    for number, (series_name, records) in enumerate(series_items, start=1):
        print(
            f"\r[2/2 导出切片] {number}/{len(series_items)} {series_name}",
            end="",
            flush=True,
        )
        first = records[0]
        try:
            image, mask, spacing, resampled_now, volume_info_now = load_image_and_mask(
                Path(first.image_path), Path(first.mask_path)
            )
            # 与完整质控切片完全一致：先对整个3D体数据计算ITK-SNAP
            # 自动窗（1%~99%），再线性映射整卷，之后才做每个椎间盘
            # 的物理空间2:1裁剪。不能在裁剪后单独重新归一化。
            intensity_low, intensity_high = itksnap_auto_window_values(image)
            normalized = normalize_with_window(
                image,
                intensity_low,
                intensity_high,
                gamma=1.0,
            )

            display_volume, display_mask = build_physical_display_volume_and_mask(
                normalized,
                mask,
                spacing,
                max_side=args.full_display_max_side,
            )
        except Exception as exc:
            for record in records:
                failures.append(
                    {
                        "stage": "extraction_volume_loading",
                        **asdict(record),
                        "error": str(exc),
                    }
                )
            continue

        for record in sorted(records, key=lambda x: x.ivd_label):
            try:
                image_out, mask_out, crop_info = extract_one_crop(
                    normalized,
                    mask,
                    record,
                    spacing,
                    fov_mm,
                    args.output_height,
                    args.output_width,
                    args.mode,
                    crop_policy=args.crop_policy,
                    mask_width_scale=args.mask_width_scale,
                    custom_expand_ratio=args.custom_expand_ratio,
                    display_volume=display_volume,
                    display_mask=display_mask,
                )
                subset = safe_component(record.subset.lower())
                filename = (
                    f"p{safe_component(record.patient)}_"
                    f"ivd{record.ivd_label:02d}_t2_"
                    f"g{record.pfirrmann_grade_original}.png"
                )
                image_rel = Path("images") / subset / filename
                mask_rel = Path("masks") / subset / filename
                save_image(image_out, args.output_dir / image_rel)
                save_mask(mask_out, args.output_dir / mask_rel)

                npy_rel: Optional[Path] = None
                if args.save_npy:
                    npy_rel = Path("arrays") / subset / filename.replace(".png", ".npy")
                    (args.output_dir / npy_rel).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    np.save(args.output_dir / npy_rel, image_out.astype(np.float32))

                row = {
                    **asdict(record),
                    "spacing_x_mm": spacing[0],
                    "spacing_y_mm": spacing[1],
                    "spacing_slice_mm": spacing[2],
                    "fixed_fov_mm": fov_mm if fov_mm is not None else np.nan,
                    "output_height": args.output_height,
                    "output_width": args.output_width,
                    "output_size": f"{args.output_width}x{args.output_height}",
                    "mode": args.mode,
                    "intensity_lower_value": intensity_low,
                    "intensity_upper_value": intensity_high,
                    "intensity_window_source": (
                        "itksnap_volume_histogram_percentile_01_99"
                    ),
                    "intensity_gamma": 1.0,
                    "intensity_normalized_before_crop": True,
                    "image_relative_path": image_rel.as_posix(),
                    "mask_relative_path": mask_rel.as_posix(),
                    "npy_relative_path": npy_rel.as_posix() if npy_rel else "",
                    "mask_resampled_to_image": bool(
                        volume_info_by_series.get(series_name, {}).get(
                            "mask_resampled_to_image", False
                        )
                        or resampled_now
                    ),
                    **volume_info_now,
                    **volume_info_by_series.get(series_name, {}),
                    **crop_info,
                }
                output_rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "stage": "disc_extraction",
                        **asdict(record),
                        "error": str(exc),
                    }
                )
    print()
    return pd.DataFrame(output_rows)


def run_self_test() -> None:
    assert parse_optional_float("[1093]") == 1093.0
    assert parse_optional_float("1901\\1902") == 1901.0
    histogram_test = np.arange(100, dtype=np.float32).reshape(2, 5, 10)
    itk_low, itk_high = itksnap_auto_window_values(histogram_test)
    expected_low, expected_high = np.percentile(
        histogram_test, [1.0, 99.0]
    )
    assert math.isclose(itk_low, float(expected_low))
    assert math.isclose(itk_high, float(expected_high))
    window_test = normalize_with_window(
        np.asarray([0.0, 50.0, 100.0], dtype=np.float32),
        low=0.0,
        high=100.0,
        gamma=1.0,
    )
    assert np.allclose(window_test, np.asarray([0.0, 0.5, 1.0]))
    gamma_test = normalize_with_window(
        np.asarray([0.0, 25.0, 100.0], dtype=np.float32),
        low=0.0,
        high=100.0,
        gamma=0.5,
    )
    assert np.allclose(gamma_test, np.asarray([0.0, 0.5, 1.0]))

    image = np.zeros((5, 80, 100), dtype=np.float32)
    image[:, 10:70, 10:90] = np.linspace(0, 1, 60, dtype=np.float32)[None, :, None]
    mask = np.zeros_like(image, dtype=np.int32)
    mask[1, 37:43, 35:65] = 201
    mask[2, 34:46, 30:70] = 201
    mask[3, 36:44, 33:67] = 201
    common_slice = choose_common_sagittal_slice(mask, 200)
    measured = measure_disc(
        mask,
        201,
        (0.5, 0.75, 4.0),
        slice_policy="common_max",
        common_slice=common_slice,
    )
    assert common_slice == 2
    assert measured["center_slice"] == 2
    assert measured["bbox_width_mm"] == 20.0
    assert measured["bbox_height_mm"] == 9.0

    middle_measured = measure_disc(
        mask,
        201,
        (0.5, 0.75, 4.0),
        slice_policy="middle",
        common_slice=mask.shape[0] // 2,
    )
    assert middle_measured["center_slice"] == mask.shape[0] // 2
    assert middle_measured["slice_selection"] == "middle_strict"

    strict_mask = np.zeros((5, 20, 20), dtype=np.int32)
    strict_mask[1, 5:10, 5:15] = 201
    try:
        measure_disc(
            strict_mask,
            201,
            (1.0, 1.0, 1.0),
            slice_policy="middle",
            common_slice=2,
        )
    except ValueError as exc:
        assert "不回退" in str(exc)
    else:
        raise AssertionError("middle策略不应回退到非中间切片")

    record = DiscRecord(
        series_name="1_t2",
        patient="1",
        subset="training",
        image_path="dummy",
        mask_path="dummy",
        ivd_label=1,
        mask_value=201,
        pfirrmann_grade_original=3,
        class_id=2,
        common_center_slice=common_slice,
        center_slice=int(measured["center_slice"]),
        slice_selection=str(measured["slice_selection"]),
        center_row=float(measured["center_row"]),
        center_col=float(measured["center_col"]),
        bbox_width_mm=float(measured["bbox_width_mm"]),
        bbox_height_mm=float(measured["bbox_height_mm"]),
        mask_voxels_3d=int(measured["mask_voxels_3d"]),
    )
    image_out, mask_out, info = extract_one_crop(
        image,
        mask,
        record,
        (0.5, 0.75, 4.0),
        None,
        112,
        224,
        "2d",
        crop_policy="bbox_expand",
        mask_width_scale=1.0,
    )
    assert image_out.shape == (112, 224)
    assert mask_out.shape == (112, 224)
    assert mask_out.sum() > 0
    assert info["mask_bbox_width_pixels"] == 40
    assert info["mask_bbox_height_pixels"] == 12
    assert info["crop_cols_original"] == 40
    assert info["crop_rows_original"] == 20
    assert info["crop_center_row"] == 39.5
    assert info["crop_center_col"] == 49.5
    assert info["bbox_expand_branch"] == "keep_length_expand_width"
    assert info["crop_pixel_aspect_ratio"] == 2.0

    # 端到端训练流程：必须先用完整3D体数据p01~p99线性映射，
    # 再进行物理空间custom_2to1裁剪，不能对裁剪结果单独重算窗宽。
    raw_training_volume = image * 1000.0 + 25.0
    training_low, training_high = itksnap_auto_window_values(
        raw_training_volume
    )
    training_normalized = normalize_with_window(
        raw_training_volume,
        training_low,
        training_high,
        gamma=1.0,
    )
    training_image_out, training_mask_out, training_info = extract_one_crop(
        training_normalized,
        mask,
        record,
        (0.5, 0.75, 4.0),
        None,
        112,
        224,
        "2d",
        crop_policy="custom_2to1",
        mask_width_scale=1.0,
        custom_expand_ratio=1.1,
    )
    assert training_image_out.shape == (112, 224)
    assert training_mask_out.shape == (112, 224)
    assert training_mask_out.sum() > 0
    assert training_info["custom_uses_physical_spacing"] is True
    assert np.isfinite(training_image_out).all()
    assert 0.0 <= float(training_image_out.min())
    assert float(training_image_out.max()) <= 1.0

    # 用户指定规则：若mask横向宽为40，裁剪必须是40×40；mask中心上下各20。
    geometry = square_geometry_from_mask_width(mask[2] == 201, 1.0)
    assert geometry["side_pixels"] == 40
    assert geometry["center_row"] == 39.5
    assert geometry["center_col"] == 49.5

    rectangle = rectangle_geometry_from_tight_bbox(
        mask[2] == 201,
        width_scale=1.2,
        output_height=112,
        output_width=224,
        spacing_x=0.5,
        spacing_y=0.75,
    )
    assert rectangle["crop_cols"] == 48
    assert rectangle["crop_rows"] == 16
    assert rectangle["crop_cols"] >= rectangle["mask_bbox_width_pixels"]
    assert rectangle["crop_rows"] >= rectangle["mask_bbox_height_pixels"]

    equal_spacing_rectangle = rectangle_geometry_from_tight_bbox(
        mask[2] == 201,
        width_scale=1.2,
        output_height=112,
        output_width=224,
        spacing_x=0.5,
        spacing_y=0.5,
    )
    assert equal_spacing_rectangle["crop_cols"] == 48
    assert equal_spacing_rectangle["crop_rows"] == 24

    # 用户明确示例：tight bbox宽20、高30 -> M=30 -> 裁剪宽60、高30。
    example_mask = np.zeros((80, 100), dtype=np.uint8)
    example_mask[20:50, 40:60] = 1
    bbox_max_rectangle = rectangle_geometry_from_bbox_max(
        example_mask, scale=1.0
    )
    assert bbox_max_rectangle["mask_bbox_width_pixels"] == 20
    assert bbox_max_rectangle["mask_bbox_height_pixels"] == 30
    assert bbox_max_rectangle["bbox_max_pixels"] == 30
    assert bbox_max_rectangle["crop_cols"] == 60
    assert bbox_max_rectangle["crop_rows"] == 30

    # 新默认安全分支测试：
    # 长60、宽20 -> 保持长60，宽补为30。
    long_mask = np.zeros((80, 100), dtype=np.uint8)
    long_mask[20:40, 20:80] = 1
    long_geometry = rectangle_geometry_expand_only(long_mask, scale=1.0)
    assert long_geometry["crop_cols"] == 60
    assert long_geometry["crop_rows"] == 30
    assert long_geometry["bbox_expand_branch"] == "keep_length_expand_width"

    # 长30、宽20虽有“长>宽”，但30<2×20；若缩成30×15会裁断。
    # 正确做法是保持宽20，把长补成40。
    middle_mask = np.zeros((80, 100), dtype=np.uint8)
    middle_mask[20:40, 20:50] = 1
    middle_geometry = rectangle_geometry_expand_only(middle_mask, scale=1.0)
    assert middle_geometry["crop_cols"] == 40
    assert middle_geometry["crop_rows"] == 20
    assert middle_geometry["bbox_expand_branch"] == "keep_width_expand_length"

    # 长20、宽30 -> 保持宽30，把长补成60。
    tall_geometry = rectangle_geometry_expand_only(
        example_mask, scale=1.0
    )
    assert tall_geometry["crop_cols"] == 60
    assert tall_geometry["crop_rows"] == 30
    assert tall_geometry["bbox_expand_branch"] == "keep_width_expand_length"

    # custom_2to1：横向明显更长时无需迭代，60×20 -> 60×30。
    custom_long = rectangle_geometry_custom_2to1(
        long_mask, expand_ratio=1.1, scale=1.0
    )
    assert custom_long["crop_cols"] == 60
    assert custom_long["crop_rows"] == 30
    assert custom_long["custom_expansion_steps"] == 0
    assert (
        custom_long["custom_branch"]
        == "physical_width_iterative_expand"
    )

    # custom_2to1：30×20需要30->33->36.3->39.93->43.923，
    # 离散化后严格得到44×22，而不是分别ceil后可能破坏2:1。
    custom_middle = rectangle_geometry_custom_2to1(
        middle_mask, expand_ratio=1.1, scale=1.0
    )
    assert custom_middle["crop_cols"] == 44
    assert custom_middle["crop_rows"] == 22
    assert custom_middle["custom_expansion_steps"] == 4
    assert custom_middle["crop_cols"] == 2 * custom_middle["crop_rows"]

    # custom_2to1：20×30直接保持高度30、横向补为60。
    custom_tall = rectangle_geometry_custom_2to1(
        example_mask, expand_ratio=1.1, scale=1.0
    )
    assert custom_tall["crop_cols"] == 60
    assert custom_tall["crop_rows"] == 30
    assert (
        custom_tall["custom_branch"]
        == "physical_keep_height_double_width"
    )

    # 非等距像素：原始bbox为20列×30行，但spacing=(1.0, 0.25)；
    # 物理尺寸为20×7.5 mm，应得到20列×40行，即20×10 mm的2:1视野。
    physical_custom = rectangle_geometry_custom_2to1(
        example_mask,
        expand_ratio=1.1,
        scale=1.0,
        spacing_x=1.0,
        spacing_y=0.25,
    )
    assert physical_custom["crop_cols"] == 20
    assert physical_custom["crop_rows"] == 40
    assert math.isclose(
        physical_custom["crop_physical_aspect_ratio"],
        2.0,
        rel_tol=0,
        abs_tol=1e-8,
    )
    assert physical_custom["custom_uses_physical_spacing"] is True

    # 211/186/58类完整切面：374×2048，spacing=(1.0,0.25)
    # 恢复方形显示像素后应为1496×2048，而不是狭长的374×2048。
    ph, pw, physical_info = physical_display_size(
        (2048, 374), 1.0, 0.25, max_side=2048
    )
    assert (ph, pw) == (2048, 1496)
    assert math.isclose(
        physical_info["display_pixel_aspect_ratio"],
        physical_info["physical_aspect_ratio"],
        rel_tol=0,
        abs_tol=1e-8,
    )

    full_colors = colorize_full_label_mask(mask[2])
    assert full_colors.shape == (80, 100, 3)
    assert np.array_equal(full_colors[35, 35], np.array([255, 45, 45]))
    full_overlay = make_full_slice_overlay(image[2], mask[2])
    assert full_overlay.shape == (80, 100, 3)
    assert full_overlay.dtype == np.uint8

    # 保留fixed_mm兼容模式。
    _, _, fixed_info = extract_one_crop(
        image,
        mask,
        record,
        (0.5, 0.75, 4.0),
        60.0,
        112,
        224,
        "2d",
        crop_policy="fixed_mm",
    )
    assert fixed_info["crop_cols_original"] == 120
    assert fixed_info["crop_rows_original"] == 40

    padded = crop_fixed(np.ones((5, 5), dtype=np.uint8), 0, 0, 7, 7)
    assert padded.shape == (7, 7)
    assert 0 < padded.sum() <= 25

    # LPS [z,y,x] -> [slice_x,row_z_flipped,col_y]
    lps_image = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    lps_mask = lps_image.copy()
    sag_image, sag_mask = canonicalize_lps_arrays(lps_image, lps_mask)
    assert sag_image.shape == (4, 2, 3)
    assert np.array_equal(sag_image, sag_mask)
    assert sag_image[0, 0, 0] == lps_image[1, 0, 0]
    assert sag_image[0, 1, 0] == lps_image[0, 0, 0]

    # crop小于原图时应平移到图内，不制造可避免的padding。
    shifted = crop_fixed(np.ones((10, 10), dtype=np.uint8), 0, 0, 6, 6)
    assert shifted.shape == (6, 6)
    assert shifted.sum() == 36
    centered = crop_fixed(
        np.ones((10, 10), dtype=np.uint8),
        0,
        0,
        6,
        6,
        shift_to_fit=False,
    )
    assert centered.shape == (6, 6)
    assert centered.sum() == 16
    assert centered_padding_fraction((10, 10), 0, 0, 6, 6, False) > 0
    print(
        "✅ self-test 通过：LPS方向统一、mask定位、"
        "overview窗宽解析、ITK-SNAP整卷1%~99%窗口与gamma显示、"
        "整卷窗宽后再做物理2:1训练裁剪、"
        "严格中间切片不回退、adaptive_ratio自适应比例裁剪、"
        "非等距像素正常比例显示、"
        "只扩展不缩小的2:1居中矩形裁剪、"
        "完整切面mask/overlay、固定毫米兼容模式、"
        "边界平移/padding和224×112缩放均正常"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.output_height <= 0 or args.output_width <= 0:
        raise ValueError("--output-height 和 --output-width 必须大于 0")
    if args.fov_mm is not None and args.fov_mm <= 0:
        raise ValueError("--fov-mm 必须大于 0")
    if args.mask_width_scale < 1.0:
        raise ValueError(
            "--mask-width-scale 不能小于1.0，否则可能截断椎间盘mask"
        )
    if args.custom_expand_ratio <= 1.0:
        raise ValueError(
            "--custom-expand-ratio 必须大于1.0，否则custom_2to1迭代"
            "可能无法结束"
        )
    if args.full_display_max_side <= 0:
        raise ValueError("--full-display-max-side 必须大于0")
    if args.full_display_gamma <= 0:
        raise ValueError("--full-display-gamma 必须大于0")
    if (
        args.crop_policy in ("custom_2to1", "bbox_expand", "bbox_max")
        and args.output_width != 2 * args.output_height
    ):
        raise ValueError(
            "custom_2to1/bbox_expand/bbox_max固定生成2:1裁剪框；"
            "为避免拉伸，必须满足"
            "--output-width = 2 × --output-height"
        )
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        raise ValueError("强度百分位必须满足 0 <= lower < upper <= 100")
    if not 0 < args.fov_percentile <= 100:
        raise ValueError("--fov-percentile 必须在 (0, 100] 内")



def get_display_mask_bbox(mask_display):
    rows, cols = np.where(mask_display > 0)
    if rows.size == 0:
        raise ValueError("empty display mask")

    r1, r2 = int(rows.min()), int(rows.max())
    c1, c2 = int(cols.min()), int(cols.max())

    return {
        "center_row": (r1+r2)/2.0,
        "center_col": (c1+c2)/2.0,
        "height": r2-r1+1,
        "width": c2-c1+1,
    }


def build_display_mask_center_crop(mask_display,
                                   output_width=224,
                                   output_height=160,
                                   scale=1.1):

    bbox = get_display_mask_bbox(mask_display)

    bw = bbox["width"]
    bh = bbox["height"]

    ratio = output_width / output_height

    if bw >= bh:
        crop_w = bw * scale
        crop_h = crop_w / ratio
        rule = "keep_width"
    else:
        crop_h = bh * scale
        crop_w = crop_h * ratio
        rule = "keep_height"

    return {
        "center_row": bbox["center_row"],
        "center_col": bbox["center_col"],
        "crop_height": int(round(crop_h)),
        "crop_width": int(round(crop_w)),
        "rule": rule,
    }


def crop_from_display_image(display_image, crop_info,
                            output_width=224,
                            output_height=160):

    h, w = display_image.shape[:2]

    cy = crop_info["center_row"]
    cx = crop_info["center_col"]

    ch = crop_info["crop_height"]
    cw = crop_info["crop_width"]

    y1 = int(round(cy-ch/2))
    y2 = int(round(cy+ch/2))
    x1 = int(round(cx-cw/2))
    x2 = int(round(cx+cw/2))

    pad_top = max(0, -y1)
    pad_left = max(0, -x1)
    pad_bottom = max(0, y2-h)
    pad_right = max(0, x2-w)

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        display_image = np.pad(
            display_image,
            ((pad_top,pad_bottom),
             (pad_left,pad_right)),
            mode="constant"
        )
        y1 += pad_top
        y2 += pad_top
        x1 += pad_left
        x2 += pad_left

    crop = display_image[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (output_width, output_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    return crop


def main() -> int:
    args = build_parser().parse_args()
    if args.output_size is not None:
        if args.output_size <= 0:
            raise ValueError("--output-size 必须大于 0")
        args.output_height = args.output_size
        args.output_width = args.output_size
    if args.self_test:
        run_self_test()
        return 0
    prepare_paths(args)
    validate_args(args)
    if sitk is None:
        raise ImportError(
            "缺少 SimpleITK。请先运行：\n"
            "pip install SimpleITK pandas numpy Pillow"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialize_output_directories(args)
    metadata_path = args.output_dir / "metadata.csv"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出已存在：{metadata_path}\n"
            "如确认需要覆盖同名导出文件，请加 --overwrite；旧目录中的无关文件不会删除。"
        )

    print("=" * 72)
    print(f"VERSION    : {SCRIPT_VERSION}")
    print("SPIDER 逐椎间盘 T2 2D/2.5D 提取")
    print(f"root       : {args.root}")
    print(f"images     : {args.images_dir}")
    print(f"masks      : {args.masks_dir}")
    print(f"overview   : {args.overview_csv}")
    print(f"gradings   : {args.gradings_csv}")
    print(f"output     : {args.output_dir}")
    print(f"mode       : {args.mode}")
    print(f"size       : {args.output_width}x{args.output_height} (W x H)")
    print(f"slice      : {args.slice_policy}")
    print(f"crop       : {args.crop_policy}")
    print(
        "full PNG   : physical ratio, "
        "ITK-SNAP volume histogram p01-p99 window, "
        f"gamma={args.full_display_gamma:g}, "
        f"max side={args.full_display_max_side}"
    )
    print(
        "train PNG  : full-volume p01-p99 window -> "
        "physical 2:1 crop -> resize"
    )
    print(f"full slices: {'yes' if args.save_full_slices else 'no'}")
    print("=" * 72)

    overview = read_table(args.overview_csv)
    gradings = read_table(args.gradings_csv)
    print("正在扫描 images 和 masks ...")
    image_index = discover_medical_files(args.images_dir)
    mask_index = discover_medical_files(args.masks_dir)
    print(
        f"发现 image 序列 {sum(map(len, image_index.values()))} 个，"
        f"mask 序列 {sum(map(len, mask_index.values()))} 个"
    )

    failures: List[Dict[str, object]] = []
    series_records, matching_failures = build_series_records(
        args, overview, image_index, mask_index
    )
    failures.extend(matching_failures)
    grading_lookup, grading_columns = build_grading_lookup(gradings)
    print(f"匹配到 T2 序列 {len(series_records)} 个")

    disc_records, volume_info_by_series = collect_disc_records(
        args,
        series_records,
        grading_lookup,
        grading_columns,
        failures,
    )
    print(f"找到有效椎间盘 {len(disc_records)} 个")

    if args.crop_policy in (
        "custom_2to1",
        "bbox_expand",
        "bbox_max",
        "bbox_ratio",
        "mask_width",
    ):
        fov_mm = None
        fov_details = {
            "source": "not_used",
            "crop_policy": args.crop_policy,
            "mask_width_scale": args.mask_width_scale,
            "custom_expand_ratio": args.custom_expand_ratio,
        }
        if args.crop_policy == "custom_2to1":
            print(
                "使用物理空间custom_2to1迭代裁剪："
                "先按spacing把mask宽高换算为毫米；横向较长时逐轮乘"
                f"{args.custom_expand_ratio:g}，直到其一半能容纳mask高度；"
                "mask纵向不短于横向时，横向直接补为纵向两倍；"
                f"统一上下文倍率={args.mask_width_scale:g}"
            )
        elif args.crop_policy == "bbox_expand":
            print(
                "使用bbox只扩展规则："
                "长≥2×宽时保持长、宽补到长的一半；"
                "否则保持宽、长补到宽的两倍"
            )
        elif args.crop_policy == "bbox_max":
            print(
                "使用bbox最大边规则："
                "M=max(mask bbox宽,高)，"
                f"裁剪宽=2M、高=M，倍率={args.mask_width_scale:g}"
            )
        elif args.crop_policy == "bbox_ratio":
            print(
                "使用紧密mask bbox比例裁剪："
                f"输出={args.output_width}:{args.output_height}，"
                f"上下文倍率={args.mask_width_scale:g}，"
                "裁剪框物理纵横比与输出一致"
            )
        else:
            print(
                "使用mask横向宽度正方形裁剪："
                f"边长 = mask宽度 × {args.mask_width_scale:g}，"
                "上下左右以mask包围盒中心对称"
            )
    elif args.fov_mm is None:
        fov_mm, fov_details = compute_auto_fov_mm(
            disc_records,
            args.fov_percentile,
            args.min_fov_mm,
            args.width_context_factor,
            args.height_context_factor,
        )
        print(
            "自动固定 FOV："
            f"{fov_mm:.1f} mm "
            f"(P{args.fov_percentile:g} 宽={fov_details['width_percentile_mm']:.1f} mm, "
            f"高={fov_details['height_percentile_mm']:.1f} mm)"
        )
    else:
        fov_mm = float(args.fov_mm)
        fov_details = {
            "source": "command_line",
            "selected_fov_mm": fov_mm,
        }
        print(f"使用指定固定 FOV：{fov_mm:.1f} mm")

    metadata = extract_and_save(
        args, disc_records, fov_mm, volume_info_by_series, failures
    )
    if metadata.empty:
        raise RuntimeError("所有样本导出均失败，请查看 failures.csv")

    metadata = add_qc_flags(metadata)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(
        args.output_dir / "failures.csv", index=False, encoding="utf-8-sig"
    )

    grade_counts = (
        metadata.groupby(["subset", "pfirrmann_grade_original"])
        .size()
        .rename("count")
        .reset_index()
    )
    grade_counts.to_csv(
        args.output_dir / "grade_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qc_path = create_qc_montage(
        metadata, args.output_dir, args.qc_count, args.seed
    )
    suspicious = metadata[metadata["qc_flag"]].copy()
    suspicious.to_csv(
        args.output_dir / "qc_suspicious.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suspicious_qc_path = create_qc_montage(
        suspicious,
        args.output_dir,
        min(args.qc_count, 60),
        args.seed,
        filename="qc_suspicious_montage.png",
    )

    summary = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "root": str(args.root),
        "sequence_regex": args.sequence_regex,
        "mode": args.mode,
        "slice_policy": args.slice_policy,
        "crop_policy": args.crop_policy,
        "mask_width_scale": args.mask_width_scale,
        "custom_expand_ratio": args.custom_expand_ratio,
        "full_display_max_side": args.full_display_max_side,
        "full_display_gamma": args.full_display_gamma,
        "full_display_window": (
            "itksnap_volume_histogram_percentile_01_99"
        ),
        "training_image_window": (
            "itksnap_volume_histogram_percentile_01_99"
        ),
        "training_image_pipeline": (
            "full_volume_window_then_physical_2to1_crop_then_resize"
        ),
        "output_height": args.output_height,
        "output_width": args.output_width,
        "output_size": f"{args.output_width}x{args.output_height}",
        "save_full_slices": bool(args.save_full_slices),
        "fixed_fov_mm": fov_mm,
        "fov_estimation": fov_details,
        "normalization_percentiles": [1.0, 99.0],
        "legacy_cli_percentiles_ignored": [
            args.lower_percentile,
            args.upper_percentile,
        ],
        "valid_samples": int(len(metadata)),
        "failed_items": int(len(failures_df)),
        "qc_flagged_samples": int(len(suspicious)),
        "series_count": int(metadata["series_name"].nunique()),
        "patient_count": int(metadata["patient"].astype(str).nunique()),
        "subset_counts": {
            str(k): int(v) for k, v in Counter(metadata["subset"]).items()
        },
        "grade_counts": {
            f"{row.subset}_grade_{int(row.pfirrmann_grade_original)}": int(row.count)
            for row in grade_counts.itertuples(index=False)
        },
        "important_note": (
            "IVD label 是从尾侧向头侧编号；不要在未核对解剖变异时强制映射为 L5-S1。"
        ),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n✅ 导出完成")
    print(f"有效样本：{len(metadata)}")
    print(f"失败记录：{len(failures_df)}")
    print(f"元数据：{metadata_path}")
    if qc_path:
        print(f"质控拼图：{qc_path}")
    if suspicious_qc_path:
        print(f"可疑样本拼图：{suspicious_qc_path}")
    if args.save_full_slices:
        print(f"完整选中切面：{args.output_dir / 'full_slices'}")
    print("\n等级分布：")
    print(grade_counts.to_string(index=False))
    print(
        "\n请先打开 qc_montage.png 检查红色 mask 是否准确覆盖椎间盘，"
        "再开始训练。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n❌ 运行失败：{exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)