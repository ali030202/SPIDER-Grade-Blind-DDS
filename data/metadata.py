from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.io import atomic_json_save


LEVELS_TOP_DOWN = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
LEVELS_BOTTOM_UP = list(reversed(LEVELS_TOP_DOWN))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATA2_PATTERN = re.compile(
    r"^p(?P<patient>.+?)_ivd(?P<ivd>\d+)(?:_t2)?_g(?P<grade>[1-5])$",
    re.IGNORECASE,
)
DATA1_PATTERN = re.compile(r"^(?P<patient>.+?)_disc(?P<disc>\d+)$", re.IGNORECASE)


def canonical_patient(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return str(int(text))
    except ValueError:
        return text.lower()


def windows_basename(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return PureWindowsPath(text).name


def normalize_grades(series: pd.Series, source_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    observed = sorted(set(int(v) for v in values.dropna().tolist()))
    if not observed:
        raise ValueError(f"{source_name} 中没有有效 Pfirrmann 等级")
    minimum, maximum = min(observed), max(observed)
    if minimum >= 1 and maximum <= 5:
        return values.astype("Int64")
    if minimum >= 0 and maximum <= 4:
        return (values + 1).astype("Int64")
    raise ValueError(
        f"{source_name} 的 Pfirrmann 等级范围不明确：{observed}；只接受1-5或0-4。"
    )


def level_mapping(order: str) -> Dict[int, str]:
    if order not in {"top-down", "bottom-up"}:
        raise ValueError(f"未知椎间盘顺序：{order}")
    levels = LEVELS_TOP_DOWN if order == "top-down" else LEVELS_BOTTOM_UP
    return {index + 1: value for index, value in enumerate(levels)}


def scan_data2_directory(directory: Path) -> Tuple[Dict[Tuple[str, int, int], Path], List[Dict[str, object]]]:
    index: Dict[Tuple[str, int, int], Path] = {}
    rejected: List[Dict[str, object]] = []
    if not directory.exists():
        return index, [{"dataset": "data2", "reason": "directory_missing", "path": str(directory)}]
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = DATA2_PATTERN.match(path.stem)
        if match is None:
            rejected.append(
                {"dataset": "data2", "reason": "filename_pattern_unrecognized", "path": str(path)}
            )
            continue
        key = (
            canonical_patient(match.group("patient")),
            int(match.group("ivd")),
            int(match.group("grade")),
        )
        if key in index:
            rejected.append(
                {
                    "dataset": "data2",
                    "reason": "duplicate_key",
                    "key": str(key),
                    "path": str(path),
                    "kept": str(index[key]),
                }
            )
            continue
        index[key] = path.resolve()
    return index, rejected


def build_data1_frame(
    csv_path: Path,
    image_dir: Path,
    order: str,
    missing_policy: str,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"data1 CSV不存在：{csv_path}")
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"patient_root": str})
    required = {"image", "pfirrmann"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"data1 CSV缺少列：{sorted(missing)}")
    grades = normalize_grades(frame["pfirrmann"], "data1")
    mapping = level_mapping(order)
    rows: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for row_index, row in frame.iterrows():
        filename = windows_basename(row["image"])
        match = DATA1_PATTERN.match(Path(filename).stem)
        if match is None:
            rejected.append(
                {"dataset": "data1", "row": int(row_index), "reason": "filename_pattern_unrecognized", "image": filename}
            )
            continue
        disc_index = int(match.group("disc"))
        if disc_index not in mapping:
            rejected.append(
                {"dataset": "data1", "row": int(row_index), "reason": "disc_outside_1_5", "disc_index": disc_index}
            )
            continue
        grade = grades.iloc[row_index]
        if pd.isna(grade):
            rejected.append(
                {"dataset": "data1", "row": int(row_index), "reason": "missing_grade"}
            )
            continue
        patient_raw = row.get("patient_root", row.get("patient_id", match.group("patient")))
        patient = canonical_patient(patient_raw)
        image_path = (image_dir / filename).resolve()
        if not image_path.exists():
            item = {
                "dataset": "data1",
                "row": int(row_index),
                "reason": "image_missing",
                "path": str(image_path),
            }
            rejected.append(item)
            if missing_policy == "error":
                raise FileNotFoundError(json.dumps(item, ensure_ascii=False))
            continue
        rows.append(
            {
                "sample_id": f"data1::{patient}::disc{disc_index:02d}",
                "dataset": "data1",
                "patient_id": f"data1::{patient}",
                "source_patient_id": patient,
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": 0,
                "disc_index": disc_index,
                "disc_level": mapping[disc_index],
                "grade": int(grade),
                "age": pd.to_numeric(row.get("age", np.nan), errors="coerce"),
                "sex": str(row.get("sex", "")).strip(),
                "source_subset": "",
                "source_row": int(row_index),
            }
        )
    return pd.DataFrame(rows), rejected


def build_data2_frame(
    csv_path: Path,
    image_dir: Path,
    mask_dir: Path,
    order: str,
    missing_policy: str,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"data2 CSV不存在：{csv_path}")
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"patient": str})
    required = {"patient", "ivd_label", "pfirrmann"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"data2 CSV缺少列：{sorted(missing)}")
    grades = normalize_grades(frame["pfirrmann"], "data2")
    mapping = level_mapping(order)
    image_index, rejected = scan_data2_directory(image_dir)
    mask_index, mask_rejected = scan_data2_directory(mask_dir)
    rejected.extend(mask_rejected)
    rows: List[Dict[str, object]] = []
    used_images = set()
    for row_index, row in frame.iterrows():
        patient = canonical_patient(row["patient"])
        disc_index_value = pd.to_numeric(row["ivd_label"], errors="coerce")
        grade = grades.iloc[row_index]
        if pd.isna(disc_index_value) or pd.isna(grade):
            rejected.append(
                {"dataset": "data2", "row": int(row_index), "reason": "missing_disc_or_grade"}
            )
            continue
        disc_index = int(disc_index_value)
        if disc_index not in mapping:
            rejected.append(
                {"dataset": "data2", "row": int(row_index), "reason": "disc_outside_1_5", "disc_index": disc_index}
            )
            continue
        key = (patient, disc_index, int(grade))
        image_path = image_index.get(key)
        if image_path is None:
            # Diagnostic fallback: locate same patient/disc but different grade.
            alternatives = [str(path) for (p, d, _), path in image_index.items() if p == patient and d == disc_index]
            item = {
                "dataset": "data2",
                "row": int(row_index),
                "reason": "image_key_missing",
                "key": str(key),
                "alternatives": alternatives[:8],
            }
            rejected.append(item)
            if missing_policy == "error":
                raise FileNotFoundError(json.dumps(item, ensure_ascii=False))
            continue
        if str(image_path) in used_images:
            rejected.append(
                {"dataset": "data2", "row": int(row_index), "reason": "duplicate_csv_mapping", "path": str(image_path)}
            )
            continue
        used_images.add(str(image_path))
        mask_path = mask_index.get(key)
        rows.append(
            {
                "sample_id": f"data2::{patient}::ivd{disc_index:02d}",
                "dataset": "data2",
                "patient_id": f"data2::{patient}",
                "source_patient_id": patient,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "has_mask": int(mask_path is not None),
                "disc_index": disc_index,
                "disc_level": mapping[disc_index],
                "grade": int(grade),
                "age": np.nan,
                "sex": "",
                "source_subset": str(row.get("subset", "")).strip().lower(),
                "source_row": int(row_index),
            }
        )
    return pd.DataFrame(rows), rejected


def _group_cell_counts(frame: pd.DataFrame) -> Tuple[List[str], Dict[str, Counter]]:
    cells = sorted(
        set(
            f"{row.dataset}|{row.disc_level}|G{int(row.grade)}"
            for row in frame.itertuples(index=False)
        )
    )
    groups: Dict[str, Counter] = {}
    for patient, group in frame.groupby("patient_id", sort=False):
        counter = Counter(
            f"{row.dataset}|{row.disc_level}|G{int(row.grade)}"
            for row in group.itertuples(index=False)
        )
        groups[str(patient)] = counter
    return cells, groups


def greedy_patient_split(
    frame: pd.DataFrame,
    seed: int = 2025,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Patient-level, approximately stratified split with hard patient capacities.

    Each patient is assigned exactly once. A greedy delta-cost objective tries to
    match dataset × level × grade cell counts while hard capacities preserve the
    requested patient proportions.
    """
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"split ratios之和必须为1，当前={ratios}")
    names = ["train", "val", "test"]
    cells, groups = _group_cell_counts(frame)
    totals = Counter()
    for counter in groups.values():
        totals.update(counter)

    patient_total = len(groups)
    raw_capacities = np.asarray(ratios, dtype=np.float64) * patient_total
    capacities = np.floor(raw_capacities).astype(int)
    remainder = patient_total - int(capacities.sum())
    fractional_order = np.argsort(-(raw_capacities - capacities))
    for index in fractional_order[:remainder]:
        capacities[index] += 1
    # For practical train/val/test usage, keep each split non-empty when possible.
    if patient_total >= 3:
        for index in range(3):
            if capacities[index] == 0:
                donor = int(np.argmax(capacities))
                capacities[donor] -= 1
                capacities[index] += 1
    capacity = {name: int(value) for name, value in zip(names, capacities)}

    target = {
        split: {cell: totals[cell] * ratio for cell in cells}
        for split, ratio in zip(names, ratios)
    }
    assigned = {split: Counter() for split in names}
    patient_counts = Counter()
    rng = np.random.default_rng(seed)

    rarity = {}
    for patient, counter in groups.items():
        rarity[patient] = sum(count / max(totals[cell], 1) for cell, count in counter.items())
    patients = list(groups)
    rng.shuffle(patients)
    patients.sort(key=lambda p: (rarity[p], sum(groups[p].values())), reverse=True)

    allocation: Dict[str, str] = {}
    for patient in patients:
        counter = groups[patient]
        candidates = [name for name in names if patient_counts[name] < capacity[name]]
        if not candidates:
            raise RuntimeError("患者划分容量耗尽，内部逻辑错误")
        best_split = None
        best_delta = float("inf")
        for split in candidates:
            delta = 0.0
            for cell in cells:
                before = assigned[split][cell]
                after = before + counter[cell]
                desired = target[split][cell]
                denominator = max(desired, 1.0)
                delta += ((after - desired) ** 2 - (before - desired) ** 2) / denominator
            # Mild sample-count term for patients containing unequal numbers of discs.
            before_samples = sum(assigned[split].values())
            after_samples = before_samples + sum(counter.values())
            desired_samples = len(frame) * ratios[names.index(split)]
            delta += 0.20 * (
                ((after_samples - desired_samples) ** 2 - (before_samples - desired_samples) ** 2)
                / max(desired_samples, 1.0)
            )
            if delta < best_delta:
                best_delta = delta
                best_split = split
        assert best_split is not None
        allocation[patient] = best_split
        assigned[best_split].update(counter)
        patient_counts[best_split] += 1

    result = frame.copy()
    result["split"] = result["patient_id"].map(allocation)
    return result

def distribution_report(frame: pd.DataFrame) -> Dict[str, object]:
    report: Dict[str, object] = {
        "samples": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "datasets": frame["dataset"].value_counts().sort_index().to_dict(),
        "grades": frame["grade"].value_counts().sort_index().to_dict(),
        "levels": frame["disc_level"].value_counts().reindex(LEVELS_TOP_DOWN, fill_value=0).to_dict(),
        "has_mask": frame["has_mask"].value_counts().sort_index().to_dict(),
    }
    if "split" in frame.columns:
        report["splits"] = {}
        for split, group in frame.groupby("split"):
            report["splits"][split] = {
                "samples": int(len(group)),
                "patients": int(group["patient_id"].nunique()),
                "grades": group["grade"].value_counts().sort_index().to_dict(),
                "levels": group["disc_level"].value_counts().reindex(LEVELS_TOP_DOWN, fill_value=0).to_dict(),
                "dataset": group["dataset"].value_counts().sort_index().to_dict(),
            }
    cross = pd.crosstab(frame["disc_level"], frame["grade"]).reindex(
        index=LEVELS_TOP_DOWN, columns=[1, 2, 3, 4, 5], fill_value=0
    )
    report["level_grade"] = {
        level: {f"G{grade}": int(cross.loc[level, grade]) for grade in cross.columns}
        for level in cross.index
    }
    return report


def prepare_metadata(
    data_root: Path,
    output_dir: Path,
    data1_csv: Optional[Path] = None,
    data2_csv: Optional[Path] = None,
    data1_order: str = "top-down",
    data2_order: str = "bottom-up",
    missing_policy: str = "skip",
    seed: int = 2025,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data1_csv = data1_csv or data_root / "data1" / "labels_with_clinical.csv"
    data2_csv = data2_csv or data_root / "data2" / "metadata_external_valid.csv"
    data1, rejected1 = build_data1_frame(
        data1_csv,
        data_root / "data1" / "images",
        data1_order,
        missing_policy,
    )
    data2, rejected2 = build_data2_frame(
        data2_csv,
        data_root / "data2" / "image",
        data_root / "data2" / "mask",
        data2_order,
        missing_policy,
    )
    frame = pd.concat([data1, data2], ignore_index=True)
    if frame.empty:
        raise RuntimeError("两个数据集清理后没有任何有效样本")
    duplicates = frame.duplicated(subset=["image_path"], keep="first")
    duplicate_rows = frame.loc[duplicates].copy()
    frame = frame.loc[~duplicates].copy().reset_index(drop=True)
    frame = greedy_patient_split(frame, seed=seed, ratios=split_ratios)
    frame["level_index"] = frame["disc_level"].map({value: i for i, value in enumerate(LEVELS_TOP_DOWN)})
    frame = frame.sort_values(["split", "dataset", "patient_id", "disc_index"]).reset_index(drop=True)
    frame.to_csv(output_dir / "prepared_metadata.csv", index=False, encoding="utf-8-sig")
    frame[["patient_id", "split"]].drop_duplicates().sort_values("patient_id").to_csv(
        output_dir / "patient_splits.csv", index=False, encoding="utf-8-sig"
    )
    rejected = rejected1 + rejected2
    if len(duplicate_rows):
        rejected.extend(
            {
                "dataset": row.dataset,
                "reason": "duplicate_image_path",
                "path": row.image_path,
            }
            for row in duplicate_rows.itertuples(index=False)
        )
    pd.DataFrame(rejected).to_csv(
        output_dir / "metadata_rejected.csv", index=False, encoding="utf-8-sig"
    )
    report = distribution_report(frame)
    report.update(
        {
            "data_root": str(data_root),
            "data1_csv": str(data1_csv),
            "data2_csv": str(data2_csv),
            "data1_order": data1_order,
            "data2_order": data2_order,
            "missing_policy": missing_policy,
            "rejected_rows": int(len(rejected)),
        }
    )
    atomic_json_save(report, output_dir / "metadata_report.json")
    return frame
