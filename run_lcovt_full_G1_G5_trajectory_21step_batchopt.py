#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LCOVT 完整推理脚本（推荐主实验：每个 source 生成 21 帧完整 G1→G5 轨迹）
========================================================================

设计目标
--------
1. 从 prepared_metadata.csv 中读取 source MRI
2. 只做一次 source latent inversion
3. 沿同一条 LCOVT 连续轨迹生成完整 G1→G5
4. 默认生成 21 帧：
      q = 1.0, 1.2, 1.4, ..., 4.8, 5.0
5. 自动包含 5 个整数锚点：
      G1, G2, G3, G4, G5
6. 每个病例自动保存：
      - source.png
      - vae_recon.png
      - 21帧单图
      - 5个整数锚点单图
      - 一张总 contact sheet
      - 逐帧 metrics CSV
7. 全部病例自动汇总：
      - selected_cases.csv
      - trajectory_metrics_all.csv
      - integer_anchor_index_all.csv
      - experiment_manifest.json
      - experiment.log

推荐用途
--------
A. 先做 30 个随机 test 样例：
   用于肉眼检查连续性、单调性、identity 保持。

B. 再做全部 test 样例：
   用于后续 6 个 AI evaluator 的正式定量评价。

依赖关系
--------
本脚本是“完整推理入口脚本”，但仍会调用你现有的 stage2 helper 脚本。
该 helper 脚本需要提供以下函数（与你当前版本一致）：
    - load_models(...)
    - resolve_image_path(...)
    - load_source_image(...)
    - tensor_to_pil(...)
    - ddim_invert_to_strength(...)
    - ddim_local_ordinal_edit(...)

常用命令
--------
1) 30 个随机 test 样例，21 帧轨迹（推荐先跑）
python run_lcovt_full_G1_G5_trajectory_21step_full.py \
  --root /home/lab/jmx/SPIDER \
  --stage2-script /home/lab/jmx/SPIDER/stage2_lcovt_v3_nlp_inspired.py \
  --stage1-script /home/lab/jmx/SPIDER/train_ordinal_latent_diffusion_v2.py \
  --checkpoint /home/lab/jmx/SPIDER/runs_ordinal_ldm_v2_224x160/checkpoints/best.pt \
  --vae-checkpoint /home/lab/jmx/SPIDER/runs_vae_v2/checkpoints/vae_best.pt \
  --metadata-csv /home/lab/jmx/SPIDER/data/prepared_metadata.csv \
  --output-dir /home/lab/jmx/SPIDER/runs_lcovt_infer_30cases_21step \
  --split test \
  --sample-mode random \
  --max-cases 30 \
  --num-trajectory-steps 21 \
  --strength 0.96 \
  --cfg 1.5 \
  --sampling-steps 120 \
  --seed 2026

2) 全部 test 样例，21 帧轨迹（正式实验）
python run_lcovt_full_G1_G5_trajectory_21step_full.py \
  --root /home/lab/jmx/SPIDER \
  --stage2-script /home/lab/jmx/SPIDER/stage2_lcovt_v3_nlp_inspired.py \
  --stage1-script /home/lab/jmx/SPIDER/train_ordinal_latent_diffusion_v2.py \
  --checkpoint /home/lab/jmx/SPIDER/runs_ordinal_ldm_v2_224x160/checkpoints/best.pt \
  --vae-checkpoint /home/lab/jmx/SPIDER/runs_vae_v2/checkpoints/vae_best.pt \
  --metadata-csv /home/lab/jmx/SPIDER/data/prepared_metadata.csv \
  --output-dir /home/lab/jmx/SPIDER/runs_lcovt_infer_alltest_21step \
  --split test \
  --sample-mode sequential \
  --max-cases 0 \
  --num-trajectory-steps 21 \
  --strength 0.96 \
  --cfg 1.5 \
  --sampling-steps 120 \
  --seed 2026
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
import importlib.util
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm


SCRIPT_VERSION = "LCOVT_FULL_G1_G5_TRAJECTORY_21STEP_BATCHOPT_V2"


# ---------------------------------------------------------------------
# Basic utils
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_python(path: Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"lcovt_full_infer_{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def parse_int_list(text: str) -> List[int]:
    vals = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sanitize_for_filename(text: str) -> str:
    s = str(text).strip()
    if not s:
        return "NA"
    allowed = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    out = "".join(allowed)
    while "__" in out:
        out = out.replace("__", "_")
    return out[:80] if out else "NA"


# ---------------------------------------------------------------------
# Performance / asynchronous I/O helpers
# ---------------------------------------------------------------------

def configure_cuda_runtime(args, device: torch.device, logger: logging.Logger) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if bool(args.allow_tf32):
        # High uses TF32 where PyTorch considers it appropriate on Ampere/Ada GPUs.
        torch.set_float32_matmul_precision("high")
    logger.info("allow_tf32     = %s", bool(args.allow_tf32))
    logger.info("cudnn_benchmark= %s", bool(args.cudnn_benchmark))


def autocast_context(device: torch.device, amp_dtype: str):
    name = str(amp_dtype).lower()
    if device.type != "cuda" or name == "none":
        return nullcontext()
    if name == "fp16":
        dtype = torch.float16
    elif name == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"unknown amp dtype: {amp_dtype}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_tensor_png(stage2, tensor_cpu: torch.Tensor, path: Path) -> None:
    ensure_dir(Path(path).parent)
    stage2.tensor_to_pil(tensor_cpu).save(path)


def _check_done_futures(done) -> None:
    for fut in done:
        # Raises the original worker exception in the main thread.
        fut.result()


def queue_io(
    executor: Optional[ThreadPoolExecutor],
    pending: Set[Future],
    max_pending: int,
    fn,
    *args,
    **kwargs,
) -> None:
    if executor is None:
        fn(*args, **kwargs)
        return
    fut = executor.submit(fn, *args, **kwargs)
    pending.add(fut)
    limit = max(1, int(max_pending))
    if len(pending) >= limit:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        _check_done_futures(done)
        pending.difference_update(done)


def flush_io(pending: Set[Future]) -> None:
    if not pending:
        return
    done, _ = wait(pending)
    _check_done_futures(done)
    pending.clear()


def write_root_tables(output_dir: Path, all_rows, all_anchor_rows) -> None:
    pd.DataFrame(all_rows).to_csv(
        output_dir / "trajectory_metrics_all.csv",
        index=False,
    )
    pd.DataFrame(all_anchor_rows).to_csv(
        output_dir / "integer_anchor_index_all.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Math / metrics helpers
# ---------------------------------------------------------------------

def tensor_image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().mean().item())


def tensor_latent_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().mean().item())


def tensor_latent_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).square().mean().sqrt().item())


def build_uniform_q_schedule(num_steps: int) -> np.ndarray:
    """
    推荐主实验：21 帧
        q = 1.0, 1.2, ..., 5.0

    为了保证 G1/G2/G3/G4/G5 五个整数锚点都能精确出现，
    这里要求 (num_steps - 1) 能被 4 整除。
    常见可用值：5, 9, 13, 17, 21, 25, ...

    默认 21 最合适。
    """
    num_steps = int(num_steps)
    if num_steps < 5:
        raise ValueError("--num-trajectory-steps must be >= 5")
    if (num_steps - 1) % 4 != 0:
        raise ValueError(
            "--num-trajectory-steps must satisfy (steps - 1) % 4 == 0, "
            "so that integer anchors G1..G5 are included exactly. "
            "Recommended: 21."
        )

    q_values = np.linspace(1.0, 5.0, num_steps, dtype=np.float64)

    for g in range(1, 6):
        if not np.any(np.isclose(q_values, float(g), atol=1e-10)):
            raise RuntimeError(f"integer anchor G{g} missing from q schedule")

    return q_values


def integer_anchor_indices(q_values: np.ndarray) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for g in range(1, 6):
        idxs = np.where(np.isclose(q_values, float(g), atol=1e-10))[0]
        if len(idxs) != 1:
            raise RuntimeError(f"Expected exactly one integer anchor for G{g}, got {len(idxs)}")
        out[g] = int(idxs[0])
    return out


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def normalize_split_name(x: str) -> str:
    x = str(x).strip().lower()
    mapping = {
        "training": "train",
        "valid": "val",
        "validation": "val",
        "testing": "test",
    }
    return mapping.get(x, x)


def load_metadata(args) -> Tuple[pd.DataFrame, Path]:
    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    df = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype={"patient_id": str})

    if "image_path" not in df.columns:
        for candidate in ["image_relative_path", "image", "path", "unified_image_path"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "image_path"})
                break

    required = {"image_path", "grade", "level_index", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    df = df.copy()
    df.insert(0, "metadata_index", np.arange(len(df), dtype=np.int64))
    df["grade"] = pd.to_numeric(df["grade"], errors="raise").astype(int)
    df["level_index"] = pd.to_numeric(df["level_index"], errors="raise").astype(int)
    df["split"] = df["split"].map(normalize_split_name)

    if "patient_id" not in df.columns:
        df["patient_id"] = ""
    if "sample_id" not in df.columns:
        df["sample_id"] = df["metadata_index"].astype(str)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    source_grades = set(parse_int_list(args.source_grades))
    df = df[df["grade"].isin(source_grades)].copy()

    if args.levels != "all":
        level_set = set(parse_int_list(args.levels))
        df = df[df["level_index"].isin(level_set)].copy()

    if len(df) == 0:
        raise RuntimeError("No cases left after split/source-grade/level filtering")

    if args.sample_mode == "random":
        df = df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    start = max(0, int(args.start_index))
    df = df.iloc[start:].reset_index(drop=True)

    if int(args.max_cases) > 0:
        df = df.iloc[: int(args.max_cases)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No cases selected after start_index/max_cases slicing")

    return df, metadata_path


# ---------------------------------------------------------------------
# Visualization / saving
# ---------------------------------------------------------------------

def save_grid(
    stage2,
    trajectory: List[Dict[str, object]],
    source_grade: int,
    level_index: int,
    patient_id: str,
    path: Path,
    columns: int = 7,
) -> None:
    """
    21 帧默认 7 列 × 3 行，比较紧凑。
    """
    if not trajectory:
        return

    images = [stage2.tensor_to_pil(item["image"]) for item in trajectory]
    w, h = images[0].size
    columns = max(1, int(columns))
    rows = int(math.ceil(len(images) / columns))
    label_h = 24
    title_h = 32

    canvas = Image.new("L", (columns * w, title_h + rows * (h + label_h)), color=0)
    draw = ImageDraw.Draw(canvas)
    title = (
        f"LCOVT full G1->G5 trajectory | pid={patient_id or 'NA'} "
        f"| native=G{source_grade} | L{level_index} | N={len(images)}"
    )
    draw.text((5, 5), title, fill=255)

    for idx, (im, item) in enumerate(zip(images, trajectory)):
        row = idx // columns
        col = idx % columns
        x = col * w
        y = title_h + row * (h + label_h)
        canvas.paste(im, (x, y))

        q = float(item["continuous_grade"])
        native = " SRC" if np.isclose(q, float(source_grade), atol=1e-8) else ""
        draw.text((x + 3, y + h + 3), f"{idx:02d} q={q:.1f}{native}", fill=255)

    ensure_dir(path.parent)
    canvas.save(path)


def save_integer_anchor_images(
    stage2,
    trajectory: List[Dict[str, object]],
    q_values: np.ndarray,
    output_dir: Path,
) -> List[Dict[str, object]]:
    """
    从完整轨迹中抽取：
        q=1.0, 2.0, 3.0, 4.0, 5.0
    保存为 integer_anchors/G1.png ... G5.png
    """
    ensure_dir(output_dir)
    anchor_map = integer_anchor_indices(q_values)
    records: List[Dict[str, object]] = []

    for grade, idx in anchor_map.items():
        item = trajectory[idx]
        q = float(item["continuous_grade"])
        out_path = output_dir / f"G{grade}.png"
        stage2.tensor_to_pil(item["image"]).save(out_path)

        rec = {
            "target_integer_grade": int(grade),
            "step_index": int(idx),
            "continuous_grade": float(q),
            "file_path": str(out_path),
        }
        records.append(rec)

    return records


# ---------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_full_trajectory(
    stage2,
    runtime: Dict,
    source_image: torch.Tensor,
    source_grade: int,
    level_index: int,
    q_values: np.ndarray,
    args,
    progress_desc: str = "G1→G5 trajectory",
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    """Generate the full ordinal trajectory with q-batched LCOVT reverse diffusion.

    The mathematical LCOVT method is unchanged.  The optimization is execution-only:
    one source inversion is shared by all q values, and several q trajectories are
    advanced through each reverse DDIM timestep together.
    """
    vae = runtime["vae"]
    model = runtime["model"]
    schedule = runtime["schedule"]
    stage1 = runtime["stage1"]
    device = source_image.device
    num_grades = int(runtime["config"].num_grades)

    source_latent, _ = vae.encode(source_image, sample=False)
    recon = vae.decode(source_latent).clamp(0.0, 1.0)

    sg = torch.full(
        (source_latent.shape[0],),
        int(source_grade),
        device=device,
        dtype=torch.long,
    )
    level = torch.full(
        (source_latent.shape[0],),
        int(level_index),
        device=device,
        dtype=torch.long,
    )

    # One inversion per source, exactly as in the original script.
    start_latent, timeline = stage2.ddim_invert_to_strength(
        stage1,
        model,
        schedule,
        source_latent,
        sg,
        level,
        sampling_steps=int(args.sampling_steps),
        strength=float(args.strength),
        inversion_guidance_scale=float(args.inversion_guidance_scale),
        x0_clip=float(args.x0_clip),
    )

    if not hasattr(stage2, "ddim_local_ordinal_edit_batch"):
        raise RuntimeError(
            "The selected --stage2-script does not provide ddim_local_ordinal_edit_batch(). "
            "Use stage2_lcovt_v3_nlp_inspired_batchopt.py generated together with this script."
        )

    trajectory: List[Dict[str, object]] = []
    q_batch_size = max(1, int(args.trajectory_batch_size))

    pbar = tqdm(
        total=len(q_values),
        desc=progress_desc,
        unit="frame",
        leave=False,
        dynamic_ncols=True,
        disable=not bool(args.progress),
    )

    for batch_start in range(0, len(q_values), q_batch_size):
        batch_end = min(len(q_values), batch_start + q_batch_size)
        q_batch_np = q_values[batch_start:batch_end]
        q_batch = torch.as_tensor(q_batch_np, device=device, dtype=torch.float32)

        edited_latents, stats_list = stage2.ddim_local_ordinal_edit_batch(
            stage1,
            model,
            schedule,
            start_latent,
            source_latent,
            q_batch,
            level,
            timeline,
            guidance_scale=float(args.cfg),
            num_grades=num_grades,
            late_identity_blend=float(args.late_identity_blend),
            late_identity_steps=int(args.late_identity_steps),
            x0_clip=float(args.x0_clip),
        )

        # Decode all q values in the current trajectory batch together.
        images = vae.decode(edited_latents).clamp(0.0, 1.0)
        batch = int(images.shape[0])
        source_image_b = source_image.expand(batch, *source_image.shape[1:])
        source_latent_b = source_latent.expand(batch, *source_latent.shape[1:])

        # Vectorized metrics: only one GPU->CPU synchronization per metric vector.
        image_l1 = (
            (images.float() - source_image_b.float()).abs().flatten(1).mean(dim=1).detach().cpu().tolist()
        )
        latent_delta = edited_latents.float() - source_latent_b.float()
        latent_l1 = latent_delta.abs().flatten(1).mean(dim=1).detach().cpu().tolist()
        latent_rms = latent_delta.square().flatten(1).mean(dim=1).sqrt().detach().cpu().tolist()

        # Move finished outputs to CPU immediately, freeing GPU memory for the next q batch.
        images_cpu = images.detach().float().cpu()
        latents_cpu = edited_latents.detach().float().cpu()

        for j, q in enumerate(q_batch_np.tolist()):
            idx = batch_start + j
            stats = stats_list[j]
            trajectory.append({
                "step_index": int(idx),
                "continuous_grade": float(q),
                "image": images_cpu[j:j + 1].clone(),
                "latent": latents_cpu[j:j + 1].clone(),
                "source_grade": int(source_grade),
                "level_index": int(level_index),
                "strength": float(args.strength),
                "cfg": float(args.cfg),
                "start_timestep": int(stats.get("start_timestep", 0)),
                "anchor_left": float(stats.get("anchor_left", q)),
                "anchor_right": float(stats.get("anchor_right", q)),
                "anchor_mix": float(stats.get("anchor_mix", 0.0)),
                "image_l1_to_source": float(image_l1[j]),
                "latent_l1_to_source": float(latent_l1[j]),
                "latent_rms_to_source": float(latent_rms[j]),
                "is_native_grade_anchor": bool(np.isclose(q, float(source_grade), atol=1e-8)),
                "is_integer_anchor": bool(np.isclose(q, np.round(q), atol=1e-8)),
            })

        if bool(args.progress):
            pbar.set_postfix_str(
                f"q={float(q_batch_np[0]):.1f}..{float(q_batch_np[-1]):.1f} B={batch}"
            )
        pbar.update(batch)

        del images, edited_latents, source_image_b, source_latent_b, latent_delta

    pbar.close()
    return source_latent.detach().float().cpu(), recon.detach().float().cpu(), trajectory


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run full 21-step LCOVT G1→G5 trajectory inference",
    )

    # 基础路径
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--stage2-script", type=Path, required=True)
    p.add_argument("--stage1-script", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument(
        "--allow-vae-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # 样本选择
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--source-grades", type=str, default="1,2,3,4,5")
    p.add_argument("--levels", type=str, default="all")
    p.add_argument("--sample-mode", choices=["sequential", "random"], default="random")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=30, help="0 means all selected")

    # 推理超参
    p.add_argument("--strength", type=float, default=0.96)
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sampling-steps", type=int, default=120)
    p.add_argument(
        "--num-trajectory-steps",
        type=int,
        default=21,
        help="Recommended 21: q=1.0,1.2,...,5.0",
    )
    p.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    p.add_argument("--late-identity-blend", type=float, default=0.0)
    p.add_argument("--late-identity-steps", type=int, default=6)
    p.add_argument("--x0-clip", type=float, default=4.5)

    # GPU吞吐优化。trajectory batch只改变执行方式，不改变LCOVT数学定义。
    p.add_argument(
        "--trajectory-batch-size",
        type=int,
        default=4,
        help="How many q trajectory points are advanced together on GPU. Try 4, then 8 on a 4090D.",
    )
    p.add_argument(
        "--amp-dtype",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Optional CUDA autocast. For scientific equivalence checks, start with none; then benchmark bf16.",
    )
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile-mode", type=str, default="reduce-overhead")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    # 保存与显示
    p.add_argument("--grid-columns", type=int, default=7)
    p.add_argument("--save-individual-steps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-integer-anchors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--io-workers", type=int, default=4,
        help="Background CPU workers for PNG/contact-sheet saving; 0 disables async I/O.",
    )
    p.add_argument(
        "--max-pending-io", type=int, default=64,
        help="Backpressure limit for queued image-save jobs.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=20,
        help="Rewrite root aggregate CSVs every N completed cases; final write is always performed.",
    )

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda")
    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    args.root = Path(args.root).expanduser().resolve()
    args.stage2_script = Path(args.stage2_script).expanduser().resolve()
    args.stage1_script = Path(args.stage1_script).expanduser().resolve()
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    args.vae_checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
    args.metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    args.output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if not (0.0 <= float(args.strength) <= 1.0):
        raise ValueError("--strength must be in [0,1]")
    if float(args.cfg) <= 0.0:
        raise ValueError("--cfg must be > 0")
    if int(args.sampling_steps) <= 0:
        raise ValueError("--sampling-steps must be > 0")
    if int(args.trajectory_batch_size) <= 0:
        raise ValueError("--trajectory-batch-size must be > 0")
    if int(args.io_workers) < 0:
        raise ValueError("--io-workers must be >= 0")

    q_values = build_uniform_q_schedule(int(args.num_trajectory_steps))
    anchor_map = integer_anchor_indices(q_values)

    set_all_seeds(int(args.seed))
    logger = setup_logger(args.output_dir)

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; switching to CPU")
        device_name = "cpu"
    device = torch.device(device_name)
    configure_cuda_runtime(args, device, logger)

    logger.info("=" * 88)
    logger.info("SCRIPT_VERSION = %s", SCRIPT_VERSION)
    logger.info("root          = %s", args.root)
    logger.info("stage2_script = %s", args.stage2_script)
    logger.info("stage1_script = %s", args.stage1_script)
    logger.info("checkpoint    = %s", args.checkpoint)
    logger.info("vae_checkpoint= %s", args.vae_checkpoint)
    logger.info("metadata_csv  = %s", args.metadata_csv)
    logger.info("output_dir    = %s", args.output_dir)
    logger.info("device        = %s", device)
    logger.info("split         = %s", args.split)
    logger.info("source_grades = %s", args.source_grades)
    logger.info("levels        = %s", args.levels)
    logger.info("sample_mode   = %s", args.sample_mode)
    logger.info("max_cases     = %d", int(args.max_cases))
    logger.info("strength      = %.4f", float(args.strength))
    logger.info("cfg           = %.4f", float(args.cfg))
    logger.info("sampling_steps= %d", int(args.sampling_steps))
    logger.info("num_steps     = %d", int(args.num_trajectory_steps))
    logger.info("q_batch_size  = %d", int(args.trajectory_batch_size))
    logger.info("amp_dtype     = %s", args.amp_dtype)
    logger.info("io_workers    = %d", int(args.io_workers))
    logger.info("checkpoint_ev = %d", int(args.checkpoint_every))
    logger.info("q schedule    = %s", ", ".join(f"{x:.1f}" for x in q_values))
    logger.info("integer anchors: %s", anchor_map)
    logger.info("=" * 88)

    stage2 = import_python(args.stage2_script, "stage2_lcovt_helper_for_full_inference")
    runtime = stage2.load_models(
        args.stage1_script,
        args.checkpoint,
        args.vae_checkpoint,
        device,
        logger,
        allow_vae_mismatch=bool(args.allow_vae_mismatch),
    )

    if bool(args.torch_compile):
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile is unavailable in this PyTorch build; continuing without it")
        else:
            logger.info("Compiling diffusion model with torch.compile(mode=%s)...", args.compile_mode)
            runtime["model"] = torch.compile(runtime["model"], mode=str(args.compile_mode))

    df, metadata_path = load_metadata(args)
    df.to_csv(args.output_dir / "selected_cases.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "method": "LCOVT local ordinal v-space interpolation (q-batched execution)",
        "trajectory_type": "one unified full G1→G5 trajectory per source",
        "num_trajectory_steps": int(args.num_trajectory_steps),
        "q_values": q_values.tolist(),
        "integer_anchor_indices": {str(k): int(v) for k, v in anchor_map.items()},
        "strength": float(args.strength),
        "cfg": float(args.cfg),
        "sampling_steps": int(args.sampling_steps),
        "trajectory_batch_size": int(args.trajectory_batch_size),
        "amp_dtype": str(args.amp_dtype),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_rows: List[Dict[str, object]] = []
    all_anchor_rows: List[Dict[str, object]] = []
    completed_since_checkpoint = 0

    io_executor: Optional[ThreadPoolExecutor]
    if int(args.io_workers) > 0:
        io_executor = ThreadPoolExecutor(
            max_workers=int(args.io_workers),
            thread_name_prefix="lcovt_io",
        )
    else:
        io_executor = None
    pending_io: Set[Future] = set()

    case_iter = tqdm(
        df.iterrows(),
        total=len(df),
        desc="All cases",
        unit="case",
        dynamic_ncols=True,
        disable=not bool(args.progress),
    )

    try:
        for local_idx, row in case_iter:
            meta_idx = int(row["metadata_index"])
            source_grade = int(row["grade"])
            level_index = int(row["level_index"])
            patient_id = str(row.get("patient_id", ""))
            sample_id = str(row.get("sample_id", ""))

            try:
                image_path = stage2.resolve_image_path(
                    str(row["image_path"]),
                    args.root,
                    metadata_path,
                )

                pid_tag = sanitize_for_filename(patient_id) if patient_id else "NA"
                sid_tag = sanitize_for_filename(sample_id) if sample_id else "NA"
                case_name = (
                    f"idx_{meta_idx:06d}"
                    f"_pid_{pid_tag}"
                    f"_sid_{sid_tag}"
                    f"_SRC_G{source_grade}"
                    f"_L{level_index}"
                )
                case_dir = ensure_dir(args.output_dir / "cases" / case_name)

                if bool(args.progress):
                    case_iter.set_postfix_str(case_name)

                grid_path = case_dir / "trajectory_full_G1_G5_21step.png"
                csv_path = case_dir / "trajectory_metrics.csv"
                anchor_csv_path = case_dir / "integer_anchor_index.csv"

                if bool(args.skip_existing) and grid_path.exists() and csv_path.exists():
                    logger.info("[%d/%d] skip existing %s", local_idx + 1, len(df), case_name)
                    old = pd.read_csv(csv_path)
                    all_rows.extend(old.to_dict("records"))
                    if anchor_csv_path.exists():
                        old_anchor = pd.read_csv(anchor_csv_path)
                        all_anchor_rows.extend(old_anchor.to_dict("records"))
                    continue

                source_cpu = stage2.load_source_image(
                    image_path,
                    runtime["config"].image_height,
                    runtime["config"].image_width,
                )
                if device.type == "cuda" and bool(args.pin_memory):
                    source_cpu = source_cpu.pin_memory()
                source_image = source_cpu.to(
                    device,
                    non_blocking=(device.type == "cuda" and bool(args.pin_memory)),
                )

                with autocast_context(device, args.amp_dtype):
                    _, recon_cpu, trajectory = generate_full_trajectory(
                        stage2=stage2,
                        runtime=runtime,
                        source_image=source_image,
                        source_grade=source_grade,
                        level_index=level_index,
                        q_values=q_values,
                        args=args,
                        progress_desc=case_name,
                    )

                # Queue source/reconstruction/contact sheet in CPU background workers.
                queue_io(
                    io_executor, pending_io, args.max_pending_io,
                    save_tensor_png, stage2, source_cpu, case_dir / "source.png",
                )
                queue_io(
                    io_executor, pending_io, args.max_pending_io,
                    save_tensor_png, stage2, recon_cpu, case_dir / "vae_recon.png",
                )
                queue_io(
                    io_executor, pending_io, args.max_pending_io,
                    save_grid,
                    stage2=stage2,
                    trajectory=trajectory,
                    source_grade=source_grade,
                    level_index=level_index,
                    patient_id=patient_id,
                    path=grid_path,
                    columns=int(args.grid_columns),
                )

                # Save all individual frames asynchronously; metrics CSV remains synchronous.
                records: List[Dict[str, object]] = []
                if bool(args.save_individual_steps):
                    step_dir = ensure_dir(case_dir / "trajectory_steps")
                else:
                    step_dir = None

                for item in trajectory:
                    rec = {k: v for k, v in item.items() if k not in {"image", "latent"}}
                    rec.update({
                        "metadata_index": meta_idx,
                        "patient_id": patient_id,
                        "sample_id": sample_id,
                        "image_path": str(image_path),
                        "split": str(row["split"]),
                    })

                    if step_dir is not None:
                        idx = int(item["step_index"])
                        q = float(item["continuous_grade"])
                        step_path = step_dir / f"step_{idx:02d}_q{q:.1f}.png"
                        queue_io(
                            io_executor, pending_io, args.max_pending_io,
                            save_tensor_png, stage2, item["image"], step_path,
                        )
                        rec["step_file_path"] = str(step_path)
                    else:
                        rec["step_file_path"] = ""
                    records.append(rec)

                pd.DataFrame(records).to_csv(csv_path, index=False)
                all_rows.extend(records)

                # Integer anchor images are references into the already-generated trajectory.
                anchor_rows: List[Dict[str, object]] = []
                if bool(args.save_integer_anchors):
                    anchor_dir = ensure_dir(case_dir / "integer_anchors")
                    trajectory_by_idx = {int(item["step_index"]): item for item in trajectory}
                    for grade, idx in anchor_map.items():
                        item = trajectory_by_idx[int(idx)]
                        q = float(item["continuous_grade"])
                        out_path = anchor_dir / f"G{grade}.png"
                        queue_io(
                            io_executor, pending_io, args.max_pending_io,
                            save_tensor_png, stage2, item["image"], out_path,
                        )
                        row_anchor = {
                            "target_integer_grade": int(grade),
                            "step_index": int(idx),
                            "continuous_grade": float(q),
                            "file_path": str(out_path),
                            "metadata_index": meta_idx,
                            "patient_id": patient_id,
                            "sample_id": sample_id,
                            "source_grade": int(source_grade),
                            "level_index": int(level_index),
                            "split": str(row["split"]),
                            "source_image_path": str(image_path),
                            "is_native_grade_anchor": bool(item["is_native_grade_anchor"]),
                        }
                        anchor_rows.append(row_anchor)

                    pd.DataFrame(anchor_rows).to_csv(anchor_csv_path, index=False)
                    all_anchor_rows.extend(anchor_rows)

                completed_since_checkpoint += 1
                checkpoint_every = int(args.checkpoint_every)
                if checkpoint_every > 0 and completed_since_checkpoint >= checkpoint_every:
                    write_root_tables(args.output_dir, all_rows, all_anchor_rows)
                    completed_since_checkpoint = 0

                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    logger.info(
                        "[%d/%d] done %s | pid=%s | native=G%d | L%d | frames=%d | "
                        "cuda_alloc=%.2fGiB reserved=%.2fGiB peak=%.2fGiB",
                        local_idx + 1, len(df), case_name,
                        patient_id if patient_id else "NA",
                        source_grade, level_index, len(q_values),
                        mem_alloc, mem_reserved, mem_peak,
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    logger.info(
                        "[%d/%d] done %s | pid=%s | native=G%d | L%d | frames=%d",
                        local_idx + 1, len(df), case_name,
                        patient_id if patient_id else "NA",
                        source_grade, level_index, len(q_values),
                    )

                del source_image, source_cpu, recon_cpu, trajectory

            except Exception as exc:
                logger.exception(
                    "[%d/%d] FAILED metadata_index=%d patient_id=%s: %s",
                    local_idx + 1,
                    len(df),
                    meta_idx,
                    patient_id,
                    exc,
                )
                if not bool(args.continue_on_error):
                    raise

        # All GPU work is finished; wait for outstanding PNG/contact-sheet jobs.
        flush_io(pending_io)
        write_root_tables(args.output_dir, all_rows, all_anchor_rows)

    finally:
        # Ensure worker threads are joined even if a foreground exception occurs.
        if io_executor is not None:
            try:
                flush_io(pending_io)
            finally:
                io_executor.shutdown(wait=True, cancel_futures=False)

    logger.info("✅ finished | cases=%d | output=%s", len(df), args.output_dir)


if __name__ == "__main__":
    main()
