#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ordinal Latent Diffusion Baseline V1
====================================

Purpose
-------
A clean baseline for generating lumbar IVD MRI directly from:

    target Pfirrmann grade (G1..G5)
    + disc level

This is NOT a grade-chain model and does NOT use a source image.

The model learns a single distribution:

    p(z | ordinal severity, disc level)

where the Pfirrmann grade is represented as a continuous ordinal scalar:

    G1=-1.0, G2=-0.5, G3=0.0, G4=0.5, G5=1.0

This allows one diffusion model to learn the full ordered grade spectrum and
avoids cumulative edge_12 -> edge_23 -> edge_34 -> edge_45 errors.

Important design choices
------------------------
1. Reuses ONLY the already-good VAE from the previous grade-chain project.
2. VAE is frozen.
3. Diffusion is trained only on REAL image latents.
4. No assessor/classifier loss.
5. No feature-bank loss.
6. No GAN.
7. No source-image conditioning.
8. No handcrafted morphology features.
9. Classifier-free guidance is supported.
10. Preview uses the SAME initial noise across G1..G5, making ordinal changes
    directly visible.

Expected project structure
--------------------------
SPIDER/
├── data/
├── model/
│   ├── __init__.py
│   ├── blocks.py
│   ├── vae.py
│   ├── ordinal_ldm.py
│   └── diffusion.py
├── train_vae_v2.py
└── train_ordinal_latent_diffusion_v1.py

The VAE and ordinal diffusion network are imported from the clean model package.
No grade-chain assessor/anatomy/discriminator code is required.

Examples
--------
Train:

python train_ordinal_latent_diffusion_v1.py \
  --mode train \
  --root /home/lab/jmx/SPIDER \
  --run-dir /home/lab/jmx/SPIDER/runs_ordinal_latent_diffusion_v1 \
  --vae-checkpoint /home/lab/jmx/SPIDER/runs_vae_v2/checkpoints/vae_best.pt

Generate a G2 image at level index 3:

python train_ordinal_latent_diffusion_v1.py \
  --mode sample \
  --root /home/lab/jmx/SPIDER \
  --run-dir /home/lab/jmx/SPIDER/runs_ordinal_latent_diffusion_v1 \
  --target-grade 2 \
  --level-index 3 \
  --num-samples 16
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Project data module + clean model package.
from data import IVDDataset, prepare_metadata
from model import (
    DiffusionSchedule,
    ImageVAE,
    OrdinalLatentDiffusionConfig,
    OrdinalLatentUNet,
    VAEConfig,
    extract,
)


SCRIPT_VERSION = "ORDINAL_LATENT_DIFFUSION_CLEAN_V2"


# ================================================================
# Basic utilities
# ================================================================


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.write_text(
        json.dumps(convert(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ordinal_latent_diffusion")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        run_dir / "train.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return autocast(device_type="cuda", dtype=torch.float16)

    from contextlib import nullcontext
    return nullcontext()


class EMA:
    def __init__(self, module: nn.Module, beta: float = 0.999):
        self.beta = float(beta)
        self.shadow = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        current = module.state_dict()
        for key, value in current.items():
            if not torch.is_floating_point(value):
                self.shadow[key] = value.detach().clone()
                continue
            self.shadow[key].mul_(self.beta).add_(
                value.detach(),
                alpha=1.0 - self.beta,
            )

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state, device: torch.device):
        self.shadow = {
            key: value.to(device)
            for key, value in state.items()
        }

    def copy_to(self, module: nn.Module):
        backup = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

        state = module.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value.to(state[key].device))
        return backup

    @staticmethod
    def restore(module: nn.Module, backup):
        state = module.state_dict()
        for key, value in backup.items():
            state[key].copy_(value.to(state[key].device))


# ================================================================
# Data
# ================================================================


def build_balanced_train_loader(
    frame: pd.DataFrame,
    dataset: IVDDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    balance_power: float,
) -> DataLoader:
    """
    Balance by (disc level, grade) cell.

    weight_i = 1 / count(cell_i)^balance_power

    balance_power:
        0.0 -> no balancing
        0.5 -> moderate
        1.0 -> approximately equal cell sampling
    """

    balance_power = float(
        max(0.0, balance_power)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    shuffle = True

    if balance_power > 0:

        cells = [
            (
                int(row.level_index),
                int(row.grade),
            )
            for row in frame.itertuples(
                index=False
            )
        ]

        counts = Counter(cells)

        weights = torch.tensor(
            [
                1.0
                / (
                    float(counts[cell])
                    ** balance_power
                )
                for cell in cells
            ],
            dtype=torch.double,
        )

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )

        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=(
            seed_worker
            if num_workers > 0
            else None
        ),
        generator=generator,
        drop_last=True,
    )


def build_eval_loader(
    dataset: IVDDataset,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


# ================================================================
# VAE loading
# ================================================================

def load_frozen_vae(
    config: OrdinalLatentDiffusionConfig,
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> ImageVAE:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到VAE checkpoint: {checkpoint_path}\n"
            "请使用你之前训练好的 runs_vae_v2/checkpoints/vae_best.pt"
        )

    vae_config = VAEConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        latent_channels=config.latent_channels,
        vae_base=config.vae_base,
    )

    vae = ImageVAE(
        vae_config
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "components" in checkpoint
    ):
        components = checkpoint[
            "components"
        ]

        if "vae" not in components:
            raise KeyError(
                f"{checkpoint_path} 的 components 中没有 'vae'"
            )

        state = components[
            "vae"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "vae" in checkpoint
    ):
        state = checkpoint[
            "vae"
        ]

    elif isinstance(checkpoint, dict):
        # Allow a pure state_dict.
        state = checkpoint

    else:
        raise TypeError(
            "无法识别VAE checkpoint格式"
        )

    message = vae.load_state_dict(
        state,
        strict=True,
    )

    vae.eval()

    for parameter in vae.parameters():
        parameter.requires_grad = False

    logger.info(
        "✅ Frozen VAE loaded: %s",
        checkpoint_path,
    )
    logger.info(
        "VAE latent_scale=%.6f | load=%s",
        float(vae.latent_scale.detach().cpu()),
        message,
    )

    return vae


@torch.no_grad()
def infer_latent_shape(
    vae: ImageVAE,
    config: OrdinalLatentDiffusionConfig,
    device: torch.device,
) -> Tuple[int, int, int]:

    dummy = torch.zeros(
        1,
        1,
        config.image_height,
        config.image_width,
        device=device,
    )

    latent, _ = vae.encode(
        dummy,
        sample=False,
    )

    return (
        int(latent.shape[1]),
        int(latent.shape[2]),
        int(latent.shape[3]),
    )


# ================================================================
# DDIM sampling from pure Gaussian noise
# ================================================================

@torch.no_grad()
def sample_latents(
    model: OrdinalLatentUNet,
    schedule: DiffusionSchedule,
    shape: Tuple[int, int, int, int],
    grade: torch.Tensor,
    level: torch.Tensor,
    steps: int = 60,
    guidance_scale: float = 2.0,
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    device = grade.device
    batch_size = shape[0]

    steps = max(
        2,
        int(steps),
    )

    if initial_noise is None:
        value = torch.randn(
            shape,
            device=device,
        )
    else:
        if tuple(initial_noise.shape) != tuple(shape):
            raise ValueError(
                f"initial_noise shape={tuple(initial_noise.shape)} "
                f"!= expected={tuple(shape)}"
            )
        value = initial_noise.clone()

    # Full generative trajectory from T-1 -> 0.
    timeline = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=device,
    ).round().long()

    # Remove accidental duplicates if steps is large.
    timeline = torch.unique_consecutive(
        timeline
    )

    drop_none = torch.zeros(
        batch_size,
        device=device,
        dtype=torch.bool,
    )

    drop_all = torch.ones(
        batch_size,
        device=device,
        dtype=torch.bool,
    )

    for index, current_scalar in enumerate(timeline):

        current = torch.full(
            (batch_size,),
            int(current_scalar.item()),
            device=device,
            dtype=torch.long,
        )

        conditional = model(
            value,
            current,
            grade,
            level,
            drop_mask=drop_none,
        )

        if guidance_scale != 1.0:

            unconditional = model(
                value,
                current,
                grade,
                level,
                drop_mask=drop_all,
            )

            predicted_v = (
                unconditional
                + float(guidance_scale)
                * (
                    conditional
                    - unconditional
                )
            )

        else:
            predicted_v = conditional

        x0 = schedule.predict_x0_from_v(
            value,
            predicted_v,
            current,
        ).clamp(
            -4.5,
            4.5,
        )

        if index == len(timeline) - 1:
            value = x0
            break

        next_scalar = timeline[
            index + 1
        ]

        next_t = torch.full(
            (batch_size,),
            int(next_scalar.item()),
            device=device,
            dtype=torch.long,
        )

        predicted_noise = schedule.predict_noise_from_v(
            value,
            predicted_v,
            current,
        )

        next_alpha = extract(
            schedule.sqrt_alpha_bar,
            next_t,
            value.shape,
        )

        next_sigma = extract(
            schedule.sqrt_one_minus_alpha_bar,
            next_t,
            value.shape,
        )

        # Deterministic DDIM update.
        value = (
            next_alpha
            * x0
            + next_sigma
            * predicted_noise
        )

    return value


@torch.no_grad()
def sample_images(
    model: OrdinalLatentUNet,
    vae: ImageVAE,
    schedule: DiffusionSchedule,
    latent_shape: Tuple[int, int, int],
    grades: torch.Tensor,
    levels: torch.Tensor,
    steps: int,
    guidance_scale: float,
    initial_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    shape = (
        grades.shape[0],
        latent_shape[0],
        latent_shape[1],
        latent_shape[2],
    )

    latent = sample_latents(
        model,
        schedule,
        shape,
        grades,
        levels,
        steps=steps,
        guidance_scale=guidance_scale,
        initial_noise=initial_noise,
    )

    images = vae.decode(
        latent
    ).clamp(
        0.0,
        1.0,
    )

    return images


# ================================================================
# Preview
# ================================================================

def tensor_to_pil(
    tensor: torch.Tensor,
) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .squeeze(0)
        .numpy()
        * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="L",
    )


@torch.no_grad()
def save_ordinal_preview(
    model: OrdinalLatentUNet,
    vae: ImageVAE,
    schedule: DiffusionSchedule,
    latent_shape: Tuple[int, int, int],
    output_path: Path,
    level_index: int,
    rows: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    """
    Every row uses one fixed latent noise seed.
    G1..G5 are generated from the SAME noise.

    Therefore differences across a row are primarily due to the
    continuous ordinal condition rather than random identity changes.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = next(
        model.parameters()
    ).device

    model.eval()
    vae.eval()

    cell_w = 224
    cell_h = 224
    label_h = 24

    canvas = Image.new(
        "L",
        (
            5 * cell_w,
            rows * (cell_h + label_h),
        ),
        color=0,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for row in range(rows):

        generator = torch.Generator(
            device=device
        ).manual_seed(
            int(seed + row)
        )

        base_noise = torch.randn(
            (
                1,
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
            ),
            generator=generator,
            device=device,
        )

        for column, grade_value in enumerate(
            range(1, 6)
        ):

            grade = torch.tensor(
                [grade_value],
                device=device,
                dtype=torch.long,
            )

            level = torch.tensor(
                [level_index],
                device=device,
                dtype=torch.long,
            )

            image = sample_images(
                model,
                vae,
                schedule,
                latent_shape,
                grade,
                level,
                steps=steps,
                guidance_scale=guidance_scale,
                initial_noise=base_noise,
            )[0]

            pil = tensor_to_pil(
                image
            ).resize(
                (cell_w, cell_h),
                Image.Resampling.BILINEAR,
            )

            x = column * cell_w
            y = row * (cell_h + label_h)

            canvas.paste(
                pil,
                (x, y),
            )

            draw.text(
                (
                    x + 4,
                    y + cell_h + 3,
                ),
                f"G{grade_value} | level={level_index}",
                fill=255,
            )

    canvas.save(
        output_path
    )


# ================================================================
# Validation
# ================================================================

@torch.no_grad()
def validate(
    model: OrdinalLatentUNet,
    vae: ImageVAE,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    min_snr_gamma: float,
    max_batches: int = 64,
) -> Dict[str, float]:

    model.eval()
    vae.eval()

    total_v = 0.0
    total_x0 = 0.0
    total_count = 0

    generator = torch.Generator(
        device=device
    ).manual_seed(
        17001
    )

    for batch_index, batch in enumerate(loader):

        if batch_index >= max_batches:
            break

        image = batch["image"].to(
            device,
            non_blocking=True,
        )
        grade = batch["grade"].to(
            device,
            non_blocking=True,
        )
        level = batch["level"].to(
            device,
            non_blocking=True,
        )

        latent, _ = vae.encode(
            image,
            sample=False,
        )

        timestep = torch.randint(
            0,
            schedule.timesteps,
            (latent.shape[0],),
            device=device,
            generator=generator,
        )

        noise = torch.randn(
            latent.shape,
            device=device,
            dtype=latent.dtype,
            generator=generator,
        )

        noisy = schedule.q_sample(
            latent,
            timestep,
            noise,
        )

        target_v = schedule.v_target(
            latent,
            noise,
            timestep,
        )

        predicted_v = model(
            noisy,
            timestep,
            grade,
            level,
            drop_mask=torch.zeros(
                latent.shape[0],
                device=device,
                dtype=torch.bool,
            ),
        )

        per_sample = (
            predicted_v
            - target_v
        ).square().mean(
            dim=(1, 2, 3)
        )

        weights = schedule.min_snr_weight(
            timestep,
            min_snr_gamma,
        )

        v_loss = (
            per_sample
            * weights
        )

        predicted_x0 = schedule.predict_x0_from_v(
            noisy,
            predicted_v,
            timestep,
        )

        x0_l1 = (
            predicted_x0
            - latent
        ).abs().mean(
            dim=(1, 2, 3)
        )

        batch_size = latent.shape[0]

        total_v += float(
            v_loss.sum()
        )
        total_x0 += float(
            x0_l1.sum()
        )
        total_count += batch_size

    return {
        "v_loss": total_v / max(total_count, 1),
        "x0_l1": total_x0 / max(total_count, 1),
    }


# ================================================================
# Checkpoint
# ================================================================

def save_checkpoint(
    path: Path,
    model: OrdinalLatentUNet,
    optimizer,
    scaler,
    ema: EMA,
    config: OrdinalLatentDiffusionConfig,
    epoch: int,
    best_score: float,
    args,
    vae_checkpoint: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "script_version":
            SCRIPT_VERSION,

        "config":
            asdict(config),

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict()
            if optimizer is not None
            else None,

        "scaler":
            scaler.state_dict()
            if scaler is not None
            else None,

        "ema":
            {
                key:
                    value.detach().cpu()
                for key, value
                in ema.state_dict().items()
            },

        "epoch":
            int(epoch),

        "best_score":
            float(best_score),

        "vae_checkpoint":
            str(vae_checkpoint),

        "vae_sha256":
            sha256_file(
                vae_checkpoint
            ),

        "arguments":
            {
                key:
                    str(value)
                    if isinstance(value, Path)
                    else value
                for key, value
                in vars(args).items()
            },
    }

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temp_path,
    )

    temp_path.replace(
        path
    )


def load_diffusion_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = OrdinalLatentDiffusionConfig(
        **checkpoint["config"]
    )

    model = OrdinalLatentUNet(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    ema_state = checkpoint.get(
        "ema"
    )

    if ema_state is not None:
        state = model.state_dict()
        for key, value in ema_state.items():
            state[key].copy_(
                value.to(
                    state[key].device
                )
            )

    model.eval()

    return model, config, checkpoint


# ================================================================
# Metadata
# ================================================================

def prepare_or_load_metadata(
    args,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    """
    Priority:
    1. --metadata-csv
    2. previous grade-chain prepared_metadata.csv
    3. build a new prepared_metadata.csv with the existing prepare_metadata()
    """

    if args.metadata_csv is not None:

        prepared_path = (
            args.metadata_csv
            .expanduser()
            .resolve()
        )

        if not prepared_path.exists():
            raise FileNotFoundError(
                prepared_path
            )

        frame = pd.read_csv(
            prepared_path,
            encoding="utf-8-sig",
            dtype={
                "patient_id": str
            },
        )

        logger.info(
            "✅ 使用指定prepared metadata: %s",
            prepared_path,
        )

    else:

        previous = (
            args.root
            / "prepared_metadata.csv"
        )

        own = (
            args.run_dir
            / "data"
            / "prepared_metadata.csv"
        )

        if (
            previous.exists()
            and not args.rebuild_data
        ):
            prepared_path = previous

            frame = pd.read_csv(
                prepared_path,
                encoding="utf-8-sig",
                dtype={
                    "patient_id": str
                },
            )

            logger.info(
                "✅ 复用grade-chain完全相同的数据划分: %s",
                prepared_path,
            )

        elif (
            own.exists()
            and not args.rebuild_data
        ):
            prepared_path = own

            frame = pd.read_csv(
                prepared_path,
                encoding="utf-8-sig",
                dtype={
                    "patient_id": str
                },
            )

        else:

            own.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            frame = prepare_metadata(
                args.data_root,
                args.run_dir / "data",
                data1_csv=args.data1_csv,
                data2_csv=args.data2_csv,
                data1_order=args.data1_order,
                data2_order=args.data2_order,
                missing_policy=args.missing_policy,
                seed=args.split_seed,
                split_ratios=(
                    args.train_fraction,
                    args.val_fraction,
                    args.test_fraction,
                ),
            )

            prepared_path = own

    required = {
        "image_path",
        "grade",
        "level_index",
        "split",
        "patient_id",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"prepared metadata缺少列: {sorted(missing)}"
        )

    train_frame = frame[
        frame["split"] == "train"
    ].reset_index(drop=True)

    val_frame = frame[
        frame["split"] == "val"
    ].reset_index(drop=True)

    test_frame = frame[
        frame["split"] == "test"
    ].reset_index(drop=True)

    if min(
        len(train_frame),
        len(val_frame),
        len(test_frame),
    ) == 0:
        raise RuntimeError(
            "train/val/test存在空划分"
        )

    logger.info(
        "samples train=%d val=%d test=%d | patients=%d/%d/%d",
        len(train_frame),
        len(val_frame),
        len(test_frame),
        train_frame.patient_id.nunique(),
        val_frame.patient_id.nunique(),
        test_frame.patient_id.nunique(),
    )

    logger.info(
        "train grade distribution=%s",
        train_frame["grade"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    return (
        train_frame,
        val_frame,
        test_frame,
        prepared_path,
    )


# ================================================================
# Train
# ================================================================

def run_train(args) -> int:

    args.root = args.root.expanduser().resolve()

    args.run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else args.root
        / "runs_ordinal_latent_diffusion_v1"
    )

    args.data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else args.root / "data"
    )

    if args.vae_checkpoint is None:
        args.vae_checkpoint = (
            args.root
            / "runs_vae_v2"
            / "checkpoints"
            / "vae_best.pt"
        )

    args.vae_checkpoint = (
        args.vae_checkpoint
        .expanduser()
        .resolve()
    )

    for folder in [
        "checkpoints",
        "metrics",
        "previews",
        "samples",
        "data",
    ]:
        (
            args.run_dir
            / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    logger = setup_logger(
        args.run_dir
    )

    seed_everything(
        args.seed,
        args.deterministic,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = OrdinalLatentDiffusionConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        latent_channels=args.latent_channels,
        vae_base=args.vae_base,
        diffusion_base=args.diffusion_base,
        condition_dim=args.condition_dim,
        diffusion_timesteps=args.diffusion_timesteps,
        dropout=args.dropout,
    )

    save_json(
        {
            "script_version":
                SCRIPT_VERSION,
            "config":
                asdict(config),
            "arguments":
                vars(args),
        },
        args.run_dir
        / "launch_config.json",
    )

    logger.info("=" * 80)
    logger.info(SCRIPT_VERSION)
    logger.info("root=%s", args.root)
    logger.info("run_dir=%s", args.run_dir)
    logger.info(
        "device=%s amp=%s",
        device,
        args.amp and device.type == "cuda",
    )
    if device.type == "cuda":
        logger.info(
            "GPU=%s",
            torch.cuda.get_device_name(0),
        )

    (
        train_frame,
        val_frame,
        test_frame,
        metadata_path,
    ) = prepare_or_load_metadata(
        args,
        logger,
    )

    train_dataset = IVDDataset(
        train_frame,
        args.image_height,
        args.image_width,
        augment=True,
        two_views=False,
    )

    val_dataset = IVDDataset(
        val_frame,
        args.image_height,
        args.image_width,
        augment=False,
        two_views=False,
    )

    test_dataset = IVDDataset(
        test_frame,
        args.image_height,
        args.image_width,
        augment=False,
        two_views=False,
    )

    train_loader = build_balanced_train_loader(
        train_frame,
        train_dataset,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.balance_power,
    )

    val_loader = build_eval_loader(
        val_dataset,
        args.batch_size,
        num_workers=0,
    )

    _ = test_dataset  # retained for the final split / later independent evaluation.

    vae = load_frozen_vae(
        config,
        args.vae_checkpoint,
        device,
        logger,
    )

    latent_shape = infer_latent_shape(
        vae,
        config,
        device,
    )

    logger.info(
        "latent shape CxHxW=%s",
        latent_shape,
    )

    model = OrdinalLatentUNet(
        config
    ).to(device)

    schedule = DiffusionSchedule(
        config.diffusion_timesteps
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    logger.info(
        "diffusion parameters=%d",
        parameter_count,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(
        device="cuda",
        enabled=(
            args.amp
            and device.type == "cuda"
        ),
    )

    ema = EMA(
        model,
        args.ema_beta,
    )

    checkpoint_dir = (
        args.run_dir
        / "checkpoints"
    )

    best_path = (
        checkpoint_dir
        / "best.pt"
    )

    last_path = (
        checkpoint_dir
        / "last.pt"
    )

    metrics_path = (
        args.run_dir
        / "metrics"
        / "diffusion.csv"
    )

    start_epoch = 1
    best_score = float("inf")
    bad_epochs = 0

    if (
        args.resume
        and last_path.exists()
    ):

        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if (
            checkpoint.get("scaler")
            is not None
            and scaler.is_enabled()
        ):
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        if checkpoint.get("ema") is not None:
            ema.load_state_dict(
                checkpoint["ema"],
                device,
            )

        start_epoch = (
            int(checkpoint["epoch"])
            + 1
        )

        best_score = float(
            checkpoint["best_score"]
        )

        bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        logger.info(
            "♻️ resume epoch=%d best=%.6f",
            start_epoch,
            best_score,
        )

    if (
        not metrics_path.exists()
        or start_epoch == 1
    ):
        metrics_path.write_text(
            (
                "epoch,"
                "train_loss,"
                "train_raw_mse,"
                "val_v_loss,"
                "val_x0_l1,"
                "score,"
                "lr\n"
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_items = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"ORDINAL-LDM "
                f"{epoch}/{args.epochs}"
            ),
        )

        for batch in progress:

            image = batch["image"].to(
                device,
                non_blocking=True,
            )

            grade = batch["grade"].to(
                device,
                non_blocking=True,
            )

            level = batch["level"].to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                latent, _ = vae.encode(
                    image,
                    sample=False,
                )

            timestep = torch.randint(
                0,
                schedule.timesteps,
                (latent.shape[0],),
                device=device,
            )

            noise = torch.randn_like(
                latent
            )

            noisy = schedule.q_sample(
                latent,
                timestep,
                noise,
            )

            target_v = schedule.v_target(
                latent,
                noise,
                timestep,
            )

            drop_mask = (
                torch.rand(
                    latent.shape[0],
                    device=device,
                )
                < args.condition_dropout
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_context(
                device,
                args.amp,
            ):

                predicted_v = model(
                    noisy,
                    timestep,
                    grade,
                    level,
                    drop_mask=drop_mask,
                )

                per_sample = (
                    predicted_v
                    - target_v
                ).square().mean(
                    dim=(1, 2, 3)
                )

                weights = (
                    schedule
                    .min_snr_weight(
                        timestep,
                        args.min_snr_gamma,
                    )
                )

                loss = (
                    per_sample
                    * weights
                ).mean()

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            ema.update(
                model
            )

            batch_size = latent.shape[0]

            total_loss += (
                float(
                    loss.detach()
                )
                * batch_size
            )

            total_mse += (
                float(
                    per_sample.mean().detach()
                )
                * batch_size
            )

            total_items += (
                batch_size
            )

            progress.set_postfix(
                loss=(
                    f"{float(loss.detach()):.4f}"
                ),
                raw=(
                    f"{float(per_sample.mean().detach()):.4f}"
                ),
            )

        train_loss = (
            total_loss
            / max(
                total_items,
                1,
            )
        )

        train_mse = (
            total_mse
            / max(
                total_items,
                1,
            )
        )

        # --------------------------------------------------------
        # Validate EMA model
        # --------------------------------------------------------

        backup = ema.copy_to(
            model
        )

        validation = validate(
            model,
            vae,
            schedule,
            val_loader,
            device,
            args.min_snr_gamma,
            max_batches=args.val_batches,
        )

        score = (
            validation["v_loss"]
            + 0.25
            * validation["x0_l1"]
        )

        improved = (
            score
            < best_score
        )

        if improved:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Preview is produced from EMA weights before restoration.
        if (
            epoch == 1
            or epoch % args.preview_every == 0
            or improved
        ):

            for level_index in (
                args.preview_levels
            ):
                save_ordinal_preview(
                    model,
                    vae,
                    schedule,
                    latent_shape,
                    (
                        args.run_dir
                        / "previews"
                        / (
                            f"epoch_{epoch:03d}"
                            f"_level_{level_index}.png"
                        )
                    ),
                    level_index=level_index,
                    rows=args.preview_rows,
                    steps=args.preview_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.preview_seed,
                )

        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                ema,
                config,
                epoch,
                best_score,
                args,
                args.vae_checkpoint,
            )

        EMA.restore(
            model,
            backup,
        )

        # Save last with non-EMA model + EMA shadow.
        payload = {
            "script_version":
                SCRIPT_VERSION,
            "config":
                asdict(config),
            "model":
                model.state_dict(),
            "optimizer":
                optimizer.state_dict(),
            "scaler":
                scaler.state_dict(),
            "ema":
                {
                    key:
                        value.detach().cpu()
                    for key, value
                    in ema.state_dict().items()
                },
            "epoch":
                int(epoch),
            "best_score":
                float(best_score),
            "bad_epochs":
                int(bad_epochs),
            "vae_checkpoint":
                str(args.vae_checkpoint),
            "vae_sha256":
                sha256_file(
                    args.vae_checkpoint
                ),
            "metadata_csv":
                str(metadata_path),
        }

        torch.save(
            payload,
            last_path,
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"{epoch},"
                f"{train_loss:.8f},"
                f"{train_mse:.8f},"
                f"{validation['v_loss']:.8f},"
                f"{validation['x0_l1']:.8f},"
                f"{score:.8f},"
                f"{current_lr:.10f}\n"
            )

        logger.info(
            "[%03d] train=%.6f raw=%.6f | "
            "val_v=%.6f x0=%.6f score=%.6f%s",
            epoch,
            train_loss,
            train_mse,
            validation["v_loss"],
            validation["x0_l1"],
            score,
            " ✅" if improved else "",
        )

        if (
            bad_epochs
            >= args.patience
        ):
            logger.info(
                "Early stopping: %d epochs without improvement",
                bad_epochs,
            )
            break

    # ------------------------------------------------------------
    # Final preview from best checkpoint
    # ------------------------------------------------------------

    best_model, best_config, checkpoint = (
        load_diffusion_checkpoint(
            best_path,
            device,
        )
    )

    if (
        asdict(best_config)
        != asdict(config)
    ):
        logger.warning(
            "best config differs from launch config"
        )

    final_schedule = DiffusionSchedule(
        best_config.diffusion_timesteps
    )

    final_vae = load_frozen_vae(
        best_config,
        Path(
            checkpoint[
                "vae_checkpoint"
            ]
        ),
        device,
        logger,
    )

    final_latent_shape = infer_latent_shape(
        final_vae,
        best_config,
        device,
    )

    for level_index in args.preview_levels:

        save_ordinal_preview(
            best_model,
            final_vae,
            final_schedule,
            final_latent_shape,
            (
                args.run_dir
                / "previews"
                / f"FINAL_level_{level_index}.png"
            ),
            level_index=level_index,
            rows=max(
                args.preview_rows,
                4,
            ),
            steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            seed=args.preview_seed,
        )

    save_json(
        {
            "script_version":
                SCRIPT_VERSION,
            "best_checkpoint":
                str(best_path),
            "best_score":
                float(
                    checkpoint[
                        "best_score"
                    ]
                ),
            "vae_checkpoint":
                checkpoint[
                    "vae_checkpoint"
                ],
            "metadata_csv":
                str(metadata_path),
            "grade_scalar_mapping":
                {
                    "G1": -1.0,
                    "G2": -0.5,
                    "G3": 0.0,
                    "G4": 0.5,
                    "G5": 1.0,
                },
        },
        args.run_dir
        / "final_manifest.json",
    )

    logger.info(
        "✅ Training complete: %s",
        best_path,
    )

    return 0


# ================================================================
# Sampling mode
# ================================================================

def run_sample(args) -> int:

    args.root = args.root.expanduser().resolve()

    args.run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else args.root
        / "runs_ordinal_latent_diffusion_v1"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else args.run_dir
        / "checkpoints"
        / "best.pt"
    )

    logger = setup_logger(
        args.run_dir
    )

    model, config, checkpoint = (
        load_diffusion_checkpoint(
            checkpoint_path,
            device,
        )
    )

    vae_checkpoint = (
        args.vae_checkpoint.expanduser().resolve()
        if args.vae_checkpoint is not None
        else Path(
            checkpoint[
                "vae_checkpoint"
            ]
        )
    )

    vae = load_frozen_vae(
        config,
        vae_checkpoint,
        device,
        logger,
    )

    schedule = DiffusionSchedule(
        config.diffusion_timesteps
    )

    latent_shape = infer_latent_shape(
        vae,
        config,
        device,
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.run_dir
        / "samples"
        / (
            f"G{args.target_grade}"
            f"_level{args.level_index}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        1
        <= args.target_grade
        <= 5
    ):
        raise ValueError(
            "--target-grade 必须是1~5"
        )

    if not (
        0
        <= args.level_index
        < config.num_levels
    ):
        raise ValueError(
            f"--level-index 必须是0~{config.num_levels - 1}"
        )

    model.eval()
    vae.eval()

    generated = 0

    while generated < args.num_samples:

        batch_size = min(
            args.sample_batch_size,
            args.num_samples
            - generated,
        )

        grades = torch.full(
            (batch_size,),
            args.target_grade,
            device=device,
            dtype=torch.long,
        )

        levels = torch.full(
            (batch_size,),
            args.level_index,
            device=device,
            dtype=torch.long,
        )

        generator = torch.Generator(
            device=device
        ).manual_seed(
            args.sample_seed
            + generated
        )

        initial_noise = torch.randn(
            (
                batch_size,
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
            ),
            generator=generator,
            device=device,
        )

        images = sample_images(
            model,
            vae,
            schedule,
            latent_shape,
            grades,
            levels,
            steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            initial_noise=initial_noise,
        )

        for offset in range(
            batch_size
        ):

            index = (
                generated
                + offset
            )

            path = (
                output_dir
                / f"{index:04d}.png"
            )

            tensor_to_pil(
                images[offset]
            ).save(path)

        generated += batch_size

        logger.info(
            "generated %d/%d",
            generated,
            args.num_samples,
        )

    logger.info(
        "✅ samples saved to: %s",
        output_dir,
    )

    return 0


# ================================================================
# Parser
# ================================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Ordinal Latent Diffusion baseline for Pfirrmann-grade MRI generation"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["train", "sample"],
        default="train",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "已有prepared_metadata.csv。未提供时先检查项目根目录的"
            "prepared_metadata.csv，否则按 data 配置重新准备。"
        ),
    )

    parser.add_argument(
        "--data1-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--data2-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--data1-order",
        choices=[
            "top-down",
            "bottom-up",
        ],
        default="top-down",
    )

    parser.add_argument(
        "--data2-order",
        choices=[
            "top-down",
            "bottom-up",
        ],
        default="bottom-up",
    )

    parser.add_argument(
        "--missing-policy",
        choices=[
            "skip",
            "error",
        ],
        default="skip",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--rebuild-data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--latent-channels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--vae-base",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--diffusion-base",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--condition-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--diffusion-timesteps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=8e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=0.12,
        help="classifier-free guidance training dropout",
    )

    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help=(
            "按(level,grade)平衡采样强度。"
            "0=不平衡；0.5=温和；1=近似完全平衡"
        ),
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--val-batches",
        type=int,
        default=64,
    )

    # ------------------------------------------------------------
    # Preview / sampling
    # ------------------------------------------------------------

    parser.add_argument(
        "--preview-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--preview-levels",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="默认预览L1-L2、L3-L4、L5-S1对应的level index",
    )

    parser.add_argument(
        "--preview-steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--preview-seed",
        type=int,
        default=3407,
    )

    parser.add_argument(
        "--sample-steps",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
    )

    # sample mode
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--target-grade",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--level-index",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser


def main() -> int:

    args = build_parser().parse_args()

    if args.mode == "train":
        return run_train(args)

    if args.mode == "sample":
        return run_sample(args)

    raise ValueError(
        args.mode
    )


if __name__ == "__main__":
    raise SystemExit(main())
