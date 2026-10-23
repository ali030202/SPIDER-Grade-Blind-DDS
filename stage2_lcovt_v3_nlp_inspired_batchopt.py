#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 Local Contrastive Ordinal Velocity Transport (LCOVT)
==========================================================

This is the single-file Stage-2 implementation for the clean VAE + Ordinal LDM.

Main V3 idea
------------
The inference design borrows the *representation steering* idea from NLP, where
contrastive hidden-state differences are used as inference-time control vectors
(e.g. Activation Addition / Contrastive Activation Addition), and the
*contrastive decoding* idea of comparing several conditioned experts.

Here the transferable idea is applied to diffusion velocity fields rather than
text activations:

    Delta v_k(z_t,t) = v(z_t,t,G_{k+1},L) - v(z_t,t,G_k,L)

For a continuous ordinal coordinate q in [1,5], only the two neighboring real
grade anchors are used:

    v_local = v_k + r * Delta v_k,  q = k + r

Thus no unseen fractional grade is fed into the Fourier conditioner in the main
method.  Long-range G1->G5 trajectories move piecewise through the learned
G1-G2-G3-G4-G5 anchor fields.

V3 additionally provides a self-calibrated inversion-depth option.  Candidate
counterfactual latents are probed under all five trained grade conditions using
the diffusion v-prediction error.  The smallest strength for which the target
grade is the best-fitting diffusion prior is selected.  This internal criterion
is for generation/calibration only; clinical evaluation should still use an
independent classifier and/or blinded readers.

No other Stage-2 Python module is required.

Supported modes
---------------
one              : one source image -> one target grade
all_grades       : one source image -> G1..G5
trajectory       : one source image -> continuous trajectory (ordinal / v-direction / latent baselines)
batch_all_grades : many images from prepared_metadata.csv -> each source -> G1..G5
batch_trajectory : many images from prepared_metadata.csv -> each source -> one continuous trajectory

Core method
-----------
1. Load frozen VAE used by the trained Ordinal LDM.
2. Encode source MRI to source latent z0.
3. DDIM inversion under source-grade condition to selected start timestep.
4. Reverse DDIM with directional v-space residual guidance:
       v_edit = v_source + lambda * (v_target - v_source)
   or for a trajectory:
       v_edit = v_source + alpha * lambda * (v_target - v_source)
5. Decode edited latent with the frozen VAE decoder.

Notes
-----
- This script is inference-only.
- It is compatible with the clean train_ordinal_latent_diffusion_v2.py runtime
  and its best.pt checkpoint.
- It does NOT retrain Stage-1.
- The network predicts velocity v, not epsilon noise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

SCRIPT_VERSION = "STAGE2_LCOVT_V3_NLP_INSPIRED_SELF_CALIBRATED_BATCHOPT_V1"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def setup_logger(output_dir: Path, filename: str = "stage2.log") -> logging.Logger:
    logger = logging.getLogger(f"stage2_all_in_one_{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / filename, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(payload, path: Path):
    def cv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    path.write_text(
        json.dumps(cv(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    array = (x.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def load_source_image(path: Path, image_height: int, image_width: int) -> torch.Tensor:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        im = im.convert("L")
        im = TF.resize(
            im,
            [int(image_height), int(image_width)],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    return (TF.pil_to_tensor(im).float() / 255.0).unsqueeze(0)


def resolve_image_path(raw_path: str, root: Path, metadata_path: Optional[Path] = None) -> Path:
    raw = str(raw_path).strip()
    p = Path(raw).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        root = Path(root).expanduser().resolve()
        candidates.extend([root / p, root / "data" / p])
        if metadata_path is not None:
            candidates.append(Path(metadata_path).expanduser().resolve().parent / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve() if candidates else p.resolve()


# ---------------------------------------------------------------------
# Stage-1 runtime loading
# ---------------------------------------------------------------------
def load_stage1_runtime(stage1_script: Path):
    stage1_script = Path(stage1_script).expanduser().resolve()
    if not stage1_script.exists():
        raise FileNotFoundError(
            f"Cannot find Stage-1 script: {stage1_script}\n"
            "Use the clean train_ordinal_latent_diffusion_v2.py."
        )

    project_root = stage1_script.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.spec_from_file_location(
        "ordinal_ldm_clean_v2_runtime_for_stage2_all_in_one", stage1_script
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Stage-1 script: {stage1_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_models(
    stage1_script: Path,
    checkpoint_path: Path,
    vae_checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
    allow_vae_mismatch: bool = False,
):
    stage1 = load_stage1_runtime(stage1_script)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    vae_checkpoint_path = Path(vae_checkpoint_path).expanduser().resolve()

    model, config, checkpoint = stage1.load_diffusion_checkpoint(checkpoint_path, device)

    expected_vae_hash = checkpoint.get("vae_sha256")
    actual_vae_hash = sha256_file(vae_checkpoint_path)
    if expected_vae_hash and expected_vae_hash != actual_vae_hash:
        msg = (
            "The VAE checkpoint does not match the VAE used when training the "
            "Ordinal LDM.\n"
            f"expected sha256={expected_vae_hash}\n"
            f"actual   sha256={actual_vae_hash}\n"
            f"VAE={vae_checkpoint_path}"
        )
        if not allow_vae_mismatch:
            raise RuntimeError(msg + "\nUse --allow-vae-mismatch only for an intentional ablation.")
        logger.warning(msg)

    vae = stage1.load_frozen_vae(config, vae_checkpoint_path, device, logger)
    schedule = stage1.DiffusionSchedule(config.diffusion_timesteps)
    latent_shape = stage1.infer_latent_shape(vae, config, device)
    model.eval()
    vae.eval()

    return {
        "stage1": stage1,
        "model": model,
        "vae": vae,
        "schedule": schedule,
        "config": config,
        "checkpoint": checkpoint,
        "latent_shape": latent_shape,
        "vae_sha256": actual_vae_hash,
    }


# ---------------------------------------------------------------------
# Diffusion helpers
# ---------------------------------------------------------------------
def build_timeline(total_timesteps: int, sampling_steps: int, device: torch.device) -> torch.Tensor:
    sampling_steps = max(2, int(sampling_steps))
    timeline = torch.linspace(0, int(total_timesteps) - 1, sampling_steps, device=device).round().long()
    timeline = torch.unique_consecutive(timeline)
    if int(timeline[0].item()) != 0:
        timeline = torch.cat([torch.zeros(1, device=device, dtype=torch.long), timeline], dim=0)
    return timeline


@torch.no_grad()
def predict_v(
    model,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    grade: torch.Tensor,
    level: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Predict v with classifier-free guidance.

    Optimized implementation: when CFG is enabled, conditional and unconditional
    branches are concatenated into ONE model forward instead of two serial forwards.
    The mathematical expression is unchanged:

        v = v_uncond + s * (v_cond - v_uncond)

    This is especially useful on large GPUs where batch=1 leaves most compute idle.
    """
    batch = int(latent.shape[0])
    drop_none = torch.zeros(batch, device=latent.device, dtype=torch.bool)

    if float(guidance_scale) == 1.0:
        return model(latent, timestep, grade, level, drop_mask=drop_none)

    latent_b = torch.cat([latent, latent], dim=0)
    timestep_b = torch.cat([timestep, timestep], dim=0)
    grade_b = torch.cat([grade, grade], dim=0)
    level_b = torch.cat([level, level], dim=0)
    drop_b = torch.cat([
        drop_none,
        torch.ones(batch, device=latent.device, dtype=torch.bool),
    ], dim=0)

    v_b = model(latent_b, timestep_b, grade_b, level_b, drop_mask=drop_b)
    v_cond, v_uncond = v_b.split(batch, dim=0)
    return v_uncond + float(guidance_scale) * (v_cond - v_uncond)


def resolve_strength(
    source_grade: int,
    target_grade: int,
    *,
    adaptive_strength: bool,
    fixed_strength: float,
    adjacent_strength: float,
    strength_per_extra_grade: float,
    max_strength: float,
) -> float:
    delta = abs(int(target_grade) - int(source_grade))
    if delta == 0:
        return 0.0
    if not adaptive_strength:
        return float(np.clip(fixed_strength, 0.0, 1.0))
    s = float(adjacent_strength) + max(delta - 1, 0) * float(strength_per_extra_grade)
    return float(np.clip(s, 0.0, float(max_strength)))


def _time_lambda_weight(current_t: int, start_t: int, schedule: str, min_ratio: float, power: float) -> float:
    schedule = str(schedule).lower()
    if start_t <= 0 or schedule == "constant":
        return 1.0
    r = float(np.clip(float(current_t) / float(start_t), 0.0, 1.0))
    min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
    if schedule == "cosine":
        shaped = 0.5 - 0.5 * math.cos(math.pi * r)
    elif schedule == "power":
        shaped = r ** max(float(power), 1e-6)
    else:
        raise ValueError(f"Unknown lambda schedule: {schedule}")
    return min_ratio + (1.0 - min_ratio) * shaped


def effective_lambda(
    *,
    base_scale: float,
    trajectory_alpha: float,
    source_grade: int,
    target_grade: int,
    current_t: int,
    start_t: int,
    lambda_schedule: str,
    lambda_min_ratio: float,
    lambda_time_power: float,
    grade_distance_gain: float,
) -> float:
    delta = abs(int(target_grade) - int(source_grade))
    grade_factor = 1.0 + max(delta - 1, 0) * float(grade_distance_gain)
    time_factor = _time_lambda_weight(
        current_t, start_t, lambda_schedule, lambda_min_ratio, lambda_time_power
    )
    return (
        max(float(base_scale), 0.0)
        * float(np.clip(trajectory_alpha, 0.0, 1.0))
        * max(grade_factor, 0.0)
        * time_factor
    )


@torch.no_grad()
def ddim_invert_to_strength(
    stage1,
    model,
    schedule,
    source_latent: torch.Tensor,
    source_grade: torch.Tensor,
    level: torch.Tensor,
    *,
    sampling_steps: int,
    strength: float,
    inversion_guidance_scale: float = 1.0,
    x0_clip: float = 4.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    strength = float(np.clip(strength, 0.0, 1.0))
    device = source_latent.device
    timeline = build_timeline(schedule.timesteps, sampling_steps, device)
    if strength <= 0.0:
        return source_latent.clone(), timeline[:1].clone()

    pivot = max(1, int(round((len(timeline) - 1) * strength)))
    used = timeline[: pivot + 1].clone()
    value = source_latent.clone()

    for i in range(len(used) - 1):
        t_scalar = used[i]
        next_scalar = used[i + 1]
        t = torch.full((value.shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
        t_next = torch.full((value.shape[0],), int(next_scalar.item()), device=device, dtype=torch.long)
        v = predict_v(
            model, value, t, source_grade, level,
            guidance_scale=inversion_guidance_scale,
        )
        x0 = schedule.predict_x0_from_v(value, v, t).clamp(-float(x0_clip), float(x0_clip))
        eps = schedule.predict_noise_from_v(value, v, t)
        a_next = stage1.extract(schedule.sqrt_alpha_bar, t_next, value.shape)
        s_next = stage1.extract(schedule.sqrt_one_minus_alpha_bar, t_next, value.shape)
        value = a_next * x0 + s_next * eps

    return value, used


@torch.no_grad()
def ddim_directional_edit(
    stage1,
    model,
    schedule,
    start_latent: torch.Tensor,
    source_latent: torch.Tensor,
    source_grade: torch.Tensor,
    target_grade: torch.Tensor,
    level: torch.Tensor,
    inversion_timeline: torch.Tensor,
    *,
    directional_scale: float = 1.0,
    trajectory_alpha: float = 1.0,
    source_guidance_scale: float = 1.0,
    target_guidance_scale: float = 1.0,
    lambda_schedule: str = "constant",
    lambda_min_ratio: float = 0.35,
    lambda_time_power: float = 1.0,
    grade_distance_gain: float = 0.0,
    late_identity_blend: float = 0.0,
    late_identity_steps: int = 6,
    x0_clip: float = 4.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = start_latent.device
    timeline = torch.flip(inversion_timeline, dims=[0])
    value = start_latent.clone()
    start_t = int(timeline[0].item()) if len(timeline) else 0
    src_g = int(source_grade[0].item())
    tgt_g = int(target_grade[0].item())
    lambdas = []

    for i, t_scalar in enumerate(timeline):
        t_int = int(t_scalar.item())
        t = torch.full((value.shape[0],), t_int, device=device, dtype=torch.long)

        v_source = predict_v(
            model, value, t, source_grade, level,
            guidance_scale=source_guidance_scale,
        )
        v_target = predict_v(
            model, value, t, target_grade, level,
            guidance_scale=target_guidance_scale,
        )

        lam = effective_lambda(
            base_scale=directional_scale,
            trajectory_alpha=trajectory_alpha,
            source_grade=src_g,
            target_grade=tgt_g,
            current_t=t_int,
            start_t=start_t,
            lambda_schedule=lambda_schedule,
            lambda_min_ratio=lambda_min_ratio,
            lambda_time_power=lambda_time_power,
            grade_distance_gain=grade_distance_gain,
        )
        lambdas.append(lam)
        v_edit = v_source + lam * (v_target - v_source)

        x0 = schedule.predict_x0_from_v(value, v_edit, t).clamp(-float(x0_clip), float(x0_clip))

        if late_identity_blend > 0.0 and late_identity_steps > 0:
            remaining = len(timeline) - i
            if remaining <= int(late_identity_steps):
                blend = float(np.clip(late_identity_blend, 0.0, 0.5))
                x0 = (1.0 - blend) * x0 + blend * source_latent

        if i == len(timeline) - 1:
            value = x0
            break

        next_scalar = timeline[i + 1]
        t_next = torch.full((value.shape[0],), int(next_scalar.item()), device=device, dtype=torch.long)
        eps = schedule.predict_noise_from_v(value, v_edit, t)
        a_next = stage1.extract(schedule.sqrt_alpha_bar, t_next, value.shape)
        s_next = stage1.extract(schedule.sqrt_one_minus_alpha_bar, t_next, value.shape)
        value = a_next * x0 + s_next * eps

    stats = {
        "lambda_min": float(min(lambdas)) if lambdas else 0.0,
        "lambda_max": float(max(lambdas)) if lambdas else 0.0,
        "lambda_mean": float(np.mean(lambdas)) if lambdas else 0.0,
        "start_timestep": int(start_t),
    }
    return value, stats


@torch.no_grad()
def ddim_conditioned_edit(
    stage1,
    model,
    schedule,
    start_latent: torch.Tensor,
    source_latent: torch.Tensor,
    grade_condition: torch.Tensor,
    level: torch.Tensor,
    inversion_timeline: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    late_identity_blend: float = 0.0,
    late_identity_steps: int = 6,
    x0_clip: float = 4.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reverse DDIM using ONE continuous ordinal grade condition.

    Important:
      `grade_condition` may be a floating-point grade coordinate such as 2.35.
      The clean Ordinal LDM conditioner maps grade -> continuous severity with
      grade.float(), so fractional grades are valid interpolation coordinates.

    This is the preferred trajectory primitive:
        g_alpha = (1-alpha) * G_source + alpha * G_target
        v_alpha = f_theta(z_t, t, g_alpha, level)
    """
    device = start_latent.device
    timeline = torch.flip(inversion_timeline, dims=[0])
    value = start_latent.clone()
    start_t = int(timeline[0].item()) if len(timeline) else 0

    for i, t_scalar in enumerate(timeline):
        t_int = int(t_scalar.item())
        t = torch.full((value.shape[0],), t_int, device=device, dtype=torch.long)
        v = predict_v(
            model,
            value,
            t,
            grade_condition,
            level,
            guidance_scale=guidance_scale,
        )
        x0 = schedule.predict_x0_from_v(value, v, t).clamp(
            -float(x0_clip), float(x0_clip)
        )

        if late_identity_blend > 0.0 and late_identity_steps > 0:
            remaining = len(timeline) - i
            if remaining <= int(late_identity_steps):
                blend = float(np.clip(late_identity_blend, 0.0, 0.5))
                x0 = (1.0 - blend) * x0 + blend * source_latent

        if i == len(timeline) - 1:
            value = x0
            break

        next_scalar = timeline[i + 1]
        t_next = torch.full(
            (value.shape[0],), int(next_scalar.item()), device=device, dtype=torch.long
        )
        eps = schedule.predict_noise_from_v(value, v, t)
        a_next = stage1.extract(schedule.sqrt_alpha_bar, t_next, value.shape)
        s_next = stage1.extract(schedule.sqrt_one_minus_alpha_bar, t_next, value.shape)
        value = a_next * x0 + s_next * eps

    stats = {
        "start_timestep": int(start_t),
        "lambda_min": 0.0,
        "lambda_max": 0.0,
        "lambda_mean": 0.0,
    }
    return value, stats



# ---------------------------------------------------------------------
# V3: NLP-inspired local contrastive ordinal velocity transport
# ---------------------------------------------------------------------
def continuous_grade_to_severity(grade_value: float, num_grades: int = 5) -> float:
    center = (float(num_grades) + 1.0) / 2.0
    half_range = max((float(num_grades) - 1.0) / 2.0, 1.0)
    return (float(grade_value) - center) / half_range


def local_grade_anchors(continuous_grade: float, num_grades: int = 5) -> Tuple[int, int, float]:
    """Return trained neighboring grade anchors (left, right, mix r)."""
    q = float(np.clip(continuous_grade, 1.0, float(num_grades)))
    left = int(math.floor(q + 1e-10))
    right = int(math.ceil(q - 1e-10))
    left = max(1, min(int(num_grades), left))
    right = max(1, min(int(num_grades), right))
    if left == right:
        return left, right, 0.0
    r = float(np.clip(q - float(left), 0.0, 1.0))
    return left, right, r


@torch.no_grad()
def predict_local_ordinal_v(
    model,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    continuous_grade: float,
    level: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    num_grades: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Piecewise local contrastive steering using ONLY trained integer grade anchors.

    This is analogous in spirit to NLP activation steering / CAA:
        steering direction = representation(positive) - representation(negative)
    Here:
        local direction = v(G_{k+1}) - v(G_k)
        v(q) = v(G_k) + r * local direction

    Importantly, the U-Net never receives an unseen fractional grade in this mode.
    """
    left, right, r = local_grade_anchors(continuous_grade, num_grades=num_grades)
    batch = latent.shape[0]
    left_t = torch.full((batch,), left, device=latent.device, dtype=torch.long)
    v_left = predict_v(model, latent, timestep, left_t, level, guidance_scale=guidance_scale)
    if left == right or r <= 1e-12:
        return v_left, {"anchor_left": float(left), "anchor_right": float(right), "anchor_mix": 0.0}
    right_t = torch.full((batch,), right, device=latent.device, dtype=torch.long)
    v_right = predict_v(model, latent, timestep, right_t, level, guidance_scale=guidance_scale)
    v_local = v_left + float(r) * (v_right - v_left)
    return v_local, {"anchor_left": float(left), "anchor_right": float(right), "anchor_mix": float(r)}


@torch.no_grad()
def ddim_local_ordinal_edit(
    stage1,
    model,
    schedule,
    start_latent: torch.Tensor,
    source_latent: torch.Tensor,
    continuous_grade: float,
    level: torch.Tensor,
    inversion_timeline: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    num_grades: int = 5,
    late_identity_blend: float = 0.0,
    late_identity_steps: int = 6,
    x0_clip: float = 4.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reverse DDIM through the local piecewise ordinal velocity field."""
    device = start_latent.device
    timeline = torch.flip(inversion_timeline, dims=[0])
    value = start_latent.clone()
    start_t = int(timeline[0].item()) if len(timeline) else 0
    anchor_info = local_grade_anchors(continuous_grade, num_grades=num_grades)

    for i, t_scalar in enumerate(timeline):
        t_int = int(t_scalar.item())
        t = torch.full((value.shape[0],), t_int, device=device, dtype=torch.long)
        v_local, _ = predict_local_ordinal_v(
            model,
            value,
            t,
            continuous_grade,
            level,
            guidance_scale=guidance_scale,
            num_grades=num_grades,
        )
        x0 = schedule.predict_x0_from_v(value, v_local, t).clamp(-float(x0_clip), float(x0_clip))

        if late_identity_blend > 0.0 and late_identity_steps > 0:
            remaining = len(timeline) - i
            if remaining <= int(late_identity_steps):
                blend = float(np.clip(late_identity_blend, 0.0, 0.5))
                x0 = (1.0 - blend) * x0 + blend * source_latent

        if i == len(timeline) - 1:
            value = x0
            break

        next_scalar = timeline[i + 1]
        t_next = torch.full((value.shape[0],), int(next_scalar.item()), device=device, dtype=torch.long)
        eps = schedule.predict_noise_from_v(value, v_local, t)
        a_next = stage1.extract(schedule.sqrt_alpha_bar, t_next, value.shape)
        s_next = stage1.extract(schedule.sqrt_one_minus_alpha_bar, t_next, value.shape)
        value = a_next * x0 + s_next * eps

    left, right, mix = anchor_info
    stats = {
        "start_timestep": int(start_t),
        "anchor_left": float(left),
        "anchor_right": float(right),
        "anchor_mix": float(mix),
        "lambda_min": 0.0,
        "lambda_max": 0.0,
        "lambda_mean": 0.0,
    }
    return value, stats



def local_grade_anchors_tensor(
    continuous_grades: torch.Tensor,
    num_grades: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized counterpart of :func:`local_grade_anchors`.

    Parameters
    ----------
    continuous_grades:
        1-D tensor containing q values, e.g. [1.0, 1.2, 1.4, 1.6].

    Returns
    -------
    left, right, mix:
        Integer trained anchors and local interpolation coefficient for every q.
    """
    q = continuous_grades.float().clamp(1.0, float(num_grades))
    left = torch.floor(q + 1e-7).long().clamp(1, int(num_grades))
    right = torch.ceil(q - 1e-7).long().clamp(1, int(num_grades))
    mix = (q - left.float()).clamp(0.0, 1.0)
    mix = torch.where(left == right, torch.zeros_like(mix), mix)
    return left, right, mix


@torch.no_grad()
def predict_local_ordinal_v_batch(
    model,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    continuous_grades: torch.Tensor,
    level: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    num_grades: int = 5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Vectorized LCOVT velocity prediction for many q values.

    For B trajectory points, both neighboring integer anchors are packed together:

        [G_left(q_1), ..., G_left(q_B), G_right(q_1), ..., G_right(q_B)]

    `predict_v` then packs conditional/unconditional CFG branches into the same
    forward.  Therefore a q-batch of 4 with CFG uses one model call whose effective
    model batch is 4 x 2 anchors x 2 CFG branches = 16.

    The LCOVT equation is unchanged:

        v(q) = v_left + r * (v_right - v_left)
    """
    if latent.ndim < 2:
        raise ValueError("latent must have a batch dimension")
    batch = int(latent.shape[0])
    q = torch.as_tensor(continuous_grades, device=latent.device, dtype=torch.float32).reshape(-1)
    if q.numel() != batch:
        raise ValueError(f"continuous_grades has {q.numel()} values but latent batch is {batch}")
    if timestep.numel() != batch:
        raise ValueError(f"timestep has {timestep.numel()} values but latent batch is {batch}")
    if level.numel() != batch:
        raise ValueError(f"level has {level.numel()} values but latent batch is {batch}")

    left, right, mix = local_grade_anchors_tensor(q, num_grades=num_grades)

    # Pack left/right trained grade conditions into one condition batch.
    latent_lr = torch.cat([latent, latent], dim=0)
    timestep_lr = torch.cat([timestep, timestep], dim=0)
    grade_lr = torch.cat([left, right], dim=0)
    level_lr = torch.cat([level, level], dim=0)

    # predict_v itself fuses conditional/unconditional CFG into one model call.
    v_lr = predict_v(
        model,
        latent_lr,
        timestep_lr,
        grade_lr,
        level_lr,
        guidance_scale=guidance_scale,
    )
    v_left, v_right = v_lr.split(batch, dim=0)

    mix_view = mix.to(dtype=v_left.dtype).view(batch, *([1] * (latent.ndim - 1)))
    v_local = v_left + mix_view * (v_right - v_left)
    return v_local, {
        "anchor_left": left,
        "anchor_right": right,
        "anchor_mix": mix,
    }


@torch.no_grad()
def ddim_local_ordinal_edit_batch(
    stage1,
    model,
    schedule,
    start_latent: torch.Tensor,
    source_latent: torch.Tensor,
    continuous_grades,
    level: torch.Tensor,
    inversion_timeline: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    num_grades: int = 5,
    late_identity_blend: float = 0.0,
    late_identity_steps: int = 6,
    x0_clip: float = 4.5,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Batched reverse DDIM through the same LCOVT field used by the scalar method.

    This function changes execution only, not the inference definition.  A single
    source inversion is replicated across a small batch of continuous q values and
    all q trajectories are advanced through every reverse timestep together.

    It is intended for high-throughput inference on GPUs such as RTX 4090/4090D.
    """
    device = start_latent.device
    q = torch.as_tensor(continuous_grades, device=device, dtype=torch.float32).reshape(-1)
    batch = int(q.numel())
    if batch <= 0:
        raise ValueError("continuous_grades must contain at least one q value")

    def _expand_first_dim(x: torch.Tensor, name: str) -> torch.Tensor:
        if int(x.shape[0]) == batch:
            return x.clone()
        if int(x.shape[0]) == 1:
            return x.expand(batch, *x.shape[1:]).clone()
        raise ValueError(f"{name} batch must be 1 or {batch}, got {x.shape[0]}")

    value = _expand_first_dim(start_latent, "start_latent")
    source_batch = _expand_first_dim(source_latent, "source_latent")

    level = level.reshape(-1)
    if level.numel() == 1:
        level_batch = level.expand(batch).clone()
    elif level.numel() == batch:
        level_batch = level.clone()
    else:
        raise ValueError(f"level must contain 1 or {batch} entries, got {level.numel()}")

    timeline = torch.flip(inversion_timeline, dims=[0])
    start_t = int(timeline[0].item()) if len(timeline) else 0
    left, right, mix = local_grade_anchors_tensor(q, num_grades=num_grades)

    for i, t_scalar in enumerate(timeline):
        t_int = int(t_scalar.item())
        t = torch.full((batch,), t_int, device=device, dtype=torch.long)
        v_local, _ = predict_local_ordinal_v_batch(
            model,
            value,
            t,
            q,
            level_batch,
            guidance_scale=guidance_scale,
            num_grades=num_grades,
        )
        x0 = schedule.predict_x0_from_v(value, v_local, t).clamp(
            -float(x0_clip), float(x0_clip)
        )

        if late_identity_blend > 0.0 and late_identity_steps > 0:
            remaining = len(timeline) - i
            if remaining <= int(late_identity_steps):
                blend = float(np.clip(late_identity_blend, 0.0, 0.5))
                x0 = (1.0 - blend) * x0 + blend * source_batch

        if i == len(timeline) - 1:
            value = x0
            break

        next_scalar = timeline[i + 1]
        t_next = torch.full(
            (batch,), int(next_scalar.item()), device=device, dtype=torch.long
        )
        eps = schedule.predict_noise_from_v(value, v_local, t)
        a_next = stage1.extract(schedule.sqrt_alpha_bar, t_next, value.shape)
        s_next = stage1.extract(schedule.sqrt_one_minus_alpha_bar, t_next, value.shape)
        value = a_next * x0 + s_next * eps

    left_cpu = left.detach().cpu().tolist()
    right_cpu = right.detach().cpu().tolist()
    mix_cpu = mix.detach().cpu().tolist()
    stats: List[Dict[str, float]] = []
    for j in range(batch):
        stats.append({
            "start_timestep": int(start_t),
            "anchor_left": float(left_cpu[j]),
            "anchor_right": float(right_cpu[j]),
            "anchor_mix": float(mix_cpu[j]),
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "lambda_mean": 0.0,
        })
    return value, stats


@torch.no_grad()
def diffusion_prior_profile(
    runtime: Dict,
    latent: torch.Tensor,
    level_index: int,
    *,
    probe_count: int = 4,
    probe_min_frac: float = 0.15,
    probe_max_frac: float = 0.75,
    probe_seed: int = 3407,
    guidance_scale: float = 1.0,
    min_snr_gamma: float = 5.0,
) -> Dict[str, object]:
    """
    Contrastive diffusion-prior probe over G1..G5.

    For the candidate z0, add known noise at several timesteps, construct the
    exact v-target, then ask the same Ordinal LDM to predict v under each grade.
    Lower error means the latent is more compatible with that conditioned prior.

    This is intentionally an INTERNAL calibration signal, not an independent
    evaluation metric.
    """
    model = runtime["model"]
    schedule = runtime["schedule"]
    total = int(schedule.timesteps)
    count = max(1, int(probe_count))
    lo = int(round(np.clip(probe_min_frac, 0.0, 1.0) * (total - 1)))
    hi = int(round(np.clip(probe_max_frac, 0.0, 1.0) * (total - 1)))
    if hi < lo:
        lo, hi = hi, lo
    probe_ts = np.linspace(lo, hi, count).round().astype(int).tolist()

    generator = torch.Generator(device=latent.device)
    generator.manual_seed(int(probe_seed))
    level = torch.full((latent.shape[0],), int(level_index), device=latent.device, dtype=torch.long)

    errors = {g: [] for g in range(1, 6)}
    for t_int in probe_ts:
        t = torch.full((latent.shape[0],), int(t_int), device=latent.device, dtype=torch.long)
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        noisy = schedule.q_sample(latent, t, noise)
        target_v = schedule.v_target(latent, noise, t)
        if hasattr(schedule, "min_snr_weight"):
            w = schedule.min_snr_weight(t, float(min_snr_gamma)).view(-1)
        else:
            w = torch.ones(latent.shape[0], device=latent.device, dtype=latent.dtype)

        for g in range(1, 6):
            grade = torch.full((latent.shape[0],), g, device=latent.device, dtype=torch.long)
            pred = predict_v(model, noisy, t, grade, level, guidance_scale=guidance_scale)
            per_sample = (pred - target_v).square().flatten(1).mean(dim=1)
            errors[g].append(float((per_sample * w).mean().item()))

    mean_errors = {g: float(np.mean(v)) for g, v in errors.items()}
    ranked = sorted(mean_errors.items(), key=lambda kv: kv[1])
    best_grade, best_error = ranked[0]
    second_grade, second_error = ranked[1]
    return {
        "errors": mean_errors,
        "best_grade": int(best_grade),
        "best_error": float(best_error),
        "second_grade": int(second_grade),
        "second_error": float(second_error),
        "best_margin": float(second_error - best_error),
        "probe_timesteps": probe_ts,
    }


def parse_strength_grid(text: str) -> List[float]:
    values = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(np.clip(float(token), 0.0, 1.0)))
    values = sorted(set(round(v, 8) for v in values))
    if not values:
        raise ValueError("strength grid is empty")
    return values


@torch.no_grad()
def make_target_endpoint(
    runtime: Dict,
    source_latent: torch.Tensor,
    source_grade: int,
    target_grade: int,
    level_index: int,
    strength: float,
    args,
) -> Dict[str, object]:
    """Generate an integer target-grade endpoint for one inversion strength."""
    model = runtime["model"]
    vae = runtime["vae"]
    schedule = runtime["schedule"]
    stage1 = runtime["stage1"]
    device = source_latent.device
    sg = torch.full((source_latent.shape[0],), int(source_grade), device=device, dtype=torch.long)
    level = torch.full((source_latent.shape[0],), int(level_index), device=device, dtype=torch.long)
    tg = torch.full((source_latent.shape[0],), int(target_grade), device=device, dtype=torch.long)

    if int(source_grade) == int(target_grade) or float(strength) <= 0.0:
        return {
            "start_latent": source_latent.clone(),
            "timeline": torch.zeros(1, device=device, dtype=torch.long),
            "edited_latent": source_latent.clone(),
            "stats": {"start_timestep": 0},
        }

    start_latent, timeline = ddim_invert_to_strength(
        stage1,
        model,
        schedule,
        source_latent,
        sg,
        level,
        sampling_steps=args.sampling_steps,
        strength=float(strength),
        inversion_guidance_scale=args.inversion_guidance_scale,
        x0_clip=args.x0_clip,
    )
    # Integer target anchor. LCOVT and pure target conditioning coincide here.
    edited_latent, stats = ddim_local_ordinal_edit(
        stage1,
        model,
        schedule,
        start_latent,
        source_latent,
        float(target_grade),
        level,
        timeline,
        guidance_scale=args.ordinal_guidance_scale,
        num_grades=int(runtime["config"].num_grades),
        late_identity_blend=args.late_identity_blend,
        late_identity_steps=args.late_identity_steps,
        x0_clip=args.x0_clip,
    )
    return {
        "start_latent": start_latent,
        "timeline": timeline,
        "edited_latent": edited_latent,
        "stats": stats,
    }


@torch.no_grad()
def self_calibrate_strength(
    runtime: Dict,
    source_latent: torch.Tensor,
    source_grade: int,
    target_grade: int,
    level_index: int,
    args,
) -> Tuple[float, pd.DataFrame, Dict[str, object]]:
    """
    Select the MINIMUM inversion strength whose candidate is best explained by
    the target grade-conditioned diffusion prior.

    Selection rule:
      1) target grade has the lowest v-prediction error among G1..G5;
      2) target-vs-runner-up margin >= --prior-margin;
      3) target improves over the source-grade error by the requested relative
         margin (--prior-relative-gain).

    If no candidate passes, select the candidate with the strongest target
    contrastive evidence and report fallback=True.
    """
    if int(source_grade) == int(target_grade):
        return 0.0, pd.DataFrame(), {"fallback": False, "reason": "same_grade"}

    strengths = parse_strength_grid(args.calibration_strength_grid)
    records = []
    candidate_cache: Dict[float, Dict[str, object]] = {}
    chosen = None

    for idx, strength in enumerate(strengths):
        endpoint = make_target_endpoint(
            runtime,
            source_latent,
            source_grade,
            target_grade,
            level_index,
            strength,
            args,
        )
        candidate_cache[strength] = endpoint
        profile = diffusion_prior_profile(
            runtime,
            endpoint["edited_latent"],
            level_index,
            probe_count=args.prior_probe_count,
            probe_min_frac=args.prior_probe_min_frac,
            probe_max_frac=args.prior_probe_max_frac,
            probe_seed=args.prior_probe_seed,
            guidance_scale=args.prior_guidance_scale,
            min_snr_gamma=args.prior_min_snr_gamma,
        )
        errs = profile["errors"]
        target_error = float(errs[int(target_grade)])
        source_error = float(errs[int(source_grade)])
        competitors = [float(v) for g, v in errs.items() if int(g) != int(target_grade)]
        second_best = min(competitors)
        target_margin = float(second_best - target_error)
        relative_gain = float((source_error - target_error) / max(abs(source_error), 1e-12))
        latent_l1 = float((endpoint["edited_latent"] - source_latent).abs().mean().item())
        accepted = (
            int(profile["best_grade"]) == int(target_grade)
            and target_margin >= float(args.prior_margin)
            and relative_gain >= float(args.prior_relative_gain)
        )
        rec = {
            "candidate_index": idx,
            "strength": float(strength),
            "accepted": bool(accepted),
            "prior_best_grade": int(profile["best_grade"]),
            "target_error": target_error,
            "source_error": source_error,
            "target_margin": target_margin,
            "relative_target_gain_vs_source": relative_gain,
            "latent_l1_to_source": latent_l1,
            "start_timestep": int(endpoint["stats"].get("start_timestep", 0)),
        }
        for g in range(1, 6):
            rec[f"prior_error_G{g}"] = float(errs[g])
        records.append(rec)
        if accepted and chosen is None:
            chosen = float(strength)
            break

    fallback = False
    reason = "first_minimum_accepted"
    if chosen is None:
        fallback = True
        reason = "no_candidate_passed; chose strongest contrastive target evidence"
        # Prefer correct best-grade candidates, then larger target margin, then smaller strength.
        ranked = sorted(
            records,
            key=lambda r: (
                int(r["prior_best_grade"] == int(target_grade)),
                float(r["target_margin"]),
                float(r["relative_target_gain_vs_source"]),
                -float(r["strength"]),
            ),
            reverse=True,
        )
        chosen = float(ranked[0]["strength"])

    table = pd.DataFrame(records)
    meta = {"fallback": fallback, "reason": reason, "chosen_strength": chosen}
    return chosen, table, meta


def choose_strength(
    runtime: Dict,
    source_latent: torch.Tensor,
    source_grade: int,
    target_grade: int,
    level_index: int,
    args,
) -> Tuple[float, Optional[pd.DataFrame], Dict[str, object]]:
    mode = str(args.strength_mode)
    if int(source_grade) == int(target_grade):
        return 0.0, None, {"mode": mode, "same_grade": True}
    if mode == "fixed":
        return float(np.clip(args.strength, 0.0, 1.0)), None, {"mode": mode}
    if mode == "delta":
        s = resolve_strength(
            source_grade,
            target_grade,
            adaptive_strength=True,
            fixed_strength=args.strength,
            adjacent_strength=args.adjacent_strength,
            strength_per_extra_grade=args.strength_per_extra_grade,
            max_strength=args.max_strength,
        )
        return s, None, {"mode": mode}
    if mode == "self_calibrated":
        s, table, meta = self_calibrate_strength(
            runtime, source_latent, source_grade, target_grade, level_index, args
        )
        meta["mode"] = mode
        return s, table, meta
    raise ValueError(f"Unknown strength_mode={mode}")


def arc_length_indices(latents: List[torch.Tensor], desired_steps: int) -> Tuple[List[int], List[float]]:
    """Pick approximately equal latent arc-length positions from a dense path."""
    n = len(latents)
    desired_steps = max(2, min(int(desired_steps), n))
    if n <= desired_steps:
        cum = [0.0]
        for i in range(1, n):
            d = float((latents[i] - latents[i - 1]).square().mean().sqrt().item())
            cum.append(cum[-1] + d)
        total = max(cum[-1], 1e-12)
        return list(range(n)), [x / total for x in cum]

    cumulative = [0.0]
    for i in range(1, n):
        d = float((latents[i] - latents[i - 1]).square().mean().sqrt().item())
        cumulative.append(cumulative[-1] + d)
    total = cumulative[-1]
    if total <= 1e-12:
        indices = np.linspace(0, n - 1, desired_steps).round().astype(int).tolist()
        return indices, np.linspace(0.0, 1.0, desired_steps).tolist()

    targets = np.linspace(0.0, total, desired_steps)
    indices = []
    arr = np.asarray(cumulative)
    for target in targets:
        idx = int(np.argmin(np.abs(arr - target)))
        if indices and idx <= indices[-1]:
            idx = min(n - 1, indices[-1] + 1)
        indices.append(idx)
    indices[-1] = n - 1
    # In rare cases strict monotonicity near the end can duplicate; repair backwards.
    for i in range(len(indices) - 2, -1, -1):
        if indices[i] >= indices[i + 1]:
            indices[i] = max(0, indices[i + 1] - 1)
    positions = [float(cumulative[i] / total) for i in indices]
    return indices, positions

def latent_lerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * z0 + a * z1


def latent_slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float, eps: float = 1e-7) -> torch.Tensor:
    """Stable SLERP in the flattened latent space, with interpolated magnitude."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return z0.clone()
    if a >= 1.0:
        return z1.clone()

    b = z0.shape[0]
    x0 = z0.reshape(b, -1)
    x1 = z1.reshape(b, -1)
    n0 = x0.norm(dim=1, keepdim=True).clamp_min(eps)
    n1 = x1.norm(dim=1, keepdim=True).clamp_min(eps)
    u0 = x0 / n0
    u1 = x1 / n1
    dot = (u0 * u1).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Near-collinear vectors: fall back to normalized linear interpolation.
    near = sin_theta.abs() < 1e-4
    w0 = torch.sin((1.0 - a) * theta) / sin_theta.clamp_min(eps)
    w1 = torch.sin(a * theta) / sin_theta.clamp_min(eps)
    direction = w0 * u0 + w1 * u1
    linear_direction = (1.0 - a) * u0 + a * u1
    direction = torch.where(near, linear_direction, direction)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(eps)

    magnitude = (1.0 - a) * n0 + a * n1
    return (direction * magnitude).reshape_as(z0)


@torch.no_grad()
def edit_one(
    runtime: Dict,
    source_image: torch.Tensor,
    source_grade_value: int,
    target_grade_value: int,
    level_index_value: int,
    *,
    sampling_steps: int,
    strength: float,
    inversion_guidance_scale: float = 1.0,
    source_guidance_scale: float = 1.0,
    target_guidance_scale: float = 1.0,
    directional_scale: float = 1.0,
    trajectory_alpha: float = 1.0,
    lambda_schedule: str = "constant",
    lambda_min_ratio: float = 0.35,
    lambda_time_power: float = 1.0,
    grade_distance_gain: float = 0.0,
    late_identity_blend: float = 0.0,
    late_identity_steps: int = 6,
    x0_clip: float = 4.5,
):
    model = runtime["model"]
    vae = runtime["vae"]
    schedule = runtime["schedule"]
    stage1 = runtime["stage1"]

    source_latent, _ = vae.encode(source_image, sample=False)
    vae_recon = vae.decode(source_latent).clamp(0.0, 1.0)
    device = source_image.device

    source_grade = torch.tensor([int(source_grade_value)], device=device, dtype=torch.long)
    target_grade = torch.tensor([int(target_grade_value)], device=device, dtype=torch.long)
    level = torch.tensor([int(level_index_value)], device=device, dtype=torch.long)

    if int(source_grade_value) == int(target_grade_value) or float(strength) <= 0.0:
        start_latent = source_latent.clone()
        edited_latent = source_latent.clone()
        edited_image = source_image.clone()
        timeline = torch.zeros(1, device=device, dtype=torch.long)
        lambda_stats = {
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "lambda_mean": 0.0,
            "start_timestep": 0,
        }
    else:
        start_latent, timeline = ddim_invert_to_strength(
            stage1,
            model,
            schedule,
            source_latent,
            source_grade,
            level,
            sampling_steps=sampling_steps,
            strength=strength,
            inversion_guidance_scale=inversion_guidance_scale,
            x0_clip=x0_clip,
        )
        edited_latent, lambda_stats = ddim_directional_edit(
            stage1,
            model,
            schedule,
            start_latent,
            source_latent,
            source_grade,
            target_grade,
            level,
            timeline,
            directional_scale=directional_scale,
            trajectory_alpha=trajectory_alpha,
            source_guidance_scale=source_guidance_scale,
            target_guidance_scale=target_guidance_scale,
            lambda_schedule=lambda_schedule,
            lambda_min_ratio=lambda_min_ratio,
            lambda_time_power=lambda_time_power,
            grade_distance_gain=grade_distance_gain,
            late_identity_blend=late_identity_blend,
            late_identity_steps=late_identity_steps,
            x0_clip=x0_clip,
        )
        edited_image = vae.decode(edited_latent).clamp(0.0, 1.0)

    return {
        "source_latent": source_latent,
        "vae_recon": vae_recon,
        "start_latent": start_latent,
        "edited_latent": edited_latent,
        "edited_image": edited_image,
        "used_timeline": timeline,
        "lambda_stats": lambda_stats,
    }


# ---------------------------------------------------------------------
# Visualization and metrics
# ---------------------------------------------------------------------
def draw_label(canvas: Image.Image, text: str, x: int, y: int):
    ImageDraw.Draw(canvas).text((x + 4, y + 4), text, fill=255)


def save_all_grades_grid(
    source,
    recon,
    outputs: Dict[int, torch.Tensor],
    strengths: Dict[int, float],
    source_grade: int,
    level: int,
    path: Path,
):
    cells = [(f"SRC G{source_grade}", tensor_to_pil(source)), ("VAE recon", tensor_to_pil(recon))]
    for g in range(1, 6):
        cells.append((f"G{g} s={strengths[g]:.2f}", tensor_to_pil(outputs[g])))
    w, h = cells[0][1].size
    lh = 26
    canvas = Image.new("L", (w * len(cells), h + lh), color=0)
    for i, (label, image) in enumerate(cells):
        x = i * w
        canvas.paste(image, (x, 0))
        draw_label(canvas, f"{label} | level={level}", x, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_trajectory_grid(source, recon, steps, sg, tg, level, path: Path):
    cells = [(f"SRC G{sg}", tensor_to_pil(source)), ("VAE recon", tensor_to_pil(recon))]
    for item in steps:
        g = item.get("metrics", {}).get("continuous_grade", None)
        if g is None:
            label = f"a={item['alpha']:.2f}"
        else:
            label = f"a={item['alpha']:.2f} g={float(g):.2f}"
        cells.append((label, tensor_to_pil(item["image"])))
    w, h = cells[0][1].size
    lh = 26
    canvas = Image.new("L", (w * len(cells), h + lh), color=0)
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(cells):
        x = i * w
        canvas.paste(im, (x, 0))
        draw.text((x + 4, h + 4), label, fill=255)
    canvas.save(path)


def metric_record(result, source_image, sg: int, tg: int, level: int, strength: float, args) -> Dict:
    image_l1 = float((result["edited_image"] - source_image).abs().mean().item())
    latent_l1 = float((result["edited_latent"] - result["source_latent"]).abs().mean().item())
    d = {
        "source_grade": sg,
        "target_grade": tg,
        "grade_delta": tg - sg,
        "level_index": level,
        "strength": float(strength),
        "sampling_steps": int(args.sampling_steps),
        "directional_scale": float(args.directional_scale),
        "lambda_schedule": args.lambda_schedule,
        "image_l1_to_source": image_l1,
        "latent_l1_to_source": latent_l1,
    }
    d.update(result["lambda_stats"])
    return d


def common_edit_kwargs(args):
    return dict(
        sampling_steps=args.sampling_steps,
        inversion_guidance_scale=args.inversion_guidance_scale,
        source_guidance_scale=args.source_guidance_scale,
        target_guidance_scale=args.target_guidance_scale,
        directional_scale=args.directional_scale,
        lambda_schedule=args.lambda_schedule,
        lambda_min_ratio=args.lambda_min_ratio,
        lambda_time_power=args.lambda_time_power,
        grade_distance_gain=args.grade_distance_gain,
        late_identity_blend=args.late_identity_blend,
        late_identity_steps=args.late_identity_steps,
        x0_clip=args.x0_clip,
    )


# ---------------------------------------------------------------------
# Batch metadata
# ---------------------------------------------------------------------
def load_metadata(args, logger):
    if args.metadata_csv is None:
        raise ValueError(f"mode={args.mode} requires --metadata-csv")
    p = args.metadata_csv.expanduser().resolve()
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {"image_path", "grade", "level_index", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prepared_metadata.csv missing columns: {sorted(missing)}")
    df = df.copy()
    df.insert(0, "metadata_index", np.arange(len(df), dtype=np.int64))
    df["grade"] = pd.to_numeric(df["grade"], errors="raise").astype(int)
    df["level_index"] = pd.to_numeric(df["level_index"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    if args.split != "all":
        df = df[df["split"] == args.split].copy()
    df = df.reset_index(drop=True)
    start = max(0, int(args.start_index))
    df = df.iloc[start:].reset_index(drop=True)
    if args.max_cases > 0:
        df = df.iloc[: args.max_cases].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No cases selected")
    logger.info("metadata=%s | split=%s | selected=%d", p, args.split, len(df))
    return df, p


# ---------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_all_grades(runtime, source_image: torch.Tensor, source_grade: int, level: int, args, case_dir: Optional[Path] = None):
    vae = runtime["vae"]
    device = source_image.device
    source_latent, _ = vae.encode(source_image, sample=False)
    recon = vae.decode(source_latent).clamp(0, 1)

    outputs: Dict[int, torch.Tensor] = {}
    strengths: Dict[int, float] = {}
    records: List[Dict] = []

    for tg in range(1, 6):
        strength, calibration_table, calibration_meta = choose_strength(
            runtime, source_latent, source_grade, tg, level, args
        )
        strengths[tg] = strength
        if calibration_table is not None and case_dir is not None:
            calibration_table.to_csv(case_dir / f"strength_calibration_G{tg}.csv", index=False)
            save_json(calibration_meta, case_dir / f"strength_calibration_G{tg}.json")

        if tg == source_grade or strength <= 0:
            edited_latent = source_latent.clone()
            edited_image = source_image.clone()
            stats = {"start_timestep": 0, "anchor_left": float(tg), "anchor_right": float(tg), "anchor_mix": 0.0}
        else:
            endpoint = make_target_endpoint(
                runtime, source_latent, source_grade, tg, level, strength, args
            )
            edited_latent = endpoint["edited_latent"]
            edited_image = vae.decode(edited_latent).clamp(0, 1)
            stats = dict(endpoint["stats"])

        result = {
            "source_latent": source_latent,
            "edited_latent": edited_latent,
            "edited_image": edited_image,
            "lambda_stats": stats,
        }
        outputs[tg] = edited_image
        rec = metric_record(result, source_image, source_grade, tg, level, strength, args)
        rec.update({
            "strength_mode": args.strength_mode,
            "calibration_fallback": bool(calibration_meta.get("fallback", False)),
            "calibration_reason": str(calibration_meta.get("reason", "")),
        })
        records.append(rec)
    return outputs, strengths, recon, records


@torch.no_grad()
def generate_trajectory(runtime, source_image: torch.Tensor, source_grade: int, target_grade: int, level: int, args, case_dir: Optional[Path] = None):
    """
    Continuous counterfactual trajectory.

    Main mode: local_ordinal
      Piecewise local contrastive ordinal velocity transport (LCOVT).  At a
      nominal continuous grade q, only the two neighboring TRAINED integer
      anchors are evaluated, then their velocity fields are interpolated.

    Baselines:
      v_direction        endpoint contrastive v_source/v_target steering
      fractional_ordinal directly feed fractional grade to conditioner
      latent_lerp        interpolate final VAE latents
      latent_slerp       spherical latent interpolation
    """
    if args.num_trajectory_steps < 2:
        raise ValueError("--num-trajectory-steps must be >= 2")

    vae = runtime["vae"]
    model = runtime["model"]
    schedule = runtime["schedule"]
    stage1 = runtime["stage1"]
    device = source_image.device
    num_grades = int(runtime["config"].num_grades)

    source_latent, _ = vae.encode(source_image, sample=False)
    recon = vae.decode(source_latent).clamp(0, 1)
    sg_long = torch.tensor([source_grade], device=device, dtype=torch.long)
    tg_long = torch.tensor([target_grade], device=device, dtype=torch.long)
    level_t = torch.tensor([level], device=device, dtype=torch.long)

    strength, calibration_table, calibration_meta = choose_strength(
        runtime, source_latent, source_grade, target_grade, level, args
    )
    if calibration_table is not None and case_dir is not None:
        calibration_table.to_csv(case_dir / "strength_calibration.csv", index=False)
        save_json(calibration_meta, case_dir / "strength_calibration.json")

    if source_grade == target_grade or strength <= 0:
        start_latent = source_latent.clone()
        timeline = torch.zeros(1, device=device, dtype=torch.long)
    else:
        start_latent, timeline = ddim_invert_to_strength(
            stage1, model, schedule, source_latent, sg_long, level_t,
            sampling_steps=args.sampling_steps,
            strength=strength,
            inversion_guidance_scale=args.inversion_guidance_scale,
            x0_clip=args.x0_clip,
        )

    target_endpoint = None
    if args.trajectory_mode in {"latent_lerp", "latent_slerp"} and source_grade != target_grade and strength > 0:
        endpoint = make_target_endpoint(
            runtime, source_latent, source_grade, target_grade, level, strength, args
        )
        target_endpoint = endpoint["edited_latent"]

    dense_count = int(args.trajectory_candidates) if args.arc_length_reparameterize else int(args.num_trajectory_steps)
    dense_count = max(int(args.num_trajectory_steps), dense_count, 2)
    alphas = np.linspace(0.0, 1.0, dense_count, dtype=np.float32)
    dense_steps = []

    for idx, alpha in enumerate(alphas.tolist()):
        continuous_grade = (1.0 - float(alpha)) * float(source_grade) + float(alpha) * float(target_grade)
        severity = continuous_grade_to_severity(continuous_grade, num_grades=num_grades)
        left, right, mix = local_grade_anchors(continuous_grade, num_grades=num_grades)

        if idx == 0 or source_grade == target_grade or strength <= 0:
            edited_latent = source_latent.clone()
            image = source_image.clone()
            stats = {"start_timestep": 0, "anchor_left": float(source_grade), "anchor_right": float(source_grade), "anchor_mix": 0.0,
                     "lambda_min": 0.0, "lambda_max": 0.0, "lambda_mean": 0.0}
        elif args.trajectory_mode == "local_ordinal":
            edited_latent, stats = ddim_local_ordinal_edit(
                stage1, model, schedule, start_latent, source_latent,
                continuous_grade, level_t, timeline,
                guidance_scale=args.ordinal_guidance_scale,
                num_grades=num_grades,
                late_identity_blend=args.late_identity_blend,
                late_identity_steps=args.late_identity_steps,
                x0_clip=args.x0_clip,
            )
            image = vae.decode(edited_latent).clamp(0, 1)
        elif args.trajectory_mode == "v_direction":
            edited_latent, stats = ddim_directional_edit(
                stage1, model, schedule, start_latent, source_latent,
                sg_long, tg_long, level_t, timeline,
                directional_scale=args.directional_scale,
                trajectory_alpha=float(alpha),
                source_guidance_scale=args.source_guidance_scale,
                target_guidance_scale=args.target_guidance_scale,
                lambda_schedule=args.lambda_schedule,
                lambda_min_ratio=args.lambda_min_ratio,
                lambda_time_power=args.lambda_time_power,
                grade_distance_gain=args.grade_distance_gain,
                late_identity_blend=args.late_identity_blend,
                late_identity_steps=args.late_identity_steps,
                x0_clip=args.x0_clip,
            )
            image = vae.decode(edited_latent).clamp(0, 1)
        elif args.trajectory_mode == "fractional_ordinal":
            grade_float = torch.tensor([continuous_grade], device=device, dtype=torch.float32)
            edited_latent, stats = ddim_conditioned_edit(
                stage1, model, schedule, start_latent, source_latent,
                grade_float, level_t, timeline,
                guidance_scale=args.ordinal_guidance_scale,
                late_identity_blend=args.late_identity_blend,
                late_identity_steps=args.late_identity_steps,
                x0_clip=args.x0_clip,
            )
            stats.update({"anchor_left": float(left), "anchor_right": float(right), "anchor_mix": float(mix)})
            image = vae.decode(edited_latent).clamp(0, 1)
        elif args.trajectory_mode == "latent_lerp":
            edited_latent = latent_lerp(source_latent, target_endpoint, float(alpha))
            image = vae.decode(edited_latent).clamp(0, 1)
            stats = {"start_timestep": int(timeline[-1].item()) if len(timeline) else 0,
                     "anchor_left": float(left), "anchor_right": float(right), "anchor_mix": float(mix),
                     "lambda_min": 0.0, "lambda_max": 0.0, "lambda_mean": 0.0}
        elif args.trajectory_mode == "latent_slerp":
            edited_latent = latent_slerp(source_latent, target_endpoint, float(alpha))
            image = vae.decode(edited_latent).clamp(0, 1)
            stats = {"start_timestep": int(timeline[-1].item()) if len(timeline) else 0,
                     "anchor_left": float(left), "anchor_right": float(right), "anchor_mix": float(mix),
                     "lambda_min": 0.0, "lambda_max": 0.0, "lambda_mean": 0.0}
        else:
            raise ValueError(f"Unknown trajectory_mode={args.trajectory_mode}")

        rec = {
            "dense_index": idx,
            "trajectory_mode": args.trajectory_mode,
            "alpha": float(alpha),
            "continuous_grade": float(continuous_grade),
            "continuous_severity": float(severity),
            "anchor_left": float(stats.get("anchor_left", left)),
            "anchor_right": float(stats.get("anchor_right", right)),
            "anchor_mix": float(stats.get("anchor_mix", mix)),
            "source_grade": int(source_grade),
            "target_grade": int(target_grade),
            "level_index": int(level),
            "strength": float(strength),
            "strength_mode": args.strength_mode,
            "image_l1_to_source": float((image - source_image).abs().mean().item()),
            "latent_l1_to_source": float((edited_latent - source_latent).abs().mean().item()),
            **{k: v for k, v in stats.items() if k not in {"anchor_left", "anchor_right", "anchor_mix"}},
        }
        dense_steps.append({"alpha": float(alpha), "image": image, "latent": edited_latent, "metrics": rec})

    # Dense raw path is useful for monotonicity and arc-length analysis.
    if case_dir is not None:
        pd.DataFrame([s["metrics"] for s in dense_steps]).to_csv(case_dir / "trajectory_dense_metrics.csv", index=False)

    if args.arc_length_reparameterize and len(dense_steps) > args.num_trajectory_steps:
        indices, arc_positions = arc_length_indices(
            [s["latent"] for s in dense_steps], args.num_trajectory_steps
        )
        selected = []
        for out_idx, (dense_idx, arc_pos) in enumerate(zip(indices, arc_positions)):
            item = dense_steps[dense_idx]
            metrics = dict(item["metrics"])
            metrics["step_index"] = int(out_idx)
            metrics["arc_position"] = float(arc_pos)
            metrics["selected_dense_index"] = int(dense_idx)
            selected.append({"alpha": item["alpha"], "image": item["image"], "latent": item["latent"], "metrics": metrics})
        steps = selected
    else:
        steps = []
        # If dense_count equals requested count, use normalized cumulative arc position for reporting.
        latents = [s["latent"] for s in dense_steps]
        indices, arc_positions = arc_length_indices(latents, len(latents))
        for out_idx, item in enumerate(dense_steps):
            metrics = dict(item["metrics"])
            metrics["step_index"] = int(out_idx)
            metrics["arc_position"] = float(arc_positions[out_idx]) if out_idx < len(arc_positions) else float(item["alpha"])
            metrics["selected_dense_index"] = int(out_idx)
            steps.append({"alpha": item["alpha"], "image": item["image"], "latent": item["latent"], "metrics": metrics})

    return recon, steps, float(strength), calibration_meta


def run_one(args, runtime, device, logger):
    if args.source_image is None or args.source_grade is None or args.target_grade is None or args.level_index is None:
        raise ValueError("mode=one needs --source-image --source-grade --target-grade --level-index")
    source = load_source_image(args.source_image, runtime["config"].image_height, runtime["config"].image_width).to(device)
    vae = runtime["vae"]
    source_latent, _ = vae.encode(source, sample=False)
    recon = vae.decode(source_latent).clamp(0, 1)
    strength, calibration_table, calibration_meta = choose_strength(
        runtime, source_latent, args.source_grade, args.target_grade, args.level_index, args
    )
    if calibration_table is not None:
        calibration_table.to_csv(args.output_dir / "strength_calibration.csv", index=False)
        save_json(calibration_meta, args.output_dir / "strength_calibration.json")

    if args.source_grade == args.target_grade or strength <= 0:
        edited_latent = source_latent.clone()
        edited_image = source.clone()
        stats = {"start_timestep": 0, "anchor_left": float(args.source_grade), "anchor_right": float(args.source_grade), "anchor_mix": 0.0}
    else:
        endpoint = make_target_endpoint(
            runtime, source_latent, args.source_grade, args.target_grade, args.level_index, strength, args
        )
        edited_latent = endpoint["edited_latent"]
        edited_image = vae.decode(edited_latent).clamp(0, 1)
        stats = endpoint["stats"]

    result = {"source_latent": source_latent, "vae_recon": recon, "edited_latent": edited_latent,
              "edited_image": edited_image, "lambda_stats": stats}
    out = args.output_dir / f"G{args.source_grade}_to_G{args.target_grade}_L{args.level_index}_s{strength:.3f}.png"
    tensor_to_pil(edited_image).save(out)
    tensor_to_pil(source).save(args.output_dir / "source.png")
    tensor_to_pil(recon).save(args.output_dir / "vae_recon.png")
    rec = metric_record(result, source, args.source_grade, args.target_grade, args.level_index, strength, args)
    rec.update({"strength_mode": args.strength_mode, "calibration_fallback": bool(calibration_meta.get("fallback", False)),
                "calibration_reason": str(calibration_meta.get("reason", ""))})
    pd.DataFrame([rec]).to_csv(args.output_dir / "stage2_results.csv", index=False)
    logger.info("saved=%s | strength=%.3f | image_L1=%.5f | start_t=%d", out, strength, rec["image_l1_to_source"], rec.get("start_timestep", 0))


def run_all_grades(args, runtime, device, logger):
    if args.source_image is None or args.source_grade is None or args.level_index is None:
        raise ValueError("mode=all_grades needs --source-image --source-grade --level-index")
    source = load_source_image(args.source_image, runtime["config"].image_height, runtime["config"].image_width).to(device)
    outputs, strengths, recon, records = generate_all_grades(
        runtime, source, args.source_grade, args.level_index, args, case_dir=args.output_dir
    )
    save_all_grades_grid(source, recon, outputs, strengths, args.source_grade, args.level_index, args.output_dir / "all_grades.png")
    if args.save_individual_targets:
        for g, im in outputs.items():
            tensor_to_pil(im).save(args.output_dir / f"target_G{g}.png")
    pd.DataFrame(records).to_csv(args.output_dir / "stage2_results.csv", index=False)
    logger.info("saved=%s", args.output_dir / "all_grades.png")


def run_trajectory(args, runtime, device, logger):
    if args.source_image is None or args.source_grade is None or args.target_grade is None or args.level_index is None:
        raise ValueError("mode=trajectory needs --source-image --source-grade --target-grade --level-index")
    source = load_source_image(args.source_image, runtime["config"].image_height, runtime["config"].image_width).to(device)
    recon, steps, strength, calibration_meta = generate_trajectory(
        runtime, source, args.source_grade, args.target_grade, args.level_index, args, case_dir=args.output_dir
    )
    save_trajectory_grid(source, recon, steps, args.source_grade, args.target_grade, args.level_index, args.output_dir / "trajectory.png")
    tensor_to_pil(source).save(args.output_dir / "source.png")
    tensor_to_pil(recon).save(args.output_dir / "vae_recon.png")
    if args.save_individual_targets:
        for item in steps:
            alpha = item["alpha"]
            idx = item["metrics"]["step_index"]
            tensor_to_pil(item["image"]).save(args.output_dir / f"step_{idx:02d}_a{alpha:.4f}.png")
    pd.DataFrame([s["metrics"] for s in steps]).to_csv(args.output_dir / "trajectory_metrics.csv", index=False)
    logger.info("saved=%s | selected_strength=%.3f | calibration=%s", args.output_dir / "trajectory.png", strength, calibration_meta)


def run_batch_all_grades(args, runtime, device, logger):
    df, metadata_path = load_metadata(args, logger)
    all_records = []
    for local_idx, row in df.iterrows():
        meta_idx = int(row["metadata_index"]); sg = int(row["grade"]); level = int(row["level_index"])
        image_path = resolve_image_path(row["image_path"], args.root, metadata_path)
        case_name = f"idx_{meta_idx:06d}_G{sg}_L{level}"
        case_dir = ensure_dir(args.output_dir / "cases" / case_name)
        final_grid = case_dir / "all_grades.png"; case_csv = case_dir / "case_results.csv"
        if args.skip_existing and final_grid.exists() and case_csv.exists():
            logger.info("[%d/%d] skip %s", local_idx + 1, len(df), case_name)
            old = pd.read_csv(case_csv); all_records.extend(old.to_dict("records")); continue
        source = load_source_image(image_path, runtime["config"].image_height, runtime["config"].image_width).to(device)
        outputs, strengths, recon, records = generate_all_grades(runtime, source, sg, level, args, case_dir=case_dir)
        save_all_grades_grid(source, recon, outputs, strengths, sg, level, final_grid)
        tensor_to_pil(source).save(case_dir / "source.png"); tensor_to_pil(recon).save(case_dir / "vae_recon.png")
        if args.save_individual_targets:
            for g, im in outputs.items(): tensor_to_pil(im).save(case_dir / f"target_G{g}.png")
        patient_id = str(row.get("patient_id", ""))
        for r in records:
            r.update({"metadata_index": meta_idx, "patient_id": patient_id, "image_path": str(image_path), "split": str(row["split"])})
        pd.DataFrame(records).to_csv(case_csv, index=False); all_records.extend(records)
        logger.info("[%d/%d] %s done", local_idx + 1, len(df), case_name)
    pd.DataFrame(all_records).to_csv(args.output_dir / "stage2_results.csv", index=False)


def run_batch_trajectory(args, runtime, device, logger):
    if args.target_grade is None: raise ValueError("mode=batch_trajectory requires --target-grade")
    df, metadata_path = load_metadata(args, logger); all_records = []
    for local_idx, row in df.iterrows():
        meta_idx = int(row["metadata_index"]); sg = int(row["grade"]); level = int(row["level_index"])
        image_path = resolve_image_path(row["image_path"], args.root, metadata_path)
        case_name = f"idx_{meta_idx:06d}_G{sg}_to_G{args.target_grade}_L{level}"
        case_dir = ensure_dir(args.output_dir / "cases" / case_name)
        final_grid = case_dir / "trajectory.png"; case_csv = case_dir / "trajectory_metrics.csv"
        if args.skip_existing and final_grid.exists() and case_csv.exists():
            logger.info("[%d/%d] skip %s", local_idx + 1, len(df), case_name)
            old = pd.read_csv(case_csv); old["metadata_index"] = meta_idx; old["patient_id"] = str(row.get("patient_id", "")); old["image_path"] = str(image_path); old["split"] = str(row["split"])
            all_records.extend(old.to_dict("records")); continue
        source = load_source_image(image_path, runtime["config"].image_height, runtime["config"].image_width).to(device)
        recon, steps, strength, calibration_meta = generate_trajectory(runtime, source, sg, args.target_grade, level, args, case_dir=case_dir)
        save_trajectory_grid(source, recon, steps, sg, args.target_grade, level, final_grid)
        tensor_to_pil(source).save(case_dir / "source.png"); tensor_to_pil(recon).save(case_dir / "vae_recon.png")
        if args.save_individual_targets:
            for item in steps:
                alpha = item["alpha"]; idx = item["metrics"]["step_index"]
                tensor_to_pil(item["image"]).save(case_dir / f"step_{idx:02d}_a{alpha:.4f}.png")
        case_records = []
        for item in steps:
            rec = dict(item["metrics"]); rec.update({"metadata_index": meta_idx, "patient_id": str(row.get("patient_id", "")), "image_path": str(image_path), "split": str(row["split"])})
            case_records.append(rec)
        pd.DataFrame(case_records).to_csv(case_csv, index=False); all_records.extend(case_records)
        logger.info("[%d/%d] %s done | s=%.3f", local_idx + 1, len(df), case_name, strength)
    pd.DataFrame(all_records).to_csv(args.output_dir / "trajectory_metrics.csv", index=False)


def verify_continuous_ordinal_conditioner(runtime: Dict, device: torch.device, logger: logging.Logger) -> None:
    """Fail fast if the loaded model does not really respond to fractional grade coordinates."""
    model = runtime["model"]
    conditioner = getattr(model, "conditioner", None)
    severity_features = getattr(conditioner, "severity_features", None)
    if severity_features is None:
        raise RuntimeError(
            "trajectory_mode=fractional_ordinal requires the clean ContinuousOrdinalConditioner "
            "with severity_features()."
        )
    with torch.no_grad():
        f_a = severity_features(torch.tensor([2.0], device=device, dtype=torch.float32))
        f_b = severity_features(torch.tensor([2.5], device=device, dtype=torch.float32))
    if torch.allclose(f_a, f_b, atol=1e-7, rtol=1e-7):
        raise RuntimeError(
            "Loaded conditioner does not distinguish grade 2.0 from 2.5; "
            "continuous ordinal trajectory is therefore not valid for this checkpoint."
        )
    logger.info("✅ Continuous ordinal conditioner check passed (fractional grades supported).")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["one", "all_grades", "trajectory", "batch_all_grades", "batch_trajectory"], default="one")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--stage1-script", type=Path, default=Path("train_ordinal_latent_diffusion_v2.py"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-vae-mismatch", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--source-image", type=Path)
    p.add_argument("--source-grade", type=int, choices=range(1, 6))
    p.add_argument("--target-grade", type=int, choices=range(1, 6))
    p.add_argument("--level-index", type=int, choices=range(0, 5))

    p.add_argument("--metadata-csv", type=Path)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-individual-targets", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--sampling-steps", type=int, default=120)
    p.add_argument("--strength-mode", choices=["fixed", "delta", "self_calibrated"], default="delta",
                   help="self_calibrated is the V3 contrastive diffusion-prior selector; delta is faster for large batches")
    p.add_argument("--strength", type=float, default=0.95)
    p.add_argument("--adjacent-strength", type=float, default=0.75)
    p.add_argument("--strength-per-extra-grade", type=float, default=0.07)
    p.add_argument("--max-strength", type=float, default=0.96)

    # Self-calibration candidates and contrastive prior probes.
    p.add_argument("--calibration-strength-grid", type=str, default="0.70,0.78,0.86,0.92,0.96,0.99")
    p.add_argument("--prior-probe-count", type=int, default=4)
    p.add_argument("--prior-probe-min-frac", type=float, default=0.15)
    p.add_argument("--prior-probe-max-frac", type=float, default=0.75)
    p.add_argument("--prior-probe-seed", type=int, default=3407)
    p.add_argument("--prior-guidance-scale", type=float, default=1.0)
    p.add_argument("--prior-min-snr-gamma", type=float, default=5.0)
    p.add_argument("--prior-margin", type=float, default=0.0,
                   help="absolute target-vs-runner-up v-loss margin required by self calibration")
    p.add_argument("--prior-relative-gain", type=float, default=0.0,
                   help="relative target-vs-source prior improvement required by self calibration")

    p.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    p.add_argument("--source-guidance-scale", type=float, default=1.0)
    p.add_argument("--target-guidance-scale", type=float, default=1.0)
    p.add_argument("--directional-scale", type=float, default=1.0)
    p.add_argument("--ordinal-guidance-scale", type=float, default=1.0,
                   help="CFG scale for integer-anchor/local ordinal reverse DDIM")

    p.add_argument("--lambda-schedule", choices=["constant", "cosine", "power"], default="constant")
    p.add_argument("--lambda-min-ratio", type=float, default=0.35)
    p.add_argument("--lambda-time-power", type=float, default=1.0)
    p.add_argument("--grade-distance-gain", type=float, default=0.0)
    p.add_argument("--late-identity-blend", type=float, default=0.0)
    p.add_argument("--late-identity-steps", type=int, default=6)
    p.add_argument("--x0-clip", type=float, default=4.5)

    p.add_argument("--num-trajectory-steps", type=int, default=17)
    p.add_argument("--trajectory-candidates", type=int, default=33,
                   help="dense path samples before arc-length reparameterization")
    p.add_argument("--arc-length-reparameterize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trajectory-mode",
                   choices=["local_ordinal", "v_direction", "fractional_ordinal", "latent_lerp", "latent_slerp"],
                   default="local_ordinal",
                   help="local_ordinal is V3 main method; others are ablations/baselines")

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main():
    args = build_parser().parse_args()
    args.root = args.root.expanduser().resolve()
    if not args.stage1_script.is_absolute():
        args.stage1_script = args.root / args.stage1_script
    args.output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    logger = setup_logger(args.output_dir, filename="stage2.log")

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; switching to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    runtime = load_models(
        args.stage1_script,
        args.checkpoint,
        args.vae_checkpoint,
        device,
        logger,
        allow_vae_mismatch=args.allow_vae_mismatch,
    )

    logger.info("%s", SCRIPT_VERSION)
    logger.info("mode=%s", args.mode)
    logger.info("stage1=%s", args.stage1_script)
    logger.info("diffusion=%s", args.checkpoint)
    logger.info("vae=%s", args.vae_checkpoint)
    logger.info("latent_shape=%s", runtime["latent_shape"])
    if args.mode in {"trajectory", "batch_trajectory"} and args.trajectory_mode == "fractional_ordinal":
        verify_continuous_ordinal_conditioner(runtime, device, logger)

    save_json(
        {
            "script_version": SCRIPT_VERSION,
            "stage1_script": args.stage1_script,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "vae_checkpoint": args.vae_checkpoint,
            "vae_sha256": runtime["vae_sha256"],
            "stage1_config": runtime["checkpoint"].get("config"),
            "arguments": vars(args),
            "method": "LCOVT: source-grade DDIM inversion + local contrastive adjacent-grade velocity steering + optional contrastive diffusion-prior strength calibration",
            "nlp_inspiration": ["Activation Addition / activation engineering", "Contrastive Activation Addition", "Contrastive Decoding / decoding-time steering"],
            "important_note": "The NLP papers motivate representation-space steering and contrastive selection; the medical diffusion formulation here is an adaptation, not a direct reproduction. Internal diffusion-prior calibration must not replace independent clinical evaluation.",
        },
        args.output_dir / "stage2_manifest.json",
    )

    if args.mode == "one":
        run_one(args, runtime, device, logger)
    elif args.mode == "all_grades":
        run_all_grades(args, runtime, device, logger)
    elif args.mode == "trajectory":
        run_trajectory(args, runtime, device, logger)
    elif args.mode == "batch_all_grades":
        run_batch_all_grades(args, runtime, device, logger)
    elif args.mode == "batch_trajectory":
        run_batch_trajectory(args, runtime, device, logger)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info("✅ Stage-2 all-in-one complete")


if __name__ == "__main__":
    main()
