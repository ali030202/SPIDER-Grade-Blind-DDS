#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train VAE V2 for lumbar IVD MRI
================================

用途
----
重新训练一个与当前 Ordinal Latent Diffusion / Stage-2 代码兼容的 VAE。

关键设计：
1. 只使用干净的 model.vae.ImageVAE；不依赖 grade_chain、assessor、GAN 或旧工程组件；
2. 默认 224x224、1 通道、latent_channels=8、vae_base=48，和你当前 Stage-1 一致；
3. 训练目标保持简单：
       L_total = L1 + lambda_ssim * (1 - SSIM) + beta_kl * KL
   不加入 assessor、GAN、手工退变特征、频谱/纹理等额外约束；
4. KL 使用 warm-up，避免训练前期把重建压坏；
5. 最终自动校准 latent_scale，使训练集 latent mean 的全局 std 约为 1；
6. 保存格式兼容 train_ordinal_latent_diffusion_v1.py 中 load_frozen_vae()：
       checkpoint["components"]["vae"]
7. 自动保存 preview、metrics、best/last checkpoint。

非常重要
--------
如果你用这个脚本重新训练了 VAE，不建议把新 VAE 直接替换到“旧的 Stage-1 best.pt”中继续推理。
旧 Stage-1 diffusion 是在旧 VAE latent 空间里训练的。新 VAE 即使结构相同，latent 分布也会变化。
正确流程通常是：

    新 VAE -> 固定新 VAE -> 重新训练 Stage-1 Ordinal Latent Diffusion -> 再做 Stage-2

推荐输入 metadata
-----------------
优先使用你之前已经固定的数据划分 prepared_metadata.csv，至少包含：

    image_path, split

其中 split 为 train / val / test。
如果还有 patient_id / grade / level_index，可以保留，但 VAE 训练本身不使用 grade。

示例
----
cd /home/lab/jmx/SPIDER

python train_vae_v2.py \
  --metadata-csv /home/lab/jmx/SPIDER/data/prepared_metadata.csv \
  --run-dir /home/lab/jmx/SPIDER/runs_vae_v2 \
  --epochs 150 \
  --batch-size 16 \
  --lr 2e-4 \
  --latent-channels 8 \
  --vae-base 48 \
  --image-size 224 \
  --amp

训练完成后：
    /home/lab/jmx/SPIDER/runs_vae_v2/checkpoints/vae_best.pt

然后重新训练 Stage-1 时使用：
    --vae-checkpoint /home/lab/jmx/SPIDER/runs_vae_v2_224x160/checkpoints/vae_best.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 仅依赖当前干净 VAE 模块。
from model.vae import VAEConfig, ImageVAE


SCRIPT_VERSION = "TRAIN_VAE_V2_CLEAN_STABLE_RECON_SSIM_KL"


# ================================================================
# 配置
# ================================================================

@dataclass
class VAETrainConfig:
    image_height: int = 224
    image_width: int = 224
    latent_channels: int = 8
    vae_base: int = 48

    epochs: int = 150
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5

    lambda_ssim: float = 0.25
    beta_kl: float = 5e-7
    kl_warmup_epochs: int = 20

    num_workers: int = 4
    seed: int = 2026
    patience: int = 30
    preview_every: int = 5
    preview_count: int = 8
    latent_calibration_batches: int = 64

    ema_beta: float = 0.999
    use_ema: bool = True


# ================================================================
# 基础工具
# ================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict, path: Path) -> None:
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    ensure_dir(path.parent)
    path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def setup_logger(run_dir: Path) -> logging.Logger:
    ensure_dir(run_dir)
    logger = logging.getLogger("train_vae_v2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(run_dir / "train_vae_v2.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
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
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


# ================================================================
# Dataset
# ================================================================

class VAEMRIDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_height: int,
        image_width: int,
        image_root: Optional[Path] = None,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, raw: str) -> Path:
        path = Path(str(raw))
        if path.is_absolute():
            return path
        if self.image_root is not None:
            return self.image_root / path
        return path

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        path = self._resolve_path(row["image_path"])
        if not path.exists():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            image = image.convert("L")
            image = TF.resize(
                image,
                [self.image_height, self.image_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.pil_to_tensor(image).float() / 255.0

        return {
            "image": tensor,
            "image_path": str(path),
        }


def prepare_splits(
    metadata_csv: Path,
    image_root: Optional[Path],
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(metadata_csv, encoding="utf-8-sig", dtype={"patient_id": str})

    if "image_path" not in frame.columns:
        # 尝试兼容常见列名。
        candidates = ["image", "path", "image_relative_path", "unified_image_path"]
        found = next((c for c in candidates if c in frame.columns), None)
        if found is None:
            raise ValueError(
                "metadata 必须包含 image_path，或至少包含以下之一: " + ", ".join(candidates)
            )
        frame = frame.rename(columns={found: "image_path"})
        logger.info("image path column: %s -> image_path", found)

    # 优先复用已有 split，避免重新划分。
    if "split" in frame.columns:
        split_values = frame["split"].astype(str).str.lower().str.strip()
        frame = frame.copy()
        frame["split"] = split_values

        train = frame[frame["split"].isin(["train", "training"])].reset_index(drop=True)
        val = frame[frame["split"].isin(["val", "valid", "validation"])].reset_index(drop=True)
        test = frame[frame["split"].isin(["test", "testing"])].reset_index(drop=True)

        if len(train) > 0 and len(val) > 0:
            if len(test) == 0:
                logger.warning("metadata 没有 test split；VAE训练本身可以继续，但最终只报告 train/val。")
            logger.info("复用 metadata 中已有 split。")
            return train, val, test

    # 若没有 split，则优先按 patient_id 划分。
    ratios = np.array([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test fraction 之和必须等于 1")

    rng = np.random.default_rng(split_seed)

    if "patient_id" in frame.columns and frame["patient_id"].notna().all():
        patients = frame["patient_id"].astype(str).unique().tolist()
        rng.shuffle(patients)
        n = len(patients)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_p = set(patients[:n_train])
        val_p = set(patients[n_train:n_train + n_val])
        test_p = set(patients[n_train + n_val:])

        pid = frame["patient_id"].astype(str)
        train = frame[pid.isin(train_p)].reset_index(drop=True)
        val = frame[pid.isin(val_p)].reset_index(drop=True)
        test = frame[pid.isin(test_p)].reset_index(drop=True)
        logger.info("metadata 无 split，按 patient_id 划分。")
    else:
        logger.warning("metadata 无 split 且无完整 patient_id，只能按图像随机划分。")
        order = np.arange(len(frame))
        rng.shuffle(order)
        n = len(order)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train = frame.iloc[order[:n_train]].reset_index(drop=True)
        val = frame.iloc[order[n_train:n_train + n_val]].reset_index(drop=True)
        test = frame.iloc[order[n_train + n_val:]].reset_index(drop=True)

    return train, val, test


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


# ================================================================
# SSIM
# ================================================================

def _gaussian_kernel_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def ssim_index(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Numerically stable differentiable SSIM for grayscale [0,1] tensors.

    IMPORTANT: SSIM is always evaluated in FP32 with autocast disabled.
    Half-precision local variance/covariance subtraction can produce large
    cancellation errors and invalid SSIM > 1, which makes (1-SSIM) negative
    and can destabilize VAE training.
    """
    if x.shape != y.shape:
        raise ValueError(f"SSIM shape mismatch: {x.shape} vs {y.shape}")

    device_type = x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x32 = x.float().clamp(0.0, 1.0)
        y32 = y.float().clamp(0.0, 1.0)
        k1d = _gaussian_kernel_1d(window_size, sigma, x32.device, torch.float32)
        kernel2d = torch.outer(k1d, k1d).view(1, 1, window_size, window_size)
        padding = window_size // 2

        # Reflect padding avoids the artificial zero-valued border used by
        # padding=window_size//2 in conv2d.
        xpad = F.pad(x32, (padding, padding, padding, padding), mode="reflect")
        ypad = F.pad(y32, (padding, padding, padding, padding), mode="reflect")

        mu_x = F.conv2d(xpad, kernel2d)
        mu_y = F.conv2d(ypad, kernel2d)
        mu_x2 = mu_x.square()
        mu_y2 = mu_y.square()
        mu_xy = mu_x * mu_y

        ex2 = F.conv2d(xpad * xpad, kernel2d)
        ey2 = F.conv2d(ypad * ypad, kernel2d)
        exy = F.conv2d(xpad * ypad, kernel2d)

        sigma_x2 = (ex2 - mu_x2).clamp_min(0.0)
        sigma_y2 = (ey2 - mu_y2).clamp_min(0.0)
        sigma_xy = exy - mu_xy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = numerator / denominator.clamp_min(1e-12)

        # The analytical SSIM upper bound is 1. Numerical rounding should never
        # be allowed to turn the reconstruction loss negative.
        ssim_map = ssim_map.clamp(min=-1.0, max=1.0)
        return ssim_map.mean()


# ================================================================
# Loss
# ================================================================

def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()


def compute_losses(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    lambda_ssim: float,
    beta_kl: float,
) -> Dict[str, torch.Tensor]:
    recon = output["image"]
    # Reconstruction statistics/losses are accumulated in FP32. The network
    # forward may still use AMP; casting here keeps the sensitive reductions
    # stable while preserving gradients.
    recon32 = recon.float()
    target32 = target.float()
    l1 = F.l1_loss(recon32, target32)
    ssim = ssim_index(recon32, target32)
    ssim_loss = (1.0 - ssim).clamp_min(0.0)
    kl = kl_divergence(output["mean"].float(), output["logvar"].float())

    total = l1 + float(lambda_ssim) * ssim_loss + float(beta_kl) * kl

    mse = F.mse_loss(recon32, target32)
    psnr = psnr_from_mse(mse)

    return {
        "loss": total,
        "l1": l1,
        "ssim": ssim,
        "ssim_loss": ssim_loss,
        "kl": kl,
        "mse": mse,
        "psnr": psnr,
    }


# ================================================================
# EMA
# ================================================================

class EMA:
    def __init__(self, module: nn.Module, beta: float):
        self.beta = float(beta)
        self.shadow = {
            k: v.detach().clone()
            for k, v in module.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        state = module.state_dict()
        for key, shadow in self.shadow.items():
            value = state[key].detach()
            shadow.mul_(self.beta).add_(value, alpha=1.0 - self.beta)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in module.state_dict().items()}
        state = module.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value.to(state[key].device, dtype=state[key].dtype))
        return backup

    @torch.no_grad()
    def restore(self, module: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        module.load_state_dict(backup, strict=True)

    def load_state_dict(self, state: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {k: v.to(device).clone() for k, v in state.items()}


# ================================================================
# Preview
# ================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


@torch.no_grad()
def save_preview(
    vae: ImageVAE,
    loader: DataLoader,
    device: torch.device,
    path: Path,
    max_items: int = 8,
) -> None:
    vae.eval()
    raw = next(iter(loader))
    images = raw["image"][:max_items].to(device, non_blocking=True)

    latent, _ = vae.encode(images, sample=False)
    recon = vae.decode(latent).clamp(0.0, 1.0)
    residual = (recon - images).abs()

    n = images.shape[0]
    cell_w = images.shape[-1]
    cell_h = images.shape[-2]
    label_h = 20
    rows = 3

    canvas = Image.new("L", (cell_w * n, (cell_h + label_h) * rows), color=0)
    draw = ImageDraw.Draw(canvas)

    for i in range(n):
        for row_idx, (label, tensor) in enumerate([
            ("input", images[i:i+1]),
            ("recon", recon[i:i+1]),
            ("abs diff", residual[i:i+1]),
        ]):
            x = i * cell_w
            y = row_idx * (cell_h + label_h)
            canvas.paste(tensor_to_pil(tensor), (x, y))
            draw.text((x + 3, y + cell_h + 2), label, fill=255)

    ensure_dir(path.parent)
    canvas.save(path)


# ================================================================
# Checkpoint
# ================================================================

def compatible_vae_checkpoint(
    vae: ImageVAE,
    train_config: VAETrainConfig,
    epoch: int,
    best_score: float,
    metadata_csv: Path,
    extra: Optional[Dict] = None,
) -> Dict:
    payload = {
        "script_version": SCRIPT_VERSION,
        "config": asdict(train_config),
        "components": {
            "vae": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
        },
        "epoch": int(epoch),
        "best_score": float(best_score),
        "latent_scale": float(vae.latent_scale.detach().cpu()),
        "metadata_csv": str(metadata_csv),
    }
    if extra:
        payload.update(extra)
    return payload


# ================================================================
# Train / Eval
# ================================================================

def kl_beta_for_epoch(epoch: int, target_beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target_beta)
    ratio = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
    return float(target_beta) * ratio


def train_one_epoch(
    vae: ImageVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: VAETrainConfig,
    epoch: int,
    amp: bool,
    ema: Optional[EMA],
) -> Dict[str, float]:
    vae.train()
    totals: Dict[str, float] = {k: 0.0 for k in ["loss", "l1", "ssim", "ssim_loss", "kl", "mse", "psnr"]}
    count = 0
    beta = kl_beta_for_epoch(epoch, cfg.beta_kl, cfg.kl_warmup_epochs)

    progress = tqdm(loader, desc=f"VAE {epoch:03d}/{cfg.epochs}")
    for raw in progress:
        images = raw["image"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            output = vae(images)
            losses = compute_losses(
                output,
                images,
                lambda_ssim=cfg.lambda_ssim,
                beta_kl=beta,
            )

        # Fail fast instead of allowing a poisoned checkpoint/EMA state.
        if not torch.isfinite(losses["loss"]):
            logging.warning(
                "Skip batch: non-finite VAE loss "
                f"loss={float(losses['loss'].detach())}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            logging.warning(
                "Skip batch: non-finite gradient norm "
                f"{float(grad_norm)}"
            )

            optimizer.zero_grad(set_to_none=True)

            # AMP状态必须推进，否则下一batch unscale会报错
            scaler.update()

            continue
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(vae)

        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        count += batch_size

        progress.set_postfix(
            loss=f"{float(losses['loss'].detach()):.4f}",
            l1=f"{float(losses['l1'].detach()):.4f}",
            ssim=f"{float(losses['ssim'].detach()):.4f}",
        )

    result = {k: v / max(count, 1) for k, v in totals.items()}
    result["kl_beta"] = beta
    return result


@torch.no_grad()
def evaluate(
    vae: ImageVAE,
    loader: DataLoader,
    device: torch.device,
    cfg: VAETrainConfig,
) -> Dict[str, float]:
    vae.eval()
    totals: Dict[str, float] = {k: 0.0 for k in ["loss", "l1", "ssim", "ssim_loss", "kl", "mse", "psnr"]}
    count = 0

    # validation 用完整 beta，但 best 主要按重建质量选。
    for raw in loader:
        images = raw["image"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        # validation 用 deterministic mean latent，避免采样噪声干扰模型选择。
        latent, encoded = vae.encode(images, sample=False)
        recon = vae.decode(latent).clamp(0.0, 1.0)
        output = {
            "image": recon,
            "mean": encoded["mean"],
            "logvar": encoded["logvar"],
        }
        losses = compute_losses(
            output,
            images,
            lambda_ssim=cfg.lambda_ssim,
            beta_kl=cfg.beta_kl,
        )

        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        count += batch_size

    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def calibrate_latent_scale(
    vae: ImageVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    logger: logging.Logger,
) -> float:
    vae.eval()

    # 先把 scale 设回 1，统计 raw mean latent 的 std。
    vae.latent_scale.fill_(1.0)

    # 用 streaming E[x], E[x^2]，避免把所有 latent 都堆进内存。
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch_idx, raw in enumerate(loader):
        images = raw["image"].to(device, non_blocking=True)
        encoded = vae.encoder(images)["mean"].detach().float()
        total_sum += float(encoded.sum().cpu())
        total_sq_sum += float(encoded.square().sum().cpu())
        total_count += int(encoded.numel())
        if batch_idx + 1 >= max_batches:
            break

    mean = total_sum / max(total_count, 1)
    variance = total_sq_sum / max(total_count, 1) - mean * mean
    std = math.sqrt(max(variance, 1e-8))
    scale = 1.0 / max(std, 1e-4)
    vae.latent_scale.fill_(scale)

    logger.info("latent_scale calibrated: %.6f | raw latent mean=%.6f std=%.6f", scale, mean, std)
    return float(scale)


# ================================================================
# LR schedule
# ================================================================

def lr_lambda_builder(total_epochs: int, warmup_epochs: int = 5):
    def fn(epoch_index: int):
        epoch = epoch_index + 1
        if epoch <= warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), 1e-3)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


# ================================================================
# Main
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a clean VAE V2 for lumbar IVD MRI",
    )

    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--vae-base", type=int, default=48)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lambda-ssim", type=float, default=0.25)
    parser.add_argument("--beta-kl", type=float, default=1e-6)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--latent-calibration-batches", type=int, default=64)

    parser.add_argument("--ema-beta", type=float, default=0.999)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="可选：从旧 VAE 初始化；默认 None 表示从头训练。",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    previews_dir = ensure_dir(run_dir / "previews")
    data_dir = ensure_dir(run_dir / "data")

    logger = setup_logger(run_dir)
    seed_everything(args.seed, deterministic=args.deterministic)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; switching to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    image_height = int(args.image_height if args.image_height is not None else args.image_size)
    image_width = int(args.image_width if args.image_width is not None else args.image_size)

    cfg = VAETrainConfig(
        image_height=image_height,
        image_width=image_width,
        latent_channels=args.latent_channels,
        vae_base=args.vae_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_ssim=args.lambda_ssim,
        beta_kl=args.beta_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        preview_every=args.preview_every,
        preview_count=args.preview_count,
        latent_calibration_batches=args.latent_calibration_batches,
        ema_beta=args.ema_beta,
        use_ema=args.use_ema,
    )

    logger.info("Script version: %s", SCRIPT_VERSION)
    logger.info("Device: %s", device)
    logger.info("Config: %s", cfg)

    train_frame, val_frame, test_frame = prepare_splits(
        args.metadata_csv.expanduser().resolve(),
        args.image_root.expanduser().resolve() if args.image_root is not None else None,
        args.split_seed,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        logger,
    )

    # 保存本次实际使用的划分，便于复现。
    split_frame = []
    for split_name, split_df in [("train", train_frame), ("val", val_frame), ("test", test_frame)]:
        temp = split_df.copy()
        temp["vae_split"] = split_name
        split_frame.append(temp)
    pd.concat(split_frame, ignore_index=True).to_csv(data_dir / "vae_v2_splits.csv", index=False)

    logger.info("Samples train=%d val=%d test=%d", len(train_frame), len(val_frame), len(test_frame))
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise RuntimeError("train 或 val 为空，无法训练。")

    train_dataset = VAEMRIDataset(train_frame, image_height, image_width, args.image_root)
    val_dataset = VAEMRIDataset(val_frame, image_height, image_width, args.image_root)
    test_dataset = VAEMRIDataset(test_frame, image_height, image_width, args.image_root) if len(test_frame) > 0 else None

    train_loader = make_loader(train_dataset, cfg.batch_size, cfg.num_workers, True, cfg.seed)
    val_loader = make_loader(val_dataset, cfg.batch_size, cfg.num_workers, False, cfg.seed + 1)
    test_loader = make_loader(test_dataset, cfg.batch_size, cfg.num_workers, False, cfg.seed + 2) if test_dataset is not None else None

    vae_cfg = VAEConfig(
        image_height=image_height,
        image_width=image_width,
        latent_channels=cfg.latent_channels,
        vae_base=cfg.vae_base,
    )
    vae = ImageVAE(vae_cfg).to(device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "components" in ckpt and "vae" in ckpt["components"]:
            state = ckpt["components"]["vae"]
        elif isinstance(ckpt, dict) and "vae" in ckpt:
            state = ckpt["vae"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise TypeError("无法识别 init-checkpoint 格式")
        msg = vae.load_state_dict(state, strict=True)
        # 重新训练时先重置 latent_scale，最后会重新标定。
        vae.latent_scale.fill_(1.0)
        logger.info("Loaded init VAE: %s | %s", args.init_checkpoint, msg)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_builder(cfg.epochs, warmup_epochs=5),
    )
    scaler = GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    ema = EMA(vae, cfg.ema_beta) if cfg.use_ema else None

    best_path = checkpoints_dir / "vae_best.pt"
    last_path = checkpoints_dir / "vae_last.pt"
    metrics_path = metrics_dir / "vae_v2.csv"

    start_epoch = 1
    best_score = float("inf")
    bad_epochs = 0

    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        vae.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        logger.info("Resume from epoch %d", start_epoch)

    header = [
        "epoch", "lr", "kl_beta",
        "train_loss", "train_l1", "train_ssim", "train_kl", "train_psnr",
        "val_loss", "val_l1", "val_ssim", "val_kl", "val_psnr", "val_score",
    ]
    if not metrics_path.exists() or start_epoch == 1:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            vae, train_loader, optimizer, scaler, device, cfg, epoch, args.amp, ema
        )

        # validation 优先评价 EMA 权重。
        backup = None
        if ema is not None:
            backup = ema.copy_to(vae)
        val_metrics = evaluate(vae, val_loader, device, cfg)

        # 模型选择只看重建质量，不让 KL 主导：L1 + 0.25*(1-SSIM)
        val_score = val_metrics["l1"] + cfg.lambda_ssim * (1.0 - val_metrics["ssim"])

        improved = val_score < best_score
        if improved:
            best_score = float(val_score)
            bad_epochs = 0
            # 此时 vae 已是 EMA 权重（如果启用），直接保存 best。
            torch.save(
                compatible_vae_checkpoint(
                    vae,
                    cfg,
                    epoch,
                    best_score,
                    args.metadata_csv,
                    extra={
                        "best_val": val_metrics,
                        "uses_ema_weights": bool(ema is not None),
                    },
                ),
                best_path,
            )
        else:
            bad_epochs += 1

        if backup is not None:
            ema.restore(vae, backup)

        # last 保存训练中的 raw 权重 + optimizer/ema，方便 resume。
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "config": asdict(cfg),
                "model": {k: v.detach().cpu() for k, v in vae.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "ema": ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "best_score": best_score,
                "bad_epochs": bad_epochs,
            },
            last_path,
        )

        lr = float(optimizer.param_groups[0]["lr"])
        row = [
            epoch, lr, train_metrics["kl_beta"],
            train_metrics["loss"], train_metrics["l1"], train_metrics["ssim"], train_metrics["kl"], train_metrics["psnr"],
            val_metrics["loss"], val_metrics["l1"], val_metrics["ssim"], val_metrics["kl"], val_metrics["psnr"], val_score,
        ]
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        logger.info(
            "[VAE %03d] train L1=%.5f SSIM=%.4f | val L1=%.5f SSIM=%.4f PSNR=%.2f | score=%.6f%s",
            epoch,
            train_metrics["l1"],
            train_metrics["ssim"],
            val_metrics["l1"],
            val_metrics["ssim"],
            val_metrics["psnr"],
            val_score,
            " ✅" if improved else "",
        )

        if epoch == 1 or epoch % cfg.preview_every == 0 or improved:
            # preview 用当前最佳视角：EMA 如果有，就临时 apply。
            preview_backup = None
            if ema is not None:
                preview_backup = ema.copy_to(vae)
            save_preview(
                vae,
                val_loader,
                device,
                previews_dir / f"epoch_{epoch:03d}.png",
                max_items=cfg.preview_count,
            )
            if preview_backup is not None:
                ema.restore(vae, preview_backup)

        scheduler.step()

        if bad_epochs >= cfg.patience:
            logger.info("Early stop: %d epochs without validation improvement.", bad_epochs)
            break

    # ============================================================
    # 最终加载 best，重新校准 latent_scale，再覆盖保存为兼容 checkpoint。
    # ============================================================
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(best_ckpt["components"]["vae"], strict=True)
    vae.to(device).eval()

    latent_scale = calibrate_latent_scale(
        vae,
        train_loader,
        device,
        cfg.latent_calibration_batches,
        logger,
    )

    final_val = evaluate(vae, val_loader, device, cfg)
    final_test = evaluate(vae, test_loader, device, cfg) if test_loader is not None else None

    final_payload = compatible_vae_checkpoint(
        vae,
        cfg,
        int(best_ckpt.get("epoch", 0)),
        float(best_ckpt.get("best_score", best_score)),
        args.metadata_csv,
        extra={
            "best_val_before_scale_calibration": best_ckpt.get("best_val"),
            "final_val": final_val,
            "final_test": final_test,
            "latent_scale_calibrated": latent_scale,
            "uses_ema_weights": best_ckpt.get("uses_ema_weights", False),
        },
    )
    torch.save(final_payload, best_path)

    save_preview(
        vae,
        val_loader,
        device,
        previews_dir / "final_best.png",
        max_items=cfg.preview_count,
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "metadata_csv": str(args.metadata_csv),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "train_config": asdict(cfg),
        "samples": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame),
        },
        "final_val": final_val,
        "final_test": final_test,
        "latent_scale": latent_scale,
        "important_note": (
            "A newly trained VAE defines a new latent space. Retrain Stage-1 ordinal diffusion "
            "with this VAE instead of directly swapping it under an old diffusion checkpoint."
        ),
    }
    save_json(manifest, run_dir / "vae_v2_manifest.json")

    logger.info("✅ Best VAE: %s", best_path)
    logger.info("✅ latent_scale=%.6f", latent_scale)
    logger.info("Final val: L1=%.5f SSIM=%.4f PSNR=%.2f", final_val["l1"], final_val["ssim"], final_val["psnr"])
    if final_test is not None:
        logger.info("Final test: L1=%.5f SSIM=%.4f PSNR=%.2f", final_test["l1"], final_test["ssim"], final_test["psnr"])
    logger.info("IMPORTANT: 新 VAE 训练完成后，请用它重新训练 Stage-1 diffusion。")


if __name__ == "__main__":
    main()
