from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConditionalResBlock,
    Downsample,
    SinusoidalTimeEmbedding,
    SpatialAttention,
    Upsample,
    group_norm,
)


@dataclass
class OrdinalLatentDiffusionConfig:
    image_height: int = 224
    image_width: int = 224

    latent_channels: int = 8
    vae_base: int = 48

    diffusion_base: int = 96
    condition_dim: int = 384
    diffusion_timesteps: int = 1000
    dropout: float = 0.05

    num_grades: int = 5
    num_levels: int = 5

    # G1..G5 are represented on an ordered scalar axis.
    severity_fourier_frequencies: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


def grade_to_scalar(
    grade: torch.Tensor,
    num_grades: int = 5,
) -> torch.Tensor:
    """Map G1..G5 to [-1, 1]."""
    grade = grade.float()
    center = (num_grades + 1.0) / 2.0
    half_range = max((num_grades - 1.0) / 2.0, 1.0)
    return (grade - center) / half_range


class ContinuousOrdinalConditioner(nn.Module):
    """time + continuous severity + disc-level conditioning."""

    def __init__(self, config: OrdinalLatentDiffusionConfig):
        super().__init__()
        self.config = config
        dim = config.condition_dim

        self.time = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        frequencies = torch.tensor(
            config.severity_fourier_frequencies,
            dtype=torch.float32,
        )
        self.register_buffer(
            "severity_frequencies",
            frequencies,
            persistent=True,
        )

        severity_input_dim = 1 + 2 * len(config.severity_fourier_frequencies)
        self.severity_mlp = nn.Sequential(
            nn.Linear(severity_input_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # real levels = 0..4; index 5 is reserved for null/unconditional use.
        self.level_embedding = nn.Embedding(config.num_levels + 1, dim)

        self.final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def severity_features(self, grade: torch.Tensor) -> torch.Tensor:
        scalar = grade_to_scalar(grade, self.config.num_grades).unsqueeze(1)
        freq = self.severity_frequencies.to(
            device=scalar.device,
            dtype=scalar.dtype,
        )[None, :]
        angle = math.pi * scalar * freq
        return torch.cat(
            [scalar, torch.sin(angle), torch.cos(angle)],
            dim=1,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        grade: torch.Tensor,
        level: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_condition = self.time_mlp(self.time(timestep))
        severity_condition = self.severity_mlp(self.severity_features(grade))
        level_condition = self.level_embedding(
            level.long().clamp(0, self.config.num_levels)
        )

        if drop_mask is not None:
            drop_mask = drop_mask.bool().view(-1, 1)
            severity_condition = torch.where(
                drop_mask,
                torch.zeros_like(severity_condition),
                severity_condition,
            )
            level_condition = torch.where(
                drop_mask,
                torch.zeros_like(level_condition),
                level_condition,
            )

        return self.final(
            time_condition + severity_condition + level_condition
        )


class OrdinalLatentUNet(nn.Module):
    """
    Pure ordinal latent diffusion denoiser.

    Input:
        noisy latent only
    Condition:
        continuous Pfirrmann severity + disc level + timestep

    No source image, no grade-chain edge id, no assessor, no discriminator.
    """

    def __init__(self, config: OrdinalLatentDiffusionConfig):
        super().__init__()
        self.config = config

        base = config.diffusion_base
        cond = config.condition_dim
        latent_channels = config.latent_channels

        self.conditioner = ContinuousOrdinalConditioner(config)

        self.input = nn.Conv2d(latent_channels, base, 3, padding=1)

        self.down0a = ConditionalResBlock(base, base, cond, config.dropout)
        self.down0b = ConditionalResBlock(base, base, cond, config.dropout)

        self.down1 = Downsample(base, base * 2)
        self.down1a = ConditionalResBlock(base * 2, base * 2, cond, config.dropout)
        self.down1b = ConditionalResBlock(base * 2, base * 2, cond, config.dropout)

        self.down2 = Downsample(base * 2, base * 4)

        self.mid1 = ConditionalResBlock(base * 4, base * 4, cond, config.dropout)
        self.attn = SpatialAttention(base * 4, heads=4)
        self.mid2 = ConditionalResBlock(base * 4, base * 4, cond, config.dropout)

        self.up1 = Upsample(base * 4, base * 2)
        self.up1a = ConditionalResBlock(base * 4, base * 2, cond, config.dropout)
        self.up1b = ConditionalResBlock(base * 2, base * 2, cond, config.dropout)

        self.up0 = Upsample(base * 2, base)
        self.up0a = ConditionalResBlock(base * 2, base, cond, config.dropout)
        self.up0b = ConditionalResBlock(base, base, cond, config.dropout)

        self.output = nn.Sequential(
            group_norm(base),
            nn.SiLU(),
            nn.Conv2d(base, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        grade: torch.Tensor,
        level: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.conditioner(
            timestep,
            grade,
            level,
            drop_mask=drop_mask,
        )

        x = self.input(noisy_latent)

        d0 = self.down0b(
            self.down0a(x, condition),
            condition,
        )

        d1 = self.down1b(
            self.down1a(self.down1(d0), condition),
            condition,
        )

        x = self.down2(d1)
        x = self.mid1(x, condition)
        x = self.attn(x)
        x = self.mid2(x, condition)

        x = self.up1(x)
        if x.shape[-2:] != d1.shape[-2:]:
            x = F.interpolate(
                x,
                size=d1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = self.up1a(torch.cat([x, d1], dim=1), condition)
        x = self.up1b(x, condition)

        x = self.up0(x)
        if x.shape[-2:] != d0.shape[-2:]:
            x = F.interpolate(
                x,
                size=d0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = self.up0a(torch.cat([x, d0], dim=1), condition)
        x = self.up0b(x, condition)

        return self.output(x)
