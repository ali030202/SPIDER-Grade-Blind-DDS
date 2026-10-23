from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest divisor <= 32."""
    groups = min(32, int(channels))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half,
            device=timestep.device,
            dtype=torch.float32,
        ) / max(half - 1, 1)
        value = timestep.float()[:, None] * exponent.exp()[None]
        embedding = torch.cat([value.sin(), value.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style residual block used by the VAE."""

    def __init__(self, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise1 = nn.Conv2d(channels, channels * 4, 1)
        self.pointwise2 = nn.Conv2d(channels * 4, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise2(F.gelu(self.pointwise1(x)))
        x = x * self.gamma

        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
            x = x * mask / keep

        return residual + x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConditionalResBlock(nn.Module):
    """Residual block with FiLM modulation from a conditioning vector."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, out_channels * 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.conv2(self.dropout(F.silu(x)))
        return residual + x


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = group_norm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        value = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attention(value, value, value, need_weights=False)
        attended = self.proj(attended).transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return x + attended
