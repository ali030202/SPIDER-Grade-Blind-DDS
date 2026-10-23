from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .blocks import ConvNeXtBlock, Downsample, Upsample, group_norm


@dataclass
class VAEConfig:
    image_height: int = 160
    image_width: int = 224
    latent_channels: int = 8
    vae_base: int = 48


class ImageEncoder(nn.Module):
    """
    224x224 -> 28x28 for the default 3 downsampling stages.

    The module names intentionally match the previous ImageVAE so old VAE
    state_dicts can still be loaded when the architecture is unchanged.
    """

    def __init__(self, base: int, latent_channels: int):
        super().__init__()
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.stage0 = nn.Sequential(ConvNeXtBlock(base), ConvNeXtBlock(base))
        self.down1 = Downsample(base, base * 2)
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(base * 2),
            ConvNeXtBlock(base * 2),
        )
        self.down2 = Downsample(base * 2, base * 4)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(base * 4),
            ConvNeXtBlock(base * 4),
        )
        self.down3 = Downsample(base * 4, base * 4)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(base * 4),
            ConvNeXtBlock(base * 4),
        )
        self.to_moments = nn.Conv2d(base * 4, latent_channels * 2, 1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        f0 = self.stage0(self.stem(image))
        f1 = self.stage1(self.down1(f0))
        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        mean, logvar = self.to_moments(f3).chunk(2, dim=1)
        logvar = logvar.clamp(-12.0, 6.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "features": (f0, f1, f2, f3),
        }


class ImageDecoder(nn.Module):
    def __init__(self, base: int, latent_channels: int):
        super().__init__()
        self.from_latent = nn.Conv2d(latent_channels, base * 4, 3, padding=1)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(base * 4),
            ConvNeXtBlock(base * 4),
        )
        self.up2 = Upsample(base * 4, base * 4)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(base * 4),
            ConvNeXtBlock(base * 4),
        )
        self.up1 = Upsample(base * 4, base * 2)
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(base * 2),
            ConvNeXtBlock(base * 2),
        )
        self.up0 = Upsample(base * 2, base)
        self.stage0 = nn.Sequential(ConvNeXtBlock(base), ConvNeXtBlock(base))
        self.output = nn.Sequential(
            group_norm(base),
            nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.from_latent(latent))
        x = self.stage2(self.up2(x))
        x = self.stage1(self.up1(x))
        x = self.stage0(self.up0(x))
        return torch.sigmoid(self.output(x))


class ImageVAE(nn.Module):
    """Minimal VAE used by the ordinal latent diffusion pipeline only."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = ImageEncoder(config.vae_base, config.latent_channels)
        self.decoder = ImageDecoder(config.vae_base, config.latent_channels)
        self.latent_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoded = self.encoder(image)
        mean, logvar = encoded["mean"], encoded["logvar"]

        if sample and self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            latent = mean

        return latent * self.latent_scale, encoded

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent / self.latent_scale.clamp_min(1e-6))

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent, encoded = self.encode(image, sample=True)
        return {
            "image": self.decode(latent),
            "latent": latent,
            "mean": encoded["mean"],
            "logvar": encoded["logvar"],
        }
