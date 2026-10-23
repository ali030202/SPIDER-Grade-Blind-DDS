from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def extract(values: torch.Tensor, timestep: torch.Tensor, shape) -> torch.Tensor:
    out = values.to(timestep.device).gather(0, timestep.long())
    return out.view(timestep.shape[0], *((1,) * (len(shape) - 1)))


@dataclass
class DiffusionSchedule:
    """
    Clean v-prediction diffusion schedule.

    It deliberately contains no grade-chain/source-image sampling logic.
    """

    timesteps: int = 1000

    def __post_init__(self) -> None:
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)

    def q_sample(
        self,
        x0: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, x0.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, x0.shape)
        return alpha * x0 + sigma * noise

    def v_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, x0.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, x0.shape)
        return alpha * noise - sigma * x0

    def predict_x0_from_v(
        self,
        noisy: torch.Tensor,
        predicted_v: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, noisy.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, noisy.shape)
        return alpha * noisy - sigma * predicted_v

    def predict_noise_from_v(
        self,
        noisy: torch.Tensor,
        predicted_v: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, noisy.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, noisy.shape)
        return sigma * noisy + alpha * predicted_v

    def min_snr_weight(
        self,
        timestep: torch.Tensor,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        snr = self.snr.to(timestep.device).gather(0, timestep.long())
        return torch.minimum(snr, snr.new_tensor(float(gamma))) / (snr + 1.0)
