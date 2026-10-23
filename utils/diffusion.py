from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-5, 0.999).float()


def extract(values: torch.Tensor, timestep: torch.Tensor, shape) -> torch.Tensor:
    out = values.to(timestep.device).gather(0, timestep.long())
    return out.view(timestep.shape[0], *((1,) * (len(shape) - 1)))


@dataclass
class DiffusionSchedule:
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

    def q_sample(self, x0: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, x0.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, x0.shape)
        return alpha * x0 + sigma * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, x0.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, x0.shape)
        return alpha * noise - sigma * x0

    def predict_x0_from_v(
        self, noisy: torch.Tensor, predicted_v: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, noisy.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, noisy.shape)
        return alpha * noisy - sigma * predicted_v

    def predict_noise_from_v(
        self, noisy: torch.Tensor, predicted_v: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        alpha = extract(self.sqrt_alpha_bar, timestep, noisy.shape)
        sigma = extract(self.sqrt_one_minus_alpha_bar, timestep, noisy.shape)
        return sigma * noisy + alpha * predicted_v

    def min_snr_weight(self, timestep: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        snr = self.snr.to(timestep.device).gather(0, timestep.long())
        # v-prediction weighting from Min-SNR training.
        return torch.minimum(snr, snr.new_tensor(float(gamma))) / (snr + 1.0)

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser,
        source_latent: torch.Tensor,
        source_grade: torch.Tensor,
        target_grade: torch.Tensor,
        level: torch.Tensor,
        transition_id: torch.Tensor,
        steps: int = 30,
        strength: float = 0.35,
        guidance_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = source_latent.device
        steps = max(2, int(steps))
        maximum_index = int(round((self.timesteps - 1) * float(strength)))
        maximum_index = max(1, min(self.timesteps - 1, maximum_index))
        timeline = torch.linspace(maximum_index, 0, steps, device=device).long()
        if noise is None:
            noise = torch.randn_like(source_latent)
        timestep = torch.full(
            (source_latent.shape[0],), maximum_index, device=device, dtype=torch.long
        )
        value = self.q_sample(source_latent, timestep, noise)

        for index, current_scalar in enumerate(timeline):
            current = torch.full_like(timestep, int(current_scalar.item()))
            conditional = denoiser(
                value,
                source_latent,
                current,
                source_grade,
                target_grade,
                level,
                transition_id,
                condition_drop=False,
            )
            if guidance_scale != 1.0:
                unconditional = denoiser(
                    value,
                    source_latent,
                    current,
                    source_grade,
                    target_grade,
                    level,
                    transition_id,
                    condition_drop=True,
                )
                predicted_v = unconditional + float(guidance_scale) * (conditional - unconditional)
            else:
                predicted_v = conditional
            x0 = self.predict_x0_from_v(value, predicted_v, current).clamp(-4.0, 4.0)
            if index == len(timeline) - 1:
                value = x0
                break
            next_scalar = timeline[index + 1]
            next_t = torch.full_like(timestep, int(next_scalar.item()))
            predicted_noise = self.predict_noise_from_v(value, predicted_v, current)
            next_alpha = extract(self.sqrt_alpha_bar, next_t, value.shape)
            next_sigma = extract(self.sqrt_one_minus_alpha_bar, next_t, value.shape)
            value = next_alpha * x0 + next_sigma * predicted_noise
        return value
