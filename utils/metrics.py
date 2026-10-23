from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinate = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).view(1, 1, kernel_size, kernel_size)


def gaussian_blur(image: torch.Tensor, kernel_size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    kernel_size = int(kernel_size) | 1
    kernel = gaussian_kernel(kernel_size, sigma, image.device, image.dtype)
    padding = kernel_size // 2
    return F.conv2d(image, kernel.expand(image.shape[1], 1, -1, -1), padding=padding, groups=image.shape[1])


def image_gradient(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., 1:] - image[..., :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def laplacian_energy(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    return F.conv2d(image, kernel, padding=1).abs().mean()


def ordinal_expected_grade(ordinal_logits: torch.Tensor) -> torch.Tensor:
    return 1.0 + torch.sigmoid(ordinal_logits).sum(dim=1)


def ordinal_targets(grades: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(1, 5, device=grades.device)[None]
    return (grades[:, None] > thresholds).float()


def ordinal_loss(
    class_logits: torch.Tensor,
    ordinal_logits: torch.Tensor,
    grades: torch.Tensor,
    ce_weight: float = 1.0,
    ordinal_weight: float = 1.0,
    monotonic_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    ce = F.cross_entropy(class_logits, grades.long() - 1)
    target = ordinal_targets(grades)
    cumulative = F.binary_cross_entropy_with_logits(ordinal_logits, target)
    # P(y>k+1) should not exceed P(y>k).
    probability = torch.sigmoid(ordinal_logits)
    monotonic = F.relu(probability[:, 1:] - probability[:, :-1]).mean()
    total = ce_weight * ce + ordinal_weight * cumulative + monotonic_weight * monotonic
    return {"loss": total, "ce": ce, "ordinal": cumulative, "monotonic": monotonic}


def content_cosine_loss(source: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(source, generated, dim=1)).mean()


def basic_generation_metrics(
    source: torch.Tensor,
    generated: torch.Tensor,
    source_content: torch.Tensor,
    generated_content: torch.Tensor,
    expected_grade: torch.Tensor,
    target_grade: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    low = (gaussian_blur(source) - gaussian_blur(generated)).abs().mean()
    content = content_cosine_loss(source_content, generated_content)
    target_mae = (expected_grade - target_grade.float()).abs().mean()
    residual = generated - source
    artifact = laplacian_energy(residual) / (laplacian_energy(source).detach() + 1e-4)
    return {
        "low_frequency_change": low,
        "content_distance": content,
        "target_mae": target_mae,
        "artifact_ratio": artifact,
    }
