from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils.io import seed_worker

def load_grayscale(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def transform_image(
    image: Image.Image,
    height: int,
    width: int,
    augment: bool,
) -> torch.Tensor:
    image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if augment:
        angle = random.uniform(-3.0, 3.0)
        translate = [
            int(random.uniform(-0.025, 0.025) * width),
            int(random.uniform(-0.025, 0.025) * height),
        ]
        scale = random.uniform(0.97, 1.03)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        image = TF.adjust_gamma(image, random.uniform(0.92, 1.08))
        image = TF.adjust_contrast(image, random.uniform(0.92, 1.08))
    tensor = TF.pil_to_tensor(image).float() / 255.0
    if augment and random.random() < 0.25:
        tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.0, 0.015)).clamp(0.0, 1.0)
    return tensor


class IVDDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        height: int = 224,
        width: int = 224,
        augment: bool = False,
        two_views: bool = False,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        self.two_views = bool(two_views)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        image = load_grayscale(row["image_path"])
        first = transform_image(image, self.height, self.width, self.augment)
        result: Dict[str, object] = {
            "image": first,
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "level": torch.tensor(int(row["level_index"]), dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "sample_id": str(row["sample_id"]),
            "image_path": str(row["image_path"]),
        }
        if self.two_views:
            result["image_view2"] = transform_image(image, self.height, self.width, True)
        return result


class AdjacentPairDataset(Dataset):
    """Random unpaired source/target samples from the same level and adjacent grades.

    The real target image is used only for domain/discriminator supervision. It is
    never used as an L1/L2 pixel target for the source patient.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        edge_weights: Mapping[int, float],
        height: int,
        width: int,
        epoch_size: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch_size = int(epoch_size or len(frame))
        self.edge_ids = sorted(int(edge) for edge, weight in edge_weights.items() if weight > 0)
        weights = np.array([float(edge_weights[edge]) for edge in self.edge_ids], dtype=np.float64)
        self.edge_probabilities = weights / weights.sum()
        self.pools: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for index, row in self.frame.iterrows():
            self.pools[(int(row["level_index"]), int(row["grade"]))].append(index)
        self.valid_levels: Dict[int, List[int]] = {}
        for edge in self.edge_ids:
            self.valid_levels[edge] = [
                level
                for level in range(5)
                if self.pools.get((level, edge)) and self.pools.get((level, edge + 1))
            ]
            if not self.valid_levels[edge]:
                raise RuntimeError(f"训练集中没有任何节段同时包含G{edge}和G{edge + 1}")

    def __len__(self) -> int:
        return self.epoch_size

    def _choose_different_patient(self, indices: Sequence[int], source_patient: str) -> int:
        candidates = [index for index in indices if str(self.frame.iloc[index]["patient_id"]) != source_patient]
        return random.choice(candidates if candidates else list(indices))

    def __getitem__(self, index: int) -> Dict[str, object]:
        edge = int(np.random.choice(self.edge_ids, p=self.edge_probabilities))
        forward = random.random() < 0.5
        source_grade = edge if forward else edge + 1
        target_grade = edge + 1 if forward else edge
        level = random.choice(self.valid_levels[edge])
        source_index = random.choice(self.pools[(level, source_grade)])
        source_row = self.frame.iloc[source_index]
        target_index = self._choose_different_patient(
            self.pools[(level, target_grade)], str(source_row["patient_id"])
        )
        target_row = self.frame.iloc[target_index]
        source_image = transform_image(
            load_grayscale(source_row["image_path"]), self.height, self.width, self.augment
        )
        target_image = transform_image(
            load_grayscale(target_row["image_path"]), self.height, self.width, self.augment
        )
        return {
            "source_image": source_image,
            "real_target_image": target_image,
            "source_grade": torch.tensor(source_grade, dtype=torch.long),
            "target_grade": torch.tensor(target_grade, dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.long),
            "edge": torch.tensor(edge, dtype=torch.long),
            "source_patient": str(source_row["patient_id"]),
            "target_patient": str(target_row["patient_id"]),
        }


class TransitionValidationDataset(Dataset):
    """Deterministic source-only validation for active adjacent transitions.

    It does not require a real target image because validation metrics assess
    target-grade response, content preservation, cycle consistency and artifacts.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        edge_weights: Mapping[int, float],
        height: int,
        width: int,
        max_items: int = 256,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.height = int(height)
        self.width = int(width)
        self.items: List[Tuple[int, int]] = []
        active_edges = sorted(int(edge) for edge, weight in edge_weights.items() if weight > 0)
        for row_index, row in self.frame.iterrows():
            grade = int(row["grade"])
            candidates = []
            for edge in active_edges:
                if grade == edge:
                    candidates.append(edge + 1)
                if grade == edge + 1:
                    candidates.append(edge)
            for target in candidates:
                self.items.append((row_index, target))
        if not self.items:
            # Extremely small validation sets may miss the active grades. Use a
            # nearest legal adjacent target solely for a diagnostic score.
            for row_index, row in self.frame.iterrows():
                grade = int(row["grade"])
                target = grade + 1 if grade < 5 else grade - 1
                self.items.append((row_index, target))
        self.items = self.items[: max(1, int(max_items))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row_index, target_grade = self.items[index]
        row = self.frame.iloc[row_index]
        source_image = transform_image(
            load_grayscale(row["image_path"]), self.height, self.width, False
        )
        return {
            "source_image": source_image,
            "source_grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "target_grade": torch.tensor(int(target_grade), dtype=torch.long),
            "level": torch.tensor(int(row["level_index"]), dtype=torch.long),
            "source_patient": str(row["patient_id"]),
        }


class LongRangePairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        height: int,
        width: int,
        epoch_size: Optional[int] = None,
        augment: bool = True,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        self.epoch_size = int(epoch_size or len(frame))
        self.pools: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for index, row in self.frame.iterrows():
            self.pools[(int(row["level_index"]), int(row["grade"]))].append(index)
        self.valid_pairs: List[Tuple[int, int, int]] = []
        for level in range(5):
            for source in range(1, 6):
                for target in range(1, 6):
                    if abs(source - target) <= 1:
                        continue
                    if self.pools.get((level, source)):
                        self.valid_pairs.append((level, source, target))
        if not self.valid_pairs:
            raise RuntimeError("没有可用于长程蒸馏的跨级样本")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> Dict[str, object]:
        level, source_grade, target_grade = random.choice(self.valid_pairs)
        source_index = random.choice(self.pools[(level, source_grade)])
        row = self.frame.iloc[source_index]
        image = transform_image(
            load_grayscale(row["image_path"]), self.height, self.width, self.augment
        )
        return {
            "source_image": image,
            "source_grade": torch.tensor(source_grade, dtype=torch.long),
            "target_grade": torch.tensor(target_grade, dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.long),
            "source_patient": str(row["patient_id"]),
        }


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    training: bool,
    seed: int,
    balanced_frame: Optional[pd.DataFrame] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    sampler = None
    shuffle = training
    if training and balanced_frame is not None:
        cells = [
            f"{row.disc_level}|G{int(row.grade)}"
            for row in balanced_frame.itertuples(index=False)
        ]
        counts = Counter(cells)
        weights = torch.tensor([1.0 / np.sqrt(counts[cell]) for cell in cells], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    generator = torch.Generator().manual_seed(seed + (1 if training else 1009))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=training if drop_last is None else drop_last,
    )
