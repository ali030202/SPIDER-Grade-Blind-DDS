from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def setup_logger(run_dir: Path, name: str = "grade_chain") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"无法JSON序列化: {type(value)!r}")


def atomic_json_save(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(temporary, path)


def atomic_torch_save(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(data), temporary)
    os.replace(temporary, path)


def safe_torch_load(path: Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({k: _json_default(v) if isinstance(v, (Path, np.generic)) else v for k, v in row.items()})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def set_trainable(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)
    module.train(value)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()


class EMA:
    def __init__(self, module: torch.nn.Module, beta: float = 0.999):
        self.beta = float(beta)
        self.shadow = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        state = module.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.beta).add_(state[key].detach(), alpha=1.0 - self.beta)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {key: value.to(device).clone() for key, value in state.items()}

    def copy_to(self, module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        current = module.state_dict()
        backup = {key: current[key].detach().clone() for key in self.shadow}
        current.update({key: value.to(current[key]) for key, value in self.shadow.items()})
        module.load_state_dict(current, strict=True)
        return backup

    @staticmethod
    def restore(module: torch.nn.Module, backup: Mapping[str, torch.Tensor]) -> None:
        current = module.state_dict()
        current.update(backup)
        module.load_state_dict(current, strict=True)
