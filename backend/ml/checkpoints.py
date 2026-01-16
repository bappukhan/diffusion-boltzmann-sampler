"""Checkpoint utilities for training and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

import torch


DEFAULT_CHECKPOINT_DIR = "checkpoints"


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata extracted from a checkpoint file."""

    name: str
    path: str
    size_bytes: int
    modified_time: str
    lattice_size: Optional[int] = None
    training_temperature: Optional[float] = None
    model_config: Optional[Dict[str, Any]] = None
    diffusion_config: Optional[Dict[str, Any]] = None
    training_meta: Optional[Dict[str, Any]] = None


def get_checkpoint_dir() -> Path:
    """Return the checkpoint directory, ensuring it exists."""
    path = Path(os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_checkpoint_name(lattice_size: int, temperature: float) -> str:
    """Format the canonical checkpoint name for a lattice/temperature pair."""
    return f"ising_{lattice_size}_T{temperature:.2f}.pt"


def format_epoch_checkpoint_name(
    lattice_size: int,
    temperature: float,
    epoch: int,
    timestamp: Optional[str] = None,
) -> str:
    """Format an epoch-specific checkpoint name."""
    stamp = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"ising_{lattice_size}_T{temperature:.2f}_epoch{epoch}_{stamp}.pt"


def sanitize_checkpoint_name(name: str) -> str:
    """Return a filename-only checkpoint name."""
    return Path(name).name


def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    """Load metadata from a checkpoint file without raising on missing keys."""
    stat = path.stat()
    metadata = {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time": str(stat.st_mtime),
    }

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return CheckpointMetadata(**metadata)

    metadata.update(
        {
            "lattice_size": checkpoint.get("lattice_size"),
            "training_temperature": checkpoint.get("training_temperature"),
            "model_config": checkpoint.get("model_config"),
            "diffusion_config": checkpoint.get("diffusion_config"),
            "training_meta": checkpoint.get("training_meta"),
        }
    )
    return CheckpointMetadata(**metadata)


def list_checkpoints() -> List[CheckpointMetadata]:
    """List checkpoint metadata sorted by modified time (newest first)."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [load_checkpoint_metadata(path) for path in checkpoint_dir.glob("*.pt")]
    checkpoints.sort(key=lambda item: item.modified_time, reverse=True)
    return checkpoints


def find_latest_checkpoint(
    lattice_size: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[CheckpointMetadata]:
    """Return the latest checkpoint matching the optional filters."""
    checkpoints = list_checkpoints()

    def matches(item: CheckpointMetadata) -> bool:
        if lattice_size is not None and item.lattice_size != lattice_size:
            return False
        if temperature is not None and item.training_temperature is not None:
            return abs(item.training_temperature - temperature) < 1e-6
        return True

    for item in checkpoints:
        if matches(item):
            return item
    return None
