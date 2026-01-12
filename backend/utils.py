"""Utility functions for the backend."""

import torch
from typing import Optional, Union
import numpy as np


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate torch device.

    Args:
        device: Device string ('cpu', 'cuda', 'mps') or None for auto-detect

    Returns:
        torch.device instance
    """
    if device is not None:
        return torch.device(device)

    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_list(
    tensor: Union[torch.Tensor, np.ndarray]
) -> Union[list, float]:
    """Convert tensor/array to Python list for JSON serialization.

    Args:
        tensor: PyTorch tensor or numpy array

    Returns:
        Python list or float
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 0:
        return float(tensor)

    return tensor.tolist()


def validate_lattice_size(size: int, max_size: int = 128) -> int:
    """Validate and constrain lattice size.

    Args:
        size: Requested lattice size
        max_size: Maximum allowed size

    Returns:
        Validated lattice size

    Raises:
        ValueError: If size is invalid
    """
    if size < 4:
        raise ValueError(f"Lattice size must be at least 4, got {size}")
    if size > max_size:
        raise ValueError(f"Lattice size must be at most {max_size}, got {size}")
    return size


def validate_temperature(
    temperature: float,
    min_temp: float = 0.1,
    max_temp: float = 10.0,
) -> float:
    """Validate temperature value.

    Args:
        temperature: Temperature value
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature

    Returns:
        Validated temperature

    Raises:
        ValueError: If temperature is out of range
    """
    if temperature < min_temp or temperature > max_temp:
        raise ValueError(
            f"Temperature must be in [{min_temp}, {max_temp}], got {temperature}"
        )
    return temperature
