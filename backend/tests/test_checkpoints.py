"""Tests for checkpoint utilities."""

from backend.ml.checkpoints import (
    format_checkpoint_name,
    format_epoch_checkpoint_name,
)


def test_format_checkpoint_name():
    """Checkpoint name includes lattice size and temperature."""
    name = format_checkpoint_name(lattice_size=8, temperature=2.27)
    assert name == "ising_8_T2.27.pt"


def test_format_epoch_checkpoint_name():
    """Epoch checkpoint name includes epoch and timestamp."""
    name = format_epoch_checkpoint_name(
        lattice_size=16,
        temperature=1.5,
        epoch=4,
        timestamp="20240101_000000",
    )
    assert name == "ising_16_T1.50_epoch4_20240101_000000.pt"
