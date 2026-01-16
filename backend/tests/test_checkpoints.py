"""Tests for checkpoint utilities."""

import os
import torch

from backend.ml.checkpoints import (
    format_checkpoint_name,
    format_epoch_checkpoint_name,
    sanitize_checkpoint_name,
    checkpoint_path_from_name,
    find_latest_checkpoint,
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


def test_sanitize_checkpoint_name():
    """Checkpoint sanitizer should drop path components."""
    assert sanitize_checkpoint_name("../../foo.pt") == "foo.pt"


def test_checkpoint_path_from_name_uses_env(monkeypatch, tmp_path):
    """Checkpoint path helper should use CHECKPOINT_DIR."""
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    path = checkpoint_path_from_name("nested/foo.pt")
    assert path.parent == tmp_path
    assert path.name == "foo.pt"


def test_find_latest_checkpoint(monkeypatch, tmp_path):
    """find_latest_checkpoint should return the newest matching entry."""
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"

    torch.save({"lattice_size": 8, "training_temperature": 2.0}, first)
    torch.save({"lattice_size": 8, "training_temperature": 2.0}, second)

    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    latest = find_latest_checkpoint(lattice_size=8)
    assert latest is not None
    assert latest.name == "second.pt"
