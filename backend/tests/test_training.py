"""Tests for training module including losses, trainer, and API endpoints."""

import pytest
import torch
import tempfile
import os
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.models.score_network import ScoreNetwork
from backend.ml.models.diffusion import DiffusionProcess
from backend.ml.training import (
    Trainer,
    ScoreMatchingLoss,
    denoising_score_matching_loss,
    sigma_weighted_loss,
    snr_weighted_loss,
    importance_sampled_loss,
    compute_loss,
    reduce_loss,
)
from backend.ml.training.trainer import (
    EarlyStopping,
    EMA,
    create_scheduler,
    compute_gradient_norm,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_score_network():
    """Create a small score network for fast testing."""
    return ScoreNetwork(
        in_channels=1,
        base_channels=8,
        time_embed_dim=16,
        num_blocks=1,
    )


@pytest.fixture
def diffusion():
    """Create a diffusion process."""
    return DiffusionProcess(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def sample_batch():
    """Create sample batch data for testing."""
    return torch.randn(4, 1, 8, 8)


@pytest.fixture
def sample_dataloader(sample_batch):
    """Create a dataloader from sample batch."""
    dataset = TensorDataset(sample_batch)
    return DataLoader(dataset, batch_size=2, shuffle=True)


@pytest.fixture
def trainer(small_score_network, diffusion):
    """Create a trainer instance."""
    return Trainer(
        score_network=small_score_network,
        diffusion=diffusion,
        learning_rate=1e-3,
    )


# ============================================================================
# Loss Function Tests
# ============================================================================


class TestComputeLoss:
    """Tests for compute_loss helper function."""

    def test_l2_loss(self):
        """L2 loss computes squared error."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="l2")
        expected = torch.tensor([0.0, 1.0, 4.0])
        assert torch.allclose(loss, expected)

    def test_l1_loss(self):
        """L1 loss computes absolute error."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="l1")
        expected = torch.tensor([0.0, 1.0, 2.0])
        assert torch.allclose(loss, expected)

    def test_huber_loss_small_error(self):
        """Huber loss is quadratic for small errors."""
        pred = torch.tensor([1.0, 1.5])
        target = torch.tensor([1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="huber", huber_delta=1.0)
        # For |diff| <= delta, huber = 0.5 * diff^2
        expected = torch.tensor([0.0, 0.5 * 0.5**2])
        assert torch.allclose(loss, expected)

    def test_huber_loss_large_error(self):
        """Huber loss is linear for large errors."""
        pred = torch.tensor([3.0])
        target = torch.tensor([0.0])
        loss = compute_loss(pred, target, loss_type="huber", huber_delta=1.0)
        # For |diff| > delta, huber = delta * (|diff| - 0.5 * delta)
        expected = torch.tensor([1.0 * (3.0 - 0.5 * 1.0)])
        assert torch.allclose(loss, expected)

    def test_invalid_loss_type(self):
        """Invalid loss type raises ValueError."""
        with pytest.raises(ValueError):
            compute_loss(torch.zeros(1), torch.zeros(1), loss_type="invalid")


class TestReduceLoss:
    """Tests for reduce_loss helper function."""

    def test_mean_reduction(self):
        """Mean reduction averages all elements."""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reduced = reduce_loss(loss, reduction="mean")
        assert reduced.item() == 2.5

    def test_sum_reduction(self):
        """Sum reduction sums all elements."""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reduced = reduce_loss(loss, reduction="sum")
        assert reduced.item() == 10.0

    def test_none_reduction(self):
        """None reduction returns original tensor."""
        loss = torch.tensor([1.0, 2.0, 3.0])
        reduced = reduce_loss(loss, reduction="none")
        assert torch.equal(reduced, loss)

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        with pytest.raises(ValueError):
            reduce_loss(torch.zeros(1), reduction="invalid")
