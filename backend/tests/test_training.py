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


class TestDSMLossFunctions:
    """Tests for denoising score matching loss functions."""

    def test_dsm_loss_returns_scalar(self, small_score_network, diffusion, sample_batch):
        """DSM loss returns a scalar tensor."""
        loss = denoising_score_matching_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_sigma_weighted_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """Sigma-weighted loss returns a scalar tensor."""
        loss = sigma_weighted_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_snr_weighted_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """SNR-weighted loss returns a scalar tensor."""
        loss = snr_weighted_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_importance_sampled_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """Importance-sampled loss returns a scalar tensor."""
        loss = importance_sampled_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_is_differentiable(self, small_score_network, diffusion, sample_batch):
        """Loss can be backpropagated."""
        loss = denoising_score_matching_loss(small_score_network, sample_batch, diffusion)
        loss.backward()

        # Check gradients exist
        for param in small_score_network.parameters():
            assert param.grad is not None


class TestScoreMatchingLossClass:
    """Tests for ScoreMatchingLoss class."""

    def test_uniform_weighting(self, small_score_network, diffusion, sample_batch):
        """Uniform weighting computes standard DSM loss."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="uniform")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_sigma_weighting(self, small_score_network, diffusion, sample_batch):
        """Sigma weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="sigma")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_snr_weighting(self, small_score_network, diffusion, sample_batch):
        """SNR weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="snr", snr_gamma=0.5)
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_importance_weighting(self, small_score_network, diffusion, sample_batch):
        """Importance weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="importance")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_invalid_weighting_raises(self, diffusion, small_score_network, sample_batch):
        """Invalid weighting raises ValueError."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="uniform")
        loss_fn.weighting = "invalid"  # Force invalid weighting
        with pytest.raises(ValueError):
            loss_fn(small_score_network, sample_batch)


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainerBasic:
    """Basic tests for Trainer class."""

    def test_trainer_initialization(self, small_score_network, diffusion):
        """Trainer initializes with correct attributes."""
        trainer = Trainer(small_score_network, diffusion, learning_rate=1e-3)
        assert trainer.model is not None
        assert trainer.diffusion is not None
        assert trainer.optimizer is not None

    def test_trainer_default_diffusion(self, small_score_network):
        """Trainer creates default diffusion if not provided."""
        trainer = Trainer(small_score_network)
        assert trainer.diffusion is not None

    def test_train_step_returns_loss(self, trainer, sample_batch):
        """Train step returns loss value."""
        loss = trainer.train_step(sample_batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_updates_history(self, trainer, sample_dataloader):
        """Train epoch updates loss history."""
        initial_len = len(trainer.history["train_loss"])
        trainer.train_epoch(sample_dataloader)
        assert len(trainer.history["train_loss"]) == initial_len + 1

    def test_evaluate_returns_loss(self, trainer, sample_dataloader):
        """Evaluate returns validation loss."""
        loss = trainer.evaluate(sample_dataloader)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_multiple_epochs(self, trainer, sample_dataloader):
        """Training for multiple epochs works."""
        history = trainer.train(sample_dataloader, epochs=3, verbose=False)
        assert len(history["train_loss"]) == 3

    def test_loss_weighting_option(self, small_score_network, diffusion):
        """Trainer accepts loss_weighting parameter."""
        trainer = Trainer(
            small_score_network,
            diffusion,
            loss_weighting="sigma",
        )
        assert trainer.loss_weighting == "sigma"
