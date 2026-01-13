"""Tests for diffusion model components."""

import pytest
import torch

from backend.ml.models import (
    ScoreNetwork,
    SinusoidalTimeEmbedding,
    ConvBlock,
    DiffusionProcess,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    get_schedule,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def score_network():
    """Create a small score network for testing."""
    return ScoreNetwork(
        in_channels=1,
        base_channels=16,
        time_embed_dim=32,
        num_blocks=2,
    )


@pytest.fixture
def diffusion():
    """Create a diffusion process for testing."""
    return DiffusionProcess(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def batch_data():
    """Create sample batch data."""
    batch_size = 4
    size = 16
    x = torch.randn(batch_size, 1, size, size)
    t = torch.rand(batch_size)
    return x, t


@pytest.fixture
def linear_schedule():
    """Create a linear noise schedule."""
    return LinearNoiseSchedule(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def cosine_schedule():
    """Create a cosine noise schedule."""
    return CosineNoiseSchedule(s=0.008)


@pytest.fixture
def sigmoid_schedule():
    """Create a sigmoid noise schedule."""
    return SigmoidNoiseSchedule(beta_min=0.1, beta_max=20.0)


# ============================================================================
# SinusoidalTimeEmbedding Tests
# ============================================================================


class TestSinusoidalTimeEmbedding:
    """Tests for time embedding module."""

    def test_output_shape(self):
        """Embedding output has correct shape."""
        dim = 64
        embed = SinusoidalTimeEmbedding(dim)
        t = torch.rand(8)
        out = embed(t)
        assert out.shape == (8, dim)

    def test_different_times_different_embeddings(self):
        """Different times produce different embeddings."""
        embed = SinusoidalTimeEmbedding(64)
        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        emb1 = embed(t1)
        emb2 = embed(t2)
        assert not torch.allclose(emb1, emb2)

    def test_same_time_same_embedding(self):
        """Same time produces same embedding."""
        embed = SinusoidalTimeEmbedding(64)
        t = torch.tensor([0.5, 0.5])
        emb = embed(t)
        assert torch.allclose(emb[0], emb[1])

    def test_batch_independence(self):
        """Embeddings are computed independently per sample."""
        embed = SinusoidalTimeEmbedding(32)
        t_batch = torch.tensor([0.0, 0.5, 1.0])
        t_single = torch.tensor([0.5])
        emb_batch = embed(t_batch)
        emb_single = embed(t_single)
        assert torch.allclose(emb_batch[1], emb_single[0])

    def test_deterministic(self):
        """Embeddings are deterministic (no randomness)."""
        embed = SinusoidalTimeEmbedding(64)
        t = torch.rand(4)
        emb1 = embed(t)
        emb2 = embed(t)
        assert torch.allclose(emb1, emb2)
