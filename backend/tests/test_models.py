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


# ============================================================================
# ConvBlock Tests
# ============================================================================


class TestConvBlock:
    """Tests for convolutional block with time conditioning."""

    def test_output_shape_same_channels(self):
        """Output shape matches input when channels are same."""
        block = ConvBlock(in_ch=32, out_ch=32, time_dim=64)
        x = torch.randn(2, 32, 16, 16)
        t_emb = torch.randn(2, 64)
        out = block(x, t_emb)
        assert out.shape == x.shape

    def test_output_shape_different_channels(self):
        """Output has correct shape when channels differ."""
        block = ConvBlock(in_ch=16, out_ch=32, time_dim=64)
        x = torch.randn(2, 16, 16, 16)
        t_emb = torch.randn(2, 64)
        out = block(x, t_emb)
        assert out.shape == (2, 32, 16, 16)

    def test_time_conditioning_effect(self):
        """Different time embeddings produce different outputs."""
        block = ConvBlock(in_ch=16, out_ch=16, time_dim=32)
        x = torch.randn(1, 16, 8, 8)
        t_emb1 = torch.randn(1, 32)
        t_emb2 = torch.randn(1, 32)
        out1 = block(x, t_emb1)
        out2 = block(x, t_emb2)
        assert not torch.allclose(out1, out2)

    def test_residual_connection(self):
        """Residual connection is functional (output differs from input)."""
        block = ConvBlock(in_ch=16, out_ch=16, time_dim=32)
        x = torch.randn(1, 16, 8, 8)
        t_emb = torch.zeros(1, 32)
        out = block(x, t_emb)
        # Output should not be exactly zero (residual adds input)
        assert out.abs().sum() > 0

    def test_residual_scaling(self):
        """Residual scaling parameter works."""
        block_scaled = ConvBlock(in_ch=16, out_ch=16, time_dim=32, residual_scale=0.1)
        block_normal = ConvBlock(in_ch=16, out_ch=16, time_dim=32, residual_scale=1.0)
        # Both should work without error
        x = torch.randn(1, 16, 8, 8)
        t_emb = torch.randn(1, 32)
        _ = block_scaled(x, t_emb)
        _ = block_normal(x, t_emb)


# ============================================================================
# ScoreNetwork Shape Tests
# ============================================================================


class TestScoreNetworkShape:
    """Tests for score network output shapes."""

    def test_output_shape_matches_input(self, score_network, batch_data):
        """Score output has same shape as input."""
        x, t = batch_data
        score = score_network(x, t)
        assert score.shape == x.shape

    def test_various_input_sizes(self, score_network):
        """Network handles various spatial sizes."""
        for size in [8, 16, 32]:
            x = torch.randn(2, 1, size, size)
            t = torch.rand(2)
            score = score_network(x, t)
            assert score.shape == x.shape

    def test_single_sample(self, score_network):
        """Network works with batch size 1."""
        x = torch.randn(1, 1, 16, 16)
        t = torch.rand(1)
        score = score_network(x, t)
        assert score.shape == x.shape

    def test_large_batch(self, score_network):
        """Network handles larger batches."""
        x = torch.randn(16, 1, 16, 16)
        t = torch.rand(16)
        score = score_network(x, t)
        assert score.shape == x.shape

    def test_non_square_input(self):
        """Network handles non-square inputs (power of 2)."""
        net = ScoreNetwork(in_channels=1, base_channels=16, num_blocks=2)
        x = torch.randn(2, 1, 16, 32)
        t = torch.rand(2)
        score = net(x, t)
        assert score.shape == x.shape
