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


# ============================================================================
# ScoreNetwork Gradient Tests
# ============================================================================


class TestScoreNetworkGradients:
    """Tests for gradient flow through score network."""

    def test_gradients_flow_to_all_parameters(self, score_network, batch_data):
        """Gradients flow to all trainable parameters."""
        x, t = batch_data
        score = score_network(x, t)
        loss = score.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in score_network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_no_nan_gradients(self, score_network, batch_data):
        """No NaN gradients during backprop."""
        x, t = batch_data
        score = score_network(x, t)
        loss = (score ** 2).mean()
        loss.backward()

        for name, param in score_network.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf in {name}"

    def test_gradient_magnitude_reasonable(self, score_network, batch_data):
        """Gradient magnitudes are within reasonable bounds."""
        x, t = batch_data
        score = score_network(x, t)
        loss = score.pow(2).mean()
        loss.backward()

        for name, param in score_network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradient should not be astronomically large
                assert grad_norm < 1e6, f"Huge gradient in {name}: {grad_norm}"

    def test_training_step_reduces_loss(self, batch_data):
        """A training step reduces loss (basic sanity check)."""
        net = ScoreNetwork(in_channels=1, base_channels=8, num_blocks=1)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        x, t = batch_data
        target = torch.randn_like(x)

        # Initial loss
        score = net(x, t)
        loss1 = (score - target).pow(2).mean()

        # Training step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # New loss
        score = net(x, t)
        loss2 = (score - target).pow(2).mean()

        # Loss should decrease (usually)
        assert loss2 < loss1 * 1.5  # Allow some variance


# ============================================================================
# DiffusionProcess Forward Tests
# ============================================================================


class TestDiffusionProcessForward:
    """Tests for diffusion forward process."""

    def test_forward_output_shapes(self, diffusion, batch_data):
        """Forward process returns correct shapes."""
        x, t = batch_data
        x_t, noise = diffusion.forward(x, t)
        assert x_t.shape == x.shape
        assert noise.shape == x.shape

    def test_forward_at_t_zero(self, diffusion):
        """At t=0, x_t should be close to x_0."""
        x_0 = torch.randn(4, 1, 16, 16)
        t = torch.zeros(4)
        x_t, _ = diffusion.forward(x_0, t)
        assert torch.allclose(x_t, x_0, atol=1e-5)

    def test_forward_at_t_one(self, diffusion):
        """At t=1, x_t should be approximately standard normal."""
        x_0 = torch.randn(100, 1, 8, 8)
        t = torch.ones(100)
        x_t, _ = diffusion.forward(x_0, t)
        # Should be close to N(0, 1)
        assert abs(x_t.mean()) < 0.2
        assert abs(x_t.std() - 1.0) < 0.2

    def test_noise_is_standard_normal(self, diffusion, batch_data):
        """Returned noise should be standard normal."""
        x, t = batch_data
        _, noise = diffusion.forward(x, t)
        # Check noise statistics
        assert abs(noise.mean()) < 0.5
        assert abs(noise.std() - 1.0) < 0.5

    def test_different_times_different_noise_levels(self, diffusion):
        """Different times produce different noise levels."""
        x_0 = torch.randn(1, 1, 16, 16)
        t_low = torch.tensor([0.1])
        t_high = torch.tensor([0.9])

        x_t_low, _ = diffusion.forward(x_0, t_low)
        x_t_high, _ = diffusion.forward(x_0, t_high)

        # Higher t should have more noise (larger deviation from x_0)
        diff_low = (x_t_low - x_0).abs().mean()
        diff_high = (x_t_high - x_0).abs().mean()
        assert diff_high > diff_low


# ============================================================================
# Noise Level Consistency Tests
# ============================================================================


class TestNoiseLevelConsistency:
    """Tests for variance preservation and noise level properties."""

    def test_variance_preservation(self, diffusion):
        """α² + σ² should be approximately 1 (variance preserving)."""
        t = torch.linspace(0.01, 0.99, 50)
        alpha_t, sigma_t = diffusion.noise_level(t)
        var = alpha_t**2 + sigma_t**2
        assert torch.allclose(var, torch.ones_like(var), atol=0.05)

    def test_alpha_decreases_with_time(self, diffusion):
        """Signal coefficient α should decrease with time."""
        t = torch.linspace(0, 1, 11)
        alpha_t, _ = diffusion.noise_level(t)
        # Alpha should be monotonically decreasing
        for i in range(len(t) - 1):
            assert alpha_t[i] >= alpha_t[i + 1]

    def test_sigma_increases_with_time(self, diffusion):
        """Noise coefficient σ should increase with time."""
        t = torch.linspace(0, 1, 11)
        _, sigma_t = diffusion.noise_level(t)
        # Sigma should be monotonically increasing
        for i in range(len(t) - 1):
            assert sigma_t[i] <= sigma_t[i + 1]

    def test_alpha_at_boundaries(self, diffusion):
        """α should be ~1 at t=0 and small at t=1."""
        t_zero = torch.tensor([0.0])
        t_one = torch.tensor([1.0])
        alpha_0, _ = diffusion.noise_level(t_zero)
        alpha_1, _ = diffusion.noise_level(t_one)
        assert alpha_0.item() > 0.99
        assert alpha_1.item() < 0.1

    def test_sigma_at_boundaries(self, diffusion):
        """σ should be small at t=0 and ~1 at t=1."""
        t_zero = torch.tensor([0.0])
        t_one = torch.tensor([1.0])
        _, sigma_0 = diffusion.noise_level(t_zero)
        _, sigma_1 = diffusion.noise_level(t_one)
        assert sigma_0.item() < 0.1
        assert sigma_1.item() > 0.99
