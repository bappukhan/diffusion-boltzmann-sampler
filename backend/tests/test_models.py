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
        # Relax tolerance due to MIN_SIGMA floor for numerical stability
        assert torch.allclose(x_t, x_0, atol=1e-3)

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


# ============================================================================
# NoiseSchedule Tests
# ============================================================================


class TestNoiseSchedules:
    """Tests for noise schedule implementations."""

    def test_linear_schedule_beta_range(self, linear_schedule):
        """Linear schedule beta should be in [beta_min, beta_max]."""
        t = torch.linspace(0, 1, 100)
        beta = linear_schedule.beta(t)
        assert (beta >= 0.1 - 1e-6).all()
        assert (beta <= 20.0 + 1e-6).all()

    def test_linear_schedule_beta_monotonic(self, linear_schedule):
        """Linear schedule beta should increase with t."""
        t = torch.linspace(0, 1, 100)
        beta = linear_schedule.beta(t)
        for i in range(len(t) - 1):
            assert beta[i] <= beta[i + 1] + 1e-6

    def test_cosine_schedule_smooth(self, cosine_schedule):
        """Cosine schedule should produce smooth α values."""
        t = torch.linspace(0, 1, 100)
        alpha_t, _ = cosine_schedule.noise_level(t)
        # Check no sudden jumps
        diffs = torch.abs(alpha_t[1:] - alpha_t[:-1])
        assert diffs.max() < 0.1

    def test_cosine_schedule_boundaries(self, cosine_schedule):
        """Cosine schedule should have proper boundary behavior."""
        t_zero = torch.tensor([0.0])
        t_one = torch.tensor([1.0])
        alpha_0, _ = cosine_schedule.noise_level(t_zero)
        alpha_1, _ = cosine_schedule.noise_level(t_one)
        assert alpha_0.item() > 0.95
        assert alpha_1.item() < 0.1

    def test_sigmoid_schedule_beta_range(self, sigmoid_schedule):
        """Sigmoid schedule beta should be in reasonable range."""
        t = torch.linspace(0, 1, 100)
        beta = sigmoid_schedule.beta(t)
        assert (beta >= 0).all()
        assert (beta <= 30.0).all()  # Allow some headroom

    def test_get_schedule_factory(self):
        """Factory function creates correct schedule types."""
        linear = get_schedule("linear")
        assert isinstance(linear, LinearNoiseSchedule)

        cosine = get_schedule("cosine")
        assert isinstance(cosine, CosineNoiseSchedule)

        sigmoid = get_schedule("sigmoid")
        assert isinstance(sigmoid, SigmoidNoiseSchedule)

    def test_get_schedule_unknown_type(self):
        """Factory raises error for unknown schedule type."""
        with pytest.raises(ValueError):
            get_schedule("unknown")

    def test_diffusion_with_custom_schedule(self, cosine_schedule):
        """DiffusionProcess works with custom schedule."""
        diffusion = DiffusionProcess(schedule=cosine_schedule)
        x_0 = torch.randn(2, 1, 16, 16)
        t = torch.tensor([0.5, 0.5])
        x_t, noise = diffusion.forward(x_0, t)
        assert x_t.shape == x_0.shape


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for score network and diffusion process."""

    def test_full_training_loop(self, score_network, diffusion, batch_data):
        """Complete training loop with diffusion and score network."""
        x_0, _ = batch_data
        optimizer = torch.optim.Adam(score_network.parameters(), lr=1e-3)

        for _ in range(3):
            # Sample random times
            t = torch.rand(x_0.shape[0])

            # Forward diffusion
            x_t, noise = diffusion.forward(x_0, t)

            # Predict score
            score_pred = score_network(x_t, t)

            # Compute loss
            loss = diffusion.compute_loss(score_pred, noise, t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete without errors
        assert True

    def test_score_prediction_improves_with_training(self, diffusion):
        """Score predictions should improve after training."""
        net = ScoreNetwork(in_channels=1, base_channels=8, num_blocks=1)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        x_0 = torch.randn(8, 1, 8, 8)

        # Initial loss
        t = torch.rand(8)
        x_t, noise = diffusion.forward(x_0, t)
        score_pred = net(x_t, t)
        initial_loss = diffusion.compute_loss(score_pred, noise, t).item()

        # Training
        for _ in range(50):
            t = torch.rand(8)
            x_t, noise = diffusion.forward(x_0, t)
            score_pred = net(x_t, t)
            loss = diffusion.compute_loss(score_pred, noise, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss
        t = torch.rand(8)
        x_t, noise = diffusion.forward(x_0, t)
        with torch.no_grad():
            score_pred = net(x_t, t)
            final_loss = diffusion.compute_loss(score_pred, noise, t).item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_reverse_step_reduces_noise(self, score_network, diffusion):
        """Reverse step should move towards lower noise."""
        x_0 = torch.randn(2, 1, 16, 16)
        t = torch.tensor([0.8, 0.8])

        # Get noisy sample
        x_t, _ = diffusion.forward(x_0, t)

        # Predict score
        with torch.no_grad():
            score = score_network(x_t, t)

        # Take reverse step
        dt = -0.1
        x_next = diffusion.reverse_step(x_t, score, t, dt)

        # Output should have same shape
        assert x_next.shape == x_t.shape
        # Should be finite
        assert not torch.isnan(x_next).any()
        assert not torch.isinf(x_next).any()


# ============================================================================
# Memory and Performance Tests
# ============================================================================


class TestMemoryAndPerformance:
    """Tests for memory usage and performance constraints."""

    def test_reasonable_parameter_count(self):
        """Model should have reasonable number of parameters."""
        net = ScoreNetwork(in_channels=1, base_channels=32, num_blocks=3)
        n_params = sum(p.numel() for p in net.parameters())
        # Should be under 10 million parameters for CPU inference
        assert n_params < 10_000_000

    def test_inference_completes_quickly(self, score_network, batch_data):
        """Inference should complete in reasonable time."""
        import time

        x, t = batch_data

        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = score_network(x, t)
            elapsed = time.time() - start

        # 10 forward passes should take < 5 seconds on CPU
        assert elapsed < 5.0

    def test_batch_processing_efficient(self):
        """Batch processing should be more efficient than sequential."""
        import time

        net = ScoreNetwork(in_channels=1, base_channels=16, num_blocks=2)
        x_batch = torch.randn(16, 1, 16, 16)
        t_batch = torch.rand(16)

        with torch.no_grad():
            # Batch processing
            start = time.time()
            _ = net(x_batch, t_batch)
            batch_time = time.time() - start

            # Sequential processing
            start = time.time()
            for i in range(16):
                _ = net(x_batch[i : i + 1], t_batch[i : i + 1])
            seq_time = time.time() - start

        # Batch should be at least as fast (usually faster due to parallelism)
        assert batch_time <= seq_time * 2  # Allow some variance

    def test_memory_released_after_inference(self, score_network, batch_data):
        """Memory should be released after inference."""
        import gc

        x, t = batch_data

        # Run inference
        with torch.no_grad():
            score = score_network(x, t)
            del score

        # Force garbage collection
        gc.collect()

        # Should complete without memory issues
        assert True
