"""Tests for DiffusionSampler."""

import pytest
import torch
import numpy as np
from backend.ml.samplers.diffusion import DiffusionSampler, PretrainedDiffusionSampler
from backend.ml.models.score_network import ScoreNetwork
from backend.ml.models.diffusion import DiffusionProcess


class TestDiffusionSamplerInit:
    """Tests for DiffusionSampler initialization."""

    def test_init_with_score_network(self):
        """Sampler should initialize with score network."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        sampler = DiffusionSampler(
            score_network=model,
            num_steps=50,
        )
        assert sampler.model is model
        assert sampler.num_steps == 50

    def test_init_with_custom_diffusion(self):
        """Sampler should accept custom diffusion process."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        diffusion = DiffusionProcess(beta_min=0.05, beta_max=15.0)
        sampler = DiffusionSampler(
            score_network=model,
            diffusion=diffusion,
            num_steps=50,
        )
        assert sampler.diffusion is diffusion

    def test_init_default_diffusion(self):
        """Sampler should create default diffusion if not provided."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        sampler = DiffusionSampler(score_network=model)
        assert sampler.diffusion is not None
        assert isinstance(sampler.diffusion, DiffusionProcess)


class TestDiffusionSamplerSample:
    """Tests for the sample method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        return DiffusionSampler(
            score_network=model,
            num_steps=20,
        )

    def test_sample_returns_tensor(self, diffusion_sampler):
        """Sample should return a tensor."""
        samples = diffusion_sampler.sample(shape=(2, 1, 8, 8))
        assert isinstance(samples, torch.Tensor)

    def test_sample_correct_shape(self, diffusion_sampler):
        """Sample should return correct shape."""
        samples = diffusion_sampler.sample(shape=(4, 1, 8, 8))
        assert samples.shape == (4, 1, 8, 8)

    def test_sample_batch_size_one(self, diffusion_sampler):
        """Sample should work with batch_size=1."""
        samples = diffusion_sampler.sample(shape=(1, 1, 8, 8))
        assert samples.shape == (1, 1, 8, 8)


class TestDiffusionSamplerDiscretize:
    """Tests for spin discretization methods."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        return DiffusionSampler(
            score_network=model,
            num_steps=10,
        )

    def test_discretize_sign_method(self, diffusion_sampler):
        """Sign method should produce ±1 values."""
        continuous = torch.randn(2, 1, 8, 8)
        discrete = diffusion_sampler.discretize_spins(continuous, method="sign")
        unique = torch.unique(discrete)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_discretize_tanh_method(self, diffusion_sampler):
        """Tanh method should produce values in [-1, 1]."""
        continuous = torch.randn(2, 1, 8, 8) * 2
        discrete = diffusion_sampler.discretize_spins(continuous, method="tanh")
        assert (discrete >= -1).all()
        assert (discrete <= 1).all()

    def test_discretize_stochastic_method(self, diffusion_sampler):
        """Stochastic method should produce ±1 values."""
        continuous = torch.randn(2, 1, 8, 8)
        discrete = diffusion_sampler.discretize_spins(continuous, method="stochastic")
        unique = torch.unique(discrete)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_discretize_gumbel_method(self, diffusion_sampler):
        """Gumbel method should produce ±1 values."""
        continuous = torch.randn(2, 1, 8, 8)
        discrete = diffusion_sampler.discretize_spins(continuous, method="gumbel")
        unique = torch.unique(discrete)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_discretize_preserves_shape(self, diffusion_sampler):
        """Discretization should preserve tensor shape."""
        continuous = torch.randn(3, 1, 8, 8)
        discrete = diffusion_sampler.discretize_spins(continuous, method="sign")
        assert discrete.shape == continuous.shape


class TestDiffusionSamplerSampleIsing:
    """Tests for sample_ising method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(in_channels=1, base_channels=16, time_embed_dim=32, num_blocks=2)
        return DiffusionSampler(
            score_network=model,
            num_steps=10,
        )

    def test_sample_ising_returns_discrete(self, diffusion_sampler):
        """sample_ising should return discrete ±1 values."""
        samples = diffusion_sampler.sample_ising(shape=(2, 1, 8, 8))
        unique = torch.unique(samples)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_sample_ising_correct_shape(self, diffusion_sampler):
        """sample_ising should return correct shape."""
        samples = diffusion_sampler.sample_ising(shape=(3, 1, 8, 8))
        assert samples.shape == (3, 1, 8, 8)


class TestDiffusionSamplerTrajectory:
    """Tests for sample_with_trajectory method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(lattice_size=8)
        return DiffusionSampler(
            score_network=model,
            num_steps=20,
        )

    def test_trajectory_is_generator(self, diffusion_sampler):
        """sample_with_trajectory should return a generator."""
        from types import GeneratorType
        trajectory = diffusion_sampler.sample_with_trajectory(shape=(1, 1, 8, 8))
        assert isinstance(trajectory, GeneratorType)

    def test_trajectory_yields_tuples(self, diffusion_sampler):
        """Trajectory should yield (tensor, time) tuples."""
        for x, t in diffusion_sampler.sample_with_trajectory(shape=(1, 1, 8, 8), yield_every=5):
            assert isinstance(x, torch.Tensor)
            assert isinstance(t, float)
            break

    def test_trajectory_time_decreases(self, diffusion_sampler):
        """Time values should decrease along trajectory."""
        times = []
        for x, t in diffusion_sampler.sample_with_trajectory(shape=(1, 1, 8, 8), yield_every=5):
            times.append(t)
        # First time should be 1.0, times should generally decrease
        assert times[0] == 1.0
        assert times[-1] < times[0]


class TestDiffusionSamplerODE:
    """Tests for sample_ode method (deterministic sampling)."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(lattice_size=8)
        return DiffusionSampler(
            score_network=model,
            num_steps=10,
        )

    def test_sample_ode_returns_tensor(self, diffusion_sampler):
        """sample_ode should return a tensor."""
        samples = diffusion_sampler.sample_ode(shape=(2, 1, 8, 8))
        assert isinstance(samples, torch.Tensor)

    def test_sample_ode_correct_shape(self, diffusion_sampler):
        """sample_ode should return correct shape."""
        samples = diffusion_sampler.sample_ode(shape=(3, 1, 8, 8))
        assert samples.shape == (3, 1, 8, 8)

    def test_sample_ode_different_solvers(self, diffusion_sampler):
        """sample_ode should support different ODE solvers."""
        for solver in ["euler", "heun", "rk4"]:
            samples = diffusion_sampler.sample_ode(shape=(2, 1, 8, 8), solver=solver)
            assert samples.shape == (2, 1, 8, 8)


class TestDiffusionSamplerPredictorCorrector:
    """Tests for sample_predictor_corrector method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(lattice_size=8)
        return DiffusionSampler(
            score_network=model,
            num_steps=10,
        )

    def test_predictor_corrector_returns_tensor(self, diffusion_sampler):
        """sample_predictor_corrector should return a tensor."""
        samples = diffusion_sampler.sample_predictor_corrector(
            shape=(2, 1, 8, 8), n_corrector_steps=2
        )
        assert isinstance(samples, torch.Tensor)

    def test_predictor_corrector_correct_shape(self, diffusion_sampler):
        """sample_predictor_corrector should return correct shape."""
        samples = diffusion_sampler.sample_predictor_corrector(
            shape=(3, 1, 8, 8), n_corrector_steps=1
        )
        assert samples.shape == (3, 1, 8, 8)


class TestDiffusionSamplerStatistics:
    """Tests for compute_sample_statistics method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        model = ScoreNetwork(lattice_size=8)
        return DiffusionSampler(
            score_network=model,
            num_steps=10,
        )

    def test_statistics_returns_dict(self, diffusion_sampler):
        """compute_sample_statistics should return a dictionary."""
        samples = torch.randn(5, 1, 8, 8)
        stats = diffusion_sampler.compute_sample_statistics(samples)
        assert isinstance(stats, dict)

    def test_statistics_contains_expected_keys(self, diffusion_sampler):
        """Statistics should contain expected keys."""
        samples = torch.randn(5, 1, 8, 8)
        stats = diffusion_sampler.compute_sample_statistics(samples)
        # Check for some expected keys
        assert "mean" in stats or "sample_mean" in stats


class TestPretrainedDiffusionSampler:
    """Tests for PretrainedDiffusionSampler (heuristic mode)."""

    def test_init_creates_sampler(self):
        """PretrainedDiffusionSampler should initialize correctly."""
        sampler = PretrainedDiffusionSampler(lattice_size=16, num_steps=50)
        assert sampler.lattice_size == 16
        assert sampler.num_steps == 50

    def test_sample_heuristic_returns_tensor(self):
        """sample_heuristic should return a tensor."""
        sampler = PretrainedDiffusionSampler(lattice_size=8, num_steps=20)
        samples = sampler.sample_heuristic(batch_size=2, temperature=2.27)
        assert isinstance(samples, torch.Tensor)

    def test_sample_heuristic_correct_shape(self):
        """sample_heuristic should return correct shape."""
        sampler = PretrainedDiffusionSampler(lattice_size=16, num_steps=20)
        samples = sampler.sample_heuristic(batch_size=3, temperature=2.27)
        assert samples.shape == (3, 1, 16, 16)

    def test_sample_heuristic_discrete_values(self):
        """sample_heuristic should produce discrete ±1 values."""
        sampler = PretrainedDiffusionSampler(lattice_size=8, num_steps=20)
        samples = sampler.sample_heuristic(batch_size=5, temperature=2.27)
        samples_2d = samples.squeeze(1)
        unique = torch.unique(samples_2d)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_sample_trajectory_generator(self):
        """sample_trajectory should return a generator."""
        from types import GeneratorType
        sampler = PretrainedDiffusionSampler(lattice_size=8, num_steps=20)
        trajectory = sampler.sample_trajectory(batch_size=1)
        assert isinstance(trajectory, GeneratorType)

    def test_sample_trajectory_yields_frames(self):
        """sample_trajectory should yield tensor frames."""
        sampler = PretrainedDiffusionSampler(lattice_size=8, num_steps=20)
        frames = list(sampler.sample_trajectory(batch_size=1, yield_every=5))
        assert len(frames) > 0
        for frame in frames:
            assert isinstance(frame, torch.Tensor)
