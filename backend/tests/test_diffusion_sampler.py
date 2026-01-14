"""Tests for DiffusionSampler."""

import pytest
import torch
import numpy as np
from backend.ml.samplers.diffusion import DiffusionSampler, PretrainedDiffusionSampler


class TestDiffusionSamplerInit:
    """Tests for DiffusionSampler initialization."""

    def test_init_with_model_and_diffusion(self):
        """Sampler should initialize with model and diffusion process."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        sampler = DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=50,
        )

        assert sampler.model is model
        assert sampler.diffusion is diffusion
        assert sampler.lattice_size == 8
        assert sampler.num_steps == 50

    def test_init_default_num_steps(self):
        """Sampler should use default 100 steps if not specified."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        sampler = DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
        )

        assert sampler.num_steps == 100


class TestDiffusionSamplerSample:
    """Tests for the sample method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=20,
        )

    def test_sample_returns_tensor(self, diffusion_sampler):
        """Sample should return a tensor."""
        samples = diffusion_sampler.sample(batch_size=2)
        assert isinstance(samples, torch.Tensor)

    def test_sample_correct_shape(self, diffusion_sampler):
        """Sample should return correct shape with channel dimension."""
        samples = diffusion_sampler.sample(batch_size=4)
        assert samples.shape == (4, 1, 8, 8)

    def test_sample_batch_size_one(self, diffusion_sampler):
        """Sample should work with batch_size=1."""
        samples = diffusion_sampler.sample(batch_size=1)
        assert samples.shape == (1, 1, 8, 8)


class TestDiffusionSamplerDiscretize:
    """Tests for spin discretization methods."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=10,
        )

    def test_discretize_sign_method(self, diffusion_sampler):
        """Sign method should produce ±1 values."""
        continuous = torch.randn(2, 1, 8, 8)
        discrete = diffusion_sampler.discretize_spins(continuous, method="sign")

        unique = torch.unique(discrete)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_discretize_tanh_method(self, diffusion_sampler):
        """Tanh method should produce values close to ±1."""
        continuous = torch.randn(2, 1, 8, 8) * 2  # Scale for clear separation
        discrete = diffusion_sampler.discretize_spins(continuous, method="tanh")

        # Values should be mostly near ±1
        assert (torch.abs(discrete) > 0.5).all()

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
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=10,
        )

    def test_sample_ising_returns_discrete(self, diffusion_sampler):
        """sample_ising should return discrete ±1 values."""
        samples = diffusion_sampler.sample_ising(batch_size=2)
        unique = torch.unique(samples)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_sample_ising_correct_shape(self, diffusion_sampler):
        """sample_ising should return correct shape."""
        samples = diffusion_sampler.sample_ising(batch_size=3)
        assert samples.shape == (3, 1, 8, 8)

    def test_sample_ising_different_methods(self, diffusion_sampler):
        """sample_ising should accept different discretization methods."""
        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            samples = diffusion_sampler.sample_ising(batch_size=2, method=method)
            assert samples.shape == (2, 1, 8, 8)


class TestDiffusionSamplerTrajectory:
    """Tests for sample_trajectory method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=20,
        )

    def test_trajectory_is_generator(self, diffusion_sampler):
        """sample_trajectory should return a generator."""
        from types import GeneratorType

        trajectory = diffusion_sampler.sample_trajectory(batch_size=1)
        assert isinstance(trajectory, GeneratorType)

    def test_trajectory_yields_tensors(self, diffusion_sampler):
        """Trajectory should yield tensors."""
        for frame in diffusion_sampler.sample_trajectory(batch_size=1, yield_every=5):
            assert isinstance(frame, torch.Tensor)
            break  # Just check first frame

    def test_trajectory_yield_count(self, diffusion_sampler):
        """Trajectory should yield correct number of frames."""
        # With num_steps=20 and yield_every=5, should get 4 frames
        frames = list(diffusion_sampler.sample_trajectory(batch_size=1, yield_every=5))
        expected = 20 // 5  # 4 frames
        assert len(frames) == expected


class TestDiffusionSamplerTrajectoryWithMetadata:
    """Tests for sample_trajectory_with_metadata method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=20,
        )

    def test_trajectory_metadata_yields_tuples(self, diffusion_sampler):
        """sample_trajectory_with_metadata should yield (step, time, state) tuples."""
        for result in diffusion_sampler.sample_trajectory_with_metadata(
            batch_size=1, num_frames=3
        ):
            assert isinstance(result, tuple)
            assert len(result) == 3
            step, time, state = result
            assert isinstance(step, int)
            assert isinstance(time, float)
            assert isinstance(state, torch.Tensor)
            break

    def test_trajectory_metadata_frame_count(self, diffusion_sampler):
        """Should yield exactly num_frames frames."""
        frames = list(
            diffusion_sampler.sample_trajectory_with_metadata(batch_size=1, num_frames=5)
        )
        assert len(frames) == 5

    def test_trajectory_metadata_time_decreases(self, diffusion_sampler):
        """Time should decrease from 1 to 0 along trajectory."""
        frames = list(
            diffusion_sampler.sample_trajectory_with_metadata(batch_size=1, num_frames=5)
        )
        times = [f[1] for f in frames]
        # Times should be decreasing (or at least non-increasing)
        for i in range(len(times) - 1):
            assert times[i] >= times[i + 1]


class TestDiffusionSamplerODE:
    """Tests for sample_ode method (deterministic sampling)."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=10,
        )

    def test_sample_ode_returns_tensor(self, diffusion_sampler):
        """sample_ode should return a tensor."""
        samples = diffusion_sampler.sample_ode(batch_size=2)
        assert isinstance(samples, torch.Tensor)

    def test_sample_ode_correct_shape(self, diffusion_sampler):
        """sample_ode should return correct shape."""
        samples = diffusion_sampler.sample_ode(batch_size=3)
        assert samples.shape == (3, 1, 8, 8)

    def test_sample_ode_different_solvers(self, diffusion_sampler):
        """sample_ode should support different ODE solvers."""
        for solver in ["euler", "heun", "rk4"]:
            samples = diffusion_sampler.sample_ode(batch_size=2, solver=solver)
            assert samples.shape == (2, 1, 8, 8)


class TestDiffusionSamplerPredictorCorrector:
    """Tests for sample_predictor_corrector method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
            num_steps=10,
        )

    def test_predictor_corrector_returns_tensor(self, diffusion_sampler):
        """sample_predictor_corrector should return a tensor."""
        samples = diffusion_sampler.sample_predictor_corrector(
            batch_size=2, corrector_steps=2
        )
        assert isinstance(samples, torch.Tensor)

    def test_predictor_corrector_correct_shape(self, diffusion_sampler):
        """sample_predictor_corrector should return correct shape."""
        samples = diffusion_sampler.sample_predictor_corrector(
            batch_size=3, corrector_steps=1
        )
        assert samples.shape == (3, 1, 8, 8)


class TestDiffusionSamplerStatistics:
    """Tests for compute_sample_statistics method."""

    @pytest.fixture
    def diffusion_sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess

        model = ScoreNetwork(lattice_size=8)
        diffusion = DiffusionProcess()
        return DiffusionSampler(
            model=model,
            diffusion=diffusion,
            lattice_size=8,
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

        expected_keys = ["mean", "std", "min", "max", "energy_mean", "energy_std"]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"


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
        # Shape should be (batch, channel, size, size)
        assert samples.shape == (3, 1, 16, 16)

    def test_sample_heuristic_discrete_values(self):
        """sample_heuristic should produce discrete ±1 values."""
        sampler = PretrainedDiffusionSampler(lattice_size=8, num_steps=20)
        samples = sampler.sample_heuristic(batch_size=5, temperature=2.27)

        # Squeeze channel dimension for checking values
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
            assert frame.shape == (1, 1, 8, 8)
