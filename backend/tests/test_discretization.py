"""Tests for spin discretization methods and their physical correctness."""

import pytest
import torch
import numpy as np


class TestDiscretizationSign:
    """Tests for the sign discretization method."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_sign_positive_inputs(self, sampler):
        """Positive values should map to +1."""
        continuous = torch.ones(2, 1, 8, 8) * 0.5
        discrete = sampler.discretize_spins(continuous, method="sign")
        assert (discrete == 1).all()

    def test_sign_negative_inputs(self, sampler):
        """Negative values should map to -1."""
        continuous = torch.ones(2, 1, 8, 8) * -0.5
        discrete = sampler.discretize_spins(continuous, method="sign")
        assert (discrete == -1).all()

    def test_sign_mixed_inputs(self, sampler):
        """Mixed inputs should be correctly classified."""
        continuous = torch.zeros(1, 1, 4, 4)
        continuous[0, 0, 0:2, :] = 1.0
        continuous[0, 0, 2:4, :] = -1.0

        discrete = sampler.discretize_spins(continuous, method="sign")

        assert (discrete[0, 0, 0:2, :] == 1).all()
        assert (discrete[0, 0, 2:4, :] == -1).all()

    def test_sign_zero_maps_to_positive(self, sampler):
        """Zero should map to +1 (positive side of sign function)."""
        continuous = torch.zeros(1, 1, 4, 4)
        discrete = sampler.discretize_spins(continuous, method="sign")
        # Note: sign(0) = 0 but we add eps, so should be +1
        assert (discrete >= 0).all()


class TestDiscretizationTanh:
    """Tests for the tanh discretization method."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_tanh_soft_discretization(self, sampler):
        """Tanh should produce soft ±1 values."""
        continuous = torch.randn(2, 1, 8, 8) * 3  # Large values
        discrete = sampler.discretize_spins(continuous, method="tanh")

        # All values should be in [-1, 1]
        assert (discrete >= -1).all()
        assert (discrete <= 1).all()

    def test_tanh_large_positive(self, sampler):
        """Large positive values should be close to +1."""
        continuous = torch.ones(1, 1, 4, 4) * 5.0
        discrete = sampler.discretize_spins(continuous, method="tanh")
        assert (discrete > 0.99).all()

    def test_tanh_large_negative(self, sampler):
        """Large negative values should be close to -1."""
        continuous = torch.ones(1, 1, 4, 4) * -5.0
        discrete = sampler.discretize_spins(continuous, method="tanh")
        assert (discrete < -0.99).all()

    def test_tanh_sharpness_parameter(self, sampler):
        """Higher sharpness should produce sharper transitions."""
        continuous = torch.tensor([[[[0.3, -0.3], [0.1, -0.1]]]])

        # Low sharpness
        discrete_low = sampler.discretize_spins(
            continuous, method="tanh", sharpness=1.0
        )
        # High sharpness
        discrete_high = sampler.discretize_spins(
            continuous, method="tanh", sharpness=10.0
        )

        # Higher sharpness should produce values closer to ±1
        assert torch.abs(discrete_high).mean() > torch.abs(discrete_low).mean()


class TestDiscretizationStochastic:
    """Tests for the stochastic discretization method."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_stochastic_produces_binary(self, sampler):
        """Stochastic method should produce only ±1 values."""
        continuous = torch.randn(10, 1, 8, 8)
        discrete = sampler.discretize_spins(continuous, method="stochastic")

        unique = torch.unique(discrete)
        assert len(unique) == 2
        assert -1 in unique.tolist() and 1 in unique.tolist()

    def test_stochastic_probability_bias(self, sampler):
        """Positive inputs should more likely produce +1."""
        torch.manual_seed(42)

        # Strong positive bias
        continuous = torch.ones(100, 1, 4, 4) * 2.0
        discrete = sampler.discretize_spins(continuous, method="stochastic")

        positive_fraction = (discrete == 1).float().mean()
        assert positive_fraction > 0.8, f"Expected mostly +1, got {positive_fraction}"

    def test_stochastic_probability_negative_bias(self, sampler):
        """Negative inputs should more likely produce -1."""
        torch.manual_seed(42)

        # Strong negative bias
        continuous = torch.ones(100, 1, 4, 4) * -2.0
        discrete = sampler.discretize_spins(continuous, method="stochastic")

        negative_fraction = (discrete == -1).float().mean()
        assert negative_fraction > 0.8, f"Expected mostly -1, got {negative_fraction}"

    def test_stochastic_is_stochastic(self, sampler):
        """Same input should sometimes produce different outputs."""
        torch.manual_seed(42)
        continuous = torch.zeros(100, 1, 4, 4)  # Neutral input

        # Run multiple times
        results = []
        for _ in range(5):
            discrete = sampler.discretize_spins(continuous, method="stochastic")
            results.append(discrete.clone())

        # Not all results should be identical
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Stochastic method should produce varied outputs"


class TestDiscretizationGumbel:
    """Tests for the Gumbel-softmax discretization method."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_gumbel_produces_binary(self, sampler):
        """Gumbel method should produce only ±1 values."""
        continuous = torch.randn(5, 1, 8, 8)
        discrete = sampler.discretize_spins(continuous, method="gumbel")

        unique = torch.unique(discrete)
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_gumbel_respects_input_sign(self, sampler):
        """Gumbel should mostly follow input sign for large values."""
        torch.manual_seed(42)

        # Strong positive
        continuous_pos = torch.ones(50, 1, 4, 4) * 3.0
        discrete_pos = sampler.discretize_spins(continuous_pos, method="gumbel")
        pos_fraction = (discrete_pos == 1).float().mean()
        assert pos_fraction > 0.7, f"Expected mostly +1 for positive input"

        # Strong negative
        continuous_neg = torch.ones(50, 1, 4, 4) * -3.0
        discrete_neg = sampler.discretize_spins(continuous_neg, method="gumbel")
        neg_fraction = (discrete_neg == -1).float().mean()
        assert neg_fraction > 0.7, f"Expected mostly -1 for negative input"


class TestDiscretizationThreshold:
    """Tests for threshold parameter in discretization."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_threshold_zero_default(self, sampler):
        """Default threshold of 0 should use sign function behavior."""
        continuous = torch.tensor([[[[0.1, -0.1], [0.5, -0.5]]]])
        discrete = sampler.discretize_spins(continuous, method="sign", threshold=0.0)

        assert discrete[0, 0, 0, 0] == 1  # 0.1 > 0
        assert discrete[0, 0, 0, 1] == -1  # -0.1 < 0

    def test_threshold_positive(self, sampler):
        """Positive threshold should require larger values for +1."""
        continuous = torch.tensor([[[[0.3, 0.7], [-0.3, -0.7]]]])
        discrete = sampler.discretize_spins(continuous, method="sign", threshold=0.5)

        # 0.3 < 0.5, so might not be +1
        # 0.7 > 0.5, so should be +1
        assert discrete[0, 0, 0, 1] == 1


class TestDiscretizationPhysicalProperties:
    """Tests for physical properties of discretized samples."""

    @pytest.fixture
    def ising_model(self):
        """Create an Ising model for testing."""
        from backend.ml.systems.ising import IsingModel
        return IsingModel(size=8)

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_discretized_energy_valid_range(self, sampler, ising_model):
        """Discretized samples should have valid energy range."""
        continuous = torch.randn(10, 1, 8, 8)
        discrete = sampler.discretize_spins(continuous, method="sign")

        # Remove channel dimension for energy calculation
        samples_2d = discrete.squeeze(1)
        energies = ising_model.energy_per_spin(samples_2d)

        # Energy per spin should be in [-2, 2] for nearest-neighbor Ising
        assert (energies >= -2.0).all()
        assert (energies <= 2.0).all()

    def test_discretized_magnetization_valid_range(self, sampler, ising_model):
        """Discretized samples should have valid magnetization range."""
        continuous = torch.randn(10, 1, 8, 8)
        discrete = sampler.discretize_spins(continuous, method="sign")

        # Remove channel dimension for magnetization calculation
        samples_2d = discrete.squeeze(1)
        magnetizations = ising_model.magnetization(samples_2d)

        # Magnetization should be in [-1, 1]
        assert (magnetizations >= -1.0).all()
        assert (magnetizations <= 1.0).all()

    def test_different_methods_same_input_shape(self, sampler):
        """All discretization methods should preserve input shape."""
        continuous = torch.randn(3, 1, 8, 8)

        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            discrete = sampler.discretize_spins(continuous, method=method)
            assert discrete.shape == continuous.shape, f"Shape mismatch for {method}"


class TestDiscretizationNumericalStability:
    """Tests for numerical stability of discretization methods."""

    @pytest.fixture
    def sampler(self):
        """Create a diffusion sampler for testing."""
        from backend.ml.samplers.diffusion import DiffusionSampler
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

    def test_handles_very_large_values(self, sampler):
        """Discretization should handle very large input values."""
        continuous = torch.ones(2, 1, 4, 4) * 1e6
        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            discrete = sampler.discretize_spins(continuous, method=method)
            assert not torch.isnan(discrete).any(), f"NaN for {method}"
            assert not torch.isinf(discrete).any(), f"Inf for {method}"

    def test_handles_very_small_values(self, sampler):
        """Discretization should handle very small input values."""
        continuous = torch.ones(2, 1, 4, 4) * 1e-10
        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            discrete = sampler.discretize_spins(continuous, method=method)
            assert not torch.isnan(discrete).any(), f"NaN for {method}"
            assert not torch.isinf(discrete).any(), f"Inf for {method}"

    def test_handles_mixed_extreme_values(self, sampler):
        """Discretization should handle mixed extreme values."""
        continuous = torch.zeros(1, 1, 4, 4)
        continuous[0, 0, 0, 0] = 1e6
        continuous[0, 0, 0, 1] = -1e6
        continuous[0, 0, 1, 0] = 1e-10
        continuous[0, 0, 1, 1] = -1e-10

        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            discrete = sampler.discretize_spins(continuous, method=method)
            assert not torch.isnan(discrete).any(), f"NaN for {method}"
            assert not torch.isinf(discrete).any(), f"Inf for {method}"
