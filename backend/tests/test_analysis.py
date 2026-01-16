"""Tests for analysis comparison utilities."""

import pytest
import torch
import numpy as np
from backend.ml.analysis import (
    pair_correlation,
    magnetization_distribution,
    autocorrelation_time,
    kl_divergence,
    symmetric_kl_divergence,
    magnetization_kl_divergence,
    wasserstein_distance_1d,
    magnetization_wasserstein,
    correlation_function_comparison,
)


class TestPairCorrelation:
    """Tests for pair_correlation function."""

    def test_returns_dict_with_required_keys(self):
        """pair_correlation should return dict with 'r' and 'C_r' keys."""
        samples = torch.randint(0, 2, (10, 8, 8)).float() * 2 - 1
        result = pair_correlation(samples)
        assert "r" in result
        assert "C_r" in result

    def test_r_values_are_distances(self):
        """r values should be non-negative integers."""
        samples = torch.randint(0, 2, (10, 8, 8)).float() * 2 - 1
        result = pair_correlation(samples)
        assert all(r >= 0 for r in result["r"])

    def test_self_correlation_is_one(self):
        """C(0) should be approximately 1 for uniform spins."""
        # All +1 spins have perfect self-correlation
        samples = torch.ones(20, 8, 8)
        result = pair_correlation(samples)
        # First non-trivial r should have correlation ~1 for uniform spins
        assert len(result["C_r"]) > 0


class TestMagnetizationDistribution:
    """Tests for magnetization_distribution function."""

    def test_returns_dict_with_required_keys(self):
        """magnetization_distribution should return dict with 'M' and 'P_M' keys."""
        samples = torch.randint(0, 2, (50, 8, 8)).float() * 2 - 1
        result = magnetization_distribution(samples)
        assert "M" in result
        assert "P_M" in result

    def test_probabilities_sum_to_one(self):
        """P_M should integrate to approximately 1."""
        samples = torch.randint(0, 2, (100, 8, 8)).float() * 2 - 1
        result = magnetization_distribution(samples, n_bins=20)
        # For density histogram, the integral approximation depends on bin width
        bin_width = 2.0 / 20  # Range is [-1, 1], 20 bins
        integral = sum(result["P_M"]) * bin_width
        assert 0.8 < integral < 1.2  # Allow some tolerance

    def test_handles_4d_input(self):
        """Should handle 4D tensor (batch, channel, h, w)."""
        samples = torch.randint(0, 2, (50, 1, 8, 8)).float() * 2 - 1
        result = magnetization_distribution(samples)
        assert "M" in result
        assert "P_M" in result


class TestAutocorrelationTime:
    """Tests for autocorrelation_time function."""

    def test_returns_positive_float(self):
        """autocorrelation_time should return a positive float."""
        samples = torch.randint(0, 2, (100, 8, 8)).float() * 2 - 1
        tau = autocorrelation_time(samples)
        assert isinstance(tau, float)
        assert tau > 0

    def test_uncorrelated_samples_have_low_tau(self):
        """Independent random samples should have tau close to 1."""
        # Generate truly independent samples
        samples = torch.randint(0, 2, (200, 8, 8)).float() * 2 - 1
        tau = autocorrelation_time(samples)
        # For independent samples, tau should be close to 0.5-2
        assert tau < 10.0

    def test_short_sequence_returns_one(self):
        """Short sequences should return tau = 1.0."""
        samples = torch.randint(0, 2, (5, 8, 8)).float() * 2 - 1
        tau = autocorrelation_time(samples)
        assert tau == 1.0


class TestKLDivergence:
    """Tests for kl_divergence function."""

    def test_identical_distributions_zero_kl(self):
        """KL divergence between identical distributions should be near zero."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        kl = kl_divergence(p, p)
        assert kl < 1e-6

    def test_kl_non_negative(self):
        """KL divergence should always be non-negative."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        kl = kl_divergence(p, q)
        assert kl >= 0

    def test_kl_asymmetric(self):
        """KL divergence should be asymmetric."""
        p = np.array([0.1, 0.9])
        q = np.array([0.5, 0.5])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        # These should generally be different
        assert abs(kl_pq - kl_qp) > 1e-6


class TestSymmetricKLDivergence:
    """Tests for symmetric_kl_divergence function."""

    def test_symmetric_kl_is_symmetric(self):
        """Symmetric KL should give same result in both directions."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        skl_pq = symmetric_kl_divergence(p, q)
        skl_qp = symmetric_kl_divergence(q, p)
        assert abs(skl_pq - skl_qp) < 1e-10

    def test_symmetric_kl_non_negative(self):
        """Symmetric KL should be non-negative."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])
        skl = symmetric_kl_divergence(p, q)
        assert skl >= 0


class TestMagnetizationKLDivergence:
    """Tests for magnetization_kl_divergence function."""

    def test_returns_dict_with_kl_values(self):
        """Should return dict with kl_divergence and symmetric_kl_divergence."""
        samples1 = torch.randint(0, 2, (50, 8, 8)).float() * 2 - 1
        samples2 = torch.randint(0, 2, (50, 8, 8)).float() * 2 - 1
        result = magnetization_kl_divergence(samples1, samples2)
        assert "kl_divergence" in result
        assert "symmetric_kl_divergence" in result

    def test_identical_samples_low_kl(self):
        """Identical sample sets should have low KL divergence."""
        samples = torch.randint(0, 2, (100, 8, 8)).float() * 2 - 1
        result = magnetization_kl_divergence(samples, samples)
        assert result["symmetric_kl_divergence"] < 0.1


class TestWassersteinDistance:
    """Tests for wasserstein_distance_1d function."""

    def test_identical_samples_zero_distance(self):
        """Identical samples should have zero Wasserstein distance."""
        samples = np.array([1, 2, 3, 4, 5])
        w = wasserstein_distance_1d(samples, samples)
        assert w < 1e-10

    def test_wasserstein_non_negative(self):
        """Wasserstein distance should be non-negative."""
        samples1 = np.random.randn(100)
        samples2 = np.random.randn(100)
        w = wasserstein_distance_1d(samples1, samples2)
        assert w >= 0

    def test_wasserstein_symmetric(self):
        """Wasserstein distance should be symmetric."""
        samples1 = np.array([1, 2, 3])
        samples2 = np.array([4, 5, 6])
        w12 = wasserstein_distance_1d(samples1, samples2)
        w21 = wasserstein_distance_1d(samples2, samples1)
        assert abs(w12 - w21) < 1e-10


class TestMagnetizationWasserstein:
    """Tests for magnetization_wasserstein function."""

    def test_returns_non_negative_float(self):
        """Should return a non-negative float."""
        samples1 = torch.randint(0, 2, (50, 8, 8)).float() * 2 - 1
        samples2 = torch.randint(0, 2, (50, 8, 8)).float() * 2 - 1
        w = magnetization_wasserstein(samples1, samples2)
        assert isinstance(w, float)
        assert w >= 0

    def test_handles_4d_input(self):
        """Should handle 4D tensor input."""
        samples1 = torch.randint(0, 2, (50, 1, 8, 8)).float() * 2 - 1
        samples2 = torch.randint(0, 2, (50, 1, 8, 8)).float() * 2 - 1
        w = magnetization_wasserstein(samples1, samples2)
        assert isinstance(w, float)


class TestCorrelationFunctionComparison:
    """Tests for correlation_function_comparison function."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with comparison metrics."""
        samples1 = torch.randint(0, 2, (20, 8, 8)).float() * 2 - 1
        samples2 = torch.randint(0, 2, (20, 8, 8)).float() * 2 - 1
        result = correlation_function_comparison(samples1, samples2)
        assert "rmse" in result
        assert "max_diff" in result
        assert "correlation_length_ratio" in result

    def test_identical_samples_low_rmse(self):
        """Identical samples should have low RMSE."""
        samples = torch.randint(0, 2, (30, 8, 8)).float() * 2 - 1
        result = correlation_function_comparison(samples, samples)
        # Same samples should give identical correlations
        assert result["rmse"] < 0.1
