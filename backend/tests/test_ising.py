"""Tests for Ising model implementation."""

import pytest
import torch
from backend.ml.systems.ising import IsingModel


class TestIsingEnergy:
    """Tests for energy computation."""

    def test_ground_state_energy(self, ising_model, all_up_spins):
        """Ground state should have minimum energy."""
        E = ising_model.energy(all_up_spins)
        # Each spin has 2 unique bonds (right and down)
        # E = -J * 2 * N = -2 * 8 * 8 = -128
        expected = -2 * 8 * 8
        assert abs(E.item() - expected) < 1e-6

    def test_energy_flip_symmetry(self, ising_model, all_up_spins, all_down_spins):
        """All-up and all-down should have same energy."""
        E_up = ising_model.energy(all_up_spins)
        E_down = ising_model.energy(all_down_spins)
        assert torch.allclose(E_up, E_down)

    def test_checkerboard_energy(self, ising_model, checkerboard_spins):
        """Checkerboard should have maximum energy (all anti-aligned)."""
        E = ising_model.energy(checkerboard_spins)
        # All neighbors are anti-aligned: E = +2 * N
        expected = 2 * 8 * 8
        assert abs(E.item() - expected) < 1e-6

    def test_energy_batch(self, ising_model):
        """Energy should work on batches."""
        spins = ising_model.random_configuration(batch_size=4)
        E = ising_model.energy(spins)
        assert E.shape == (4,)


class TestIsingLocalEnergyDiff:
    """Tests for local energy difference computation."""

    def test_local_energy_diff_consistency(self, ising_model, random_spins):
        """Local energy diff should equal full energy change."""
        spins = random_spins.clone()
        i, j = 3, 5

        E_before = ising_model.energy(spins)
        dE = ising_model.local_energy_diff(spins, i, j)

        spins[i, j] *= -1
        E_after = ising_model.energy(spins)

        assert torch.allclose(dE, E_after - E_before, atol=1e-5)

    def test_local_energy_diff_ground_state(self, ising_model, all_up_spins):
        """Flipping spin in ground state should cost energy."""
        dE = ising_model.local_energy_diff(all_up_spins, 0, 0)
        # Each spin has 4 neighbors in 2D
        # Flipping costs 2 * s * J * sum(neighbors) = 2 * 1 * 1 * 4 = 8
        assert dE.item() == 8.0


class TestIsingMagnetization:
    """Tests for magnetization computation."""

    def test_all_up_magnetization(self, ising_model, all_up_spins):
        """All-up should have M = +1."""
        M = ising_model.magnetization(all_up_spins)
        assert abs(M.item() - 1.0) < 1e-6

    def test_all_down_magnetization(self, ising_model, all_down_spins):
        """All-down should have M = -1."""
        M = ising_model.magnetization(all_down_spins)
        assert abs(M.item() - (-1.0)) < 1e-6

    def test_checkerboard_magnetization(self, ising_model, checkerboard_spins):
        """Checkerboard should have M = 0."""
        M = ising_model.magnetization(checkerboard_spins)
        assert abs(M.item()) < 1e-6


class TestIsingRandomConfiguration:
    """Tests for random configuration generation."""

    def test_random_configuration_shape(self, ising_model):
        """Random config should have correct shape."""
        spins = ising_model.random_configuration(batch_size=1)
        assert spins.shape == (1, 8, 8)

    def test_random_configuration_values(self, ising_model):
        """Random config should only contain ±1."""
        spins = ising_model.random_configuration(batch_size=10)
        unique = torch.unique(spins)
        assert len(unique) == 2
        assert -1 in unique
        assert 1 in unique

    def test_random_configuration_batch(self, ising_model):
        """Batch random configs should be independent."""
        spins = ising_model.random_configuration(batch_size=100)
        # With 100 samples, not all should be identical
        assert not torch.all(spins[0] == spins[1])


class TestIsingCriticalTemperature:
    """Tests for critical temperature constant."""

    def test_critical_temperature_value(self):
        """T_c should be approximately 2.269."""
        assert abs(IsingModel.T_CRITICAL - 2.269) < 0.01

    def test_critical_temperature_formula(self):
        """T_c should satisfy 2/ln(1+√2)."""
        expected = 2.0 / torch.log(torch.tensor(1.0 + 2**0.5)).item()
        assert abs(IsingModel.T_CRITICAL - expected) < 1e-6
