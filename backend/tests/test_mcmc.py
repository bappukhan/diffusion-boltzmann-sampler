"""Tests for Metropolis-Hastings MCMC sampler."""

import pytest
import torch
from backend.ml.samplers.mcmc import MetropolisHastings
from backend.ml.systems.ising import IsingModel


class TestMetropolisHastingsInit:
    """Tests for MetropolisHastings initialization."""

    def test_init_stores_model(self, ising_model):
        """Sampler should store the Ising model reference."""
        sampler = MetropolisHastings(ising_model, temperature=2.0)
        assert sampler.model is ising_model

    def test_init_stores_temperature(self, ising_model):
        """Sampler should store the temperature."""
        sampler = MetropolisHastings(ising_model, temperature=2.5)
        assert sampler.temperature == 2.5

    def test_init_computes_beta(self, ising_model):
        """Sampler should compute inverse temperature beta = 1/T."""
        sampler = MetropolisHastings(ising_model, temperature=2.0)
        assert abs(sampler.beta - 0.5) < 1e-10

    def test_init_high_temperature_low_beta(self, ising_model):
        """High temperature should give low beta."""
        sampler = MetropolisHastings(ising_model, temperature=10.0)
        assert sampler.beta == 0.1

    def test_init_low_temperature_high_beta(self, ising_model):
        """Low temperature should give high beta."""
        sampler = MetropolisHastings(ising_model, temperature=0.5)
        assert sampler.beta == 2.0

    def test_init_zero_temperature_infinite_beta(self, ising_model):
        """Zero temperature should give infinite beta."""
        sampler = MetropolisHastings(ising_model, temperature=0.0)
        assert sampler.beta == float("inf")


class TestMetropolisHastingsStep:
    """Tests for single Metropolis step."""

    def test_step_returns_tensor(self, mcmc_sampler, random_spins):
        """Step should return a tensor."""
        result = mcmc_sampler.step(random_spins.clone())
        assert isinstance(result, torch.Tensor)

    def test_step_preserves_shape(self, mcmc_sampler, random_spins):
        """Step should preserve spin configuration shape."""
        spins = random_spins.clone()
        result = mcmc_sampler.step(spins)
        assert result.shape == random_spins.shape

    def test_step_preserves_spin_values(self, mcmc_sampler, random_spins):
        """Step should only produce ±1 values."""
        spins = random_spins.clone()
        for _ in range(100):
            spins = mcmc_sampler.step(spins)
        unique = torch.unique(spins)
        assert len(unique) <= 2
        assert all(v in [-1, 1] for v in unique.tolist())

    def test_step_modifies_at_most_one_spin(self, mcmc_sampler, random_spins):
        """Single step should flip at most one spin."""
        spins_before = random_spins.clone()
        spins_after = mcmc_sampler.step(spins_before.clone())
        diff = (spins_before != spins_after).sum()
        assert diff <= 1


class TestEnergyLoweringAcceptance:
    """Tests for energy-lowering move acceptance (always accepted)."""

    def test_energy_lowering_move_accepted(self, ising_model):
        """Energy-lowering moves should always be accepted at any temperature."""
        # Create checkerboard (high energy state)
        spins = torch.ones(8, 8)
        spins[::2, 1::2] = -1
        spins[1::2, ::2] = -1

        # At T=0, only energy-lowering moves are accepted
        sampler = MetropolisHastings(ising_model, temperature=0.01)

        # Track energy over many steps
        initial_energy = ising_model.energy(spins).item()
        for _ in range(1000):
            spins = sampler.step(spins)
            current_energy = ising_model.energy(spins).item()
            # Energy should never increase significantly at low T
            assert current_energy <= initial_energy + 1e-6

    def test_low_temperature_converges_to_ground_state(self, ising_model):
        """At very low temperature, system should converge to ground state."""
        # Start from random configuration
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        sampler = MetropolisHastings(ising_model, temperature=0.1)

        # Run many sweeps
        for _ in range(200):
            spins = sampler.sweep(spins)

        # Should be close to ground state (all +1 or all -1)
        mag = abs(ising_model.magnetization(spins).item())
        assert mag > 0.9, f"Expected |M| > 0.9, got {mag}"
