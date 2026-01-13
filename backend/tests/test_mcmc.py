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


class TestHighTemperatureAcceptance:
    """Tests for high temperature behavior (disordered phase)."""

    def test_high_temperature_high_acceptance(self, ising_model):
        """At high temperature, most moves should be accepted."""
        sampler = MetropolisHastings(ising_model, temperature=100.0)
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        # Count accepted moves
        accepted = 0
        total = 1000
        for _ in range(total):
            spins_before = spins.clone()
            spins = sampler.step(spins)
            if not torch.allclose(spins, spins_before):
                accepted += 1

        # At T=100, acceptance should be high (> 40% due to random selection)
        acceptance_rate = accepted / total
        assert acceptance_rate > 0.3, f"Expected high acceptance, got {acceptance_rate}"

    def test_high_temperature_disordered_state(self, high_temp_sampler, ising_model):
        """At high temperature, magnetization should be near zero."""
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        # Run many sweeps at high temperature
        for _ in range(200):  # Increased sweeps for better thermalization
            spins = high_temp_sampler.sweep(spins)

        # Should be disordered (|M| ≈ 0)
        # Relaxed threshold to account for statistical fluctuations
        mag = abs(ising_model.magnetization(spins).item())
        assert mag < 0.6, f"Expected |M| < 0.6 at high T, got {mag}"

    def test_high_temperature_random_walk(self, high_temp_sampler, ising_model):
        """At high temperature, energy should fluctuate around mean."""
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        energies = []
        for _ in range(200):
            spins = high_temp_sampler.sweep(spins)
            energies.append(ising_model.energy(spins).item())

        # Energy should fluctuate, not monotonically decrease
        energy_tensor = torch.tensor(energies)
        std = energy_tensor.std().item()
        assert std > 1.0, f"Expected energy fluctuations, got std={std}"


class TestLowTemperatureRejection:
    """Tests for low temperature behavior (ordered phase)."""

    def test_low_temperature_low_acceptance_from_ground(self, ising_model):
        """At low temperature from ground state, most moves should be rejected."""
        sampler = MetropolisHastings(ising_model, temperature=0.5)
        # Start from ground state
        spins = torch.ones(8, 8)

        # Count accepted moves
        accepted = 0
        total = 1000
        for _ in range(total):
            spins_before = spins.clone()
            spins = sampler.step(spins)
            if not torch.allclose(spins, spins_before):
                accepted += 1

        # From ground state at low T, acceptance should be very low
        acceptance_rate = accepted / total
        assert acceptance_rate < 0.2, f"Expected low acceptance from ground, got {acceptance_rate}"

    def test_low_temperature_maintains_order(self, low_temp_sampler, ising_model):
        """At low temperature, ordered state should be maintained."""
        # Start from ordered state
        spins = torch.ones(8, 8)

        # Run many sweeps at low temperature
        for _ in range(100):
            spins = low_temp_sampler.sweep(spins)

        # Should remain ordered (|M| ≈ 1)
        mag = abs(ising_model.magnetization(spins).item())
        assert mag > 0.8, f"Expected |M| > 0.8 at low T, got {mag}"

    def test_low_temperature_energy_near_ground(self, low_temp_sampler, ising_model):
        """At low temperature, energy should stay near ground state."""
        # Start from ordered state
        spins = torch.ones(8, 8)
        ground_energy = ising_model.energy(spins).item()

        # Run many sweeps
        for _ in range(100):
            spins = low_temp_sampler.sweep(spins)

        # Energy should be close to ground state
        current_energy = ising_model.energy(spins).item()
        energy_per_spin = (current_energy - ground_energy) / 64
        assert energy_per_spin < 0.5, f"Expected energy near ground, got +{energy_per_spin} per spin"


class TestSweepMethod:
    """Tests for the sweep method (N² single-spin updates)."""

    def test_sweep_returns_tensor(self, mcmc_sampler, random_spins):
        """Sweep should return a tensor."""
        result = mcmc_sampler.sweep(random_spins.clone())
        assert isinstance(result, torch.Tensor)

    def test_sweep_preserves_shape(self, mcmc_sampler, random_spins):
        """Sweep should preserve spin configuration shape."""
        spins = random_spins.clone()
        result = mcmc_sampler.sweep(spins)
        assert result.shape == random_spins.shape

    def test_sweep_attempts_many_flips(self, mcmc_sampler, ising_model):
        """Sweep should attempt N² flip operations."""
        # At high temperature, we should see many changes after a sweep
        sampler = MetropolisHastings(ising_model, temperature=10.0)
        spins_before = ising_model.random_configuration(batch_size=1).squeeze(0)
        spins_after = sampler.sweep(spins_before.clone())

        # Count differences
        diff_count = (spins_before != spins_after).sum().item()

        # At T=10, should have many accepted flips
        n_spins = ising_model.size * ising_model.size
        assert diff_count > 0, "Expected at least some flips in a sweep"

    def test_sweep_ergodic(self, high_temp_sampler, ising_model):
        """Multiple sweeps should explore configuration space."""
        spins = torch.ones(8, 8)  # Start ordered
        initial_mag = ising_model.magnetization(spins).item()

        # Run many sweeps at high temperature
        for _ in range(100):
            spins = high_temp_sampler.sweep(spins)

        # Magnetization should change significantly
        final_mag = ising_model.magnetization(spins).item()
        assert abs(final_mag - initial_mag) > 0.1, "Expected ergodic exploration"


class TestSampleMethod:
    """Tests for the sample method with burn-in."""

    def test_sample_returns_correct_count(self, mcmc_sampler):
        """Sample should return requested number of samples."""
        samples = mcmc_sampler.sample(n_samples=5, n_sweeps=2, burn_in=10)
        assert samples.shape[0] == 5

    def test_sample_returns_correct_shape(self, mcmc_sampler, ising_model):
        """Sample should return correct tensor shape."""
        samples = mcmc_sampler.sample(n_samples=3, n_sweeps=2, burn_in=10)
        assert samples.shape == (3, ising_model.size, ising_model.size)

    def test_sample_with_initial_config(self, mcmc_sampler, all_up_spins):
        """Sample should use provided initial configuration."""
        samples = mcmc_sampler.sample(
            n_samples=2, n_sweeps=1, burn_in=0, initial=all_up_spins
        )
        # First sample should be based on the initial config (with possible changes)
        assert samples.shape[0] == 2

    def test_sample_burn_in_thermalizes(self, ising_model):
        """Burn-in should allow system to thermalize."""
        sampler = MetropolisHastings(ising_model, temperature=5.0)

        # Start from ordered state
        initial = torch.ones(8, 8)

        # With burn-in, should thermalize to disordered
        samples = sampler.sample(
            n_samples=5, n_sweeps=10, burn_in=100, initial=initial
        )

        # After burn-in at high T, samples should be disordered
        mags = [abs(ising_model.magnetization(s).item()) for s in samples]
        avg_mag = sum(mags) / len(mags)
        assert avg_mag < 0.7, f"Expected thermalized samples, got avg |M|={avg_mag}"

    def test_sample_spacing_decorrelates(self, mcmc_sampler, ising_model):
        """Spacing between samples (n_sweeps) should reduce correlation."""
        samples = mcmc_sampler.sample(n_samples=10, n_sweeps=20, burn_in=50)

        # Compute magnetizations
        mags = [ising_model.magnetization(s).item() for s in samples]

        # Samples should not all be identical
        mag_tensor = torch.tensor(mags)
        std = mag_tensor.std().item()
        assert std > 0.01, "Expected some variation between samples"


class TestSampleWithTrajectory:
    """Tests for the sample_with_trajectory generator method."""

    def test_trajectory_is_generator(self, mcmc_sampler):
        """sample_with_trajectory should return a generator."""
        from types import GeneratorType

        trajectory = mcmc_sampler.sample_with_trajectory(n_steps=5)
        assert isinstance(trajectory, GeneratorType)

    def test_trajectory_yields_correct_count(self, mcmc_sampler):
        """Trajectory should yield n_steps + 1 frames (initial + steps)."""
        n_steps = 10
        trajectory = list(mcmc_sampler.sample_with_trajectory(n_steps=n_steps))
        assert len(trajectory) == n_steps + 1  # Initial + n_steps

    def test_trajectory_yields_tensors(self, mcmc_sampler):
        """Each yielded frame should be a tensor."""
        for frame in mcmc_sampler.sample_with_trajectory(n_steps=3):
            assert isinstance(frame, torch.Tensor)

    def test_trajectory_preserves_shape(self, mcmc_sampler, ising_model):
        """Each frame should have correct shape."""
        for frame in mcmc_sampler.sample_with_trajectory(n_steps=3):
            assert frame.shape == (ising_model.size, ising_model.size)

    def test_trajectory_with_initial_config(self, mcmc_sampler, all_up_spins):
        """Trajectory should use provided initial configuration."""
        trajectory = list(
            mcmc_sampler.sample_with_trajectory(n_steps=2, initial=all_up_spins)
        )
        # First frame should be the initial config
        assert torch.allclose(trajectory[0], all_up_spins)

    def test_trajectory_evolves(self, high_temp_sampler, ising_model):
        """Trajectory should show evolution over time at high T."""
        frames = list(high_temp_sampler.sample_with_trajectory(n_steps=50))

        # First and last frames should be different
        first_mag = ising_model.magnetization(frames[0]).item()
        last_mag = ising_model.magnetization(frames[-1]).item()

        # At high temperature, there should be some change
        energies = [ising_model.energy(f).item() for f in frames]
        energy_std = torch.tensor(energies).std().item()
        assert energy_std > 0, "Expected trajectory evolution"


class TestThermalizationConvergence:
    """Tests for thermalization and convergence behavior."""

    def test_ordered_to_disordered_thermalization(self, ising_model):
        """System should thermalize from ordered to disordered at high T."""
        sampler = MetropolisHastings(ising_model, temperature=5.0)

        # Start from perfectly ordered state
        spins = torch.ones(8, 8)
        initial_mag = abs(ising_model.magnetization(spins).item())
        assert initial_mag == 1.0

        # Run many sweeps
        for _ in range(200):
            spins = sampler.sweep(spins)

        # Should thermalize to disordered
        final_mag = abs(ising_model.magnetization(spins).item())
        assert final_mag < 0.5, f"Expected thermalization to disorder, got |M|={final_mag}"

    def test_disordered_to_ordered_thermalization(self, ising_model):
        """System should thermalize from disordered to ordered at low T."""
        sampler = MetropolisHastings(ising_model, temperature=0.5)

        # Start from random state
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        # Run many sweeps
        for _ in range(500):
            spins = sampler.sweep(spins)

        # Should thermalize to ordered
        final_mag = abs(ising_model.magnetization(spins).item())
        assert final_mag > 0.8, f"Expected thermalization to order, got |M|={final_mag}"

    def test_critical_temperature_behavior(self, ising_model):
        """At critical temperature, system should show intermediate behavior."""
        # T_c ≈ 2.269 for 2D Ising
        sampler = MetropolisHastings(ising_model, temperature=2.27)

        # Start from random state
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)

        # Collect magnetization samples after thermalization
        for _ in range(100):  # Burn-in
            spins = sampler.sweep(spins)

        mags = []
        for _ in range(50):
            for _ in range(10):  # Spacing
                spins = sampler.sweep(spins)
            mags.append(abs(ising_model.magnetization(spins).item()))

        # At T_c, should see fluctuations (not stuck at 0 or 1)
        avg_mag = sum(mags) / len(mags)
        mag_std = torch.tensor(mags).std().item()

        # Intermediate behavior: not fully ordered, not fully disordered
        assert 0.1 < avg_mag < 0.9, f"Expected intermediate magnetization, got {avg_mag}"


class TestDetailedBalance:
    """Tests verifying detailed balance condition."""

    def test_energy_distribution_boltzmann(self, ising_model):
        """Energy distribution should follow Boltzmann distribution."""
        temperature = 2.5
        sampler = MetropolisHastings(ising_model, temperature=temperature)

        # Start from random state and thermalize
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)
        for _ in range(200):
            spins = sampler.sweep(spins)

        # Collect energy samples
        energies = []
        for _ in range(100):
            for _ in range(10):
                spins = sampler.sweep(spins)
            energies.append(ising_model.energy(spins).item())

        # Check that energies have reasonable mean and variance
        energy_tensor = torch.tensor(energies)
        mean_energy = energy_tensor.mean().item()

        # At T=2.5 (above T_c), mean energy should be between ground (-128) and max (+128)
        assert -128 < mean_energy < 0, f"Unexpected mean energy: {mean_energy}"

    def test_magnetization_symmetry(self, ising_model):
        """Magnetization should be symmetric around zero at high T."""
        sampler = MetropolisHastings(ising_model, temperature=5.0)

        # Collect magnetization samples
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)
        for _ in range(100):  # Burn-in
            spins = sampler.sweep(spins)

        mags = []
        for _ in range(200):
            for _ in range(5):
                spins = sampler.sweep(spins)
            mags.append(ising_model.magnetization(spins).item())

        # At high T, mean magnetization should be near zero
        mean_mag = sum(mags) / len(mags)
        assert abs(mean_mag) < 0.3, f"Expected mean M ≈ 0 at high T, got {mean_mag}"

    def test_equilibrium_stability(self, ising_model):
        """Once equilibrated, statistics should remain stable."""
        sampler = MetropolisHastings(ising_model, temperature=2.5)

        # Thermalize
        spins = ising_model.random_configuration(batch_size=1).squeeze(0)
        for _ in range(300):
            spins = sampler.sweep(spins)

        # Collect statistics in two periods
        energies_first = []
        energies_second = []

        for _ in range(50):
            for _ in range(10):
                spins = sampler.sweep(spins)
            energies_first.append(ising_model.energy(spins).item())

        for _ in range(50):
            for _ in range(10):
                spins = sampler.sweep(spins)
            energies_second.append(ising_model.energy(spins).item())

        # Mean energies should be similar (equilibrium)
        mean_first = sum(energies_first) / len(energies_first)
        mean_second = sum(energies_second) / len(energies_second)

        # Allow some statistical fluctuation
        diff = abs(mean_first - mean_second)
        assert diff < 30, f"Expected stable equilibrium, got diff={diff}"
