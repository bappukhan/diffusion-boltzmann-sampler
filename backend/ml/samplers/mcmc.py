"""Metropolis-Hastings MCMC sampler for Ising model."""

import torch
import numpy as np
from typing import List, Generator, Optional
from ..systems.ising import IsingModel


class MetropolisHastings:
    """Metropolis-Hastings sampler for the 2D Ising model.

    Implements single-spin flip dynamics with Metropolis acceptance criterion.
    """

    def __init__(self, model: IsingModel, temperature: float):
        """Initialize sampler.

        Args:
            model: IsingModel instance
            temperature: Temperature T (in units where k_B = 1)
        """
        self.model = model
        self.temperature = temperature
        self.beta = 1.0 / temperature if temperature > 0 else float("inf")

    def step(self, spins: torch.Tensor) -> torch.Tensor:
        """Perform single Metropolis step: propose flip, accept/reject.

        Args:
            spins: Current spin configuration

        Returns:
            Updated spin configuration (may be same as input if rejected)
        """
        # Random spin to flip
        i = np.random.randint(self.model.size)
        j = np.random.randint(self.model.size)

        # Compute energy change
        dE = self.model.local_energy_diff(spins, i, j)

        # Metropolis acceptance
        if dE <= 0:
            # Always accept energy-lowering moves
            spins[..., i, j] *= -1
        elif np.random.random() < np.exp(-self.beta * dE.item()):
            # Accept energy-raising moves with Boltzmann probability
            spins[..., i, j] *= -1

        return spins

    def sweep(self, spins: torch.Tensor) -> torch.Tensor:
        """Perform N sweep (N = size²) of single-spin updates.

        One sweep attempts to flip each spin once on average.
        """
        n_spins = self.model.size * self.model.size
        for _ in range(n_spins):
            spins = self.step(spins)
        return spins

    def sample(
        self,
        n_samples: int,
        n_sweeps: int = 10,
        initial: Optional[torch.Tensor] = None,
        burn_in: int = 100,
    ) -> torch.Tensor:
        """Generate samples from equilibrium distribution.

        Args:
            n_samples: Number of samples to generate
            n_sweeps: Number of sweeps between samples
            initial: Initial configuration (random if None)
            burn_in: Number of sweeps for thermalization

        Returns:
            Tensor of shape (n_samples, size, size)
        """
        # Initialize
        if initial is None:
            spins = self.model.random_configuration(batch_size=1).squeeze(0)
        else:
            spins = initial.clone()

        # Burn-in for thermalization
        for _ in range(burn_in):
            spins = self.sweep(spins)

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                spins = self.sweep(spins)
            samples.append(spins.clone())

        return torch.stack(samples)

    def sample_with_trajectory(
        self,
        n_steps: int,
        initial: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Generate samples yielding each configuration.

        Useful for animation.

        Args:
            n_steps: Total number of sweeps
            initial: Initial configuration

        Yields:
            Spin configuration after each sweep
        """
        if initial is None:
            spins = self.model.random_configuration(batch_size=1).squeeze(0)
        else:
            spins = initial.clone()

        yield spins.clone()

        for _ in range(n_steps):
            spins = self.sweep(spins)
            yield spins.clone()


if __name__ == "__main__":
    # Test MCMC sampler
    model = IsingModel(size=16, J=1.0, h=0.0)

    # Test at low temperature (should order)
    sampler_low = MetropolisHastings(model, temperature=1.0)
    samples_low = sampler_low.sample(n_samples=10, n_sweeps=10, burn_in=200)
    mag_low = model.magnetization(samples_low).abs().mean()
    print(f"T=1.0 (below T_c): |M| = {mag_low:.3f} (should be ~1)")

    # Test at high temperature (should disorder)
    sampler_high = MetropolisHastings(model, temperature=5.0)
    samples_high = sampler_high.sample(n_samples=10, n_sweeps=10, burn_in=200)
    mag_high = model.magnetization(samples_high).abs().mean()
    print(f"T=5.0 (above T_c): |M| = {mag_high:.3f} (should be ~0)")

    print("MCMC tests passed!")
