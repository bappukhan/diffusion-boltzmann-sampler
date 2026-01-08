"""2D Ising model implementation with periodic boundary conditions."""

import torch
from typing import Tuple, Optional


class IsingModel:
    """2D Ising model on a square lattice with periodic boundary conditions.

    The Hamiltonian is:
        H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    where s_i ∈ {-1, +1}, J is the coupling constant, and h is the external field.

    Attributes:
        size: Linear size of the square lattice (size × size spins)
        J: Coupling constant (J > 0 is ferromagnetic)
        h: External magnetic field
        T_critical: Critical temperature for 2D Ising (2/ln(1+√2) ≈ 2.269)
    """

    T_CRITICAL = 2.0 / torch.log(torch.tensor(1.0 + 2**0.5)).item()  # ≈ 2.269

    def __init__(self, size: int, J: float = 1.0, h: float = 0.0):
        """Initialize Ising model.

        Args:
            size: Linear size of square lattice
            J: Coupling constant (default 1.0 for ferromagnetic)
            h: External field (default 0.0)
        """
        self.size = size
        self.J = J
        self.h = h

    def random_configuration(self, batch_size: int = 1) -> torch.Tensor:
        """Generate random spin configuration.

        Args:
            batch_size: Number of configurations to generate

        Returns:
            Tensor of shape (batch_size, size, size) with values in {-1, +1}
        """
        return torch.randint(0, 2, (batch_size, self.size, self.size)).float() * 2 - 1

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute total energy of spin configurations.

        Args:
            spins: Tensor of shape (..., size, size) with values in {-1, +1}

        Returns:
            Tensor of shape (...,) containing energy for each configuration
        """
        # Nearest neighbor interactions with periodic boundaries
        right = torch.roll(spins, -1, dims=-1)
        down = torch.roll(spins, -1, dims=-2)

        # Interaction energy: -J Σ s_i s_j
        interaction = -self.J * (spins * right + spins * down).sum(dim=(-1, -2))

        # Field energy: -h Σ s_i
        field = -self.h * spins.sum(dim=(-1, -2))

        return interaction + field

    def local_energy_diff(self, spins: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """Compute energy change from flipping spin at (i, j).

        This is O(1) instead of O(N²) for full energy recomputation.

        Args:
            spins: Current spin configuration
            i, j: Position of spin to flip

        Returns:
            Energy difference ΔE = E(flipped) - E(current)
        """
        s = spins[..., i, j]

        # Sum of neighboring spins
        neighbors = (
            spins[..., (i + 1) % self.size, j]
            + spins[..., (i - 1) % self.size, j]
            + spins[..., i, (j + 1) % self.size]
            + spins[..., i, (j - 1) % self.size]
        )

        # ΔE = 2 * s * (J * Σ_neighbors + h)
        return 2 * s * (self.J * neighbors + self.h)

    def magnetization(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute mean magnetization per spin.

        Args:
            spins: Tensor of shape (..., size, size)

        Returns:
            Tensor of shape (...,) containing magnetization per spin in [-1, 1]
        """
        return spins.mean(dim=(-1, -2))

    def energy_per_spin(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute energy per spin."""
        return self.energy(spins) / (self.size * self.size)

    def to_continuous(self, spins: torch.Tensor) -> torch.Tensor:
        """Convert discrete spins to continuous values for diffusion.

        Maps {-1, +1} to continuous space suitable for Gaussian noise.
        """
        return spins.unsqueeze(1)  # Add channel dimension

    def from_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous values back to discrete spins.

        Applies sign function to threshold at 0.
        """
        return torch.sign(x.squeeze(1))


if __name__ == "__main__":
    # Quick test
    model = IsingModel(size=8, J=1.0, h=0.0)

    # Ground state: all spins aligned
    all_up = torch.ones(8, 8)
    E = model.energy(all_up)
    expected = -2 * 8 * 8  # Each spin has 2 unique bonds
    print(f"Ground state energy: {E.item()}, Expected: {expected}")
    assert abs(E.item() - expected) < 1e-6, "Energy calculation incorrect!"

    # Random configuration
    random = model.random_configuration(batch_size=4)
    print(f"Random config shape: {random.shape}")
    print(f"Random energies: {model.energy(random)}")
    print(f"Random magnetizations: {model.magnetization(random)}")

    print("All tests passed!")
