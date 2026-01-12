"""Abstract base class for physical systems."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class PhysicalSystem(ABC):
    """Abstract base class for physical systems that can be sampled.

    All physical systems must implement methods for computing energy,
    generating configurations, and converting between discrete/continuous
    representations for diffusion models.
    """

    @abstractmethod
    def energy(self, configuration: torch.Tensor) -> torch.Tensor:
        """Compute the energy of a configuration.

        Args:
            configuration: System configuration tensor

        Returns:
            Energy value(s) for the configuration(s)
        """
        pass

    @abstractmethod
    def random_configuration(self, batch_size: int = 1) -> torch.Tensor:
        """Generate random configuration(s).

        Args:
            batch_size: Number of configurations to generate

        Returns:
            Random configuration tensor
        """
        pass

    @abstractmethod
    def energy_per_particle(self, configuration: torch.Tensor) -> torch.Tensor:
        """Compute energy per particle/spin.

        Args:
            configuration: System configuration tensor

        Returns:
            Energy per particle for normalization
        """
        pass

    @abstractmethod
    def to_continuous(self, configuration: torch.Tensor) -> torch.Tensor:
        """Convert discrete configuration to continuous for diffusion.

        Args:
            configuration: Discrete configuration

        Returns:
            Continuous representation suitable for Gaussian noise
        """
        pass

    @abstractmethod
    def from_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous values back to discrete configuration.

        Args:
            x: Continuous tensor from diffusion process

        Returns:
            Discrete configuration
        """
        pass

    @property
    @abstractmethod
    def num_particles(self) -> int:
        """Return the number of particles/spins in the system."""
        pass

    @property
    @abstractmethod
    def configuration_shape(self) -> Tuple[int, ...]:
        """Return the shape of a single configuration."""
        pass
