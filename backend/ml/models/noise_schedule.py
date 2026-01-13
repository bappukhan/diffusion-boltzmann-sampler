"""Noise schedules for diffusion models.

Provides various beta schedules that control the rate of noise addition
during the forward diffusion process.

Available schedules:
- LinearNoiseSchedule: Standard linear schedule from DDPM
- CosineNoiseSchedule: Improved cosine schedule from OpenAI
- SigmoidNoiseSchedule: Smooth S-curve transition with configurable steepness

Use get_schedule() factory to create schedules by name.
"""

import torch
import math
from abc import ABC, abstractmethod
from typing import Tuple


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules.

    A noise schedule defines how the noise rate β(t) varies over time t ∈ [0, 1].
    For the VP-SDE, the forward process is:
        dx = -0.5 β(t) x dt + √β(t) dW
    """

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise rate β(t) at time t.

        Args:
            t: Time values in [0, 1], shape (batch,) or scalar

        Returns:
            β(t) values with same shape as t
        """
        pass

    def noise_level(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute signal and noise coefficients at time t.

        For the VP-SDE, we have:
            x_t = α_t x_0 + σ_t ε,  where ε ~ N(0, I)

        Args:
            t: Time values in [0, 1]

        Returns:
            (α_t, σ_t) coefficients
        """
        # Default numerical integration (can be overridden for analytic forms)
        integral = self._integrate_beta(t)
        alpha_t = torch.exp(-0.5 * integral)
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_t**2, min=1e-8))
        return alpha_t, sigma_t

    def _integrate_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Integrate β(s) from 0 to t.

        Default implementation uses numerical integration.
        Subclasses should override with analytic forms when available.
        """
        # Simple trapezoidal integration
        n_steps = 100
        dt = t / n_steps
        integral = torch.zeros_like(t)

        for i in range(n_steps):
            t_i = i * dt
            t_ip1 = (i + 1) * dt
            integral = integral + 0.5 * (self.beta(t_i) + self.beta(t_ip1)) * dt

        return integral


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule: β(t) = β_min + t(β_max - β_min).

    This is the standard schedule used in DDPM. The linear schedule
    provides a simple, interpretable progression from low to high noise.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        """Initialize linear schedule.

        Args:
            beta_min: Minimum noise rate at t=0
            beta_max: Maximum noise rate at t=1
        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute linear β(t)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integrate_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Analytic integral of linear schedule.

        ∫₀ᵗ β(s) ds = β_min * t + 0.5 * (β_max - β_min) * t²
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule from 'Improved Denoising Diffusion' paper.

    Uses a cosine schedule for α_t, which provides smoother transitions
    and better sample quality, especially for images.

    α_t = cos(π/2 * (t + s) / (1 + s))² / cos(π/2 * s / (1 + s))²

    where s is a small offset to prevent singularities.
    """

    def __init__(self, s: float = 0.008):
        """Initialize cosine schedule.

        Args:
            s: Small offset to prevent β from being too large near t=1
        """
        self.s = s
        # Precompute normalization factor
        self._alpha_0 = self._f(torch.tensor(0.0)).item()

    def _f(self, t: torch.Tensor) -> torch.Tensor:
        """Compute f(t) = cos(π/2 * (t + s) / (1 + s))²."""
        angle = math.pi / 2 * (t + self.s) / (1 + self.s)
        return torch.cos(angle) ** 2

    def noise_level(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute α_t and σ_t directly from cosine schedule."""
        alpha_t_sq = self._f(t) / self._alpha_0
        alpha_t = torch.sqrt(torch.clamp(alpha_t_sq, min=1e-8, max=1.0))
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_t_sq, min=1e-8))
        return alpha_t, sigma_t

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute β(t) from the derivative of log α_t².

        β(t) = -d/dt log α_t² = -2 d/dt log α_t
        """
        # Use finite difference for numerical stability
        eps = 1e-4
        alpha_t, _ = self.noise_level(t)
        alpha_t_eps, _ = self.noise_level(t + eps)

        # β ≈ -2 * (log α_{t+ε} - log α_t) / ε
        beta_t = -2 * (torch.log(alpha_t_eps + 1e-8) - torch.log(alpha_t + 1e-8)) / eps
        return torch.clamp(beta_t, min=0.0, max=100.0)


class SigmoidNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with configurable steepness.

    β(t) = sigmoid(start + (end - start) * t) * (β_max - β_min) + β_min

    This schedule provides a smooth S-curve transition that can be adjusted
    to spend more or less time in low/high noise regimes.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        start: float = -6.0,
        end: float = 6.0,
    ):
        """Initialize sigmoid schedule.

        Args:
            beta_min: Minimum noise rate
            beta_max: Maximum noise rate
            start: Sigmoid input at t=0 (more negative = slower start)
            end: Sigmoid input at t=1 (more positive = faster end)
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.start = start
        self.end = end

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-based β(t)."""
        sigmoid_input = self.start + (self.end - self.start) * t
        sigmoid_val = torch.sigmoid(sigmoid_input)
        return self.beta_min + (self.beta_max - self.beta_min) * sigmoid_val


def get_schedule(
    schedule_type: str = "linear",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    **kwargs,
) -> NoiseSchedule:
    """Factory function to create noise schedules.

    Args:
        schedule_type: Type of schedule ("linear", "cosine", or "sigmoid")
        beta_min: Minimum noise rate (for linear and sigmoid)
        beta_max: Maximum noise rate (for linear and sigmoid)
        **kwargs: Additional arguments for specific schedules

    Returns:
        NoiseSchedule instance

    Raises:
        ValueError: If schedule_type is unknown
    """
    schedule_type = schedule_type.lower()

    if schedule_type == "linear":
        return LinearNoiseSchedule(beta_min=beta_min, beta_max=beta_max)
    elif schedule_type == "cosine":
        s = kwargs.get("s", 0.008)
        return CosineNoiseSchedule(s=s)
    elif schedule_type == "sigmoid":
        start = kwargs.get("start", -6.0)
        end = kwargs.get("end", 6.0)
        return SigmoidNoiseSchedule(
            beta_min=beta_min, beta_max=beta_max, start=start, end=end
        )
    else:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Choose from: linear, cosine, sigmoid"
        )
