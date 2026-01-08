"""Diffusion process for score-based generative modeling."""

import torch
from typing import Tuple


class DiffusionProcess:
    """Variance Preserving (VP) Stochastic Differential Equation diffusion.

    Forward process: dx = -0.5 β(t) x dt + √β(t) dW
    This gradually adds noise to data until it becomes pure Gaussian.

    The noise schedule is linear: β(t) = β_min + t(β_max - β_min)
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        """Initialize diffusion process.

        Args:
            beta_min: Minimum noise rate
            beta_max: Maximum noise rate
        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise rate β(t) at time t.

        Args:
            t: Time values in [0, 1]

        Returns:
            β(t) values
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def noise_level(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute signal and noise coefficients at time t.

        For the VP-SDE, we have:
            x_t = α_t x_0 + σ_t ε,  where ε ~ N(0, I)

        Args:
            t: Time values in [0, 1]

        Returns:
            (α_t, σ_t) coefficients
        """
        # Integral of beta(s) from 0 to t
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

        # α_t = exp(-0.5 ∫β(s)ds)
        alpha_t = torch.exp(-0.5 * integral)

        # σ_t = √(1 - α_t²) for variance-preserving
        sigma_t = torch.sqrt(1 - alpha_t**2 + 1e-8)

        return alpha_t, sigma_t

    def forward(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) = N(α_t x_0, σ_t² I).

        Args:
            x_0: Clean data of shape (batch, channels, height, width)
            t: Time values of shape (batch,)

        Returns:
            (x_t, noise) where x_t is noisy data and noise is the added noise
        """
        alpha_t, sigma_t = self.noise_level(t)

        # Reshape for broadcasting
        alpha_t = alpha_t[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]

        # Sample noise
        noise = torch.randn_like(x_0)

        # Compute x_t
        x_t = alpha_t * x_0 + sigma_t * noise

        return x_t, noise

    def score_target(self, noise: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """Compute target score for denoising score matching.

        The score of q(x_t | x_0) is: ∇log q(x_t|x_0) = -noise/σ_t

        Args:
            noise: The noise added to get x_t
            sigma_t: Noise level at time t

        Returns:
            Target score
        """
        return -noise / (sigma_t + 1e-8)


if __name__ == "__main__":
    # Test diffusion process
    diffusion = DiffusionProcess()

    # Test noise levels
    t = torch.linspace(0, 1, 11)
    alpha_t, sigma_t = diffusion.noise_level(t)
    print("Time | Alpha | Sigma")
    for i in range(len(t)):
        print(f"{t[i]:.1f}  | {alpha_t[i]:.3f} | {sigma_t[i]:.3f}")

    # Verify α²+σ²≈1 (variance preservation)
    var = alpha_t**2 + sigma_t**2
    print(f"\nα² + σ² (should be ~1): {var}")

    # Test forward process
    x_0 = torch.randn(4, 1, 16, 16)
    t = torch.rand(4)
    x_t, noise = diffusion.forward(x_0, t)

    print(f"\nInput shape: {x_0.shape}")
    print(f"Noisy shape: {x_t.shape}")
    print(f"Noise shape: {noise.shape}")

    # At t=0, x_t ≈ x_0
    t_zero = torch.zeros(4)
    x_t_0, _ = diffusion.forward(x_0, t_zero)
    diff_0 = (x_t_0 - x_0).abs().max()
    print(f"\nAt t=0, max |x_t - x_0|: {diff_0:.6f} (should be ~0)")

    # At t=1, x_t ≈ N(0, 1)
    t_one = torch.ones(4)
    x_t_1, _ = diffusion.forward(x_0, t_one)
    print(f"At t=1, mean: {x_t_1.mean():.3f}, std: {x_t_1.std():.3f} (should be ~0, ~1)")

    print("\nDiffusion process tests passed!")
