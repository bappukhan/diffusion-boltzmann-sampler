"""Loss functions for denoising score matching training.

This module provides various loss functions and weighting schemes
for training score-based diffusion models.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.diffusion import DiffusionProcess

# Numerical stability constant
EPS = 1e-8

# Type alias for loss types
LossType = Literal["l1", "l2", "huber"]


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: LossType = "l2",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute element-wise loss between predictions and targets.

    Args:
        pred: Predicted values
        target: Target values
        loss_type: Type of loss ("l1", "l2", or "huber")
        huber_delta: Delta parameter for Huber loss

    Returns:
        Element-wise loss tensor (same shape as input)
    """
    if loss_type == "l2":
        return (pred - target) ** 2
    elif loss_type == "l1":
        return torch.abs(pred - target)
    elif loss_type == "huber":
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = huber_delta * (abs_diff - 0.5 * huber_delta)
        return torch.where(abs_diff <= huber_delta, quadratic, linear)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def denoising_score_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    diffusion: "DiffusionProcess",
) -> torch.Tensor:
    """Compute denoising score matching (DSM) loss.

    DSM loss: E_t E_{x_t|x_0} ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²

    For Gaussian diffusion: ∇log p(x_t|x_0) = -noise/σ_t

    Args:
        model: Score network s_θ(x, t)
        x_0: Clean samples of shape (batch, channels, height, width)
        diffusion: Diffusion process for forward noising

    Returns:
        Scalar loss value
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample time uniformly from [0, 1]
    t = torch.rand(batch_size, device=device)

    # Get noisy samples via forward diffusion
    x_t, noise = diffusion.forward(x_0, t)

    # Get noise level for computing target score
    _, sigma_t = diffusion.noise_level(t)

    # Target score: ∇log p(x_t|x_0) = -noise/σ_t
    target = diffusion.score_target(noise, sigma_t[:, None, None, None])

    # Predicted score from model
    pred = model(x_t, t)

    # MSE loss
    loss = ((pred - target) ** 2).mean()

    return loss


def sigma_weighted_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    diffusion: "DiffusionProcess",
) -> torch.Tensor:
    """Compute sigma-weighted denoising score matching loss.

    Weights the loss by σ² to balance contributions across noise levels.
    This helps the model learn equally well at all noise scales.

    Loss: E_t E_{x_t|x_0} σ_t² ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²

    Args:
        model: Score network s_θ(x, t)
        x_0: Clean samples of shape (batch, channels, height, width)
        diffusion: Diffusion process for forward noising

    Returns:
        Scalar loss value
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample time uniformly
    t = torch.rand(batch_size, device=device)

    # Get noisy samples
    x_t, noise = diffusion.forward(x_0, t)

    # Get noise level
    _, sigma_t = diffusion.noise_level(t)
    sigma_t_expanded = sigma_t[:, None, None, None]

    # Target score
    target = diffusion.score_target(noise, sigma_t_expanded)

    # Predicted score
    pred = model(x_t, t)

    # Sigma-weighted MSE: multiply by σ² to balance across noise levels
    # This is equivalent to predicting noise ε instead of score
    weights = sigma_t_expanded ** 2
    loss = (weights * (pred - target) ** 2).mean()

    return loss


def snr_weighted_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    diffusion: "DiffusionProcess",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute SNR-weighted denoising score matching loss.

    Weights the loss by SNR^gamma where SNR = α²/σ².
    Higher gamma emphasizes low-noise (high SNR) timesteps.

    Loss: E_t E_{x_t|x_0} SNR(t)^γ ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²

    Args:
        model: Score network s_θ(x, t)
        x_0: Clean samples of shape (batch, channels, height, width)
        diffusion: Diffusion process for forward noising
        gamma: SNR weighting exponent (default: 1.0)

    Returns:
        Scalar loss value
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample time uniformly
    t = torch.rand(batch_size, device=device)

    # Get noisy samples
    x_t, noise = diffusion.forward(x_0, t)

    # Get noise levels
    alpha_t, sigma_t = diffusion.noise_level(t)

    # Compute SNR = α²/σ²
    snr = (alpha_t ** 2) / (sigma_t ** 2 + EPS)
    snr_weights = snr ** gamma

    # Expand for broadcasting
    sigma_t_expanded = sigma_t[:, None, None, None]
    snr_weights_expanded = snr_weights[:, None, None, None]

    # Target score
    target = diffusion.score_target(noise, sigma_t_expanded)

    # Predicted score
    pred = model(x_t, t)

    # SNR-weighted MSE
    loss = (snr_weights_expanded * (pred - target) ** 2).mean()

    return loss


def importance_sampled_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    diffusion: "DiffusionProcess",
    t_min: float = 0.001,
    t_max: float = 0.999,
) -> torch.Tensor:
    """Compute DSM loss with importance-sampled time distribution.

    Uses truncated time range to avoid numerical issues at t=0 and t=1,
    and applies importance sampling weights based on the time distribution.

    Args:
        model: Score network s_θ(x, t)
        x_0: Clean samples of shape (batch, channels, height, width)
        diffusion: Diffusion process for forward noising
        t_min: Minimum time value (default: 0.001)
        t_max: Maximum time value (default: 0.999)

    Returns:
        Scalar loss value
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # Sample time from truncated uniform distribution
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min

    # Get noisy samples
    x_t, noise = diffusion.forward(x_0, t)

    # Get noise level
    _, sigma_t = diffusion.noise_level(t)
    sigma_t_expanded = sigma_t[:, None, None, None]

    # Target score
    target = diffusion.score_target(noise, sigma_t_expanded)

    # Predicted score
    pred = model(x_t, t)

    # Importance weight for truncated distribution
    # If sampling uniformly from [t_min, t_max], weight by (t_max - t_min)
    # to correct for the truncation
    weight = t_max - t_min
    loss = weight * ((pred - target) ** 2).mean()

    return loss


# Type alias for weighting schemes
WeightingType = Literal["uniform", "sigma", "snr", "importance"]


class ScoreMatchingLoss(nn.Module):
    """Configurable score matching loss for training diffusion models.

    Provides a unified interface for various DSM loss variants with
    configurable weighting schemes.
    """

    def __init__(
        self,
        diffusion: "DiffusionProcess",
        weighting: WeightingType = "uniform",
        snr_gamma: float = 1.0,
        t_min: float = 0.001,
        t_max: float = 0.999,
    ):
        """Initialize ScoreMatchingLoss.

        Args:
            diffusion: Diffusion process for forward noising
            weighting: Loss weighting scheme:
                - "uniform": Standard DSM loss
                - "sigma": Weight by σ² (equivalent to noise prediction)
                - "snr": Weight by SNR^gamma
                - "importance": Truncated time distribution
            snr_gamma: Exponent for SNR weighting (default: 1.0)
            t_min: Minimum time for importance sampling (default: 0.001)
            t_max: Maximum time for importance sampling (default: 0.999)
        """
        super().__init__()
        self.diffusion = diffusion
        self.weighting = weighting
        self.snr_gamma = snr_gamma
        self.t_min = t_min
        self.t_max = t_max

    def forward(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        """Compute score matching loss.

        Args:
            model: Score network s_θ(x, t)
            x_0: Clean samples of shape (batch, channels, height, width)

        Returns:
            Scalar loss value
        """
        if self.weighting == "uniform":
            return denoising_score_matching_loss(model, x_0, self.diffusion)
        elif self.weighting == "sigma":
            return sigma_weighted_loss(model, x_0, self.diffusion)
        elif self.weighting == "snr":
            return snr_weighted_loss(model, x_0, self.diffusion, self.snr_gamma)
        elif self.weighting == "importance":
            return importance_sampled_loss(
                model, x_0, self.diffusion, self.t_min, self.t_max
            )
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting}")
