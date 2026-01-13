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
