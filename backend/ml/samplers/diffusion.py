"""Diffusion sampler using trained score network."""

import torch
from typing import Generator, Tuple, Optional
from ..models.diffusion import DiffusionProcess


class DiffusionSampler:
    """Reverse diffusion sampler using trained score network.

    Generates samples by reversing the noising process using
    the learned score function.
    """

    def __init__(
        self,
        score_network: torch.nn.Module,
        diffusion: Optional[DiffusionProcess] = None,
        num_steps: int = 100,
        device: str = "cpu",
    ):
        """Initialize diffusion sampler.

        Args:
            score_network: Trained score network s_θ(x, t)
            diffusion: Diffusion process (default: standard VP-SDE)
            num_steps: Number of discretization steps
            device: Device to run on
        """
        self.model = score_network.to(device)
        self.model.eval()
        self.diffusion = diffusion or DiffusionProcess()
        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion.

        Uses Euler-Maruyama discretization of the reverse SDE:
            dx = [σ²(t) s_θ(x,t)] dt + σ(t) dW

        Args:
            shape: Shape of samples (batch, channels, height, width)
            temperature: Sampling temperature (scales noise)

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Time step
        dt = 1.0 / self.num_steps

        # Reverse diffusion
        for i in range(self.num_steps, 0, -1):
            t = torch.full((shape[0],), i / self.num_steps, device=self.device)

            # Get noise level
            _, sigma_t = self.diffusion.noise_level(t)
            sigma = sigma_t[0].item()

            # Score prediction
            score = self.model(x, t)

            # Reverse SDE step (Euler-Maruyama)
            # Mean update: x += σ² * score * dt
            drift = (sigma**2) * score * dt

            # Stochastic term (skip at final step)
            if i > 1:
                noise = torch.randn_like(x)
                diffusion_term = sigma * (dt**0.5) * noise * temperature
            else:
                diffusion_term = 0

            x = x + drift + diffusion_term

        return x

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        shape: Tuple[int, ...],
        temperature: float = 1.0,
        yield_every: int = 1,
    ) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """Generate samples yielding intermediate states.

        Useful for animation and visualization.

        Args:
            shape: Shape of samples
            temperature: Sampling temperature
            yield_every: Yield every N steps (for efficiency)

        Yields:
            (x_t, t) tuples
        """
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / self.num_steps

        yield x.clone(), 1.0

        for i in range(self.num_steps, 0, -1):
            t_val = i / self.num_steps
            t = torch.full((shape[0],), t_val, device=self.device)

            _, sigma_t = self.diffusion.noise_level(t)
            sigma = sigma_t[0].item()

            score = self.model(x, t)

            drift = (sigma**2) * score * dt
            if i > 1:
                noise = torch.randn_like(x)
                diffusion_term = sigma * (dt**0.5) * noise * temperature
            else:
                diffusion_term = 0

            x = x + drift + diffusion_term

            # Yield intermediate states
            if i % yield_every == 0 or i == 1:
                yield x.clone(), (i - 1) / self.num_steps


class PretrainedDiffusionSampler(DiffusionSampler):
    """Diffusion sampler that can be used without a trained model.

    Uses simple heuristics for demonstration purposes.
    """

    def __init__(
        self,
        lattice_size: int = 32,
        num_steps: int = 100,
        device: str = "cpu",
    ):
        """Initialize with a dummy score network."""
        from ..models.score_network import ScoreNetwork

        # Create a small network for demo
        network = ScoreNetwork(
            in_channels=1,
            base_channels=16,  # Smaller for CPU
            time_embed_dim=32,
            num_blocks=2,
        )

        super().__init__(
            score_network=network,
            num_steps=num_steps,
            device=device,
        )
        self.lattice_size = lattice_size

    @torch.no_grad()
    def sample_heuristic(
        self,
        batch_size: int = 1,
        temperature: float = 2.27,
    ) -> torch.Tensor:
        """Generate samples using physics-informed heuristic.

        For demonstration, uses a simple annealing approach
        without a trained model.
        """
        shape = (batch_size, 1, self.lattice_size, self.lattice_size)

        # Start from noise
        x = torch.randn(shape, device=self.device)

        # Simple annealing (not a real diffusion model)
        for i in range(self.num_steps, 0, -1):
            t = i / self.num_steps

            # Reduce noise level
            noise_scale = t**0.5
            signal_scale = (1 - t**0.5)

            # Threshold toward discrete values
            x_discrete = torch.sign(x)

            # Interpolate
            x = signal_scale * x_discrete + noise_scale * torch.randn_like(x) * 0.5

        return torch.sign(x)


if __name__ == "__main__":
    # Test with heuristic sampler
    sampler = PretrainedDiffusionSampler(lattice_size=16)

    print("Testing heuristic sampling...")
    samples = sampler.sample_heuristic(batch_size=4)
    print(f"Sample shape: {samples.shape}")
    print(f"Values in {{-1, +1}}: {((samples == 1) | (samples == -1)).all()}")
    print(f"Mean magnetization: {samples.mean():.3f}")

    print("\nTesting trajectory generation...")
    for x, t in sampler.sample_with_trajectory((1, 1, 16, 16), yield_every=25):
        print(f"t={t:.2f}, x range: [{x.min():.2f}, {x.max():.2f}]")

    print("\nDiffusion sampler tests passed!")
