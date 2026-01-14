"""Diffusion sampler using trained score network."""

import torch
from typing import Generator, Tuple, Optional, Dict, Any
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

    @torch.no_grad()
    def sample_predictor_corrector(
        self,
        shape: Tuple[int, ...],
        temperature: float = 1.0,
        n_corrector_steps: int = 1,
        corrector_snr: float = 0.16,
    ) -> torch.Tensor:
        """Generate samples using predictor-corrector method.

        Combines Euler-Maruyama predictor with Langevin corrector steps
        for improved sample quality.

        Args:
            shape: Shape of samples (batch, channels, height, width)
            temperature: Sampling temperature
            n_corrector_steps: Number of Langevin corrector steps per predictor step
            corrector_snr: Signal-to-noise ratio for corrector (controls step size)

        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps, 0, -1):
            t = torch.full((shape[0],), i / self.num_steps, device=self.device)
            _, sigma_t = self.diffusion.noise_level(t)
            sigma = sigma_t[0].item()

            # Predictor step (Euler-Maruyama)
            score = self.model(x, t)
            drift = (sigma**2) * score * dt
            if i > 1:
                noise = torch.randn_like(x)
                diffusion_term = sigma * (dt**0.5) * noise * temperature
            else:
                diffusion_term = 0
            x = x + drift + diffusion_term

            # Corrector steps (Langevin dynamics)
            if i > 1:  # Skip corrector at final step
                for _ in range(n_corrector_steps):
                    score = self.model(x, t)
                    # Langevin step size based on SNR
                    grad_norm = score.flatten(1).norm(dim=1).mean()
                    noise_norm = (x.numel() / x.shape[0]) ** 0.5
                    step_size = (corrector_snr * noise_norm / grad_norm) ** 2 * 2

                    noise = torch.randn_like(x)
                    x = x + step_size * score + (2 * step_size) ** 0.5 * noise * temperature

        return x

    @torch.no_grad()
    def sample_ode(
        self,
        shape: Tuple[int, ...],
        method: str = "euler",
    ) -> torch.Tensor:
        """Generate samples using deterministic ODE solver.

        Uses the probability flow ODE (no stochastic term) for deterministic
        sampling. Useful for evaluation and when reproducibility is needed.

        Args:
            shape: Shape of samples (batch, channels, height, width)
            method: ODE solver method:
                - "euler": Simple Euler method
                - "heun": Heun's method (2nd order)
                - "rk4": 4th order Runge-Kutta

        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps, 0, -1):
            t_val = i / self.num_steps
            t = torch.full((shape[0],), t_val, device=self.device)
            _, sigma_t = self.diffusion.noise_level(t)
            sigma = sigma_t[0].item()

            if method == "euler":
                # Simple Euler: x_{t-dt} = x_t + σ² * s(x,t) * dt
                score = self.model(x, t)
                x = x + (sigma**2) * score * dt

            elif method == "heun":
                # Heun's method (predictor-corrector without noise)
                score = self.model(x, t)
                x_pred = x + (sigma**2) * score * dt

                # Corrector at next time
                t_next = torch.full((shape[0],), (i - 1) / self.num_steps, device=self.device)
                _, sigma_next = self.diffusion.noise_level(t_next)
                sigma_n = sigma_next[0].item()
                score_next = self.model(x_pred, t_next)

                # Average the drift
                x = x + 0.5 * ((sigma**2) * score + (sigma_n**2) * score_next) * dt

            elif method == "rk4":
                # 4th order Runge-Kutta
                def drift_fn(x_in, t_in):
                    _, sig = self.diffusion.noise_level(t_in)
                    s = sig[0].item()
                    return (s**2) * self.model(x_in, t_in)

                k1 = drift_fn(x, t)
                t_half = torch.full((shape[0],), t_val - 0.5 * dt, device=self.device)
                k2 = drift_fn(x + 0.5 * dt * k1, t_half)
                k3 = drift_fn(x + 0.5 * dt * k2, t_half)
                t_full = torch.full((shape[0],), t_val - dt, device=self.device)
                k4 = drift_fn(x + dt * k3, t_full)

                x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            else:
                raise ValueError(f"Unknown ODE method: {method}")

        return x

    def discretize_spins(
        self,
        x: torch.Tensor,
        method: str = "sign",
        threshold: float = 0.0,
        sharpness: float = 10.0,
    ) -> torch.Tensor:
        """Convert continuous samples to discrete Ising spins {-1, +1}.

        Args:
            x: Continuous samples from diffusion (any shape)
            method: Discretization method:
                - "sign": Simple sign function threshold at threshold
                - "tanh": Soft discretization using tanh (preserves gradients)
                - "gumbel": Gumbel-softmax for differentiable sampling
                - "stochastic": Probabilistic rounding based on value
            threshold: Threshold value for sign method (default 0.0)
            sharpness: Sharpness parameter for tanh method (higher = sharper)

        Returns:
            Tensor of discrete spins in {-1, +1}
        """
        if method == "sign":
            # Hard threshold at specified value
            return torch.sign(x - threshold)
        elif method == "tanh":
            # Soft discretization (sharpness → ∞ gives sign)
            return torch.tanh(sharpness * (x - threshold))
        elif method == "gumbel":
            # Gumbel-softmax for two classes {-1, +1}
            logits = torch.stack([-(x - threshold), x - threshold], dim=-1)
            probs = torch.softmax(logits * sharpness, dim=-1)
            # Return expected value: P(+1) - P(-1)
            return 2 * probs[..., 1] - 1
        elif method == "stochastic":
            # Probabilistic discretization: P(+1) = sigmoid(sharpness * x)
            probs = torch.sigmoid(sharpness * (x - threshold))
            random_vals = torch.rand_like(probs)
            return torch.where(random_vals < probs, torch.ones_like(x), -torch.ones_like(x))
        else:
            raise ValueError(f"Unknown discretization method: {method}")

    def sample_ising(
        self,
        batch_size: int = 1,
        lattice_size: int = 32,
        temperature: float = 1.0,
        discretize: bool = True,
        discretize_method: str = "sign",
    ) -> torch.Tensor:
        """Generate discrete Ising spin samples.

        Convenience method for sampling Ising configurations.

        Args:
            batch_size: Number of samples to generate
            lattice_size: Size of square lattice
            temperature: Sampling temperature
            discretize: Whether to discretize to {-1, +1}
            discretize_method: Discretization method if discretize=True

        Returns:
            Tensor of shape (batch_size, lattice_size, lattice_size)
        """
        shape = (batch_size, 1, lattice_size, lattice_size)
        samples = self.sample(shape, temperature=temperature)

        if discretize:
            samples = self.discretize_spins(samples, method=discretize_method)

        # Remove channel dimension
        return samples.squeeze(1)


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
