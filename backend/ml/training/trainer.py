"""Training loop for denoising score matching."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable, Dict, List, Union
from tqdm import tqdm

from ..models.score_network import ScoreNetwork
from ..models.diffusion import DiffusionProcess
from ..systems.ising import IsingModel
from ..samplers.mcmc import MetropolisHastings
from .losses import ScoreMatchingLoss, WeightingType


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated
    with exponential moving average during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict:
        """Return EMA state for checkpointing."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)


class Trainer:
    """Denoising score matching trainer for Ising model."""

    def __init__(
        self,
        score_network: ScoreNetwork,
        diffusion: Optional[DiffusionProcess] = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        loss_weighting: WeightingType = "uniform",
    ):
        """Initialize trainer.

        Args:
            score_network: Score network to train
            diffusion: Diffusion process (default: VP-SDE)
            learning_rate: Learning rate for Adam optimizer
            device: Device to train on
            loss_weighting: Loss weighting scheme ("uniform", "sigma", "snr", "importance")
        """
        self.model = score_network.to(device)
        self.diffusion = diffusion or DiffusionProcess()
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_weighting = loss_weighting

        # Create loss function using the losses module
        self.loss_fn = ScoreMatchingLoss(
            diffusion=self.diffusion,
            weighting=loss_weighting,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def score_matching_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute denoising score matching loss.

        L = E_t E_{x_t|x_0} ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²

        For Gaussian diffusion: ∇log p(x_t|x_0) = -noise/σ_t

        Args:
            x_0: Clean samples (batch, channels, height, width)

        Returns:
            Scalar loss value
        """
        batch_size = x_0.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)

        # Get noisy samples
        x_t, noise = self.diffusion.forward(x_0, t)

        # Get noise level for weighting
        _, sigma_t = self.diffusion.noise_level(t)

        # Target score
        target = self.diffusion.score_target(noise, sigma_t[:, None, None, None])

        # Predicted score
        pred = self.model(x_t, t)

        # MSE loss (optionally weighted by σ²)
        loss = ((pred - target) ** 2).mean()

        return loss

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step.

        Args:
            batch: Batch of samples

        Returns:
            Loss value
        """
        self.model.train()
        batch = batch.to(self.device)

        self.optimizer.zero_grad()
        loss = self.score_matching_loss(batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        callback: Optional[Callable[[float], None]] = None,
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            callback: Optional callback called with each batch loss

        Returns:
            Average loss for epoch
        """
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle TensorDataset format

            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            if callback:
                callback(loss)

        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation data.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(self.device)
            loss = self.score_matching_loss(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)

        return avg_loss

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            dataloader: Training data loader
            epochs: Number of epochs
            val_dataloader: Optional validation data loader
            verbose: Whether to print progress

        Returns:
            Training history
        """
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            train_loss = self.train_epoch(dataloader)

            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)

            if verbose:
                msg = f"Epoch {epoch+1}: train_loss={train_loss:.4f}"
                if val_dataloader:
                    msg += f", val_loss={val_loss:.4f}"
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix_str(msg.split(": ")[1])

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})


def generate_training_data(
    model: IsingModel,
    temperature: float,
    n_samples: int,
    n_sweeps: int = 10,
    burn_in: int = 500,
) -> torch.Tensor:
    """Generate training data using MCMC.

    Args:
        model: Ising model
        temperature: Sampling temperature
        n_samples: Number of samples to generate
        n_sweeps: Sweeps between samples
        burn_in: Burn-in sweeps

    Returns:
        Tensor of shape (n_samples, 1, size, size)
    """
    sampler = MetropolisHastings(model, temperature)
    samples = sampler.sample(n_samples, n_sweeps, burn_in=burn_in)

    # Add channel dimension
    return samples.unsqueeze(1).float()


if __name__ == "__main__":
    # Quick training test
    print("Generating training data...")
    ising = IsingModel(size=16)
    data = generate_training_data(ising, temperature=2.27, n_samples=100, burn_in=200)
    print(f"Training data shape: {data.shape}")

    # Create dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create model and trainer
    model = ScoreNetwork(in_channels=1, base_channels=16, num_blocks=2)
    trainer = Trainer(model, learning_rate=1e-3)

    # Train for a few epochs
    print("\nTraining...")
    history = trainer.train(dataloader, epochs=5, verbose=True)

    print(f"\nFinal loss: {history['train_loss'][-1]:.4f}")
    print("Training test passed!")
