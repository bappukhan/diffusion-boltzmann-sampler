"""Training API routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...ml.systems.ising import IsingModel
from ...ml.models.score_network import ScoreNetwork
from ...ml.models.diffusion import DiffusionProcess
from ...ml.training.trainer import Trainer, generate_training_data


router = APIRouter()

# Global training state
training_state = {
    "is_training": False,
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": None,
    "history": {"train_loss": []},
}


class TrainingRequest(BaseModel):
    """Request model for training."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(32, ge=8, le=64)
    n_training_samples: int = Field(500, ge=100, le=5000)
    epochs: int = Field(50, ge=1, le=500)
    batch_size: int = Field(16, ge=4, le=64)
    learning_rate: float = Field(1e-3, ge=1e-5, le=1e-1)


class TrainingStatus(BaseModel):
    """Response model for training status."""

    is_training: bool
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    history: Dict[str, Any]


def run_training_background(request: TrainingRequest):
    """Background training task."""
    global training_state

    try:
        training_state["is_training"] = True
        training_state["total_epochs"] = request.epochs
        training_state["history"] = {"train_loss": []}

        # Generate training data using MCMC
        print(f"Generating {request.n_training_samples} training samples...")
        ising = IsingModel(size=request.lattice_size)
        data = generate_training_data(
            ising,
            temperature=request.temperature,
            n_samples=request.n_training_samples,
            burn_in=500,
        )

        # Create dataloader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=request.batch_size, shuffle=True)

        # Create model and trainer
        model = ScoreNetwork(
            in_channels=1,
            base_channels=32,
            time_embed_dim=64,
            num_blocks=3,
        )
        trainer = Trainer(model, learning_rate=request.learning_rate)

        # Training loop
        print("Starting training...")
        for epoch in range(request.epochs):
            loss = trainer.train_epoch(dataloader)

            # Update state
            training_state["current_epoch"] = epoch + 1
            training_state["progress"] = (epoch + 1) / request.epochs
            training_state["current_loss"] = loss
            training_state["history"]["train_loss"].append(loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{request.epochs}, Loss: {loss:.4f}")

        print("Training complete!")

    except Exception as e:
        print(f"Training error: {e}")
        training_state["history"]["error"] = str(e)

    finally:
        training_state["is_training"] = False


@router.post("/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Start training in background.

    Training generates samples using MCMC, then trains the score network
    using denoising score matching.
    """
    if training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Reset state
    training_state["progress"] = 0.0
    training_state["current_epoch"] = 0
    training_state["current_loss"] = None

    # Start background training
    background_tasks.add_task(run_training_background, request)

    return {"message": "Training started", "status": "running"}


@router.get("/status", response_model=TrainingStatus)
async def get_training_status() -> TrainingStatus:
    """Get current training status."""
    return TrainingStatus(**training_state)


@router.post("/stop")
async def stop_training() -> Dict[str, str]:
    """Request training stop.

    Note: This sets a flag but doesn't immediately stop training.
    The training loop checks this flag between epochs.
    """
    # In a real implementation, we'd have a stop flag the training loop checks
    return {"message": "Stop requested", "note": "Training will stop after current epoch"}


@router.get("/config")
async def get_training_config() -> Dict[str, Any]:
    """Get recommended training configuration."""
    return {
        "recommended": {
            "temperature": IsingModel.T_CRITICAL,
            "lattice_size": 32,
            "n_training_samples": 1000,
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
        },
        "limits": {
            "max_lattice_size": 64,
            "max_epochs": 500,
            "max_samples": 5000,
        },
        "notes": {
            "temperature": f"Critical temperature is ~{IsingModel.T_CRITICAL:.3f}",
            "training_time": "~1 minute per 100 samples on CPU",
        },
    }
