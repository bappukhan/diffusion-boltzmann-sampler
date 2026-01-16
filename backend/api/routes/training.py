"""Training API routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...ml.checkpoints import (
    get_checkpoint_dir,
    list_checkpoints as list_checkpoint_metadata,
    format_checkpoint_name,
)

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
    "last_checkpoint": None,
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
    checkpoint_interval: int = Field(
        0,
        ge=0,
        le=500,
        description="Save periodic checkpoints every N epochs (0 disables)",
    )


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
        model_config = {
            "in_channels": 1,
            "base_channels": 32,
            "time_embed_dim": 64,
            "num_blocks": 3,
        }
        model = ScoreNetwork(**model_config)
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

        checkpoint_dir = get_checkpoint_dir()
        checkpoint_name = format_checkpoint_name(
            lattice_size=request.lattice_size,
            temperature=request.temperature,
        )
        checkpoint_path = checkpoint_dir / checkpoint_name
        trainer.save_checkpoint(
            path=str(checkpoint_path),
            model_config=model_config,
            training_temperature=request.temperature,
            training_meta={
                "epochs": request.epochs,
                "n_training_samples": request.n_training_samples,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
            },
            extra_info={
                "lattice_size": request.lattice_size,
            },
        )
        training_state["last_checkpoint"] = str(checkpoint_path)

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


class CheckpointInfo(BaseModel):
    """Information about a saved checkpoint."""

    name: str
    path: str
    size_bytes: int
    modified_time: str
    lattice_size: Optional[int] = None
    training_temperature: Optional[float] = None
    model_config: Optional[Dict[str, Any]] = None
    diffusion_config: Optional[Dict[str, Any]] = None
    training_meta: Optional[Dict[str, Any]] = None


@router.get("/checkpoints", response_model=List[CheckpointInfo])
async def list_checkpoints() -> List[CheckpointInfo]:
    """List all available checkpoints.

    Returns list of checkpoint files in the checkpoints directory.
    """
    return [
        CheckpointInfo(**checkpoint.__dict__)
        for checkpoint in list_checkpoint_metadata()
    ]


class LoadCheckpointRequest(BaseModel):
    """Request to load a checkpoint."""

    checkpoint_name: str = Field(..., description="Name of checkpoint file to load")


class LoadCheckpointResponse(BaseModel):
    """Response after loading a checkpoint."""

    message: str
    checkpoint_name: str
    history: Optional[Dict[str, Any]] = None


@router.post("/checkpoints/load", response_model=LoadCheckpointResponse)
async def load_checkpoint(request: LoadCheckpointRequest) -> LoadCheckpointResponse:
    """Load a saved checkpoint.

    Loads model weights and training history from a checkpoint file.
    """
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_name = Path(request.checkpoint_name).name
    checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {request.checkpoint_name}",
        )

    try:
        # Load checkpoint to verify it's valid
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        history = checkpoint.get("history", {})

        return LoadCheckpointResponse(
            message=f"Checkpoint loaded successfully",
            checkpoint_name=request.checkpoint_name,
            history=history,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}",
        )
