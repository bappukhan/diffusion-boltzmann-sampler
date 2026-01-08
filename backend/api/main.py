"""FastAPI application for Diffusion Boltzmann Sampler."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch

from .routes import sampling_router, training_router, analysis_router
from ..ml.systems.ising import IsingModel
from ..ml.models.score_network import ScoreNetwork
from ..ml.models.diffusion import DiffusionProcess
from ..ml.samplers.mcmc import MetropolisHastings


# Global state for ML models
class AppState:
    """Application state holding ML models."""

    def __init__(self):
        self.device = "cpu"
        self.ising_model: IsingModel = None
        self.score_network: ScoreNetwork = None
        self.diffusion: DiffusionProcess = None
        self.is_training = False
        self.training_progress = 0.0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    # Initialize default models
    state.ising_model = IsingModel(size=32)
    state.diffusion = DiffusionProcess()
    state.score_network = ScoreNetwork(
        in_channels=1,
        base_channels=32,
        time_embed_dim=64,
        num_blocks=3,
    ).to(state.device)

    print("Models initialized")
    yield

    # Cleanup
    print("Shutting down")


app = FastAPI(
    title="Diffusion Boltzmann Sampler",
    description="Neural sampling from Boltzmann distributions using score-based diffusion models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sampling_router, prefix="/sample", tags=["Sampling"])
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return {
        "lattice_size": state.ising_model.size if state.ising_model else 32,
        "T_critical": IsingModel.T_CRITICAL,
        "device": state.device,
        "is_training": state.is_training,
        "training_progress": state.training_progress,
    }


@app.websocket("/ws/sample")
async def websocket_sample(websocket: WebSocket):
    """WebSocket endpoint for streaming sampling results.

    Expects JSON message with:
    - temperature: float
    - lattice_size: int
    - sampler: "mcmc" or "diffusion"
    - num_steps: int
    """
    await websocket.accept()

    try:
        # Receive parameters
        params = await websocket.receive_json()
        temperature = params.get("temperature", 2.27)
        lattice_size = params.get("lattice_size", 32)
        sampler_type = params.get("sampler", "mcmc")
        num_steps = params.get("num_steps", 100)

        # Create model for this request
        ising = IsingModel(size=lattice_size)

        if sampler_type == "mcmc":
            # MCMC sampling with trajectory
            sampler = MetropolisHastings(ising, temperature)
            for spins in sampler.sample_with_trajectory(n_steps=num_steps):
                await websocket.send_json({
                    "type": "frame",
                    "spins": spins.tolist(),
                    "energy": ising.energy_per_spin(spins).item(),
                    "magnetization": ising.magnetization(spins).item(),
                })
                await asyncio.sleep(0.01)  # Small delay for animation
        else:
            # Diffusion sampling with trajectory
            from ..ml.samplers.diffusion import PretrainedDiffusionSampler

            sampler = PretrainedDiffusionSampler(
                lattice_size=lattice_size,
                num_steps=num_steps,
            )

            # Use heuristic sampling for demo
            shape = (1, 1, lattice_size, lattice_size)
            for x, t in sampler.sample_with_trajectory(shape, yield_every=max(1, num_steps // 50)):
                spins = torch.sign(x[0, 0])  # Discretize
                await websocket.send_json({
                    "type": "frame",
                    "spins": spins.tolist(),
                    "t": t,
                    "energy": ising.energy_per_spin(spins).item(),
                    "magnetization": ising.magnetization(spins).item(),
                })
                await asyncio.sleep(0.02)

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


def get_state() -> AppState:
    """Get global app state (for dependency injection in routes)."""
    return state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
