"""WebSocket handlers for real-time sampling visualization."""

import asyncio
from typing import Dict, Any, List, Set

from fastapi import WebSocket, WebSocketDisconnect
import torch

from ..ml.systems.ising import IsingModel
from ..ml.samplers.mcmc import MetropolisHastings


class ConnectionManager:
    """Manage WebSocket connections for real-time updates.

    Handles connection lifecycle, broadcasting, and client tracking.
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket to connect
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: The WebSocket to disconnect
        """
        async with self._lock:
            self.active_connections.discard(websocket)

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """Send JSON data to a specific WebSocket.

        Args:
            websocket: The target WebSocket
            data: JSON-serializable data to send

        Returns:
            True if send succeeded, False if connection was lost
        """
        try:
            await websocket.send_json(data)
            return True
        except Exception:
            await self.disconnect(websocket)
            return False

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast JSON data to all connected clients.

        Args:
            data: JSON-serializable data to send
        """
        async with self._lock:
            connections = list(self.active_connections)

        for websocket in connections:
            await self.send_json(websocket, data)

    @property
    def connection_count(self) -> int:
        """Return number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()


async def sample_websocket_handler(
    websocket: WebSocket,
    state: Any,
) -> None:
    """Handle WebSocket connection for streaming sampling results.

    Expects JSON message with:
    - temperature: float
    - lattice_size: int
    - sampler: "mcmc" or "diffusion"
    - num_steps: int

    Args:
        websocket: The WebSocket connection
        state: Application state containing models
    """
    await manager.connect(websocket)

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
            await _stream_mcmc_samples(
                websocket, ising, temperature, num_steps
            )
        else:
            await _stream_diffusion_samples(
                websocket, ising, lattice_size, num_steps
            )

        await manager.send_json(websocket, {"type": "done"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await manager.send_json(websocket, {"type": "error", "message": str(e)})
    finally:
        await manager.disconnect(websocket)


async def _stream_mcmc_samples(
    websocket: WebSocket,
    ising: IsingModel,
    temperature: float,
    num_steps: int,
) -> None:
    """Stream MCMC sampling results frame by frame.

    Args:
        websocket: The WebSocket connection
        ising: Ising model instance
        temperature: Sampling temperature
        num_steps: Number of MCMC steps
    """
    sampler = MetropolisHastings(ising, temperature)

    for step, spins in enumerate(sampler.sample_with_trajectory(n_steps=num_steps)):
        frame_data: Dict[str, Any] = {
            "type": "frame",
            "spins": spins.tolist(),
            "energy": ising.energy_per_spin(spins).item(),
            "magnetization": ising.magnetization(spins).item(),
            "step": step,
            "total_steps": num_steps,
            "progress": (step + 1) / num_steps,
        }
        if not await manager.send_json(websocket, frame_data):
            break  # Connection lost
        await asyncio.sleep(0.01)  # Small delay for animation


async def _stream_diffusion_samples(
    websocket: WebSocket,
    ising: IsingModel,
    lattice_size: int,
    num_steps: int,
) -> None:
    """Stream diffusion sampling results frame by frame.

    Args:
        websocket: The WebSocket connection
        ising: Ising model instance
        lattice_size: Size of the lattice
        num_steps: Number of diffusion steps
    """
    from ..ml.samplers.diffusion import PretrainedDiffusionSampler

    sampler = PretrainedDiffusionSampler(
        lattice_size=lattice_size,
        num_steps=num_steps,
    )

    # Use heuristic sampling for demo
    shape = (1, 1, lattice_size, lattice_size)
    yield_every = max(1, num_steps // 50)

    for step, (x, t) in enumerate(
        sampler.sample_with_trajectory(shape, yield_every=yield_every)
    ):
        spins = torch.sign(x[0, 0])  # Discretize
        frame_data: Dict[str, Any] = {
            "type": "frame",
            "spins": spins.tolist(),
            "t": t,
            "energy": ising.energy_per_spin(spins).item(),
            "magnetization": ising.magnetization(spins).item(),
            "step": step,
            "total_steps": num_steps // yield_every + 1,
            "progress": 1.0 - t,
        }
        if not await manager.send_json(websocket, frame_data):
            break  # Connection lost
        await asyncio.sleep(0.02)
