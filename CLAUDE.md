# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Score-based diffusion models for sampling from Boltzmann distributions in statistical mechanics. The system trains neural networks to learn the score function (gradient of log probability) for physical systems like the 2D Ising model, enabling faster sampling than traditional MCMC methods.

## Common Commands

### Development
```bash
# Start backend (FastAPI on port 8000)
make backend
# Or directly:
cd backend && uvicorn api.main:app --reload --port 8000

# Start frontend (Vite on port 5173)
make frontend
# Or directly:
cd frontend && npm run dev
```

### Testing
```bash
# All tests
make test

# Backend tests only
make test-backend
# Or: pytest backend/tests/ -v

# Single test file
pytest backend/tests/test_ising.py -v

# Single test
pytest backend/tests/test_ising.py::TestIsingModel::test_ground_state_energy -v

# Frontend tests
make test-frontend
# Or: cd frontend && npm run test:run
```

### Linting & Formatting
```bash
# Lint backend
make lint-backend
# Or: ruff check backend/ && mypy backend/ --ignore-missing-imports

# Lint frontend
cd frontend && npm run lint

# Format backend
make format-backend
# Or: black backend/ && isort backend/
```

## Architecture

### Three-Layer Design
```
Frontend (React/TypeScript)  →  Backend (FastAPI)  →  ML Engine (PyTorch)
     Plotly visualizations        REST + WebSocket      Score networks
     Zustand state                 Pydantic models       Diffusion samplers
```

### Backend Structure (backend/)
- **api/main.py**: FastAPI app with lifespan management, WebSocket endpoint
- **api/websocket.py**: WebSocket handlers with ConnectionManager for streaming samples
  - Streams frame data with noise level (sigma), diffusion time (t), and metadata
  - Supports both MCMC and diffusion samplers
- **api/routes/**: Endpoints split by domain (sampling.py, training.py, analysis.py)
- **ml/systems/ising.py**: 2D Ising model with energy, magnetization, score functions
- **ml/models/**: Score network (U-Net), diffusion process, noise schedules
- **ml/samplers/**: MCMC (Metropolis-Hastings) and diffusion samplers
- **ml/training/**: Score matching loss and trainer
- **ml/analysis/**: Correlation functions, distribution comparisons, statistics

### Key ML Concepts
- **Score function**: `s(x) = ∇log p(x) = -∇E(x)/kT` (force field / temperature)
- **Training**: Denoising score matching - learn to predict noise added at time t
- **Sampling**: Reverse SDE from pure noise using learned score
- **Discretization**: Convert continuous diffusion output to discrete spins {-1, +1}

### Frontend Structure (frontend/src/)
- **components/**: IsingVisualizer (heatmap), DiffusionAnimation, CorrelationPlot, ControlPanel
  - **DiffusionProgressVisualization**: Visual timeline showing denoising trajectory phases
  - **MCMCvsDiffusionComparison**: Side-by-side comparison of sampling methods
  - **ComparisonPanel**: Reusable panel with SVG spin grid renderer
- **hooks/**: Custom React hooks
  - **useWebSocket**: WebSocket connection with reconnection and state management
  - **useSynchronizedPlayback**: Coordinated playback across multiple frame sequences
  - **useHealthCheck**: Backend connection monitoring
- **store/simulationStore.ts**: Zustand store for temperature, lattice size, sampler type, frame metadata
- **store/selectors.ts**: Derived state selectors including diffusion phase and metadata
- **services/api.ts**: REST and WebSocket communication with backend

### Global State
The backend uses `AppState` in `api/main.py` to hold:
- `ising_model`: Current Ising model instance
- `score_network`: Trained score network
- `diffusion`: Diffusion process parameters
- Training status flags

Access via `get_state()` function for dependency injection in routes.

## Testing Patterns

### Ising Model Tests
Verify energy calculations with known ground states (all +1 or all -1 should give E = -2JN where N = size²).

### API Tests
Use FastAPI's TestClient:
```python
from fastapi.testclient import TestClient
from backend.api.main import app
client = TestClient(app)
```

### ML Tests
Test score network output shapes match input shapes. Test diffusion sampler produces valid trajectories.

## Physical Constants
- **Critical temperature**: `T_c ≈ 2.269` (2D Ising model phase transition)
- **Default lattice size**: 32×32
- **Default diffusion steps**: 100

## Type Aliases (backend/ml/types.py)
Custom type aliases for clarity: `SpinConfiguration`, `EnergyTensor`, `ScoreTensor`, `Temperature`, `DiffusionTime`
