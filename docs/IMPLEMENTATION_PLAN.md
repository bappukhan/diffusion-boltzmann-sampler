# Implementation Plan: Diffusion Models as Boltzmann Samplers

## Executive Summary

This project implements score-based diffusion models to sample from Boltzmann distributions in statistical mechanics. The system provides a FastAPI backend serving PyTorch models with a React frontend for interactive visualization.

**Expert Role:** ML Engineer / Computational Physics Engineer
**Rationale:** This project requires deep understanding of generative models (diffusion, score matching), statistical mechanics (Boltzmann distributions, phase transitions), and full-stack development (FastAPI, React, real-time visualization).

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         React Frontend                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │    │
│  │  │ Ising        │  │ Diffusion    │  │ Correlation  │  │ Control    │  │    │
│  │  │ Visualizer   │  │ Animation    │  │ Plots        │  │ Panel      │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │ HTTP/WebSocket                                    │
└──────────────────────────────┼──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Backend                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │    │
│  │  │ /sampling    │  │ /training    │  │ /analysis    │  │ /websocket │  │    │
│  │  │ endpoints    │  │ endpoints    │  │ endpoints    │  │ streaming  │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                              │                                                   │
└──────────────────────────────┼──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ML LAYER                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         PyTorch Engine                                   │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │    │
│  │  │ Score Network    │  │ Diffusion        │  │ Physical Systems    │   │    │
│  │  │ (U-Net/MLP)      │  │ Scheduler        │  │ (Ising, L-J)        │   │    │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────────┘   │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │    │
│  │  │ MCMC Baseline    │  │ Trainer          │  │ Analysis Tools      │   │    │
│  │  │ (Metropolis)     │  │                  │  │                     │   │    │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Interaction                    Backend Processing                ML Computation
─────────────────                   ──────────────────                ──────────────

[Set Temperature] ──────────────► [Validate Params] ─────────────► [Update Model]
        │                                  │                              │
        ▼                                  ▼                              ▼
[Click "Sample"] ──────────────► [POST /sampling]  ─────────────► [Run Diffusion]
        │                                  │                              │
        │                                  ▼                              ▼
        │                          [WebSocket Stream] ◄──────────── [Yield Steps]
        │                                  │
        ▼                                  ▼
[View Animation] ◄──────────────── [Send Frames]
        │
        ▼
[Compare w/ MCMC] ─────────────► [GET /analysis/compare] ────────► [Compute Stats]
        │                                  │                              │
        ▼                                  ▼                              ▼
[View Correlation] ◄─────────────── [Return JSON]  ◄──────────────── [g(r), M]
```

### Component Interactions

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Frontend      │         │   Backend       │         │   ML Engine     │
│   Components    │         │   Services      │         │   Modules       │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│                 │  REST   │                 │         │                 │
│ ControlPanel ───┼────────►│ SamplingRouter ─┼────────►│ DiffusionSampler│
│                 │         │                 │         │                 │
│                 │   WS    │                 │         │                 │
│ DiffusionAnim◄──┼─────────┤ WebSocketMgr◄──┼─────────┤ (yields states) │
│                 │         │                 │         │                 │
│                 │  REST   │                 │         │                 │
│ CorrelationPlot◄┼─────────┤ AnalysisRouter◄┼─────────┤ AnalysisTools   │
│                 │         │                 │         │                 │
│                 │  REST   │                 │         │                 │
│ IsingVisualizer◄┼─────────┤ SystemRouter◄──┼─────────┤ IsingModel      │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## Technology Selection

### Core Stack

| Component | Technology | Version | Rationale | Alternatives Considered |
|-----------|------------|---------|-----------|------------------------|
| ML Framework | PyTorch | 2.0+ | Industry standard, excellent autodiff, eager mode debugging | JAX (steeper learning curve) |
| Backend | FastAPI | 0.100+ | Async native, auto OpenAPI docs, Pydantic validation | Flask (no async), Django (too heavy) |
| Frontend | React | 18+ | Component reusability, large ecosystem | Vue (smaller ecosystem), Svelte (less mature) |
| Language | TypeScript | 5.0+ | Type safety prevents runtime errors | JavaScript (no types) |
| Visualization | Plotly.js | 2.0+ | Interactive plots, animation, 3D support | D3 (lower level), Chart.js (less flexible) |
| State | Zustand | 4.0+ | Minimal boilerplate, no provider wrapping | Redux (verbose), Context (re-render issues) |
| Styling | Tailwind CSS | 3.0+ | Rapid prototyping, consistent design | CSS Modules (more boilerplate), Styled Components (runtime cost) |

### ML-Specific Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Score Network (Ising) | U-Net with self-attention | Captures spatial correlations in lattice |
| Score Network (L-J) | MLP with periodic embeddings | Handles continuous coordinates, translation invariance |
| Noise Schedule | Linear β schedule | Simple, well-understood, easy to tune |
| MCMC Baseline | Metropolis-Hastings | Gold standard for Ising, easy to implement |

### Trade-offs and Fallbacks

| Decision | Trade-off | Fallback |
|----------|-----------|----------|
| CPU-only | Slower training/inference | If too slow, add optional GPU detection |
| Single-node | No distributed training | For larger systems, could add Ray |
| WebSocket streaming | More complex than polling | Fall back to polling with progress endpoint |

---

## Phased Implementation Plan

### Phase 0: Foundation (Estimated: 2-3 days)

**Objective:** Set up development environment and basic project structure.

**Scope:**
- `backend/api/main.py` - FastAPI app with health check
- `backend/ml/systems/base.py` - Abstract base class for physical systems
- `frontend/` - Vite + React + TypeScript scaffold
- Docker Compose for development (optional)

**Deliverables:**
- [ ] FastAPI server running on port 8000
- [ ] React dev server running on port 5173
- [ ] Health check endpoint returns `{"status": "healthy"}`
- [ ] Frontend displays "Connected to backend"

**Verification:**
```bash
# Backend
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Frontend
npm run dev
# Visit http://localhost:5173 - should show connection status
```

**Definition of Done:**
- Backend starts without errors
- Frontend builds without TypeScript errors
- API documentation available at `/docs`

**Code Skeleton:**

```python
# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Diffusion Boltzmann Sampler",
    description="Neural sampling from Boltzmann distributions",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

---

### Phase 1: Ising Model Core (Estimated: 3-4 days)

**Objective:** Implement 2D Ising model with energy computation and MCMC baseline.

**Scope:**
- `backend/ml/systems/ising.py` - Ising model class
- `backend/ml/samplers/mcmc.py` - Metropolis-Hastings sampler
- `backend/api/routes/sampling.py` - Sampling endpoint
- Unit tests for energy computation

**Deliverables:**
- [ ] `IsingModel` class with `energy()`, `magnetization()`, `local_energy()`
- [ ] `MetropolisHastings` sampler with configurable temperature
- [ ] `/sample/mcmc` endpoint returning samples
- [ ] Tests verifying energy at T=0 (ground state) and T=∞ (random)

**Verification:**
```python
# Test: Ground state at T→0 should be all +1 or all -1
model = IsingModel(size=16, J=1.0, h=0.0)
sampler = MetropolisHastings(model, temperature=0.1)
samples = sampler.sample(n_samples=100, n_steps=1000)
final_mag = abs(samples[-1].mean())
assert final_mag > 0.95, "Should be near ground state"
```

**Technical Challenges:**
1. **Periodic boundary conditions** - Must wrap indices correctly
2. **Energy difference computation** - Only compute local changes for efficiency
3. **Thermalization** - Need sufficient burn-in before collecting samples

**Code Skeleton:**

```python
# backend/ml/systems/ising.py
import torch
from typing import Tuple

class IsingModel:
    """2D Ising model on a square lattice with periodic boundaries."""

    def __init__(self, size: int, J: float = 1.0, h: float = 0.0):
        self.size = size
        self.J = J  # Coupling strength
        self.h = h  # External field

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute total energy: E = -J Σ s_i s_j - h Σ s_i"""
        # Nearest neighbor interactions (periodic)
        right = torch.roll(spins, -1, dims=-1)
        down = torch.roll(spins, -1, dims=-2)
        interaction = -self.J * (spins * right + spins * down).sum(dim=(-1, -2))
        field = -self.h * spins.sum(dim=(-1, -2))
        return interaction + field

    def local_energy_diff(self, spins: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """Energy change from flipping spin at (i,j)."""
        s = spins[..., i, j]
        neighbors = (
            spins[..., (i+1) % self.size, j] +
            spins[..., (i-1) % self.size, j] +
            spins[..., i, (j+1) % self.size] +
            spins[..., i, (j-1) % self.size]
        )
        return 2 * s * (self.J * neighbors + self.h)

    def magnetization(self, spins: torch.Tensor) -> torch.Tensor:
        """Mean magnetization per spin."""
        return spins.mean(dim=(-1, -2))

    def score(self, spins: torch.Tensor, temperature: float) -> torch.Tensor:
        """True score function: ∇log p = -∇E / T"""
        # For discrete spins, this is the expected flip direction
        # Implemented via softmax over {-1, +1} for each site
        raise NotImplementedError("Score for discrete system")
```

---

### Phase 2: Score Network Architecture (Estimated: 4-5 days)

**Objective:** Implement the neural network that learns the score function.

**Scope:**
- `backend/ml/models/score_network.py` - U-Net style architecture
- `backend/ml/models/diffusion.py` - Forward/reverse diffusion processes
- `backend/ml/models/noise_schedule.py` - β schedule implementations

**Deliverables:**
- [ ] `ScoreNetwork` with time conditioning
- [ ] `DiffusionProcess` with forward noising and reverse sampling
- [ ] Tests verifying output shapes and gradient flow

**Verification:**
```python
# Test: Score network output shape matches input
net = ScoreNetwork(in_channels=1, time_embed_dim=64)
x = torch.randn(4, 1, 32, 32)  # Batch of 32x32 lattices
t = torch.rand(4)  # Time values in [0, 1]
score = net(x, t)
assert score.shape == x.shape, "Score should match input shape"
```

**Technical Challenges:**
1. **Time embedding** - Must condition score on diffusion time
2. **Architecture size** - Balance expressiveness vs CPU speed
3. **Numerical stability** - Score can be large at low noise levels

**Code Skeleton:**

```python
# backend/ml/models/score_network.py
import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class ScoreNetwork(nn.Module):
    """U-Net style score network for lattice systems."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_embed_dim: int = 64,
        num_blocks: int = 3
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(ConvBlock(ch, out_ch, time_embed_dim))
            ch = out_ch

        # Middle
        self.middle = ConvBlock(ch, ch, time_embed_dim)

        # Decoder (with skip connections)
        self.decoder = nn.ModuleList()
        for i in range(num_blocks - 1, -1, -1):
            out_ch = base_channels * (2 ** i) if i > 0 else in_channels
            self.decoder.append(ConvBlock(ch * 2, out_ch, time_embed_dim))
            ch = out_ch

        self.final = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))

        # Encoder with skip connections
        skips = []
        h = x
        for block in self.encoder:
            h = block(h, t_emb)
            skips.append(h)

        h = self.middle(h, t_emb)

        # Decoder
        for block, skip in zip(self.decoder, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.final(h)
```

---

### Phase 3: Training Loop (Estimated: 3-4 days)

**Objective:** Implement denoising score matching training.

**Scope:**
- `backend/ml/training/trainer.py` - Training loop with logging
- `backend/ml/training/losses.py` - Score matching loss
- `backend/api/routes/training.py` - Training endpoints

**Deliverables:**
- [ ] `ScoreMatchingLoss` computing denoising objective
- [ ] `Trainer` class with training loop
- [ ] `/training/start` endpoint to initiate training
- [ ] `/training/status` endpoint to check progress

**Verification:**
```python
# Test: Loss decreases over training
trainer = Trainer(model, ising, learning_rate=1e-3)
initial_loss = trainer.evaluate()
trainer.train(epochs=10)
final_loss = trainer.evaluate()
assert final_loss < initial_loss, "Loss should decrease"
```

**Technical Challenges:**
1. **Ground truth score** - For Ising, need to compute ∇log p properly
2. **Noise level sampling** - Uniform vs importance-weighted
3. **Gradient clipping** - Prevent exploding gradients

**Code Skeleton:**

```python
# backend/ml/training/trainer.py
import torch
from torch.optim import Adam
from typing import Optional, Callable
from tqdm import tqdm

class Trainer:
    """Denoising score matching trainer."""

    def __init__(
        self,
        score_network: torch.nn.Module,
        physical_system,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        self.model = score_network.to(device)
        self.system = physical_system
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def score_matching_loss(
        self,
        x_0: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Denoising score matching loss.

        L = E_t E_{x_t|x_0} ||s_θ(x_t, t) - ∇log p(x_t|x_0)||²

        For Gaussian noising: ∇log p(x_t|x_0) = -(x_t - α_t x_0) / σ_t²
        """
        batch_size = x_0.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)

        # Noise schedule (linear)
        alpha_t = 1 - t
        sigma_t = t

        # Sample x_t from q(x_t|x_0) = N(α_t x_0, σ_t² I)
        noise = torch.randn_like(x_0)
        x_t = alpha_t[:, None, None, None] * x_0 + sigma_t[:, None, None, None] * noise

        # Target score: ∇log p(x_t|x_0) = -noise / σ_t
        target_score = -noise / (sigma_t[:, None, None, None] + 1e-8)

        # Predicted score
        pred_score = self.model(x_t, t)

        # MSE loss weighted by σ_t²
        loss = ((pred_score - target_score) ** 2).mean()
        return loss

    def train_epoch(
        self,
        dataloader,
        temperature: float,
        callback: Optional[Callable] = None
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.score_matching_loss(batch, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

            if callback:
                callback(loss.item())

        return total_loss / len(dataloader)
```

---

### Phase 4: Diffusion Sampling (Estimated: 3-4 days)

**Objective:** Implement reverse diffusion sampling from trained model.

**Scope:**
- `backend/ml/samplers/diffusion.py` - Reverse SDE sampler
- Integration with existing API routes
- Comparison utilities with MCMC

**Deliverables:**
- [ ] `DiffusionSampler` with configurable steps
- [ ] `/sample/diffusion` endpoint
- [ ] Generator yielding intermediate states for animation

**Verification:**
```python
# Test: Generated samples have correct energy distribution
sampler = DiffusionSampler(trained_model, steps=100)
samples = sampler.sample(n_samples=1000, temperature=2.0)
energies = [ising.energy(s) for s in samples]
# Compare histogram to MCMC baseline
```

**Code Skeleton:**

```python
# backend/ml/samplers/diffusion.py
import torch
from typing import Generator, Tuple

class DiffusionSampler:
    """Reverse diffusion sampler using trained score network."""

    def __init__(
        self,
        score_network: torch.nn.Module,
        num_steps: int = 100,
        device: str = "cpu"
    ):
        self.model = score_network.to(device)
        self.model.eval()
        self.num_steps = num_steps
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Reverse diffusion
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps, 0, -1):
            t = torch.full((shape[0],), i / self.num_steps, device=self.device)

            # Score prediction
            score = self.model(x, t)

            # Noise schedule derivatives
            sigma_t = t[0].item()

            # Reverse SDE step
            # dx = σ² s_θ(x,t) dt + σ dW
            noise = torch.randn_like(x) if i > 1 else 0
            x = x + (sigma_t ** 2) * score * dt + sigma_t * (dt ** 0.5) * noise

        return x

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        shape: Tuple[int, ...],
        temperature: float = 1.0
    ) -> Generator[torch.Tensor, None, None]:
        """Generate samples yielding intermediate states."""
        x = torch.randn(shape, device=self.device)
        yield x.clone()

        dt = 1.0 / self.num_steps
        for i in range(self.num_steps, 0, -1):
            t = torch.full((shape[0],), i / self.num_steps, device=self.device)
            score = self.model(x, t)
            sigma_t = t[0].item()
            noise = torch.randn_like(x) if i > 1 else 0
            x = x + (sigma_t ** 2) * score * dt + sigma_t * (dt ** 0.5) * noise
            yield x.clone()
```

---

### Phase 5: React Frontend Core (Estimated: 4-5 days)

**Objective:** Build interactive frontend with visualization components.

**Scope:**
- `frontend/src/components/IsingVisualizer.tsx`
- `frontend/src/components/ControlPanel.tsx`
- `frontend/src/services/api.ts`
- State management with Zustand

**Deliverables:**
- [ ] Ising lattice visualization (color-coded spins)
- [ ] Temperature slider with phase transition marker
- [ ] Lattice size selector
- [ ] API service for backend communication

**Verification:**
- Render 32x32 lattice in < 100ms
- Temperature slider updates immediately (no lag)
- API errors display user-friendly messages

**Code Skeleton:**

```typescript
// frontend/src/components/IsingVisualizer.tsx
import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

interface IsingVisualizerProps {
  spins: number[][];
  size: number;
}

export const IsingVisualizer: React.FC<IsingVisualizerProps> = ({ spins, size }) => {
  const plotData = useMemo(() => [{
    z: spins,
    type: 'heatmap' as const,
    colorscale: [
      [0, '#3b82f6'],   // Blue for -1
      [1, '#ef4444'],   // Red for +1
    ],
    showscale: false,
    hoverongaps: false,
  }], [spins]);

  const layout = useMemo(() => ({
    width: 400,
    height: 400,
    margin: { t: 20, b: 20, l: 20, r: 20 },
    xaxis: { visible: false },
    yaxis: { visible: false, scaleanchor: 'x' },
  }), []);

  return (
    <div className="border rounded-lg p-4 bg-white shadow">
      <h3 className="text-lg font-semibold mb-2">Spin Configuration</h3>
      <Plot data={plotData} layout={layout} config={{ displayModeBar: false }} />
    </div>
  );
};
```

```typescript
// frontend/src/store/simulationStore.ts
import { create } from 'zustand';

interface SimulationState {
  temperature: number;
  latticeSize: number;
  spins: number[][] | null;
  isRunning: boolean;
  samplerType: 'mcmc' | 'diffusion';

  setTemperature: (temp: number) => void;
  setLatticeSize: (size: number) => void;
  setSpins: (spins: number[][]) => void;
  setIsRunning: (running: boolean) => void;
  setSamplerType: (type: 'mcmc' | 'diffusion') => void;
}

export const useSimulationStore = create<SimulationState>((set) => ({
  temperature: 2.27,  // Critical temperature
  latticeSize: 32,
  spins: null,
  isRunning: false,
  samplerType: 'diffusion',

  setTemperature: (temperature) => set({ temperature }),
  setLatticeSize: (latticeSize) => set({ latticeSize }),
  setSpins: (spins) => set({ spins }),
  setIsRunning: (isRunning) => set({ isRunning }),
  setSamplerType: (samplerType) => set({ samplerType }),
}));
```

---

### Phase 6: Diffusion Animation (Estimated: 3-4 days)

**Objective:** Visualize the reverse diffusion process in real-time.

**Scope:**
- `frontend/src/components/DiffusionAnimation.tsx`
- WebSocket integration for streaming
- Animation controls (play/pause/step)

**Deliverables:**
- [ ] Animated lattice showing denoising process
- [ ] Step-by-step control
- [ ] Noise level indicator
- [ ] Side-by-side MCMC vs Diffusion comparison

**Code Skeleton:**

```typescript
// frontend/src/components/DiffusionAnimation.tsx
import React, { useState, useEffect, useRef } from 'react';
import { IsingVisualizer } from './IsingVisualizer';
import { useSimulationStore } from '../store/simulationStore';

export const DiffusionAnimation: React.FC = () => {
  const [frames, setFrames] = useState<number[][][]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef<number>();

  const { temperature, latticeSize, samplerType } = useSimulationStore();

  useEffect(() => {
    if (isPlaying && currentFrame < frames.length - 1) {
      animationRef.current = window.setTimeout(() => {
        setCurrentFrame(f => f + 1);
      }, 50); // 20fps
    }
    return () => clearTimeout(animationRef.current);
  }, [isPlaying, currentFrame, frames.length]);

  const handleSample = async () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/sample`);
    const newFrames: number[][][] = [];

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'frame') {
        newFrames.push(data.spins);
        setFrames([...newFrames]);
      } else if (data.type === 'done') {
        ws.close();
      }
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({
        temperature,
        lattice_size: latticeSize,
        sampler: samplerType,
        num_steps: 100,
      }));
    };
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <button
          onClick={handleSample}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Generate Sample
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
      </div>

      {frames.length > 0 && (
        <>
          <IsingVisualizer spins={frames[currentFrame]} size={latticeSize} />
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={currentFrame}
            onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
            className="w-full"
          />
          <p className="text-sm text-gray-600">
            Step {currentFrame + 1} / {frames.length}
          </p>
        </>
      )}
    </div>
  );
};
```

---

### Phase 7: Analysis & Correlation Functions (Estimated: 3-4 days)

**Objective:** Implement statistical analysis comparing neural vs MCMC.

**Scope:**
- `backend/ml/analysis/correlation.py` - Correlation function computations
- `frontend/src/components/CorrelationPlot.tsx`
- Magnetization distribution plots

**Deliverables:**
- [ ] Pair correlation function `g(r)` computation
- [ ] Magnetization histogram
- [ ] Autocorrelation time comparison
- [ ] Interactive comparison plots

**Code Skeleton:**

```python
# backend/ml/analysis/correlation.py
import torch
import numpy as np
from typing import List, Dict

def pair_correlation_ising(samples: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute spin-spin correlation function C(r) = <s_0 s_r>."""
    # samples: (n_samples, size, size)
    n_samples, size, _ = samples.shape
    max_r = size // 2

    correlations = np.zeros(max_r)
    counts = np.zeros(max_r)

    for sample in samples:
        for i in range(size):
            for j in range(size):
                for di in range(-max_r, max_r + 1):
                    for dj in range(-max_r, max_r + 1):
                        r = int(np.sqrt(di**2 + dj**2))
                        if 0 < r < max_r:
                            ni, nj = (i + di) % size, (j + dj) % size
                            correlations[r] += sample[i, j] * sample[ni, nj]
                            counts[r] += 1

    correlations /= np.maximum(counts, 1)
    distances = np.arange(max_r)

    return {"r": distances, "C_r": correlations}


def magnetization_distribution(samples: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute magnetization distribution P(M)."""
    magnetizations = samples.mean(dim=(-1, -2)).numpy()
    hist, bin_edges = np.histogram(magnetizations, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return {"M": bin_centers, "P_M": hist}


def autocorrelation_time(samples: torch.Tensor, observable: str = "magnetization") -> float:
    """Estimate integrated autocorrelation time."""
    if observable == "magnetization":
        obs = samples.mean(dim=(-1, -2)).numpy()
    else:
        raise ValueError(f"Unknown observable: {observable}")

    n = len(obs)
    mean = obs.mean()
    var = obs.var()

    if var < 1e-10:
        return 1.0

    # Compute autocorrelation
    autocorr = np.correlate(obs - mean, obs - mean, mode='full')[n-1:]
    autocorr /= var * np.arange(n, 0, -1)

    # Integrate until first negative value
    tau = 0.5
    for i in range(1, n):
        if autocorr[i] < 0:
            break
        tau += autocorr[i]

    return tau
```

---

### Phase 8: Lennard-Jones Extension (Estimated: 4-5 days)

**Objective:** Extend system to continuous Lennard-Jones fluid.

**Scope:**
- `backend/ml/systems/lennard_jones.py`
- `backend/ml/models/score_network_continuous.py` - MLP architecture
- `frontend/src/components/LJVisualizer.tsx`

**Deliverables:**
- [ ] Lennard-Jones potential and force computation
- [ ] Continuous score network with periodic embeddings
- [ ] Particle position visualization
- [ ] Radial distribution function g(r)

**Technical Challenges:**
1. **Periodic boundaries** - Minimum image convention
2. **Cutoff radius** - Truncate potential for efficiency
3. **Softening** - Avoid singularity at r→0

**Code Skeleton:**

```python
# backend/ml/systems/lennard_jones.py
import torch
from typing import Tuple

class LennardJonesSystem:
    """2D Lennard-Jones fluid with periodic boundaries."""

    def __init__(
        self,
        n_particles: int,
        box_size: float,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        cutoff: float = 2.5
    ):
        self.n_particles = n_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff * sigma

    def minimum_image(self, dr: torch.Tensor) -> torch.Tensor:
        """Apply minimum image convention for periodic boundaries."""
        return dr - self.box_size * torch.round(dr / self.box_size)

    def pairwise_distances(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute all pairwise distances with minimum image."""
        # positions: (..., n_particles, 2)
        r_i = positions[..., :, None, :]  # (..., n, 1, 2)
        r_j = positions[..., None, :, :]  # (..., 1, n, 2)
        dr = self.minimum_image(r_i - r_j)  # (..., n, n, 2)
        dist = torch.sqrt((dr ** 2).sum(dim=-1) + 1e-10)  # (..., n, n)
        return dist, dr

    def energy(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total Lennard-Jones energy."""
        dist, _ = self.pairwise_distances(positions)

        # Mask self-interactions and apply cutoff
        mask = (dist > 0) & (dist < self.cutoff)
        dist_masked = torch.where(mask, dist, torch.ones_like(dist))

        # LJ potential: 4ε[(σ/r)^12 - (σ/r)^6]
        r6 = (self.sigma / dist_masked) ** 6
        r12 = r6 ** 2
        potential = 4 * self.epsilon * (r12 - r6)
        potential = torch.where(mask, potential, torch.zeros_like(potential))

        # Sum over pairs (avoiding double counting)
        return 0.5 * potential.sum(dim=(-1, -2))

    def forces(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute forces on all particles: F = -∇E."""
        dist, dr = self.pairwise_distances(positions)

        mask = (dist > 0) & (dist < self.cutoff)
        dist_masked = torch.where(mask, dist, torch.ones_like(dist))

        # Force magnitude: 24ε/r [2(σ/r)^12 - (σ/r)^6]
        r6 = (self.sigma / dist_masked) ** 6
        r12 = r6 ** 2
        force_mag = 24 * self.epsilon / dist_masked * (2 * r12 - r6)
        force_mag = torch.where(mask, force_mag, torch.zeros_like(force_mag))

        # Force vectors
        unit_vectors = dr / (dist_masked[..., None] + 1e-10)
        forces = (force_mag[..., None] * unit_vectors).sum(dim=-2)

        return forces

    def score(self, positions: torch.Tensor, temperature: float) -> torch.Tensor:
        """True score function: ∇log p = F / (kT)."""
        return self.forces(positions) / temperature
```

---

### Phase 9: Polish & Documentation (Estimated: 2-3 days)

**Objective:** Final polish, error handling, and documentation.

**Scope:**
- Error handling throughout
- Loading states and skeletons
- Responsive design
- API documentation
- User guide

**Deliverables:**
- [ ] Comprehensive error handling
- [ ] Loading indicators
- [ ] Mobile-responsive layout
- [ ] API documentation at `/docs`
- [ ] User guide in README

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning | Mitigation |
|------|------------|--------|---------------|------------|
| Score network fails to converge | Medium | High | Loss plateaus early | Start with smaller network, add skip connections |
| CPU too slow for interactive use | Medium | Medium | Sampling takes > 10s | Reduce lattice size, cache trained models |
| WebSocket disconnections | Low | Medium | Frequent reconnects | Add reconnection logic, fallback to polling |
| Phase transition instability | Medium | Low | NaN in energy | Add numerical safeguards, temperature bounds |
| Frontend performance with large lattices | Medium | Medium | Laggy visualization | Use canvas instead of SVG, reduce update frequency |

---

## Testing Strategy

### Unit Tests

```python
# backend/tests/test_ising.py
import pytest
import torch
from ml.systems.ising import IsingModel

class TestIsingModel:
    def test_ground_state_energy(self):
        """All spins aligned should give minimum energy."""
        model = IsingModel(size=8, J=1.0, h=0.0)
        all_up = torch.ones(1, 8, 8)
        all_down = -torch.ones(1, 8, 8)

        E_up = model.energy(all_up)
        E_down = model.energy(all_down)

        # Should be equal (Z2 symmetry)
        assert torch.allclose(E_up, E_down)
        # Should be negative (ferromagnetic)
        assert E_up < 0

    def test_energy_flip_symmetry(self):
        """Flipping all spins should preserve energy at h=0."""
        model = IsingModel(size=8, J=1.0, h=0.0)
        spins = torch.randint(0, 2, (1, 8, 8)) * 2 - 1

        E_original = model.energy(spins)
        E_flipped = model.energy(-spins)

        assert torch.allclose(E_original, E_flipped)

    def test_local_energy_diff_consistency(self):
        """Local energy diff should match full recomputation."""
        model = IsingModel(size=8, J=1.0, h=0.0)
        spins = torch.randint(0, 2, (1, 8, 8)).float() * 2 - 1

        E_before = model.energy(spins)

        i, j = 3, 5
        dE_local = model.local_energy_diff(spins, i, j)

        spins_flipped = spins.clone()
        spins_flipped[0, i, j] *= -1
        E_after = model.energy(spins_flipped)

        dE_full = E_after - E_before
        assert torch.allclose(dE_local, dE_full)
```

### Integration Tests

```python
# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_mcmc_sampling():
    response = client.post("/sample/mcmc", json={
        "temperature": 2.27,
        "lattice_size": 16,
        "n_samples": 10,
        "n_steps": 100
    })
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) == 10
```

### First Three Tests to Write

1. **`test_ising_ground_state_energy`** - Verify energy computation correctness
2. **`test_mcmc_detailed_balance`** - MCMC satisfies detailed balance
3. **`test_score_network_output_shape`** - Neural network produces correct output shape

---

## First Concrete Task

### File to Create

`backend/ml/systems/ising.py`

### Function Signature

```python
def energy(self, spins: torch.Tensor) -> torch.Tensor:
    """Compute total Ising model energy.

    Args:
        spins: Tensor of shape (..., size, size) with values in {-1, +1}

    Returns:
        Tensor of shape (...,) containing energy for each configuration

    Energy formula:
        E = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    where the first sum is over nearest neighbors with periodic boundaries.
    """
```

### Starter Code

```python
# backend/ml/systems/ising.py
"""2D Ising model implementation."""

import torch
from typing import Tuple


class IsingModel:
    """2D Ising model on a square lattice with periodic boundary conditions.

    The Hamiltonian is:
        H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    where s_i ∈ {-1, +1}, J is the coupling constant, and h is the external field.

    Attributes:
        size: Linear size of the square lattice (size × size spins)
        J: Coupling constant (J > 0 is ferromagnetic)
        h: External magnetic field
    """

    def __init__(self, size: int, J: float = 1.0, h: float = 0.0):
        """Initialize Ising model.

        Args:
            size: Linear size of square lattice
            J: Coupling constant (default 1.0 for ferromagnetic)
            h: External field (default 0.0)
        """
        self.size = size
        self.J = J
        self.h = h

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute total energy of spin configurations.

        Args:
            spins: Tensor of shape (..., size, size) with values in {-1, +1}

        Returns:
            Tensor of shape (...,) containing energy for each configuration
        """
        # TODO: Implement this
        # Hint: Use torch.roll for periodic boundary conditions
        # Right neighbor: torch.roll(spins, -1, dims=-1)
        # Down neighbor: torch.roll(spins, -1, dims=-2)
        raise NotImplementedError("Implement energy computation")

    def magnetization(self, spins: torch.Tensor) -> torch.Tensor:
        """Compute mean magnetization per spin.

        Args:
            spins: Tensor of shape (..., size, size)

        Returns:
            Tensor of shape (...,) containing magnetization per spin
        """
        return spins.mean(dim=(-1, -2))


if __name__ == "__main__":
    # Quick test
    model = IsingModel(size=8, J=1.0, h=0.0)

    # Ground state: all spins aligned
    all_up = torch.ones(8, 8)
    E = model.energy(all_up)
    print(f"Ground state energy: {E.item()}")
    # Expected: -2 * J * size^2 = -2 * 1 * 64 = -128
```

### Verification Method

```bash
cd backend
python -c "
import torch
from ml.systems.ising import IsingModel

model = IsingModel(size=8, J=1.0, h=0.0)
all_up = torch.ones(8, 8)
E = model.energy(all_up)
expected = -2 * 8 * 8  # Each of 64 spins has 2 unique bonds, each contributing -J
print(f'Energy: {E.item()}, Expected: {expected}')
assert abs(E.item() - expected) < 1e-6, 'Energy calculation incorrect!'
print('Test passed!')
"
```

### First Commit Message

```
feat(ising): implement 2D Ising model energy computation

Add IsingModel class with:
- Energy calculation using periodic boundary conditions
- Magnetization per spin computation
- Support for batched configurations

The energy formula E = -J Σ s_i s_j - h Σ s_i is computed
efficiently using torch.roll for neighbor access.
```

---

## Summary Timeline

| Phase | Description | Est. Duration |
|-------|-------------|---------------|
| 0 | Foundation & Setup | 2-3 days |
| 1 | Ising Model Core | 3-4 days |
| 2 | Score Network Architecture | 4-5 days |
| 3 | Training Loop | 3-4 days |
| 4 | Diffusion Sampling | 3-4 days |
| 5 | React Frontend Core | 4-5 days |
| 6 | Diffusion Animation | 3-4 days |
| 7 | Analysis & Correlations | 3-4 days |
| 8 | Lennard-Jones Extension | 4-5 days |
| 9 | Polish & Documentation | 2-3 days |

**Total Estimated: 32-41 days**

---

## Dependencies Graph

```
Phase 0 (Foundation)
    │
    ├──► Phase 1 (Ising Core)
    │        │
    │        ├──► Phase 2 (Score Network)
    │        │        │
    │        │        └──► Phase 3 (Training)
    │        │                  │
    │        │                  └──► Phase 4 (Diffusion Sampling)
    │        │                            │
    │        │                            └──► Phase 7 (Analysis)
    │        │                                      │
    │        │                                      └──► Phase 8 (L-J Extension)
    │        │
    │        └──► Phase 5 (Frontend Core)
    │                  │
    │                  └──► Phase 6 (Animation)
    │
    └──► Phase 9 (Polish) ◄─── All phases
```
