# API Documentation

This document describes the REST API and WebSocket endpoints for the Diffusion Boltzmann Sampler.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

Check if the API is running and get version info.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "title": "Diffusion Boltzmann Sampler",
  "features": {
    "mcmc_sampling": true,
    "diffusion_sampling": true,
    "websocket_streaming": true
  }
}
```

### Configuration

Get current application configuration.

```http
GET /config
```

**Response:**
```json
{
  "lattice_size": 32,
  "T_critical": 2.269,
  "device": "cpu",
  "is_training": false,
  "training_progress": 0.0
}
```

---

## Sampling Endpoints

### MCMC Sampling

Generate samples using Metropolis-Hastings Monte Carlo.

```http
POST /sample/mcmc
```

**Request Body:**
```json
{
  "temperature": 2.27,
  "lattice_size": 32,
  "n_samples": 10,
  "n_sweeps": 10,
  "burn_in": 100
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 2.27 | Temperature (0.1-10.0) |
| `lattice_size` | int | 32 | Lattice size (8-128) |
| `n_samples` | int | 10 | Number of samples |
| `n_sweeps` | int | 10 | Sweeps between samples |
| `burn_in` | int | 100 | Burn-in sweeps |

**Response:**
```json
{
  "samples": [[[1, -1, ...], ...], ...],
  "energies": [-1.5, -1.4, ...],
  "magnetizations": [0.1, -0.05, ...],
  "temperature": 2.27,
  "lattice_size": 32
}
```

### Diffusion Sampling

Generate samples using the trained diffusion model.

```http
POST /sample/diffusion
```

**Request Body:**
```json
{
  "temperature": 2.27,
  "lattice_size": 32,
  "n_samples": 1,
  "num_steps": 100,
  "checkpoint_name": "ising_32_T2.27.pt",
  "use_trained_model": true,
  "discretize": true,
  "discretization_method": "sign"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 2.27 | Temperature (0.1-10.0) |
| `lattice_size` | int | 32 | Lattice size (8-64) |
| `n_samples` | int | 1 | Number of samples (1-10) |
| `num_steps` | int | 100 | Diffusion steps (10-500) |
| `checkpoint_path` | string | null | Explicit checkpoint path |
| `checkpoint_name` | string | null | Checkpoint file in `CHECKPOINT_DIR` |
| `use_trained_model` | bool | false | Load latest matching checkpoint |
| `discretize` | bool | true | Discretize to ±1 spins |
| `discretization_method` | string | sign | sign, tanh, gumbel, stochastic |

### Random Configuration

Get a random spin configuration.

```http
GET /sample/random?lattice_size=32
```

**Response:**
```json
{
  "spins": [[1, -1, ...], ...],
  "energy": -0.5,
  "magnetization": 0.02,
  "lattice_size": 32
}
```

### Ground State

Get a ground state configuration.

```http
GET /sample/ground_state?lattice_size=32&positive=true
```

---

## Training Endpoints

### List Checkpoints

```http
GET /training/checkpoints
```

### Latest Checkpoint

```http
GET /training/checkpoints/latest?lattice_size=32&temperature=2.27
```

---

## Analysis Endpoints

### Compare Samplers

Compare MCMC and diffusion samplers.

```http
POST /analysis/compare
```

**Request Body:**
```json
{
  "temperature": 2.27,
  "lattice_size": 32,
  "n_samples": 100
}
```

---

## WebSocket Endpoints

### Streaming Sampling

Stream sampling results in real-time.

```
ws://localhost:8000/ws/sample
```

**Send Parameters:**
```json
{
  "temperature": 2.27,
  "lattice_size": 32,
  "sampler": "mcmc",
  "num_steps": 100
}
```

**Receive Frames:**
```json
{
  "type": "frame",
  "spins": [[1, -1, ...], ...],
  "energy": -1.5,
  "magnetization": 0.1
}
```

**Completion:**
```json
{
  "type": "done"
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

---

## Error Handling

All endpoints return errors in this format:

```json
{
  "detail": "Error message"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `422` - Validation error
- `500` - Internal server error

---

## Examples

### cURL Examples

**Check health:**
```bash
curl http://localhost:8000/health
```

**Get random configuration:**
```bash
curl http://localhost:8000/sample/random?lattice_size=16
```

**Generate MCMC samples:**
```bash
curl -X POST http://localhost:8000/sample/mcmc \
  -H "Content-Type: application/json" \
  -d '{"temperature": 2.27, "lattice_size": 32, "n_samples": 5}'
```

### Python Examples

```python
import requests

# Check health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Generate MCMC samples
response = requests.post(
    "http://localhost:8000/sample/mcmc",
    json={
        "temperature": 2.27,
        "lattice_size": 32,
        "n_samples": 10,
    },
)
data = response.json()
print(f"Generated {len(data['samples'])} samples")
```

### JavaScript Examples

```javascript
// Check health
const health = await fetch('http://localhost:8000/health').then(r => r.json());
console.log(health.status);

// WebSocket sampling
const ws = new WebSocket('ws://localhost:8000/ws/sample');
ws.onopen = () => {
  ws.send(JSON.stringify({
    temperature: 2.27,
    lattice_size: 32,
    sampler: 'mcmc',
    num_steps: 100,
  }));
};
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'frame') {
    console.log('Energy:', data.energy);
  }
};
```
