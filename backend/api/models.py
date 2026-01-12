"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# =============================================================================
# Health and Config
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    title: str = Field(..., description="API title")
    features: Dict[str, bool] = Field(
        default_factory=dict, description="Available features"
    )


class ConfigResponse(BaseModel):
    """Configuration response."""

    lattice_size: int = Field(..., description="Current lattice size")
    T_critical: float = Field(..., description="Critical temperature")
    device: str = Field(..., description="Compute device")
    is_training: bool = Field(..., description="Training in progress")
    training_progress: float = Field(..., description="Training progress 0-1")


# =============================================================================
# Sampling
# =============================================================================


class SampleRequestBase(BaseModel):
    """Base model for sampling requests."""

    temperature: float = Field(
        2.27, ge=0.1, le=10.0, description="Temperature (T_c ≈ 2.27)"
    )
    lattice_size: int = Field(32, ge=8, le=128, description="Lattice size")
    n_samples: int = Field(1, ge=1, le=100, description="Number of samples")


class MCMCSampleRequest(SampleRequestBase):
    """MCMC sampling request."""

    n_sweeps: int = Field(
        10, ge=1, le=100, description="Sweeps between samples"
    )
    burn_in: int = Field(100, ge=0, le=1000, description="Burn-in sweeps")


class DiffusionSampleRequest(SampleRequestBase):
    """Diffusion sampling request."""

    num_steps: int = Field(
        100, ge=10, le=500, description="Diffusion steps"
    )


class SampleResponse(BaseModel):
    """Sampling response."""

    samples: List[List[List[float]]] = Field(
        ..., description="Spin configurations (n_samples, size, size)"
    )
    energies: List[float] = Field(..., description="Energy per sample")
    magnetizations: List[float] = Field(
        ..., description="Magnetization per sample"
    )
    temperature: float = Field(..., description="Temperature used")
    lattice_size: int = Field(..., description="Lattice size used")


class RandomConfigResponse(BaseModel):
    """Random configuration response."""

    spins: List[List[float]] = Field(..., description="Spin configuration")
    energy: float = Field(..., description="Energy per spin")
    magnetization: float = Field(..., description="Magnetization")
    lattice_size: int = Field(..., description="Lattice size")


# =============================================================================
# WebSocket Messages
# =============================================================================


class WebSocketParams(BaseModel):
    """WebSocket sampling parameters."""

    temperature: float = Field(2.27, description="Temperature")
    lattice_size: int = Field(32, description="Lattice size")
    sampler: str = Field("mcmc", description="Sampler type: mcmc or diffusion")
    num_steps: int = Field(100, description="Number of steps")


class WebSocketFrame(BaseModel):
    """WebSocket frame message."""

    type: str = Field("frame", description="Message type")
    spins: List[List[float]] = Field(..., description="Current configuration")
    energy: float = Field(..., description="Current energy")
    magnetization: float = Field(..., description="Current magnetization")
    t: Optional[float] = Field(None, description="Diffusion time (if applicable)")


class WebSocketDone(BaseModel):
    """WebSocket completion message."""

    type: str = Field("done", description="Message type")


class WebSocketError(BaseModel):
    """WebSocket error message."""

    type: str = Field("error", description="Message type")
    message: str = Field(..., description="Error message")


# =============================================================================
# Analysis
# =============================================================================


class CompareRequest(BaseModel):
    """Comparison analysis request."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(32, ge=8, le=64)
    n_samples: int = Field(100, ge=10, le=500)


class CorrelationData(BaseModel):
    """Correlation function data."""

    r: List[float] = Field(..., description="Distance values")
    C_r: List[float] = Field(..., description="Correlation values")


class DistributionData(BaseModel):
    """Distribution data for histograms."""

    values: List[float] = Field(..., description="Bin centers")
    probabilities: List[float] = Field(..., description="Probabilities")
