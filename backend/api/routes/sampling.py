"""Sampling API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch

from ...ml.systems.ising import IsingModel
from ...ml.samplers.mcmc import MetropolisHastings


router = APIRouter()


class MCMCSampleRequest(BaseModel):
    """Request model for MCMC sampling."""

    temperature: float = Field(2.27, ge=0.1, le=10.0, description="Temperature")
    lattice_size: int = Field(32, ge=8, le=128, description="Lattice size")
    n_samples: int = Field(10, ge=1, le=1000, description="Number of samples")
    n_sweeps: int = Field(10, ge=1, le=100, description="Sweeps between samples")
    burn_in: int = Field(100, ge=0, le=1000, description="Burn-in sweeps")


class DiffusionSampleRequest(BaseModel):
    """Request model for diffusion sampling."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(32, ge=8, le=64)
    n_samples: int = Field(1, ge=1, le=10)
    num_steps: int = Field(100, ge=10, le=500)
    checkpoint_path: Optional[str] = Field(
        None, description="Path to trained model checkpoint"
    )
    checkpoint_name: Optional[str] = Field(
        None, description="Checkpoint file name in the checkpoint directory"
    )
    use_trained_model: bool = Field(
        False, description="Whether to use trained model if available"
    )
    discretize: bool = Field(
        True, description="Whether to discretize output to ±1 spins"
    )
    discretization_method: str = Field(
        "sign", description="Method: sign, tanh, gumbel, stochastic"
    )


class SampleResponse(BaseModel):
    """Response model for sampling."""

    samples: List[List[List[float]]]  # (n_samples, size, size)
    energies: List[float]
    magnetizations: List[float]
    temperature: float
    lattice_size: int


@router.post("/mcmc", response_model=SampleResponse)
async def sample_mcmc(request: MCMCSampleRequest) -> SampleResponse:
    """Generate samples using Metropolis-Hastings MCMC.

    This is the gold-standard baseline for Ising model sampling.
    """
    try:
        # Create model and sampler
        model = IsingModel(size=request.lattice_size)
        sampler = MetropolisHastings(model, temperature=request.temperature)

        # Generate samples
        samples = sampler.sample(
            n_samples=request.n_samples,
            n_sweeps=request.n_sweeps,
            burn_in=request.burn_in,
        )

        # Compute observables
        energies = model.energy_per_spin(samples).tolist()
        magnetizations = model.magnetization(samples).tolist()

        return SampleResponse(
            samples=samples.tolist(),
            energies=energies,
            magnetizations=magnetizations,
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffusion", response_model=SampleResponse)
async def sample_diffusion(request: DiffusionSampleRequest) -> SampleResponse:
    """Generate samples using diffusion model.

    Uses the trained score network for reverse diffusion sampling.
    Supports loading trained model checkpoints for high-quality samples.
    """
    try:
        from ...ml.samplers.diffusion import DiffusionSampler, PretrainedDiffusionSampler

        # Choose sampler based on request
        if request.checkpoint_path:
            # Load from explicit checkpoint path
            sampler = DiffusionSampler.from_checkpoint(
                checkpoint_path=request.checkpoint_path,
                num_steps=request.num_steps,
            )
            # Generate samples with trained model
            if request.discretize:
                samples = sampler.sample_ising(
                    batch_size=request.n_samples,
                    method=request.discretization_method,
                )
            else:
                samples = sampler.sample(batch_size=request.n_samples)
        elif request.use_trained_model:
            # Try to find a trained model automatically
            import os
            checkpoints_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ml", "checkpoints"
            )
            checkpoint_file = os.path.join(
                checkpoints_dir, f"ising_{request.lattice_size}.pt"
            )
            if os.path.exists(checkpoint_file):
                sampler = DiffusionSampler.from_checkpoint(
                    checkpoint_path=checkpoint_file,
                    num_steps=request.num_steps,
                )
                if request.discretize:
                    samples = sampler.sample_ising(
                        batch_size=request.n_samples,
                        method=request.discretization_method,
                    )
                else:
                    samples = sampler.sample(batch_size=request.n_samples)
            else:
                # Fall back to heuristic mode
                sampler = PretrainedDiffusionSampler(
                    lattice_size=request.lattice_size,
                    num_steps=request.num_steps,
                )
                samples = sampler.sample_heuristic(
                    batch_size=request.n_samples,
                    temperature=request.temperature,
                )
        else:
            # Use heuristic mode (demo/untrained)
            sampler = PretrainedDiffusionSampler(
                lattice_size=request.lattice_size,
                num_steps=request.num_steps,
            )
            samples = sampler.sample_heuristic(
                batch_size=request.n_samples,
                temperature=request.temperature,
            )

        # Remove channel dimension if present
        if len(samples.shape) == 4:
            samples = samples.squeeze(1)

        # Create model for observables
        model = IsingModel(size=request.lattice_size)
        energies = model.energy_per_spin(samples).tolist()
        magnetizations = model.magnetization(samples).tolist()

        return SampleResponse(
            samples=samples.tolist(),
            energies=energies,
            magnetizations=magnetizations,
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random")
async def get_random_configuration(
    lattice_size: int = 32,
) -> dict:
    """Get a random spin configuration."""
    model = IsingModel(size=lattice_size)
    spins = model.random_configuration(batch_size=1).squeeze(0)

    return {
        "spins": spins.tolist(),
        "energy": model.energy_per_spin(spins.unsqueeze(0)).item(),
        "magnetization": model.magnetization(spins.unsqueeze(0)).item(),
        "lattice_size": lattice_size,
    }


@router.get("/ground_state")
async def get_ground_state(
    lattice_size: int = 32,
    positive: bool = True,
) -> dict:
    """Get a ground state configuration (all +1 or all -1)."""
    model = IsingModel(size=lattice_size)

    if positive:
        spins = torch.ones(lattice_size, lattice_size)
    else:
        spins = -torch.ones(lattice_size, lattice_size)

    return {
        "spins": spins.tolist(),
        "energy": model.energy_per_spin(spins.unsqueeze(0)).item(),
        "magnetization": model.magnetization(spins.unsqueeze(0)).item(),
        "lattice_size": lattice_size,
    }


class CompareRequest(BaseModel):
    """Request model for comparing samplers."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(16, ge=8, le=64)
    n_samples: int = Field(50, ge=10, le=200)
    mcmc_sweeps: int = Field(20, ge=5, le=100)
    mcmc_burn_in: int = Field(200, ge=50, le=1000)
    diffusion_steps: int = Field(100, ge=50, le=500)


class CompareResponse(BaseModel):
    """Response model for sampler comparison."""

    temperature: float
    lattice_size: int
    n_samples: int
    summary: dict
    basic_statistics: dict
    kl_divergence: dict
    wasserstein: dict
    correlation: dict


@router.post("/compare", response_model=CompareResponse)
async def compare_samplers(request: CompareRequest) -> CompareResponse:
    """Compare diffusion sampler against MCMC baseline.

    Generates samples from both samplers and computes comprehensive
    comparison metrics including KL divergence, Wasserstein distance,
    and correlation function analysis.
    """
    try:
        from ...ml.samplers.diffusion import PretrainedDiffusionSampler
        from ...ml.analysis import comprehensive_comparison

        # Create model and MCMC sampler
        model = IsingModel(size=request.lattice_size)
        mcmc_sampler = MetropolisHastings(model, temperature=request.temperature)

        # Generate MCMC samples (reference)
        mcmc_samples = mcmc_sampler.sample(
            n_samples=request.n_samples,
            n_sweeps=request.mcmc_sweeps,
            burn_in=request.mcmc_burn_in,
        )

        # Generate diffusion samples
        diffusion_sampler = PretrainedDiffusionSampler(
            lattice_size=request.lattice_size,
            num_steps=request.diffusion_steps,
        )
        diffusion_samples = diffusion_sampler.sample_heuristic(
            batch_size=request.n_samples,
            temperature=request.temperature,
        )

        # Remove channel dimension if present
        if len(diffusion_samples.shape) == 4:
            diffusion_samples = diffusion_samples.squeeze(1)

        # Compute comprehensive comparison
        comparison = comprehensive_comparison(
            samples1=mcmc_samples,
            samples2=diffusion_samples,
            ising_model=model,
        )

        # Add sampler names to summary
        summary = comparison["summary"]
        summary["name_a"] = "MCMC"
        summary["name_b"] = "Diffusion"

        return CompareResponse(
            temperature=request.temperature,
            lattice_size=request.lattice_size,
            n_samples=request.n_samples,
            summary=summary,
            basic_statistics=comparison["basic_statistics"],
            kl_divergence=comparison["kl_divergence"],
            wasserstein=comparison["wasserstein"],
            correlation=comparison["correlation"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrajectoryRequest(BaseModel):
    """Request model for diffusion trajectory."""

    lattice_size: int = Field(16, ge=8, le=64)
    num_steps: int = Field(100, ge=20, le=500)
    num_frames: int = Field(20, ge=5, le=100)
    frame_spacing: str = Field(
        "log", description="Frame spacing: linear, log, cosine"
    )
    checkpoint_path: Optional[str] = Field(None)


class TrajectoryFrame(BaseModel):
    """A single frame in a diffusion trajectory."""

    step: int
    time: float
    spins: List[List[float]]
    energy: float
    magnetization: float


class TrajectoryResponse(BaseModel):
    """Response model for diffusion trajectory."""

    lattice_size: int
    num_steps: int
    num_frames: int
    frame_spacing: str
    frames: List[TrajectoryFrame]


@router.post("/trajectory", response_model=TrajectoryResponse)
async def sample_trajectory(request: TrajectoryRequest) -> TrajectoryResponse:
    """Generate a diffusion sampling trajectory with intermediate states.

    Returns a sequence of frames showing the reverse diffusion process
    from noise to a sample, useful for visualization and animation.
    """
    try:
        from ...ml.samplers.diffusion import DiffusionSampler, PretrainedDiffusionSampler

        # Create model for observables
        model = IsingModel(size=request.lattice_size)

        frames = []

        if request.checkpoint_path:
            # Use trained model with metadata trajectory
            sampler = DiffusionSampler.from_checkpoint(
                checkpoint_path=request.checkpoint_path,
                num_steps=request.num_steps,
            )
            for step, time, state in sampler.sample_trajectory_with_metadata(
                batch_size=1,
                num_frames=request.num_frames,
                frame_spacing=request.frame_spacing,
            ):
                # Remove batch and channel dimensions
                spins = state.squeeze(0).squeeze(0)
                frames.append(TrajectoryFrame(
                    step=step,
                    time=time,
                    spins=spins.tolist(),
                    energy=model.energy_per_spin(spins.unsqueeze(0)).item(),
                    magnetization=model.magnetization(spins.unsqueeze(0)).item(),
                ))
        else:
            # Use heuristic mode with basic trajectory
            sampler = PretrainedDiffusionSampler(
                lattice_size=request.lattice_size,
                num_steps=request.num_steps,
            )
            shape = (1, 1, request.lattice_size, request.lattice_size)
            yield_every = max(1, request.num_steps // request.num_frames)
            trajectory = list(sampler.sample_with_trajectory(
                shape=shape,
                yield_every=yield_every,
            ))

            for i, (state, time_val) in enumerate(trajectory):
                step = i * yield_every
                # Remove batch and channel dimensions
                spins = state.squeeze(0).squeeze(0)
                frames.append(TrajectoryFrame(
                    step=step,
                    time=time_val,
                    spins=spins.tolist(),
                    energy=model.energy_per_spin(spins.unsqueeze(0)).item(),
                    magnetization=model.magnetization(spins.unsqueeze(0)).item(),
                ))

        return TrajectoryResponse(
            lattice_size=request.lattice_size,
            num_steps=request.num_steps,
            num_frames=len(frames),
            frame_spacing=request.frame_spacing,
            frames=frames,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StatisticsRequest(BaseModel):
    """Request model for computing sample statistics."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(16, ge=8, le=64)
    n_samples: int = Field(100, ge=20, le=500)
    sampler: str = Field("mcmc", description="Sampler type: mcmc or diffusion")
    mcmc_sweeps: int = Field(20, ge=5, le=100)
    mcmc_burn_in: int = Field(200, ge=50, le=1000)
    diffusion_steps: int = Field(100, ge=50, le=500)


class StatisticsResponse(BaseModel):
    """Response model for sample statistics."""

    temperature: float
    lattice_size: int
    n_samples: int
    sampler: str
    energy: dict  # mean, std, min, max
    magnetization: dict  # mean, std, min, max, abs_mean
    specific_heat: float
    susceptibility: float
    binder_cumulant: float


@router.post("/statistics", response_model=StatisticsResponse)
async def compute_statistics(request: StatisticsRequest) -> StatisticsResponse:
    """Compute comprehensive statistics for samples from a given sampler.

    Returns thermodynamic observables including energy, magnetization,
    specific heat, susceptibility, and Binder cumulant.
    """
    try:
        import numpy as np

        # Create model
        model = IsingModel(size=request.lattice_size)

        # Generate samples based on sampler type
        if request.sampler == "mcmc":
            mcmc_sampler = MetropolisHastings(model, temperature=request.temperature)
            samples = mcmc_sampler.sample(
                n_samples=request.n_samples,
                n_sweeps=request.mcmc_sweeps,
                burn_in=request.mcmc_burn_in,
            )
        elif request.sampler == "diffusion":
            from ...ml.samplers.diffusion import PretrainedDiffusionSampler

            sampler = PretrainedDiffusionSampler(
                lattice_size=request.lattice_size,
                num_steps=request.diffusion_steps,
            )
            samples = sampler.sample_heuristic(
                batch_size=request.n_samples,
                temperature=request.temperature,
            )
            if len(samples.shape) == 4:
                samples = samples.squeeze(1)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sampler: {request.sampler}. Use 'mcmc' or 'diffusion'."
            )

        # Compute observables
        energies = model.energy_per_spin(samples).numpy()
        magnetizations = model.magnetization(samples).numpy()

        # Energy statistics
        energy_stats = {
            "mean": float(np.mean(energies)),
            "std": float(np.std(energies)),
            "min": float(np.min(energies)),
            "max": float(np.max(energies)),
        }

        # Magnetization statistics
        mag_stats = {
            "mean": float(np.mean(magnetizations)),
            "std": float(np.std(magnetizations)),
            "min": float(np.min(magnetizations)),
            "max": float(np.max(magnetizations)),
            "abs_mean": float(np.mean(np.abs(magnetizations))),
        }

        # Thermodynamic quantities
        n_spins = request.lattice_size ** 2
        beta = 1.0 / request.temperature

        # Specific heat: C = β² (<E²> - <E>²) / N
        specific_heat = float(
            beta ** 2 * (np.mean(energies ** 2) - np.mean(energies) ** 2) * n_spins
        )

        # Susceptibility: χ = β N (<M²> - <|M|>²)
        susceptibility = float(
            beta * n_spins * (np.mean(magnetizations ** 2) - np.mean(np.abs(magnetizations)) ** 2)
        )

        # Binder cumulant: U = 1 - <M⁴> / (3<M²>²)
        m2_mean = np.mean(magnetizations ** 2)
        m4_mean = np.mean(magnetizations ** 4)
        binder_cumulant = float(1.0 - m4_mean / (3.0 * m2_mean ** 2 + 1e-10))

        return StatisticsResponse(
            temperature=request.temperature,
            lattice_size=request.lattice_size,
            n_samples=request.n_samples,
            sampler=request.sampler,
            energy=energy_stats,
            magnetization=mag_stats,
            specific_heat=specific_heat,
            susceptibility=susceptibility,
            binder_cumulant=binder_cumulant,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
