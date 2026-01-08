"""Analysis API routes for comparing samplers."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from ...ml.systems.ising import IsingModel
from ...ml.samplers.mcmc import MetropolisHastings
from ...ml.samplers.diffusion import PretrainedDiffusionSampler
from ...ml.analysis import (
    pair_correlation,
    magnetization_distribution,
    autocorrelation_time,
    energy_histogram,
    compare_distributions,
)


router = APIRouter()


class AnalysisRequest(BaseModel):
    """Request model for analysis."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(32, ge=8, le=64)
    n_samples: int = Field(100, ge=10, le=1000)


class CorrelationResponse(BaseModel):
    """Response for correlation function."""

    r: List[float]
    C_r: List[float]
    temperature: float
    lattice_size: int


class DistributionResponse(BaseModel):
    """Response for distribution data."""

    values: List[float]
    probabilities: List[float]
    name: str
    temperature: float


class ComparisonResponse(BaseModel):
    """Response for sampler comparison."""

    mcmc: Dict[str, Any]
    diffusion: Dict[str, Any]
    comparison_metrics: Dict[str, float]
    temperature: float
    lattice_size: int


@router.post("/correlation/mcmc", response_model=CorrelationResponse)
async def get_mcmc_correlation(request: AnalysisRequest) -> CorrelationResponse:
    """Compute spin-spin correlation function from MCMC samples."""
    try:
        model = IsingModel(size=request.lattice_size)
        sampler = MetropolisHastings(model, temperature=request.temperature)
        samples = sampler.sample(n_samples=request.n_samples, burn_in=500)

        corr = pair_correlation(samples)

        return CorrelationResponse(
            r=corr["r"],
            C_r=corr["C_r"],
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/magnetization/mcmc", response_model=DistributionResponse)
async def get_mcmc_magnetization(request: AnalysisRequest) -> DistributionResponse:
    """Compute magnetization distribution from MCMC samples."""
    try:
        model = IsingModel(size=request.lattice_size)
        sampler = MetropolisHastings(model, temperature=request.temperature)
        samples = sampler.sample(n_samples=request.n_samples, burn_in=500)

        dist = magnetization_distribution(samples)

        return DistributionResponse(
            values=dist["M"],
            probabilities=dist["P_M"],
            name="P(M) - MCMC",
            temperature=request.temperature,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/energy/mcmc", response_model=DistributionResponse)
async def get_mcmc_energy(request: AnalysisRequest) -> DistributionResponse:
    """Compute energy distribution from MCMC samples."""
    try:
        model = IsingModel(size=request.lattice_size)
        sampler = MetropolisHastings(model, temperature=request.temperature)
        samples = sampler.sample(n_samples=request.n_samples, burn_in=500)

        dist = energy_histogram(samples, model)

        return DistributionResponse(
            values=dist["E"],
            probabilities=dist["P_E"],
            name="P(E/N) - MCMC",
            temperature=request.temperature,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_samplers(request: AnalysisRequest) -> ComparisonResponse:
    """Compare MCMC and diffusion samplers.

    Generates samples from both methods and computes comparison metrics.
    """
    try:
        model = IsingModel(size=request.lattice_size)

        # Generate MCMC samples
        mcmc_sampler = MetropolisHastings(model, temperature=request.temperature)
        mcmc_samples = mcmc_sampler.sample(n_samples=request.n_samples, burn_in=500)

        # Generate diffusion samples
        diff_sampler = PretrainedDiffusionSampler(lattice_size=request.lattice_size)
        diff_samples = diff_sampler.sample_heuristic(
            batch_size=request.n_samples,
            temperature=request.temperature,
        )
        if len(diff_samples.shape) == 4:
            diff_samples = diff_samples.squeeze(1)

        # Compute statistics for both
        mcmc_mag = magnetization_distribution(mcmc_samples)
        mcmc_energy = energy_histogram(mcmc_samples, model)
        mcmc_corr = pair_correlation(mcmc_samples)
        mcmc_tau = autocorrelation_time(mcmc_samples)

        diff_mag = magnetization_distribution(diff_samples)
        diff_energy = energy_histogram(diff_samples, model)
        diff_corr = pair_correlation(diff_samples)

        # Comparison metrics
        comparison = compare_distributions(mcmc_samples, diff_samples, model)

        return ComparisonResponse(
            mcmc={
                "magnetization": mcmc_mag,
                "energy": mcmc_energy,
                "correlation": mcmc_corr,
                "autocorrelation_time": mcmc_tau,
                "mean_mag": comparison["samples1_mean_mag"],
                "var_mag": comparison["samples1_var_mag"],
            },
            diffusion={
                "magnetization": diff_mag,
                "energy": diff_energy,
                "correlation": diff_corr,
                "mean_mag": comparison["samples2_mean_mag"],
                "var_mag": comparison["samples2_var_mag"],
            },
            comparison_metrics=comparison,
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase_diagram")
async def get_phase_diagram_data(
    lattice_size: int = 32,
    n_temps: int = 10,
    n_samples_per_temp: int = 50,
) -> Dict[str, Any]:
    """Generate data for phase diagram visualization.

    Computes magnetization vs temperature across the phase transition.
    """
    try:
        import numpy as np

        model = IsingModel(size=lattice_size)
        T_c = IsingModel.T_CRITICAL

        # Temperature range around critical point
        temperatures = np.linspace(1.0, 4.0, n_temps).tolist()
        mean_mags = []
        std_mags = []

        for T in temperatures:
            sampler = MetropolisHastings(model, temperature=T)
            samples = sampler.sample(n_samples=n_samples_per_temp, burn_in=300)
            mags = model.magnetization(samples).abs()
            mean_mags.append(mags.mean().item())
            std_mags.append(mags.std().item())

        return {
            "temperatures": temperatures,
            "mean_magnetization": mean_mags,
            "std_magnetization": std_mags,
            "T_critical": T_c,
            "lattice_size": lattice_size,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/autocorrelation")
async def get_autocorrelation_comparison(
    temperature: float = 2.27,
    lattice_size: int = 32,
    n_steps: int = 1000,
) -> Dict[str, Any]:
    """Compare autocorrelation times between MCMC and diffusion.

    Lower autocorrelation time means more independent samples per step.
    """
    try:
        model = IsingModel(size=lattice_size)

        # MCMC autocorrelation
        sampler = MetropolisHastings(model, temperature=temperature)
        mcmc_samples = sampler.sample(n_samples=n_steps, n_sweeps=1, burn_in=500)
        mcmc_tau = autocorrelation_time(mcmc_samples)

        return {
            "mcmc_tau": mcmc_tau,
            "diffusion_tau": 1.0,  # Diffusion samples are independent by design
            "speedup_factor": mcmc_tau,  # How many times faster is diffusion
            "temperature": temperature,
            "lattice_size": lattice_size,
            "note": "τ=1 means perfectly independent samples",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
