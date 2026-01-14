"""Statistical analysis functions for comparing samplers."""

import torch
import numpy as np
from typing import Dict, List


def pair_correlation(samples: torch.Tensor) -> Dict[str, List[float]]:
    """Compute spin-spin correlation function C(r) = <s_0 s_r>.

    Args:
        samples: Tensor of shape (n_samples, size, size)

    Returns:
        Dictionary with 'r' (distances) and 'C_r' (correlations)
    """
    samples_np = samples.numpy() if isinstance(samples, torch.Tensor) else samples
    n_samples, size, _ = samples_np.shape
    max_r = size // 2

    correlations = np.zeros(max_r)
    counts = np.zeros(max_r)

    # Sample a subset of reference points for efficiency
    n_ref = min(100, size * size)
    ref_indices = np.random.choice(size * size, n_ref, replace=False)

    for sample in samples_np:
        for idx in ref_indices:
            i, j = divmod(idx, size)
            ref_spin = sample[i, j]

            # Compute correlations at all distances
            for di in range(-max_r, max_r + 1):
                for dj in range(-max_r, max_r + 1):
                    r = int(np.sqrt(di**2 + dj**2))
                    if 0 < r < max_r:
                        ni, nj = (i + di) % size, (j + dj) % size
                        correlations[r] += ref_spin * sample[ni, nj]
                        counts[r] += 1

    # Normalize
    correlations = np.divide(
        correlations, counts, where=counts > 0, out=np.zeros_like(correlations)
    )

    return {"r": list(range(max_r)), "C_r": correlations.tolist()}


def magnetization_distribution(
    samples: torch.Tensor, n_bins: int = 50
) -> Dict[str, List[float]]:
    """Compute magnetization distribution P(M).

    Args:
        samples: Tensor of shape (n_samples, size, size)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with 'M' (magnetization values) and 'P_M' (probabilities)
    """
    # Compute magnetization for each sample
    if len(samples.shape) == 4:  # (batch, channel, h, w)
        mags = samples.mean(dim=(-1, -2, -3)).numpy()
    else:  # (batch, h, w)
        mags = samples.mean(dim=(-1, -2)).numpy()

    # Create histogram
    hist, bin_edges = np.histogram(mags, bins=n_bins, density=True, range=(-1, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {"M": bin_centers.tolist(), "P_M": hist.tolist()}


def autocorrelation_time(
    samples: torch.Tensor, observable: str = "magnetization"
) -> float:
    """Estimate integrated autocorrelation time.

    Args:
        samples: Time-ordered sequence of samples
        observable: Which observable to use ("magnetization" or "energy")

    Returns:
        Integrated autocorrelation time τ_int
    """
    # Compute observable time series
    if observable == "magnetization":
        if len(samples.shape) == 4:
            obs = samples.mean(dim=(-1, -2, -3)).numpy()
        else:
            obs = samples.mean(dim=(-1, -2)).numpy()
    else:
        raise ValueError(f"Unknown observable: {observable}")

    n = len(obs)
    if n < 10:
        return 1.0

    mean = obs.mean()
    var = obs.var()

    if var < 1e-10:
        return 1.0

    # Compute autocorrelation function
    obs_centered = obs - mean
    autocorr = np.correlate(obs_centered, obs_centered, mode="full")
    autocorr = autocorr[n - 1 :]  # Keep only non-negative lags
    autocorr = autocorr / (var * np.arange(n, 0, -1))  # Normalize

    # Integrate until first negative value (or use window)
    tau_int = 0.5  # Include lag 0 contribution
    for i in range(1, min(n, 100)):
        if autocorr[i] < 0:
            break
        tau_int += autocorr[i]

    return float(tau_int)


def energy_histogram(
    samples: torch.Tensor, ising_model, n_bins: int = 50
) -> Dict[str, List[float]]:
    """Compute energy histogram.

    Args:
        samples: Tensor of shape (n_samples, size, size) or (n_samples, 1, size, size)
        ising_model: IsingModel instance
        n_bins: Number of histogram bins

    Returns:
        Dictionary with 'E' (energies) and 'P_E' (probabilities)
    """
    # Handle channel dimension
    if len(samples.shape) == 4:
        samples = samples.squeeze(1)

    # Compute energies
    energies = ising_model.energy_per_spin(samples).numpy()

    # Create histogram
    hist, bin_edges = np.histogram(energies, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {"E": bin_centers.tolist(), "P_E": hist.tolist()}


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Reference distribution (e.g., MCMC samples)
        q: Approximating distribution (e.g., diffusion samples)
        epsilon: Small value for numerical stability

    Returns:
        KL divergence value (non-negative, 0 if identical)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize to ensure valid probability distributions
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)

    # Add epsilon for numerical stability
    p = p + epsilon
    q = q + epsilon

    # Re-normalize
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute symmetric KL divergence (Jensen-Shannon-like).

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value for numerical stability

    Returns:
        Symmetric KL divergence: 0.5 * (D_KL(P||Q) + D_KL(Q||P))
    """
    return 0.5 * (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon))


def magnetization_kl_divergence(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    n_bins: int = 50,
) -> Dict[str, float]:
    """Compute KL divergence between magnetization distributions.

    Args:
        samples1: Reference samples (e.g., MCMC)
        samples2: Test samples (e.g., diffusion)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with kl_divergence and symmetric_kl_divergence
    """
    dist1 = magnetization_distribution(samples1, n_bins)
    dist2 = magnetization_distribution(samples2, n_bins)

    p = np.array(dist1["P_M"])
    q = np.array(dist2["P_M"])

    return {
        "kl_divergence": kl_divergence(p, q),
        "symmetric_kl_divergence": symmetric_kl_divergence(p, q),
    }


def energy_kl_divergence(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
    n_bins: int = 50,
) -> Dict[str, float]:
    """Compute KL divergence between energy distributions.

    Args:
        samples1: Reference samples (e.g., MCMC)
        samples2: Test samples (e.g., diffusion)
        ising_model: IsingModel instance
        n_bins: Number of histogram bins

    Returns:
        Dictionary with kl_divergence and symmetric_kl_divergence
    """
    dist1 = energy_histogram(samples1, ising_model, n_bins)
    dist2 = energy_histogram(samples2, ising_model, n_bins)

    p = np.array(dist1["P_E"])
    q = np.array(dist2["P_E"])

    return {
        "kl_divergence": kl_divergence(p, q),
        "symmetric_kl_divergence": symmetric_kl_divergence(p, q),
    }


def compare_distributions(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
) -> Dict[str, float]:
    """Compare two sets of samples statistically.

    Args:
        samples1: First set of samples (e.g., MCMC)
        samples2: Second set of samples (e.g., diffusion)
        ising_model: IsingModel instance

    Returns:
        Dictionary of comparison metrics
    """
    # Magnetization statistics
    mag1 = magnetization_distribution(samples1)
    mag2 = magnetization_distribution(samples2)

    # Energy statistics
    e1 = energy_histogram(samples1, ising_model)
    e2 = energy_histogram(samples2, ising_model)

    # Compute mean and variance
    if len(samples1.shape) == 4:
        m1 = samples1.mean(dim=(-1, -2, -3))
        m2 = samples2.mean(dim=(-1, -2, -3))
    else:
        m1 = samples1.mean(dim=(-1, -2))
        m2 = samples2.mean(dim=(-1, -2))

    return {
        "mag_mean_diff": abs(m1.mean().item() - m2.mean().item()),
        "mag_var_diff": abs(m1.var().item() - m2.var().item()),
        "samples1_mean_mag": m1.mean().item(),
        "samples2_mean_mag": m2.mean().item(),
        "samples1_var_mag": m1.var().item(),
        "samples2_var_mag": m2.var().item(),
    }


if __name__ == "__main__":
    from ..systems.ising import IsingModel
    from ..samplers.mcmc import MetropolisHastings

    # Generate test samples
    model = IsingModel(size=16)
    sampler = MetropolisHastings(model, temperature=2.27)
    samples = sampler.sample(n_samples=100, n_sweeps=10, burn_in=200)

    print("Testing correlation functions...")
    corr = pair_correlation(samples)
    print(f"Distances: {corr['r'][:5]}")
    print(f"Correlations: {[f'{c:.3f}' for c in corr['C_r'][:5]]}")

    print("\nTesting magnetization distribution...")
    mag_dist = magnetization_distribution(samples)
    print(f"M range: [{min(mag_dist['M']):.2f}, {max(mag_dist['M']):.2f}]")

    print("\nTesting autocorrelation time...")
    tau = autocorrelation_time(samples)
    print(f"τ_int = {tau:.2f}")

    print("\nTesting energy histogram...")
    e_hist = energy_histogram(samples, model)
    print(f"E/N range: [{min(e_hist['E']):.2f}, {max(e_hist['E']):.2f}]")

    print("\nAll analysis tests passed!")
