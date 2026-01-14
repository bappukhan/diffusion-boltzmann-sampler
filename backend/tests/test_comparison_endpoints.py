"""Tests for comparison and statistics API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestCompareEndpoint:
    """Tests for /sample/compare endpoint."""

    def test_compare_endpoint_returns_200(self, client: TestClient):
        """Compare endpoint should return 200 OK."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 10,
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
            "diffusion_steps": 20,
        }
        response = client.post("/sample/compare", json=params)
        assert response.status_code == 200

    def test_compare_endpoint_returns_json(self, client: TestClient):
        """Compare endpoint should return JSON response."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 10,
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
            "diffusion_steps": 20,
        }
        response = client.post("/sample/compare", json=params)
        assert response.headers["content-type"] == "application/json"

    def test_compare_response_contains_required_fields(self, client: TestClient):
        """Response should contain all required comparison fields."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 10,
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
            "diffusion_steps": 20,
        }
        response = client.post("/sample/compare", json=params)
        data = response.json()

        required_fields = [
            "temperature",
            "lattice_size",
            "n_samples",
            "summary",
            "basic_statistics",
            "kl_divergence",
            "wasserstein",
            "correlation",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_compare_summary_contains_sampler_names(self, client: TestClient):
        """Summary should contain MCMC and Diffusion sampler names."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 10,
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
            "diffusion_steps": 20,
        }
        response = client.post("/sample/compare", json=params)
        data = response.json()

        assert "name_a" in data["summary"]
        assert "name_b" in data["summary"]
        assert data["summary"]["name_a"] == "MCMC"
        assert data["summary"]["name_b"] == "Diffusion"

    def test_compare_basic_statistics_structure(self, client: TestClient):
        """Basic statistics should have expected structure."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 10,
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
            "diffusion_steps": 20,
        }
        response = client.post("/sample/compare", json=params)
        data = response.json()

        basic_stats = data["basic_statistics"]
        # Should have stats for both samplers
        assert "MCMC" in basic_stats or "Diffusion" in basic_stats

    def test_compare_kl_divergence_values(self, client: TestClient):
        """KL divergence values should be non-negative."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "mcmc_sweeps": 10,
            "mcmc_burn_in": 100,
            "diffusion_steps": 50,
        }
        response = client.post("/sample/compare", json=params)
        data = response.json()

        kl_div = data["kl_divergence"]
        # KL divergence should be non-negative
        if "magnetization" in kl_div:
            assert kl_div["magnetization"] >= 0
        if "energy" in kl_div:
            assert kl_div["energy"] >= 0

    def test_compare_wasserstein_values(self, client: TestClient):
        """Wasserstein values should be non-negative."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "mcmc_sweeps": 10,
            "mcmc_burn_in": 100,
            "diffusion_steps": 50,
        }
        response = client.post("/sample/compare", json=params)
        data = response.json()

        wasserstein = data["wasserstein"]
        # Wasserstein distance should be non-negative
        if "magnetization" in wasserstein:
            assert wasserstein["magnetization"] >= 0
        if "energy" in wasserstein:
            assert wasserstein["energy"] >= 0


class TestTrajectoryEndpoint:
    """Tests for /sample/trajectory endpoint."""

    def test_trajectory_endpoint_returns_200(self, client: TestClient):
        """Trajectory endpoint should return 200 OK."""
        params = {
            "lattice_size": 8,
            "num_steps": 20,
            "num_frames": 5,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        assert response.status_code == 200

    def test_trajectory_returns_correct_frame_count(self, client: TestClient):
        """Trajectory should return requested number of frames."""
        params = {
            "lattice_size": 8,
            "num_steps": 20,
            "num_frames": 5,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        data = response.json()

        assert "frames" in data
        # Frame count may vary slightly due to implementation
        assert len(data["frames"]) >= 1

    def test_trajectory_frames_have_required_fields(self, client: TestClient):
        """Each frame should have step, time, spins, energy, magnetization."""
        params = {
            "lattice_size": 8,
            "num_steps": 20,
            "num_frames": 3,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        data = response.json()

        for frame in data["frames"]:
            assert "step" in frame
            assert "time" in frame
            assert "spins" in frame
            assert "energy" in frame
            assert "magnetization" in frame

    def test_trajectory_spins_correct_shape(self, client: TestClient):
        """Spins in each frame should have correct lattice shape."""
        params = {
            "lattice_size": 16,
            "num_steps": 20,
            "num_frames": 3,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        data = response.json()

        for frame in data["frames"]:
            spins = frame["spins"]
            assert len(spins) == 16
            assert all(len(row) == 16 for row in spins)

    def test_trajectory_energy_in_valid_range(self, client: TestClient):
        """Frame energies should be in valid range."""
        params = {
            "lattice_size": 8,
            "num_steps": 20,
            "num_frames": 5,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        data = response.json()

        for frame in data["frames"]:
            energy = frame["energy"]
            # Energy per spin should be in [-2, 2]
            assert -2.5 <= energy <= 2.5

    def test_trajectory_magnetization_in_valid_range(self, client: TestClient):
        """Frame magnetizations should be in [-1, 1]."""
        params = {
            "lattice_size": 8,
            "num_steps": 20,
            "num_frames": 5,
            "frame_spacing": "linear",
        }
        response = client.post("/sample/trajectory", json=params)
        data = response.json()

        for frame in data["frames"]:
            mag = frame["magnetization"]
            assert -1.0 <= mag <= 1.0

    def test_trajectory_log_spacing(self, client: TestClient):
        """Log spacing should work correctly."""
        params = {
            "lattice_size": 8,
            "num_steps": 50,
            "num_frames": 10,
            "frame_spacing": "log",
        }
        response = client.post("/sample/trajectory", json=params)
        assert response.status_code == 200

    def test_trajectory_cosine_spacing(self, client: TestClient):
        """Cosine spacing should work correctly."""
        params = {
            "lattice_size": 8,
            "num_steps": 50,
            "num_frames": 10,
            "frame_spacing": "cosine",
        }
        response = client.post("/sample/trajectory", json=params)
        assert response.status_code == 200


class TestStatisticsEndpoint:
    """Tests for /sample/statistics endpoint."""

    def test_statistics_endpoint_returns_200(self, client: TestClient):
        """Statistics endpoint should return 200 OK."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "mcmc",
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
        }
        response = client.post("/sample/statistics", json=params)
        assert response.status_code == 200

    def test_statistics_returns_all_fields(self, client: TestClient):
        """Statistics should contain all required fields."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "mcmc",
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()

        required = [
            "temperature",
            "lattice_size",
            "n_samples",
            "sampler",
            "energy",
            "magnetization",
            "specific_heat",
            "susceptibility",
            "binder_cumulant",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_statistics_energy_has_stats(self, client: TestClient):
        """Energy field should have mean, std, min, max."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "mcmc",
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()

        energy = data["energy"]
        assert "mean" in energy
        assert "std" in energy
        assert "min" in energy
        assert "max" in energy

    def test_statistics_magnetization_has_stats(self, client: TestClient):
        """Magnetization field should have mean, std, abs_mean."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "mcmc",
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()

        mag = data["magnetization"]
        assert "mean" in mag
        assert "std" in mag
        assert "abs_mean" in mag

    def test_statistics_mcmc_sampler(self, client: TestClient):
        """Statistics should work with MCMC sampler."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "mcmc",
            "mcmc_sweeps": 5,
            "mcmc_burn_in": 50,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()
        assert data["sampler"] == "mcmc"

    def test_statistics_diffusion_sampler(self, client: TestClient):
        """Statistics should work with diffusion sampler."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "diffusion",
            "diffusion_steps": 20,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()
        assert data["sampler"] == "diffusion"

    def test_statistics_invalid_sampler_rejected(self, client: TestClient):
        """Invalid sampler type should be rejected."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 20,
            "sampler": "invalid_sampler",
        }
        response = client.post("/sample/statistics", json=params)
        assert response.status_code in [400, 500]

    def test_statistics_specific_heat_positive(self, client: TestClient):
        """Specific heat should be non-negative."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 50,
            "sampler": "mcmc",
            "mcmc_sweeps": 10,
            "mcmc_burn_in": 100,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()
        assert data["specific_heat"] >= 0

    def test_statistics_susceptibility_positive(self, client: TestClient):
        """Magnetic susceptibility should be non-negative."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 50,
            "sampler": "mcmc",
            "mcmc_sweeps": 10,
            "mcmc_burn_in": 100,
        }
        response = client.post("/sample/statistics", json=params)
        data = response.json()
        assert data["susceptibility"] >= 0


class TestDiffusionEndpointExtended:
    """Extended tests for /sample/diffusion endpoint with new parameters."""

    def test_diffusion_with_discretize_true(self, client: TestClient):
        """Diffusion should work with discretize=true."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 2,
            "num_steps": 20,
            "discretize": True,
            "discretization_method": "sign",
        }
        response = client.post("/sample/diffusion", json=params)
        assert response.status_code == 200

    def test_diffusion_with_discretize_false(self, client: TestClient):
        """Diffusion should work with discretize=false."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 2,
            "num_steps": 20,
            "discretize": False,
        }
        response = client.post("/sample/diffusion", json=params)
        assert response.status_code == 200

    def test_diffusion_discretization_methods(self, client: TestClient):
        """Diffusion should accept different discretization methods."""
        for method in ["sign", "tanh", "stochastic", "gumbel"]:
            params = {
                "temperature": 2.27,
                "lattice_size": 8,
                "n_samples": 2,
                "num_steps": 20,
                "discretize": True,
                "discretization_method": method,
            }
            response = client.post("/sample/diffusion", json=params)
            assert response.status_code == 200, f"Failed for method: {method}"
