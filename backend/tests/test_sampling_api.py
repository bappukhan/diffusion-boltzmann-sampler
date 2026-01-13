"""Tests for sampling API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestMCMCEndpointBasic:
    """Basic tests for /sample/mcmc endpoint."""

    def test_mcmc_endpoint_returns_200(self, client: TestClient, fast_mcmc_params):
        """MCMC endpoint should return 200 OK for valid request."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        assert response.status_code == 200

    def test_mcmc_endpoint_returns_json(self, client: TestClient, fast_mcmc_params):
        """MCMC endpoint should return JSON response."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        assert response.headers["content-type"] == "application/json"

    def test_mcmc_endpoint_accepts_post(self, client: TestClient, fast_mcmc_params):
        """MCMC endpoint should only accept POST requests."""
        # POST should work
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        assert response.status_code == 200

        # GET should not be allowed
        response = client.get("/sample/mcmc")
        assert response.status_code == 405  # Method Not Allowed


class TestMCMCEndpointTemperature:
    """Tests for temperature parameter handling."""

    def test_mcmc_uses_requested_temperature(self, client: TestClient):
        """Response should include the requested temperature."""
        params = {
            "temperature": 3.5,
            "lattice_size": 8,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()
        assert data["temperature"] == 3.5

    def test_mcmc_low_temperature_valid(self, client: TestClient):
        """Low temperature (0.1) should be accepted."""
        params = {
            "temperature": 0.1,
            "lattice_size": 8,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 200

    def test_mcmc_high_temperature_valid(self, client: TestClient):
        """High temperature (10.0) should be accepted."""
        params = {
            "temperature": 10.0,
            "lattice_size": 8,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 200

    def test_mcmc_invalid_temperature_rejected(self, client: TestClient):
        """Temperature outside valid range should be rejected."""
        # Too low
        params = {
            "temperature": 0.05,  # Below minimum of 0.1
            "lattice_size": 8,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 422  # Validation error


class TestMCMCResponseFormat:
    """Tests for response format validation."""

    def test_response_contains_samples(self, client: TestClient, fast_mcmc_params):
        """Response should contain samples array."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        data = response.json()
        assert "samples" in data
        assert isinstance(data["samples"], list)

    def test_response_samples_correct_count(self, client: TestClient):
        """Response should contain requested number of samples."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 3,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()
        assert len(data["samples"]) == 3

    def test_response_samples_correct_shape(self, client: TestClient):
        """Each sample should have correct lattice shape."""
        params = {
            "temperature": 2.27,
            "lattice_size": 16,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()

        for sample in data["samples"]:
            assert len(sample) == 16  # 16 rows
            assert all(len(row) == 16 for row in sample)  # 16 columns

    def test_response_contains_energies(self, client: TestClient, fast_mcmc_params):
        """Response should contain energies array."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        data = response.json()
        assert "energies" in data
        assert isinstance(data["energies"], list)
        assert len(data["energies"]) == fast_mcmc_params["n_samples"]

    def test_response_contains_magnetizations(self, client: TestClient, fast_mcmc_params):
        """Response should contain magnetizations array."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        data = response.json()
        assert "magnetizations" in data
        assert isinstance(data["magnetizations"], list)
        assert len(data["magnetizations"]) == fast_mcmc_params["n_samples"]

    def test_response_contains_lattice_size(self, client: TestClient, fast_mcmc_params):
        """Response should include lattice_size."""
        response = client.post("/sample/mcmc", json=fast_mcmc_params)
        data = response.json()
        assert "lattice_size" in data
        assert data["lattice_size"] == fast_mcmc_params["lattice_size"]


class TestRandomEndpoint:
    """Tests for /sample/random endpoint."""

    def test_random_endpoint_returns_200(self, client: TestClient):
        """Random endpoint should return 200 OK."""
        response = client.get("/sample/random")
        assert response.status_code == 200

    def test_random_endpoint_returns_spins(self, client: TestClient):
        """Random endpoint should return spins array."""
        response = client.get("/sample/random")
        data = response.json()
        assert "spins" in data
        assert isinstance(data["spins"], list)

    def test_random_endpoint_default_size(self, client: TestClient):
        """Random endpoint should use default lattice_size=32."""
        response = client.get("/sample/random")
        data = response.json()
        assert data["lattice_size"] == 32
        assert len(data["spins"]) == 32
        assert all(len(row) == 32 for row in data["spins"])

    def test_random_endpoint_custom_size(self, client: TestClient):
        """Random endpoint should accept custom lattice_size."""
        response = client.get("/sample/random?lattice_size=16")
        data = response.json()
        assert data["lattice_size"] == 16
        assert len(data["spins"]) == 16

    def test_random_endpoint_includes_observables(self, client: TestClient):
        """Random endpoint should include energy and magnetization."""
        response = client.get("/sample/random")
        data = response.json()
        assert "energy" in data
        assert "magnetization" in data
        assert isinstance(data["energy"], (int, float))
        assert isinstance(data["magnetization"], (int, float))

    def test_random_spins_are_plus_minus_one(self, client: TestClient):
        """Random spins should only contain ±1 values."""
        response = client.get("/sample/random?lattice_size=8")
        data = response.json()
        for row in data["spins"]:
            for spin in row:
                assert spin in [-1, 1], f"Invalid spin value: {spin}"
