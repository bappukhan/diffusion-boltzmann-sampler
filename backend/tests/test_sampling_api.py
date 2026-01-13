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
