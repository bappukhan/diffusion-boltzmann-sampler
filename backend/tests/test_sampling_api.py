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
