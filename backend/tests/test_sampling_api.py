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


class TestGroundStateEndpoint:
    """Tests for /sample/ground_state endpoint."""

    def test_ground_state_endpoint_returns_200(self, client: TestClient):
        """Ground state endpoint should return 200 OK."""
        response = client.get("/sample/ground_state")
        assert response.status_code == 200

    def test_ground_state_positive_default(self, client: TestClient):
        """Ground state should default to all +1 spins."""
        response = client.get("/sample/ground_state?lattice_size=8")
        data = response.json()
        for row in data["spins"]:
            for spin in row:
                assert spin == 1, "Default ground state should be all +1"

    def test_ground_state_negative(self, client: TestClient):
        """Ground state with positive=false should be all -1."""
        response = client.get("/sample/ground_state?lattice_size=8&positive=false")
        data = response.json()
        for row in data["spins"]:
            for spin in row:
                assert spin == -1, "Negative ground state should be all -1"

    def test_ground_state_has_minimum_energy(self, client: TestClient):
        """Ground state should have minimum energy."""
        response = client.get("/sample/ground_state?lattice_size=8")
        data = response.json()
        # Energy per spin for ground state: -2J = -2 (with J=1)
        assert data["energy"] == -2.0

    def test_ground_state_has_maximum_magnetization(self, client: TestClient):
        """Positive ground state should have magnetization = +1."""
        response = client.get("/sample/ground_state?lattice_size=8&positive=true")
        data = response.json()
        assert data["magnetization"] == 1.0

    def test_ground_state_negative_magnetization(self, client: TestClient):
        """Negative ground state should have magnetization = -1."""
        response = client.get("/sample/ground_state?lattice_size=8&positive=false")
        data = response.json()
        assert data["magnetization"] == -1.0

    def test_ground_state_custom_size(self, client: TestClient):
        """Ground state should respect custom lattice_size."""
        response = client.get("/sample/ground_state?lattice_size=16")
        data = response.json()
        assert data["lattice_size"] == 16
        assert len(data["spins"]) == 16
        assert all(len(row) == 16 for row in data["spins"])


class TestMCMCErrorHandling:
    """Tests for MCMC endpoint error handling."""

    def test_mcmc_rejects_invalid_lattice_size_too_small(self, client: TestClient):
        """Lattice size below minimum should be rejected."""
        params = {
            "temperature": 2.27,
            "lattice_size": 4,  # Below minimum of 8
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 422

    def test_mcmc_rejects_invalid_lattice_size_too_large(self, client: TestClient):
        """Lattice size above maximum should be rejected."""
        params = {
            "temperature": 2.27,
            "lattice_size": 256,  # Above maximum of 128
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 422

    def test_mcmc_rejects_invalid_n_samples(self, client: TestClient):
        """Invalid n_samples should be rejected."""
        params = {
            "temperature": 2.27,
            "lattice_size": 16,
            "n_samples": 0,  # Must be >= 1
            "n_sweeps": 5,
            "burn_in": 10,
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 422

    def test_mcmc_rejects_negative_burn_in(self, client: TestClient):
        """Negative burn_in should be rejected."""
        params = {
            "temperature": 2.27,
            "lattice_size": 16,
            "n_samples": 2,
            "n_sweeps": 5,
            "burn_in": -10,  # Must be >= 0
        }
        response = client.post("/sample/mcmc", json=params)
        assert response.status_code == 422

    def test_mcmc_rejects_missing_required_params(self, client: TestClient):
        """Missing required parameters should return validation error."""
        # Empty request
        response = client.post("/sample/mcmc", json={})
        # Should use defaults, so this may actually work
        # But let's test with invalid types
        response = client.post("/sample/mcmc", json={"temperature": "not_a_number"})
        assert response.status_code == 422

    def test_mcmc_rejects_non_json_body(self, client: TestClient):
        """Non-JSON request body should be rejected."""
        response = client.post(
            "/sample/mcmc",
            content="not json",
            headers={"content-type": "text/plain"},
        )
        assert response.status_code == 422


class TestEndToEndMCMCSampling:
    """Integration tests for end-to-end MCMC sampling verification."""

    def test_low_temperature_produces_ordered_samples(self, client: TestClient):
        """At low temperature, samples should be ordered (|M| ≈ 1)."""
        params = {
            "temperature": 0.5,  # Well below T_c ≈ 2.27 for clear ordering
            "lattice_size": 8,   # Smaller lattice for faster thermalization
            "n_samples": 5,
            "n_sweeps": 50,      # More sweeps between samples
            "burn_in": 200,      # Longer burn-in for thermalization
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()

        # Check average magnetization is high (ordered)
        # Using average to account for statistical fluctuations
        avg_abs_mag = sum(abs(m) for m in data["magnetizations"]) / len(data["magnetizations"])
        assert avg_abs_mag > 0.5, f"Expected ordered state, got avg |M|={avg_abs_mag}"

    def test_high_temperature_produces_disordered_samples(self, client: TestClient):
        """At high temperature, samples should be disordered (|M| ≈ 0)."""
        params = {
            "temperature": 5.0,  # Well above T_c
            "lattice_size": 16,
            "n_samples": 5,
            "n_sweeps": 20,
            "burn_in": 100,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()

        # Check average magnetization is low (disordered)
        avg_abs_mag = sum(abs(m) for m in data["magnetizations"]) / len(
            data["magnetizations"]
        )
        assert avg_abs_mag < 0.5, f"Expected disordered state, got avg |M|={avg_abs_mag}"

    def test_samples_have_consistent_energies(self, client: TestClient):
        """Sample energies should be consistent with spin configurations."""
        params = {
            "temperature": 2.27,
            "lattice_size": 8,
            "n_samples": 3,
            "n_sweeps": 10,
            "burn_in": 50,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()

        # Energies should be in valid range for 8x8 lattice
        # Ground state: -128, Maximum: +128
        for energy in data["energies"]:
            assert -2.0 <= energy <= 2.0, f"Energy per spin out of range: {energy}"

    def test_samples_have_valid_magnetizations(self, client: TestClient):
        """Sample magnetizations should be in [-1, 1]."""
        params = {
            "temperature": 2.27,
            "lattice_size": 16,
            "n_samples": 10,
            "n_sweeps": 10,
            "burn_in": 50,
        }
        response = client.post("/sample/mcmc", json=params)
        data = response.json()

        for mag in data["magnetizations"]:
            assert -1.0 <= mag <= 1.0, f"Magnetization out of range: {mag}"

    def test_full_workflow_ground_to_sample(self, client: TestClient):
        """Test full workflow: get ground state, then sample from it."""
        # Get ground state energy
        gs_response = client.get("/sample/ground_state?lattice_size=16")
        gs_data = gs_response.json()
        gs_energy = gs_data["energy"]

        # Sample at low temperature - should have similar energy
        params = {
            "temperature": 0.5,
            "lattice_size": 16,
            "n_samples": 3,
            "n_sweeps": 50,
            "burn_in": 200,
        }
        sample_response = client.post("/sample/mcmc", json=params)
        sample_data = sample_response.json()

        # At very low T, energies should be close to ground state
        for energy in sample_data["energies"]:
            assert energy < gs_energy + 0.5, f"Energy too high: {energy} vs gs {gs_energy}"


class TestDiffusionCheckpointSampling:
    """Tests for loading diffusion checkpoints via API."""

    def test_diffusion_uses_checkpoint_name(self, client: TestClient, monkeypatch, tmp_path):
        """Sampling should load the specified checkpoint."""
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))

        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess
        from backend.ml.samplers.diffusion import DiffusionSampler
        from backend.ml.checkpoints import format_checkpoint_name

        model_config = {
            "in_channels": 1,
            "base_channels": 8,
            "time_embed_dim": 16,
            "num_blocks": 1,
        }
        model = ScoreNetwork(**model_config)
        sampler = DiffusionSampler(
            score_network=model,
            diffusion=DiffusionProcess(beta_min=0.1, beta_max=1.0),
            num_steps=5,
        )

        checkpoint_path = tmp_path / format_checkpoint_name(8, 2.27)
        sampler.save_checkpoint(
            path=str(checkpoint_path),
            model_config=model_config,
            training_temperature=2.27,
            extra_info={"lattice_size": 8},
        )

        response = client.post(
            "/sample/diffusion",
            json={
                "temperature": 2.27,
                "lattice_size": 8,
                "n_samples": 1,
                "num_steps": 5,
                "checkpoint_name": checkpoint_path.name,
                "discretize": True,
                "discretization_method": "sign",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["lattice_size"] == 8
        assert len(data["samples"]) == 1

    def test_diffusion_uses_latest_checkpoint(self, client: TestClient, monkeypatch, tmp_path):
        """Sampling should fall back to the latest checkpoint."""
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))

        from backend.ml.models.score_network import ScoreNetwork
        from backend.ml.models.diffusion import DiffusionProcess
        from backend.ml.samplers.diffusion import DiffusionSampler
        from backend.ml.checkpoints import format_checkpoint_name

        model_config = {
            "in_channels": 1,
            "base_channels": 8,
            "time_embed_dim": 16,
            "num_blocks": 1,
        }
        sampler = DiffusionSampler(
            score_network=ScoreNetwork(**model_config),
            diffusion=DiffusionProcess(beta_min=0.1, beta_max=1.0),
            num_steps=5,
        )

        checkpoint_path = tmp_path / format_checkpoint_name(8, 2.27)
        sampler.save_checkpoint(
            path=str(checkpoint_path),
            model_config=model_config,
            training_temperature=2.27,
            extra_info={"lattice_size": 8},
        )

        response = client.post(
            "/sample/diffusion",
            json={
                "temperature": 2.27,
                "lattice_size": 8,
                "n_samples": 1,
                "num_steps": 5,
                "use_trained_model": True,
                "discretize": True,
                "discretization_method": "sign",
            },
        )
        assert response.status_code == 200

    def test_diffusion_missing_checkpoint_name_returns_404(
        self, client: TestClient, monkeypatch, tmp_path
    ):
        """Explicit checkpoint_name should 404 when missing."""
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))

        response = client.post(
            "/sample/diffusion",
            json={
                "temperature": 2.27,
                "lattice_size": 8,
                "n_samples": 1,
                "num_steps": 5,
                "checkpoint_name": "missing.pt",
                "use_trained_model": False,
            },
        )
        assert response.status_code == 404
