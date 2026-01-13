"""Tests for Metropolis-Hastings MCMC sampler."""

import pytest
import torch
from backend.ml.samplers.mcmc import MetropolisHastings
from backend.ml.systems.ising import IsingModel


class TestMetropolisHastingsInit:
    """Tests for MetropolisHastings initialization."""

    def test_init_stores_model(self, ising_model):
        """Sampler should store the Ising model reference."""
        sampler = MetropolisHastings(ising_model, temperature=2.0)
        assert sampler.model is ising_model

    def test_init_stores_temperature(self, ising_model):
        """Sampler should store the temperature."""
        sampler = MetropolisHastings(ising_model, temperature=2.5)
        assert sampler.temperature == 2.5

    def test_init_computes_beta(self, ising_model):
        """Sampler should compute inverse temperature beta = 1/T."""
        sampler = MetropolisHastings(ising_model, temperature=2.0)
        assert abs(sampler.beta - 0.5) < 1e-10

    def test_init_high_temperature_low_beta(self, ising_model):
        """High temperature should give low beta."""
        sampler = MetropolisHastings(ising_model, temperature=10.0)
        assert sampler.beta == 0.1

    def test_init_low_temperature_high_beta(self, ising_model):
        """Low temperature should give high beta."""
        sampler = MetropolisHastings(ising_model, temperature=0.5)
        assert sampler.beta == 2.0

    def test_init_zero_temperature_infinite_beta(self, ising_model):
        """Zero temperature should give infinite beta."""
        sampler = MetropolisHastings(ising_model, temperature=0.0)
        assert sampler.beta == float("inf")
