"""Application configuration module."""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support.

    All settings can be overridden via environment variables with
    the same name (case-insensitive).
    """

    # API Settings
    api_title: str = "Diffusion Boltzmann Sampler"
    api_description: str = "Neural sampling from Boltzmann distributions"
    api_version: str = "1.0.0"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # ML Settings
    default_lattice_size: int = 32
    max_lattice_size: int = 128
    default_temperature: float = 2.27  # Critical temperature
    default_device: str = "cpu"

    # Diffusion Settings
    default_diffusion_steps: int = 100
    max_diffusion_steps: int = 500

    # MCMC Settings
    default_mcmc_sweeps: int = 10
    default_burn_in: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
