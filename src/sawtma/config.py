"""Application configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Configuration values shared by the application components."""

    app_name: str = "SawtMA"
    environment: str = "development"


settings = Settings()

