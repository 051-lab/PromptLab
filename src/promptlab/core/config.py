"""Configuration management for PromptLab."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = "PromptLab"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # API settings
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    # Database settings
    database_url: str = "sqlite+aiosqlite:///./promptlab.db"
    database_echo: bool = False

    # LiteLLM settings
    litellm_api_key: Optional[str] = None
    litellm_model: str = "gpt-3.5-turbo"
    litellm_temperature: float = 0.7
    litellm_max_tokens: int = 2048
    litellm_timeout: int = 30

    # API Keys for different LLM providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # Security
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 30

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
