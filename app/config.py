"""Environment configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file early to ensure environment variables are set
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    anthropic_api_key: str
    vault_path: Path
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: str = "app://obsidian.md"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed origins as a list."""
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
