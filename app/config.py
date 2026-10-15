"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 15.0

    database_path: Path = PROJECT_ROOT / "data" / "meditrack.db"

    cache_ttl_seconds: int = 300
    cache_max_entries: int = 256

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
