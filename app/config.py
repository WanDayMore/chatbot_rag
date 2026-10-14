from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Gemini Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"



    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Cloudflare Turnstile
    TURNSTILE_ENABLED: bool = False
    TURNSTILE_SECRET_KEY: str = ""

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 3
    RATE_LIMIT_PER_HOUR: int = 10
    RATE_LIMIT_PER_DAY: int = 20
    DAILY_GLOBAL_HARD_CAP: int = 500

    # Semantic Cache
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL_DAYS: int = 30

    # Storage Paths
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CACHE_DB_PATH: str = "./data/cache.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
