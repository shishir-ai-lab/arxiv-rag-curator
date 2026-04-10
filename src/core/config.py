"""
Central configuration for the arXiv RAG Curator.

All settings are read from environment variables or a .env file.
Never hardcode secrets or connection strings in code.

Usage:
    from arxiv_rag_curator.core.config import settings
    print(settings.database_url)
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment / .env file.

    Pydantic validates types automatically:
      - If POSTGRES_PORT is set to "not_a_number", it raises a clear error
      - If a required field is missing, it raises a clear error
    """

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="ragdb")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")

    # ── OpenSearch ──────────────────────────────────────────────────────────
    opensearch_host: str = Field(default="localhost")
    opensearch_port: int = Field(default=9200)

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_host: str = Field(default="localhost")
    ollama_port: int = Field(default=11434)
    ollama_model: str = Field(default="llama3.2:3b")

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # ── Computed properties ─────────────────────────────────────────────────
    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible PostgreSQL URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        """Async PostgreSQL URL (for asyncpg / SQLAlchemy async)."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")

    @property
    def opensearch_url(self) -> str:
        return f"http://{self.opensearch_host}:{self.opensearch_port}"

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"   # don't raise if .env has unknown vars


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached settings instance.

    Using lru_cache means Settings() is only constructed once per process,
    avoiding repeated file reads and environment lookups.
    """
    return Settings()


# Module-level singleton — import this directly in most places
settings = get_settings()