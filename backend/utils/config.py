"""
backend/utils/config.py
========================
Centralised application configuration using Pydantic Settings.
All values are read from environment variables / .env file.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ─────────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "huggingface"] = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_BASE_URL: str | None = None
    HUGGINGFACE_API_KEY: str = ""
    HF_SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"

    # ── Whisper ASR ───────────────────────────────────────────
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large"] = "tiny"
    WHISPER_DEVICE: Literal["cpu", "cuda"] = "cpu"

    # ── Text Processing ───────────────────────────────────────
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── FAISS / Embeddings ────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/embeddings/faiss.index"

    # ── Application ───────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 7860
    DEBUG: bool = True

    # ── File Storage ──────────────────────────────────────────
    UPLOAD_DIR: str = "data/videos"
    AUDIO_DIR: str = "data/audio"
    TRANSCRIPT_DIR: str = "data/transcripts"
    SUMMARY_DIR: str = "data/summaries"
    OUTPUT_DIR: str = "outputs"
    MAX_FILE_SIZE_MB: int = 500

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Singleton instance used throughout the app
settings: Settings = get_settings()
