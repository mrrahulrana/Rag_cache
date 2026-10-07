from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").casefold()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    ollama_base_url: str
    embed_model: str
    chat_model: str
    embed_dim: int
    similarity_threshold: float
    semantic_answer_threshold: float
    top_k: int
    chat_summary_limit: int
    recent_messages: int
    summary_batch: int
    chunk_size: int
    chunk_overlap: int
    answer_ttl_seconds: int
    embedding_ttl_seconds: int
    retrieval_ttl_seconds: int
    llm_ttl_seconds: int
    admin_username: str
    admin_password: str
    admin_first_name: str
    admin_last_name: str
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str
    upload_dir: Path

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        return cls(
            database_url=_env(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/rag_cache",
            ),
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            embed_model=_env("EMBED_MODEL", "nomic-embed-text"),
            chat_model=_env("CHAT_MODEL", "llama3.2:1b"),
            embed_dim=_env_int("EMBED_DIM", 768),
            similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.6),
            semantic_answer_threshold=_env_float("SEMANTIC_ANSWER_THRESHOLD", 0.92),
            top_k=_env_int("TOP_K", 4),
            chat_summary_limit=_env_int("CHAT_SUMMARY_LIMIT", 20),
            recent_messages=_env_int("RECENT_MESSAGES", 8),
            summary_batch=4,
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            answer_ttl_seconds=_env_int("ANSWER_TTL_SECONDS", 60 * 60 * 24),
            embedding_ttl_seconds=_env_int("EMBEDDING_TTL_SECONDS", 60 * 60 * 24 * 30),
            retrieval_ttl_seconds=_env_int("RETRIEVAL_TTL_SECONDS", 60 * 60 * 24),
            llm_ttl_seconds=_env_int("LLM_TTL_SECONDS", 60 * 60 * 24),
            admin_username=_env("ADMIN_USERNAME", "admin"),
            admin_password=_env("ADMIN_PASSWORD", "change-me"),
            admin_first_name=_env("ADMIN_FIRST_NAME", "Admin"),
            admin_last_name=_env("ADMIN_LAST_NAME", "User"),
            langsmith_tracing=_env_bool("LANGSMITH_TRACING", False),
            langsmith_api_key=_env("LANGSMITH_API_KEY", ""),
            langsmith_project=_env("LANGSMITH_PROJECT", "rag-cache"),
            upload_dir=ROOT / "data" / "uploads",
        )
