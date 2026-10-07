from __future__ import annotations

import logging
import os

from rag_cache.auth import ensure_admin
from rag_cache.cache import CacheStore, connect_cache
from rag_cache.config import Settings
from rag_cache.db import Database
from rag_cache.graph import build_ask_graph
from rag_cache.ingest import DocumentIngestor
from rag_cache.memory import ChatMemory
from rag_cache.models import OllamaModels, ollama_status

logger = logging.getLogger(__name__)

_CONTEXT: AppContext | None = None


class AppContext:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        cache: CacheStore,
        models: OllamaModels,
        ingestor: DocumentIngestor,
        memory: ChatMemory,
        redis_ok: bool,
        redis_error: str | None,
        ollama_ok: bool,
        ollama_error: str | None,
    ):
        self.settings = settings
        self.db = db
        self.cache = cache
        self.models = models
        self.ingestor = ingestor
        self.memory = memory
        self.redis_ok = redis_ok
        self.redis_error = redis_error
        self.ollama_ok = ollama_ok
        self.ollama_error = ollama_error

    @classmethod
    def create(cls) -> AppContext:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        settings = Settings.load()
        _configure_langsmith(settings)
        db = Database(settings.database_url, settings.embed_dim)
        db.init()
        ensure_admin(db, settings)
        cache, redis_ok, redis_error = connect_cache(settings.redis_url)
        models = OllamaModels(settings)
        ollama_ok, ollama_error = ollama_status(settings)
        ingestor = DocumentIngestor(settings, db, cache, models)
        graph = build_ask_graph(settings, db, cache, models)

        def ask(*, user_id: str, session_id: str, question: str) -> dict:
            if not ollama_ok:
                raise RuntimeError(ollama_error or "Ollama is not available.")
            state = graph.invoke(
                {"question": question, "user_id": user_id, "session_id": session_id},
                config={
                    "run_name": "rag_ask",
                    "tags": ["rag-cache"],
                    "metadata": {"session_id": session_id, "user_id": user_id},
                },
            )
            return {
                "answer": state.get("answer") or "",
                "sources": state.get("sources") or [],
                "cache_trace": state.get("cache_trace") or {},
                "standalone_question": state.get("standalone_question") or question,
            }

        memory = ChatMemory(settings, db, models, ask)
        logger.info(
            "RAG Cache ready (redis=%s, ollama=%s)",
            "up" if redis_ok else "down",
            "up" if ollama_ok else "down",
        )
        return cls(
            settings=settings,
            db=db,
            cache=cache,
            models=models,
            ingestor=ingestor,
            memory=memory,
            redis_ok=redis_ok,
            redis_error=redis_error,
            ollama_ok=ollama_ok,
            ollama_error=ollama_error,
        )


def get_context() -> AppContext:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = AppContext.create()
    return _CONTEXT


def _configure_langsmith(settings: Settings) -> None:
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        return
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
