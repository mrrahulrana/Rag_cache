from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from rag_cache.cache import CacheStore
from rag_cache.config import Settings
from rag_cache.db import Database
from rag_cache.models import OllamaModels
from rag_cache.prompts import NO_INFO_ANSWER, is_no_info
from rag_cache.sources import build_context, to_source
from rag_cache.textutil import content_hash, sha256_text

logger = logging.getLogger(__name__)


class AskState(TypedDict, total=False):
    question: str
    user_id: str
    session_id: str
    standalone_question: str
    corpus_version: int
    query_embedding: list[float]
    chunks: list[dict]
    answer: str
    sources: list[dict]
    cache_trace: dict


def build_ask_graph(settings: Settings, db: Database, cache: CacheStore, models: OllamaModels):
    def prepare(state: AskState) -> dict:
        question = " ".join(state["question"].split())
        session = None
        if state.get("session_id") and state.get("user_id"):
            session = db.get_chat_session(state["session_id"], state["user_id"])
        messages = db.list_chat_messages(state["session_id"]) if session else []
        summary = (session or {}).get("summary") or ""
        start = int((session or {}).get("summarized_message_count") or 0)
        window = messages[start:]
        if len(window) > settings.recent_messages:
            window = window[-settings.recent_messages :]
        if summary or window:
            standalone = models.rewrite_question(question, summary, window)
        else:
            standalone = question
        return {
            "question": question,
            "standalone_question": standalone,
            "corpus_version": db.corpus_version(),
            "cache_trace": {},
            "sources": [],
            "chunks": [],
        }

    def exact_cache(state: AskState) -> dict:
        hit = cache.get_json(_answer_key(state, settings))
        if hit:
            trace = _trace(
                state,
                exact_answer="hit",
                query_embedding="skipped",
                semantic_answer="skipped",
                retrieval="skipped",
                llm="skipped",
            )
            return {"answer": hit["answer"], "sources": hit.get("sources") or [], "cache_trace": trace}
        return {"cache_trace": _trace(state, exact_answer="miss")}

    def embed(state: AskState) -> dict:
        question = state["standalone_question"]
        digest = content_hash(question)
        cached = cache.get_embedding("query", settings.embed_model, digest)
        if cached and len(cached) == settings.embed_dim:
            return {
                "query_embedding": cached,
                "cache_trace": _trace(state, query_embedding="hit"),
            }
        vector = models.embed_query(question)
        cache.set_embedding(
            "query",
            settings.embed_model,
            digest,
            vector,
            settings.embedding_ttl_seconds,
        )
        return {
            "query_embedding": vector,
            "cache_trace": _trace(state, query_embedding="miss"),
        }

    def semantic_cache(state: AskState) -> dict:
        hit = db.search_semantic_answer(
            state["query_embedding"],
            int(state["corpus_version"]),
            settings.semantic_answer_threshold,
        )
        if not hit:
            return {"cache_trace": _trace(state, semantic_answer="miss")}
        sources = hit.get("sources") or []
        _store_exact(cache, settings, state, hit["answer"], sources)
        trace = _trace(
            state,
            semantic_answer="hit",
            retrieval="skipped",
            llm="skipped",
        )
        return {"answer": hit["answer"], "sources": sources, "cache_trace": trace}

    def retrieve(state: AskState) -> dict:
        key = _retrieval_key(state, settings)
        cached = cache.get_json(key)
        if cached is not None:
            chunks = cached
            trace = _trace(state, retrieval="hit")
        else:
            chunks = db.search_chunks(
                state["query_embedding"],
                settings.similarity_threshold,
                settings.top_k,
            )
            cache.set_json(key, _cacheable_chunks(chunks), settings.retrieval_ttl_seconds)
            trace = _trace(state, retrieval="miss")
        if not chunks:
            _store_exact(cache, settings, state, NO_INFO_ANSWER, [])
            trace["llm"] = "skipped"
            return {
                "chunks": [],
                "answer": NO_INFO_ANSWER,
                "sources": [],
                "cache_trace": trace,
            }
        return {"chunks": chunks, "cache_trace": trace}

    def generate(state: AskState) -> dict:
        chunks = state.get("chunks") or []
        key = _llm_key(state, settings, chunks)
        cached = cache.get_json(key)
        if cached:
            answer = cached["answer"]
            trace = _trace(state, llm="hit")
        else:
            answer = models.answer(state["standalone_question"], build_context(chunks))
            if not answer:
                answer = NO_INFO_ANSWER
            cache.set_json(key, {"answer": answer}, settings.llm_ttl_seconds)
            trace = _trace(state, llm="miss")
        if is_no_info(answer):
            answer = NO_INFO_ANSWER
            sources: list[dict] = []
        else:
            sources = [to_source(chunk) for chunk in chunks]
            db.save_semantic_answer(
                corpus_version=int(state["corpus_version"]),
                query_text=state["standalone_question"],
                embedding=state["query_embedding"],
                answer=answer,
                sources=sources,
            )
        _store_exact(cache, settings, state, answer, sources)
        logger.info("ask cache trace: %s", trace)
        return {"answer": answer, "sources": sources, "cache_trace": trace}

    graph = StateGraph(AskState)
    graph.add_node("prepare", prepare)
    graph.add_node("exact_cache", exact_cache)
    graph.add_node("embed", embed)
    graph.add_node("semantic_cache", semantic_cache)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "exact_cache")
    graph.add_conditional_edges(
        "exact_cache",
        lambda state: "done" if state.get("answer") else "embed",
        {"done": END, "embed": "embed"},
    )
    graph.add_edge("embed", "semantic_cache")
    graph.add_conditional_edges(
        "semantic_cache",
        lambda state: "done" if state.get("answer") else "retrieve",
        {"done": END, "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "done" if state.get("answer") else "generate",
        {"done": END, "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph.compile()


def _trace(state: AskState, **flags: str) -> dict:
    trace = dict(state.get("cache_trace") or {})
    trace.update(flags)
    return trace


def _normalized(question: str) -> str:
    return " ".join(question.split()).casefold()


def _answer_key(state: AskState, settings: Settings) -> str:
    digest = sha256_text(_normalized(state["standalone_question"]))
    return f"ans:{state['corpus_version']}:{settings.chat_model}:{digest}"


def _retrieval_key(state: AskState, settings: Settings) -> str:
    raw = f"{settings.top_k}|{settings.similarity_threshold}|{_normalized(state['standalone_question'])}"
    digest = sha256_text(raw)
    return f"ret:{state['corpus_version']}:{settings.embed_model}:{digest}"


def _llm_key(state: AskState, settings: Settings, chunks: list[dict]) -> str:
    identity = "|".join(chunk.get("content_hash", "") for chunk in chunks)
    digest = sha256_text(f"{_normalized(state['standalone_question'])}|{identity}")
    return f"llm:{state['corpus_version']}:{settings.chat_model}:{digest}"


def _store_exact(cache: CacheStore, settings: Settings, state: AskState, answer: str, sources: list[dict]) -> None:
    cache.set_json(
        _answer_key(state, settings),
        {"answer": answer, "sources": sources},
        settings.answer_ttl_seconds,
    )


def _cacheable_chunks(chunks: list[dict]) -> list[dict[str, Any]]:
    stored = []
    for chunk in chunks:
        stored.append(
            {
                "id": str(chunk.get("id", "")),
                "document_name": chunk.get("document_name"),
                "document_version": chunk.get("document_version"),
                "section": chunk.get("section"),
                "page_number": chunk.get("page_number"),
                "content": chunk.get("content"),
                "content_hash": chunk.get("content_hash"),
                "metadata": chunk.get("metadata") or {},
                "similarity": chunk.get("similarity"),
            }
        )
    return stored
