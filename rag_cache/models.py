from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_cache.config import Settings
from rag_cache.prompts import ANSWER_SYSTEM, REWRITE_SYSTEM, SUMMARY_SYSTEM


class OllamaModels:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )
        self.chat = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=400,
        )

    def embed_query(self, text: str) -> list[float]:
        vector = self.embeddings.embed_query(f"search_query: {text}")
        self._check_dim(vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.embeddings.embed_documents([f"search_document: {text}" for text in texts])
        for vector in vectors:
            self._check_dim(vector)
        return vectors

    def rewrite_question(self, question: str, summary: str, recent_messages: list[dict]) -> str:
        if not summary and not recent_messages:
            return question.strip()
        transcript = _format_messages(recent_messages)
        response = self.chat.invoke(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(
                    content=(
                        f"Summary:\n{summary or 'None'}\n\n"
                        f"Recent messages:\n{transcript or 'None'}\n\n"
                        f"Latest question:\n{question.strip()}"
                    )
                ),
            ]
        )
        rewritten = _message_text(response).splitlines()[0].strip() if _message_text(response) else ""
        if not rewritten or len(rewritten) > 500:
            return question.strip()
        return rewritten

    def answer(self, question: str, context: str) -> str:
        response = self.chat.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=f"Question:\n{question}\n\nDocument excerpts:\n{context}"),
            ]
        )
        return _message_text(response)

    def summarize(self, existing_summary: str, messages: list[dict]) -> str:
        response = self.chat.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(
                    content=(
                        f"Existing summary:\n{existing_summary or 'None'}\n\n"
                        f"Messages to fold in:\n{_format_messages(messages)}"
                    )
                ),
            ]
        )
        summary = _message_text(response).strip()
        if not summary:
            raise RuntimeError("The chat model returned an empty summary.")
        return summary

    def _check_dim(self, vector: list[float]) -> None:
        if len(vector) != self.settings.embed_dim:
            raise RuntimeError(
                f"{self.settings.embed_model} returned {len(vector)} dimensions; "
                f"EMBED_DIM is {self.settings.embed_dim}."
            )


def ollama_status(settings: Settings) -> tuple[bool, str | None]:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Ollama is not reachable at {settings.ollama_base_url}: {exc}"
    names = [str(item.get("name", "")) for item in payload.get("models", [])]
    missing = [
        model
        for model in (settings.embed_model, settings.chat_model)
        if not _model_present(names, model)
    ]
    if missing:
        joined = ", ".join(missing)
        return False, f"Ollama is missing {joined}. Pull the model and restart the app."
    return True, None


def _model_present(names: list[str], wanted: str) -> bool:
    for name in names:
        if name == wanted or name.startswith(f"{wanted}:") or name.split(":")[0] == wanted:
            return True
    return False


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
