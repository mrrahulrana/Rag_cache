from __future__ import annotations

import logging

from rag_cache.config import Settings
from rag_cache.db import Database
from rag_cache.models import OllamaModels

logger = logging.getLogger(__name__)


class ChatMemory:
    def __init__(self, settings: Settings, db: Database, models: OllamaModels, ask):
        self.settings = settings
        self.db = db
        self.models = models
        self.ask = ask

    def converse(self, *, user_id: str, session_id: str | None, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Enter a question.")
        created = False
        if session_id is None:
            title = question.splitlines()[0][:80]
            session = self.db.create_chat_session(user_id, title)
            session_id = session["id"]
            created = True
        else:
            session = self.db.get_chat_session(session_id, user_id)
            if session is None:
                raise PermissionError("That chat session belongs to another user or does not exist.")
            if session["title"] == "New chat":
                self.db.rename_chat_session(session_id, question.splitlines()[0][:80])

        result = self.ask(user_id=user_id, session_id=session_id, question=question)
        self.db.add_chat_message(session_id=session_id, role="user", content=question)
        self.db.add_chat_message(
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            sources=result.get("sources") or [],
            cache_trace=result.get("cache_trace") or {},
        )
        summarized = self.maybe_summarize(session_id, user_id)
        return {
            "session_id": session_id,
            "created_session": created,
            "answer": result["answer"],
            "sources": result.get("sources") or [],
            "cache_trace": result.get("cache_trace") or {},
            "summarized": summarized,
        }

    def maybe_summarize(self, session_id: str, user_id: str) -> bool:
        session = self.db.get_chat_session(session_id, user_id)
        if session is None:
            return False
        messages = self.db.list_chat_messages(session_id)
        total = len(messages)
        if total < self.settings.chat_summary_limit:
            return False
        older_end = total - self.settings.recent_messages
        if older_end <= 0:
            return False
        pending = older_end - int(session["summarized_message_count"])
        if pending < self.settings.summary_batch:
            return False
        start = int(session["summarized_message_count"])
        fold = messages[start:older_end]
        if not fold:
            return False
        try:
            summary = self.models.summarize(session.get("summary") or "", fold)
        except Exception:
            logger.exception("Chat summary failed for session %s", session_id)
            return False
        self.db.update_chat_summary(session_id, summary, older_end)
        return True
