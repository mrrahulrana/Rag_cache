"""Ingest a sample policy, ask a grounded question, then confirm the answer cache and an incremental update."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_cache.context import get_context
from rag_cache.prompts import NO_INFO_ANSWER

POLICY = """OVERVIEW
The north warehouse policy covers blue crates stored in aisle N4.

SHIPPING
Shipping leaves the dock at 06:30 and requires form NX-42.

RETURNS
Returns are accepted within 14 days with receipt number R-9.
"""

UPDATED = POLICY.replace("within 14 days", "within 30 days")
FILENAME = "Warehouse Policy.txt"


def main() -> None:
    ctx = get_context()
    admin = ctx.db.get_user_by_username(ctx.settings.admin_username)
    if admin is None or admin["role"] != "admin":
        raise SystemExit("Admin user is missing. Set ADMIN_PASSWORD in .env and restart this script.")

    first = ctx.ingestor.ingest_bytes(FILENAME, POLICY.encode("utf-8"), str(admin["id"]))
    print("FIRST", first.status, first.version, "embedded", first.embedded_count, "reused", first.reused_count)
    second = ctx.ingestor.ingest_bytes(FILENAME, POLICY.encode("utf-8"), str(admin["id"]))
    print("SECOND", second.status, second.version)
    if second.status != "unchanged":
        raise SystemExit("Re-uploading the same file should not create a version.")

    user = ctx.db.get_user_by_username("smoke-user")
    if user is None:
        from rag_cache.auth import register_user

        user = register_user(
            ctx.db,
            first_name="Smoke",
            last_name="User",
            username="smoke-user",
            password="smoke-pass-123",
            confirm_password="smoke-pass-123",
        )
    else:
        user = {
            "id": str(user["id"]),
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "username": user["username"],
            "role": user["role"],
        }

    question = "What form is required for shipping?"
    first_answer = ctx.memory.converse(user_id=user["id"], session_id=None, question=question)
    print("ANSWER", first_answer["answer"])
    print("TRACE", first_answer["cache_trace"])
    print("SOURCES", first_answer["sources"])
    if NO_INFO_ANSWER in first_answer["answer"]:
        rows = ctx.db.search_chunks(ctx.models.embed_query(question), 0.0, 4)
        print("TOP SCORES", [(row["similarity"], row["section"], row["content"][:80]) for row in rows])
        raise SystemExit("Expected a grounded answer for the shipping form.")
    if not first_answer["sources"]:
        raise SystemExit("Expected source metadata on the grounded answer.")

    repeat = ctx.memory.converse(
        user_id=user["id"],
        session_id=first_answer["session_id"],
        question=question,
    )
    print("REPEAT TRACE", repeat["cache_trace"])
    if repeat["cache_trace"].get("exact_answer") != "hit" and repeat["cache_trace"].get("semantic_answer") != "hit":
        raise SystemExit("The repeated question should hit the exact or semantic answer cache.")

    updated = ctx.ingestor.ingest_bytes(FILENAME, UPDATED.encode("utf-8"), str(admin["id"]))
    print(
        "UPDATED",
        updated.status,
        "version",
        updated.version,
        "embedded",
        updated.embedded_count,
        "reused",
        updated.reused_count,
        "removed",
        updated.removed_count,
    )
    if updated.status != "created" or updated.reused_count < 1 or updated.embedded_count < 1:
        raise SystemExit("The edited document should reuse unchanged chunks and embed only the new text.")

    unknown = ctx.memory.converse(
        user_id=user["id"],
        session_id=None,
        question="What is the capital of France?",
    )
    print("UNKNOWN", unknown["answer"])
    if unknown["answer"] != NO_INFO_ANSWER:
        raise SystemExit("Out-of-corpus questions must use the exact fallback sentence.")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
