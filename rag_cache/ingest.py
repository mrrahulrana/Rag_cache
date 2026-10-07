from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from rag_cache.cache import CacheStore
from rag_cache.config import Settings
from rag_cache.db import Database
from rag_cache.loaders import IngestError, TextChunk, parse_file
from rag_cache.models import OllamaModels

ProgressCallback = Callable[[int, int, str], None]

_META_KEYS = ("source_type", "sheet_name", "row_start", "row_end", "slide")


@dataclass
class IngestResult:
    name: str
    version: int
    status: str
    chunk_count: int
    embedded_count: int
    reused_count: int
    removed_count: int
    message: str


class DocumentIngestor:
    def __init__(self, settings: Settings, db: Database, cache: CacheStore, models: OllamaModels):
        self.settings = settings
        self.db = db
        self.cache = cache
        self.models = models

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        user_id: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestResult:
        display_name = Path(filename).name
        if not display_name or display_name in {".", ".."}:
            raise IngestError("The file needs a name.")
        extension = Path(display_name).suffix.lower()
        file_hash = hashlib.sha256(data).hexdigest()
        preview = self.db.get_active_document(display_name)
        if preview and preview["file_hash"] == file_hash:
            return _unchanged_result(display_name, preview)

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / _safe_filename(display_name)
            path.write_bytes(data)
            parsed = parse_file(path, self.settings)
        if not parsed:
            raise IngestError(
                "No text could be extracted. Scanned PDFs and images need Tesseract OCR installed locally."
            )

        with self.db.document_lock(display_name) as conn:
            existing = self.db.get_active_document(display_name, conn)
            existing_chunks = self.db.active_chunks(display_name, conn) if existing else []
            if existing and (
                existing["file_hash"] == file_hash or same_chunk_version(parsed, existing_chunks)
            ):
                return _unchanged_result(display_name, existing, chunk_count=len(existing_chunks))

            ordered, embed_at = _assign_embeddings(
                parsed,
                existing_chunks,
                self.db.embeddings_for_hashes(
                    [chunk.content_hash for chunk in parsed],
                    conn,
                ),
                self.cache,
                self.settings.embed_model,
                self.settings.embed_dim,
            )
            embedded_count = len(embed_at)
            if embedded_count:
                _embed_missing(
                    ordered,
                    embed_at,
                    self.models,
                    self.cache,
                    self.settings,
                    on_progress,
                )
            elif on_progress:
                on_progress(1, 1, "Reusing stored embeddings for unchanged text")

            reused_count = len(ordered) - embedded_count
            removed_count = _removed_count(parsed, existing_chunks)
            saved = self.db.save_document_version(
                conn,
                name=display_name,
                file_hash=file_hash,
                extension=extension or "unknown",
                uploaded_by=user_id,
                chunks=[
                    {
                        "content": chunk.text,
                        "content_hash": chunk.content_hash,
                        "section": chunk.section,
                        "page_number": chunk.page_number,
                        "metadata": chunk.metadata,
                        "embedding": embedding,
                    }
                    for chunk, embedding, _reused in ordered
                ],
                embedded_count=embedded_count,
                reused_count=reused_count,
            )
            stored = _store_upload(self.settings.upload_dir, saved["id"], display_name, data)
            conn.execute(
                "UPDATE documents SET stored_path = %s WHERE id = %s",
                (str(stored), saved["id"]),
            )

        return IngestResult(
            name=display_name,
            version=int(saved["version"]),
            status="created",
            chunk_count=len(ordered),
            embedded_count=embedded_count,
            reused_count=reused_count,
            removed_count=removed_count,
            message=(
                f"{display_name} version {saved['version']} is ready. "
                f"Embedded {embedded_count} changed chunk(s), reused {reused_count} unchanged chunk(s), "
                f"and retired {removed_count} chunk(s) from the previous version."
            ),
        )


def _unchanged_result(name: str, existing: dict, chunk_count: int | None = None) -> IngestResult:
    count = int(existing.get("chunk_count", 0) if chunk_count is None else chunk_count)
    version = int(existing["version"])
    return IngestResult(
        name=name,
        version=version,
        status="unchanged",
        chunk_count=count,
        embedded_count=0,
        reused_count=count,
        removed_count=0,
        message=(
            f"{name} is already at version {version}. "
            "No content changes were detected, so nothing was re-embedded."
        ),
    )


def same_chunk_version(new_chunks: list[TextChunk], existing: list[dict]) -> bool:
    if len(new_chunks) != len(existing):
        return False
    for new, old in zip(new_chunks, existing):
        if new.content_hash != old["content_hash"]:
            return False
        if (new.section or None) != (old.get("section") or None):
            return False
        if new.page_number != old.get("page_number"):
            return False
        if _stable_meta(new.metadata) != _stable_meta(old.get("metadata") or {}):
            return False
    return True


def _stable_meta(metadata: dict) -> tuple:
    return tuple(metadata.get(key) for key in _META_KEYS)


def _removed_count(new_chunks: list[TextChunk], existing: list[dict]) -> int:
    new_counts = Counter(chunk.content_hash for chunk in new_chunks)
    old_counts = Counter(chunk["content_hash"] for chunk in existing)
    removed = old_counts - new_counts
    return sum(removed.values())


def _assign_embeddings(
    new_chunks: list[TextChunk],
    existing_chunks: list[dict],
    stored_vectors: dict[str, list[float]],
    cache: CacheStore,
    embed_model: str,
    embed_dim: int,
) -> tuple[list[tuple[TextChunk, list[float] | None, bool]], list[int]]:
    known: dict[str, list[list[float]]] = {}
    for chunk in existing_chunks:
        vector = chunk.get("embedding") or []
        if len(vector) == embed_dim:
            known.setdefault(chunk["content_hash"], []).append(vector)
    for content_hash, vector in stored_vectors.items():
        if len(vector) == embed_dim:
            known.setdefault(content_hash, []).append(vector)

    ordered: list[tuple[TextChunk, list[float] | None, bool]] = []
    embed_at: list[int] = []
    for chunk in new_chunks:
        bucket = known.get(chunk.content_hash) or []
        if bucket:
            ordered.append((chunk, bucket.pop(0), True))
            continue
        cached = cache.get_embedding("doc", embed_model, chunk.content_hash)
        if cached and len(cached) == embed_dim:
            ordered.append((chunk, cached, True))
            continue
        embed_at.append(len(ordered))
        ordered.append((chunk, None, False))
    return ordered, embed_at


def _embed_missing(
    ordered: list[tuple[TextChunk, list[float] | None, bool]],
    embed_at: list[int],
    models: OllamaModels,
    cache: CacheStore,
    settings: Settings,
    on_progress: ProgressCallback | None,
) -> None:
    batch_size = 8
    total = len(embed_at)
    done = 0
    for offset in range(0, total, batch_size):
        indexes = embed_at[offset : offset + batch_size]
        texts = [ordered[index][0].text for index in indexes]
        vectors = models.embed_documents(texts)
        if len(vectors) != len(indexes):
            raise IngestError("The embedding model returned an unexpected number of vectors.")
        for index, vector in zip(indexes, vectors):
            chunk = ordered[index][0]
            cache.set_embedding(
                "doc",
                settings.embed_model,
                chunk.content_hash,
                vector,
                settings.embedding_ttl_seconds,
            )
            ordered[index] = (chunk, vector, False)
            done += 1
        if on_progress:
            on_progress(done, total, f"Embedding changed text {done} of {total}")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name).strip(" .")
    return cleaned or "document"


def _store_upload(root: Path, document_id: str, filename: str, data: bytes) -> Path:
    destination = root / document_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / _safe_filename(filename)
    path.write_bytes(data)
    return path
