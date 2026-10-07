from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse, urlunparse
from uuid import UUID, uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from rag_cache.textutil import format_vector, parse_vector

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    pass


class Database:
    def __init__(self, database_url: str, embed_dim: int):
        self.database_url = database_url
        self.embed_dim = embed_dim
        self.pool: ConnectionPool | None = None

    def init(self) -> None:
        self._ensure_database()
        self.pool = ConnectionPool(
            self.database_url,
            min_size=1,
            max_size=8,
            kwargs={"autocommit": False, "application_name": "rag_cache"},
            open=True,
        )
        self.pool.wait(timeout=15)
        with self._connection() as conn:
            with conn.transaction():
                for statement in _schema_statements(self.embed_dim):
                    conn.execute(statement)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        if self.pool is None:
            raise DatabaseError("Database pool is not open.")
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            yield conn

    @contextmanager
    def document_lock(self, document_name: str) -> Iterator[psycopg.Connection]:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"document:{document_name}",),
                )
                yield conn

    def create_user(
        self,
        *,
        first_name: str,
        last_name: str,
        username: str,
        password_hash: str,
        role: str,
    ) -> dict:
        user_id = uuid4()
        with self._connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO users (id, first_name, last_name, username, password_hash, role)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, first_name, last_name, username, role, created_at
                    """,
                    (user_id, first_name, last_name, username, password_hash, role),
                ).fetchone()
        return _public_user(row)

    def get_user_by_username(self, username: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, first_name, last_name, username, password_hash, role, created_at
                FROM users
                WHERE lower(username) = lower(%s)
                """,
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def has_admin(self) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT 1 AS ok FROM users WHERE role = 'admin' LIMIT 1").fetchone()
        return row is not None

    def corpus_version(self, conn: psycopg.Connection | None = None) -> int:
        if conn is None:
            with self._connection() as owned:
                return self.corpus_version(owned)
        row = conn.execute("SELECT version FROM corpus_state WHERE id = 1").fetchone()
        if row is None:
            raise DatabaseError("Corpus version is missing.")
        return int(row["version"])

    def active_chunk_count(self) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM chunks WHERE is_active = true"
            ).fetchone()
        return int(row["n"])

    def get_active_document(self, name: str, conn: psycopg.Connection | None = None) -> dict | None:
        query = """
            SELECT id, name, version, file_hash, extension, status, chunk_count,
                   embedded_count, reused_count, stored_path, uploaded_by, created_at
            FROM documents
            WHERE name = %s AND status = 'active'
        """
        if conn is None:
            with self._connection() as owned:
                row = owned.execute(query, (name,)).fetchone()
        else:
            row = conn.execute(query, (name,)).fetchone()
        return dict(row) if row else None

    def active_chunks(self, document_name: str, conn: psycopg.Connection | None = None) -> list[dict]:
        query = """
            SELECT content_hash, section, page_number, metadata, embedding::text AS embedding
            FROM chunks
            WHERE document_name = %s AND is_active = true
            ORDER BY chunk_index
        """
        if conn is None:
            with self._connection() as owned:
                rows = owned.execute(query, (document_name,)).fetchall()
        else:
            rows = conn.execute(query, (document_name,)).fetchall()
        chunks = []
        for row in rows:
            item = dict(row)
            item["embedding"] = parse_vector(item["embedding"])
            item["metadata"] = item["metadata"] or {}
            chunks.append(item)
        return chunks

    def embeddings_for_hashes(
        self,
        hashes: list[str],
        conn: psycopg.Connection | None = None,
    ) -> dict[str, list[float]]:
        if not hashes:
            return {}
        query = """
            SELECT DISTINCT ON (content_hash) content_hash, embedding::text AS embedding
            FROM chunks
            WHERE content_hash = ANY(%s) AND embedding IS NOT NULL
            ORDER BY content_hash, created_at DESC
        """
        if conn is None:
            with self._connection() as owned:
                rows = owned.execute(query, (hashes,)).fetchall()
        else:
            rows = conn.execute(query, (hashes,)).fetchall()
        return {row["content_hash"]: parse_vector(row["embedding"]) for row in rows}

    def save_document_version(
        self,
        conn: psycopg.Connection,
        *,
        name: str,
        file_hash: str,
        extension: str,
        uploaded_by: str | None,
        chunks: list[dict],
        embedded_count: int,
        reused_count: int,
    ) -> dict:
        current = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM documents WHERE name = %s",
            (name,),
        ).fetchone()
        version = int(current["version"]) + 1
        conn.execute(
            "UPDATE documents SET status = 'superseded' WHERE name = %s AND status = 'active'",
            (name,),
        )
        conn.execute(
            "UPDATE chunks SET is_active = false WHERE document_name = %s AND is_active = true",
            (name,),
        )
        document_id = uuid4()
        conn.execute(
            """
            INSERT INTO documents (
                id, name, version, file_hash, extension, status,
                chunk_count, embedded_count, reused_count, uploaded_by
            )
            VALUES (%s, %s, %s, %s, %s, 'active', %s, %s, %s, %s)
            """,
            (
                document_id,
                name,
                version,
                file_hash,
                extension,
                len(chunks),
                embedded_count,
                reused_count,
                uploaded_by,
            ),
        )
        for index, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, document_name, document_version, section, page_number,
                    chunk_index, content, content_hash, metadata, embedding, is_active
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, true)
                """,
                (
                    uuid4(),
                    document_id,
                    name,
                    version,
                    chunk.get("section"),
                    chunk.get("page_number"),
                    index,
                    chunk["content"],
                    chunk["content_hash"],
                    Json(chunk.get("metadata") or {}),
                    format_vector(chunk["embedding"]),
                ),
            )
        conn.execute(
            """
            UPDATE corpus_state
            SET version = version + 1, updated_at = now()
            WHERE id = 1
            """
        )
        conn.execute("ANALYZE chunks")
        return {"id": str(document_id), "name": name, "version": version}

    def list_documents(self) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT d.id, d.name, d.version, d.status, d.extension, d.chunk_count,
                       d.embedded_count, d.reused_count, d.created_at,
                       u.username AS uploaded_by
                FROM documents d
                LEFT JOIN users u ON u.id = d.uploaded_by
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [_jsonable(row) for row in rows]

    def list_chunk_previews(self, document_id: str) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_index, section, page_number, content_hash, metadata,
                       left(content, 220) AS preview
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()
        return [_jsonable(row) for row in rows]

    def search_chunks(self, embedding: list[float], threshold: float, top_k: int) -> list[dict]:
        vector = format_vector(embedding)
        with self._connection() as conn:
            with conn.transaction():
                conn.execute("SET LOCAL hnsw.ef_search = 100")
                rows = conn.execute(
                    """
                    SELECT id, document_name, document_version, section, page_number,
                           content, metadata, content_hash,
                           (1 - (embedding <=> %s::vector)) AS similarity
                    FROM chunks
                    WHERE is_active = true
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> %s::vector)) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, vector, threshold, vector, top_k),
                ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["similarity"] = float(item["similarity"])
            item["metadata"] = item["metadata"] or {}
            results.append(item)
        return results

    def search_semantic_answer(
        self,
        embedding: list[float],
        corpus_version: int,
        threshold: float,
    ) -> dict | None:
        vector = format_vector(embedding)
        with self._connection() as conn:
            with conn.transaction():
                conn.execute("SET LOCAL hnsw.ef_search = 100")
                row = conn.execute(
                    """
                    SELECT answer, sources,
                           (1 - (query_embedding <=> %s::vector)) AS similarity
                    FROM semantic_answers
                    WHERE corpus_version = %s
                      AND (1 - (query_embedding <=> %s::vector)) > %s
                    ORDER BY query_embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (vector, corpus_version, vector, threshold, vector),
                ).fetchone()
        if row is None:
            return None
        return {
            "answer": row["answer"],
            "sources": row["sources"] or [],
            "similarity": float(row["similarity"]),
        }

    def save_semantic_answer(
        self,
        *,
        corpus_version: int,
        query_text: str,
        embedding: list[float],
        answer: str,
        sources: list[dict],
    ) -> None:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO semantic_answers (
                        id, corpus_version, query_text, query_embedding, answer, sources
                    )
                    VALUES (%s, %s, %s, %s::vector, %s, %s)
                    """,
                    (
                        uuid4(),
                        corpus_version,
                        query_text,
                        format_vector(embedding),
                        answer,
                        Json(sources),
                    ),
                )

    def create_chat_session(self, user_id: str, title: str) -> dict:
        session_id = uuid4()
        with self._connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id, title)
                    VALUES (%s, %s, %s)
                    RETURNING id, user_id, title, summary, summarized_message_count,
                              created_at, updated_at
                    """,
                    (session_id, user_id, title),
                ).fetchone()
        return _jsonable(row)

    def list_chat_sessions(self, user_id: str) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, title, summary, summarized_message_count,
                       created_at, updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [_jsonable(row) for row in rows]

    def get_chat_session(self, session_id: str, user_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title, summary, summarized_message_count,
                       created_at, updated_at
                FROM chat_sessions
                WHERE id = %s AND user_id = %s
                """,
                (session_id, user_id),
            ).fetchone()
        return _jsonable(row) if row else None

    def list_chat_messages(self, session_id: str) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, sources, cache_trace, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [_jsonable(row) for row in rows]

    def add_chat_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        cache_trace: dict | None = None,
    ) -> dict:
        message_id = uuid4()
        with self._connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO chat_messages (id, session_id, role, content, sources, cache_trace)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, session_id, role, content, sources, cache_trace, created_at
                    """,
                    (
                        message_id,
                        session_id,
                        role,
                        content,
                        Json(sources) if sources is not None else None,
                        Json(cache_trace) if cache_trace is not None else None,
                    ),
                ).fetchone()
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = now() WHERE id = %s",
                    (session_id,),
                )
        return _jsonable(row)

    def rename_chat_session(self, session_id: str, title: str) -> None:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(
                    "UPDATE chat_sessions SET title = %s, updated_at = now() WHERE id = %s",
                    (title, session_id),
                )

    def update_chat_summary(
        self,
        session_id: str,
        summary: str,
        summarized_message_count: int,
    ) -> None:
        with self._connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET summary = %s,
                        summarized_message_count = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (summary, summarized_message_count, session_id),
                )

    def _ensure_database(self) -> None:
        admin_url, dbname = _split_database_url(self.database_url)
        if not dbname:
            raise DatabaseError("DATABASE_URL must include a database name.")
        try:
            with psycopg.connect(admin_url, autocommit=True, connect_timeout=8) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,),
                ).fetchone()
                if exists is None:
                    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                    logger.info("Created database %s", dbname)
        except psycopg.Error as exc:
            detail = " ".join(str(exc).split())
            lowered = detail.casefold()
            if "password" in lowered or "passwort" in lowered or "authent" in lowered:
                detail = "PostgreSQL rejected the username or password in DATABASE_URL."
            raise DatabaseError(
                "Could not connect to PostgreSQL. Set DATABASE_URL in .env to your local "
                "PostgreSQL 15 user, password, host, and port. "
                + detail
            ) from exc


def _schema_statements(embed_dim: int) -> list[str]:
    if embed_dim <= 0:
        raise DatabaseError("EMBED_DIM must be a positive integer.")
    vector = f"vector({int(embed_dim)})"
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_idx ON users (lower(username))",
        """
        CREATE TABLE IF NOT EXISTS corpus_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        INSERT INTO corpus_state (id, version)
        VALUES (1, 1)
        ON CONFLICT (id) DO NOTHING
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            file_hash TEXT NOT NULL,
            extension TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
            chunk_count INTEGER NOT NULL DEFAULT 0,
            embedded_count INTEGER NOT NULL DEFAULT 0,
            reused_count INTEGER NOT NULL DEFAULT 0,
            stored_path TEXT,
            uploaded_by UUID REFERENCES users (id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (name, version)
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS documents_one_active_name
            ON documents (name) WHERE status = 'active'
        """,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            document_name TEXT NOT NULL,
            document_version INTEGER NOT NULL,
            section TEXT,
            page_number INTEGER,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding {vector},
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash)",
        "CREATE INDEX IF NOT EXISTS chunks_document_name_idx ON chunks (document_name)",
        "CREATE INDEX IF NOT EXISTS chunks_active_idx ON chunks (is_active)",
        f"""
        CREATE TABLE IF NOT EXISTS semantic_answers (
            id UUID PRIMARY KEY,
            corpus_version INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            query_embedding {vector} NOT NULL,
            answer TEXT NOT NULL,
            sources JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS semantic_answers_version_idx ON semantic_answers (corpus_version)",
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT 'New chat',
            summary TEXT,
            summarized_message_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS chat_sessions_user_idx ON chat_sessions (user_id, updated_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sources JSONB,
            cache_trace JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, created_at)",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)",
        """
        CREATE INDEX IF NOT EXISTS semantic_answers_embedding_hnsw
            ON semantic_answers USING hnsw (query_embedding vector_cosine_ops)
        """,
    ]


def _split_database_url(database_url: str) -> tuple[str, str]:
    parsed = urlparse(database_url)
    dbname = parsed.path.lstrip("/").split("?")[0]
    admin = parsed._replace(path="/postgres")
    return urlunparse(admin), dbname


def _public_user(row: dict | None) -> dict:
    if row is None:
        raise DatabaseError("User insert did not return a row.")
    return {
        "id": str(row["id"]),
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "username": row["username"],
        "role": row["role"],
    }


def _jsonable(row: dict | None) -> dict:
    if row is None:
        return {}
    item = dict(row)
    for key, value in list(item.items()):
        if isinstance(value, UUID):
            item[key] = str(value)
    return item
