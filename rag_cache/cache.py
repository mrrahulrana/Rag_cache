from __future__ import annotations

import json
import logging
from typing import Any

import redis

logger = logging.getLogger(__name__)


class CacheStore:
    def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError

    def get_embedding(self, kind: str, model: str, content_hash: str) -> list[float] | None:
        raise NotImplementedError

    def set_embedding(
        self,
        kind: str,
        model: str,
        content_hash: str,
        vector: list[float],
        ttl_seconds: int,
    ) -> None:
        raise NotImplementedError


class NullCache(CacheStore):
    def get_json(self, key: str) -> Any | None:
        return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def get_embedding(self, kind: str, model: str, content_hash: str) -> list[float] | None:
        return None

    def set_embedding(
        self,
        kind: str,
        model: str,
        content_hash: str,
        vector: list[float],
        ttl_seconds: int,
    ) -> None:
        return None


class RedisCache(CacheStore):
    def __init__(self, url: str):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()

    def get_json(self, key: str) -> Any | None:
        raw = self.client.get(_prefixed(key))
        if raw is None:
            return None
        return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.setex(_prefixed(key), ttl_seconds, json.dumps(value))

    def get_embedding(self, kind: str, model: str, content_hash: str) -> list[float] | None:
        payload = self.get_json(_embedding_key(kind, model, content_hash))
        if not isinstance(payload, list):
            return None
        return [float(item) for item in payload]

    def set_embedding(
        self,
        kind: str,
        model: str,
        content_hash: str,
        vector: list[float],
        ttl_seconds: int,
    ) -> None:
        self.set_json(_embedding_key(kind, model, content_hash), vector, ttl_seconds)


def _prefixed(key: str) -> str:
    return key if key.startswith("ragcache:") else f"ragcache:{key}"


def _embedding_key(kind: str, model: str, content_hash: str) -> str:
    return f"emb:v1:{model}:{kind}:{content_hash}"


def connect_cache(url: str) -> tuple[CacheStore, bool, str | None]:
    try:
        cache = RedisCache(url)
    except Exception as exc:
        logger.warning("Redis is unavailable; cache layers will miss. %s", exc)
        return NullCache(), False, str(exc)
    return cache, True, None
