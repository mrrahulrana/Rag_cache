from __future__ import annotations

import hashlib
import math


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\x00", "").split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def format_vector(values: list[float]) -> str:
    parts: list[str] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Embedding contains a non-finite value")
        parts.append(repr(number))
    return "[" + ",".join(parts) + "]"


def parse_vector(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [float(part) for part in text.split(",") if part.strip()]
