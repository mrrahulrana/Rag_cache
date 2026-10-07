from __future__ import annotations


def format_source(source: dict) -> str:
    name = source.get("document_name") or "Document"
    version = source.get("document_version")
    parts = [f"{name} · v{version}" if version is not None else name]
    section = source.get("section")
    if section:
        parts.append(str(section))
    page = source.get("page_number")
    if page:
        parts.append(f"page {page}")
    metadata = source.get("metadata") or {}
    if metadata.get("row_start") and metadata.get("row_end"):
        parts.append(f"rows {metadata['row_start']}–{metadata['row_end']}")
    similarity = source.get("similarity")
    if similarity is not None:
        parts.append(f"similarity {float(similarity):.2f}")
    return " · ".join(parts)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        header = format_source(chunk)
        blocks.append(f"[{index}] {header}\n{chunk['content']}")
    return "\n\n".join(blocks)


def to_source(chunk: dict) -> dict:
    content = chunk.get("content") or ""
    snippet = content if len(content) <= 400 else content[:400] + "..."
    similarity = chunk.get("similarity")
    return {
        "document_name": chunk.get("document_name"),
        "document_version": chunk.get("document_version"),
        "section": chunk.get("section"),
        "page_number": chunk.get("page_number"),
        "similarity": round(float(similarity), 4) if similarity is not None else None,
        "metadata": chunk.get("metadata") or {},
        "snippet": snippet,
    }
