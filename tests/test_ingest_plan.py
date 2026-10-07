from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_cache.cache import NullCache
from rag_cache.db import _schema_statements
from rag_cache.ingest import _assign_embeddings, _removed_count, same_chunk_version
from rag_cache.loaders import TextChunk
from rag_cache.prompts import NO_INFO_ANSWER, is_no_info
from rag_cache.sources import format_source
from rag_cache.textutil import content_hash, format_vector, parse_vector


class TextTests(unittest.TestCase):
    def test_content_hash_ignores_whitespace_differences(self) -> None:
        self.assertEqual(content_hash("Blue  crates"), content_hash("Blue crates"))

    def test_vector_round_trip(self) -> None:
        values = [0.25, -1.5, 0.0]
        self.assertEqual(parse_vector(format_vector(values)), values)

    def test_schema_uses_json_object_default(self) -> None:
        statements = "\n".join(_schema_statements(768))
        self.assertIn("vector(768)", statements)
        self.assertIn("'{}'::jsonb", statements)
        self.assertNotIn("'{{}}'", statements)


class IngestPlanTests(unittest.TestCase):
    def test_reuses_unchanged_chunk_and_marks_new_text(self) -> None:
        kept = TextChunk(text="Shipping leaves at 06:30.", section="SHIPPING", page_number=1, metadata={"source_type": "text"})
        changed = TextChunk(text="Returns are accepted within 30 days.", section="RETURNS", page_number=2, metadata={"source_type": "text"})
        existing = [
            {
                "content_hash": kept.content_hash,
                "section": "SHIPPING",
                "page_number": 1,
                "metadata": {"source_type": "text"},
                "embedding": [0.1, 0.2],
            }
        ]
        ordered, embed_at = _assign_embeddings([kept, changed], existing, {}, NullCache(), "nomic-embed-text", 2)
        self.assertEqual(embed_at, [1])
        self.assertEqual(ordered[0][1], [0.1, 0.2])
        self.assertTrue(ordered[0][2])
        self.assertIsNone(ordered[1][1])
        self.assertEqual(_removed_count([kept, changed], existing), 0)

    def test_same_version_detects_metadata_change(self) -> None:
        chunk = TextChunk(text="Dock hours", section="HOURS", page_number=1, metadata={"source_type": "text"})
        existing = [
            {
                "content_hash": chunk.content_hash,
                "section": "HOURS",
                "page_number": 4,
                "metadata": {"source_type": "text"},
                "embedding": [1.0],
            }
        ]
        self.assertFalse(same_chunk_version([chunk], existing))
        existing[0]["page_number"] = 1
        self.assertTrue(same_chunk_version([chunk], existing))

    def test_removed_chunks_are_counted(self) -> None:
        kept = TextChunk(text="Keep me", section=None, page_number=1, metadata={})
        old = TextChunk(text="Delete me", section=None, page_number=1, metadata={})
        removed = _removed_count([kept], [{"content_hash": kept.content_hash}, {"content_hash": old.content_hash}])
        self.assertEqual(removed, 1)


class AnswerTests(unittest.TestCase):
    def test_no_info_detection(self) -> None:
        self.assertTrue(is_no_info(f"Note: {NO_INFO_ANSWER}"))
        self.assertFalse(is_no_info("Shipping uses form NX-42."))

    def test_source_format_includes_version_section_and_page(self) -> None:
        label = format_source(
            {
                "document_name": "Warehouse Policy.pdf",
                "document_version": 2,
                "section": "SHIPPING",
                "page_number": 3,
                "similarity": 0.8123,
                "metadata": {},
            }
        )
        self.assertIn("Warehouse Policy.pdf · v2", label)
        self.assertIn("SHIPPING", label)
        self.assertIn("page 3", label)
        self.assertIn("0.81", label)


if __name__ == "__main__":
    unittest.main()
