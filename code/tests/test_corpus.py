"""Tests for corpus frontmatter parsing, chunking, and indexing behavior."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

loaded_code_module = sys.modules.get("code")
if loaded_code_module is not None and not hasattr(loaded_code_module, "__path__"):
    del sys.modules["code"]

from code.config import CHUNK_TOKENS
from code.retrieval.corpus import (
    ParsedDocument,
    build_company_index,
    chunk_markdown,
    chunks_for_document,
    estimate_tokens,
    parse_document,
    parse_frontmatter,
    stable_doc_id,
)


def test_frontmatter_parses_on_sample_fixture() -> None:
    """Frontmatter metadata and markdown body are separated correctly."""

    raw = """---
title: "Sample Article"
source_url: "https://example.com/help"
breadcrumbs:
  - "Root"
  - "Leaf"
last_updated: "2026-05-01"
---

# Sample Article

Body text.
"""

    metadata, body = parse_frontmatter(raw)

    assert metadata["title"] == "Sample Article"
    assert metadata["source_url"] == "https://example.com/help"
    assert metadata["breadcrumbs"] == ["Root", "Leaf"]
    assert metadata["last_updated"] == "2026-05-01"
    assert body.startswith("# Sample Article")


def test_chunker_respects_chunk_token_upper_bound() -> None:
    """Large markdown sections are split under the configured token estimate."""

    body = "## Big Section\n\n" + ("abcd " * (CHUNK_TOKENS + 100))

    chunks = chunk_markdown(body, max_tokens=CHUNK_TOKENS, overlap_chars=120)

    assert len(chunks) > 1
    assert all(estimate_tokens(text) <= CHUNK_TOKENS for _heading, text in chunks)


def test_doc_ids_are_stable_across_runs() -> None:
    """The same relative path and chunk index produce the same id."""

    first = stable_doc_id("claude/example.md", 3)
    second = stable_doc_id("claude/example.md", 3)

    assert first == second
    assert len(first) == 12


def test_chunks_for_document_have_stable_ids() -> None:
    """Chunk ids remain stable for the same parsed document."""

    sample_document = ParsedDocument(
        rel_path="claude/sample.md",
        company="claude",
        title="Sample",
        source_url="https://example.com/sample",
        breadcrumbs=["Claude"],
        last_updated="2026-05-01",
        body="# Sample\n\n## Setup\n\nUse the documented setup flow.",
    )
    first = [chunk.doc_id for chunk in chunks_for_document(sample_document)]
    second = [chunk.doc_id for chunk in chunks_for_document(sample_document)]

    assert first == second


def test_empty_and_malformed_files_are_skipped_with_warning(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    """Bad corpus files warn and return None instead of crashing."""

    data_dir = tmp_path / "data"
    company_dir = data_dir / "claude"
    company_dir.mkdir(parents=True)
    empty_path = company_dir / "empty.md"
    malformed_path = company_dir / "bad.md"
    empty_path.write_text("", encoding="utf-8")
    malformed_path.write_text("---\ntitle: [unterminated\n---\n# Bad", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        empty_result = parse_document(empty_path, data_dir=data_dir)
        malformed_result = parse_document(malformed_path, data_dir=data_dir)

    assert empty_result is None
    assert malformed_result is None
    assert "Skipping empty markdown file" in caplog.text
    assert "Skipping malformed markdown file" in caplog.text


def test_parse_document_normalizes_frontmatter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Parsed documents keep normalized source metadata."""

    data_dir = tmp_path / "data"
    company_dir = data_dir / "visa"
    company_dir.mkdir(parents=True)
    path = company_dir / "sample.md"
    path.write_text(
        """---
title: "Visa Sample"
source_url: "https://example.com/visa"
last_updated_iso: "2026-05-01T00:00:00Z"
breadcrumbs:
  - "Visa"
---
# Visa Sample

## Help
Use the support form.
""",
        encoding="utf-8",
    )

    document = parse_document(path, data_dir=data_dir)

    assert document is not None
    assert document.company == "visa"
    assert document.rel_path == "visa/sample.md"
    assert document.title == "Visa Sample"
    assert document.source_url == "https://example.com/visa"
    assert document.last_updated == "2026-05-01T00:00:00Z"


def test_missing_company_directory_warns_and_returns_zero(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    """Missing company directories warn and produce no chunks."""

    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    data_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        count = build_company_index("missing", data_dir=data_dir, index_dir=index_dir)

    assert count == 0
    assert "Company corpus directory is missing" in caplog.text
    assert not (index_dir / "missing").exists()
