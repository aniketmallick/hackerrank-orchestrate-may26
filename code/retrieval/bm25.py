"""BM25 keyword retrieval over the local per-company corpus index."""

from __future__ import annotations

import json
import logging
import pickle
import re
import string
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from code.config import COMPANIES, COMPANY_LABELS, INDEX_DIR

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\W+")
MARKDOWN_PUNCTUATION = str.maketrans({char: " " for char in string.punctuation})


def tokenize(text: str) -> list[str]:
    """Lowercase text, strip markdown punctuation, and split on non-word runs."""

    normalized = text.lower().translate(MARKDOWN_PUNCTUATION)
    return [token for token in TOKEN_RE.split(normalized) if token]


def search(query: str, company: str, k: int) -> list[tuple[str, float]]:
    """Return the top-k BM25 doc ids and scores for one company."""

    normalized_company = _normalize_company(company)
    if k <= 0 or not query.strip():
        return []

    index = _load_or_build_bm25(normalized_company)
    scores = index["bm25"].get_scores(tokenize(query))
    doc_ids: list[str] = index["doc_ids"]
    ranked = sorted(
        ((doc_ids[idx], float(score)) for idx, score in enumerate(scores)),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(doc_id, score) for doc_id, score in ranked[:k] if score > 0]


def _load_or_build_bm25(company: str) -> dict[str, Any]:
    cache_path = INDEX_DIR / company / "bm25.pkl"
    chunks_path = INDEX_DIR / company / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No BM25 index for '{company}'; run build-index first.")
    chunks_mtime_ns = chunks_path.stat().st_mtime_ns

    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("chunks_mtime_ns") == chunks_mtime_ns:
                return cached
        except (OSError, pickle.PickleError, AttributeError, EOFError) as exc:
            logger.warning("Ignoring invalid BM25 cache %s: %s", cache_path, exc)

    records = _load_chunk_records(company)
    doc_ids = [str(record["doc_id"]) for record in records]
    tokenized_corpus = [tokenize(_searchable_text(record)) for record in records]
    bm25 = BM25Okapi(tokenized_corpus)
    built = {
        "chunks_mtime_ns": chunks_mtime_ns,
        "doc_ids": doc_ids,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(built, handle)
    return built


def _load_chunk_records(company: str) -> list[dict[str, Any]]:
    chunks_path = INDEX_DIR / company / "chunks.jsonl"
    records: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _searchable_text(record: dict[str, Any]) -> str:
    fields = (
        record.get("title"),
        record.get("heading"),
        " ".join(record.get("breadcrumbs") or []),
        record.get("rel_path"),
        record.get("text"),
    )
    return "\n".join(str(field) for field in fields if field)


def _normalize_company(company: str) -> str:
    normalized = company.strip().lower()
    labels = {label.lower(): key for key, label in COMPANY_LABELS.items()}
    normalized = labels.get(normalized, normalized)
    if normalized not in COMPANIES:
        raise ValueError(f"Unknown company for BM25 retrieval: {company}")
    return normalized
