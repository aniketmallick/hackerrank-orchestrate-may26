"""Hybrid retrieval using BM25, dense cosine search, and RRF fusion."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from code.config import BM25_K, COMPANIES, COMPANY_LABELS, DENSE_K, FUSED_K, INDEX_DIR, RRF_C
from code.schema import Passage

from code.retrieval import bm25, dense

DocId = str


def reciprocal_rank_fusion(rankings: list[list[DocId]], c: int = RRF_C) -> list[tuple[DocId, float]]:
    """Fuse ranked doc ids with reciprocal rank fusion."""

    scores: dict[DocId, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (c + rank))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def search(query: str, company: str | None) -> list[Passage]:
    """Return top fused passages with per-source BM25, dense, and fused scores."""

    companies = list(COMPANIES) if company is None else [_normalize_company(company)]
    rankings: list[list[DocId]] = []
    bm25_scores: dict[DocId, float] = {}
    dense_scores: dict[DocId, float] = {}
    records_by_doc_id: dict[DocId, dict[str, Any]] = {}

    for company_name in companies:
        records_by_doc_id.update(_load_records_by_doc_id(company_name))

        bm25_results = bm25.search(query, company_name, BM25_K)
        dense_results = dense.search(query, company_name, DENSE_K)

        bm25_scores.update(dict(bm25_results))
        dense_scores.update(dict(dense_results))
        rankings.append([doc_id for doc_id, _score in bm25_results])
        rankings.append([doc_id for doc_id, _score in dense_results])

    fused = reciprocal_rank_fusion(rankings, c=RRF_C)[:FUSED_K]
    return [
        _passage_from_record(
            records_by_doc_id[doc_id],
            bm25_score=bm25_scores.get(doc_id),
            dense_score=dense_scores.get(doc_id),
            fused_score=fused_score,
        )
        for doc_id, fused_score in fused
        if doc_id in records_by_doc_id
    ]


def _load_records_by_doc_id(company: str) -> dict[DocId, dict[str, Any]]:
    return dict(_cached_records_by_doc_id(company, _manifest_corpus_hash()))


@lru_cache(maxsize=8)
def _cached_records_by_doc_id(company: str, corpus_hash: str) -> dict[DocId, dict[str, Any]]:
    records: dict[DocId, dict[str, Any]] = {}
    chunks_path = INDEX_DIR / company / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No hybrid passage index for '{company}'; run build-index first.")
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["doc_id"])] = record
    return records


def _manifest_corpus_hash() -> str:
    manifest_path = INDEX_DIR / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_hash = payload.get("corpus_hash")
    if not isinstance(corpus_hash, str) or not corpus_hash:
        raise ValueError(f"Missing corpus_hash in {manifest_path}")
    return corpus_hash


def _passage_from_record(
    record: dict[str, Any],
    bm25_score: float | None,
    dense_score: float | None,
    fused_score: float,
) -> Passage:
    return Passage(
        doc_id=str(record["doc_id"]),
        company=str(record["company"]),
        rel_path=str(record["rel_path"]),
        title=record.get("title"),
        source_url=record.get("source_url"),
        breadcrumbs=list(record.get("breadcrumbs") or []),
        last_updated=record.get("last_updated"),
        heading=record.get("heading"),
        text=str(record["text"]),
        bm25_score=bm25_score,
        dense_score=dense_score,
        fused_score=fused_score,
    )


def _normalize_company(company: str) -> str:
    normalized = company.strip().lower()
    labels = {label.lower(): key for key, label in COMPANY_LABELS.items()}
    normalized = labels.get(normalized, normalized)
    if normalized not in COMPANIES:
        raise ValueError(f"Unknown company for hybrid retrieval: {company}")
    return normalized
