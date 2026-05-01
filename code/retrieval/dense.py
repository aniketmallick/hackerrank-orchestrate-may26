"""Dense cosine retrieval over persisted per-company numpy embeddings.

Embeddings are cached under ``index/<company>/embeddings.npy`` with parallel
doc ids in ``embeddings_doc_ids.json``. The JSON metadata stores the current
``index/manifest.json`` corpus hash; when that hash is unchanged, rebuilds
reuse the vectors and avoid re-embedding the corpus.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from code.config import COMPANIES, COMPANY_LABELS, DENSE_MODEL, INDEX_DIR

logger = logging.getLogger(__name__)


def search(query: str, company: str, k: int) -> list[tuple[str, float]]:
    """Return the top-k dense doc ids and cosine scores for one company."""

    normalized_company = _normalize_company(company)
    if k <= 0 or not query.strip():
        return []

    embedding_index = load_or_build_embeddings(normalized_company)
    query_vector = _encode_texts([_bge_query(query)])[0]
    scores = embedding_index.embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:k]
    return [(embedding_index.doc_ids[idx], float(scores[idx])) for idx in top_indices]


def build_all_embeddings() -> dict[str, bool]:
    """Ensure all company embeddings exist; return company -> cache hit."""

    results: dict[str, bool] = {}
    for company in COMPANIES:
        results[company] = load_or_build_embeddings(company).cache_hit
    return results


class EmbeddingIndex:
    """In-memory dense index for one company."""

    def __init__(self, doc_ids: list[str], embeddings: np.ndarray, cache_hit: bool) -> None:
        self.doc_ids = doc_ids
        self.embeddings = embeddings
        self.cache_hit = cache_hit


def load_or_build_embeddings(company: str) -> EmbeddingIndex:
    """Load cached embeddings when the manifest hash matches, otherwise rebuild."""

    normalized_company = _normalize_company(company)
    embeddings_path = INDEX_DIR / normalized_company / "embeddings.npy"
    doc_ids_path = INDEX_DIR / normalized_company / "embeddings_doc_ids.json"
    corpus_hash = _manifest_corpus_hash()
    records = _load_chunk_records(normalized_company)
    expected_doc_ids = [str(record["doc_id"]) for record in records]

    cached_doc_ids = _load_cached_doc_ids(doc_ids_path, corpus_hash)
    if embeddings_path.exists() and cached_doc_ids == expected_doc_ids:
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(expected_doc_ids):
            logger.info("Dense embedding cache hit for %s at corpus hash %s", normalized_company, corpus_hash)
            return EmbeddingIndex(expected_doc_ids, embeddings.astype(np.float32, copy=False), cache_hit=True)

    texts = [_searchable_text(record) for record in records]
    embeddings = _encode_texts([_bge_passage(text) for text in texts])
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    metadata = {
        "corpus_hash": corpus_hash,
        "model": DENSE_MODEL,
        "doc_ids": expected_doc_ids,
    }
    doc_ids_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Dense embeddings rebuilt for %s at corpus hash %s", normalized_company, corpus_hash)
    return EmbeddingIndex(expected_doc_ids, embeddings, cache_hit=False)


@lru_cache(maxsize=1)
def _model() -> Any:
    _set_random_seeds()
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(DENSE_MODEL, local_files_only=True)
    except OSError:
        logger.info("Dense model %s not found in local cache; downloading once.", DENSE_MODEL)
        return SentenceTransformer(DENSE_MODEL)


def _encode_texts(texts: list[str]) -> np.ndarray:
    vectors = _model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _set_random_seeds() -> None:
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except ImportError:
        logger.debug("Torch is unavailable; numpy seed still set.")


def _load_cached_doc_ids(path: Path, corpus_hash: str) -> list[str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid dense doc-id cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("corpus_hash") != corpus_hash or payload.get("model") != DENSE_MODEL:
        return None
    doc_ids = payload.get("doc_ids")
    if not isinstance(doc_ids, list) or not all(isinstance(item, str) for item in doc_ids):
        return None
    return doc_ids


def _manifest_corpus_hash() -> str:
    manifest_path = INDEX_DIR / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_hash = payload.get("corpus_hash")
    if not isinstance(corpus_hash, str) or not corpus_hash:
        raise ValueError(f"Missing corpus_hash in {manifest_path}")
    return corpus_hash


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


def _bge_query(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query.strip()}"


def _bge_passage(text: str) -> str:
    return text.strip()


def _normalize_company(company: str) -> str:
    normalized = company.strip().lower()
    labels = {label.lower(): key for key, label in COMPANY_LABELS.items()}
    normalized = labels.get(normalized, normalized)
    if normalized not in COMPANIES:
        raise ValueError(f"Unknown company for dense retrieval: {company}")
    return normalized
