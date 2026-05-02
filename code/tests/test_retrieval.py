"""Tests for BM25, dense, and hybrid retrieval."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

loaded_code_module = sys.modules.get("code")
if loaded_code_module is not None and not hasattr(loaded_code_module, "__path__"):
    del sys.modules["code"]

from code.retrieval import bm25, dense, hybrid


def _rel_path_for_doc_id(company: str, doc_id: str) -> str:
    chunks_path = ROOT_DIR / "index" / company / "chunks.jsonl"
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["doc_id"] == doc_id:
                return str(row["rel_path"])
    raise AssertionError(f"Unknown doc_id for {company}: {doc_id}")


def test_rrf_synthetic() -> None:
    """RRF assigns the hand-computed sum of reciprocal rank contributions."""

    fused = hybrid.reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], c=60)
    scores = dict(fused)

    assert math.isclose(scores["a"], (1 / 61) + (1 / 63))
    assert math.isclose(scores["b"], (1 / 62) + (1 / 61))
    assert math.isclose(scores["c"], 1 / 63)
    assert math.isclose(scores["d"], 1 / 62)
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "d", "c"]


def test_bm25_lost_card() -> None:
    """Visa lost/stolen card searches surface consumer travel support docs."""

    results = bm25.search("lost stolen visa card", company="Visa", k=10)
    paths = [_rel_path_for_doc_id("visa", doc_id) for doc_id, _score in results]

    assert any("travel-support" in path or "consumer" in path for path in paths)


def test_dense_paraphrase() -> None:
    """Dense retrieval beats BM25 on the account-access paraphrase."""

    query = "I cannot get into my account"
    target_path = "hackerrank/hackerrank_community/account-settings/manage-account/2403570133-update-or-reset-password.md"

    dense_results = dense.search(query, company="HackerRank", k=20)
    bm25_results = bm25.search(query, company="HackerRank", k=20)

    dense_paths = [_rel_path_for_doc_id("hackerrank", doc_id) for doc_id, _score in dense_results]
    bm25_paths = [_rel_path_for_doc_id("hackerrank", doc_id) for doc_id, _score in bm25_results]

    assert target_path in dense_paths
    assert dense_paths.index(target_path) < (bm25_paths.index(target_path) if target_path in bm25_paths else 20)


def test_hybrid_fanout_for_None() -> None:
    """An ambiguous query with no company fan-outs across distinct companies."""

    passages = hybrid.search("account login password support", company=None)
    companies = {passage.company for passage in passages}

    assert len(passages) > 0
    assert len(companies) >= 2


def test_bm25_only_search_returns_passages_without_dense_scores() -> None:
    """The no-rerank path uses lexical retrieval only."""

    passages = hybrid.search_bm25_only("lost stolen visa card", company="Visa")

    assert passages
    assert passages[0].bm25_score is not None
    assert passages[0].dense_score is None
    assert passages[0].fused_score == 0.0
