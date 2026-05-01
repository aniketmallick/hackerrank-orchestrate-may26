"""Evaluation metrics for pipeline predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rouge_score import rouge_scorer

from code.config import COST_RATES_USD_PER_MTOK

DEFAULT_RESPONSE_THRESHOLD = 0.4


def exact_match(predicted: object, expected: object) -> bool:
    """Case-insensitive exact match after trimming whitespace."""

    return _normalize(predicted) == _normalize(expected)


def response_similarity(
    pred_text: str,
    exp_text: str,
    *,
    threshold: float = DEFAULT_RESPONSE_THRESHOLD,
) -> dict[str, float | bool]:
    """Compute ROUGE-L and character n-gram overlap for response text."""

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(exp_text or "", pred_text or "")["rougeL"].fmeasure
    char_overlap = _char_ngram_overlap(pred_text, exp_text)
    return {
        "rouge_l": rouge_l,
        "char_ngram_overlap": char_overlap,
        "below_threshold": rouge_l < threshold,
    }


def cost_estimator(token_log: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> float:
    """Estimate USD cost from token usage using pinned per-million-token rates."""

    if token_log is None:
        return 0.0
    entries = [token_log] if isinstance(token_log, Mapping) else list(token_log)
    total = 0.0
    for entry in entries:
        model = str(entry.get("model", "default"))
        rates = COST_RATES_USD_PER_MTOK.get(model, COST_RATES_USD_PER_MTOK["default"])
        input_tokens = int(entry.get("input_tokens", entry.get("prompt_tokens", 0)) or 0)
        output_tokens = int(entry.get("output_tokens", entry.get("completion_tokens", 0)) or 0)
        total += (input_tokens / 1_000_000) * rates["input"]
        total += (output_tokens / 1_000_000) * rates["output"]
    return total


def _normalize(value: object) -> str:
    return "" if value is None else str(value).strip().lower()


def _char_ngram_overlap(pred_text: str, exp_text: str, n: int = 3) -> float:
    pred = _char_ngrams(pred_text, n)
    expected = _char_ngrams(exp_text, n)
    if not pred and not expected:
        return 1.0
    if not pred or not expected:
        return 0.0
    return len(pred & expected) / len(pred | expected)


def _char_ngrams(text: str, n: int) -> set[str]:
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}
