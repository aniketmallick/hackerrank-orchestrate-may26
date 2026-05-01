"""Tests for evaluation metrics."""

from __future__ import annotations

import math

from code.config import MODEL
from code.eval.metrics import cost_estimator, exact_match, response_similarity


def test_response_similarity_wires_rouge_score() -> None:
    """ROUGE-L should reward highly overlapping responses."""

    result = response_similarity("Reset your password from account settings.", "Reset your password in account settings.")

    assert result["rouge_l"] > 0.7
    assert result["char_ngram_overlap"] > 0.5
    assert result["below_threshold"] is False


def test_exact_match_trims_and_ignores_case() -> None:
    """Enum matching ignores casing and trailing whitespace."""

    assert exact_match("none", "None ")
    assert exact_match("REPLIED", "replied")
    assert exact_match(" Product_Issue ", "product_issue")


def test_cost_estimator_uses_pinned_per_mtok_rates() -> None:
    """Known token counts produce the expected dollar estimate."""

    cost = cost_estimator(
        [
            {"model": MODEL, "input_tokens": 1_000_000, "output_tokens": 1_000_000},
            {"model": "unknown-model", "prompt_tokens": 500_000, "completion_tokens": 100_000},
        ]
    )

    assert math.isclose(cost, 21.0)
