"""Tests for grounding LLM call configuration."""

from __future__ import annotations

from code.config import GROUNDING_MAX_TOKENS
from code.schema import Passage, TicketInput
from code.stages import grounding, preflight


def test_grounding_uses_configured_token_cap(monkeypatch) -> None:
    """Grounding passes the configured max token cap to Anthropic."""

    captured: dict[str, int] = {}

    def fake_call(*_args, **kwargs):
        captured["max_tokens"] = kwargs["max_tokens"]
        return {
            "response": "This answer cites the retrieved support passage.",
            "cited_doc_ids": ["doc-1"],
            "status_proposal": "replied",
            "request_type": "product_issue",
            "no_evidence": False,
        }

    monkeypatch.setattr("code.stages.grounding.llm.call_structured", fake_call)
    ticket = TicketInput(issue="How do I reset an invite?", subject="Invite", company="HackerRank")
    passage = Passage(
        doc_id="doc-1",
        company="hackerrank",
        rel_path="hackerrank/screen/example.md",
        product_area_key="screen",
        text="Reset invitation links from the candidate invite page.",
    )

    grounding.answer(ticket, [passage], preflight.run(ticket))

    assert captured["max_tokens"] == GROUNDING_MAX_TOKENS
