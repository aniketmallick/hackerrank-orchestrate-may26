"""Tests for grounding LLM call configuration."""

from __future__ import annotations

from code.config import GROUNDING_MAX_TOKENS
from code.schema import Passage, TicketInput
from code.stages import grounding, preflight


def _visa_passage(doc_id: str, text: str, rel_path: str = "visa/support/consumer/travelers-cheques.md") -> Passage:
    return Passage(
        doc_id=doc_id,
        company="visa",
        rel_path=rel_path,
        product_area_key="travelers_cheques",
        text=text,
    )


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


def test_grounding_context_includes_citicorp_phone_number(monkeypatch) -> None:
    """P0-B: passage with Citicorp Freephone number is passed verbatim in the context."""

    captured: dict[str, str] = {}

    def fake_call(system: str, user: str, *_args, **_kwargs):
        captured["system"] = system
        captured["user"] = user
        return {
            "response": "Call Citicorp Freephone: 1-800-645-6556 or Collect: 1-813-623-1709.",
            "cited_doc_ids": ["doc-visa-1"],
            "status_proposal": "replied",
            "request_type": "product_issue",
            "no_evidence": False,
        }

    monkeypatch.setattr("code.stages.grounding.llm.call_structured", fake_call)
    ticket = TicketInput(
        issue="How do I contact Citicorp about my travelers cheque?", subject="Citicorp contact", company="Visa"
    )
    passage = _visa_passage(
        "doc-visa-1",
        "### Citicorp\n\nFreephone: 1-800-645-6556\nCollect: 1-813-623-1709\nMonday-Friday, 6:30 am to 2:30 pm EST",
    )

    grounding.answer(ticket, [passage], preflight.run(ticket))

    assert "1-800-645-6556" in captured["user"]
    assert "verbatim" in captured["system"]


def test_grounding_context_includes_visa_india_lost_card_number(monkeypatch) -> None:
    """P0-B: passage with Visa India lost-card number is passed verbatim in the context."""

    captured: dict[str, str] = {}

    def fake_call(system: str, user: str, *_args, **_kwargs):
        captured["user"] = user
        return {
            "response": "Report a lost card by calling Visa at 000-800-100-1219.",
            "cited_doc_ids": ["doc-visa-2"],
            "status_proposal": "replied",
            "request_type": "product_issue",
            "no_evidence": False,
        }

    monkeypatch.setattr("code.stages.grounding.llm.call_structured", fake_call)
    ticket = TicketInput(
        issue="How do I report a lost Visa card in India?", subject="Lost Visa India", company="Visa"
    )
    passage = _visa_passage(
        "doc-visa-2",
        "Report a lost card by calling Visa at 000-800-100-1219.",
        rel_path="visa/support.md",
    )

    grounding.answer(ticket, [passage], preflight.run(ticket))

    assert "000-800-100-1219" in captured["user"]
