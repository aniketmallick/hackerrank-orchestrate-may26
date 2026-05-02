"""Tests for draft validation and high-sensitivity verifier behavior."""

from __future__ import annotations

from code.config import VERIFIER_MAX_TOKENS, VERIFIER_MODEL
from code.schema import AnswerDraft, Passage
from code.stages import preflight, validator


def _draft() -> AnswerDraft:
    return AnswerDraft(
        response="Use the documented account recovery flow and escalate ownership disputes to support.",
        cited_doc_ids=["doc-1"],
        product_area="account_access",
        status_proposal="replied",
        request_type="product_issue",
        no_evidence=False,
    )


def _passage() -> Passage:
    return Passage(
        doc_id="doc-1",
        company="hackerrank",
        rel_path="account.md",
        product_area_key="community",
        text="Account recovery requires documented ownership verification before access is restored.",
    )


def test_high_sensitivity_unsupported_verifier_forces_escalation(monkeypatch) -> None:
    """Verifier failure overrides an otherwise valid high-sensitivity draft."""

    monkeypatch.setattr(
        "code.stages.validator.verify_supported",
        lambda response, cited_passages_text: validator.VerifierResult(
            supported=False,
            reasoning="The response promises an access outcome not present in the cited passage.",
        ),
    )
    flags = preflight.run({"issue": "Restore my account access.", "subject": "Account access", "company": "HackerRank"})

    final = validator.check(_draft(), [_passage()], flags, sensitivity="high")

    assert final.status == "escalated"
    assert "verifier=escalated" in final.justification


def test_low_sensitivity_skips_verifier(monkeypatch) -> None:
    """The extra Haiku judge is only paid for high-sensitivity tickets."""

    def fail_verify(*_args, **_kwargs):
        raise AssertionError("verifier should not run")

    monkeypatch.setattr("code.stages.validator.verify_supported", fail_verify)
    flags = preflight.run({"issue": "Reset an invite link.", "subject": "Invite", "company": "HackerRank"})

    final = validator.check(_draft(), [_passage()], flags, sensitivity="low")

    assert final.status == "replied"
    assert "verifier=ok" in final.justification


def test_verify_supported_uses_verifier_token_cap(monkeypatch) -> None:
    """Verifier passes the configured Haiku model and max token cap."""

    captured: dict[str, object] = {}

    def fake_call(*_args, **kwargs):
        captured.update(kwargs)
        return {"supported": True}

    monkeypatch.setattr("code.stages.validator.llm.call_structured", fake_call)

    result = validator.verify_supported("Supported answer.", "Cited passage text.")

    assert result.supported is True
    assert captured["model"] == VERIFIER_MODEL
    assert captured["max_tokens"] == VERIFIER_MAX_TOKENS
