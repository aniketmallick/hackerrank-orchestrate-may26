"""Validation and finalization for grounded answer drafts."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from code import llm
from code.config import REQUEST_TYPES, STATUSES, VERIFIER_MAX_TOKENS, VERIFIER_MODEL
from code.schema import AnswerDraft, FinalOutput, Passage, PreflightFlags


class VerifierResult(BaseModel):
    """LLM judge result for high-sensitivity grounded answers."""

    supported: bool
    reasoning: str = ""


def check(
    draft: AnswerDraft,
    passages: list[Passage],
    preflight: PreflightFlags,
    retry_fn: Callable[[], AnswerDraft] | None = None,
    sensitivity: str = "low",
) -> FinalOutput:
    """Validate a draft and return an evaluator-ready final output."""

    if preflight.is_adversarial or preflight.is_pleasantry:
        raise AssertionError("Pleasantry/adversarial tickets must short-circuit before validation.")

    candidate = draft
    errors = _validation_errors(candidate, passages, preflight)
    if errors and retry_fn is not None:
        candidate = retry_fn()
        errors = _validation_errors(candidate, passages, preflight)

    verifier_tag = "ok"
    if errors:
        return _escalated(preflight, candidate, passages, "; ".join(errors), verifier_tag=verifier_tag)

    if sensitivity == "high":
        verifier = verify_supported(candidate.response, _cited_passages_text(candidate, passages))
        if not verifier.supported:
            return _escalated(
                preflight,
                candidate,
                passages,
                f"verifier_unsupported:{verifier.reasoning}",
                verifier_tag="escalated",
            )

    status = "escalated" if preflight.has_live_id else candidate.status_proposal
    cited_doc_ids = _valid_citations(candidate.cited_doc_ids, passages)
    reason = "grounded_answer"
    product_area = "" if status == "escalated" else _snake_case(candidate.product_area)
    return FinalOutput(
        issue=preflight.issue,
        subject=preflight.subject,
        company=preflight.original_company,
        response=candidate.response.strip(),
        product_area=product_area,
        status=status,
        request_type=candidate.request_type,
        justification=_justification(
            status,
            preflight,
            product_area,
            cited_doc_ids,
            reason,
            verifier_tag=verifier_tag,
        ),
    )


VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "supported": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["supported", "reasoning"],
}


def verify_supported(response: str, cited_passages_text: str) -> VerifierResult:
    """Use a low-cost Haiku judge to check support for sensitive answers."""

    system = "\n".join(
        [
            "You are a strict support-answer verifier.",
            "Return exactly one structured_output tool call.",
            "Decide whether the response is fully supported by the cited passages.",
            "Do not use outside knowledge. Unsupported promises, invented facts, or over-specific claims mean supported=false.",
        ]
    )
    user = "\n".join(
        [
            "Response:",
            response,
            "Cited passages:",
            cited_passages_text,
        ]
    )
    payload = llm.call_structured(system, user, VERIFIER_SCHEMA, max_tokens=VERIFIER_MAX_TOKENS, model=VERIFIER_MODEL)
    payload.setdefault("reasoning", "")
    return VerifierResult(**payload)


def _validation_errors(draft: AnswerDraft, passages: list[Passage], preflight: PreflightFlags) -> list[str]:
    errors: list[str] = []
    if draft.status_proposal not in STATUSES:
        errors.append("invalid_status")
    if draft.request_type not in REQUEST_TYPES:
        errors.append("invalid_request_type")
    if set(draft.cited_doc_ids) - {passage.doc_id for passage in passages}:
        errors.append("invalid_citation")
    token_count = len(re.findall(r"\S+", draft.response))
    if token_count <= 8 or token_count >= 800:
        errors.append("response_length")
    if preflight.has_live_id and draft.status_proposal != "escalated":
        errors.append("live_id_requires_escalation")
    try:
        AnswerDraft.model_validate(draft.model_dump())
    except ValidationError:
        errors.append("schema")
    return errors


def _escalated(
    preflight: PreflightFlags,
    draft: AnswerDraft,
    passages: list[Passage],
    reason: str,
    *,
    verifier_tag: str = "ok",
) -> FinalOutput:
    cited_doc_ids = _valid_citations(draft.cited_doc_ids, passages) or [passages[0].doc_id] if passages else []
    product_area = ""
    return FinalOutput(
        issue=preflight.issue,
        subject=preflight.subject,
        company=preflight.original_company,
        response="I need to escalate this to a human support specialist because the available documentation does not safely support a direct answer.",
        product_area=product_area,
        status="escalated",
        request_type=draft.request_type if draft.request_type in REQUEST_TYPES else "product_issue",
        justification=_justification("escalated", preflight, product_area, cited_doc_ids, reason, verifier_tag=verifier_tag),
    )


def _cited_passages_text(draft: AnswerDraft, passages: list[Passage]) -> str:
    cited_doc_ids = set(_valid_citations(draft.cited_doc_ids, passages))
    cited = [passage for passage in passages if passage.doc_id in cited_doc_ids]
    if not cited:
        cited = passages[:3]
    return "\n\n".join(
        f"[{passage.doc_id}] {passage.title or passage.heading or passage.rel_path}\n{passage.text}" for passage in cited
    )


def _valid_citations(cited_doc_ids: list[str], passages: list[Passage]) -> list[str]:
    allowed = {passage.doc_id for passage in passages}
    return [doc_id for doc_id in cited_doc_ids if doc_id in allowed]


def _snake_case(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized


def _justification(
    status: str,
    preflight: PreflightFlags,
    product_area: str,
    cited_doc_ids: list[str],
    reason: str,
    *,
    verifier_tag: str = "ok",
) -> str:
    cited = ",".join(cited_doc_ids[:3])
    return (
        f"decision={status}; route={preflight.normalized_company}/{product_area}; "
        f"cited={cited}; reason={reason}; verifier={verifier_tag}"
    )
