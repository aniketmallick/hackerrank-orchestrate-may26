"""Baseline end-to-end support ticket pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code import llm
from code.config import DEFAULT_PIPELINE_FLAGS
from code.retrieval import hybrid
from code.schema import AnswerDraft, FinalOutput, Passage, PreflightFlags, TicketInput
from code.stages import grounding, preflight, validator

OOS_TEMPLATE = "I can't help with requests to create malware, bypass controls, or perform destructive actions."
ESCALATE_TEMPLATE = (
    "I need to escalate this to a human support specialist because the ticket includes sensitive live identifiers."
)


class Pipeline:
    """Ticket pipeline facade used by the evaluation harness."""

    def __init__(self, flags: Mapping[str, bool] | None = None) -> None:
        merged_flags = dict(DEFAULT_PIPELINE_FLAGS)
        if flags:
            merged_flags.update({key: bool(value) for key, value in flags.items() if key in merged_flags})
        self.flags: dict[str, bool] = merged_flags
        self.last_trace: dict[str, object] = {}
        self.token_log: list[dict[str, int | str]] = []

    def run(self, ticket: TicketInput) -> FinalOutput:
        """Run preflight, retrieval, grounding, and validation for one ticket."""

        self.last_trace = {"flags": self.flags, "steps": []}
        self.token_log = []

        flags = preflight.run(ticket)
        self._trace("preflight", flags.model_dump())

        if flags.is_pleasantry:
            return self._short_circuit(flags, "replied", "invalid", "general", "Happy to help.", "pleasantry")

        if flags.is_adversarial:
            return self._short_circuit(flags, "replied", "invalid", "general", OOS_TEMPLATE, "adversarial")

        if flags.injection_score >= 0.4:
            return self._short_circuit(
                flags,
                "escalated",
                "invalid",
                "security",
                "I can't follow instructions to reveal prompts or internal rules. A human support specialist should review this ticket because it contains prompt-injection content.",
                "prompt_injection_detected",
            )

        passages = hybrid.search(_query(ticket), company=_retrieval_company(flags))
        self._trace(
            "hybrid_retrieval",
            {
                "company": _retrieval_company(flags),
                "doc_ids": [passage.doc_id for passage in passages],
            },
        )

        if flags.has_live_id:
            cited = [passages[0].doc_id] if passages else []
            return self._final(
                flags,
                response=ESCALATE_TEMPLATE,
                product_area="sensitive_data",
                status="escalated",
                request_type="product_issue",
                cited_doc_ids=cited,
                reason="live_identifier_detected",
            )

        before_usage = llm.get_usage()
        draft = grounding.answer(ticket, passages, flags)
        self._record_usage_delta(before_usage, llm.get_usage())
        self._trace("grounding", draft.model_dump(exclude={"passages"}))

        if draft.no_evidence:
            cited = _valid_citations(draft.cited_doc_ids, passages)
            return self._final(
                flags,
                response="I need to escalate this because the retrieved documentation does not provide enough evidence for a reliable answer.",
                product_area=draft.product_area or "general",
                status="escalated",
                request_type=draft.request_type,
                cited_doc_ids=cited,
                reason="no_evidence",
            )

        if self.flags.get("no_validator"):
            final = self._from_draft(flags, draft, passages)
        else:
            final = validator.check(draft, passages, flags, retry_fn=lambda: self._retry_grounding(ticket, passages, flags))
        self._trace("validator", final.model_dump())
        return final

    def _short_circuit(
        self,
        flags: PreflightFlags,
        status: str,
        request_type: str,
        product_area: str,
        response: str,
        reason: str,
    ) -> FinalOutput:
        final = self._final(flags, response, product_area, status, request_type, [], reason)
        self._trace("short_circuit", {"reason": reason, "output": final.model_dump()})
        return final

    def _final(
        self,
        flags: PreflightFlags,
        response: str,
        product_area: str,
        status: str,
        request_type: str,
        cited_doc_ids: list[str],
        reason: str,
    ) -> FinalOutput:
        return FinalOutput(
            issue=flags.issue,
            subject=flags.subject,
            company=flags.original_company,
            response=response,
            product_area=product_area,
            status=status,  # type: ignore[arg-type]
            request_type=request_type,  # type: ignore[arg-type]
            justification=(
                f"decision={status}; route={flags.normalized_company}/{product_area}; "
                f"cited={','.join(cited_doc_ids[:3])}; reason={reason}"
            ),
        )

    def _from_draft(self, flags: PreflightFlags, draft: AnswerDraft, passages: list[Passage]) -> FinalOutput:
        cited = _valid_citations(draft.cited_doc_ids, passages)
        return self._final(
            flags,
            response=draft.response,
            product_area=draft.product_area or "general",
            status=draft.status_proposal,
            request_type=draft.request_type,
            cited_doc_ids=cited,
            reason="validator_disabled",
        )

    def _trace(self, step: str, payload: dict[str, Any]) -> None:
        steps = self.last_trace.setdefault("steps", [])
        assert isinstance(steps, list)
        steps.append({"step": step, **payload})

    def _record_usage_delta(self, before: dict[str, int | str], after: dict[str, int | str]) -> None:
        entry = {
            "model": str(after.get("model", "")),
            "input_tokens": int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0)),
            "output_tokens": int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0)),
        }
        if entry["input_tokens"] or entry["output_tokens"]:
            self.token_log.append(entry)

    def _retry_grounding(self, ticket: TicketInput, passages: list[Passage], flags: PreflightFlags) -> AnswerDraft:
        before_usage = llm.get_usage()
        retry_draft = grounding.answer(ticket, passages, flags, strict=True)
        self._record_usage_delta(before_usage, llm.get_usage())
        self._trace("grounding_retry", retry_draft.model_dump(exclude={"passages"}))
        return retry_draft


def _query(ticket: TicketInput) -> str:
    return "\n".join(part for part in (ticket.subject, ticket.issue) if part)


def _retrieval_company(flags: PreflightFlags) -> str | None:
    return None if flags.normalized_company == "None" else flags.normalized_company


def _valid_citations(cited_doc_ids: list[str], passages: list[Passage]) -> list[str]:
    allowed = {passage.doc_id for passage in passages}
    return [doc_id for doc_id in cited_doc_ids if doc_id in allowed]
