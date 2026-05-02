"""Baseline end-to-end support ticket pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code import llm
from code.config import DEFAULT_PIPELINE_FLAGS, VISA_FUSED_K
from code.retrieval import hybrid
from code.schema import AnswerDraft, FinalOutput, Passage, PreflightFlags, RoutingDecision, TicketInput
from code.stages import grounding, preflight, routing, validator

BENIGN_OOS_TEMPLATE = "I am sorry, this is out of scope from my capabilities"
ADVERSARIAL_TEMPLATE = "I can't help with requests to create malware, bypass controls, or perform destructive actions."
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
            return self._short_circuit(flags, "replied", "invalid", "", "Happy to help.", "pleasantry")

        if flags.is_adversarial:
            return self._short_circuit(flags, "replied", "invalid", "", ADVERSARIAL_TEMPLATE, "adversarial")

        if flags.has_live_id:
            return self._final(
                flags,
                response=ESCALATE_TEMPLATE,
                product_area="",
                status="escalated",
                request_type="product_issue",
                cited_doc_ids=[],
                reason="live billing identifier present",
            )

        if flags.injection_score >= 0.4:
            return self._short_circuit(
                flags,
                "escalated",
                "invalid",
                "",
                "I can't follow instructions to reveal prompts or internal rules. A human support specialist should review this ticket because it contains prompt-injection content.",
                "prompt_injection_detected",
            )

        before_usage = llm.get_usage()
        route = _preflight_route(flags) if self.flags.get("no_routing") else routing.classify(ticket, flags)
        route = route.model_copy(update={"sensitivity": _conservative_sensitivity(route, flags)})
        self._record_usage_delta(before_usage, llm.get_usage())
        if flags.normalized_company == "None" and route.resolved_company:
            flags = flags.model_copy(update={"normalized_company": route.resolved_company})
        self._trace("routing", route.model_dump())

        if route.scope == "pleasantry":
            return self._short_circuit(flags, "replied", route.request_type or "invalid", "", "Happy to help.", "pleasantry")

        if route.scope == "out_of_scope_benign":
            return self._short_circuit(
                flags,
                "replied",
                route.request_type or "invalid",
                "",
                BENIGN_OOS_TEMPLATE,
                "out_of_scope",
            )

        if route.scope == "adversarial":
            return self._short_circuit(
                flags,
                "replied",
                route.request_type or "invalid",
                "",
                ADVERSARIAL_TEMPLATE,
                "adversarial",
            )

        if route.injection_attempt and route.scope == "in_scope":
            # Injection embedded in a legitimate support request.
            # The <untrusted_user_input> quarantine in grounding neutralises it.
            # Proceed with normal retrieval+grounding flow.
            self._trace("injection_quarantined", {"reason": "injection_attempt_with_legitimate_intent"})

        passages = _retrieve(
            _query(ticket),
            company=_retrieval_company(flags),
            no_rerank=self.flags.get("no_rerank", False),
        )
        self._trace(
            "hybrid_retrieval",
            {
                "company": _retrieval_company(flags),
                "doc_ids": [passage.doc_id for passage in passages],
                "top_fused_score": passages[0].fused_score if passages else None,
            },
        )

        if route.scope == "ambiguous_underspecified" and not _has_high_confidence_retrieval(passages):
            cited = [passages[0].doc_id] if passages else []
            return self._final(
                flags,
                response="I need to escalate this because the request is underspecified and the available documentation does not identify the affected product or action confidently.",
                product_area="",
                status="escalated",
                request_type=route.request_type or "bug",
                cited_doc_ids=cited,
                reason="ambiguous_underspecified",
            )

        before_usage = llm.get_usage()
        draft = grounding.answer(ticket, passages, flags, intents=route.intents)
        self._record_usage_delta(before_usage, llm.get_usage())
        draft = _normalize_supported_status(draft, passages, flags)
        draft = _with_deterministic_product_area(draft, passages)
        self._trace("grounding", draft.model_dump(exclude={"passages"}))

        if draft.no_evidence:
            cited = _valid_citations(draft.cited_doc_ids, passages)
            return self._final(
                flags,
                response="I need to escalate this because the retrieved documentation does not provide enough evidence for a reliable answer.",
                product_area="",
                status="escalated",
                request_type=route.request_type or draft.request_type,
                cited_doc_ids=cited,
                reason="no_evidence",
            )

        if route.request_type is not None:
            draft = draft.model_copy(update={"request_type": route.request_type})

        if self.flags.get("no_validator"):
            final = self._from_draft(flags, draft, passages)
        else:
            before_usage = llm.get_usage()
            final = validator.check(
                draft,
                passages,
                flags,
                retry_fn=lambda: self._retry_grounding(ticket, passages, flags, route.intents),
                sensitivity=route.sensitivity,
            )
            self._record_usage_delta(before_usage, llm.get_usage())
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
            product_area="" if draft.status_proposal == "escalated" else draft.product_area,
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

    def _retry_grounding(
        self,
        ticket: TicketInput,
        passages: list[Passage],
        flags: PreflightFlags,
        intents: list[str],
    ) -> AnswerDraft:
        retry_draft = grounding.answer(ticket, passages, flags, strict=True, intents=intents)
        retry_draft = _normalize_supported_status(retry_draft, passages, flags)
        retry_draft = _with_deterministic_product_area(retry_draft, passages)
        self._trace("grounding_retry", retry_draft.model_dump(exclude={"passages"}))
        return retry_draft


def _query(ticket: TicketInput) -> str:
    return "\n".join(part for part in (ticket.subject, ticket.issue) if part)


def _retrieval_company(flags: PreflightFlags) -> str | None:
    return None if flags.normalized_company == "None" else flags.normalized_company


def _retrieve(query: str, company: str | None, no_rerank: bool) -> list[Passage]:
    fused_k = VISA_FUSED_K if company == "Visa" else None
    if no_rerank:
        return hybrid.search_bm25_only(query, company=company, fused_k=fused_k)
    return hybrid.search(query, company=company, fused_k=fused_k)


def _preflight_route(flags: PreflightFlags) -> RoutingDecision:
    if flags.is_pleasantry:
        scope = "pleasantry"
    elif flags.is_adversarial:
        scope = "adversarial"
    elif flags.is_empty:
        scope = "ambiguous_underspecified"
    else:
        scope = "in_scope"
    sensitivity = "high" if flags.has_live_id else "medium" if flags.is_sensitive else "low"
    return RoutingDecision(
        scope=scope,
        intents=[_query_text(flags)],
        sensitivity=sensitivity,  # type: ignore[arg-type]
        resolved_company=None if flags.normalized_company == "None" else flags.normalized_company,
        request_type="invalid" if scope in {"pleasantry", "adversarial"} else "bug" if scope == "ambiguous_underspecified" else None,
        rationale="preflight_only",
    )


def _query_text(flags: PreflightFlags) -> str:
    return " ".join(part for part in (flags.subject, flags.issue) if part).strip() or "unspecified support request"


def _conservative_sensitivity(route: RoutingDecision, flags: PreflightFlags) -> str:
    if route.sensitivity != "high":
        return route.sensitivity
    text = _query_text(flags).lower()
    high_risk_markers = (
        "account access",
        "restore my access",
        "restore access",
        "locked account",
        "cannot log in",
        "can't log in",
        "password",
        "fraud",
        "identity theft",
        "security disclosure",
        "vulnerability",
        "specific transaction",
        "transaction id",
        "charge id",
        "payment dispute",
    )
    if any(marker in text for marker in high_risk_markers):
        return "high"
    return "medium"


def _has_high_confidence_retrieval(passages: list[Passage]) -> bool:
    if not passages:
        return False
    top = passages[0]
    fused = top.fused_score or 0.0
    bm25_score = top.bm25_score or 0.0
    dense_score = top.dense_score or 0.0
    return fused >= 0.02 or bm25_score >= 2.0 or dense_score >= 0.6


def _with_deterministic_product_area(draft: AnswerDraft, passages: list[Passage]) -> AnswerDraft:
    if draft.status_proposal == "escalated":
        return draft.model_copy(update={"product_area": ""})
    return draft.model_copy(update={"product_area": _derive_product_area(draft.cited_doc_ids, passages)})


def _normalize_supported_status(draft: AnswerDraft, passages: list[Passage], flags: PreflightFlags) -> AnswerDraft:
    if draft.status_proposal != "escalated" or draft.no_evidence or flags.has_live_id:
        return draft
    if not _valid_citations(draft.cited_doc_ids, passages):
        return draft
    return draft.model_copy(update={"status_proposal": "replied"})


def _derive_product_area(cited_doc_ids: list[str], passages: list[Passage]) -> str:
    passages_by_doc_id = {passage.doc_id: passage for passage in passages}
    for doc_id in cited_doc_ids:
        passage = passages_by_doc_id.get(doc_id)
        if passage is not None:
            return passage.product_area_key or ""
    return ""


def _valid_citations(cited_doc_ids: list[str], passages: list[Passage]) -> list[str]:
    allowed = {passage.doc_id for passage in passages}
    return [doc_id for doc_id in cited_doc_ids if doc_id in allowed]
