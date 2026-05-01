"""Pydantic models for tickets, routing, retrieved passages, and outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from code.config import FINAL_OUTPUT_HEADER

CompanyInput = Literal["HackerRank", "Claude", "Visa", "None"]
Status = Literal["replied", "escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]


class TicketInput(BaseModel):
    """Input support ticket row."""

    issue: str
    subject: str = ""
    company: CompanyInput | None = None


class PreflightFlags(BaseModel):
    """Fast safety and validity signals checked before retrieval."""

    is_empty: bool = False
    is_out_of_scope: bool = False
    is_sensitive: bool = False
    requires_human: bool = False
    reasons: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """Classification and routing decision for a ticket."""

    company: CompanyInput | None = None
    status: Status
    product_area: str
    request_type: RequestType
    justification: str


class Passage(BaseModel):
    """Grounding passage extracted from the local support corpus."""

    doc_id: str
    company: str
    rel_path: str
    title: str | None = None
    source_url: str | None = None
    breadcrumbs: list[str] = Field(default_factory=list)
    last_updated: str | None = None
    heading: str | None = None
    text: str


class AnswerDraft(BaseModel):
    """Intermediate answer grounded in one or more passages."""

    response: str
    justification: str
    passages: list[Passage] = Field(default_factory=list)


class FinalOutput(BaseModel):
    """Final CSV output row with the exact required column order."""

    model_config = ConfigDict(extra="forbid")

    issue: str
    subject: str
    company: CompanyInput | None = None
    response: str
    product_area: str
    status: Status
    request_type: RequestType
    justification: str

    @classmethod
    def csv_header(cls) -> tuple[str, ...]:
        """Return the exact output CSV header expected by the evaluator."""

        return FINAL_OUTPUT_HEADER

    def to_csv_row(self) -> dict[str, str]:
        """Serialize this output in evaluator column order."""

        data = self.model_dump()
        return {column: "" if data[column] is None else str(data[column]) for column in FINAL_OUTPUT_HEADER}
