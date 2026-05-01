"""Placeholder pipeline for end-to-end measurement plumbing."""

from __future__ import annotations

from collections.abc import Mapping

from code.config import DEFAULT_PIPELINE_FLAGS
from code.schema import FinalOutput, TicketInput


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
        """Return the hardcoded stub output while preserving input columns."""

        self.last_trace = {
            "flags": self.flags,
            "steps": [],
        }
        self.token_log = []
        return FinalOutput(
            issue=ticket.issue,
            subject=ticket.subject,
            company=ticket.company,
            response="(stub)",
            product_area="",
            status="replied",
            request_type="invalid",
            justification="(stub)",
        )
