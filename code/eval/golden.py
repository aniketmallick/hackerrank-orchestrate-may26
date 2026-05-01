"""Golden sample loading and normalization utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from code.schema import TicketInput


@dataclass(frozen=True)
class ExpectedOutput:
    """Expected output row with display values and comparison values split."""

    issue: str
    subject: str
    company: str | None
    response: str
    product_area: str
    status: str
    request_type: str
    display: dict[str, str]
    normalized: dict[str, str]


def load_sample(path: str | Path) -> list[tuple[TicketInput, ExpectedOutput]]:
    """Load sample rows into ticket inputs and expected outputs."""

    rows: list[tuple[TicketInput, ExpectedOutput]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            issue = raw.get("Issue", "")
            subject = raw.get("Subject", "")
            company = normalize_company(raw.get("Company", ""))
            ticket = TicketInput(issue=issue, subject=subject, company=company)
            expected = ExpectedOutput(
                issue=issue,
                subject=subject,
                company=company,
                response=raw.get("Response", ""),
                product_area=raw.get("Product Area", ""),
                status=raw.get("Status", ""),
                request_type=raw.get("Request Type", ""),
                display={
                    "company": raw.get("Company", ""),
                    "product_area": raw.get("Product Area", ""),
                    "status": raw.get("Status", ""),
                    "request_type": raw.get("Request Type", ""),
                },
                normalized={
                    "company": "" if company is None else company,
                    "product_area": _normalize_enum(raw.get("Product Area", "")),
                    "status": _normalize_enum(raw.get("Status", "")),
                    "request_type": _normalize_enum(raw.get("Request Type", "")),
                },
            )
            rows.append((ticket, expected))
    return rows


def normalize_company(value: str | None) -> str | None:
    """Normalize blank and literal None company values to None."""

    company = (value or "").strip()
    if not company or company.lower() == "none":
        return None
    return company


def _normalize_enum(value: object) -> str:
    return "" if value is None else str(value).strip().lower()
