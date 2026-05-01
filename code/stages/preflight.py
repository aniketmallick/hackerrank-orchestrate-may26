"""Deterministic ticket preflight checks."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any

from langdetect import LangDetectException, detect

from code.schema import CompanyInput, PreflightFlags, TicketInput

COMPANY_MAP: dict[str, CompanyInput] = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
    "none": "None",
    "": "None",
}
LIVE_ID_RE = re.compile(r"(?<!\w)(?:cs_live_[A-Za-z0-9_*-]+|pi_live_[A-Za-z0-9_*-]+|sk_live_[A-Za-z0-9_*-]+|ch_live_[A-Za-z0-9_*-]+|\d(?:[\s-]?\d){12,18})(?!\w)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
ROLE_TAG_RE = re.compile(r"<\|im_(?:start|end)\|>|</?(?:system|assistant|user)>", re.IGNORECASE)
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{200,}={0,2})(?![A-Za-z0-9+/=])")
INJECTION_KEYWORDS = (
    "ignore previous",
    "system prompt",
    "show me your prompt",
    "afficher toutes les",
    "muestra todas las",
    "internal rules",
    "ignorez toutes les instructions",
    "instructions précédentes",
    "prompt secreto",
)
PLEASANTRY_WORDS = {
    "thanks",
    "thank",
    "thankyou",
    "thank-you",
    "thx",
    "merci",
    "gracias",
    "appreciate",
    "hello",
    "hi",
}
INTENT_WORDS = {
    "reset",
    "login",
    "log",
    "password",
    "billing",
    "payment",
    "test",
    "assessment",
    "invite",
    "error",
    "issue",
    "not",
    "cannot",
    "can't",
    "failed",
    "problem",
    "refund",
}
ADVERSARIAL_PHRASES = (
    "delete all files",
    "give me malware",
    "write a virus",
    "write me malware",
    "create malware",
    "steal credentials",
    "exfiltrate",
    "bypass owner approval",
)


def normalize_company(raw: object) -> CompanyInput:
    """Normalize raw company input to the supported display labels."""

    normalized = "" if raw is None else str(raw).strip().lower()
    return COMPANY_MAP.get(normalized, "None")


def detect_language(text: str) -> str:
    """Detect language, defaulting to English when langdetect cannot decide."""

    if not text.strip():
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def injection_score(text: str) -> float:
    """Return a 0..1 prompt-injection score from deterministic signals."""

    lowered = text.lower()
    hits = sum(1 for keyword in INJECTION_KEYWORDS if keyword in lowered)
    if ROLE_TAG_RE.search(text):
        hits += 2
    if _contains_base64_blob(text):
        hits += 2
    return min(1.0, hits / 5)


def is_pleasantry(text: str) -> bool:
    """Return True for short gratitude/greeting text with no support intent."""

    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens or len(tokens) > 6:
        return False
    if any(token in INTENT_WORDS for token in tokens):
        return False
    return any(token in PLEASANTRY_WORDS for token in tokens)


def is_adversarial(text: str) -> bool:
    """Return True for clear off-topic destructive or ToS-violating requests."""

    lowered = text.lower()
    return any(phrase in lowered for phrase in ADVERSARIAL_PHRASES)


def run(ticket: TicketInput | Mapping[str, Any]) -> PreflightFlags:
    """Run deterministic checks over a ticket."""

    ticket_input = _coerce_ticket(ticket)
    text = "\n".join(part for part in (ticket_input.subject, ticket_input.issue) if part)
    normalized_company = normalize_company(ticket_input.company)
    has_live_id = bool(LIVE_ID_RE.search(text))
    has_email_phone = bool(EMAIL_RE.search(text) or PHONE_RE.search(text))
    reasons: list[str] = []
    if has_live_id:
        reasons.append("live_id")
    if has_email_phone:
        reasons.append("email_or_phone")

    score = injection_score(text)
    if score > 0:
        reasons.append("prompt_injection_signal")

    adversarial = is_adversarial(text)
    if adversarial:
        reasons.append("adversarial")

    pleasantry = is_pleasantry(text)
    if pleasantry:
        reasons.append("pleasantry")

    return PreflightFlags(
        issue=ticket_input.issue,
        subject=ticket_input.subject,
        original_company=ticket_input.company,
        normalized_company=normalized_company,
        language=detect_language(text),
        has_live_id=has_live_id,
        has_email_phone=has_email_phone,
        injection_score=score,
        is_pleasantry=pleasantry,
        is_adversarial=adversarial,
        is_empty=not bool(text.strip()),
        is_sensitive=has_live_id or has_email_phone,
        requires_human=has_live_id,
        reasons=reasons,
    )


def _coerce_ticket(ticket: TicketInput | Mapping[str, Any]) -> TicketInput:
    if isinstance(ticket, TicketInput):
        return ticket
    return TicketInput(
        issue=str(ticket.get("issue") or ticket.get("Issue") or ""),
        subject=str(ticket.get("subject") or ticket.get("Subject") or ""),
        company=normalize_company(ticket.get("company") or ticket.get("Company")),
    )


def _contains_base64_blob(text: str) -> bool:
    for match in BASE64_RE.finditer(text):
        blob = match.group(1)
        try:
            base64.b64decode(blob, validate=True)
        except binascii.Error:
            continue
        return True
    return False
