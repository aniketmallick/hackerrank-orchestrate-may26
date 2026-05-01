"""Shared configuration constants for the support triage agent."""

from pathlib import Path
from typing import Final

ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
CODE_DIR: Final[Path] = ROOT_DIR / "code"
DATA_DIR: Final[Path] = ROOT_DIR / "data"
INDEX_DIR: Final[Path] = ROOT_DIR / "index"
SUPPORT_TICKETS_DIR: Final[Path] = ROOT_DIR / "support_tickets"
OUTPUT_CSV: Final[Path] = SUPPORT_TICKETS_DIR / "output.csv"

MODEL: Final[str] = "claude-sonnet-4-5-20250929"
DENSE_MODEL: Final[str] = "BAAI/bge-base-en-v1.5"
TEMP: Final[float] = 0.0
EVAL_TZ: Final[str] = "Asia/Kolkata"
COST_RATES_USD_PER_MTOK: Final[dict[str, dict[str, float]]] = {
    MODEL: {"input": 3.0, "output": 15.0},
    "default": {"input": 3.0, "output": 15.0},
}

CHUNK_TOKENS: Final[int] = 900
OVERLAP: Final[int] = 120
BM25_K: Final[int] = 20
DENSE_K: Final[int] = 20
FUSED_K: Final[int] = 6
RRF_C: Final[int] = 60

COMPANIES: Final[tuple[str, ...]] = ("hackerrank", "claude", "visa")
COMPANY_LABELS: Final[dict[str, str]] = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}
INPUT_COMPANIES: Final[tuple[str, ...]] = ("HackerRank", "Claude", "Visa", "None")
STATUSES: Final[tuple[str, ...]] = ("replied", "escalated")
REQUEST_TYPES: Final[tuple[str, ...]] = (
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
)
DEFAULT_PIPELINE_FLAGS: Final[dict[str, bool]] = {
    "no_routing": False,
    "no_rerank": False,
    "no_validator": False,
}
FINAL_OUTPUT_HEADER: Final[tuple[str, ...]] = (
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
)
