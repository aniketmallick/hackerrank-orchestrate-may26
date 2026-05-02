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
VERIFIER_MODEL: Final[str] = "claude-haiku-4-5"
DENSE_MODEL: Final[str] = "BAAI/bge-base-en-v1.5"
TEMP: Final[float] = 0.0
ROUTING_MAX_TOKENS: Final[int] = 256
GROUNDING_MAX_TOKENS: Final[int] = 1024
VERIFIER_MAX_TOKENS: Final[int] = 128
GROUNDING_PASSAGE_CHAR_LIMIT: Final[int] = 1200
EVAL_TZ: Final[str] = "Asia/Kolkata"
COST_RATES_USD_PER_MTOK: Final[dict[str, dict[str, float]]] = {
    MODEL: {"input": 3.0, "output": 15.0},
    VERIFIER_MODEL: {"input": 1.0, "output": 5.0},
    "default": {"input": 3.0, "output": 15.0},
}

CHUNK_TOKENS: Final[int] = 900
OVERLAP: Final[int] = 120
VISA_CHUNK_TOKENS: Final[int] = 600
VISA_OVERLAP: Final[int] = 200
BM25_K: Final[int] = 20
DENSE_K: Final[int] = 20
FUSED_K: Final[int] = 6
VISA_FUSED_K: Final[int] = 8
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
PRODUCT_AREA_LABELS: Final[dict[tuple[str, str], str]] = {
    ("hackerrank", "chakra"): "chakra",
    ("hackerrank", "engage"): "engage",
    ("hackerrank", "general-help"): "general_help",
    ("hackerrank", "hackerrank_community"): "community",
    ("hackerrank", "index.md"): "general_help",
    ("hackerrank", "integrations"): "integrations",
    ("hackerrank", "interviews"): "interviews",
    ("hackerrank", "library"): "library",
    ("hackerrank", "screen"): "screen",
    ("hackerrank", "settings"): "settings",
    ("hackerrank", "skillup"): "skillup",
    ("hackerrank", "uncategorized"): "uncategorized",
    ("claude", "amazon-bedrock"): "bedrock",
    ("claude", "claude"): "claude",
    ("claude", "claude/account-management"): "account_management",
    ("claude", "claude/conversation-management"): "conversation_management",
    ("claude", "claude/features-and-capabilities"): "features_and_capabilities",
    ("claude", "claude/get-started-with-claude"): "get_started_with_claude",
    ("claude", "claude/personalization-and-settings"): "personalization_and_settings",
    ("claude", "claude/troubleshooting"): "troubleshooting",
    ("claude", "claude/usage-and-limits"): "usage_and_limits",
    ("claude", "claude-api-and-console"): "claude_api",
    ("claude", "claude-code"): "claude_code",
    ("claude", "claude-desktop"): "claude_desktop",
    ("claude", "claude-for-education"): "education",
    ("claude", "claude-for-government"): "government",
    ("claude", "claude-for-nonprofits"): "nonprofits",
    ("claude", "claude-in-chrome"): "claude_in_chrome",
    ("claude", "claude-mobile-apps"): "claude_mobile_apps",
    ("claude", "connectors"): "connectors",
    ("claude", "identity-management-sso-jit-scim"): "identity_management",
    ("claude", "index.md"): "claude",
    ("claude", "privacy-and-legal"): "privacy",
    ("claude", "pro-and-max-plans"): "pro_and_max",
    ("claude", "safeguards"): "safeguards",
    ("claude", "team-and-enterprise-plans"): "team_and_enterprise",
    ("visa", "index.md"): "general_support",
    ("visa", "support.md"): "general_support",
    ("visa", "support/consumer.md"): "general_support",
    ("visa", "support/consumer/checkout-fees-contact-form"): "general_support",
    ("visa", "support/consumer/travel-support"): "travel_support",
    ("visa", "support/consumer/travelers-cheques"): "travelers_cheques",
    ("visa", "support/consumer/visa-rules"): "visa_rules",
    ("visa", "support/merchant.md"): "merchant",
    ("visa", "support/small-business/data-security"): "data_security",
    ("visa", "support/small-business/dispute-resolution"): "dispute_resolution",
    ("visa", "support/small-business/fraud-protection"): "fraud_protection",
    ("visa", "support/small-business/regulations-fees"): "regulations_fees",
    ("visa", "support/small-business/travelers-cheques"): "travelers_cheques",
}
EMPTY_PRODUCT_AREA_STATUSES: Final[set[str]] = {
    "escalated",
    "pleasantry_response",
    "adversarial_response",
}
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
