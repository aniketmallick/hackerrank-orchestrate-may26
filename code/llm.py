"""Thin Anthropic client wrapper placeholder."""

from __future__ import annotations

import logging
import os

from code.config import MODEL

logger = logging.getLogger(__name__)


def get_api_key() -> str | None:
    """Read the Anthropic API key from the environment."""

    return os.getenv("ANTHROPIC_API_KEY")


def health_check() -> None:
    """Log the configured model name without making a network call."""

    _ = get_api_key()
    logger.info("Anthropic model configured: %s", MODEL)
