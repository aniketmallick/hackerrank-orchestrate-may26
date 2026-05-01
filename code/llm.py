"""Anthropic structured-output client wrapper."""

from __future__ import annotations

import time
import logging
import os
from typing import Any

from anthropic import APIStatusError, Anthropic, RateLimitError
from dotenv import load_dotenv

from code.config import MODEL, TEMP

logger = logging.getLogger(__name__)
load_dotenv()

_USAGE: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_CLIENT: Anthropic | None = None


def get_api_key() -> str | None:
    """Read the Anthropic API key from the environment."""

    return os.getenv("ANTHROPIC_API_KEY")


def health_check() -> None:
    """Log the configured model name without making a network call."""

    _ = get_api_key()
    logger.info("Anthropic model configured: %s", MODEL)


def get_usage() -> dict[str, int | str]:
    """Return per-process Anthropic usage counters."""

    return {"model": MODEL, **_USAGE}


def reset_usage() -> None:
    """Reset per-process Anthropic usage counters."""

    _USAGE.update({"calls": 0, "input_tokens": 0, "output_tokens": 0})


def call_structured(system: str, user: str, tool_schema: dict[str, Any], max_tokens: int = 1024) -> dict[str, Any]:
    """Call Anthropic tool use and return the forced JSON object."""

    tool = {
        "name": "structured_output",
        "description": "Return the requested structured JSON object.",
        "input_schema": tool_schema,
    }
    delay = 1.0
    last_error: Exception | None = None

    for attempt in range(5):
        try:
            response = _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=TEMP,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "structured_output"},
            )
            _record_usage(response)
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_input = getattr(block, "input", None)
                    if isinstance(tool_input, dict):
                        return tool_input
            raise ValueError("Anthropic response did not contain structured tool_use input.")
        except (RateLimitError, APIStatusError) as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code not in {429, 529} or attempt >= 4:
                raise
            logger.warning("Anthropic transient status %s; retrying attempt %s/4", status_code, attempt + 1)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Anthropic structured call failed after retries.") from last_error


def _record_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    _USAGE["calls"] += 1
    _USAGE["input_tokens"] += input_tokens
    _USAGE["output_tokens"] += output_tokens


def _get_client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(api_key=get_api_key())
    return _CLIENT
