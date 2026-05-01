"""Command-line entry point for the HackerRank Orchestrate support agent."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from code.retrieval.corpus import build_index

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(prog="python -m code.main")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser("build-index", help="Build the local markdown corpus index")
    build_index_parser.set_defaults(func=_build_index_command)

    subparsers.add_parser("run", help="Planned: run the ticket agent")
    subparsers.add_parser("eval", help="Planned: evaluate predictions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.error(f"Command '{args.command}' is planned but not implemented yet")
    return int(args.func(args))


def _build_index_command(_args: argparse.Namespace) -> int:
    counts = build_index()
    for company, count in counts.items():
        print(f"{company}: {count} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
