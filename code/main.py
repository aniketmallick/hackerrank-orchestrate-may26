"""Command-line entry point for the HackerRank Orchestrate support agent."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from code.config import FUSED_K
from code.retrieval import dense, hybrid
from code.retrieval.corpus import build_index

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(prog="python -m code.main")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser("build-index", help="Build the local markdown corpus index")
    build_index_parser.set_defaults(func=_build_index_command)

    retrieve_parser = subparsers.add_parser("retrieve", help="Run hybrid retrieval for a query")
    retrieve_parser.add_argument("--company", required=True, help="Company label or None")
    retrieve_parser.add_argument("--query", required=True, help="Support query to retrieve passages for")
    retrieve_parser.set_defaults(func=_retrieve_command)

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
    cache_hits = dense.build_all_embeddings()
    for company, cache_hit in cache_hits.items():
        state = "cache hit; unchanged corpus hash" if cache_hit else "rebuilt"
        print(f"{company}: dense embeddings {state}")
    return 0


def _retrieve_command(args: argparse.Namespace) -> int:
    company = None if args.company.strip().lower() == "none" else args.company
    passages = hybrid.search(args.query, company=company)
    for idx, passage in enumerate(passages[:FUSED_K], start=1):
        print(f"{idx}. {passage.company} {passage.rel_path} [{passage.doc_id}]")
        print(
            "   scores "
            f"bm25={_format_score(passage.bm25_score)} "
            f"dense={_format_score(passage.dense_score)} "
            f"fused={_format_score(passage.fused_score)}"
        )
        heading = f" - {passage.heading}" if passage.heading else ""
        title = passage.title or "Untitled"
        print(f"   {title}{heading}")
        snippet = " ".join(passage.text.split())[:300]
        print(f"   {snippet}")
    return 0


def _format_score(score: float | None) -> str:
    if score is None:
        return "None"
    return f"{score:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
