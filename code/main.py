"""Command-line entry point for the HackerRank Orchestrate support agent."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from code.config import FINAL_OUTPUT_HEADER, FUSED_K
from code.pipeline import Pipeline
from code.retrieval import dense, hybrid
from code.retrieval.corpus import build_index
from code.schema import TicketInput
from code.stages.preflight import normalize_company

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

    run_parser = subparsers.add_parser("run", help="Run the ticket agent on a CSV")
    run_parser.add_argument("--input", required=True, help="Input support-ticket CSV")
    run_parser.add_argument("--output", required=True, help="Output predictions CSV")
    run_parser.set_defaults(func=_run_command)

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


def _run_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    trace_path = output_path.with_name("trace.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline()

    with input_path.open(newline="", encoding="utf-8") as input_handle:
        rows = list(csv.DictReader(input_handle))

    with output_path.open("w", newline="", encoding="utf-8") as output_handle, trace_path.open(
        "w", encoding="utf-8"
    ) as trace_handle:
        writer = csv.DictWriter(output_handle, fieldnames=FINAL_OUTPUT_HEADER)
        writer.writeheader()
        for row_index, row in enumerate(tqdm(rows, desc="tickets"), start=1):
            ticket = _ticket_from_row(row)
            final = pipeline.run(ticket)
            csv_row = final.to_csv_row()
            # Preserve the original company string verbatim (including "None").
            csv_row["company"] = row.get("Company") or row.get("company") or ""
            writer.writerow(csv_row)
            trace_handle.write(
                json.dumps(
                    {
                        "row": row_index,
                        "input": ticket.model_dump(),
                        "prediction": final.model_dump(),
                        "trace": pipeline.last_trace,
                        "token_log": pipeline.token_log,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote predictions: {output_path}")
    print(f"Wrote trace: {trace_path}")
    return 0


def _format_score(score: float | None) -> str:
    if score is None:
        return "None"
    return f"{score:.6f}"


def _ticket_from_row(row: dict[str, str]) -> TicketInput:
    normalized_company = normalize_company(row.get("company") or row.get("Company"))
    return TicketInput(
        issue=row.get("issue") or row.get("Issue") or "",
        subject=row.get("subject") or row.get("Subject") or "",
        company=None if normalized_company == "None" else normalized_company,
    )


if __name__ == "__main__":
    raise SystemExit(main())
