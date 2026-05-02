"""CLI evaluation harness for the placeholder support pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from code.config import EVAL_TZ, FINAL_OUTPUT_HEADER, ROOT_DIR
from code.eval.golden import ExpectedOutput, load_sample, normalize_company
from code.eval.metrics import cost_estimator, exact_match, response_similarity
from code.pipeline import Pipeline
from code.schema import TicketInput

MATCH_COLUMNS = ("status", "request_type", "product_area", "response")
FUZZ_PATH = ROOT_DIR / "code" / "eval" / "fuzz.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the harness CLI parser."""

    parser = argparse.ArgumentParser(prog="python -m code.eval.harness")
    parser.add_argument(
        "--sample",
        default=str(ROOT_DIR / "support_tickets" / "sample_support_tickets.csv"),
        help="Golden sample CSV path.",
    )
    parser.add_argument("--no-routing", action="store_true", help="Disable routing for ablation.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking for ablation.")
    parser.add_argument("--no-validator", action="store_true", help="Disable validation for ablation.")
    parser.add_argument("--ablate", action="store_true", help="Run full/no-routing/no-rerank/no-validator ablations.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to eval/runs/<timestamp>.")
    parser.add_argument("--fuzz", action="store_true", help="Also run adversarial fuzz tickets for inspection.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run sample and optional fuzz evaluation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else ROOT_DIR / "eval" / "runs" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "no_routing": args.no_routing,
        "no_rerank": args.no_rerank,
        "no_validator": args.no_validator,
    }
    if args.ablate:
        summary = run_ablation(Path(args.sample), out_dir)
        print(f"Wrote ablation summary: {out_dir / 'summary.md'}")
    else:
        summary = run_sample(Path(args.sample), out_dir, flags)
        print(f"Wrote predictions: {out_dir / 'predictions.csv'}")
        print(f"Wrote summary: {out_dir / 'summary.md'}")
        print(f"Wrote trace: {out_dir / 'trace.jsonl'}")

    if args.fuzz:
        fuzz_report = run_fuzz(FUZZ_PATH, out_dir, flags)
        print(f"Wrote fuzz report: {fuzz_report}")

    print(summary)
    return 0


def run_sample(sample_path: Path, out_dir: Path, flags: dict[str, bool]) -> str:
    """Run the pipeline against the golden sample and write artifacts."""

    cases = load_sample(sample_path)
    predictions_path = out_dir / "predictions.csv"
    trace_path = out_dir / "trace.jsonl"
    pipeline = Pipeline(flags=flags)
    fieldnames = list(FINAL_OUTPUT_HEADER) + [
        "status_match",
        "request_type_match",
        "product_area_match",
        "response_match",
        "rouge_l",
        "char_ngram_overlap",
        "response_below_threshold",
    ]
    match_counts = Counter()
    confusion: dict[str, Counter[tuple[str, str]]] = {
        "status": Counter(),
        "request_type": Counter(),
    }
    low_similarity_rows: list[dict[str, object]] = []
    per_row_breakdown: list[dict[str, object]] = []
    token_log: list[dict[str, int | str]] = []

    with predictions_path.open("w", newline="", encoding="utf-8") as predictions_handle, trace_path.open(
        "w", encoding="utf-8"
    ) as trace_handle:
        writer = csv.DictWriter(predictions_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, (ticket, expected) in enumerate(cases, start=1):
            predicted = pipeline.run(ticket)
            predicted_row = predicted.to_csv_row()
            expected_values = _expected_values(expected)
            status_match = exact_match(predicted.status, expected_values["status"])
            request_type_match = exact_match(predicted.request_type, expected_values["request_type"])
            product_area_match = exact_match(predicted.product_area, expected_values["product_area"])
            similarity = response_similarity(predicted.response, expected.response)
            response_match = not bool(similarity["below_threshold"])

            matches = {
                "status": status_match,
                "request_type": request_type_match,
                "product_area": product_area_match,
                "response": response_match,
            }
            match_counts.update({column: int(matched) for column, matched in matches.items()})
            confusion["status"][(expected.normalized["status"], predicted.status)] += 1
            confusion["request_type"][(expected.normalized["request_type"], predicted.request_type)] += 1
            if bool(similarity["below_threshold"]):
                low_similarity_rows.append(
                    {
                        "row": row_index,
                        "subject": ticket.subject,
                        "rouge_l": similarity["rouge_l"],
                        "char_ngram_overlap": similarity["char_ngram_overlap"],
                    }
                )
            token_log.extend(pipeline.token_log)
            per_row_breakdown.append(
                {
                    "row": row_index,
                    "company": ticket.company or "",
                    "status": _match_cell(predicted.status, expected_values["status"], status_match),
                    "request_type": _match_cell(
                        predicted.request_type,
                        expected_values["request_type"],
                        request_type_match,
                    ),
                    "product_area": _match_cell(
                        predicted.product_area,
                        expected_values["product_area"],
                        product_area_match,
                    ),
                    "rouge_l": float(similarity["rouge_l"]),
                    "preflight": _trace_value(pipeline.last_trace, "preflight", "reasons"),
                    "routing": _trace_value(pipeline.last_trace, "routing", "scope"),
                    "top_doc_id": _top_doc_id(pipeline.last_trace),
                }
            )

            writer.writerow(
                {
                    **predicted_row,
                    "status_match": status_match,
                    "request_type_match": request_type_match,
                    "product_area_match": product_area_match,
                    "response_match": response_match,
                    "rouge_l": f"{float(similarity['rouge_l']):.6f}",
                    "char_ngram_overlap": f"{float(similarity['char_ngram_overlap']):.6f}",
                    "response_below_threshold": similarity["below_threshold"],
                }
            )
            trace_handle.write(
                json.dumps(
                    {
                        "row": row_index,
                        "input": ticket.model_dump(),
                        "prediction": predicted.model_dump(),
                        "matches": matches,
                        "trace": pipeline.last_trace,
                        "token_log": pipeline.token_log,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = _render_summary(
        len(cases),
        match_counts,
        confusion,
        low_similarity_rows,
        cost_estimator(token_log),
        per_row_breakdown,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    return summary


def run_ablation(sample_path: Path, out_dir: Path) -> str:
    """Run four pipeline variants and write a compact comparison table."""

    variants = [
        ("full", {"no_routing": False, "no_rerank": False, "no_validator": False}),
        ("no-routing", {"no_routing": True, "no_rerank": False, "no_validator": False}),
        ("no-rerank", {"no_routing": False, "no_rerank": True, "no_validator": False}),
        ("no-validator", {"no_routing": False, "no_rerank": False, "no_validator": True}),
    ]
    rows: list[dict[str, object]] = []
    for name, flags in variants:
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        run_sample(sample_path, variant_dir, flags)
        rows.append(_score_predictions(variant_dir / "predictions.csv", name))

    full_summary = (out_dir / "full" / "summary.md").read_text(encoding="utf-8")
    summary = full_summary + "\n" + _render_ablation_summary(rows)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    return summary


def run_fuzz(fuzz_path: Path, out_dir: Path, flags: dict[str, bool]) -> Path:
    """Run adversarial tickets and write a human-inspection report."""

    pipeline = Pipeline(flags=flags)
    report_path = out_dir / "fuzz_report.md"
    rows = _load_fuzz_inputs(fuzz_path)
    lines = ["# Fuzz Report", "", f"Rows: {len(rows)}", ""]
    for row_index, ticket in enumerate(rows, start=1):
        predicted = pipeline.run(ticket)
        lines.extend(
            [
                f"## Row {row_index}",
                "",
                f"- Subject: {ticket.subject}",
                f"- Company: {ticket.company or ''}",
                f"- Issue: {ticket.issue}",
                f"- Status: {predicted.status}",
                f"- Request Type: {predicted.request_type}",
                f"- Product Area: {predicted.product_area}",
                f"- Response: {predicted.response}",
                f"- Justification: {predicted.justification}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _render_summary(
    total_rows: int,
    match_counts: Counter[str],
    confusion: dict[str, Counter[tuple[str, str]]],
    low_similarity_rows: list[dict[str, object]],
    total_cost: float,
    per_row_breakdown: list[dict[str, object]],
) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        f"Rows evaluated: {total_rows}",
        "",
        "## Per-Column Accuracy",
        "",
        "| Column | Accuracy | Matches | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for column in MATCH_COLUMNS:
        matches = match_counts[column]
        accuracy = matches / total_rows if total_rows else 0.0
        lines.append(f"| {column} | {accuracy:.3f} | {matches} | {total_rows} |")

    lines.extend(["", "## Status Confusion Matrix", ""])
    lines.extend(_render_confusion(confusion["status"]))
    lines.extend(["", "## Request Type Confusion Matrix", ""])
    lines.extend(_render_confusion(confusion["request_type"]))
    lines.extend(["", "## Low-Similarity Rows", ""])
    if low_similarity_rows:
        lines.extend(["| Row | Subject | ROUGE-L | Char 3-gram overlap |", "| ---: | --- | ---: | ---: |"])
        for row in low_similarity_rows:
            lines.append(
                f"| {row['row']} | {_escape_table(str(row['subject']))} | "
                f"{float(row['rouge_l']):.3f} | {float(row['char_ngram_overlap']):.3f} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Per-row breakdown", ""])
    lines.extend(
        [
            "| Row | Company | Status (P/E) | RT (P/E) | PA (P/E) | Resp ROUGE | Preflight | Routing | Top doc_id |",
            "| ---: | --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in per_row_breakdown:
        lines.append(
            f"| {row['row']} | {_escape_table(str(row['company']))} | "
            f"{_escape_table(str(row['status']))} | {_escape_table(str(row['request_type']))} | "
            f"{_escape_table(str(row['product_area']))} | {float(row['rouge_l']):.3f} | "
            f"{_escape_table(str(row['preflight']))} | {_escape_table(str(row['routing']))} | "
            f"{_escape_table(str(row['top_doc_id']))} |"
        )
    average_cost = total_cost / total_rows if total_rows else 0.0
    lines.extend(
        [
            "",
            "## Cost Estimate",
            "",
            f"Total estimated cost: ${total_cost:.6f}",
            f"Average estimated cost per ticket: ${average_cost:.6f}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_confusion(counter: Counter[tuple[str, str]]) -> list[str]:
    if not counter:
        return ["No rows."]
    expected_labels = sorted({expected for expected, _predicted in counter})
    predicted_labels = sorted({predicted for _expected, predicted in counter})
    lines = ["| Expected \\ Predicted | " + " | ".join(predicted_labels) + " |"]
    lines.append("| --- | " + " | ".join("---:" for _label in predicted_labels) + " |")
    for expected in expected_labels:
        counts = [str(counter[(expected, predicted)]) for predicted in predicted_labels]
        lines.append(f"| {expected} | " + " | ".join(counts) + " |")
    return lines


def _load_fuzz_inputs(path: Path) -> list[TicketInput]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            TicketInput(
                issue=row.get("Issue", ""),
                subject=row.get("Subject", ""),
                company=normalize_company(row.get("Company", "")),
            )
            for row in reader
        ]


def _expected_values(expected: ExpectedOutput) -> dict[str, str]:
    return {
        "status": expected.normalized["status"],
        "request_type": expected.normalized["request_type"],
        "product_area": expected.normalized["product_area"],
    }


def _score_predictions(path: Path, variant: str) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    scored: dict[str, object] = {"variant": variant, "total": total}
    for column in MATCH_COLUMNS:
        matches = sum(1 for row in rows if str(row.get(f"{column}_match", "")).lower() == "true")
        scored[f"{column}_accuracy"] = matches / total if total else 0.0
    return scored


def _match_cell(predicted: object, expected: object, matched: bool) -> str:
    marker = "✓" if matched else "✗"
    return f"{predicted or ''}/{expected or ''} {marker}"


def _trace_value(trace: dict[str, object], step: str, key: str) -> str:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return ""
    for item in steps:
        if isinstance(item, dict) and item.get("step") == step:
            value = item.get(key)
            if isinstance(value, list):
                return ",".join(str(part) for part in value)
            return "" if value is None else str(value)
    return ""


def _top_doc_id(trace: dict[str, object]) -> str:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return ""
    for item in steps:
        if not isinstance(item, dict) or item.get("step") != "hybrid_retrieval":
            continue
        doc_ids = item.get("doc_ids")
        if isinstance(doc_ids, list) and doc_ids:
            return str(doc_ids[0])
    return ""


def _render_ablation_summary(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Ablation Summary",
        "",
        "| Variant | Status | Request Type | Product Area | Response | Delta vs Full |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    full = rows[0]
    flat_variants: list[str] = []
    for row in rows:
        deltas = []
        moved = False
        for column in MATCH_COLUMNS:
            accuracy = float(row[f"{column}_accuracy"])
            baseline = float(full[f"{column}_accuracy"])
            delta = accuracy - baseline
            if abs(delta) > 0.0005:
                moved = True
            deltas.append(f"{column} {delta:+.3f}")
        if row["variant"] != "full" and not moved:
            flat_variants.append(str(row["variant"]))
        lines.append(
            f"| {row['variant']} | "
            f"{float(row['status_accuracy']):.3f} | "
            f"{float(row['request_type_accuracy']):.3f} | "
            f"{float(row['product_area_accuracy']):.3f} | "
            f"{float(row['response_accuracy']):.3f} | "
            f"{', '.join(deltas)} |"
        )
    if flat_variants:
        lines.extend(["", "Flat ablations: " + ", ".join(flat_variants) + "."])
    return "\n".join(lines) + "\n"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _timestamp() -> str:
    return datetime.now(ZoneInfo(EVAL_TZ)).strftime("%Y%m%d-%H%M%S-%f")


if __name__ == "__main__":
    raise SystemExit(main())
