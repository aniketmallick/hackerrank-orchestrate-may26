"""Tests for deterministic product-area derivation."""

from __future__ import annotations

import csv
from code.config import DATA_DIR, PRODUCT_AREA_LABELS, ROOT_DIR
from code.pipeline import Pipeline
from code.retrieval import corpus
from code.schema import AnswerDraft, Passage, TicketInput
from code.stages.preflight import normalize_company


def _passage(doc_id: str, rel_path: str, product_area_key: str, fused_score: float = 0.02) -> Passage:
    company = rel_path.split("/", maxsplit=1)[0]
    return Passage(
        doc_id=doc_id,
        company=company,
        rel_path=rel_path,
        title=rel_path,
        product_area_key=product_area_key,
        text=f"Support passage for {product_area_key}.",
        fused_score=fused_score,
    )


def test_product_area_labels_cover_all_markdown_paths() -> None:
    """Every indexed markdown path must resolve to a hand-authored label."""

    missing: list[tuple[str, str, str]] = []
    for path in corpus.iter_markdown_files(DATA_DIR):
        rel_path = path.relative_to(DATA_DIR).as_posix()
        company, lookup_key = corpus.product_area_lookup_key(rel_path)
        if (company, lookup_key) not in PRODUCT_AREA_LABELS:
            missing.append((rel_path, company, lookup_key))

    assert missing == []


def test_chunks_persist_product_area_key() -> None:
    """Chunks carry the deterministic product-area key into chunks.jsonl."""

    document = corpus.parse_document(
        DATA_DIR / "hackerrank" / "screen" / "test-settings" / "9672590042-modifying-general-settings-for-tests.md"
    )
    assert document is not None

    chunks = corpus.chunks_for_document(document)

    assert chunks
    assert {chunk.product_area_key for chunk in chunks} == {"screen"}


def test_pipeline_derives_product_area_from_top_cited_passage(monkeypatch) -> None:
    """The LLM's free-form product_area is ignored."""

    passages = [
        _passage("low", "hackerrank/library/example.md", "library", fused_score=0.01),
        _passage("high", "hackerrank/screen/example.md", "screen", fused_score=0.05),
    ]
    monkeypatch.setattr("code.pipeline.hybrid.search", lambda *_args, **_kwargs: passages)
    monkeypatch.setattr(
        "code.pipeline.grounding.answer",
        lambda *_args, **_kwargs: AnswerDraft(
            response="This answer is grounded in the cited screen passage.",
            cited_doc_ids=["low", "high"],
            product_area="wrong_llm_label",
            status_proposal="replied",
            request_type="product_issue",
            no_evidence=False,
        ),
    )

    result = Pipeline(flags={"no_routing": True}).run(
        TicketInput(issue="How long do tests stay active?", subject="Tests", company="HackerRank")
    )

    assert result.product_area == "screen"


def test_company_set_sample_rows_derive_expected_product_area(monkeypatch) -> None:
    """The seven sample rows with company set derive sample-anchored labels."""

    expected_by_query_part = {
        "Test Active in the system": ("screen", "hackerrank/screen/example.md"),
        "many default versions of roles": ("screen", "hackerrank/screen/example.md"),
        "extra time added": ("screen", "hackerrank/screen/example.md"),
        "google login on hackerrank community": ("community", "hackerrank/hackerrank_community/example.md"),
        "Card stolen": ("general_support", "visa/support/consumer.md"),
    }
    expected_by_issue_prefix = {
        "One of my claude conversations": ("privacy", "claude/privacy-and-legal/example.md"),
        "I bought Visa Traveller's Cheques": ("travel_support", "visa/support/consumer/travel-support/example.md"),
    }

    sample_rows = _company_set_sample_rows()

    def fake_search(query: str, company: str | None):
        for prefix, (_label, rel_path) in expected_by_issue_prefix.items():
            if prefix in query:
                return [_passage("doc-1", rel_path, corpus.product_area_key_for_rel_path(rel_path), fused_score=0.05)]
        for part, (label, rel_path) in expected_by_query_part.items():
            if part in query:
                return [_passage("doc-1", rel_path, label, fused_score=0.05)]
        raise AssertionError(f"Unhandled sample query: {query}")
        return [_passage("doc-1", rel_path, label, fused_score=0.05)]

    monkeypatch.setattr("code.pipeline.hybrid.search", fake_search)
    monkeypatch.setattr(
        "code.pipeline.grounding.answer",
        lambda *_args, **_kwargs: AnswerDraft(
            response="This stub response is long enough for validator checks.",
            cited_doc_ids=["doc-1"],
            product_area="llm_should_not_win",
            status_proposal="replied",
            request_type="product_issue",
            no_evidence=False,
        ),
    )

    results = [Pipeline(flags={"no_routing": True}).run(ticket) for ticket, _expected in sample_rows]

    assert [result.product_area for result in results] == [expected for _ticket, expected in sample_rows]


def _company_set_sample_rows() -> list[tuple[TicketInput, str]]:
    path = ROOT_DIR / "support_tickets" / "sample_support_tickets.csv"
    wanted: list[tuple[TicketInput, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            company = normalize_company(row.get("Company", ""))
            if company == "None":
                continue
            ticket = TicketInput(issue=row.get("Issue", ""), subject=row.get("Subject", ""), company=company)
            wanted.append((ticket, row.get("Product Area", "")))
    return wanted
