"""Build a deterministic markdown chunk index from the local support corpus."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm

from code.config import CHUNK_TOKENS, COMPANIES, DATA_DIR, INDEX_DIR, OVERLAP, PRODUCT_AREA_LABELS

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ParsedDocument:
    """Markdown document body with normalized metadata."""

    rel_path: str
    company: str
    title: str | None
    source_url: str | None
    breadcrumbs: list[str]
    last_updated: str | None
    body: str


@dataclass(frozen=True)
class CorpusChunk:
    """Serializable chunk emitted to the on-disk corpus index."""

    doc_id: str
    company: str
    rel_path: str
    chunk_idx: int
    title: str | None
    source_url: str | None
    breadcrumbs: list[str]
    last_updated: str | None
    product_area_key: str
    heading: str | None
    text: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    """Estimate token count with a deterministic character heuristic.

    The indexer intentionally avoids a tokenizer dependency at this stage.
    It uses the common rough estimate of four characters per token and rounds
    up so chunks stay conservatively within the configured upper bound.
    """

    stripped = text.strip()
    if not stripped:
        return 0
    return math.ceil(len(stripped) / CHARS_PER_TOKEN)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter when present and return metadata plus body."""

    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing_idx: int | None = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = idx
            break

    if closing_idx is None:
        raise ValueError("frontmatter block is missing a closing delimiter")

    raw_frontmatter = "\n".join(lines[1:closing_idx])
    body = "\n".join(lines[closing_idx + 1 :]).strip()
    parsed = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must parse to a mapping")
    return parsed, body


def iter_markdown_files(data_dir: Path = DATA_DIR) -> Iterable[Path]:
    """Yield all markdown corpus files under the configured company folders."""

    for company in COMPANIES:
        company_dir = data_dir / company
        if not company_dir.exists():
            logger.warning("Company corpus directory is missing: %s", company_dir)
            continue
        yield from sorted(company_dir.rglob("*.md"))


def parse_document(path: Path, data_dir: Path = DATA_DIR) -> ParsedDocument | None:
    """Parse a markdown corpus file, returning None when it should be skipped."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("Skipping unreadable markdown file %s: %s", path, exc)
        return None

    if not text.strip():
        logger.warning("Skipping empty markdown file: %s", path)
        return None

    try:
        metadata, body = parse_frontmatter(text)
    except (yaml.YAMLError, ValueError) as exc:
        logger.warning("Skipping malformed markdown file %s: %s", path, exc)
        return None

    if not body.strip():
        logger.warning("Skipping markdown file with empty body: %s", path)
        return None

    rel_path = path.relative_to(data_dir).as_posix()
    company = rel_path.split("/", maxsplit=1)[0]
    title = _coerce_optional_str(metadata.get("title")) or _extract_h1(body)
    source_url = _coerce_optional_str(metadata.get("source_url"))
    breadcrumbs = _coerce_str_list(metadata.get("breadcrumbs"))
    last_updated = _first_present(
        metadata,
        ("last_updated", "last_updated_iso", "last_updated_exact", "last_updated_relative"),
    )
    return ParsedDocument(
        rel_path=rel_path,
        company=company,
        title=title,
        source_url=source_url,
        breadcrumbs=breadcrumbs,
        last_updated=last_updated,
        body=body.strip(),
    )


def chunk_markdown(
    body: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap_chars: int = OVERLAP,
) -> list[tuple[str | None, str]]:
    """Split markdown by H2/H3 sections and pack chunks below max_tokens."""

    max_chars = max_tokens * CHARS_PER_TOKEN
    sections = _split_sections(body)
    chunks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_text = ""

    for heading, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        if estimate_tokens(section_text) > max_tokens:
            if current_text:
                chunks.append((current_heading, current_text.strip()))
                current_text = ""
                current_heading = None
            chunks.extend((heading, part) for part in _split_large_text(section_text, max_chars, overlap_chars))
            continue

        proposed = _join_chunk_text(current_text, section_text)
        if current_text and estimate_tokens(proposed) > max_tokens:
            chunks.append((current_heading, current_text.strip()))
            prefix = _overlap_suffix(current_text, overlap_chars)
            current_text = _join_chunk_text(prefix, section_text)
            current_heading = heading
        else:
            current_text = proposed
            current_heading = current_heading or heading

    if current_text.strip():
        chunks.append((current_heading, current_text.strip()))

    bounded_chunks: list[tuple[str | None, str]] = []
    for heading, text in chunks:
        if estimate_tokens(text) <= max_tokens:
            bounded_chunks.append((heading, text))
        else:
            bounded_chunks.extend((heading, part) for part in _split_large_text(text, max_chars, overlap_chars))
    return bounded_chunks


def build_company_index(company: str, data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> int:
    """Build chunks.jsonl for one company and return the emitted chunk count."""

    company_dir = data_dir / company
    if not company_dir.exists():
        logger.warning("Company corpus directory is missing: %s", company_dir)
        return 0

    output_dir = index_dir / company
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.jsonl"

    chunk_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in tqdm(sorted(company_dir.rglob("*.md")), desc=f"Indexing {company}", unit="file"):
            document = parse_document(path, data_dir=data_dir)
            if document is None:
                continue
            for chunk in chunks_for_document(document):
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")
                chunk_count += 1
    return chunk_count


def chunks_for_document(document: ParsedDocument) -> list[CorpusChunk]:
    """Create stable chunk records for a parsed markdown document."""

    chunks: list[CorpusChunk] = []
    for chunk_idx, (heading, text) in enumerate(chunk_markdown(document.body)):
        doc_id = stable_doc_id(document.rel_path, chunk_idx)
        chunks.append(
            CorpusChunk(
                doc_id=doc_id,
                company=document.company,
                rel_path=document.rel_path,
                chunk_idx=chunk_idx,
                title=document.title,
                source_url=document.source_url,
                breadcrumbs=document.breadcrumbs,
                last_updated=document.last_updated,
                product_area_key=product_area_key_for_rel_path(document.rel_path),
                heading=heading,
                text=text,
                token_estimate=estimate_tokens(text),
            )
        )
    return chunks


def stable_doc_id(rel_path: str, chunk_idx: int) -> str:
    """Return a deterministic 12-character chunk id."""

    return hashlib.sha1(f"{rel_path}#{chunk_idx}".encode("utf-8")).hexdigest()[:12]


def product_area_key_for_rel_path(rel_path: str) -> str:
    """Return the canonical product-area label for a corpus-relative path."""

    company, lookup_key = product_area_lookup_key(rel_path)
    label = PRODUCT_AREA_LABELS.get((company, lookup_key))
    if label is None:
        logger.warning("No product area label for corpus path: %s", rel_path)
        return ""
    return label


def product_area_lookup_key(rel_path: str) -> tuple[str, str]:
    """Return the hand-authored PRODUCT_AREA_LABELS lookup key for a path."""

    parts = Path(rel_path).as_posix().split("/")
    if len(parts) < 2:
        return rel_path, ""
    company = parts[0]
    remainder = parts[1:]

    if company == "hackerrank":
        return company, remainder[0]

    if company == "claude":
        if remainder[0] == "claude" and len(remainder) >= 3:
            return company, f"claude/{remainder[1]}"
        return company, remainder[0]

    if company == "visa":
        candidates: list[str] = []
        if remainder[-1].endswith(".md"):
            candidates.append("/".join(remainder))
            stem_remainder = [*remainder[:-1], remainder[-1].removesuffix(".md")]
            candidates.append("/".join(stem_remainder))
        for depth in range(len(remainder), 0, -1):
            candidates.append("/".join(remainder[:depth]))
        for candidate in candidates:
            if (company, candidate) in PRODUCT_AREA_LABELS:
                return company, candidate
        return company, "/".join(remainder)

    return company, remainder[0]


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> dict[str, int]:
    """Build all company chunk indexes and write the corpus manifest."""

    index_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for company in COMPANIES:
        counts[company] = build_company_index(company, data_dir=data_dir, index_dir=index_dir)

    manifest = {
        "corpus_hash": corpus_hash(data_dir),
        "companies": counts,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_chars": OVERLAP,
    }
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


def corpus_hash(data_dir: Path = DATA_DIR) -> str:
    """Hash sorted relative markdown paths and mtimes for change detection."""

    digest = hashlib.sha256()
    for path in sorted(data_dir.rglob("*.md")):
        rel_path = path.relative_to(data_dir).as_posix()
        stat = path.stat()
        digest.update(f"{rel_path}\t{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [(None, body)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, body[: matches[0].start()].strip()))

    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        heading = match.group(2).strip()
        sections.append((heading, body[match.start() : end].strip()))
    return sections


def _split_large_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    normalized = text.strip()
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        part = normalized[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)
    return parts


def _join_chunk_text(left: str, right: str) -> str:
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    return f"{left.strip()}\n\n{right.strip()}"


def _overlap_suffix(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    return text.strip()[-overlap_chars:].strip()


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_present(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _coerce_optional_str(metadata.get(key))
        if value:
            return value
    return None


def _extract_h1(body: str) -> str | None:
    match = H1_RE.search(body)
    if not match:
        return None
    return match.group(1).strip()
