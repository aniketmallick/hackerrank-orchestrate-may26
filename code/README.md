# Support Agent Scaffold

This directory contains the Python entry point and corpus indexer for the HackerRank Orchestrate support triage agent.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On macOS systems where `python` is not installed, use `python3` for the same commands.

Set `ANTHROPIC_API_KEY` in `.env` or in your shell environment. The current LLM wrapper is only a health-check stub and does not make model calls.

## Build The Corpus Index

```bash
python -m code.main build-index
```

This walks `data/hackerrank`, `data/claude`, and `data/visa`, parses markdown frontmatter where present, chunks the markdown body by H2/H3 sections, and writes:

- `index/<company>/chunks.jsonl`
- `index/manifest.json`

Text chunks are stored as JSONL; numpy `.npy` files will hold embedding vectors in a later prompt.

Chunk sizing uses a deterministic character heuristic, `ceil(chars / 4)`, with `CHUNK_TOKENS=900` and `OVERLAP=120` characters.

Planned commands are visible in `--help`, but only `build-index` is implemented in this scaffold.
