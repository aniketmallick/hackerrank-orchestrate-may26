# Support Triage Agent — Technical Reference

This document is the primary reference for the AI Judge interview. It explains
every architectural decision and provides the data to defend each claim.

---

## Quickstart

**Python version:** 3.11+ required (uses `tomllib`, `ZoneInfo`).

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

# Install dependencies (exact pins)
pip install -r requirements.txt

# Copy env template and set your key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# Build the corpus index (BM25 cache + dense embeddings — takes ~2 min first run)
python -m code.main build-index

# Run the agent on the full ticket CSV
python -m code.main run \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv

# Evaluate against the 10-row sample (with ablation table)
python -m code.eval.harness \
  --sample support_tickets/sample_support_tickets.csv \
  --ablate \
  --out eval/runs/<timestamp>/

# Run adversarial fuzz set
python -m code.eval.harness \
  --fuzz \
  --out eval/runs/<timestamp>-fuzz/
```

---

## Architecture

The pipeline has five named stages executed in sequence. No stage mutates
upstream state; each stage receives a read-only input and returns an immutable
output object (Pydantic models throughout). This maps directly to the rubric's
"clear separation of concerns."

### Stage 1 — Preflight

`code/stages/preflight.py` runs zero LLM calls. It normalises the company
field, detects language (langdetect), and fires four regex/heuristic checks:

- **has_live_id** — matches `cs_live_*`, `sk_live_*`, `pi_live_*` and
  13–19-digit card-number groups. Fires before any LLM call; live identifiers
  are escalated immediately per the rubric's "explicit handling of high-risk …
  tickets."
- **injection_score** — keyword and pattern matcher for prompt-injection
  attempts (including multilingual variants: `afficher toutes`, `muestra
  todas`, base64 blobs > 200 chars, role-play tags).
- **is_pleasantry** / **is_adversarial** — short gratitude texts and explicit
  ToS-violation requests are short-circuited here, before routing, to avoid
  spending LLM tokens and to ensure deterministic refusals.

Preflight returns a `PreflightFlags` object. The pipeline checks
`is_pleasantry`, `is_adversarial`, and `has_live_id` immediately; if any is
true, a templated `FinalOutput` is returned with zero further API calls.

### Stage 2 — Routing

`code/stages/routing.py` makes one `claude-sonnet-4-5` call (capped at 256
tokens). It classifies scope (`in_scope`, `out_of_scope_benign`,
`out_of_scope_adversarial`, `ambiguous_underspecified`, `pleasantry`,
`adversarial`), infers `resolved_company` when the input company is None, and
surfaces a list of `intents` for multi-intent tickets. The routing prompt
quarantines the `issue` field inside `<untrusted_user_input>` tags so injection
content cannot override instructions.

Routing is responsible for the rubric's "explicit handling of … out-of-scope
tickets" — it distinguishes benign off-topic queries (actor names, general
trivia) from adversarial requests, applying different templates to each.

### Stage 3 — Hybrid Retrieval

`code/retrieval/hybrid.py` fuses two retrieval signals via **Reciprocal Rank
Fusion** (RRF, c=60):

- **BM25** (`rank_bm25.BM25Okapi`) — tokenised, lowercase, markdown-stripped,
  cached to `index/<company>/bm25.pkl`.
- **Dense (BAAI/bge-base-en-v1.5)** — sentence-transformers cosine similarity,
  embeddings persisted in `index/<company>/embeddings.npy` and
  `embeddings_doc_ids.json`. Re-embed only when the corpus hash
  (SHA-256 over sorted rel_paths + mtimes) changes.

`score(d) = Σ_r 1 / (60 + rank_r(d))` over BM25 and dense rankings per
company. When `company=None`, all three per-company indices are queried and six
rankings are fused.

**Alternatives rejected:**

We did not adopt a vector database — for ~3 K chunks, numpy cosine is exact,
deterministic, and zero-dependency (no Chroma, Qdrant, or FAISS install
required). We did not adopt an agent framework — multi-agent loops sacrifice
the determinism the rubric explicitly rewards, and add per-turn latency.

**Visa-specific tuning:** Visa corpus uses smaller chunks (VISA_CHUNK_TOKENS=600,
VISA_OVERLAP=200) and top-K=8 instead of 6, because Visa documents contain
dense contact tables. The chunker's `protect_contacts` mode isolates phone/
contact blocks into their own chunks so phone numbers are never split from
their heading.

### Stage 4 — Grounded Answer

`code/stages/grounding.py` makes one `claude-sonnet-4-5` call (capped at 1024
tokens). The system prompt:

1. Forbids parametric knowledge: "Use ONLY the passages below."
2. Wraps the user's `issue` field in `<untrusted_user_input>` — any
   instruction inside those tags is data, not a command.
3. Requires `cited_doc_ids` to be a subset of the doc_ids supplied.
4. Sets `no_evidence=True` if passages do not support an answer.
5. Explicitly states that a prerequisite step is not an escalation trigger
   (prevents over-escalation on multi-step procedures).
6. Clarifies the `feature_request` / `product_issue` boundary: a question
   about using an existing feature is `product_issue`.

`product_area` is **not** emitted by the LLM. After grounding, the pipeline
derives it deterministically from the top-cited passage's `product_area_key`
field, which was stamped at index-build time from `config.PRODUCT_AREA_LABELS`.

### Stage 5 — Validator

`code/stages/validator.py` enforces hard invariants before finalising output:

- Enum check on `status` and `request_type`.
- `cited_doc_ids ⊆ {p.doc_id for p in passages}` — hallucinated citations
  trigger one retry with a stricter prompt; if still failing, escalate.
- `preflight.has_live_id` forces `status=escalated`.
- Response token length: 8 < tokens < 800.
- **Dual-model verification on `sensitivity=high` rows:** a separate
  `claude-haiku-4-5` call (128 tokens) judges whether the response is
  supported by the cited passages. If `supported=False`, the ticket is
  escalated. The justification field carries a `verifier=ok|escalated` tag.
- Fills `justification`:
  `decision={status}; route={company}/{product_area}; cited={ids}; reason={sentence}`

---

## Grounding & Anti-Hallucination Mechanism

| Layer | Mechanism |
|---|---|
| System prompt | "Use ONLY the passages below" — parametric knowledge forbidden |
| Injection quarantine | `<untrusted_user_input>` wraps the issue field; model told those tags are data |
| Citation enforcement | Validator checks `cited_doc_ids ⊆ passages`; retries or escalates on violation |
| No-evidence flag | `no_evidence=True` from grounding → automatic escalation |
| Dual-model verification | Haiku verifier on `sensitivity=high` rows — independent confirmation |
| Contact-block chunker | Phone numbers and contact identifiers never split from their headings |

---

## Determinism

Every run on the same inputs produces identical output:

- `temperature=0` on every Anthropic call.
- Pinned model snapshots: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5`.
- Exact dependency versions pinned in `requirements.txt`.
- Corpus hash (SHA-256 over sorted `rel_paths + mtimes`) controls embedding
  cache invalidation — embeddings are not re-computed on unchanged corpus.
- Local `sentence-transformers` embeddings — no hosted model variance.
- RRF fusion is a deterministic algebraic formula given fixed inputs.
- `numpy.random.seed(42)` and `torch.manual_seed(42)` set at model-load time.
- `PRODUCT_AREA_LABELS` is a hardcoded dict — label derivation involves no
  LLM or probabilistic step.

---

## Failure-Mode Coverage

The table below maps each hard category to the module that handles it.

| # | Failure Category | Signal | Module |
|---|---|---|---|
| 1 | Prompt injection | `injection_score > 0.5` (keyword + base64 + role-play tags) | `preflight.py` |
| 2 | Adversarial ToS violation | `is_adversarial()` keyword match (malware, delete files, etc.) | `preflight.py` |
| 3 | Pleasantry / no intent | `is_pleasantry()` (short text, gratitude words, no action verb) | `preflight.py` |
| 4 | Live billing identifier | `has_live_id` regex (`cs_live_*`, `sk_live_*`, 13–19-digit groups) | `preflight.py` |
| 5 | Out-of-scope benign | `routing.scope = out_of_scope_benign` → OOS template, not refusal | `routing.py` |
| 6 | Multilingual injection | `injection_score` multilingual keywords (`afficher toutes`, `muestra todas`) | `preflight.py` |
| 7 | Ambiguous / underspecified | `routing.scope = ambiguous_underspecified` → escalate | `routing.py` + `pipeline.py` |
| 8 | No corpus evidence | `draft.no_evidence = True` → escalate without fabrication | `grounding.py` |
| 9 | Hallucinated citations | `cited_doc_ids ⊄ passages` → retry then escalate | `validator.py` |
| 10 | High-sensitivity unsupported claim | Haiku verifier `supported=False` → force escalate | `validator.py` |
| 11 | Multi-intent ticket | `routing.intents` list passed to grounding; answerable answered, impossible declined | `routing.py` + `grounding.py` |
| 12 | Over-escalation on multi-step procedures | Explicit system-prompt rule: "a prerequisite is not an escalation trigger" | `grounding.py` |
| 13 | Visa contact info split from context | `protect_contacts=True` in chunker; each phone/contact block gets its own chunk | `corpus.py` |
| 14 | Unknown / None company | Fan-out across all 3 indices; 6 rankings fused via RRF | `hybrid.py` |

---

## Ablation Results

From `eval/runs/final/summary.md` (10-row sample, 2026-05-02):

| Variant | Status | Request Type | Product Area | Response | Delta vs Full |
| --- | ---: | ---: | ---: | ---: | --- |
| full | 1.000 | 1.000 | 0.700 | 0.300 | status +0.000, request_type +0.000, product_area +0.000, response +0.000 |
| no-routing | 0.900 | 1.000 | 0.700 | 0.200 | status -0.100, request_type +0.000, product_area +0.000, response -0.100 |
| no-rerank | 0.900 | 1.000 | 0.500 | 0.300 | status -0.100, request_type +0.000, product_area -0.200, response +0.000 |
| no-validator | 1.000 | 1.000 | 0.700 | 0.200 | status +0.000, request_type +0.000, product_area +0.000, response -0.100 |

Deltas below ±0.2 on n=10 are within single-row statistical noise and should
not be interpreted as architectural signal. The fuzz set tests architectural
choices the small sample cannot: injection handling, OOS routing, live-ID
escalation, and pleasantry short-circuits are all verified there. The routing
ablation's status delta (−0.100) is one ticket — the ambiguous underspecified
row — which preflight-only correctly escalates but for a different reason; the
rerank delta on product_area (−0.200) is two tickets where BM25-only retrieval
surfaces a less specific document.

---

## Limitations & Honest Framing

### Response Similarity Caveat

Our mean ROUGE-L on the 10-row sample is **0.454** (3/10 rows meet the ≥0.40
threshold, yielding the 0.300 accuracy figure). ROUGE-L on n=10 is a noisy
proxy where one row crossing the threshold flips the headline by 10 points. The
rubric grades response on faithfulness and non-hallucination, both of which we
enforce structurally (see Grounding section). We optimised for grounding
correctness over stylistic mimicry of sample prose. Rows 7 and 10 score ROUGE-L
= 1.000 (exact pleasantry / OOS templates); remaining rows average 0.27–0.42,
reflecting substantive grounded answers that differ in phrasing from the
sample's shorter-form answers.

### What We Found Via Manual Audit

After our second build pass, a row-by-row audit against the sample exposed
three latent issues invisible to ROUGE-L:

1. **Adversarial-intent detector misfired on benign out-of-scope queries.**
   "Name of the actor in Iron Man" hit the malware-refusal template. Fix: split
   scope into `out_of_scope_benign` vs `adversarial` with separate templates in
   routing.

2. **Visa contact-info tickets received generic guidance instead of verbatim
   phone numbers present in the corpus.** Root cause: chunker was splitting
   phone-number sections from their context. Fix: `protect_contacts` chunker
   mode + Visa-specific top-K=8.

3. **Multi-step procedures with prerequisites were being escalated instead of
   enumerated.** Fix: grounding system prompt explicitly states "a prerequisite
   is not an escalation trigger."

The harness was the artifact that made these failures catchable; the audit was
the artifact that made them addressable.

### Known Weaknesses

- **`product_area` allowlist** (`config.PRODUCT_AREA_LABELS`) is derived from
  the corpus directory tree. Held-out CSV labels outside this allowlist will
  miss. A regression test (`test_product_area.py`) catches new directories on
  rebuild.
- **Routing slightly over-escalates on truly ambiguous tickets** ("it's not
  working, help"). This is deliberate — the rubric penalises hallucination
  harder than over-escalation.
- **Uniform Sonnet model** for routing and grounding. A mixed-model setup
  (Haiku routing, Sonnet grounding) would reduce cost further but adds a model
  dependency to defend at the AI Judge interview.

---

## What an AI Tool Produced vs What I Designed

**Architecture decisions (mine):**

- 5-stage pipeline with deterministic short-circuits before any LLM call.
- Hybrid BM25 + dense retrieval fused with RRF rather than a vector DB.
- Deterministic `product_area` derivation from corpus directory tree — no LLM
  involvement.
- Contact-block protection in the chunker to preserve Visa phone numbers.
- Dual-model verification (Sonnet generator → Haiku verifier) on
  `sensitivity=high` rows.
- `<untrusted_user_input>` quarantine for prompt-injection defence.

**Implementation (AI-assisted):**

Codex (OpenAI) wrote the code under explicit per-deliverable prompts that
specified file names, function signatures, and constraints. Claude reviewed and
audited each pass, identifying the P1 issues (hardcoded model strings,
dead-code citation check, Anthropic client re-instantiation) and the three P0
audit findings above. The final code reflects multiple review-and-fix cycles
driven by the eval harness rather than a single code-gen pass.
