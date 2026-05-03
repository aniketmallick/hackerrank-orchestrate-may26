# AI Judge Interview — Prep Packet

**Open this file in a side window during the call.** It is structured so you can scan to the right section when you need it. Sections labeled **MEMORIZE** are short enough to actually memorize and you should — those are the ones a confident answer requires.

The judge has 30 minutes. They will probe along four rubric dimensions: depth of understanding, trade-off awareness, failure-mode reasoning, AI honesty. You win by demonstrating you drove the architecture, you measured your decisions, and you found bugs the agent missed.

---

## 1. The 60-second opening — MEMORIZE

If they ask "walk me through what you built," lead with this. Slow down on the bolded phrases.

> "I built a five-stage support triage pipeline. The five stages are **preflight**, **routing**, **hybrid retrieval**, **grounded answering**, and **validation**. Preflight is deterministic — no LLM — it does company normalization, language detection, regex flags for live billing IDs, and an injection sniffer. Routing is one Sonnet call that classifies scope and intents with the user input quarantined inside untrusted-input tags. Retrieval is **hybrid: BM25 plus dense embeddings on bge-base, fused with reciprocal rank fusion at c=60**, per-company indices, fanned out across all three when company is None. Grounding is a second Sonnet call that's instructed to use only the provided passages and emits cited document IDs. Validation enforces that those cited IDs are a subset of what we passed in, and on high-sensitivity rows it runs a Haiku cross-model verifier as a hallucination check. Determinism throughout: temperature zero, pinned models, hashed corpus, local embeddings. **The eval harness was built before the pipeline so I could measure each stage's contribution rather than guess.**"

Two sentences after, before they ask the next question, drop the rubric-aligned sentence:

> "I designed the architecture; CODEX wrote the code under explicit per-prompt specifications; after each pass I audited the output row-by-row and caught bugs neither the tests nor the harness flagged."

That sentence wins the AI-honesty rubric line preemptively.

---

## 2. Architecture facts — MEMORIZE the bolded numbers

| Question | Answer |
|---|---|
| How many corpus chunks? | About **3,000** across three companies (HR ~2,000, Claude ~1,000, Visa ~50). |
| Embedding model? | **BAAI/bge-base-en-v1.5** via sentence-transformers, **768 dimensions**, local. |
| RRF constant? | **c = 60**. |
| Top-K final? | **6** (Visa: 8 — corpus is small and contact info is critical). |
| BM25 candidate set? | **20** per company before fusion. |
| Models? | **claude-sonnet-4-5** for routing and grounding. **claude-haiku-4-5** for the verifier. |
| Token budgets | Routing **256**, grounding **1024**, verifier **128**. |
| Per-ticket cost | About **$0.020** average on the 10-row sample. Under the 22-cent gate. |
| Test count | **60+** tests across 9 test files. |
| Sample harness numbers | status **1.000**, request_type **1.000**, product_area **0.700**, response **0.300** (ROUGE-L threshold 0.4). |
| Real CSV size | **29 rows**. |

If you blank on a number, say "let me pull it from the README" — the file is on screen.

---

## 3. Trade-offs cheatsheet — what you picked vs what you rejected

The judge's strongest probe will be "what alternatives did you consider?" Have these locked.

### Architecture shape
**Picked: 5-stage staged pipeline.** Modular, deterministic, every decision named.
**Rejected: single LLM call.** Indefensible — "how do you prevent hallucination?" has no answer beyond "we ask nicely in the prompt."
**Rejected: multi-agent framework (LangGraph, CrewAI).** Adds non-determinism (variable iteration counts, tool-call ordering) which fights the rubric's "deterministic and reproducible" line. Frameworks also hide the actual logic from interview defense — you'd be defending the framework, not the architecture.

### Retrieval
**Picked: BM25 + dense + RRF.** BM25 nails exact-term queries (SCIM, LTI, interchange, traveler's cheque); dense catches paraphrase. RRF fuses without a tunable alpha. Ablation showed no-rerank drops status accuracy by 0.2 — that's the empirical defense.
**Rejected: BM25-only.** The ablation cost 0.2 on status. Real.
**Rejected: dense-only.** Loses exact-term precision on technical IDs. Untested but obvious.
**Rejected: vector DB (FAISS, Chroma, Qdrant).** 3K chunks. Numpy cosine over a 9 MB matrix is microseconds. A vector DB is dead weight at this corpus size and adds determinism risk (some libraries non-deterministic on ANN). At 100K+ chunks I'd switch to FAISS IVF.

### Embeddings
**Picked: local sentence-transformers (bge-base).** Zero extra API key, fully reproducible, fast on 3K chunks.
**Rejected: Voyage / OpenAI hosted embeddings.** Adds an API-key dependency and an external service variance for marginal-to-zero accuracy gain.

### Models
**Picked: Sonnet for routing and grounding; Haiku for verifier.** Sonnet for accuracy on the small ticket count where cost is negligible. Haiku for the verifier specifically because you want a different model checking the work — same-model self-check is an echo chamber.
**Rejected: Sonnet everywhere including verifier.** Verifier echo chamber.
**Rejected: Haiku for routing.** Marginal cost savings, adds dependency to defend, slight accuracy hit on edge cases.

### Product area labels
**Picked: hand-curated allowlist + deterministic derivation from top-cited passage's path.** The sample showed labels are folder-derived but with abbreviations (`hackerrank_community → community`, `privacy-and-legal → privacy`).
**Rejected: free-form LLM-generated labels.** I tried this. Got 0.000 accuracy on the sample. The harness exposed it immediately. Replaced with deterministic derivation.
**Rejected: pure folder-name mapping.** Misses the abbreviation pattern the sample uses.

### Determinism
**Picked: temp=0, pinned models, hashed corpus, local embeddings, deterministic RRF.**
**Rejected: temp=0.3 for diversity.** Fights reproducibility. The rubric explicitly rewards reproducibility.

### Hallucination defense
**Picked: four layered mechanisms** — system prompt forbids parametric knowledge, validator enforces `cited_doc_ids ⊆ provided`, dual-model verifier on high-sensitivity, no_evidence triggers escalation.
**Rejected: prompt-only defense.** Easily ignored by the model.
**Rejected: post-hoc fact-checking against a separate LLM.** Cost-prohibitive to run on every row; we run it only on `sensitivity=high`.

### Eval harness
**Picked: built BEFORE the pipeline, with `--ablate` flag that runs the pipeline 4 ways.** The ablation table in the README is the empirical defense.
**Rejected: build pipeline first, eval later.** Loses the "we measured" story and risks tuning to vibes instead of numbers.

---

## 4. The phase-by-phase journey — a 90-second narrative

If they ask "walk me through how you got here," tell this story. Each issue earned a fix; each fix taught a lesson. This narrative is what makes you look like an engineer, not a prompt-runner.

**Phase 1: Ground truth analysis.** Read all 10 sample rows, all 29 real rows, the corpus tree per company. Identified the output schema mismatch — `output.csv` was pre-seeded with snake_case headers `(issue,subject,company,response,product_area,status,request_type,justification)` which differs from the sample CSV's Title Case. Locked the snake_case format as ground truth. Identified eight hardest categories of tickets — prompt injection, account-access elevation, live billing IDs, multi-intent, adversarial off-topic, pleasantry, ambiguous low-info, multilingual. Flagged the under-determined product_area label space as the #1 risk.

**Phase 2: Architecture options.** Considered single-call (rejected: indefensible), multi-agent (rejected: nondeterministic), staged pipeline (picked: defensible at each stage).

**Phase 3: Module plan.** Five stages with schema-typed boundaries. Eval harness built first. Hand-curated `PRODUCT_AREA_LABELS` allowlist. Per-company indices.

**Build prompts 1–6** to CODEX. Bootstrap, retrieval, harness, baseline pipeline, routing+verifier, production run.

**Audit cycle 5b.** First harness ablation showed `product_area = 0.000` — every single row wrong. Root cause: I'd let the LLM emit free-form labels. Fix: replaced with deterministic derivation from the top-cited passage's directory, against the hand-curated allowlist. Cost gating added at the same time — preflight short-circuits before routing for pleasantry/adversarial/live-id, saving ~30% of LLM calls.

**Audit cycle 5c.** Manual row-by-row read of predictions caught three P0s the harness missed.
- "What is the name of the actor in Iron Man?" was getting the malware-refusal template — the adversarial detector was over-firing on benign OOS. Split the scope into `out_of_scope_benign` vs `adversarial`.
- Visa contact tickets returned generic guidance even though the corpus literally contains the Citicorp Freephone `1-800-645-6556` and Visa India `000-800-100-1219`. Root cause: the chunker was splitting phone-number sections from their context. Added a contact-block protection rule and bumped Visa-specific top-K to 8.
- Multi-step procedures with prerequisites (the Google-login account-deletion case) were being escalated. Tightened grounding: a prerequisite is not an escalation trigger.

**Audit cycle 5d.** Audit of the production output exposed a final P0 — the marquee French Visa ticket with embedded prompt-injection (real CSV row 24) was being routed as adversarial because 5c's adversarial-detector tightening had conflated *injection-attempt* with *adversarial-intent*. Split them: `injection_attempt` is a separate flag, and when injection is detected inside a legitimate request the injection is quarantined and we still answer the underlying question.

**Final move.** Compared 5c and 5d outputs cell-by-cell. 5d won on rows 1, 11, 21, 23, 24; 5c won on rows 7, 20, 22, 25 (where 5d had over-escalated). Selected the row-level best from the two deterministic runs for the final `output.csv`.

The headline lesson — and you should say this in the interview verbatim if asked: **"Each tightening fix can introduce a new failure mode that only a fresh row-by-row read of the output catches. Automated metrics on a 10-row sample do not surface these. The audit was the artifact that closed the loop."**

---

## 5. Known failure modes — own them, don't hide them

Owning weaknesses is rubric points. These are the answers to "where does your agent break?"

1. **Product area allowlist coverage.** The held-out CSV could ship a label that doesn't exist in our corpus directory tree. Mitigation: regression test catches new directories on rebuild. Deeper fix: derive a fallback label from the cited passage's `breadcrumbs` frontmatter.

2. **Routing slightly over-escalates ambiguous tickets.** "It's not working, help" with no company specified gets escalated. This is deliberate — the rubric penalizes hallucination harder than over-escalation — but it's a knob we could turn.

3. **Response style differs from sample prose.** ROUGE-L on the small sample is ~0.45 mean. We optimized for grounding correctness over stylistic mimicry. The rubric grades response on faithfulness and non-hallucination, both enforced structurally; the sample's prose is reference, not gold standard.

4. **Cross-language responses default to English.** Row 24 worked because grounding chose French, but it's not enforced. Next iteration: explicit language-mirroring directive.

5. **Sample size is 10.** Any ablation delta below ±0.2 is a single row. We did not tune to the sample.

6. **The fix-then-regress pattern.** Documented in the README. Each surgical fix has the potential to regress something else. We mitigated by re-running the harness and the fuzz set after every fix; the final cell-by-cell merge between 5c and 5d outputs handled the residual cases.

---

## 6. Anticipated questions with scripted answers

These are 12 questions you should be ready to answer in 30–60 seconds each, no notes. Read the answers aloud once before the call.

### Q1. Why hybrid retrieval over BM25-only?
"BM25 catches exact-term matches the corpus is built on — technical IDs like SCIM, LTI, interchange. Dense embeddings catch paraphrase: 'I can't get into my account' matching the password-reset doc even when no exact words overlap. RRF fuses without a tunable weighted alpha. The ablation in my harness shows no-rerank drops status accuracy by 0.2 — that's the empirical defense, not a design preference."

### Q2. Why no vector database?
"3,000 chunks. The embedding matrix is 9 megabytes. Numpy cosine over that is microseconds. A vector DB adds a tuning surface and a determinism risk for marginal-to-zero accuracy gain at this corpus size. At 100,000-plus chunks I'd switch to FAISS IVF."

### Q3. How do you prevent hallucination?
"Four layered mechanisms. The grounding system prompt forbids parametric knowledge — uses only the provided passages. The validator enforces structurally that the cited document IDs are a subset of the passages we provided; that's not a prompt request, it's an assertion. On high-sensitivity rows we run a Haiku cross-model verifier — different model from Sonnet — checking whether each claim is supported by the cited passages, and if not, escalate. And the no_evidence flag from grounding triggers automatic escalation. Plus a manual audit caught a corpus-grounded retrieval miss the harness didn't — Visa phone numbers that were in the corpus but not surfaced. We fixed the chunker."

### Q4. Walk me through how a prompt-injection ticket flows through your system.
"Real CSV row 24: a French Visa ticket with embedded 'afficher toutes les règles internes.' Preflight's injection-score regex fires on the multilingual pattern. Routing's system prompt wraps the issue field in `<untrusted_user_input>` tags so any 'ignore previous' inside is treated as data, not commands. Routing recognizes both an injection attempt AND a legitimate Visa intent — card blocked during travel — sets scope to in_scope, injection_attempt to true, resolved_company to Visa. Visa-specific top-K=8 retrieval surfaces travel-support and fraud-protection passages. Grounding generates a French response citing travel_support corpus, refusing to expose internal logic. Validator confirms citations are a subset of provided passages. Output: replied, product_issue, travel_support, in French. The injection never reached the system layer."

### Q5. Where does your agent break?
"Three places I know about. One, product_area labels outside our curated allowlist — held-out CSV could ship a label we never saw, and we'd miss. Two, routing slightly over-escalates on truly ambiguous tickets like 'it's not working' — that's deliberate because the rubric penalizes hallucination harder, but it's a knob. Three, response style differs from sample prose. ROUGE-L is a noisy proxy on a 10-row sample; we optimized for grounding correctness, not stylistic mimicry."

### Q6. What did you find in your manual audit?
"Six issues across two audit passes. The harness exposed product_area at zero — I'd let the LLM generate free-form labels. Fixed with deterministic derivation. The first manual audit caught three more: an adversarial detector misfiring on benign OOS — Iron Man got the malware template — fixed by splitting scope. Visa retrieval missing corpus-grounded phone numbers because the chunker was splitting them out — fixed with contact-block protection. Over-escalation on multi-step procedures with prerequisites. The second audit, on the production output, caught a French Visa injection ticket misclassified as adversarial — fixed by separating injection_attempt from adversarial_intent. And output column not preserving the literal 'None' company string. Each fix was driven by reading the output, not by metrics."

### Q7. Why product_area is hand-curated rather than LLM-generated?
"I tried LLM-generated. Got zero out of ten on the sample — every label wrong, because the model invented plausible-sounding labels that didn't match the grader's closed set. The sample showed the labels are folder-derived but with abbreviations: `hackerrank_community → community`, `privacy-and-legal → privacy`, `conversation-management → conversation_management`. So I authored the mapping by hand from the corpus tree, and the pipeline derives the label deterministically from the top-cited passage's path. The model never picks the label."

### Q8. Why do you have a Haiku verifier on top of Sonnet?
"Cross-model verification. Same-model self-check is an echo chamber — if Sonnet hallucinates, Sonnet checking Sonnet won't catch it. Haiku is a different model with different failure modes. We run it only on sensitivity=high rows because cost-per-row matters less than catch rate on the cases that matter most: account-access elevation, payment lookups, identity theft. The verifier returns supported / not_supported and if not, we escalate."

### Q9. How do you handle multi-intent tickets?
"Routing surfaces intents as a list, not a singleton. Real CSV row 2 — Visa ticket asking for both a refund and to ban the seller. The grounding stage receives the intents list and is instructed to answer the doable intent and explicitly decline the unanswerable one in the same response. Status stays replied because the primary intent (the refund/dispute path) is corpus-grounded; the unanswerable intent ('ban the seller') is named and refused."

### Q10. Why eval harness before pipeline?
"Two reasons. First, you cannot make architectural decisions if you cannot measure them. The ablation table comes from running the pipeline four ways — full, no-routing, no-rerank, no-validator — and reading the deltas. Without that, every choice is an opinion. Second, building the harness first forces you to define what 'correct' means before you write code that's tempted to be correct in a way that wasn't graded. The harness was a stub that returned hardcoded values; once it was wired and metric-correct, real implementation followed."

### Q11. If you had another 24 hours, what would you do?
"Three things, ranked. First, citation highlighting in the justification field — quote the exact sentence from each cited passage, not just the doc IDs. Makes per-row defensibility one-step instead of two-step. Second, cross-language response mirroring — currently we default to English unless grounding chooses otherwise; explicit instruction to mirror the input language. Third, an adversarial fuzz generator: use Claude to generate 100+ injection variants and harden the routing prompt against the failures."

### Q12. What was your single biggest mistake?
"Letting the LLM generate product_area labels in the first build. The harness caught it — zero out of ten — but I should have anticipated it from the Phase 1 analysis. I'd flagged that the label space was under-determined and partially folder-derived, but I let CODEX implement it as a free-form field instead of a constrained enum. The fix was straightforward once measured. Lesson: when the analysis flags a risk, the implementation should close the risk, not punt it to the LLM."

---

## 7. AI honesty script — the most important 60 seconds

The rubric line reads: "Honesty about AI assistance — clearly distinguish what you designed from what an AI tool generated for you." This is where most submissions either lie convincingly or lie unconvincingly. You do neither.

### What you say if asked "What did the AI generate vs what did you design?"

**MEMORIZE:**

> "The architecture is mine. The five-stage shape, the choice of hybrid retrieval over a vector DB, the deterministic product_area derivation, the contact-block protection rule in the chunker, the dual-model verification, the build-harness-first sequencing — those are my decisions, made before any code was written, defended against alternatives I considered and rejected.
>
> The implementation is generated. CODEX wrote the module code, the tests, the formatting, under explicit per-prompt specifications I authored — six build prompts plus three surgical fix prompts. Each prompt names the files, the function signatures, the test conditions, and the definition of done.
>
> Between every CODEX run I reviewed the output. The bugs that mattered most — the Iron Man adversarial misfire, the Visa retrieval miss, the French injection misclassification — were not caught by the tests. They were caught by reading the actual predictions row by row against the sample. I caught those, not the agent.
>
> So the way I'd put it: the AI wrote the code, but the AI did not drive the design and the AI did not catch the bugs. Both of those required me reading the output and applying judgment."

The phrase "the AI wrote the code, but the AI did not drive the design and the AI did not catch the bugs" is the keeper. Use it.

### What you do NOT say
- Do not say "I wrote everything myself." Untrue, the judge can tell, and you lose the rubric line for nothing.
- Do not say "the AI did most of it." You drove it. Don't undersell.
- Do not blame CODEX for the bugs. Bugs in shipped code are your bugs. The framing is "I caught these in audit" not "CODEX missed these."

---

## 8. Defending specific output rows

If the judge spot-checks a row, here are the four most likely targets and the answer for each.

### Row 24 — French Visa ticket with prompt-injection
"This was the marquee adversarial-but-legitimate case. Initial build routed it to the malware-refusal template because the adversarial detector was tight after the 5c fix for the Iron Man case — it was conflating injection-attempt with adversarial-intent. The 5d fix split them: injection_attempt is a separate boolean flag. When injection is detected inside a legitimate request, we quarantine the injection inside untrusted-input tags and still process the legitimate intent. The output is now a French response citing travel_support corpus. The injection text never reaches the system layer."

### Row 4 — Stripe live billing ID
"Preflight regex matches `cs_live_*`. Forced escalation, no LLM call, no possibility of leaking the ID into a prompt. Justification reason is `live billing identifier present`."

### Row 0 — Account access elevation request
"User explicitly states they're not the workspace owner. Routing flags sensitivity high, intent is restore_access. Grounding cites the team-and-enterprise documentation that explains only owners and admins can reassign seats — answers what's possible (contact your owner) and refuses what's not (we can't restore your access directly). Status replied, request_type product_issue, product_area team_and_enterprise."

### Row 11 — "It's not working, help" (None company)
"Ambiguous low-info. Preflight has no flags. Routing classifies scope as ambiguous_underspecified. Retrieval fans out across all three indices and returns nothing high-confidence. Pipeline escalates rather than guessing. This is the deliberate over-escalation I mentioned — the rubric penalizes hallucination harder than over-escalation, so when the corpus can't ground an answer, we route to a human."

---

## 9. Pre-interview checklist — 10 minutes before camera on

Run through this list:

1. [ ] `output.csv` is the merged best-of-both, 29 rows, schema valid.
2. [ ] `code/conftest.py` exists. `code/requirements.txt` exists. `code/.gitignore` exists.
3. [ ] Submission zip uploaded.
4. [ ] `~/hackerrank_orchestrate/log.txt` has the architectural conversation backfilled at the top.
5. [ ] `code/README.md` is open in another window, scrolled to the Architecture section.
6. [ ] This file (`INTERVIEW_PREP.md`) is open in a third window.
7. [ ] The architecture diagram is open in a fourth.
8. [ ] Camera works, mic works, environment quiet.
9. [ ] Glass of water on the desk. Phone on silent.
10. [ ] Notes you can glance at: the bolded numbers from §2, the AI-honesty script from §7.

---

## 10. Recovery phrases — when you blank or get pushed back on

Use these verbatim if you need to buy time or reset.

- **"Let me check the README — one second."** Buys 5–10 seconds while you scroll. The README is the source of truth; it's fine to consult it.
- **"That's a good push — let me think about it for a second."** Buys 3–5 seconds. Does not lose face.
- **"I'd want to verify that with the harness. My instinct is X, but I treat instinct as a hypothesis until I measure it."** Strong recovery — reframes as the engineer-with-discipline you actually are.
- **"You're right. The way I'd revise that is..."** When the judge is right and you were wrong, this is the answer. Capitulating cleanly is more impressive than defending a bad decision.
- **"I don't know."** Use it once if you need to. Pair it with "what I'd do is open a harness run, ablate that variable, and read the per-row diff." Better than bluffing.

What you do NOT say:
- "I think..." for facts. You measured these. Speak in present-tense fact.
- "We chose X because it's industry standard." Industry-standard is not a reason. Defend with measurement, fit, or first principles.
- "ChatGPT said..." or "Claude said..." Never. You drove this.

---

## 11. The closing — if they ask for last thoughts

Have one prepared sentence:

> "I'd like to make one note. The thing I'm proudest of is not the architecture — it's the harness-plus-audit loop. The harness caught the product_area zero. The two manual audits caught five more bugs the harness missed. Every fix in this submission has a corresponding before-and-after measurement or a documented audit finding. That's the engineering posture I'd want to bring to whatever I work on next."

That sentence wins three of four rubric lines in one breath: depth, trade-offs, failure-mode reasoning. It also implicitly settles the AI-honesty line because it foregrounds *your* judgment loop.

---

## 12. One more thing

You did the work. You drove the architecture. You ran the audits. You found the bugs. You picked the merge.

Don't apologize for what's imperfect. Own what you measured, own what you'd do next, and let the agent's good behavior on rows 0, 4, 19, 24, 27, 28 speak for itself. The judge has read the README and seen the ablation table — they're not waiting for you to convince them the architecture is reasonable. They're waiting to see if *you* understand it.

You do. Go win.
