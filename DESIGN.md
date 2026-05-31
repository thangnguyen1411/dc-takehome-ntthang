# System Design — Self-Improving Scientific Data Curation Agent

## 1. Problem

Scientific QA benchmarks (paper excerpt + question + candidate answer) are
generated faster than humans can audit them. Bad items take several forms: the
answer contradicts the source, hallucinates entities, reasons weakly, or makes
claims the evidence never supports. We need an **autonomous agent** that grades
each item against its evidence, classifies the failure, **repairs repairable
answers**, scores its own reliability, and emits a **ranked, annotated queue** —
so a human curator reviews the riskiest items first instead of the raw batch.

## 2. Design goals

- Modular stages with obvious data flow and typed boundaries.
- **Evidence-grounded** verdicts — judge only against supplied evidence, never the
  model's outside knowledge.
- A real iterative improvement loop, not a single pass.
- Deterministic, rule-based checks for things an LLM can't be trusted to police.
- Measurable quality against a held-out gold label.
- Runnable and **testable without network access** (the full suite runs keyless).

Everything below is implemented; §10 lists what is intentionally left as future
work.

## 3. Architecture

A linear pipeline with one embedded loop. Each stage is a small, single-purpose
module; state flows through typed pydantic models
(`Task → Verdict → Iteration → CurationResult`).

```
CSV ─▶ Ingest ─▶ Deterministic checks ─▶ Verify ─▶ [Refine loop] ─▶ Evaluate ─▶ Triage ─▶ JSON + report + queue
                       (rule-based)        (LLM)    reflect→regenerate→re-verify   (vs gold)   (rank)
```

**Why linear-with-loop, not a graph framework (LangGraph / AutoGen / CrewAI)?**
The control flow is genuinely linear with a single bounded loop. A graph runtime
would add a dependency and indirection without buying expressiveness at this
scale. Custom orchestration keeps the whole flow readable in `pipeline.py` +
`refiner.py`. If branching multiplied (parallel verifiers, tool routing) a graph
runtime would start to pay off — noted as future work.

`Pipeline.build(client, config)` is the composition root: it wires the
(optional retriever →) verifier → reflector → refiner → pipeline object graph so
callers don't have to.

### 3.1 Data model

State flows through a small set of immutable pydantic models (`models.py`); they
are the single source of truth for the I/O contract, and tool-forced LLM output is
validated into them so malformed responses are a hard error, not silent drift.

```text
Task                  # ingested input — one QA item to curate
  task_id: str
  question: str
  reference_context: str          # given, AUTHORITATIVE evidence (CSV column)
  relevant_knowledge: str = ""    # retrieved evidence (empty unless RAG ran)
  candidate_answer: str
  paper_title / domain: str = ""
  ground_truth_signal: str | None # hidden gold label — never shown to the LLM
  evidence_block() -> str         # renders PRIMARY/SUPPLEMENTARY sections for prompts

RubricScores          # orthogonal 0–1 axes, independent of the single verdict
  faithfulness / completeness / specificity: float

Verdict               # one verification result (validated: verdict ∈ vocabulary)
  verdict: str                    # supported|contradicted|hallucinated|weak_reasoning|unsupported
  confidence: float
  question_quality_score: float
  rubric: RubricScores
  detected_issues: list[str]
  rationale: str

ReflectionPlan        # the fix plan produced before a regeneration
  diagnosis / what_to_change / what_to_keep: str

Iteration             # one pass of the refine loop
  step: int
  answer: str
  verdict: Verdict
  refined: bool
  reflection: ReflectionPlan | None   # None for the initial (step 0) pass

RetrievedSnippet      # a traceable piece of retrieved evidence
  text: str; score: float; source: str   # e.g. "corpus#3" / "pubmed:35392976"

CurationResult        # final per-task output (machine-readable; superset of brief schema)
  task_id, question_quality_score, answer_confidence
  detected_issues: list[str]
  agent_reasoning_summary: str
  refinement_applied: bool
  final_answer: str
  evidence: list[str]
  final_verdict: str
  rubric_scores: RubricScores
  retrieved_evidence: list[RetrievedSnippet]
  iterations: list[Iteration]              # the full reflect→regenerate→verify trace
  ground_truth_signal / verdict_matches_ground_truth   # filled only by the eval harness
```

Three details encode design decisions: `reference_context` and `relevant_knowledge`
are **kept separate** so retrieval never overwrites given evidence (precedence is
applied at prompt time by `evidence_block()`); `ground_truth_signal` lives on the
input but is structurally walled off from every prompt; and `iterations` preserves
the *entire* reasoning trace, so any verdict is auditable back to the evidence
tokens that produced it.

## 4. Stages

1. **Ingest** (`ingest.py`) — parse the CSV, enforce required columns, produce
   `Task` objects. `ground_truth_signal` is loaded but **never passed to the LLM**
   — held out for evaluation only.

2. **Deterministic verifiers** (`checks.py`) — rule-based, no LLM:
   empty question/answer/evidence (`invalid_schema`, `missing_evidence`),
   cross-task duplicate detection via normalized-question hashing
   (`duplicate_task`), and **hallucinated citations** — references in the answer
   (`[12]`, `(Smith et al., 2020)`, DOIs, PMIDs) that do not appear in the
   evidence (`hallucinated_citation`). These are reproducible and unit-tested
   without a key; the LLM is never asked to police structure it can't reliably see.

3. **Verify** (`verifier.py`) — the core agent. A pinned system prompt defines the
   verdict taxonomy and forbids outside knowledge. Output is forced through a
   tool-use JSON schema and validated by pydantic, so malformed output is a hard
   error, not silent drift. Returns `verdict`, `confidence`,
   `question_quality_score`, three rubric axes, `detected_issues`, and a
   `rationale`. Three interchangeable verifier modes plug in behind one protocol —
   see §6.

4. **Refine** (`refiner.py` + `reflector.py`) — the self-improvement mechanism.
   The candidate answer is verified once, then a **critic gate** fires when the
   verdict is bad *or* confidence is below threshold (**confidence-based
   escalation**). Each iteration runs **reflect → regenerate → re-verify**:
   - **Reflect** (`Reflector`) plans the fix — a structured `ReflectionPlan`
     (diagnosis / what-to-change / what-to-keep), reasoning *about* the failure,
     kept separate from judging and from writing.
   - **Regenerate** executes that plan and is **history-aware and escalating**
     (*retry-with-improved-prompt*): each retry sees *every* prior failed attempt
     ("do not repeat these mistakes"), and the instruction firms up per pass —
     tier 1 "fix it", tier 2 "quote the evidence verbatim", tier 3 "be maximally
     conservative".
   - **Re-verify** judges the new answer; the loop repeats until `supported` ∧
     confident, or the iteration cap (default 3, so the escalation tiers are
     reachable). Every pass — including its reflection plan — is recorded as an
     `Iteration`, so the full reasoning trace is preserved.

   An answer that genuinely cannot be supported by its evidence exits
   **still-flagged** rather than being forced to `supported` — the trace makes
   that honest failure visible.

5. **Evaluate** (`evaluate.py`) — scores the agent's verdict on the *original*
   answer (iteration 0) against the held-out label, builds a gold→predicted
   confusion table, and reports **confidence calibration** (mean confidence when
   right vs wrong; an empty bucket renders `n/a` rather than a misleading `0.00`,
   with sample counts). Verdicts are scored on the original answer because
   refinement intentionally changes the final verdict — scoring the post-fix
   verdict would mask detection quality.

6. **Triage** (`triage.py`) — ranks every result into a worst-first review queue by
   a transparent `priority_score`, emitted as `review_queue.md` + `review_queue.csv`
   with the top items embedded in the report. See §7.

### 4.1 Refine loop — sequence

The self-improvement loop is the one piece of non-linear control flow. `Refiner`
verifies once, then a critic gate decides whether to enter the reflect → regenerate
→ re-verify cycle, bounded by `max_refine_iterations` (default 3). The verifier may
be single / panel / debate — the loop is agnostic.

```
Pipeline        Refiner            Verifier         Reflector
   │               │                  │                │
   │  refine(task) │                  │                │
   │ ─────────────▶│                  │                │
   │               │  verify(answer)  │                │   ← step 0: judge the
   │               │ ────────────────▶│                │     ORIGINAL answer
   │               │   Verdict v0     │                │
   │               │ ◀────────────────│                │
   │               │                  │                │
   │        ┌──────┴── gate: needs_refinement(v0)? ────────────┐
   │        │  v0 is bad  OR  confidence < threshold            │
   │        │  (else: stop — record v0, return)                 │
   │        └──────┬────────────────────────────────────────────┘
   │               │                                   │
   │      ╔════════╪═══ loop while needs_refinement & step < cap ═══╗
   │      ║        │  reflect(task, history)           │            ║
   │      ║        │ ─────────────────────────────────▶│            ║
   │      ║        │       ReflectionPlan (diagnosis,   │            ║
   │      ║        │ ◀──── what_to_change, what_to_keep)│            ║
   │      ║        │                                    │            ║
   │      ║        │  regenerate(task, history, plan, tier)         ║
   │      ║        │  (escalating prompt; sees ALL prior failures)  ║
   │      ║        │  ───────────────────────▶ [LLM] ── new answer  ║
   │      ║        │                                                ║
   │      ║        │  verify(new answer) ────▶ Verifier ── Verdict  ║
   │      ║        │  append Iteration(step, answer, verdict, plan) ║
   │      ╚════════╪════════════════════════════════════════════════╝
   │               │
   │   iterations[]│  (full trace: every answer + verdict + plan)
   │ ◀─────────────│
```

Key invariants: the gate reads **v0's** verdict for the scored result (eval uses
iteration 0); each loop pass adds exactly **one** re-verification (no double
counting); regeneration is **history-aware** (every prior failed attempt is in the
prompt) and **escalating** (tier 1 "fix it" → tier 2 "quote verbatim" → tier 3 "be
maximally conservative"); and the loop exits the moment the answer is `supported` ∧
confident *or* the cap is hit — an unfixable answer leaves still-flagged.

### 4.2 Worked example — one task through every stage

A real run (`outputs/rag_demo`, task `ret_001`, single verifier, RAG over the
local corpus) shows the loop earning its keep on a deliberately wrong answer:

**Input** — a question with *no given evidence*, so retrieval supplies it:
- *Question:* "What is the mechanism of statins, and how does LDL reduction relate
  to major vascular event risk?"
- *Candidate answer:* "Statins **activate** HMG-CoA reductase to **accelerate**
  cholesterol synthesis, and each 1 mmol/L **rise** in LDL lowers major vascular
  events by ~21–22%." — directionally inverted on both counts.

**Retrieve** — `KeywordRetriever` pulls the matching corpus snippet (score above
threshold) into `relevant_knowledge`: *"Statins inhibit HMG-CoA reductase … roughly
a 21–22% reduction in major vascular events per 1 mmol/L reduction in LDL."*

**Verify (step 0)** → `contradicted`, confidence **0.99**.
Rationale: *"evidence states statins 'inhibit' … the answer claims they 'activate'
… it also inverts the LDL relationship."* The critic gate fires (bad verdict).

**Reflect** → `ReflectionPlan.diagnosis`: *"contradicts the evidence on two core
facts — (1) mechanism: inhibit, not activate; (2) the LDL relationship is a
reduction, not a rise,"* with a concrete `what_to_change`.

**Regenerate (step 1)** → executes the plan using only the evidence: *"Statins
inhibit HMG-CoA reductase … each 1 mmol/L reduction in LDL is associated with
roughly a 21–22% reduction in major vascular events."*

**Re-verify** → `supported`, confidence **0.98**. Gate is satisfied; loop exits
after one refinement pass.

**Evaluate** — scores the *initial* verdict (`contradicted`) against gold
(`contradicted`) → ✓ match. (Scoring the final `supported` would have hidden that
the agent correctly caught the original flaw.)

**Triage** — `final_verdict = supported`, no open issues, high confidence →
`priority_score` near 0 → sinks to the bottom of the review queue. Had it exited
still-`contradicted`, it would have risen to the top instead.

The full arc — both answers, the verdict at each step, and the reflection plan — is
preserved in `CurationResult.iterations`, so a curator can audit exactly how the
fix was reached.

## 5. Data-quality thinking

The taxonomy is chosen to be *actionable*, each mapping to a distinct remediation:

| Issue | Signal | Remediation |
|---|---|---|
| `contradicted` | answer opposes evidence | regenerate from evidence |
| `hallucinated` | entities/relations absent from evidence | regenerate, strip invented claims |
| `weak_reasoning` | on-topic but misattributes the central finding | regenerate emphasizing the main result |
| `unsupported` | claim not entailed by context | flag for human / down-rank |
| `weak_question` | ambiguous/vague/unanswerable question | flag question for rewrite |
| `low_confidence` | agent unsure after refinement | route to human review |
| `low_faithfulness` / `incomplete` / `overclaim` | a rubric axis below threshold | targeted fix per axis |
| `invalid_schema`, `duplicate_task`, `hallucinated_citation` | structural | deterministic, pre-/post-LLM |
| `panel_disagreement` / `debate_unresolved` | judges split / unresolved | strong route-to-human signal |

### Multi-axis rubric

Alongside the single verdict, the verifier scores three **orthogonal** axes
(0.0–1.0): **faithfulness** (claims grounded in, not contradicting, the evidence),
**completeness** (captures the central finding vs a minor point), and
**specificity** (precise without over-claiming). These decompose *why* an answer
is weak ("faithful but incomplete") instead of forcing one mutually-exclusive
category, and each axis below `rubric_threshold` (0.6) emits its own actionable tag
(`low_faithfulness` / `incomplete` / `overclaim`). The single verdict is retained
for held-out scoring and the refine gate.

### Evidence precedence (RAG)

When retrieval augments a task, the given `reference_context` is rendered as
**PRIMARY EVIDENCE (authoritative)** and retrieved snippets as **SUPPLEMENTARY
RETRIEVED CONTEXT (must not override the primary)**. This rule is embedded in the
verify / reflect / regenerate system prompts so retrieval can *add* context
without ever overriding the source of truth. A snippet is only used if its
relevance score clears `retrieval_threshold` (0.25); a task that had no given
evidence and gets no qualifying snippet is tagged `retrieval_failed` so the
verdict stays honest.

## 6. Verification modes

Three implementations satisfy one `AnswerVerifier` protocol
(`verify(task, answer) -> Verdict`), so the refine loop is agnostic to which is
used. Mode is chosen at the CLI; the rest of the pipeline is unchanged.

- **Single** (`Verifier`) — one tool-forced call. Sampling **temperature defaults
  to 0** (configurable), so verification is reproducible: the same input yields the
  same verdict, which matters for an audit tool whose results must be comparable
  across runs.
- **Panel** (`PanelVerifier`) — N judges (any mix of providers) verify
  **independently**, then majority-vote. Independent judges reduce shared blind
  spots: a single self-judging model can rubber-stamp its own style of error,
  whereas a cross-model panel must agree. Disagreement is *tallied* and tagged
  `panel_disagreement`.
- **Debate** (`DebateVerifier`) — N judges verify, then over up to
  `debate_rounds` rounds each judge sees the others' rationales and may revise,
  stopping early on consensus. Debate tries to *resolve* disagreement through
  argument rather than merely tally it; it tags `verdict_changed_in_debate` (the
  majority moved off its opening position) and `debate_unresolved` (no consensus —
  the strongest route-to-human signal). Needs ≥ 2 judges.

All three share one combine path: majority vote with a **conservative tie-break**
(prefer flagging a problem over `supported`, since a false flag is cheaper than a
missed bad answer in curation; among tied flagged labels, take the one its voters
were most confident about), confidence averaged over the winning voters, rubric
axes averaged across all judges, and detected issues unioned.

**Cost.** Each `verify()` call is one debate/panel; the refiner calls it once per
pass. Worst case ≈ `(1 + refine_passes) × n_judges × (1 + debate_rounds)` LLM
calls per task, bounded by early-exit on consensus and on refinement. Panel and
debate are therefore opt-in for high-stakes batches.

## 7. Triage — the ranked review queue

The deliverable is a ranked queue, so `triage.py` folds every risk signal the
pipeline already produced into one transparent `priority_score`:

```
priority_score = verdict_risk + Σ(detected-issue risk) + low-confidence penalty
```

- **verdict_risk** — `error` 100, `contradicted`/`hallucinated` 30, `unsupported`
  20, `weak_reasoning` 15, `supported` 0 (severity order: false > fabricated >
  ungrounded > weak > fine).
- **issue_risk** — judge disagreement ranks highest (the strongest "a human must
  decide" signal): `debate_unresolved` 40, `panel_disagreement` 35,
  `hallucinated_citation` 30, … down to `weak_question` 10. Unknown tags score 0.
- **confidence penalty** — `round((1 − confidence) × 20)`.

Items bucket into **HIGH ≥ 40 / MEDIUM ≥ 15 / LOW**, carry a plain-English reason
per tag, and sort worst-first (ties → lower confidence → task_id, stable). It is
pure post-processing — no LLM, no gold labels — so the queue is meaningful on
unlabeled data too. The weights and cutoffs are deliberate, transparent
placeholders: the *ordering* is grounded in the verdict severity and the
disagreement-routes-to-human principle; the exact integers are best calibrated
against real human triage decisions (future work).

## 8. Engineering

- **Typed throughout.** pydantic models are the single source of truth for the
  output contract; tool-forced structured output + validation means malformed LLM
  output is a hard error, not silent drift.
- **Pluggable by `typing.Protocol`.** `LLMClient` (Anthropic/OpenAI),
  `AnswerVerifier` (single/panel/debate), and `Retriever` (keyword/PubMed/
  composite) are structural protocols — real clients and test doubles are
  interchangeable, which is what lets the full pipeline (including the loop) run in
  CI with **no API key** (110 tests, fully offline).
- **Prompt caching** on the static system prompt + tool schema (identical across
  tasks) — N full reads become 1 + (N−1) cache hits.
- **Temperature 0** by default for reproducible verification; configurable.
- **Per-task error isolation** — one task's failure is recorded as a
  `processing_error` result and the batch continues.
- **Centralized config** (`config.py`) — thresholds, models, iteration cap,
  temperature, retrieval settings in one place.

### 8.1 Failure modes & resilience

The system is built to **degrade, not crash** — a single bad item or a flaky
dependency must never sink the batch.

| Failure | Where | Handling |
|---|---|---|
| LLM API error / timeout (one task) | `pipeline.run` | caught per task → emitted as a `processing_error` `CurationResult`; the batch continues |
| Malformed / off-schema LLM output | `llm.py` + pydantic | tool-forced schema + validation reject it as a hard error rather than letting bad data flow downstream |
| Model returns a partial reflection plan | `ReflectionPlan` | fields default to `""`; the regenerator falls back to the failure rationale instead of crashing |
| PubMed / network down or rate-limited | `PubMedRetriever` | any exception degrades to an empty result; the task proceeds on given evidence |
| Retrieval finds nothing relevant | `pipeline` | evidence-less task tagged `retrieval_failed`, kept honest rather than answered from outside knowledge |
| Empty / structurally invalid task | `checks.py` | flagged deterministically (`invalid_schema` / `missing_evidence`) before any LLM call |
| Unfixable answer (evidence can't support it) | refine loop | exits still-flagged after the iteration cap; the trace shows why, no forced `supported` |
| Missing API key | `build_client` / `make_client` | fails fast with a clear message naming the env var to set |
| Top-level API/setup error | `cli.py` | reported cleanly (no stack dump) with a hint to check the key/credits |

The recurring principle: **boundaries contain failure.** Deterministic checks run
before the LLM; the LLM boundary validates structure; the retriever and per-task
loop swallow their own faults into typed, visible signals (`processing_error`,
`retrieval_failed`) rather than exceptions — so the run always produces a complete,
auditable output set.

## 9. Results & error analysis

Run on the 10-row benchmark with `claude-sonnet-4-6` (held-out `ground_truth_signal`):

- **Exact 4-way category accuracy: ~50%.** Harsh, because the categories overlap
  (a reversed claim is *both* a contradiction and a fabrication).
- **Binary supported-vs-flagged accuracy: ~70%.** The more meaningful number for
  "did the agent catch a problem at all."
- **Calibration holds**: mean confidence is higher when correct than when wrong.

The disagreements are instructive, and most are *not agent errors* — the agent is
**stricter than the benchmark's own labels**, surfacing genuine over-claims (e.g.
flagging a "strongest"/"most predictive" superlative the evidence calls merely
"significant"). Two takeaways drive the design: (1) the agent surfacing
over-claims is exactly the "flag unsupported claims" capability we want; (2) the
contradicted/hallucinated/weak boundary is inherently fuzzy, so binary
supported-vs-flagged is the fairer headline and category labels are best treated
as soft.

**Multi-judge runs are stricter still.** A cross-model panel (or debate) flags
mild over-claims one Opus pass would let through, which *lowers* accuracy against
the gold labels but is the desired curation behaviour — and the
`panel_disagreement` / `debate_unresolved` tags route exactly those contested
items to a human. The trade-off is cost (N× to N×(1+rounds) the verify spend) and
a higher false-flag rate, which is why multi-judge modes are opt-in.

> Important caveat on "accuracy": when the agent's verdict disagrees with gold, the
> metric counts it wrong by definition, but the gold label is a *reference*, not
> ground truth — several disagreements are the agent correctly catching a lenient
> or mislabeled benchmark item. The disagreement itself is the signal worth
> surfacing, which is why those items rise in the triage queue.

## 10. Trade-offs & future work

**Trade-offs**
- **Bounded iterations** (default 3): caps cost/latency; a hard case may exit
  still-imperfect — the trace makes that visible rather than hiding it.
- **Self-verification**: by default the same model family judges and regenerates,
  risking shared blind spots — mitigable via `PanelVerifier`/`DebateVerifier`
  (cross-provider). Regeneration still uses the primary model, so a residual
  shared-style bias remains.
- **Keyword retrieval, not embeddings**: `KeywordRetriever` uses IDF-weighted term
  overlap rather than a vector store — chosen so retrieval runs offline in CI with
  no embedding service. A vector DB is a drop-in `Retriever` upgrade.

**Future work**
- **Vector-DB retrieval** — an embedding-based `Retriever` behind the same protocol.
- **Claim-level decomposition** — split multi-claim answers into atomic claims and
  verify each, for span-level feedback.
- **Self-consistency** — sample one model N times at temperature > 0 and reuse the
  existing `_aggregate`/`_majority_vote` (the temperature knob already exists).
- **Learned triage weights** — calibrate `priority_score` against human triage
  labels instead of hand-set constants.
- **Cost/latency accounting** and **async/batch parallelism** — stages are pure per
  task (only dedup is cross-task), so throughput is trivially parallelizable.
- **Graph runtime** — adopt LangGraph if branching/tool-routing grows past the
  current single bounded loop.

## 11. Repository map

```
src/curation_agent/
  ingest · checks · verifier · reflector · refiner · retriever ·
  evaluate · triage · generator · pipeline · cli · models · config · llm
tests/               offline suite (scripted LLM doubles, mocked HTTP) — 110 tests, no key
data/tasks.csv       10-row biomedical benchmark
data/corpus/         biomedical evidence snippets for retrieval
outputs/             per-task JSON, results.json, report.md, review_queue.{md,csv}
```
