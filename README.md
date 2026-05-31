# Self-Improving AI Agent for Scientific Data Curation

A lightweight, autonomous pipeline that audits scientific QA data. Given
`(question, evidence, candidate answer)` triples, it **judges** whether each
answer is faithful to its evidence, **classifies** how it fails, **repairs** the
fixable answers through an iterative loop, scores its own reliability against
held-out labels, and emits a **ranked, annotated review queue** so a human curator
audits the riskiest items first instead of the raw batch.

Everything is **evidence-grounded** — an answer is judged only against its
supplied evidence, never the model's outside knowledge — which is the property
that makes it useful for curation rather than general QA.

---

## Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. add your key
cp .env.example .env          # then edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. run the 10-task biomedical benchmark
PYTHONPATH=src python -m curation_agent.cli --csv data/tasks.csv --out outputs
```

Outputs land in `outputs/`: per-task JSON, `results.json`, `report.md`,
`review_queue.md`, and `review_queue.csv`.

> Requires Python 3.11+ and an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` with
> `--provider openai`). The **test suite runs with no key** (see below).

---

## What it does, stage by stage

```
CSV ─▶ Ingest ─▶ Deterministic checks ─▶ Verify ─▶ [Refine loop] ─▶ Evaluate ─▶ Triage ─▶ JSON + report + queue
                       (rule-based)        (LLM)    reflect→regenerate→re-verify   (vs gold)   (rank)
```

1. **Ingest** (`ingest.py`) — parse the task CSV into typed `Task` objects. The
   gold label (`ground_truth_signal`) is loaded but **never shown to the model** —
   held out for evaluation only.
2. **Deterministic checks** (`checks.py`) — rule-based, no LLM: empty
   question/answer/evidence (`invalid_schema`, `missing_evidence`), cross-task
   duplicates (`duplicate_task`), and citations absent from the evidence
   (`hallucinated_citation`).
3. **Verify** (`verifier.py`) — the core judge. A tool-forced LLM call returns a
   structured `Verdict`: one label (`supported / contradicted / hallucinated /
   weak_reasoning / unsupported`), a confidence, a `question_quality_score`, three
   rubric axes (faithfulness / completeness / specificity), detected issues, and a
   rationale grounded in specific evidence tokens. Three interchangeable verifier
   modes (single / panel / debate) plug in behind one interface — see
   [Verification modes](#verification-modes) below.
4. **Refine** (`refiner.py` + `reflector.py`) — the self-improvement loop. When a
   verdict is bad *or* low-confidence, the agent **reflects** (diagnoses the
   failure), **regenerates** a corrected answer, and **re-verifies** — up to 3
   escalating rounds. Every pass is recorded, so the full reasoning trace is
   preserved.
5. **Evaluate** (`evaluate.py`) — scores the agent's *initial* verdict against the
   held-out gold label: exact + binary accuracy, a confusion table, and
   **confidence calibration** (is it more confident when right?).
6. **Triage** (`triage.py`) — ranks every task into a worst-first review queue by
   a transparent `priority_score`, so a curator works top-down.

Orchestration is plain Python (`pipeline.py`) — a linear flow with one embedded
loop. No graph framework: the control flow is genuinely linear with a single
bounded loop, and custom orchestration keeps it readable. See `DESIGN.md` for the
full rationale and trade-offs.

---

## Verification modes

The verify stage has three interchangeable implementations, all satisfying one
`AnswerVerifier` protocol (`verify(task, answer) -> Verdict`), so the refine loop
is agnostic to which is used. Pick a mode at the CLI; the rest of the pipeline is
unchanged.

| Mode | Enable with | What it does | Cost / task | Best for |
|---|---|---|---|---|
| **Single** | *(default)* | One model judges the answer once. | 1 call | fast, cheap baseline |
| **Panel** | `--panel a,b,c` | N judges (any mix of providers) judge **independently**, then majority-vote. | N calls | catching one model's blind spots |
| **Debate** | `--panel a,b,c --debate` | N judges judge, then **see each other's rationales and revise** over rounds until they agree or `--debate-rounds` is hit. | N × (1 + rounds) | high-stakes / contested items |

- **Single** (`Verifier`) — the workhorse. One tool-forced call, returns the
  `Verdict` directly. Sampling temperature is configurable and defaults to `0`, so
  verification is reproducible (same input → same verdict).
- **Panel** (`PanelVerifier`) — independent judges *vote* but never interact. A
  cross-provider panel (e.g. Anthropic + OpenAI) must agree, which surfaces
  over-claims a single self-judging model would rubber-stamp. Disagreement is
  *tallied* and tagged `panel_disagreement` — a strong "route to a human" signal.
  Confidence is the mean of the judges who chose the winning verdict; rubric axes
  are averaged across all judges; detected issues are unioned.
- **Debate** (`DebateVerifier`) — judges *deliberate*: each round, every judge is
  shown the others' latest verdict + rationale and may change its mind. Debate
  tries to **resolve** disagreement through argument rather than merely tally it —
  a judge with a more evidence-faithful reading can move the others. It stops early
  on consensus; it tags `verdict_changed_in_debate` if the majority shifts and
  `debate_unresolved` if the judges never converge (the strongest send-to-human
  signal). Needs a panel of ≥ 2.

All three share the same combine logic — majority vote with a **conservative
tie-break** (prefer flagging a problem over `supported`, since a false flag is
cheaper than a missed bad answer in curation), averaged rubric axes, and a union
of detected issues — so verdicts are comparable across modes. Whichever mode is
chosen, it drops into the refine loop unchanged: the loop just calls
`verify(task, answer)` and doesn't know or care how many models stand behind it.

---

## Detected issue categories

The agent classifies failures into actionable categories (LLM-judged + rule-based):

| Category | Source | Meaning |
|---|---|---|
| `contradicted` | verifier | answer opposes the evidence |
| `hallucinated` | verifier | introduces entities/relations absent from the evidence |
| `weak_reasoning` | verifier | on-topic but shallow / misattributes the main finding |
| `unsupported` | verifier | claim the evidence neither confirms nor denies |
| `weak_question` | verifier | vague / ambiguous / unanswerable question |
| `low_confidence` | threshold | agent still unsure after refinement |
| `low_faithfulness` / `incomplete` / `overclaim` | rubric | a rubric axis fell below threshold |
| `missing_evidence` / `invalid_schema` | deterministic | structural problems |
| `duplicate_task` | deterministic | normalized-question collision |
| `hallucinated_citation` | deterministic | cites a reference absent from the evidence |
| `panel_disagreement` / `debate_unresolved` | multi-judge | judges split / couldn't agree |
| `retrieval_failed` | retrieval | nothing relevant found for an evidence-less task |

---

## Structured output

Each task produces a machine-readable `CurationResult` (superset of the brief's
schema):

```json
{
  "task_id": "task_001",
  "question_quality_score": 0.82,
  "answer_confidence": 0.74,
  "detected_issues": ["unsupported", "missing_citation"],
  "agent_reasoning_summary": "...",
  "refinement_applied": true,
  "final_answer": "...",
  "evidence": ["..."],
  "final_verdict": "supported",
  "rubric_scores": {"faithfulness": 0.9, "completeness": 0.8, "specificity": 0.7},
  "retrieved_evidence": [{"text": "...", "score": 0.61, "source": "pubmed:35392976"}],
  "iterations": []
}
```

`iterations` holds the full reflect → regenerate → verify trace (one entry per
pass), so every refinement step is auditable.

---

## Key capabilities

| Capability | Flag / module | Notes |
|---|---|---|
| **Self-improvement loop** | `refiner.py` | reflect → regenerate → re-verify, history-aware & escalating |
| **Deterministic verifiers** | `checks.py` | rule-based, unit-tested, no LLM |
| **Rubric grading** | `verifier.py` | faithfulness / completeness / specificity |
| **Outcome-based evaluation** | `evaluate.py` | vs held-out gold; confusion + calibration |
| **Confidence calibration** | `evaluate.py` | mean confidence when right vs wrong |
| **Multi-agent voting** | `--panel a,b,c` | independent judges, majority vote |
| **Multi-agent debate** | `--debate` | judges see each other's rationales and revise |
| **RAG retrieval** | `--retrieve --corpus PATH` / `--pubmed` | local corpus + live PubMed, combinable |
| **Benchmark generation** | `python -m curation_agent.generate` | synthesize labeled tasks from evidence |
| **Ranked review queue** | `triage.py` | worst-first `priority_score`, md + csv |

### Example commands

```bash
# Cross-model panel (needs OPENAI_API_KEY too)
PYTHONPATH=src python -m curation_agent.cli --csv data/tasks.csv \
  --panel "anthropic:claude-opus-4-8,openai:gpt-4o"

# Multi-agent debate
PYTHONPATH=src python -m curation_agent.cli --csv data/tasks.csv \
  --panel "anthropic:claude-opus-4-8,anthropic:claude-sonnet-4-6" --debate

# RAG: local corpus + live PubMed
PYTHONPATH=src python -m curation_agent.cli --csv data/pubmed_demo_tasks.csv \
  --retrieve --corpus data/corpus --pubmed

# Generate a fresh labeled benchmark from the corpus
PYTHONPATH=src python -m curation_agent.generate --corpus data/corpus \
  --per-evidence 1 --out data/generated_tasks.csv

# Trace every LLM request/response (debugging)
PYTHONPATH=src python -m curation_agent.cli --csv data/tasks.csv --verbose --llm-log
```

Useful flags: `--out DIR`, `--provider {anthropic,openai}`, `--model ID`,
`--debate-rounds N`, `--results-file`, `--report-file`, `--queue-file`.

---

## Tests

The full suite is **offline** — scripted LLM doubles, mocked HTTP, fake
retrievers — so it runs in CI with **no API key**:

```bash
PYTHONPATH=src python -m pytest -q     # 110 tests
```

---

## Project layout

```
src/curation_agent/
  ingest.py      load + validate the task CSV
  checks.py      deterministic, rule-based verifiers
  verifier.py    LLM judge: Verifier / PanelVerifier / DebateVerifier
  reflector.py   diagnoses a failure → a structured fix plan
  refiner.py     the reflect → regenerate → re-verify loop
  retriever.py   KeywordRetriever / PubMedRetriever / CompositeRetriever (RAG)
  evaluate.py    scoring vs held-out gold: accuracy, confusion, calibration
  triage.py      ranked review queue (priority_score)
  generator.py   benchmark generation from evidence
  pipeline.py    composition root + orchestration
  cli.py         command-line entry point
  models.py      pydantic data contracts
  config.py      thresholds, models, flags
  llm.py         Anthropic/OpenAI clients (tool-forced structured output)
data/
  tasks.csv               10-task biomedical benchmark (tab-separated)
  pubmed_demo_tasks.csv   2-task RAG demo
  corpus/                 biomedical evidence snippets for retrieval
tests/                    offline test suite (no API key)
DESIGN.md                 architecture, trade-offs, results & error analysis
```

---

## Design notes

- **Typed boundaries** — pydantic models are the single source of truth for the
  output contract; malformed LLM output is a hard error, not silent drift.
- **Pluggable by Protocol** — `LLMClient`, `AnswerVerifier`, and `Retriever` are
  `typing.Protocol`s, so real clients and test doubles are interchangeable and the
  full pipeline (including the loop) runs keyless in CI.
- **Evidence precedence** — when retrieval augments a task, given evidence is
  marked PRIMARY (authoritative) and retrieved context SUPPLEMENTARY, so retrieval
  never overrides the source of truth.
- **Honest failure** — an answer that can't be supported by its evidence exits
  still-flagged rather than being forced to `supported`; the trace makes that
  visible.

See **`DESIGN.md`** for the architecture rationale, results, error analysis, and
scaling/trade-off discussion.
```
