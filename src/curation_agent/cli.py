"""Command-line entry point.

    python -m curation_agent.cli --csv data/tasks.csv

Runs the full pipeline, writes per-task JSON + an aggregate results file, scores
against ground truth, and writes a markdown report. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .evaluate import Evaluator
from .ingest import Ingestor
from .llm import build_client
from .pipeline import Pipeline
from .triage import prioritize
from .triage import render_csv as render_queue_csv
from .triage import render_markdown as render_queue_markdown
from .triage import render_summary as render_queue_summary


def _print_verbose(curated) -> None:
    """Print the per-task reasoning trace: each refinement pass and, for a panel,
    each verifier's vote (recovered from the combined rationale)."""
    for it in curated.iterations:
        verdict = it.verdict
        label = "initial" if it.step == 0 else f"refine pass {it.step}"
        print(
            f"[curation]     {label}: {verdict.verdict} (conf {verdict.confidence:.2f})",
            flush=True,
        )
        # The reflection plan that led to this pass (retries only).
        if it.reflection is not None:
            print(f"[curation]       reflect: {it.reflection.diagnosis}", flush=True)
            print(f"[curation]       fix: {it.reflection.what_to_change}", flush=True)
        # A panel rationale is the per-verifier votes joined by ' | '; a single
        # verifier is one plain rationale. Either way, one line per part.
        for part in verdict.rationale.split(" | "):
            print(f"[curation]       {part}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scientific data curation agent")
    parser.add_argument("--csv", default="data/tasks.csv", help="path to the task CSV")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument(
        "--results-file", default="results.json",
        help="filename for the aggregate JSON results (in --out; default: results.json)",
    )
    parser.add_argument(
        "--report-file", default="report.md",
        help="filename for the markdown report (in --out; default: report.md)",
    )
    parser.add_argument(
        "--queue-file", default="review_queue.md",
        help="filename for the ranked human-review queue (in --out; default: review_queue.md)",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"], default="anthropic",
        help="model provider (default: anthropic)",
    )
    parser.add_argument("--model", default=None, help="override the provider's default model id")
    parser.add_argument(
        "--panel",
        default=None,
        help="comma-separated verifier panel of provider:model specs for cross-checking, "
        "e.g. 'anthropic:claude-opus-4-8,openai:gpt-4o' (default: single verifier). "
        "Each provider listed needs its own API key.",
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="make the --panel judges debate (see each other's rationales and revise "
        "over rounds) instead of voting independently; needs a panel of >= 2",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=2,
        help="max revision rounds when --debate is set (default 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print each refinement pass and every verifier's vote per task",
    )
    parser.add_argument(
        "--llm-log",
        action="store_true",
        help="log every raw LLM request and response, tagged by provider:model "
        "(very verbose; for debugging the model interactions)",
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="augment every task's evidence with retrieved snippets (RAG). "
        "Pick source(s) with --corpus and/or --pubmed; both can be combined.",
    )
    parser.add_argument(
        "--corpus", default="",
        help="retrieve from this local corpus (a .txt file or folder of .txt files). "
        "Omit to use no corpus.",
    )
    parser.add_argument(
        "--pubmed",
        action="store_true",
        help="retrieve live abstracts from PubMed (requires network access); "
        "combine with --corpus to use both",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    panel = tuple(spec.strip() for spec in args.panel.split(",") if spec.strip()) if args.panel else ()
    if args.debate and len(panel) < 2:
        print("[curation] --debate needs a --panel of at least two verifiers.", file=sys.stderr)
        return 1
    # A source is used only if selected: --corpus PATH enables the corpus,
    # --pubmed enables PubMed. Either or both.
    config = Config(
        provider=args.provider, model=args.model, verifier_panel=panel,
        debate=args.debate, debate_rounds=args.debate_rounds,
        llm_log=args.llm_log, retrieve=args.retrieve, corpus_path=args.corpus,
        use_corpus=bool(args.corpus), use_pubmed=args.pubmed,
    )
    if config.retrieve and not (config.use_corpus or config.use_pubmed):
        print("[curation] --retrieve needs a source: pass --corpus PATH and/or --pubmed.",
              file=sys.stderr)
        return 1
    try:
        client = build_client(config)
    except RuntimeError as exc:
        print(f"[curation] {exc}", file=sys.stderr)
        return 1
    primary = f"{config.provider}:{config.resolved_model}"
    verifier_backend = " + ".join(config.verifier_panel) if config.verifier_panel else primary
    if config.debate:
        verifier_backend += f" (debate, {config.debate_rounds} rounds)"
    elif config.verifier_panel:
        verifier_backend += " (panel vote)"
    print(f"[curation] verifier: {verifier_backend} | regeneration: {primary}")
    if config.retrieve:
        sources = ([f"corpus:{config.corpus_path}"] if config.use_corpus else []) \
            + (["pubmed"] if config.use_pubmed else [])
        print(f"[curation] retrieval: ON ({' + '.join(sources)})")

    tasks = Ingestor().load(args.csv)
    print(f"[curation] loaded {len(tasks)} tasks from {args.csv}")

    # Per-task progress logging. The pipeline stays I/O-free; the CLI owns output.
    timer = {"start": 0.0}

    def on_task_start(index: int, total: int, task) -> None:
        print("[curation] " + "-" * 60, flush=True)
        print(f"[curation] [{index}/{total}] {task.task_id}: verifying...", flush=True)
        timer["start"] = time.monotonic()

    def on_task_done(index: int, total: int, curated) -> None:
        elapsed = time.monotonic() - timer["start"]
        flags = []
        if curated.refinement_applied:
            flags.append("refined")
        if curated.detected_issues:
            flags.append(f"issues: {', '.join(curated.detected_issues)}")
        suffix = f" ({'; '.join(flags)})" if flags else ""
        print(
            f"[curation] [{index}/{total}] {curated.task_id}: "
            f"{curated.final_verdict} (conf {curated.answer_confidence:.2f}) "
            f"[{elapsed:.1f}s]{suffix}",
            flush=True,
        )
        if args.verbose:
            _print_verbose(curated)

    print(f"[curation] verifying {len(tasks)} tasks (this calls the model per task)...", flush=True)
    run_start = time.monotonic()
    try:
        results = Pipeline.build(client, config).run(
            tasks, on_task_start=on_task_start, on_task_done=on_task_done
        )
    except Exception as exc:  # top-level boundary: report cleanly, no stack dump
        print(f"[curation] ERROR talking to the model: {exc}", file=sys.stderr)
        print(f"[curation] check {config.api_key_env} and your account credits.", file=sys.stderr)
        return 1
    print(f"[curation] " + "-" * 60, flush=True)
    print(f"[curation] verified {len(results)} tasks in {time.monotonic() - run_start:.1f}s", flush=True)

    # Evaluate first so the ground-truth annotations are present when results
    # are serialized.
    evaluator = Evaluator(results, tasks)
    report = evaluator.evaluate()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = out_dir / f"{result.task_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    aggregate = out_dir / args.results_file
    aggregate.write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )

    # Ranked human-review queue: triage every result worst-first so a curator
    # reviews the riskiest items before the clean ones. Written as markdown + CSV,
    # with the top few embedded in the main report.
    queue = prioritize(results)
    (out_dir / args.queue_file).write_text(render_queue_markdown(queue), encoding="utf-8")
    queue_csv = Path(args.queue_file).with_suffix(".csv").name
    (out_dir / queue_csv).write_text(render_queue_csv(queue), encoding="utf-8")

    report_md = evaluator.render_markdown(report, queue_summary=render_queue_summary(queue, top_n=3))
    (out_dir / args.report_file).write_text(report_md, encoding="utf-8")

    print(
        f"[curation] wrote {len(results)} task files + {args.results_file} + "
        f"{args.report_file} + {args.queue_file} + {queue_csv} to {out_dir}/"
    )
    high = sum(1 for it in queue if it.level == "HIGH")
    medium = sum(1 for it in queue if it.level == "MEDIUM")
    print(f"[curation] review queue: {high} HIGH, {medium} MEDIUM (top of {args.queue_file})")
    print(
        f"[curation] verdict accuracy {report.accuracy:.0%} "
        f"({report.correct}/{report.scored}); refinements applied: {report.refinements}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
