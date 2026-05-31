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
        help="augment every task's evidence with relevant snippets from the corpus (RAG)",
    )
    parser.add_argument(
        "--corpus", default="data/corpus",
        help="knowledge corpus for --retrieve: a .txt file or folder of .txt files",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    panel = tuple(spec.strip() for spec in args.panel.split(",") if spec.strip()) if args.panel else ()
    config = Config(
        provider=args.provider, model=args.model, verifier_panel=panel,
        llm_log=args.llm_log, retrieve=args.retrieve, corpus_path=args.corpus,
    )
    try:
        client = build_client(config)
    except RuntimeError as exc:
        print(f"[curation] {exc}", file=sys.stderr)
        return 1
    primary = f"{config.provider}:{config.resolved_model}"
    verifier_backend = " + ".join(config.verifier_panel) if config.verifier_panel else primary
    print(f"[curation] verifier: {verifier_backend} | regeneration: {primary}")
    if config.retrieve:
        print(f"[curation] retrieval: ON (corpus: {config.corpus_path})")

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

    report_md = evaluator.render_markdown(report)
    (out_dir / args.report_file).write_text(report_md, encoding="utf-8")

    print(
        f"[curation] wrote {len(results)} task files + {args.results_file} + "
        f"{args.report_file} to {out_dir}/"
    )
    print(
        f"[curation] verdict accuracy {report.accuracy:.0%} "
        f"({report.correct}/{report.scored}); refinements applied: {report.refinements}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
