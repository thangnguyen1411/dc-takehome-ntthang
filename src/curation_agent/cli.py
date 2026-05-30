"""Command-line entry point.

    python -m curation_agent.cli --csv data/tasks.csv

Runs the full pipeline, writes per-task JSON + an aggregate results file, scores
against ground truth, and writes a markdown report. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .evaluate import Evaluator
from .ingest import Ingestor
from .llm import build_client
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scientific data curation agent")
    parser.add_argument("--csv", default="data/tasks.csv", help="path to the task CSV")
    parser.add_argument("--out", default="outputs", help="output directory")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    config = Config()
    try:
        client = build_client(config)
    except RuntimeError as exc:
        print(f"[curation] {exc}", file=sys.stderr)
        return 1
    print(f"[curation] LLM backend: anthropic:{config.model}")

    tasks = Ingestor().load(args.csv)
    print(f"[curation] loaded {len(tasks)} tasks from {args.csv}")

    try:
        results = Pipeline.build(client, config).run(tasks)
    except Exception as exc:  # top-level boundary: report cleanly, no stack dump
        print(f"[curation] ERROR talking to the model: {exc}", file=sys.stderr)
        print("[curation] check ANTHROPIC_API_KEY and your account credits.", file=sys.stderr)
        return 1

    # Evaluate first so the ground-truth annotations are present when results
    # are serialized.
    evaluator = Evaluator(results, tasks)
    report = evaluator.evaluate()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = out_dir / f"{result.task_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    aggregate = out_dir / "results.json"
    aggregate.write_text(
        json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8"
    )

    report_md = evaluator.render_markdown(report)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"[curation] wrote {len(results)} task files + results.json + report.md to {out_dir}/")
    print(
        f"[curation] verdict accuracy {report.accuracy:.0%} "
        f"({report.correct}/{report.scored}); refinements applied: {report.refinements}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
