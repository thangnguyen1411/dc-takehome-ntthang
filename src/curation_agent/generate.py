"""CLI to generate a labeled benchmark CSV from a knowledge corpus.

    python -m curation_agent.generate --corpus data/corpus --out data/generated_tasks.csv

Reads evidence snippets from a corpus (a file or a folder of `.txt` files), and
for each generates one task per target verdict (cycling through the taxonomy),
writing a tab-separated CSV in the same format the curation pipeline ingests.
Requires an API key.
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import cycle
from pathlib import Path

from .config import Config
from .corpus import load_corpus
from .generator import BenchmarkGenerator
from .llm import build_client
from .models import Task

# A balanced rotation of target verdicts so a generated set isn't all one flaw.
DEFAULT_TARGETS = ["supported", "contradicted", "hallucinated", "weak_reasoning", "unsupported"]

CSV_COLUMNS = [
    "task_id", "paper_title", "domain", "question",
    "reference_context", "candidate_answer", "ground_truth_signal",
]


def _write_csv(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for t in tasks:
            writer.writerow({
                "task_id": t.task_id,
                "paper_title": t.paper_title,
                "domain": t.domain,
                "question": t.question,
                "reference_context": t.reference_context,
                "candidate_answer": t.candidate_answer,
                "ground_truth_signal": t.ground_truth_signal or "",
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a labeled curation benchmark")
    parser.add_argument(
        "--corpus", default="data/corpus",
        help="evidence corpus: a .txt file or a folder of .txt files (one snippet per line)",
    )
    parser.add_argument("--out", default="data/generated_tasks.csv", help="output CSV path")
    parser.add_argument("--per-evidence", type=int, default=1, help="tasks to generate per evidence snippet")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"], default="anthropic",
        help="model provider (default: anthropic)",
    )
    parser.add_argument("--model", default=None, help="override the provider's default model id")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    config = Config(provider=args.provider, model=args.model)
    try:
        client = build_client(config)
    except RuntimeError as exc:
        print(f"[generate] {exc}", file=sys.stderr)
        return 1

    try:
        snippets = load_corpus(args.corpus)
    except FileNotFoundError as exc:
        print(f"[generate] {exc}", file=sys.stderr)
        return 1
    if not snippets:
        print(f"[generate] no evidence snippets found in {args.corpus}", file=sys.stderr)
        return 1

    generator = BenchmarkGenerator(client)
    targets = cycle(DEFAULT_TARGETS)
    tasks: list[Task] = []
    counter = 0
    total = len(snippets) * args.per_evidence
    print(f"[generate] {len(snippets)} snippets x {args.per_evidence} = {total} tasks "
          f"via {config.provider}:{config.resolved_model}", flush=True)

    for snippet in snippets:
        for _ in range(args.per_evidence):
            counter += 1
            target = next(targets)
            task_id = f"gen_{counter:03d}"
            try:
                task = generator.generate(task_id, snippet, target)
            except Exception as exc:  # isolate per-task: skip and continue
                print(f"[generate] {task_id} ({target}) failed: {exc}", file=sys.stderr)
                continue
            tasks.append(task)
            print(f"[generate] {task_id} [{target}]: {task.question}", flush=True)

    if not tasks:
        print("[generate] no tasks generated", file=sys.stderr)
        return 1

    _write_csv(Path(args.out), tasks)
    print(f"[generate] wrote {len(tasks)} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
