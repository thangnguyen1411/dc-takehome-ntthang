"""Stage 1: ingest the task CSV into validated `Task` objects."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import Task


class Ingestor:
    """Loads and validates the task CSV, producing `Task` objects.

    Stateless apart from the required-column contract; kept as a class so every
    pipeline stage has a uniform object-oriented shape.
    """

    REQUIRED_COLUMNS = {"task_id", "question", "reference_context", "candidate_answer"}

    def load(self, csv_path: str | Path) -> list[Task]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"task CSV not found: {path}")

        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")
            return [self._row_to_task(row) for row in reader]

    @staticmethod
    def _row_to_task(row: dict[str, str]) -> Task:
        return Task(
            task_id=row["task_id"].strip(),
            paper_title=row.get("paper_title", "").strip(),
            domain=row.get("domain", "").strip(),
            question=row["question"].strip(),
            reference_context=row["reference_context"].strip(),
            candidate_answer=row["candidate_answer"].strip(),
            ground_truth_signal=(row.get("ground_truth_signal") or "").strip() or None,
        )
