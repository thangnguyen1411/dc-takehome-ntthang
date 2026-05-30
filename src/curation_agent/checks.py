"""Stage 4: deterministic, rule-based verifiers.

These run without an LLM and catch structural problems the model should not be
trusted to police: schema validity, duplicate tasks, and missing evidence. They
are fast, reproducible, and unit-tested without an API key.
"""

from __future__ import annotations

import re

from .models import Task


class DeterministicChecker:
    """Rule-based checks for one batch of tasks.

    Built from the full task list so it can flag cross-task duplicates; the
    duplicate set is computed once in the constructor and reused per task.
    """

    def __init__(self, tasks: list[Task]) -> None:
        self._duplicate_ids = self._duplicate_task_ids(self._find_duplicates(tasks))

    def issues_for(self, task: Task) -> list[str]:
        """All rule-based issues for one task, including cross-task duplication."""
        issues = self.schema_issues(task)
        if task.task_id in self._duplicate_ids:
            issues.append("duplicate_task")
        return issues

    @staticmethod
    def normalized_question(question: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation — for dedup hashing."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", question.lower())).strip()

    @staticmethod
    def schema_issues(task: Task) -> list[str]:
        """Structural problems with a single task."""
        issues: list[str] = []
        if not task.question:
            issues.append("invalid_schema:empty_question")
        if not task.candidate_answer:
            issues.append("invalid_schema:empty_answer")
        if not task.reference_context:
            issues.append("missing_evidence")
        return issues

    @classmethod
    def _find_duplicates(cls, tasks: list[Task]) -> dict[str, list[str]]:
        """Map a normalized question to the task_ids sharing it (size > 1 only)."""
        seen: dict[str, list[str]] = {}
        for task in tasks:
            seen.setdefault(cls.normalized_question(task.question), []).append(task.task_id)
        return {q: ids for q, ids in seen.items() if len(ids) > 1}

    @staticmethod
    def _duplicate_task_ids(duplicates: dict[str, list[str]]) -> set[str]:
        """Flatten the duplicate map into the set of task_ids that are duplicated."""
        return {task_id for ids in duplicates.values() for task_id in ids}
