"""Orchestrates the stages into per-task `CurationResult` objects.

`Pipeline` owns the refine loop (which owns the verifier) and the deterministic
checker, and assembles their outputs. `Pipeline.build(client, config)` is the
composition root that wires the object graph so callers don't have to.
"""

from __future__ import annotations

from .checks import DeterministicChecker
from .config import Config
from .llm import LLMClient
from .models import CurationResult, Iteration, Task
from .refiner import Refiner
from .verifier import Verifier


class Pipeline:
    """Runs every task through deterministic checks + the verify/refine loop."""

    def __init__(self, refiner: Refiner, config: Config) -> None:
        self._refiner = refiner
        self._config = config

    @classmethod
    def build(cls, client: LLMClient, config: Config) -> "Pipeline":
        """Composition root: assemble the verifier -> refiner -> pipeline graph."""
        verifier = Verifier(client)
        refiner = Refiner(client, verifier, config)
        return cls(refiner, config)

    def run(self, tasks: list[Task]) -> list[CurationResult]:
        checker = DeterministicChecker(tasks)
        results: list[CurationResult] = []
        for task in tasks:
            try:
                results.append(self._curate_task(task, checker))
            except Exception as exc:  # isolate per-task: record and continue
                results.append(self._error_result(task, exc))
        return results

    def _curate_task(self, task: Task, checker: DeterministicChecker) -> CurationResult:
        det_issues = checker.issues_for(task)
        iterations = self._refiner.refine(task)

        final = iterations[-1]
        refined = len(iterations) > 1
        derived = self._derived_issues(
            iterations[0].verdict.question_quality_score, final.verdict.confidence
        )
        all_issues = sorted(set(det_issues) | set(final.verdict.detected_issues) | set(derived))

        return CurationResult(
            task_id=task.task_id,
            question_quality_score=final.verdict.question_quality_score,
            answer_confidence=final.verdict.confidence,
            detected_issues=all_issues,
            agent_reasoning_summary=self._summarize(iterations, refined),
            refinement_applied=refined,
            final_answer=final.answer,
            evidence=[task.reference_context],
            final_verdict=final.verdict.verdict,
            iterations=iterations,
        )

    def _derived_issues(self, question_quality_score: float, confidence: float) -> list[str]:
        """Threshold-driven issue tags not produced directly by the verifier.

        `weak_question` flags an ambiguous or low-quality question — judged on the
        initial pass, since refinement changes the answer, not the question.
        `low_confidence` flags an output the agent is still unsure about *after*
        refinement, so it reads the final confidence.
        """
        issues: list[str] = []
        if question_quality_score < self._config.question_quality_threshold:
            issues.append("weak_question")
        if confidence < self._config.confidence_threshold:
            issues.append("low_confidence")
        return issues

    @staticmethod
    def _summarize(iterations: list[Iteration], refined: bool) -> str:
        first, last = iterations[0].verdict, iterations[-1].verdict
        if not refined:
            return f"Answer judged '{first.verdict}' (confidence {first.confidence:.2f}). {first.rationale}"
        return (
            f"Initial verdict '{first.verdict}' (confidence {first.confidence:.2f}) triggered refinement. "
            f"After {len(iterations) - 1} pass(es) the answer is '{last.verdict}' "
            f"(confidence {last.confidence:.2f}). {last.rationale}"
        )

    @staticmethod
    def _error_result(task: Task, exc: Exception) -> CurationResult:
        """A placeholder result so one task's failure doesn't discard the batch."""
        return CurationResult(
            task_id=task.task_id,
            question_quality_score=0.0,
            answer_confidence=0.0,
            detected_issues=["processing_error"],
            agent_reasoning_summary=f"Curation failed: {type(exc).__name__}: {exc}",
            refinement_applied=False,
            final_answer=task.candidate_answer,
            evidence=[task.reference_context],
            final_verdict="error",
            iterations=[],
        )
