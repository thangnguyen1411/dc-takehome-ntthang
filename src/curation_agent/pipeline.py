"""Orchestrates the stages into per-task `CurationResult` objects.

`Pipeline` owns the refine loop (which owns the verifier) and the deterministic
checker, and assembles their outputs. `Pipeline.build(client, config)` is the
composition root that wires the object graph so callers don't have to.
"""

from __future__ import annotations

from .checks import DeterministicChecker
from .config import Config
from .corpus import load_corpus
from .llm import LLMClient, make_client
from .models import CurationResult, Iteration, RubricScores, Task
from .reflector import Reflector
from .refiner import Refiner
from .retriever import KeywordRetriever, Retriever
from .verifier import AnswerVerifier, PanelVerifier, Verifier


class Pipeline:
    """Runs every task through deterministic checks + the verify/refine loop."""

    def __init__(self, refiner: Refiner, config: Config, retriever: Retriever | None = None) -> None:
        self._refiner = refiner
        self._config = config
        self._retriever = retriever

    @classmethod
    def build(cls, client: LLMClient, config: Config) -> "Pipeline":
        """Composition root: assemble the (optional retriever ->) verifier ->
        refiner -> pipeline graph.

        `client` is the primary Anthropic client, used for reflection and
        regeneration. Verification uses a single `Verifier` over that same client
        by default, or a cross-checking `PanelVerifier` when
        `config.verifier_panel` is set. A `KeywordRetriever` over `corpus_path` is
        attached only when `config.retrieve` is set.
        """
        verifier = cls._build_verifier(client, config)
        reflector = Reflector(client)
        refiner = Refiner(client, verifier, reflector, config)
        retriever = cls._build_retriever(config)
        return cls(refiner, config, retriever)

    @staticmethod
    def _build_retriever(config: Config) -> Retriever | None:
        if not config.retrieve:
            return None
        corpus = load_corpus(config.corpus_path)
        return KeywordRetriever(corpus, threshold=config.retrieval_threshold)

    @staticmethod
    def _build_verifier(client: LLMClient, config: Config) -> AnswerVerifier:
        if not config.verifier_panel:
            return Verifier(client)
        verifiers = [Verifier(make_client(spec, config), name=spec) for spec in config.verifier_panel]
        return PanelVerifier(verifiers)

    def run(self, tasks: list[Task], on_task_start=None, on_task_done=None) -> list[CurationResult]:
        """Curate every task, returning one `CurationResult` each.

        Optional callbacks let a caller report progress without the pipeline doing
        any I/O itself (keeping it a pure library component):
        `on_task_start(index, total, task)` fires before a task is curated, and
        `on_task_done(index, total, curated)` fires after. Both use a 1-based index.
        """
        checker = DeterministicChecker(tasks)
        total = len(tasks)
        results: list[CurationResult] = []
        for index, task in enumerate(tasks, start=1):
            if on_task_start is not None:
                on_task_start(index, total, task)
            try:
                curated = self._curate_task(task, checker)
            except Exception as exc:  # isolate per-task: record and continue
                curated = self._error_result(task, exc)
            results.append(curated)
            if on_task_done is not None:
                on_task_done(index, total, curated)
        return results

    def _curate_task(self, task: Task, checker: DeterministicChecker) -> CurationResult:
        # Optional retrieval: when configured, fetch corpus snippets into the
        # task's `relevant_knowledge` (its given `reference_context` is untouched).
        task, retrieved, retrieval_issues = self._maybe_retrieve(task)

        det_issues = checker.issues_for(task)
        iterations = self._refiner.refine(task)

        final = iterations[-1]
        refined = len(iterations) > 1
        derived = self._derived_issues(
            iterations[0].verdict.question_quality_score, final.verdict.confidence
        )
        rubric_issues = self._rubric_issues(final.verdict.rubric)
        all_issues = sorted(
            set(det_issues) | set(final.verdict.detected_issues)
            | set(derived) | set(rubric_issues) | set(retrieval_issues)
        )

        return CurationResult(
            task_id=task.task_id,
            question_quality_score=final.verdict.question_quality_score,
            answer_confidence=final.verdict.confidence,
            detected_issues=all_issues,
            agent_reasoning_summary=self._summarize(iterations, refined),
            refinement_applied=refined,
            final_answer=final.answer,
            evidence=self._evidence_list(task),
            final_verdict=final.verdict.verdict,
            rubric_scores=final.verdict.rubric,
            retrieved_evidence=retrieved,
            iterations=iterations,
        )

    def _maybe_retrieve(self, task: Task):
        """When retrieval is on, fetch corpus snippets into `relevant_knowledge`.

        Returns (effective_task, retrieved_snippets, issue_tags). The task's given
        `reference_context` is never touched — retrieved text lands in the separate
        `relevant_knowledge` field, and `Task.evidence_block()` combines the two
        with PRIMARY/SUPPLEMENTARY precedence at prompt time. Only snippets
        clearing the relevance threshold are kept; if a task had no given evidence
        and nothing clears it, `retrieval_failed` is tagged so the verdict stays
        honest.
        """
        if self._retriever is None:
            return task, [], []
        had_evidence = bool(task.reference_context.strip())
        snippets = self._retriever.retrieve(task.question, k=self._config.retrieval_top_k)
        if not snippets:
            # Nothing relevant; only a problem if the task had no evidence to begin with.
            return task, [], [] if had_evidence else ["retrieval_failed"]
        retrieved_text = "\n".join(s.text for s in snippets)
        return task.model_copy(update={"relevant_knowledge": retrieved_text}), snippets, ["evidence_retrieved"]

    @staticmethod
    def _evidence_list(task: Task) -> list[str]:
        """The evidence actually used, as a clean list — given first, then any
        retrieved knowledge as a separate entry (no inline section labels)."""
        evidence = [task.reference_context] if task.reference_context.strip() else []
        if task.relevant_knowledge.strip():
            evidence.append(task.relevant_knowledge)
        return evidence

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

    def _rubric_issues(self, rubric: RubricScores) -> list[str]:
        """Map low rubric axes to actionable tags (orthogonal to the verdict).

        Each axis below `rubric_threshold` gets a distinct tag so a curator can
        see the *kind* of weakness: unfaithful, incomplete, or over-claimed.
        """
        issues: list[str] = []
        if rubric.faithfulness < self._config.rubric_threshold:
            issues.append("low_faithfulness")
        if rubric.completeness < self._config.rubric_threshold:
            issues.append("incomplete")
        if rubric.specificity < self._config.rubric_threshold:
            issues.append("overclaim")
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
