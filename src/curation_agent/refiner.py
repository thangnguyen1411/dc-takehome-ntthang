"""Stage 3: the self-improvement loop.

verify -> critic gate -> regenerate -> re-verify, repeated until the answer is
`supported` with confidence above threshold, or the iteration cap is hit. Every
pass is recorded as an `Iteration` so the reasoning trace is fully auditable.
"""

from __future__ import annotations

from .config import BAD_VERDICTS, Config
from .llm import LLMClient
from .models import Iteration, Task, Verdict
from .verifier import AnswerVerifier


class Refiner:
    """Runs the critic -> regenerate -> re-verify loop over a single task.

    Depends on an `AnswerVerifier` (a single `Verifier` or a `PanelVerifier`) to
    judge, and the raw `LLMClient` to regenerate, with thresholds and the
    iteration cap read from `Config`.
    """

    SYSTEM = """You rewrite a scientific answer so it is fully faithful to the
evidence. Use ONLY the supplied evidence. Correct any contradiction or
hallucination, state the evidence's central finding directly, and do not add
claims the evidence does not support. Return only the corrected answer."""

    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    def __init__(self, client: LLMClient, verifier: AnswerVerifier, config: Config) -> None:
        self._client = client
        self._verifier = verifier
        self._config = config

    def refine(self, task: Task) -> list[Iteration]:
        """Run the verify/refine loop and return the full iteration trace."""
        iterations: list[Iteration] = []
        current_answer = task.candidate_answer

        latest_verdict = self._verifier.verify(task, current_answer)
        iterations.append(Iteration(step=0, answer=current_answer, verdict=latest_verdict, refined=False))

        step = 0
        while self._needs_refinement(latest_verdict) and step < self._config.max_refine_iterations:
            step += 1
            current_answer = self._regenerate_answer(task, latest_verdict, current_answer)
            latest_verdict = self._verifier.verify(task, current_answer)
            iterations.append(Iteration(step=step, answer=current_answer, verdict=latest_verdict, refined=True))

        return iterations

    def _needs_refinement(self, verdict: Verdict) -> bool:
        return verdict.verdict in BAD_VERDICTS or verdict.confidence < self._config.confidence_threshold

    def _regenerate_answer(self, task: Task, prior_verdict: Verdict, prior_answer: str) -> str:
        prompt = (
            f"QUESTION: {task.question}\n\n"
            f"EVIDENCE: {task.reference_context}\n\n"
            f"PRIOR ANSWER: {prior_answer}\n\n"
            f"PROBLEM TYPE: {prior_verdict.verdict}\n"
            f"WHY IT FAILED: {prior_verdict.rationale}"
        )
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=prompt,
            schema=self.SCHEMA,
            tool_name="emit_answer",
        )
        return raw["answer"].strip()
