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

    # Per-attempt guidance, escalating in strictness. Each retry's prompt is
    # "improved" by appending a firmer instruction, so a fix that failed once is
    # not simply re-attempted the same way. The last tier is reused if the loop
    # ever runs more passes than there are tiers (see `_escalation_for`).
    ESCALATION = [
        # Tier 1 (pass 1): a normal corrective rewrite.
        "Rewrite the answer to fix the problem above, using only the evidence.",
        # Tier 2 (pass 2): the first fix failed - demand verbatim grounding.
        "Your previous fix STILL failed. Quote the evidence's central finding "
        "verbatim and make only claims directly stated in it.",
        # Tier 3 (pass 3+): repeated failures - be maximally conservative.
        "Multiple fixes have failed. Be maximally conservative: if the evidence "
        "does not clearly state something, omit it entirely rather than infer it.",
    ]

    def refine(self, task: Task) -> list[Iteration]:
        """Run the verify/refine loop and return the full iteration trace."""
        iterations: list[Iteration] = []
        current_answer = task.candidate_answer

        latest_verdict = self._verifier.verify(task, current_answer)
        iterations.append(Iteration(step=0, answer=current_answer, verdict=latest_verdict, refined=False))

        step = 0
        while self._needs_refinement(latest_verdict) and step < self._config.max_refine_iterations:
            step += 1
            # Pass the whole failure history so a retry learns from every prior
            # attempt, not just the most recent one.
            current_answer = self._regenerate_answer(task, iterations, step)
            latest_verdict = self._verifier.verify(task, current_answer)
            iterations.append(Iteration(step=step, answer=current_answer, verdict=latest_verdict, refined=True))

        return iterations

    def _needs_refinement(self, verdict: Verdict) -> bool:
        return verdict.verdict in BAD_VERDICTS or verdict.confidence < self._config.confidence_threshold

    def _regenerate_answer(self, task: Task, history: list[Iteration], attempt: int) -> str:
        """Rewrite the answer, given every prior failed attempt and an
        attempt-scaled instruction (retry-with-improved-prompt)."""
        attempts = "\n".join(
            f"ATTEMPT {it.step + 1} produced: {it.answer!r}\n"
            f"  -> judged '{it.verdict.verdict}': {it.verdict.rationale}"
            for it in history
        )
        prompt = (
            f"QUESTION: {task.question}\n\n"
            f"EVIDENCE: {task.reference_context}\n\n"
            f"PRIOR FAILED ATTEMPTS (do not repeat these mistakes):\n{attempts}\n\n"
            f"{self._escalation_for(attempt)}"
        )
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=prompt,
            schema=self.SCHEMA,
            tool_name="emit_answer",
        )
        return raw["answer"].strip()

    def _escalation_for(self, attempt: int) -> str:
        """The instruction for this retry; firmer with each pass, clamped to the
        strongest tier once attempts exceed the number of tiers."""
        index = min(attempt - 1, len(self.ESCALATION) - 1)
        return self.ESCALATION[index]
