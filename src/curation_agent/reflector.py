"""Stage 3a: the reflection step.

Between judging (verify) and acting (regenerate), the `Reflector` reasons about
*how* to fix a failed answer and emits a structured `ReflectionPlan`. It does not
judge (the verifier does) and does not write the answer (the regenerator does)
it plans the correction, so the strategy is explicit and auditable.
"""

from __future__ import annotations

from .llm import LLMClient
from .models import Iteration, ReflectionPlan, Task


class Reflector:
    """Produces a corrective fix plan for a failed answer via one LLM call."""

    SYSTEM = """You are a correction strategist for scientific answers. You are
given a question, its evidence, and one or more prior answers that FAILED
verification (with the reason each failed). Do NOT rewrite the answer yourself
and do NOT re-judge it. Instead, plan the fix:
- diagnosis: state precisely why the prior attempt(s) failed against the evidence.
- what_to_change: the concrete correction the rewrite must make, grounded only in
  the evidence (e.g. "drop the superlative 'strongest' which evidence says only 'significant'").
- what_to_keep: any part of the prior answer that was already correct and should be preserved.
Be specific and evidence-anchored; this plan will guide a separate rewrite step."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "what_to_change": {"type": "string"},
            "what_to_keep": {"type": "string"},
        },
        # Nothing is strictly required: a model occasionally returns a partial
        # plan, and a partial plan is still useful — far better than failing the
        # whole task. `ReflectionPlan` defaults the missing fields to "".
        "required": [],
    }

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def reflect(self, task: Task, history: list[Iteration]) -> ReflectionPlan:
        attempts = "\n".join(
            f"ATTEMPT {it.step + 1} produced: {it.answer!r}\n"
            f"  -> judged '{it.verdict.verdict}': {it.verdict.rationale}"
            for it in history
        )
        prompt = (
            f"QUESTION: {task.question}\n\n"
            f"EVIDENCE: {task.reference_context}\n\n"
            f"FAILED ATTEMPTS:\n{attempts}\n\n"
            f"Plan how to fix the answer."
        )
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=prompt,
            schema=self.SCHEMA,
            tool_name="emit_reflection",
        )
        return ReflectionPlan(**raw)
