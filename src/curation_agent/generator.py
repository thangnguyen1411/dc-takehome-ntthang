"""Benchmark generation: synthesize labeled QA tasks from evidence snippets.

Given a piece of evidence and a *target* verdict, the `BenchmarkGenerator` asks
the model to produce a question and a candidate answer that exhibits exactly that
flaw (or is faithful, for `supported`). The target verdict becomes the held-out
gold label, so the generated tasks plug straight into the curation pipeline and
its evaluation - closing a generate -> curate -> verify loop: generate a task
with a *known* flaw, run the agent, and check it catches the flaw we injected.

Generation is intentionally the inverse of verification: the verifier judges an
answer against evidence; the generator crafts an answer to earn a chosen verdict.
"""

from __future__ import annotations

from .llm import LLMClient
from .models import Task

# What kind of answer to synthesize, and how, per target gold label. Mirrors the
# verifier's taxonomy so a generated flaw maps onto a verdict the agent can catch.
FLAW_INSTRUCTIONS: dict[str, str] = {
    "supported": "Write an answer that is fully faithful to the evidence - every "
    "claim directly backed by it, no over-claiming.",
    "contradicted": "Write an answer that directly contradicts the evidence "
    "(assert the opposite of what it states).",
    "hallucinated": "Write an answer that introduces a specific entity or "
    "relationship absent from the evidence (a fabricated gene, drug, or finding).",
    "weak_reasoning": "Write an answer that is on-topic but misattributes the "
    "central finding or over-emphasizes a minor point as the main result.",
    "unsupported": "Write an answer that makes a claim the evidence neither "
    "confirms nor denies e.g. an unsupported superlative or extrapolation.",
}


class BenchmarkGenerator:
    """Synthesizes labeled `Task`s from evidence, one per requested target verdict."""

    SYSTEM = """You generate items for a scientific QA benchmark. Given a piece of
EVIDENCE and a TARGET verdict, produce: a plausible PAPER TITLE the evidence could
come from; a short DOMAIN tag (e.g. 'oncology', 'genetics', 'cardiology'); a
QUESTION the evidence can speak to; and a CANDIDATE ANSWER that exhibits exactly
the target's flaw (or is faithful, for 'supported'). The answer must be
plausible-sounding so it is a genuine test not obviously wrong. Do not reference
the target verdict in the text. Ground the question in the evidence; craft the
answer to match the requested verdict."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "paper_title": {"type": "string"},
            "domain": {"type": "string"},
            "question": {"type": "string"},
            "candidate_answer": {"type": "string"},
        },
        "required": ["paper_title", "domain", "question", "candidate_answer"],
    }

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(
        self,
        task_id: str,
        evidence: str,
        target_verdict: str,
        *,
        paper_title: str = "",
        domain: str = "",
    ) -> Task:
        """Generate one labeled `Task` whose gold label is `target_verdict`.

        `paper_title`/`domain` override the model's suggestions when supplied;
        otherwise the model's generated values are used.
        """
        if target_verdict not in FLAW_INSTRUCTIONS:
            raise ValueError(
                f"unknown target_verdict {target_verdict!r}; "
                f"choose from {sorted(FLAW_INSTRUCTIONS)}"
            )
        prompt = (
            f"EVIDENCE: {evidence}\n\n"
            f"TARGET: {target_verdict}\n"
            f"INSTRUCTION: {FLAW_INSTRUCTIONS[target_verdict]}"
        )
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=prompt,
            schema=self.SCHEMA,
            tool_name="emit_task",
        )
        return Task(
            task_id=task_id,
            paper_title=paper_title or raw.get("paper_title", "").strip(),
            domain=domain or raw.get("domain", "").strip(),
            question=raw["question"].strip(),
            reference_context=evidence.strip(),
            candidate_answer=raw["candidate_answer"].strip(),
            ground_truth_signal=target_verdict,
        )
