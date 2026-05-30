"""Stage 2: the verification agent.

Given a question, its evidence, and a candidate answer, the `Verifier` returns a
structured `Verdict`. The system prompt pins it to the evidence only and defines
the issue taxonomy; the user prompt carries the labelled task fields.
"""

from __future__ import annotations

from .llm import LLMClient
from .models import Task, Verdict


class Verifier:
    """Judges a candidate answer against its evidence via a tool-forced LLM call."""

    SYSTEM = """You are a rigorous scientific data-curation verifier.

You judge whether a CANDIDATE ANSWER is faithful to the EVIDENCE for a QUESTION.
Use ONLY the supplied evidence. Do not rely on outside knowledge, and never
reward an answer for being plausible if the evidence does not support it.

Assign exactly one verdict:
- supported: every claim in the answer is backed by the evidence.
- contradicted: the answer asserts something the evidence directly opposes.
- hallucinated: the answer introduces entities or relationships absent from the
  evidence.
- weak_reasoning: the answer is on-topic but shallow, misattributes the main
  finding, or emphasizes a minor point over the evidence's central claim.
- unsupported: the answer makes a claim the evidence neither confirms nor denies.

Also rate question_quality_score: 1.0 for a clear, specific, answerable
question; lower it for vague, ambiguous, multi-part, or unanswerable questions,
and add a weak_question tag to detected_issues when the score is low. List
detected_issues using the verdict names plus optional tags like missing_citation
or unsupported_claim. Give a one to three sentence rationale grounded in
specific evidence tokens."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supported", "contradicted", "hallucinated", "weak_reasoning", "unsupported"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "question_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
            "detected_issues": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "confidence", "question_quality_score", "detected_issues", "rationale"],
    }

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def verify(self, task: Task, answer: str) -> Verdict:
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=self._prompt(task.question, task.reference_context, answer),
            schema=self.SCHEMA,
            tool_name="emit_verdict",
        )
        return Verdict(**raw)

    @staticmethod
    def _prompt(question: str, evidence: str, answer: str) -> str:
        return f"QUESTION: {question}\n\nEVIDENCE: {evidence}\n\nANSWER: {answer}"
