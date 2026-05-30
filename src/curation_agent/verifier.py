"""Stage 2: the verification agent.

Given a question, its evidence, and a candidate answer, a verifier returns a
structured `Verdict`. `Verifier` is a single-model judge; `PanelVerifier` runs
several judges (optionally across providers) and votes. Both satisfy the
`AnswerVerifier` protocol, so the refine loop is agnostic to which is used.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from .llm import LLMClient
from .models import RubricScores, Task, Verdict


class AnswerVerifier(Protocol):
    """Anything that can judge an answer against its task and return a Verdict."""

    def verify(self, task: Task, answer: str) -> Verdict:
        ...


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
and add a weak_question tag to detected_issues when the score is low.

Separately, score three ORTHOGONAL rubric axes from 0.0 to 1.0 — judge each
independently of the single verdict and of each other:
- faithfulness: are the answer's claims grounded in, and not contradicting, the
  evidence? (a contradiction or fabrication drives this down)
- completeness: does the answer capture the evidence's central finding, rather
  than a minor or tangential point?
- specificity: is the answer appropriately precise WITHOUT over-claiming — no
  unsupported superlatives ("strongest", "most"), hedges, or vagueness?

List detected_issues using the verdict names plus optional tags like
missing_citation or unsupported_claim. Give a one to three sentence rationale
grounded in specific evidence tokens."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supported", "contradicted", "hallucinated", "weak_reasoning", "unsupported"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "question_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
            "rubric": {
                "type": "object",
                "properties": {
                    "faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
                    "completeness": {"type": "number", "minimum": 0, "maximum": 1},
                    "specificity": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["faithfulness", "completeness", "specificity"],
            },
            "detected_issues": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "confidence", "question_quality_score", "rubric", "detected_issues", "rationale"],
    }

    def __init__(self, client: LLMClient, name: str = "verifier") -> None:
        self._client = client
        self.name = name

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


class PanelVerifier:
    """Runs several verifiers independently and combines them into one Verdict.

    Independent judges (ideally across providers) reduce shared blind spots: a
    single self-judging model can rubber-stamp its own style of error, whereas a
    cross-model panel must agree. The combination rules:

    - verdict: majority vote. On a tie, prefer flagging a problem over `supported`
      (a false flag is cheaper than a missed bad answer in curation); among tied
      flagged labels, take the one its voters were most confident about.
    - confidence: mean confidence of the verifiers that voted for the chosen verdict.
    - question_quality_score & rubric axes: mean across all verifiers.
    - detected_issues: union across all verifiers, plus `panel_disagreement` when
      the verifiers were not unanimous (a strong "send to a human" signal).
    - rationale: each verifier's rationale, attributed by name.
    """

    def __init__(self, verifiers: list[Verifier]) -> None:
        if not verifiers:
            raise ValueError("PanelVerifier needs at least one verifier")
        self._verifiers = verifiers

    def verify(self, task: Task, answer: str) -> Verdict:
        verdicts = [v.verify(task, answer) for v in self._verifiers]
        if len(verdicts) == 1:
            return verdicts[0]
        return self._combine(verdicts)

    def _combine(self, verdicts: list[Verdict]) -> Verdict:
        chosen = self._vote(verdicts)
        agreed = [v for v in verdicts if v.verdict == chosen]
        unanimous = len({v.verdict for v in verdicts}) == 1

        issues = {issue for v in verdicts for issue in v.detected_issues}
        if not unanimous:
            issues.add("panel_disagreement")

        return Verdict(
            verdict=chosen,
            confidence=self._mean(v.confidence for v in agreed),
            question_quality_score=self._mean(v.question_quality_score for v in verdicts),
            rubric=RubricScores(
                faithfulness=self._mean(v.rubric.faithfulness for v in verdicts),
                completeness=self._mean(v.rubric.completeness for v in verdicts),
                specificity=self._mean(v.rubric.specificity for v in verdicts),
            ),
            detected_issues=sorted(issues),
            rationale=self._combine_rationales(verdicts),
        )

    def _vote(self, verdicts: list[Verdict]) -> str:
        counts = Counter(v.verdict for v in verdicts)
        top_count = counts.most_common(1)[0][1]
        leaders = [label for label, c in counts.items() if c == top_count]
        if len(leaders) == 1:
            return leaders[0]
        # Tie. Prefer flagging a problem over 'supported'; among the remaining
        # candidates, pick the label its voters were on average most confident
        # about. A tie means >= 2 *distinct* labels, so dropping 'supported'
        # always leaves at least one candidate.
        candidates = [label for label in leaders if label != "supported"]
        return max(candidates, key=lambda label: self._mean(
            v.confidence for v in verdicts if v.verdict == label
        ))

    def _combine_rationales(self, verdicts: list[Verdict]) -> str:
        parts = [
            f"[{vf.name}] judged '{v.verdict}' ({v.confidence:.2f}): {v.rationale}"
            for vf, v in zip(self._verifiers, verdicts)
        ]
        return " | ".join(parts)

    @staticmethod
    def _mean(values) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0
