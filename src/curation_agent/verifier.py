"""Stage 2: the verification agent.

Given a question, its evidence, and a candidate answer, a verifier returns a
structured `Verdict`. Three implementations satisfy the `AnswerVerifier`
protocol, so the refine loop is agnostic to which is used:

- `Verifier` — a single-model judge.
- `PanelVerifier` — several judges vote *independently* (no cross-talk).
- `DebateVerifier` — several judges *debate*: each sees the others' rationales
  each round and may revise, before a final vote.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from .llm import LLMClient
from .models import RubricScores, Task, Verdict
from .prompts import EVIDENCE_PRECEDENCE_RULE


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
grounded in specific evidence tokens.

""" + EVIDENCE_PRECEDENCE_RULE

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

    def verify(self, task: Task, answer: str, peer_review: str = "") -> Verdict:
        """Judge `answer` against `task`'s evidence.

        `peer_review` (optional) carries other judges' assessments during a
        `DebateVerifier` round; when present it is appended to the prompt so this
        judge can reconsider. Empty for a normal single-shot verification.
        """
        raw = self._client.complete_structured(
            system=self.SYSTEM,
            prompt=self._prompt(task, answer, peer_review),
            schema=self.SCHEMA,
            tool_name="emit_verdict",
        )
        return Verdict(**raw)

    @staticmethod
    def _prompt(task: Task, answer: str, peer_review: str = "") -> str:
        # `evidence_block()` carries its own section header(s) — given evidence and
        # retrieved context are separate, self-labeled sections, not one EVIDENCE blob.
        prompt = f"QUESTION: {task.question}\n\n{task.evidence_block()}\n\nANSWER: {answer}"
        if peer_review:
            prompt += f"\n\n{peer_review}"
        return prompt


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _majority_vote(verdicts: list[Verdict]) -> str:
    """Most-voted verdict label, with a conservative tie-break.

    On a tie, prefer flagging a problem over `supported` (a false flag is cheaper
    than a missed bad answer in curation); among the tied flagged labels, take the
    one its voters were on average most confident about. A tie means >= 2 *distinct*
    labels, so dropping `supported` always leaves at least one candidate.
    """
    counts = Counter(v.verdict for v in verdicts)
    top_count = counts.most_common(1)[0][1]
    leaders = [label for label, c in counts.items() if c == top_count]
    if len(leaders) == 1:
        return leaders[0]
    candidates = [label for label in leaders if label != "supported"]
    return max(candidates, key=lambda label: _mean(
        v.confidence for v in verdicts if v.verdict == label
    ))


def _aggregate(names: list[str], verdicts: list[Verdict], extra_issues: set[str]) -> Verdict:
    """Combine several judges' verdicts into one (shared by panel and debate).

    - verdict: majority vote (see `_majority_vote`).
    - confidence: mean confidence of the judges that voted for the chosen verdict.
    - question_quality_score & rubric axes: mean across all judges.
    - detected_issues: union across all judges, plus any `extra_issues`.
    - rationale: every judge's rationale, attributed by name.
    """
    chosen = _majority_vote(verdicts)
    agreed = [v for v in verdicts if v.verdict == chosen]
    issues = {issue for v in verdicts for issue in v.detected_issues} | extra_issues
    rationale = " | ".join(
        f"[{name}] judged '{v.verdict}' ({v.confidence:.2f}): {v.rationale}"
        for name, v in zip(names, verdicts)
    )
    return Verdict(
        verdict=chosen,
        confidence=_mean(v.confidence for v in agreed),
        question_quality_score=_mean(v.question_quality_score for v in verdicts),
        rubric=RubricScores(
            faithfulness=_mean(v.rubric.faithfulness for v in verdicts),
            completeness=_mean(v.rubric.completeness for v in verdicts),
            specificity=_mean(v.rubric.specificity for v in verdicts),
        ),
        detected_issues=sorted(issues),
        rationale=rationale,
    )


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
        unanimous = len({v.verdict for v in verdicts}) == 1
        extra = set() if unanimous else {"panel_disagreement"}
        return _aggregate([vf.name for vf in self._verifiers], verdicts, extra)


class DebateVerifier:
    """Several judges *debate* a verdict over a few rounds, then vote.

    Where `PanelVerifier` collects independent one-shot votes, here the judges see
    each other. The aim is to *resolve* disagreement through argument rather than
    merely tally it: a judge confronted with a more evidence-faithful reading can
    change its mind, and a judge that's right can defend its position.

    - Round 0: every judge verifies independently.
    - Rounds 1..max_rounds: each judge re-verifies, shown the *other* judges'
      latest verdicts + rationales, and may revise. The loop stops early the moment
      all judges agree (consensus).
    - Final verdict: majority vote over the last round (same conservative tie-break
      as the panel). Issue tags record what the debate did:
        - `debate_unresolved` — judges never reached consensus within max_rounds.
        - `verdict_changed_in_debate` — debate moved the majority off its round-0
          position (a strong signal the item is genuinely contestable).

    Cost is roughly `n_judges * (1 + rounds)` verifications, so it's opt-in for
    high-stakes batches. Needs >= 2 judges (one judge has no one to debate).
    """

    def __init__(self, verifiers: list[Verifier], max_rounds: int = 2, log: bool = False) -> None:
        if len(verifiers) < 2:
            raise ValueError("DebateVerifier needs at least two verifiers to debate")
        self._verifiers = verifiers
        self._max_rounds = max_rounds
        self._log = log

    def verify(self, task: Task, answer: str) -> Verdict:
        verdicts = [v.verify(task, answer) for v in self._verifiers]  # round 0
        opening_majority = _majority_vote(verdicts)
        self._log_round(0, verdicts)

        rounds = 0
        while not self._consensus(verdicts) and rounds < self._max_rounds:
            rounds += 1
            verdicts = [
                v.verify(task, answer, peer_review=self._peer_review(i, verdicts))
                for i, v in enumerate(self._verifiers)
            ]
            self._log_round(rounds, verdicts)

        extra: set[str] = set()
        if not self._consensus(verdicts):
            extra.add("debate_unresolved")
        if _majority_vote(verdicts) != opening_majority:
            extra.add("verdict_changed_in_debate")
        return _aggregate([vf.name for vf in self._verifiers], verdicts, extra)

    @staticmethod
    def _consensus(verdicts: list[Verdict]) -> bool:
        return len({v.verdict for v in verdicts}) == 1

    def _log_round(self, rnd: int, verdicts: list[Verdict]) -> None:
        """One greppable line per round in the --llm-log stream, tagged
        `[LLM-debate]` so it groups with the refiner's `[LLM-step]` markers."""
        if not self._log:
            return
        votes = ", ".join(
            f"{vf.name}={v.verdict}({v.confidence:.2f})"
            for vf, v in zip(self._verifiers, verdicts)
        )
        state = "consensus" if self._consensus(verdicts) else "no consensus"
        label = "round 0 (opening)" if rnd == 0 else f"round {rnd}"
        print(f"[LLM-debate] {label}: {votes} [{state}]", flush=True)

    def _peer_review(self, idx: int, verdicts: list[Verdict]) -> str:
        """Render the *other* judges' current positions for judge `idx` to weigh."""
        others = "\n".join(
            f"- [{vf.name}] argues '{v.verdict}' (confidence {v.confidence:.2f}): {v.rationale}"
            for j, (vf, v) in enumerate(zip(self._verifiers, verdicts)) if j != idx
        )
        return (
            "OTHER REVIEWERS ASSESSED THIS SAME ANSWER AGAINST THE SAME EVIDENCE:\n"
            f"{others}\n\n"
            "Weigh their reasoning against your own. If an argument is more faithful "
            "to the evidence than your current verdict, revise accordingly; otherwise "
            "hold your position and explain why it is better grounded. Re-issue your "
            "verdict now, using ONLY the supplied evidence."
        )
