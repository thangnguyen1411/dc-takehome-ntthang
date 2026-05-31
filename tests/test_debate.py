"""Tests for DebateVerifier — judges that revise across rounds. No API key.

Uses a scripted judge double whose verdict can change per call, so we can drive
consensus, persistent disagreement, and verdict flips deterministically.
"""

import pytest
from pytest import approx

from curation_agent.models import Task, Verdict
from curation_agent.verifier import DebateVerifier


def _task() -> Task:
    return Task(
        task_id="task_d",
        question="Which biomarker?",
        reference_context="Biomarker CYP2E1 was elevated.",
        candidate_answer="CYP2E1 was elevated.",
    )


def _verdict(label, conf=0.8, rubric=(1.0, 1.0, 1.0), issues=None, rationale="") -> Verdict:
    f, c, s = rubric
    return Verdict(
        verdict=label, confidence=conf, question_quality_score=0.8,
        rubric={"faithfulness": f, "completeness": c, "specificity": s},
        detected_issues=issues or [], rationale=rationale,
    )


class ScriptedDebater:
    """A judge that returns a preset verdict per verify() call (round by round)."""

    def __init__(self, name: str, verdicts: list[Verdict]) -> None:
        self.name = name
        self._verdicts = verdicts
        self.calls = 0
        self.peer_reviews: list[str] = []

    def verify(self, task, answer, peer_review: str = "") -> Verdict:
        self.peer_reviews.append(peer_review)
        verdict = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return verdict


def test_needs_at_least_two_judges():
    with pytest.raises(ValueError):
        DebateVerifier([ScriptedDebater("solo", [_verdict("supported")])])


def test_immediate_consensus_runs_no_debate_rounds():
    a = ScriptedDebater("a", [_verdict("supported", 0.9)])
    b = ScriptedDebater("b", [_verdict("supported", 0.8)])
    out = DebateVerifier([a, b], max_rounds=2).verify(_task(), "ans")

    assert out.verdict == "supported"
    assert out.confidence == approx(0.85) # mean of agreeing judges
    assert a.calls == 1 and b.calls == 1 # round 0 only — no debate needed
    assert "debate_unresolved" not in out.detected_issues
    assert "verdict_changed_in_debate" not in out.detected_issues


def test_debate_converges_and_flags_verdict_change():
    # round 0: 2 supported vs 1 contradicted -> majority supported.
    # round 1: all three converge on contradicted -> consensus, majority flipped.
    a = ScriptedDebater("a", [_verdict("supported", 0.7), _verdict("contradicted", 0.9)])
    b = ScriptedDebater("b", [_verdict("supported", 0.7), _verdict("contradicted", 0.9)])
    c = ScriptedDebater("c", [_verdict("contradicted", 0.95), _verdict("contradicted", 0.95)])
    out = DebateVerifier([a, b, c], max_rounds=2).verify(_task(), "ans")

    assert out.verdict == "contradicted"
    assert a.calls == 2 and b.calls == 2 and c.calls == 2 # one debate round, then stop
    assert "verdict_changed_in_debate" in out.detected_issues
    assert "debate_unresolved" not in out.detected_issues # they agreed in the end


def test_persistent_disagreement_is_tagged_unresolved():
    a = ScriptedDebater("a", [_verdict("supported", 0.6)])      # never budges
    b = ScriptedDebater("b", [_verdict("contradicted", 0.9)])   # never budges
    out = DebateVerifier([a, b], max_rounds=2).verify(_task(), "ans")

    assert a.calls == 3 and b.calls == 3 # round 0 + 2 debate rounds
    assert "debate_unresolved" in out.detected_issues
    assert out.verdict == "contradicted" # tie -> prefer flagging
    assert out.confidence == 0.9 # mean of the chosen-verdict voters


def test_peer_review_carries_other_judges_positions():
    a = ScriptedDebater("a", [_verdict("supported", 0.7, rationale="looks fine")])
    b = ScriptedDebater("b", [_verdict("contradicted", 0.9, rationale="evidence opposes it")])
    DebateVerifier([a, b], max_rounds=1).verify(_task(), "ans")

    # round 0 had no peer review; round 1 should show the *other* judge's stance
    assert a.peer_reviews[0] == "" and b.peer_reviews[0] == ""
    assert "contradicted" in a.peer_reviews[1] and "evidence opposes it" in a.peer_reviews[1]
    assert "[b]" in a.peer_reviews[1]
    assert "[a]" not in a.peer_reviews[1] # a does not see its own position


def test_max_rounds_zero_behaves_like_a_vote():
    a = ScriptedDebater("a", [_verdict("supported", 0.8)])
    b = ScriptedDebater("b", [_verdict("contradicted", 0.9)])
    out = DebateVerifier([a, b], max_rounds=0).verify(_task(), "ans")

    assert a.calls == 1 and b.calls == 1 # no debate rounds at all
    assert "debate_unresolved" in out.detected_issues
    assert out.verdict == "contradicted"


def test_debate_logs_each_round_when_enabled(capsys):
    a = ScriptedDebater("a", [_verdict("supported", 0.7), _verdict("contradicted", 0.9)])
    b = ScriptedDebater("b", [_verdict("contradicted", 0.9), _verdict("contradicted", 0.9)])
    DebateVerifier([a, b], max_rounds=2, log=True).verify(_task(), "ans")
    out = capsys.readouterr().out

    assert "[LLM-debate] round 0 (opening): a=supported(0.70)" in out
    assert "[no consensus]" in out # round 0 disagreed
    assert "[LLM-debate] round 1:" in out
    assert "[consensus]" in out # round 1 agreed


def test_debate_is_silent_without_log_flag(capsys):
    a = ScriptedDebater("a", [_verdict("supported", 0.9)])
    b = ScriptedDebater("b", [_verdict("supported", 0.8)])
    DebateVerifier([a, b]).verify(_task(), "ans") # log defaults to False
    assert "[LLM-debate]" not in capsys.readouterr().out
