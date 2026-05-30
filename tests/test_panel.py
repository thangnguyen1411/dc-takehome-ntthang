"""Tests for the cross-checking PanelVerifier — no API key, fully deterministic."""

from pytest import approx

from curation_agent.models import Task, Verdict
from curation_agent.verifier import PanelVerifier


def _task() -> Task:
    return Task(
        task_id="task_p",
        question="Which biomarker?",
        reference_context="Biomarker CYP2E1 was elevated.",
        candidate_answer="CYP2E1 was elevated.",
    )


class FixedVerifier:
    """A stand-in verifier that always returns a preset Verdict, with a name."""

    def __init__(self, name: str, verdict: Verdict) -> None:
        self.name = name
        self._verdict = verdict

    def verify(self, task, answer) -> Verdict:
        return self._verdict


def _verdict(label, conf, rubric=(1.0, 1.0, 1.0), issues=None, rationale="") -> Verdict:
    f, c, s = rubric
    return Verdict(
        verdict=label,
        confidence=conf,
        question_quality_score=0.8,
        rubric={"faithfulness": f, "completeness": c, "specificity": s},
        detected_issues=issues or [],
        rationale=rationale,
    )


def test_panel_unanimous_agreement():
    panel = PanelVerifier([
        FixedVerifier("anthropic", _verdict("supported", 0.9)),
        FixedVerifier("openai", _verdict("supported", 0.8)),
    ])
    out = panel.verify(_task(), "ans")
    assert out.verdict == "supported"
    assert out.confidence == approx(0.85)  # mean of agreeing voters
    assert "panel_disagreement" not in out.detected_issues


def test_panel_disagreement_prefers_flagging_and_tags():
    # one says supported, one says contradicted -> tie -> prefer the flag
    panel = PanelVerifier([
        FixedVerifier("anthropic", _verdict("supported", 0.7)),
        FixedVerifier("openai", _verdict("contradicted", 0.9)),
    ])
    out = panel.verify(_task(), "ans")
    assert out.verdict == "contradicted"          # conservative: flag over supported
    assert "panel_disagreement" in out.detected_issues
    assert out.confidence == 0.9                   # mean of voters who chose 'contradicted'


def test_panel_majority_wins():
    panel = PanelVerifier([
        FixedVerifier("a", _verdict("supported", 0.9)),
        FixedVerifier("b", _verdict("supported", 0.8)),
        FixedVerifier("c", _verdict("contradicted", 0.95)),
    ])
    out = panel.verify(_task(), "ans")
    assert out.verdict == "supported"              # 2 vs 1
    assert "panel_disagreement" in out.detected_issues  # still not unanimous


def test_panel_averages_rubric_and_unions_issues():
    panel = PanelVerifier([
        FixedVerifier("a", _verdict("supported", 0.9, rubric=(1.0, 1.0, 0.8), issues=["x"])),
        FixedVerifier("b", _verdict("supported", 0.9, rubric=(0.6, 1.0, 0.4), issues=["y"])),
    ])
    out = panel.verify(_task(), "ans")
    assert out.rubric.faithfulness == approx(0.8)  # mean(1.0, 0.6)
    assert out.rubric.specificity == approx(0.6)   # mean(0.8, 0.4)
    assert set(out.detected_issues) >= {"x", "y"}  # union of both


def test_panel_of_one_passes_through():
    only = _verdict("hallucinated", 0.77, issues=["hallucinated"])
    panel = PanelVerifier([FixedVerifier("solo", only)])
    out = panel.verify(_task(), "ans")
    assert out.verdict == "hallucinated"
    assert out.confidence == 0.77
    assert "panel_disagreement" not in out.detected_issues
