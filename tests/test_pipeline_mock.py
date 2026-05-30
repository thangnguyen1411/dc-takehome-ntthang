"""Pipeline tests using a scripted fake LLM — no API key, fully deterministic."""

from curation_agent.config import Config
from curation_agent.evaluate import Evaluator
from curation_agent.models import Task
from curation_agent.pipeline import Pipeline


class ScriptedLLM:
    """Returns 'contradicted' once, then 'supported' on regeneration.

    Exercises the full critic -> regenerate -> re-verify loop deterministically.
    """

    def __init__(self):
        self.verify_calls = 0

    def complete_structured(self, *, system, prompt, schema, tool_name):
        if tool_name == "emit_answer":
            return {"answer": "Corrected answer grounded in evidence."}
        self.verify_calls += 1
        if self.verify_calls == 1:
            return {
                "verdict": "contradicted",
                "confidence": 0.9,
                "question_quality_score": 0.8,
                "detected_issues": ["contradicted"],
                "rationale": "answer opposes evidence",
            }
        return {
            "verdict": "supported",
            "confidence": 0.95,
            "question_quality_score": 0.8,
            "detected_issues": [],
            "rationale": "now matches evidence",
        }


def _task(gt="contradicted") -> Task:
    return Task(
        task_id="task_x",
        question="Which biomarker?",
        reference_context="Biomarker CYP2E1 was elevated.",
        candidate_answer="Troponin-I was the biomarker.",
        ground_truth_signal=gt,
    )


def test_refinement_loop_runs_and_improves():
    results = Pipeline.build(ScriptedLLM(), Config()).run([_task()])
    r = results[0]
    assert r.refinement_applied is True
    assert r.final_verdict == "supported"
    assert len(r.iterations) == 2
    assert r.iterations[0].verdict.verdict == "contradicted"


def test_eval_scores_initial_verdict_against_ground_truth():
    tasks = [_task(gt="contradicted")]
    results = Pipeline.build(ScriptedLLM(), Config()).run(tasks)
    report = Evaluator(results, tasks).evaluate()
    # initial verdict (contradicted) matches the gold label
    assert report.scored == 1
    assert report.correct == 1
    assert results[0].verdict_matches_ground_truth is True


def test_weak_question_and_low_confidence_are_flagged():
    class WeakQ:
        def complete_structured(self, *, system, prompt, schema, tool_name):
            if tool_name == "emit_answer":
                return {"answer": "still cannot answer from the given evidence"}
            return {
                "verdict": "unsupported",
                "confidence": 0.3,
                "question_quality_score": 0.2,
                "detected_issues": [],
                "rationale": "vague question, evidence does not address it",
            }

    results = Pipeline.build(WeakQ(), Config()).run([_task(gt="weak_reasoning")])
    issues = results[0].detected_issues
    assert "weak_question" in issues
    assert "low_confidence" in issues


def test_rubric_low_specificity_is_flagged_and_recorded():
    class OverClaim:
        def complete_structured(self, *, system, prompt, schema, tool_name):
            return {
                "verdict": "supported",
                "confidence": 0.9,
                "question_quality_score": 0.9,
                "rubric": {"faithfulness": 0.9, "completeness": 0.8, "specificity": 0.3},
                "detected_issues": [],
                "rationale": "grounded but over-claims a superlative",
            }

    results = Pipeline.build(OverClaim(), Config()).run([_task(gt="supported")])
    r = results[0]
    # low specificity (0.3 < 0.6) becomes the "overclaim" tag
    assert "overclaim" in r.detected_issues
    assert "low_faithfulness" not in r.detected_issues  # 0.9 is fine
    # rubric scores flow through to the structured output
    assert r.rubric_scores.specificity == 0.3
    assert r.rubric_scores.faithfulness == 0.9


def test_verdict_defaults_rubric_when_omitted():
    # a judge that omits `rubric` should still validate (defaults to 1.0s)
    class NoRubric:
        def complete_structured(self, *, system, prompt, schema, tool_name):
            return {
                "verdict": "supported",
                "confidence": 0.95,
                "question_quality_score": 0.9,
                "detected_issues": [],
                "rationale": "matches",
            }

    results = Pipeline.build(NoRubric(), Config()).run([_task(gt="supported")])
    assert results[0].rubric_scores.specificity == 1.0
    assert "overclaim" not in results[0].detected_issues


def test_supported_answer_skips_refinement():
    class AllGood:
        def complete_structured(self, *, system, prompt, schema, tool_name):
            return {
                "verdict": "supported",
                "confidence": 0.95,
                "question_quality_score": 0.9,
                "detected_issues": [],
                "rationale": "matches",
            }

    results = Pipeline.build(AllGood(), Config()).run([_task(gt="supported")])
    assert results[0].refinement_applied is False
    assert len(results[0].iterations) == 1
