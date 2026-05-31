"""Tests for the triage / ranked review queue. Pure logic, no API key."""

import csv
import io

from curation_agent.models import CurationResult
from curation_agent.triage import (
    prioritize,
    render_csv,
    render_markdown,
    render_summary,
    score_result,
)


def _result(task_id, verdict="supported", confidence=1.0, issues=None) -> CurationResult:
    return CurationResult(
        task_id=task_id,
        question_quality_score=1.0,
        answer_confidence=confidence,
        detected_issues=issues or [],
        agent_reasoning_summary="",
        refinement_applied=False,
        final_answer="ans",
        evidence=["e"],
        final_verdict=verdict,
    )


def test_clean_supported_high_confidence_is_low_priority():
    item = score_result(_result("t", verdict="supported", confidence=1.0))
    assert item.priority_score == 0
    assert item.level == "LOW"
    assert item.reasons == []


def test_bad_verdict_raises_score_and_reasons():
    item = score_result(_result("t", verdict="contradicted", confidence=0.9))
    # 30 (contradicted) + confidence penalty round(0.1*20)=2
    assert item.priority_score == 32
    assert item.level == "MEDIUM"
    assert "final verdict 'contradicted'" in item.reasons


def test_issue_tags_add_risk_and_explanations():
    item = score_result(_result("t", verdict="supported", confidence=1.0,
                                issues=["panel_disagreement", "hallucinated_citation"]))
    assert item.priority_score == 35 + 30 # both issue weights, no penalty
    assert item.level == "HIGH"
    assert "judges split on the verdict" in item.reasons
    assert "cites a reference absent from the evidence" in item.reasons


def test_low_confidence_adds_penalty():
    clean = score_result(_result("t", confidence=1.0)).priority_score
    unsure = score_result(_result("t", confidence=0.0)).priority_score
    assert unsure - clean == 20  # full penalty at zero confidence


def test_unknown_issue_tag_scores_zero_but_is_ignored_gracefully():
    item = score_result(_result("t", issues=["some_future_tag"]))
    assert item.priority_score == 0 # unknown tag adds nothing
    assert item.reasons == [] # and no fabricated reason


def test_prioritize_sorts_worst_first():
    results = [
        _result("clean", verdict="supported", confidence=1.0),
        _result("worst", verdict="contradicted", confidence=0.5, issues=["debate_unresolved"]),
        _result("mid", verdict="supported", confidence=1.0, issues=["incomplete"]),
    ]
    ranked = [it.task_id for it in prioritize(results)]
    assert ranked == ["worst", "mid", "clean"]


def test_prioritize_tie_break_is_stable_by_confidence_then_id():
    # same score (clean supported), tie broken by lower confidence then task_id
    results = [
        _result("b", confidence=0.9),
        _result("a", confidence=0.9),
        _result("c", confidence=0.5),
    ]
    ranked = [it.task_id for it in prioritize(results)]
    assert ranked == ["c", "a", "b"]


def test_render_markdown_lists_counts_and_rows():
    results = [
        _result("hi", verdict="contradicted", issues=["panel_disagreement"]),
        _result("lo", verdict="supported", confidence=1.0),
    ]
    md = render_markdown(prioritize(results))
    assert "# Review Queue" in md
    assert "HIGH: **1**" in md
    assert "| 1 | hi |" in md # worst-first: hi ranked #1
    assert "| 2 | lo |" in md


def test_summary_lists_only_flagged_top_n():
    results = [
        _result("hi", verdict="contradicted", issues=["panel_disagreement"]),
        _result("mid", verdict="supported", confidence=1.0, issues=["incomplete"]),
        _result("clean", verdict="supported", confidence=1.0),
    ]
    summary = render_summary(prioritize(results), top_n=2)
    assert "## Top 2 review-queue items" in summary
    assert "**hi**" in summary and "**mid**" in summary
    assert "**clean**" not in summary # score 0 -> not flagged


def test_summary_empty_when_nothing_flagged():
    results = [_result("a", confidence=1.0), _result("b", confidence=1.0)]
    assert render_summary(prioritize(results)) == "" # clean batch adds no noise


def test_csv_has_header_and_quotes_reason_commas():
    results = [_result("hi", verdict="contradicted", issues=["panel_disagreement"])]
    text = render_csv(prioritize(results))
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["rank", "task_id", "priority_score", "level", "verdict", "confidence", "reasons"]
    assert rows[1][1] == "hi" and rows[1][3] == "HIGH"
    # reasons cell holds the semicolon-joined explanations as a single field
    assert "judges split on the verdict" in rows[1][6]
