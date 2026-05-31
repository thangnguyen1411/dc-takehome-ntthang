"""Stage 6: triage — rank curated results into a human review queue.

The project's goal is that a curator reviews a *ranked, annotated* queue rather
than the raw batch. This module folds every risk signal the pipeline already
produced — the final verdict, the agent's confidence, the detected-issue tags,
and the rubric axes — into one transparent `priority_score`, then sorts tasks
worst-first. It is pure post-processing: no LLM calls, and it does not need the
held-out gold labels, so the queue is meaningful even on unlabeled data.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .models import CurationResult

# Base risk per FINAL verdict. After refinement most items are `supported`, so
# this mostly separates the genuinely-unfixable ones; an unknown label is
# treated as mildly risky (10) rather than ignored.
_VERDICT_RISK: dict[str, int] = {
    "error": 100,
    "contradicted": 30,
    "hallucinated": 30,
    "unsupported": 20,
    "weak_reasoning": 15,
    "supported": 0,
}

# Extra risk per detected-issue tag. Judge disagreement ranks highest: it is the
# strongest "a human must decide" signal. Tags not listed add 0 to the score
# (they may still appear as reasons).
_ISSUE_RISK: dict[str, int] = {
    "processing_error": 100,
    "debate_unresolved": 40,
    "panel_disagreement": 35,
    "hallucinated_citation": 30,
    "low_faithfulness": 25,
    "missing_evidence": 25,
    "retrieval_failed": 20,
    "overclaim": 15,
    "low_confidence": 15,
    "incomplete": 10,
    "weak_question": 10,
}

# Plain-English explanation per tag, shown in the queue so a curator sees *why*
# an item is flagged without opening the per-task JSON.
_REASON: dict[str, str] = {
    "processing_error": "curation failed to run",
    "debate_unresolved": "judges could not reach consensus",
    "panel_disagreement": "judges split on the verdict",
    "hallucinated_citation": "cites a reference absent from the evidence",
    "low_faithfulness": "claims weakly grounded in the evidence",
    "missing_evidence": "no evidence to judge against",
    "retrieval_failed": "retrieval found nothing relevant",
    "overclaim": "answer over-claims (low specificity)",
    "low_confidence": "agent still unsure after refinement",
    "incomplete": "misses the evidence's central finding",
    "weak_question": "question is vague or unanswerable",
}

# Score cutoffs for a coarse triage level.
_HIGH = 40
_MEDIUM = 15


@dataclass
class ReviewItem:
    task_id: str
    priority_score: int
    level: str  # HIGH | MEDIUM | LOW
    verdict: str
    confidence: float
    reasons: list[str]


def _confidence_penalty(confidence: float) -> int:
    """Lower confidence -> more risk: 0 at full confidence, up to 20 at zero."""
    return round((1.0 - confidence) * 20)


def score_result(result: CurationResult) -> ReviewItem:
    """Compute a transparent priority_score + human-readable reasons for one task."""
    score = _VERDICT_RISK.get(result.final_verdict, 10)
    score += sum(_ISSUE_RISK.get(issue, 0) for issue in result.detected_issues)
    score += _confidence_penalty(result.answer_confidence)

    reasons: list[str] = []
    if result.final_verdict not in ("supported", ""):
        reasons.append(f"final verdict '{result.final_verdict}'")
    reasons += [_REASON[issue] for issue in result.detected_issues if issue in _REASON]

    level = "HIGH" if score >= _HIGH else "MEDIUM" if score >= _MEDIUM else "LOW"
    return ReviewItem(
        task_id=result.task_id,
        priority_score=score,
        level=level,
        verdict=result.final_verdict,
        confidence=result.answer_confidence,
        reasons=reasons,
    )


def prioritize(results: list[CurationResult]) -> list[ReviewItem]:
    """Rank results worst-first (highest priority_score).

    Ties broken by lower confidence, then task_id, so the ordering is stable and
    reproducible across runs.
    """
    items = [score_result(r) for r in results]
    items.sort(key=lambda it: (-it.priority_score, it.confidence, it.task_id))
    return items


def render_markdown(items: list[ReviewItem]) -> str:
    """Render the ranked queue as a markdown table, highest-priority first."""
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for it in items:
        counts[it.level] += 1

    lines = [
        "# Review Queue",
        "",
        "Tasks ranked by `priority_score` (highest risk first). Work HIGH before "
        "MEDIUM/LOW. Score = final-verdict risk + detected-issue risk + a "
        "low-confidence penalty.",
        "",
        f"- HIGH: **{counts['HIGH']}**  |  MEDIUM: **{counts['MEDIUM']}**  |  LOW: **{counts['LOW']}**",
        "",
        "| # | task | priority | level | verdict | conf | why |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, it in enumerate(items, start=1):
        why = "; ".join(it.reasons) if it.reasons else "—"
        lines.append(
            f"| {rank} | {it.task_id} | {it.priority_score} | {it.level} "
            f"| {it.verdict} | {it.confidence:.2f} | {why} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_summary(items: list[ReviewItem], top_n: int = 3) -> str:
    """A compact 'top N to review' block for embedding in the main report.

    Lists only the highest-priority items that actually carry risk (score > 0);
    returns '' when nothing is flagged, so a clean batch adds no noise.
    """
    flagged = [it for it in items if it.priority_score > 0][:top_n]
    if not flagged:
        return ""
    lines = [
        f"## Top {len(flagged)} review-queue items",
        "",
        "Highest-priority tasks for human review (full ranking in the review queue):",
        "",
    ]
    for rank, it in enumerate(flagged, start=1):
        why = "; ".join(it.reasons) if it.reasons else "—"
        lines.append(f"{rank}. **{it.task_id}** ({it.level}, score {it.priority_score}) — {why}")
    lines.append("")
    return "\n".join(lines)


def render_csv(items: list[ReviewItem]) -> str:
    """The ranked queue as CSV, for spreadsheet triage. `reasons` is one cell
    (semicolon-joined); csv quoting handles any commas inside it."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "task_id", "priority_score", "level", "verdict", "confidence", "reasons"])
    for rank, it in enumerate(items, start=1):
        writer.writerow([
            rank, it.task_id, it.priority_score, it.level, it.verdict,
            f"{it.confidence:.2f}", "; ".join(it.reasons),
        ])
    return buffer.getvalue()
