"""Data models shared across pipeline stages.

`Task` is the ingested input. `Verdict` is one verification result. `Iteration`
records a single pass of the refine loop. `CurationResult` is the final
machine-readable output emitted per task.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .config import VALID_VERDICTS


class Task(BaseModel):
    """One scientific QA item to curate."""

    task_id: str
    paper_title: str = ""
    domain: str = ""
    question: str
    reference_context: str
    candidate_answer: str
    # The hidden gold label. Held out from the agent; used only by the eval
    # harness to score the agent's verdicts.
    ground_truth_signal: Optional[str] = None


class RubricScores(BaseModel):
    """Orthogonal 0-1 quality axes, scored independently of the single verdict.

    The single `verdict` is one mutually-exclusive label; these axes decompose
    *why* an answer is good or bad along independent dimensions, so a reviewer
    sees, e.g., "faithful but incomplete" rather than one blurry category.
    Defaults are 1.0 so a judge (or test double) that omits them reads as clean.
    """

    # Are the answer's claims grounded in — and not contradicting — the evidence?
    faithfulness: float = Field(default=1.0, ge=0.0, le=1.0)
    # Does the answer capture the evidence's central finding (vs a minor point)?
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    # Is it appropriately precise without over-claiming (no unbacked superlatives)?
    specificity: float = Field(default=1.0, ge=0.0, le=1.0)


class Verdict(BaseModel):
    """Structured judgement of an answer against its evidence."""

    verdict: str = Field(description="supported|contradicted|hallucinated|weak_reasoning|unsupported")
    confidence: float = Field(ge=0.0, le=1.0)
    question_quality_score: float = Field(ge=0.0, le=1.0)
    # Multi-axis rubric scores complementing the single verdict label.
    rubric: RubricScores = Field(default_factory=RubricScores)
    detected_issues: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict_in_vocabulary(cls, value: str) -> str:
        if value not in VALID_VERDICTS:
            raise ValueError(
                f"verdict {value!r} not in {sorted(VALID_VERDICTS)}"
            )
        return value


class ReflectionPlan(BaseModel):
    """The Reflector's fix plan for one retry — reasoning *about* the failure,
    produced as a distinct step between judging (verify) and acting (regenerate).

    Separating "decide how to fix" from "write the fix" makes the correction
    strategy explicit and auditable rather than buried in the rewrite prompt.

    All fields default to empty so a model that returns a partial plan (e.g.
    omits `what_to_change`) degrades to a usable plan instead of crashing the
    task — the regenerator falls back to the failure rationale in that case.
    """

    diagnosis: str = ""        # what specifically went wrong in the prior answer(s)
    what_to_change: str = ""   # the concrete corrective approach for the rewrite
    what_to_keep: str = ""     # parts already correct, to preserve


class Iteration(BaseModel):
    """A single step of the self-improvement loop."""

    step: int
    answer: str
    verdict: Verdict
    refined: bool = False
    # The reflection plan that *led to* this answer (None for the initial pass,
    # which is the original candidate and was not reflected upon).
    reflection: Optional[ReflectionPlan] = None


class CurationResult(BaseModel):
    """Final per-task output. Matches the assignment's required schema."""

    task_id: str
    question_quality_score: float
    answer_confidence: float
    detected_issues: list[str]
    agent_reasoning_summary: str
    refinement_applied: bool
    final_answer: str
    evidence: list[str]
    final_verdict: str
    # Multi-axis rubric scores from the final verdict (decomposed quality view).
    rubric_scores: RubricScores = Field(default_factory=RubricScores)
    iterations: list[Iteration] = Field(default_factory=list)
    # Populated only by the eval harness, never by the agent.
    ground_truth_signal: Optional[str] = None
    verdict_matches_ground_truth: Optional[bool] = None
