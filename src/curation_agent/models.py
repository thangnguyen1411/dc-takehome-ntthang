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


class Verdict(BaseModel):
    """Structured judgement of an answer against its evidence."""

    verdict: str = Field(description="supported|contradicted|hallucinated|weak_reasoning|unsupported")
    confidence: float = Field(ge=0.0, le=1.0)
    question_quality_score: float = Field(ge=0.0, le=1.0)
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


class Iteration(BaseModel):
    """A single step of the self-improvement loop."""

    step: int
    answer: str
    verdict: Verdict
    refined: bool = False


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
    iterations: list[Iteration] = Field(default_factory=list)
    # Populated only by the eval harness, never by the agent.
    ground_truth_signal: Optional[str] = None
    verdict_matches_ground_truth: Optional[bool] = None
