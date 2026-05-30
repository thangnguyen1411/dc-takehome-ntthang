"""Central configuration for the curation pipeline.

Thresholds and model choices live here so a reviewer can tune behaviour in one
place without reading the orchestration code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model: str = "claude-opus-4-8"
    # Below this confidence the refine loop is triggered even if the verdict
    # itself looks acceptable.
    confidence_threshold: float = 0.75
    # Below this question_quality_score the question is flagged weak/ambiguous.
    question_quality_threshold: float = 0.5
    # Hard cap on critic -> regenerate -> re-verify cycles per task.
    max_refine_iterations: int = 2
    max_tokens: int = 1024

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


# Verdicts that mean the candidate answer is not faithful to the evidence and
# should trigger a refinement attempt.
BAD_VERDICTS: frozenset[str] = frozenset(
    {"contradicted", "hallucinated", "weak_reasoning", "unsupported"}
)

VALID_VERDICTS: frozenset[str] = BAD_VERDICTS | {"supported"}
