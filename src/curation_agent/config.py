"""Central configuration for the curation pipeline.

Thresholds and model choices live here so a reviewer can tune behaviour in one
place without reading the orchestration code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default model + API-key env var per provider, used when the CLI doesn't
# override the model explicitly.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",
}
API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class Config:
    provider: str = "anthropic"
    # None means "use the provider's default model" (see resolved_model).
    model: str | None = None
    # Verifier panel: each entry is a "provider:model" spec. Empty = single-verifier
    # over the primary client (default). A panel of >1 runs independent
    # verifications and majority-votes the verdict; a cross-provider panel
    # (e.g. anthropic + openai) reduces shared-blind-spot risk.
    verifier_panel: tuple[str, ...] = ()
    # Below this confidence the refine loop is triggered even if the verdict
    # itself looks acceptable.
    confidence_threshold: float = 0.75
    # Below this question_quality_score the question is flagged weak/ambiguous.
    question_quality_threshold: float = 0.5
    # Below this, a rubric axis (faithfulness/completeness/specificity) is flagged.
    rubric_threshold: float = 0.6
    # Hard cap on critic -> regenerate -> re-verify cycles per task. Set to 3 so
    # the escalating retry prompts (see Refiner.ESCALATION) can actually exercise
    # their firmer tiers on hard cases.
    max_refine_iterations: int = 3
    max_tokens: int = 1024
    # When True, each client prints every LLM request + response, tagged by
    # provider:model (see log_llm_call). Enabled via --llm-log; verbose and noisy.
    llm_log: bool = False
    # Retrieval (RAG): when True, every task's reference_context is augmented with
    # relevant snippets from the corpus at `corpus_path` (given evidence is kept,
    # retrieved snippets appended). A snippet is only used if its relevance score
    # clears `retrieval_threshold`; a task that had no evidence and gets no
    # qualifying snippet is tagged `retrieval_failed` so the verdict stays honest.
    retrieve: bool = False
    corpus_path: str = "data/corpus"
    retrieval_threshold: float = 0.25
    # How many corpus snippets to pull per task (those clearing the threshold).
    retrieval_top_k: int = 2

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    @property
    def api_key_env(self) -> str:
        return API_KEY_ENV[self.provider]

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))


# Verdicts that mean the candidate answer is not faithful to the evidence and
# should trigger a refinement attempt.
BAD_VERDICTS: frozenset[str] = frozenset(
    {"contradicted", "hallucinated", "weak_reasoning", "unsupported"}
)

VALID_VERDICTS: frozenset[str] = BAD_VERDICTS | {"supported"}
