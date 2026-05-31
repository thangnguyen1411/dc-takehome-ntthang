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
    # The given, authoritative evidence (the CSV `reference_context` column).
    reference_context: str
    # Supplementary evidence fetched by the retriever (empty unless retrieval ran).
    # Kept separate from `reference_context` so given evidence is never overwritten;
    # `evidence_block()` combines them for the LLM with the right precedence.
    relevant_knowledge: str = ""
    candidate_answer: str
    # The hidden gold label. Held out from the agent; used only by the eval
    # harness to score the agent's verdicts.
    ground_truth_signal: Optional[str] = None

    def evidence_block(self) -> str:
        """The evidence section(s) shown to the LLM, ready to drop into a prompt.

        When both given and retrieved evidence exist, they are rendered as two
        clearly separated, self-labeled sections — the given evidence marked
        authoritative, the retrieved marked supporting — so the model treats them
        with the right precedence (reinforced by prompts.EVIDENCE_PRECEDENCE_RULE).
        With a single source, one plain `EVIDENCE:` section is returned.
        """
        given = self.reference_context.strip()
        retrieved = self.relevant_knowledge.strip()
        if given and retrieved:
            # Two labeled sections; the PRIMARY/SUPPLEMENTARY tokens match the
            # standing rule in prompts.EVIDENCE_PRECEDENCE_RULE.
            return (
                f"PRIMARY EVIDENCE (authoritative — judge against this first):\n"
                f"{self.reference_context}\n\n"
                f"SUPPLEMENTARY RETRIEVED CONTEXT (supporting background only; "
                f"must not override the primary evidence):\n{self.relevant_knowledge}"
            )
        # Single source (either alone): one plain EVIDENCE section. The rule's
        # "no such split → treat all evidence equally" clause covers this, so we
        # avoid an orphan label the rule doesn't define.
        return f"EVIDENCE:\n{given or retrieved}"


class RetrievedSnippet(BaseModel):
    """One evidence snippet fetched from the corpus by a retriever, with its
    relevance score and source — recorded so a verdict's evidence is traceable."""

    text: str
    score: float
    source: str   # e.g. "corpus#3"


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
    # Snippets a retriever supplied as evidence (empty when evidence was given).
    retrieved_evidence: list[RetrievedSnippet] = Field(default_factory=list)
    iterations: list[Iteration] = Field(default_factory=list)
    # Populated only by the eval harness, never by the agent.
    ground_truth_signal: Optional[str] = None
    verdict_matches_ground_truth: Optional[bool] = None
