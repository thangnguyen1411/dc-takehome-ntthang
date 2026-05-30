"""Tests for the escalating, history-aware retry loop - no API key needed."""

import dataclasses

from curation_agent.config import Config
from curation_agent.models import Task
from curation_agent.reflector import Reflector
from curation_agent.refiner import Refiner
from curation_agent.verifier import Verifier


def _task() -> Task:
    return Task(
        task_id="task_r",
        question="Which biomarker?",
        reference_context="Biomarker CYP2E1 was elevated.",
        candidate_answer="Troponin-I was the biomarker.",
    )


def _bad(label="contradicted"):
    return {
        "verdict": label,
        "confidence": 0.9,
        "question_quality_score": 0.8,
        "detected_issues": [label],
        "rationale": f"answer is {label}",
    }


def _good():
    return {
        "verdict": "supported",
        "confidence": 0.95,
        "question_quality_score": 0.8,
        "detected_issues": [],
        "rationale": "now grounded",
    }


class RecordingLLM:
    """Fails the first two verifications, then succeeds. Records every regen prompt
    so we can assert the retry prompt carries history + escalates."""

    def __init__(self):
        self.verify_calls = 0
        self.regen_prompts: list[str] = []
        self.regen_count = 0

    def complete_structured(self, *, system, prompt, schema, tool_name):
        if tool_name == "emit_reflection":
            return {"diagnosis": "wrong marker named", "what_to_change": "ground in CYP2E1"}
        if tool_name == "emit_answer":
            self.regen_count += 1
            self.regen_prompts.append(prompt)
            return {"answer": f"attempt-{self.regen_count} answer"}
        self.verify_calls += 1
        # verify #1 and #2 fail (with distinct labels), #3 passes
        if self.verify_calls == 1:
            return _bad("contradicted")
        if self.verify_calls == 2:
            return _bad("unsupported")
        return _good()


def _refiner(client) -> Refiner:
    config = dataclasses.replace(Config(), max_refine_iterations=3)
    return Refiner(client, Verifier(client), Reflector(client), config)


def test_retry_accumulates_failure_history():
    client = RecordingLLM()
    iterations = _refiner(client).refine(_task())

    # 3 iterations: initial (contradicted) -> retry (unsupported) -> retry (supported)
    assert [it.verdict.verdict for it in iterations] == ["contradicted", "unsupported", "supported"]

    # second regen prompt must reference BOTH prior attempts (the candidate and
    # the first regenerated answer), proving history accumulates.
    second_prompt = client.regen_prompts[1]
    assert "Troponin-I was the biomarker." in second_prompt   # original candidate
    assert "attempt-1 answer" in second_prompt                # first regeneration
    assert "contradicted" in second_prompt and "unsupported" in second_prompt


def test_retry_prompt_escalates_in_strictness():
    client = RecordingLLM()
    _refiner(client).refine(_task())

    first, second = client.regen_prompts[0], client.regen_prompts[1]
    # tier 1 vs tier 2 wording differs and gets firmer
    assert Refiner.ESCALATION[0] in first
    assert Refiner.ESCALATION[1] in second
    assert first != second


def test_escalation_clamps_to_strongest_tier():
    r = _refiner(RecordingLLM())
    # attempts beyond the number of tiers reuse the last (strongest) tier
    assert r._escalation_for(99) == Refiner.ESCALATION[-1]
    assert r._escalation_for(1) == Refiner.ESCALATION[0]


class PartialReflectionLLM(RecordingLLM):
    """Like RecordingLLM but the reflector omits `what_to_change` (and even
    diagnosis on later calls) — simulating a model returning a partial plan."""

    def complete_structured(self, *, system, prompt, schema, tool_name):
        if tool_name == "emit_reflection":
            return {"diagnosis": "only a diagnosis, no change field"}
        return super().complete_structured(
            system=system, prompt=prompt, schema=schema, tool_name=tool_name
        )


def test_partial_reflection_does_not_crash_and_degrades():
    # A reflection missing `what_to_change` must NOT raise; the task still runs
    # and the regen prompt omits the missing field instead of failing.
    client = PartialReflectionLLM()
    iterations = _refiner(client).refine(_task())

    assert [it.verdict.verdict for it in iterations] == ["contradicted", "unsupported", "supported"]
    # plan is still recorded, with the field that was provided
    assert iterations[1].reflection.diagnosis == "only a diagnosis, no change field"
    assert iterations[1].reflection.what_to_change == ""
    # the prompt shows the diagnosis but no "what to change" line
    assert "diagnosis: only a diagnosis" in client.regen_prompts[0]
    assert "what to change:" not in client.regen_prompts[0]


def test_empty_reflection_omits_fix_plan_block():
    from curation_agent.models import ReflectionPlan

    assert Refiner._fix_plan_block(ReflectionPlan()) == ""
    block = Refiner._fix_plan_block(ReflectionPlan(diagnosis="d"))
    assert "FIX PLAN:" in block and "diagnosis: d" in block


def test_reflection_is_produced_and_fed_into_regeneration():
    client = RecordingLLM()
    iterations = _refiner(client).refine(_task())

    # the initial pass has no reflection; every retry does
    assert iterations[0].reflection is None
    assert all(it.reflection is not None for it in iterations[1:])
    assert iterations[1].reflection.diagnosis == "wrong marker named"

    # the reflection plan is injected into the regeneration prompt (reflect -> act)
    assert "FIX PLAN:" in client.regen_prompts[0]
    assert "ground in CYP2E1" in client.regen_prompts[0]
