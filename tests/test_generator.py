"""Tests for the benchmark generator - no API key, fully deterministic."""

import pytest

from curation_agent.generator import FLAW_INSTRUCTIONS, BenchmarkGenerator
from curation_agent.models import Task


class ScriptedGenLLM:
    """Returns a fixed question/answer and records the prompt it was given."""

    def __init__(self):
        self.last_prompt = ""
        self.last_tool = ""

    def complete_structured(self, *, system, prompt, schema, tool_name):
        self.last_prompt = prompt
        self.last_tool = tool_name
        return {
            "paper_title": "A study of Marker X",
            "domain": "oncology",
            "question": "Which marker was elevated?",
            "candidate_answer": "Marker X was elevated.",
        }


EVIDENCE = "Marker X was elevated in the treated cohort."


def test_generate_returns_task_labeled_with_target_verdict():
    gen = BenchmarkGenerator(ScriptedGenLLM())
    task = gen.generate("gen_001", EVIDENCE, "contradicted", domain="oncology")

    assert isinstance(task, Task)
    assert task.task_id == "gen_001"
    assert task.reference_context == EVIDENCE
    assert task.domain == "oncology"
    # the target verdict becomes the held-out gold label
    assert task.ground_truth_signal == "contradicted"
    assert task.question and task.candidate_answer


def test_generate_uses_the_emit_task_tool_and_includes_target_in_prompt():
    client = ScriptedGenLLM()
    BenchmarkGenerator(client).generate("gen_002", EVIDENCE, "hallucinated")
    assert client.last_tool == "emit_task"
    assert "TARGET: hallucinated" in client.last_prompt
    assert EVIDENCE in client.last_prompt
    # the flaw instruction for that target is passed to the model
    assert FLAW_INSTRUCTIONS["hallucinated"] in client.last_prompt


def test_unknown_target_verdict_raises():
    gen = BenchmarkGenerator(ScriptedGenLLM())
    with pytest.raises(ValueError):
        gen.generate("gen_003", EVIDENCE, "definitely_not_a_verdict")


def test_every_taxonomy_label_is_generatable():
    # the generator must cover the full verdict taxonomy the verifier can emit
    gen = BenchmarkGenerator(ScriptedGenLLM())
    for label in ("supported", "contradicted", "hallucinated", "weak_reasoning", "unsupported"):
        task = gen.generate("gen_x", EVIDENCE, label)
        assert task.ground_truth_signal == label


def test_generated_task_round_trips_through_validation():
    # a generated Task is a real, valid Task usable by the pipeline/eval
    gen = BenchmarkGenerator(ScriptedGenLLM())
    task = gen.generate("gen_004", EVIDENCE, "supported")
    # re-validate via pydantic to confirm it satisfies the model contract
    assert Task(**task.model_dump()).ground_truth_signal == "supported"


def test_read_evidence_skips_comments_and_blanks(tmp_path):
    from curation_agent.generate import _read_evidence

    f = tmp_path / "ev.txt"
    f.write_text(
        "# a comment\n"
        "\n"
        "First real snippet.\n"
        "   # indented comment\n"
        "Second real snippet.\n",
        encoding="utf-8",
    )
    assert _read_evidence(f) == ["First real snippet.", "Second real snippet."]


def test_bundled_evidence_file_is_readable():
    # the shipped data/evidence.txt parses to real snippets (comments stripped)
    from pathlib import Path

    from curation_agent.generate import _read_evidence

    snippets = _read_evidence(Path(__file__).resolve().parents[1] / "data" / "evidence.txt")
    assert len(snippets) >= 14
    assert all(not s.startswith("#") for s in snippets)
