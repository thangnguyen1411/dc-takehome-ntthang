"""Tests for stripping leaked tool-call markup from LLM output (no API key)."""

from curation_agent.llm import _strip_tool_artifacts


def test_strips_trailing_invoke_and_parameter_tags():
    # the exact shape seen in a real run: answer text + leaked closing tags
    raw = {"answer": "Statins inhibit HMG-CoA reductase.</parameter>\n</invoke>\n"}
    assert _strip_tool_artifacts(raw) == {"answer": "Statins inhibit HMG-CoA reductase."}


def test_strips_mid_string_parameter_tags():
    # a reflection that crammed all fields into one with literal <parameter> tags
    raw = {"diagnosis": "Wrong direction.</parameter>\n<parameter name=\"what_to_change\">Reverse it."}
    cleaned = _strip_tool_artifacts(raw)
    assert "</parameter>" not in cleaned["diagnosis"]
    assert "<parameter" not in cleaned["diagnosis"]
    assert "Wrong direction." in cleaned["diagnosis"]
    assert "Reverse it." in cleaned["diagnosis"]


def test_preserves_legitimate_angle_brackets():
    # scientific text with comparison operators must be untouched
    raw = {"rationale": "The effect was significant (p < 0.05) and HR > 1.2."}
    assert _strip_tool_artifacts(raw) == raw


def test_recurses_into_nested_structures():
    raw = {
        "verdict": "supported",
        "detected_issues": ["a</parameter>", "b"],
        "nested": {"note": "ok</invoke>"},
    }
    cleaned = _strip_tool_artifacts(raw)
    assert cleaned["detected_issues"] == ["a", "b"]
    assert cleaned["nested"]["note"] == "ok"


def test_non_string_values_pass_through():
    raw = {"confidence": 0.9, "flag": True, "count": 3}
    assert _strip_tool_artifacts(raw) == raw
