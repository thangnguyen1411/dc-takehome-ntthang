"""Tests for the --llm-log terminal logging helper — no API key needed."""

from curation_agent.llm import log_llm_call


def test_log_llm_call_prints_request_and_response(capsys):
    log_llm_call(
        "anthropic:claude-sonnet-4-6",
        "emit_verdict",
        "QUESTION: q\nEVIDENCE: e",
        {"verdict": "supported", "confidence": 0.9},
    )
    out = capsys.readouterr().out
    # every line carries provider:model + direction + tool
    assert "[LLM-anthropic:claude-sonnet-4-6 REQUEST emit_verdict] QUESTION: q" in out
    assert "[LLM-anthropic:claude-sonnet-4-6 REQUEST emit_verdict] EVIDENCE: e" in out
    assert (
        '[LLM-anthropic:claude-sonnet-4-6 RESPONSE emit_verdict] '
        '{"verdict": "supported", "confidence": 0.9}' in out
    )


def test_every_line_is_fully_tagged_for_grepping(capsys):
    log_llm_call("openai:gpt-4o", "emit_answer", "a\nb\nc", {"answer": "x"})
    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    # 3 request lines + 1 response line, each fully self-describing
    assert all(ln.startswith("[LLM-openai:gpt-4o ") for ln in out_lines)
    assert sum("REQUEST emit_answer]" in ln for ln in out_lines) == 3
    assert sum("RESPONSE emit_answer]" in ln for ln in out_lines) == 1
