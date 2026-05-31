"""The clients must forward a sampling temperature to the API (default 0 for
reproducible verification). Uses a fake SDK client; no network or key."""

import types

from curation_agent.llm import AnthropicLLM, OpenAILLM


def _fake_anthropic_response():
    block = types.SimpleNamespace(type="tool_use", name="emit", input={"ok": 1})
    return types.SimpleNamespace(content=[block])


def _fake_openai_response():
    call = types.SimpleNamespace(function=types.SimpleNamespace(arguments='{"ok": 1}'))
    message = types.SimpleNamespace(tool_calls=[call])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _patch_anthropic(monkeypatch, captured):
    client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=lambda **kw: (captured.update(kw), _fake_anthropic_response())[1]
        )
    )
    monkeypatch.setattr(
        AnthropicLLM, "__init__",
        lambda self, *a, **k: (
            setattr(self, "_client", client),
            setattr(self, "_model", "m"),
            setattr(self, "_max_tokens", 16),
            setattr(self, "_log", False),
            setattr(self, "_temperature", k.get("temperature", 0.0)),
        ) and None,
    )


def _patch_openai(monkeypatch, captured):
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: (captured.update(kw), _fake_openai_response())[1]
            )
        )
    )
    monkeypatch.setattr(
        OpenAILLM, "__init__",
        lambda self, *a, **k: (
            setattr(self, "_client", client),
            setattr(self, "_model", "m"),
            setattr(self, "_max_tokens", 16),
            setattr(self, "_log", False),
            setattr(self, "_temperature", k.get("temperature", 0.0)),
        ) and None,
    )


def test_anthropic_defaults_to_temperature_zero(monkeypatch):
    captured = {}
    _patch_anthropic(monkeypatch, captured)
    AnthropicLLM("m", 16).complete_structured(system="s", prompt="p", schema={}, tool_name="emit")
    assert captured["temperature"] == 0.0


def test_anthropic_forwards_custom_temperature(monkeypatch):
    captured = {}
    _patch_anthropic(monkeypatch, captured)
    AnthropicLLM("m", 16, temperature=0.7).complete_structured(
        system="s", prompt="p", schema={}, tool_name="emit"
    )
    assert captured["temperature"] == 0.7


def test_openai_defaults_to_temperature_zero(monkeypatch):
    captured = {}
    _patch_openai(monkeypatch, captured)
    OpenAILLM("m", 16).complete_structured(system="s", prompt="p", schema={}, tool_name="emit")
    assert captured["temperature"] == 0.0
