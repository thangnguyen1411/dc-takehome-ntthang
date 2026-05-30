"""Thin Anthropic client wrapper with structured (tool-forced) JSON output.

The wrapper exposes one method, `complete_structured`, that forces the model to return an
object matching a JSON schema via tool use, then validates it. It is kept behind
the `LLMClient` Protocol so tests can substitute their own scripted doubles.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        ...


class AnthropicLLM:
    """Calls the real Anthropic API and forces a single tool call as output.

    Conforms to `LLMClient` structurally (via the `structured` method) rather
    than by inheritance.
    """

    def __init__(self, model: str, max_tokens: int) -> None:
        from anthropic import Anthropic  # imported lazily so offline mode needs no SDK

        self._client = Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        tool = {
            "name": tool_name,
            "description": "Return the structured result.",
            "input_schema": schema,
            # The tool schema is also identical across tasks; caching it extends
            # the cached prefix (system + tools) at no extra cost.
            "cache_control": {"type": "ephemeral"},
        }
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            # Cache the static rubric system prompt: it is identical across all
            # 10 tasks, so this turns 10 full prompt reads into 1 + 9 cache hits.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input)
        raise RuntimeError(f"model did not call tool {tool_name!r}")


def build_client(config) -> AnthropicLLM:
    """Return the real Anthropic client, requiring an API key to be present.

    The return type is the concrete `AnthropicLLM`, which satisfies the
    `LLMClient` Protocol that downstream stages depend on.
    """
    if not config.has_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment or a .env "
            "file (see .env.example) before running the curation agent."
        )
    return AnthropicLLM(config.model, config.max_tokens)
