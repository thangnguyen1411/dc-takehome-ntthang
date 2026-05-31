"""LLM client wrappers with structured (tool-forced) JSON output.

Each wrapper exposes one method, `complete_structured`, that forces the model to
return an object matching a JSON schema via tool/function calling, then returns
it. Both Anthropic and OpenAI implementations conform to the `LLMClient`
Protocol structurally, so the pipeline is provider-agnostic and tests can
substitute their own scripted doubles.
"""

from __future__ import annotations

import json
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


def log_llm_call(label: str, tool_name: str, prompt: str, result: dict[str, Any]) -> None:
    """Print one LLM request + response to the terminal.

    Every line carries the full tag `[LLM-<provider:model> <REQUEST|RESPONSE> <tool>]`
    so each line is self-describing and greppable (e.g. filter by provider, by
    direction, or by tool). Called from inside each client (the only place the raw
    prompt/response exist) when logging is enabled via the CLI --llm-log flag.
    """
    req = f"[LLM-{label} REQUEST {tool_name}]"
    res = f"[LLM-{label} RESPONSE {tool_name}]"
    for line in prompt.splitlines():
        print(f"{req} {line}", flush=True)
    print(f"{res} {json.dumps(result, ensure_ascii=False)}", flush=True)


class AnthropicLLM:
    """Calls the real Anthropic API and forces a single tool call as output.

    Conforms to `LLMClient` structurally (via the `structured` method) rather
    than by inheritance.
    """

    def __init__(self, model: str, max_tokens: int, log: bool = False) -> None:
        from anthropic import Anthropic  # imported lazily so offline mode needs no SDK

        self._client = Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._log = log

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
                result = dict(block.input)
                if self._log:
                    log_llm_call(f"anthropic:{self._model}", tool_name, prompt, result)
                return result
        raise RuntimeError(f"model did not call tool {tool_name!r}")


class OpenAILLM:
    """Calls the OpenAI API and forces a single function call as output.

    Conforms to `LLMClient` structurally. Uses OpenAI function calling: the JSON
    schema is the function's parameters, and the model is forced to call it, so
    the returned arguments are the structured result.
    """

    def __init__(self, model: str, max_tokens: int, log: bool = False) -> None:
        from openai import OpenAI  # imported lazily so Anthropic-only runs need no SDK

        self._client = OpenAI()
        self._model = model
        self._max_tokens = max_tokens
        self._log = log

    def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "Return the structured result.",
                "parameters": schema,
            },
        }
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        calls = response.choices[0].message.tool_calls
        if not calls:
            raise RuntimeError(f"model did not call function {tool_name!r}")
        result = json.loads(calls[0].function.arguments)
        if self._log:
            log_llm_call(f"openai:{self._model}", tool_name, prompt, result)
        return result


def build_client(config) -> LLMClient:
    """Return the client for the configured provider, requiring its API key.

    The concrete client (`AnthropicLLM` or `OpenAILLM`) satisfies the
    `LLMClient` Protocol that downstream stages depend on.
    """
    if not config.has_api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set. Add it to your environment or a "
            ".env file (see .env.example) before running the curation agent."
        )
    if config.provider == "openai":
        return OpenAILLM(config.resolved_model, config.max_tokens, log=config.llm_log)
    return AnthropicLLM(config.resolved_model, config.max_tokens, log=config.llm_log)


def make_client(spec: str, config) -> LLMClient:
    """Build a client from a "provider:model" spec, e.g. "openai:gpt-4o".

    If the model part is omitted ("openai"), the provider's default model is used.
    Used by the verifier panel to assemble a mix of providers; each provider's
    own API key must be set.
    """
    import os

    from .config import API_KEY_ENV, DEFAULT_MODELS

    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"unknown verifier provider {provider!r} in spec {spec!r}")
    if not os.environ.get(API_KEY_ENV[provider]):
        raise RuntimeError(
            f"{API_KEY_ENV[provider]} is not set (needed for the '{spec}' verifier)."
        )
    model = model or DEFAULT_MODELS[provider]
    if provider == "openai":
        return OpenAILLM(model, config.max_tokens, log=config.llm_log)
    return AnthropicLLM(model, config.max_tokens, log=config.llm_log)
