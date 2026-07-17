"""Pluggable LLM backend used by the listwise reranker and the LLM generator.

`IdentityLLMClient` is the zero-dependency, zero-API-key default: it makes the
listwise/generation stages runnable everywhere, at the cost of not actually
using an LLM. Swap to `OpenAICompatibleClient` or `AnthropicClient` (or add a
new backend implementing `LLMClient`) once credentials are available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str: ...


class IdentityLLMClient(LLMClient):
    """Fallback client: returns an empty completion. Callers must treat a
    blank/unparsable response as "no-op" and fall back to a deterministic
    heuristic (see `LLMListwiseReranker` and `ExtractiveGenerator`)."""

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        return ""


class OpenAICompatibleClient(LLMClient):
    """Minimal client for any OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


class AnthropicClient(LLMClient):
    """Client backed by the `anthropic` package (lazy-imported, optional dep)."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package: pip install trecrag[llm]"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        self.model = model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
