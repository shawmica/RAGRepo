"""LLM abstraction — Groq backend (free) or deterministic mock."""
from __future__ import annotations
import os, json, re, time
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    thinking: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        from groq import Groq
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Set GROQ_API_KEY environment variable.\n"
                "Get a free key at https://console.groq.com — no credit card needed."
            )
        self.client = Groq(api_key=key)
        self.model = os.environ.get("GROQ_MODEL", model)

    def chat(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> LLMResponse:
        api_msgs: list[dict[str, Any]] = []
        if system:
            api_msgs.append({"role": "system", "content": system})
        for m in messages:
            api_msgs.append({"role": m.role, "content": m.content})

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=api_msgs,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                return LLMResponse(
                    content=text,
                    model=self.model,
                    input_tokens=getattr(usage, "prompt_tokens", 0),
                    output_tokens=getattr(usage, "completion_tokens", 0),
                )
            except Exception as e:
                message = str(e).lower()
                if "rate_limit" in message or "429" in message:
                    delay = 2 ** attempt
                    time.sleep(min(delay, 10))
                    continue
                raise
        raise RuntimeError(
            "Groq rate limit exceeded. Please try again in a few seconds or switch to a smaller model."
        )


class MockLLM:
    """Deterministic mock — no API key needed, used for offline testing."""

    def __init__(self):
        self.model = "mock-llm"
        self._calls: list[dict] = []
        self.responses: list[str] = []
        self._idx = 0

    def chat(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        self._calls.append({"system": system, "last_user": last_user})

        if self.responses and self._idx < len(self.responses):
            text = self.responses[self._idx]
            self._idx += 1
            return LLMResponse(content=text, model=self.model)

        sys_lower = system.lower()
        chunk_ids = re.findall(r"\[chunk_\d+\]", last_user + system)
        is_selection = (
            "json array" in sys_lower
            or "available chunks" in last_user.lower()
            or ("reply with" in sys_lower and "chunk" in sys_lower)
        )
        if is_selection:
            ids = [c.strip("[]") for c in chunk_ids[:3]] if chunk_ids else ["chunk_0"]
            return LLMResponse(content=json.dumps(ids), model=self.model)
        if chunk_ids:
            ref = chunk_ids[0].strip("[]")
            return LLMResponse(
                content=f"Based on {ref}: This is a mock answer. [source: {ref}]",
                model=self.model,
            )
        return LLMResponse(
            content="This is a mock answer with no citations.",
            model=self.model,
        )

    @property
    def calls(self) -> list[dict]:
        return self._calls

    def reset(self):
        self._calls.clear()
        self._idx = 0


def get_llm(mock: bool = False, model: str = "llama-3.3-70b-versatile") -> GroqLLM | MockLLM:
    if mock or not os.environ.get("GROQ_API_KEY"):
        return MockLLM()
    return GroqLLM(model=model)
