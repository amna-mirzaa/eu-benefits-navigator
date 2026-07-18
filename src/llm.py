"""Groq LLM client with role-based model routing (fast vs. reasoning)."""

import json
from functools import lru_cache
from typing import Literal

from groq import Groq

from src import config

Role = Literal["fast", "reasoning"]


class LLMClient:
    """Thin wrapper so the rest of the app never talks to the Groq SDK directly.

    Two roles map to two different Groq-hosted models: `fast` for parsing/writing
    tasks, `reasoning` for the eligibility agent's step-by-step chain of thought.
    Kept small and swappable so a different provider (e.g. Ollama) could
    implement the same `complete` signature later.
    """

    def __init__(self) -> None:
        self._client = Groq(api_key=config.GROQ_API_KEY) if config.has_groq_key() else None

    def _model_for(self, role: Role) -> str:
        return config.GROQ_REASONING_MODEL if role == "reasoning" else config.GROQ_FAST_MODEL

    def complete(
        self,
        messages: list[dict],
        role: Role = "fast",
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if self._client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self._model_for(role),
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        messages: list[dict],
        role: Role = "fast",
        temperature: float = 0.2,
    ) -> dict:
        raw = self.complete(messages, role=role, temperature=temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
            raise

    def complete_with_reasoning(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> tuple[str, str]:
        """Runs the `reasoning` model and returns (answer, reasoning_trace).

        Groq's `openai/gpt-oss-120b` exposes reasoning via `reasoning_effort`
        (depth of the chain-of-thought) and `include_reasoning` (whether it's
        returned at all), landing in `message.reasoning` separately from the
        final `message.content`. Falls back to extracting a `<think>...</think>`
        block manually in case a differently-configured model is swapped in.
        """
        if self._client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        response = self._client.chat.completions.create(
            model=self._model_for("reasoning"),
            messages=messages,
            temperature=temperature,
            reasoning_effort="high",
            include_reasoning=True,
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return content, reasoning
        if "<think>" in content and "</think>" in content:
            start = content.find("<think>") + len("<think>")
            end = content.find("</think>")
            return content[end + len("</think>") :].strip(), content[start:end].strip()
        return content, ""


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
