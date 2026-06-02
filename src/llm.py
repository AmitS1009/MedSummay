from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._api_key = os.getenv("GROQ_API_KEY")
        self._client: Groq | None = None

    @property
    def client(self) -> Groq:
        if not self._api_key:
            raise LLMError("GROQ_API_KEY is missing. Add it to .env before running the live agent.")
        if self._client is None:
            self._client = Groq(
                api_key=self._api_key,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def estimate_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(message.get("content", "")) for message in messages) // 4

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise LLMError("Groq returned an empty response")
                return content.strip()
            except Exception as exc:  # Groq exposes several transport/API exception types.
                last_error = exc
                if getattr(exc, "status_code", None) in {400, 401, 403, 404, 422}:
                    raise LLMError(f"Groq request rejected: {exc}", retryable=False) from exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise LLMError(
            f"Groq call failed after {self.max_retries} attempts: {last_error}",
            retryable=True,
        )

    def call_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        content = self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Groq returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Groq JSON response must be an object")
        return parsed
