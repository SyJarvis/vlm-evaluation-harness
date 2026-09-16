from __future__ import annotations

import json
from typing import Any


class OpenAIJudgeClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0,
        top_p: float = 1,
        seed: int = 42,
        response_format: dict[str, Any] | None = None,
        timeout: float = 300,
        client: Any | None = None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )
        self._client = client
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.response_format = response_format or {"type": "json_object"}

    def complete_json(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=self.response_format,
            n=1,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("judge API returned an empty message")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("judge API response must be a JSON object")
        return parsed

    def public_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "response_format": self.response_format,
        }
