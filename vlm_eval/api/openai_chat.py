from __future__ import annotations

from typing import Any

from ..sample import EvalSample
from ..utils import image_to_data_url


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 128,
        temperature: float = 0,
        timeout: float = 300,
        max_retries: int = 5,
        image_detail: str = "auto",
        extra_body: dict[str, Any] | None = None,
        client: Any | None = None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        self._client = client
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_detail = image_detail
        self.extra_body = extra_body or {}

    def complete(self, sample: EvalSample) -> str:
        content = []
        for part in sample.content:
            if part.type == "text":
                content.append({"type": "text", "text": str(part.value)})
            else:
                image_url = {"url": image_to_data_url(part.value)}
                if self.image_detail != "auto":
                    image_url["detail"] = self.image_detail
                content.append({"type": "image_url", "image_url": image_url})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request.update(sample.request_options)
        if self.extra_body:
            request["extra_body"] = self.extra_body
        response = self._client.chat.completions.create(**request)
        choice = response.choices[0]
        answer = str(choice.message.content or "").strip()
        if not answer:
            # Reasoning models spend max_tokens on hidden thinking and can return
            # blank content. Scoring that as a wrong answer silently deflates results.
            raise RuntimeError(
                "API returned no answer text "
                f"(finish_reason={choice.finish_reason}); raise --max-tokens"
            )
        return answer

    def public_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "image_detail": self.image_detail,
            "extra_body": self.extra_body,
        }
