from __future__ import annotations

import threading
import time
from typing import Any

from ..sample import EvalSample
from ..utils import image_to_data_url

# Rate-limit backoff policy: wait long enough to outlive provider quota windows
# (the OpenAI SDK's built-in retries are too short for sustained limits).
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_BASE_WAIT = 8.0
RATE_LIMIT_MAX_WAIT = 60.0


class _RequestPacer:
    """Thread-safe minimum interval between request starts (RPM pacing)."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                self._next_allowed += self._min_interval
            else:
                self._next_allowed = now + self._min_interval
        if wait > 0:
            time.sleep(wait)


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
        rpm: int | None = None,
        client: Any | None = None,
    ):
        if rpm is not None and rpm < 1:
            raise ValueError("rpm must be a positive integer")
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
        self.rpm = rpm
        self._pacer = _RequestPacer(60.0 / rpm) if rpm else None

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
        response = self._request_with_backoff(request)
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

    def _request_with_backoff(self, request: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            if self._pacer is not None:
                self._pacer.acquire()
            try:
                return self._client.chat.completions.create(**request)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= RATE_LIMIT_RETRIES:
                    raise
                delay = min(
                    RATE_LIMIT_BASE_WAIT * 2**attempt, RATE_LIMIT_MAX_WAIT
                )
                delay = max(delay, _retry_after_seconds(exc))
                time.sleep(delay)
                attempt += 1

    def public_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "image_detail": self.image_detail,
            "extra_body": self.extra_body,
            "rpm": self.rpm,
        }


def _is_rate_limit_error(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "rate limit" in message or "1302" in message


def _retry_after_seconds(exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return 0.0
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return 0.0
