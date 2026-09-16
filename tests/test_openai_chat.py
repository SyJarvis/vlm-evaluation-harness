from types import SimpleNamespace
import time

import pytest
from PIL import Image

from vlm_eval.api import OpenAIChatClient, OpenAIJudgeClient
from vlm_eval.sample import ContentPart, EvalSample


class BlankCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=""),
                    finish_reason="length",
                )
            ]
        )


class BlankOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=BlankCompletions())


class RecordingCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content="B")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RecordingOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=RecordingCompletions())


def test_openai_client_sends_interleaved_multimodal_content_without_system_prompt():
    fake = RecordingOpenAI()
    client = OpenAIChatClient(
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="local",
        max_tokens=32,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        client=fake,
    )
    sample = EvalSample(
        id="1",
        content=[
            ContentPart.text("before"),
            ContentPart.image(Image.new("RGB", (2, 2), "red")),
            ContentPart.text("after"),
        ],
        answer="B",
        question_type="multiple-choice",
        request_options={"max_tokens": 1000, "top_p": 1, "seed": 42},
    )

    response = client.complete(sample)

    request = fake.chat.completions.requests[0]
    content = request["messages"][0]["content"]
    assert response == "B"
    assert request["messages"] == [{"role": "user", "content": content}]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request["max_tokens"] == 1000
    assert request["top_p"] == 1
    assert request["seed"] == 42
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_blank_answer_is_reported_as_an_error():
    client = OpenAIChatClient(
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="local",
        client=BlankOpenAI(),
    )
    sample = EvalSample(
        id="1",
        content=[ContentPart.text("question")],
        answer="B",
        question_type="multiple-choice",
    )

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        client.complete(sample)


class JSONCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content='{"extracted_answer": "B", "score": 1}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class JSONOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=JSONCompletions())


def test_openai_judge_requests_official_json_protocol():
    fake = JSONOpenAI()
    client = OpenAIJudgeClient(
        model="gpt-4o-2024-05-13",
        base_url="https://api.openai.com/v1",
        api_key="secret",
        client=fake,
    )

    result = client.complete_json("grade this", max_tokens=256)

    request = fake.chat.completions.requests[0]
    assert result == {"extracted_answer": "B", "score": 1}
    assert request["messages"] == [{"role": "user", "content": "grade this"}]
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 256
    assert request["temperature"] == 0
    assert request["top_p"] == 1
    assert request["seed"] == 42


# --- Rate limiting: pacing and 429 backoff ---


class Fake429(Exception):
    status_code = 429

    def __init__(self, headers=None):
        super().__init__("Error code: 429 - {'error': {'code': '1302', "
                         "'message': '您的账户已达到速率限制'}}")
        self.response = SimpleNamespace(headers=headers or {})


class FlakyCompletions:
    """Fail with 429 for the first `failures` calls, then succeed."""

    def __init__(self, failures, headers=None):
        self.failures = failures
        self.headers = headers
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise Fake429(headers=self.headers)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RateLimitedCompletions:
    def create(self, **kwargs):
        raise Fake429()


class BoomCompletions:
    def create(self, **kwargs):
        raise ValueError("boom")


def _sample() -> EvalSample:
    return EvalSample(
        id="1",
        content=[ContentPart.text("question")],
        answer="B",
        question_type="multiple-choice",
    )


def _client(completions, **kwargs):
    return OpenAIChatClient(
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="local",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        **kwargs,
    )


def test_rate_limit_error_retries_with_exponential_backoff(monkeypatch):
    delays = []
    monkeypatch.setattr(
        "vlm_eval.api.openai_chat.time.sleep", lambda s: delays.append(s)
    )
    fake = FlakyCompletions(failures=2)
    client = _client(fake)

    assert client.complete(_sample()) == "ok"
    assert fake.calls == 3
    assert delays == [8.0, 16.0]


def test_rate_limit_retry_honors_retry_after_header(monkeypatch):
    delays = []
    monkeypatch.setattr(
        "vlm_eval.api.openai_chat.time.sleep", lambda s: delays.append(s)
    )
    fake = FlakyCompletions(failures=1, headers={"Retry-After": "30"})
    client = _client(fake)

    assert client.complete(_sample()) == "ok"
    assert delays == [30.0]


def test_rate_limit_error_raises_after_exhausting_retries(monkeypatch):
    delays = []
    monkeypatch.setattr(
        "vlm_eval.api.openai_chat.time.sleep", lambda s: delays.append(s)
    )
    fake = RateLimitedCompletions()
    client = _client(fake)

    with pytest.raises(Fake429):
        client.complete(_sample())
    assert len(delays) == 6
    assert delays == [8.0, 16.0, 32.0, 60.0, 60.0, 60.0]


def test_non_rate_limit_errors_are_not_retried(monkeypatch):
    delays = []
    monkeypatch.setattr(
        "vlm_eval.api.openai_chat.time.sleep", lambda s: delays.append(s)
    )
    fake = BoomCompletions()
    client = _client(fake)

    with pytest.raises(ValueError, match="boom"):
        client.complete(_sample())
    assert delays == []


def test_pacer_spaces_request_starts_by_rpm_interval():
    from vlm_eval.api.openai_chat import _RequestPacer

    pacer = _RequestPacer(min_interval=0.1)
    start = time.monotonic()
    pacer.acquire()
    pacer.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08


def test_client_rejects_invalid_rpm():
    with pytest.raises(ValueError, match="rpm"):
        _client(RecordingCompletions(), rpm=0)


def test_public_config_records_rpm():
    client = _client(RecordingCompletions(), rpm=30)
    assert client.public_config()["rpm"] == 30
