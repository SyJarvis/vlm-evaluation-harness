from types import SimpleNamespace

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
