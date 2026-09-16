from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ContentPart:
    type: Literal["text", "image"]
    value: Any

    @classmethod
    def text(cls, value: str) -> "ContentPart":
        return cls("text", value)

    @classmethod
    def image(cls, value: Any) -> "ContentPart":
        return cls("image", value)


@dataclass
class EvalSample:
    id: str
    content: list[ContentPart]
    answer: Any
    question_type: str
    choices: list[str] = field(default_factory=list)
    index_to_answer: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_options: dict[str, Any] = field(default_factory=dict)

    def audit_prompt(self) -> str:
        parts = []
        image_index = 0
        for part in self.content:
            if part.type == "text":
                parts.append(str(part.value))
            else:
                image_index += 1
                parts.append(f"<image {image_index}>")
        return "".join(parts)
