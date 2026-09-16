from __future__ import annotations

import ast
import base64
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .sample import ContentPart


IMAGE_TOKEN_RE = re.compile(r"<image\s+(\d+)>", re.IGNORECASE)


def parse_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("options must decode to a list")
        return [str(item) for item in parsed]
    raise TypeError(f"unsupported options type: {type(value).__name__}")


def choice_mapping(options: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    options = list(options)
    if len(options) > 26:
        raise ValueError("at most 26 options are supported")
    labels = [chr(ord("A") + index) for index in range(len(options))]
    return labels, dict(zip(labels, options))


def numbered_image_content(
    text: str,
    row: dict[str, Any],
    *,
    fallback_position: str = "before",
) -> list[ContentPart]:
    """Replace official ``<image N>`` tokens with interleaved image parts."""
    content: list[ContentPart] = []
    cursor = 0
    matches = list(IMAGE_TOKEN_RE.finditer(text))
    for match in matches:
        if match.start() > cursor:
            content.append(ContentPart.text(text[cursor:match.start()]))
        key = f"image_{int(match.group(1))}"
        image = row.get(key)
        if image is None:
            raise ValueError(f"prompt references missing image field {key}")
        content.append(ContentPart.image(image))
        cursor = match.end()
    if matches:
        if cursor < len(text):
            content.append(ContentPart.text(text[cursor:]))
        return content

    images = [row.get(f"image_{index}") for index in range(1, 8)]
    images = [image for image in images if image is not None]
    text_part = ContentPart.text(text)
    image_parts = [ContentPart.image(image) for image in images]
    if fallback_position == "after":
        return [text_part, *image_parts]
    return [*image_parts, text_part]


def image_to_data_url(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _bytes_to_data_url(value["bytes"], "image/png")
        if value.get("path"):
            value = value["path"]

    if isinstance(value, (str, Path)):
        path = Path(value)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return _bytes_to_data_url(path.read_bytes(), mime)

    if isinstance(value, bytes):
        return _bytes_to_data_url(value, "image/png")

    if isinstance(value, Image.Image):
        image = value
        if image.mode not in {"RGB", "RGBA", "L"}:
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return _bytes_to_data_url(buffer.getvalue(), "image/png")

    raise TypeError(f"unsupported image type: {type(value).__name__}")


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                latest[str(record["id"])] = record
    return latest


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
