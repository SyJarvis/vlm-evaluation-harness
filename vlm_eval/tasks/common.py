from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any


def parse_multiple_choice(
    response: str,
    choices: list[str],
    index_to_answer: dict[str, str],
    *,
    rng: random.Random | None = None,
) -> str | None:
    """Parse a choice using the official MMMU/MMMU-Pro strategy."""
    response = response or ""
    answer_matches = re.findall(
        r"answer\s*:\s*\*?\*?\s*\(?([A-Z])\)?",
        response,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        candidate = answer_matches[-1].upper()
        if candidate in choices:
            return candidate

    stripped = response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        stripped = stripped.strip(char)
    padded = f" {stripped} "

    candidates = [choice for choice in choices if f"({choice})" in padded]
    bracketed = bool(candidates)
    matched_by_text = False
    if not candidates:
        candidates = [
            choice
            for choice in choices
            if re.search(rf"(?<![A-Za-z]){re.escape(choice)}(?![A-Za-z])", padded)
        ]
    if not candidates and len(padded.split()) > 5:
        candidates = [
            choice
            for choice, answer in index_to_answer.items()
            if answer and answer.lower() in padded.lower()
        ]
        matched_by_text = bool(candidates)

    if not candidates:
        return rng.choice(choices) if rng is not None and choices else None
    if len(candidates) == 1:
        return candidates[0]

    positions = []
    for candidate in candidates:
        if matched_by_text:
            position = padded.lower().rfind(index_to_answer[candidate].lower())
        elif bracketed:
            position = padded.rfind(f"({candidate})")
        else:
            matches = list(
                re.finditer(
                    rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z])", padded
                )
            )
            position = matches[-1].start() if matches else -1
        positions.append(position)
    return candidates[max(range(len(candidates)), key=positions.__getitem__)]


def parse_open_response(response: str) -> list[str | float]:
    def key_subresponses(value: str) -> list[str]:
        value = value.strip().strip(".").lower()
        parts = re.split(r"\.\s(?=[A-Z])|\n", value)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        results = []
        for index, part in enumerate(parts):
            current_indicators = indicators + (["="] if index == len(parts) - 1 else [])
            candidates = [
                part.split(indicator)[-1].strip()
                for indicator in current_indicators
                if indicator in part
            ]
            if candidates:
                shortest = min(candidates, key=len)
                if shortest not in {":", ",", ".", "!", "?", ";", "'"}:
                    results.append(shortest)
        return results or [value]

    key_parts = key_subresponses(response or "")
    predictions: list[str | float] = list(key_parts)
    for part in key_parts:
        predictions.extend(_extract_numbers(part))

    normalized: list[str | float] = []
    for prediction in predictions:
        normalized.extend(_normalize_str(str(prediction)))
    return list(dict.fromkeys(normalized))


def eval_open_answer(answer: Any, predictions: list[str | float]) -> bool:
    answers = answer if isinstance(answer, list) else [answer]
    normalized_answers: list[str | float] = []
    for item in answers:
        normalized_answers.extend(_normalize_str(str(item)))
    for prediction in predictions:
        if isinstance(prediction, str):
            if any(
                isinstance(answer_item, str) and answer_item in prediction
                for answer_item in normalized_answers
            ):
                return True
        elif prediction in normalized_answers:
            return True
    return False


def grouped_accuracy(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        value = record.get("metadata", {}).get(key)
        if value is not None:
            grouped[str(value)].append(bool(record.get("correct")))
    return {
        name: {
            "num": len(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for name, values in sorted(grouped.items())
    }


def _normalize_str(value: str) -> list[str | float]:
    value = value.strip()
    try:
        return [round(float(value.replace(",", "")), 2)]
    except ValueError:
        lowered = value.lower()
        if len(lowered) == 1:
            return [f" {lowered}", f"{lowered} "]
        return [lowered]


def _extract_numbers(value: str) -> list[str]:
    patterns = [
        r"-?\b\d{1,3}(?:,\d{3})+\b",
        r"-?\d+(?:\.\d+)?[eE][+-]?\d+",
        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])",
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, value))
    return results
