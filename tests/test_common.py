import random

from vlm_eval.tasks.common import (
    eval_open_answer,
    parse_multiple_choice,
    parse_open_response,
)


def test_parse_multiple_choice_prefers_last_answer_tag():
    prediction = parse_multiple_choice(
        "I first considered A. Answer: C",
        ["A", "B", "C", "D"],
        {"A": "one", "B": "two", "C": "three", "D": "four"},
    )

    assert prediction == "C"


def test_parse_multiple_choice_uses_seeded_official_fallback():
    rng = random.Random(42)

    prediction = parse_multiple_choice("unknown", ["A", "B", "C", "D"], {}, rng=rng)

    assert prediction == "A"


def test_open_answer_numeric_normalization_matches_official_behavior():
    predictions = parse_open_response("Therefore, the answer is 1,234.004")

    assert eval_open_answer("1234", predictions)
