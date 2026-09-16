from PIL import Image

from vlm_eval.prompts import MMMU_PRO_DIRECT_VISION, MMSTAR_ANSWER_HINT
from vlm_eval.tasks.base import DatasetConfig
from vlm_eval.tasks.mmmu import MMMUAdapter
from vlm_eval.tasks.mmmu_pro import MMMUProAdapter
from vlm_eval.tasks.mmstar import MMStarAdapter


def config(subset: str = "all", prompt_mode: str = "direct") -> DatasetConfig:
    return DatasetConfig(
        source="local",
        dataset_path="unused",
        subset=subset,
        split="test",
        prompt_mode=prompt_mode,
    )


def test_mmmu_build_sample_uses_official_prompt_and_image_position():
    image = Image.new("RGB", (2, 2), "red")
    row = {
        "id": "validation_Accounting_1",
        "question": "<image 1> What is shown?",
        "options": "['one', 'two']",
        "answer": "B",
        "question_type": "multiple-choice",
        "image_1": image,
        "subfield": "Accounting",
        "topic_difficulty": "Medium",
    }

    sample = MMMUAdapter().build_sample(row, config())

    assert [part.type for part in sample.content[:2]] == ["image", "text"]
    assert sample.content[0].value is image
    assert sample.audit_prompt() == (
        "<image 1> What is shown?\n"
        "(A) one\n"
        "(B) two\n\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert sample.metadata["subject"] == "Accounting"


def test_mmmu_pro_standard_preserves_referenced_image_order():
    image_1 = Image.new("RGB", (2, 2), "red")
    image_2 = Image.new("RGB", (2, 2), "blue")
    row = {
        "id": "test_Accounting_1",
        "question": "Choose the matching image.",
        "options": "['<image 2>', '<image 1>']",
        "answer": "A",
        "subject": "Accounting",
        "image_1": image_1,
        "image_2": image_2,
    }

    sample = MMMUProAdapter().build_sample(
        row, config(subset="standard (10 options)")
    )

    images = [part.value for part in sample.content if part.type == "image"]
    assert images == [image_2, image_1]
    assert "A. " in sample.audit_prompt()
    assert sample.audit_prompt().endswith(
        "Answer with the option letter from the given choices directly."
    )


def test_mmmu_pro_vision_uses_official_direct_prompt():
    image = Image.new("RGB", (2, 2), "white")
    row = {
        "id": "test_Art_1",
        "options": "['one', 'two']",
        "answer": "B",
        "subject": "Art",
        "image": image,
    }

    sample = MMMUProAdapter().build_sample(row, config(subset="vision"))

    assert sample.content[0].value == MMMU_PRO_DIRECT_VISION
    assert sample.content[1].value is image


def test_mmstar_does_not_duplicate_embedded_options():
    image = Image.new("RGB", (2, 2), "white")
    question = "Question?\nOptions: A: one, B: two, C: three, D: four."
    row = {
        "index": 7,
        "question": question,
        "image": image,
        "answer": "C",
        "category": "logical reasoning",
        "l2_category": "common reasoning",
    }

    sample = MMStarAdapter().build_sample(row, config(subset="val"))

    assert sample.audit_prompt().count("Options:") == 1
    assert sample.audit_prompt().endswith(MMSTAR_ANSWER_HINT)
    assert sample.content[0].value is image
