import pytest

from vlm_eval.cli import _build_judge_client, _parse_tasks, build_parser


def test_cli_omitted_tasks_defaults_to_all():
    args = build_parser().parse_args(["--model", "qwen3.5-vl"])

    assert _parse_tasks(args.tasks) == [
        "babyvision",
        "chartqa",
        "docvqa",
        "mmlu",
        "mmmu",
        "mmmu_pro",
        "mmstar",
        "charxiv",
    ]


def test_cli_all_selects_all_tasks():
    args = build_parser().parse_args(
        ["--model", "qwen3.5-vl", "--tasks", "all"]
    )

    assert _parse_tasks(args.tasks) == [
        "babyvision",
        "chartqa",
        "docvqa",
        "mmlu",
        "mmmu",
        "mmmu_pro",
        "mmstar",
        "charxiv",
    ]


def test_cli_accepts_comma_separated_tasks():
    args = build_parser().parse_args(
        [
            "--model",
            "qwen3.5-vl",
            "--tasks",
            "mmstar, mmmu",
            "--hub",
            "hf",
        ]
    )

    assert _parse_tasks(args.tasks) == ["mmstar", "mmmu"]


def test_charxiv_validation_requires_separate_judge_key():
    args = build_parser().parse_args(
        ["--model", "qwen3.5-vl", "--tasks", "charxiv"]
    )
    args.judge_api_key = None

    with pytest.raises(ValueError, match="OPENAI_JUDGE_API_KEY"):
        _build_judge_client(args, ["charxiv"])


def test_charxiv_public_test_does_not_require_judge_key():
    args = build_parser().parse_args(
        ["--model", "qwen3.5-vl", "--tasks", "charxiv", "--split", "test"]
    )
    args.judge_api_key = None

    assert _build_judge_client(args, ["charxiv"]) is None


@pytest.mark.parametrize("legacy_option", ["--task", "--dataset", "--source"])
def test_cli_no_longer_accepts_legacy_options(legacy_option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--model", "qwen3.5-vl", legacy_option, "mmstar"]
        )
