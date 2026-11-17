import pytest
import yaml

from vlm_eval.run_config import (
    ConfigError,
    config_sha256,
    dump_config,
    judge_api_key,
    resolve_config,
)


def _raw(**overrides):
    raw = {
        "model": {"name": "qwen3.5-vl", "max_tokens": 64},
        "runtime": {"concurrency": 4},
        "tasks": [
            "mmstar",
            {
                "id": "mmmu_pro_vision",
                "name": "mmmu_pro",
                "subset": "vision",
                "prompt_mode": "cot",
                "generation": {"max_tokens": 2048},
                "runtime": {"rpm": 30},
            },
        ],
    }
    raw.update(overrides)
    return raw


def test_tasks_inherit_defaults_and_apply_overrides():
    resolved = resolve_config(_raw())
    mmstar, mmmu_pro = resolved["tasks"]

    assert mmstar["id"] == "mmstar"
    assert mmstar["dataset_path"] == "Lin-Chen/MMStar"
    assert mmstar["split"] == "val"
    assert mmstar["generation"]["max_tokens"] == 64
    assert mmstar["runtime"] == {"concurrency": 4, "rpm": None, "limit": None}
    assert mmmu_pro["subset"] == "vision"
    assert mmmu_pro["prompt_mode"] == "cot"
    assert mmmu_pro["generation"]["max_tokens"] == 2048
    assert mmmu_pro["runtime"] == {"concurrency": 4, "rpm": 30, "limit": None}
    assert resolved["run"] == {"name": "qwen3.5-vl", "output_root": "runs"}


def test_cli_limit_overrides_every_task():
    resolved = resolve_config(_raw(), limit=3)

    assert [task["runtime"]["limit"] for task in resolved["tasks"]] == [3, 3]


def test_resolving_dumped_config_is_stable():
    resolved = resolve_config(_raw(), limit=3)
    reloaded = resolve_config(yaml.safe_load(dump_config(resolved)))

    assert reloaded == resolved
    assert config_sha256(reloaded) == config_sha256(resolved)


def test_dumped_config_holds_env_names_not_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-model-secret")
    monkeypatch.setenv("OPENAI_JUDGE_API_KEY", "sk-judge-secret")
    resolved = resolve_config(_raw(tasks=["charxiv"]))
    dumped = dump_config(resolved)

    assert judge_api_key(resolved) == "sk-judge-secret"
    assert "sk-model-secret" not in dumped
    assert "sk-judge-secret" not in dumped
    assert "OPENAI_JUDGE_API_KEY" in dumped


def test_duplicate_task_ids_are_rejected():
    with pytest.raises(ConfigError, match="duplicate task ids: mmstar"):
        resolve_config(_raw(tasks=["mmstar", {"name": "mmstar"}]))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tasks": ["nope"]}, "unsupported task"),
        ({"tasks": [{"name": "mmstar", "max_token": 1}]}, "unknown keys max_token"),
        ({"tasks": [{"name": "mmstar", "hub": "local"}]}, "dataset_path is required"),
        ({"tasks": []}, "non-empty list"),
        ({"model": {}}, "model.name is required"),
    ],
)
def test_invalid_configs_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        resolve_config(_raw(**overrides))


def test_charxiv_validation_requires_judge_key(monkeypatch):
    monkeypatch.delenv("OPENAI_JUDGE_API_KEY", raising=False)
    resolved = resolve_config(_raw(tasks=["charxiv"]))

    with pytest.raises(ConfigError, match="OPENAI_JUDGE_API_KEY"):
        judge_api_key(resolved)


def test_charxiv_public_test_does_not_require_judge_key(monkeypatch):
    monkeypatch.delenv("OPENAI_JUDGE_API_KEY", raising=False)
    resolved = resolve_config(_raw(tasks=[{"name": "charxiv", "split": "test"}]))

    assert judge_api_key(resolved) is None
