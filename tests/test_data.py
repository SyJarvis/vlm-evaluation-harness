import json

from vlm_eval.data import load_rows


def test_local_jsonl_source_loads_without_network(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"id": "one", "answer": "A"}) + "\n")

    rows = load_rows(
        source="local",
        dataset_path=str(path),
        subset=None,
        split="test",
        cache_dir=str(tmp_path / "cache"),
    )

    assert rows[0]["id"] == "one"
