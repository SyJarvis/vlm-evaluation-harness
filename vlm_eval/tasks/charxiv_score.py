from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..utils import load_latest_jsonl


def score_charxiv(
    records: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    judge_client: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calls = _build_judge_calls(records, spec)
    completed = _run_judge_calls(calls, spec, judge_client, output_dir)
    scores = _decode_scores(calls, completed)

    evaluated = []
    for record in sorted(records, key=lambda item: item["sequence"]):
        result = scores[record["id"]]
        evaluated.append(
            {
                **record,
                "prediction": result["extracted_answer"],
                "score": result["score"],
                "correct": result["score"] == 1,
            }
        )

    descriptive = [
        item for item in evaluated if item["metadata"]["mode"] == "descriptive"
    ]
    reasoning = [
        item for item in evaluated if item["metadata"]["mode"] == "reasoning"
    ]
    summary = {
        "dataset": "charxiv",
        "num_samples": len(evaluated),
        "num_charts": len(
            {item["metadata"]["figure_id"] for item in evaluated}
        ),
        "scored": True,
        "protocol_comparable": (
            judge_client.model == spec["judge"]["model"]
        ),
        "num_judge_calls": len(calls),
        "num_judge_errors": sum(bool(item.get("error")) for item in completed.values()),
        "descriptive": _descriptive_stats(descriptive, spec),
        "reasoning": _reasoning_stats(reasoning, spec),
    }
    return evaluated, summary


def _build_judge_calls(records, spec):
    ordered = sorted(records, key=lambda item: item["sequence"])
    calls = []
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        if record["metadata"]["mode"] == "descriptive":
            groups[int(record["metadata"]["qid"])].append(record)

    batch_size = int(spec["judge"]["descriptive_batch_size"])
    for qid in range(1, 20):
        for batch_index, offset in enumerate(range(0, len(groups[qid]), batch_size)):
            batch = groups[qid][offset : offset + batch_size]
            calls.append(
                {
                    "id": f"descriptive:q{qid}:{batch_index:04d}",
                    "kind": "descriptive",
                    "sample_ids": [item["id"] for item in batch],
                    "prompt": _descriptive_grading_prompt(qid, batch, spec),
                }
            )

    for record in ordered:
        if record["metadata"]["mode"] != "reasoning":
            continue
        calls.append(
            {
                "id": f"judge:{record['id']}",
                "kind": "reasoning",
                "sample_ids": [record["id"]],
                "prompt": _reasoning_grading_prompt(record, spec),
            }
        )
    return calls


def _descriptive_grading_prompt(qid, batch, spec):
    keys = []
    for index in range(1, len(batch) + 1):
        keys.extend((f"extract_answer_T{index}", f"score_T{index}"))
    descriptive = spec["descriptive"]
    prefix = (
        descriptive["grading_prefix"]
        .replace("<|NUM_TRIPLETS|>", str(len(batch)))
        .replace("<|OVERARCHING_QUESTION|>", descriptive["grading_question_map"][str(qid)])
        .replace("<|JSON_KEYS|>", str(keys))
    )
    rubric_name = descriptive["rubric_by_qid"][str(qid)]
    examples = descriptive["grading_rubrics"][rubric_name]
    inputs = ""
    for index, record in enumerate(batch, start=1):
        inputs += (
            f"T{index}:\n"
            f"Response {index}: {record.get('response') or ''}\n"
            f"Ground Truth {index}: {record['answer']}\n\n"
        )
    return prefix + examples + inputs


def _reasoning_grading_prompt(record, spec):
    reasoning = spec["reasoning"]
    category = str(record["metadata"]["inst_category"])
    instruction = reasoning["grading_instructions"][category]
    instruction = (
        instruction.replace("<|question|>", record["metadata"]["raw_question"])
        .replace("<|ground_truth|>", str(record["answer"]))
        .replace("<|response|>", record.get("response") or "")
    )
    return reasoning["grading_prefix"] + instruction


def _run_judge_calls(calls, spec, judge_client, output_dir):
    path = output_dir / "judge_calls.jsonl"
    latest = load_latest_jsonl(path)
    completed = {}
    with path.open("a", encoding="utf-8") as handle:
        progress = tqdm(calls, desc="charxiv judge", unit="call")
        for call in progress:
            cached = latest.get(call["id"])
            if _usable_cached_call(cached, call):
                completed[call["id"]] = cached
                continue

            result, error, max_tokens = _grade_with_retries(call, spec, judge_client)
            stored = {
                **call,
                "result": result,
                "error": error,
                "max_tokens": max_tokens,
            }
            handle.write(json.dumps(stored, ensure_ascii=False) + "\n")
            handle.flush()
            latest[call["id"]] = stored
            completed[call["id"]] = stored
    return completed


def _usable_cached_call(cached, planned):
    return bool(
        cached
        and not cached.get("error")
        and cached.get("sample_ids") == planned["sample_ids"]
        and cached.get("prompt") == planned["prompt"]
        and isinstance(cached.get("result"), dict)
    )


def _grade_with_retries(call, spec, judge_client):
    judge = spec["judge"]
    max_tokens = int(judge["initial_max_tokens"])
    maximum = int(judge["maximum_max_tokens"])
    last_error = None
    for _ in range(int(judge["max_retries"])):
        try:
            result = judge_client.complete_json(call["prompt"], max_tokens=max_tokens)
            _verify_result(call, result)
            return result, None, max_tokens
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if "Unterminated string starting at" in str(exc):
                if max_tokens >= maximum:
                    break
                max_tokens = min(maximum, max_tokens * 2)
    return _dummy_result(call), last_error, max_tokens


def _verify_result(call, result):
    if call["kind"] == "reasoning":
        if "extracted_answer" not in result or "score" not in result:
            raise ValueError("reasoning judge result is missing required keys")
        return
    for index in range(1, len(call["sample_ids"]) + 1):
        answer_key = f"extract_answer_T{index}"
        score_key = f"score_T{index}"
        if answer_key not in result or score_key not in result:
            raise ValueError(f"descriptive judge result is missing {answer_key}/{score_key}")
        if result[score_key] not in (0, 1):
            raise ValueError(f"descriptive judge score must be 0 or 1: {score_key}")


def _dummy_result(call):
    if call["kind"] == "reasoning":
        return {"extracted_answer": "Failed to parse response", "score": -1}
    result = {}
    for index in range(1, len(call["sample_ids"]) + 1):
        result[f"extract_answer_T{index}"] = "Failed to parse response"
        result[f"score_T{index}"] = -1
    return result


def _decode_scores(calls, completed):
    scores = {}
    for call in calls:
        result = completed[call["id"]]["result"]
        if call["kind"] == "reasoning":
            scores[call["sample_ids"][0]] = {
                "extracted_answer": result["extracted_answer"],
                "score": result["score"],
            }
            continue
        for index, sample_id in enumerate(call["sample_ids"], start=1):
            scores[sample_id] = {
                "extracted_answer": result[f"extract_answer_T{index}"],
                "score": result[f"score_T{index}"],
            }
    return scores


def _descriptive_stats(records, spec):
    by_question = {f"Q{qid}": [] for qid in range(1, 20)}
    categories = list(dict.fromkeys(spec["descriptive"]["category_by_qid"].values()))
    by_category = {name: [] for name in categories}
    by_subplot = {name: [] for name in spec["aggregation"]["subplot_groups"].values()}
    by_subject = {name: [] for name in spec["aggregation"]["subjects"].values()}
    by_year = {name: [] for name in spec["aggregation"]["years"].values()}
    overall = []
    invalid = 0

    for record in records:
        score, was_invalid = _normalized_score(record["score"])
        invalid += was_invalid
        metadata = record["metadata"]
        qid = str(metadata["qid"])
        overall.append(score)
        by_question[f"Q{qid}"].append(score)
        by_category[spec["descriptive"]["category_by_qid"][qid]].append(score)
        by_subplot[_subplot_group(metadata["num_subplots"])].append(score)
        by_subject[spec["aggregation"]["subjects"][metadata["category"]]].append(score)
        by_year[spec["aggregation"]["years"][str(metadata["year"])]].append(score)

    return {
        "Overall Score": _percentage(overall),
        "By Question": _percentages(by_question),
        "By Category": _percentages(by_category),
        "By Subplot": _percentages(by_subplot),
        "By Subject": _percentages(by_subject),
        "By Year": _percentages(by_year),
        "N_valid": len(overall),
        "N_invalid": invalid,
        "Question Type": "Descriptive",
    }


def _reasoning_stats(records, spec):
    by_answer_type = {name: [] for name in spec["reasoning"]["answer_types"].values()}
    by_source = {name: [] for name in spec["reasoning"]["sources"].values()}
    by_subplot = {name: [] for name in spec["aggregation"]["subplot_groups"].values()}
    by_subject = {name: [] for name in spec["aggregation"]["subjects"].values()}
    by_year = {name: [] for name in spec["aggregation"]["years"].values()}
    overall = []
    invalid = 0

    for record in records:
        score, was_invalid = _normalized_score(record["score"])
        invalid += was_invalid
        metadata = record["metadata"]
        overall.append(score)
        by_answer_type[
            spec["reasoning"]["answer_types"][str(metadata["inst_category"])]
        ].append(score)
        by_source[
            spec["reasoning"]["sources"][str(metadata["qa_source"])]
        ].append(score)
        by_subplot[_subplot_group(metadata["num_subplots"])].append(score)
        by_subject[spec["aggregation"]["subjects"][metadata["category"]]].append(score)
        by_year[spec["aggregation"]["years"][str(metadata["year"])]].append(score)

    return {
        "Overall Score": _percentage(overall),
        "By Answer Type": _percentages(by_answer_type),
        "By Source": _percentages(by_source),
        "By Subject": _percentages(by_subject),
        "By Year": _percentages(by_year),
        "By Subplot": _percentages(by_subplot),
        "N_valid": len(overall),
        "N_invalid": invalid,
        "Question Type": "Reasoning",
    }


def _normalized_score(score):
    if score in (0, 1):
        return int(score), 0
    return 0, 1


def _subplot_group(num_subplots):
    value = int(num_subplots)
    if value == 1:
        return "1 Subplot"
    if value <= 4:
        return "2-4 Subplots"
    return "5+ Subplots"


def _percentages(groups):
    return {name: _percentage(values) for name, values in groups.items()}


def _percentage(values):
    return round(100 * sum(values) / len(values), 2) if values else 0
