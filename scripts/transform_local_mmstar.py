"""Transform local MMStar@val-n500-stratified-seed0-v1 to harness-compatible JSONL.

Output columns: question, A, B, C, D, image, answer, index, category, l2_category
"""
import json
import re
from pathlib import Path

SRC = Path("/home/runke.zhong/workspace/datasets/vlppl_raw/MMStar@val-n500-stratified-seed0-v1")
DST = SRC / "harness.jsonl"

ANSWER_HINT = "Answer with the option's letter from the given choices directly."


def parse_text(text: str):
    text = text.replace(ANSWER_HINT, "").strip()
    text = re.sub(r"^Hint:.*?\n\s*", "", text).strip()
    text = re.sub(r"^Question:\s*", "", text).strip()

    # Try format: (A) ... (B) ... (C) ... (D) ... (possibly 3 options)
    m = re.search(
        r"(?:^|\n)\s*\(A\)\s*(.*?)\s*\n\s*\(B\)\s*(.*?)(?:\s*\n\s*\(C\)\s*(.*?))?(?:\s*\n\s*\(D\)\s*(.*?))?(?=\n|$)",
        text,
        re.DOTALL,
    )
    if m:
        question = text[: m.start()].strip()
        opts = [m.group(i).strip() if m.group(i) else "" for i in range(1, 5)]
        return (question, *opts)

    # Try format: Options: A: ..., B: ..., C: ..., D: ...
    m = re.search(
        r"Options:\s*A:\s*(.*?)\s*,\s*B:\s*(.*?)\s*,\s*C:\s*(.*?)\s*,\s*D:\s*(.*?)(?=\n|$)",
        text,
        re.DOTALL,
    )
    if m:
        question = text[: m.start()].strip()
        opts = [m.group(i).strip() if m.group(i) else "" for i in range(1, 5)]
        return (question, *opts)

    raise ValueError(f"Could not parse options from (len={len(text)}): {text!r}")


def main():
    seen = set()
    with open(SRC / "samples.jsonl") as f, open(DST, "w") as out:
        for line in f:
            sample = json.loads(line)
            sid = sample["id"]
            assert sid not in seen, f"Duplicate id: {sid}"
            seen.add(sid)

            user_msg = sample["messages"][0]
            img_entry = [c for c in user_msg["content"] if c["type"] == "image"][0]
            img_path = str((SRC / img_entry["path"]).resolve())
            text_entry = [c for c in user_msg["content"] if c["type"] == "text"][0]
            question, opt_a, opt_b, opt_c, opt_d = parse_text(text_entry["text"])

            answer = sample["messages"][1]["content"].strip().upper()
            idx = int(sid.rsplit("-", 1)[-1])
            meta = sample["metadata"]

            row = {
                "question": question,
                "A": opt_a,
                "B": opt_b,
                "C": opt_c,
                "D": opt_d,
                "image": img_path,
                "answer": answer,
                "index": idx,
                "category": meta.get("category"),
                "l2_category": meta.get("l2_category"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(seen)} rows to {DST}")


if __name__ == "__main__":
    main()
