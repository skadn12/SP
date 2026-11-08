#!/usr/bin/env python3
"""Project only public question text from the sealed training rows for dedup audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "training_set_rebuild_20261104_run01/training_export_v1/TRAIN.jsonl"
OUTPUT = RUN / "semantic_review/SEALED_TRAIN_QUESTION_PROJECTION.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not OUTPUT.exists(), "NO_OVERWRITE"
    raw = SOURCE.read_bytes()
    assert sha(raw) == "7f1084873230556dc7307d2d3e831509fd71ecc9fcd7bda7c613494703d1ef46"
    rows = []
    for line in raw.splitlines():
        record = json.loads(line)
        user_messages = [message["content"] for message in record["messages"] if message["role"] == "user"]
        assert len(user_messages) == 1
        tail = user_messages[0].rsplit("\nQuestion:\n", 1)
        assert len(tail) == 2
        question_and_options, separator, instruction = tail[1].partition("\n\nOutput only one <box>")
        assert separator and instruction
        assert any(marker in question_and_options for marker in ("\n\nOptions:\n", "\n\nItems:\n"))
        rows.append({"candidate_id": record["candidate_id"], "root_id": record["root_id"],
                     "question_type": record["question_type"], "question_and_displayed_options": question_and_options,
                     "content_sha256": sha(question_and_options.encode("utf-8"))})
    assert len(rows) == 121 and len({row["candidate_id"] for row in rows}) == 121
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump({"source_sha256": sha(raw), "record_count": len(rows),
                   "projection": "candidate id, root, type, question and displayed options only; no answers or scores",
                   "items": rows}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "records": len(rows), "sha256": sha(OUTPUT.read_bytes())}))


if __name__ == "__main__":
    main()
