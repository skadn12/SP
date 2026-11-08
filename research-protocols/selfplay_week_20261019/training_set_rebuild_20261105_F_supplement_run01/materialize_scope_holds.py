#!/usr/bin/env python3
"""Bind independent pre-evaluation scope holds into new evaluation inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
DECISION_PATH = RUN / "PRE_EVALUATION_QUALITY_ADJUDICATION.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    decision = read(DECISION_PATH)
    holds = [row for row in decision["items"] if row["decision"] == "HOLD"]
    for batch in sorted({row["batch"] for row in holds}):
        parent = RUN / "evaluation" / batch
        output = RUN / "evaluation" / (batch + "_scope_adjudicated")
        assert not output.exists(), "NO_OVERWRITE_OR_RETRY"
        assert not (parent / "solver/REQUESTS.json").exists(), "PARENT_ALREADY_PREPARED"
        files = ("ITEMS_PRIVATE.json", "QUALITY_FINAL.json", "SOURCE_FREEZE.json", "SOURCE_QUALIFICATION.json")
        values = {name: read(parent / name) for name in files}
        items = {row["candidate_id"]: row for row in values["ITEMS_PRIVATE.json"]}
        quality = {row["candidate_id"]: row for row in values["QUALITY_FINAL.json"]["items"]}
        for hold in (row for row in holds if row["batch"] == batch):
            item, qrow = items[hold["candidate_id"]], quality[hold["candidate_id"]]
            assert item["source_sha256"] == hold["source_sha256"] == qrow["source_sha256"]
            assert item["review_id"] == hold["review_id"] == qrow["review_id"]
            assert qrow["content_pass"] is True
            qrow["prior_quality_decision"] = copy.deepcopy(qrow)
            qrow["content_pass"] = False
            qrow["status"] = "HOLD_ADJUDICATED"
            qrow["hold_reasons"] = ["independent_evidence_scope_review"]
            qrow["hard_issues"] = [{"issue": "evidence_or_question_scope", "reason": hold["reason"]}]
            qrow["quality_adjudication"] = {"path": str(DECISION_PATH), "sha256": sha(DECISION_PATH),
                                             "adjudication_id": decision["adjudication_id"]}
        output.mkdir(parents=True)
        for name, value in values.items():
            write(output / name, value)
        paths = [parent / name for name in files] + [DECISION_PATH, Path(__file__).resolve()]
        write(output / "ADJUDICATION_INPUT_BINDINGS.json", {
            "parent_batch": str(parent),
            "files": [{"path": str(path), "sha256": sha(path)} for path in paths],
            "content_or_gold_changed": False,
        })
        print(json.dumps({"batch": batch, "output": str(output),
                          "additional_holds": [row["candidate_id"] for row in holds if row["batch"] == batch]}))


if __name__ == "__main__":
    main()
