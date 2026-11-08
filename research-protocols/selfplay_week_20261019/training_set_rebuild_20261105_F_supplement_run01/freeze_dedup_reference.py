#!/usr/bin/env python3
"""Create a score-free semantic comparison corpus from historical questions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
PRIOR = RUN.parent / "training_set_rebuild_20261104_run01"
LEGACY = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json")
OUT = RUN / "semantic_review"
OUT.mkdir(exist_ok=True)
inputs = []
records = {}


def read(path):
    raw = path.read_bytes()
    inputs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def projection(item, origin):
    keys = ("candidate_id", "root_id", "question_type", "question", "options", "items", "event_identity", "cognitive_task", "natural_information_need", "relation_plan", "query_scope", "query_pairs", "answer_semantics", "evidence", "option_evidence", "item_evidence")
    result = {key: item[key] for key in keys if key in item}
    result["record_origin"] = origin
    return result


for path in sorted(PRIOR.glob("new_evaluation/*/ITEMS_PRIVATE.json")):
    for item in read(path):
        key = item["candidate_id"]
        if key not in records:
            records[key] = projection(item, "previous_FG_candidate")
for item in read(LEGACY)["questions"]:
    key = item["candidate_id"]
    assert key not in records
    records[key] = projection(item, "legacy_363")

train_path = PRIOR / "training_export_v1/TRAIN.jsonl"
raw = train_path.read_bytes()
actual = hashlib.sha256(raw).hexdigest()
assert actual == "7f1084873230556dc7307d2d3e831509fd71ecc9fcd7bda7c613494703d1ef46"
training_rows = [json.loads(line) for line in raw.splitlines() if line]
assert len(training_rows) == 121
inputs.append({"path": str(train_path), "sha256": actual, "parseable_records": 121})

roots = {row["root_id"] for row in records.values()}
for root in sorted(roots):
    rows = [row for row in records.values() if row["root_id"] == root]
    with (OUT / f"REFERENCE_{root}.json").open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
with (OUT / "REFERENCE_FREEZE.json").open("x", encoding="utf-8") as stream:
    json.dump({"records": len(records), "roots": len(roots), "input_bindings": inputs,
               "score_free_projection": True, "training_export_verified": True,
               "damaged_selection_json_used": False,
               "projection_excludes_gold_solver_scores_and_review_decisions": True}, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"records": len(records), "roots": len(roots), "legacy": sum(x["record_origin"] == "legacy_363" for x in records.values())}))
