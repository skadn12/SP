"""Bind existing records and prepare an offline semantic-review packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PRIOR = WEEK / "day6_diagnostic_reliability_v11"
DAY5 = WEEK / "day5_training_v1" / "train_diagnosis_prepare"


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(rows):
    for row in rows:
        if bind(row["path"]) != row:
            raise RuntimeError("INPUT_BINDING_CHANGED:" + row["path"])


def write_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    if any((HERE / name).exists() for name in ("AUDIT_PACKET.json", "INPUT_MANIFEST.json")):
        raise RuntimeError("NO_OVERWRITE_AUDIT_INPUTS")
    check(read(PRIOR / "FINAL_MANIFEST.json")["files"])
    requests = read(PRIOR / "REQUESTS.json")["requests"]
    if len(requests) != 12 or len({r["root_id"] for r in requests}) != 6:
        raise RuntimeError("SCOPE_DRIFT")
    baseline = read(DAY5 / "private/S0_BASELINE_BINDINGS.json")
    check([baseline["gold_binding"]])
    gold_path = Path(baseline["gold_binding"]["path"])
    gold = {q["candidate_id"]: q for q in read(gold_path)["questions"]}
    annotations = {q["candidate_id"]: q for q in read(DAY5 / "SELECTION.json")["question_annotations"]}
    bound_paths = {
        HERE / "PROTOCOL_ZH.md", Path(__file__), PRIOR / "FINAL_MANIFEST.json",
        PRIOR / "REQUESTS.json", PRIOR / "DECISION.json", PRIOR / "FORMAT_AUDIT.json",
        PRIOR / "INDEPENDENT_POSTEXECUTION_AUDIT.json", DAY5 / "POLICY.json",
        DAY5 / "SELECTION.json", DAY5 / "private/S0_BASELINE_BINDINGS.json", gold_path,
        DAY5 / "public/DIRECT_DISPLAY_MAPS.json", DAY5.parent / "DECISION.json",
    }
    questions = []
    for request in requests:
        source_path = Path(request["source_path"])
        if bind(source_path)["sha256"] != request["source_sha256"]:
            raise RuntimeError("SOURCE_CHANGED")
        source = read(source_path)
        turns = {turn["turn_ref"]: {"source_ordinal_1based": i + 1, **turn}
                 for i, turn in enumerate(source["turns"])}
        field = "options" if request["question_type"] == "multi_select" else "items"
        display = request["display"][field]
        canonical = gold[request["candidate_id"]]
        if canonical["root_id"] != request["root_id"] or canonical[field].values() == []:
            raise RuntimeError("QUESTION_ID_DRIFT")
        if set(canonical[field].values()) != set(display.values()) or len(set(display.values())) != len(display):
            raise RuntimeError("CANONICAL_DISPLAY_CONTENT_MISMATCH")
        reverse = {text: label for label, text in display.items()}
        display_gold = [reverse[canonical[field][label]] for label in canonical["gold"]]
        prompt_path = Path(request["prompt_path"])
        if bind(prompt_path)["sha256"] != request["prompt_sha256"]:
            raise RuntimeError("PROMPT_CHANGED")
        prompt = prompt_path.read_text()
        prompt_tail = prompt.split("Question (", 1)[1].split("For each displayed label,", 1)[0]
        row = {
            "candidate_id": request["candidate_id"], "request_id": request["request_id"],
            "root_id": request["root_id"], "question_type": request["question_type"],
            "question_text": canonical["question"], "exact_prompt_question_section": "Question (" + prompt_tail,
            "display": display, "display_gold": display_gold,
            "gold_mapping_method": "exact content string equality; canonical label names are never assumed to be display labels",
            "source_binding": bind(source_path), "prompt_binding": bind(prompt_path),
            "annotation": annotations[request["candidate_id"]], "models": {},
        }
        bound_paths.update((source_path, prompt_path, Path(request["schema_path"])))
        for role in ("A1", "B1"):
            raw_path = PRIOR / "runs" / role / "responses" / (request["request_id"] + ".json")
            raw = read(raw_path)
            if raw["request_id"] != request["request_id"] or raw["role"] != role:
                raise RuntimeError("RAW_ID_DRIFT")
            parsed = json.loads(raw["raw_output"])
            cited = {label: [turns[ref] for ref in item["turn_refs"]]
                     for label, item in parsed["item_evidence"].items()}
            row["models"][role] = {"raw_binding": bind(raw_path), "parsed": parsed,
                                    "actual_cited_turns": cited}
            bound_paths.add(raw_path)
            bound_paths.add(PRIOR / "runs" / role / "RAW_RESPONSE_FREEZE.json")
        questions.append(row)
    packet = {"scope": "OFFLINE_POSTHOC_SOURCE_SEMANTIC_REVIEW_ONLY", "questions": questions,
              "question_count": 12, "response_count": 24, "item_judgment_count": 144,
              "new_model_requests": 0, "training_runs": 0}
    write_new(HERE / "AUDIT_PACKET.json", packet)
    bound_paths.add(HERE / "AUDIT_PACKET.json")
    bindings = [bind(path) for path in sorted(bound_paths)]
    write_new(HERE / "INPUT_MANIFEST.json", {
        "scope": packet["scope"], "new_model_requests": 0,
        "files": bindings, "file_count": len(bindings),
        "gold_used_only_for_post_generation_source_audit": True,
        "prior_freeze_preserved": True,
    })
    check(bindings)
    print(json.dumps({"status": "AUDIT_PACKET_READY", "bound_files": len(bindings),
                      "questions": 12, "responses": 24, "items": 144}))


if __name__ == "__main__":
    main()
