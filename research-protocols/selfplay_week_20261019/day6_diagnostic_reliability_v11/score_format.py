"""Offline format scoring; preserve the frozen runner and all raw records."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

import run as frozen


FLAGS = (
    "json_parse_valid", "json_schema_valid", "all_labels_present",
    "all_turn_refs_exist", "final_answer_unique",
    "final_answer_shape_valid", "final_answer_labels_valid",
)


def inspect_response(request, response):
    row = {key: request[key] for key in (
        "request_id", "candidate_id", "root_id", "question_type")}
    row.update({key: False for key in FLAGS})
    row.update(status=response["status"], errors=[])
    try:
        obj = json.loads(response["raw_output"])
        row["json_parse_valid"] = True
    except (TypeError, ValueError) as exc:
        row["errors"].append("JSON_PARSE:" + str(exc)[:350])
        row["mechanically_valid"] = False
        return row
    try:
        jsonschema.validate(obj, frozen.read(Path(request["schema_path"])))
        row["json_schema_valid"] = True
    except jsonschema.ValidationError as exc:
        row["errors"].append("SCHEMA:" + str(exc)[:350])
    labels = request["labels"]
    evidence = obj.get("item_evidence") if isinstance(obj, dict) else None
    if isinstance(evidence, dict):
        row["all_labels_present"] = set(evidence) == set(labels)
        refs_well_formed, refs = True, []
        for item in evidence.values():
            if not isinstance(item, dict) or not isinstance(item.get("turn_refs"), list):
                refs_well_formed = False
            else:
                refs.extend(item["turn_refs"])
        refset = set(request["source_turn_refs"])
        row["all_turn_refs_exist"] = refs_well_formed and all(
            isinstance(ref, str) and ref in refset for ref in refs)
    answer = obj.get("final_answer") if isinstance(obj, dict) else None
    if isinstance(answer, list):
        strings = all(isinstance(label, str) for label in answer)
        row["final_answer_unique"] = strings and len(answer) == len(set(answer))
        row["final_answer_labels_valid"] = strings and all(label in labels for label in answer)
        row["final_answer_shape_valid"] = (
            len(answer) == len(labels) if request["question_type"] == "ordering"
            else 2 <= len(answer) <= 5)
        row["final_answer"] = answer
    for flag in FLAGS[2:]:
        if not row[flag]:
            row["errors"].append(flag.upper())
    row["mechanically_valid"] = all(row[flag] for flag in FLAGS)
    return row


def main():
    target = frozen.HERE / "FORMAT_AUDIT.json"
    if target.exists():
        raise RuntimeError("NO_OVERWRITE_FORMAT_AUDIT")
    freeze, data = frozen.verify()
    requests = data["requests"]
    expected_ids = {request["request_id"] for request in requests}
    if len(expected_ids) != 12:
        raise RuntimeError("REQUEST_DENOMINATOR_DRIFT")
    report = {
        "status": "SCORED_FORMAT_ONLY",
        "scope": "NO_GOLD_NO_SEMANTIC_CORRECTNESS",
        "scorer": frozen.binding(Path(__file__)),
        "frozen_runner": frozen.binding(frozen.HERE / "run.py"),
        "preparation_freeze": frozen.binding(frozen.FREEZE_PATH),
        "scoring_fix": "Historical records have no disposition field. Compare only their frozen schema_valid boolean; do not infer disposition.",
        "original_scoring_failure": frozen.binding(frozen.HERE / "FORMAT_SCORING.log"),
        "models": {},
    }
    for role in ("A1", "B1"):
        run_dir = frozen.HERE / "runs" / role
        raw_freeze_path = run_dir / "RAW_RESPONSE_FREEZE.json"
        raw_freeze = frozen.read(raw_freeze_path)
        frozen.check(raw_freeze["files"] + [raw_freeze["preparation_freeze"], raw_freeze["audit"]])
        frozen.audit_check(freeze, Path(raw_freeze["audit"]["path"]))
        raw_records = [frozen.read(Path(item["path"])) for item in raw_freeze["files"]]
        responses = {item["request_id"]: item for item in raw_records}
        if len(raw_records) != 12 or set(responses) != expected_ids:
            raise RuntimeError("RESPONSE_ID_OR_DENOMINATOR_DRIFT:" + role)
        rows = [inspect_response(request, responses[request["request_id"]]) for request in requests]
        legacy_path = frozen.PREV / "models" / role / "private" / "SEMANTIC_READOUT.json"
        # Read only the historical format verdict for the matched candidate IDs.
        legacy = frozen.read(legacy_path)
        matched = {row["candidate_id"]: [] for row in rows}
        for old in legacy["questions"]:
            if old["candidate_id"] in matched:
                matched[old["candidate_id"]].append(old["schema_valid"])
        for row in rows:
            old_values = matched[row["candidate_id"]]
            if len(old_values) != 1 or type(old_values[0]) is not bool:
                raise RuntimeError("LEGACY_MATCH_OR_FORMAT_VERDICT_INVALID")
            row["legacy_schema_valid"] = old_values[0]
        metrics = {
            "planned": 12,
            "submitted": raw_freeze["submitted_requests"],
            "returned": raw_freeze["returned_requests"],
            "automatic_retry": raw_freeze["automatic_retry"],
            "raw_response_freeze": frozen.binding(raw_freeze_path),
            "legacy_readout": frozen.binding(legacy_path),
            "json_parse_valid": sum(row["json_parse_valid"] for row in rows),
            "json_schema_valid": sum(row["json_schema_valid"] for row in rows),
            "all_labels_present": sum(row["all_labels_present"] for row in rows),
            "all_turn_refs_exist": sum(row["all_turn_refs_exist"] for row in rows),
            "unique_final_answers": sum(row["final_answer_unique"] for row in rows),
            "valid_final_answer_shapes": sum(row["final_answer_shape_valid"] for row in rows),
            "valid_final_answer_labels": sum(row["final_answer_labels_valid"] for row in rows),
            "mechanically_valid": sum(row["mechanically_valid"] for row in rows),
            "legacy_schema_valid_same_questions": sum(row["legacy_schema_valid"] for row in rows),
            "legacy_invalid_same_questions": sum(not row["legacy_schema_valid"] for row in rows),
            "wall_seconds": raw_freeze["wall_seconds"],
            "model_load_seconds": raw_freeze["model_load_seconds"],
            "by_question": rows,
        }
        gate = {
            "schema_valid_at_least_11": metrics["json_schema_valid"] >= 11,
            "labels_present_all_12": metrics["all_labels_present"] == 12,
            "turn_refs_valid_all_12": metrics["all_turn_refs_exist"] == 12,
            "answers_unique_all_12": metrics["unique_final_answers"] == 12,
            "answer_shapes_valid_all_12": metrics["valid_final_answer_shapes"] == 12,
            "answer_labels_valid_all_12": metrics["valid_final_answer_labels"] == 12,
        }
        gate["passed"] = all(gate.values())
        metrics["engineering_gate"] = gate
        report["models"][role] = metrics
    report["status"] = "PASS_FORMAT_ONLY_GATE" if all(
        metrics["engineering_gate"]["passed"] for metrics in report["models"].values()
    ) else "FAIL_FORMAT_ONLY_GATE"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "models": {
        role: {key: value for key, value in metrics.items() if key != "by_question"}
        for role, metrics in report["models"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
