"""Read frozen v98 artifacts and write an explicit stage/accounting receipt.

No network, output editing, semantic grading, or promotion is performed here.
Independent quality decisions remain in the separately authored review files.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parents[1]
RUN = HERE / "expansion_engine/runs/four_roots_B_scene01"


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(RUN / "MANIFEST.json")
    report = read(RUN / "REPORT.json")
    allocation = read(RUN / "UNIQUE_EVENT_ALLOCATION.json")
    roots = manifest["roots"]
    rows = {root: {"root_id": root,
        "source_ids": [x["source_id"] for x in allocation["assignments"][root]],
        "request_roles": Counter(), "mechanically_valid_plans": 0,
        "plans_cleared_for_realization": 0, "realizer_responses": 0,
        "raw_generated_turns": 0, "raw_generated_candidates": [],
        "event_reviews_valid": 0, "scene_reviews_valid": 0,
        "feedback_requests": [], "failures": []} for root in roots}
    roles, finishes, usage = Counter(), Counter(), Counter()
    request_rows, incomplete = [], []
    for folder in sorted((RUN / "requests").iterdir()):
        proof = read(folder / "PROOF.json")
        role, root = proof["role"], proof.get("root_id")
        roles[role] += 1
        target = rows.get(root)
        if target is not None:
            target["request_roles"][role] += 1
        responses = sorted(folder.glob("HTTP_RESPONSE_*.json"))
        for path in responses:
            response = read(path)
            meta = response.get("metadata", {})
            finish = meta.get("finish_reason")
            finishes[str(finish)] += 1
            for key, value in meta.get("usage", {}).items():
                if isinstance(value, int):
                    usage[key] += value
            if finish != "stop":
                incomplete.append({"path": str(path), "finish_reason": finish,
                                   "output_tokens": meta.get("output_tokens")})
        if target is not None and role in ("planner", "realizer"):
            receipt = proof.get("feedback_receipt", {})
            payload = read(folder / "REQUEST.json")
            included = "[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]" in payload["messages"][0]["content"]
            target["feedback_requests"].append({"request": str(folder), "role": role,
                "receipt_present": bool(receipt), "literal_block_present": included,
                "feedback_appended": receipt.get("feedback_appended"),
                "current_generation_feedback_used": receipt.get("current_generation_feedback_used")})
        if target is not None:
            if (folder / "CANDIDATE_PLAN.json").exists():
                target["mechanically_valid_plans"] += 1
            if (folder / "EVENT_STATE_DECISION.json").exists():
                target["plans_cleared_for_realization"] += int(read(folder / "EVENT_STATE_DECISION.json")["status"] == "PASS")
            if role == "event_state_extractor" and (folder / "VALIDATION.json").exists():
                target["event_reviews_valid"] += int(read(folder / "VALIDATION.json")["status"] == "PASS")
            if role == "scene_reviewer" and (folder / "REVIEW.json").exists():
                target["scene_reviews_valid"] += int(read(folder / "REVIEW.json").get("schema_valid") is True)
            if role == "realizer" and (folder / "RESPONSE.json").exists():
                target["realizer_responses"] += 1
                try:
                    body = json.loads(read(folder / "RESPONSE.json")["text"])
                except (ValueError, TypeError):
                    body = {}
                turns = body.get("turns", [])
                count = len(turns) if isinstance(turns, list) else 0
                target["raw_generated_turns"] += count
                target["raw_generated_candidates"].append({"request": str(folder), "turns": count,
                    "acceptance": read(folder / "ACCEPTANCE.json") if (folder / "ACCEPTANCE.json").exists() else None})
        request_rows.append({"path": str(folder), "root_id": root, "role": role,
                             "http_responses": len(responses)})
    for result in report["routes"]:
        row = rows[result["root_id"]]
        row.update({"model_accepted_scenes": result["accepted_scenes"],
                    "model_accepted_new_turns": result["added_turns"],
                    "input_turns": result["input_turns"], "failures": result["failures"]})
    feedback = [q for row in rows.values() for q in row["feedback_requests"]]
    totals = {"assigned_roots": len(roots),
        "card_extraction_requests": roles["dataset_extractor"],
        "usable_cards": len(read(RUN / "dataset_cards_result.json")["accepted_cards"]),
        "roots_with_planner_requests": sum(bool(row["request_roles"]["planner"]) for row in rows.values()),
        "roots_with_plans_cleared_for_realization": sum(bool(row["plans_cleared_for_realization"]) for row in rows.values()),
        "roots_with_realizer_requests": sum(bool(row["request_roles"]["realizer"]) for row in rows.values()),
        "roots_with_generated_turns": sum(bool(row["raw_generated_turns"]) for row in rows.values()),
        "raw_generated_candidates": sum(len(row["raw_generated_candidates"]) for row in rows.values()),
        "raw_generated_turns_across_all_attempts": sum(row["raw_generated_turns"] for row in rows.values()),
        "model_accepted_scenes": report["accepted_scenes"],
        "model_accepted_new_turns": sum(row["model_accepted_new_turns"] for row in rows.values()),
        "http_attempts": report["http_attempts"], "http_max": manifest["http_max"],
        "question_calls": report["question_calls"], "solver_calls": report["solver_calls"]}
    result = {"version": manifest["version"], "run_status": report["status"],
        "totals": totals, "roles": roles, "finish_reasons": finishes, "usage": usage,
        "incomplete_responses": incomplete,
        "actual_previous_training_feedback": {"generator_requests": len(feedback),
            "all_have_receipt_and_literal_block": all(q["receipt_present"] and q["literal_block_present"] and q["feedback_appended"] is True for q in feedback),
            "all_exclude_current_generation_feedback": all(q["current_generation_feedback_used"] is False for q in feedback)},
        "roots": list(rows.values()), "requests": request_rows,
        "semantic_certified_by_this_collector": False,
        "source_selection_bias": read(HERE / "expansion_engine/SOURCE_SELECTION.json")["selection_bias"]}
    output = HERE / "evidence/STAGE_RESULTS.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"totals": totals, "roles": roles, "finish_reasons": finishes,
                      "feedback": result["actual_previous_training_feedback"],
                      "receipt_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
