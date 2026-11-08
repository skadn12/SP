"""Bind rejected original source reviews to the unchanged question freeze.

This is a read-only verification followed by one immutable addendum write.  It
does not accept sources/questions, invoke a model, or alter original returns.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

BASE = Path(__file__).resolve().parents[1]

def bind(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

def read(path): return json.loads(Path(path).read_text())

def verify_tree(value, verified):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            expected = {k: value[k] for k in ("path", "sha256", "bytes")}
            assert bind(value["path"]) == expected, ("BINDING_DRIFT", value["path"])
            verified[value["path"]] = expected
        for item in value.values(): verify_tree(item, verified)
    elif isinstance(value, list):
        for item in value: verify_tree(item, verified)

def main():
    out = BASE / "QUESTION_FREEZE_SOURCE_REVIEW_ADDENDUM_V1.json"
    assert not out.exists(), "ADDENDUM_ALREADY_FROZEN"
    protocol = read(BASE / "PROTOCOL_FREEZE_V1.json")
    freeze = read(BASE / "QUESTION_FREEZE.json")
    gate = read(BASE / "FIRST_WAVE_ENGINEERING_GATE.json")
    assert freeze["planned_slots"] == 112 and gate["planned_slots"] == 56
    assert gate["status"] == "FAIL"
    verified = {}; rejection_rows = []; ambiguity_rows = []; scene_rows = []
    for root in protocol["first_wave_roots"]:
        for arm in ("GENERIC", "TARGETED"):
            unit = BASE / "sources" / root / arm
            for scene in protocol["scene_sequence"]:
                sd = unit / scene.replace(":", "_")
                row = {"root_id": root, "condition": arm, "scene_ref": scene,
                       "draft_exists": (sd / "DRAFT.json").exists(),
                       "accepted": (sd / "ACCEPTED.json").exists(),
                       "failure_record_exists": (sd / "FAILURE.json").exists()}
                row["unresolved_terminal_failure"] = row["failure_record_exists"] and not row["accepted"]
                review_dir = BASE / "cli_calls" / f"{root}__{arm}__{scene.replace(':','_')}__REVIEW"
                rp = review_dir / "PARSED_RESPONSE.json"
                if rp.exists():
                    raw = read(rp)
                    row["raw_review_binding"] = bind(rp)
                    row["original_review_decision"] = raw["decision"]
                    row["original_source_or_gold_ambiguity_failure"] = raw.get("source_or_gold_ambiguity_failure")
                    row["original_reported_turn_count"] = raw.get("source_turn_count")
                    row["actual_source_turn_count"] = len(read(sd / "CUMULATIVE_SOURCE.json")["turns"])
                    if raw.get("source_or_gold_ambiguity_failure") is True or "SOURCE_AMBIGUITY" in raw.get("failure_codes", []):
                        ambiguity_rows.append({"root_id": root, "condition": arm, "scene_ref": scene,
                                               "raw_review_binding": bind(rp),
                                               "materialized_review_binding": bind(sd / "REVIEW.json") if (sd / "REVIEW.json").exists() else None,
                                               "original_review_decision": raw["decision"],
                                               "original_failure_codes": raw.get("failure_codes", [])})
                for path in sorted(sd.glob("*REJECTION.json")):
                    rejected = read(path)
                    assert rejected["root_id"] == root and rejected["condition"] == arm and rejected["scene_ref"] == scene
                    assert rejected["accepted"] is False and rejected["semantic_retry"] is False
                    verify_tree(rejected, verified)
                    rejection_rows.append({"root_id": root, "condition": arm, "scene_ref": scene,
                                           "status": rejected["status"], "rejection_binding": bind(path),
                                           "original_review_decision": rejected.get("review_decision")})
                scene_rows.append(row)
    keys = [(r["root_id"], r["condition"], r["scene_ref"]) for r in ambiguity_rows]
    assert len(keys) == len(set(keys)), "AMBIGUITY_SCENE_COUNT_DUPLICATED"
    assert len(ambiguity_rows) == gate["source_ambiguity_failure_count"] == 4
    assert len(rejection_rows) == 3
    assert {r["rejection_binding"]["path"] for r in rejection_rows} == {r["path"] for r in gate["source_review_rejection_bindings"]}
    assert len(scene_rows) == 48
    value = {"schema_version": "oet1_question_freeze_source_review_addendum_v1", "status": "FROZEN_SOURCE_REVIEW_EVIDENCE_ADDENDUM",
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "question_freeze_binding": bind(BASE / "QUESTION_FREEZE.json"),
             "engineering_gate_binding": bind(BASE / "FIRST_WAVE_ENGINEERING_GATE.json"),
             "production_integrity_binding": bind(BASE / "PRODUCTION_ENGINEERING_AUDIT_FINAL.json"),
             "question_freeze_original_bytes_unchanged": True,
             "planned_new_sources": 16, "planned_new_scenes": 96, "planned_new_question_slots": 112,
             "first_wave_planned_sources": 8, "first_wave_planned_scenes": 48, "first_wave_planned_question_slots": 56,
             "first_wave_counts": {"draft_scenes": sum(r["draft_exists"] for r in scene_rows),
                                   "accepted_scenes": sum(r["accepted"] for r in scene_rows),
                                   "failure_record_scenes": sum(r["failure_record_exists"] for r in scene_rows),
                                   "unresolved_terminal_failure_scenes": sum(r["unresolved_terminal_failure"] for r in scene_rows),
                                   "fully_accepted_sources": sum((BASE / "sources" / root / arm / "FINAL_SOURCE.json").exists() for root in protocol["first_wave_roots"] for arm in ("GENERIC", "TARGETED")),
                                   "admitted_questions": gate["overall"]["admitted"]},
             "source_ambiguity_failure_scene_count": len(ambiguity_rows),
             "source_ambiguity_scenes": ambiguity_rows,
             "explicit_rejection_scene_count": len(rejection_rows), "explicit_rejections": rejection_rows,
             "source_scene_audit": scene_rows, "verified_rejection_artifact_bindings": list(verified.values()),
             "all_rejection_bindings_verified": True, "rejection_identity_bound_to_directory": True,
             "semantic_retries": 0, "data_or_model_calls": 0, "question_admission_changes": 0,
             "new_root_primary_objective_completed": False,
             "scope": "Source-review failures remain distinct from QFP2-only historical pilot training.",
             "addendum_code_binding": bind(__file__)}
    with out.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"binding": bind(out), "counts": value["first_wave_counts"]}, ensure_ascii=False))

if __name__ == "__main__": main()
