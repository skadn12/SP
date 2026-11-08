"""Finite first-wave sweep of unmaterialized native-domain READY returns.

No model call is possible in this script. Semantic FAIL/HOLD, wrong quotations,
unknown references, completed reviews, and non-native failures are not rescued.
"""
from pathlib import Path
import importlib.util
import json

BASE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def recover():
    fix = module("oet1_native_sweep_validator", BASE / "scripts/source_unit_v3.py")
    driver = module("oet1_native_sweep_frozen_prompts", BASE / "scripts/production_driver.py")
    control = module("oet1_native_sweep_control", BASE / "scripts/control_v3.py")
    rows = []
    roots = fix.read(BASE / "PROTOCOL_FREEZE_V1.json")["first_wave_roots"]
    for root in roots:
        for arm in ("GENERIC", "TARGETED"):
            u = BASE / "sources" / root / arm
            for scene in fix.SCENES[3:]:
                sd = u / scene.replace(":", "_")
                if not (sd / "FAILURE.json").exists() or (sd / "REVIEW.json").exists() or (sd / "ACCEPTED.json").exists():
                    continue
                call_dir = BASE / "cli_calls" / f"{root}__{arm}__{scene.replace(':','_')}__REVIEW"
                row = {"root_id": root, "condition": arm, "scene_ref": scene, "status": "NOT_RECOVERED"}
                if not (call_dir / "RESULT.json").exists() or not (sd / "CUMULATIVE_SOURCE.json").exists():
                    row["reason"] = "NO_COMPLETED_REVIEW_RETURN_OR_SOURCE"; rows.append(row); continue
                response = fix.read(call_dir / "PARSED_RESPONSE.json")
                if response.get("decision") != "READY" or response.get("source_or_gold_ambiguity_failure") is not False:
                    row["reason"] = "SEMANTIC_FAIL_HOLD_OR_AMBIGUITY_IS_TERMINAL"; rows.append(row); continue
                packet = fix.read(u / "SOURCE_AUTHOR_INPUT.json"); source = fix.read(sd / "CUMULATIVE_SOURCE.json")
                _, native, _ = fix.evidence_domains(packet, source, scene)
                failure = fix.read(sd / "FAILURE.json")
                if failure.get("error") not in {repr(ref) for ref in native} or "KeyError" not in failure.get("traceback", "") or failure.get("semantic_retry") is not False:
                    row["reason"] = "ORIGINAL_FAILURE_IS_NOT_EXACT_AUTHORIZED_NATIVE_LOOKUP_KEYERROR"; rows.append(row); continue
                try:
                    # Validate completely before any materialization is written.
                    fix.validate_evidence(response, packet, source, scene)
                    before = {str(p.resolve()): fix.bind(p) for p in sorted(call_dir.rglob("*")) if p.is_file()}
                    result = fix.read(call_dir / "RESULT.json")
                    review, validation = fix.materialize_review(driver, root, arm, scene, response, result, require_ready=True)
                    control.accept(root, arm, scene)
                    after = {str(p.resolve()): fix.bind(p) for p in sorted(call_dir.rglob("*")) if p.is_file()}
                    assert before == after, "ORIGINAL_CALL_ARTIFACTS_CHANGED"
                    receipt = {"schema_version": "oet1_native_reference_engineering_recovery_v2", "status": "COMPLETED_MECHANICAL_STAGE_ONLY",
                               "created_at_utc": fix.now(), "root_id": root, "condition": arm, "scene_ref": scene,
                               "classification": "UNEXECUTED_REVIEW_MATERIALIZATION_VALIDATOR_DOMAIN_BUG",
                               "original_failure_binding": fix.bind(sd / "FAILURE.json"), "original_result_binding": fix.bind(call_dir / "RESULT.json"),
                               "original_raw_binding": result["raw_response"], "review_binding": review, "native_evidence_validation": validation,
                               "accepted_binding": fix.bind(sd / "ACCEPTED.json"), "raw_semantic_fields_unchanged": True,
                               "source_turn_refs_only_in_read_scope": True, "original_failure_preserved": True,
                               "no_new_semantic_request": True, "cli_artifact_bindings": list(before.values()),
                               "review_semantic_request_attempts": len(list(call_dir.glob("attempt_*/START.json"))),
                               "original_cli_artifacts_before_after_equal": True, "author_or_reviewer_calls_launched": 0,
                               "descendant_cli_calls_launched": 0, "recovery_code_binding": fix.bind(__file__),
                               "validator_code_binding": fix.bind(BASE / "scripts/source_unit_v3.py")}
                    row.update(status="RECOVERED_MECHANICAL_ONLY", recovery_binding=fix.save(sd / "ENGINEERING_RECOVERY.json", receipt))
                except Exception as error:
                    row["reason"] = "STRICT_EVIDENCE_OR_PROVENANCE_CHECK_REJECTED"; row["error"] = str(error)
                rows.append(row)
    return {"schema_version": "oet1_finite_native_ready_recovery_sweep_v2", "created_at_utc": fix.now(),
            "rows": rows, "recovered": sum(r["status"] == "RECOVERED_MECHANICAL_ONLY" for r in rows),
            "author_or_reviewer_calls_launched": 0, "code_binding": fix.bind(__file__)}


if __name__ == "__main__":
    value = recover(); fix = module("oet1_native_sweep_io", BASE / "scripts/source_unit_v3.py")
    print(json.dumps({"receipt": fix.save(BASE / "NATIVE_READY_RECOVERY_SWEEP_V2.json", value), "recovered": value["recovered"]}, ensure_ascii=False))
