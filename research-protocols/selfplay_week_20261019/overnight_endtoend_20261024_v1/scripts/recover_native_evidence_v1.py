"""Recover one unexecuted mechanical materialization, never a semantic retry."""
from pathlib import Path
import importlib.util
import json

BASE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def main():
    fix = module("oet1_native_validator_recovery", BASE / "scripts/source_unit_v3.py")
    driver = module("oet1_frozen_driver_for_recovery", BASE / "scripts/production_driver.py")
    control = module("oet1_frozen_control_for_recovery", BASE / "scripts/control.py")
    root = "cc-train-episode-24719"; arm = "GENERIC"; scene = "OET1:B:S01"
    sd = BASE / "sources" / root / arm / scene.replace(":", "_")
    receipt_path = sd / "ENGINEERING_RECOVERY.json"
    assert not receipt_path.exists(), "RECOVERY_ALREADY_RECORDED"
    assert not (sd / "REVIEW.json").exists() and not (sd / "ACCEPTED.json").exists(), "MATERIALIZATION_ALREADY_EXECUTED"
    failure = fix.read(sd / "FAILURE.json")
    assert failure["error"] == "'MUL0526.json:9'" and "KeyError" in failure["traceback"]
    assert failure["semantic_retry"] is False, "UNEXPECTED_ORIGINAL_FAILURE"
    call_dir = BASE / "cli_calls" / f"{root}__{arm}__OET1_B_S01__REVIEW"
    before = {str(p.resolve()): fix.bind(p) for p in sorted(call_dir.rglob("*")) if p.is_file()}
    attempts = sorted(call_dir.glob("attempt_*/START.json"))
    assert len(attempts) == 1 and attempts[0].parent.name == "attempt_01", "SEMANTIC_REQUEST_MULTIPLICITY_CHANGED"
    response = fix.read(call_dir / "PARSED_RESPONSE.json"); result = fix.read(call_dir / "RESULT.json")
    assert response["decision"] == "READY" and response.get("source_or_gold_ambiguity_failure") is False
    review_binding, validation_binding = fix.materialize_review(driver, root, arm, scene, response, result, require_ready=True)
    accepted = control.accept(root, arm, scene)
    after = {str(p.resolve()): fix.bind(p) for p in sorted(call_dir.rglob("*")) if p.is_file()}
    assert before == after, "ORIGINAL_CLI_ARTIFACTS_CHANGED_OR_NEW_REQUEST_MADE"
    review = fix.read(sd / "REVIEW.json")
    assert all(review[k] == v for k, v in response.items()), "SEMANTIC_REVIEW_CHANGED"
    receipt = {"schema_version": "oet1_native_reference_engineering_recovery_v1", "status": "COMPLETED_MECHANICAL_STAGE_ONLY",
               "created_at_utc": fix.now(), "root_id": root, "condition": arm, "scene_ref": scene,
               "classification": "UNEXECUTED_REVIEW_MATERIALIZATION_VALIDATOR_DOMAIN_BUG",
               "reason": "Original READY response cited the exact assigned native utterance MUL0526.json:9; source-only lookup raised KeyError before REVIEW creation.",
               "original_failure_binding": fix.bind(sd / "FAILURE.json"), "original_result_binding": fix.bind(call_dir / "RESULT.json"),
               "original_raw_binding": result["raw_response"], "review_binding": review_binding,
               "native_evidence_validation": validation_binding, "accepted_binding": fix.bind(sd / "ACCEPTED.json"),
               "source_binding": fix.bind(sd / "CUMULATIVE_SOURCE.json"), "draft_binding": fix.bind(sd / "DRAFT.json"),
               "source_turn_refs_only_in_read_scope": True, "original_failure_preserved": True,
               "raw_semantic_fields_unchanged": True, "no_new_semantic_request": True,
               "review_semantic_request_attempts": len(attempts), "original_cli_artifacts_before_after_equal": True,
               "cli_artifact_bindings": list(before.values()), "recovery_code_binding": fix.bind(__file__),
               "validator_code_binding": fix.bind(BASE / "scripts/source_unit_v3.py"),
               "author_or_reviewer_calls_launched": 0, "descendant_cli_calls_launched": 0}
    binding = fix.save(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": binding, "accepted": accepted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
