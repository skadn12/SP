#!/usr/bin/env python3
"""Check saved F179186 wrapper/finalizer interfaces with no model dispatch.

All scientific-shaped test outputs live under TemporaryDirectory and are
deleted. Only an append-only engineering receipt is published. Existing
scheduler dependencies, authoring artifacts and review decisions are read only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import content_continuation_adapter as adapter
import finalize_report as report
import question_driver as q
import resume_authorized_pipeline as scheduler


def main() -> int:
    ready_path = BASE / "F179186_SAVED_SHAPE_GOLD_VERIFICATION_READY.json"
    ready = q.read(ready_path)
    if (ready.get("status") != "READY_ENGINEERING_ONLY"
            or ready.get("root_id") != "cc-train-episode-179186" or ready.get("arm") != "F"
            or ready.get("serialization_variant") != "f179186_shape_final_revision_one_r03_alias_no_global_copy"
            or ready.get("semantic_calls_started_by_engineering") != 0
            or ready.get("semantic_admission_asserted") is not False):
        raise RuntimeError("DEDICATED_F179186_ENGINEERING_READY_IDENTITY_DRIFT")
    import f179186_saved_shape_gold_verification as wrapper
    if ready.get("wrapper_binding") != q.binding(wrapper.__file__):
        raise RuntimeError("DEDICATED_F179186_WRAPPER_BINDING_DRIFT")
    for field in ("wrapper_binding", "frozen_saved_verifier_helper_binding",
                  "frozen_RAM_adapter_binding", "engineering_checks_binding"):
        q.verify_binding(ready[field])
    scheduler_ready = q.read(scheduler.READY)
    for row in scheduler_ready["dependency_bindings"]:
        scheduler.verify(row)
    q.SOURCE_SELECTION_PATH = wrapper.SELECTION
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    _, sources = q.load_selection(wrapper.SELECTION)
    source = sources[wrapper.ROOT]
    before = wrapper.saved.actual_requests(wrapper.ROOT, wrapper.ARM)
    frozen = wrapper.inspect_inputs(source)
    reviewer, _, clone = wrapper.build_review()
    candidates = q.read(wrapper.INPUT_WORK / "CANDIDATES_FINAL.json")
    calls = []

    class NoDispatch:
        def call(self, key, prompt):
            if key != wrapper.saved.call_id(wrapper.ROOT, wrapper.ARM):
                raise AssertionError("Unexpected semantic call identity in engineering fixture")
            calls.append({"call_id": key, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            raise RuntimeError("ENGINEERING_INTERFACE_MOCK_NO_MODEL_DISPATCH")

    cache = BASE / "runtime_cache/f179186_content_interface"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="interface_", dir=cache) as directory:
        out = Path(directory)
        q.publish(out / "RECOVERY_INPUT_FREEZE.json", frozen)
        data = reviewer(wrapper.ROOT, wrapper.ARM, candidates, source, wrapper.SELECTION, out, frozen, NoDispatch())
        result = wrapper.publish_variant_ready(wrapper.ROOT, wrapper.ARM, wrapper.INPUT_WORK, out, frozen, data)
        checker = adapter.Continuations.__new__(adapter.Continuations)
        checker.inputs = []
        observed = checker.validate_ready(out / adapter.READY_NAME)
        assert len(calls) == 1
        assert result["serialization_variant"] == wrapper.VARIANT
        assert observed["root_id"] == wrapper.ROOT and observed["arm"] == wrapper.ARM
        assert observed["supersedes_ready_binding"] == q.binding(wrapper.INPUT_WORK / adapter.READY_NAME)
        assert len(q.read(out / "CANDIDATES_FINAL.json")) == 8
        assert (out / "CANDIDATES_FINAL.json").read_bytes() == (wrapper.INPUT_WORK / "CANDIDATES_FINAL.json").read_bytes()
        assert len(data["reviews"]) == 8 and all(row["decision"] == "HOLD" for row in data["reviews"])
        assert all(row["admitted"] is False for row in data["gate_validations"])
        assert not (out / "CONTENT_VERIFIER_CALL_revision_01.json").exists()
        retained = len(checker.inputs)
        exact_normalizer = q.read(out / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json")
        assert exact_normalizer["adapter_binding"] == q.binding(wrapper.__file__)
        assert len(exact_normalizer["changes"]) == 8
    assert wrapper.saved.actual_requests(wrapper.ROOT, wrapper.ARM) == before
    for row in scheduler_ready["dependency_bindings"]:
        scheduler.verify(row)
    now = dt.datetime.now(dt.timezone.utc)
    receipt = {"schema_version": "fgr20_F179186_content_interface_engineering_check_v1",
        "at_utc": now.isoformat(), "status": "PASS_ENGINEERING_ONLY",
        "root_id": wrapper.ROOT, "arm": wrapper.ARM, "serialization_variant": wrapper.VARIANT,
        "script_binding": report.bind(Path(__file__)), "wrapper_ready_binding": report.bind(ready_path),
        "wrapper_binding": q.binding(wrapper.__file__),
        "existing_content_adapter_binding": q.binding(adapter.__file__),
        "scheduler_ready_binding": report.bind(scheduler.READY),
        "scheduler_dependency_count_reverified_before_and_after": len(scheduler_ready["dependency_bindings"]),
        "exact_saved_source_and_final_revision_candidates_read": True,
        "exact_final_candidate_bytes_unchanged": True, "final_candidate_count": 8,
        "final_mock_review_count": 8, "mock_no_response_HOLD_count": 8,
        "temporary_READY_passes_actual_frozen_adapter": True,
        "retained_explicit_input_binding_count": retained,
        "supersession_to_original_shape_READY_verified": True,
        "saved_eight_single_R03_alias_changes": 8, "global_attestation_copies": 0,
        "concrete_wrapper_identity_in_normalization_verified": True,
        "mock_only_first_verifier_reached": calls, "clone": clone,
        "actual_request_registry_unchanged": True, "temporary_outputs_removed": True,
        "semantic_calls_started": 0, "semantic_judgments_authored": 0,
        "actual_content_freeze_report_or_solver_launched": False,
        "existing_scheduler_dependency_files_modified": False}
    path = BASE / "mechanical_audits" / ("F179186_CONTENT_INTERFACE_ENGINEERING_CHECK_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    report.publish_immutable(path, receipt)
    print(json.dumps(report.bind(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
