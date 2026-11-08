"""Resume one rejected review by adding only the missing source-first attestation.

The first RAW_JUDGMENT and SAVE_FAILURE remain untouched. This adapter supplies
no semantic judgment and does not establish that the executor read any text or
avoided private answers; the executing reviewer makes that explicit declaration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_correction(original, corrected):
    require(isinstance(original, dict) and isinstance(corrected, dict), "OBJECTS_REQUIRED")
    att = original.get("read_attestation")
    require(isinstance(att, dict), "ORIGINAL_ATTESTATION_REQUIRED")
    require("private_answers_read_before_save" not in att, "ONLY_MISSING_FIELD_CAN_BE_ADDED")
    expected = deepcopy(original)
    expected["read_attestation"]["private_answers_read_before_save"] = False
    require(corrected == expected, "ONLY_SOURCE_FIRST_FALSE_ADDITION_ALLOWED")


def load_engine(path):
    path = Path(path).resolve()
    require(path.name in {"review_v2.py", "review_v3.py"}, "SUPPORTED_BOUND_ENGINE_REQUIRED")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("attestation_resume_bound_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resume(batch, root, phase, executor, judgment, engine_path, check_only=False):
    require(phase in {"asset", "public"}, "ONLY_SOURCE_FIRST_QA_REVIEW_PHASES")
    engine = load_engine(engine_path)
    data = engine.inputs(batch, root, phase, executor, None)
    directory = engine.read_directory(data)
    context_path = directory / "READ_CONTEXT.json"
    context = engine.read(context_path)
    require((context.get("root_id"), context.get("phase"), context.get("scene_ref"), context.get("executor_id")) ==
            (root, phase, None, executor), "READ_CONTEXT_IDENTITY_MISMATCH")
    require(context.get("input_bindings") == data["input_bindings"], "ORIGINAL_READ_INPUTS_CHANGED")
    for item in [*context["input_bindings"], context["input_render_binding"]]:
        engine.verify_binding(item, data["unit"], "ORIGINAL_READ_CONTEXT_INPUT")
    require(context["input_render_binding"] == engine.binding(directory / "INPUT_RENDER.txt"), "RENDER_PATH_CHANGED")
    require((directory / "INPUT_RENDER.txt").read_text(encoding="utf-8") == engine.render(data), "ORIGINAL_RENDER_CHANGED")
    archive = directory / "RAW_JUDGMENT.json"
    failure_path = directory / "SAVE_FAILURE.json"
    failure = engine.read(failure_path)
    require(failure.get("status") == "FIRST_JUDGMENT_REJECTED_PRESERVED" and
            failure.get("executor_id") == executor and
            failure.get("error") == "EXPLICIT_SOURCE_FIRST_ATTESTATION_REQUIRED", "ONLY_RECORDED_MISSING_ATTESTATION_FAILURE")
    require(failure.get("raw_judgment_binding") == engine.binding(archive), "ORIGINAL_RAW_BINDING_CHANGED")
    original, corrected = engine.read(archive), engine.read(judgment)
    validate_correction(original, corrected)
    attestation = engine.validate_attestation(corrected, data)
    supplied = engine.validate_qa(corrected, data)
    provenance = {"schema_version": "missing_source_first_attestation_resume_v1",
                  "original_raw_judgment_binding": engine.binding(archive),
                  "original_save_failure_binding": engine.binding(failure_path),
                  "corrected_submission_binding": engine.binding(judgment),
                  "bound_validation_engine": engine.binding(engine_path),
                  "resume_adapter_binding": engine.binding(__file__),
                  "only_added_field": "read_attestation.private_answers_read_before_save",
                  "added_value": False, "semantic_dictionary_unchanged": True,
                  "private_file_access_proven_by_program": False}
    if check_only:
        return {"status": "MECHANICAL_RESUME_CHECK_PASS", **provenance}
    correction_dir = directory / "attestation_resume_v1"
    correction_dir.mkdir(exist_ok=False)
    corrected_archive = correction_dir / "CORRECTED_JUDGMENT.json"
    with corrected_archive.open("xb") as stream:
        stream.write(Path(judgment).read_bytes())
    finished = engine.now()
    require(context["started_at_utc"] <= finished, "INVALID_READ_INTERVAL")
    role = "non_author_source_first_public_asset_reviewer" if phase == "asset" else "non_author_source_first_final_public_question_reviewer"
    review = {"schema_version": "codex_assigned_source_review_v2", "root_id": root,
              "reviewer": executor, "review_role": role, "backend": "CURRENT_CODEX_SESSION",
              "exact_model_id": None, "external_semantic_certification": False,
              "started_at_utc": context["started_at_utc"], "finished_at_utc": finished, "created_at_utc": finished,
              "source_sha256": data["source_binding"]["sha256"], "full_source_read": True,
              "turn_refs_read": list(data["turns"]), "all_continuation_constraints_read": True,
              "constraints_binding": data["constraints_binding"], "role_assignments_binding": data["role_assignments_binding"],
              "read_context_binding": engine.binding(context_path), "input_render_binding": context["input_render_binding"],
              "raw_judgment_binding": engine.binding(corrected_archive), "helper_binding": engine.binding(__file__),
              "executor_read_attestation": attestation, "identity_verification_scope": engine.IDENTITY_LIMITATION,
              "read_attestation_scope": engine.READ_LIMITATION, "semantic_review_is_programmatic": False,
              "public_sha256": data["public_binding"]["sha256"], "private_answers_read_before_save": False,
              "independent_of_question_author": True, "attestation_resume_provenance": provenance, **supplied}
    if phase == "public":
        review.update(asset_judgments_sha256=data["asset_judgments_binding"]["sha256"], final_public_questions_actually_reread=True)
    engine.save_new(data["output"], review)
    engine.save_new(correction_dir / "SAVE_RECEIPT.json", {"status": "RECORDED_CORRECTED_ATTESTATION_NOT_EXTERNAL_CERTIFICATION",
                    "created_at_utc": engine.now(), "executor_id": executor, "root_id": root, "phase": phase,
                    "read_context_binding": engine.binding(context_path), "output_binding": engine.binding(data["output"]),
                    "corrected_archive_binding": engine.binding(corrected_archive), **provenance})
    return {"status": "ATTESTATION_RESUME_SAVED", "output": str(data["output"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--phase", choices=["asset", "public"], required=True)
    p.add_argument("--executor", required=True)
    p.add_argument("--judgment", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    import json
    print(json.dumps(resume(args.batch, args.root, args.phase, args.executor, args.judgment, args.engine, args.check_only), ensure_ascii=False))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
