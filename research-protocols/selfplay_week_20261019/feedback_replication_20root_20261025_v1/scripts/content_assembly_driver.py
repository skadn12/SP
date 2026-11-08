#!/usr/bin/env python3
"""Freeze actual initial and final content using the unchanged question driver.

Invoked without arguments after source promotion and question stages settle.
This wrapper changes no authored content or semantic reviewer conclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q  # noqa: E402
import assemble_initial_content as initial  # noqa: E402
import content_continuation_adapter as continuation  # noqa: E402


def adapter_launch_bindings(base: Path = BASE) -> list[Path]:
    """Bind actual launch provenance, including the approved future v2 scope.

    This validates recorded engineering identity only.  It never transforms a
    prompt, candidate, Gold, review decision or finalizer gate result.
    """
    paths = []
    for family in ("review_schema_adapter", "review_schema_adapter_v2"):
        launches = sorted((base / family / "launches").glob("*.json"))
        approved = None
        if family.endswith("_v2") and launches:
            ready = base / "REVIEW_SCHEMA_ADAPTER_V2_READY.json"
            approved = q.read(ready)
            script = base / "scripts/review_schema_adapter_v2.py"
            if (approved.get("schema_version") != "fgr20_ram_review_schema_v2_ready_v1"
                    or approved.get("status") != "READY_ENGINEERING_ONLY"
                    or approved.get("adapter_binding") != initial.binding(script)
                    or approved["adapter_binding"]["sha256"] != "3d8bf15b6e42655fb4ca5d53fc0fde00f2da7970f3dafae1660c53987b469645"
                    or approved.get("semantic_calls_started_by_engineering") != 0
                    or approved.get("semantic_admission_asserted") is not False):
                raise RuntimeError("ASSEMBLY_V2_ADAPTER_APPROVAL_DRIFT")
            for row in initial.nested_bindings(approved):
                q.verify_binding(row)
                paths.append(Path(row["path"]))
            paths.append(ready)
        for path in launches:
            doc = q.read(path)
            if approved is not None:
                targets = [target for target in approved.get("future_targets", [])
                           if (target.get("driver"), target.get("root_id"), target.get("arm"))
                           == (doc.get("driver"), doc.get("root_id"), doc.get("arm"))]
                if (doc.get("schema_version") != "fgr20_ram_review_schema_v2_launch_v1"
                        or doc.get("adapter_binding") != approved["adapter_binding"]
                        or doc.get("frozen_v1_adapter_binding") != approved["frozen_v1_adapter_binding"]
                        or doc.get("future_unstarted_stage_only") is not True
                        or doc.get("raw_and_old_validation_and_review_artifacts_preserved") is not True
                        or doc.get("extra_semantic_budget") != 0
                        or doc.get("semantic_admission_asserted") is not False
                        or len(targets) != 1
                        or targets[0].get("stage_request_count_at_freeze") != 0
                        or shlex.split(targets[0].get("command", "")) != [
                            "/root/miniconda3/bin/python", str(base / "scripts/review_schema_adapter_v2.py"),
                            "launch", "--driver", doc.get("driver"), "--", *doc.get("driver_argv", [])]):
                    raise RuntimeError("ASSEMBLY_V2_LAUNCH_SCOPE_OR_BINDING_DRIFT:" + str(path))
            for row in initial.nested_bindings(doc):
                q.verify_binding(row)
                paths.append(Path(row["path"]))
            paths.append(path)
    return sorted(set(paths))


def extra_bindings() -> list[Path]:
    interim = BASE / "source_stage_interim" / "five_reviewed_sources"
    paths = [Path(__file__).resolve(), Path(initial.__file__).resolve(), Path(continuation.__file__).resolve(),
             BASE / "scripts" / "question_driver_recovery.py",
             BASE / "scripts" / "solver_runtime_adapter.py",
             BASE / "scripts" / "solver_cpu_length_precheck.py",
             BASE / "scripts" / "production_driver.py",
             interim / "SOURCE_SELECTION.json", interim / "SOURCE_FREEZE.json", interim / "SOURCE_FAILURES.json",
             BASE / "ANALYSIS_DRIVER_FREEZE.json",
             BASE / "RUNTIME_INTERPRETER_ADAPTATION.json",
             BASE / "QUESTION_DRIVER_AUDIT_READY.json"]
    source_publication = BASE / "SOURCE_FILE_PUBLICATION.json"
    if source_publication.is_file():
        paths.append(source_publication)
    for name in ("scripts/question_plan_schema_continuation_v2.py",
                 "scripts/question_plan_schema_continuation_v3.py",
                 "scripts/question_candidate_shape_continuation.py",
                 "scripts/review_schema_adapter.py",
                 "scripts/review_schema_adapter_v2.py",
                 "scripts/unused_failed_lineage_adapter.py",
                 "scripts/saved_revision_gold_verification.py",
                 "scripts/g136202_saved_global_gold_verification.py",
                 "scripts/g109164_saved_global7_gold_verification.py",
                 "scripts/f179186_saved_shape_gold_verification.py",
                 "QUESTION_PLAN_SCHEMA_CONTINUATION_V2_READY.json",
                 "QUESTION_PLAN_SCHEMA_CONTINUATION_V3_READY.json",
                 "QUESTION_CANDIDATE_SHAPE_CONTINUATION_READY.json",
                 "REVIEW_SCHEMA_ADAPTER_READY.json",
                 "REVIEW_SCHEMA_ADAPTER_V2_READY.json",
                 "QUESTION_DRIVER_ENGINEERING_HANDOFF_V2.json",
                 "UNUSED_FAILED_LINEAGE_ADAPTER_READY.json",
                 "G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                 "G109164_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json",
                 "F179186_SAVED_SHAPE_GOLD_VERIFICATION_READY.json",
                 "PRODUCTION_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                 "PRODUCTION_G109_QUESTION_PROCESS_GUARD_READY.json",
                 "PRODUCTION_G109_QUESTION_PROCESS_GUARD_HISTORY_LOCATOR.json",
                 "SAVED_REVISION_GOLD_VERIFICATION_READY.json"):
        path = BASE / name
        if path.is_file():
            paths.append(path)
    process_guard_ready = BASE / "PRODUCTION_QUESTION_PROCESS_GUARD_ENGINEERING_READY.json"
    if process_guard_ready.is_file():
        paths.append(process_guard_ready)
    paths.extend(adapter_launch_bindings())
    paths.extend(BASE / name for name in initial.OUTPUT_NAMES)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"REQUIRED_ACTUAL_ASSEMBLY_BINDING_MISSING:{path}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        result = initial.self_check()
        result["continuation_path_adapter_self_check"] = continuation.self_check()
        result["wrapper_invokes_only_initial_aggregation_and_frozen_qdriver_finalize"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    audit = q.read(BASE / "QUESTION_DRIVER_AUDIT_READY.json")
    q.verify_binding(audit["driver_binding"])
    analysis_freeze = q.read(BASE / "ANALYSIS_DRIVER_FREEZE.json")
    if analysis_freeze.get("status") != "FROZEN_BEFORE_SOLVER":
        raise RuntimeError("ANALYSIS_DRIVER_NOT_FROZEN")
    q.verify_binding(analysis_freeze["script_binding"])
    q.verify_binding(analysis_freeze["analysis_plan_binding"])
    q.SOURCE_SELECTION_PATH = BASE / "SOURCE_SELECTION_FINAL.json"
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    continuations = continuation.Continuations()
    receipt = initial.assemble(q.SOURCE_SELECTION_PATH, continuations.initial_work,
                               continuations.inputs + [initial.binding(Path(continuation.__file__).resolve())])
    q.EXTRA_DRIVER_BINDINGS = list(q.EXTRA_DRIVER_BINDINGS) + extra_bindings()
    freeze = continuation.finalize(continuations)
    print(json.dumps({"initial_assembly": receipt, "final_content_freeze": freeze}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
