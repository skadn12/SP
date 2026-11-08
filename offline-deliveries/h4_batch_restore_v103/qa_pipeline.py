"""v103 orchestration of unchanged v98 mixed QA, stopping before Solver.

This file routes existing commands and binds actual accepted expansions. It
does not create semantic READY judgments, public facts, answer keys, reviews,
or replacement dialogue. All semantic records must already have been saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HOME = Path(__file__).resolve().parent
V98 = HOME.parent / "h4_event_identity_correction_pilot_v98"
RUN = HOME / "expansion_engine/runs/four_roots_B_scene01"
PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")
sys.dont_write_bytecode = True


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def save_identical(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        need(path.read_bytes() == data, "refuse_overwrite_changed_record:" + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def copy_identical(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        need(destination.read_bytes() == source.read_bytes(), "reused_code_changed:" + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())


def execute(argv, cwd, label):
    """Retain each invocation's stdout, including failures, without rerunning it."""
    logs = HOME / "evidence/qa_execution"
    logs.mkdir(parents=True, exist_ok=True)
    ordinal = 1
    while (logs / f"{ordinal:03d}-{label}.log").exists():
        ordinal += 1
    path = logs / f"{ordinal:03d}-{label}.log"
    with path.open("xb") as stream:
        result = subprocess.run([str(PYTHON), "-B", *map(str, argv)], cwd=cwd,
                                stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"created_at_utc": now(), "argv": list(map(str, argv)), "cwd": str(cwd),
               "exit_code": result.returncode, "log": bind(path),
               "solver_or_training_requested": False}
    save_identical(path.with_suffix(".json"), receipt)
    if result.returncode:
        raise RuntimeError("qa_command_failed:" + str(path))
    return receipt


def bootstrap():
    reused = []
    for name in ("materialize_expansion.py", "build_fresh_shard.py"):
        source, target = V98 / "evidence" / name, HOME / "evidence" / name
        copy_identical(source, target)
        reused.append({"upstream": bind(source), "installed": bind(target), "byte_identical": True})
    for source in sorted((V98 / "qa_engine").glob("*.py")):
        target = HOME / "qa_engine" / source.name
        need(target.read_bytes() == source.read_bytes(), "qa_engine_not_original:" + source.name)
        reused.append({"upstream": bind(source), "installed": bind(target), "byte_identical": True})
    receipt = HOME / "evidence/QA_CODE_REUSE.json"
    value = {"version": "v103_qa_code_reuse_v1", "qa_semantics_version": "unchanged_v98",
             "record_version_note": "v98_* record schema versions describe the unchanged validator contract; all runtime paths belong to this new v103 delivery.",
             "reused": reused, "qa_rule_changes": [], "network_calls": 0,
             "solver_calls": 0, "training_allowed": False}
    save_identical(receipt, value)
    return value


def bind_expansion():
    """The existing scope validator expects final actual run bindings."""
    bootstrap()
    manifest, report, audit = [read(RUN / n) for n in ("MANIFEST.json", "REPORT.json", "FINAL_AUDIT.json")]
    roots = manifest["roots"]
    need(len(roots) == len(set(roots)) == 4, "four_preselected_roots_required")
    need(audit["status"] == "PASS" and audit["failures"] == [] and audit["semantic_certified"] is False,
         "expansion_final_structural_audit_required")
    plan = {"version": "v98_four_root_semantic_review_mixed_v1", "runtime_delivery": "v103",
            "candidate_root_budget": 4, "candidate_roots": roots, "source_split": "train",
            "question_types": ["multi_select", "ordering"], "training_allowed": False,
            "formal_training_admission": False, "max_generation_http_total": 12,
            "max_solver_calls_per_model_total": 456, "solver_execution_in_this_batch": False,
            "exclude_repeated_roots": [], "exclude_known_hold_roots": [],
            "upstream_expansion_run": {name: bind(RUN / name) for name in
                                       ("MANIFEST.json", "REPORT.json", "FINAL_AUDIT.json")},
            "selection_disclosure": "Use only this run's four frozen roots and its independently reviewed READY subset, in original order. Preserve all expansion and review failures. No historical-source fallback, new QA rule or claim of blind sampling.",
            "source_ready_certified_by_this_binding": False}
    sys.path.insert(0, str(HOME / "qa_engine"))
    import batch_scope
    batch_scope.validate_plan(plan)
    save_identical(HOME / "PILOT_PLAN.json", plan)
    return plan


def materialize():
    bind_expansion()
    return execute([HOME / "evidence/materialize_expansion.py"], HOME, "materialize")


def ready_status():
    """Validate existing source-review records; never write a READY decision."""
    plan = read(HOME / "PILOT_PLAN.json")
    sys.path.insert(0, str(HOME / "qa_engine"))
    import fresh_parent
    result = []
    for root in plan["candidate_roots"]:
        number = root.rsplit("-", 1)[-1]
        provenance = HOME / "evidence/expansion_provenance" / ("root" + number + "_PROVENANCE.json")
        if not provenance.exists():
            result.append({"root_id": root, "status": "NO_ACCEPTED_SOURCE"})
            continue
        lineage = read(provenance)
        review_path = HOME / "inputs" / ("root" + number + "_SOURCE_REVIEW.json")
        if not review_path.exists():
            result.append({"root_id": root, "status": "AWAITING_FULL_SOURCE_REVIEW",
                           "source": lineage["output_source"], "review_path": str(review_path)})
            continue
        review = fresh_parent.validate_source_review({"root_id": root, "source": lineage["output_source"],
                                                       "review_path": str(review_path)})
        result.append({"root_id": root, "status": review["decision"],
                       "source": lineage["output_source"], "review": bind(review_path)})
    return {"roots": result, "ready_roots": [r["root_id"] for r in result if r["status"] == "READY"],
            "semantic_judgments_authored_by_harness": False}


def prepare_inputs(shard_id, multi_prior, ordering_prior):
    """Use saved six-fact judgments and two explicit six-turn plans per READY root."""
    status = ready_status()
    need(not any(r["status"] == "AWAITING_FULL_SOURCE_REVIEW" for r in status["roots"]),
         "finish_full_source_reviews_before_selecting_ready_subset")
    rows = []
    for root in status["ready_roots"]:
        number = root.rsplit("-", 1)[-1]
        prefix = HOME / "inputs" / ("root" + number)
        rows.append({"root_id": root,
                     "public_parent": str(prefix) + "_PUBLIC_PARENT.json",
                     "parent_judgments": str(prefix) + "_PARENT_JUDGMENTS.json",
                     "source_review": str(prefix) + "_SOURCE_REVIEW.json",
                     "expanded_source_provenance": str(HOME / "evidence/expansion_provenance" /
                                                        ("root" + number + "_PROVENANCE.json")),
                     "public_prior_paths": [str(Path(x).resolve()) for x in multi_prior],
                     "public_ordering_prior_paths": [str(Path(x).resolve()) for x in ordering_prior]})
    plans = HOME / "inputs/SCENE_PLANS.json"
    need(plans.exists(), "saved_ordering_anchor_plans_required_even_if_empty")
    spec = {"version": "v98_ready_subset_shard_spec_v1", "shard_id": shard_id, "roots": rows,
            "scene_plans": str(plans), "plan_path": str(HOME / "PILOT_PLAN.json")}
    spec_path = HOME / "inputs/SHARD_SPEC.json"
    save_identical(spec_path, spec)
    execute([HOME / "evidence/build_fresh_shard.py", spec_path, "--check"], HOME, "check_inputs")
    return execute([HOME / "evidence/build_fresh_shard.py", spec_path], HOME, "build_shard")


def batch_action(action, shard_id, slot=None, round_id="initial"):
    allowed = {"prepare", "verify", "generate", "finish-asset-review", "finalize-multiselect",
               "finish-multiselect-review", "finish-ordering-review", "freeze"}
    need(action in allowed, "solver_and_training_outside_v103_qa_harness")
    shard = HOME / "shards" / shard_id
    need(shard.resolve().parent == (HOME / "shards").resolve(), "unsafe_shard_id")
    argv = ["batch.py", action]
    if slot is not None:
        argv += ["--slot", slot, "--round", round_id]
    return execute(argv, shard, action.replace("-", "_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "bind-expansion", "materialize", "ready-status",
                        "prepare-inputs", "prepare", "verify", "generate", "finish-asset-review",
                        "finalize-multiselect", "finish-multiselect-review", "finish-ordering-review", "freeze"))
    parser.add_argument("--shard-id", default="shard_01")
    parser.add_argument("--slot")
    parser.add_argument("--round", choices=("initial", "supplement"), default="initial")
    parser.add_argument("--multiselect-prior", action="append", default=[])
    parser.add_argument("--ordering-prior", action="append", default=[])
    args = parser.parse_args()
    if args.action == "bootstrap": result = bootstrap()
    elif args.action == "bind-expansion": result = bind_expansion()
    elif args.action == "materialize": result = materialize()
    elif args.action == "ready-status": result = ready_status()
    elif args.action == "prepare-inputs": result = prepare_inputs(args.shard_id, args.multiselect_prior, args.ordering_prior)
    else: result = batch_action(args.action, args.shard_id, args.slot, args.round)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
