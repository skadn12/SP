"""Collect two frozen roots with 20 slots each, then evaluate both.

Preparation and capacity checks are local only. Stages execute once, with no
automatic replay, replacement slots, or current Solver feedback to the writer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
BASE = REPORT.parent
CODE = BASE / "h4_sequential_ab_v55_qa"
PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")
sys.path.insert(0, str(CODE))
import batch

gen = batch.gen
ROOTS = ("135466", "210046")
SLOTS_PER_ROOT = 20
VERSION = "h4_question_collection_v55_40_slots_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_binding(row):
    if binding(row["path"]) != row:
        raise ValueError("frozen_file_changed:" + row["path"])


def batch_run(short):
    return CODE / "runs" / f"root{short}_collect20_01"


def marker(stage, suffix):
    return REPORT / f"{stage}.{suffix}.json"


def begin(stage):
    if any(marker(stage, suffix).exists() for suffix in ("STARTED", "COMPLETE", "ERROR")):
        raise ValueError("stage_already_recorded_no_replay:" + stage)
    save(marker(stage, "STARTED"), {"version": VERSION, "stage": stage,
         "started_at_utc": now(), "automatic_replay": False})


def record_error(stage, error, completed, current=None):
    if not marker(stage, "ERROR").exists():
        save(marker(stage, "ERROR"), {"version": VERSION, "stage": stage,
             "failed_at_utc": now(), "error_type": type(error).__name__,
             "error": str(error), "traceback": traceback.format_exc(),
             "completed_roots": completed, "current_operation": current,
             "completed_artifacts_preserved": True, "automatic_replay": False})


def public_question(question):
    field = "options" if question["question_type"] == "multi_select" else "items"
    values = question[field]
    if not isinstance(values, dict) or len(values) != 6 or not all(isinstance(v, str) for v in values.values()):
        raise ValueError("historical_public_question_requires_six_texts")
    return {key: deepcopy(question[key]) for key in
            ("candidate_id", "root_id", "question_type", "question", field)}


def whole_set_key(question):
    field = "options" if question["question_type"] == "multi_select" else "items"
    return question["question_type"], tuple(sorted(" ".join(text.split()).casefold()
                                                   for text in question[field].values()))


def build_prior(short):
    root = "cc-train-episode-" + short
    source_paths = []
    for version, count in (("v51", 5), ("v52b", 3), ("v53", 1), ("v54", 1)):
        paths = sorted((BASE / f"h4_sequential_ab_{version}_qa/runs").glob(
            f"root{short}_*_generation/MECHANICAL_POOL.json"))
        if len(paths) != count:
            raise ValueError(f"historical_prior_file_count_changed:{root}:{version}:{len(paths)}")
        source_paths.extend(paths)
    questions, duplicates, seen = [], [], {}
    original_count = 0

    def add(question):
        nonlocal original_count
        original_count += 1
        if question["root_id"] != root:
            raise ValueError("historical_prior_root_mismatch")
        public = public_question(question)
        key = whole_set_key(public)
        if key in seen:
            duplicates.append({"candidate_id": public["candidate_id"],
                               "retained_candidate_id": seen[key]})
        else:
            seen[key] = public["candidate_id"]
            questions.append(public)

    for path in source_paths:
        for question in read(path)["questions"]:
            add(question)
    historical_rejected_paths = []
    if short == "135466":
        response = (BASE / "h4_sequential_ab_v54_qa/runs/root135466_rules4_01_generation/requests/"
                    "001-135466-merged-multi-generate-1/RESPONSE.json")
        raw = json.loads(read(response)["text"])
        expected = [f"{root}-v52-v54_root135466_rules4_01-merged-multi-b01-{index:02d}"
                    for index in (1, 2)]
        if raw["root_id"] != root or [q["candidate_id"] for q in raw["questions"]] != expected:
            raise ValueError("historical_five_correct_response_identity_changed")
        for draft in raw["questions"]:
            if len(draft["options"]) != 6 or sum(option["relation"] == "supports" for option in draft["options"]) != 5:
                raise ValueError("historical_five_correct_response_shape_changed")
            add({"candidate_id": draft["candidate_id"], "root_id": root,
                 "question_type": "multi_select", "question": draft["question"],
                 "options": {label: option["proposition"] for label, option in zip("ABCDEF", draft["options"])}})
        historical_rejected_paths.append(response)
    path = REPORT / "priors" / f"{root}_v51_v52b_v53_v54_public.json"
    save(path, {"questions": questions, "source_files": [binding(p) for p in source_paths],
         "historical_rejected_response_files": [binding(p) for p in historical_rejected_paths],
         "original_rows": original_count, "unique_whole_sets": len(questions),
         "collapsed_exact_sets": duplicates,
         "projection": "unchanged_public_text_only_no_gold_no_evidence_no_solver",
         "historical_rejected_text_is_dedup_inventory_not_semantic_admission": True})
    return path, questions, source_paths + historical_rejected_paths


def capacity_preflight(rows):
    """Prepare disposable local manifests, never actual generation chunk dirs."""
    _, _, annotate, capacity, dev = gen.base.transport()
    tokenizer = dev.tokenizer_from(str(gen.base.TOKENIZER))
    checks = []
    for row in rows:
        short, root = row["short_root"], row["root_id"]
        scratch = REPORT / "capacity_scratch" / f"root{short}_c01_generation"
        first = row["focus_plan"]["chunks"][0]
        gen.prepare(scratch, Path(row["history"]["path"]), roots=[root], slots=2,
                    prior_pools=[Path(row["prior"]["path"])], prompt_arm="current",
                    comparison_id=first["comparison_id"],
                    slot_focus_map=first["slot_focus_by_candidate_id"])
        for kind in gen.TYPES:
            payload, receipt = gen.make_payload(scratch, root, kind, gen.initial_context(scratch, root, kind))
            profile = read(scratch / "provider_profile.json")
            profile["output_limit"] = payload["max_tokens"]
            profile["request_parameters"]["max_tokens"] = payload["max_tokens"]
            result = annotate(capacity(profile, tokenizer, payload["messages"], payload["max_tokens"]))
            snapshot = REPORT / "initial_capacity" / short / f"{kind}.json"
            save(snapshot, {"payload": payload, "receipt": receipt, "capacity": result,
                            "scratch_manifest": binding(scratch / "MANIFEST.json")})
            checks.append({"root_id": root, "question_type": kind, **result,
                           "snapshot": binding(snapshot)})
        gen.verify(scratch)
    value = {"checks": checks, "all_pass": all(row["status"] == "PASS" for row in checks),
             "scope": "first_chunk_actual_prompt_with_all_frozen_priors",
             "later_chunks_and_repairs_checked_again_by_transport": True,
             "full_source_and_frozen_feedback_untruncated": True, "remote_http_attempts": 0}
    save(REPORT / "CAPACITY_CHECK.json", value)
    if not value["all_pass"]:
        raise ValueError("initial_capacity_blocked_before_any_remote_call")
    return value


def prepare():
    begin("preparation")
    rows, prepared = [], []
    current = None
    try:
        for short in ROOTS:
            root = "cc-train-episode-" + short
            current = {"root_id": root, "action": "prepare_public_inventory"}
            run = batch_run(short)
            if run.exists() or any(path.exists() for i in range(1, 6) for path in batch.chunk_paths(run, i)):
                raise ValueError("planned_batch_or_chunk_already_exists:" + str(run))
            history = BASE / f"h4_sequential_ab_v51/runs/root{short}_a3b3_01/CANDIDATE_HISTORY.json"
            _, sources, excluded = gen.checked_sources(history, [root])
            if len(sources) != 1 or excluded:
                raise ValueError("requires_one_complete_frozen_root")
            prior, questions, source_paths = build_prior(short)
            focus = gen.slot_focus.build_plan(root, run.name, sources[0]["source"]["turns"], slots=SLOTS_PER_ROOT)
            rows.append({"short_root": short, "root_id": root, "history": binding(history),
                         "prior": binding(prior), "prior_source_files": [binding(p) for p in source_paths],
                         "public_prior_rows": len(questions), "batch_run": str(run), "focus_plan": focus})
        current = {"action": "local_initial_capacity_preflight"}
        capacity_result = capacity_preflight(rows)
        for row in rows:
            current = {"root_id": row["root_id"], "action": "batch_prepare"}
            run = Path(row["batch_run"])
            batch.prepare(run, Path(row["history"]["path"]), prior_pools=[Path(row["prior"]["path"])],
                          root=row["root_id"], focus_plan=row["focus_plan"], candidate_slots=SLOTS_PER_ROOT)
            row["batch_manifest"] = binding(run / "MANIFEST.json")
            prepared.append(row["root_id"])
        plan = {"version": VERSION, "created_at_utc": now(), "roots": rows,
                "candidate_slots": 40, "slots_per_root": 20, "slots_per_type_per_root": 10,
                "remote_http_max": 60, "solver_full_calls_max": 600,
                "all_generation_before_any_collection_solver": True,
                "current_solver_feedback_to_writer": False,
                "full_v51_source_and_frozen_previous_round_feedback": True,
                "prior_scope": "all_v51_v52b_v53_v54_mechanical_public_sets_plus_two_135466_historical_five_correct_drafts_and_default_history",
                "maximum_mechanical_repairs_per_type_per_chunk": 1, "replacement_slots": 0,
                "minimum_remote_completion_gap_seconds": 35, "training_calls": 0,
                "training_allowed": False, "cardinality_rule": "2_to_6_correct_per_question",
                "fixed_cardinality_quota": False, "all_six_individual_item_allowed": True,
                "all_six_batch_is_distribution_diagnostic": True,
                "capacity_check": binding(REPORT / "CAPACITY_CHECK.json"),
                "generation_code": binding(CODE / "generate.py"),
                "orchestrator": binding(Path(__file__)), "automatic_replay": False}
        save(REPORT / "EXPERIMENT_PLAN.json", plan)
        save(marker("preparation", "COMPLETE"), {"version": VERSION, "completed_at_utc": now(),
             "experiment_plan": binding(REPORT / "EXPERIMENT_PLAN.json"), "prepared_roots": prepared,
             "remote_http_attempts": 0, "solver_calls": 0})
        print(json.dumps({"event": "prepared", "candidate_slots": 40,
              "public_prior_rows": [r["public_prior_rows"] for r in rows],
              "max_initial_context": max(c["required_context_tokens"] for c in capacity_result["checks"])}), flush=True)
    except BaseException as error:
        record_error("preparation", error, prepared, current)
        raise


def verify_plan():
    complete = read(marker("preparation", "COMPLETE"))
    verify_binding(complete["experiment_plan"])
    plan = read(REPORT / "EXPERIMENT_PLAN.json")
    if plan["version"] != VERSION or marker("preparation", "ERROR").exists():
        raise ValueError("preparation_incomplete_or_changed")
    for key in ("capacity_check", "generation_code", "orchestrator"):
        verify_binding(plan[key])
    if not read(plan["capacity_check"]["path"])["all_pass"]:
        raise ValueError("initial_capacity_not_passed")
    for row in plan["roots"]:
        for value in (row["history"], row["prior"], row["batch_manifest"], *row["prior_source_files"]):
            verify_binding(value)
        batch.verify(Path(row["batch_run"]))
    return plan


def command(action, run):
    log = REPORT / "logs" / f"{run.name}_{action}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "-B", "-u", str(CODE / "batch.py"), action, "--run", str(run)]
    print(json.dumps({"event": "batch_action_started", "action": action, "run": str(run),
                      "log": str(log), "started_at_utc": now()}), flush=True)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(cmd, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    return binding(log)


def execute(stage):
    plan = verify_plan()
    if stage == "evaluation":
        completion = read(marker("generation", "COMPLETE"))
        if marker("generation", "ERROR").exists() or [row["root_id"] for row in completion["roots"]] != [row["root_id"] for row in plan["roots"]]:
            raise ValueError("all_root_generation_not_complete")
        for row in completion["roots"]:
            verify_binding(row["completion"])
            for item in row["logs"]:
                verify_binding(item)
        for row in plan["roots"]:
            batch._generation_complete(Path(row["batch_run"]))
    else:
        if marker("evaluation", "STARTED").exists():
            raise ValueError("evaluation_already_started_before_generation")
    begin(stage)
    summaries, current = [], None
    try:
        for row in plan["roots"]:
            verify_plan()
            run = Path(row["batch_run"])
            logs = []
            for action in (("generate",) if stage == "generation" else ("evaluate", "finalize", "verify")):
                current = {"root_id": row["root_id"], "action": action, "batch_run": str(run)}
                logs.append(command(action, run))
            if stage == "generation":
                record = read(run / "GENERATION_COMPLETE.json")
                summary = {"root_id": row["root_id"], "completion": binding(run / "GENERATION_COMPLETE.json"),
                           **{key: record[key] for key in ("candidate_slots", "qwen_http_attempts", "mechanical_pass", "ready_for_solver")}}
            else:
                record = read(run / "training_pool/REPORT.json")
                summary = {"root_id": row["root_id"], "completion": binding(run / "FINALIZATION_COMPLETE.json"),
                           "pool_report": binding(run / "training_pool/REPORT.json"),
                           **{key: record[key] for key in ("full_solver_calls", "new_unique_added", "by_type")}}
            summary["logs"] = logs
            summaries.append(summary)
            print(json.dumps({"event": stage + "_root_complete", **summary}, ensure_ascii=False), flush=True)
        completion = {"version": VERSION, "stage": stage, "completed_at_utc": now(),
                      "experiment_plan": binding(REPORT / "EXPERIMENT_PLAN.json"), "roots": summaries}
        if stage == "evaluation":
            completion["generation_completion"] = binding(marker("generation", "COMPLETE"))
        save(marker(stage, "COMPLETE"), completion)
    except BaseException as error:
        record_error(stage, error, summaries, current)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generation", "evaluation", "verify"))
    action = parser.parse_args().action

    def interrupted(signum, frame):
        raise KeyboardInterrupt("interrupted_by_signal_" + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    if action == "prepare":
        prepare()
    elif action == "verify":
        plan = verify_plan()
        print(json.dumps({"event": "verified", "roots": [row["root_id"] for row in plan["roots"]]}))
    else:
        execute(action)


if __name__ == "__main__":
    main()
