"""Collect 12 MC candidates for one configured fresh A/B root with frozen v56 rules.

No automatic stage replay. Content review must be complete and frozen before any
Solver starts. This module changes orchestration only, never core QA rules.
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
CODE = BASE / "h4_sequential_ab_v57_qa"
PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")
CONFIG_PATH = REPORT / "COLLECTION_CONFIG.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
ROOT = CONFIG["root_id"]
SHORT_ROOT = ROOT.rsplit("-", 1)[-1]
HISTORY = Path(CONFIG["history"])
VERSION = "h4_new_root_collection_v57_12_slots_v1"
CHUNK_COUNT = 6
SLOTS = 2
MAX_HTTP = 12
MAX_SOLVER = 216


def modules():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import generate
    import evaluate_and_pool
    return generate, evaluate_and_pool


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
             "completed_chunks": completed, "current_operation": current,
             "completed_artifacts_preserved": True, "automatic_replay": False})


def public_question(question):
    kind = question["question_type"]
    if kind not in ("multi_select", "ordering"):
        raise ValueError("historical_question_type_invalid")
    field = "options" if kind == "multi_select" else "items"
    values = question[field]
    if not isinstance(values, dict) or len(values) != 6 or not all(isinstance(v, str) for v in values.values()):
        raise ValueError("historical_public_question_requires_six_texts")
    if question["root_id"] != ROOT:
        raise ValueError("historical_question_root_mismatch")
    return {key: deepcopy(question[key]) for key in
            ("candidate_id", "root_id", "question_type", "question", field)}


def whole_set_key(question):
    field = "options" if question["question_type"] == "multi_select" else "items"
    return question["question_type"], tuple(sorted(" ".join(text.split()).casefold()
                                                   for text in question[field].values()))


def project_prior(paths, target):
    questions, duplicates, seen = [], [], {}
    original_count = 0
    for path in paths:
        obj = read(path)
        rows = obj if isinstance(obj, list) else obj.get("questions", obj.get("candidates", []))
        for question in rows:
            if question.get("root_id") != ROOT:
                continue
            original_count += 1
            public = public_question(question)
            key = whole_set_key(public)
            if key in seen:
                duplicates.append({"candidate_id": public["candidate_id"],
                                   "retained_candidate_id": seen[key]})
            else:
                seen[key] = public["candidate_id"]
                questions.append(public)
    save(target, {"questions": questions, "source_files": [binding(p) for p in paths],
         "original_rows": original_count, "unique_whole_sets": len(questions),
         "collapsed_exact_sets": duplicates,
         "projection": "unchanged_public_text_only_no_gold_no_evidence_no_solver"})
    return questions


def historical_paths():
    # Explicitly freeze all prior accepted mechanical sets for this root.
    # A new root may have zero modern QA pools; default previous-round pools
    # and feedback are still loaded by the unchanged generate.prepare method.
    paths = []
    for directory in sorted(BASE.glob("h4_sequential_ab_v*_qa")):
        if directory == CODE:
            continue
        for path in sorted((directory / "runs").glob("*/MECHANICAL_POOL.json")):
            obj = read(path)
            rows = obj if isinstance(obj, list) else obj.get("questions", obj.get("candidates", []))
            if any(q.get("root_id") == ROOT for q in rows):
                paths.append(path)
    paths.extend(Path(p).resolve() for p in CONFIG.get("extra_prior_pools", []))
    return sorted(set(paths))


def focus_chunks(gen, turns):
    focus_path = Path(CONFIG["focus_topics"])
    focus = read(focus_path)
    if focus.get("root_id") != ROOT or len(focus.get("chunks", [])) != CHUNK_COUNT:
        raise ValueError("focus_topics_requires_configured_root_and_six_chunks")
    chunks, hints = [], []
    for index, request in enumerate(focus["chunks"], 1):
        if request.get("index") != index or len(request.get("slots", [])) != SLOTS:
            raise ValueError("focus_chunk_requires_index_and_two_slots")
        comparison_id = CONFIG["batch_id"] + f"_c{index:02d}"
        ids = gen.slot_focus.candidate_ids(ROOT, comparison_id, "multi_select", SLOTS)
        mapping = {}
        for qid, slot in zip(ids, request["slots"]):
            if set(slot) != {"preferred_scene_refs", "focus"}:
                raise ValueError("focus_slot_only_scene_refs_and_neutral_topic")
            mapping[qid] = {**deepcopy(slot), "requirement": gen.slot_focus.REQUIREMENT}
            hints.append(slot["focus"])
        gen.slot_focus.validate_map(mapping, ids, turns, allow_empty=False)
        chunks.append({"index": index, "comparison_id": comparison_id,
            "generation_run": str(CODE / "runs" / (comparison_id + "_generation")),
            "evaluation_run": str(CODE / "runs" / (comparison_id + "_evaluation")),
            "candidate_ids": ids, "slot_focus_by_candidate_id": mapping})
    if len(set(hints)) != CHUNK_COUNT * SLOTS:
        raise ValueError("twelve_distinct_soft_focus_topics_required")
    return chunks


def cumulative_prior_path(index):
    return REPORT / "priors" / f"root{SHORT_ROOT}_c{index:02d}_cumulative_public.json"


def verify_copy_snapshot():
    snapshot = read(CODE / "COPY_BINDINGS.json")
    for row in snapshot["files"]:
        for key in ("source", "destination"):
            if binding(row[key])["sha256"] != row["sha256"]:
                raise ValueError("frozen_v56_core_snapshot_changed:" + row[key])
    return binding(CODE / "COPY_BINDINGS.json")


def reviewed_content():
    generation = completion("generation")
    ready, pools, pool_by_id = [], [], {}
    for chunk in verify_plan()["chunks"]:
        pool = Path(chunk["generation_run"]) / "MECHANICAL_POOL.json"
        bound = binding(pool)
        pools.append(bound)
        rows = read(pool)["questions"]
        for question in rows:
            if question["candidate_id"] in pool_by_id:
                raise ValueError("duplicate_mechanical_candidate_id")
            pool_by_id[question["candidate_id"]] = bound["sha256"]
        ready.extend(rows)
    path = REPORT / "INDEPENDENT_CONTENT_AUDIT.json"
    audit = read(path)
    if audit.get("solver_scores_used") is not False:
        raise ValueError("content_review_must_be_solver_blind")
    rows = audit.get("questions", [])
    indexed = {q["candidate_id"]: q for q in rows}
    if len(indexed) != len(rows) or set(indexed) != {q["candidate_id"] for q in ready}:
        raise ValueError("content_review_must_cover_complete_mechanical_pool")
    for question in ready:
        row = indexed[question["candidate_id"]]
        detail = row.get("audit", row)
        status = row.get("core_status", detail.get("core_status"))
        if status not in ("CORE_USABLE", "KNOWN_SUBSTANTIVE_ISSUE"):
            raise ValueError("content_review_status_missing")
        if row.get("root_id") != question["root_id"] or row.get("question_type") != question["question_type"]:
            raise ValueError("content_review_identity_mismatch")
        if detail.get("source_sha256") != question["source_sha256"]:
            raise ValueError("content_review_source_binding_missing_or_changed")
        if detail.get("mechanical_pool_sha256") != pool_by_id[question["candidate_id"]]:
            raise ValueError("content_review_mechanical_pool_binding_missing_or_changed")
    return {"audit": binding(path), "generation_completion": binding(marker("generation", "COMPLETE")),
            "mechanical_pools": pools, "reviewed_questions": len(rows),
            "checked_without_solver_scores": True, "generation_or_gold_edited": False,
            "reviewed_status_counts": {key: sum(indexed[q["candidate_id"]].get("core_status", indexed[q["candidate_id"]].get("audit", {}).get("core_status")) == key for q in ready)
                                      for key in ("CORE_USABLE", "KNOWN_SUBSTANTIVE_ISSUE")}}


def freeze_review():
    verify_plan()
    if marker("evaluation", "STARTED").exists():
        raise ValueError("cannot_freeze_content_review_after_evaluation_start")
    for chunk in verify_plan()["chunks"]:
        if Path(chunk["evaluation_run"]).exists():
            raise ValueError("evaluation_run_already_exists_before_review_freeze")
    value = reviewed_content()
    save(REPORT / "CONTENT_REVIEW.COMPLETE.json", {"version": VERSION,
        "completed_at_utc": now(), "all_generation_complete": True,
        "before_any_solver": True, **value})
    print(json.dumps({"event": "content_review_frozen", "reviewed_questions": value["reviewed_questions"]}))


def verify_review():
    frozen = read(REPORT / "CONTENT_REVIEW.COMPLETE.json")
    current = reviewed_content()
    if frozen.get("version") != VERSION or frozen.get("before_any_solver") is not True:
        raise ValueError("content_review_freeze_policy_changed")
    if any(frozen.get(key) != value for key, value in current.items()):
        raise ValueError("frozen_content_review_changed")
    return frozen


def capacity_check(gen, run, target):
    _, _, annotate, capacity, dev = gen.base.transport()
    tokenizer = dev.tokenizer_from(str(gen.base.TOKENIZER))
    payload, receipt = gen.make_payload(run, ROOT, "multi_select", gen.initial_context(run, ROOT, "multi_select"))
    if payload["max_tokens"] != 4096:
        raise ValueError("v56_actual_output_budget_changed")
    profile = read(run / "provider_profile.json")
    profile["output_limit"] = payload["max_tokens"]
    profile["request_parameters"]["max_tokens"] = payload["max_tokens"]
    result = annotate(capacity(profile, tokenizer, payload["messages"], payload["max_tokens"]))
    save(target, {"payload": payload, "receipt": receipt, "capacity": result,
                  "generation_manifest": binding(run / "MANIFEST.json"),
                  "remote_http_attempts": 0, "source_and_feedback_truncated": False})
    gen.verify(run)
    if result["status"] != "PASS":
        raise ValueError("capacity_blocked_before_remote_call:" + str(target))
    return result


def prepare():
    verify_copy_snapshot()
    # Read-only prerequisites before recording preparation attempt.
    modules()[0].checked_sources(HISTORY, [ROOT])
    focus_chunks(modules()[0], read(HISTORY)["turns"])
    begin("preparation")
    current = None
    try:
        gen, _ = modules()
        if getattr(gen, "GENERATION_TYPES", None) != ("multi_select",):
            raise ValueError("v56_must_generate_multiselect_only")
        current = {"action": "verify_frozen_source"}
        _, sources, excluded = gen.checked_sources(HISTORY, [ROOT])
        if len(sources) != 1 or excluded:
            raise ValueError("requires_one_complete_frozen_root")
        chunks = focus_chunks(gen, sources[0]["source"]["turns"])
        if any(Path(c[key]).exists() for c in chunks for key in ("generation_run", "evaluation_run")):
            raise ValueError("planned_chunk_already_exists")
        paths = historical_paths()
        prior = REPORT / "priors" / f"root{SHORT_ROOT}_historical_public.json"
        questions = project_prior(paths, prior)
        focus_path = REPORT / "FROZEN_FOCUS_PLAN.json"
        save(focus_path, {"root_id": ROOT, "slots": 12, "admission_gate": False,
                          "fixed_correct_count_per_slot": False, "chunks": chunks})
        scratch = REPORT / "capacity_scratch" / f"root{SHORT_ROOT}_c01_generation"
        current = {"action": "initial_capacity_preflight"}
        gen.prepare(scratch, HISTORY, roots=[ROOT], slots=SLOTS, prior_pools=[prior],
                    prompt_arm="current", comparison_id=chunks[0]["comparison_id"],
                    slot_focus_map=chunks[0]["slot_focus_by_candidate_id"])
        result = capacity_check(gen, scratch, REPORT / "INITIAL_CAPACITY.json")
        plan = {"version": VERSION, "created_at_utc": now(), "root_id": ROOT,
                "history": binding(HISTORY), "source_turn_count": len(sources[0]["source"]["turns"]),
                "historical_prior": binding(prior), "prior_source_files": [binding(p) for p in paths],
                "public_prior_rows": len(questions), "frozen_focus_plan": binding(focus_path),
                "chunks": chunks, "candidate_slots": 12, "slots_per_chunk": 2,
                "question_types": ["multi_select"], "remote_http_max": MAX_HTTP,
                "solver_full_calls_max": MAX_SOLVER, "all_generation_before_any_solver": True,
                "current_solver_feedback_to_writer": False,
                "full_source_and_frozen_previous_round_feedback": True,
                "content_review_required_before_any_solver": True,
                "configuration": binding(CONFIG_PATH),
                "source_focus_topics": binding(CONFIG["focus_topics"]),
                "copy_snapshot": verify_copy_snapshot(),
                "report_scripts": [binding(p) for p in sorted(REPORT.glob("*.py"))],
                "previous_result_bindings": binding(CONFIG["previous_result_bindings"]),
                "baseline_usable_pool": binding(CONFIG["baseline_usable_pool"]),
                "within_collection_prior": "unchanged_public_statements_from_completed_mechanical_pools",
                "maximum_mechanical_repairs_per_chunk": 1, "replacement_slots": 0,
                "minimum_remote_completion_gap_seconds": 35, "training_calls": 0,
                "training_allowed": False, "cardinality_rule": "2_to_6_correct_per_question",
                "collection_guidance": "supplement_evidence_grounded_2_3_4_correct_items_without_fixed_quota",
                "fixed_cardinality_quota": False, "all_six_individual_item_allowed": True,
                "actual_max_output_tokens": 4096,
                "capacity_check": binding(REPORT / "INITIAL_CAPACITY.json"),
                "generation_code": binding(CODE / "generate.py"),
                "code_files": [binding(p) for p in sorted(CODE.glob("*.py"))],
                "orchestrator": binding(Path(__file__)), "automatic_replay": False}
        save(REPORT / "EXPERIMENT_PLAN.json", plan)
        save(marker("preparation", "COMPLETE"), {"version": VERSION, "completed_at_utc": now(),
             "experiment_plan": binding(REPORT / "EXPERIMENT_PLAN.json"),
             "remote_http_attempts": 0, "solver_calls": 0})
        print(json.dumps({"event": "prepared", "candidate_slots": 12,
              "public_prior_rows": len(questions), "required_context_tokens": result["required_context_tokens"]}), flush=True)
    except BaseException as error:
        record_error("preparation", error, [], current)
        raise


def verify_plan():
    complete = read(marker("preparation", "COMPLETE"))
    verify_binding(complete["experiment_plan"])
    plan = read(REPORT / "EXPERIMENT_PLAN.json")
    if plan["version"] != VERSION or marker("preparation", "ERROR").exists():
        raise ValueError("preparation_incomplete_or_changed")
    for key in ("history", "historical_prior", "frozen_focus_plan", "capacity_check", "generation_code", "orchestrator"):
        verify_binding(plan[key])
    for key in ("configuration", "source_focus_topics", "copy_snapshot", "previous_result_bindings", "baseline_usable_pool"):
        verify_binding(plan[key])
    verify_copy_snapshot()
    for value in plan["prior_source_files"] + plan["code_files"] + plan["report_scripts"]:
        verify_binding(value)
    if read(plan["capacity_check"]["path"])["capacity"]["status"] != "PASS":
        raise ValueError("initial_capacity_not_passed")
    if plan["chunks"] != read(plan["frozen_focus_plan"]["path"])["chunks"]:
        raise ValueError("frozen_focus_chunks_changed")
    return plan


def command(script, action, run, generation=None):
    log = REPORT / "logs" / f"{run.name}_{action}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "-B", "-u", str(CODE / script), action, "--run", str(run)]
    if generation is not None:
        cmd.extend(["--generation", str(generation)])
    print(json.dumps({"event": "action_started", "script": script, "action": action,
                      "run": str(run), "log": str(log), "started_at_utc": now()}), flush=True)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(cmd, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    return binding(log)


def completion(stage):
    value = read(marker(stage, "COMPLETE"))
    if marker(stage, "ERROR").exists() or value["version"] != VERSION:
        raise ValueError("stage_incomplete_or_changed:" + stage)
    for row in value["chunks"]:
        for item in row["bindings"]:
            verify_binding(item)
    if len(value["chunks"]) != CHUNK_COUNT:
        raise ValueError("stage_chunk_count_changed:" + stage)
    return value


def generate():
    plan = verify_plan()
    if marker("evaluation", "STARTED").exists():
        raise ValueError("evaluation_already_started_before_generation")
    begin("generation")
    summaries, current = [], None
    gen, _ = modules()
    cumulative_paths = [Path(plan["historical_prior"]["path"])]
    total_http = 0
    try:
        for chunk in plan["chunks"]:
            verify_plan()
            run = Path(chunk["generation_run"])
            current = {"chunk": chunk["index"], "action": "prepare_actual_cumulative_chunk"}
            prior = cumulative_prior_path(chunk["index"])
            questions = project_prior(cumulative_paths, prior)
            gen.prepare(run, HISTORY, roots=[ROOT], slots=SLOTS, prior_pools=[prior],
                        prompt_arm="current", comparison_id=chunk["comparison_id"],
                        slot_focus_map=chunk["slot_focus_by_candidate_id"])
            capacity_path = REPORT / "capacity" / f"c{chunk['index']:02d}_initial.json"
            capacity = capacity_check(gen, run, capacity_path)
            current["action"] = "generate"
            log = command("generate.py", "generate", run)
            result = read(run / "REPORT.json")
            total_http += result["http_attempts"]
            if result["candidate_slots"] != SLOTS or result["http_attempts"] > 2 or total_http > MAX_HTTP:
                raise ValueError("generation_budget_exceeded")
            pool = run / "MECHANICAL_POOL.json"
            cumulative_paths.append(pool)
            summary = {"index": chunk["index"], "generation_run": str(run),
                       "public_prior_rows": len(questions),
                       "initial_required_context_tokens": capacity["required_context_tokens"],
                       **{key: result[key] for key in ("candidate_slots", "http_attempts", "mechanical_pass", "ready_for_solver")},
                       "bindings": [binding(run / "MANIFEST.json"), binding(run / "REPORT.json"),
                                    binding(pool), binding(run / "SOLVER_POOL.json"), binding(prior), binding(capacity_path), log]}
            summaries.append(summary)
            print(json.dumps({"event": "generation_chunk_complete", **summary}, ensure_ascii=False), flush=True)
        save(marker("generation", "COMPLETE"), {"version": VERSION, "stage": "generation",
             "completed_at_utc": now(), "experiment_plan": binding(REPORT / "EXPERIMENT_PLAN.json"),
             "chunks": summaries, "http_attempts": total_http, "solver_calls": 0,
             "candidate_slots": 12, "mechanical_pass": sum(r["mechanical_pass"] for r in summaries),
             "ready_for_solver": sum(r["ready_for_solver"] for r in summaries)})
    except BaseException as error:
        record_error("generation", error, summaries, current)
        raise


def evaluate():
    plan = verify_plan()
    completion("generation")
    verify_review()
    begin("evaluation")
    summaries, current = [], None
    _, pipeline = modules()
    total_calls = 0
    try:
        for chunk in plan["chunks"]:
            verify_plan()
            completion("generation")
            verify_review()
            generation_run, run = Path(chunk["generation_run"]), Path(chunk["evaluation_run"])
            current = {"chunk": chunk["index"], "action": "preflight"}
            logs = [command("evaluate_and_pool.py", "preflight", run, generation_run)]
            manifest = read(run / "MANIFEST.json")
            if manifest["full_calls_planned"] > 36 or total_calls + manifest["full_calls_planned"] > MAX_SOLVER:
                raise ValueError("solver_budget_exceeded_before_live")
            for action in ("live", "finalize", "verify"):
                current["action"] = action
                logs.append(command("evaluate_and_pool.py", action, run))
            pipeline.verify(run)
            total_calls += manifest["full_calls_planned"]
            report_path = run / "training_pool/REPORT.json"
            report = read(report_path)
            summary = {"index": chunk["index"], "evaluation_run": str(run),
                       "candidate_count": manifest["candidate_count"],
                       "full_solver_calls": manifest["full_calls_planned"],
                       "pool_report": report,
                       "bindings": [binding(run / "MANIFEST.json"), binding(report_path),
                                    binding(run / "training_pool/MANIFEST.json"), *logs]}
            summaries.append(summary)
            print(json.dumps({"event": "evaluation_chunk_complete", "index": chunk["index"],
                              "full_solver_calls": manifest["full_calls_planned"], "pool_report": report}), flush=True)
        save(marker("evaluation", "COMPLETE"), {"version": VERSION, "stage": "evaluation",
             "completed_at_utc": now(), "experiment_plan": binding(REPORT / "EXPERIMENT_PLAN.json"),
             "generation_completion": binding(marker("generation", "COMPLETE")),
             "content_review_completion": binding(REPORT / "CONTENT_REVIEW.COMPLETE.json"),
             "chunks": summaries, "full_solver_calls": total_calls, "training_calls": 0})
    except BaseException as error:
        record_error("evaluation", error, summaries, current)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generation", "freeze-review", "evaluation", "verify"))
    action = parser.parse_args().action

    def interrupted(signum, frame):
        raise KeyboardInterrupt("interrupted_by_signal_" + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    if action == "prepare":
        prepare()
    elif action == "generation":
        generate()
    elif action == "freeze-review":
        freeze_review()
    elif action == "evaluation":
        evaluate()
    else:
        plan = verify_plan()
        for stage in ("generation", "evaluation"):
            if marker(stage, "COMPLETE").exists():
                completion(stage)
        print(json.dumps({"event": "verified", "root_id": plan["root_id"], "candidate_slots": 12}))


if __name__ == "__main__":
    main()
