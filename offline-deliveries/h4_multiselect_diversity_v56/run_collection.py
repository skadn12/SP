"""Collect 12 multiselect candidates on frozen root 138343, then evaluate.

All scene preferences and historical public statements are frozen before calls.
Later chunks also receive unchanged public statements from completed chunks.
No current Solver data reaches the writer. Stages execute once without replay.
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
CODE = BASE / "h4_sequential_ab_v56_qa"
PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")
ROOT = "cc-train-episode-138343"
HISTORY = BASE / "h4_sequential_ab_v51/runs/root138343_a3b3_01/CANDIDATE_HISTORY.json"
VERSION = "h4_multiselect_diversity_v56_12_slots_v1"
CHUNK_COUNT = 6
SLOTS = 2
MAX_HTTP = 12
MAX_SOLVER = 216

# Frozen neutral topics, never option text, evidence labels, Gold, or Solver data.
SCENE_TOPICS = (
    ("CANDIDATE:A:S01", (
        "The hotel, train, and theater arrangements and timing explicitly reported in this scene.",
        "Emily's chosen meeting arrangement and gallery interest, compared with alternatives offered in this scene.")),
    ("CANDIDATE:B:S01", (
        "The requested cuisine, city area, price preference, and named restaurant alternatives within this restaurant exchange.",
        "The booking request and the explicitly reported completed booking details, scoped to this restaurant exchange.")),
    ("CANDIDATE:A:S02", (
        "The gallery observations, sketches, and details that capture Emily's interest in this exchange.",
        "James's bistro preparations, offered navigation and support, and their intended next meeting as stated in this exchange.")),
    ("CANDIDATE:B:S02", (
        "The HR policy, Emily's reaction, and her manager's response to her request for role flexibility.",
        "The petition's current preparation and circulation plan, James's conditional hopes, and the help he offers.")),
    ("CANDIDATE:A:S03", (
        "The announced comedy troupe, performance format, and named theater in this show-details exchange.",
        "The dinner reservation, Emily's preferred meeting arrangement, and ticket or email commitments in this exchange.")),
    ("CANDIDATE:B:S03", (
        "The confirmed train and hotel details, included breakfast, and Emily's gallery and snack plans in this exchange.",
        "The gallery walking distance, weekend menu, dinner, show, and ticket details explicitly reported in this exchange.")),
)


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
    paths = sorted((BASE / "h4_sequential_ab_v51_qa/runs").glob(
        "root138343_focus20_01_c*_generation/MECHANICAL_POOL.json"))
    if len(paths) != 5:
        raise ValueError("v51_historical_prior_file_count_changed")
    # Check earlier accepted mechanical pools by actual root, not directory name.
    for version in range(43, 51):
        for path in sorted((BASE / f"h4_sequential_ab_v{version}_qa/runs").glob("*/MECHANICAL_POOL.json")):
            obj = read(path)
            rows = obj if isinstance(obj, list) else obj.get("questions", obj.get("candidates", []))
            if any(q.get("root_id") == ROOT for q in rows):
                paths.append(path)
    return paths


def focus_chunks(gen, turns):
    chunks = []
    for index, (scene, topics) in enumerate(SCENE_TOPICS, 1):
        comparison_id = f"root138343_diversity12_01_c{index:02d}"
        ids = gen.slot_focus.candidate_ids(ROOT, comparison_id, "multi_select", SLOTS)
        mapping = {qid: {"preferred_scene_refs": [scene], "focus": topic,
                        "requirement": gen.slot_focus.REQUIREMENT}
                   for qid, topic in zip(ids, topics)}
        gen.slot_focus.validate_map(mapping, ids, turns, allow_empty=False)
        chunks.append({"index": index, "comparison_id": comparison_id,
                       "generation_run": str(CODE / "runs" / (comparison_id + "_generation")),
                       "evaluation_run": str(CODE / "runs" / (comparison_id + "_evaluation")),
                       "candidate_ids": ids, "slot_focus_by_candidate_id": mapping})
    if len(chunks) != CHUNK_COUNT or len({v["focus"] for c in chunks for v in c["slot_focus_by_candidate_id"].values()}) != CHUNK_COUNT * SLOTS:
        raise ValueError("frozen_focus_count_invalid")
    return chunks


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
        prior = REPORT / "priors" / "root138343_historical_public.json"
        questions = project_prior(paths, prior)
        focus_path = REPORT / "FROZEN_FOCUS_PLAN.json"
        save(focus_path, {"root_id": ROOT, "slots": 12, "admission_gate": False,
                          "fixed_correct_count_per_slot": False, "chunks": chunks})
        scratch = REPORT / "capacity_scratch" / "root138343_c01_generation"
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
                "full_v51_source_and_frozen_previous_round_feedback": True,
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
    for value in plan["prior_source_files"] + plan["code_files"]:
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
            prior = REPORT / "priors" / f"root138343_c{chunk['index']:02d}_cumulative_public.json"
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
    begin("evaluation")
    summaries, current = [], None
    _, pipeline = modules()
    total_calls = 0
    try:
        for chunk in plan["chunks"]:
            verify_plan()
            completion("generation")
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
             "chunks": summaries, "full_solver_calls": total_calls, "training_calls": 0})
    except BaseException as error:
        record_error("evaluation", error, summaries, current)
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
    elif action == "generation":
        generate()
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
