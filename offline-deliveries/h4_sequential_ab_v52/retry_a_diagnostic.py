"""Bounded live diagnostic: retry 210046 A2/A3 using frozen v51 baseline+A1.

prepare/verify/audit are offline. Only explicit live invokes the existing shared
locked transport. The result is isolated and never merged into v51 or its QA.
"""
from copy import deepcopy
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import candidate_runner as c
from candidate_request_audit import audit_requests

HERE = Path(__file__).resolve().parent
ROOT = "cc-train-episode-210046"
ROUTE = c.r.ROUTES[0]
SCENES = ["S02", "S03"]
SCOPE = "targeted_A_retry_diagnostic"
SEED = HERE.parent / "h4_sequential_ab_v51/runs/root210046_a3b3_01"
read, write, require, digest = c.r.base.read, c.r.base.write, c.r.base.require, c.r.base.digest


def seed_inputs():
    # All frozen v51 bytes, including its final bindings, were snapshotted before
    # this implementation. Do not use its final A+B history as retry input.
    c.r.base.verify_bindings(read(HERE / "BASELINE_FILES.json")["files"])
    c.r.base.verify_bindings(read(SEED / "COLLECTION_BINDINGS.json")["files"])
    source = c.r.frozen_original_r01(ROOT)
    baseline = read(SEED / "FORMAL_BASELINE.json")["turns"]
    require(baseline == source["original"]["turns"] and len(baseline) == 110,
            "retry_original_baseline_changed")
    rows = [row for row in read(SEED / "CANDIDATE_RECORDS.json")
            if row["route"] == ROUTE and row["scene_id"] == "S01"
            and row["disposition"] == "retained_pending_review"]
    require(len(rows) == 1, "retry_seed_A1_not_unique")
    row = rows[0]
    raw = read(Path(row["request"]) / "RESPONSE.json")["text"]
    obj, failures, warnings, duplicate = c.inspect_realization(
        raw, c.identity(ROOT, ROUTE, "S01"), baseline, baseline, None)
    review = read(Path(row["review_request"]) / "RESPONSE.json")["text"]
    decision = c.contract.decide_candidate(mechanical_failures=failures,
        duplicate_result=duplicate, review_raw=review, review_error=row["review_error"],
        expected_identity=c.identity(ROOT, ROUTE, "S01"))
    require(obj == row["realization"] and decision == row["decision"]
            and decision["continue_candidate_history"], "retry_seed_A1_not_raw_retained_output")
    history = c.append_candidate(baseline, ROOT, ROUTE, obj)
    require(len(history) == 123, "retry_seed_turn_count_changed")
    require(history == read(SEED / "candidates" / ROOT / "A/S02/INPUT.json")["history"],
            "retry_seed_not_exact_pre_A2_history")
    seed = {"root_id": ROOT, "seed_run": str(SEED), "baseline_sha256": digest(baseline),
            "retained_A1_record_sha256": digest(row), "initial_history_sha256": digest(history),
            "initial_history_turns": 123, "turns": history,
            "policy": "exact_original_R01_plus_v51_retained_A1_only_no_B",
            "training_pool_eligible": False}
    return baseline, seed, source


def prepare(run):
    run = Path(run).resolve()
    require(run.parent == HERE / "runs" and not run.exists(), "invalid_or_existing_retry_run")
    baseline, seed, source = seed_inputs()
    profile, transport, tokenizer = c.r.transport_inputs()
    run.mkdir(parents=True)
    shutil.copytree(SEED / "feedback", run / "feedback")
    write(run / "FORMAL_BASELINE.json", {"root_id": ROOT, "turns": baseline,
        "history_sha256": digest(baseline), "policy": c.BASELINE_POLICY,
        "original_source_path": str(source["path"]), "original_source_sha256": source["sha256"]})
    write(run / "INITIAL_HISTORY.json", seed)
    write(run / "provider_profile.json", profile)
    policy = {"version": c.VERSION, "scope": SCOPE, "entrypoint": "retry_a_diagnostic.py",
        "root_id": ROOT, "routes": [ROUTE], "scene_ids": SCENES,
        "model": c.r.MODEL, "logical_max": 8, "http_max": 16,
        "max_plan_attempts": 1, "max_realizer_attempts": 2, "max_review_attempts": 1,
        "max_card_attempts": 0, "source_reservations": 0,
        "minimum_http_gap_after_response_seconds": 35,
        "transport_retry_policy": c.TRANSPORT_RETRY_POLICY,
        "current_task_contract": c.CURRENT_TASK_CONTRACT,
        "formal_history_sha256": digest(baseline), "initial_history_sha256": digest(seed["turns"]),
        "seed_run": str(SEED), "training_pool_eligible": False, "training_started": False,
        "fresh_collection_yield_claim": False, "merge_into_v51": False}
    write(run / "PREVIEW.json", policy)
    # Bind the predecessor snapshot itself and every file it references, as
    # well as current source/transport/tokenizer and exact frozen feedback.
    files = list(HERE.glob("*.py")) + [HERE / "POLICY.md", HERE / "BASELINE_FILES.json"]
    files += [Path(row["path"]) for row in read(HERE / "BASELINE_FILES.json")["files"]]
    files += [Path(row["path"]) for row in read(SEED / "MANIFEST.json")["files"]]
    files += list(c.r.LEGACY.glob("*.py")) + transport + tokenizer + [source["path"]]
    files += [p for p in run.rglob("*") if p.is_file()]
    manifest = {**policy, "files": c.r.base.bindings(files), "prepared_at": c.r.now()}
    write(run / "MANIFEST.json", manifest)
    snapshots = []
    for i, row in enumerate(manifest["files"]):
        if Path(row["path"]).suffix == ".py":
            dest = run / "execution_code" / f"{i:04d}_{Path(row['path']).name}"
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(row["path"], dest)
            snapshots.append({"path": str(dest), "sha256": row["sha256"]})
    write(run / "EXECUTION_CODE_SNAPSHOT.json", {"files": snapshots})
    verify(run)
    return {"status": "PREPARED", "run": str(run), "logical_max": 8, "http_max": 16,
            "initial_history_turns": 123, "generation_calls": 0, "B_calls": 0}


def verify(run):
    run = Path(run).resolve()
    require(run.parent == HERE / "runs", "retry_run_outside_current_directory")
    m = read(run / "MANIFEST.json")
    c.r.base.verify_bindings(m["files"])
    c.r.base.verify_bindings(read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"])
    preview = read(run / "PREVIEW.json")
    require({k: v for k, v in m.items() if k not in ("files", "prepared_at")} == preview,
            "retry_manifest_policy_changed")
    require(m["version"] == c.VERSION and m["scope"] == SCOPE
            and m["root_id"] == ROOT and m["routes"] == [ROUTE] and m["scene_ids"] == SCENES
            and m["logical_max"] == 8 and m["http_max"] == 16 and m["model"] == c.r.MODEL
            and m["minimum_http_gap_after_response_seconds"] == 35
            and m["transport_retry_policy"] == c.TRANSPORT_RETRY_POLICY
            and m["current_task_contract"] == c.CURRENT_TASK_CONTRACT
            and m["training_pool_eligible"] is False and m["training_started"] is False
            and m["fresh_collection_yield_claim"] is False and m["merge_into_v51"] is False,
            "retry_scope_or_policy_changed")
    bound = {Path(row["path"]).resolve() for row in m["files"]}
    required = set(HERE.glob("*.py")) | {HERE / "POLICY.md", HERE / "BASELINE_FILES.json"}
    required |= {run / name for name in ("PREVIEW.json", "FORMAL_BASELINE.json",
                                         "INITIAL_HISTORY.json", "provider_profile.json")}
    required |= {p for p in (run / "feedback").rglob("*") if p.is_file()}
    required |= {Path(row["path"]) for row in read(HERE / "BASELINE_FILES.json")["files"]}
    required |= {Path(row["path"]) for row in read(SEED / "MANIFEST.json")["files"]}
    require(required <= bound, "retry_inputs_not_bound")
    baseline, seed, source = seed_inputs()
    require(read(run / "INITIAL_HISTORY.json") == seed, "retry_initial_history_changed")
    require(read(run / "FORMAL_BASELINE.json") == {"root_id": ROOT, "turns": baseline,
        "history_sha256": digest(baseline), "policy": c.BASELINE_POLICY,
        "original_source_path": str(source["path"]), "original_source_sha256": source["sha256"]},
        "retry_baseline_changed")
    require(m["formal_history_sha256"] == digest(baseline)
            and m["initial_history_sha256"] == digest(seed["turns"]), "retry_history_digest_changed")
    original_feedback = {p.relative_to(SEED / "feedback"): p.read_bytes()
                         for p in (SEED / "feedback").rglob("*") if p.is_file()}
    actual_feedback = {p.relative_to(run / "feedback"): p.read_bytes()
                       for p in (run / "feedback").rglob("*") if p.is_file()}
    require(original_feedback == actual_feedback, "retry_frozen_feedback_changed")
    require(not (run / "SOURCE_RESERVATIONS.json").exists()
            and not (run / "extractions").exists(), "retry_B_activity_forbidden")
    profile, *_ = c.r.base.transport()
    require(profile() == read(run / "provider_profile.json"), "provider_profile_changed")
    return m, baseline, seed["turns"]


def summary(records, baseline, initial, history):
    counter = c.token_counter()
    measure = lambda turns: c.metrics.measure_turns(turns, counter)[0]
    buckets = {name: {"scenes": 0, "turns": 0, "body_tokens": 0}
               for name in ("generated", "retained", "quarantined", "rejected")}
    mapping = {"retained_pending_review": "retained", "quarantined": "quarantined", "hard_reject": "rejected"}
    for row in records:
        obj = row["realization"]
        counts = measure(obj.get("turns") if isinstance(obj, dict) else None)
        for name in ("generated", mapping[row["disposition"]]):
            for key, value in {"scenes": int(counts["turns"] > 0), **counts}.items():
                buckets[name][key] += value
    return {"scope": SCOPE, "new_A_responses_only": buckets,
            "original_plus_R01": measure(baseline), "seed_A1": measure(initial[len(baseline):]),
            "diagnostic_cumulative": measure(history), "B_new_body_tokens": 0,
            "training_admitted_body_tokens": 0, "merged_into_v51": False,
            "body_token_definition": "sum frozen local tokenizer encode(turn.text); no prompt/metadata tokens"}


def live(run):
    from candidate_transport import Caller
    run = Path(run).resolve()
    with c.r.LOCK_PATH.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        m, baseline, initial = verify(run)
        require(not (run / "LIVE_STARTED.json").exists(), "already_started_no_automatic_replay")
        write(run / "LIVE_STARTED.json", {"version": c.VERSION, "scope": SCOPE, "started_at": c.r.now()})
        caller, history, records = Caller(run), deepcopy(initial), []
        for scene in SCENES:
            rows, history = c.collect_slot(caller, run, ROOT, ROUTE, scene,
                history, baseline, provenance=SCOPE)
            records.extend(rows)
        write(run / "CANDIDATE_RECORDS.json", records)
        write(run / "CANDIDATE_HISTORY.json", {"root_id": ROOT, "turns": history,
            "formal_baseline_turns": len(baseline), "seed_history_turns": len(initial),
            "scope": SCOPE, "training_pool_eligible": False})
        report = {"version": c.VERSION, "scope": SCOPE, "root_id": ROOT,
            "metrics": summary(records, baseline, initial, history),
            "shortfalls": [scene for scene in SCENES if not any(
                row["scene_id"] == scene and row["decision"]["continue_candidate_history"] for row in records)],
            "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
            "B_calls": 0, "solver_calls": 0, "question_calls": 0,
            "training_pool_eligible": False, "training_started": False}
        write(run / "CANDIDATE_REPORT.json", report)
        verify(run)
        write(run / "COLLECTION_BINDINGS.json", {"files": c.r.base.bindings(
            [p for p in run.rglob("*") if p.is_file()]), "semantic_certified": False})
        return report


def audit(run):
    run = Path(run).resolve()
    m, baseline, initial = verify(run)
    c.r.base.verify_bindings(read(run / "COLLECTION_BINDINGS.json")["files"])
    records, history = read(run / "CANDIDATE_RECORDS.json"), deepcopy(initial)
    order = []
    for row in records:
        require(row["root_id"] == ROOT and row["route"] == ROUTE and row["scene_id"] in SCENES
                and row["provenance"] == SCOPE, "retry_record_scope_changed")
        scene = row["scene_id"]
        order.append(SCENES.index(scene))
        slot = read(run / "candidates" / ROOT / "A" / scene / "INPUT.json")
        require(slot["history"] == history and slot["card"] is None, "retry_cumulative_history_changed")
        raw = read(Path(row["request"]) / "RESPONSE.json")["text"]
        obj, failures, warnings, dup = c.inspect_realization(
            raw, c.identity(ROOT, ROUTE, scene), history, baseline, None)
        review = read(Path(row["review_request"]) / "RESPONSE.json")["text"] if row["review_request"] else None
        decision = c.contract.decide_candidate(mechanical_failures=failures, duplicate_result=dup,
            review_raw=review, review_error=row["review_error"], expected_identity=c.identity(ROOT, ROUTE, scene))
        require(obj == row["realization"] and failures == row["mechanical_failures"]
                and warnings == row["quality_warnings"] and dup == row["duplicate_result"]
                and review == row["raw_review"] and decision == row["decision"]
                and decision["disposition"] == row["disposition"], "retry_decision_replay_changed")
        if decision["continue_candidate_history"]:
            history = c.append_candidate(history, ROOT, ROUTE, obj)
    require(order == sorted(order), "retry_scene_order_changed")
    require(history == read(run / "CANDIDATE_HISTORY.json")["turns"], "retry_final_history_changed")
    replay = audit_requests(run, records, baseline)
    for p in (run / "requests").glob("*/PROOF.json"):
        proof = read(p)
        require(proof.get("route") == ROUTE and proof.get("scene_id") in SCENES
                and proof.get("role") in ("planner", "realizer", "reviewer"), "retry_call_scope_changed")
    report = read(run / "CANDIDATE_REPORT.json")
    require(report["metrics"] == summary(records, baseline, initial, history), "retry_metrics_changed")
    require(report["logical_calls"] == len(list((run / "requests").glob("*/REQUEST.json"))) <= 8
            and report["http_attempts"] == len(list((run / "requests").glob("*/HTTP_ATTEMPT*.json"))) <= 16,
            "retry_budget_receipts_changed")
    return {"status": "ARTIFACT_REPLAY_VERIFIED", "scope": SCOPE,
            "request_replay": replay, "metrics": report["metrics"],
            "semantic_certified": False, "training_pool_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "live", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.command](args.run)
    if args.command == "verify":
        result = {"status": "PREPARED_BINDINGS_VERIFIED", "scope": SCOPE,
                  "logical_max": result[0]["logical_max"], "http_max": result[0]["http_max"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
