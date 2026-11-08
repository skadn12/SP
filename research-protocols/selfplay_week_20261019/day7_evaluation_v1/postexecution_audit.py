"""Independent standard-library audit; run only after root authorizes sealed TEST.

No model imports/calls and no modification of bound inputs or existing results.
The --authorized-after-sealing flag records the operator's explicit handoff.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROLES = ("S0", "A1", "B1")
BOX = re.compile(r"<box>([A-Z][A-Z0-9_]*(?:,[A-Z][A-Z0-9_]*)*)</box>\Z")
HASH_CACHE = {}
VERIFIED = {}


def require(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    path = Path(path).resolve()
    sig = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    if sig not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        HASH_CACHE[sig] = digest.hexdigest()
    return HASH_CACHE[sig]


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(ref):
    got = binding(ref["path"])
    require(all(got[k] == ref[k] for k in ("path", "bytes", "sha256")),
            "BINDING_CHANGED:" + str(ref["path"]))
    VERIFIED[got["path"]] = got


def unique(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "DUPLICATE:" + key)
    return result


def instant(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "NAIVE_TIME")
    return parsed


def near(got, expected, label):
    require(isinstance(got, (int, float)) and abs(got - expected) <= 1e-12, label)


def mean(values):
    return statistics.mean(list(values))


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def compute(raw, req, gold, mapping, choices):
    """Third scoring path: literal finite outputs and exact rational math."""
    text = raw.get("raw_output")
    match = BOX.fullmatch(text) if isinstance(text, str) else None
    labels = match.group(1).split(",") if match else []
    syntax = bool(labels) and len(labels) == len(set(labels)) and set(labels) <= set(mapping)
    pred = [mapping[x] for x in labels] if syntax else None
    valid = (raw["status"] == "RETURNED" and syntax
             and raw["metadata"]["finish_reason"] in ("stop", "eos") and text in choices)
    if req["question_type"] == "ordering":
        valid = valid and len(pred) == len(gold) and set(pred) == set(gold)
    exact, partial = False, Fraction(0)
    if valid and req["question_type"] == "multi_select":
        p, g = set(pred), set(gold)
        partial = Fraction(len(p & g), len(g))
        if len(g) != len(mapping):
            partial -= Fraction(len(p - g), len(mapping) - len(g))
        exact, pred = p == g, sorted(p)
    elif valid:
        positions = {x: i for i, x in enumerate(pred)}
        agreements = sum(positions[a] < positions[b] for i, a in enumerate(gold) for b in gold[i + 1:])
        pairs = len(gold) * (len(gold) - 1) // 2
        partial = Fraction(2 * agreements - pairs, pairs)
        exact = pred == gold
    return {"valid": bool(valid), "exact": bool(exact), "partial_score": float(partial),
            "prediction_content_ids": pred if valid else None, "raw_status": raw["status"]}


def group_inventory(questions, gold, admission):
    groups = {}
    partitions = defaultdict(list)
    for cid, q in questions.items():
        partitions[(q["cohort_id"], gold[cid]["reporting_role"])].append(cid)
    expected = {
        ("historical115_TEST", "HISTORICAL_REGRESSION_PRIMARY"): (64, 1032),
        ("historical115_TEST", "GOLD6_CONTROL_ONLY"): (11, 198),
        ("previous2MC_TEST", "SEPARATE_PREVIOUS_MC_SUPPLEMENT"): (2, 36),
        ("supplement12_TEST", "SEPARATE_MULTITYPE_SUPPLEMENT"): (12, 180),
    }
    require(set(partitions) == set(expected), "REPORT_PARTITIONS")
    for (cohort, report), ids in partitions.items():
        prefix = cohort + "/" + report + "/"
        require(len(ids) == expected[(cohort, report)][0], "PARTITION_Q_COUNT")
        groups[prefix + "ALL"] = ids
        for typ in ("multi_select", "ordering"):
            selected = [cid for cid in ids if questions[cid]["question_type"] == typ]
            if selected:
                groups[prefix + typ] = selected
        for n in range(2, 7):
            selected = [cid for cid in ids if questions[cid]["question_type"] == "multi_select"
                        and len(gold[cid]["gold_content_ids"]) == n]
            if selected:
                groups[prefix + "multi_select_gold_" + str(n) + ("_control" if n == 6 else "")] = selected
        for n in (4, 5, 6):
            selected = [cid for cid in ids if questions[cid]["question_type"] == "ordering"
                        and len(questions[cid]["items"]) == n]
            if selected:
                groups[prefix + "ordering_arity_" + str(n)] = selected
        if cohort == "supplement12_TEST":
            for field in ("planned_design", "relation_class"):
                for value in {admission[cid].get(field) for cid in ids}:
                    selected = [cid for cid in ids if admission[cid].get(field) == value]
                    groups[prefix + field + "_" + str(value)] = selected
    return groups, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-after-sealing", action="store_true", required=True)
    args = parser.parse_args()
    require(args.authorized_after_sealing, "ROOT_HANDOFF_REQUIRED")
    # Check existence without opening any TEST result before all roles are sealed.
    required = [HERE / "PAIRED_AGGREGATE.json", HERE / "REPORT_ZH.md", HERE / "ENDPOINT_TERMINALS.json"]
    for role in ROLES:
        required += [HERE / "models" / role / name for name in
                     ("RAW_RESPONSE_FREEZE.json", "SCORING_STARTED.json", "AGGREGATE.json", "private/SCORED_REQUESTS.json")]
    require(all(p.is_file() for p in required), "ALL_THREE_RAW_AND_SCORES_AND_COMPARE_MUST_EXIST")
    require(not (HERE / "POSTEXECUTION_AUDIT.json").exists(), "AUDIT_ALREADY_EXISTS")
    for path in required + [HERE / "ENDPOINT_LAUNCH.json", HERE / "INDEPENDENT_PREFLIGHT.json"]:
        verify(binding(path))

    preflight = read(HERE / "INDEPENDENT_PREFLIGHT.json")
    require(preflight["status"] == "PASS", "PREFLIGHT_NOT_PASS")
    for ref in preflight["audited_files"] + preflight["source_freeze_bindings"]:
        verify(ref)
    launch = read(HERE / "ENDPOINT_LAUNCH.json")
    terminals = read(HERE / "ENDPOINT_TERMINALS.json")
    require(launch["audit_sha256"] == sha(HERE / "INDEPENDENT_PREFLIGHT.json")
            and launch["launcher_sha256"] == sha(HERE / "execute_endpoints.py"), "LAUNCH_AUDIT_BINDING")
    require(launch["roles"] == list(ROLES) and launch["sequential"] is True
            and launch["automatic_retry"] is False and launch["private_gold_values_read"] is False,
            "LAUNCH_CONTRACT")
    require(all(int(v) <= 1000 for v in launch["gpu_memory_used_mib_before_start"]), "GPU_IDLE_CHECK")
    require(instant(preflight["reviewed_at_utc"]) <= instant(launch["started_at_utc"]), "REVIEW_MUST_PRECEDE_LAUNCH")
    require(terminals["all_roles_raw_sealed"] is True and terminals["automatic_retry"] is False
            and set(terminals["exit_codes"]) == set(ROLES), "ENDPOINT_TERMINALS")

    frozen = read(HERE / "TEST_REQUEST_FREEZE.json")
    for ref in frozen["input_bindings"] + frozen["artifact_bindings"] + [frozen["wrapper"]]:
        verify(ref)
    requests = unique(lines(HERE / "TEST_REQUESTS.jsonl"), "request_id")
    public = unique(lines(HERE / "TEST_PUBLIC_QUESTIONS.jsonl"), "candidate_id")
    maps = read(HERE / "TEST_DISPLAY_MAPS.json")
    cpu = unique(read(HERE / "CPU_PREFLIGHT.json")["prompt_checks"], "prompt_id")
    require(len(requests) == 1446 and len(public) == 89, "FROZEN_DENOMINATORS")
    require(all(r["split"] == "test" and r["branch"] == "direct" for r in requests.values()), "TEST_ONLY")
    role_state = {}
    max_sealed = None
    previous_sealed = instant(launch["started_at_utc"])
    for role in ROLES:
        run = HERE / "models" / role
        start, rawfreeze = read(run / "STARTED.json"), read(run / "RAW_RESPONSE_FREEZE.json")
        verify(binding(run / "STARTED.json"))
        manifest = read(run / "EXECUTION_MANIFEST.json")
        aggregate = read(run / "AGGREGATE.json")
        for ref in ([start["execution_manifest"], start["review"], rawfreeze["execution_manifest"], rawfreeze["request_freeze"],
                     manifest["request_freeze"], manifest["historical_DEV_identity_manifest"], manifest["endpoint_proof"]]
                    + manifest["endpoint_evidence_bindings"] + manifest["model_bindings"]
                    + manifest["tokenizer_bindings"] + manifest["code_bindings"]
                    + [aggregate[k] for k in ("execution_manifest", "raw_response_freeze", "private_scores")]):
            verify(ref)
        require(start["review"] == binding(HERE / "INDEPENDENT_PREFLIGHT.json")
                and start["private_gold_values_parsed"] is False, "START_REVIEW_AND_GOLD_GATE")
        require(instant(manifest["created_at_utc"]) <= instant(preflight["reviewed_at_utc"]), "ENDPOINT_FIXED_BEFORE_REVIEW")
        dev = read(manifest["historical_DEV_identity_manifest"]["path"])
        for key in ("model_bindings", "tokenizer_bindings", "decoder_contract", "runtime_versions", "model_directory"):
            require(manifest[key] == dev[key], "DEV_TEST_IDENTITY:" + role + ":" + key)
        require(manifest["model_role"] == role and manifest["automatic_retry"] is False
                and manifest["request_count"] == manifest["test_requests"] == 1446, "ENDPOINT_SCOPE")
        require(rawfreeze["status"] == "COMMITTED" and rawfreeze["planned_request_count"] == 1446
                and rawfreeze["gold_opened_by_live"] is False, "RAW_SEAL_CONTRACT")
        expected_paths = {str((run / "responses" / (rid + ".json")).resolve()) for rid in requests}
        require(len(rawfreeze["files"]) == 1446 and {r["path"] for r in rawfreeze["files"]} == expected_paths,
                "RAW_EXACT_REQUEST_BOUNDARY")
        require({str(p.resolve()) for p in (run / "responses").glob("*.json")} == expected_paths, "EXTRA_RAW_RECORDS")
        for ref in rawfreeze["files"]:
            verify(ref)
        started, sealed = instant(start["started_at_utc"]), instant(rawfreeze["created_at_utc"])
        require(previous_sealed <= started <= sealed, "ENDPOINT_SEQUENTIAL_ORDER:" + role)
        previous_sealed = sealed
        max_sealed = sealed if max_sealed is None else max(max_sealed, sealed)
        submitted = {}
        completed_batch_ids = set()
        for path in sorted((run / "batches").glob("*_SUBMITTED.json")):
            packet = read(path)
            verify(binding(path))
            batch_id = path.name.split("_")[0]
            offset = int(batch_id.removeprefix("batch")) * 8
            expected_ids = list(requests)[offset:offset + 8]
            require(packet["request_ids"] == expected_ids and packet["automatic_retry"] is False, "BATCH_NATIVE_ORDER_OR_RETRY")
            submitted_at = instant(packet["submitted_at_utc"])
            require(started <= submitted_at <= sealed, "BATCH_TIME")
            for rid in packet["request_ids"]:
                require(rid not in submitted, "REQUEST_SUBMITTED_MULTIPLE_TIMES")
                submitted[rid] = packet["submitted_at_utc"]
            complete_path = path.with_name(batch_id + "_COMPLETE.json")
            if complete_path.exists():
                packet_complete = read(complete_path)
                verify(binding(complete_path))
                require(packet_complete["request_ids"] == packet["request_ids"]
                        and submitted_at <= instant(packet_complete["completed_at_utc"]) <= sealed, "BATCH_COMPLETION")
                completed_batch_ids.add(batch_id)
        require(len(list((run / "batches").glob("*_COMPLETE.json"))) == len(completed_batch_ids), "UNPAIRED_COMPLETE_BATCH")
        require(aggregate["denominator_questions"] == 89 and aggregate["denominator_requests"] == 1446
                and aggregate["checkpoint_selected_by_test"] is False and aggregate["challenger_release"] is False
                and aggregate["cohorts_have_separate_denominators"] is True, "NATIVE_AGGREGATE_CONTRACT")
        role_state[role] = {"run": run, "start": started, "seal": sealed, "manifest": manifest,
                            "rawfreeze": rawfreeze, "aggregate": aggregate, "submitted": submitted}
    require(max_sealed <= instant(terminals["completed_at_utc"]), "TERMINALS_BEFORE_ALL_SEALS")
    # Gold access is allowed only after verifying every role's complete seal.
    for role, state in role_state.items():
        scoring = read(state["run"] / "SCORING_STARTED.json")
        verify(binding(state["run"] / "SCORING_STARTED.json"))
        require(scoring["all_raw_records_committed_before_this_scorer_read_gold"] is True
                and max_sealed <= instant(scoring["started_at_utc"])
                <= instant(state["aggregate"]["scored_at_utc"]), "ALL_RAW_BEFORE_ANY_SCORE")
        verify(scoring["raw_response_freeze"])
        require(scoring["raw_response_freeze"] == binding(state["run"] / "RAW_RESPONSE_FREEZE.json"), "SCORING_RAW_BINDING")

    gold = unique(lines(HERE / "private/TEST_GOLD.jsonl"), "candidate_id")
    choices = read(HERE / "FINITE_CHOICES.json")
    require(set(gold) == set(public), "GOLD_QUESTION_BOUNDARY")
    admission = unique(read(HERE.parent / "day3_eval_supplement_v2/CONTENT_ADMISSION.json")["records"], "candidate_id")
    groups, expected_partitions = group_inventory(public, gold, admission)
    paired = read(HERE / "PAIRED_AGGREGATE.json")
    for ref in paired["bindings"] + [paired["report"]]:
        verify(ref)
    require(paired["independent_recalculation"]["status"] == "PASS"
            and paired["independent_recalculation"]["requests"] == 4338
            and paired["independent_recalculation"]["question_endpoint_pairs"] == 267
            and paired["independent_recalculation"]["model_requests"] == 0, "PAIRED_INDEPENDENT_COUNTS")
    require(paired["primary_question_count"] == 64 and paired["primary_planned_requests_per_model"] == 1032
            and paired["total_planned_requests"] == 4338 and paired["completed_training_stages"] == 1
            and paired["missing_models"] == {"A2": "NOT_PRODUCED", "B2": "NOT_PRODUCED"}, "PAIRED_SCOPE")
    for key in ("no_pooled_89_question_primary_metric", "no_unseen_root_claim", "no_causal_feedback_claim", "historically_exposed"):
        require(paired[key] is True, "CLAIM_BOUNDARY:" + key)
    require(paired["checkpoint_selected_by_test"] is False and paired["challenger_release"] is False, "TEST_USE_POLICY")
    require(all(instant(s["aggregate"]["scored_at_utc"]) <= instant(paired["created_at_utc"]) for s in role_state.values()), "COMPARISON_BEFORE_SCORES")

    models, role_summaries = {}, {}
    for role, state in role_state.items():
        scored = read(state["run"] / "private/SCORED_REQUESTS.json")
        stored_requests = unique(scored["requests"], "request_id")
        stored_questions = unique(scored["questions"], "candidate_id")
        require(set(stored_requests) == set(requests) and set(stored_questions) == set(public), "NATIVE_SCORED_BOUNDARIES")
        results, byquestion, statuses = {}, defaultdict(list), Counter()
        for rid, req in requests.items():
            rawpath = state["run"] / "responses" / (rid + ".json")
            raw = read(rawpath)
            require(all(raw[k] == expected for k, expected in
                        (("request_id", rid), ("candidate_id", req["candidate_id"]), ("cohort_id", req["cohort_id"]), ("model_role", role))), "RAW_IDENTITY")
            committed = instant(raw["response_committed_at_utc"])
            require(state["start"] <= committed <= state["seal"], "RAW_COMMIT_TIME")
            if raw["status"] == "RETURNED":
                require(raw["request_submitted_at_utc"] == state["submitted"][rid]
                        and instant(raw["request_submitted_at_utc"]) <= committed, "RAW_SUBMISSION_TIME")
                require(raw["model_directory"] == state["manifest"]["model_directory"]
                        and raw["prompt_sha256"] == req["prompt_sha256"] and raw["seed"] == req["sampling"]["seed"]
                        and raw["metadata"]["sampling"] == req["sampling"] and raw["gold_opened_by_solver"] is False, "RAW_REQUEST_CONTRACT")
                require(len(raw["prompt_token_ids"]) == req["prompt_tokens"] == raw["metadata"]["prompt_token_count"]
                        and canonical_hash(raw["prompt_token_ids"]) == cpu[req["prompt_id"]]["token_ids_sha256"]
                        and len(raw["completion_token_ids"]) == raw["metadata"]["completion_token_count"] <= 32, "RAW_TOKEN_CONTRACT")
            else:
                require(raw["status"] == ("BACKEND_ERROR" if rid in state["submitted"] else "NOT_SUBMITTED")
                        and raw["raw_output"] is None, "FAILURE_FIXED_DENOMINATOR")
            cid = req["candidate_id"]
            result = compute(raw, req, gold[cid]["gold_content_ids"], maps[req["prompt_id"]]["display_to_content"], choices[req["constraint_key"]])
            native = stored_requests[rid]
            require(all(native[key] == result[key] for key in ("valid", "exact", "prediction_content_ids")), "THIRD_SCORER_DISAGREES")
            near(native["partial_score"], result["partial_score"], "THIRD_PARTIAL_DISAGREES")
            verify(native["response_binding"])
            require(native["response_binding"] == binding(rawpath), "SCORE_RESPONSE_BINDING")
            for key in ("candidate_id", "cohort_id", "root_id", "question_type", "reporting_role", "variant_id", "sample_index"):
                require(native[key] == req[key], "NATIVE_REQUEST_METADATA:" + key)
            results[rid] = {**req, **result}
            byquestion[cid].append(result)
            statuses[raw["status"]] += 1
        question_results = {}
        for cid, rs in byquestion.items():
            q = stored_questions[cid]
            computed = {"exact_rate": mean(r["exact"] for r in rs), "partial_score_mean": mean(r["partial_score"] for r in rs)}
            for key in computed:
                near(q[key], computed[key], "QUESTION_SUMMARY_DISAGREES")
            require(q["planned_requests"] == len(rs) and q["valid_requests"] == sum(r["valid"] for r in rs)
                    and q["any_correct"] == any(r["exact"] for r in rs) and q["all_correct"] == all(r["exact"] for r in rs), "QUESTION_COUNTS_DISAGREE")
            question_results[cid] = {**public[cid], **computed, "planned_requests": len(rs)}
        terminal_name = "COMPLETE.json" if state["aggregate"]["status"] == "COMPLETE" else "FAILED.json"
        terminal_path = state["run"] / terminal_name
        terminal = read(terminal_path)
        verify(binding(terminal_path))
        require(terminal == state["aggregate"]["runtime"]
                and terminal["actual_submitted_requests"] == len(state["submitted"])
                and terminal["returned_requests"] == statuses["RETURNED"], "RUNTIME_TERMINAL_COUNTS")
        if terminal_name == "COMPLETE.json":
            require(not (state["run"] / "FAILED.json").exists() and len(state["submitted"]) == 1446
                    and statuses["RETURNED"] == 1446 and terminals["exit_codes"][role] == 0,
                    "COMPLETE_ROLE_CONTRACT")
        require(set(paired["models"][role]["groups"]) == set(groups), "PAIRED_GROUP_INVENTORY")
        for name, ids in groups.items():
            rows = [r for r in results.values() if r["candidate_id"] in set(ids)]
            g = paired["models"][role]["groups"][name]
            roots = defaultdict(list)
            for cid in ids:
                roots[public[cid]["root_id"]].append(question_results[cid]["exact_rate"])
            require(g["questions"] == len(ids) and g["planned_requests"] == len(rows)
                    and g["valid_requests"] == sum(r["valid"] for r in rows)
                    and g["invalid_requests"] == sum(not r["valid"] for r in rows)
                    and g["exact_requests"] == sum(r["exact"] for r in rows)
                    and g["raw_status_counts"] == dict(Counter(r["raw_status"] for r in rows)), "PAIRED_GROUP_COUNTS")
            near(g["question_equal_exact"], mean(question_results[cid]["exact_rate"] for cid in ids), "PAIRED_QUESTION_EXACT")
            near(g["question_equal_partial"], mean(question_results[cid]["partial_score_mean"] for cid in ids), "PAIRED_QUESTION_PARTIAL")
            near(g["root_equal_exact"], mean(mean(v) for v in roots.values()), "PAIRED_ROOT_EQUAL")
            uncertainty = g["root_descriptive_uncertainty"]
            require(uncertainty["root_count"] == len(roots), "ROOT_UNCERTAINTY_COUNT")
            if len(roots) < 3:
                require(uncertainty["lower_95"] is None and uncertainty["upper_95"] is None, "TOO_FEW_ROOTS_INTERVAL")
            if name.endswith("/ALL"):
                key = tuple(name.split("/")[:2])
                require((len(ids), len(rows)) == expected_partitions[key], "SEPARATE_DENOMINATORS")
            if name in state["aggregate"]["groups"]:
                native_group = state["aggregate"]["groups"][name]
                for key in ("question_equal_exact", "question_equal_partial", "root_equal_exact"):
                    near(native_group[key], g[key], "NATIVE_GROUP_DISAGREEMENT")
        models[role] = {"requests": results, "questions": question_results}
        role_summaries[role] = {"request_count": len(results), "question_count": len(question_results),
                                "raw_status_counts": dict(statuses), "submitted_once_requests": len(state["submitted"]),
                                "exit_code": terminals["exit_codes"][role], "aggregate_status": state["aggregate"]["status"],
                                "raw_sealed_at_utc": state["rawfreeze"]["created_at_utc"]}

    for left, right in (("B1", "A1"), ("A1", "S0"), ("B1", "S0")):
        direction = left + "_minus_" + right
        require(set(paired["comparisons"][direction]) == set(groups), "PAIRED_COMPARISON_GROUPS")
        for name, ids in groups.items():
            delta = [models[left]["questions"][cid]["exact_rate"] - models[right]["questions"][cid]["exact_rate"] for cid in ids]
            comparison = paired["comparisons"][direction][name]
            near(comparison["question_equal_exact_difference"], mean(delta), "PAIRED_DELTA")
            require(comparison["improved_questions"] == sum(d > 0 for d in delta)
                    and comparison["worsened_questions"] == sum(d < 0 for d in delta)
                    and comparison["unchanged_questions"] == sum(d == 0 for d in delta), "PAIRED_Q_TRANSITIONS")
            transitions = Counter()
            for rid, a in models[left]["requests"].items():
                if a["candidate_id"] not in set(ids):
                    continue
                b = models[right]["requests"][rid]
                label = lambda r: "invalid" if not r["valid"] else ("correct" if r["exact"] else "wrong")
                transitions[label(b) + "_to_" + label(a)] += 1
            require(comparison["paired_request_transitions"] == dict(transitions)
                    and comparison["planned_paired_requests"] == sum(transitions.values()), "PAIRED_REQUEST_TRANSITIONS")

    report = (HERE / "REPORT_ZH.md").read_text()
    for name, ids in groups.items():
        if not name.endswith(("/ALL", "/multi_select", "/ordering")):
            continue
        gs = [paired["models"][role]["groups"][name] for role in ROLES]
        expected_row = (f"| {name} | {len(ids)} | {gs[0]['planned_requests']} | "
                        + " | ".join(f"{100 * g['question_equal_exact']:.2f}%" for g in gs) + " |")
        require(expected_row in report, "REPORT_MAIN_TABLE_ROW:" + name)
    for text in ("A2/B2未产出", "不提供合并89题作为主指标", "不能解释为全新 root 泛化", "不能单独证明反馈的因果收益"):
        require(text in report, "REPORT_CLAIM_BOUNDARY")

    outputs = [binding(p) for p in required] + [binding(HERE / "ENDPOINT_LAUNCH.json"),
               binding(HERE / "INDEPENDENT_PREFLIGHT.json"), binding(__file__)]
    result = {
        "schema_version": "day7_independent_postexecution_audit_v1", "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_handoff": "Root explicitly authorized audit after all three raw freezes, native scoring and comparison",
        "model_requests_by_auditor": 0, "native_scorer_imported": False, "compare_module_imported": False,
        "verified_request_endpoint_pairs": 4338, "verified_question_endpoint_pairs": 267,
        "preflight_bindings_still_match": True, "source_and_output_bindings_still_match": True,
        "all_three_models_match_frozen_DEV_identities": True,
        "all_model_weight_shards_fully_rehashed_after_execution": True,
        "sequential_no_retry_batch_audit": True, "all_raw_sealed_before_first_scoring": True,
        "third_scoring_path_matches_native_requests_and_questions": True,
        "paired_groups_report_tables_and_transitions_match": True,
        "role_summaries": role_summaries,
        "report_partitions": {"/".join(k): {"questions": v[0], "requests_per_endpoint": v[1]} for k, v in expected_partitions.items()},
        "unique_verified_bound_file_count": len(VERIFIED),
        "verified_bound_files": list(VERIFIED.values()), "audit_output_bindings": outputs,
        "limits": ["Gold isolation is supported by bound execution gates, recorded declarations and timestamp ordering; no operating-system file-access trace was collected.",
                   "Exposed overlapping roots, one completed training stage; no feedback-causal or unseen-root claim.",
                   "Repeated submissions are checked at application batch-log level; this is not a network-level trace."],
    }
    with (HERE / "POSTEXECUTION_AUDIT.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "verified_request_endpoint_pairs", "verified_question_endpoint_pairs", "unique_verified_bound_file_count", "role_summaries")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
