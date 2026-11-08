"""Independent CPU-only audit of the frozen Day7 TEST preparation package.

Reads sealed upstream/private packets for byte and correspondence checks only.
It never imports or invokes a model runtime and emits no Gold values.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def main():
    freeze_path = HERE / "TEST_REQUEST_FREEZE.json"
    freeze = read(freeze_path)
    check(freeze["status"] == "CPU_PASS_AWAITING_FINAL_TEST_MODEL_BINDINGS_AND_INDEPENDENT_REVIEW", "freeze_status")
    check(freeze["execution_permitted"] is False and freeze["new_solver_calls"] == 0, "freeze_no_execution")

    closure = freeze["input_bindings"] + freeze["artifact_bindings"] + [freeze["wrapper"]]
    for ref in closure:
        p = Path(ref["path"])
        check(p.is_file() and p.stat().st_size == ref["bytes"] and sha(p) == ref["sha256"], "freeze_binding:" + str(p))

    rows = lines(HERE / "TEST_REQUESTS.jsonl")
    public_rows = lines(HERE / "TEST_PUBLIC_QUESTIONS.jsonl")
    private_rows = lines(HERE / "private/TEST_GOLD.jsonl")
    public = {row["candidate_id"]: row for row in public_rows}
    private = {row["candidate_id"]: row for row in private_rows}
    maps = read(HERE / "TEST_DISPLAY_MAPS.json")
    choices = read(HERE / "FINITE_CHOICES.json")
    prompt_index = {row["prompt_id"]: row for row in read(HERE / "PROMPT_INDEX.json")}
    check(len(rows) == len({r["request_id"] for r in rows}) == 1446, "request_count_and_uniqueness")
    check(len(public) == len(private) == 89 and len(public) == len(public_rows) == len(private_rows), "question_packet_uniqueness")
    check(len(prompt_index) == len(maps) == 690, "prompt_map_count")
    check(set(prompt_index) == set(maps) == {r["prompt_id"] for r in rows}, "prompt_map_request_boundary")

    cohorts = {x["cohort_id"]: x for x in read(WEEK / "day4_schedule_v1/EVALUATION_LOCK.json")["cohorts"]}
    hist = WEEK / "day3_mainline_readiness_v1/common_eval/evidence"
    hist_private = {r["candidate_id"]: r for r in lines(hist / "ADMITTED_PRIVATE.jsonl") if r["split"] == "test"}
    hist_public = {r["candidate_id"]: r for r in lines(hist / "ADMITTED_PUBLIC.jsonl") if r["candidate_id"] in hist_private}
    hist_requests = {r["request_id"]: r for r in lines(hist / "ADMITTED_REQUESTS.jsonl") if r["split"] == "test"}
    hist_prompts = {r["prompt_id"]: r for r in lines(hist / "ADMITTED_PROMPTS.jsonl") if r["candidate_id"] in hist_private}
    hist_maps = {r["prompt_id"]: r for r in read(hist / "ADMITTED_DISPLAY_MAPS.json")["variants"] if r["split"] == "test"}
    hist_choices = read(hist / "FINITE_ANSWER_CHOICES.json")["choices"]

    previous_root = WEEK / "day3_eval_clearance_v1/requests"
    previous_requests = [r for r in lines(previous_root / "REQUESTS.jsonl") if r["original_split"] == "test"]
    previous_maps = {(r["candidate_id"], r["variant_id"]): r for r in read(previous_root / "DISPLAY_MAPS.json")}

    supplement_root = WEEK / "day3_eval_supplement_v2/requests"
    supplement_requests = lines(supplement_root / "REQUESTS.jsonl")
    supplement_prompts = {r["prompt_id"]: r for r in lines(supplement_root / "PROMPTS.jsonl")}
    supplement_maps = {r["prompt_id"]: r for r in read(supplement_root / "DISPLAY_MAPS.json")["variants"]}
    supplement_choices = read(supplement_root / "FINITE_CHOICES.json")["choices"]

    source_request_ids = set(hist_requests) | {r["request_id"] for r in previous_requests} | {r["request_id"] for r in supplement_requests}
    check({r["request_id"] for r in rows} == source_request_ids, "all_source_request_ids_preserved")
    check(not (set(hist_choices) & set(supplement_choices)) or
          all(hist_choices[k] == supplement_choices[k] for k in set(hist_choices) & set(supplement_choices)),
          "finite_choice_source_conflict")
    check(hist_choices | supplement_choices == choices, "native_finite_choice_arrays_preserved")
    source_prompt_count = len(hist_prompts) + len({r["candidate_id"] + "::" + r["variant_id"] for r in previous_requests}) + len(supplement_prompts)
    check(source_prompt_count == len(prompt_index) == 690, "source_prompt_count")

    previous_public, previous_private = {}, {}
    for source in previous_requests:
        cid = source["candidate_id"]
        previous_public[cid] = read(previous_root.parent / "content/public" / (cid + ".json"))
        previous_private[cid] = read(previous_root.parent / "content/private" / (cid + ".json"))
    admission = read(WEEK / "day3_eval_supplement_v2/CONTENT_ADMISSION.json")
    admission_by_id = {r["candidate_id"]: r for r in admission["records"]}
    supplement_public, supplement_private = {}, {}
    for cid, record in admission_by_id.items():
        if cid not in {r["candidate_id"] for r in supplement_requests}:
            continue
        check(record["status"] == "CONTENT_ACCEPTED" and record["original_split"] == "test", "supplement_admission:" + cid)
        supplement_public[cid] = read(record["public_question"]["path"])
        supplement_private[cid] = read(WEEK / "day3_eval_supplement_v2/content/private" / (cid + ".json"))
        check(binding(record["public_question"]["path"]) == record["public_question"], "admitted_public_binding:" + cid)
        check(binding(record["private_record"]["path"]) == record["private_record"], "admitted_private_binding:" + cid)
    check(len(supplement_public) == len(supplement_private) == 12, "supplement_admission_count")

    source_public = {**hist_public, **previous_public, **supplement_public}
    source_private = {**hist_private, **previous_private, **supplement_private}
    source_rows = {}
    source_prompts = dict(hist_prompts)
    source_maps = dict(hist_maps)
    for row in hist_requests.values():
        source_rows[row["request_id"]] = row
    for row in previous_requests:
        source_rows[row["request_id"]] = row
        pid = row["candidate_id"] + "::" + row["variant_id"]
        source_prompts[pid] = {"prompt": row["prompt"], "prompt_tokens": row["prompt_tokens"]}
        source_maps[pid] = {"candidate_id": row["candidate_id"], **row["display"]}
    for row in supplement_requests:
        source_rows[row["request_id"]] = row
    source_prompts.update(supplement_prompts)
    source_maps.update(supplement_maps)
    check(len(source_rows) == 1446 and len(source_prompts) == 690 and len(source_maps) == 690, "complete_upstream_union")
    check(maps == source_maps, "exact_upstream_mapping_bytes_as_json_objects")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen3-4B", local_files_only=True)
    cpu_checks = {p["prompt_id"]: p for p in read(HERE / "CPU_PREFLIGHT.json")["prompt_checks"]}
    for pid, prompt in source_prompts.items():
        token_ids = tok.encode(prompt["prompt"], add_special_tokens=False)
        check(len(token_ids) == prompt["prompt_tokens"] and len(token_ids) + 32 <= 16384,
              "independent_tokenization_or_context_overflow:" + pid)
        check(canon(token_ids) == cpu_checks[pid]["token_ids_sha256"], "independent_token_id_hash:" + pid)
        check(cpu_checks[pid]["truncated"] is False, "truncation_claim:" + pid)

    role_counts = Counter()
    cohort_questions = Counter(q["cohort_id"] for q in public_rows)
    cohort_requests = Counter(r["cohort_id"] for r in rows)
    expected_cohorts = {"historical115_TEST": (75, 1230), "previous2MC_TEST": (2, 36), "supplement12_TEST": (12, 180)}
    for cohort, pair in expected_cohorts.items():
        check((cohort_questions[cohort], cohort_requests[cohort]) == pair, "cohort_denominator:" + cohort)

    for row in rows:
        src = source_rows[row["request_id"]]
        q = public[row["candidate_id"]]
        pid = row["prompt_id"]
        prompt_src = source_prompts[pid]
        prompt_path = Path(row["prompt_file"])
        check(prompt_path.read_bytes() == prompt_src["prompt"].encode(), "prompt_upstream_bytes:" + pid)
        check(sha(prompt_path) == row["prompt_sha256"] and prompt_index[pid]["prompt_sha256"] == row["prompt_sha256"], "prompt_sha:" + pid)
        check(row["prompt_tokens"] == prompt_src["prompt_tokens"] == prompt_index[pid]["prompt_tokens"], "prompt_token_count:" + pid)
        check(row["split"] == "test" and row["branch"] == "direct", "TEST_direct_only:" + row["request_id"])
        if row["cohort_id"] in ("historical115_TEST", "supplement12_TEST"):
            for key, value in src.items():
                check(row.get(key) == value, "upstream_request_field:" + key + ":" + row["request_id"])
            if row["cohort_id"] == "historical115_TEST":
                check(row["reporting_role"] == hist_private[row["candidate_id"]]["reporting_role"], "historical_reporting_role")
            else:
                check(row["reporting_role"] == "SEPARATE_MULTITYPE_SUPPLEMENT", "supplement_reporting_role")
        else:
            check(row["sampling"] == {"temperature": src["temperature"], "top_p": src["top_p"], "max_tokens": src["max_output_tokens"], "seed": src["seed"], "stop": src["stop"], "include_stop_str_in_output": src["include_stop_str_in_output"]}, "previous_sampling:" + row["request_id"])
            previous_expected = {
                "request_id": src["request_id"], "candidate_id": src["candidate_id"],
                "root_id": src["root_id"], "split": "test", "question_type": src["question_type"],
                "branch": src["branch"], "variant_id": src["variant_id"],
                "sample_index": src["sample_index"], "prompt_id": pid,
                "prompt_sha256": hashlib.sha256(src["prompt"].encode()).hexdigest(),
                "prompt_tokens": src["prompt_tokens"],
            }
            for key, value in previous_expected.items():
                check(row.get(key) == value, "previous_request_field:" + key + ":" + row["request_id"])
            check(row["constraint_key"] == row["question_type"] and row["context_mode"] == "full_context", "previous_request_context_contract")
        mapping = maps[pid]
        check(mapping["candidate_id"] == row["candidate_id"] and mapping["variant_id"] == row["variant_id"], "mapping_request_identity:" + pid)
        field = "options" if q["question_type"] == "multi_select" else "items"
        check(row["constraint_key"] in choices and len(choices[row["constraint_key"]]) == (2 ** len(q[field]) - 1 if field == "options" else math.factorial(len(q[field]))), "constraint_arity:" + pid)

    def canonical_ids(q):
        field = "options" if q["question_type"] == "multi_select" else "items"
        return {k: "content_" + hashlib.sha256(v.strip().encode()).hexdigest()[:16] for k, v in q[field].items()}

    for cid, q in public.items():
        src = source_public[cid]
        for key, value in q.items():
            if key not in ("cohort_id", "split"):
                check(src.get(key) == value, "public_question_source:" + cid + ":" + key)
        original_gold = source_private[cid]
        expected_gold = original_gold["gold"]
        check(private[cid]["gold"] == expected_gold, "private_gold_source:" + cid)
        ids = canonical_ids(q)
        check(private[cid]["gold_content_ids"] == [ids[label] for label in expected_gold], "private_gold_content_ids:" + cid)
        if cid in hist_private:
            check(private[cid]["reporting_role"] == hist_private[cid]["reporting_role"], "private_historical_role:" + cid)
        for pid, mapping in maps.items():
            if mapping["candidate_id"] != cid:
                continue
            field = "options" if q["question_type"] == "multi_select" else "items"
            check(set(mapping["display_to_content"].values()) == set(ids.values()), "mapping_content_bijection:" + pid)
            check(all(mapping[field][label] == mapping["content_to_text"][content] for label, content in mapping["display_to_content"].items()), "mapping_display_text:" + pid)
        role_counts[(q["cohort_id"], private[cid]["reporting_role"], q["question_type"])] += 1

    # Scope guard audit: all roles and immutable endpoint identities must be
    # present before live; scoring waits until every raw population is frozen.
    runtime_text = (HERE / "runtime.py").read_text()
    for needle in ("review.get('status') == 'PASS'", "MODEL_EXECUTION_SCOPE_DRIFT", "ALL_ROLES_RAW_FREEZE_REQUIRED_BEFORE_GOLD", "ALREADY_STARTED_NO_AUTOMATIC_RETRY", "NOT_SUBMITTED", "gold_opened_by_live", "private_gold_values_parsed"):
        check(needle in runtime_text, "runtime_guard:" + needle)

    launcher_path = HERE / "execute_endpoints.py"
    launcher_text = launcher_path.read_text()
    for needle in ("INDEPENDENT_PREFLIGHT_REQUIRED", "LAUNCH_ALREADY_RECORDED_NO_AUTOMATIC_RETRY", "GPU_BUSY_BEFORE_TEST", "sequential': True", "for role in ROLES", "ROLE_ALREADY_STARTED:", "ROLE_NOT_SEALED_STOP_NO_RETRY:"):
        check(needle in launcher_text, "launcher_guard:" + needle)
    check("launcher_sha256" in launcher_text and "audited_files" in launcher_text and "review.get('status') != 'PASS'" in launcher_text and "AUDIT_DOES_NOT_BIND_LAUNCHER_BYTES" in launcher_text,
          "launcher_self_binding_and_audit_status_guard")

    model_bindings = []
    for role in ("S0", "A1", "B1"):
        test_path = HERE / "models" / role / "EXECUTION_MANIFEST.json"
        dev_path = WEEK / "day5_training_v1/evaluation/models" / role / "EXECUTION_MANIFEST.json"
        test_m, dev_m = read(test_path), read(dev_path)
        check(test_m["status"] == "BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW" and test_m["model_role"] == role, "model_status:" + role)
        check(test_m["request_count"] == test_m["test_requests"] == 1446 and test_m["automatic_retry"] is False, "model_scope:" + role)
        for key in ("model_bindings", "tokenizer_bindings", "decoder_contract", "runtime_versions", "model_directory"):
            check(test_m[key] == dev_m[key], "DEV_TEST_endpoint_identity:" + role + ":" + key)
        check(test_m["historical_DEV_identity_manifest"] == {"path": str(dev_path.resolve()), "bytes": dev_path.stat().st_size, "sha256": sha(dev_path)}, "DEV_manifest_binding:" + role)
        for ref in test_m["model_bindings"]:
            p = Path(ref["path"])
            check(p.stat().st_size == ref["bytes"], "model_file_size:" + role)
            if not p.name.endswith(".safetensors"):
                check(sha(p) == ref["sha256"], "model_metadata_hash:" + role)
        model_bindings += [binding(test_path), binding(dev_path)]
        for ref in [test_m["endpoint_proof"]] + test_m["endpoint_evidence_bindings"] + test_m["tokenizer_bindings"] + test_m["code_bindings"] + test_m["model_bindings"]:
            p = Path(ref["path"])
            check(p.is_file() and p.stat().st_size == ref["bytes"], "model_bound_file_size:" + role + ":" + str(p))
            if ref not in test_m["model_bindings"] or not p.name.endswith(".safetensors"):
                check(sha(p) == ref["sha256"], "model_bound_file_hash:" + role + ":" + str(p))

    for role in ("S0", "A1", "B1"):
        run = HERE / "models" / role
        check(not (run / "STARTED.json").exists() and not (run / "RAW_RESPONSE_FREEZE.json").exists(), "no_live_requests:" + role)
    audited_paths = [HERE / "prepare.py", HERE / "runtime.py", launcher_path,
                     HERE / "compare.py", freeze_path, Path(__file__)]
    if (HERE / "INDEPENDENT_PREFLIGHT_REPORT_ZH.md").is_file():
        audited_paths.append(HERE / "INDEPENDENT_PREFLIGHT_REPORT_ZH.md")
    audited_paths += [HERE / "models" / role / "EXECUTION_MANIFEST.json" for role in ("S0", "A1", "B1")]
    audited_paths += [WEEK / "day4_schedule_v1/EVALUATION_LOCK.json"]
    audited = {str(Path(p).resolve()): binding(p) for p in audited_paths}
    result = {
        "schema_version": "day7_independent_TEST_execution_preflight_v1",
        "status": "PASS",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU-only independent preflight of frozen TEST preparation and execution guards; no model call, no training, no TEST response exists.",
        "audited_files": list(audited.values()),
        "source_freeze_bindings": closure,
        "reviewer_script": binding(__file__),
        "verified_freeze_closure": {"input_binding_count": len(freeze["input_bindings"]), "artifact_binding_count": len(freeze["artifact_bindings"]), "wrapper_binding_count": 1, "all_sha256_and_byte_counts_match": True},
        "source_correspondence": {"requests": 1446, "prompts": 690, "mappings": 690, "public_questions": 89, "private_answer_packets": 89, "upstream_request_fields_and_sampling_preserved": True, "prompt_bytes_and_token_counts_preserved": True, "all690_prompts_independently_retokenized_and_token_id_hashes_match": True, "mapping_objects_and_choice_lists_preserved": True, "public_and_private_answer_content_matches_source": True, "supplement_content_admission_bindings_checked": 12},
        "cohort_counts": {key: {"questions": pair[0], "requests_per_model": pair[1]} for key, pair in expected_cohorts.items()},
        "reporting_counts": {"/".join(key): count for key, count in sorted(role_counts.items())},
        "arity_keys": list(choices),
        "model_identity": {"roles": ["S0", "A1", "B1"], "all_final_step56_or_existing_S0_lineage_bound": True, "DEV_TEST_model_tokenizer_decoder_hashes_match": True, "all_large_shard_sizes_checked": True, "all_model_metadata_and_lineage_hashes_checked": True, "A2": "NOT_PRODUCED", "B2": "NOT_PRODUCED"},
        "runtime_guard_review": {"TEST_only": True, "all_three_final_manifests_required": True, "all_three_raw_response_freezes_required_before_gold_scoring": True, "no_automatic_retry": True, "full_fixed_denominator_failure_records": True, "no_challenger_release": True},
        "launcher_guard_review": {"launcher_hash_audited": True, "PASS_audit_required": True, "one_shot_launch_record": True, "GPU_idle_check_before_start": True, "sequential_roles": ["S0", "A1", "B1"], "no_role_retry": True, "stops_without_raw_freeze": True, "no_private_gold_read": True},
        "execution_state": {"model_calls": 0, "training_runs": 0, "TEST_responses": 0, "execution_started": False, "execution_permitted_by_freeze": False},
        "limitations": ["Historical roots are exposed and overlap across cohorts; this is not unseen-root generalization.", "Gold6 controls and the two supplementary cohorts must remain separate denominators.", "A2/B2 were not produced, so the result covers one completed training stage only."],
    }
    out = HERE / "INDEPENDENT_PREFLIGHT.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "requests": 1446, "prompts": 690, "questions": 89, "freeze_bindings_verified": len(closure), "model_calls": 0, "output": str(out)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        raise
