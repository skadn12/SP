"""Independent CPU-only public-package preflight; never opens a private sidecar."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PILOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT / "runtime"))
import common as c

SOURCE_HASHES = {
    "cc-train-episode-246750": "5fbb9c88a810cce22ede8cbf3aede643aed1a329316ba03c6f3b0acee81c26ac",
    "cc-train-episode-70315": "b10cecd1fdcfa683fc911fdd8958c4a8f78a962561e14c7466e01ae88b75023c",
    "cc-train-episode-96462": "ed2f6810a8ab42c99741c0894100cb1656fe1d340ea9b7b7f5ad06875c6b1bbf",
    "cc-train-episode-24719": "95bce29d8831fad7707763149fc51062d66a338809718b8b53c6b468c9ab23f6",
    "cc-train-episode-38601": "21f9e349fcd2c735e1f526e2969f6f7bbcffa4a7ec2f6ded5905c7ee9bf8e2a0",
    "cc-train-episode-74644": "5882a1077e2424eb73425c7c95b7b45ddacd055e05d30434726b81fe1362953a",
}
SLOT_KEYS = {"slot_id", "candidate_id", "root_id", "root_index", "arm", "question_type", "item_count"}
FORBIDDEN_KEYS = {"gold", "gold_count", "evidence", "citations", "retrieved_card_ids", "expected_B_card_ids", "B_feedback", "target_labels"}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def scan_policy(value):
    if isinstance(value, dict):
        check(not set(value) & FORBIDDEN_KEYS, "PRIVATE_POLICY_FIELD")
        for child in value.values():
            scan_policy(child)
    elif isinstance(value, list):
        for child in value:
            scan_policy(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PILOT / "runtime/direct")
    parser.add_argument("--output-directory", type=Path, default=PILOT / "audit")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_directory.resolve()
    check(run == (PILOT / "runtime/direct").resolve(), "ONLY_DESIGNATED_PUBLIC_DIRECT_PACKAGE")
    check(not c.below(output, run), "AUDIT_MUST_NOT_MUTATE_FROZEN_RUN")
    check(not (run / "STARTED.json").exists(), "EXECUTION_ALREADY_STARTED")
    # This verifier reads public run bindings and implementation/model files only.
    manifest = c.verify(run, check_models=True)
    frozen = c.read(run / "PREFLIGHT_FREEZE.json")
    payload = c.read(run / "PUBLIC_QUESTIONS.json")
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    token_ids = c.read(run / "PROMPT_TOKEN_IDS.json")
    policy = c.read(run / "PROTOCOL.json")
    admission = c.read(run / "CONTENT_ADMISSION.json")
    check(set(payload) == {"questions"}, "PUBLIC_WRAPPER_KEYS")
    check(set(policy) == {"schema_version", "solver", "budget", "slots"}, "PUBLIC_POLICY_TOP_LEVEL")
    check(policy["schema_version"] == "day2_pilot_public_execution_policy_v1", "PUBLIC_POLICY_SCHEMA")
    scan_policy(policy)
    check(len(policy["slots"]) == 24 and all(set(slot) <= SLOT_KEYS for slot in policy["slots"]), "PUBLIC_SLOT_WHITELIST")
    check(all({"slot_id", "candidate_id", "root_id", "arm", "question_type", "item_count"} <= set(slot) for slot in policy["slots"]), "PUBLIC_SLOT_REQUIRED_KEYS")
    slots = {slot["candidate_id"]: slot for slot in policy["slots"]}
    check(len(slots) == 24, "DUPLICATE_FROZEN_SLOT")
    expected_solver = {"name": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
        "MC_mappings": 6, "MC_samples_each_mapping": 3, "ordering_mappings": 12,
        "ordering_samples_each_mapping": 1, "temperature": .8, "top_p": .9,
        "max_tokens": 32, "enable_thinking": False, "max_model_len": 16384,
        "source_truncation": False, "automatic_retry": False}
    check(all(policy["solver"][key] == value for key, value in expected_solver.items()), "FROZEN_SOLVER_CONTRACT")
    check(policy["budget"]["initial_question_slots"] == 24 and policy["budget"]["max_direct_requests"] == 360, "FROZEN_BUDGET")
    questions = payload["questions"]
    qmap = {q["candidate_id"]: q for q in questions}
    check(len(qmap) == len(questions) == manifest["question_count"] <= 24, "QUESTION_COUNT")
    check(set(qmap) <= set(slots), "UNKNOWN_CANDIDATE")
    counts = collections.Counter(q["question_type"] for q in questions)
    check(counts["multi_select"] <= 12 and counts["ordering"] <= 12, "TYPE_CAP")
    check(manifest["request_count"] == 18 * counts["multi_select"] + 12 * counts["ordering"] <= 360, "REQUEST_FORMULA")
    check(manifest["question_type_counts"] == dict(counts), "TYPE_COUNT_METADATA")
    check(manifest["planned_question_slots"] == manifest["production_denominator"] == 24, "PLANNED_DENOMINATOR")
    check(manifest["production_admitted_numerator"] == len(questions), "ADMITTED_NUMERATOR")
    check(manifest["missing_candidate_ids"] == sorted(set(slots) - set(qmap)), "MISSING_CANDIDATE_SET")
    check(admission["status"] == "PASS" and set(admission["candidate_ids"]) == set(qmap), "CONTENT_ADMISSION")
    check(admission["admitted_count"] == len(questions) and admission["planned_slots"] == 24, "ADMISSION_COUNTS")
    check(admission["public_questions_sha256"] == c.sha(run / "PUBLIC_QUESTIONS.json"), "ADMISSION_PUBLIC_HASH")
    check(manifest["batch_size"] == 8 and manifest["max_model_len"] == 16384 and manifest["prefix_caching"] is False, "BACKEND_LIMITS")
    check(manifest["source_truncation"] is False and manifest["automatic_retry"] is False, "NO_TRUNCATION_OR_RETRY")
    check(manifest["gold_parsed_by_prepare"] is False and manifest["gold_opened_by_prepare"] is False, "PREPARE_GOLD_PARSE")
    check(manifest["random_display"]["gold_blind"] is True, "GOLD_BLIND_MAPPING_DECLARATION")
    seed = policy["solver"]["seed_base"]
    check(seed == manifest["sampling_seed_start"] == manifest["random_display"]["seed"], "FROZEN_SEEDS")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, solver = c.a.native()
    native_decoder = solver.generation_config("multi_select")
    check(manifest["decoder_contract"] == native_decoder, "NATIVE_DECODER_IDENTITY")
    sources, variants, source_summary = {}, {}, []
    for q in questions:
        c.a.public_question(q)
        qid, root = q["candidate_id"], q["root_id"]
        slot = slots[qid]
        check(root in SOURCE_HASHES and slot["root_id"] == root and slot["question_type"] == q["question_type"], "SLOT_IDENTITY")
        check(slot["arm"] in {"A", "B"} and slot["item_count"] == 6, "SLOT_ARM_OR_ITEMS")
        field = "options" if q["question_type"] == "multi_select" else "items"
        check(len(q[field]) == 6, "SIX_ITEMS")
        source_path = run / "sources" / (root + ".json")
        check(c.sha(source_path) == q["source_sha256"] == SOURCE_HASHES[root], "BOUND_SOURCE_HASH")
        if root not in sources:
            source = c.read(source_path)
            check(source["root_id"] == root and len({t["turn_ref"] for t in source["turns"]}) == len(source["turns"]), "SOURCE_ROOT_OR_REFS")
            sources[root] = source
            source_summary.append({"root_id": root, "sha256": c.sha(source_path), "full_history_turn_count": len(source["turns"])})
        check(maps["public_to_canonical"][qid] == c.a.content_ids(q), "CANONICAL_CONTENT_HASHES")
        variants[qid] = {v["variant_id"]: v for v in c.a.variants(q, seed, 6 if field == "options" else 12)}
    check(manifest["root_count"] == len(sources), "ROOT_COUNT")
    rows = manifest["requests"]
    ids = {row["request_id"] for row in rows}
    check(len(ids) == len(rows) == manifest["request_count"], "REQUEST_IDS")
    check(set(maps) == {"requests", "public_to_canonical"} and set(maps["requests"]) == set(token_ids) == ids, "REQUEST_MAPPING_AND_TOKEN_COVERAGE")
    check(set(maps["public_to_canonical"]) == set(qmap), "CANONICAL_COVERAGE")
    seen, prompt_cache, sample_counts = set(), {}, collections.Counter()
    for index, row in enumerate(rows):
        qid, rid = row["candidate_id"], row["request_id"]
        q = qmap[qid]
        field = "options" if q["question_type"] == "multi_select" else "items"
        variant = variants[qid][row["variant_id"]]
        check(row["branch"] == "direct" and row["root_id"] == q["root_id"] and row["question_type"] == q["question_type"], "REQUEST_IDENTITY")
        check(row["display"] == {field: variant[field]} and maps["requests"][rid] == variant, "DETERMINISTIC_REBINDING")
        mapping = variant["display_to_content"]
        check(len(mapping) == len(set(mapping.values())) == 6 and set(mapping.values()) == set(maps["public_to_canonical"][qid].values()), "ALL_MAPPING_BIJECTIONS")
        sample_key = (qid, row["variant_id"], row["slot"])
        check(sample_key not in seen, "DUPLICATE_SAMPLE_SLOT")
        seen.add(sample_key)
        check(row["slot"] in (range(3) if field == "options" else range(1)), "SAMPLE_SLOT")
        sample_counts[qid] += 1
        check(rid == f"{qid}__{row['variant_id']}__direct__s{row['slot']:02d}", "REQUEST_NAME")
        check(row["seed"] == seed + index and row["max_output_tokens"] == 32, "REQUEST_SEED_OR_MAX_TOKENS")
        check(row["source_sha256"] == q["source_sha256"] and row["source_turns"] == len(sources[q["root_id"]]["turns"]), "REQUEST_SOURCE_METADATA")
        cache_key = (qid, row["variant_id"])
        if cache_key not in prompt_cache:
            # Exact reconstruction rules out any extra Gold/evidence/card injection.
            prompt = c.a.direct_prompt(tokenizer, q, variant, sources[q["root_id"]])
            ordered_rows = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in sources[q["root_id"]]["turns"])
            check(ordered_rows in prompt, "FULL_ORDERED_HISTORY")
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_cache[cache_key] = (prompt, encoded)
        prompt, encoded = prompt_cache[cache_key]
        actual = (run / "prompts" / (rid + ".txt")).read_text()
        check(actual == prompt and hashlib.sha256(actual.encode()).hexdigest() == row["prompt_sha256"], "EXACT_SOURCE_ONLY_PROMPT")
        check(token_ids[rid] == encoded and row["prompt_tokens"] == len(encoded), "EXACT_PROMPT_TOKEN_IDS")
        check(len(encoded) + 32 <= 16384, "FULL_CONTEXT_LIMIT")
        params = c.a.sampling_params(row, native_decoder["structured_output_api"])
        check(params.temperature == .8 and params.top_p == .9 and params.max_tokens == 32 and params.seed == row["seed"], "ACTUAL_SAMPLING_PARAMS")
        check(params.stop == ["</box>"] and params.include_stop_str_in_output is True, "BOX_STOP_CONTRACT")
        check(len(c.a.choices(row["question_type"], row["display"][field])) == (63 if field == "options" else 720), "FINITE_SYNTAX_CHOICES")
    check(all(sample_counts[qid] == (18 if q["question_type"] == "multi_select" else 12) for qid, q in qmap.items()), "PER_QUESTION_SAMPLES")
    check(manifest["max_prompt_tokens"] == max((row["prompt_tokens"] for row in rows), default=0), "MAX_PROMPT_METADATA")
    check(manifest["minimum_context_headroom"] == min((16384-row["prompt_tokens"]-32 for row in rows), default=None), "HEADROOM_METADATA")
    expected_files = {run / name for name in ["PUBLIC_QUESTIONS.json", "DISPLAY_MAPPINGS.json", "PROMPT_TOKEN_IDS.json", "PROTOCOL.json", "CONTENT_ADMISSION.json", "manifest.json", "PREFLIGHT_FREEZE.json"]}
    expected_files |= {run / "sources" / (root + ".json") for root in sources}
    expected_files |= {run / "prompts" / (rid + ".txt") for rid in ids}
    actual_files = {path.resolve() for path in run.rglob("*") if path.is_file()}
    check(actual_files == expected_files and all(not path.is_symlink() for path in run.rglob("*")), "EXACT_PUBLIC_FILE_SET")
    frozen_paths = {Path(row["path"]).resolve() for row in frozen["files"]}
    check(len(frozen_paths) == len(frozen["files"]) and frozen_paths == expected_files - {run / "PREFLIGHT_FREEZE.json"}, "FROZEN_FILE_COVERAGE")
    check({Path(row["path"]).resolve() for row in manifest["public_input_bindings"]} == expected_files - {run / "manifest.json", run / "PREFLIGHT_FREEZE.json"}, "MANIFEST_PUBLIC_BINDING_COVERAGE")
    check(frozen["planned_slots"] == 24 and frozen["questions"] == len(questions) and frozen["requests"] == len(rows) and frozen["inference_started"] is False and frozen["public_only"] is True, "FREEZE_METADATA")
    public_bindings = [c.bind(path) for path in sorted(expected_files)]
    detail = {"schema_version": "day2_independent_public_preflight_detail_v1", "status": "PASS", "created_at_utc": c.now(),
        "reviewer_scope": "A author, independent of runtime implementation; all A Gold exposure disclosed; no B private Gold or private sidecar read.",
        "cpu_only": True, "model_backend_created": False, "new_solver_calls": 0,
        "read_scope": "Only files inside designated runtime/direct plus bound implementation, tokenizer/model files and library metadata; original authoring freeze/private bindings are not dereferenced.",
        "code_review_binding": c.bind(output / "RUNTIME_CODE_REVIEW.json"), "checker_binding": c.bind(__file__),
        "runtime_code_bindings": [c.bind(path) for path in sorted((PILOT / "runtime").glob("*.py"))],
        "question_count": len(questions), "request_count": len(rows), "question_type_counts": dict(counts), "planned_slots": 24,
        "missing_candidate_ids": manifest["missing_candidate_ids"], "sources": source_summary,
        "distinct_prompt_count": len(prompt_cache), "all_request_mappings_and_prompts_verified": len(rows),
        "max_prompt_tokens": manifest["max_prompt_tokens"], "minimum_context_headroom": manifest["minimum_context_headroom"],
        "checks": ["all frozen hashes and exact public file sets", "whitelisted public questions and policy; no Gold/evidence/cards", "all full-history source rows in original order", "exact native prompt reconstruction", "all encoded tokens exact and context bounded", "all seeded canonical/display bijections", "per-question mappings/samples and <=360 total", "fixed model/tokenizer/native decoder and actual CPU SamplingParams", "no STARTED or private file in direct"],
        "total_api_token_cost": None}
    # The executable PASS intentionally contains bindings only to inside-direct files.
    passed = {"schema_version": "day2_independent_public_preflight_pass_v1", "status": "PASS", "created_at_utc": c.now(),
        "public_only": True, "cpu_only": True, "new_solver_calls": 0, "audited_files": public_bindings,
        "questions": len(questions), "requests": len(rows), "detailed_report_filename": "PUBLIC_PREFLIGHT_DETAIL.json"}
    c.save(output / "PUBLIC_PREFLIGHT_DETAIL.json", detail)
    c.save(output / "PUBLIC_PREFLIGHT_PASS.json", passed)
    print(json.dumps({"status": "PASS", "questions": len(questions), "requests": len(rows), "pass_binding": c.bind(output / "PUBLIC_PREFLIGHT_PASS.json"), "detail_binding": c.bind(output / "PUBLIC_PREFLIGHT_DETAIL.json")}))


if __name__ == "__main__":
    main()
