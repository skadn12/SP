"""Rebuild and validate v5 public requests on CPU; never issue model requests."""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path
import common as c

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=c.HERE.parent / "direct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run / "CPU_PREFLIGHT.json"
    c.require(not output.exists(), "PREFLIGHT_OUTPUT_EXISTS_NO_OVERWRITE")
    manifest = c.verify(run)
    c.require(not (run / "STARTED.json").exists(), "SOLVER_ALREADY_STARTED")
    public = c.read(run / "PUBLIC_QUESTIONS.json")
    c.require(set(public) == {"questions"}, "PUBLIC_WRAPPER_INVALID")
    questions = public["questions"]
    c.require(len(questions) == c.QUESTION_COUNT, "PUBLIC_QUESTION_COUNT_MISMATCH")
    question_map = {q["candidate_id"]: q for q in questions}
    c.require(len(question_map) == c.QUESTION_COUNT, "DUPLICATE_QUESTION")
    mappings = c.read(run / "DISPLAY_MAPPINGS.json")
    tokens = c.read(run / "PROMPT_TOKEN_IDS.json")
    rows = manifest["requests"]
    request_map = {x["request_id"]: x for x in rows}
    c.require(set(request_map) == set(tokens) == set(mappings["requests"]), "REQUEST_ARTIFACT_SET_MISMATCH")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    expected_counts = collections.Counter()
    expected_root = collections.Counter()
    max_prompt_tokens = 0
    rebuilt_ids = []
    checked = []
    for q in questions:
        c.a.public_question(q)
        qid = q["candidate_id"]
        field = "options" if q["question_type"] == "multi_select" else "items"
        n = len(q[field])
        c.require(n == (6 if field == "options" else 4), "ITEM_COUNT_MISMATCH")
        canonical = c.a.content_ids(q)
        c.require(mappings["public_to_canonical"][qid] == canonical, "CANONICAL_MAPPING_DRIFT")
        source_path = run / "sources" / (q["root_id"] + ".json")
        c.require(c.sha(source_path) == q["source_sha256"], "STAGED_SOURCE_DRIFT")
        source = c.read(source_path)
        c.require(source["root_id"] == q["root_id"], "SOURCE_ROOT_MISMATCH")
        refs = [t["turn_ref"] for t in source["turns"]]
        c.require(len(refs) == len(set(refs)), "DUPLICATE_SOURCE_REF")
        variants = c.a.variants(q, c.DISPLAY_SEED, 6 if field == "options" else 12)
        orders = [tuple(v["display_order_content_ids"]) for v in variants]
        c.require(len(set(orders)) == len(variants), "DUPLICATE_DISPLAY_PERMUTATION")
        for variant in variants:
            prompt = c.a.direct_prompt(tokenizer, q, variant, source)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            c.require(len(ids) + 32 <= manifest["max_model_len"], "CONTEXT_OVERFLOW")
            c.require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source["turns"]), "MISSING_SOURCE_TURN")
            c.require(q["question"] in prompt and all(value in prompt for value in variant[field].values()), "PROMPT_CONTENT_INCOMPLETE")
            finite = c.a.choices(q["question_type"], variant[field])
            c.require(len(finite) == (2 ** n - 1 if field == "options" else math.factorial(n)), "FINITE_CHOICE_COUNT_INVALID")
            c.require(len(set(finite)) == len(finite), "DUPLICATE_FINITE_CHOICE")
            c.require(all(text.startswith("<box>") and text.endswith("</box>") for text in finite), "FINITE_CHOICE_FORMAT_INVALID")
            for slot in range(3 if field == "options" else 1):
                rid = f"{qid}__{variant['variant_id']}__direct__s{slot:02d}"
                rebuilt_ids.append(rid)
                c.require(rid in request_map, "MISSING_REQUEST")
                row = request_map[rid]
                c.require(row["candidate_id"] == qid and row["root_id"] == q["root_id"] and row["question_type"] == q["question_type"], "REQUEST_IDENTITY_DRIFT")
                c.require(row["branch"] == "direct" and row["slot"] == slot and row["variant_id"] == variant["variant_id"], "REQUEST_VARIANT_DRIFT")
                c.require(row["seed"] == c.SAMPLING_SEED + len(rebuilt_ids) - 1, "SEED_SEQUENCE_DRIFT")
                c.require(row["display"] == {field: variant[field]} and mappings["requests"][rid] == variant, "DISPLAY_MAPPING_DRIFT")
                p = run / "prompts" / (rid + ".txt")
                c.require(p.read_text() == prompt and c.sha(p) == row["prompt_sha256"], "PROMPT_BYTES_DRIFT")
                c.require(tokens[rid] == ids and row["prompt_tokens"] == len(ids), "PROMPT_TOKEN_DRIFT")
                c.require(row["source_turns"] == len(source["turns"]) and row["source_sha256"] == q["source_sha256"], "REQUEST_SOURCE_DRIFT")
                c.require(row["max_output_tokens"] == 32, "OUTPUT_BUDGET_DRIFT")
                c.a.sampling_params(row, manifest["decoder_contract"]["structured_output_api"])
                max_prompt_tokens = max(max_prompt_tokens, len(ids))
        expected_counts[q["question_type"]] += 1
        expected_root[q["root_id"]] += 1
        checked.append({"candidate_id": qid, "root_id": q["root_id"], "question_type": q["question_type"],
            "source_turns": len(source["turns"]), "item_count": n, "unique_display_permutations": len(variants),
            "samples_each_permutation": 3 if field == "options" else 1, "finite_choice_count": len(finite)})
    c.require(dict(expected_counts) == c.TYPE_COUNTS and dict(expected_root) == c.ROOT_COUNTS, "QUESTION_COUNTS_DRIFT")
    c.require(rebuilt_ids == [x["request_id"] for x in rows] and len(rebuilt_ids) == c.REQUEST_COUNT, "REQUEST_SEQUENCE_DRIFT")
    c.require(manifest["max_prompt_tokens"] == max_prompt_tokens, "MAX_TOKEN_SUMMARY_DRIFT")
    report = {"schema_version": "live_extension_v5_cpu_preflight_v1", "status": "PASS", "created_at_utc": c.now(),
        "cpu_only": True, "model_requests_issued": 0, "gpu_solver_started": False, "gold_parsed_by_preflight": False,
        "independent_review_required": True, "independent_review_completed": False,
        "question_count": c.QUESTION_COUNT, "root_count": len(expected_root), "question_type_counts": dict(expected_counts),
        "request_count": c.REQUEST_COUNT, "direct_mc_requests": 180, "direct_ordering_requests": 48,
        "display_seed": c.DISPLAY_SEED, "sampling_seed_start": c.SAMPLING_SEED,
        "max_prompt_tokens": max_prompt_tokens, "minimum_context_headroom": 16384 - max_prompt_tokens - 32,
        "checks_passed": ["bound question freeze and public export hashes", "14 public questions and 2 roots", "MC2/3/4/5/6 frozen source admission retained",
            "228 unique request IDs and seeds", "6 MC and 12 ordering distinct mappings per question", "63 MC and 24 ordering native finite choices",
            "all source turns preserved in every prompt", "public native prompt byte and token regeneration", "16384 context budget without truncation",
            "temperature .8 top_p .9 max_tokens 32 enable_thinking false", "no retry and no inference"],
        "questions": checked, "audited_files": [c.bind(run / "PREFLIGHT_FREEZE.json"), c.bind(run / "manifest.json")],
        "script_binding": c.bind(Path(__file__))}
    c.save(output, report)
    print(json.dumps({k: report[k] for k in ("status", "question_count", "request_count", "max_prompt_tokens", "minimum_context_headroom", "model_requests_issued")}), flush=True)

if __name__ == "__main__":
    main()
