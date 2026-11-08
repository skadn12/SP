"""CPU-only v5 preparation from a Gold-free public export."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import shutil
import sys
import common as c

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--public-export-freeze", type=Path)
    parser.add_argument("--question-freeze", type=Path, default=c.BASE / "QUESTION_FREEZE.json")
    parser.add_argument("--run", type=Path, default=c.HERE.parent / "direct")
    args = parser.parse_args()
    c.environment()
    run = args.run.resolve()
    c.require(not run.exists(), "RUN_EXISTS_NO_OVERWRITE")
    public_path = args.public.resolve()
    freeze_path = args.question_freeze.resolve()
    question_freeze = c.read(freeze_path)
    c.require(question_freeze.get("status") == "FROZEN_BEFORE_SOLVER", "QUESTION_FREEZE_REQUIRED")
    c.require(question_freeze["gates"]["content_review_status"] == "PASS", "CONTENT_REVIEW_REQUIRED")
    c.require(question_freeze["policy"]["solver_score_filtering"] is False, "NO_SOLVER_FILTERING")
    c.require(question_freeze["policy"]["full_source_no_truncation"] is True, "NO_SOURCE_TRUNCATION")
    # These are byte/hash checks, not JSON parsing of questions or private Gold.
    c.verify_bindings(list(c.all_bindings(question_freeze)))
    payload = c.read(public_path)
    c.require(set(payload) == {"questions"}, "PUBLIC_WRAPPER_WHITELIST")
    questions = payload["questions"]
    c.require(len(questions) == c.QUESTION_COUNT, "QUESTION_COUNT_MISMATCH")
    frozen_slots = {x["candidate_id"]: x for x in question_freeze["questions"]}
    c.require(len(frozen_slots) == c.QUESTION_COUNT and set(frozen_slots) == {q["candidate_id"] for q in questions}, "FROZEN_CANDIDATE_SET_MISMATCH")
    if args.public_export_freeze:
        export = c.read(args.public_export_freeze)
        c.verify_bindings(list(c.all_bindings(export)))
        c.require(any(x["sha256"] == c.sha(public_path) and Path(x["path"]).resolve() == public_path for x in c.all_bindings(export)), "EXPORT_NOT_BOUND_TO_PUBLIC")
        c.require(any(x["sha256"] == c.sha(freeze_path) and Path(x["path"]).resolve() == freeze_path for x in c.all_bindings(export)), "EXPORT_NOT_BOUND_TO_QUESTION_FREEZE")
    else:
        c.require(False, "PUBLIC_EXPORT_FREEZE_REQUIRED")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, solver = c.a.native()
    decoder = solver.generation_config("multi_select")
    expected = {"temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False, "structured_output_version": "finite_choice_set_v1"}
    c.require(all(decoder[k] == v for k, v in expected.items()), "NATIVE_DECODER_CHANGED")
    rows, prompts, tokens, mappings, canonical, source_paths = [], {}, {}, {}, {}, {}
    counts = collections.Counter()
    root_counts = collections.Counter()
    for q in questions:
        c.a.public_question(q)
        qid, root = c.safe_id(q["candidate_id"]), c.safe_id(q["root_id"])
        fs = frozen_slots[qid]
        c.require(fs["root_id"] == root and fs["question_type"] == q["question_type"], "FROZEN_IDENTITY_MISMATCH")
        c.require(fs["source_binding"]["sha256"] == q["source_sha256"], "FROZEN_SOURCE_MISMATCH")
        source = c.rt.source_check(q)
        source_paths[root] = Path(q["source_path"])
        field = "options" if q["question_type"] == "multi_select" else "items"
        c.require(len(q[field]) == (6 if field == "options" else 4), "FROZEN_ITEM_COUNT_MISMATCH")
        variants = c.a.variants(q, c.DISPLAY_SEED, 6 if field == "options" else 12)
        canonical[qid] = c.a.content_ids(q)
        counts[q["question_type"]] += 1
        root_counts[root] += 1
        for variant in variants:
            prompt = c.a.direct_prompt(tokenizer, q, variant, source)
            c.require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source["turns"]), "INCOMPLETE_SOURCE_IN_PROMPT:" + qid)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            c.require(len(ids) + 32 <= 16384, "FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:" + qid)
            finite = c.a.choices(q["question_type"], variant[field])
            c.require(len(finite) == (63 if field == "options" else 24), "FINITE_CHOICE_SET_INVALID")
            for slot in range(3 if field == "options" else 1):
                rid = f"{qid}__{variant['variant_id']}__direct__s{slot:02d}"
                row = {"request_id": rid, "candidate_id": qid, "root_id": root,
                    "question_type": q["question_type"], "branch": "direct", "variant_id": variant["variant_id"],
                    "display": {field: variant[field]}, "slot": slot, "seed": c.SAMPLING_SEED + len(rows),
                    "prompt_sha256": c.rt.textsha(prompt), "prompt_tokens": len(ids),
                    "source_turns": len(source["turns"]), "source_sha256": q["source_sha256"], "max_output_tokens": 32}
                c.a.sampling_params(row, decoder["structured_output_api"])
                rows.append(row)
                prompts[rid], tokens[rid], mappings[rid] = prompt, ids, variant
    c.require(dict(counts) == c.TYPE_COUNTS and dict(root_counts) == c.ROOT_COUNTS, "FROZEN_COUNTS_MISMATCH")
    c.require(len(rows) == c.REQUEST_COUNT, "FROZEN_REQUEST_COUNT_MISMATCH")
    c.require(len({x["request_id"] for x in rows}) == len(rows) and len({x["seed"] for x in rows}) == len(rows), "DUPLICATE_REQUEST_ID_OR_SEED")
    model_bindings = c.rt.models()
    execution_policy = {"schema_version": "live_extension_v5_solver_execution_freeze_v1", "status": "FROZEN_BEFORE_SOLVER",
        "created_at_utc": c.now(), "question_freeze_binding": c.bind(freeze_path), "public_questions_binding": c.bind(public_path),
        "public_export_freeze_binding": c.bind(args.public_export_freeze),
        "solver": {"name": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
            "MC_mappings": 6, "MC_samples_each_mapping": 3, "ordering_mappings": 12, "ordering_samples_each_mapping": 1,
            "temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False, "max_model_len": 16384,
            "display_seed": c.DISPLAY_SEED, "sampling_seed": c.SAMPLING_SEED, "source_truncation": False, "automatic_retry": False},
        "budget": {"planned_question_slots": 14, "question_type_counts": c.TYPE_COUNTS, "max_direct_requests": c.REQUEST_COUNT},
        "solver_score_filtering": False, "raw_responses_freeze_before_gold_read": True}
    run.mkdir(parents=True)
    c.save(run / "PUBLIC_QUESTIONS.json", payload)
    c.save(run / "DISPLAY_MAPPINGS.json", {"requests": mappings, "public_to_canonical": canonical})
    c.save(run / "PROMPT_TOKEN_IDS.json", tokens)
    c.save(run / "EXECUTION_FREEZE.json", execution_policy)
    public_slots = [{"candidate_id": q["candidate_id"], "root_id": q["root_id"],
        "question_type": q["question_type"], "item_count": len(q.get("options", q.get("items")))} for q in questions]
    c.save(run / "PROTOCOL.json", {"schema_version": "live_extension_v5_public_execution_policy_v1",
        "solver": execution_policy["solver"], "budget": execution_policy["budget"], "slots": public_slots})
    c.save(run / "CONTENT_ADMISSION.json", {"status": "PASS", "namespace": "train.feedback",
        "question_freeze_binding": c.bind(freeze_path), "public_questions_sha256": c.sha(public_path),
        "candidate_ids": [q["candidate_id"] for q in questions], "planned_slots": c.QUESTION_COUNT,
        "admitted_count": len(questions), "production_planned_slots": c.QUESTION_COUNT,
        "missing_candidate_ids": [], "solver_score_filtering": False})
    for root, source_path in source_paths.items():
        target = run / "sources" / (root + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, target)
    for rid, prompt in prompts.items():
        target = run / "prompts" / (rid + ".txt")
        target.parent.mkdir(exist_ok=True)
        target.write_text(prompt)
    staged = sorted(p for p in run.rglob("*") if p.is_file())
    manifest = {"schema_version": "live_extension_v5_solver_manifest_v1", "status": "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT",
        "prepared_at_utc": c.now(), "branch": "direct", "solver_model": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
        "namespace": "train.feedback/diagnostic_only", "use_as_final_eval": False,
        "denominator_scope": "14 content-admitted v5 diagnostic questions and all 228 fixed requests; no solver-based admission",
        "planned_question_slots": c.QUESTION_COUNT, "missing_candidate_ids": [],
        "production_denominator": c.QUESTION_COUNT, "production_admitted_numerator": c.QUESTION_COUNT,
        "production_unadmitted_slots": 0,
        "question_count": len(questions), "question_type_counts": dict(counts), "root_count": len(source_paths),
        "request_count": len(rows), "requests": rows, "decoder_contract": decoder, "batch_size": 8,
        "max_model_len": 16384, "prefix_caching": False, "automatic_retry": False, "source_truncation": False,
        "gold_opened_by_prepare": False, "gold_parsed_by_prepare": False, "training_started": False,
        "gold_access_scope": "prepare and live parse public questions only; original private question bytes are hashed only; score may parse Gold only after raw freeze",
        "question_freeze_binding": c.bind(freeze_path), "execution_freeze_binding": c.bind(run / "EXECUTION_FREEZE.json"),
        "random_display": {"seed": c.DISPLAY_SEED, "gold_blind": True, "algorithm": "v15_independent_seeded_rebinding_dynamic4_5_6"},
        "sampling_seed_start": c.SAMPLING_SEED,
        "public_input_bindings": [c.bind(p) for p in staged],
        "input_bindings": [c.bind(public_path), c.bind(freeze_path), c.bind(args.public_export_freeze)] + list(c.all_bindings(question_freeze)),
        "model_bindings": model_bindings, "code_bindings": [c.bind(p) for p in c.code_paths()], "versions": c.rt.versions(), "python_prefix": sys.prefix,
        "max_prompt_tokens": max(x["prompt_tokens"] for x in rows), "minimum_context_headroom": min(16384 - x["prompt_tokens"] - 32 for x in rows)}
    c.save(run / "manifest.json", manifest)
    c.save(run / "PREFLIGHT_FREEZE.json", {"status": manifest["status"], "created_at_utc": c.now(), "files": [c.bind(p) for p in sorted(run.rglob("*")) if p.is_file()],
        "questions": len(questions), "requests": len(rows), "inference_started": False, "public_only": True})
    print(json.dumps({k: manifest[k] for k in ("status", "question_count", "question_type_counts", "request_count", "max_prompt_tokens", "minimum_context_headroom")}), flush=True)

if __name__ == "__main__":
    main()
