"""CPU preparation; Gold bytes go to an independent private sidecar, never parsed."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import shutil
import sys
import common as c


def prepare(args):
    c.environment()
    run = args.run.resolve()
    private = args.private_output.resolve() if args.private_output else run.parent / (run.name + "_private")
    c.require(not run.exists() and not private.exists(), "NO_OVERWRITE_RUN_OR_PRIVATE_SIDECAR")
    c.require(not c.below(private, run) and not c.below(run, private), "PRIVATE_SIDECAR_MUST_BE_DISJOINT")
    public_path, gold_path, protocol_path = (path.resolve() for path in (args.public, args.private, args.protocol))
    c.require(public_path != gold_path, "PUBLIC_AND_PRIVATE_MUST_BE_SEPARATE")
    payload = c.read(public_path)
    c.require(set(payload) == {"questions"}, "PUBLIC_WRAPPER_WHITELIST")
    questions = payload["questions"]
    c.require(isinstance(questions, list) and len(questions) == c.DIAGNOSTIC_QUESTIONS, "ADMITTED_QUESTION_CAP_EXCEEDED")
    c.require(len({q["candidate_id"] for q in questions}) == len(questions), "DUPLICATE_CANDIDATE_ID")
    policy = c.read(protocol_path)
    frozen_solver = policy["solver"]
    frozen_expected = {"name": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
        "MC_mappings": 6, "MC_samples_each_mapping": 3, "ordering_mappings": 12,
        "ordering_samples_each_mapping": 1, "temperature": .8, "top_p": .9,
        "max_tokens": 32, "enable_thinking": False, "max_model_len": 16384,
        "source_truncation": False, "automatic_retry": False}
    c.require(all(frozen_solver[key] == value for key, value in frozen_expected.items()), "EXECUTION_FREEZE_DECODER_MISMATCH")
    c.require(policy["budget"]["initial_question_slots"] == c.DIAGNOSTIC_QUESTIONS and policy["budget"]["max_direct_requests"] == c.DIRECT_REQUESTS and policy["budget"]["production_planned_slots"] == c.PRODUCTION_PLANNED_SLOTS, "EXECUTION_FREEZE_BUDGET_MISMATCH")
    display_seed = args.display_seed if args.display_seed is not None else frozen_solver["display_seed"]
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else frozen_solver["sampling_seed"]
    c.require(display_seed == frozen_solver["display_seed"] == c.DISPLAY_SEED and sampling_seed == frozen_solver["sampling_seed"] == c.SAMPLING_SEED, "FROZEN_SEED_OVERRIDE_FORBIDDEN")
    slots = {slot["candidate_id"]: slot for slot in policy["slots"]}
    c.require(len(slots) == c.DIAGNOSTIC_QUESTIONS and {q["candidate_id"] for q in questions} == set(slots), "FROZEN_SLOT_SET_MISMATCH")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, solver = c.a.native()
    decoder = solver.generation_config("multi_select")
    expected = {"temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False, "structured_output_version": "finite_choice_set_v1"}
    c.require(all(decoder[key] == value for key, value in expected.items()), "NATIVE_DECODER_CHANGED")
    rows, prompts, tokens, maps, canonical, source_paths = [], {}, {}, {}, {}, {}
    for question in questions:
        c.a.public_question(question)
        qid, root = c.safe_id(question["candidate_id"]), c.safe_id(question["root_id"])
        slot_design = slots[qid]
        c.require(slot_design["root_id"] == root and slot_design["question_type"] == question["question_type"], "FROZEN_SLOT_IDENTITY_MISMATCH")
        source = c.rt.source_check(question)
        c.require(slot_design["source"]["sha256"] == question["source_sha256"], "FROZEN_SLOT_SOURCE_MISMATCH")
        if root in source_paths:
            c.require(c.sha(source_paths[root]) == question["source_sha256"], "CONFLICTING_SOURCE_VERSIONS")
        source_paths[root] = Path(question["source_path"])
        field = "options" if question["question_type"] == "multi_select" else "items"
        c.require(len(question[field]) == slot_design["item_count"] == 6, "FROZEN_SIX_ITEMS_REQUIRED")
        variants = c.a.variants(question, display_seed, 6 if field == "options" else 12)
        canonical[qid] = c.a.content_ids(question)
        for variant in variants:
            prompt = c.a.direct_prompt(tokenizer, question, variant, source)
            c.require(all(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" in prompt for turn in source["turns"]), "INCOMPLETE_SOURCE_IN_PROMPT")
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            c.require(len(ids) + 32 <= 16384, "FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:" + qid)
            for slot in range(3 if field == "options" else 1):
                rid = f"{qid}__{variant['variant_id']}__direct__s{slot:02d}"
                row = {"request_id": rid, "candidate_id": qid, "root_id": root,
                       "question_type": question["question_type"], "branch": "direct",
                       "variant_id": variant["variant_id"], "display": {field: variant[field]},
                       "slot": slot, "seed": sampling_seed + len(rows),
                       "prompt_sha256": c.rt.textsha(prompt), "prompt_tokens": len(ids),
                       "source_turns": len(source["turns"]), "source_sha256": question["source_sha256"],
                       "max_output_tokens": 32}
                c.a.sampling_params(row, decoder["structured_output_api"])
                rows.append(row)
                prompts[rid], tokens[rid], maps[rid] = prompt, ids, variant
    counts = collections.Counter(q["question_type"] for q in questions)
    c.require(dict(counts) == c.TYPE_COUNTS, "FROZEN_TYPE_SLOT_CAP_EXCEEDED")
    c.require(len(rows) == 18 * counts["multi_select"] + 12 * counts["ordering"] == c.DIRECT_REQUESTS, "FROZEN_REQUEST_CAP_EXCEEDED")
    c.require(len({row["request_id"] for row in rows}) == len(rows), "DUPLICATE_REQUEST_ID")
    c.require(len({row["seed"] for row in rows}) == len(rows), "DUPLICATE_SAMPLING_SEED")
    model_bindings = c.rt.models()
    admission = None
    if args.admission:
        admission = c.read(args.admission)
        c.require(admission.get("status") in ("PASS", "FROZEN", "CONTENT_FROZEN") and str(admission.get("namespace", "")).startswith("train.feedback"), "CONTENT_ADMISSION_REQUIRED")
        c.require({row["candidate_id"] for row in admission["questions"]} == {q["candidate_id"] for q in questions}, "ADMISSION_CANDIDATE_SET_MISMATCH")
        c.require(admission["public_questions"]["sha256"] == c.sha(public_path) and admission["private_questions"]["sha256"] == c.sha(gold_path), "ADMISSION_QA_BINDING_MISMATCH")
        c.require(admission.get("planned_slots") == c.DIAGNOSTIC_QUESTIONS and admission.get("admitted_count") == len(questions) and admission.get("production_planned_slots") == c.PRODUCTION_PLANNED_SLOTS, "ADMISSION_COUNTS_MISMATCH")
    run.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    c.save(run / "PUBLIC_QUESTIONS.json", payload)
    c.save(run / "DISPLAY_MAPPINGS.json", {"requests": maps, "public_to_canonical": canonical})
    c.save(run / "PROMPT_TOKEN_IDS.json", tokens)
    # The execution-side policy is public: quotas, cards, plans and review
    # evidence stay in the original authoring freeze outside this run.
    public_slot_keys = ("slot_id", "candidate_id", "root_id", "root_index",
                        "arm", "question_type", "item_count")
    public_policy = {
        "schema_version": "day3_solver187_public_execution_policy_v1",
        "solver": policy["solver"],
        "budget": policy["budget"],
        "slots": [{key: slot[key] for key in public_slot_keys if key in slot}
                  for slot in policy["slots"]],
    }
    c.save(run / "PROTOCOL.json", public_policy)
    c.save(run / "CONTENT_ADMISSION.json", {"status": "PASS" if admission else "NOT_PROVIDED",
           "namespace": "train.feedback", "candidate_ids": [q["candidate_id"] for q in questions],
           "public_questions_sha256": c.sha(public_path), "admission_sha256": c.sha(args.admission) if admission else None,
           "planned_slots": c.DIAGNOSTIC_QUESTIONS, "admitted_count": len(questions),
           "production_planned_slots": c.PRODUCTION_PLANNED_SLOTS,
           "missing_candidate_ids": sorted(set(slots) - {q["candidate_id"] for q in questions})})
    for root, source_path in source_paths.items():
        target = run / "sources" / (root + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, target)
    for rid, prompt in prompts.items():
        target = run / "prompts" / (rid + ".txt")
        target.parent.mkdir(exist_ok=True)
        target.write_text(prompt)
    counts = collections.Counter(q["question_type"] for q in questions)
    staged = sorted(path for path in run.rglob("*") if path.is_file())
    manifest = {"status": "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "prepared_at_utc": c.now(),
        "branch": "direct", "solver_model": "S0_M_clean", "model": str(c.MODEL),
        "tokenizer": str(c.TOKENIZER), "namespace": "train.feedback/diagnostic_only", "use_as_final_eval": False,
        "denominator_scope": "187 content-admitted diagnostic questions; production accounting separately retains all240 slots",
        "planned_question_slots": c.DIAGNOSTIC_QUESTIONS, "question_count": len(questions), "question_type_counts": dict(counts), "request_count": len(rows),
        "missing_candidate_ids": sorted(set(slots) - {q["candidate_id"] for q in questions}),
        "production_denominator": c.PRODUCTION_PLANNED_SLOTS, "production_admitted_numerator": len(questions),
        "production_unadmitted_slots": c.PRODUCTION_PLANNED_SLOTS - len(questions),
        "root_count": len(source_paths), "requests": rows, "decoder_contract": decoder,
        "batch_size": 8, "max_model_len": 16384, "prefix_caching": False,
        "automatic_retry": False, "source_truncation": False, "gold_opened_by_prepare": False,
        "gold_bytes_copied_by_prepare": True, "gold_parsed_by_prepare": False, "training_started": False,
        "protocol_original_binding": c.bind(protocol_path),
        "gold_access_scope": "Gold bytes only copied to independent private sidecar; live opens public inputs only; score parses Gold after committed terminal raw freeze.",
        "random_display": {"seed": display_seed, "gold_blind": True, "algorithm": "v15_independent_seeded_rebinding_dynamic4_5_6"},
        "sampling_seed_start": sampling_seed, "public_input_bindings": [c.bind(path) for path in staged],
        "model_bindings": model_bindings, "code_bindings": [c.bind(path) for path in c.code_paths()],
        "versions": c.rt.versions(), "python_prefix": sys.prefix,
        "max_prompt_tokens": max((row["prompt_tokens"] for row in rows), default=0),
        "minimum_context_headroom": min((16384 - row["prompt_tokens"] - 32 for row in rows), default=None)}
    c.save(run / "manifest.json", manifest)
    c.save(run / "PREFLIGHT_FREEZE.json", {"status": manifest["status"], "created_at_utc": c.now(),
        "files": [c.bind(path) for path in sorted(run.rglob("*")) if path.is_file()],
        "planned_slots": c.DIAGNOSTIC_QUESTIONS, "questions": len(questions), "requests": len(rows), "inference_started": False, "public_only": True})
    shutil.copy2(gold_path, private / "PRIVATE_QUESTIONS.json")
    (private / "PRIVATE_QUESTIONS.json").chmod(0o600)
    c.save(private / "PRIVATE_BINDINGS.json", {"created_at_utc": c.now(),
        "public_manifest_binding": c.bind(run / "manifest.json"),
        "public_preflight_binding": c.bind(run / "PREFLIGHT_FREEZE.json"),
        "original_private_binding": c.bind(gold_path),
        "private_answer_binding": c.bind(private / "PRIVATE_QUESTIONS.json"),
        "private_json_parsed_by_prepare": False})
    (private / "PRIVATE_BINDINGS.json").chmod(0o600)
    print(json.dumps({key: manifest[key] for key in ("status", "question_count", "question_type_counts", "request_count", "max_prompt_tokens", "minimum_context_headroom")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--protocol", type=Path, default=c.HERE.parent / "EXECUTION_FREEZE.json")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--display-seed", type=int)
    parser.add_argument("--sampling-seed", type=int)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
