"""CPU prepare from admitted public questions; private Gold is copied, not parsed."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import common as c


def prepare(args):
    c.environment()
    run = args.run.resolve()
    private = args.private_output.resolve() if args.private_output else run.parent / (run.name + "_private")
    c.require(not run.exists() and not private.exists(), "NO_OVERWRITE_RUN_OR_PRIVATE")
    c.require(not c.below(private, run) and not c.below(run, private), "DISJOINT_PRIVATE_SIDECAR_REQUIRED")
    public_path, gold_path, protocol_path = [path.resolve() for path in (args.public, args.private, args.protocol)]
    c.require(public_path != gold_path, "SEPARATE_PUBLIC_AND_GOLD_REQUIRED")
    snapshots = {}
    def public_snapshot(path):
        path = path.resolve()
        raw = path.read_bytes()
        snapshots[path] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return json.loads(raw)
    payload, policy, admission = public_snapshot(public_path), public_snapshot(protocol_path), public_snapshot(args.admission)
    frozen_private_binding = c.bind(gold_path)  # Hashes bytes; never parses JSON.
    c.require(set(payload) == {"questions"}, "PUBLIC_WRAPPER_WHITELIST")
    questions = payload["questions"]
    budget, solver = policy["budget"], policy["solver"]
    c.require(isinstance(questions, list) and len(questions) == budget["initial_question_slots"] <= 24, "ADMITTED_COUNT_DIFFERS_FROM_FREEZE")
    c.require(len({q["candidate_id"] for q in questions}) == len(questions), "DUPLICATE_CANDIDATE_ID")
    slots = {slot["candidate_id"]: slot for slot in policy["slots"]}
    c.require(len(slots) == len(policy["slots"]) == budget["production_planned_slots"] == 24, "EXPECTED_24_UNIQUE_PRODUCTION_SLOTS")
    c.require(len({slot["slot_id"] for slot in slots.values()}) == 24, "EXPECTED_24_UNIQUE_SLOT_IDS")
    c.require(len({slot["root_id"] for slot in slots.values()}) == 4, "EXPECTED_FOUR_PRODUCTION_ROOTS")
    c.require({q["candidate_id"] for q in questions} <= set(slots), "UNPLANNED_QUESTION")
    expected = c.SOLVER_FIXED
    c.require(all(solver[key] == value for key, value in expected.items()), "FROZEN_SOLVER_CONTRACT_MISMATCH")
    c.require(type(solver["display_seed"]) is int and type(solver["sampling_seed"]) is int, "INTEGER_SEEDS_REQUIRED")
    c.require(budget["max_direct_requests"] == sum(c.a.request_count(q) for q in questions), "FROZEN_ARITY_REQUEST_BUDGET_MISMATCH")
    c.require(admission.get("status") in ("PASS", "FROZEN", "CONTENT_FROZEN") and str(admission.get("namespace", "")).startswith("train.feedback"), "CONTENT_ADMISSION_REQUIRED")
    c.require({q["candidate_id"] for q in admission["questions"]} == {q["candidate_id"] for q in questions}, "ADMISSION_CANDIDATE_MISMATCH")
    c.require(admission["public_questions"]["sha256"] == snapshots[public_path]["sha256"] and admission["private_questions"]["sha256"] == frozen_private_binding["sha256"], "ADMISSION_QA_HASH_MISMATCH")
    c.require(admission["planned_slots"] == 24 and admission["admitted_count"] == len(questions), "ADMISSION_COUNT_MISMATCH")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, native_solver = c.a.native()
    decoder = native_solver.generation_config("multi_select")
    c.require(all(decoder[key] == expected[key] for key in ("temperature", "top_p", "max_tokens", "enable_thinking")), "NATIVE_DECODER_CHANGED")
    rows, prompts, tokens, mappings, canonical, sources = [], {}, {}, {}, {}, {}
    for question in questions:
        c.a.public_question(question)
        qid, root = c.safe_id(question["candidate_id"]), c.safe_id(question["root_id"])
        slot = slots[qid]
        field = "options" if question["question_type"] == "multi_select" else "items"
        c.require(slot["root_id"] == root and slot["question_type"] == question["question_type"] and slot["item_count"] == len(question[field]), "FROZEN_SLOT_IDENTITY_MISMATCH")
        source = c.rt.source_check(question)
        c.require(Path(slot["source"]["path"]).resolve() == Path(question["source_path"]).resolve() and slot["source"]["sha256"] == question["source_sha256"], "FROZEN_SOURCE_MISMATCH")
        if root in sources:
            c.require(c.sha(sources[root]) == question["source_sha256"], "ROOT_SOURCE_VERSION_CONFLICT")
        sources[root] = Path(question["source_path"])
        canonical[qid] = c.a.content_ids(question)
        variants = c.a.variants(question, solver["display_seed"], c.a.mapping_count(question))
        for variant in variants:
            text = c.a.direct_prompt(tokenizer, question, variant, source)
            c.require(all(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" in text for turn in source["turns"]), "INCOMPLETE_SOURCE_PROMPT")
            ids = tokenizer.encode(text, add_special_tokens=False)
            c.require(len(ids) + 32 <= 16384, "FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:" + qid)
            for sample in range(3 if field == "options" else 1):
                rid = f"{qid}__{variant['variant_id']}__direct__s{sample:02d}"
                row = {"request_id": rid, "candidate_id": qid, "root_id": root, "question_type": question["question_type"],
                    "branch": "direct", "variant_id": variant["variant_id"], "display": {field: variant[field]}, "slot": sample,
                    "seed": solver["sampling_seed"] + len(rows), "prompt_sha256": c.rt.textsha(text), "prompt_tokens": len(ids),
                    "source_turns": len(source["turns"]), "source_sha256": question["source_sha256"], "max_output_tokens": 32}
                c.a.sampling_params(row, decoder["structured_output_api"])
                rows.append(row)
                prompts[rid], tokens[rid], mappings[rid] = text, ids, variant
    c.require(len(rows) == budget["max_direct_requests"], "PREPARED_REQUEST_COUNT_DRIFT")
    c.verify_bindings(list(snapshots.values()) + [frozen_private_binding])
    run.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    c.save(run / "PUBLIC_QUESTIONS.json", payload)
    c.save(run / "DISPLAY_MAPPINGS.json", {"requests": mappings, "public_to_canonical": canonical})
    c.save(run / "PROMPT_TOKEN_IDS.json", tokens)
    # Cards, Gold, evidence, authoring reviews and metadata never enter execution.
    slot_keys = ("slot_id", "candidate_id", "root_id", "question_type", "item_count")
    c.save(run / "PROTOCOL.json", {"schema_version": "day3_crossroot_public_policy_v1", "namespace": admission["namespace"], "solver": {key: solver[key] for key in (*c.SOLVER_FIXED, "display_seed", "sampling_seed")},
        "budget": {key: budget[key] for key in ("initial_question_slots", "production_planned_slots", "max_direct_requests")},
        "slots": [{key: slot[key] for key in slot_keys if key in slot} for slot in policy["slots"]]})
    admitted_ids = {q["candidate_id"] for q in questions}
    missing = sorted(set(slots) - admitted_ids)
    c.save(run / "CONTENT_ADMISSION.json", {"status": "PASS", "namespace": admission["namespace"],
        "candidate_ids": sorted(admitted_ids), "public_questions_sha256": snapshots[public_path]["sha256"], "admission_sha256": snapshots[args.admission.resolve()]["sha256"],
        "planned_slots": 24, "admitted_count": len(questions), "production_planned_slots": 24, "missing_candidate_ids": missing})
    for root, source_path in sources.items():
        dest = run / "sources" / (root + ".json")
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, dest)
    for rid, prompt in prompts.items():
        dest = run / "prompts" / (rid + ".txt")
        dest.parent.mkdir(exist_ok=True)
        with dest.open("x") as stream:
            stream.write(prompt)
    staged = [path for path in sorted(run.rglob("*")) if path.is_file()]
    counts = dict(collections.Counter(q["question_type"] for q in questions))
    arities = collections.Counter(q["question_type"] + ":" + str(len(q["options"] if q["question_type"] == "multi_select" else q["items"])) for q in questions)
    manifest = {"status": "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "prepared_at_utc": c.now(), "branch": "direct",
        "solver_model": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
        "namespace": admission["namespace"], "use_as_final_eval": False, "training_started": False,
        "denominator_scope": "Content-admitted diagnostic questions; all24 production slots and four planned roots separately retained.",
        "planned_question_slots": len(questions), "question_count": len(questions), "question_type_counts": counts,
        "question_arity_counts": dict(arities), "request_count": len(rows), "missing_candidate_ids": missing,
        "production_denominator": 24, "production_admitted_numerator": len(questions), "production_unadmitted_slots": len(missing),
        "planned_root_count": 4, "root_count": len(sources), "requests": rows, "decoder_contract": decoder,
        "batch_size": 8, "max_model_len": 16384, "gpu_memory_utilization": .8, "prefix_caching": False, "automatic_retry": False, "source_truncation": False,
        "gold_opened_by_prepare": False, "gold_bytes_copied_by_prepare": True, "gold_parsed_by_prepare": False,
        "protocol_original_binding": snapshots[protocol_path], "gold_access_scope": "Prepare copies private bytes into separate sidecar without JSON parsing; live reads public inputs only; scorer first parses Gold after all terminal responses are frozen.",
        "random_display": {"seed": solver["display_seed"], "gold_blind": True, "algorithm": "v15_independent_seeded_rebinding_local_MC6_order3_6"},
        "sampling_seed_start": solver["sampling_seed"], "public_input_bindings": [c.bind(path) for path in staged],
        "model_bindings": c.rt.models(), "code_bindings": [c.bind(path) for path in c.code_paths()], "versions": c.rt.versions(),
        "python_prefix": sys.prefix, "max_prompt_tokens": max((row["prompt_tokens"] for row in rows), default=0),
        "minimum_context_headroom": min((16384-row["prompt_tokens"]-32 for row in rows), default=None),
        "scoring_contract": {"exact": "canonical content set for MC; complete canonical content sequence for permutation",
            "invalid": "zero exact and ordering pairs on fixed call denominators; unknown MC selections/FP/FN remain null",
            "ordering_pair_total": "n*(n-1)/2 per request; report arity strata", "question_equal_primary": True,
            "root_equal_secondary": True, "permutation_counts": "MC6x3; ordering min(12,n!)x1", "A_B_effect_estimate": False}}
    c.save(run / "manifest.json", manifest)
    c.save(run / "PREFLIGHT_FREEZE.json", {"status": manifest["status"], "created_at_utc": c.now(),
        "files": [c.bind(path) for path in sorted(run.rglob("*")) if path.is_file()],
        "planned_slots": 24, "questions": len(questions), "requests": len(rows), "inference_started": False, "public_only": True})
    shutil.copy2(gold_path, private / "PRIVATE_QUESTIONS.json")
    (private / "PRIVATE_QUESTIONS.json").chmod(0o600)
    copied_binding = c.bind(private / "PRIVATE_QUESTIONS.json")
    c.require(all(copied_binding[key] == frozen_private_binding[key] for key in ("sha256", "bytes")), "PRIVATE_COPY_DIFFERS_FROM_FROZEN_ADMISSION")
    c.save(private / "PRIVATE_BINDINGS.json", {"created_at_utc": c.now(), "public_manifest_binding": c.bind(run / "manifest.json"),
        "public_preflight_binding": c.bind(run / "PREFLIGHT_FREEZE.json"), "original_private_binding": frozen_private_binding,
        "private_answer_binding": copied_binding, "private_json_parsed_by_prepare": False})
    (private / "PRIVATE_BINDINGS.json").chmod(0o600)
    print(json.dumps({key: manifest[key] for key in ("status", "question_count", "question_type_counts", "question_arity_counts", "request_count", "max_prompt_tokens", "minimum_context_headroom")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "private", "run", "protocol", "admission"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
