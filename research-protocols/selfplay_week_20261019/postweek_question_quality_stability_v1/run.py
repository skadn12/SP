"""Train-only cross-root replication of the four question repairs.

Reuse the frozen pilot capture/validation code without changing its artifacts.
The GPU entry point opens public prepared requests only; scoring runs afterward.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
BASE_PATH = WEEK / "postweek_question_quality_pilot_v1/run.py"
spec = importlib.util.spec_from_file_location("stability_frozen_pilot_reuse", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = HERE / "execution"
NEW = HERE / "CANDIDATES_PRIVATE.json"
REVIEW = HERE / "CONTENT_REVIEW_AI.json"
SELECTION = HERE / "SELECTION_FREEZE.json"
read, save, bind, check = base.read, base.save, base.bind, base.check
gate, diagnostic = base.gate, base.diagnostic
CONDITIONS = ("FULL", "QUESTION_ONLY")
VARIANTS = ("perm_00", "perm_01")
DISPLAY_SEED = 2026102302
MAX_TOKENS, MODEL_LEN = 3072, 16384
PUBLIC_KEYS = base.PUBLIC_KEYS | {"variant_id"}


def sampling_seed(candidate_id):
    return int.from_bytes(hashlib.sha256(("quality_stability_v1:" + candidate_id).encode()).digest()[:4], "big") % 2147483647


def displays(candidate):
    """Reject only duplicate displays, independently of answer values."""
    seen = set()
    for index, variant in enumerate(VARIANTS):
        seed = DISPLAY_SEED + index
        skipped = []
        while True:
            public, private = gate.prepare_display(candidate, seed)
            permutation = tuple(private["display_to_canonical"].values())
            if permutation not in seen:
                break
            skipped.append(seed)
            seed += len(VARIANTS)
        seen.add(permutation)
        private["duplicate_display_seed_skips"] = skipped
        yield variant, public, private


def prepare():
    assert not OUT.exists(), "NO_PREPARATION_OVERWRITE"
    selection = read(SELECTION)
    roots = {row["root_id"]: row for row in selection["roots"]}
    candidates = read(NEW)
    reviews = read(REVIEW)["reviews"]
    by_id = {row["candidate_id"]: row for row in reviews}
    assert len(by_id) == len(reviews) == len(candidates) == 28, "ALL_28_DRAFTS_AND_REVIEWS_REQUIRED"
    expected = {f"QQS1-{rid.split('-')[-1]}-{slot}" for rid in roots for slot in selection["slots_per_root"]}
    assert set(by_id) == {c["candidate_id"] for c in candidates} == expected, "FROZEN_SLOT_SET_DRIFT"
    admitted, ledger = [], []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        review = by_id[cid]
        assert review["candidate_sha256"] == base.fingerprint(candidate), "REVIEW_BINDING_DRIFT:" + cid
        assert candidate["author_id"] == review["author_id"] != review["reviewer_id"], "INDEPENDENT_REVIEW_REQUIRED"
        assert review["decision"] in ("PASS", "HOLD", "FAIL")
        source_binding = candidate["source_binding"]
        selected = roots[candidate["root_id"]]
        assert source_binding == {k: selected[k] for k in ("path", "sha256", "bytes")}, "SOURCE_SELECTION_DRIFT"
        check(source_binding)
        if review["decision"] == "PASS":
            gate._validate(candidate)
            assert 4 <= len(candidate["items"]) <= 6 if candidate["question_type"] == "ordering" else len(candidate["options"]) == 6
            admitted.append(candidate)
        ledger.append({"candidate_id": cid, "root_id": candidate["root_id"], "decision": review["decision"],
            "initial_decision": review.get("initial_decision", review["decision"]),
            "initial_content_decision": review["initial_content_decision"],
            "candidate_sha256": base.fingerprint(candidate), "author_id": candidate["author_id"], "reviewer_id": review["reviewer_id"]})
    from transformers import AutoTokenizer
    import xgrammar as xgr
    endpoint = read(diagnostic.ENDPOINT)
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    staged, sources = [], {}
    for candidate in admitted:
        source = read(candidate["source_binding"]["path"])
        sources[candidate["source_binding"]["path"]] = candidate["source_binding"]
        refs = [t["turn_ref"] for t in source["turns"]]
        assert source["root_id"] == candidate["root_id"] and len(refs) == len(set(refs))
        for variant, public, private in displays(candidate):
            field = "options" if candidate["question_type"] == "multi_select" else "items"
            for condition in CONDITIONS:
                row = {k: v for k, v in public.items() if k != field}
                row.update(display=public[field], labels=list(public[field]), variant_id=variant,
                    cohort="new", condition=condition, sampling_seed=sampling_seed(candidate["candidate_id"]),
                    request_id=f"{candidate['candidate_id']}__{variant}__{condition}__B1_stability_v1")
                prompt = diagnostic.make_prompt(tokenizer, row, row["display"], source) if condition == "FULL" else base.question_only_prompt(tokenizer, row)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                assert len(tokens) + MAX_TOKENS <= MODEL_LEN, "CONTEXT_OVERFLOW_NO_TRUNCATION:" + row["request_id"]
                schema = diagnostic.repair.schema(row["labels"], refs, row["question_type"]) if condition == "FULL" else base.question_only_schema(row["labels"], row["question_type"])
                compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
                scoring = {"request_id": row["request_id"], "candidate_id": candidate["candidate_id"],
                    "cohort": "new", "parent_candidate_id": None, "strictly_paired": False,
                    "variant_id": variant, "presentation": private}
                staged.append((row, prompt, tokens, schema, scoring))
    assert len(staged) == len(admitted) * 4 <= 112 and len({r[0]["request_id"] for r in staged}) == len(staged)
    OUT.mkdir()
    paths = (Path(__file__), BASE_PATH, HERE / "PROTOCOL_ZH.md", SELECTION, base.GATE_PATH,
        base.DIAGNOSTIC_PATH, base.CAPTURE_PATH, diagnostic.REPAIR_PATH, diagnostic.ENDPOINT)
    bindings = [bind(p) for p in paths] + list(sources.values())
    rows, token_ids, private_rows = [], {}, []
    for row, prompt, tokens, schema, scoring in staged:
        rid = row["request_id"]
        pp, sp = OUT / "prompts" / (rid + ".txt"), OUT / "schemas" / (rid + ".json")
        pp.parent.mkdir(exist_ok=True)
        with pp.open("x", encoding="utf-8") as stream:
            stream.write(prompt)
        save(sp, schema)
        row.update(prompt_path=str(pp), schema_path=str(sp), prompt_tokens=len(tokens))
        assert set(row) == PUBLIC_KEYS
        rows.append(row); token_ids[rid] = tokens; private_rows.append(scoring)
        bindings.extend((bind(pp), bind(sp)))
    save(OUT / "PUBLIC_REQUESTS.json", {"requests": rows})
    save(OUT / "PROMPT_TOKEN_IDS.json", token_ids)
    bindings.extend(bind(OUT / name) for name in ("PUBLIC_REQUESTS.json", "PROMPT_TOKEN_IDS.json"))
    coverage = {rid: gate.coverage_report([c for c in admitted if c["root_id"] == rid]) for rid in roots}
    private_path = OUT / "PRIVATE_SCORING.json"
    save(private_path, {"requests": private_rows, "bindings": [bind(NEW), bind(REVIEW)],
        "public_requests_binding": bind(OUT / "PUBLIC_REQUESTS.json"), "prompt_token_ids_binding": bind(OUT / "PROMPT_TOKEN_IDS.json"),
        "new_admission": ledger, "new_mc_coverage": gate.coverage_report(admitted, minimum_per_count=4),
        "coverage_by_root": coverage, "planned_questions": {"new": 28}, "submitted_new_drafts": len(candidates)})
    private_path.chmod(0o600)
    save(OUT / "PREPARATION.json", {"requests": len(rows), "model_role": "B1", "bindings": bindings,
        "private_scoring_binding": bind(private_path), "max_output_tokens": MAX_TOKENS, "max_model_len": MODEL_LEN,
        "temperature": .8, "top_p": .9, "display_seed": DISPLAY_SEED, "variants": list(VARIANTS),
        "conditions": list(CONDITIONS), "fixed_order": [r["request_id"] for r in rows],
        "automatic_retry": False, "source_truncation": False, "native_hidden_thinking": False,
        "independent_requests_no_shared_history_or_memory": True, "generation_reads_private_scoring": False,
        "historical_scores_reusable": False, "namespace": "train_only_crossroot_question_quality_stability"})
    print(json.dumps({"prepared": len(rows), "questions": len(admitted), "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
        "coverage_by_root": {rid: c["coverage_complete"] for rid, c in coverage.items()}}))


def public_preparation():
    prep = read(OUT / "PREPARATION.json")
    private_paths = {NEW.resolve(), REVIEW.resolve(), (OUT / "PRIVATE_SCORING.json").resolve()}
    for bound in prep["bindings"]:
        assert Path(bound["path"]).resolve() not in private_paths, "PRIVATE_INPUT_IN_WORKER_BINDINGS"
        check(bound)
    rows, tokens = read(OUT / "PUBLIC_REQUESTS.json")["requests"], read(OUT / "PROMPT_TOKEN_IDS.json")
    ids = [r["request_id"] for r in rows]
    assert 0 < len(rows) == prep["requests"] == len(set(ids)) <= 112
    assert ids == prep["fixed_order"] and set(ids) == set(tokens)
    required = [OUT / "PUBLIC_REQUESTS.json", OUT / "PROMPT_TOKEN_IDS.json", Path(__file__), BASE_PATH]
    for row in rows:
        assert set(row) == PUBLIC_KEYS and row["condition"] in CONDITIONS and row["variant_id"] in VARIANTS
        assert row["cohort"] == "new" and row["labels"] == list(row["display"])
        assert len(tokens[row["request_id"]]) == row["prompt_tokens"]
        assert row["prompt_tokens"] + MAX_TOKENS <= MODEL_LEN
        required.extend(Path(row[k]) for k in ("prompt_path", "schema_path"))
    assert all(bind(path) in prep["bindings"] for path in required)
    return prep, rows, tokens


original_score = base.score_record


def score_record(row, presentation, answer, valid):
    result = original_score(row, presentation, answer, valid)
    result["variant_id"] = row["variant_id"]
    return result


def summarize(scores, private):
    cells = [{"condition": c, "variant_id": v, **base.aggregate([s for s in scores if s["condition"] == c and s["variant_id"] == v])}
        for c in CONDITIONS for v in VARIANTS]
    conditions = [{"condition": c, **base.aggregate([s for s in scores if s["condition"] == c])} for c in CONDITIONS]
    buckets = [{"condition": c, "gold_count": count, **base.aggregate([s for s in scores if s["condition"] == c and s["gold_count"] == count])}
        for c in CONDITIONS for count in (2, 3, 4, 5, 6)]
    root_cells = [{"root_id": root, "condition": c, **base.aggregate([s for s in scores if s["root_id"] == root and s["condition"] == c])}
        for root in sorted({s["root_id"] for s in scores}) for c in CONDITIONS]
    variant_pairs = []
    for cid in sorted({s["candidate_id"] for s in scores}):
        for condition in CONDITIONS:
            pair = sorted([s for s in scores if s["candidate_id"] == cid and s["condition"] == condition], key=lambda s:s["variant_id"])
            assert len(pair) == 2
            a, b = pair
            canonical_same = a["valid"] and b["valid"] and (set(a["prediction"]) == set(b["prediction"]) if a["question_type"] == "multi_select" else a["prediction"] == b["prediction"])
            variant_pairs.append({"candidate_id": cid, "condition": condition, "question_type": a["question_type"],
                "both_valid": a["valid"] and b["valid"], "canonical_prediction_same": canonical_same,
                "both_exact": a["exact"] and b["exact"], "neither_exact": not a["exact"] and not b["exact"],
                "accuracy_changed": a["exact"] != b["exact"], "first_prediction": a["prediction"], "second_prediction": b["prediction"]})
    first_pass = sum(r["initial_decision"] == "PASS" for r in private["new_admission"])
    semantic_pass = sum(r["initial_content_decision"] == "PASS" for r in private["new_admission"])
    return {"requests": len(scores), "valid": sum(s["valid"] for s in scores), "correct": sum(s["exact"] for s in scores),
        "planned_questions": 28, "admitted_questions": len(private["new_admission"]) - sum(r["decision"] != "PASS" for r in private["new_admission"]),
        "initial_content_pass": first_pass, "initial_content_pass_rate": first_pass / 28,
        "initial_semantic_content_pass": semantic_pass, "initial_semantic_content_pass_rate": semantic_pass / 28,
        "initial_content_stability_threshold": .9, "initial_content_stability_threshold_met": first_pass / 28 >= .9,
        "coverage_by_root": private["coverage_by_root"], "new_mc_coverage": private["new_mc_coverage"],
        "cells": cells, "conditions": conditions, "new_mc_gold_buckets": buckets, "root_cells": root_cells,
        "variant_pairs": variant_pairs, "new_admission": private["new_admission"],
        "question_only_both_variants_correct": [p["candidate_id"] for p in variant_pairs if p["condition"] == "QUESTION_ONLY" and p["both_exact"]],
        "all_frozen_requests_in_exact_denominator": True, "invalid_mc_prediction_for_descriptive_fp_fn": "empty_set",
        "semantic_reason_review": "NOT_AUTOMATICALLY_CERTIFIED", "training_admission": False, "feedback_eligible": False,
        "historical_scores_reused": False, "new_dialogue_turns": 0,
        "inference_limits": "Four selected Train roots, two stochastic permutations each. Separately assess mechanical stability, content yield and possible shortcuts; no population or causal feedback claim."}


# Only the imported instance is adapted; files and functions in the frozen pilot
# are unchanged. All reused functions resolve these globals in that instance.
base.OUT = OUT
base.public_preparation = public_preparation
base.score_record = score_record
base.summarize = summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": lambda: base.run(args.execute_gpu), "analyze": base.analyze}[args.action]()
