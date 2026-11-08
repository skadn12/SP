"""Synthetic, CPU-only tests. Never opens author questions or private answers."""
from __future__ import annotations
import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import adapter as a
import common as c
import execute
import prepare
import score


def synthetic(typ, n):
    labels = a.labels(typ, n)
    field = "options" if typ == "multi_select" else "items"
    return {"candidate_id": "synthetic-" + typ + str(n), "root_id": "synthetic-root", "question_type": typ,
        "question": "Synthetic format test.", field: {label: f"Distinct synthetic item {i}" for i, label in enumerate(labels)},
        "source_path": "/synthetic/source.json", "source_sha256": "0" * 64}


def integration(check):
    # Synthetic only: real tokenizer/prompt/decoder, no model engine. Model
    # bindings are omitted for this temporary fixture, never for production.
    with tempfile.TemporaryDirectory(prefix="crossroot-runtime-selftest-") as directory:
        base = Path(directory)
        source_paths = {}
        for r in range(4):
            root = f"fixture_root_{r}"
            path = base / (root + ".json")
            c.save(path, {"root_id": root, "turns": [{"turn_ref": "S1:T1", "speaker": "Alpha", "text": "Synthetic integration statement."}]})
            source_paths[root] = path
        questions, private_questions, slots = [], [], []
        admitted = [("multi_select", 6, 0), ("ordering", 3, 1), ("ordering", 4, 1), ("ordering", 5, 2), ("ordering", 6, 2)]
        for typ, n, r in admitted:
            q = synthetic(typ, n)
            q.update(root_id=f"fixture_root_{r}", source_path=str(source_paths[f"fixture_root_{r}"]), source_sha256=c.sha(source_paths[f"fixture_root_{r}"]))
            questions.append(q)
            field = "options" if typ == "multi_select" else "items"
            private_questions.append(dict(q, gold=["A", "C", "F"] if typ == "multi_select" else list(reversed(q[field]))))
        for r in range(4):
            subset = [q for q in questions if q["root_id"] == f"fixture_root_{r}"]
            for s in range(6):
                q = subset[s] if s < len(subset) else None
                slots.append({"slot_id": f"fixture_slot_{r}_{s}", "candidate_id": q["candidate_id"] if q else f"unadmitted_{r}_{s}",
                    "root_id": f"fixture_root_{r}", "question_type": q["question_type"] if q else "ordering", "item_count": len(q.get("options", q.get("items"))) if q else 3,
                    "source": c.bind(source_paths[f"fixture_root_{r}"])})
        public_path, private_path, protocol_path, admission_path = [base / name for name in ("public.json", "private.json", "protocol.json", "admission.json")]
        c.save(public_path, {"questions": questions})
        c.save(private_path, {"questions": private_questions})
        c.save(protocol_path, {"solver": dict(c.SOLVER_FIXED, display_seed=2026102101, sampling_seed=300000),
            "budget": {"initial_question_slots": 5, "production_planned_slots": 24, "max_direct_requests": 60}, "slots": slots})
        c.save(admission_path, {"status": "PASS", "namespace": "train.feedback/synthetic_cpu_selftest", "questions": [{"candidate_id": q["candidate_id"]} for q in questions],
            "public_questions": c.bind(public_path), "private_questions": c.bind(private_path), "planned_slots": 24, "admitted_count": 5})
        run, sidecar = base / "run", base / "sidecar"
        original_models = c.rt.models
        try:
            c.rt.models = lambda: []
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare(SimpleNamespace(run=run, private_output=sidecar, public=public_path, private=private_path, protocol=protocol_path, admission=admission_path))
        finally:
            c.rt.models = original_models
        manifest = c.verify(run, check_models=False)
        check("synthetic_prepare_verify_60_fixed_calls_24_slots_4_planned_3_admitted_roots", manifest["request_count"] == 60 and manifest["production_denominator"] == 24 and manifest["root_count"] == 3 and manifest["planned_root_count"] == 4)
        check("synthetic_private_copy_exact_frozen_bytes", private_path.read_bytes() == (sidecar / "PRIVATE_QUESTIONS.json").read_bytes())
        # Rebinding checksums in this disposable fixture must not hide an
        # invalid seed schedule from the independent reconstruction.
        original_manifest = (run / "manifest.json").read_bytes()
        original_freeze = (run / "PREFLIGHT_FREEZE.json").read_bytes()
        altered = json.loads(original_manifest)
        altered["requests"][0]["seed"] += 1000000
        (run / "manifest.json").write_text(json.dumps(altered))
        rebound = json.loads(original_freeze)
        rebound["files"] = [c.bind(item["path"]) for item in rebound["files"]]
        (run / "PREFLIGHT_FREEZE.json").write_text(json.dumps(rebound))
        try:
            c.verify(run, check_models=False)
        except RuntimeError as error:
            check("synthetic_rebound_hashes_do_not_hide_wrong_seed_schedule", "DETERMINISTIC_REQUEST_SCHEDULE_DRIFT" in str(error))
        else:
            raise AssertionError("invalid seed accepted")
        (run / "manifest.json").write_bytes(original_manifest)
        (run / "PREFLIGHT_FREEZE.json").write_bytes(original_freeze)
        try:
            score.score(run, sidecar)
        except FileNotFoundError:
            check("synthetic_pre_raw_score_gate_no_gold_start", not (run / "SCORING_STARTED.json").exists())
        else:
            raise AssertionError("pre-raw scoring accepted")
        for row in manifest["requests"]:
            c.save(run / "responses" / (row["request_id"] + ".json"), execute.terminal(row, "NOT_SUBMITTED", usage_status="NO_REQUEST_SENT"))
        c.save(run / "FAILED.json", {"status": "SYNTHETIC_NOT_SUBMITTED", "wall_seconds": 0, "model_load_wall_seconds": 0})
        c.freeze_raw(run, manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            score.score(run, sidecar)
        scored = c.read(run / "SCORES.json")
        check("synthetic_all_invalid_fixed_denominator_60_calls_5_questions_24_slots", scored["denominator_calls"] == 60 and scored["denominator_questions"] == 5 and len(scored["planned_slot_outcomes"]) == 24 and scored["valid_calls"] == scored["exact_calls"] == 0)
        check("synthetic_invalid_all_ordering_pairs_zero", all(record["metrics"]["correct_pairs"] == 0 for record in scored["requests"] if record["question_type"] == "ordering"))
        check("synthetic_order3_and_order6_arity_reports", scored["by_arity"]["ordering:3"]["calls"] == 6 and scored["by_arity"]["ordering:6"]["calls"] == 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=c.HERE / "CPU_SELFTEST.json")
    args = parser.parse_args()
    c.environment()
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    for n in (3, 4, 5, 6):
        q = synthetic("ordering", n)
        a.public_question(q)
        variants = a.variants(q, 2026102101, a.mapping_count(q))
        check(f"order{n}_unique_display_count", len({tuple(v["display_order_content_ids"]) for v in variants}) == min(12, __import__('math').factorial(n)))
        choices = a.choices("ordering", q["items"])
        check(f"order{n}_finite_choice_count", len(choices) == __import__('math').factorial(n))
        canonical = a.content_ids(q)
        gold = list(reversed(q["items"]))
        for variant in variants:
            inverse = {content: label for label, content in variant["display_to_content"].items()}
            displayed = [inverse[canonical[label]] for label in gold]
            row = {"request_id": q["candidate_id"] + variant["variant_id"], "candidate_id": q["candidate_id"],
                "root_id": q["root_id"], "question_type": "ordering", "variant_id": variant["variant_id"],
                "slot": 0, "seed": 7, "display": {"items": variant["items"]}}
            raw = {"status": "RETURNED", "raw_output": "<box>" + ",".join(displayed) + "</box>",
                "metadata": {"finish_reason": "stop"}, "response_committed_at_utc": c.now()}
            scored = score.score_one(row, raw, variant, canonical, gold)
            check(f"order{n}_{variant['variant_id']}_gold_remap", scored["valid"] and scored["exact"] and scored["metrics"]["total_pairs"] == n*(n-1)//2 and scored["metrics"]["correct_pairs"] == n*(n-1)//2)
            invalid = score.score_one(row, dict(raw, raw_output="<box>" + ",".join(displayed[:-1]) + "</box>"), variant, canonical, gold)
            check(f"order{n}_{variant['variant_id']}_incomplete_fixed_zero", not invalid["valid"] and not invalid["exact"] and invalid["metrics"]["correct_pairs"] == 0 and invalid["metrics"]["total_pairs"] == n*(n-1)//2)
        check(f"order{n}_deterministic_gold_blind_shuffle", variants == a.variants(q, 2026102101, a.mapping_count(q)))
    q = synthetic("multi_select", 6)
    check("MC6_63_legal_nonempty_sets", len(a.choices("multi_select", q["options"])) == 63)
    check("MC6_18_planned_calls", a.request_count(q) == 18)
    canonical = a.content_ids(q)
    for variant in a.variants(q, 2026102101, 6):
        inverse = {content: label for label, content in variant["display_to_content"].items()}
        gold = ["A", "C", "F"]
        labels = sorted(inverse[canonical[label]] for label in gold)
        row = {"request_id": "syntheticMC"+variant["variant_id"], "candidate_id": q["candidate_id"], "root_id": q["root_id"],
            "question_type": "multi_select", "variant_id": variant["variant_id"], "slot": 0, "seed": 7, "display": {"options": variant["options"]}}
        raw = {"status": "RETURNED", "raw_output": "<box>" + ",".join(labels) + "</box>", "metadata": {"finish_reason": "stop"}, "response_committed_at_utc": c.now()}
        result = score.score_one(row, raw, variant, canonical, gold)
        check(variant["variant_id"]+"_MC_gold_remap", result["valid"] and result["exact"] and result["metrics"]["tp"] == 3 and result["metrics"]["fp"] == result["metrics"]["fn"] == 0)
        invalid = score.score_one(row, dict(raw, status="BACKEND_ERROR"), variant, canonical, gold)
        check(variant["variant_id"]+"_MC_invalid_unknown_FP_FN", not invalid["exact"] and invalid["metrics"]["fp"] is invalid["metrics"]["fn"] is None and invalid["metrics"]["jaccard"] == 0)
    # Native structured params compile/verify on CPU for every arity; no LLM load.
    _, solver = a.native()
    decoder = solver.generation_config("multi_select")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    source = {"turns": [{"turn_ref": "S1:T1", "speaker": "Alpha", "text": "Synthetic source statement."}]}
    for typ, n in [("multi_select", 6), ("ordering", 3), ("ordering", 4), ("ordering", 5), ("ordering", 6)]:
        q = synthetic(typ, n)
        field = "options" if typ == "multi_select" else "items"
        variant = a.variants(q, 1, 1)[0]
        row = {"question_type": typ, "branch": "direct", "display": {field: variant[field]}, "seed": 33}
        params = a.sampling_params(row, decoder["structured_output_api"])
        check(f"{typ}{n}_native_params_CPU", params.max_tokens == 32 and params.temperature == .8 and params.top_p == .9 and params.seed == 33)
        prompt = a.direct_prompt(tokenizer, q, variant, source)
        check(f"{typ}{n}_full_source_prompt", "[S1:T1] Alpha: Synthetic source statement." in prompt and all(text in prompt for text in q[field].values()))
        check(f"{typ}{n}_format_tokens_within32", max(len(tokenizer.encode(choice, add_special_tokens=False)) for choice in a.choices(typ, q[field])) <= 32)
        if n >= 4:
            check(f"{typ}{n}_native_finite_vocabulary_unchanged", a.choices(typ, q[field]) == a.old.choices(typ, q[field]))
    integration(check)
    result = {"status": "PASS", "created_at_utc": c.now(), "synthetic_only": True, "inference_calls": 0,
        "author_private_gold_json_read": False, "synthetic_gold_json_read_only_after_synthetic_raw_freeze": True,
        "fixture_model_binding_check": "omitted for disposable CPU fixture; production prepare/verify require actual checkpoint bindings",
        "checks": len(checks), "tests": checks, "versions": c.rt.versions(),
        "code_bindings": [c.bind(path) for path in sorted(c.HERE.glob("*.py"))]}
    c.save(args.report, result)
    print(json.dumps({"status": "PASS", "checks": len(checks), "inference_calls": 0}))


if __name__ == "__main__":
    main()
