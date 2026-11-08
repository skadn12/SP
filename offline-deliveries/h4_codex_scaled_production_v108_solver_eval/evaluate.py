"""Bind sealed v108 to the unchanged v15 full-only paired Solver protocol.

The local protocol copy changes only its generator guard and contract path.
No generation, decoding, permutation, scoring, or selection code is replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BATCH = HERE.parent / "h4_codex_scaled_production_v108"
NATIVE = HERE.parent / "h4_feedback_training_pool_v15_qa/evaluation"
MODELS = {"trained": "M_clean_policy", "base": "M0"}
PACKAGES = ("torch", "vllm", "transformers", "tokenizers", "xgrammar")


def now():
    return datetime.now(timezone.utc).isoformat()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def evaluation_counts(accounting):
    """Read only the predeclared sealed count contract; never infer a target."""
    raw = accounting.get("evaluation_counts")
    need(isinstance(raw, dict), "EVALUATION_COUNTS_REQUIRED")
    need(set(raw) == {"candidate_count", "type_counts", "multi_gold_counts", "calls_per_model"},
         "EVALUATION_COUNT_FIELDS_INVALID")
    total, calls = raw["candidate_count"], raw["calls_per_model"]
    need(type(total) is int and total > 0 and type(calls) is int and calls > 0,
         "EVALUATION_TOTAL_OR_CALL_COUNT_INVALID")
    types, gold = raw["type_counts"], raw["multi_gold_counts"]
    need(isinstance(types, dict) and set(types) <= {"multi_select", "ordering"}
         and all(type(v) is int and v >= 0 for v in types.values()), "TYPE_COUNT_CONTRACT_INVALID")
    need(isinstance(gold, dict) and set(gold) <= set("23456")
         and all(type(v) is int and v >= 0 for v in gold.values()), "GOLD_COUNT_CONTRACT_INVALID")
    normalized_types = {key: types.get(key, 0) for key in ("multi_select", "ordering")}
    normalized_gold = {key: gold[key] for key in sorted(gold) if gold[key]}
    need(sum(normalized_types.values()) == total
         and sum(normalized_gold.values()) == normalized_types["multi_select"],
         "EVALUATION_COUNT_SUM_MISMATCH")
    need(calls == 18 * normalized_types["multi_select"] + 12 * normalized_types["ordering"],
         "EVALUATION_CALL_FORMULA_MISMATCH")
    return {"candidate_count": total, "type_counts": normalized_types,
            "multi_gold_counts": normalized_gold, "calls_per_model": calls}


def observed_counts(questions):
    types = Counter(q["question_type"] for q in questions)
    return {"candidate_count": len(questions),
            "type_counts": {key: types.get(key, 0) for key in ("multi_select", "ordering")},
            "multi_gold_counts": dict(sorted(Counter(str(len(q["gold"])) for q in questions
                                                     if q["question_type"] == "multi_select").items())),
            "calls_per_model": sum(18 if q["question_type"] == "multi_select" else 12 for q in questions)}


def enforce_budget(counts, plan):
    cap = plan.get("solver_calls_per_model_max")
    need(type(cap) is int and cap > 0 and counts["calls_per_model"] <= cap,
         "PRECALL_SOLVER_CALL_BUDGET_EXCEEDED_OR_INVALID")
    need(counts["candidate_count"] <= plan["planned_new_questions_max"] + plan["planned_controls_max"],
         "PRECALL_CANDIDATE_BUDGET_EXCEEDED")


def content_accounting(accounting, total):
    passed = accounting.get("new_questions", -1) + accounting.get("gold6_controls", -1)
    excluded = accounting.get("excluded_final_candidates")
    need(accounting.get("status") == "CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_VALIDATED"
         and type(excluded) is int and excluded >= 0 and passed == total
         and accounting.get("public_candidates") == passed + excluded,
         "BATCH_PASS_EXCLUDED_ACCOUNTING_MISMATCH")


def exposure_contract():
    frozen = read(BATCH / "EXECUTION_INPUTS_FROZEN.json")
    for key, name in (("precall_plan", "PRECALL_PLAN.json"), ("run_config", "RUN_CONFIG.json"),
                      ("content_contract", "evidence/CONTENT_CONTRACT.md"),
                      ("source_reservations", "SOURCE_RESERVATIONS.json")):
        need(frozen[key]["path"] == str(BATCH / name), "EXECUTION_INPUT_BINDING_PATH_MISMATCH")
        verify([frozen[key]])
    audit = read(BATCH / "EXPOSURE_AUDIT.json")
    plan = read(BATCH / "PRECALL_PLAN.json")
    selected = [r["root_id"] for g in plan["groups"] for r in g["roots"]]
    need(audit.get("status") == "PASS" and audit.get("purpose") == "TRAINING_SIDE_DATA_PRODUCTION"
         and audit.get("selected_root_ids") == selected and len(selected) == 12,
         "FIXED_TRAINING_ROOT_CONTRACT_INVALID")
    need(audit.get("first_round_training_exposure") is True
         and audit.get("v106_validation_reused_for_training") is False,
         "TRAINING_EXPOSURE_NOT_DISCLOSED")
    for key in ("holdout_root_overlap", "holdout_source_id_overlap", "holdout_source_record_overlap"):
        need(audit.get(key) == [], "HOLDOUT_OVERLAP_DECLARED:" + key)
    identity = read(BATCH / "evidence/SOURCE_IDENTITY_BINDINGS.json")
    selection = identity["selection_manifest"]
    need(Path(selection["path"]) == PROJECT / "SP-820-publish/first_training_pipeline/data/selection_manifest_h4_unexpanded122.json",
         "ORIGINAL_SELECTION_BINDING_PATH_INVALID")
    verify([selection, audit["clean_training_manifest"], *audit["holdout_manifests"], *identity["inputs"]])
    training = {json.loads(line)["root_id"] for line in Path(audit["clean_training_manifest"]["path"]).read_text().splitlines()}
    holdout = {json.loads(line)["root_id"] for row in audit["holdout_manifests"]
               for line in Path(row["path"]).read_text().splitlines()}
    need(set(selected) <= training and not set(selected) & holdout, "TRAINING_HOLDOUT_ROOT_ISOLATION_INVALID")
    index = {r["root_id"]: r for r in read(selection["path"])["records"]}
    bound = {r["root_id"]: r for r in identity["inputs"]}
    need(len(bound) == len(identity["inputs"]) and set(bound) == set(selected) | holdout
         and set(identity["selected_root_ids"]) == set(selected)
         and set(identity["holdout_root_ids"]) == holdout, "IDENTITY_BINDING_ROOT_SET_INVALID")
    originals = {}
    for rid, row in bound.items():
        need(Path(row["path"]) == Path(index[rid]["input_path"]).resolve(), "ORIGINAL_INPUT_BINDING_PATH_MISMATCH")
        originals[rid] = read(row["path"])["source"]
    declared = {r["root_id"]: r for r in audit["selected_roots"]}
    for rid in selected:
        need(originals[rid]["source_id"] == declared[rid]["original_source_id"]
             and originals[rid]["source_record_sha256"] == declared[rid]["original_source_record_sha256"],
             "ORIGINAL_SOURCE_IDENTITY_MISMATCH")
    for key in ("source_id", "source_record_sha256"):
        need(not {originals[r][key] for r in selected} & {originals[r][key] for r in holdout},
             "ORIGINAL_SOURCE_HOLDOUT_OVERLAP:" + key)
    reservations = read(BATCH / "SOURCE_RESERVATIONS.json")["allocation"]
    fixed_sources = {}
    for row in reservations:
        need(sha(row["baseline_path"]) == row["baseline_sha256"], "FIXED_BASELINE_CHANGED")
        folder = BATCH / "roots" / ("root" + row["root_id"].rsplit("-", 1)[1])
        if not (folder / "CUMULATIVE_SOURCE.json").exists():
            continue
        source = read(folder / "CUMULATIVE_SOURCE.json")
        baseline = read(row["baseline_path"])["turns"]
        need(source["turns"][:len(baseline)] == baseline, "HISTORICAL_PREFIX_CHANGED")
        need(12 <= len(source["turns"]) - len(baseline) <= 16, "EXPANSION_LENGTH_CHANGED")
        fixed_sources[row["root_id"]] = source["turns"]
    return fixed_sources


def self_test():
    good = {"evaluation_counts": {"candidate_count": 3, "type_counts": {"multi_select": 2, "ordering": 1},
            "multi_gold_counts": {"2": 1, "3": 1, "6": 0}, "calls_per_model": 48}}
    expected = evaluation_counts(good)
    need(expected["multi_gold_counts"] == {"2": 1, "3": 1}, "SELF_TEST_ZERO_NORMALIZATION")
    need(evaluation_counts({"evaluation_counts": {"candidate_count": 1, "type_counts": {"ordering": 1},
         "multi_gold_counts": {}, "calls_per_model": 12}})["type_counts"]["multi_select"] == 0,
         "SELF_TEST_MISSING_ZERO_TYPE")
    rows = [{"question_type": "multi_select", "gold": ["A", "B"]},
            {"question_type": "multi_select", "gold": ["A", "B", "C"]},
            {"question_type": "ordering", "gold": [f"I{i}" for i in range(1, 7)]}]
    need(observed_counts(rows) == expected, "SELF_TEST_OBSERVED_COUNTS")
    for key, replacement in (("candidate_count", True), ("candidate_count", 4), ("calls_per_model", 49),
                             ("multi_gold_counts", {"1": 2}), ("type_counts", {"unknown": 3})):
        altered = {**good["evaluation_counts"], key: replacement}
        try:
            evaluation_counts({"evaluation_counts": altered})
        except ValueError:
            pass
        else:
            raise AssertionError("INVALID_COUNT_CONTRACT_ACCEPTED:" + key)
    plan = {"solver_calls_per_model_max": 48, "planned_new_questions_max": 3, "planned_controls_max": 0}
    enforce_budget(expected, plan)
    for changed in ({**plan, "solver_calls_per_model_max": 47}, {**plan, "planned_new_questions_max": 2}):
        try:
            enforce_budget(expected, changed)
        except ValueError:
            pass
        else:
            raise AssertionError("OVER_BUDGET_ACCEPTED")
    content_accounting({"status": "CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_VALIDATED",
        "new_questions": 2, "gold6_controls": 1, "public_candidates": 4, "excluded_final_candidates": 1}, 3)
    print(json.dumps({"self_test": "PASS", "cases": 12, "model_calls": 0, "file_writes": 0}))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(rows):
    for row in rows:
        need(sha(row["path"]) == row["sha256"], "FROZEN_FILE_CHANGED:" + row["path"])


def sealed_files():
    manifest = read(BATCH / "COMPLETE_MANIFEST.json")
    need(manifest["status"] == "SEALED_CONTENT_ONLY", "SOURCE_NOT_SEALED")
    paths = [row["path"] for row in manifest["files"]]
    required = {"RUN_CONFIG.json", "FINAL_BATCH_ACCOUNTING.json", "EXPOSURE_AUDIT.json",
                "PRECALL_PLAN.json", "EXECUTION_INPUTS_FROZEN.json", "EVALUATION_COUNTS.json", "evidence/SOURCE_IDENTITY_BINDINGS.json",
                "NEW_PUBLIC_QUESTIONS.json", "NEW_PRIVATE_GOLD.json",
                "CONTROL_PUBLIC_QUESTIONS.json", "CONTROL_PRIVATE_GOLD.json"}
    need(len(paths) == len(set(paths)) and required <= set(paths), "SOURCE_SEAL_FILE_SET_INVALID")
    rows = [bind(BATCH / "COMPLETE_MANIFEST.json")]
    for row in manifest["files"]:
        path = BATCH / row["path"]
        need(path.is_relative_to(BATCH) and path.stat().st_size == row["bytes"]
             and sha(path) == row["sha256"], "SEALED_CONTENT_CHANGED:" + str(path))
        rows.append({"path": str(path), "sha256": row["sha256"]})
    return rows


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def native():
    sys.path.insert(0, str(NATIVE))
    result = module("v108_unchanged_native_runner", NATIVE / "run_solver.py")
    local = module("v108_adapter_protocol", HERE / "adapter_protocol.py")
    result.validate_pool = local.validate_pool
    return result


def prepare():
    need(not (HERE / "ADAPTER_BINDINGS.json").exists(), "ALREADY_PREPARED_NO_OVERWRITE")
    frozen = sealed_files()
    accounting = read(BATCH / "FINAL_BATCH_ACCOUNTING.json")
    expected = evaluation_counts(read(BATCH / "EVALUATION_COUNTS.json"))
    enforce_budget(expected, read(BATCH / "PRECALL_PLAN.json"))
    roots = exposure_contract()
    content_accounting(accounting, expected["candidate_count"])
    config = read(BATCH / "RUN_CONFIG.json")
    need(config["backend"] == "CURRENT_CODEX_SESSION" and config["exact_model_id"] is None
         and config["batch_id"] == BATCH.name,
         "GENERATOR_PROVENANCE_CHANGED")
    original = (NATIVE / "protocol.py").read_text(encoding="utf-8")
    old_guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    new_guard = ('if (provenance.get("backend") != "CURRENT_CODEX_SESSION"\n'
                 '            or provenance.get("exact_model_id") is not None\n'
                 '            or provenance.get("source_batch_id") != "h4_codex_scaled_production_v108"):')
    old_path = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    new_path = "CONTRACT_DIR = Path(" + repr(str(NATIVE.parent)) + ")"
    need(original.count(old_guard) == original.count(old_path) == 1, "PROTOCOL_ADAPTER_PATCH_DRIFT")
    adapted = original.replace(old_guard, new_guard).replace(old_path, new_path)
    with (HERE / "adapter_protocol.py").open("x", encoding="utf-8") as stream:
        stream.write(adapted)
    public = read(BATCH / "NEW_PUBLIC_QUESTIONS.json")["questions"] + read(
        BATCH / "CONTROL_PUBLIC_QUESTIONS.json")["questions"]
    private = read(BATCH / "NEW_PRIVATE_GOLD.json")["questions"] + read(
        BATCH / "CONTROL_PRIVATE_GOLD.json")["questions"]
    public_ids, private_ids = [q["candidate_id"] for q in public], [q["candidate_id"] for q in private]
    need(len(public_ids) == len(set(public_ids)) == len(private_ids) == len(set(private_ids)) == expected["candidate_count"]
         and set(public_ids) == set(private_ids), "PUBLIC_PRIVATE_CANDIDATE_SET_MISMATCH")
    gold = {q["candidate_id"]: q for q in private}
    questions, receipt = [], []
    for q in public:
        p = gold[q["candidate_id"]]
        need(q["root_id"] in roots, "ROOT_NOT_IN_FIXED_SELECTION")
        source_path = (PROJECT / q["source_path"]).resolve(strict=True)
        need(source_path.is_relative_to(BATCH) and sha(source_path) == q["source_sha256"], "SOURCE_CHANGED")
        source = read(source_path)
        need(source["root_id"] == q["root_id"], "SOURCE_ROOT_MISMATCH")
        need(source["turns"] == roots[q["root_id"]], "SOURCE_CHANGED_FROM_FIXED_HISTORICAL_DIALOGUE")
        refs = {t["turn_ref"]: t for t in source["turns"]}
        need(len(refs) == len(source["turns"]), "DUPLICATE_SOURCE_TURN")
        root_path = source_path.parent
        root_public = read(root_path / "PUBLIC_QUESTIONS.json")["questions"]
        need(q in root_public, "BATCH_ROOT_PUBLIC_MISMATCH")
        review = read(root_path / "FINAL_CROSSCHECK.json")
        matches = [r for r in review["results"] if r["candidate_id"] == q["candidate_id"]]
        need(len(matches) == 1 and matches[0]["status"] == "CONTENT_PASS"
             and matches[0]["source_derived_answer"] == p["gold"], "FINAL_REVIEW_GOLD_MISMATCH")
        labels = tuple("ABCDEF") if q["question_type"] == "multi_select" else tuple(f"I{i}" for i in range(1, 7))
        need(set(p["evidence"]) == set(labels), "INCOMPLETE_PRIVATE_EVIDENCE")
        evidence = {}
        for label in labels:
            citations = p["evidence"][label]
            citations = citations if isinstance(citations, list) else [citations]
            need(bool(citations), "EMPTY_CITATIONS")
            for citation in citations:
                turn = refs.get(citation.get("turn_ref"))
                need(turn is not None and citation["quote"] and citation["quote"] in turn["text"], "NONVERBATIM_CITATION")
                need(citation.get("speaker", turn["speaker"]) == turn["speaker"], "CITATION_SPEAKER_MISMATCH")
            evidence[label] = {**citations[0], "speaker": refs[citations[0]["turn_ref"]]["speaker"]}
            if q["question_type"] == "multi_select":
                evidence[label].update(relation="supports" if label in p["gold"] else "contradicts",
                                       reason="Relation copied from sealed private Gold and CONTENT_PASS review; no new semantic judgment.")
        normalized = {**q, "source_path": str(source_path), "gold": p["gold"], "role": p["role"]}
        key = "option_evidence" if q["question_type"] == "multi_select" else "item_evidence"
        normalized[key] = evidence
        if q["question_type"] == "ordering":
            normalized["order_anchor_spans"] = evidence
        questions.append(normalized)
        receipt.append({"candidate_id": q["candidate_id"], "public_unchanged_except_absolute_source_path": True,
                        "original_private": p, "source": bind(source_path),
                        "final_crosscheck": bind(root_path / "FINAL_CROSSCHECK.json")})
    need(observed_counts(questions) == expected, "SEALED_EVALUATION_COUNTS_MISMATCH")
    provenance = {"backend": config["backend"], "exact_model_id": None,
                  "source_batch_id": config["batch_id"], "source_batch_manifest": bind(BATCH / "COMPLETE_MANIFEST.json"),
                  "semantic_review_performed": True, "reviewer_backend": "CURRENT_CODEX_SESSION",
                  "same_inherited_model_and_shared_session_background": True}
    write(HERE / "EVALUATION_POOL.json", {"status": "SEALED_CONTENT_READY_FOR_SOLVER_EVALUATION",
          "candidate_count": len(questions), "provenance": provenance, "questions": questions})
    write(HERE / "ADAPTATION_RECEIPT.json", {"prepared_at_utc": now(), "questions": receipt,
          "native_protocol": bind(NATIVE / "protocol.py"), "adapter_protocol": bind(HERE / "adapter_protocol.py"),
          "changes": ["Generator provenance guard accepts only sealed v108 CURRENT_CODEX_SESSION; exact model remains unknown.",
                      "CONTRACT_DIR points to unchanged native v15 mechanical gates."],
          "solver_protocol": "v15_full_only_random_mcq6x3_order12x1_v1", "ground_calls": 0,
          "evaluation_counts": expected, "exposure_audit": bind(BATCH / "EXPOSURE_AUDIT.json"),
          "sampling": "Twelve fixed clean-training roots, disjoint from common dev/test by root and original source; controlled production pilot, not an untouched test.",
          "independent_untouched_test": False,
          "new_semantic_judgments": False, "public_text_edits": False, "training_started": False})
    # Validate before the first tokenizer or model call; preserve unchanged native gates.
    n = native()
    _, checked, _ = n.validate_pool(HERE / "EVALUATION_POOL.json")
    need(checked == questions, "VALIDATOR_ALTERED_QUESTIONS")
    paths = [Path(__file__), HERE / "audit_results.py", HERE / "adapter_protocol.py",
             HERE / "EVALUATION_POOL.json", HERE / "ADAPTATION_RECEIPT.json"]
    paths += [NATIVE / name for name in ("run_solver.py", "protocol.py", "scoring.py", "selection.py")]
    paths += list(NATIVE.parent.glob("*.py"))
    write(HERE / "ADAPTER_BINDINGS.json", {"prepared_at_utc": now(), "status": "PASS",
          "source_batch_files": frozen, "files": [bind(p) for p in sorted(set(paths))],
          "candidate_count": expected["candidate_count"], "calls_per_model": expected["calls_per_model"],
          "evaluation_counts": expected, "models": MODELS,
          "automatic_retry": False, "training_started": False})
    print(json.dumps({"status": "ADAPTER_READY", **expected}), flush=True)


def verify_adapter():
    binding = read(HERE / "ADAPTER_BINDINGS.json")
    verify(binding["source_batch_files"])
    verify(binding["files"])
    expected = evaluation_counts(read(BATCH / "EVALUATION_COUNTS.json"))
    enforce_budget(expected, read(BATCH / "PRECALL_PLAN.json"))
    need(binding.get("evaluation_counts") == expected
         and binding.get("candidate_count") == expected["candidate_count"]
         and binding.get("calls_per_model") == expected["calls_per_model"], "FROZEN_COUNT_BINDING_MISMATCH")
    exposure_contract()
    return binding


def preflight():
    binding = verify_adapter()
    expected = binding["evaluation_counts"]
    n = native()
    for side, model in MODELS.items():
        n.preflight(HERE / "EVALUATION_POOL.json", HERE / "run" / side, solver_model=model)
    manifests = {side: read(HERE / "run" / side / "manifest.json") for side in MODELS}
    left, right = manifests.values()
    allowed = {"prepared_at_utc", "model", "solver_model", "solver_identity"}
    need({k: v for k, v in left.items() if k not in allowed} == {k: v for k, v in right.items() if k not in allowed}, "PAIRED_INPUTS_DIFFER")
    need(left["request_count"] == right["request_count"] == expected["calls_per_model"], "CALL_COUNT_CHANGED")
    for side in MODELS:
        run = HERE / "run" / side
        need(read(run / "questions.json") == read(HERE / "run" / "trained" / "questions.json"), "PAIRED_QUESTIONS_DIFFER")
        need(read(run / "gold.json") == read(HERE / "run" / "trained" / "gold.json"), "PAIRED_GOLD_DIFFER")
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    paths = [p for side in MODELS for p in (HERE / "run" / side).rglob("*") if p.is_file()]
    verify_adapter()
    write(HERE / "PAIRED_PREFLIGHT.json", {"status": "PASS", "prepared_at_utc": now(),
          "files": [bind(p) for p in sorted(paths)], "adapter_binding": bind(HERE / "ADAPTER_BINDINGS.json"),
          "packages": packages, "python_prefix": sys.prefix, "candidate_count": expected["candidate_count"],
          "calls_per_model": expected["calls_per_model"], "evaluation_counts": expected, "same_prompts_permutations_seeds_decoder_scoring": True,
          "max_prompt_tokens": max(r["prompt_tokens"] for r in left["requests"]), "training_started": False})
    print(json.dumps({"status": "PAIRED_PREFLIGHT_PASS", "calls_per_model": expected["calls_per_model"],
                      "max_prompt_tokens": max(r["prompt_tokens"] for r in left["requests"])}), flush=True)


def live(side):
    verify_adapter()
    pre = read(HERE / "PAIRED_PREFLIGHT.json")
    verify(pre["files"])
    verify([pre["adapter_binding"]])
    need(pre["python_prefix"] == sys.prefix, "PYTHON_PREFIX_CHANGED")
    need(pre["packages"] == {name: importlib.metadata.version(name) for name in PACKAGES}, "PACKAGE_VERSION_CHANGED")
    marker = HERE / (side.upper() + "_EXECUTION")
    need(not marker.with_suffix(".STARTED.json").exists(), "NO_AUTOMATIC_RERUN")
    start = time.monotonic()
    write(marker.with_suffix(".STARTED.json"), {"started_at_utc": now(), "model": MODELS[side],
                                               "paired_preflight": bind(HERE / "PAIRED_PREFLIGHT.json")})
    n = native()
    n.live(HERE / "run" / side)
    n.report(HERE / "run" / side)
    verify_adapter()
    verify(pre["files"])
    write(marker.with_suffix(".COMPLETE.json"), {"completed_at_utc": now(), "wall_seconds": time.monotonic() - start,
          "model": MODELS[side], "calls": read(HERE / "run" / side / "COMPLETE.json")["actual_call_count"],
          "training_started": False, "new_question_generation_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("prepare", "preflight", "live"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--side", choices=tuple(MODELS))
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase == "prepare":
        prepare()
    elif args.phase == "preflight":
        preflight()
    else:
        need(args.phase == "live" and args.side is not None, "LIVE_PHASE_AND_SIDE_REQUIRED")
        live(args.side)

