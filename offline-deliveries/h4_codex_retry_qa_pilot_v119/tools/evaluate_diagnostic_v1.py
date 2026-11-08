"""Paired, content-only Solver diagnostics for the v119 procedure-hold batch.

Usage (preflight/live use the proven local4b Python environment):
  python tools/evaluate_diagnostic_v1.py prepare
  python tools/evaluate_diagnostic_v1.py preflight
  python tools/evaluate_diagnostic_v1.py live --side base
  python tools/evaluate_diagnostic_v1.py live --side trained

The native v15 runner, permutations, seeds, decoding, scoring and selection are
unchanged. Only its input provenance validator is adapted, as in v118. Native
selection is a diagnostic result and never grants training/pool eligibility.
No existing v119 artifact, frozen tool or v118 artifact is overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
NATIVE = B.parent / "h4_feedback_training_pool_v15_qa/evaluation"
REFERENCE = B.parent / "h4_codex_fullscale_ab_v118/tools/evaluate.py"
SCOPE = B / "evaluations/diagnostic"
POLICY = B / "DIAGNOSTIC_CLOSEOUT_POLICY.json"
EXPORTER = B / "scripts/codex_evidence_export.py"
MODELS = {"base": "M0", "trained": "M_clean_policy"}
POLICY_VERSION = "v15_full_only_random_mcq6x3_order12x1_v1"
SEED = 2026100800
LIMITATIONS = {
    "evaluation_scope": "CONTENT_ONLY_DIAGNOSTIC",
    "batch_status": "CONTENT_AUDITED_PROCEDURE_HOLD",
    "role_deviation_status": "FROZEN_REVIEWER_ASSIGNMENT_NOT_FOLLOWED_NO_RETROACTIVE_COMPLIANCE",
    "batch_procedural_success": False,
    "training_pool_eligible": False,
    "training_started": False,
    "training_promotion": False,
    "training_pool_merge_started": False,
    "native_selection_confers_training_eligibility": False,
}
EXPORT_INPUTS = (
    "CUMULATIVE_SOURCE.json", "PUBLIC_QUESTIONS.json", "PRIVATE_GOLD.json",
    "PUBLIC_SOURCE_JUDGMENTS.json", "ASSET_SOURCE_JUDGMENTS.json",
    "PRIVATE_DESIGN.json", "FINAL_CROSSCHECK.json", "PUBLIC_ASSETS.json",
    "FINAL_BLIND_SOURCE_REVIEW.json", "ORDERING_ANCHOR_PLAN.json",
    "QA_DRAFT.json", "COMPOSITION_RECEIPT.json",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def verify(rows):
    for row in rows:
        path = Path(row["path"])
        require(path.is_absolute() and path.is_file(), "BOUND_FILE_MISSING:" + str(path))
        require(sha(path) == row["sha256"], "BOUND_FILE_CHANGED:" + str(path))


def unique_bindings(rows):
    by_path = {}
    for row in rows:
        require(row["path"] not in by_path or by_path[row["path"]] == row,
                "CONFLICTING_BINDING:" + row["path"])
        by_path[row["path"]] = row
    return [by_path[path] for path in sorted(by_path)]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "MODULE_LOAD_FAILED:" + str(path))
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def adapter_text():
    """Exactly the same two provenance/path substitutions used by v118."""
    original = (NATIVE / "protocol.py").read_text(encoding="utf-8")
    guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    replacement = ('if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or '
                   'provenance.get("source_batch_id") != ' + repr(B.name) + '):')
    location = "CONTRACT_DIR = Path(__file__).resolve().parent.parent"
    require(original.count(guard) == 1, "NATIVE_PROVENANCE_GUARD_CHANGED")
    require(original.count(location) == 1, "NATIVE_CONTRACT_LOCATION_CHANGED")
    result = original.replace(guard, replacement).replace(
        location, "CONTRACT_DIR = Path(" + repr(str(NATIVE.parent)) + ")")
    compile(result, "adapter_protocol.py", "exec")
    return result


def native(scope):
    """Load unchanged native functions, replacing validate_pool only."""
    sys.path.insert(0, str(NATIVE))
    for name in ("protocol", "scoring", "selection"):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == NATIVE / (name + ".py"),
                    "NATIVE_MODULE_NAME_COLLISION:" + name)
    obj = module("v119_diagnostic_native_runner", NATIVE / "run_solver.py")
    require(obj.POLICY_VERSION == POLICY_VERSION and obj.SEED == SEED,
            "NATIVE_POLICY_OR_SEED_CHANGED")
    protocol = module("v119_diagnostic_adapter_protocol", scope / "adapter_protocol.py")
    require((scope / "adapter_protocol.py").read_text(encoding="utf-8") == adapter_text(),
            "ADAPTER_EXCEEDS_PROVENANCE_ONLY_CHANGE")
    obj.validate_pool = protocol.validate_pool
    return obj


def policy_bindings():
    policy = read(POLICY)
    for key, value in LIMITATIONS.items():
        if key in policy:
            require(policy[key] == value, "DIAGNOSTIC_POLICY_CHANGED:" + key)
    require(policy.get("role_deviation_status") == LIMITATIONS["role_deviation_status"],
            "ROLE_DEVIATION_MUST_REMAIN_DISCLOSED")
    require(policy.get("training_pool_eligible") is False and
            policy.get("batch_procedural_success") is False and
            policy.get("new_qa_generation") is False,
            "DIAGNOSTIC_SCOPE_MUST_NOT_PROMOTE_OR_REGENERATE")
    rows = [value for value in policy.values() if isinstance(value, dict) and
            set(("path", "sha256")) <= set(value)]
    verify(rows)
    master = read(B / "MASTER_FREEZE.json")
    verify(master["files"])
    return [bind(POLICY), bind(B / "MASTER_FREEZE.json"), *rows]


def native_code_bindings():
    """Bind native/gate/probe code, without executing generation or model loading."""
    sys.path.insert(0, str(NATIVE))
    obj = module("v119_diagnostic_binding_native_runner", NATIVE / "run_solver.py")
    require(obj.POLICY_VERSION == POLICY_VERSION and obj.SEED == SEED,
            "NATIVE_POLICY_OR_SEED_CHANGED")
    probe = obj.load_probe("v119_diagnostic_binding_probe")
    paths = [*NATIVE.glob("*.py"), *NATIVE.parent.glob("*.py"), obj.BASE_RUNNER]
    # Collect local imports of the native probe, including prompt/decoder
    # constants, shared backends, and package __init__ files. No response or
    # later diagnostic output is part of this immutable code input set.
    worktree = probe.WORKTREE.resolve()
    for imported in list(sys.modules.values()):
        value = getattr(imported, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.suffix == ".py" and path.is_relative_to(worktree):
                paths.append(path)
    paths.extend([obj.MODEL_AUDIT, obj.TRAINED_AUDIT])
    return [bind(path) for path in paths]


def prepare():
    require(not SCOPE.exists(), "DIAGNOSTIC_SCOPE_EXISTS_NO_OVERWRITE")
    bindings = policy_bindings()
    manifest = read(B / "MANIFEST.json")
    require(manifest["batch_id"] == B.name and manifest["retry_limit_per_root"] == 1,
            "UNEXPECTED_RETRY_BATCH")
    exporter = module("v119_diagnostic_lossless_exporter", EXPORTER)
    questions, roots, candidate_ids = [], [], set()
    for row in manifest["roots"]:
        unit = Path(row["unit_path"]).resolve(strict=True)
        folder = Path(row["root_path"]).resolve(strict=True)
        require(unit.parent == B / "units" and folder.parent.parent == unit,
                "ROOT_OUTSIDE_RETRY_BATCH:" + row["root_id"])
        for name in EXPORT_INPUTS:
            require((folder / name).is_file(), "WAIT_FOR_FINAL_CONTENT_AUDIT:" + str(folder / name))
        check = read(folder / "FINAL_CROSSCHECK.json")
        require(check["private_sha256"] == sha(folder / "PRIVATE_GOLD.json"),
                "FINAL_PRIVATE_HASH_MISMATCH:" + row["root_id"])
        pass_ids = {q["candidate_id"] for q in check["results"] if q["status"] == "CONTENT_PASS"}
        exported = exporter.export_root_questions(folder)
        require({q["candidate_id"] for q in exported} == pass_ids,
                "CONTENT_PASS_EXPORT_COVERAGE:" + row["root_id"])
        expected_source = folder / "CUMULATIVE_SOURCE.json"
        for question in exported:
            path = Path(question["source_path"]).resolve(strict=True)
            require(path == expected_source and sha(path) == question["source_sha256"],
                    "EXPORT_SOURCE_NOT_THIS_ROOT:" + question["candidate_id"])
            require(question["root_id"] == row["root_id"], "EXPORT_ROOT_MISMATCH")
            require(question["candidate_id"] not in candidate_ids, "DUPLICATE_CANDIDATE")
            candidate_ids.add(question["candidate_id"])
            # Serialize absolute path metadata only. Never rewrite a question,
            # choice, Gold answer, evidence span or original source file.
            question["source_path"] = str(path)
        questions.extend(exported)
        counts = Counter(q["question_type"] for q in exported)
        roots.append({"root_id": row["root_id"], "status": "CONTENT_AUDITED_PROCEDURE_HOLD",
                      "content_pass": len(exported), "question_types": dict(counts),
                      "excluded_content_candidates": len(check["results"]) - len(exported),
                      "unfilled_composition_slots": read(folder / "COMPOSITION_RECEIPT.json")["unfilled_slots"],
                      "procedural_success": False, "training_pool_eligible": False})
        bindings += [bind(folder / name) for name in EXPORT_INPUTS]
        bindings += [bind(unit / "inputs/CONTINUATION_CONSTRAINTS.json"), bind(unit / "RUN_CONFIG.json")]
    bindings += [bind(B / name) for name in ("MANIFEST.json", "ROLE_ASSIGNMENTS.json", "RETRY_LEDGER.json")]
    bindings += [bind(__file__), bind(EXPORTER), bind(REFERENCE),
                 bind(B / "tools/finalize_diagnostic_v1.py")]
    bindings += native_code_bindings()
    # All required inputs were read and validated before creating output files.
    SCOPE.mkdir(parents=True)
    with (SCOPE / "adapter_protocol.py").open("x", encoding="utf-8") as handle:
        handle.write(adapter_text())
    save(SCOPE / "DIAGNOSTIC_SCOPE.json", {
        "created_at_utc": now(), **LIMITATIONS, "policy_binding": bind(POLICY),
        "reason": "Original asset reviews did not follow frozen reviewer assignments; additional content audit does not retroactively certify the original procedure.",
        "native_policy_unchanged": True, "automatic_retry": False,
    })
    save(SCOPE / "EVALUATION_POOL.json", {
        "status": "CONTENT_AUDITED_PROCEDURE_HOLD", **LIMITATIONS,
        "provenance": {"backend": "CURRENT_CODEX_SESSION", "source_batch_id": B.name,
                       "exact_model_id": None, "same_inherited_model": True,
                       "evaluation_scope": "CONTENT_ONLY_DIAGNOSTIC"}, "questions": questions,
    })
    bindings += [bind(SCOPE / name) for name in ("adapter_protocol.py", "EVALUATION_POOL.json", "DIAGNOSTIC_SCOPE.json")]
    files = unique_bindings(bindings)
    verify(files)
    if questions:
        native(SCOPE).validate_pool(SCOPE / "EVALUATION_POOL.json")
    calls = sum(18 if q["question_type"] == "multi_select" else 12 for q in questions)
    save(SCOPE / "ADAPTER_BINDINGS.json", {
        "schema_version": "v119_new_diagnostic_adapter_bindings_v1", "created_at_utc": now(),
        "binding_origin": "NEW_DIAGNOSTIC_BINDINGS_NOT_ORIGINAL_MASTER_FREEZE",
        **LIMITATIONS, "files": files, "terminal_roots": roots,
        "candidate_count": len(questions), "calls_per_model": calls,
        "paired_models": MODELS, "seed": SEED, "native_policy_version": POLICY_VERSION,
        "full_slots": {"multi_select": "6 permutations x 3 rollouts", "ordering": "12 permutations x 1 rollout"},
        "native_decoding_permutations_scoring_selection_unchanged": True,
        "bound_files_exclude_future_responses_reports_and_unrelated_diagnostics": True,
    })
    print(json.dumps({"status": "DIAGNOSTIC_READY" if questions else "EMPTY_CONTENT_POOL",
                      **LIMITATIONS, "roots": len(roots), "questions": len(questions), "calls_per_model": calls}))


def frozen_adapter():
    policy_bindings()
    adapter = read(SCOPE / "ADAPTER_BINDINGS.json")
    require(adapter.get("binding_origin") == "NEW_DIAGNOSTIC_BINDINGS_NOT_ORIGINAL_MASTER_FREEZE",
            "INVALID_DIAGNOSTIC_BINDING_ORIGIN")
    require(all(adapter.get(key) == value for key, value in LIMITATIONS.items()),
            "DIAGNOSTIC_LIMITATIONS_CHANGED")
    verify(adapter["files"])
    require(adapter["candidate_count"] > 0, "EMPTY_DIAGNOSTIC_POOL")
    return adapter


def preflight():
    adapter = frozen_adapter()
    require(not (SCOPE / "run").exists() and not (SCOPE / "PAIRED_PREFLIGHT.json").exists(),
            "PAIRED_PREFLIGHT_EXISTS_NO_OVERWRITE")
    obj = native(SCOPE)
    for side, model in MODELS.items():
        obj.preflight(SCOPE / "EVALUATION_POOL.json", SCOPE / "run" / side, solver_model=model)
    manifests = [read(SCOPE / "run" / side / "manifest.json") for side in MODELS]
    require(all(m["request_count"] == adapter["calls_per_model"] for m in manifests), "PREFLIGHT_CALL_COUNT")
    require(manifests[0]["requests"] == manifests[1]["requests"], "PAIRED_REQUESTS_OR_SEEDS_DIFFER")
    require(manifests[0]["decoder_contract"] == manifests[1]["decoder_contract"], "PAIRED_DECODERS_DIFFER")
    for name in ("questions.json", "gold.json", "canonical_bindings.json"):
        require(read(SCOPE / "run/base" / name) == read(SCOPE / "run/trained" / name), "PAIRED_INPUTS_DIFFER:" + name)
    maximum = max(row["prompt_tokens"] for m in manifests for row in m["requests"])
    require(maximum + 32 <= 16384, "PROMPT_EXCEEDS_NATIVE_CONTEXT")
    # Freeze preflight input artifacts only. Responses/results/start markers are
    # intentionally absent, so completing one side cannot invalidate the other.
    files = []
    for side in MODELS:
        run = SCOPE / "run" / side
        files += [bind(run / name) for name in ("manifest.json", "questions.json", "gold.json", "canonical_bindings.json", "solver_identity.json")]
        files += [bind(path) for path in (run / "sources").glob("*.json")]
        files += [bind(path) for path in (run / "prompts").glob("*.txt")]
    verify(adapter["files"])
    save(SCOPE / "PAIRED_PREFLIGHT.json", {
        "status": "DIAGNOSTIC_PREFLIGHT_PASS", "created_at_utc": now(), **LIMITATIONS,
        "files": unique_bindings(files), "adapter_binding": bind(SCOPE / "ADAPTER_BINDINGS.json"),
        "candidate_count": adapter["candidate_count"], "calls_per_model": adapter["calls_per_model"],
        "max_prompt_tokens": maximum, "paired_requests_seeds_prompts_and_gold_identical": True,
        "automatic_retry": False,
    })
    print(json.dumps({"status": "DIAGNOSTIC_PAIRED_PREFLIGHT_PASS", **LIMITATIONS,
                      "max_prompt_tokens": maximum, "calls_per_model": adapter["calls_per_model"]}))


def live(side):
    require(side in MODELS, "LIVE_SIDE_REQUIRED")
    adapter = frozen_adapter()
    pre = read(SCOPE / "PAIRED_PREFLIGHT.json")
    verify(pre["files"])
    verify([pre["adapter_binding"]])
    require(all(pre.get(key) == value for key, value in LIMITATIONS.items()), "PREFLIGHT_SCOPE_CHANGED")
    run = SCOPE / "run" / side
    manifest = read(run / "manifest.json")
    require(manifest["solver_model"] == MODELS[side], "SOLVER_SIDE_MODEL_MISMATCH")
    require(manifest["request_count"] == adapter["calls_per_model"], "SOLVER_CALL_COUNT_MISMATCH")
    obj = native(SCOPE)
    # Fail before a started marker if environment, fixed model audit or code
    # inputs fail native verification. No automatic retry is introduced.
    require(Path(sys.prefix) == obj.EXPECTED_VENV, "USE_PROVEN_LOCAL4B_VENV")
    obj._verify_manifest(run, manifest)
    save(SCOPE / (side.upper() + "_STARTED.json"), {
        "started_at_utc": now(), **LIMITATIONS, "side": side,
        "solver_model": MODELS[side], "automatic_retry": False,
        "adapter_binding": bind(SCOPE / "ADAPTER_BINDINGS.json"),
        "paired_preflight_binding": bind(SCOPE / "PAIRED_PREFLIGHT.json"),
    })
    obj.live(run)
    obj.report(run)
    verify(pre["files"])
    verify(adapter["files"])
    complete = read(run / "COMPLETE.json")
    require(complete["actual_call_count"] == adapter["calls_per_model"], "INCOMPLETE_CALL_SET")
    save(SCOPE / (side.upper() + "_COMPLETE.json"), {
        "completed_at_utc": now(), **LIMITATIONS, "side": side,
        "solver_model": MODELS[side], "actual_calls": complete["actual_call_count"],
        "native_complete_binding": bind(run / "COMPLETE.json"),
        "native_results_binding": bind(run / "RESULTS.json"),
        "interpretation": "Native selection scores are diagnostics only; no candidate is promoted to training or a formal pool.",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "live"))
    parser.add_argument("--side", choices=tuple(MODELS))
    args = parser.parse_args(argv)
    if args.action == "live":
        parser.error("live requires --side base|trained") if args.side is None else live(args.side)
    else:
        if args.side is not None:
            parser.error("--side is for live; preflight prepares both paired sides")
        prepare() if args.action == "prepare" else preflight()


if __name__ == "__main__":
    main()
