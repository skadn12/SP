"""Groundless full-only evaluation.

Usage (inside the proven local 4B environment):
  python run_solver.py preflight --pool POOL.json --run RUN --solver-model M_clean_policy
  python run_solver.py live --run RUN
  python run_solver.py report --run RUN

The preflight plans only full-context calls. It never invokes no-context or
ground evaluation and never applies a hidden stability/context-gain gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation')
DELIVERIES = HERE.parent
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
MODEL_PATHS = {
    "M0": TOKENIZER,
    "M_clean_policy": Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model"),
}
EXPECTED_VENV = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b")
BASE_RUNNER = DELIVERIES.parent / "h4_event_ledger_m0_pilot_v1/run.py"
SEED = 2026100800
MODEL_AUDIT = DELIVERIES.parent / "h4_multiselect_partial_admission_v1/followup_model_identity.json"
MODEL_AUDIT_SHA256 = "59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035"
TRAINED_AUDIT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/model_identity_report.json")
TRAINED_AUDIT_SHA256 = "add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0"
POLICY_VERSION = "v15_full_only_random_mcq6x3_order12x1_v1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from protocol import make_variants, prompt, read, sha, validate_pool
from scoring import score_response
from selection import audit_complete_call_set, select


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("REFUSE_OVERWRITE:" + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_probe(name: str):
    spec = importlib.util.spec_from_file_location(name, BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_identity(solver_model: str = "M_clean_policy") -> dict:
    """Bind an explicitly named model to its immutable historical audit."""
    if solver_model not in MODEL_PATHS:
        raise RuntimeError("UNSUPPORTED_SOLVER_MODEL")
    if not MODEL_AUDIT.is_file() or sha(MODEL_AUDIT) != MODEL_AUDIT_SHA256:
        raise RuntimeError("BASE_MODEL_AUDIT_CHANGED_OR_MISSING")
    base = read(MODEL_AUDIT)
    if Path(base.get("model_path", "")).resolve() != TOKENIZER.resolve() or not base.get("files"):
        raise RuntimeError("MODEL_AUDIT_PATH_MISMATCH")
    identity = {"model_path": str(MODEL_PATHS[solver_model]), "model_identifier": solver_model,
                "base_model_identifier": "Qwen/Qwen3-4B", "tokenizer_path": str(TOKENIZER),
                "audit_path": str(MODEL_AUDIT), "audit_sha256": MODEL_AUDIT_SHA256,
                "weight_revision": base.get("weight_revision"),
                "checkpoint_files": base["files"], "checked_at_utc": now(),
                "tokenizer_audit_path": str(MODEL_AUDIT), "tokenizer_audit_sha256": MODEL_AUDIT_SHA256}
    if solver_model == "M_clean_policy":
        if not TRAINED_AUDIT.is_file() or sha(TRAINED_AUDIT) != TRAINED_AUDIT_SHA256:
            raise RuntimeError("TRAINED_MODEL_AUDIT_CHANGED_OR_MISSING")
        trained = read(TRAINED_AUDIT)["models"][solver_model]
        model_path = MODEL_PATHS[solver_model]
        if (trained.get("inference_model_path") != str(model_path)
                or trained.get("merged_model_identity", {}).get("path") != str(model_path)):
            raise RuntimeError("TRAINED_MODEL_AUDIT_PATH_MISMATCH")
        checkpoint = [{"path": str(model_path / row["name"]), "name": row["name"],
                       "bytes": row["size_bytes"], "sha256": row["sha256"]}
                      for row in trained["merged_model_identity"]["files"]]
        # Match the trained reevaluation's base-tokenizer prompt contract.
        tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}
        checkpoint.extend(row for row in base["files"] if row["name"] in tokenizer_names)
        identity.update(audit_path=str(TRAINED_AUDIT), audit_sha256=TRAINED_AUDIT_SHA256,
                        checkpoint_files=checkpoint, adapter_path=trained["adapter_path"],
                        model_fingerprint_sha256=trained["model_fingerprint_sha256"],
                        adapter_enabled=True, adapter_already_merged=True)
    return identity


def _validate_model(identity: dict) -> None:
    solver_model = identity.get("model_identifier")
    if (solver_model not in MODEL_PATHS or identity.get("model_path") != str(MODEL_PATHS[solver_model])
            or identity.get("tokenizer_path") != str(TOKENIZER)):
        raise RuntimeError("MODEL_IDENTITY_PATH_INVALID")
    audited = model_identity(solver_model)
    if (identity.get("checkpoint_files") != audited["checkpoint_files"]
            or identity.get("audit_sha256") != audited["audit_sha256"]):
        raise RuntimeError("MODEL_IDENTITY_DOES_NOT_MATCH_FIXED_AUDIT")
    if not identity.get("checkpoint_files"):
        raise RuntimeError("MODEL_CHECKPOINT_AUDIT_REQUIRED")
    for row in identity["checkpoint_files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError("MODEL_CHECKPOINT_CHANGED:" + str(path))


def _variants_for(question: dict) -> tuple[list[dict], int]:
    count = 6 if question["question_type"] == "multi_select" else 12
    paired_question = dict(question)
    paired_question["candidate_id"] = question["original_candidate_id"]
    return make_variants(paired_question, seed=SEED, count=count), count


def preflight(pool_path: Path, run: Path, *, require_ordering_review: bool = False,
              require_model: bool = True, solver_model: str = "M_clean_policy") -> None:
    if require_model and Path(sys.prefix) != EXPECTED_VENV:
        raise RuntimeError("USE_PROVEN_LOCAL4B_VENV")
    if run.exists():
        raise RuntimeError("RUN_ALREADY_EXISTS_NO_OVERWRITE")
    pool_path = pool_path.resolve(strict=True)
    pool, questions, sources = validate_pool(pool_path, require_ordering_review=require_ordering_review)
    identity = model_identity(solver_model) if require_model else {
        "model_path": str(MODEL_PATHS[solver_model]), "model_identifier": solver_model,
        "tokenizer_path": str(TOKENIZER), "skipped": True}
    probe = load_probe("groundless_preflight_probe")
    decoder = probe.solver.generation_config("multi_select")
    required_decoder = {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                        "enable_thinking": False, "structured_output_version": "finite_choice_set_v1"}
    if any(decoder.get(key) != value for key, value in required_decoder.items()):
        raise RuntimeError("DECODER_CONTRACT_MISMATCH")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    except Exception:
        if require_model:
            raise
        tokenizer = None
    requests: list[dict] = []
    prompts: dict[str, str] = {}
    for question in questions:
        variants, count = _variants_for(question)
        source = sources[question["root_id"]]["source"]
        field = "options" if question["question_type"] == "multi_select" else "items"
        labels = list(variant_labels := (tuple("ABCDEF") if field == "options" else tuple(f"I{i}" for i in range(1, 7))))
        text_to_content = {text: content_id for content_id, text in variants[0]["content_to_text"].items()}
        question["gold_content_ids"] = [text_to_content[question[field][label]] for label in question["gold"]]
        for variant in variants:
            choices = probe.contract.structured_choices(question["question_type"], variant)
            expected_choices = 63 if question["question_type"] == "multi_select" else 720
            if len(choices) != expected_choices:
                raise RuntimeError("STRUCTURED_CHOICE_SET_MISMATCH:" + question["candidate_id"])
            # Reuse the proven chat-template contract. The public perturbation
            # check ensures Gold/evidence/content mappings cannot affect it.
            dialogue = probe.contract.dialogue_rows(source)
            text = probe.contract.build_prompt(tokenizer, question, variant, dialogue,
                                               context_mode="full_context") if tokenizer is not None else prompt(question, variant, source)
            if tokenizer is not None:
                public = {key: question[key] for key in ("question_type", "question")}
                public_variant = {"options" if field == "options" else "items": variant[field]}
                public_text = probe.contract.build_prompt(tokenizer, public, public_variant, dialogue,
                                                          context_mode="full_context")
                if text != public_text:
                    raise RuntimeError("PROMPT_METADATA_LEAK:" + question["candidate_id"])
            if tokenizer is not None:
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if tokens + 32 > 16384:
                    raise RuntimeError("PROMPT_EXCEEDS_MODEL_CONTEXT:" + question["candidate_id"])
            else:
                tokens = None
            rollouts = 3 if field == "options" else 1
            for slot in range(rollouts):
                request_id = f"{question['candidate_id']}__{variant['variant_id']}__full_context__s{slot:02d}"
                requests.append({"request_id": request_id, "candidate_id": question["candidate_id"],
                                 "root_id": question["root_id"], "question_type": question["question_type"],
                                 "variant": variant, "mode": "full_context", "slot": slot,
                                 "seed": read(Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_wording_minfix_v114/PAIRING_INPUTS.json'))["original_request_seeds"][variant["variant_id"] + "|" + str(slot)], "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "prompt_tokens": tokens})
                prompts[request_id] = text
    expected = sum(6 * 3 if q["question_type"] == "multi_select" else 12 for q in questions)
    if len(requests) != expected:
        raise RuntimeError("FULL_REQUEST_PLAN_COUNT_MISMATCH")
    _validate_model(identity) if require_model else None
    frozen_sources = {str(Path("sources") / f"{root}.json"): row["sha256"] for root, row in sources.items()}
    from scripts.current_method.restart_n5_spell import backends, vllm_backends, structured_qa_box_protocol
    code_paths = (Path(__file__), HERE / "protocol.py", HERE / "scoring.py", HERE / "selection.py",
                  BASE_RUNNER, Path(probe.solver.__file__), Path(probe.contract.__file__),
                  Path(backends.__file__), Path(vllm_backends.__file__), Path(structured_qa_box_protocol.__file__))
    frozen_code = {str(path): sha(path) for path in code_paths}
    run.mkdir(parents=True)
    public_questions = []
    bindings = []
    for question in questions:
        public = {k: v for k, v in question.items() if k not in {"gold", "gold_content_ids"}}
        public_questions.append(public)
        field = "options" if question["question_type"] == "multi_select" else "items"
        variants, _ = _variants_for(question)
        first = variants[0]
        text_to_content = {text: content_id for content_id, text in first["content_to_text"].items()}
        label_to_content = {label: text_to_content[text] for label, text in question[field].items()}
        evidence_key = "option_evidence" if field == "options" else "item_evidence"
        bindings.append({"candidate_id": question["candidate_id"], "root_id": question["root_id"],
                         "content_id_by_label": label_to_content,
                         "gold_content_ids": question["gold_content_ids"],
                         "evidence_by_content_id": {
                             label_to_content[label]: citation
                             for label, citation in question[evidence_key].items()
                         }})
    write_once(run / "questions.json", {"candidates": public_questions})
    write_once(run / "gold.json", {q["candidate_id"]: q["gold_content_ids"] for q in questions})
    write_once(run / "canonical_bindings.json", {"candidates": bindings})
    for root, row in sources.items():
        target = run / "sources" / f"{root}.json"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(row["path"], target)
    for request_id, text in prompts.items():
        path = run / "prompts" / f"{request_id}.txt"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
    write_once(run / "solver_identity.json", identity)
    manifest = {"status": "PREFLIGHT_PASS", "prepared_at_utc": now(), "policy_version": POLICY_VERSION,
                "execution": "full_only_no_ground_v1", "model": identity["model_path"],
                "solver_model": solver_model, "tokenizer": str(TOKENIZER), "candidate_count": len(questions),
                "request_count": len(requests), "request_count_semantics": "Full-context calls only; no ground calls planned.",
                "full_slots": {"multi_select": "6 permutations x 3 rollouts", "ordering": "12 permutations x 1 rollout"},
                "random_display": {"version": "independent_seeded_permutation_v2", "seed": SEED,
                                   "gold_blind": True, "cyclic_rebinding": False, "shape_filter": False},
                "generator_provenance": pool.get("provenance"), "solver_identity": identity,
                "decoder_contract": decoder,
                "input_sha256": {"pool": sha(pool_path), **frozen_sources,
                                 "questions.json": sha(run / "questions.json"),
                                 "gold.json": sha(run / "gold.json"),
                                 "canonical_bindings.json": sha(run / "canonical_bindings.json")}, "pool_path": str(pool_path),
                "pool_sha256": sha(pool_path), "code_sha256": frozen_code, "requests": requests,
                "training_started": False, "training_promotion": False, "new_generation_calls": 0}
    write_once(run / "manifest.json", manifest)
    print(json.dumps({"status": "PREFLIGHT_PASS", "questions": len(questions), "full_calls": len(requests)}, ensure_ascii=False))


def _verify_manifest(run: Path, manifest: dict) -> None:
    if manifest.get("policy_version") != POLICY_VERSION:
        raise RuntimeError("POLICY_VERSION_MISMATCH")
    for path, expected in manifest["code_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("CODE_CHANGED:" + path)
    for name, expected in manifest["input_sha256"].items():
        path = Path(manifest["pool_path"]) if name == "pool" else run / name
        if sha(path) != expected:
            raise RuntimeError("INPUT_CHANGED:" + str(path))
    identity = read(run / "solver_identity.json")
    if identity != manifest.get("solver_identity"):
        raise RuntimeError("MODEL_IDENTITY_MANIFEST_MISMATCH")
    if (manifest.get("model") != identity.get("model_path")
            or manifest.get("solver_model") != identity.get("model_identifier")
            or manifest.get("tokenizer") != identity.get("tokenizer_path")):
        raise RuntimeError("MANIFEST_MODEL_PATH_MISMATCH")
    if identity.get("audit_path") and sha(identity["audit_path"]) != identity.get("audit_sha256"):
        raise RuntimeError("MODEL_AUDIT_CHANGED")
    _validate_model(identity)
    if manifest.get("decoder_contract", {}).get("enable_thinking") is not False:
        raise RuntimeError("DECODER_CONTRACT_MISMATCH")


def live(run: Path) -> None:
    if Path(sys.prefix) != EXPECTED_VENV:
        raise RuntimeError("USE_PROVEN_LOCAL4B_VENV")
    if (run / "STARTED.json").exists():
        raise RuntimeError("ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    manifest = read(run / "manifest.json"); _verify_manifest(run, manifest)
    probe = load_probe("groundless_live_probe")
    if probe.solver.generation_config("multi_select") != manifest.get("decoder_contract"):
        raise RuntimeError("DECODER_CONTRACT_CHANGED")
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    questions = read(run / "questions.json")["candidates"]
    gold = read(run / "gold.json")
    q_by_id = {q["candidate_id"]: q for q in questions}
    requests = {row["request_id"]: row for row in manifest["requests"]}
    write_once(run / "STARTED.json", {"started_at_utc": now(), "manifest_sha256": sha(run / "manifest.json")})
    executed: list[str] = []; backend = None
    try:
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        probe.solver.ensure_vllm_environment()
        backend = VLLMSolverBackend(model_path=Path(manifest["model"]), tokenizer_path=TOKENIZER,
                                   max_model_len=16384, temperature=0.8, top_p=0.9, max_tokens=32,
                                   gpu_memory_utilization=0.80)
        for row in manifest["requests"]:
            prompt_path = run / "prompts" / f"{row['request_id']}.txt"
            if sha(prompt_path) != row["prompt_sha256"]:
                raise RuntimeError("PROMPT_CHANGED:" + row["request_id"])
            request = SolverRequest(request_id=row["request_id"], qa_id=row["candidate_id"],
                                    evaluation_mode="full_context", order_id=row["variant"]["variant_id"],
                                    target_slot=row["slot"], retry_index=0, sampling_seed=row["seed"],
                                    options=row["variant"].get("options") or row["variant"].get("items"),
                                    canonical_answer_text="", prompt_sha256=row["prompt_sha256"],
                                    prompt=prompt_path.read_text(encoding="utf-8"))
            response = probe.solver._structured_generate(backend, request, question_type=row["question_type"])
            q = dict(q_by_id[row["candidate_id"]]); q["gold_content_ids"] = gold[row["candidate_id"]]
            score = score_response(q, row, response.raw_text)
            decoder_errors = [key for key in ("temperature", "top_p", "max_tokens", "structured_output_api",
                                              "structured_output_version")
                              if response.metadata.get(key) != manifest["decoder_contract"].get(key)]
            if response.model_identifier != manifest["model"]:
                decoder_errors.append("model_identifier")
            valid = (response.metadata.get("finish_reason") in ("stop", "eos")
                     and score.get("scored") is True and not decoder_errors)
            record = {"request_id": row["request_id"], "candidate_id": row["candidate_id"], "root_id": row["root_id"],
                      "solver_model": manifest["solver_model"], "model_path": manifest["model"],
                      "question_type": row["question_type"], "variant_id": row["variant"]["variant_id"],
                      "mode": "full_context", "slot": row["slot"], "seed": row["seed"], "raw_output": response.raw_text,
                      "metadata": response.metadata, "score": score, "evaluation_valid": valid,
                      "data_errors": [] if valid else (["DECODER_METADATA_MISMATCH:" + key for key in decoder_errors]
                                                        or ["INCOMPLETE_OR_UNPARSEABLE_RESPONSE"])}
            write_once(run / "responses" / f"{row['request_id']}.json", record); executed.append(row["request_id"])
        write_once(run / "COMPLETE.json", {"status": "COMPLETE", "actual_call_count": len(executed),
                                           "training_started": False, "training_promotion": False})
    except Exception as error:
        write_once(run / "FAILED.json", {"status": "FAILED", "error": str(error), "traceback": traceback.format_exc(), "executed_request_ids": executed})
        raise
    finally:
        if backend is not None: backend.close()


def report(run: Path) -> None:
    manifest = read(run / "manifest.json"); questions = read(run / "questions.json")["candidates"]
    records = [read(path) for path in sorted((run / "responses").glob("*.json"))]
    output = []
    for q in questions:
        rows = [row for row in records if row["candidate_id"] == q["candidate_id"]]
        variant_count = 6 if q["question_type"] == "multi_select" else 12
        variants = [f"perm_{index:02d}" for index in range(variant_count)]
        scores = {variant: [Fraction(row["score"]["partial_score_fraction"]) for row in rows if row["variant_id"] == variant and row.get("evaluation_valid")] for variant in variants}
        valid = sum(len(values) for values in scores.values())
        exact = [row for row in rows if row.get("evaluation_valid") and row["score"].get("exact_match")]
        call_errors = audit_complete_call_set(manifest, records, candidate_id=q["candidate_id"])
        if valid != len(rows):
            call_errors.append("PARSE_OR_CALL_FAILURE")
        selection = select(scores, variants, exact_rate=Fraction(len(exact), valid) if valid else None,
                           data_errors=call_errors)
        selection.update({"candidate_id": q["candidate_id"], "root_id": q["root_id"], "question_type": q["question_type"],
                          "expected_full_calls": 18 if q["question_type"] == "multi_select" else 12,
                          "actual_full_calls": len(rows)})
        output.append(selection)
    result = {"policy_version": manifest["policy_version"], "solver_model": manifest["solver_model"],
              "model_path": manifest["model"], "ground_calls": 0, "questions": output,
              "call_set_errors": audit_complete_call_set(manifest, records),
              "selected_questions": [row["candidate_id"] for row in output if row["selected_by_policy"] is True]}
    write_once(run / "RESULTS.json", result)
    (run / "REPORT.md").write_text("# Groundless full-only evaluation\n\n" + "\n".join(
        f"- `{row['candidate_id']}`: `{row['selection_reason']}`, S_full=`{row['full_partial_mean']}`" for row in output), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="phase", required=True)
    pre = sub.add_parser("preflight"); pre.add_argument("--pool", type=Path, required=True); pre.add_argument("--run", type=Path, required=True)
    pre.add_argument("--require-ordering-review", action="store_true"); pre.add_argument("--skip-model-check", action="store_true")
    pre.add_argument("--solver-model", choices=tuple(MODEL_PATHS), default="M_clean_policy")
    for name in ("live", "report"):
        cmd = sub.add_parser(name); cmd.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "preflight": preflight(args.pool, args.run, require_ordering_review=args.require_ordering_review, require_model=not args.skip_model_check, solver_model=args.solver_model)
    elif args.phase == "live": live(args.run)
    else: report(args.run)


if __name__ == "__main__": main()
