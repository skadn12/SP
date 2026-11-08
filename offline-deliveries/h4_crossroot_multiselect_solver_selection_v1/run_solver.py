"""Freeze and evaluate audited cross-root six-option multiselect questions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
DEFAULT_POOL = DELIVERIES / "h4_crossroot_multiselect_content_review_v1/READY_FOR_EVALUATION.json"
BASE_RUNNER = DELIVERIES / "h4_event_ledger_m0_pilot_v1/run.py"
PRIOR_RUNNER = DELIVERIES / "h4_ordering_large_model_solver_selection_v1/run_solver.py"
POLICY_PATH = DELIVERIES / "h4_multiselect_ground_full_selection_v1/selection.py"
SCORER_PATH = DELIVERIES / "h4_multiselect_partial_admission_v1/scoring.py"
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
EXPECTED_VENV = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from protocol import FULL_SLOTS, GROUND_SLOTS, VARIANTS, display, object_sha, read, request_plan, sha, validate_pool


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prior = module("multiselect_prior_identity", PRIOR_RUNNER)
policy = module("crossroot_multiselect_policy", POLICY_PATH)
scorer = module("crossroot_multiselect_scorer", SCORER_PATH)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise RuntimeError("FROZEN_FILE_CHANGED:" + path)


def verify_model(identity):
    for row in identity["files"]:
        if prior.file_stat(row["path"]) != {key: row[key] for key in
                                             ("bytes", "mtime_ns", "ctime_ns", "inode", "device")}:
            raise RuntimeError("MODEL_FILE_STAT_CHANGED:" + row["path"])


def preflight(pool_path, run):
    if Path(sys.prefix) != EXPECTED_VENV:
        raise RuntimeError("USE_LOCAL4B_VENV")
    if run.exists():
        raise RuntimeError("RUN_ALREADY_EXISTS_NO_OVERWRITE")
    pool_path = pool_path.resolve(strict=True)
    pool, questions, sources = validate_pool(pool_path)
    identity = prior.current_model_identity()
    probe = module("multiselect_preflight_probe", BASE_RUNNER)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    config = probe.solver.generation_config("multi_select")
    required = {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                "enable_thinking": False,
                "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                "structured_output_version": "finite_choice_set_v1"}
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError("DECODING_CONTRACT_MISMATCH")

    def make_prompt(question, variant, source, mode):
        choices = probe.contract.structured_choices("multi_select", variant)
        if len(choices) != 63 or any(not choice.startswith("<box>") for choice in choices):
            raise RuntimeError("STRUCTURED_CHOICES_INVALID")
        dialogue = probe.contract.dialogue_rows(source)
        prompt = probe.contract.build_prompt(tokenizer, question, variant, dialogue,
                                             context_mode=mode)
        public = {key: question[key] for key in ("question_type", "question")}
        if prompt != probe.contract.build_prompt(tokenizer, public,
                                                 {"options": variant["options"]}, dialogue,
                                                 context_mode=mode):
            raise RuntimeError("PROMPT_METADATA_LEAK")
        if len(tokenizer.encode(prompt)) + config["max_tokens"] > 16384:
            raise RuntimeError("PROMPT_EXCEEDS_MODEL_CONTEXT:" + question["candidate_id"])
        return prompt

    requests, prompts = request_plan(questions, sources, make_prompt)
    for row in requests:
        row["prompt_tokens"] = len(tokenizer.encode(prompts[row["request_id"]]))
    verify_model(identity)
    protected = {str(pool_path): sha(pool_path), str(POLICY_PATH): sha(POLICY_PATH),
                 str(SCORER_PATH): sha(SCORER_PATH)}
    protected.update({row["path"]: row["sha256"] for row in sources.values()})
    verify(protected)
    run.mkdir(parents=True)
    write_once(run / "questions.json", {"candidates": questions})
    write_once(run / "gold.json", {q["candidate_id"]: q["gold"] for q in questions})
    for root, row in sources.items():
        destination = run / "sources" / (root + ".json")
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(row["path"], destination)
    for request_id, prompt in prompts.items():
        path = run / "prompts" / (request_id + ".txt")
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(prompt)
    write_once(run / "solver_identity.json", identity)
    packages = ("torch", "vllm", "transformers", "tokenizers", "xgrammar")
    code_paths = [Path(__file__), HERE / "protocol.py", HERE / "report_results.py",
                  PRIOR_RUNNER, BASE_RUNNER, POLICY_PATH, SCORER_PATH,
                  Path(probe.contract.__file__), Path(probe.solver.__file__),
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/vllm_backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py"]
    frozen = {str(Path("sources") / (root + ".json")): sha(run / "sources" / (root + ".json"))
              for root in sources}
    frozen.update({name: sha(run / name) for name in ("questions.json", "gold.json")})
    write_once(run / "manifest.json", {
        "status": "PREFLIGHT_PASS", "prepared_at_utc": now(),
        "execution": "multiselect_ground_k3_adaptive_k5_full_6x3_v1",
        "model": str(MODEL), "candidate_count": len(questions),
        "request_count": len(requests), "maximum_preplanned_slots_per_question": 23,
        "request_count_semantics": "Maximum preplanned slots, not actual Solver calls.",
        "ground_display": "perm_00", "ground_initial_slots": [0, 1, 2],
        "ground_extension_slots": [3, 4], "full_variant_ids": list(VARIANTS),
        "full_rollouts_per_variant": len(FULL_SLOTS), "display_rule": "forward_cyclic_left_v1",
        **required, "generation_config": config, "max_model_len": 16384,
        "gpu_memory_utilization": 0.80, "generator_provenance": pool["provenance"],
        "solver_identity": identity, "policy_version": policy.VERSION,
        "ground_rule_version": policy.GROUND_VERSION,
        "seed_schedule": "preplanned_request_index_v1", "input_sha256": frozen,
        "pool_path": str(pool_path), "pool_sha256": sha(pool_path),
        "source_paths_by_root": {root: row["path"] for root, row in sources.items()},
        "source_turn_count_by_root": {root: row["turn_count"] for root, row in sources.items()},
        "code_sha256": {str(path): sha(path) for path in code_paths},
        "historical_snapshot": protected, "requests": requests,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "training_started": False, "training_promotion": False,
        "new_generation_calls": 0,
    })
    print(json.dumps({"status": "PREFLIGHT_PASS", "questions": len(questions),
                      "maximum_calls": len(requests),
                      "maximum_prompt_tokens": max(row["prompt_tokens"] for row in requests),
                      "roots": {root: row["turn_count"] for root, row in sources.items()},
                      "model": str(MODEL)}), flush=True)


def live(run):
    manifest = read(run / "manifest.json")
    if (run / "STARTED.json").exists():
        raise RuntimeError("ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    if Path(sys.prefix) != EXPECTED_VENV or sys.prefix != manifest["python_prefix"]:
        raise RuntimeError("PYTHON_RUNTIME_CHANGED")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("PACKAGE_CHANGED:" + name)
    verify({str(run / name): digest for name, digest in manifest["input_sha256"].items()})
    verify(manifest["code_sha256"])
    verify(manifest["historical_snapshot"])
    verify_model(manifest["solver_identity"])
    probe = module("multiselect_live_probe", BASE_RUNNER)
    if probe.solver.generation_config("multi_select") != manifest["generation_config"]:
        raise RuntimeError("DECODING_CHANGED")
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    questions = read(run / "questions.json")["candidates"]
    requests = {(row["candidate_id"], row["mode"], row["variant"]["variant_id"], row["slot"]): row
                for row in manifest["requests"]}
    prompts = {}
    for row in manifest["requests"]:
        prompt_path = run / "prompts" / (row["request_id"] + ".txt")
        if sha(prompt_path) != row["prompt_sha256"]:
            raise RuntimeError("PROMPT_CHANGED:" + row["request_id"])
        root_key = str(Path("sources") / (row["root_id"] + ".json"))
        if row["source_sha256"] != manifest["input_sha256"][root_key]:
            raise RuntimeError("REQUEST_SOURCE_BINDING_MISMATCH:" + row["request_id"])
        prompts[row["request_id"]] = prompt_path.read_text(encoding="utf-8")
    executed, statuses = [], []
    write_once(run / "STARTED.json", {"started_at_utc": now(),
                                      "maximum_request_count": len(requests),
                                      "manifest_sha256": sha(run / "manifest.json")})
    backend = None
    try:
        backend = probe.solver.create_backend(model_path=MODEL, tokenizer_path=MODEL)

        def evaluate(question, mode, shift, slot):
            key = question["candidate_id"], mode, VARIANTS[shift], slot
            row = requests[key]
            request = SolverRequest(
                request_id=row["request_id"], qa_id=row["candidate_id"],
                evaluation_mode=mode, order_id=row["variant"]["variant_id"],
                target_slot=slot, retry_index=0, sampling_seed=row["seed"],
                options=row["variant"]["options"], canonical_answer_text="",
                prompt_sha256=row["prompt_sha256"], prompt=prompts[row["request_id"]])
            response = probe.solver._structured_generate(backend, request,
                                                         question_type="multi_select")
            raw = {"request_id": row["request_id"], "raw_output": response.raw_text}
            score = scorer.score_response(question, row, raw)
            metadata = {**response.metadata, "enable_thinking": False}
            errors = []
            if metadata.get("finish_reason") not in ("stop", "eos") or score["scoring_status"] != "SCORED":
                errors.append("INCOMPLETE_OR_UNPARSEABLE_RESPONSE")
            for field in ("temperature", "top_p", "max_tokens", "structured_output_api",
                          "structured_output_version"):
                if metadata.get(field) != manifest["generation_config"].get(field):
                    errors.append("DECODING_MISMATCH:" + field)
            valid = not errors
            j = (Fraction(score["tp"], score["g"]) - Fraction(score["fp"], score["n"] - score["g"])
                 if valid else None)
            record = {"request_id": row["request_id"], "candidate_id": row["candidate_id"],
                      "root_id": row["root_id"], "source_sha256": row["source_sha256"],
                      "mode": mode, "variant_id": row["variant"]["variant_id"],
                      "slot": slot, "seed": row["seed"], "prompt_sha256": row["prompt_sha256"],
                      "raw_output": response.raw_text, "model_identifier": response.model_identifier,
                      "metadata": metadata, "displayed_gold": row["variant"]["displayed_gold"],
                      "score": score, "partial_score_fraction": str(j) if j is not None else None,
                      "evaluation_valid": valid, "data_errors": errors}
            write_once(run / "responses" / (row["request_id"] + ".json"), record)
            executed.append(row["request_id"])
            print(json.dumps({"completed_calls": len(executed), "request_id": row["request_id"],
                              "valid": valid, "exact": score["exact_match"] if valid else None,
                              "J": str(j) if j is not None else None}), flush=True)
            return score["exact_match"] if valid else None

        for question in questions:
            start = len(executed)
            initial = [evaluate(question, "no_context", 0, slot) for slot in range(3)]
            ground = policy.ground_decision(initial)
            if ground["status"] == "NEEDS_EXTENSION":
                extra = [evaluate(question, "no_context", 0, slot) for slot in (3, 4)]
                ground = policy.ground_decision(initial, extra)
            if ground["ground_pass"] is True:
                for shift in range(6):
                    for slot in FULL_SLOTS:
                        evaluate(question, "full_context", shift, slot)
                status, skipped = "GROUND_PASS_ALL_FULL_EVALUATED", None
            elif ground["ground_pass"] is False:
                status, skipped = "GROUND_FILTERED_FULL_SKIPPED", "GROUND_FILTERED"
            else:
                status, skipped = "EVALUATION_INCOMPLETE_FULL_SKIPPED", "GROUND_EVALUATION_INCOMPLETE"
            row = {"candidate_id": question["candidate_id"], "root_id": question["root_id"],
                   "ground": ground, "status": status,
                   "executed_request_ids": executed[start:], "full_skipped_reason": skipped}
            statuses.append(row)
            write_once(run / "question_status" / (question["candidate_id"] + ".json"), row)
            print(json.dumps({"question_complete": question["candidate_id"],
                              "ground": ground, "status": status,
                              "actual_calls": len(executed) - start}), flush=True)
    except Exception as error:
        write_once(run / "FAILED.json", {"failed_at_utc": now(), "error": str(error),
                                          "traceback": traceback.format_exc(),
                                          "executed_request_ids": executed,
                                          "completed_question_statuses": statuses})
        raise
    finally:
        if backend is not None:
            backend.close()
    verify_model(manifest["solver_identity"])
    verify(manifest["historical_snapshot"])
    write_once(run / "QUESTION_STATUS.json", {"questions": statuses})
    write_once(run / "COMPLETE.json", {
        "completed_at_utc": now(), "executed_request_ids": executed,
        "actual_call_count": len(executed), "response_count": len(executed),
        "maximum_preplanned_slots": len(requests),
        "unused_preplanned_slots": len(requests) - len(executed),
        "model_files_stat_unchanged_after_run": True,
        "historical_files_unchanged": True, "training_started": False,
        "training_promotion": False, "new_generation_calls": 0})
    print(json.dumps({"status": "COMPLETE", "actual_calls": len(executed),
                      "ground_pass_count": sum(row["ground"]["ground_pass"] is True
                                               for row in statuses)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "live"))
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--run", type=Path, default=HERE / "run")
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight(args.pool, args.run)
    else:
        live(args.run)


if __name__ == "__main__":
    main()
