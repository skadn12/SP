"""Evaluate the frozen remote-model ordering pool with an adaptive ground gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
RUN = HERE / "run"
POOL = DELIVERIES / "h4_ordering_large_model_generation_v1/READY_FOR_EVALUATION.json"
SOURCE = DELIVERIES / "h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
SOURCE_SHA256 = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
BASE_RUNNER = DELIVERIES / "h4_event_ledger_m0_pilot_v1/run.py"
POLICY = DELIVERIES / "h4_ordering_ground_full_selection_v1/selection.py"
OLD_IDENTITY = DELIVERIES / "h4_multiselect_partial_admission_v1/followup_model_identity.json"
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
EXPECTED_VENV = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b")
BASE_ORDERS = ("412356", "345162", "256431", "624513", "531624", "163245")
DISPLAY_ORDERS = BASE_ORDERS + tuple(order[::-1] for order in BASE_ORDERS)
SEED_BASE = 2026100600

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_stat(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def check_model_stats(identity):
    for row in identity["files"]:
        if file_stat(row["path"]) != {key: row[key] for key in file_stat(row["path"])}:
            raise RuntimeError("MODEL_FILE_CHANGED:" + row["path"])


def current_model_identity():
    previous = read(OLD_IDENTITY)
    if Path(previous["model_path"]).resolve() != MODEL.resolve():
        raise RuntimeError("MODEL_IDENTITY_PATH_MISMATCH")
    files = []
    for old in previous["files"]:
        path = Path(old["path"])
        before = file_stat(path)
        digest = sha(path)
        if file_stat(path) != before:
            raise RuntimeError("MODEL_CHANGED_WHILE_HASHING:" + str(path))
        if digest != old["sha256"] or before["bytes"] != old["bytes"]:
            raise RuntimeError("MODEL_DIFFERS_FROM_FIXED_REVISION:" + str(path))
        files.append({"path": str(path), "name": old["name"], "sha256": digest, **before})
    return {
        "model_path": str(MODEL), "model_identifier": "Qwen/Qwen3-4B",
        "model_resolved_path": str(MODEL.resolve()), "audited_at_utc": now(),
        "model_and_tokenizer_fingerprint_sha256": previous["model_and_tokenizer_fingerprint_sha256"],
        "fingerprint_basis": "All critical file SHA-256 and byte sizes freshly match the prior fixed-revision audit.",
        "fresh_critical_files_sha256": object_sha([{key: row[key] for key in ("name", "bytes", "sha256")}
                                                  for row in files]),
        "weight_revision": previous["weight_revision"],
        "evidence_strength": "CONTEMPORANEOUS_CRITICAL_FILE_HASHES_MATCH_PREVIOUS_FIXED_REVISION_AUDIT",
        "prior_identity_path": str(OLD_IDENTITY), "prior_identity_sha256": sha(OLD_IDENTITY),
        "files": files,
    }


def historical_snapshot():
    roots = [POOL.parent, POLICY.parent,
             DELIVERIES / "h4_259192_manual_qa_v2",
             DELIVERIES / "h4_multiselect_partial_admission_v1",
             DELIVERIES / "h4_multiselect_ground_full_selection_v1"]
    return {str(path): sha(path) for root in roots for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts}


def verify_files(mapping):
    for name, expected in mapping.items():
        if sha(name) != expected:
            raise RuntimeError("FROZEN_FILE_CHANGED:" + name)


def preflight():
    if Path(sys.prefix) != EXPECTED_VENV:
        raise RuntimeError("USE_LOCAL4B_VENV")
    if RUN.exists():
        raise RuntimeError("RUN_ALREADY_EXISTS_NO_OVERWRITE")
    if sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("SOURCE_HASH_MISMATCH")
    pool = read(POOL)
    questions = pool["questions"]
    expected_ids = [f"259192-qwen36-order-{number:02d}" for number in range(2, 7)]
    if [q["candidate_id"] for q in questions] != expected_ids:
        raise RuntimeError("FIVE_READY_CANDIDATES_REQUIRED")
    keys = [f"I{number}" for number in range(1, 7)]
    source = read(SOURCE)
    if source["root_id"] != "cc-train-episode-259192" or len(source["turns"]) != 188:
        raise RuntimeError("FROZEN_ROOT_OR_TURN_COUNT_MISMATCH")
    for question in questions:
        if (question["question_type"] != "ordering" or list(question["items"]) != keys
                or question["gold"] != keys or question["root_id"] != source["root_id"]):
            raise RuntimeError("READY_QUESTION_CONTRACT_MISMATCH")
    snapshot = historical_snapshot()
    identity = current_model_identity()
    probe = load_module("ordering_large_model_original_probe", BASE_RUNNER)
    policy = load_module("ordering_large_model_policy", POLICY)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    dialogue = probe.contract.dialogue_rows(source)
    config = probe.solver.generation_config("ordering")
    for key, expected in {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                          "enable_thinking": False,
                          "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                          "structured_output_version": "finite_choice_set_v1"}.items():
        if config[key] != expected:
            raise RuntimeError("DECODING_CONTRACT_MISMATCH:" + key)
    prompts, requests = {}, []
    for question in questions:
        for mode, indexes in (("no_context", range(5)), ("full_context", range(12))):
            for index in indexes:
                variant_id = f"perm_{index:02d}"
                variant = probe.variant(question, variant_id, [f"I{n}" for n in DISPLAY_ORDERS[index]])
                if len(probe.contract.structured_choices("ordering", variant)) != 720:
                    raise RuntimeError("FINITE_CHOICE_SET_MISMATCH")
                policy.ordering_metrics(question["gold"], question["gold"])
                prompt = probe.contract.build_prompt(tokenizer, question, variant, dialogue, context_mode=mode)
                # Metadata perturbations must not affect the Solver-visible prompt.
                public = {key: question[key] for key in ("question_type", "question")}
                public_variant = {"items": variant["items"]}
                if prompt != probe.contract.build_prompt(tokenizer, public, public_variant, dialogue, context_mode=mode):
                    raise RuntimeError("PROMPT_METADATA_LEAK")
                token_count = len(tokenizer.encode(prompt))
                if token_count + config["max_tokens"] > 16384:
                    raise RuntimeError("PROMPT_EXCEEDS_MODEL_CONTEXT")
                request_id = f"{question['candidate_id']}__{variant_id}__{mode}__s00"
                prompts[request_id] = prompt
                requests.append({
                    "request_id": request_id, "candidate_id": question["candidate_id"],
                    "question_type": "ordering", "variant": variant, "mode": mode, "slot": 0,
                    "seed": SEED_BASE + len(requests),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": token_count,
                })
    if len(requests) != 85 or len(prompts) != 85:
        raise RuntimeError("FROZEN_REQUEST_PLAN_MISMATCH")
    check_model_stats(identity)
    verify_files(snapshot)
    RUN.mkdir()
    write_once(RUN / "questions.json", {"candidates": questions})
    write_once(RUN / "gold.json", {q["candidate_id"]: q["gold"] for q in questions})
    shutil.copy2(SOURCE, RUN / "source.json")
    for request_id, prompt in prompts.items():
        path = RUN / "prompts" / (request_id + ".txt")
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(prompt)
    write_once(RUN / "solver_identity.json", identity)
    package_names = ("torch", "vllm", "transformers", "tokenizers", "xgrammar")
    code_paths = [Path(__file__), BASE_RUNNER, POLICY, Path(probe.contract.__file__),
                  Path(probe.solver.__file__), probe.WORKTREE / "scripts/current_method/restart_n5_spell/backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/vllm_backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py",
                  Path(probe.solver.__file__).parent / "__init__.py"]
    write_once(RUN / "manifest.json", {
        "status": "PREFLIGHT_PASS", "prepared_at_utc": now(), "model": str(MODEL),
        "execution": "ground_first_adaptive_v1", "candidate_count": 5, "request_count": 85,
        "request_count_semantics": "Maximum preplanned slots, not actual calls.",
        "rollouts_per_position": 1, "temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
        "enable_thinking": False, "generation_config": config,
        "max_model_len": 16384, "gpu_memory_utilization": 0.80,
        "generator_provenance": pool["provenance"], "solver_identity": identity,
        "policy_version": policy.VERSION, "ground_rule_version": policy.GROUND_VERSION,
        "display_orders": list(DISPLAY_ORDERS), "seed_schedule": "preplanned_request_index_v1",
        "input_sha256": {name: sha(RUN / name) for name in ("questions.json", "gold.json", "source.json")},
        "pool_path": str(POOL), "pool_sha256": sha(POOL),
        "source_paths": {"questions": str(POOL), "source": str(SOURCE)},
        "code_sha256": {str(path): sha(path) for path in code_paths},
        "historical_snapshot": snapshot, "requests": requests,
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "environment": {name: os.environ.get(name) for name in
                        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_USE_FLASHINFER_SAMPLER", "TOKENIZERS_PARALLELISM")},
        "authorization": "User requested the next step after large-model generation: Solver screening of the five ready ordering candidates.",
        "training_started": False, "training_promotion": False, "new_generation_calls": 0,
    })
    print(json.dumps({"status": "PREFLIGHT_PASS", "questions": 5, "maximum_calls": 85,
                      "maximum_prompt_tokens": max(row["prompt_tokens"] for row in requests),
                      "source_turns": len(dialogue), "model": str(MODEL)}), flush=True)


def live():
    manifest = read(RUN / "manifest.json")
    if (RUN / "STARTED.json").exists():
        raise RuntimeError("ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    if Path(sys.prefix) != EXPECTED_VENV or sys.prefix != manifest["python_prefix"]:
        raise RuntimeError("PYTHON_RUNTIME_CHANGED")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("PACKAGE_CHANGED:" + name)
    verify_files({str(RUN / name): digest for name, digest in manifest["input_sha256"].items()})
    verify_files(manifest["code_sha256"])
    verify_files(manifest["historical_snapshot"])
    check_model_stats(manifest["solver_identity"])
    probe = load_module("ordering_large_model_live_probe", BASE_RUNNER)
    policy = load_module("ordering_large_model_live_policy", POLICY)
    if probe.solver.generation_config("ordering") != manifest["generation_config"]:
        raise RuntimeError("DECODING_CHANGED")
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    questions = read(RUN / "questions.json")["candidates"]
    prompts = {}
    for row in manifest["requests"]:
        prompt = (RUN / "prompts" / (row["request_id"] + ".txt")).read_text(encoding="utf-8")
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise RuntimeError("PROMPT_CHANGED:" + row["request_id"])
        prompts[row["request_id"]] = prompt
    requests = {(row["candidate_id"], row["mode"], row["variant"]["variant_id"]): row
                for row in manifest["requests"]}
    executed, statuses = [], []
    write_once(RUN / "STARTED.json", {"started_at_utc": now(), "maximum_request_count": 85,
                                     "manifest_sha256": sha(RUN / "manifest.json")})
    backend = None
    try:
        backend = probe.solver.create_backend(model_path=MODEL, tokenizer_path=MODEL)

        def evaluate(question, mode, index):
            row = requests[question["candidate_id"], mode, f"perm_{index:02d}"]
            request = SolverRequest(
                request_id=row["request_id"], qa_id=row["candidate_id"], evaluation_mode=mode,
                order_id=row["variant"]["variant_id"], target_slot=0, retry_index=0,
                sampling_seed=row["seed"], options=row["variant"]["items"], canonical_answer_text="",
                prompt_sha256=row["prompt_sha256"], prompt=prompts[row["request_id"]],
            )
            response = probe.solver._structured_generate(backend, request, question_type="ordering")
            score = probe.contract.score_output(response.raw_text, question, row["variant"])
            valid = (response.metadata.get("finish_reason") in ("stop", "eos")
                     and score.get("scored") is True)
            metrics = None
            errors = []
            if valid:
                try:
                    metrics = policy.ordering_metrics(score["canonical_prediction"], question["gold"])
                except ValueError as error:
                    valid = False
                    errors.append(str(error))
            if not valid and not errors:
                errors.append("INCOMPLETE_OR_UNPARSEABLE_RESPONSE")
            record = {
                "request_id": row["request_id"], "candidate_id": row["candidate_id"],
                "mode": mode, "variant_id": row["variant"]["variant_id"], "seed": row["seed"],
                "prompt_sha256": row["prompt_sha256"], "raw_output": response.raw_text,
                "model_identifier": response.model_identifier,
                "metadata": {**response.metadata, "enable_thinking": False},
                "displayed_gold": row["variant"]["displayed_gold"], "score": score,
                "ordering_metrics": metrics, "evaluation_valid": valid, "data_errors": errors,
            }
            write_once(RUN / "responses" / (row["request_id"] + ".json"), record)
            executed.append(row["request_id"])
            print(json.dumps({"completed_calls": len(executed), "request_id": row["request_id"],
                              "valid": valid, "exact": metrics["exact_match"] if valid else None,
                              "J": metrics["partial_score"]["fraction"] if valid else None}), flush=True)
            return metrics["exact_match"] if valid else None

        for question in questions:
            start = len(executed)
            initial = [evaluate(question, "no_context", index) for index in range(3)]
            ground = policy.ground_decision(initial)
            if ground["status"] == "NEEDS_EXTENSION":
                extra = [evaluate(question, "no_context", index) for index in (3, 4)]
                ground = policy.ground_decision(initial, extra)
            if ground["ground_pass"] is True:
                # No early rejection: all fixed full displays are evaluated.
                for index in range(12):
                    evaluate(question, "full_context", index)
                status, skipped = "GROUND_PASS_ALL_FULL_EVALUATED", None
            elif ground["ground_pass"] is False:
                status, skipped = "GROUND_FILTERED_FULL_SKIPPED", "GROUND_FILTERED"
            else:
                status, skipped = "EVALUATION_INCOMPLETE_FULL_SKIPPED", "GROUND_EVALUATION_INCOMPLETE"
            row = {"candidate_id": question["candidate_id"], "ground": ground, "status": status,
                   "executed_request_ids": executed[start:], "full_skipped_reason": skipped}
            statuses.append(row)
            write_once(RUN / "question_status" / (question["candidate_id"] + ".json"), row)
            print(json.dumps({"question_complete": question["candidate_id"], "ground": ground,
                              "status": status, "actual_calls": len(executed) - start}), flush=True)
    except Exception as error:
        write_once(RUN / "FAILED.json", {"failed_at_utc": now(), "error": str(error),
                                         "traceback": traceback.format_exc(),
                                         "executed_request_ids": executed,
                                         "completed_question_statuses": statuses})
        raise
    finally:
        if backend is not None:
            backend.close()
    check_model_stats(manifest["solver_identity"])
    verify_files(manifest["historical_snapshot"])
    write_once(RUN / "QUESTION_STATUS.json", {"questions": statuses})
    write_once(RUN / "COMPLETE.json", {
        "completed_at_utc": now(), "executed_request_ids": executed,
        "actual_call_count": len(executed), "response_count": len(executed),
        "maximum_preplanned_slots": 85, "unused_preplanned_slots": 85 - len(executed),
        "model_files_stat_unchanged_after_run": True, "historical_files_unchanged": True,
        "training_started": False, "training_promotion": False, "new_generation_calls": 0,
    })
    print(json.dumps({"status": "COMPLETE", "actual_calls": len(executed),
                      "ground_pass_count": sum(row["ground"]["ground_pass"] is True for row in statuses)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "live"))
    args = parser.parse_args()
    {"preflight": preflight, "live": live}[args.phase]()
