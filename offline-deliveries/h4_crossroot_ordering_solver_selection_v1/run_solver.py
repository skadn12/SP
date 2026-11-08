"""Freeze and evaluate audited six-item ordering questions across source roots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
RUN = HERE / "run"
POOL = DELIVERIES / "h4_crossroot_ordering_label_rebind_v1/READY_FOR_EVALUATION.json"
AUDIT = POOL.parent / "LABEL_MAPPING_AUDIT.json"
BATCH = DELIVERIES / "h4_crossroot_ordering_batch_v2"
BASE_RUNNER = DELIVERIES / "h4_event_ledger_m0_pilot_v1/run.py"
POLICY = DELIVERIES / "h4_ordering_ground_full_selection_v1/selection.py"
PRIOR_RUNNER = DELIVERIES / "h4_ordering_large_model_solver_selection_v1/run_solver.py"
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
EXPECTED_VENV = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b")
BASE_ORDERS = ("412356", "345162", "256431", "624513", "531624", "163245")
DISPLAY_ORDERS = BASE_ORDERS + tuple(order[::-1] for order in BASE_ORDERS)
SEED_BASE = 2026100700

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_module("crossroot_prior_runner", PRIOR_RUNNER)
read = prior.read
sha = prior.sha
write_once = prior.write_once
file_stat = prior.file_stat


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise RuntimeError("FROZEN_FILE_CHANGED:" + path)


def verify_model(identity):
    for row in identity["files"]:
        actual = file_stat(row["path"])
        if actual != {key: row[key] for key in actual}:
            raise RuntimeError("MODEL_FILE_STAT_CHANGED:" + row["path"])


def source_path(root_id):
    return BATCH / root_id / "source.json"


def checked_inputs():
    pool, audit = read(POOL), read(AUDIT)
    questions = pool["questions"]
    if (pool.get("status") != "CONTENT_AUDITED_LABEL_REBOUND_NOT_SOLVER_SCREENED"
            or pool.get("candidate_count") != len(questions) or len(questions) != 2
            or audit.get("candidate_count") != len(questions)):
        raise RuntimeError("AUDITED_TWO_CANDIDATE_POOL_REQUIRED")
    provenance = pool["provenance"]
    if (provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B"
            or provenance.get("returned_model") != "Qwen/Qwen3.6-35B-A3B"
            or provenance.get("codex_role") != "DETERMINISTIC_LABEL_REBIND_ONLY"):
        raise RuntimeError("GENERATOR_PROVENANCE_MISMATCH")
    audit_rows = {row["derived_candidate_id"]: row for row in audit["candidates"]}
    ids = [q["candidate_id"] for q in questions]
    if len(ids) != len(set(ids)) or set(ids) != set(audit_rows):
        raise RuntimeError("POOL_AUDIT_ID_MISMATCH")
    keys = [f"I{n}" for n in range(1, 7)]
    policy = load_module("crossroot_input_policy", POLICY)
    sources = {}
    for question in questions:
        qid, root_id = question["candidate_id"], question["root_id"]
        row = audit_rows[qid]
        path = source_path(root_id)
        if (row["root_id"] != root_id or row["source_sha256"] != sha(path)
                or row["original_model_provenance"]["returned_model"] != provenance["returned_model"]):
            raise RuntimeError("CANDIDATE_SOURCE_OR_MODEL_MISMATCH:" + qid)
        source = read(path)
        if (source["root_id"] != root_id or len(source["turns"]) != row["source_turn_count"]
                or row["original_files_sha256"].get(str(path)) != sha(path)):
            raise RuntimeError("SOURCE_ROOT_OR_TURN_COUNT_MISMATCH:" + qid)
        verify(row["original_files_sha256"])
        if (question["question_type"] != "ordering" or list(question["items"]) != keys
                or len(question["gold"]) != 6 or set(question["gold"]) != set(keys)
                or set(question["item_evidence"]) != set(keys)):
            raise RuntimeError("SIX_ITEM_ORDERING_CONTRACT_MISMATCH:" + qid)
        policy.ordering_metrics(question["gold"], question["gold"])
        refs = {turn["turn_ref"]: turn for turn in source["turns"]}
        for evidence in question["item_evidence"].values():
            if (evidence["turn_ref"] not in refs
                    or evidence["quote"] not in refs[evidence["turn_ref"]]["text"]):
                raise RuntimeError("EVIDENCE_SOURCE_MISMATCH:" + qid)
        sources[root_id] = {"path": path, "source": source, "sha256": sha(path),
                            "turn_count": len(source["turns"])}
    if len(sources) != len(questions):
        raise RuntimeError("TWO_DISTINCT_ROOTS_REQUIRED")
    return pool, audit, questions, sources


def preflight():
    if Path(sys.prefix) != EXPECTED_VENV:
        raise RuntimeError("USE_LOCAL4B_VENV")
    if RUN.exists():
        raise RuntimeError("RUN_ALREADY_EXISTS_NO_OVERWRITE")
    pool, audit, questions, sources = checked_inputs()
    identity = prior.current_model_identity()
    probe = load_module("crossroot_preflight_probe", BASE_RUNNER)
    policy = load_module("crossroot_preflight_policy", POLICY)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    config = probe.solver.generation_config("ordering")
    required_config = {"temperature": 0.8, "top_p": 0.9, "max_tokens": 32,
                       "enable_thinking": False,
                       "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)",
                       "structured_output_version": "finite_choice_set_v1"}
    if any(config.get(key) != value for key, value in required_config.items()):
        raise RuntimeError("DECODING_CONTRACT_MISMATCH")
    prompts, requests = {}, []
    for question in questions:
        root_id = question["root_id"]
        dialogue = probe.contract.dialogue_rows(sources[root_id]["source"])
        for mode, indexes in (("no_context", range(5)), ("full_context", range(12))):
            for index in indexes:
                variant_id = f"perm_{index:02d}"
                variant = probe.variant(question, variant_id,
                                        [f"I{n}" for n in DISPLAY_ORDERS[index]])
                if len(probe.contract.structured_choices("ordering", variant)) != 720:
                    raise RuntimeError("FINITE_CHOICE_SET_MISMATCH")
                prompt = probe.contract.build_prompt(tokenizer, question, variant, dialogue,
                                                     context_mode=mode)
                public = {key: question[key] for key in ("question_type", "question")}
                if prompt != probe.contract.build_prompt(tokenizer, public,
                        {"items": variant["items"]}, dialogue, context_mode=mode):
                    raise RuntimeError("PROMPT_METADATA_LEAK")
                tokens = len(tokenizer.encode(prompt))
                if tokens + config["max_tokens"] > 16384:
                    raise RuntimeError("PROMPT_EXCEEDS_MODEL_CONTEXT:" + root_id)
                request_id = f"{question['candidate_id']}__{variant_id}__{mode}__s00"
                prompts[request_id] = prompt
                requests.append({"request_id": request_id,
                                 "candidate_id": question["candidate_id"], "root_id": root_id,
                                 "source_sha256": sources[root_id]["sha256"],
                                 "question_type": "ordering", "variant": variant,
                                 "mode": mode, "slot": 0,
                                 "seed": SEED_BASE + len(requests),
                                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "prompt_tokens": tokens})
    if len(requests) != len(questions) * 17 or len(prompts) != len(requests):
        raise RuntimeError("REQUEST_PLAN_MISMATCH")
    verify_model(identity)
    protected = {str(path): digest for row in audit["candidates"]
                 for path, digest in row["original_files_sha256"].items()}
    protected.update({str(POOL): sha(POOL), str(AUDIT): sha(AUDIT), str(POLICY): sha(POLICY)})
    verify(protected)
    RUN.mkdir()
    write_once(RUN / "questions.json", {"candidates": questions})
    write_once(RUN / "gold.json", {q["candidate_id"]: q["gold"] for q in questions})
    (RUN / "sources").mkdir()
    for root_id, row in sources.items():
        shutil.copy2(row["path"], RUN / "sources" / (root_id + ".json"))
    for request_id, prompt in prompts.items():
        path = RUN / "prompts" / (request_id + ".txt")
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(prompt)
    write_once(RUN / "solver_identity.json", identity)
    packages = ("torch", "vllm", "transformers", "tokenizers", "xgrammar")
    code_paths = [Path(__file__), PRIOR_RUNNER, BASE_RUNNER, POLICY,
                  Path(probe.contract.__file__), Path(probe.solver.__file__),
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/vllm_backends.py",
                  probe.WORKTREE / "scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py"]
    source_inputs = {str(Path("sources") / (root_id + ".json")):
                     sha(RUN / "sources" / (root_id + ".json")) for root_id in sources}
    source_inputs.update({name: sha(RUN / name) for name in ("questions.json", "gold.json")})
    write_once(RUN / "manifest.json", {
        "status": "PREFLIGHT_PASS", "prepared_at_utc": now(),
        "execution": "ground_first_adaptive_v1", "model": str(MODEL),
        "candidate_count": len(questions), "request_count": len(requests),
        "request_count_semantics": "Maximum preplanned slots, not actual calls.",
        "rollouts_per_position": 1, **required_config, "generation_config": config,
        "max_model_len": 16384, "gpu_memory_utilization": 0.80,
        "generator_provenance": pool["provenance"],
        "label_mapping_audit_path": str(AUDIT), "label_mapping_audit_sha256": sha(AUDIT),
        "solver_identity": identity, "policy_version": policy.VERSION,
        "ground_rule_version": policy.GROUND_VERSION,
        "display_orders": list(DISPLAY_ORDERS),
        "seed_schedule": "preplanned_request_index_v1",
        "input_sha256": source_inputs,
        "pool_path": str(POOL), "pool_sha256": sha(POOL),
        "source_paths_by_root": {root_id: str(row["path"]) for root_id, row in sources.items()},
        "source_turn_count_by_root": {root_id: row["turn_count"] for root_id, row in sources.items()},
        "code_sha256": {str(path): sha(path) for path in code_paths},
        "historical_snapshot": protected, "requests": requests,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "environment": {name: os.environ.get(name) for name in
                        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                         "VLLM_USE_FLASHINFER_SAMPLER", "TOKENIZERS_PARALLELISM")},
        "training_started": False, "training_promotion": False,
        "new_generation_calls": 0,
    })
    print(json.dumps({"status": "PREFLIGHT_PASS", "questions": len(questions),
                      "maximum_calls": len(requests),
                      "maximum_prompt_tokens": max(row["prompt_tokens"] for row in requests),
                      "roots": {root_id: row["turn_count"] for root_id, row in sources.items()},
                      "model": str(MODEL)}), flush=True)


def live():
    manifest = read(RUN / "manifest.json")
    if (RUN / "STARTED.json").exists():
        raise RuntimeError("ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    if Path(sys.prefix) != EXPECTED_VENV or sys.prefix != manifest["python_prefix"]:
        raise RuntimeError("PYTHON_RUNTIME_CHANGED")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("PACKAGE_CHANGED:" + name)
    verify({str(RUN / name): digest for name, digest in manifest["input_sha256"].items()})
    verify(manifest["code_sha256"])
    verify(manifest["historical_snapshot"])
    verify_model(manifest["solver_identity"])
    probe = load_module("crossroot_live_probe", BASE_RUNNER)
    policy = load_module("crossroot_live_policy", POLICY)
    if probe.solver.generation_config("ordering") != manifest["generation_config"]:
        raise RuntimeError("DECODING_CHANGED")
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    questions = read(RUN / "questions.json")["candidates"]
    prompts = {}
    for row in manifest["requests"]:
        prompt = (RUN / "prompts" / (row["request_id"] + ".txt")).read_text(encoding="utf-8")
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise RuntimeError("PROMPT_CHANGED:" + row["request_id"])
        if row["source_sha256"] != manifest["input_sha256"][str(
                Path("sources") / (row["root_id"] + ".json"))]:
            raise RuntimeError("REQUEST_SOURCE_BINDING_MISMATCH:" + row["request_id"])
        prompts[row["request_id"]] = prompt
    requests = {(row["candidate_id"], row["mode"], row["variant"]["variant_id"]): row
                for row in manifest["requests"]}
    executed, statuses = [], []
    write_once(RUN / "STARTED.json", {"started_at_utc": now(),
                                     "maximum_request_count": len(requests),
                                     "manifest_sha256": sha(RUN / "manifest.json")})
    backend = None
    try:
        backend = probe.solver.create_backend(model_path=MODEL, tokenizer_path=MODEL)

        def evaluate(question, mode, index):
            row = requests[question["candidate_id"], mode, f"perm_{index:02d}"]
            request = SolverRequest(
                request_id=row["request_id"], qa_id=row["candidate_id"],
                evaluation_mode=mode, order_id=row["variant"]["variant_id"],
                target_slot=0, retry_index=0, sampling_seed=row["seed"],
                options=row["variant"]["items"], canonical_answer_text="",
                prompt_sha256=row["prompt_sha256"], prompt=prompts[row["request_id"]])
            response = probe.solver._structured_generate(backend, request, question_type="ordering")
            score = probe.contract.score_output(response.raw_text, question, row["variant"])
            valid = (response.metadata.get("finish_reason") in ("stop", "eos")
                     and score.get("scored") is True)
            metrics, errors = None, []
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
                "root_id": row["root_id"], "source_sha256": row["source_sha256"],
                "mode": mode, "variant_id": row["variant"]["variant_id"],
                "seed": row["seed"], "prompt_sha256": row["prompt_sha256"],
                "raw_output": response.raw_text,
                "model_identifier": response.model_identifier,
                "metadata": {**response.metadata, "enable_thinking": False},
                "displayed_gold": row["variant"]["displayed_gold"],
                "score": score, "ordering_metrics": metrics,
                "evaluation_valid": valid, "data_errors": errors,
            }
            write_once(RUN / "responses" / (row["request_id"] + ".json"), record)
            executed.append(row["request_id"])
            print(json.dumps({"completed_calls": len(executed),
                              "request_id": row["request_id"], "valid": valid,
                              "exact": metrics["exact_match"] if valid else None,
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
                for index in range(12):
                    evaluate(question, "full_context", index)
                status, skipped = "GROUND_PASS_ALL_FULL_EVALUATED", None
            elif ground["ground_pass"] is False:
                status, skipped = "GROUND_FILTERED_FULL_SKIPPED", "GROUND_FILTERED"
            else:
                status, skipped = "EVALUATION_INCOMPLETE_FULL_SKIPPED", "GROUND_EVALUATION_INCOMPLETE"
            row = {"candidate_id": question["candidate_id"], "root_id": question["root_id"],
                   "ground": ground, "status": status,
                   "executed_request_ids": executed[start:], "full_skipped_reason": skipped}
            statuses.append(row)
            write_once(RUN / "question_status" / (question["candidate_id"] + ".json"), row)
            print(json.dumps({"question_complete": question["candidate_id"],
                              "ground": ground, "status": status,
                              "actual_calls": len(executed) - start}), flush=True)
    except Exception as error:
        write_once(RUN / "FAILED.json", {"failed_at_utc": now(), "error": str(error),
                                         "traceback": traceback.format_exc(),
                                         "executed_request_ids": executed,
                                         "completed_question_statuses": statuses})
        raise
    finally:
        if backend is not None:
            backend.close()
    verify_model(manifest["solver_identity"])
    verify(manifest["historical_snapshot"])
    write_once(RUN / "QUESTION_STATUS.json", {"questions": statuses})
    write_once(RUN / "COMPLETE.json", {
        "completed_at_utc": now(), "executed_request_ids": executed,
        "actual_call_count": len(executed), "response_count": len(executed),
        "maximum_preplanned_slots": len(requests),
        "unused_preplanned_slots": len(requests) - len(executed),
        "model_files_stat_unchanged_after_run": True,
        "historical_files_unchanged": True, "training_started": False,
        "training_promotion": False, "new_generation_calls": 0,
    })
    print(json.dumps({"status": "COMPLETE", "actual_calls": len(executed),
                      "ground_pass_count": sum(row["ground"]["ground_pass"] is True
                                               for row in statuses)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "live"))
    args = parser.parse_args()
    {"preflight": preflight, "live": live}[args.phase]()
