from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "postweek_source_citation_control_v1"
MODEL_MANIFEST = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json")
EXECUTION_ENV = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
EXPECTED_RUNTIME = {"torch": "2.11.0", "transformers": "5.9.0", "tokenizers": "0.22.2", "vllm": "0.21.0", "jsonschema": "4.26.0"}
sys.dont_write_bytecode = True


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_binding(row: dict) -> None:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise RuntimeError("HASH_BINDING_MISMATCH:" + str(path))


def versions() -> dict:
    return {name: importlib.metadata.version(name) for name in EXPECTED_RUNTIME}


def cpu_verify(*, compile_schemas: bool = True):
    """Full pre-run integrity and CPU-only schema/runtime validation."""
    if Path(sys.prefix).resolve() != Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b").resolve():
        raise RuntimeError("USE_FROZEN_LOCAL4B_ENVIRONMENT")
    if versions() != EXPECTED_RUNTIME:
        raise RuntimeError("RUNTIME_VERSION_MISMATCH")
    manifest = read(HERE / "EXECUTION_MANIFEST.json")
    if manifest["status"] != "EXECUTION_AUTHORIZED_NOT_STARTED":
        raise RuntimeError("EXECUTION_MANIFEST_STATUS_INVALID")
    for row in manifest["bindings"]:
        check_binding(row)
    audit = read(HERE / "RUNNER_PREFLIGHT.json")
    if audit.get("status") != "PASS_READ_ONLY" or audit.get("model_inference_performed") is not False:
        raise RuntimeError("INDEPENDENT_RUNNER_PREFLIGHT_FAILED")
    audited_files = audit.get("audited_files", [])
    expected_audit_paths = {str((HERE / name).resolve()) for name in (
        "run_control.py", "score_control.py", "unblind_scores.py",
        "prepare_blind_review.py", "finalize_execution.py", "verify_execution.py", "USER_EXECUTION_DECISION.json")}
    if not expected_audit_paths.issubset({row["path"] for row in audited_files}):
        raise RuntimeError("INDEPENDENT_AUDIT_CORE_FILES_MISSING")
    for row in audited_files:
        check_binding(row)
    auth = read(HERE / "USER_EXECUTION_DECISION.json")
    if auth["authorized_scope"]["maximum_model_requests"] != 24 or auth["authorized_scope"]["automatic_retries"] != 0:
        raise RuntimeError("EXECUTION_AUTH_SCOPE_INVALID")
    parent_freeze = read(PARENT / "FREEZE.json")
    if parent_freeze.get("status") != "FROZEN_NOT_EXECUTED" or parent_freeze.get("execution_authorized") is not False:
        raise RuntimeError("PARENT_FREEZE_DRIFT")
    if sha(PARENT / "FINAL_MANIFEST.json") != parent_freeze["manifest_binding"]["sha256"]:
        raise RuntimeError("PARENT_MANIFEST_HASH_DRIFT")
    if sha(PARENT / "FREEZE.json") != manifest["parent_freeze_sha256"]:
        raise RuntimeError("PARENT_FREEZE_HASH_DRIFT")
    requests = read(PARENT / "REQUESTS.json")["requests"]
    run_order = read(PARENT / "RUN_ORDER.json")["run_order_request_ids"]
    if len(requests) != 24 or len(run_order) != 24 or len(set(run_order)) != 24:
        raise RuntimeError("REQUEST_DENOMINATOR_INVALID")
    by_id = {row["request_id"]: row for row in requests}
    if set(by_id) != set(run_order):
        raise RuntimeError("RUN_ORDER_REQUEST_SET_MISMATCH")
    for request_id in run_order:
        row = by_id[request_id]
        check_binding({"path": row["prompt_path"], "sha256": row["prompt_sha256"], "bytes": Path(row["prompt_path"]).stat().st_size})
        check_binding({"path": row["schema_path"], "sha256": row["schema_sha256"], "bytes": Path(row["schema_path"]).stat().st_size})
        if row["max_output_tokens"] != 1024 or row["max_model_len"] != 16384:
            raise RuntimeError("REQUEST_DECODER_CONTRACT_DRIFT")
    endpoint = read(MODEL_MANIFEST)
    if endpoint["model_role"] != "B1" or endpoint["status"] != "BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW":
        raise RuntimeError("B1_ENDPOINT_MANIFEST_INVALID")
    if endpoint["model_directory"] != "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model":
        raise RuntimeError("MODEL_PATH_DRIFT")
    proof = read(Path(endpoint["endpoint_proof"]["path"]))
    if proof["model_role"] != "B1" or proof["fixed_final_optimizer_step"] != 56 or proof["completed_optimizer_steps"] != 56:
        raise RuntimeError("B1_ENDPOINT_PROOF_INVALID")
    model_bindings = endpoint["model_bindings"] + endpoint["tokenizer_bindings"]
    for row in model_bindings:
        check_binding(row)
    if compile_schemas:
        import xgrammar as xgr
        from transformers import AutoTokenizer
        from vllm import SamplingParams, sampling_params
        from vllm.v1.structured_output.backend_xgrammar import has_xgrammar_unsupported_json_features
        tokenizer_path = Path(endpoint["tokenizer_directory"])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
        compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=False)
        compiled = set()
        for row in requests:
            schema = read(Path(row["schema_path"]))
            if row["schema_sha256"] in compiled:
                continue
            if has_xgrammar_unsupported_json_features(schema):
                raise RuntimeError("XGRAMMAR_UNSUPPORTED_SCHEMA:" + row["request_id"])
            compiler.compile_json_schema(json.dumps(schema, ensure_ascii=False), any_whitespace=True)
            params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024,
                seed=row["sampling_seed"], structured_outputs=sampling_params.StructuredOutputsParams(json=schema))
            if getattr(params, "structured_outputs", None) is None:
                raise RuntimeError("STRUCTURED_OUTPUT_NOT_BOUND")
            compiled.add(row["schema_sha256"])
        if len(compiled) != 3:
            raise RuntimeError("DISTINCT_SCHEMA_COUNT_MISMATCH")
    return manifest, requests, run_order, endpoint


def fsync_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def bind(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def live() -> None:
    for key, value in EXECUTION_ENV.items():
        os.environ[key] = value
    manifest, requests, run_order, endpoint = cpu_verify()
    output_root = HERE / "outputs"
    if output_root.exists():
        raise RuntimeError("OUTPUT_DIRECTORY_EXISTS_NO_OVERWRITE_OR_RETRY")
    by_id = {row["request_id"]: row for row in requests}
    start = time.perf_counter()
    run_dir = output_root / "B1"
    responses_dir = run_dir / "responses"
    run_dir.mkdir(parents=True)
    responses_dir.mkdir()
    freeze_path = HERE / "EXECUTION_MANIFEST.json"
    fsync_json(run_dir / "STARTED.json", {
        "status": "STARTED", "started_at_unix": time.time(),
        "execution_manifest": bind(freeze_path), "request_count": 24,
        "execution_order_sha256": sha(PARENT / "RUN_ORDER.json"),
        "gold_accessed_by_runner": False,
    })
    model = None
    submitted: list[str] = []
    completed: list[str] = []
    failure = None
    model_load_seconds = None
    try:
        backend_binding = next(row for row in endpoint["code_bindings"] if Path(row["path"]).name == "vllm_backends.py")
        backend_path = Path(backend_binding["path"])
        repo_root = backend_path.parents[3]
        for row in endpoint["code_bindings"]:
            check_binding(row)
        sys.path.insert(0, str(repo_root))
        from scripts.current_method.restart_n5_spell import vllm_backends
        if Path(vllm_backends.__file__).resolve() != backend_path.resolve():
            raise RuntimeError("BACKEND_IMPORT_PATH_MISMATCH")
        model = vllm_backends.VLLMSolverBackend(
            model_path=Path(endpoint["model_directory"]),
            tokenizer_path=Path(endpoint["tokenizer_directory"]), max_model_len=16384,
            temperature=0.8, top_p=0.9, max_tokens=1024, gpu_memory_utilization=0.8)
        model_load_seconds = model.load_latency_seconds
        if model.model_identifier != endpoint["model_directory"]:
            raise RuntimeError("MODEL_IDENTIFIER_MISMATCH")
        from vllm import SamplingParams, sampling_params
        for position, request_id in enumerate(run_order):
            row = by_id[request_id]
            prompt_path, schema_path = Path(row["prompt_path"]), Path(row["schema_path"])
            if sha(prompt_path) != row["prompt_sha256"] or sha(schema_path) != row["schema_sha256"]:
                raise RuntimeError("PROMPT_OR_SCHEMA_HASH_MISMATCH:" + request_id)
            schema = read(schema_path)
            params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1024,
                seed=row["sampling_seed"], structured_outputs=sampling_params.StructuredOutputsParams(json=schema))
            prompt = prompt_path.read_text()
            submit_path = run_dir / (f"{position:02d}_SUBMITTED.json")
            fsync_json(submit_path, {
                "position": position, "request_id": request_id,
                "prompt_sha256": row["prompt_sha256"], "schema_sha256": row["schema_sha256"],
                "seed": row["sampling_seed"], "sampling": {
                    "temperature": 0.8, "top_p": 0.9, "max_tokens": 1024,
                    "structured_json": True, "prefix_caching": False,
                    "automatic_retry": False,
                }, "submitted_at_unix": time.time(),
            })
            submitted.append(request_id)
            outputs = model.engine.generate([prompt], [params], use_tqdm=False)
            output = outputs[0] if outputs else None
            completions = list(output.outputs) if output is not None else []
            first = completions[0] if completions else None
            record = {
                "position": position, "request_id": request_id,
                "candidate_id": row["candidate_id"], "root_id": row["root_id"],
                "prompt_sha256": row["prompt_sha256"], "schema_sha256": row["schema_sha256"],
                "sampling_seed": row["sampling_seed"],
                "status": "RETURNED" if first is not None else "MISSING_OUTPUT",
                "raw_output": first.text if first is not None else None,
                "prompt_token_ids": list(output.prompt_token_ids) if output is not None else None,
                "all_completions": [{"text": c.text, "token_ids": list(c.token_ids),
                    "finish_reason": c.finish_reason, "stop_reason": c.stop_reason} for c in completions],
                "metadata": {"finish_reason": first.finish_reason if first is not None else None,
                    "stop_reason": first.stop_reason if first is not None else None,
                    "completion_tokens": len(first.token_ids) if first is not None else None,
                    "structured_json": True, "temperature": 0.8, "top_p": 0.9,
                    "max_tokens": 1024, "gold_accessed_by_runner": False},
            }
            fsync_json(responses_dir / (request_id + ".json"), record)
            if first is not None:
                completed.append(request_id)
            if len(outputs) != 1 or len(completions) != 1:
                raise RuntimeError("OUTPUT_COUNT_CONTRACT_MISMATCH_AFTER_RAW_COMMIT")
            if len(output.prompt_token_ids) != row["prompt_tokens"]:
                raise RuntimeError("PROMPT_TOKEN_COUNT_MISMATCH_AFTER_RAW_COMMIT")
            if getattr(output, "prompt", prompt) != prompt:
                raise RuntimeError("PROMPT_BYTES_MISMATCH_AFTER_RAW_COMMIT")
            print(json.dumps({"position": position + 1, "planned": 24,
                "returned": len(completed), "request_id": request_id}), flush=True)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if model is not None:
            try:
                model.close()
            except BaseException as exc:
                failure = failure or {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        for position, request_id in enumerate(run_order):
            path = responses_dir / (request_id + ".json")
            if not path.exists():
                fsync_json(path, {"position": position, "request_id": request_id,
                    "candidate_id": by_id[request_id]["candidate_id"], "root_id": by_id[request_id]["root_id"],
                    "status": "BACKEND_ERROR" if request_id in submitted else "NOT_SUBMITTED",
                    "usage_status": "UNKNOWN_NOT_ZERO" if request_id in submitted else "NO_REQUEST_SENT", "raw_output": None,
                    "failure": failure})
        files = [bind(p) for p in sorted(responses_dir.glob("*.json"))]
        returned = sum(read(Path(x["path"])).get("status") == "RETURNED" for x in files)
        status = "COMMITTED" if len(files) == 24 else "INCOMPLETE_COMMIT"
        fsync_json(run_dir / "RAW_RESPONSE_FREEZE.json", {
            "status": status, "role": "B1", "planned_requests": 24,
            "submitted_requests": len(submitted), "returned_requests": returned,
            "request_ids_in_frozen_order": run_order,
            "automatic_retry": False, "files": files,
            "execution_manifest": bind(HERE / "EXECUTION_MANIFEST.json"),
            "parent_freeze": bind(PARENT / "FREEZE.json"),
            "failure": failure, "wall_seconds": time.perf_counter() - start,
            "model_load_seconds": model_load_seconds,
        })
        fsync_json(run_dir / ("COMPLETE.json" if failure is None and returned == 24 else "FAILED.json"), {
            "role": "B1", "status": "COMPLETE" if failure is None and returned == 24 else "FAILED",
            "failure": failure, "submitted": len(submitted), "returned": returned,
        })
    if failure:
        raise RuntimeError("RUN_FAILED_RAW_RESPONSES_COMMITTED:" + failure["error"])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"verify", "live"}:
        raise SystemExit("Usage: run_control.py verify|live")
    for key, value in EXECUTION_ENV.items():
        os.environ[key] = value
    if sys.argv[1] == "verify":
        result = cpu_verify()
        print(json.dumps({"status": "PASS_CPU_ONLY", "request_count": 24,
            "run_order_count": len(result[2]), "model_weights_loaded": False}))
    else:
        live()
