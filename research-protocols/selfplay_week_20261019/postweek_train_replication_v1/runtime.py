"""One-shot, B1-only runtime for the post-week Train-side replication.

The runtime deliberately has no scoring path and never opens private Gold.
CPU preflight verifies the complete frozen endpoint and every generation
input; live writes raw records for the fixed 12-request denominator.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
VENV = REPO / ".venv-local4b"
FREEZE_PATH = HERE / "PREPARATION_FREEZE.json"
REQUESTS_PATH = HERE / "REQUESTS.json"
ADMISSION_PATH = HERE / "SOURCE_ADMISSION.json"
RUN_DIR = HERE / "runs" / "B1"
VLLM_BACKEND_SUFFIX = ("scripts", "current_method", "restart_n5_spell", "vllm_backends.py")
EXEC_ENV = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
PACKAGES = ("torch", "transformers", "tokenizers", "vllm", "jsonschema")
REQUESTS_N = 12
ROOTS_N = 6
BATCH_SIZE = 4
MODEL_LEN = 16384
MAX_TOKENS = 1024

for _key, _value in EXEC_ENV.items():
    os.environ.setdefault(_key, _value)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_bindings(rows, label: str = "BOUND_FILE_CHANGED") -> None:
    require(isinstance(rows, list), "BINDING_LIST_REQUIRED:" + label)
    for row in rows:
        path = Path(row["path"])
        require(path.is_file(), f"{label}:MISSING:{path}")
        require(path.stat().st_size == int(row["bytes"]), f"{label}:BYTES:{path}")
        require(sha(path) == row["sha256"], f"{label}:SHA256:{path}")


def package_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def all_frozen_bindings(freeze: dict) -> list[dict]:
    rows = list(freeze["artifacts"]) + list(freeze["inputs"])
    endpoint = freeze["endpoint"]
    for key in ("model_bindings", "tokenizer_bindings", "code_bindings", "endpoint_evidence_bindings"):
        rows.extend(endpoint[key])
    return rows


def expected_backend(endpoint: dict) -> Path:
    matches = [Path(row["path"]).resolve() for row in endpoint["code_bindings"]
               if Path(row["path"]).name == "vllm_backends.py"]
    require(len(matches) == 1, "FROZEN_BACKEND_BINDING_COUNT")
    backend = matches[0]
    require(backend.parts[-4:] == VLLM_BACKEND_SUFFIX, "FROZEN_BACKEND_SOURCE_PATH_MISMATCH")
    require(backend.is_file(), "FROZEN_BACKEND_MISSING")
    transitive = backend.with_name("backends.py")
    require(any(Path(row["path"]).resolve() == transitive for row in endpoint["code_bindings"]),
            "TRANSITIVE_BACKEND_SOURCE_UNBOUND")
    for init_path in (backend.parents[1] / "__init__.py", backend.parent / "__init__.py"):
        require(any(Path(row["path"]).resolve() == init_path.resolve() for row in endpoint["code_bindings"]),
                "BACKEND_PACKAGE_INIT_UNBOUND")
    return backend


def verify_backend_prefix_cache_disabled(path: Path) -> None:
    """Verify the actual imported backend passes the frozen cache setting."""
    calls = [node for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "LLM"]
    require(len(calls) == 1, "BACKEND_LLM_CALL_COUNT_CHANGED")
    settings = {item.arg: item.value for item in calls[0].keywords if item.arg}
    value = settings.get("enable_prefix_caching")
    require(isinstance(value, ast.Constant) and value.value is False,
            "BACKEND_PREFIX_CACHING_NOT_EXPLICITLY_DISABLED")


def audited_file_map(audit: dict) -> dict[str, dict]:
    result = {}
    for row in audit.get("audited_files", []):
        result[str(Path(row["path"]).resolve())] = row
    return result


def verify_audit(audit_path: Path, *, require_admission: bool) -> dict:
    audit = read(audit_path)
    require(audit.get("status") == "PASS", "INDEPENDENT_PREFLIGHT_NOT_PASS")
    records = audited_file_map(audit)
    mandatory = [FREEZE_PATH, REQUESTS_PATH, Path(__file__)]
    if require_admission:
        mandatory.append(ADMISSION_PATH)
    for path in mandatory:
        expected = binding(path)
        row = records.get(str(path.resolve()))
        require(row is not None and row.get("sha256") == expected["sha256"] and
                row.get("bytes") == expected["bytes"], "AUDIT_DOES_NOT_BIND_CURRENT_BYTES:" + str(path))
    # A live audit must also bind all source-admission inputs and the endpoint
    # manifests/model files that were independently checked before launch.
    freeze = read(FREEZE_PATH)
    for row in all_frozen_bindings(freeze):
        path = str(Path(row["path"]).resolve())
        seen = records.get(path)
        require(seen is not None and seen.get("sha256") == row["sha256"] and
                seen.get("bytes") == row["bytes"], "AUDIT_OMITS_FROZEN_BINDING:" + path)
    return audit


def import_v11_interface():
    """Load the exact frozen v11 helper source, then verify the imported path."""
    path = WEEK / "day6_diagnostic_reliability_v11" / "prepare.py"
    module_name = "postweek_v11_prepare_interface"
    module = sys.modules.get(module_name)
    if module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, path)
        require(spec is not None and spec.loader is not None, "V11_PREPARE_IMPORT_FAILED")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), "V11_HELPER_IMPORT_PATH_MISMATCH")
    return module


def verify_v11_source_binding(freeze: dict):
    helper_path = WEEK / "day6_diagnostic_reliability_v11" / "prepare.py"
    rows = all_frozen_bindings(freeze)
    matching = [row for row in rows if Path(row["path"]).resolve() == helper_path.resolve()]
    require(len(matching) == 1, "V11_PROMPT_SCHEMA_HELPER_NOT_BOUND_EXACTLY_ONCE")
    require(matching[0]["sha256"] == sha(helper_path) and matching[0]["bytes"] == helper_path.stat().st_size,
            "V11_PROMPT_SCHEMA_HELPER_BINDING_DRIFT")


def validate_denominator(requests: list[dict]) -> None:
    require(len(requests) == REQUESTS_N, "REQUEST_DENOMINATOR_NOT_12")
    roots = sorted({row["root_id"] for row in requests})
    require(len(roots) == ROOTS_N, "ROOT_DENOMINATOR_NOT_6")
    candidates = {}
    for row in requests:
        require(row["question_type"] == "ordering", "NON_ORDERING_REQUEST_PRESENT")
        require(row["variant_id"] in ("perm_00", "perm_01"), "NON_ORIGINAL_MAPPING_PRESENT")
        require(Path(row["prompt_path"]).is_file() and Path(row["schema_path"]).is_file(),
                "PROMPT_OR_SCHEMA_MISSING:" + row["request_id"])
        require(int(row.get("max_output_tokens", MAX_TOKENS)) == MAX_TOKENS,
                "MAX_OUTPUT_TOKENS_DRIFT")
        candidates.setdefault(row["candidate_id"], []).append(row)
    require(len(candidates) == ROOTS_N, "EXPECTED_ONE_CANDIDATE_PER_ROOT")
    for candidate_id, rows in candidates.items():
        require(len(rows) == 2, "EXPECTED_TWO_MAPPINGS:" + candidate_id)
        require({row["variant_id"] for row in rows} == {"perm_00", "perm_01"},
                "MAPPING_PAIR_INCOMPLETE:" + candidate_id)
        require(len({row["root_id"] for row in rows}) == 1, "CANDIDATE_ROOT_DRIFT")
        require(len({row["source_sha256"] for row in rows}) == 1, "PAIRED_SOURCE_DRIFT")
        require(all(isinstance(row.get("question"), str) and row["question"] for row in rows),
                "REQUEST_QUESTION_TEXT_REQUIRED:" + candidate_id)
        require(len({row["question"] for row in rows}) == 1, "PAIRED_QUESTION_DRIFT")
    require(len({row["request_id"] for row in requests}) == REQUESTS_N, "DUPLICATE_REQUEST_ID")


def check_source_admission(requests: list[dict], admission: dict) -> None:
    require(admission.get("status") == "PASS", "SOURCE_ADMISSION_NOT_PASS")
    # Require an explicit per-question decision, not merely a package-level flag.
    cases = admission.get("cases", admission.get("admissions", admission.get("questions")))
    require(isinstance(cases, list), "SOURCE_ADMISSION_CASE_LIST_REQUIRED")
    by_candidate = {}
    for item in cases:
        candidate = item.get("candidate_id")
        require(candidate and candidate not in by_candidate, "DUPLICATE_OR_MISSING_ADMISSION_CANDIDATE")
        by_candidate[candidate] = item
    expected = {r["candidate_id"] for r in requests}
    require(set(by_candidate) == expected, "SOURCE_ADMISSION_CANDIDATE_SET_MISMATCH")
    for candidate in sorted(expected):
        item = by_candidate[candidate]
        require(item.get("status", item.get("disposition")) in ("PASS", "ADMITTED", "QUALIFIED"),
                "SOURCE_CASE_NOT_ADMITTED:" + candidate)
        require(item.get("root_id") in {r["root_id"] for r in requests if r["candidate_id"] == candidate},
                "SOURCE_ADMISSION_ROOT_MISMATCH:" + candidate)
        require(item.get("source_sha256") in {r["source_sha256"] for r in requests if r["candidate_id"] == candidate},
                "SOURCE_ADMISSION_SOURCE_HASH_MISMATCH:" + candidate)


def verify(*, audit_path: Path | None = None, live_gate: bool = False):
    require(Path(sys.prefix).resolve() == VENV.resolve(), "USE_FROZEN_LOCAL4B_ENVIRONMENT")
    require(all(os.environ.get(k) == v for k, v in EXEC_ENV.items()), "VLLM_EXECUTION_ENV_MISMATCH")
    freeze = read(FREEZE_PATH)
    requests_doc = read(REQUESTS_PATH)
    requests = requests_doc.get("requests")
    require(isinstance(requests, list), "REQUESTS_ARRAY_REQUIRED")
    validate_denominator(requests)
    require(freeze.get("status") == "INPUTS_FROZEN_NOT_EXECUTED", "PREPARATION_FREEZE_STATUS_INVALID")
    verify_bindings(all_frozen_bindings(freeze), "FROZEN_INPUT_CHANGED")
    endpoint = freeze["endpoint"]
    require(Path(freeze["python_prefix"]).resolve() == VENV.resolve(), "FROZEN_PYTHON_PREFIX_MISMATCH")
    current_versions = package_versions()
    require(freeze["runtime_versions"] == current_versions, "RUNTIME_VERSION_DRIFT")
    require(Path(endpoint["model_directory"]).is_dir() and Path(endpoint["tokenizer_directory"]).is_dir(),
            "ENDPOINT_DIRECTORY_MISSING")
    backend_path = expected_backend(endpoint)
    verify_backend_prefix_cache_disabled(backend_path)
    backend_root = backend_path.parents[3]
    sys.path.insert(0, str(backend_root))
    backend_module = importlib.import_module("scripts.current_method.restart_n5_spell.vllm_backends")
    require(Path(backend_module.__file__).resolve() == backend_path.resolve(), "IMPORTED_BACKEND_SOURCE_MISMATCH")
    verify_v11_source_binding(freeze)

    from transformers import AutoTokenizer
    import msgspec
    import xgrammar as xgr
    from vllm import SamplingParams
    from vllm import sampling_params
    from vllm.v1.structured_output.backend_xgrammar import has_xgrammar_unsupported_json_features
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=False)
    v11 = import_v11_interface()
    for row in requests:
        prompt_path, schema_path, source_path = map(Path, (row["prompt_path"], row["schema_path"], row["source_path"]))
        require(sha(prompt_path) == row["prompt_sha256"], "PROMPT_HASH_MISMATCH:" + row["request_id"])
        require(sha(schema_path) == row["schema_sha256"], "SCHEMA_HASH_MISMATCH:" + row["request_id"])
        require(sha(source_path) == row["source_sha256"], "SOURCE_HASH_MISMATCH:" + row["request_id"])
        prompt = prompt_path.read_text()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        token_count = len(token_ids)
        require(token_count == int(row["prompt_tokens"]), "PROMPT_TOKEN_COUNT_MISMATCH:" + row["request_id"])
        token_ids_sha = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
        require(token_ids_sha == row["prompt_token_ids_sha256"],
                "PROMPT_TOKEN_IDS_HASH_MISMATCH:" + row["request_id"])
        require(token_count + MAX_TOKENS <= MODEL_LEN, "CONTEXT_OVERFLOW:" + row["request_id"])
        schema = read(schema_path)
        require(not has_xgrammar_unsupported_json_features(schema),
                "XGRAMMAR_UNSUPPORTED_SCHEMA:" + row["request_id"])
        compiler.compile_json_schema(json.dumps(schema, ensure_ascii=False), any_whitespace=True)
        params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=MAX_TOKENS, seed=row["seed"],
                                structured_outputs=sampling_params.StructuredOutputsParams(json=schema))
        require(getattr(params, "structured_outputs", None) is not None, "VLLM_STRUCTURED_OUTPUT_NOT_SET")
        json.dumps(msgspec.to_builtins(params), ensure_ascii=False)

        # Prompt must be exactly reproducible from the frozen question,
        # display and source using the previous validated v11 helper.
        question = row["question"]
        source = read(source_path)
        display = row["display"]
        q = {"question_type": "ordering", "question": question}
        regenerated = v11.make_prompt(tokenizer, q, display, source)
        require(regenerated.encode() == prompt_path.read_bytes(), "V11_PROMPT_REGENERATION_MISMATCH:" + row["request_id"])
        expected_schema = v11.schema(list(row["labels"]), [t["turn_ref"] for t in source["turns"]], "ordering")
        require(schema == expected_schema, "V11_SCHEMA_REGENERATION_MISMATCH:" + row["request_id"])
    if live_gate:
        require(audit_path is not None, "INDEPENDENT_AUDIT_REQUIRED")
        audit = verify_audit(audit_path, require_admission=True)
        check_source_admission(requests, read(ADMISSION_PATH))
    elif audit_path is not None:
        audit = verify_audit(audit_path, require_admission=False)
    else:
        audit = None
    return freeze, requests_doc, backend_path, audit


def gpu_idle() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        raise RuntimeError("CANNOT_VERIFY_GPU_IDLE:" + str(exc)) from exc
    used = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    require(bool(used), "NO_GPU_REPORTED")
    require(max(used) < 1000, "GPU_NOT_IDLE_MIB:" + repr(used))
    return {"memory_used_mib": used, "checked_at_utc": utcnow()}


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def live(audit_path: Path) -> None:
    freeze, request_doc, backend_path, audit = verify(audit_path=audit_path, live_gate=True)
    requests = request_doc["requests"]
    idle = gpu_idle()
    require(not RUN_DIR.exists(), "ONE_SHOT_ALREADY_STARTED_NO_RETRY")
    # mkdir is the atomic one-shot guard. A crash after this point remains a
    # consumed run and must not be resumed or silently retried.
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    write_new(RUN_DIR / "STARTED.json", {
        "status": "STARTED", "role": "B1", "started_at_utc": utcnow(),
        "request_count": len(requests), "freeze": binding(FREEZE_PATH),
        "audit": binding(audit_path), "source_admission": binding(ADMISSION_PATH),
        "gpu_idle_precheck": idle, "gold_opened": False, "automatic_retry": False,
    })
    submitted: set[str] = set()
    returned: set[str] = set()
    failure = None
    backend = None
    started = time.perf_counter()
    model_load_seconds = None
    try:
        from vllm import SamplingParams, sampling_params
        import msgspec
        module = importlib.import_module("scripts.current_method.restart_n5_spell.vllm_backends")
        require(Path(module.__file__).resolve() == backend_path.resolve(), "LIVE_BACKEND_IMPORT_PATH_MISMATCH")
        backend = module.VLLMSolverBackend(
            model_path=Path(freeze["endpoint"]["model_directory"]),
            tokenizer_path=Path(freeze["endpoint"]["tokenizer_directory"]),
            max_model_len=MODEL_LEN, temperature=0.8, top_p=0.9, max_tokens=MAX_TOKENS,
            gpu_memory_utilization=0.8,
        )
        model_load_seconds = backend.load_latency_seconds
        require(str(backend.model_identifier) == str(Path(freeze["endpoint"]["model_directory"]).resolve()),
                "LOADED_MODEL_DIRECTORY_MISMATCH")
        responses_dir = RUN_DIR / "responses"
        for offset in range(0, len(requests), BATCH_SIZE):
            batch = requests[offset:offset + BATCH_SIZE]
            prompts, params = [], []
            for row in batch:
                prompt_path, schema_path = Path(row["prompt_path"]), Path(row["schema_path"])
                require(sha(prompt_path) == row["prompt_sha256"] and sha(schema_path) == row["schema_sha256"],
                        "LIVE_REQUEST_BYTES_CHANGED:" + row["request_id"])
                prompts.append(prompt_path.read_text())
                params.append(SamplingParams(
                    temperature=0.8, top_p=0.9, max_tokens=MAX_TOKENS, seed=row["seed"],
                    structured_outputs=sampling_params.StructuredOutputsParams(json=read(schema_path)),
                ))
            batch_id = f"batch{offset // BATCH_SIZE:02d}"
            batch_submitted_at = utcnow()
            write_new(RUN_DIR / f"{batch_id}_SUBMITTED.json", {
                "request_ids": [row["request_id"] for row in batch],
                "sampling_params": [msgspec.to_builtins(item) for item in params],
                "submitted_at_utc": batch_submitted_at, "automatic_retry": False,
            })
            submitted.update(row["request_id"] for row in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            for index, row in enumerate(batch):
                out = outputs[index] if index < len(outputs) else None
                completions = list(out.outputs) if out else []
                first = completions[0] if completions else None
                record = {
                    "request_id": row["request_id"], "candidate_id": row["candidate_id"],
                    "root_id": row["root_id"], "question_type": row["question_type"], "role": "B1",
                    "variant_id": row["variant_id"], "seed": row["seed"],
                    "prompt_sha256": row["prompt_sha256"], "schema_sha256": row["schema_sha256"],
                    "source_sha256": row["source_sha256"], "status": "RETURNED" if first else "MISSING_OUTPUT",
                    "raw_output": first.text if first else None,
                    "prompt_token_ids": list(out.prompt_token_ids) if out else None,
                    "all_completions": [{
                        "text": completion.text, "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                    } for completion in completions],
                    "sampling": {"temperature": 0.8, "top_p": 0.9, "max_tokens": MAX_TOKENS,
                                 "seed": row["seed"], "structured_json": True,
                                 "sampler": "native_vllm_sampler", "worker_method": "spawn"},
                    "metadata": {
                        "finish_reason": first.finish_reason if first else None,
                        "stop_reason": first.stop_reason if first else None,
                        "prompt_tokens": len(out.prompt_token_ids) if out else None,
                        "completion_tokens": len(first.token_ids) if first else None,
                        "model_directory": str(Path(freeze["endpoint"]["model_directory"]).resolve()),
                        "tokenizer_directory": str(Path(freeze["endpoint"]["tokenizer_directory"]).resolve()),
                        "submitted_at_utc": batch_submitted_at, "returned_at_utc": utcnow(),
                        "gold_opened_by_runtime": False,
                    },
                }
                write_new(responses_dir / (row["request_id"] + ".json"), record)
                if first:
                    returned.add(row["request_id"])
            if len(outputs) != len(batch):
                raise RuntimeError("OUTPUT_COUNT_MISMATCH_RAW_RECORDS_COMMITTED")
            print(json.dumps({"status": "BATCH_RECORDED", "batch": batch_id,
                              "returned": len(returned), "planned": REQUESTS_N}), flush=True)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as exc:
                failure = failure or {"type": type(exc).__name__, "error": str(exc),
                                      "traceback": traceback.format_exc()}
        responses_dir = RUN_DIR / "responses"
        responses_dir.mkdir(exist_ok=True)
        for row in requests:
            response_path = responses_dir / (row["request_id"] + ".json")
            if not response_path.exists():
                was_submitted = row["request_id"] in submitted
                write_new(response_path, {
                    "request_id": row["request_id"], "candidate_id": row["candidate_id"],
                    "root_id": row["root_id"], "question_type": row["question_type"], "role": "B1",
                    "variant_id": row["variant_id"], "seed": row["seed"],
                    "status": "BACKEND_ERROR" if was_submitted else "NOT_SUBMITTED",
                    "usage_status": "UNKNOWN_NOT_ZERO" if was_submitted else "NO_REQUEST_SENT",
                    "raw_output": None, "failure": failure,
                    "gold_opened_by_runtime": False,
                })
        raw_files = [binding(p) for p in sorted(responses_dir.glob("*.json"))]
        terminal = "FAILED" if failure else "COMPLETE"
        write_new(RUN_DIR / "RAW_RESPONSE_FREEZE.json", {
            "status": "COMMITTED", "terminal_status": terminal, "role": "B1",
            "planned_requests": REQUESTS_N, "submitted_requests": len(submitted),
            "returned_requests": len(returned), "automatic_retry": False,
            "files": raw_files, "preparation_freeze": binding(FREEZE_PATH),
            "audit": binding(audit_path), "source_admission": binding(ADMISSION_PATH),
            "failure": failure, "wall_seconds": time.perf_counter() - started,
            "model_load_seconds": model_load_seconds, "gold_opened": False,
        })
        write_new(RUN_DIR / ("FAILED.json" if failure else "COMPLETE.json"), {
            "status": terminal, "role": "B1", "planned": REQUESTS_N,
            "submitted": len(submitted), "returned": len(returned), "failure": failure,
        })
    if failure:
        raise RuntimeError("RUN_FAILED_RAW_RECORDS_FROZEN:" + failure["error"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--audit", type=Path)
    run = sub.add_parser("live")
    run.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        freeze, request_doc, backend, audit = verify(audit_path=args.audit)
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS", "role": "B1", "requests": len(request_doc["requests"]),
                          "roots": len({r['root_id'] for r in request_doc['requests']}),
                          "backend_import": str(backend), "audit_checked": bool(audit),
                          "runtime_sha256": sha(Path(__file__))}, ensure_ascii=False))
    elif args.command == "live":
        live(args.audit)


if __name__ == "__main__":
    main()
