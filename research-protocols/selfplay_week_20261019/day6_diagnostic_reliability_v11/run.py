from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PREV = WEEK / "day5_training_v1" / "train_diagnosis_prepare"
FREEZE_PATH = HERE / "PREPARATION_FREEZE_V11.json"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
sys.dont_write_bytecode = True


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def check(rows):
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError("FROZEN_FILE_CHANGED:" + str(path))


def package_bindings(freeze):
    result = list(freeze["files"] + freeze["upstream"] + freeze["sources"] + freeze["prompts"] +
        freeze["tokenizer_bindings"] + freeze["code_bindings"] + [freeze["selection_root_file"]] +
        freeze["legacy_readout_bindings"])
    for endpoint in freeze["endpoints"].values():
        result += [endpoint["manifest"], endpoint["endpoint_manifest"], endpoint["endpoint_proof"]]
        result += endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["endpoint_evidence_bindings"]
        result += endpoint["code_bindings"]
    return result


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "vllm", "jsonschema")}


def audit_check(freeze, audit_path):
    audit = read(audit_path)
    expected = binding(FREEZE_PATH)
    if audit.get("status") != "PASS" or audit.get("scope") != "DAY6_DIAGNOSTIC_INTERFACE_PREFLIGHT":
        raise RuntimeError("INDEPENDENT_PREFLIGHT_REQUIRED")
    records = audit.get("audited_files", [])
    if expected not in records:
        raise RuntimeError("PREFLIGHT_MUST_BIND_THIS_FREEZE")
    check(records)


def verify(audit_path=None):
    expected_prefix = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("USE_FROZEN_LOCAL4B_ENVIRONMENT")
    freeze = read(FREEZE_PATH)
    if freeze["status"] != "CPU_PREPARED_AWAITING_INDEPENDENT_PREFLIGHT":
        raise RuntimeError("PREPARATION_STATE_DRIFT")
    expected_exec_env = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    if any(os.environ.get(key) != value for key, value in expected_exec_env.items()):
        raise RuntimeError("VLLM_EXECUTION_ENV_MISMATCH")
    check(package_bindings(freeze))
    protocol = read(HERE / "PROTOCOL.json")
    data = read(HERE / "REQUESTS.json")
    if protocol["status"] != "PREPARED_PENDING_INDEPENDENT_PREFLIGHT" or len(data["requests"]) != 12:
        raise RuntimeError("REQUEST_SCOPE_DRIFT")
    if {r["question_type"] for r in data["requests"]} != {"multi_select", "ordering"}:
        raise RuntimeError("QUESTION_TYPES_MISSING")
    current_versions = versions()
    if protocol["runtime_versions"] != current_versions or freeze["runner_runtime_versions"] != current_versions:
        raise RuntimeError("RUNNER_RUNTIME_VERSION_MISMATCH")
    if protocol.get("execution_env") != expected_exec_env or freeze.get("execution_env") != expected_exec_env:
        raise RuntimeError("FROZEN_MULTIPROCESS_METHOD_MISMATCH")
    for role, ep in freeze["endpoints"].items():
        if ep["python_prefix"] != str(expected_prefix) or any(current_versions[k] != v for k, v in ep["runtime_versions"].items()):
            raise RuntimeError("ENDPOINT_RUNTIME_MISMATCH:" + role)
        backend_bindings = [Path(row["path"]).resolve() for row in ep["code_bindings"]
            if Path(row["path"]).name == "vllm_backends.py"]
        if len(backend_bindings) != 1 or backend_bindings[0].parts[-4:] != (
            "scripts", "current_method", "restart_n5_spell", "vllm_backends.py"):
            raise RuntimeError("FROZEN_BACKEND_SOURCE_PATH_MISMATCH:" + role)
        backend_source = backend_bindings[0]
        transitive = backend_source.with_name("backends.py")
        if not any(Path(row["path"]).resolve() == transitive for row in ep["code_bindings"]):
            raise RuntimeError("TRANSITIVE_BACKEND_SOURCE_UNBOUND:" + role)
        for init_path in (backend_source.parents[1] / "__init__.py", backend_source.parent / "__init__.py"):
            if not any(Path(row["path"]).resolve() == init_path.resolve() for row in ep["code_bindings"]):
                raise RuntimeError("BACKEND_PACKAGE_INIT_UNBOUND:" + role)
        sys.path.insert(0, str(backend_source.parents[3]))
        backend_module = importlib.import_module("scripts.current_method.restart_n5_spell.vllm_backends")
        if Path(backend_module.__file__).resolve() != backend_source:
            raise RuntimeError("IMPORTED_BACKEND_SOURCE_MISMATCH:" + role)
    from transformers import AutoTokenizer
    import msgspec
    import xgrammar as xgr
    from vllm import SamplingParams, sampling_params
    from vllm.v1.structured_output.backend_xgrammar import has_xgrammar_unsupported_json_features
    tokenizer = AutoTokenizer.from_pretrained(str(freeze["tokenizer_path"]), local_files_only=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=False)
    for row in data["requests"]:
        schema = read(Path(row["schema_path"]))
        if has_xgrammar_unsupported_json_features(schema):
            raise RuntimeError("XGRAMMAR_UNSUPPORTED_SCHEMA:" + row["request_id"])
        compiler.compile_json_schema(json.dumps(schema, ensure_ascii=False), any_whitespace=True)
        params = SamplingParams(temperature=.8, top_p=.9, max_tokens=1024, seed=row["seed"],
            structured_outputs=sampling_params.StructuredOutputsParams(json=schema))
        if getattr(params, "structured_outputs", None) is None:
            raise RuntimeError("VLLM_STRUCTURED_JSON_NOT_BOUND")
        json.dumps(msgspec.to_builtins(params), ensure_ascii=False)
    if audit_path:
        audit_check(freeze, audit_path)
    return freeze, data


def live(role: str, audit_path: Path):
    freeze, data = verify(audit_path)
    if role not in ("A1", "B1"):
        raise RuntimeError("UNKNOWN_ROLE")
    run_dir = HERE / "runs" / role
    if run_dir.exists():
        raise RuntimeError("NO_OVERWRITE_OR_AUTOMATIC_RETRY")
    endpoint = freeze["endpoints"][role]
    backend_files = [Path(row["path"]) for row in endpoint["code_bindings"]
        if Path(row["path"]).name == "vllm_backends.py"]
    if len(backend_files) != 1 or backend_files[0].parts[-4:] != (
        "scripts", "current_method", "restart_n5_spell", "vllm_backends.py"):
        raise RuntimeError("FROZEN_BACKEND_SOURCE_PATH_MISMATCH")
    backend_root = backend_files[0].parents[3]
    if not backend_root.is_dir():
        raise RuntimeError("FROZEN_BACKEND_SOURCE_ROOT_MISSING")
    sys.path.insert(0, str(backend_root))
    requests = data["requests"]
    run_dir.mkdir(parents=True)
    (run_dir / "STARTED.json").write_text(json.dumps({
        "status": "STARTED", "role": role, "started_at_unix": time.time(),
        "freeze": binding(FREEZE_PATH), "audit": binding(audit_path),
        "request_count": len(requests), "gold_opened": False,
    }, indent=2) + "\n")
    backend = None
    submitted, completed = set(), set()
    failed = None
    start = time.perf_counter()
    load_seconds = None
    try:
        import msgspec
        from scripts.current_method.restart_n5_spell import vllm_backends
        if Path(vllm_backends.__file__).resolve() != backend_files[0].resolve():
            raise RuntimeError("IMPORTED_BACKEND_SOURCE_MISMATCH")
        VLLMSolverBackend = vllm_backends.VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=Path(endpoint["model_directory"]),
            tokenizer_path=Path(endpoint["tokenizer_directory"]), max_model_len=16384,
            temperature=.8, top_p=.9, max_tokens=1024, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        if str(backend.model_identifier) != endpoint["model_directory"]:
            raise RuntimeError("MODEL_DIRECTORY_MISMATCH")
        for offset in range(0, len(requests), 4):
            batch = requests[offset:offset + 4]
            prompts = []
            params = []
            for row in batch:
                prompt_path = Path(row["prompt_path"])
                schema_path = Path(row["schema_path"])
                if sha(prompt_path) != row["prompt_sha256"] or sha(schema_path) != row["schema_sha256"]:
                    raise RuntimeError("REQUEST_INPUT_HASH_MISMATCH")
                prompts.append(prompt_path.read_text())
                params.append(SamplingParams(temperature=.8, top_p=.9, max_tokens=1024, seed=row["seed"],
                    structured_outputs=sampling_params.StructuredOutputsParams(json=read(schema_path))))
            batch_id = f"batch{offset // 4:04d}"
            (run_dir / (batch_id + "_SUBMITTED.json")).write_text(json.dumps({
                "request_ids": [r["request_id"] for r in batch], "sampling": [msgspec.to_builtins(p) for p in params],
                "submitted_at_unix": time.time(), "automatic_retry": False,
            }, indent=2) + "\n")
            submitted.update(r["request_id"] for r in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            for i, row in enumerate(batch):
                out = outputs[i] if i < len(outputs) else None
                completions = list(out.outputs) if out else []
                first = completions[0] if completions else None
                record = {
                    "request_id": row["request_id"], "candidate_id": row["candidate_id"], "root_id": row["root_id"],
                    "question_type": row["question_type"], "role": role,
                    "prompt_sha256": row["prompt_sha256"], "schema_sha256": row["schema_sha256"],
                    "seed": row["seed"], "status": "RETURNED" if first else "MISSING_OUTPUT",
                    "raw_output": first.text if first else None,
                    "prompt_token_ids": out.prompt_token_ids if out else None,
                    "all_completions": [{"text": c.text, "token_ids": c.token_ids, "finish_reason": c.finish_reason,
                        "stop_reason": c.stop_reason} for c in completions],
                    "metadata": {"finish_reason": first.finish_reason if first else None,
                        "stop_reason": first.stop_reason if first else None,
                        "completion_tokens": len(first.token_ids) if first else None,
                        "structured_json": True, "temperature": .8, "top_p": .9, "max_tokens": 1024,
                        "gold_opened_by_solver": False},
                }
                path = run_dir / "responses" / (row["request_id"] + ".json")
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                if first:
                    completed.add(row["request_id"])
            if len(outputs) != len(batch):
                raise RuntimeError("OUTPUT_COUNT_MISMATCH_AFTER_RAW_COMMIT")
            print(json.dumps({"role": role, "completed": len(completed), "planned": len(requests)}), flush=True)
    except BaseException as exc:
        failed = {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as exc:
                failed = failed or {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        for row in requests:
            path = run_dir / "responses" / (row["request_id"] + ".json")
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"request_id": row["request_id"], "candidate_id": row["candidate_id"],
                    "root_id": row["root_id"], "question_type": row["question_type"], "role": role,
                    "status": "BACKEND_ERROR" if row["request_id"] in submitted else "NOT_SUBMITTED",
                    "usage_status": "UNKNOWN_NOT_ZERO" if row["request_id"] in submitted else "NO_REQUEST_SENT",
                    "raw_output": None, "failure": failed}, ensure_ascii=False, indent=2) + "\n")
        files = [binding(p) for p in sorted((run_dir / "responses").glob("*.json"))]
        (run_dir / "RAW_RESPONSE_FREEZE.json").write_text(json.dumps({
            "status": "COMMITTED", "role": role, "planned_requests": 12,
            "submitted_requests": len(submitted), "returned_requests": len(completed),
            "automatic_retry": False, "files": files,
            "preparation_freeze": binding(FREEZE_PATH),
            "audit": binding(audit_path), "failure": failed,
            "wall_seconds": time.perf_counter() - start, "model_load_seconds": load_seconds,
        }, indent=2) + "\n")
        (run_dir / ("FAILED.json" if failed else "COMPLETE.json")).write_text(json.dumps({
            "role": role, "status": "FAILED" if failed else "COMPLETE", "failure": failed,
            "submitted": len(submitted), "returned": len(completed),
        }, indent=2) + "\n")
    if failed:
        raise RuntimeError("RUN_FAILED_RECORDS_FROZEN:" + failed["error"])


def score():
    import jsonschema
    freeze, data = verify()
    report = {"status": "SCORED_FORMAT_ONLY", "scope": "NO_GOLD_NO_SEMANTIC_CORRECTNESS", "models": {}}
    for role in ("A1", "B1"):
        run_dir = HERE / "runs" / role
        raw_freeze = read(run_dir / "RAW_RESPONSE_FREEZE.json")
        check(raw_freeze["files"] + [raw_freeze["preparation_freeze"], raw_freeze["audit"]])
        audit_check(freeze, Path(raw_freeze["audit"]["path"]))
        responses = {read(Path(x["path"]))["request_id"]: read(Path(x["path"])) for x in raw_freeze["files"]}
        if len(responses) != 12:
            raise RuntimeError("RAW_RESPONSE_DENOMINATOR_DRIFT")
        rows = []
        for request in data["requests"]:
            raw = responses[request["request_id"]]
            item = {"request_id": request["request_id"], "candidate_id": request["candidate_id"],
                "root_id": request["root_id"], "question_type": request["question_type"],
                "status": raw["status"], "json_schema_valid": False, "all_labels_present": False,
                "all_turn_refs_exist": False, "final_answer_unique": False,
                "final_answer_shape_valid": False, "final_answer_labels_valid": False,
                "errors": []}
            if raw.get("raw_output") is not None:
                try:
                    obj = json.loads(raw["raw_output"])
                    labels = request["labels"]
                    try:
                        jsonschema.validate(obj, read(Path(request["schema_path"])))
                        item["json_schema_valid"] = True
                    except Exception as exc:
                        item["errors"].append("SCHEMA:" + type(exc).__name__ + ":" + str(exc)[:350])
                    evidence = obj.get("item_evidence") if isinstance(obj, dict) else None
                    if isinstance(evidence, dict):
                        item["all_labels_present"] = set(evidence) == set(labels)
                        refset = set(request["source_turn_refs"])
                        refs = []
                        refs_well_formed = True
                        for ev in evidence.values():
                            if not isinstance(ev, dict) or not isinstance(ev.get("turn_refs"), list):
                                refs_well_formed = False
                                continue
                            refs.extend(ev["turn_refs"])
                        item["all_turn_refs_exist"] = refs_well_formed and all(
                            isinstance(ref, str) and ref in refset for ref in refs)
                    final_answer = obj.get("final_answer") if isinstance(obj, dict) else None
                    if isinstance(final_answer, list):
                        answer_strings = all(isinstance(label, str) for label in final_answer)
                        item["final_answer_unique"] = answer_strings and len(final_answer) == len(set(final_answer))
                        item["final_answer_labels_valid"] = answer_strings and all(label in labels for label in final_answer)
                        if request["question_type"] == "ordering":
                            item["final_answer_shape_valid"] = len(final_answer) == len(labels)
                        else:
                            item["final_answer_shape_valid"] = 2 <= len(final_answer) <= 5
                    if not item["all_labels_present"]:
                        item["errors"].append("ITEM_LABEL_COVERAGE")
                    if not item["all_turn_refs_exist"]:
                        item["errors"].append("INVALID_TURN_REF")
                    if not item["final_answer_unique"]:
                        item["errors"].append("DUPLICATE_FINAL_LABEL")
                    if not item["final_answer_shape_valid"]:
                        item["errors"].append("FINAL_ANSWER_SHAPE")
                    if not item["final_answer_labels_valid"]:
                        item["errors"].append("FINAL_ANSWER_LABEL_DOMAIN")
                    if isinstance(final_answer, list):
                        item["final_answer"] = final_answer
                except Exception as exc:
                    item["errors"].append("JSON_PARSE:" + type(exc).__name__ + ":" + str(exc)[:350])
            rows.append(item)
        legacy = read(PREV / "models" / role / "private" / "SEMANTIC_READOUT.json")
        legacy_by = {q["candidate_id"]: q for q in legacy["questions"]}
        for row in rows:
            old = legacy_by[row["candidate_id"]]
            row["legacy_schema_valid"] = old["schema_valid"]
            row["legacy_disposition"] = old["disposition"]
        metrics = {
            "planned": len(rows), "json_schema_valid": sum(x["json_schema_valid"] for x in rows),
            "all_labels_present": sum(x["all_labels_present"] for x in rows),
            "all_turn_refs_exist": sum(x["all_turn_refs_exist"] for x in rows),
            "unique_final_answers": sum(x.get("final_answer_unique", False) for x in rows),
            "valid_final_answer_shapes": sum(x.get("final_answer_shape_valid", False) for x in rows),
            "mechanically_valid": sum(
                x["json_schema_valid"] and x["all_labels_present"] and x["all_turn_refs_exist"]
                and x.get("final_answer_unique", False) and x.get("final_answer_shape_valid", False)
                and x.get("final_answer_labels_valid", False)
                for x in rows
            ),
            "legacy_schema_valid_same_questions": sum(x["legacy_schema_valid"] for x in rows),
            "legacy_invalid_same_questions": sum(not x["legacy_schema_valid"] for x in rows),
            "by_question": rows,
        }
        metrics["engineering_gate"] = {
            "schema_valid_at_least_11": metrics["json_schema_valid"] >= 11,
            "labels_present_all_12": metrics["all_labels_present"] == 12,
            "turn_refs_valid_all_12": metrics["all_turn_refs_exist"] == 12,
            "answers_unique_all_12": metrics["unique_final_answers"] == 12,
            "answer_shapes_valid_all_12": metrics["valid_final_answer_shapes"] == 12,
            "answer_labels_valid_all_12": sum(x.get("final_answer_labels_valid", False) for x in rows) == 12,
        }
        metrics["engineering_gate"]["passed"] = all(metrics["engineering_gate"].values())
        report["models"][role] = metrics
    report["status"] = "PASS_FORMAT_ONLY_GATE" if all(
        report["models"][role]["engineering_gate"]["passed"] for role in ("A1", "B1")
    ) else "FAIL_FORMAT_ONLY_GATE"
    path = HERE / "FORMAT_AUDIT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "models": {
        role: {k: v for k, v in value.items() if k != "by_question"}
        for role, value in report["models"].items()}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    live_p = sub.add_parser("live")
    live_p.add_argument("--role", choices=("A1", "B1"), required=True)
    live_p.add_argument("--audit", type=Path, required=True)
    sub.add_parser("score-format")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
        print("PASS_CPU_PREPARED_NOT_EXECUTED")
    elif args.command == "live":
        live(args.role, args.audit)
    else:
        score()


if __name__ == "__main__":
    main()
