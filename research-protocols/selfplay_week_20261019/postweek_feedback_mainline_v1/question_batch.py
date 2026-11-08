"""Run two content-reviewed tasks through the accepted feedback source handoff."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
SPEC = importlib.util.spec_from_file_location("operation_reason_interface", WEEK / "postweek_diversity_reason_v1/run.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
read, save, bind, check = diagnostic.read, diagnostic.save, diagnostic.bind, diagnostic.check
OUT = HERE / "question_execution"


def public_preparation():
    """Validate only public worker inputs; never open the private manifest."""
    prep = read(OUT / "PREPARATION.json")
    for bound in prep["bindings"]:
        check(bound)
    rows = read(OUT / "PUBLIC_REQUESTS.json")["requests"]
    ids = read(OUT / "PROMPT_TOKEN_IDS.json")
    request_ids = [row["request_id"] for row in rows]
    assert prep["requests"] == len(rows) == len(set(request_ids)) == 2, "REQUEST_COUNT_DRIFT"
    assert set(ids) == set(request_ids), "PROMPT_TOKEN_REQUEST_SET_DRIFT"
    required = [OUT / "PUBLIC_REQUESTS.json", OUT / "PROMPT_TOKEN_IDS.json"]
    required.extend(Path(row[key]) for row in rows for key in ("prompt_path", "schema_path"))
    assert all(bind(path) in prep["bindings"] for path in required), "UNBOUND_PUBLIC_INPUT"
    assert all(len(ids[row["request_id"]]) == row["prompt_tokens"] for row in rows), "PROMPT_TOKEN_COUNT_DRIFT"
    return prep, rows, ids


def response_errors(raw, prompt, token_ids):
    errors = []
    results = raw.get("returned_results", [])
    if raw.get("returned_result_count") != 1 or len(results) != 1:
        errors.append("RETURNED_RESULT_COUNT_MISMATCH")
    for result in results:
        if result.get("prompt") != prompt:
            errors.append("RETURNED_PROMPT_DRIFT")
        if result.get("prompt_token_ids") != token_ids:
            errors.append("PROMPT_TOKEN_IDS_DRIFT")
        if len(result.get("outputs", [])) != 1:
            errors.append("COMPLETION_COUNT_MISMATCH")
    if results and results[0].get("outputs"):
        first, comp = results[0], results[0]["outputs"][0]
        for key in ("raw_output", "finish_reason", "stop_reason", "output_token_ids"):
            if raw.get(key) != comp.get(key):
                errors.append("RAW_COMPLETION_FIELD_DRIFT:" + key)
        if raw.get("prompt_token_ids") != first.get("prompt_token_ids"):
            errors.append("RAW_PROMPT_TOKEN_FIELD_DRIFT")
    return errors


def returned_record(row, results, prompt, token_ids, endpoint):
    """Keep every returned text before rejecting any response shape or drift."""
    returned = [{"prompt": result.prompt, "prompt_token_ids": list(result.prompt_token_ids or []),
        "outputs": [{"raw_output": comp.text, "finish_reason": comp.finish_reason,
            "stop_reason": comp.stop_reason, "output_token_ids": list(comp.token_ids)}
            for comp in result.outputs]} for result in results]
    raw = {"request_id": row["request_id"], "status": "RETURNED", "returned_result_count": len(results),
        "returned_results": returned, "raw_output": None, "model_role": "B1",
        "model_directory": endpoint["model_directory"], "sampling_seed": row["sampling_seed"], "gold_read": False}
    if returned:
        raw["prompt_token_ids"] = returned[0]["prompt_token_ids"]
        if returned[0]["outputs"]:
            raw.update(returned[0]["outputs"][0])
    raw["output_validation_errors"] = response_errors(raw, prompt, token_ids)
    return raw


def prepare():
    from transformers import AutoTokenizer
    import xgrammar as xgr
    assert not OUT.exists(), "NO_OVERWRITE"
    freeze = read(HERE / "QUESTION_EXECUTION_FREEZE.json")
    review_path = HERE / "questions/CONTENT_REVIEW_AI.json"
    review_doc = read(review_path)
    assert review_doc["question_execution_freeze_binding"] == bind(HERE / "QUESTION_EXECUTION_FREEZE.json")
    reviews = {r["candidate_id"]: r for r in review_doc["reviews"]}
    endpoint = read(diagnostic.ENDPOINT)
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    staged = []
    for slot in freeze["slots"]:
        cid, root_id = slot["candidate_id"], slot["root_id"]
        candidate_path = HERE / "questions" / (cid + ".json")
        candidate = read(candidate_path)
        reviewed = reviews[cid]
        assert reviewed["candidate_binding"] == bind(candidate_path)
        assert reviewed.get("status", reviewed.get("decision")) in ("PASS", "READY")
        assert reviewed["reviewer_type"] == "ai_agent"
        assert reviewed["reviewer_id"] != candidate["author_id"]
        ready_path = HERE / "live_run_v2" / root_id / "QUESTION_READY_SOURCE.json"
        ready = read(ready_path)
        assert ready["status"] == "SOURCE_READY_FOR_QUESTION_AUTHORING"
        assert candidate["source_binding"] == ready["source_binding"]
        check(ready["source_binding"])
        source = read(ready["source_binding"]["path"])
        assert source["root_id"] == root_id
        assert all(candidate[k] == slot[k] for k in ("candidate_id", "root_id", "question_type", "task_operation"))
        field = "options" if candidate["question_type"] == "multi_select" else "items"
        keys = list(candidate[field])
        shuffled = random.Random(slot["display_seed"]).sample(keys, len(keys))
        canonical = {chr(65 + i): key for i, key in enumerate(shuffled)}
        display = {label: candidate[field][key] for label, key in canonical.items()}
        row = {key: candidate[key] for key in ("candidate_id", "root_id", "question_type", "task_operation", "question")}
        row.update(request_id=cid + "__B1_mainline_reason_v1", source_path=ready["source_binding"]["path"],
            source_sha256=ready["source_binding"]["sha256"], source_binding=ready["source_binding"],
            display=display, display_to_canonical=canonical, labels=list(display), sampling_seed=slot["sampling_seed"],
            question_ready_binding=bind(ready_path))
        prompt = diagnostic.make_prompt(tokenizer, row, display, source)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(tokens) + 3072 <= 16384, "CONTEXT_OVERFLOW_NO_TRUNCATION"
        output_schema = diagnostic.repair.schema(row["labels"], [t["turn_ref"] for t in source["turns"]], row["question_type"])
        compiler.compile_json_schema(json.dumps(output_schema), any_whitespace=True)
        staged.append((row, prompt, tokens, output_schema, candidate_path))
    assert len(staged) == 2
    OUT.mkdir()
    bindings = [bind(HERE / "QUESTION_EXECUTION_FREEZE.json"), bind(Path(__file__)), bind(diagnostic.ENDPOINT), bind(diagnostic.REPAIR_PATH), bind(diagnostic.HERE / "run.py")]
    rows, token_ids, private_rows = [], {}, []
    for row, prompt, tokens, output_schema, candidate_path in staged:
        rid = row["request_id"]
        prompt_path, schema_path = OUT / (rid + ".txt"), OUT / (rid + "_schema.json")
        with prompt_path.open("x", encoding="utf-8") as stream:
            stream.write(prompt)
        save(schema_path, output_schema)
        row.update(prompt_path=str(prompt_path), schema_path=str(schema_path), prompt_tokens=len(tokens))
        rows.append(row)
        token_ids[rid] = tokens
        private_rows.append({"request_id": rid, "candidate_id": row["candidate_id"],
            "candidate_binding": bind(candidate_path), "question_ready_binding": row["question_ready_binding"],
            "source_binding": row["source_binding"]})
        bindings.extend([bind(prompt_path), bind(schema_path), row["source_binding"], row["question_ready_binding"]])
    save(OUT / "PUBLIC_REQUESTS.json", {"requests": rows})
    save(OUT / "PROMPT_TOKEN_IDS.json", token_ids)
    bindings.extend(bind(OUT / name) for name in ("PUBLIC_REQUESTS.json", "PROMPT_TOKEN_IDS.json"))
    save(OUT / "PRIVATE_SCORING_BINDINGS.json", {"requests": private_rows,
        "content_review_binding": bind(review_path), "execution_freeze_binding": bind(HERE / "QUESTION_EXECUTION_FREEZE.json"),
        "public_requests_binding": bind(OUT / "PUBLIC_REQUESTS.json"),
        "prompt_token_ids_binding": bind(OUT / "PROMPT_TOKEN_IDS.json")})
    save(OUT / "PREPARATION.json", {"requests": 2, "model_role": "B1", "bindings": bindings,
        "generation_reads_private_questions": False, "automatic_retry": False, "max_output_tokens": 3072,
        "private_scoring_binding": bind(OUT / "PRIVATE_SCORING_BINDINGS.json"),
        "content_review_binding": bind(review_path), "public_worker_bindings_exclude_private_question_files": True})
    print(json.dumps({"prepared": 2, "prompt_tokens": [r["prompt_tokens"] for r in rows]}))


def run(execute_gpu):
    assert execute_gpu, "GPU_FLAG_REQUIRED"
    assert not (OUT / "RAW_RESPONSE_FREEZE.json").exists() and not (OUT / "STARTED.json").exists(), "NO_RERUN_OR_OVERWRITE"
    for key, val in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = val
    prep, rows, token_ids = public_preparation()
    endpoint = read(diagnostic.ENDPOINT)
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve()
    for bound in endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]:
        check(bound)
    backend_path = next(Path(b["path"]) for b in endpoint["code_bindings"] if Path(b["path"]).name == "vllm_backends.py")
    sys.path.insert(0, str(backend_path.parents[3]))
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
    from vllm import SamplingParams, sampling_params
    save(OUT / "STARTED.json", {"time_unix": time.time(), "preparation": bind(OUT / "PREPARATION.json"),
        "private_scoring_binding": prep["private_scoring_binding"], "requests": 2,
        "request_ids": [row["request_id"] for row in rows]})
    backend, failure, submitted = None, None, set()
    started = time.perf_counter()
    try:
        backend = VLLMSolverBackend(model_path=Path(endpoint["model_directory"]), tokenizer_path=Path(endpoint["tokenizer_directory"]),
            max_model_len=16384, temperature=.8, top_p=.9, max_tokens=3072, gpu_memory_utilization=.8)
        for row in rows:
            rid = row["request_id"]
            save(OUT / "submitted" / (rid + ".json"), {"request_id": rid, "seed": row["sampling_seed"], "time_unix": time.time()})
            submitted.add(rid)
            params = SamplingParams(temperature=.8, top_p=.9, max_tokens=3072, seed=row["sampling_seed"],
                structured_outputs=sampling_params.StructuredOutputsParams(json=read(row["schema_path"])))
            prompt = Path(row["prompt_path"]).read_text()
            results = backend.engine.generate([prompt], [params], use_tqdm=False)
            raw = returned_record(row, results, prompt, token_ids[rid], endpoint)
            save(OUT / "raw" / (rid + ".json"), raw)
            assert not raw["output_validation_errors"], ";".join(raw["output_validation_errors"])
            print("RETURNED " + rid, flush=True)
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                failure = (failure or "") + traceback.format_exc()
        for row in rows:
            path = OUT / "raw" / (row["request_id"] + ".json")
            if not path.exists():
                save(path, {"request_id": row["request_id"], "status": "BACKEND_ERROR" if row["request_id"] in submitted else "NOT_SUBMITTED", "raw_output": None, "failure": failure})
        raw_paths = sorted((OUT / "raw").glob("*.json"))
        save(OUT / "RAW_RESPONSE_FREEZE.json", {"requests": 2, "wall_seconds": time.perf_counter() - started,
            "started_binding": bind(OUT / "STARTED.json"), "request_ids": [row["request_id"] for row in rows],
            "returned_requests": sum(read(path)["status"] == "RETURNED" for path in raw_paths),
            "failure": failure, "bindings": [bind(p) for p in raw_paths]})
    if failure:
        raise RuntimeError(failure)


def analyze():
    frozen = read(OUT / "RAW_RESPONSE_FREEZE.json")
    assert frozen["requests"] == 2 and len(frozen["bindings"]) == 2
    prep, rows, token_ids = public_preparation()
    expected_ids = [row["request_id"] for row in rows]
    expected_paths = {str((OUT / "raw" / (rid + ".json")).resolve()) for rid in expected_ids}
    bound_paths = [bound["path"] for bound in frozen["bindings"]]
    assert len(set(bound_paths)) == 2 and set(bound_paths) == expected_paths, "RAW_FREEZE_SET_DRIFT"
    assert {str(path.resolve()) for path in (OUT / "raw").glob("*.json")} == expected_paths, "RAW_DIRECTORY_SET_DRIFT"
    assert frozen["request_ids"] == expected_ids, "RAW_REQUEST_SET_DRIFT"
    for bound in frozen["bindings"]:
        check(bound)
    assert frozen["returned_requests"] == sum(read(path)["status"] == "RETURNED" for path in expected_paths), "RETURNED_COUNT_DRIFT"
    assert frozen["started_binding"] == bind(OUT / "STARTED.json"), "STARTED_BINDING_DRIFT"
    start = read(OUT / "STARTED.json")
    assert start["requests"] == 2 and start["request_ids"] == expected_ids, "STARTED_REQUEST_SET_DRIFT"
    assert start["preparation"] == bind(OUT / "PREPARATION.json"), "PREPARATION_BINDING_DRIFT"
    assert start["private_scoring_binding"] == prep["private_scoring_binding"], "PRIVATE_SCORING_CHAIN_DRIFT"
    # The worker only copies this opaque binding; private files open after raw sealing.
    check(prep["private_scoring_binding"])
    private = read(prep["private_scoring_binding"]["path"])
    for key in ("content_review_binding", "execution_freeze_binding", "public_requests_binding", "prompt_token_ids_binding"):
        check(private[key])
    assert private["content_review_binding"] == prep["content_review_binding"], "CONTENT_REVIEW_BINDING_DRIFT"
    assert private["public_requests_binding"] == bind(OUT / "PUBLIC_REQUESTS.json"), "PRIVATE_PUBLIC_REQUEST_DRIFT"
    assert private["prompt_token_ids_binding"] == bind(OUT / "PROMPT_TOKEN_IDS.json"), "PRIVATE_PROMPT_TOKENS_DRIFT"
    assert private["execution_freeze_binding"] in prep["bindings"], "EXECUTION_FREEZE_CHAIN_DRIFT"
    assert [item["request_id"] for item in private["requests"]] == expected_ids, "PRIVATE_REQUEST_SET_DRIFT"
    scoring = {item["request_id"]: item for item in private["requests"]}
    reviews_doc = read(private["content_review_binding"]["path"])
    assert reviews_doc["question_execution_freeze_binding"] == private["execution_freeze_binding"]
    reviews = {item["candidate_id"]: item for item in reviews_doc["reviews"]}
    for row in rows:
        item = scoring[row["request_id"]]
        check(item["candidate_binding"])
        assert item["candidate_id"] == row["candidate_id"], "PRIVATE_CANDIDATE_ID_DRIFT"
        assert item["source_binding"] == row["source_binding"] and item["question_ready_binding"] == row["question_ready_binding"]
        assert reviews[row["candidate_id"]]["candidate_binding"] == item["candidate_binding"], "REVIEWED_CANDIDATE_DRIFT"
    records, scores = [], []
    for row in rows:
        raw_path = OUT / "raw" / (row["request_id"] + ".json")
        raw = read(raw_path)
        record = {key: row[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "task_operation")}
        record.update(raw_binding=bind(raw_path), feedback_eligible=False, semantic_review="PENDING")
        try:
            assert raw["request_id"] == row["request_id"], "RAW_REQUEST_ID_DRIFT"
            assert "output_validation_errors" in raw and not raw["output_validation_errors"], "OUTPUT_VALIDATION_ERRORS"
            assert raw["status"] == "RETURNED" and raw["finish_reason"] == "stop"
            assert not response_errors(raw, Path(row["prompt_path"]).read_text(), token_ids[row["request_id"]]), "RETURNED_RESPONSE_DRIFT"
            parsed, answer = diagnostic.repair.validate(raw["raw_output"], row, read(row["schema_path"]))
            prediction = [row["display_to_canonical"][label] for label in answer]
            record.update(status="STRUCTURALLY_VALID_REASONS_SAVED", parsed=parsed, answer_labels=answer,
                canonical_prediction=prediction, request=row, source_binding=bind(row["source_path"]),
                schema_binding=bind(row["schema_path"]), mechanical_consistency=diagnostic.mechanical_consistency(row, parsed, answer))
        except Exception as error:
            record.update(status="INVALID", error=str(error))
        records.append(record)
        # Private labels are opened only after every generation response is sealed.
        candidate_binding = scoring[row["request_id"]]["candidate_binding"]
        candidate = read(candidate_binding["path"])
        gold, pred = candidate["gold"], record.get("canonical_prediction", [])
        exact = record["status"] != "INVALID" and (set(gold) == set(pred) if row["question_type"] == "multi_select" else gold == pred)
        scores.append({"candidate_id": row["candidate_id"], "task_operation": row["task_operation"], "prediction": pred,
            "gold": gold, "exact": exact, "candidate_binding": candidate_binding})
    save(OUT / "REASON_REVIEW_PACKET.json", {"requests": records, "gold_read_by_generator": False,
        "purpose": "Two feedback-routed task realizations; exploratory integration check, not a causal experiment."})
    save(OUT / "SCORES.json", {"requests": 2, "correct": sum(s["exact"] for s in scores), "records": scores,
        "raw_freeze_binding": bind(OUT / "RAW_RESPONSE_FREEZE.json"), "gold_read_after_freeze": True,
        "private_scoring_binding": prep["private_scoring_binding"], "started_binding": frozen["started_binding"]})
    print(json.dumps({"requests": 2, "valid": sum(r["status"] != "INVALID" for r in records), "correct": sum(s["exact"] for s in scores)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": lambda: run(args.execute_gpu), "analyze": analyze}[args.action]()
