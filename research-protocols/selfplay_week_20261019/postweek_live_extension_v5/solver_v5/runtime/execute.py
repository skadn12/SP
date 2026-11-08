"""Public-only native direct evaluation; every planned request gets one terminal row."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import common as c


def terminal(row, status, **extra):
    return {key: row[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "branch", "variant_id", "slot", "seed", "prompt_sha256")} | {
        "status": status, "raw_output": None, "response_committed_at_utc": c.now(),
        "model_path": str(c.MODEL), "gold_read_by_solver": False,
        "automatic_retry": False, "source_truncation": False, **extra}


def live(run, audit):
    run = Path(run).resolve()
    c.require(not (run / "STARTED.json").exists(), "ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    manifest = c.verify(run, audit)
    admission = c.read(run / "CONTENT_ADMISSION.json")
    c.require(admission["status"] == "PASS", "SOURCE_CONTENT_ADMISSION_REQUIRED_BEFORE_GPU")
    rows = manifest["requests"]
    expected_token_ids = c.read(run / "PROMPT_TOKEN_IDS.json")
    c.save(run / "STARTED.json", {"started_at_utc": c.now(), "manifest_binding": c.bind(run / "manifest.json"),
           "independent_audit_binding": c.bind(audit), "gold_opened": False, "public_only": True})
    if not rows:
        c.save(run / "COMPLETE.json", {"status": "COMPLETE", "completed_at_utc": c.now(),
            "actual_submitted_requests": 0, "returned_requests": 0, "model_load_wall_seconds": 0,
            "wall_seconds": 0, "gold_opened": False, "model_backend_created": False})
        (run / "responses").mkdir()
        c.freeze_raw(run, manifest)
        return
    backend, submitted, done = None, set(), set()
    started, load_seconds, failure = time.perf_counter(), None, None
    try:
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        _, solver = c.a.native()
        solver.ensure_vllm_environment()
        backend = VLLMSolverBackend(model_path=c.MODEL, tokenizer_path=c.TOKENIZER,
            max_model_len=16384, temperature=.8, top_p=.9, max_tokens=32, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        c.require(str(backend.model_identifier) == manifest["model"], "BACKEND_MODEL_MISMATCH")
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            batch_id = f"batch{offset // 8:04d}"
            prompts, params = [], []
            for row in batch:
                path = run / "prompts" / (row["request_id"] + ".txt")
                c.require(c.sha(path) == row["prompt_sha256"], "PROMPT_CHANGED")
                prompts.append(path.read_text())
                params.append(c.a.sampling_params(row, manifest["decoder_contract"]["structured_output_api"]))
            submitted_at, batch_start = c.now(), time.perf_counter()
            c.save(run / "batches" / (batch_id + "_SUBMITTED.json"), {"submitted_at_utc": submitted_at,
                   "request_ids": [row["request_id"] for row in batch],
                   "sampling_params": [param.to_dict() if hasattr(param, "to_dict") else repr(param) for param in params],
                   "automatic_retry": False})
            submitted.update(row["request_id"] for row in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            # Preserve every returned byte before checking structural drift.
            for index, row in enumerate(batch):
                if index >= len(outputs):
                    record = terminal(row, "BACKEND_ERROR", error="OUTPUT_COUNT_MISMATCH_MISSING_RETURN",
                        request_submitted_at_utc=submitted_at, usage=None, usage_status="UNKNOWN_NOT_ZERO")
                else:
                    output = outputs[index]
                    completions = list(output.outputs)
                    primary = completions[0] if completions else None
                    record = terminal(row, "RETURNED", raw_output=primary.text if primary else None,
                        request_submitted_at_utc=submitted_at,
                        prompt_token_ids=list(output.prompt_token_ids),
                        completion_token_ids=list(primary.token_ids) if primary else None,
                        output_validation_errors=[], validation_status="DEFERRED_TO_FROZEN_SIDECAR",
                        all_completions=[{"text": item.text, "token_ids": list(item.token_ids),
                            "finish_reason": item.finish_reason, "stop_reason": item.stop_reason} for item in completions],
                        metadata={"prompt_token_count": len(output.prompt_token_ids),
                            "completion_token_count": len(primary.token_ids) if primary else None,
                            "finish_reason": primary.finish_reason if primary else None,
                            "stop_reason": primary.stop_reason if primary else None,
                            "temperature": .8, "top_p": .9, "max_tokens": 32,
                            "structured_output_api": manifest["decoder_contract"]["structured_output_api"],
                            "enable_thinking": False, "backend_request_id": output.request_id,
                            "cached_tokens": getattr(output, "num_cached_tokens", None)})
                c.save(run / "responses" / (row["request_id"] + ".json"), record)
                done.add(row["request_id"])
                # Preserve all returned text/tokens first, then check against the
                # exact pre-frozen token sequence, not merely its length.
                validation = []
                if index >= len(outputs):
                    validation.append("MISSING_BACKEND_RETURN")
                else:
                    if output.prompt != prompts[index]:
                        validation.append("OUTPUT_PROMPT_TEXT_DRIFT")
                    if list(output.prompt_token_ids) != expected_token_ids[row["request_id"]]:
                        validation.append("OUTPUT_PROMPT_TOKEN_IDS_DRIFT")
                    if len(completions) != 1:
                        validation.append("COMPLETION_COUNT_MISMATCH")
                c.save(run / "response_validation" / (row["request_id"] + ".json"), {
                    "request_id": row["request_id"], "checked_after_raw_commit": True,
                    "errors": validation, "response_binding": c.bind(run / "responses" / (row["request_id"] + ".json"))})
            if len(outputs) > len(batch):
                c.save(run / "batches" / (batch_id + "_EXTRA_OUTPUTS.json"), {
                    "error": "OUTPUT_COUNT_MISMATCH_EXTRA_RETURN", "extra_outputs": [
                        {"request_id": output.request_id, "prompt": output.prompt,
                         "prompt_token_ids": list(output.prompt_token_ids), "completions": [
                             {"text": item.text, "token_ids": list(item.token_ids), "finish_reason": item.finish_reason,
                              "stop_reason": item.stop_reason} for item in output.outputs]}
                        for output in outputs[len(batch):]]})
            c.require(len(outputs) == len(batch), "OUTPUT_COUNT_MISMATCH")
            c.save(run / "batches" / (batch_id + "_COMPLETE.json"), {"completed_at_utc": c.now(),
                "request_ids": [row["request_id"] for row in batch], "wall_seconds": time.perf_counter() - batch_start})
            print(json.dumps({"status": "LIVE", "completed": len(done), "planned": len(rows)}), flush=True)
    except BaseException as error:
        failure = {"error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as error:
                failure = failure or {"error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc()}
        if failure:
            c.save(run / "FAILED.json", {"status": "FAILED", "failed_at_utc": c.now(),
                "submitted_request_count": len(submitted), "returned_request_count": len(done),
                "model_load_wall_seconds": load_seconds, "wall_seconds": time.perf_counter() - started,
                "automatic_retry": False, **failure})
            for row in rows:
                if row["request_id"] in done:
                    continue
                sent = row["request_id"] in submitted
                c.save(run / "responses" / (row["request_id"] + ".json"), terminal(row,
                    "BACKEND_ERROR" if sent else "NOT_SUBMITTED", usage=None,
                    usage_status="UNKNOWN_NOT_ZERO" if sent else "NO_REQUEST_SENT", **failure))
        else:
            c.save(run / "COMPLETE.json", {"status": "COMPLETE", "completed_at_utc": c.now(),
                "actual_submitted_requests": len(submitted), "returned_requests": len(done),
                "model_load_wall_seconds": load_seconds, "wall_seconds": time.perf_counter() - started,
                "gold_opened": False})
        c.freeze_raw(run, manifest)
    if failure:
        raise RuntimeError("LIVE_FAILED_WITH_FIXED_DENOMINATOR: " + failure["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "live"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = c.verify(args.run, args.audit)
        print(json.dumps({"status": "PASS", "public_only": True, "requests": manifest["request_count"]}))
    else:
        live(args.run, args.audit)


if __name__ == "__main__":
    main()
