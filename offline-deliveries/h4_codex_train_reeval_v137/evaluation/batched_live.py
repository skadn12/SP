"""One-shot v137 scheduling adapter; native prompts, decoding and scores stay fixed.

The only generation change from the frozen v15 runner is scheduling eight
requests in one ``LLM.generate`` call. Every request still has its own seed and
SamplingParams. Identical numerical outputs to serial execution are not claimed.
No inference is performed by importing this module.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


BATCH_SIZE = 8
EXECUTION_VERSION = "v137_fixed_order_vllm_batch8_v1"


def execution_contract():
    return {
        "version": EXECUTION_VERSION,
        "batch_size": BATCH_SIZE,
        "batch_order": "contiguous frozen manifest order; identical for both models",
        "request_sampling_params": "one independent SamplingParams per request",
        "automatic_retry": False,
        "score_based_filtering": False,
        "gold_used_for_generation": False,
        "native_decoder_contract_changed": False,
        "serial_numerical_equivalence_not_claimed": True,
        "prefix_caching": False,
    }


def _params(probe, row, expected_api):
    """Mirror native _structured_generate using only public labels and a seed."""
    from vllm import SamplingParams, sampling_params

    question_type = row["question_type"]
    if question_type not in {"multi_select", "ordering"}:
        raise RuntimeError("UNSUPPORTED_STRUCTURED_QUESTION_TYPE:" + question_type)
    public_items = row["variant"].get("options") or row["variant"].get("items")
    choices = probe.solver.structured_box_choices(
        question_type, allowed_item_ids=public_items.keys(),
    )
    expected_count = 63 if question_type == "multi_select" else 720
    if len(choices) != expected_count:
        raise RuntimeError("STRUCTURED_CHOICE_SET_MISMATCH:" + row["request_id"])
    api_name, _ = probe.solver.detect_structured_output_api()
    if api_name != expected_api:
        raise RuntimeError("STRUCTURED_OUTPUT_API_CHANGED")
    common = {
        "temperature": probe.solver.STRUCTURED_TEMPERATURE,
        "top_p": probe.solver.TOP_P,
        "max_tokens": probe.solver.MAX_TOKENS,
        "seed": row["seed"],
        "stop": ["</box>"],
        "include_stop_str_in_output": True,
    }
    if api_name.startswith("structured_outputs="):
        params = SamplingParams(
            **common,
            structured_outputs=sampling_params.StructuredOutputsParams(choice=choices),
        )
    else:
        params = SamplingParams(
            **common,
            guided_decoding=sampling_params.GuidedDecodingParams(choice=choices),
        )
    return params


def _native_metadata(probe, row, output, completion, api_name):
    """Keep precisely the response metadata fields emitted by native decoding."""
    return {
        "sampling_seed": row["seed"],
        "prompt_token_count": len(output.prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "temperature": probe.solver.STRUCTURED_TEMPERATURE,
        "top_p": probe.solver.TOP_P,
        "max_tokens": probe.solver.MAX_TOKENS,
        "structured_output_api": api_name,
        "structured_output_version": probe.solver.STRUCTURED_OUTPUT_VERSION,
    }


def live(ev, side):
    """Run one independently approved model side, persist native-shaped responses."""
    runner = ev.frozen(require_independent=True)
    ev.require(Path(sys.prefix) == runner.EXPECTED_VENV, "USE_PROVEN_LOCAL4B_VENV")
    ev.require(side in ev.MODELS, "UNKNOWN_MODEL_SIDE")
    run = ev.HERE / "run" / side
    ev.require(not (run / "STARTED.json").exists(), "ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    ev.require(not (run / "COMPLETE.json").exists() and not (run / "FAILED.json").exists(),
               "TERMINAL_RUN_EXISTS_NO_AUTOMATIC_RERUN")
    ev.require(not any((run / "responses").glob("*.json")), "ORPHAN_RESPONSES_NO_AUTOMATIC_RERUN")
    manifest = ev.read(run / "manifest.json")
    runner._verify_manifest(run, manifest)
    probe = runner.load_probe("v137_batched_live_probe_" + side)
    ev.require(probe.solver.generation_config("multi_select") == manifest["decoder_contract"],
               "DECODER_CONTRACT_CHANGED")
    ev.require(manifest["solver_model"] == ev.MODELS[side], "MODEL_SIDE_MANIFEST_MISMATCH")
    rows = manifest["requests"]
    ids = [row["request_id"] for row in rows]
    ev.require(len(ids) == len(set(ids)) == manifest["request_count"], "REQUEST_IDS_OR_COUNT_INVALID")

    # The scoring dictionaries are never passed into vLLM or used by _params.
    questions = ev.read(run / "questions.json")["candidates"]
    gold = ev.read(run / "gold.json")
    q_by_id = {q["candidate_id"]: q for q in questions}
    contract = execution_contract()
    started = time.perf_counter()
    ev.save(run / "STARTED.json", {
        "started_at_utc": ev.now(), "manifest_sha256": ev.sha(run / "manifest.json"),
        "execution_contract": contract,
    })
    executed = []
    submitted = []
    backend = None
    try:
        ev.save(run / "BATCH_EXECUTION.json", {
            **contract, "manifest_sha256": ev.sha(run / "manifest.json"),
            "planned_requests": len(rows),
            "planned_batches": (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE,
            "decoder_contract": manifest["decoder_contract"],
        })
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend

        probe.solver.ensure_vllm_environment()
        backend = VLLMSolverBackend(
            model_path=Path(manifest["model"]), tokenizer_path=runner.TOKENIZER,
            max_model_len=16384, temperature=0.8, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80,
        )
        ev.require(str(backend.model_identifier) == manifest["model"], "BACKEND_MODEL_IDENTITY_MISMATCH")
        api_name = manifest["decoder_contract"]["structured_output_api"]
        for offset in range(0, len(rows), BATCH_SIZE):
            batch = rows[offset:offset + BATCH_SIZE]
            prompts = []
            params = []
            for row in batch:
                prompt_path = run / "prompts" / (row["request_id"] + ".txt")
                ev.require(ev.sha(prompt_path) == row["prompt_sha256"], "PROMPT_CHANGED:" + row["request_id"])
                prompts.append(prompt_path.read_text(encoding="utf-8"))
                params.append(_params(probe, row, api_name))
            batch_id = offset // BATCH_SIZE
            batch_stem = "batch" + str(batch_id).zfill(4)
            batch_started = time.perf_counter()
            ev.save(run / "batches" / (batch_stem + "_SUBMITTED.json"), {
                "batch_id": batch_id, "submitted_at_utc": ev.now(),
                "request_ids": [row["request_id"] for row in batch],
                "prompt_sha256": [row["prompt_sha256"] for row in batch],
                "seeds": [row["seed"] for row in batch],
                "request_count": len(batch), "automatic_retry": False,
            })
            submitted.extend(row["request_id"] for row in batch)
            # vLLM documents one-to-one SamplingParams and input-order outputs.
            # These are already rendered chat strings; do not re-template them.
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            ev.require(len(outputs) == len(batch), "BATCH_OUTPUT_COUNT_MISMATCH")
            for row, expected_prompt, output in zip(batch, prompts, outputs):
                ev.require(output.prompt == expected_prompt, "BATCH_PROMPT_ORDER_MISMATCH:" + row["request_id"])
                ev.require(len(output.prompt_token_ids) == row["prompt_tokens"],
                           "PROMPT_TOKEN_COUNT_CHANGED:" + row["request_id"])
                ev.require(len(output.outputs) == 1, "COMPLETION_COUNT_MISMATCH:" + row["request_id"])
                completion = output.outputs[0]
                metadata = _native_metadata(probe, row, output, completion, api_name)
                q = dict(q_by_id[row["candidate_id"]])
                q["gold_content_ids"] = gold[row["candidate_id"]]
                score = runner.score_response(q, row, completion.text)
                decoder_errors = [
                    key for key in ("temperature", "top_p", "max_tokens", "structured_output_api",
                                    "structured_output_version")
                    if metadata.get(key) != manifest["decoder_contract"].get(key)
                ]
                if backend.model_identifier != manifest["model"]:
                    decoder_errors.append("model_identifier")
                valid = (metadata.get("finish_reason") in ("stop", "eos")
                         and score.get("scored") is True and not decoder_errors)
                record = {
                    "request_id": row["request_id"], "candidate_id": row["candidate_id"],
                    "root_id": row["root_id"], "solver_model": manifest["solver_model"],
                    "model_path": manifest["model"], "question_type": row["question_type"],
                    "variant_id": row["variant"]["variant_id"], "mode": "full_context",
                    "slot": row["slot"], "seed": row["seed"], "raw_output": completion.text,
                    "metadata": metadata, "score": score, "evaluation_valid": valid,
                    "data_errors": [] if valid else (
                        ["DECODER_METADATA_MISMATCH:" + key for key in decoder_errors]
                        or ["INCOMPLETE_OR_UNPARSEABLE_RESPONSE"]
                    ),
                }
                ev.save(run / "responses" / (row["request_id"] + ".json"), record)
                executed.append(row["request_id"])
            ev.save(run / "batches" / (batch_stem + "_COMPLETE.json"), {
                "batch_id": batch_id, "completed_at_utc": ev.now(),
                "request_ids": [row["request_id"] for row in batch],
                "request_count": len(batch), "wall_seconds": time.perf_counter() - batch_started,
                "automatic_retry": False,
            })
            if batch_id % 10 == 0 or len(executed) == len(rows):
                print(json.dumps({"status": "LIVE", "side": side, "completed_requests": len(executed),
                                  "planned_requests": len(rows), "batch_size": BATCH_SIZE}), flush=True)
        ev.require(executed == ids and submitted == ids, "COMPLETED_REQUEST_ORDER_OR_COUNT_MISMATCH")
        ev.save(run / "COMPLETE.json", {
            "status": "COMPLETE", "actual_call_count": len(executed),
            "training_started": False, "training_promotion": False,
            "completed_at_utc": ev.now(), "wall_seconds": time.perf_counter() - started,
            "execution_contract": contract,
        })
    except Exception as error:
        ev.save(run / "FAILED.json", {
            "status": "FAILED", "error": str(error), "traceback": traceback.format_exc(),
            "executed_request_ids": executed, "submitted_request_ids": submitted,
            "failed_at_utc": ev.now(), "automatic_retry": False,
            "execution_contract": contract,
        })
        raise
    finally:
        if backend is not None:
            backend.close()
    runner.report(run)
