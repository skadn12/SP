#!/usr/bin/env python3
"""Read-only, four-request diagnostic of the frozen endpoint with a larger decode cap.

This deliberately does not modify the frozen evaluation, its reports, or prior runs.
It answers whether the 32-token cap alone explains invalid output on a tiny fixed
subset. Outputs are diagnostic only and must not be used for scientific metrics.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
TRAIN = TASK / "training"
sys.path.insert(0, str(TRAIN))

spec = importlib.util.spec_from_file_location("oet1_frozen_runner_diagnostic", TRAIN / "runner.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

SCHEDULE = TRAIN / "schedule/SCHEDULE_FREEZE.json"
OUTPUT = Path(__file__).resolve().parent / "diagnostic_output"
MAX_NEW_TOKENS = 128


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_with_larger_cap(model, tokenizer, ids):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompt_length = len(ids)
    require = runner.require
    require(prompt_length + MAX_NEW_TOKENS <= 16384, "DIAGNOSTIC_PROMPT_CAP")
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)

    class StopOnClosedBox(StoppingCriteria):
        def __call__(self, generated, scores, **kwargs):
            tail = tokenizer.decode(generated[0, prompt_length:].tolist(), skip_special_tokens=True)
            return "</box>" in tail

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnClosedBox()]),
        )
    token_ids = generated[0, prompt_length:].detach().cpu().tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True), token_ids


def main():
    import torch

    frozen, tokenizer, _, _ = runner.verify_schedule(SCHEDULE, "GENERIC")
    eval_frozen, requests, private = runner.eval_inputs(Path(frozen["evaluation_freeze"]["path"]), "final")
    by_type = {"multi_select": [], "ordering": []}
    for request in requests:
        if request["variant_id"] == "perm_00" and request["question_type"] in by_type:
            if len(by_type[request["question_type"]]) < 2:
                by_type[request["question_type"]].append(request)
    selected = by_type["multi_select"] + by_type["ordering"]
    runner.require(len(selected) == 4, "EXPECTED_TWO_MC_AND_TWO_ORDER_REQUESTS")

    models = [
        ("B1", None),
        ("B1_GENERIC", TASK / "training/runs/GENERIC/adapter/adapter_model.pt"),
        ("B1_TARGETED", TASK / "training/runs/TARGETED/adapter/adapter_model.pt"),
    ]
    result = {
        "schema_version": "oet1_followup_eval_cap_diagnostic_v1",
        "diagnostic_only": True,
        "scientific_metrics_claimed": False,
        "frozen_evaluation_unchanged": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop_at_first_closed_box": True,
        "started_at_utc": runner.now(),
        "schedule_freeze": runner.binding(SCHEDULE),
        "evaluation_freeze": runner.binding(Path(eval_frozen["protocol"]["path"])),
        "selected_requests": [
            {k: request[k] for k in ("request_id", "candidate_id", "question_type", "variant_id", "prompt_sha256")}
            for request in selected
        ],
        "models": {},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for role, adapter_path in models:
        identity = {
            "logical_model": role,
            "base_model_directory": str(runner.MODEL),
            "base_model_bindings": frozen["model_bindings"],
        }
        if adapter_path is not None:
            identity["adapter_sha256"] = file_hash(adapter_path)
        model = runner.init_model(frozen, trainable=False, adapter_path=adapter_path)
        model.eval()
        model.config.use_cache = True
        rows = []
        for request in selected:
            prompt = Path(request["prompt_file"]).read_text()
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            runner.require(runner.canonical_sha(ids) == request["prompt_token_ids_sha256"], "PROMPT_TOKEN_DRIFT")
            started = time.monotonic()
            raw, token_ids = generate_with_larger_cap(model, tokenizer, ids)
            score = runner.score_one(raw, request, private[request["candidate_id"]])
            row = {
                "request_id": request["request_id"],
                "candidate_id": request["candidate_id"],
                "question_type": request["question_type"],
                "raw_output": raw,
                "output_tokens": len(token_ids),
                "ended_by_cap": len(token_ids) == MAX_NEW_TOKENS,
                "score_diagnostic": score,
                "wall_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            print(json.dumps({"model": role, "request": row}, ensure_ascii=False), flush=True)
        result["models"][role] = rows
        del model
        gc.collect()
        torch.cuda.empty_cache()

    result["finished_at_utc"] = runner.now()
    out_path = OUTPUT / "RESULT.json"
    with out_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "DIAGNOSTIC_ONLY_COMPLETED", "result_path": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
