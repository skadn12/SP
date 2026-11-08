#!/usr/bin/env python3
"""Versioned RoPE compatibility replay; frozen OET1 inputs remain read-only."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
TRAIN = TASK / "training"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("oet1_frozen_runner_config_replay", TRAIN / "runner.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
from model_compat import load_compatible_config

SCHEDULE = TRAIN / "schedule/SCHEDULE_FREEZE.json"
AMENDMENT = HERE / "CONFIG_BRIDGE_AMENDMENT_V2.json"
LOAD_RECEIPTS = []


def init_model(frozen, trainable, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM

    runner.require(torch.cuda.is_available(), "CUDA_REQUIRED")
    runner.require(frozen["model_directory"] == str(runner.MODEL), "WRONG_B1_BASE")
    config, receipt = load_compatible_config(runner.MODEL, runner.TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(
        str(runner.MODEL), config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True, attn_implementation="sdpa",
    )
    runner.require(model.config.rope_theta == 1000000, "LOADED_ROPE_THETA_DRIFT")
    native = runner.native()
    native.freeze_base_model(model)
    if trainable or adapter_path:
        params = frozen["training"]
        count = native.inject_lora(model, set(runner.TARGETS.split(",")),
                                   params["rank"], params["alpha"], params["dropout"])
        runner.require(count == 252, "EXPECTED_252_LORA_LINEAR_TARGETS")
        if adapter_path:
            adapter_path = Path(adapter_path).resolve()
            runner.require(HERE in adapter_path.parents, "OLD_WRONG_ROPE_ADAPTER_FORBIDDEN")
            runner.load_lora(model, torch.load(adapter_path, map_location="cpu", weights_only=True))
    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(torch.device("cuda"))
    model.train(trainable)
    LOAD_RECEIPTS.append({"config_bridge": receipt, "trainable": trainable,
                          "adapter_path": str(adapter_path) if adapter_path else None})
    return model


ORIGINAL_EVALUATE = runner.evaluate_model


def evaluate_model(model, tokenizer, evaluation_freeze, scope, output, identity, step=None):
    identity = {**identity, "configuration_amendment": runner.binding(AMENDMENT),
                "effective_rope_theta": model.config.rope_theta,
                "corrected_replay": True}
    return ORIGINAL_EVALUATE(model, tokenizer, evaluation_freeze, scope, output, identity, step)


runner.init_model = init_model
runner.evaluate_model = evaluate_model


def seal_amendment():
    bindings = [runner.binding(p) for p in (Path(__file__), HERE / "model_compat.py")]
    value = {
        "schema_version": "oet1_rope_compatibility_amendment_v1",
        "status": "FROZEN_BEFORE_CORRECTED_REPLAY",
        "schedule_freeze": runner.binding(SCHEDULE),
        "code_bindings": bindings,
        "change": "Map checkpoint rope_parameters.rope_theta to runtime rope_theta in memory.",
        "effective_rope_theta": 1000000,
        "previous_runtime_rope_theta": 10000,
        "training_schedule_changed": False,
        "evaluation_requests_or_gold_changed": False,
        "strict_parser_changed": False,
        "max_new_tokens": 32,
        "old_adapter_resume_forbidden": True,
        "old_results_disposition": "Invalid runtime configuration; retained for audit only.",
        "scientific_scope": "Exploratory old-QFP2 fallback replay, not a new-source causal experiment.",
        "prior_amendment": runner.binding(HERE / "CONFIG_BRIDGE_AMENDMENT.json"),
        "v2_change": "Move newly injected smoke LoRA to CUDA. Preserve short-control failures separately from target-format engineering gate; no accuracy-based gate or protocol tuning.",
        "test_requests": 0,
    }
    if AMENDMENT.exists():
        runner.require(runner.read(AMENDMENT) == value, "AMENDMENT_CODE_OR_PROTOCOL_DRIFT")
    else:
        runner.save(AMENDMENT, value)


def smoke(frozen, tokenizer):
    import torch

    output = HERE / "smoke_v2"
    runner.require(not output.exists(), "SMOKE_ALREADY_STARTED")
    output.mkdir()
    _, requests, private = runner.eval_inputs(Path(frozen["evaluation_freeze"]["path"]), "final")
    selected = []
    for kind in ("multi_select", "ordering"):
        selected += [r for r in requests if r["variant_id"] == "perm_00" and r["question_type"] == kind][:2]
    control_messages = [
        [
            {"role": "system", "content": "答案必须使用<box>...</box>格式，只输出答案，不要输出解释。多选答案用逗号分隔。"},
            {"role": "user", "content": "甲说：红球和蓝球属于我，绿球不属于我。选择所有正确说法。\nA.红球属于甲\nB.绿球属于甲\nC.红球不属于甲\nD.蓝球属于甲\nE.蓝球不属于甲\nF.三个球都属于甲"},
        ],
        [
            {"role": "system", "content": "答案必须使用<box>...</box>格式，只输出答案，不要输出解释。排序答案用逗号分隔字母。"},
            {"role": "user", "content": "按数字从小到大排列以下选项。\nA.3\nB.1\nC.2"},
        ],
    ]
    rows = []
    runner.seed_all(torch, frozen["training"]["seed"])
    model = init_model(frozen, False)
    for index, messages in enumerate(control_messages):
        prompt = runner.native().render_chat(tokenizer, messages, True)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        raw, generated = runner.generate_answer(model, tokenizer, ids)
        expected = ("<box>A,D</box>", "<box>B,C,A</box>")[index]
        row = {"scope": "SHORT_CONTROL", "control_id": index, "raw_output": raw,
               "output_token_ids": generated, "expected": expected, "correct": raw.strip() == expected}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    before = []
    for request in selected:
        ids = tokenizer(Path(request["prompt_file"]).read_text(), add_special_tokens=False).input_ids
        runner.require(runner.canonical_sha(ids) == request["prompt_token_ids_sha256"], "PROMPT_TOKEN_DRIFT")
        raw, generated = runner.generate_answer(model, tokenizer, ids)
        row = {"scope": "FROZEN_DEV_DIAGNOSTIC", "request_id": request["request_id"],
               "question_type": request["question_type"], "prompt_tokens": len(ids),
               "raw_output": raw, "output_token_ids": generated,
               "score": runner.score_one(raw, request, private[request["candidate_id"]])}
        rows.append(row)
        before.append(generated)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # A fresh LoRA must preserve B1 before its first update.
    runner.seed_all(torch, frozen["training"]["seed"])
    params = frozen["training"]
    count = runner.native().inject_lora(model, set(runner.TARGETS.split(",")),
                                      params["rank"], params["alpha"], params["dropout"])
    runner.require(count == 252, "EXPECTED_252_LORA_LINEAR_TARGETS")
    model.to(torch.device("cuda"))
    model.eval()
    after = []
    for request in selected:
        ids = tokenizer(Path(request["prompt_file"]).read_text(), add_special_tokens=False).input_ids
        _, generated = runner.generate_answer(model, tokenizer, ids)
        after.append(generated)
    gate = all(r["score"]["parse_valid"] for r in rows[2:]) and before == after
    result = {
        "status": "PASS" if gate else "FAIL_NEEDS_FURTHER_DIAGNOSIS",
        "diagnostic_only": True, "configuration_amendment": runner.binding(AMENDMENT),
        "target_task_generation_sanity_pass": gate,
        "short_control_all_correct": all(r["correct"] for r in rows[:2]),
        "gate_scope": "Frozen task format validity plus unchanged outputs under zero fresh LoRA; not answer accuracy or general instruction following.",
        "model_load_receipts": LOAD_RECEIPTS, "rows": rows,
        "fresh_lora_preserves_all_four_output_token_sequences": before == after,
        "max_new_tokens": 32, "runtime_versions": runner.versions(),
        "evaluation_freeze": frozen["evaluation_freeze"], "completed_at_utc": runner.now(),
    }
    runner.save(output / "RESULT.json", result)
    runner.require(gate, "CORRECTED_GENERATION_SANITY_FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train", "evaluate"))
    parser.add_argument("--arm", choices=("GENERIC", "TARGETED"), default="GENERIC")
    parser.add_argument("--role", choices=("B1", "B1_GENERIC", "B1_TARGETED"), default="B1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seal_amendment()
    frozen, tokenizer, rows, features = runner.verify_schedule(SCHEDULE, args.arm)
    if args.mode != "smoke":
        runner.require(runner.read(HERE / "smoke_v2/RESULT.json")["status"] == "PASS", "SMOKE_PASS_REQUIRED")
    with open("/tmp/gxy-selfplay-gpu.lock", "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "smoke":
            smoke(frozen, tokenizer)
            return
        if args.mode == "train":
            output = HERE / "corrected_replay/runs" / args.arm
            call = SimpleNamespace(output=output, schedule_freeze=SCHEDULE, arm=args.arm, resume=args.resume)
            runner.train(call, frozen, tokenizer, rows, features)
        else:
            output = HERE / "corrected_replay/final_evaluation" / args.role
            training_result = None if args.role == "B1" else HERE / "corrected_replay/runs" / args.role.removeprefix("B1_") / "RESULT.json"
            if training_result:
                runner.require((training_result.parent / "CORRECTED_RUN_RECEIPT.json").exists(), "CORRECTED_TRAINING_RECEIPT_REQUIRED")
            call = SimpleNamespace(output=output, schedule_freeze=SCHEDULE, role=args.role, training_result=training_result)
            runner.evaluate_endpoint(call, frozen, tokenizer)
        runner.save(output / "CORRECTED_RUN_RECEIPT.json", {
            "status": "COMPLETED", "mode": args.mode, "configuration_amendment": runner.binding(AMENDMENT),
            "model_load_receipts": LOAD_RECEIPTS, "completed_at_utc": runner.now(),
            "result": runner.binding(output / ("RESULT.json" if args.mode == "train" else "SUMMARY.json")),
        })


if __name__ == "__main__":
    main()
