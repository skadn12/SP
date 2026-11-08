"""Run the serial two-policy Round-0 LoRA training smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_text, canonical_json, sha256_file
from . import (
    BASE_MODEL,
    DATA_RUN,
    DEFAULT_RUN_ROOT,
    EXPECTED_OPTIMIZER_STEPS,
    RECIPE_VERSION,
    REQUIRED_MAX_LENGTH,
    REPO_ROOT,
    SMOKE_EPOCHS,
    SMOKE_VERSION,
    SOURCE_RECIPE_REPORT,
    TRAINER,
)
from .builder import audit_loss_mask, build_smoke_manifests, load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=DATA_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, text: str) -> None:
    atomic_text(path, text)
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        offset = end
    return rows


def model_snapshot(model_dir: Path) -> dict[str, Any]:
    files = {
        path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted(model_dir.iterdir()) if path.is_file()
    }
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"model_snapshot_incomplete:{model_dir}")
    return {"model_dir": str(model_dir), "files": files}


def run_training(*, policy: str, train_file: Path, output_dir: Path, log_path: Path, gradient_path: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.train_smoke_runner",
        "--model", str(BASE_MODEL), "--tokenizer", str(BASE_MODEL), "--train-file", str(train_file),
        "--output-dir", str(output_dir), "--max-length", str(REQUIRED_MAX_LENGTH),
        "--epochs", str(SMOKE_EPOCHS), "--batch-size", str(recipe["batch_size"]),
        "--grad-accum", str(recipe["grad_accum"]), "--lr", str(recipe["lr"]),
        "--min-lr-ratio", str(recipe["min_lr_ratio"]), "--warmup-ratio", str(recipe["warmup_ratio"]),
        "--weight-decay", str(recipe["weight_decay"]), "--max-grad-norm", str(recipe["max_grad_norm"]),
        "--lora-rank", str(recipe["lora_rank"]), "--lora-alpha", str(recipe["lora_alpha"]),
        "--lora-dropout", str(recipe["lora_dropout"]), "--target-modules", ",".join(recipe["target_modules"]),
        "--seed", str(recipe["seed"]),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SMOKE_GRADIENT_AUDIT_PATH"] = str(gradient_path)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    log_text = log_path.read_text(encoding="utf-8")
    parsed = parse_json_lines(log_text)
    finish = next((row for row in reversed(parsed) if row.get("finished") is True), {})
    initial = next((row for row in parsed if "total_update_steps" in row), {})
    step_logs = [row for row in parsed if "train_loss_recent" in row]
    gradient = load_json(gradient_path) if gradient_path.is_file() else {}
    adapter_config = output_dir / "adapter_config.json"
    adapter_state = output_dir / "adapter_model.pt"
    adapter_saved = adapter_config.is_file() and adapter_state.is_file()
    status = "PASS" if (
        result.returncode == 0 and finish.get("finished") is True
        and initial.get("total_update_steps") == EXPECTED_OPTIMIZER_STEPS
        and gradient.get("optimizer_steps") == EXPECTED_OPTIMIZER_STEPS
        and gradient.get("backward_calls") == 16
        and gradient.get("all_losses_finite") is True
        and gradient.get("all_gradients_finite_nonzero") is True
        and adapter_saved
    ) else "FAIL"
    report = {
        "training_smoke_report_version": f"{SMOKE_VERSION}_training_report_v1",
        "policy": policy, "return_code": result.returncode,
        "command": command, "log_path": str(log_path), "log_sha256": file_sha(log_path),
        "gradient_audit_path": str(gradient_path), "gradient_audit_sha256": file_sha(gradient_path) if gradient_path.is_file() else None,
        "initial_summary": initial, "finish_summary": finish, "step_logs": step_logs,
        "optimizer_steps_observed": gradient.get("optimizer_steps"), "backward_calls_observed": gradient.get("backward_calls"),
        "all_losses_finite": gradient.get("all_losses_finite", False),
        "all_gradients_finite_nonzero": gradient.get("all_gradients_finite_nonzero", False),
        "adapter_saved": adapter_saved,
        "adapter_config_sha256": file_sha(adapter_config) if adapter_config.is_file() else None,
        "adapter_state_sha256": file_sha(adapter_state) if adapter_state.is_file() else None,
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "status": status,
    }
    return report


def main() -> int:
    args = parse_args()
    data_run = args.data_run.resolve()
    run_root = args.run_root.resolve()
    model_sha_before = model_snapshot(BASE_MODEL)
    built = build_smoke_manifests(data_run=data_run, run_root=run_root)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    clean_audit = audit_loss_mask(built["clean_smoke"], tokenizer=tokenizer, policy="M_clean_policy")
    inclusive_audit = audit_loss_mask(built["inclusive_smoke"], tokenizer=tokenizer, policy="M_inclusive_policy")
    write_json(run_root / "clean_loss_mask_audit.json", clean_audit)
    write_json(run_root / "inclusive_loss_mask_audit.json", inclusive_audit)
    source_recipe = load_json(SOURCE_RECIPE_REPORT)
    if source_recipe.get("status") != "success" or source_recipe.get("merged_ok") is not True or source_recipe.get("trainer_finished_summary", {}).get("finished") is not True:
        raise RuntimeError("verified_source_recipe_evidence_not_successful")
    verified_adapter = Path(source_recipe["outputs"]["adapter_dir"])
    if not (verified_adapter / "adapter_config.json").is_file() or not (verified_adapter / "adapter_model.pt").is_file():
        raise RuntimeError(f"verified_source_recipe_adapter_missing:{verified_adapter}")
    hp = source_recipe["hyperparameters"]
    recipe = {
        "recipe_version": RECIPE_VERSION,
        "source_recipe_report": str(SOURCE_RECIPE_REPORT),
        "source_recipe_report_sha256": sha256_file(SOURCE_RECIPE_REPORT),
        "source_recipe_success_evidence": {
            "status": source_recipe["status"], "finished": source_recipe["trainer_finished_summary"]["finished"],
            "global_step": source_recipe["trainer_finished_summary"]["global_step"], "merged_ok": source_recipe["merged_ok"],
            "adapter_dir": str(verified_adapter),
            "adapter_config_sha256": sha256_file(verified_adapter / "adapter_config.json"),
            "adapter_state_sha256": sha256_file(verified_adapter / "adapter_model.pt"),
        },
        "trainer_path": str(TRAINER), "trainer_sha256": sha256_file(TRAINER),
        "model": str(BASE_MODEL), "model_config_sha256": sha256_file(BASE_MODEL / "config.json"),
        "tokenizer": str(BASE_MODEL), "tokenizer_config_sha256": sha256_file(BASE_MODEL / "tokenizer_config.json"),
        "max_length": REQUIRED_MAX_LENGTH,
        "epochs": SMOKE_EPOCHS,
        "batch_size": int(hp["batch_size"]), "grad_accum": int(hp["grad_accum"]),
        "lr": float(hp["lr"]), "min_lr_ratio": float(hp["min_lr_ratio"]),
        "warmup_ratio": float(hp["warmup_ratio"]), "weight_decay": float(hp["weight_decay"]),
        "max_grad_norm": float(hp["max_grad_norm"]), "lora_rank": int(hp["lora_rank"]),
        "lora_alpha": int(hp["lora_alpha"]), "lora_dropout": float(hp["lora_dropout"]),
        "target_modules": list(hp["target_modules"]), "seed": int(hp["seed"]),
        "optimizer": "torch.optim.AdamW", "scheduler": "trainer.lr_at_step cosine with warmup",
        "optimizer_defaults": {"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "bf16": True, "gradient_checkpointing": True,
        "loss_mask": "existing train_qwen3_lora_sft.py tokenize_record assistant-only prefix mask",
        "smoke_schedule_reason": "one epoch over 16 records with batch_size=1 and grad_accum=4 gives four optimizer steps; every record is used once",
        "forbidden_changes": ["learning_rate", "lora_parameters", "optimizer", "scheduler", "batch_size", "grad_accum", "loss_mask"],
    }
    recipe_fp_payload = {key: value for key, value in recipe.items() if key not in {"model", "tokenizer", "source_recipe_report", "trainer_path"}}
    recipe_fp_payload["trainer_sha256"] = recipe["trainer_sha256"]
    recipe_fp = hashlib.sha256(canonical_json(recipe_fp_payload).encode()).hexdigest()
    recipe["recipe_fingerprint"] = recipe_fp
    write_json(run_root / "training_recipe.json", recipe)
    write_json(run_root / "training_recipe_fingerprint.json", {
        "training_recipe_fingerprint_version": f"{SMOKE_VERSION}_fingerprint_v1",
        "fingerprint": recipe_fp, "same_for_both_policies": True,
        "fingerprint_payload": recipe_fp_payload,
    })
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    for policy, manifest_name in (("M_clean_policy", "clean_smoke_manifest.jsonl"), ("M_inclusive_policy", "inclusive_smoke_manifest.jsonl")):
        output_dir = run_root / ("clean_adapter" if policy == "M_clean_policy" else "inclusive_adapter")
        reports[policy] = run_training(
            policy=policy, train_file=run_root / manifest_name, output_dir=output_dir,
            log_path=logs / ("clean_train.log" if policy == "M_clean_policy" else "inclusive_train.log"),
            gradient_path=logs / ("clean_gradient_audit.json" if policy == "M_clean_policy" else "inclusive_gradient_audit.json"),
            recipe=recipe,
        )
        atomic_text(logs / ("clean_train.log.sha256" if policy == "M_clean_policy" else "inclusive_train.log.sha256"), reports[policy]["log_sha256"] + "\n")
        atomic_text(logs / ("clean_gradient_audit.json.sha256" if policy == "M_clean_policy" else "inclusive_gradient_audit.json.sha256"), reports[policy]["gradient_audit_sha256"] + "\n")
        write_json(run_root / ("clean_training_smoke_report.json" if policy == "M_clean_policy" else "inclusive_training_smoke_report.json"), reports[policy])
        if reports[policy]["status"] != "PASS":
            raise RuntimeError(f"training_smoke_failed:{policy}")
    reload_reports: dict[str, Any] = {}
    for policy, manifest_name in (("M_clean_policy", "clean_smoke_manifest.jsonl"), ("M_inclusive_policy", "inclusive_smoke_manifest.jsonl")):
        out = run_root / ("clean_adapter_reload_test.json" if policy == "M_clean_policy" else "inclusive_adapter_reload_test.json")
        command = [sys.executable, "-m", "scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.reload_and_generate", "--model", str(BASE_MODEL), "--tokenizer", str(BASE_MODEL), "--adapter-dir", str(run_root / ("clean_adapter" if policy == "M_clean_policy" else "inclusive_adapter")), "--manifest", str(run_root / manifest_name), "--output", str(out), "--max-length", str(REQUIRED_MAX_LENGTH)]
        result = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0 and not out.is_file():
            raise RuntimeError(f"adapter_reload_process_failed:{policy}:{result.returncode}:{result.stdout[-1000:]}")
        reload_reports[policy] = load_json(out)
        write_json(out, reload_reports[policy])
        if reload_reports[policy].get("status") != "PASS":
            raise RuntimeError(f"adapter_reload_failed:{policy}")
    model_sha_after = model_snapshot(BASE_MODEL)
    source_sha_after = {name: sha256_file(data_run / name) for name in built["source_snapshot"]["files"]}
    source_sha_drift = source_sha_after != built["source_snapshot"]["files"]
    model_sha_drift = model_sha_after != model_sha_before
    if source_sha_drift or model_sha_drift:
        raise RuntimeError(f"frozen_source_drift:source={source_sha_drift}:model={model_sha_drift}")
    final = {
        "final_smoke_report_version": f"{SMOKE_VERSION}_final_v1", "status": "READY_FOR_FULL_TRAINING",
        "training_recipe_fingerprint": recipe_fp, "recipe_fingerprint_equal": True,
        "sample_count_per_policy": 16, "shared_count": 12, "matched_replacement_count": 4,
        "optimizer_steps_per_policy": {policy: reports[policy]["optimizer_steps_observed"] for policy in reports},
        "max_length": REQUIRED_MAX_LENGTH, "truncated_sample_count": clean_audit["truncated_sample_count"] + inclusive_audit["truncated_sample_count"],
        "assistant_supervision_missing": clean_audit["assistant_supervision_missing"] + inclusive_audit["assistant_supervision_missing"],
        "loss_mask_status": {"M_clean_policy": clean_audit["status"], "M_inclusive_policy": inclusive_audit["status"]},
        "loss_finite": all(reports[policy]["all_losses_finite"] for policy in reports),
        "gradient_finite_nonzero": all(reports[policy]["all_gradients_finite_nonzero"] for policy in reports),
        "adapter_saved_and_reloaded": all(reports[policy]["adapter_saved"] and reload_reports[policy]["status"] == "PASS" for policy in reports),
        "train_dev_test_candidate_root_leakage": {"candidate_overlap": 0, "root_overlap": 0},
        "source_jsonl_manifest_sha_drift": source_sha_drift,
        "base_model_file_sha_drift": model_sha_drift,
        "base_model_snapshot": model_sha_before,
        "model_training_scope": "smoke_only_16_rows_each_serial_4_steps",
        "formal_training_started": False, "dev_test_evaluation_started": False,
        "solver_calls": 0, "external_api_calls": 0, "semantic_auditor_calls": 0,
        "sample_details": {
            "M_clean_policy": built["clean_smoke"], "M_inclusive_policy": built["inclusive_smoke"],
        },
    }
    write_json(run_root / "final_smoke_report.json", final)
    md = "\n".join([
        "# Round-0 B0 Training Smoke", "",
        f"Status: **{final['status']}**", "",
        "Both policies used the verified Qwen3-4B custom LoRA SFT recipe with only the frozen `--max-length 8192` input contract and the one-epoch smoke schedule.",
        "No formal 386-row training, Dev/Test evaluation, Solver call, external API, or semantic auditor was started.", "",
        "- Samples per policy: `16` (`12` shared Clean + `4` matched replacements)",
        "- Optimizer steps: `4` per policy, serial, same seed and recipe fingerprint",
        f"- Truncated samples: `{final['truncated_sample_count']}`; missing assistant supervision: `{final['assistant_supervision_missing']}`",
        f"- Loss finite: `{final['loss_finite']}`; nonzero finite gradients: `{final['gradient_finite_nonzero']}`",
        f"- Adapter save/reload/legal B0 generation: `{final['adapter_saved_and_reloaded']}`",
        "",
        "Main outputs are in this run directory; every manifest and major JSON has a SHA-256 sidecar.",
    ]) + "\n"
    write_text(run_root / "final_smoke_report.md", md)
    print(json.dumps({k: v for k, v in final.items() if k != "sample_details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
