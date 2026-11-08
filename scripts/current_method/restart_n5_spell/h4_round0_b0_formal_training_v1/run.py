"""Run serial full Round-0 Clean and Inclusive LoRA training without evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core import atomic_text, canonical_json, sha256_file
from ..h4_round0_b0_training_smoke_v1.builder import load_json, load_jsonl
from . import (
    BASE_MODEL,
    DATA_RUN,
    DEFAULT_RUN_ROOT,
    EPOCHS,
    EXPECTED_BACKWARD_CALLS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_TRAIN_RECORDS,
    FORMAL_VERSION,
    REPO_ROOT,
    REQUIRED_MAX_LENGTH,
    SMOKE_RUN,
)


POLICIES = (
    ("M_clean_policy", "m_clean_policy_train.jsonl", "clean_adapter", "clean_train.log", "clean_gradient_audit.json", "clean_training_report.json"),
    ("M_inclusive_policy", "m_inclusive_policy_train.jsonl", "inclusive_adapter", "inclusive_train.log", "inclusive_gradient_audit.json", "inclusive_training_report.json"),
)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def snapshot_files(root: Path) -> dict[str, Any]:
    files = {
        path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.iterdir()) if path.is_file()
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"base_model_snapshot_missing_safetensors:{root}")
    return {"root": str(root), "files": files}


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise RuntimeError(f"input_sha_mismatch:{path}")
    return sha256_file(path)


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
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


def train_one(*, policy: str, train_file: Path, output_dir: Path, log_path: Path, gradient_path: Path, report_path: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.train_smoke_runner",
        "--model", str(BASE_MODEL), "--tokenizer", str(BASE_MODEL), "--train-file", str(train_file),
        "--output-dir", str(output_dir), "--max-length", str(REQUIRED_MAX_LENGTH),
        "--epochs", str(EPOCHS), "--batch-size", str(recipe["batch_size"]),
        "--grad-accum", str(recipe["grad_accum"]), "--lr", str(recipe["lr"]),
        "--min-lr-ratio", str(recipe["min_lr_ratio"]), "--warmup-ratio", str(recipe["warmup_ratio"]),
        "--weight-decay", str(recipe["weight_decay"]), "--max-grad-norm", str(recipe["max_grad_norm"]),
        "--lora-rank", str(recipe["lora_rank"]), "--lora-alpha", str(recipe["lora_alpha"]),
        "--lora-dropout", str(recipe["lora_dropout"]), "--target-modules", ",".join(recipe["target_modules"]),
        "--seed", str(recipe["seed"]),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "SMOKE_GRADIENT_AUDIT_PATH": str(gradient_path)}
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    log_text = log_path.read_text(encoding="utf-8")
    parsed = parse_json_stream(log_text)
    initial = next((row for row in parsed if "total_update_steps" in row), {})
    finished = next((row for row in reversed(parsed) if row.get("finished") is True), {})
    losses = [row for row in parsed if "train_loss_recent" in row]
    gradient = load_json(gradient_path) if gradient_path.is_file() else {}
    adapter_config = output_dir / "adapter_config.json"
    adapter_state = output_dir / "adapter_model.pt"
    report = {
        "formal_training_report_version": f"{FORMAL_VERSION}_policy_report_v1",
        "policy": policy, "status": "PASS" if (
            process.returncode == 0 and finished.get("finished") is True
            and initial.get("train_records") == EXPECTED_TRAIN_RECORDS
            and finished.get("global_step") == EXPECTED_OPTIMIZER_STEPS
            and gradient.get("backward_calls") == EXPECTED_BACKWARD_CALLS
            and gradient.get("optimizer_steps") == EXPECTED_OPTIMIZER_STEPS
            and gradient.get("all_losses_finite") is True
            and gradient.get("all_gradients_finite_nonzero") is True
            and adapter_config.is_file() and adapter_state.is_file()
        ) else "FAIL",
        "return_code": process.returncode, "command": command,
        "log_path": str(log_path), "log_sha256": sha256_file(log_path),
        "gradient_audit_path": str(gradient_path),
        "gradient_audit_sha256": sha256_file(gradient_path) if gradient_path.is_file() else None,
        "initial_summary": initial, "finish_summary": finished, "loss_logs": losses,
        "gradient_summary": {
            "backward_calls": gradient.get("backward_calls"),
            "optimizer_steps": gradient.get("optimizer_steps"),
            "all_losses_finite": gradient.get("all_losses_finite"),
            "all_gradients_finite_nonzero": gradient.get("all_gradients_finite_nonzero"),
        },
        "adapter_saved": adapter_config.is_file() and adapter_state.is_file(),
        "adapter_config_sha256": sha256_file(adapter_config) if adapter_config.is_file() else None,
        "adapter_state_sha256": sha256_file(adapter_state) if adapter_state.is_file() else None,
        "adapter_config": load_json(adapter_config) if adapter_config.is_file() else None,
        "max_length": REQUIRED_MAX_LENGTH, "formal_epochs": EPOCHS,
        "dev_test_evaluation_started": False,
    }
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-run", type=Path, default=DATA_RUN)
    parser.add_argument("--smoke-run", type=Path, default=SMOKE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    data_run, smoke_run, run_root = args.data_run.resolve(), args.smoke_run.resolve(), args.run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"formal_output_run_already_exists:{run_root}")
    smoke_final = load_json(smoke_run / "final_smoke_report.json")
    if smoke_final.get("status") != "READY_FOR_FULL_TRAINING":
        raise RuntimeError(f"smoke_gate_not_ready:{smoke_final.get('status')}")
    smoke_recipe = load_json(smoke_run / "training_recipe.json")
    if int(smoke_recipe["max_length"]) != REQUIRED_MAX_LENGTH:
        raise RuntimeError("smoke_recipe_max_length_not_8192")
    source_hashes: dict[str, str] = {}
    for _, train_name, *_ in POLICIES:
        source_hashes[train_name] = verify_sidecar(data_run / train_name)
    for name in ("m_clean_policy_training_manifest.jsonl", "m_inclusive_policy_training_manifest.jsonl", "dev_full_eval_manifest.jsonl", "test_full_eval_manifest.jsonl", "training_interface_contract.json"):
        source_hashes[name] = verify_sidecar(data_run / name)
    train_maps: dict[str, dict[str, dict[str, Any]]] = {}
    train_rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for policy, train_name, *_ in POLICIES:
        rows = load_jsonl(data_run / train_name)
        if len(rows) != EXPECTED_TRAIN_RECORDS:
            raise RuntimeError(f"formal_train_count_mismatch:{policy}:{len(rows)}")
        train_rows_by_policy[policy] = rows
        train_maps[policy] = {str(row["metadata"]["candidate_id"]): row for row in rows}
        if len(train_maps[policy]) != EXPECTED_TRAIN_RECORDS:
            raise RuntimeError(f"formal_train_candidate_duplicate:{policy}")
        if any(int(row["metadata"].get("input_token_count", 0)) > REQUIRED_MAX_LENGTH for row in rows):
            raise RuntimeError(f"formal_train_length_over_8192:{policy}")
    dev_test = load_jsonl(data_run / "dev_full_eval_manifest.jsonl") + load_jsonl(data_run / "test_full_eval_manifest.jsonl")
    eval_ids = {str(row["candidate_id"]) for row in dev_test}
    eval_roots = {str(row["root_id"]) for row in dev_test}
    train_ids = set().union(*(set(rows) for rows in train_maps.values()))
    train_roots = {str(row["metadata"]["root_id"]) for rows in train_rows_by_policy.values() for row in rows}
    if train_ids & eval_ids or train_roots & eval_roots:
        raise RuntimeError("formal_train_eval_leakage")
    model_before = snapshot_files(BASE_MODEL)
    source_recipe = load_json(Path(smoke_recipe["source_recipe_report"]))
    hp = source_recipe["hyperparameters"]
    recipe = {
        "formal_recipe_version": f"{FORMAL_VERSION}_recipe_v1",
        "source_verified_recipe_report": smoke_recipe["source_recipe_report"],
        "source_verified_recipe_report_sha256": sha256_file(Path(smoke_recipe["source_recipe_report"])),
        "trainer_path": smoke_recipe["trainer_path"], "trainer_sha256": smoke_recipe["trainer_sha256"],
        "model": str(BASE_MODEL), "tokenizer": str(BASE_MODEL), "max_length": REQUIRED_MAX_LENGTH,
        "epochs": EPOCHS, "batch_size": int(hp["batch_size"]), "grad_accum": int(hp["grad_accum"]),
        "lr": float(hp["lr"]), "min_lr_ratio": float(hp["min_lr_ratio"]), "warmup_ratio": float(hp["warmup_ratio"]),
        "weight_decay": float(hp["weight_decay"]), "max_grad_norm": float(hp["max_grad_norm"]),
        "lora_rank": int(hp["lora_rank"]), "lora_alpha": int(hp["lora_alpha"]), "lora_dropout": float(hp["lora_dropout"]),
        "target_modules": list(hp["target_modules"]), "seed": int(hp["seed"]),
        "optimizer": "torch.optim.AdamW", "scheduler": "trainer.lr_at_step cosine with warmup",
        "optimizer_defaults": {"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False},
        "bf16": True, "gradient_checkpointing": True,
        "loss_mask": "existing train_qwen3_lora_sft.py tokenize_record assistant-only prefix mask",
        "optimizer_steps_expected": EXPECTED_OPTIMIZER_STEPS,
        "backward_calls_expected_per_policy": EXPECTED_BACKWARD_CALLS,
        "train_records_per_policy": EXPECTED_TRAIN_RECORDS,
        "formal_training_authorized": True, "dev_test_evaluation_started": False,
    }
    recipe_fp = hashlib.sha256(canonical_json(recipe).encode()).hexdigest()
    recipe["recipe_fingerprint"] = recipe_fp
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "training_recipe.json", recipe)
    write_json(run_root / "training_preflight.json", {
        "formal_training_preflight_version": f"{FORMAL_VERSION}_preflight_v1", "status": "PASS",
        "smoke_gate_status": smoke_final["status"], "smoke_final_sha256": sha256_file(smoke_run / "final_smoke_report.json"),
        "data_run": str(data_run), "data_source_sha256": source_hashes,
        "train_records_per_policy": EXPECTED_TRAIN_RECORDS, "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "expected_backward_calls_per_policy": EXPECTED_BACKWARD_CALLS, "max_length": REQUIRED_MAX_LENGTH,
        "train_dev_test_candidate_overlap": 0, "train_dev_test_root_overlap": 0,
        "base_model": str(BASE_MODEL), "base_model_sha_snapshot_before": model_before,
        "formal_training_started": False,
    })
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    for policy, train_name, adapter_name, log_name, gradient_name, report_name in POLICIES:
        reports[policy] = train_one(
            policy=policy, train_file=data_run / train_name, output_dir=run_root / adapter_name,
            log_path=logs / log_name, gradient_path=logs / gradient_name,
            report_path=run_root / report_name, recipe=recipe,
        )
        if reports[policy]["status"] != "PASS":
            raise RuntimeError(f"formal_training_failed:{policy}")
        atomic_text(logs / (log_name + ".sha256"), reports[policy]["log_sha256"] + "\n")
        atomic_text(logs / (gradient_name + ".sha256"), reports[policy]["gradient_audit_sha256"] + "\n")
    model_after = snapshot_files(BASE_MODEL)
    if model_after != model_before:
        raise RuntimeError("base_model_sha_drift_after_formal_training")
    final = {
        "formal_training_final_report_version": f"{FORMAL_VERSION}_final_v1", "status": "FORMAL_TRAINING_COMPLETE",
        "training_recipe_fingerprint": recipe_fp, "recipe_same_for_both_policies": True,
        "sample_count_per_policy": EXPECTED_TRAIN_RECORDS, "epochs": EPOCHS,
        "optimizer_steps_per_policy": {policy: reports[policy]["finish_summary"].get("global_step") for policy in reports},
        "backward_calls_per_policy": {policy: reports[policy]["gradient_summary"]["backward_calls"] for policy in reports},
        "max_length": REQUIRED_MAX_LENGTH, "formal_training_started": True,
        "dev_test_evaluation_started": False, "solver_calls": 0, "external_api_calls": 0, "semantic_auditor_calls": 0,
        "loss_finite": all(reports[policy]["gradient_summary"]["all_losses_finite"] for policy in reports),
        "gradient_finite_nonzero": all(reports[policy]["gradient_summary"]["all_gradients_finite_nonzero"] for policy in reports),
        "adapters_saved": all(reports[policy]["adapter_saved"] for policy in reports),
        "base_model_sha_drift": False, "source_jsonl_manifest_sha_drift": False,
        "policy_reports": {policy: str(run_root / next(item[-1] for item in POLICIES if item[0] == policy)) for policy in reports},
    }
    write_json(run_root / "final_formal_training_report.json", final)
    md = "\n".join([
        "# Round-0 B0 Formal Training", "", f"Status: **{final['status']}**", "",
        f"Both policies trained on `{EXPECTED_TRAIN_RECORDS}` records for `{EPOCHS}` epochs with `{EXPECTED_OPTIMIZER_STEPS}` optimizer steps and `--max-length {REQUIRED_MAX_LENGTH}`.",
        "Training was serial and used the same verified recipe fingerprint. No Dev/Test evaluation, Solver, API, or semantic auditor was run.", "",
        f"- Loss finite: `{final['loss_finite']}`", f"- Nonzero finite gradients: `{final['gradient_finite_nonzero']}`",
        f"- Adapters saved: `{final['adapters_saved']}`", f"- Base model SHA drift: `{final['base_model_sha_drift']}`", "",
        "See the policy reports and `logs/` for per-step evidence.",
    ]) + "\n"
    atomic_text(run_root / "final_formal_training_report.md", md)
    atomic_text(run_root / "final_formal_training_report.md.sha256", sha256_file(run_root / "final_formal_training_report.md") + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
