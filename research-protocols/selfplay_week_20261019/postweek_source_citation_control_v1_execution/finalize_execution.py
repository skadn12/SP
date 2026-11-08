from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "postweek_source_citation_control_v1"
MODEL_MANIFEST = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> None:
    if (HERE / "EXECUTION_MANIFEST.json").exists() or (HERE / "outputs").exists():
        raise RuntimeError("EXECUTION_ALREADY_FROZEN_OR_STARTED_NO_OVERWRITE")
    auth = json.loads((HERE / "USER_EXECUTION_DECISION.json").read_text())
    if auth.get("authorization_is_separate_from_parent_protocol_freeze") is not True:
        raise RuntimeError("SEPARATE_EXECUTION_DECISION_REQUIRED")
    if auth["authorized_scope"]["maximum_model_requests"] != 24 or auth["authorized_scope"]["automatic_retries"] != 0:
        raise RuntimeError("EXECUTION_SCOPE_INVALID")
    audit_path = HERE / "RUNNER_PREFLIGHT.json"
    if not audit_path.is_file():
        raise RuntimeError("INDEPENDENT_RUNNER_PREFLIGHT_MISSING")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS_READ_ONLY" or audit.get("model_inference_performed") is not False:
        raise RuntimeError("INDEPENDENT_RUNNER_PREFLIGHT_FAILED")
    expected_audit_paths = {str((HERE / name).resolve()) for name in (
        "run_control.py", "score_control.py", "unblind_scores.py",
        "prepare_blind_review.py", "finalize_execution.py", "verify_execution.py", "USER_EXECUTION_DECISION.json")}
    audited_files = audit.get("audited_files", [])
    if not expected_audit_paths.issubset({row["path"] for row in audited_files}):
        raise RuntimeError("INDEPENDENT_AUDIT_CORE_FILES_MISSING")
    for row in audited_files:
        p = Path(row["path"])
        if not p.is_file() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise RuntimeError("AUDITED_FILE_CHANGED:" + str(p))
    parent_freeze = json.loads((PARENT / "FREEZE.json").read_text())
    if parent_freeze.get("status") != "FROZEN_NOT_EXECUTED" or parent_freeze.get("execution_authorized") is not False:
        raise RuntimeError("PARENT_FREEZE_CHANGED")
    parent_manifest = json.loads((PARENT / "FINAL_MANIFEST.json").read_text())
    requests = json.loads((PARENT / "REQUESTS.json").read_text())["requests"]
    run_order = json.loads((PARENT / "RUN_ORDER.json").read_text())["run_order_request_ids"]
    if len(requests) != 24 or len(run_order) != 24:
        raise RuntimeError("REQUEST_DENOMINATOR_CHANGED")
    by_id = {r["request_id"]: r for r in requests}
    if set(by_id) != set(run_order):
        raise RuntimeError("REQUEST_ORDER_SET_CHANGED")

    local_names = [
        "USER_EXECUTION_DECISION.json", "RUNNER_PREFLIGHT.json", "run_control.py",
        "score_control.py", "prepare_blind_review.py", "unblind_scores.py", "finalize_execution.py", "verify_execution.py",
    ]
    local_files = [HERE / n for n in local_names]
    public_parent_names = [
        "PROTOCOL.json", "REQUESTS.json", "RUN_ORDER.json", "CPU_PREFLIGHT.json",
        "INDEPENDENT_PREFLIGHT.json", "RUNTIME_BINDING.json", "FINAL_MANIFEST.json", "FREEZE.json",
    ]
    public_parent_files = [PARENT / n for n in public_parent_names]
    prompt_schema_files = []
    for row in requests:
        prompt_schema_files.extend([Path(row["prompt_path"]), Path(row["schema_path"])])
    endpoint_manifest = json.loads(MODEL_MANIFEST.read_text())
    endpoint_paths = [MODEL_MANIFEST, Path(endpoint_manifest["endpoint_proof"]["path"])]
    endpoint_paths += [Path(row["path"]) for row in endpoint_manifest["model_bindings"]]
    endpoint_paths += [Path(row["path"]) for row in endpoint_manifest["tokenizer_bindings"]]
    endpoint_paths += [Path(row["path"]) for row in endpoint_manifest["code_bindings"]]
    if any(not p.is_file() for p in local_files + public_parent_files + prompt_schema_files + endpoint_paths):
        raise RuntimeError("MISSING_EXECUTION_FREEZE_INPUT")
    bindings = [bind(p) for p in local_files + public_parent_files + prompt_schema_files + endpoint_paths]
    if len({row["path"] for row in bindings}) != len(bindings):
        dedup = {row["path"]: row for row in bindings}
        bindings = list(dedup.values())
    manifest = {
        "schema_version": "source_citation_control_execution_manifest_v1",
        "status": "EXECUTION_AUTHORIZED_NOT_STARTED",
        "authorized_by": "current user instruction, interpreted in immediate conversational context",
        "request_count_maximum": 24,
        "ordered_request_ids": run_order,
        "model_role": "B1",
        "separate_from_unchanged_parent_freeze": True,
        "automatic_retries": 0,
        "request_execution": "One request per generate call in frozen randomized order; one model load.",
        "sampling": {"temperature": 0.8, "top_p": 0.9, "max_tokens": 1024,
            "structured_json": True, "enable_thinking": False,
            "prefix_caching": False, "automatic_retry": False},
        "runtime_versions": {"torch": "2.11.0", "transformers": "5.9.0",
            "tokenizers": "0.22.2", "vllm": "0.21.0", "jsonschema": "4.26.0"},
        "parent_freeze_sha256": sha(PARENT / "FREEZE.json"),
        "parent_final_manifest_sha256": sha(PARENT / "FINAL_MANIFEST.json"),
        "answer_key_opened_by_runner": False,
        "bindings": bindings,
    }
    with (HERE / "EXECUTION_MANIFEST.json").open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": manifest["status"], "bound_files": len(bindings),
        "request_count_maximum": 24, "automatic_retries": 0}))


if __name__ == "__main__":
    main()
