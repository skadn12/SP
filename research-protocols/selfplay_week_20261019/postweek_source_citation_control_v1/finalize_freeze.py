from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path, *, logical_path: str | None = None) -> dict:
    path = path.resolve()
    return {
        "path": logical_path or str(path),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    requests = read_json(HERE / "REQUESTS.json")
    if requests["status"] != "FROZEN_NOT_EXECUTED" or requests["request_count"] != 24:
        raise RuntimeError("REQUESTS_NOT_READY")
    if len(requests["requests"]) != 24:
        raise RuntimeError("REQUEST_COUNT_MISMATCH")
    cpu = read_json(HERE / "CPU_PREFLIGHT.json")
    if cpu["status"] != "PASS_NOT_EXECUTED" or not cpu["all_prompts_fit"]:
        raise RuntimeError("CPU_PREFLIGHT_FAILED")
    if cpu["model_loaded"] or cpu["inference_performed"] or cpu["gold_or_answer_key_read"]:
        raise RuntimeError("UNEXPECTED_EXECUTION_OR_GOLD_ACCESS")

    audit_path = HERE / "INDEPENDENT_PREFLIGHT.json"
    if not audit_path.is_file():
        raise RuntimeError("INDEPENDENT_PREFLIGHT_MISSING")
    audit = read_json(audit_path)
    if audit.get("status") != "PASS_READ_ONLY" or audit.get("model_inference_performed") is not False:
        raise RuntimeError("INDEPENDENT_PREFLIGHT_FAILED_OR_EXECUTED")

    protocol = read_json(HERE / "PROTOCOL.json")
    protocol["status"] = "FROZEN_NOT_EXECUTED"
    protocol["new_model_calls_authorized_by_freeze"] = False
    write_json(HERE / "PROTOCOL.json", protocol)

    # Bind all local public artifacts, private scoring materials, source files,
    # and external model/runtime dependencies. The manifest excludes itself and
    # FREEZE.json to avoid circular hashes; FREEZE.json binds the manifest.
    local_files = [
        HERE / "PROTOCOL_ZH.md",
        HERE / "PROTOCOL.json",
        HERE / "prepare_control.py",
        HERE / "finalize_freeze.py",
        HERE / "verify_freeze.py",
        HERE / "SOURCE_ADJUDICATION.json",
        HERE / "EXPOSURE_LEDGER.json",
        HERE / "REQUESTS.json",
        HERE / "RUN_ORDER.json",
        HERE / "CPU_PREFLIGHT.json",
        HERE / "RUNTIME_BINDING.json",
        audit_path,
        HERE / "private" / "PRIVATE_SCORE_KEY.json",
        *sorted((HERE / "source_review_packets").glob("*.json")),
        *sorted((HERE / "prompts").glob("*.txt")),
        *sorted((HERE / "schemas").glob("*.json")),
    ]
    if any(not p.is_file() for p in local_files):
        raise RuntimeError("MISSING_LOCAL_FREEZE_ARTIFACT")
    local_bindings = [binding(p, logical_path=str(p.relative_to(HERE))) for p in local_files]
    external_paths = [
        Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_production_v1/sources/cc-train-episode-136202.json"),
        Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_production_v1/sources/cc-train-episode-147997.json"),
        Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_production_v1/sources/cc-train-episode-156136.json"),
        WEEK / "day4_schedule_v1" / "TRAIN_POOL_B.jsonl",
        WEEK / "day5_base_full_train_eval_v1" / "public" / "DIRECT_REQUESTS.json",
        WEEK / "day5_training_v1" / "evaluation" / "models" / "B1" / "EXECUTION_MANIFEST.json",
        WEEK / "day5_training_v1" / "evaluation" / "endpoint_proofs" / "B1_EXECUTION.json",
        WEEK / "day6_diagnostic_reliability_v11" / "prepare.py",
        WEEK / "day6_diagnostic_reliability_v11" / "run.py",
        WEEK / "day6_diagnostic_reliability_v11" / "PROTOCOL.json",
        Path("/root/autodl-tmp/Qwen3-4B/tokenizer.json"),
        Path("/root/autodl-tmp/Qwen3-4B/tokenizer_config.json"),
    ]
    if any(not p.is_file() for p in external_paths):
        raise RuntimeError("MISSING_EXTERNAL_DEPENDENCY")
    external_bindings = [binding(p) for p in external_paths]
    manifest = {
        "schema_version": "source_citation_control_final_manifest_v1",
        "status": "FROZEN_NOT_EXECUTED",
        "local_artifacts": local_bindings,
        "external_dependencies": external_bindings,
        "request_count": 24,
        "matched_pair_count": 12,
        "model_inference_performed": False,
        "responses_present": False,
        "gold_access_during_preparation": False,
    }
    write_json(HERE / "FINAL_MANIFEST.json", manifest)
    freeze = {
        "schema_version": "source_citation_control_freeze_v1",
        "created_at_utc": "2026-10-22",
        "status": "FROZEN_NOT_EXECUTED",
        "execution_authorized": False,
        "new_model_calls_authorized_by_this_freeze": 0,
        "historical_probe_reopened": False,
        "protocol_binding": binding(HERE / "PROTOCOL.json", logical_path="PROTOCOL.json"),
        "manifest_binding": binding(HERE / "FINAL_MANIFEST.json", logical_path="FINAL_MANIFEST.json"),
        "preflight_binding": binding(HERE / "CPU_PREFLIGHT.json", logical_path="CPU_PREFLIGHT.json"),
        "source_adjudication_binding": binding(HERE / "SOURCE_ADJUDICATION.json", logical_path="SOURCE_ADJUDICATION.json"),
        "exposure_ledger_binding": binding(HERE / "EXPOSURE_LEDGER.json", logical_path="EXPOSURE_LEDGER.json"),
        "request_binding": binding(HERE / "REQUESTS.json", logical_path="REQUESTS.json"),
        "run_order_binding": binding(HERE / "RUN_ORDER.json", logical_path="RUN_ORDER.json"),
        "private_score_key_binding": binding(HERE / "private" / "PRIVATE_SCORE_KEY.json", logical_path="private/PRIVATE_SCORE_KEY.json"),
        "independent_preflight_binding": binding(audit_path, logical_path="INDEPENDENT_PREFLIGHT.json"),
        "request_count": 24,
        "matched_pairs": 12,
        "roots": ["cc-train-episode-136202", "cc-train-episode-147997", "cc-train-episode-156136"],
        "model_loaded": False,
        "inference_performed": False,
        "responses_present": False,
        "gold_or_answer_key_read_for_model_input": False,
        "independent_preflight_completed": True,
        "independent_preflight_status": audit["status"],
        "next_step": "Freeze is complete; any execution requires a separate explicit decision and is not authorized by this freeze.",
    }
    write_json(HERE / "FREEZE.json", freeze)
    print(json.dumps({
        "status": freeze["status"], "request_count": 24, "matched_pairs": 12,
        "local_artifacts": len(local_bindings), "external_dependencies": len(external_bindings),
        "execution_authorized": False,
    }))


if __name__ == "__main__":
    main()
