"""Read-only audit for the completed feedback R02 single-root smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import VERSION
from .preflight import RUN_BASE, _load, _sha
from ..live_backends import CHALLENGER_DEFAULT_TEMPERATURE


LIVE_ROOT = RUN_BASE / "h4_round1_feedback_r02_single_root_v1_live_6"
PREFLIGHT_ROOT = RUN_BASE / "h4_round1_feedback_r02_single_root_v1_live_7"
OLD_ROOT = RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_3"
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
ROOT_ID = "cc-train-episode-254852"
USER_ID = "gxy-selfplay:round1-paired-feedback:2338d0e1c349:cc-train-episode-254852"
SESSION_ID = ROOT_ID + ":memorax:R02-feedback"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sidecar_coverage(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and not path.name.endswith(".sha256"))
    present = [path for path in files
               if path.with_name(path.name + ".sha256").is_file()]
    missing = [str(path.relative_to(root)) for path in files
               if not path.with_name(path.name + ".sha256").is_file()]
    return {
        "file_count": len(files),
        "sidecar_count": len(present),
        "missing_sidecar_count": len(missing),
        "missing_sidecar_examples": missing[:20],
        "files": [
            {"path": str(path.relative_to(root)), "sha256": _sha(path)}
            for path in files
        ],
        "complete": not missing,
    }


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_root}")
    report = _load(LIVE_ROOT / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")
    freeze = _load(LIVE_ROOT / "run_freeze_manifest.json")
    config = _load(CONFIG_PATH)
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    generation = config.get("generation") if isinstance(config.get("generation"), dict) else {}
    effective_temperature = float(
        api.get("temperature") if api.get("temperature") is not None
        else CHALLENGER_DEFAULT_TEMPERATURE
    )
    preflight_contract = _load(PREFLIGHT_ROOT / "r02_feedback_contract.json")
    child_intent = _load(LIVE_ROOT / "memory/child_add_intent.json")
    child_completion = _load(LIVE_ROOT / "memory/child_add_completion.json")
    visibility = _load(LIVE_ROOT / "memory/child_visibility_audit.json")
    skipped = _load(LIVE_ROOT / "memory/parent_add_skipped.json")
    events = _load_jsonl(LIVE_ROOT / "ledger/request_events.jsonl")
    completed = [row for row in events if row.get("status") == "completed"]
    identities = sorted({json.dumps(row.get("backend_identity") or {},
                                      sort_keys=True) for row in completed})
    observed = [row.get("backend_identity", {}).get("temperature")
                for row in completed]
    planner_prompt = (LIVE_ROOT / "raw/prompts/planner_attempt_01.txt").read_text(
        encoding="utf-8")
    child_refs = [str(ref) for ref in child_intent.get("turn_refs") or []]
    r02_only = bool(child_refs) and all(
        re.fullmatch(r"R02:S\d+:T\d+", ref) for ref in child_refs)
    payload = child_intent.get("payload") or {}
    window = payload.get("window_context") or []
    window_roles_valid = all(row.get("role") == "user" for row in window
                             if isinstance(row, dict))
    feedback_payload_keys = {"feedback_card_exposed", "feedback_card_sha256",
                             "feedback_version", "feedback_card"}
    payload_metadata = payload.get("metadata") or {}
    feedback_payload_keys_present = sorted(
        feedback_payload_keys.intersection(payload)
        | (feedback_payload_keys.intersection(payload_metadata)
           if isinstance(payload_metadata, dict) else set()))
    old_freeze = OLD_ROOT / "shards/shard-00/roots/cc-train-episode-254852/attempt-01/run_freeze_manifest.json"
    old_freeze_value = _load(old_freeze)
    old_api_temperature = (old_freeze_value.get("api_configuration") or {}).get("temperature")
    checks = {
        "run_completed": report.get("status") == "completed" and not report.get("failure_stage") and not report.get("runtime_error"),
        "root_identity": report.get("root_id") == ROOT_ID,
        "model_identity_present": len(identities) == 1 and json.loads(identities[0]).get("model") == "Qwen/Qwen3-14B",
        "observed_api_temperature_matches_effective": bool(observed) and all(value == effective_temperature for value in observed),
        "preflight_api_temperature_matches_effective": (preflight_contract.get("model") or {}).get("api_temperature") == effective_temperature,
        "generation_temperature_recorded_separately": (preflight_contract.get("model") or {}).get("generation_temperature") == generation.get("temperature"),
        "historical_recipe_temperature_matches": old_api_temperature == effective_temperature,
        "config_sha_matches_run_freeze": freeze.get("config_sha256") == _sha(CONFIG_PATH),
        "single_child_add": report.get("child_add_calls") == 1,
        "parent_add_skipped": report.get("parent_add_calls") == 0 and skipped.get("parent_add_calls") == 0 and report.get("parent_add_skipped") is True,
        "child_add_success": child_completion.get("final_status") == "success" and child_completion.get("verified_complete") is True,
        "child_visibility_bucket_valid": visibility.get("valid") is True and visibility.get("bucket_visibility_verified") is True,
        "feedback_session_frozen": child_intent.get("session_id") == SESSION_ID and child_intent.get("user_id") == USER_ID,
        "r02_only_child_refs": r02_only,
        "r02_window_roles_valid": window_roles_valid,
        "memorax_payload_excludes_feedback_metadata": not feedback_payload_keys_present,
        "feedback_card_in_planner_prompt": "FROZEN ROUND-1 FEEDBACK CARD" in planner_prompt,
        "qa_solver_zero": report.get("qa_calls") == 0 and report.get("solver_calls") == 0,
        "no_request_failures": len(completed) == 5 and all(row.get("paid_generation_count") == 1 for row in completed),
    }
    artifact_coverage = _sidecar_coverage(LIVE_ROOT)
    output_root.mkdir(parents=True, exist_ok=False)
    external_inventory = {
        "manifest_version": VERSION + "_historical_artifact_inventory_v1",
        "source_root": str(LIVE_ROOT.resolve()),
        "source_run_is_immutable": True,
        "file_count": artifact_coverage["file_count"],
        "files": artifact_coverage["files"],
    }
    _write(output_root / "historical_artifact_sha256_manifest.json",
           external_inventory)
    artifact_coverage["external_inventory_path"] = str(
        (output_root / "historical_artifact_sha256_manifest.json").resolve())
    artifact_coverage["external_inventory_sha256"] = _sha(
        output_root / "historical_artifact_sha256_manifest.json")
    result = {
        "audit_version": VERSION + "_completed_run_audit_v1",
        "status": "PASS_HISTORICAL_SIDECAR_EXTERNALIZED" if all(checks.values()) else "BLOCKED",
        "ready_for_remaining_roots": all(checks.values()),
        "root_id": ROOT_ID,
        "source_run": {"path": str(LIVE_ROOT.resolve()), "report_sha256": _sha(LIVE_ROOT / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")},
        "preflight_contract": {"path": str(PREFLIGHT_ROOT / "r02_feedback_contract.json"), "sha256": _sha(PREFLIGHT_ROOT / "r02_feedback_contract.json")},
        "recipe": {
            "model": "Qwen/Qwen3-14B",
            "base_url": api.get("base_url"),
            "config_sha256": _sha(CONFIG_PATH),
            "api_temperature_declared": api.get("temperature"),
            "api_temperature_effective": effective_temperature,
            "generation_temperature": generation.get("temperature"),
            "historical_runner_api_temperature": old_api_temperature,
            "observed_completed_request_temperatures": sorted(set(observed)),
            "completed_request_count": len(completed),
            "backend_identities": [json.loads(identity) for identity in identities],
        },
        "memory": {
            "child_add_calls": report.get("child_add_calls"),
            "parent_add_calls": report.get("parent_add_calls"),
            "parent_add_skipped": report.get("parent_add_skipped"),
            "session_id": child_intent.get("session_id"),
            "user_id": child_intent.get("user_id"),
            "turn_count": len(child_refs),
            "r02_only_refs": r02_only,
            "completion_status": child_completion.get("final_status"),
            "verified_complete": child_completion.get("verified_complete"),
            "bucket_visibility_verified": visibility.get("bucket_visibility_verified"),
            "semantic_visibility_verified": visibility.get("semantic_visibility_verified"),
        },
        "feedback": {
            "planner_marker_present": "FROZEN ROUND-1 FEEDBACK CARD" in planner_prompt,
            "feedback_card_exposed": child_intent.get("feedback_card_exposed") is True,
            "payload_forbidden_keys_present": feedback_payload_keys_present,
            "provenance_storage": "local audit only",
        },
        "prohibited_calls": {"qa_calls": report.get("qa_calls"), "solver_calls": report.get("solver_calls")},
        "checks": checks,
        "artifact_sidecar_coverage": artifact_coverage,
        "caveat": "The completed live_6 runner predated adapter sidecar finalization; its source files are hashed in this audit but are not modified.",
    }
    _write(output_root / "r02_feedback_smoke_audit.json", result)
    _write(output_root / "source_file_hashes.json", {
        "config": {"path": str(CONFIG_PATH.resolve()), "sha256": _sha(CONFIG_PATH)},
        "live_report": {"path": str((LIVE_ROOT / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json").resolve()), "sha256": _sha(LIVE_ROOT / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")},
        "child_add_intent": {"path": str((LIVE_ROOT / "memory/child_add_intent.json").resolve()), "sha256": _sha(LIVE_ROOT / "memory/child_add_intent.json")},
        "child_add_completion": {"path": str((LIVE_ROOT / "memory/child_add_completion.json").resolve()), "sha256": _sha(LIVE_ROOT / "memory/child_add_completion.json")},
        "child_visibility_audit": {"path": str((LIVE_ROOT / "memory/child_visibility_audit.json").resolve()), "sha256": _sha(LIVE_ROOT / "memory/child_visibility_audit.json")},
    })
    md = "\n".join([
        "# R02 Feedback Single-Root Smoke Audit",
        "",
        f"- Runtime contract: `{result['status']}`",
        f"- Ready for remaining roots: `{result['ready_for_remaining_roots']}`",
        f"- API temperature: `{effective_temperature}`; generation temperature: `{generation.get('temperature')}`",
        f"- Requests: `{len(completed)}` completed, all model/backend identities equal",
        "- Memory: one R02-feedback child Add succeeded; Parent Add skipped",
        f"- QA/Solver calls: `{report.get('qa_calls')}/{report.get('solver_calls')}`",
        f"- Historical live_6 sidecar coverage: `{artifact_coverage['sidecar_count']}/{artifact_coverage['file_count']}`",
        "- The live_6 source directory was not modified.",
        "",
    ])
    (output_root / "final_r02_feedback_smoke_audit.md").write_text(md, encoding="utf-8")
    (output_root / "final_r02_feedback_smoke_audit.md.sha256").write_text(
        _sha(output_root / "final_r02_feedback_smoke_audit.md") + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=RUN_BASE / "h4_round1_feedback_r02_single_root_v1_audit_live_1")
    args = parser.parse_args(argv)
    result = run(args.output_root.resolve())
    print(json.dumps({"status": result["status"], "ready_for_remaining_roots": result["ready_for_remaining_roots"], "failed_checks": [key for key, value in result["checks"].items() if not value]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
