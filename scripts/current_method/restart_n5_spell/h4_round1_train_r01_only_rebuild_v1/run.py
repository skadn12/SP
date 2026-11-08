"""Plan and execute a clean R01-only Memorax namespace rebuild for Train.

The source Train namespaces already contain an uncontrolled R02. This command
creates a new namespace per remaining root and writes exactly Parent(R00) and
R01. It never writes to the source namespace and never generates R02.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..h4_round1_feedback_r02_batch_v1.bootstrap import run as bootstrap_run
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (
    memorax_user_id,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SOURCE_MANIFEST = RUN_BASE / (
    "h4_round1_cumulative_continuation_fresh_mining_v1_live_47/"
    "round1_root_manifest.jsonl")
PILOT_ROOTS = RUN_BASE / (
    "h4_round1_feedback_compiler_v1_live_1/selected_10_roots.jsonl")
SOURCE_RUNS = RUN_BASE / "h4_unexpanded122_memory_event_k3_v1"
DEFAULT_RUN_ROOT = RUN_BASE / (
    "h4_round1_train_r01_only_rebuild_v1_live_1")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")

ALGORITHM_VERSION = "h4_round1_train_r01_only_rebuild_v1"
NAMESPACE_PREFIX = "gxy-selfplay:round1-paired-feedback:rebuild-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2,
                      sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    side_tmp.write_text(_sha(path) + "\n", encoding="utf-8")
    os.replace(side_tmp, sidecar)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows), encoding="utf-8")
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    side_tmp.write_text(_sha(path) + "\n", encoding="utf-8")
    os.replace(side_tmp, sidecar)


def _remap_intent(intent: Mapping[str, Any], *, root_id: str,
                  user_id: str, branch: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(intent))
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"intent_payload_missing:{root_id}")
    payload["user_id"] = user_id
    if str(payload.get("session_id") or "") != result.get("session_id"):
        raise RuntimeError(f"intent_payload_session_mismatch:{root_id}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"intent_metadata_missing:{root_id}")
    if str(metadata.get("root_id") or "") != root_id:
        raise RuntimeError(f"intent_metadata_root_mismatch:{root_id}")
    result["root_id"] = root_id
    result["user_id"] = user_id
    result["task_id"] = None
    result["request_id"] = (
        f"{ALGORITHM_VERSION}-{branch}-{root_id}-r"
        f"{int(result.get('round_index') or 0):02d}-"
        f"{_canonical_sha({'root_id': root_id, 'branch': branch, 'round_index': result.get('round_index'), 'source': result.get('source_dialogue_sha256')})[:12]}"
    )
    result["payload_sha256"] = _canonical_sha(payload)
    result["reconstruction_branch"] = branch
    result["live_submission"] = False
    result["api_key_in_artifact"] = False
    return result


def _source_paths(row: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    r01_path = Path(str(row.get("memory_intent_path") or ""))
    if not r01_path.is_file():
        raise RuntimeError(f"r01_intent_missing:{row.get('root_id')}")
    parent_path = r01_path.with_name("parent_add_intent.json")
    r01_completion = r01_path.with_name("child_add_completion.json")
    parent_completion = r01_path.with_name("parent_add_completion.json")
    for path in (parent_path, r01_completion, parent_completion):
        if not path.is_file():
            raise RuntimeError(f"memory_artifact_missing:{path}")
    return parent_path, r01_path, parent_completion, r01_completion


def _build_plan(run_root: Path) -> dict[str, Any]:
    manifest_rows = _jsonl(SOURCE_MANIFEST)
    pilot_rows = _jsonl(PILOT_ROOTS)
    pilot_ids = [str(row.get("root_id") or "") for row in pilot_rows]
    if len(manifest_rows) != 70 or len(set(row.get("root_id") for row in manifest_rows)) != 70:
        raise RuntimeError("train_manifest_must_have_70_unique_roots")
    if len(pilot_ids) != 10 or len(set(pilot_ids)) != 10:
        raise RuntimeError("pilot_manifest_must_have_10_unique_roots")
    manifest = {str(row["root_id"]): row for row in manifest_rows}
    if not set(pilot_ids) <= set(manifest):
        raise RuntimeError("pilot_root_not_in_train_manifest")
    pending_ids = [root_id for root_id in manifest if root_id not in set(pilot_ids)]
    if len(pending_ids) != 60:
        raise RuntimeError("pending_train_root_count_not_60")

    rows: list[dict[str, Any]] = []
    namespace_values: set[str] = set()
    source_users: set[str] = set()
    for index, root_id in enumerate(pending_ids):
        source = manifest[root_id]
        if str(source.get("split") or "") != "train":
            raise RuntimeError(f"non_train_root_in_train_manifest:{root_id}")
        dialogue_path = Path(str(source.get("canonical_dialogue_path") or ""))
        if not dialogue_path.is_file() or _sha(dialogue_path) != source.get(
                "canonical_dialogue_sha256"):
            raise RuntimeError(f"canonical_dialogue_sha_mismatch:{root_id}")
        dialogue = _load(dialogue_path)
        if str(dialogue.get("root_id") or "") != root_id:
            raise RuntimeError(f"dialogue_root_mismatch:{root_id}")
        refs = [str(turn.get("turn_ref") or "")
                for turn in dialogue.get("turns") or []]
        if not refs or len(refs) != len(set(refs)):
            raise RuntimeError(f"dialogue_refs_invalid:{root_id}")
        if any("R02" in ref.upper() for ref in refs):
            raise RuntimeError(f"r02_ref_in_r01_dialogue:{root_id}")
        parent_path, r01_path, parent_completion_path, r01_completion_path = _source_paths(source)
        parent = _load(parent_path)
        r01 = _load(r01_path)
        parent_completion = _load(parent_completion_path)
        r01_completion = _load(r01_completion_path)
        if parent.get("round_index") != 0 or r01.get("round_index") != 1:
            raise RuntimeError(f"round_index_invalid:{root_id}")
        if parent.get("root_id") != root_id or r01.get("root_id") != root_id:
            raise RuntimeError(f"intent_root_mismatch:{root_id}")
        if parent_completion.get("final_status") != "success" or not parent_completion.get("verified_complete"):
            raise RuntimeError(f"parent_completion_not_verified:{root_id}")
        if r01_completion.get("final_status") != "success" or not r01_completion.get("verified_complete"):
            raise RuntimeError(f"r01_completion_not_verified:{root_id}")
        if parent.get("source_dialogue_sha256") != dialogue.get("parent_dialogue_sha256"):
            raise RuntimeError(f"parent_source_sha_mismatch:{root_id}")
        if r01.get("source_dialogue_sha256") != dialogue.get("complete_dialogue_sha256"):
            raise RuntimeError(f"r01_source_sha_mismatch:{root_id}")
        source_user = str(r01.get("user_id") or "")
        if not source_user:
            raise RuntimeError(f"source_user_missing:{root_id}")
        source_users.add(source_user)
        digest = hashlib.sha256(
            f"{ALGORITHM_VERSION}|{root_id}|{source.get('complete_dialogue_sha256')}".encode()
        ).hexdigest()[:16]
        namespace = f"{NAMESPACE_PREFIX}:{digest}"
        if namespace in namespace_values:
            raise RuntimeError(f"namespace_collision:{root_id}")
        namespace_values.add(namespace)
        user_id = memorax_user_id(root_id, namespace=namespace)
        parent_intent = _remap_intent(parent, root_id=root_id,
                                       user_id=user_id, branch="train_rebuild")
        r01_intent = _remap_intent(r01, root_id=root_id,
                                   user_id=user_id, branch="train_rebuild")
        rows.append({
            "algorithm_version": ALGORITHM_VERSION,
            "status": "REQUIRED_BEFORE_R02",
            "selection_index": index,
            "root_id": root_id,
            "split": "train",
            "namespace": namespace,
            "user_id": user_id,
            "source_user_id": source_user,
            "source_dialogue_path": str(dialogue_path),
            "source_dialogue_sha256": _sha(dialogue_path),
            "source_complete_dialogue_sha256": dialogue.get("complete_dialogue_sha256"),
            "source_parent_dialogue_sha256": dialogue.get("parent_dialogue_sha256"),
            "source_parent_intent_path": str(parent_path),
            "source_r01_intent_path": str(r01_path),
            "source_parent_completion_path": str(parent_completion_path),
            "source_r01_completion_path": str(r01_completion_path),
            "parent_add_intent": parent_intent,
            "r01_add_intent": r01_intent,
            "memory_write_contract": {
                "parent_session": f"{root_id}:memorax:R00",
                "r01_session": f"{root_id}:memorax:R01",
                "messages_scope": "R00_then_R01_only",
                "window_context_scope": "source_intent_only",
                "source_namespace_reuse_forbidden": True,
                "r02_generation": 0,
            },
        })
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "status": "PASS",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": _sha(SOURCE_MANIFEST),
        "pilot_manifest": str(PILOT_ROOTS),
        "pilot_manifest_sha256": _sha(PILOT_ROOTS),
        "source_memory_run": str(SOURCE_RUNS),
        "source_train_root_count": 70,
        "pilot_root_count": 10,
        "pending_root_count": len(rows),
        "parent_add_count": len(rows),
        "r01_add_count": len(rows),
        "r02_add_count": 0,
        "source_namespace_reuse_forbidden": True,
        "unique_source_users": len(source_users),
        "unique_target_namespaces": len(namespace_values),
        "rows": rows,
    }


def plan_only(run_root: Path) -> dict[str, Any]:
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(run_root)
    rows = plan.pop("rows")
    _write(run_root / "preflight_report.json", {
        **{key: value for key, value in plan.items() if key != "rows"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_SERIAL_R01_REBUILD",
    })
    _write_jsonl(run_root / "r01_train_rebuild_plan.jsonl", rows)
    _write(run_root / "rebuild_plan_summary.json", {
        "algorithm_version": ALGORITHM_VERSION,
        "status": "PASS",
        "root_count": len(rows),
        "parent_add_count": len(rows),
        "r01_add_count": len(rows),
        "r02_add_count": 0,
        "pilot_excluded_root_count": 10,
        "pilot_excluded_root_ids": sorted(
            str(row["root_id"]) for row in _jsonl(PILOT_ROOTS)),
        "target_namespace_prefix": NAMESPACE_PREFIX,
        "source_namespace_reuse_forbidden": True,
    })
    return {"status": "OFFLINE_PLAN_ONLY", "run_root": str(run_root),
            "root_count": len(rows), "parent_add_count": len(rows),
            "r01_add_count": len(rows), "r02_add_count": 0}


def execute(run_root: Path, credentials: Path,
            allow_insecure_http: bool) -> dict[str, Any]:
    plan = _jsonl(run_root / "r01_train_rebuild_plan.jsonl")
    preflight = _load(run_root / "preflight_report.json")
    if preflight.get("status") != "PASS" or len(plan) != 60:
        raise RuntimeError("offline_plan_not_ready_or_count_invalid")
    # Adapt the immutable plan to the existing fail-closed serial bootstrap.
    bootstrap_plan = run_root / "bootstrap_plan"
    bootstrap_plan.mkdir()
    bootstrap_rows = []
    for row in plan:
        bootstrap_rows.append({
            "root_id": row["root_id"],
            "namespace": row["namespace"],
            "user_id": row["user_id"],
            "status": row["status"],
            "parent_add_intent": row["parent_add_intent"],
            "r01_add_intent": row["r01_add_intent"],
        })
    bootstrap_preflight = {
        "status": "PASS",
        "decision": "pass",
        "formal_root_count": len(bootstrap_rows),
        "algorithm_version": ALGORITHM_VERSION,
        "source_plan_sha256": _sha(run_root / "r01_train_rebuild_plan.jsonl"),
        "source_namespace_reuse_forbidden": True,
    }
    _write(bootstrap_plan / "preflight_report.json", bootstrap_preflight)
    _write_jsonl(bootstrap_plan / "r01_feedback_bootstrap_plan.jsonl",
                 bootstrap_rows)
    execution_root = run_root / "execution"
    result = bootstrap_run(
        plan_root=bootstrap_plan,
        output_root=execution_root,
        credentials_path=credentials,
        allow_insecure_http=allow_insecure_http,
        live=True,
    )
    _write(run_root / "final_rebuild_report.json", {
        "algorithm_version": ALGORITHM_VERSION,
        "status": result.get("status"),
        "requested_root_count": 60,
        "passed_root_count": len(result.get("passed_root_ids") or []),
        "quarantined_root_count": len(result.get("quarantined_root_ids") or []),
        "parent_add_calls": result.get("parent_add_calls"),
        "r01_child_add_calls": result.get("r01_child_add_calls"),
        "r02_generation": 0,
        "r02_add_calls": 0,
        "source_namespace_modified": False,
        "automatic_retry": False,
        "bootstrap_result": result,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("plan", "live"), default="plan")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--memorax-credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.phase == "plan":
        result = plan_only(run_root)
    else:
        result = execute(run_root, args.memorax_credentials.resolve(),
                         args.allow_insecure_memorax_http)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {
        "OFFLINE_PLAN_ONLY", "PASS", "PARTIAL_QUARANTINE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
