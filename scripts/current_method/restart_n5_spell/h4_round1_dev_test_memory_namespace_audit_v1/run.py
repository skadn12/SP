"""Audit the current Memorax namespace for every frozen Dev/Test root.

This command only uses Memorax bucket search. It never calls the Add endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.client import (
    MemoraxClient,
    load_memorax_config,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.memory_context import (
    normalize_bucket_snapshot,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (
    build_bucket_search_payload,
)


DEFAULT_FREEZE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_b0_matched_training_v1_live_2")
DEFAULT_SPLIT_MANIFEST = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl")
DEFAULT_R01_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_unexpanded122_memory_event_k3_v1")
DEFAULT_R02_MANIFEST = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_cumulative_continuation_fresh_mining_v1_live_47/"
    "round1_root_manifest.jsonl")
DEFAULT_RUNS_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
DEFAULT_RUN_ROOT = DEFAULT_RUNS_ROOT / (
    "h4_round1_dev_test_memory_namespace_audit_v1_live_1")

AUDIT_VERSION = "h4_round1_dev_test_memory_namespace_audit_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True) + "\n")
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def _namespace(user_id: str, root_id: str) -> str:
    suffix = f":{root_id}"
    if not user_id.endswith(suffix):
        raise ValueError(f"user_id_root_suffix_mismatch:{root_id}")
    return user_id[:-len(suffix)]


def _intent_inventory(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs_root.glob("**/*add_intent*.json")):
        try:
            value = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not value.get("user_id"):
            continue
        inventory[str(value["user_id"])].append({
            "path": str(path),
            "file_sha256": _sha(path),
            "root_id": str(value.get("root_id") or ""),
            "round_index": value.get("round_index"),
            "session_id": str(value.get("session_id") or ""),
            "payload_sha256": str(value.get("payload_sha256") or ""),
        })
    return inventory


def _search(client: MemoraxClient, *, root_id: str, user_id: str,
            label: str) -> dict[str, Any]:
    namespace = _namespace(user_id, root_id)
    query = build_bucket_search_payload(
        root_id=root_id, limit=100, max_buckets_per_type=500,
        user_namespace=namespace)
    if query["payload"]["user_id"] != user_id:
        raise ValueError(f"compiled_user_id_mismatch:{root_id}")
    response = client.search_buckets(
        query["payload"], request_id=f"{AUDIT_VERSION}-{label}-{root_id}")
    return normalize_bucket_snapshot(
        response, root_id=root_id, user_id=user_id)


def _select_authority(root_id: str, source_path: Path,
                      r01_run: Path) -> tuple[Path, list[dict[str, Any]]]:
    physical_root = source_path.parent.parent
    reports: list[dict[str, Any]] = []
    for path in sorted(r01_run.glob(
            f"**/roots/{root_id}*/reports/"
            "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json")):
        value = _load(path)
        reports.append({
            "path": str(path),
            "status": str(value.get("status") or ""),
            "user_id": str(value.get("user_id") or ""),
            "physical_root": str(path.parent.parent),
        })
    matching = [row for row in reports
                if Path(row["physical_root"]) == physical_root]
    if len(matching) != 1:
        raise ValueError(f"authoritative_report_not_unique:{root_id}")
    return physical_root, reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    freeze_paths = {
        "dev": args.freeze_root / "dev_full_eval_manifest.jsonl",
        "test": args.freeze_root / "test_full_eval_manifest.jsonl",
    }
    freeze_rows = {split: _load_jsonl(path)
                   for split, path in freeze_paths.items()}
    expected_candidates = {"dev": 485, "test": 507}
    expected_roots = {"dev": 15, "test": 16}
    errors: list[str] = []
    roots: dict[str, dict[str, Any]] = {}
    for split in ("dev", "test"):
        rows = freeze_rows[split]
        if len(rows) != expected_candidates[split]:
            errors.append(f"{split}_candidate_count_mismatch")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["root_id"])].append(row)
        if len(grouped) != expected_roots[split]:
            errors.append(f"{split}_root_count_mismatch")
        for root_id, group in grouped.items():
            if root_id in roots:
                errors.append(f"split_root_overlap:{root_id}")
                continue
            dialogue_paths = {str(row["dialogue_path"]) for row in group}
            dialogue_shas = {str(row["dialogue_sha256"]) for row in group}
            if len(dialogue_paths) != 1 or len(dialogue_shas) != 1:
                errors.append(f"freeze_dialogue_not_unique:{root_id}")
                continue
            roots[root_id] = {
                "split": split,
                "candidate_count": len(group),
                "eval_dialogue_path": next(iter(dialogue_paths)),
                "eval_dialogue_sha256": next(iter(dialogue_shas)),
            }

    split_rows = _load_jsonl(args.split_manifest)
    split_index = {str(row["root_id"]): row for row in split_rows}
    for root_id, root in roots.items():
        split_row = split_index.get(root_id)
        if not split_row or split_row.get("split") != root["split"]:
            errors.append(f"root_split_manifest_mismatch:{root_id}")
        elif split_row.get("candidate_count") != root["candidate_count"]:
            errors.append(f"root_candidate_count_mismatch:{root_id}")

    r02_rows = _load_jsonl(args.r02_manifest)
    r02_roots = {str(row["root_id"]) for row in r02_rows}
    r02_overlap = sorted(set(roots) & r02_roots)
    if r02_overlap:
        errors.append("dev_test_r02_manifest_overlap")

    preflight = {
        "audit_version": AUDIT_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "read_only_contract": {
            "memorax_allowed_endpoint": "/v1/memory-buckets/search",
            "memorax_add_calls_allowed": 0,
            "qa_solver_expansion_calls_allowed": 0,
        },
        "freeze": {
            split: {
                "manifest_path": str(path),
                "manifest_sha256": _sha(path),
                "candidate_count": len(freeze_rows[split]),
                "root_count": len({row["root_id"]
                                   for row in freeze_rows[split]}),
            } for split, path in freeze_paths.items()
        },
        "root_split_manifest_path": str(args.split_manifest),
        "root_split_manifest_sha256": _sha(args.split_manifest),
        "r02_manifest_path": str(args.r02_manifest),
        "r02_manifest_sha256": _sha(args.r02_manifest),
        "r02_train_root_count": len(r02_roots),
        "dev_test_r02_root_overlap": r02_overlap,
    }
    _write(run_root / "preflight_report.json", preflight)
    if errors:
        raise RuntimeError("preflight_failed:" + ",".join(errors))

    inventory = _intent_inventory(args.runs_root)
    client = MemoraxClient(load_memorax_config(
        credentials_path=args.credentials,
        allow_insecure_http=args.allow_insecure_http))
    manifest: list[dict[str, Any]] = []
    orphan_rows: list[dict[str, Any]] = []
    current_snapshots: list[dict[str, Any]] = []

    for root_id in sorted(roots, key=lambda value:
                          (roots[value]["split"], value)):
        root = roots[root_id]
        eval_dialogue_path = Path(root["eval_dialogue_path"])
        source_metadata_path = eval_dialogue_path.parent / "source_metadata.json"
        source_metadata = _load(source_metadata_path)
        source_path = Path(str(source_metadata["source_path"]))
        physical_root, reports = _select_authority(
            root_id, source_path, args.r01_run)
        report_path = physical_root / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json"
        report = _load(report_path)
        memory_root = physical_root / "memory"
        parent_intent_path = memory_root / "parent_add_intent.json"
        child_intent_path = memory_root / "child_add_intent.json"
        parent_completion_path = memory_root / "parent_add_completion.json"
        child_completion_path = memory_root / "child_add_completion.json"
        saved_snapshot_path = memory_root / "post_generation_bucket_snapshot.json"
        parent_intent = _load(parent_intent_path)
        child_intent = _load(child_intent_path)
        parent_completion = _load(parent_completion_path)
        child_completion = _load(child_completion_path)
        saved_snapshot = _load(saved_snapshot_path)
        eval_dialogue = _load(eval_dialogue_path)
        source_dialogue = _load(source_path)
        user_id = str(report["user_id"])
        namespace = _namespace(user_id, root_id)
        sessions = inventory.get(user_id, [])
        session_ids = sorted({row["session_id"] for row in sessions})
        r02_intents = [row for row in sessions if row["round_index"] == 2
                       or ":R02" in row["session_id"]]
        current = _search(client, root_id=root_id, user_id=user_id,
                          label="authority")
        current_snapshots.append({
            "root_id": root_id,
            "split": root["split"],
            "user_id": user_id,
            "namespace": namespace,
            "bucket_counts": current["bucket_counts"],
            "snapshot_sha256": current["snapshot_sha256"],
        })

        root_errors: list[str] = []
        checks = {
            "report_completed": report.get("status") == "completed",
            "report_memory_write_verified":
                report.get("memory_write_verified") is True,
            "report_memory_visibility_verified":
                report.get("memory_visibility_verified") is True,
            "parent_add_count_one": report.get("parent_add_calls") == 1,
            "r01_add_count_one": report.get("child_add_calls") == 1,
            "user_id_consistent": parent_intent.get("user_id") == user_id
                and child_intent.get("user_id") == user_id,
            "r00_session_correct": parent_intent.get("round_index") == 0
                and parent_intent.get("session_id") ==
                f"{root_id}:memorax:R00",
            "r01_session_correct": child_intent.get("round_index") == 1
                and child_intent.get("session_id") ==
                f"{root_id}:memorax:R01",
            "parent_add_verified":
                parent_completion.get("verified_complete") is True
                and parent_completion.get("final_status") == "success",
            "r01_add_verified": child_completion.get(
                "verified_complete") is True
                and child_completion.get("final_status") == "success",
            "eval_dialogue_file_sha_matches":
                _sha(eval_dialogue_path) == root["eval_dialogue_sha256"],
            "source_metadata_file_matches":
                _sha(source_path) == source_metadata.get("source_sha256"),
            "canonical_dialogue_content_matches":
                eval_dialogue.get("complete_dialogue_sha256") ==
                source_dialogue.get("complete_dialogue_sha256"),
            "saved_snapshot_identity_matches":
                saved_snapshot.get("root_id") == root_id
                and saved_snapshot.get("user_id") == user_id,
            "current_snapshot_equals_frozen_r01":
                current.get("snapshot_sha256") ==
                saved_snapshot.get("snapshot_sha256"),
            "no_r02_intent_for_authority_user_id": not r02_intents,
        }
        for name, passed in checks.items():
            if not passed:
                root_errors.append(name)
        row = {
            "audit_version": AUDIT_VERSION,
            "root_id": root_id,
            "split": root["split"],
            "candidate_count": root["candidate_count"],
            "authority_status": "PASS" if not root_errors else "FAIL",
            "errors": root_errors,
            "user_id": user_id,
            "namespace": namespace,
            "known_session_ids": session_ids,
            "r02_intent_count": len(r02_intents),
            "r02_manifest_member": root_id in r02_roots,
            "bucket_counts": current["bucket_counts"],
            "bucket_total": sum(current["bucket_counts"].values()),
            "current_snapshot_sha256": current["snapshot_sha256"],
            "frozen_r01_snapshot_sha256": saved_snapshot.get(
                "snapshot_sha256"),
            "checks": checks,
            "paths": {
                "eval_dialogue": str(eval_dialogue_path),
                "source_metadata": str(source_metadata_path),
                "canonical_parent_r01": str(source_path),
                "r01_report": str(report_path),
                "parent_add_intent": str(parent_intent_path),
                "r01_add_intent": str(child_intent_path),
                "frozen_r01_snapshot": str(saved_snapshot_path),
            },
            "sha256": {
                "eval_dialogue": _sha(eval_dialogue_path),
                "source_metadata": _sha(source_metadata_path),
                "canonical_parent_r01": _sha(source_path),
                "r01_report": _sha(report_path),
                "parent_add_intent": _sha(parent_intent_path),
                "r01_add_intent": _sha(child_intent_path),
                "frozen_r01_snapshot_file": _sha(saved_snapshot_path),
            },
        }
        manifest.append(row)

        for other in reports:
            if Path(other["physical_root"]) == physical_root:
                continue
            orphan_user_id = other["user_id"]
            orphan_snapshot = _search(
                client, root_id=root_id, user_id=orphan_user_id,
                label="orphan") if orphan_user_id else None
            orphan_rows.append({
                "audit_version": AUDIT_VERSION,
                "root_id": root_id,
                "split": root["split"],
                "authority": False,
                "attempt_status": other["status"],
                "user_id": orphan_user_id,
                "namespace": _namespace(orphan_user_id, root_id)
                if orphan_user_id else "",
                "report_path": other["path"],
                "report_sha256": _sha(Path(other["path"])),
                "bucket_counts": orphan_snapshot["bucket_counts"]
                if orphan_snapshot else {},
                "bucket_total": sum(orphan_snapshot["bucket_counts"].values())
                if orphan_snapshot else 0,
                "snapshot_sha256": orphan_snapshot["snapshot_sha256"]
                if orphan_snapshot else "",
                "isolated_from_authority": orphan_user_id != user_id,
            })

    authority_failures = [row for row in manifest
                          if row["authority_status"] != "PASS"]
    call_paths = Counter(str(call.get("path") or "")
                         for call in client.calls)
    non_search_calls = sum(count for path, count in call_paths.items()
                           if path != "/v1/memory-buckets/search")
    split_summary: dict[str, Any] = {}
    for split in ("dev", "test"):
        rows = [row for row in manifest if row["split"] == split]
        bucket_totals = Counter()
        for row in rows:
            bucket_totals.update(row["bucket_counts"])
        split_summary[split] = {
            "root_count": len(rows),
            "candidate_count": sum(row["candidate_count"] for row in rows),
            "authority_pass_count": sum(
                row["authority_status"] == "PASS" for row in rows),
            "current_equals_frozen_r01_count": sum(
                row["checks"]["current_snapshot_equals_frozen_r01"]
                for row in rows),
            "r02_intent_count": sum(row["r02_intent_count"] for row in rows),
            "bucket_counts_across_namespaces": dict(bucket_totals),
        }
    summary = {
        "audit_version": AUDIT_VERSION,
        "status": "DEV_TEST_MEMORY_NAMESPACE_AUDIT_PASS"
        if not authority_failures and not non_search_calls else
        "DEV_TEST_MEMORY_NAMESPACE_AUDIT_FAIL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "root_count": len(manifest),
        "candidate_count": sum(row["candidate_count"] for row in manifest),
        "split_summary": split_summary,
        "authority_namespace_count": len({row["user_id"]
                                          for row in manifest}),
        "authority_failure_count": len(authority_failures),
        "current_equals_frozen_r01_count": sum(
            row["checks"]["current_snapshot_equals_frozen_r01"]
            for row in manifest),
        "authority_r02_intent_count": sum(row["r02_intent_count"]
                                           for row in manifest),
        "r02_train_manifest_overlap_count": len(r02_overlap),
        "orphan_failed_attempt_namespace_count": len(orphan_rows),
        "orphan_namespaces_isolated_count": sum(
            row["isolated_from_authority"] for row in orphan_rows),
        "memorax_call_count": len(client.calls),
        "memorax_call_paths": dict(call_paths),
        "memorax_non_search_call_count": non_search_calls,
        "memorax_add_call_count": call_paths.get("/v1/memories/add", 0),
        "qa_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "decision": "READY_FOR_DEV_TEST_R01_MEMORY_FORK_PLANNING"
        if not authority_failures and not non_search_calls else "BLOCKED",
    }

    _write_jsonl(run_root / "dev_test_memory_namespace_manifest.jsonl",
                 manifest)
    _write_jsonl(run_root / "orphan_attempt_namespace_manifest.jsonl",
                 orphan_rows)
    _write_jsonl(run_root / "current_memory_snapshot_manifest.jsonl",
                 current_snapshots)
    _write(run_root / "memorax_read_call_audit.json", {
        "audit_version": AUDIT_VERSION,
        "calls": client.calls,
        "path_counts": dict(call_paths),
        "only_bucket_search": non_search_calls == 0,
        "add_call_count": call_paths.get("/v1/memories/add", 0),
        "authorization_header_recorded": False,
        "api_key_recorded": False,
    })
    _write(run_root / "final_namespace_audit_report.json", summary)
    markdown = [
        "# Dev/Test Memorax Namespace Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Split | Roots | Candidates | Authority pass | Current == frozen R01 | R02 intents |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("dev", "test"):
        row = split_summary[split]
        markdown.append(
            f"| {split} | {row['root_count']} | {row['candidate_count']} | "
            f"{row['authority_pass_count']} | "
            f"{row['current_equals_frozen_r01_count']} | "
            f"{row['r02_intent_count']} |")
    markdown += [
        "",
        f"- Authority namespaces: {summary['authority_namespace_count']}",
        f"- Isolated failed-attempt namespaces: {summary['orphan_failed_attempt_namespace_count']}",
        f"- Memorax bucket-search calls: {summary['memorax_call_count']}",
        f"- Memorax Add calls: {summary['memorax_add_call_count']}",
        f"- Decision: `{summary['decision']}`",
        "",
    ]
    md_path = run_root / "final_namespace_audit_report.md"
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    md_path.with_name(md_path.name + ".sha256").write_text(
        _sha(md_path) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--split-manifest", type=Path,
                        default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--r01-run", type=Path, default=DEFAULT_R01_RUN)
    parser.add_argument("--r02-manifest", type=Path,
                        default=DEFAULT_R02_MANIFEST)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-http", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
