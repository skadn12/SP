"""Read-only capture of the current R01 bucket payloads.

This is deliberately separate from the bootstrap runner.  It performs two
Search calls only, one for each already-populated namespace, and never calls
Add or status endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SMOKE = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_3")
DEFAULT_PREFLIGHT = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_2")
DEFAULT_OUTPUT = RUN_BASE / (
    "h4_round1_feedback_memory_semantic_parity_v1_live_3")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
VERSION = "h4_round1_feedback_memory_semantic_parity_v1_readonly_capture_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_name(path.name + ".sha256").write_text(
        _sha(path) + "\n", encoding="utf-8")


def run(*, smoke_root: Path, preflight_root: Path, output_root: Path,
        credentials: Path, allow_insecure_http: bool) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    smoke = _load(smoke_root / "final_feedback_memory_smoke_report.json")
    preflight = _load(preflight_root / "preflight_report.json")
    mapping = _load(preflight_root / "namespace_mapping.json")
    if smoke.get("status") != "ROUND1_R01_BOOTSTRAP_SMOKE_PASS":
        raise RuntimeError("bootstrap_smoke_not_passed")
    root_id = str(smoke.get("root_id") or preflight.get("root_id") or "")
    if not root_id or set(mapping) != {"control", "feedback"}:
        raise RuntimeError("namespace_mapping_invalid")

    client = MemoraxClient(load_memorax_config(
        credentials_path=credentials,
        allow_insecure_http=allow_insecure_http))
    captures: dict[str, Any] = {}
    for branch in ("control", "feedback"):
        namespace = str(mapping[branch]["namespace"])
        user_id = str(mapping[branch]["user_id"])
        query = build_bucket_search_payload(
            root_id=root_id, limit=100, max_buckets_per_type=500,
            user_namespace=namespace)
        if query["payload"].get("user_id") != user_id:
            raise RuntimeError(f"search_user_id_mismatch:{branch}")
        request_id = f"{VERSION}-{branch}-{root_id}-search"
        response = client.search_buckets(query["payload"], request_id=request_id)
        snapshot = normalize_bucket_snapshot(
            response, root_id=root_id, user_id=user_id)
        snapshot_path = output_root / f"{branch}_r01_bucket_snapshot.json"
        _write(snapshot_path, snapshot)
        captures[branch] = {
            "namespace": namespace,
            "user_id": user_id,
            "request_id": request_id,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": _sha(snapshot_path),
            "bucket_counts": snapshot["bucket_counts"],
            "bucket_total": sum(snapshot["bucket_counts"].values()),
        }
    manifest = {
        "capture_version": VERSION,
        "captured_at": _now(),
        "root_id": root_id,
        "smoke_root": str(smoke_root),
        "preflight_root": str(preflight_root),
        "branches": captures,
        "memorax_call_count": len(client.calls),
        "search_calls": len(client.calls),
        "add_calls": 0,
        "status_calls": 0,
        "solver_calls": 0,
        "qa_calls": 0,
        "semantic_auditor_calls": 0,
        "test_access": 0,
    }
    _write(output_root / "readonly_snapshot_capture_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--preflight-root", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memorax-credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        smoke_root=args.smoke_root.resolve(),
        preflight_root=args.preflight_root.resolve(),
        output_root=args.output_root.resolve(),
        credentials=args.memorax_credentials.resolve(),
        allow_insecure_http=args.allow_insecure_memorax_http,
    )
    print(json.dumps({
        "status": "READONLY_R01_SNAPSHOT_CAPTURE_PASS",
        "root_id": result["root_id"],
        "memorax_call_count": result["memorax_call_count"],
        "add_calls": result["add_calls"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

