"""Verify a completed run against its original code hashes and saved code bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_archive(run):
    run = Path(run).resolve()
    manifest_path = run / "MANIFEST.json"
    manifest = read(manifest_path)
    index = read(run / "CODE_SNAPSHOT_INDEX.json")
    require(digest(manifest_path) == index["source_manifest_sha256"], "original_manifest_changed")
    snapshot_rows = {row["path"]: row for row in index["bindings"]}
    require(len(snapshot_rows) == len(index["bindings"]), "duplicate_snapshot_binding")
    relocated = []
    for row in manifest["files"]:
        original = Path(row["path"])
        if original.suffix == ".py":
            saved = snapshot_rows.get(row["path"])
            require(saved and saved["sha256"] == row["sha256"], "snapshot_manifest_mismatch")
            snapshot = Path(saved["snapshot_path"]).resolve()
            require(snapshot.is_relative_to(run / "code_snapshot"), "snapshot_outside_archive")
            require(digest(snapshot) == row["sha256"], "frozen_code_snapshot_changed")
            if not original.exists() or digest(original) != row["sha256"]:
                relocated.append(row["path"])
        else:
            require(digest(original) == row["sha256"], "frozen_artifact_changed:" + row["path"])
    proof = read(run / "LIVE_BINDINGS_VERIFIED.json")
    require(proof["all_original_bindings_verified_after_live_before_legacy_restore"] is True,
            "missing_post_live_verification")
    require(digest(run / "REPORT.json") == proof["report_sha256"], "completed_report_changed")
    report = read(run / "REPORT.json")
    attempts = list(run.glob("requests/*/HTTP_ATTEMPT.json"))
    require(len(attempts) == report["http_attempts"], "http_count_mismatch")
    for attempt in attempts:
        actual = read(attempt)
        request = read(attempt.parent / "REQUEST.json")
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(canonical).hexdigest() == actual["payload_sha256"], "request_payload_changed")
        require(actual["actual_payload_equals_frozen"] is True, "actual_payload_mismatch")
    for row in report["routes"]:
        root, route = row["root_id"], row["route"]
        letter = "A" if route == "old_event_continuation" else "B"
        original = read(run / "sources" / f"{root}.json")["turns"]
        dialogue = read(run / "roots" / root / letter / "cumulative_dialogue.json")["turns"]
        require(dialogue[:len(original)] == original, "branch_base_changed")
        require(all(t["turn_ref"].startswith(f"R02:{letter}:") for t in dialogue[len(original):]),
                "other_branch_in_history")
        require(read(run / "roots" / root / letter / "RESULT.json") == row, "result_report_mismatch")
    return {"status": "PASS", "verification": "original_manifest_with_exact_code_snapshots",
            "snapshotted_code_files": len(snapshot_rows), "relocated_code_files": relocated,
            "http_attempts": len(attempts), "accepted_scenes": report["accepted_scenes"],
            "historical_reports_overwritten": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.run), ensure_ascii=False, indent=2))
