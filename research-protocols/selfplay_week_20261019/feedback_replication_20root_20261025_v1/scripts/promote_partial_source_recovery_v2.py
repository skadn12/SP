"""Promote a verified partial recovery selection to the canonical source freeze.

The prior canonical source outputs are preserved byte-for-byte under snapshot
names; this is only metadata promotion, never source-content editing.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import source_call_recovery as recovery  # noqa: E402
import source_driver  # noqa: E402


def promote() -> dict:
    selection_path = BASE / "SOURCE_SELECTION_RECOVERY_V2.json"
    freeze_path = BASE / "SOURCE_FREEZE_RECOVERY_V2.json"
    failures_path = BASE / "SOURCE_FAILURES_RECOVERY_V2.json"
    if not all(path.is_file() for path in (selection_path, freeze_path, failures_path)):
        raise RuntimeError("Recovery v2 selection, freeze, and failures must be published first")
    selection, freeze, failures = map(recovery.read, (selection_path, freeze_path, failures_path))
    for key, path in (("source_selection_binding", selection_path), ("failures_binding", failures_path)):
        if freeze.get(key) != recovery.bind(path):
            raise RuntimeError(f"Recovery freeze has stale {key}")
    if freeze.get("status") not in ("PASS", "PARTIAL"):
        raise RuntimeError("Unsupported recovery freeze status")
    roots = source_driver.validate_preparation()
    configured = [row["root_id"] for row in roots]
    selected_ids = [row.get("root_id") for row in selection.get("roots", [])]
    if (selection.get("split") != "train" or len(set(selected_ids)) != len(selected_ids)
            or selected_ids != [rid for rid in configured if rid in set(selected_ids)]):
        raise RuntimeError("Recovery selection has duplicate or non-frozen root order")
    if freeze.get("planned_root_count") != len(configured) or len(configured) != 16:
        raise RuntimeError("Recovery freeze planned denominator drift")
    if (freeze.get("passed_root_count") != len(selected_ids)
            or freeze.get("failed_root_count") != 16 - len(selected_ids)
            or len(failures.get("failed_roots", [])) != 16 - len(selected_ids)):
        raise RuntimeError("Recovery freeze counts do not match its selection/failures")
    if freeze.get("status") != ("PASS" if len(selected_ids) == 16 else "PARTIAL"):
        raise RuntimeError("Recovery freeze status/count mismatch")
    for entry in selection.get("roots", []):
        if not recovery._binding_matches(entry):
            raise RuntimeError(f"Selected source binding drift: {entry}")
        source = recovery.read(entry["path"])
        if source.get("root_id") != entry["root_id"] or not isinstance(source.get("turns"), list):
            raise RuntimeError("Selected recovery source root/shape mismatch")
    for binding in freeze.get("root_outcomes", []):
        if not recovery._binding_matches(binding):
            raise RuntimeError("Root resolution binding drift")
        row = recovery.read(binding["path"])
        if row.get("root_id") not in configured:
            raise RuntimeError("Root resolution outside frozen selection")

    lock_path = BASE / ".source_driver.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Source authoring/recovery process is still active") from exc
        snapshots = []
        mapping = ((selection_path, BASE / "SOURCE_SELECTION_FINAL.json"),
                   (freeze_path, BASE / "SOURCE_FREEZE.json"),
                   (failures_path, BASE / "SOURCE_FAILURES.json"))
        for _, target in mapping:
            if not target.exists():
                raise RuntimeError(f"Missing original canonical artifact: {target}")
            snapshot = BASE / f"SOURCE_DRIVER_INITIAL_{target.name}"
            old_binding = recovery.bind(target)
            if snapshot.exists():
                if snapshot.read_bytes() != target.read_bytes():
                    raise RuntimeError(f"Initial canonical snapshot conflicts with current bytes: {snapshot}")
            else:
                snapshot.write_bytes(target.read_bytes())
            snapshots.append({"original": old_binding, "snapshot": recovery.bind(snapshot)})
        for source, target in mapping:
            temporary = target.with_suffix(target.suffix + ".recovery_v2.tmp")
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, target)
        receipt = {"schema_version": "fgr20_partial_source_recovery_promotion_v2",
                   "status": freeze["status"], "passed_roots": len(selected_ids),
                   "planned_roots": 16, "recovery_selection": recovery.bind(selection_path),
                   "recovery_freeze": recovery.bind(freeze_path),
                   "recovery_failures": recovery.bind(failures_path), "snapshots": snapshots,
                   "canonical_selection": recovery.bind(BASE / "SOURCE_SELECTION_FINAL.json"),
                   "canonical_freeze": recovery.bind(BASE / "SOURCE_FREEZE.json"),
                   "canonical_failures": recovery.bind(BASE / "SOURCE_FAILURES.json"),
                   "old_source_artifacts_preserved_as_snapshots": True,
                   "root_substitution": False, "parent_source_used_as_final": False}
        return recovery.write_immutable(BASE / "SOURCE_RECOVERY_CANONICAL_PROMOTION_V2.json", receipt)


if __name__ == "__main__":
    print(json.dumps(promote(), ensure_ascii=False, indent=2))
