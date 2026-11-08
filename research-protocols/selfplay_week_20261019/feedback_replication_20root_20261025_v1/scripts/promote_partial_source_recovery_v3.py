"""Promote bound recovery metadata without editing source or semantic decisions.

Run after the source continuation batches have finished and their final
publisher has written SOURCE_*_RECOVERY_V3.json.  This script preserves the prior canonical
metadata byte-for-byte, adds the actual PASS recovery outcomes required by the
question driver, and rebinds the canonical selection/failure filenames.  It
never calls a model or creates a source/review/PASS decision.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import source_call_recovery as recovery  # noqa: E402
import source_driver  # noqa: E402

RECEIPT = BASE / "SOURCE_RECOVERY_CANONICAL_PROMOTION_V3.json"
INTENT = BASE / "SOURCE_RECOVERY_CANONICAL_PROMOTION_V3_INTENT.json"
CANONICAL_NAMES = ("SOURCE_SELECTION_FINAL.json", "SOURCE_FREEZE.json", "SOURCE_FAILURES.json")
CONTINUED_ROOTS = ("24719", "25652", "45849", "75241")


def recovery_names(version: int) -> tuple[str, str, str]:
    return tuple(f"SOURCE_{kind}_RECOVERY_V{version}.json"
                 for kind in ("SELECTION", "FREEZE", "FAILURES"))


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def bytes_binding(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def file_binding(value: dict[str, Any]) -> dict[str, Any]:
    """Selection rows carry root_id; only the three binding fields identify bytes."""
    return {key: value.get(key) for key in ("path", "sha256", "bytes")}


def verify(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or recovery.bind(value.get("path", "")) != file_binding(value):
        raise RuntimeError(f"Artifact binding drift: {value}")


def validate_counts(configured: list[str], selection: dict[str, Any],
                    freeze: dict[str, Any], failures: dict[str, Any]) -> list[str]:
    rows = selection.get("roots")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("Recovery selection roots must be a list of objects")
    ids = [row.get("root_id") for row in rows]
    if (selection.get("split") != "train" or len(configured) != 16
            or len(set(configured)) != 16 or len(set(ids)) != len(ids)
            or ids != [rid for rid in configured if rid in set(ids)]):
        raise RuntimeError("Recovery selection has duplicate, unplanned, or reordered roots")
    if (freeze.get("planned_root_count") != 16
            or freeze.get("passed_root_count") != len(ids)
            or freeze.get("failed_root_count") != 16 - len(ids)
            or freeze.get("status") != ("PASS" if len(ids) == 16 else "PARTIAL")):
        raise RuntimeError("Recovery freeze denominator/status drift")
    failed_rows = failures.get("failed_roots")
    if not isinstance(failed_rows, list):
        raise RuntimeError("Recovery failures must preserve every failed root")
    failed_ids = [row.get("root_id") for row in failed_rows if isinstance(row, dict)]
    if (len(failed_ids) != len(failed_rows)
            or failed_ids != [rid for rid in configured if rid not in set(ids)]
            or failures.get("planned_roots") != 16
            or failures.get("passed_roots") != len(ids)):
        raise RuntimeError("Recovery failure roster/count drift")
    return ids


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def ensure_idle() -> None:
    active = []
    for path in sorted((BASE / "process_registry").glob("*.json")):
        row = recovery.read(path)
        if row.get("active") is not True:
            continue
        pid = row.get("pid")
        if isinstance(pid, int) and process_start(pid) == str(row.get("start_token")):
            active.append({"pid": pid, "registry": str(path)})
    if active:
        raise RuntimeError(f"Fresh semantic calls are still active: {active}")


def verify_outcome(path: Path, root: str) -> dict[str, Any]:
    row = recovery.read(path)
    if row.get("root_id") != root:
        raise RuntimeError(f"Source outcome root mismatch: {path}")
    for bound in row.get("artifacts", []):
        verify(bound)
    return row


def canonical_freeze(freeze: dict[str, Any], selection_binding: dict[str, Any],
                     failure_binding: dict[str, Any], original_outcomes: list[dict[str, Any]],
                     upstream: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(freeze)
    result["source_selection_binding"] = selection_binding
    result["failures_binding"] = failure_binding
    # These are existing independent-review outcomes, not new engineering PASSes.
    result["original_root_outcomes"] = original_outcomes
    result["canonical_recovery_promotion_v3"] = upstream
    return result


def validate_inputs(version: int) -> dict[str, Any]:
    ensure_idle()
    for root in CONTINUED_ROOTS:
        path = BASE / "source_recovery_continuations_v2" / f"cc-train-episode-{root}" / "CONTINUATION_OUTCOME.json"
        if not path.is_file():
            raise RuntimeError(f"Source continuation batch has not finished: {root}")
    names = recovery_names(version)
    paths = [BASE / name for name in names]
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"Publish SOURCE_*_RECOVERY_V{version}.json after its source batch exits")
    selection, freeze, failures = [recovery.read(path) for path in paths]
    for key, path in (("source_selection_binding", paths[0]), ("failures_binding", paths[2])):
        if freeze.get(key) != recovery.bind(path):
            raise RuntimeError(f"Recovery freeze has stale {key}")
    roots = source_driver.validate_preparation()
    configured = [row["root_id"] for row in roots]
    ids = validate_counts(configured, selection, freeze, failures)
    verify(freeze["parent_selection_binding"])
    if freeze["parent_selection_binding"] != recovery.bind(BASE / "PARENT_SOURCE_SELECTION.json"):
        raise RuntimeError("Recovery parent selection differs from frozen parent selection")
    resolution_ids = []
    selected = {row["root_id"]: file_binding(row) for row in selection["roots"]}
    for bound in freeze.get("root_outcomes", []):
        verify(bound)
        row = recovery.read(bound["path"])
        rid = row.get("root_id")
        resolution_ids.append(rid)
        resolved = row.get("resolved_source_binding", row.get("source_binding"))
        if rid in selected:
            if row.get("status") != "PASS" or resolved != selected[rid]:
                raise RuntimeError(f"Recovery resolution does not match admitted source: {rid}")
        elif row.get("status") == "PASS" or resolved is not None:
            raise RuntimeError(f"Recovery resolution claims an unselected source: {rid}")
    if resolution_ids != configured:
        raise RuntimeError("Recovery freeze must bind exactly one ordered resolution per planned root")

    originals = []
    actual_records = []
    for rid in configured:
        original = BASE / "source_recovery" / rid / "ROOT_RECOVERY.json"
        if original.is_file():
            row = verify_outcome(original, rid)
            originals.append(recovery.bind(original))
            actual_records.append((rid, str(original), row))
        continued = BASE / "source_recovery_continuations_v2" / rid / "CONTINUATION_OUTCOME.json"
        if continued.is_file():
            row = verify_outcome(continued, rid)
            bound = recovery.bind(continued)
            if bound not in freeze.get("continuation_v2_outcomes", []):
                raise RuntimeError(f"V2 freeze omits a completed continuation: {rid}")
            actual_records.append((rid, str(continued), row))
    known = {path for _, path, _ in actual_records}
    for field in ("original_root_outcomes", "continuation_v2_outcomes", "continuation_outcomes"):
        for bound in freeze.get(field, []):
            verify(bound)
            path = Path(bound["path"])
            row = recovery.read(path)
            rid = row.get("root_id")
            if rid not in configured:
                raise RuntimeError(f"Unplanned source outcome in {field}: {rid}")
            row = verify_outcome(path, rid)
            if str(path) not in known:
                actual_records.append((rid, str(path), row))
                known.add(str(path))
    # This is an existing engineering contract, not semantic adjudication.
    import question_driver
    for entry in selection["roots"]:
        verify(entry)
        rid = entry["root_id"]
        if not any(r == rid and row.get("status") == "PASS"
                   and row.get("source_binding") == selected[rid]
                   for r, _, row in actual_records):
            raise RuntimeError(f"No actual saved PASS outcome binds selected source: {rid}")
        source = recovery.read(entry["path"])
        question_driver._validate_extended_source(rid, entry, source, actual_records)
    merged_originals = {bound["path"]: bound for bound in freeze.get("original_root_outcomes", [])}
    merged_originals.update({bound["path"]: bound for bound in originals})
    return {"selection": selection, "freeze": freeze, "failures": failures,
            "selected_ids": ids, "original_outcomes": list(merged_originals.values()),
            "recovery_names": names, "recovery_version": version,
            "upstream_bindings": {name: recovery.bind(BASE / name) for name in names},
            "question_driver_binding": recovery.bind(Path(question_driver.__file__))}


def expected_outputs(inputs: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    selection_raw = (BASE / inputs["recovery_names"][0]).read_bytes()
    failures_raw = (BASE / inputs["recovery_names"][2]).read_bytes()
    upstream = {"schema_version": "fgr20_canonical_source_metadata_promotion_v3",
                "source_content_modified": False, "semantic_decisions_modified": False,
                "prior_recovery_artifacts_untouched": True,
                "promoter_binding": recovery.bind(Path(__file__)),
                "recovery_input_version": inputs["recovery_version"],
                "question_driver_contract_binding": inputs["question_driver_binding"],
                "recovery_inputs": inputs["upstream_bindings"]}
    freeze = canonical_freeze(
        inputs["freeze"], bytes_binding(BASE / CANONICAL_NAMES[0], selection_raw),
        bytes_binding(BASE / CANONICAL_NAMES[2], failures_raw),
        inputs["original_outcomes"], upstream)
    raw = {CANONICAL_NAMES[0]: selection_raw, CANONICAL_NAMES[1]: encoded(freeze),
           CANONICAL_NAMES[2]: failures_raw}
    return raw, freeze


def verify_receipt() -> dict[str, Any]:
    receipt = recovery.read(RECEIPT)
    verify(receipt["promoter_binding"])
    for field in ("canonical_bindings", "recovery_input_bindings"):
        for bound in receipt[field]:
            verify(bound)
    for row in receipt["preserved_prior_canonical"]:
        verify(row["snapshot"])
        if {k: row["original"][k] for k in ("sha256", "bytes")} != {
                k: row["snapshot"][k] for k in ("sha256", "bytes")}:
            raise RuntimeError("Canonical snapshot is not byte-for-byte identical to prior artifact")
    return receipt


def promote(*, check_only: bool = False, version: int = 3) -> dict[str, Any]:
    if RECEIPT.exists():
        return {"idempotent_reuse": True, "receipt_binding": recovery.bind(RECEIPT),
                "receipt": verify_receipt()}
    for path in (BASE / "CONTENT_FREEZE.json", BASE / "execution" / "STARTED.json"):
        if path.exists():
            raise RuntimeError(f"Refusing source metadata changes after downstream freeze/start: {path}")
    lock_path = BASE / ".source_driver.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Source authoring/recovery lock is still held") from exc
        inputs = validate_inputs(version)
        outputs, freeze = expected_outputs(inputs)
        expected = [bytes_binding(BASE / name, outputs[name]) for name in CANONICAL_NAMES]
        if check_only:
            return {"schema_version": "fgr20_source_promotion_v3_preflight",
                    "status": "PASS", "writes": 0, "model_calls": 0,
                    "passed_roots": len(inputs["selected_ids"]), "planned_roots": 16,
                    "recovery_input_version": version,
                    "canonical_expected_bindings": expected,
                    "promoter_binding": recovery.bind(Path(__file__))}
        if INTENT.exists():
            intent = recovery.read(INTENT)
            if intent["expected_canonical_bindings"] != expected:
                raise RuntimeError("Promotion inputs changed after write-once intent")
        else:
            snapshots = []
            for name in CANONICAL_NAMES:
                target = BASE / name
                snapshot = BASE / f"SOURCE_DRIVER_INITIAL_V3_{name}"
                if not target.is_file():
                    raise RuntimeError(f"Missing prior canonical artifact: {target}")
                old = target.read_bytes()
                if snapshot.exists():
                    if snapshot.read_bytes() != old:
                        raise RuntimeError(f"Pre-existing snapshot differs from prior canonical bytes: {snapshot}")
                else:
                    with snapshot.open("xb") as stream:
                        stream.write(old)
                snapshots.append({"original": bytes_binding(target, old),
                                  "snapshot": recovery.bind(snapshot)})
            intent = {"schema_version": "fgr20_source_promotion_v3_intent",
                      "expected_canonical_bindings": expected,
                      "preserved_prior_canonical": snapshots,
                      "recovery_input_bindings": list(inputs["upstream_bindings"].values()),
                      "promoter_binding": recovery.bind(Path(__file__))}
            recovery.write_immutable(INTENT, intent)
        for row in intent["preserved_prior_canonical"]:
            verify(row["snapshot"])
        old_by_name = {Path(row["original"]["path"]).name: row["original"]
                       for row in intent["preserved_prior_canonical"]}
        for name in CANONICAL_NAMES:
            target = BASE / name
            actual = recovery.bind(target)
            wanted = bytes_binding(target, outputs[name])
            if actual == wanted:
                continue
            if actual != old_by_name[name]:
                raise RuntimeError(f"Canonical bytes changed outside this promotion: {target}")
            temporary = target.with_name(target.name + ".promotion_v3.tmp")
            with temporary.open("wb") as stream:
                stream.write(outputs[name])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        # Validate the public downstream interface after all three metadata files exist.
        import question_driver
        question_driver.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
        _, sources = question_driver.load_selection(BASE / "SOURCE_SELECTION_FINAL.json")
        if list(sources) != inputs["selected_ids"]:
            raise RuntimeError("Canonical question-driver loader returned a different root roster")
        receipt = {"schema_version": "fgr20_partial_source_recovery_promotion_v3",
                   "status": freeze["status"], "passed_roots": len(sources), "planned_roots": 16,
                   "recovery_input_version": version,
                   "canonical_bindings": [recovery.bind(BASE / name) for name in CANONICAL_NAMES],
                   "recovery_input_bindings": intent["recovery_input_bindings"],
                   "preserved_prior_canonical": intent["preserved_prior_canonical"],
                   "promoter_binding": intent["promoter_binding"],
                   "intent_binding": recovery.bind(INTENT),
                   "question_driver_binding": inputs["question_driver_binding"],
                   "question_driver_load_selection": "PASS",
                   "old_source_artifacts_preserved_as_snapshots": True,
                   "source_content_modified": False, "semantic_decisions_modified": False,
                   "root_substitution": False, "parent_source_used_as_final": False,
                   "model_calls": 0}
        recovery.write_immutable(RECEIPT, receipt)
        return {"receipt_binding": recovery.bind(RECEIPT), "receipt": receipt}


def self_check() -> dict[str, Any]:
    roots = [f"cc-train-episode-{n}" for n in range(16)]
    selected = roots[::3]
    selection = {"split": "train", "roots": [{"root_id": rid, "path": "/example",
                    "sha256": "a" * 64, "bytes": 1} for rid in selected]}
    freeze = {"planned_root_count": 16, "passed_root_count": len(selected),
              "failed_root_count": 16 - len(selected), "status": "PARTIAL"}
    failures = {"planned_roots": 16, "passed_roots": len(selected),
                "failed_roots": [{"root_id": rid} for rid in roots if rid not in selected]}
    assert validate_counts(roots, selection, freeze, failures) == selected
    assert "root_id" not in file_binding(selection["roots"][0])
    rejected = 0
    for bad in (list(reversed(selection["roots"])), selection["roots"] + selection["roots"][:1],
                selection["roots"] + [{"root_id": "cc-train-episode-unplanned"}]):
        try:
            validate_counts(roots, {"split": "train", "roots": bad}, freeze, failures)
        except RuntimeError:
            rejected += 1
    assert rejected == 3
    old = copy.deepcopy(freeze)
    sb = {"path": "/canonical/SOURCE_SELECTION_FINAL.json", "sha256": "b" * 64, "bytes": 2}
    fb = {"path": "/canonical/SOURCE_FAILURES.json", "sha256": "c" * 64, "bytes": 3}
    actual_passes = [{"path": "/bound/ROOT_RECOVERY.json", "sha256": "d" * 64, "bytes": 4}]
    promoted = canonical_freeze(freeze, sb, fb, actual_passes, {"model_calls": 0})
    assert freeze == old and promoted["status"] == old["status"]
    assert promoted["source_selection_binding"] == sb and promoted["failures_binding"] == fb
    assert promoted["original_root_outcomes"] == actual_passes
    return {"schema_version": "fgr20_source_promotion_v3_self_check", "status": "PASS",
            "tests": ["selection_metadata_binding_projection", "fixed_root_order_and_counts",
                      "reordered_duplicate_unplanned_roots_rejected", "canonical_binding_paths",
                      "actual_outcomes_preserved", "no_status_or_semantic_changes"],
            "filesystem_mutations": 0, "model_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Validate all inputs without promotion")
    group.add_argument("--self-check", action="store_true", help="Pure-memory metadata tests; no calls or writes")
    parser.add_argument("--recovery-version", type=int, choices=(2, 3), default=3,
                        help="Published recovery metadata version; final V3 is the default")
    args = parser.parse_args()
    result = self_check() if args.self_check else promote(check_only=args.check, version=args.recovery_version)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
