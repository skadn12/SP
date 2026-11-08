"""Frozen fresh-evaluation phase and source-allocation gates; no model calls.

The post-dev closure is written once by the parent only after an independent
audit. Bindings attest immutable files on shared storage; they do not establish
cryptographic identities or automatically certify semantic judgments.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

TERMINAL_STATES = {"BASELINE_HOLD", "MATCHING_HOLD", "EXPANSION_STOPPED", "QA_COMPLETE", "QA_FAILED"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item, expected=None):
    require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "PHASE_BINDING_FIELDS")
    path = Path(item["path"])
    require(path.is_absolute() and binding(path) == item, "PHASE_BINDING_CHANGED:" + str(path))
    require(expected is None or path.resolve() == Path(expected).resolve(), "PHASE_BINDING_WRONG_PATH")
    return path.resolve()


def timestamp(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0, "PHASE_TIMESTAMP_UTC_REQUIRED")
    return value


def frozen_binding(master, path):
    master, path = Path(master).resolve(), Path(path).resolve()
    freeze = read(master / "MASTER_FREEZE.json")
    require(freeze.get("status") == "FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS", "PHASE_MASTER_NOT_READY")
    rows = [r for r in freeze["files"] if r.get("path") == str(path)]
    require(len(rows) == 1, "PHASE_UNIQUE_FROZEN_BINDING_REQUIRED:" + str(path))
    verify(rows[0], path)
    return rows[0]


def source_assignments(master):
    """Read only frozen allocation metadata, never other roots' event text."""
    master = Path(master).resolve()
    frozen_binding(master, __file__)
    manifest_path = master / "MANIFEST.json"
    pool_path = master / "evidence/SOURCE_POOL_FREEZE.json"
    frozen_binding(master, manifest_path)
    frozen_binding(master, pool_path)
    roots = read(manifest_path)["roots"]
    require(len(roots) == 31 and len({r["root_id"] for r in roots}) == 31, "SOURCE_FIXED_31_ROOTS_REQUIRED")
    rows = read(pool_path).get("candidate_assignments")
    require(isinstance(rows, list) and len(rows) == 31 and {r["root_id"] for r in rows} == {r["root_id"] for r in roots},
            "SOURCE_EXACT_ROOT_ASSIGNMENTS_REQUIRED")
    consumed = read(pool_path)["consumed_source_ids"]
    require(isinstance(consumed, list) and all(isinstance(s, str) and s for s in consumed), "SOURCE_HISTORICAL_UNION_REQUIRED")
    used = []
    for row in rows:
        ids = row.get("candidate_source_ids")
        require(isinstance(ids, list) and len(ids) >= 3 and all(isinstance(s, str) and s.startswith("multiwoz22:train:") for s in ids),
                "SOURCE_NATIVE_TRAIN_CANDIDATES_REQUIRED")
        unit = master / "units" / ("root" + row["root_id"].rsplit("-", 1)[1])
        expected = unit / "inputs/EVENT_POOL.json"
        require(row.get("event_pool_binding") == frozen_binding(master, expected), "SOURCE_POOL_NOT_FROZEN_FOR_ROOT")
        used.extend(ids)
    require(len(used) == len(set(used)), "SOURCE_CANDIDATES_OVERLAP_ROOTS")
    require(not set(used).intersection(consumed), "SOURCE_CANDIDATE_HISTORICALLY_CONSUMED")
    return {r["root_id"]: r for r in rows}


def check_candidate_pool(master, root, pool):
    row = source_assignments(master)[root]
    events = pool.get("events")
    require(isinstance(events, list), "SOURCE_POOL_EVENTS_REQUIRED")
    ids = ["multiwoz22:train:" + e["dialogue_id"] for e in events]
    require(len(ids) == len(set(ids)) and set(ids) == set(row["candidate_source_ids"]), "SOURCE_POOL_CANDIDATE_SET_CHANGED")
    return row


def require_phase(master, root):
    """Test roots remain closed until all 15 dev roots have audited terminals."""
    master = Path(master).resolve()
    frozen_binding(master, __file__)
    manifest_path = master / "MANIFEST.json"
    frozen_binding(master, manifest_path)
    manifest = read(manifest_path)
    matches = [r for r in manifest["roots"] if r["root_id"] == root]
    require(len(matches) == 1 and matches[0].get("split") in {"dev", "test"}, "PHASE_ROOT_NOT_ASSIGNED")
    if matches[0]["split"] == "dev":
        return None
    path = master / "evidence/DEV_PHASE_CLOSED.json"
    require(path.is_file(), "TEST_PHASE_CLOSED_UNTIL_ALL_DEV_TERMINAL")
    closure = read(path)
    expected = {"schema_version": "v124_dev_phase_closed_v1", "status": "DEV_COMPLETE_BEFORE_TEST_READS",
                "batch_id": master.name, "solver_run": False, "score_based_selection": False}
    for key, value in expected.items():
        require(type(closure.get(key)) is type(value) and closure.get(key) == value, "DEV_CLOSURE_FIELD:" + key)
    verify(closure["master_freeze_binding"], master / "MASTER_FREEZE.json")
    verify(closure["manifest_binding"], manifest_path)
    accounting_path = verify(closure["terminal_accounting_binding"], master / "evidence/DEV_TERMINAL_ACCOUNTING.json")
    audit_path = verify(closure["independent_audit_binding"], master / "evidence/DEV_PHASE_CLOSE_AUDIT.json")
    accounting, audit = read(accounting_path), read(audit_path)
    require(accounting.get("schema_version") == "v124_dev_terminal_accounting_v1" and accounting.get("status") == "ALL_DEV_TERMINAL"
            and accounting.get("solver_run") is False and accounting.get("score_based_selection") is False, "DEV_ACCOUNTING_POLICY")
    require(audit.get("status") == "PASS" and audit.get("terminal_accounting_binding") == closure["terminal_accounting_binding"],
            "DEV_CLOSURE_INDEPENDENT_AUDIT_REQUIRED")
    dev = {r["root_id"] for r in manifest["roots"] if r["split"] == "dev"}
    rows = accounting.get("roots")
    require(len(dev) == 15 and isinstance(rows, list) and len(rows) == 15 and {r["root_id"] for r in rows} == dev,
            "DEV_ACCOUNTING_EXACT_15_ROOTS")
    for row in rows:
        require(row.get("terminal_state") in TERMINAL_STATES, "DEV_ROOT_NOT_TERMINAL")
        evidence = row.get("terminal_evidence_bindings")
        require(isinstance(evidence, list) and evidence, "DEV_TERMINAL_EVIDENCE_REQUIRED")
        for item in evidence:
            verify(item)
    master_at = timestamp(read(master / "MASTER_FREEZE.json")["created_at_utc"])
    closed_at = timestamp(closure["created_at_utc"])
    require(master_at <= timestamp(accounting["created_at_utc"]) <= timestamp(audit["created_at_utc"]) <= closed_at <= datetime.now(timezone.utc),
            "DEV_CLOSURE_TIME_ORDER")
    return binding(path)
