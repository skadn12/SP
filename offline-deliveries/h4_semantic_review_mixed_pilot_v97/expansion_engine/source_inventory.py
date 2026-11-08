"""Consume source events at reservation, before any model request.

Historical v1/v2 accepted events and v4/v5/v6/v7/v8/v9/v11/v12 reservations remain consumed. A v15
reservation survives failure or interruption and belongs to one run/root/scene
for all its retries. All reservation generations share the existing schema.
The live runner must hold the shared v2 inventory lock around collect/reserve.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parents[1]
V1_RUNS = DELIVERIES / "h4_r02_dualroute_feedback_v1" / "runs"
V2_RUNS = DELIVERIES / "h4_r02_source_unique_v2" / "runs"
V4_RUNS = DELIVERIES / "h4_r02_independent_routes_v4" / "runs"
V5_RUNS = DELIVERIES / "h4_r02_stage_repair_v5" / "runs"
V6_RUNS = DELIVERIES / "h4_r02_card_contract_v6" / "runs"
V7_RUNS = DELIVERIES / "h4_r02_latest_state_v7" / "runs"
V8_RUNS = DELIVERIES / "h4_r02_event_state_v8" / "runs"
V9_RUNS = DELIVERIES / "h4_r02_reference_contract_v9" / "runs"
V11_RUNS = DELIVERIES / "h4_r02_yield_pilot_v11" / "runs"
V12_RUNS = DELIVERIES / "h4_r02_yield_pilot_v12" / "runs"
V15_RUNS = DELIVERIES / "h4_feedback_training_pool_v15_expansion" / "runs"
V90_RUNS = DELIVERIES / "h4_end_to_end_expansion_mixed_v90" / "expansion_engine" / "runs"
V91_RUNS = DELIVERIES / "h4_planner_cardinality_expansion_batch_v91" / "expansion_engine" / "runs"
V92_RUNS = DELIVERIES / "h4_card_fidelity_event_consistency_minfix_v92" / "expansion_engine" / "runs"
V93_RUNS = DELIVERIES / "h4_review_reference_minfix_v93" / "expansion_engine" / "runs"
V94_RUNS = DELIVERIES / "h4_dependency_plan_field_evidence_minfix_v94" / "expansion_engine" / "runs"
V95_RUNS = DELIVERIES / "h4_controlled_six_root_expansion_v95" / "expansion_engine" / "runs"
SCHEMA_VERSION = "h4.r02.source_reservations.v4"
_spec = importlib.util.spec_from_file_location(
    "r02_inventory_legacy_unique", V2_RUNS.parent / "unique_events.py")
assert _spec and _spec.loader
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
event_key = legacy.event_key
verify_cards_for_root = legacy.verify_cards_for_root


def _row(item: dict[str, Any], *, run: Path, root: str, scene: str) -> dict[str, Any]:
    key = event_key(item)
    source = item.get("lineage", item)
    digest = source.get("source_item_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid_source_item_sha256")
    source_id = item.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("invalid_source_id")
    if not isinstance(root, str) or not root or not isinstance(scene, str) or not scene:
        raise ValueError("invalid_reservation_slot")
    return {
        "dataset": key[0], "split": key[1], "item_id": key[2],
        "source_id": source_id, "source_item_sha256": digest,
        "run": str(run.resolve()), "root_id": root, "scene_id": scene,
    }


def _owner(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(Path(row["run"]).resolve()), row["root_id"], row["scene_id"]


def _assignment_rows(run: Path, assignments: dict[str, list[dict]]) -> list[dict]:
    rows = []
    events, slots = set(), set()
    for root, items in assignments.items():
        for index, item in enumerate(items, 1):
            if item.get("root_id", root) != root:
                raise ValueError("assignment_root_mismatch")
            if "run" in item and Path(item["run"]).resolve() != run.resolve():
                raise ValueError("assignment_run_mismatch")
            row = _row(item, run=run, root=root,
                       scene=item.get("scene_id", f"S{index:02d}"))
            if event_key(row) in events:
                raise ValueError("source_event_assigned_twice")
            if _owner(row) in slots:
                raise ValueError("reservation_slot_assigned_twice")
            events.add(event_key(row))
            slots.add(_owner(row))
            rows.append(row)
    return rows


def _merge(occupied: dict[tuple[str, str, str], dict], row: dict) -> None:
    """Keep every historical owner; conflicts consume an event, never free it.

    accepted_uses remains one row per event for the legacy allocator. The
    representative is accompanied by all original owner/hash/source bindings;
    no historical collision is silently overwritten or treated as valid reuse.
    New assignments still use validate_available and cannot reuse any owner.
    """
    key = event_key(row)
    old = occupied.get(key)
    clean = {name: deepcopy(value) for name, value in row.items() if name != "historical_owners"}
    if old is None:
        occupied[key] = {**clean, "historical_owners": [deepcopy(clean)]}
        return
    owners = deepcopy(old.get("historical_owners", [{name: value for name, value in old.items()
                                                   if name != "historical_owners"}]))
    identity = (_owner(clean), clean["source_item_sha256"], clean["source_id"])
    match = next((index for index, item in enumerate(owners)
                  if (_owner(item), item["source_item_sha256"], item["source_id"]) == identity), None)
    if match is None:
        owners.append(clean)
    elif clean.get("status") == "accepted" or owners[match].get("status") != "accepted":
        owners[match] = clean
    owners.sort(key=lambda item: (_owner(item), item["source_item_sha256"], item["source_id"]))
    representative = deepcopy(owners[0])
    occupied[key] = {**representative, "historical_owners": owners}


def collect_prior(current_runs: Path = HERE / "runs", *, v1_runs: Path = V1_RUNS,
                  v2_runs: Path = V2_RUNS, v4_runs: Path = V4_RUNS,
                  v5_runs: Path = V5_RUNS, v6_runs: Path = V6_RUNS,
                  v7_runs: Path = V7_RUNS, v8_runs: Path = V8_RUNS,
                  v9_runs: Path = V9_RUNS, v11_runs: Path = V11_RUNS,
                  v12_runs: Path = V12_RUNS, v15_runs: Path = V15_RUNS,
                  v90_runs: Path = V90_RUNS, v91_runs: Path = V91_RUNS,
                  v92_runs: Path = V92_RUNS,
                  v93_runs: Path = V93_RUNS,
                  v94_runs: Path = V94_RUNS, v95_runs: Path = V95_RUNS,
                  historical_archive_runs: Path = DELIVERIES) -> dict[str, Any]:
    """Return legacy-compatible usage including failed/incomplete reservations.

    Recovery reports in v2 contain delta routes; the legacy reader validates
    their evidence against the recorded source/card manifests. Historical v4,
    v5, v6, v7, v8, v9, v11, v12 v15 and current v90 runs are scanned; acceptance and reservation for the same
    slot count as one occupied source event. A versioned delivery directory
    never resets the global source inventory. Tests can isolate each generation
    by passing temporary directories for every runs argument.
    """
    prior = legacy.collect_prior_usage(legacy.prior_runs(v1_runs, v2_runs))
    occupied: dict[tuple[str, str, str], dict] = {}
    evidence = {row["path"]: row for row in prior["prior_evidence"]}
    prior_runs = set(prior["prior_runs"])
    reserved = []
    for item in prior["accepted_uses"]:
        row = _row(item, run=Path(item["run"]), root=item["root_id"], scene=item["scene_id"])
        row.update(status="accepted", origin="historical_v1_v2")
        if "card_id" in item:
            row["card_id"] = item["card_id"]
        _merge(occupied, row)
    # Resolve and deduplicate to avoid counting a reservation twice if callers
    # supply the same directory through multiple arguments (including symlinks).
    run_paths: dict[Path, str] = {}
    for directory, generation in ((Path(v4_runs), "v4"), (Path(v5_runs), "v5"),
                                  (Path(v6_runs), "v6"), (Path(v7_runs), "v7"),
                                  (Path(v8_runs), "v8"), (Path(v9_runs), "v9"),
                                  (Path(v11_runs), "v11"),
                                  (Path(v12_runs), "v12"),
                                  (Path(v15_runs), "v15"),
                                  (Path(v90_runs), "v90"),
                                  (Path(v91_runs), "v91"),
                                  (Path(v92_runs), "v92"),
                                  (Path(v93_runs), "v93"),
                                  (Path(v94_runs), "v94"),
                                  (Path(v95_runs), "v95"),
                                  (Path(current_runs), "v97")):
        if directory.exists():
            for run in directory.iterdir():
                if run.is_dir():
                    run_paths.setdefault(run.resolve(), generation)
    # Every historical reservation generation is permanently consumed. Enumerate
    # actual reservation files instead of maintaining an incomplete version list.
    archive = Path(historical_archive_runs)
    if archive.exists():
        for reservation_path in sorted(archive.rglob("SOURCE_RESERVATIONS.json")):
            run_paths.setdefault(reservation_path.parent.resolve(), "historical_archive")
    for run, generation in sorted(run_paths.items()):
        path = run / "SOURCE_RESERVATIONS.json"
        reservations = {}
        if path.is_file():
            payload = legacy.read(path)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("reservation_schema_changed:" + str(path))
            if Path(payload["run"]).resolve() != run.resolve():
                raise ValueError("reservation_run_changed:" + str(path))
            evidence[str(path.resolve())] = legacy._record(path)
            prior_runs.add(str(run.resolve()))
            slots = set()
            for item in payload["reservations"]:
                if Path(item["run"]).resolve() != run.resolve():
                    raise ValueError("reservation_run_changed:" + str(path))
                row = _row(item, run=run, root=item["root_id"], scene=item["scene_id"])
                key = event_key(row)
                if key in reservations or _owner(row) in slots:
                    raise ValueError("duplicate_event_or_slot_in_reservations")
                slots.add(_owner(row))
                row.update(status="reserved", origin=f"{generation}_reservation")
                reservations[key] = row
                reserved.append(deepcopy(row))
                _merge(occupied, row)
        elif (run / "LIVE_STARTED.json").exists():
            raise ValueError("live_run_missing_source_reservations:" + str(run))
        for report_path in (run / "REPORT.json", run / "UNIQUE_EVENT_REPORT.json"):
            if not report_path.is_file():
                continue
            report = legacy.read(report_path)
            evidence[str(report_path.resolve())] = legacy._record(report_path)
            prior_runs.add(str(run.resolve()))
            for item in report.get("accepted_uses", []):
                key = event_key(item)
                reservation = reservations.get(key)
                if reservation is None:
                    raise ValueError("accepted_event_missing_reservation:" + str(report_path))
                source = {**reservation, **item}
                row = _row(source, run=run, root=source["root_id"], scene=source["scene_id"])
                if "run" in item and Path(item["run"]).resolve() != run.resolve():
                    raise ValueError("accepted_event_run_mismatch")
                row.update(status="accepted", origin=f"{generation}_accepted")
                if "card_id" in item:
                    row["card_id"] = item["card_id"]
                _merge(occupied, row)
    return {
        "prior_runs": sorted(prior_runs),
        "prior_evidence": sorted(evidence.values(), key=lambda row: row["path"]),
        # This name is retained for legacy allocate(); it includes reservations.
        "accepted_uses": [occupied[key] for key in sorted(occupied)],
        "reserved_uses": reserved,
        "historical_conflicts": [
            {"dataset": key[0], "split": key[1], "item_id": key[2],
             "owners": deepcopy(occupied[key]["historical_owners"]),
             "reasons": [reason for reason, present in (
                 ("multiple_slots", len({_owner(row) for row in occupied[key]["historical_owners"]}) > 1),
                 ("multiple_source_hashes", len({row["source_item_sha256"] for row in occupied[key]["historical_owners"]}) > 1),
                 ("multiple_source_ids", len({row["source_id"] for row in occupied[key]["historical_owners"]}) > 1)) if present]}
            for key in sorted(occupied) if len(occupied[key]["historical_owners"]) > 1],
        "inventory_scan_scope": "all recursive historical SOURCE_RESERVATIONS.json plus explicit legacy accepted reports",
        "consumption_policy": "reservation_before_model_request_never_release",
    }


def validate_available(assignments: dict[str, list[dict]], prior: dict) -> None:
    """Check global identity, not card ID, paraphrase, or selected source span."""
    consumed = {event_key(row): row for row in prior["accepted_uses"]}
    # The placeholder run is used only to detect duplicate root/scene slots.
    rows = _assignment_rows(HERE / "_validation_only", {
        root: [{key: value for key, value in row.items() if key != "run"} for row in items]
        for root, items in assignments.items()
    })
    for row in rows:
        old = consumed.get(event_key(row))
        if old:
            if old.get("source_item_sha256") != row["source_item_sha256"]:
                raise ValueError("source_event_hash_changed")
            raise ValueError("source_event_already_consumed")


def reserve(run: Path, assignments: dict[str, list[dict]], prior: dict) -> dict[str, Any]:
    """Write one immutable reservation; reuse only its identical slot binding.

    Call under the shared inventory lock with a freshly collected prior.  A
    later failed scene or transport call does not delete or free reservations.
    """
    run = run.resolve()
    rows = _assignment_rows(run, assignments)
    payload = {"schema_version": SCHEMA_VERSION, "run": str(run),
               "consumption_policy": "reservation_before_model_request_never_release",
               "reservations": rows}
    path = run / "SOURCE_RESERVATIONS.json"
    if path.exists():
        existing = legacy.read(path)
        if existing != payload:
            raise ValueError("existing_reservations_changed")
        # Retries within this slot may use the reservation, never a second slot.
        own = {event_key(row): row for row in rows}
        for representative in prior["accepted_uses"]:
            for old in representative.get("historical_owners", [representative]):
                row = own.get(event_key(old))
                if row and (_owner(row) != _owner(old) or
                            row["source_item_sha256"] != old.get("source_item_sha256") or
                            row["source_id"] != old.get("source_id")):
                    raise ValueError("existing_reservation_conflicts_with_prior")
        return existing
    validate_available(assignments, prior)
    for item in prior.get("prior_evidence", []):
        if legacy.sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("historical_usage_evidence_changed:" + item["path"])
    run.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a mistaken second writer from replacing use.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload
