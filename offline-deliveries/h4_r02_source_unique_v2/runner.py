"""Next-version R02 adapter: globally unique dataset events, no API calls in prepare."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import unique_events


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "h4_r02_dualroute_feedback_v1"
if str(V1) not in sys.path:
    sys.path.insert(0, str(V1))
spec = importlib.util.spec_from_file_location("r02_frozen_v1_runner", V1 / "runner.py")
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

VERSION = "h4_r02_source_unique_v2"
V1_RUNS = V1 / "runs"
V2_RUNS = HERE / "runs"


def _verify_v2(run: Path) -> dict:
    manifest = base.read(run / "V2_MANIFEST.json")
    base.require(manifest["version"] == VERSION, "v2_manifest_version_changed")
    base.verify_bindings(manifest["files"])
    base.verify_bindings(base.read(run / "manifest.json")["files"])
    snapshot = base.read(run / "UNIQUE_EVENT_ALLOCATION.json")
    unique_events.verify_allocation(snapshot, base.read(run / "dataset_sources.json"), base.ROOTS)
    return snapshot


def prepare(run: Path, *, v1_runs: Path = V1_RUNS, v2_runs: Path = V2_RUNS) -> dict:
    base.require(not run.exists(), "run_already_exists")
    sources = base.dataset_cards.freeze_source_subset()
    prior = unique_events.collect_prior_usage(unique_events.prior_runs(v1_runs, v2_runs))
    snapshot = unique_events.allocate(sources, base.ROOTS, prior)
    base.prepare(run)
    base.require(base.read(run / "dataset_sources.json")["manifest_sha256"] == sources["manifest_sha256"],
                 "source_changed_during_prepare")
    base.write(run / "UNIQUE_EVENT_ALLOCATION.json", snapshot)
    paths = [HERE / "runner.py", HERE / "unique_events.py", V1 / "runner.py",
             run / "manifest.json", run / "UNIQUE_EVENT_ALLOCATION.json"]
    paths += [Path(row["path"]) for row in prior["prior_evidence"]]
    base.write(run / "V2_MANIFEST.json", {
        "version": VERSION, "v1_transport_and_dialogue_contract": base.VERSION,
        "identity_fields": ["dataset", "split", "item_id"],
        "one_source_item_across_roots_and_completed_runs": True,
        "only_accepted_prior_scenes_consume_items": True,
        "assigned_B_slots": snapshot["assigned_slots"],
        "target_B_slots": snapshot["target_slots"],
        "files": base.bindings(paths),
    })
    _verify_v2(run)
    return {"status": "PASS", "http_calls": 0,
            "assigned_B_slots": snapshot["assigned_slots"],
            "target_B_slots": snapshot["target_slots"],
            "excluded_prior_source_ids": snapshot["excluded_prior_source_ids"],
            "per_root": {root: [row["source_id"] for row in rows]
                         for root, rows in snapshot["assignments"].items()}}


def _check_new_prior_use(snapshot: dict) -> None:
    current = unique_events.collect_prior_usage(unique_events.prior_runs(V1_RUNS, V2_RUNS))
    prior = {unique_events.event_key(row) for row in snapshot["prior"]["accepted_uses"]}
    newly_used = {unique_events.event_key(row) for row in current["accepted_uses"]} - prior
    assigned = {unique_events.event_key(row) for rows in snapshot["assignments"].values() for row in rows}
    if newly_used & assigned:
        raise ValueError("allocated_event_consumed_by_newer_completed_run")


def report(run: Path, snapshot: dict) -> dict:
    original = base.report(run)
    cards = {card["card_id"]: card for card in base.read(run / "dataset_cards_result.json")["accepted_cards"]}
    accepted = []
    keys = set()
    for route in original["routes"]:
        if route["route"] != "dataset_new_event":
            continue
        root = route["root_id"]
        root_cards = [cards[scene["dataset_card_id"]] for scene in route["scene_metadata"]]
        unique_events.verify_cards_for_root(root_cards, snapshot, root)
        for scene, card in zip(route["scene_metadata"], root_cards):
            key = unique_events.event_key(card)
            if key in keys:
                raise ValueError("source_event_accepted_twice_in_run")
            keys.add(key)
            accepted.append({"root_id": root, "scene_id": scene["scene_id"],
                             "card_id": card["card_id"], "source_id": card["source_id"],
                             "dataset": key[0], "split": key[1], "item_id": key[2]})
    assigned = [row for rows in snapshot["assignments"].values() for row in rows]
    result = {
        "version": VERSION, "v1_report_sha256": unique_events.sha256(run / "REPORT.json"),
        "assignment_sha256": unique_events.sha256(run / "UNIQUE_EVENT_ALLOCATION.json"),
        "prior_accepted_uses": len(snapshot["prior"]["accepted_uses"]),
        "excluded_prior_source_ids": snapshot["excluded_prior_source_ids"],
        "assigned": snapshot["assignments"], "assigned_B_slots": len(assigned),
        "target_B_slots": snapshot["target_slots"],
        "accepted_uses": accepted, "used_source_ids": [row["source_id"] for row in accepted],
        "assigned_but_unused_source_ids": [row["source_id"] for row in assigned
                                           if unique_events.event_key(row) not in keys],
        "remaining_unassigned_source_ids": snapshot["unassigned_source_ids"],
        "global_unique_event_check": "PASS", "semantic_certification": False,
    }
    base.write(run / "UNIQUE_EVENT_REPORT.json", result)
    return result


def cards_for_root(cards: list[dict], snapshot: dict, root: str) -> list[dict]:
    root_keys = {unique_events.event_key(row) for row in snapshot["assignments"][root]}
    selected = [card for card in cards if unique_events.event_key(card) in root_keys]
    unique_events.verify_cards_for_root(selected, snapshot, root)
    return selected


def live(run: Path) -> dict:
    # Serialize v2 pilots so another completed run cannot consume this allocation mid-run.
    V2_RUNS.mkdir(parents=True, exist_ok=True)
    with (V2_RUNS / ".unique_event_live.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        snapshot = _verify_v2(run)
        _check_new_prior_use(snapshot)
        base.write(run / "LIVE_STARTED.json", {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "version": VERSION, "authorization": "USER_REQUEST_R02_SMALL_BATCH",
            "v2_manifest_sha256": unique_events.sha256(run / "V2_MANIFEST.json"),
            "no_replay_after_ambiguous_attempt": True,
        })
        caller = base.Caller(run)
        manifest = base.read(run / "dataset_sources.json")
        assigned_ids = {row["source_id"] for rows in snapshot["assignments"].values() for row in rows}
        cards, rejections = [], []
        for batch, dataset in enumerate(("allenai/soda", "MultiWOZ_2.2"), 1):
            ids = [item["source_id"] for item in manifest["items"]
                   if item["dataset"] == dataset and item["source_id"] in assigned_ids]
            if not ids:
                continue
            subset = base.dataset_cards.select_source_subset(manifest, ids)
            payload = base.dataset_cards.build_extraction_request(subset)
            proof = {"role": "dataset_extractor", "source_manifest_sha256": manifest["manifest_sha256"],
                     "source_ids": ids, "source_unique_allocation_sha256": unique_events.sha256(run / "UNIQUE_EVENT_ALLOCATION.json")}
            try:
                raw, directory = caller.call(f"dataset-{batch}", payload, proof)
                validation = base.dataset_cards.validate_extracted_cards(raw, subset)
                base.write(directory / "VALIDATION.json", validation)
                cards.extend(validation["accepted_cards"])
                rejections.extend(validation["rejected_records"])
            except RuntimeError as error:
                rejections.append({"scope": "batch", "dataset": dataset, "reason": str(error)})
        base.require(len({card["card_id"] for card in cards}) == len(cards), "cross_batch_card_id_collision")
        base.write(run / "dataset_cards_result.json", {"accepted_cards": cards, "rejected_records": rejections})
        for root in base.ROOTS:
            base.run_root(caller, run, root, cards_for_root(cards, snapshot, root))
        return report(run, snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "live"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    outcome = prepare(args.run.resolve()) if args.action == "prepare" else live(args.run.resolve())
    print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
