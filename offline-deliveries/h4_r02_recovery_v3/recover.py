"""Recover R02 by reusing only identical requests and rebuilding downstream histories."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "h4_r02_source_unique_v2"
sys.path.insert(0, str(V2))
spec = importlib.util.spec_from_file_location("recovery_unique_v2", V2 / "runner.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
base = v2.base
unique_events = v2.unique_events
REUSE_RUNS = [
    base.HERE / "runs" / "pilot_01",
    base.HERE / "runs" / "pilot_02",
    HERE.parent / "h4_r02_504_retry_v1" / "runs" / "pilot_02_retry_01",
]
# Keep reports in the v2 inventory so subsequent runs see accepted source usage.
DEFAULT_RUN = V2 / "runs" / "recovery_01"
VERSION = "h4_r02_recovery_v3"
NEW_HTTP_MAX = 22


def cache_entries():
    entries = {}
    for source in REUSE_RUNS:
        base.require((source / "REPORT.json").exists(), "reuse_run_incomplete")
        for directory in sorted((source / "requests").iterdir()):
            if not all((directory / f).exists() for f in ("RESPONSE.json", "HTTP_ATTEMPT.json")):
                continue
            payload = base.read(directory / "REQUEST.json")
            proof = base.read(directory / "PROOF.json")
            attempt = base.read(directory / "HTTP_ATTEMPT.json")
            response = base.read(directory / "RESPONSE.json")
            digest = base.digest(payload)
            base.require(proof["payload_sha256"] == attempt["payload_sha256"] == digest
                         and attempt["actual_payload_equals_frozen"] is True,
                         "unverified_reuse_request")
            base.require(response["metadata"]["returned_model"] == base.dataset_cards.MODEL,
                         "reuse_model_mismatch")
            # The old B requests belong to obsolete histories and used MultiWOZ events.
            # Exclude them outright in addition to enforcing exact payload identity.
            if proof.get("route") == "dataset_new_event":
                continue
            entries[digest] = directory
    return entries


class RecoveryCaller(base.Caller):
    def __init__(self, run, entries):
        super().__init__(run)
        self.reuse = dict(entries)

    def call(self, label, payload, proof):
        if base.digest(payload) not in self.reuse:
            base.require(self.http_counter < NEW_HTTP_MAX, "recovery_new_http_budget_exhausted")
        return super().call(label, payload, proof)


def prepare(run):
    base.require(run.parent == v2.V2_RUNS, "recovery_run_must_be_in_unique_event_inventory")
    entries = cache_entries()
    result = v2.prepare(run)
    frozen = [HERE / "recover.py"]
    frozen.extend(p for directory in entries.values() for p in directory.glob("*.json"))
    frozen.extend(source / "REPORT.json" for source in REUSE_RUNS)
    base.write(run / "RECOVERY_MANIFEST.json", {
        "version": VERSION, "reuse_runs": [str(p) for p in REUSE_RUNS],
        "cache": {key: str(value) for key, value in entries.items()},
        "files": base.bindings(frozen), "new_http_max": NEW_HTTP_MAX,
        "normal_new_http_estimate": 11, "automatic_transport_retries": 0,
        "exact_request_payload_required_for_reuse": True,
        "all_old_B_cache_entries_excluded": True,
        "target_A_scenes_per_root": 3, "available_B_source_events_per_root": 1,
        "B_receives_actual_accepted_A": True,
        "historical_outputs_unchanged": True,
    })
    return {**result, "new_http_max": NEW_HTTP_MAX, "cache_entries": len(entries)}


def verify(run):
    snapshot = v2._verify_v2(run)
    manifest = base.read(run / "RECOVERY_MANIFEST.json")
    base.require(manifest["version"] == VERSION and manifest["new_http_max"] == NEW_HTTP_MAX,
                 "recovery_contract_changed")
    base.verify_bindings(manifest["files"])
    entries = cache_entries()
    base.require({key: str(value) for key, value in entries.items()} == manifest["cache"],
                 "reuse_cache_changed")
    return snapshot, entries


def live(run):
    with (v2.V2_RUNS / ".unique_event_live.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        snapshot, entries = verify(run)
        v2._check_new_prior_use(snapshot)
        base.write(run / "LIVE_STARTED.json", {
            "version": VERSION, "started_at": datetime.now(timezone.utc).isoformat(),
            "authorization": "USER_REQUEST_RESOLVE_RECOVERED_R02_RESULTS",
            "new_http_max": NEW_HTTP_MAX, "automatic_transport_retries": 0,
        })
        caller = RecoveryCaller(run, entries)
        manifest = base.read(run / "dataset_sources.json")
        assigned_ids = {row["source_id"] for rows in snapshot["assignments"].values() for row in rows}
        cards, rejected = [], []
        for batch, dataset in enumerate(("allenai/soda", "MultiWOZ_2.2"), 1):
            ids = [row["source_id"] for row in manifest["items"]
                   if row["dataset"] == dataset and row["source_id"] in assigned_ids]
            if not ids:
                continue
            subset = base.dataset_cards.select_source_subset(manifest, ids)
            payload = base.dataset_cards.build_extraction_request(subset)
            # Dataset extraction has already succeeded; this recovery must reuse it.
            base.require(base.digest(payload) in entries, "validated_dataset_extraction_not_in_cache")
            proof = {"role": "dataset_extractor", "source_manifest_sha256": manifest["manifest_sha256"],
                     "source_ids": ids, "source_unique_allocation_sha256": unique_events.sha256(run / "UNIQUE_EVENT_ALLOCATION.json")}
            raw, directory = caller.call(f"dataset-{batch}", payload, proof)
            validation = base.dataset_cards.validate_extracted_cards(raw, subset)
            base.write(directory / "VALIDATION.json", validation)
            cards.extend(validation["accepted_cards"])
            rejected.extend(validation["rejected_records"])
        base.require(len(cards) == 3 and not rejected, "recovered_dataset_cards_invalid")
        base.require(len({card["card_id"] for card in cards}) == len(cards), "duplicate_recovered_card_id")
        base.write(run / "dataset_cards_result.json", {"accepted_cards": cards, "rejected_records": rejected})
        for root in base.ROOTS:
            assigned_cards = v2.cards_for_root(cards, snapshot, root)
            base.require(len(assigned_cards) == 1, "expected_one_unique_B_source_per_root")
            base.run_root(caller, run, root, assigned_cards)
        # Verify all source assignments before base.report writes its completed marker.
        accepted_keys = set()
        for root in base.ROOTS:
            result = base.read(run / "roots" / root / "B" / "RESULT.json")
            lookup = {card["card_id"]: card for card in cards}
            accepted_cards = [lookup[row["dataset_card_id"]] for row in result["scene_metadata"]]
            unique_events.verify_cards_for_root(accepted_cards, snapshot, root)
            for card in accepted_cards:
                key = unique_events.event_key(card)
                base.require(key not in accepted_keys, "source_event_accepted_twice")
                accepted_keys.add(key)
        unique_report = v2.report(run, snapshot)
        report = base.read(run / "REPORT.json")
        base.write(run / "RECOVERY_REPORT.json", {
            "version": VERSION, "http_attempts": report["http_attempts"],
            "reused_responses": report["reused_responses"],
            "accepted_scenes": report["accepted_scenes"], "added_turns": report["added_turns"],
            "feasible_target_scenes": 12, "new_http_max": NEW_HTTP_MAX,
            "global_unique_event_check": unique_report["global_unique_event_check"],
            "all_B_handoffs_verified": report["all_B_handoffs_verified"],
            "per_root": {root: {"A": next(r["accepted_scenes"] for r in report["routes"]
                if r["root_id"] == root and r["route"] == base.ROUTES[0]),
                "B": next(r["accepted_scenes"] for r in report["routes"]
                if r["root_id"] == root and r["route"] == base.ROUTES[1])} for root in base.ROOTS},
            "semantic_certification": False, "question_generation_calls": 0,
            "solver_calls": 0, "training_calls": 0,
        })
        return base.read(run / "RECOVERY_REPORT.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "live"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    result = prepare(run) if args.action == "prepare" else live(run) if args.action == "live" else {"status": "PASS", "cached_requests": len(verify(run)[1])}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
