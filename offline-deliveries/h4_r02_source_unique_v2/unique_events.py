"""Source-item identity, historical usage, and deterministic R02 allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "h4.r02.source_unique.v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def event_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """Span, paraphrase and card ID cannot make one source event new again."""
    if "lineage" in item:
        item = item["lineage"]
    key = tuple(item[field] for field in ("dataset", "split", "item_id"))
    if any(not isinstance(part, str) or not part for part in key):
        raise ValueError("invalid_source_event_identity")
    return key


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError("missing_historical_evidence:" + str(path))
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def prior_runs(v1_runs: Path, v2_runs: Path) -> list[Path]:
    runs: list[Path] = []
    for base in (v1_runs, v2_runs):
        if base.exists():
            for path in base.iterdir():
                if not path.is_dir():
                    continue
                if (path / "LIVE_STARTED.json").is_file() and not (path / "REPORT.json").is_file():
                    raise ValueError("incomplete_prior_live_run_usage_unknown:" + str(path))
                if (path / "REPORT.json").is_file():
                    runs.append(path)
    return sorted(set(runs))


def collect_prior_usage(runs: list[Path]) -> dict[str, Any]:
    """Only accepted B scenes consume a source event; plans and failed scenes do not."""
    evidence: list[dict[str, str]] = []
    used: list[dict[str, Any]] = []
    for run in runs:
        report = run / "REPORT.json"
        cards_path = run / "dataset_cards_result.json"
        source_path = run / "dataset_sources.json"
        evidence.extend(_record(p) for p in (report, cards_path, source_path))
        cards = {card["card_id"]: card for card in read(cards_path)["accepted_cards"]}
        sources = {item["source_id"]: item for item in read(source_path)["items"]}
        routes = [route for route in read(report)["routes"] if route["route"] == "dataset_new_event"]
        for route in routes:
            result = run / "roots" / route["root_id"] / "B" / "RESULT.json"
            evidence.append(_record(result))
            actual = read(result)
            if actual != route:
                raise ValueError("historical_report_result_disagree:" + str(result))
            scenes = actual["scene_metadata"]
            if len(scenes) != actual["accepted_scenes"]:
                raise ValueError("historical_scene_count_disagree:" + str(result))
            for scene in scenes:
                card_id = scene["dataset_card_id"]
                card = cards.get(card_id)
                if card is None:
                    raise ValueError("historical_accepted_card_missing:" + str(result))
                source = sources.get(card["source_id"])
                if source is None or event_key(source) != event_key(card):
                    raise ValueError("historical_card_source_mismatch:" + str(result))
                if card["lineage"]["source_item_sha256"] != source["source_item_sha256"]:
                    raise ValueError("historical_card_source_hash_mismatch:" + str(result))
                used.append({
                    "dataset": event_key(card)[0], "split": event_key(card)[1],
                    "item_id": event_key(card)[2], "source_id": card["source_id"],
                    "source_item_sha256": source["source_item_sha256"],
                    "run": str(run.resolve()), "root_id": route["root_id"],
                    "scene_id": scene["scene_id"], "card_id": card_id,
                })
    return {"prior_runs": [str(run.resolve()) for run in runs],
            "prior_evidence": sorted(evidence, key=lambda row: row["path"]),
            "accepted_uses": used}


def allocate(manifest: dict[str, Any], roots: list[str], prior: dict[str, Any], slots_per_root: int = 3) -> dict[str, Any]:
    if not roots or len(roots) != len(set(roots)) or slots_per_root < 1:
        raise ValueError("invalid_root_allocation_request")
    seen: set[tuple[str, str, str]] = set()
    unique_items = []
    for item in manifest["items"]:
        key = event_key(item)
        if key in seen:
            raise ValueError("duplicate_source_event_in_manifest")
        seen.add(key)
        unique_items.append(item)
    consumed = {event_key(row) for row in prior["accepted_uses"]}
    available = [item for item in unique_items if event_key(item) not in consumed]
    assignments: dict[str, list[dict[str, Any]]] = {root: [] for root in roots}
    # Round robin prevents a small remaining pool from being monopolized by root 1.
    for item in available:
        eligible = [root for root in roots if len(assignments[root]) < slots_per_root]
        if not eligible:
            break
        root = min(eligible, key=lambda value: (len(assignments[value]), roots.index(value)))
        assignments[root].append({
            "dataset": item["dataset"], "split": item["split"], "item_id": item["item_id"],
            "source_id": item["source_id"], "source_file_sha256": item["source_file_sha256"],
            "source_item_sha256": item["source_item_sha256"],
        })
    assigned = {event_key(item) for rows in assignments.values() for item in rows}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "prior": prior,
        "roots": roots,
        "slots_per_root": slots_per_root,
        "assignments": assignments,
        "excluded_prior_source_ids": [item["source_id"] for item in unique_items if event_key(item) in consumed],
        "unassigned_source_ids": [item["source_id"] for item in available if event_key(item) not in assigned],
        "assigned_slots": len(assigned),
        "target_slots": slots_per_root * len(roots),
    }


def verify_allocation(snapshot: dict[str, Any], manifest: dict[str, Any], roots: list[str]) -> None:
    if snapshot["source_manifest_sha256"] != manifest["manifest_sha256"] or snapshot["roots"] != roots:
        raise ValueError("allocation_source_or_roots_changed")
    expected = allocate(manifest, roots, snapshot["prior"], snapshot["slots_per_root"])
    if expected != snapshot:
        raise ValueError("allocation_changed")
    for evidence in snapshot["prior"]["prior_evidence"]:
        if sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("historical_usage_evidence_changed:" + evidence["path"])


def verify_cards_for_root(cards: list[dict[str, Any]], snapshot: dict[str, Any], root: str) -> None:
    allowed = {event_key(row): row for row in snapshot["assignments"][root]}
    seen: set[tuple[str, str, str]] = set()
    for card in cards:
        key = event_key(card)
        row = allowed.get(key)
        if row is None or row["source_id"] != card["source_id"]:
            raise ValueError("card_source_not_allocated_to_root")
        if row["source_item_sha256"] != card["lineage"]["source_item_sha256"]:
            raise ValueError("allocated_card_source_hash_changed")
        if key in seen:
            raise ValueError("duplicate_event_card_for_root")
        seen.add(key)
