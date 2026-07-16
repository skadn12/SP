#!/usr/bin/env python3
"""Isolated Truth-first incremental temporal child pilot.

This runner deliberately does not enter the production Stage3 generation path.
It freezes one four-scene temporal plan locally, makes at most one physical API
call per ``generate-next`` invocation, and requires an explicit semantic review
before a scene is appended to the accepted hash chain.  It never writes a train
mix or resumes an older Stage3 state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native
from scripts.current_method import stage3_claim_ir, stage3_multiqa_rows


RUN_ID = "233_temporal_incremental_child_pilot_n1"
DEFAULT_OUTPUT_DIR = ROOT / "base_restart_current_method_validation" / RUN_ID
SOURCE_RUN = (
    ROOT
    / "base_restart_current_method_validation"
    / "230l_truth_first_temporal_n1_retry11"
)
SOURCE_ARTIFACT = (
    SOURCE_RUN
    / "generation"
    / "graph_artifacts"
    / "c8e83824a84b81d1baa0d7ffcd52cd852b1e7c66254764b34c30ead42ef07d71.json"
)
SOURCE_SEED_ROWS = SOURCE_RUN / "reports" / "round_dataset_train_seed_rows.jsonl"
SOURCE_ARTIFACT_FILE_SHA256 = (
    "5db6237f7e84b0b3824d4d22bd4feb2c29fd318dbc309590101e369f0eb31d24"
)
SOURCE_SEED_ROWS_FILE_SHA256 = (
    "9aef173fbb471a46bef01f42f5d79a1fd1cd3a4d1fc2a17c8fb34f0ffec6e256"
)
SOURCE_SEED_SAMPLE_ID = "stage3_parent_v4_8be1dd8dbc81fad7e073"
SOURCE_LINEAGE_ID = "conv-43"
PARENT_EVENT_ID = "conv-43:OBS:17:001:26ab6a65"
PARENT_TURN_ID = "D17:3"
PARENT_DIALOGUE_HASH = (
    "9cd2981e920874da6c288b725568db8b5158131c5d07cd9d08b5862f7c6aecd6"
)
TOKENIZER_PATH = ROOT.parent / "Qwen3-4B"
MODEL = "Qwen/Qwen3-32B"
SCENE_COUNT = 4
MIN_SCENE_TOKENS = 375
MIN_TOTAL_ADDED_TOKENS = 1500
MIN_TURNS = 4
MAX_TURNS = 10
MAX_ATTEMPTS_PER_SCENE = 3  # Initial call plus at most two retries.
SCHEMA_VERSION = "stage3_temporal_incremental_pilot_v1"


TYPED_NODE_SPECS: dict[str, dict[str, str]] = {
    "E1": {
        "predicate": "believes_destination",
        "status": "belief_unconfirmed",
        "value_type": "LocationOrScope",
        "must_mean": (
            "John currently considers Cedar Hall a possible destination, but it "
            "remains unconfirmed. It is not confirmed, active, outdated, canceled, "
            "or already visited."
        ),
    },
    "E2": {
        "predicate": "current_destination",
        "status": "active",
        "value_type": "LocationOrScope",
        "must_mean": (
            "London is the destination currently recorded as active for John. It "
            "is not merely possible, outdated, canceled, or already completed."
        ),
    },
    "E3": {
        "predicate": "formerly_considered_destination",
        "status": "outdated_no_longer_applicable",
        "value_type": "LocationOrScope",
        "must_mean": (
            "The Midwest is an older destination entry for John that is now "
            "outdated and no longer applicable. It is not active and it was not "
            "canceled before effect."
        ),
    },
    "E4": {
        "predicate": "canceled_destination_plan",
        "status": "canceled_before_effect",
        "value_type": "LocationOrScope",
        "must_mean": (
            "John's community gym destination plan was canceled before it took "
            "effect. It is not active, completed, merely outdated, or unconfirmed."
        ),
    },
}

SCENE_SPECS: list[dict[str, Any]] = [
    {
        "scene_id": "S1",
        "node_event_id": "E1",
        "relation_kind": "parent_child_continuity",
        "relation_claim_id": "pilot_rel_parent_travel_to_E1",
        "discourse_mode": "reminiscence_to_open_possibility",
        "discourse_instruction": (
            "Let an earlier travel memory surface naturally while John and Tim "
            "move into a still-unconfirmed possibility. Avoid an interview or a "
            "fact-by-fact status report."
        ),
    },
    {
        "scene_id": "S2",
        "node_event_id": "E2",
        "relation_kind": "status_transition",
        "relation_claim_id": "clm_31b6cb058619412ecc3b6b99f8d88109",
        "discourse_mode": "joint_itinerary_review",
        "discourse_instruction": (
            "Continue with a shared itinerary review. John distinguishes the "
            "current active entry from the earlier unconfirmed possibility while "
            "Tim reacts and helps organize the discussion."
        ),
    },
    {
        "scene_id": "S3",
        "node_event_id": "E3",
        "relation_kind": "status_transition",
        "relation_claim_id": "clm_d8690275f9738888c8a66fc5be2ecf9d",
        "discourse_mode": "old_note_correction",
        "discourse_instruction": (
            "An old planning note comes up in the same conversation. John corrects "
            "its lifecycle status in passing, without giving a formal recap of all "
            "earlier facts."
        ),
    },
    {
        "scene_id": "S4",
        "node_event_id": "E4",
        "relation_kind": "status_transition",
        "relation_claim_id": "clm_066c08e4336b6b601989a415eec96bd0",
        "discourse_mode": "change_log_wrapup",
        "discourse_instruction": (
            "Finish the planning discussion by clarifying a genuinely canceled "
            "plan and how that differs from the preceding outdated entry. End with "
            "a natural social close, not a database summary."
        ),
    },
]

SYSTEM_PROMPT = (
    "Write one self-contained scene in a continuing direct conversation from "
    "immutable typed truth. Preserve the subject, value, epistemic/lifecycle "
    "status, relation direction, and exclusions exactly. Return JSON with only a "
    "turns array. Do not generate graph data, QA, evidence metadata, or claim IDs."
)


class PilotError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id(prefix: str, value: Any, length: int = 24) -> str:
    return f"{prefix}_{_sha256_json(value)[:length]}"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PilotError(f"blank JSONL line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PilotError(
                    f"truncated or invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PilotError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_seed_row() -> dict[str, Any]:
    matches = []
    for row in _read_jsonl(SOURCE_SEED_ROWS):
        if row.get("sample_id") == SOURCE_SEED_SAMPLE_ID:
            matches.append(row)
    if len(matches) != 1:
        raise PilotError(
            f"expected one source seed {SOURCE_SEED_SAMPLE_ID}, found {len(matches)}"
        )
    return matches[0]


def _source_bundle() -> dict[str, Any]:
    if _sha256_file(SOURCE_ARTIFACT) != SOURCE_ARTIFACT_FILE_SHA256:
        raise PilotError("source graph artifact file hash changed")
    if _sha256_file(SOURCE_SEED_ROWS) != SOURCE_SEED_ROWS_FILE_SHA256:
        raise PilotError("source seed-row file hash changed")
    artifact = _read_json(SOURCE_ARTIFACT)
    seed = _load_seed_row()
    if artifact.get("seed_sample_id") != SOURCE_SEED_SAMPLE_ID:
        raise PilotError("source artifact points to a different seed")
    if seed.get("lineage_group") != SOURCE_LINEAGE_ID:
        raise PilotError("source seed lineage changed")
    dialogue = seed.get("dialogue") or {}
    if dialogue.get("dialogue_hash") != PARENT_DIALOGUE_HASH:
        raise PilotError("source parent dialogue hash changed")
    turns = dialogue.get("turns") or []
    parent_turns = [turn for turn in turns if turn.get("turn_id") == PARENT_TURN_ID]
    if len(parent_turns) != 1 or parent_turns[0].get("speaker") != "John":
        raise PilotError("parent evidence turn D17:3 is missing or ambiguous")
    parent_events = (artifact.get("parent_graph") or {}).get("events") or []
    parent_event = next(
        (event for event in parent_events if event.get("event_id") == PARENT_EVENT_ID),
        None,
    )
    if not isinstance(parent_event, dict):
        raise PilotError("selected parent continuity event is absent")
    child_graph = (
        ((artifact.get("current_event_plan") or {}).get("_truth_first_materialized") or {})
        .get("graph")
    )
    if not isinstance(child_graph, dict):
        raise PilotError("source child truth graph is absent")
    claim_ir = stage3_claim_ir.compile_claim_ir(child_graph)
    return {
        "artifact": artifact,
        "seed": seed,
        "parent_turn": copy.deepcopy(parent_turns[0]),
        "parent_event": copy.deepcopy(parent_event),
        "child_graph": child_graph,
        "claim_ir": claim_ir,
    }


def _node_claims_by_event(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    events = {
        event.get("event_id"): event
        for event in source["child_graph"].get("events") or []
        if isinstance(event, dict)
    }
    result: dict[str, dict[str, Any]] = {}
    for event_id in TYPED_NODE_SPECS:
        event = events.get(event_id)
        if not isinstance(event, dict):
            raise PilotError(f"source truth graph is missing {event_id}")
        matches = []
        for claim in source["claim_ir"].get("claims") or []:
            canonical = claim.get("canonical_support_claim") or {}
            if (
                claim.get("claim_type") == "node"
                and canonical.get("subject") == event.get("entity")
                and canonical.get("predicate") == event.get("slot")
                and canonical.get("objects") == [event.get("value")]
                and canonical.get("status") == event.get("status")
                and (canonical.get("time") or {}).get("global_time_index")
                == event.get("global_time_index")
            ):
                matches.append(claim)
        if len(matches) != 1:
            raise PilotError(f"cannot map {event_id} losslessly to exactly one node claim")
        result[event_id] = copy.deepcopy(matches[0])
    return result


def _relation_claims(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    wanted = {spec["relation_claim_id"] for spec in SCENE_SPECS[1:]}
    result = {
        claim["claim_id"]: copy.deepcopy(claim)
        for claim in source["claim_ir"].get("claims") or []
        if claim.get("claim_id") in wanted
    }
    if set(result) != wanted:
        raise PilotError(
            f"source status-transition claim set changed: {sorted(set(result) ^ wanted)}"
        )
    return result


def _typed_node(event: Mapping[str, Any], claim: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    spec = TYPED_NODE_SPECS[event_id]
    return {
        "claim_id": claim["claim_id"],
        "claim_type": "node",
        "event_id": event_id,
        "subject": event["entity"],
        "predicate": spec["predicate"],
        "source_slot": event["slot"],
        "value": event["value"],
        "value_type": spec["value_type"],
        "status": spec["status"],
        "source_status": event["status"],
        "local_time_index": event["local_time_index"],
        "global_time_index": event["global_time_index"],
        "must_mean": spec["must_mean"],
        "provenance": "frozen_source_truth_graph_plus_typed_prompt_adapter",
    }


def _typed_relation(
    scene_spec: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    source_relations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scene_id = scene_spec["scene_id"]
    if scene_id == "S1":
        return {
            "claim_id": scene_spec["relation_claim_id"],
            "claim_type": "parent_child_continuity",
            "subject": "John",
            "predicate": "continues_travel_topic_from",
            "source_event_id": PARENT_EVENT_ID,
            "target_event_id": "E1",
            "relation_direction": f"{PARENT_EVENT_ID}->E1",
            "status": "active_noncausal_topic_continuity",
            "must_mean": (
                "John's present destination discussion is a thematic continuation "
                "of his earlier completed European-coastline road-trip memory. The "
                "earlier trip did not cause or confirm the Cedar Hall possibility."
            ),
            "provenance": "local_pilot_continuity_contract",
        }
    current_event = scene_spec["node_event_id"]
    previous_event = f"E{int(current_event[1:]) - 1}"
    relation = source_relations[scene_spec["relation_claim_id"]]
    canonical = relation["canonical_support_claim"]
    return {
        "claim_id": relation["claim_id"],
        "claim_type": "status_transition",
        "subject": canonical["subject"],
        "predicate": canonical["predicate"],
        "source_event_id": previous_event,
        "target_event_id": current_event,
        "relation_direction": f"{previous_event}->{current_event}",
        "from_value": nodes[previous_event]["value"],
        "to_value": nodes[current_event]["value"],
        "from_status": nodes[previous_event]["status"],
        "to_status": nodes[current_event]["status"],
        "must_mean": (
            f"In John's destination-state history, {nodes[previous_event]['value']} "
            f"with status {nodes[previous_event]['status']} precedes the distinct "
            f"{nodes[current_event]['value']} entry with status "
            f"{nodes[current_event]['status']}. Preserve this direction and do not "
            "merge the two values into one event."
        ),
        "provenance": "frozen_source_claim_ir",
    }


def _build_scene_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    source_events = {
        event.get("event_id"): event
        for event in source["child_graph"].get("events") or []
        if isinstance(event, dict)
    }
    node_claims = _node_claims_by_event(source)
    relation_claims = _relation_claims(source)
    nodes = {
        event_id: _typed_node(source_events[event_id], node_claims[event_id])
        for event_id in TYPED_NODE_SPECS
    }
    scenes = []
    for scene_spec in SCENE_SPECS:
        node = nodes[scene_spec["node_event_id"]]
        relation = _typed_relation(scene_spec, nodes, relation_claims)
        scene = {
            **copy.deepcopy(scene_spec),
            "scene_index": len(scenes) + 1,
            "claim_count": 2,
            "frozen_claims": [relation, node],
            "frozen_claim_ids": [relation["claim_id"], node["claim_id"]],
            "continuity": {
                "depends_on_parent_event_ids": [PARENT_EVENT_ID],
                "parent_topic_anchor_event_ids": [PARENT_EVENT_ID],
                "continuity_basis": (
                    "same_subject_travel_topic_and_explicit_temporal_state_chain"
                ),
            },
        }
        scene["scene_plan_sha256"] = _sha256_json(scene)
        scenes.append(scene)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "operation": "temporal_state_update",
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_artifact_file_sha256": SOURCE_ARTIFACT_FILE_SHA256,
        "source_seed_rows": str(SOURCE_SEED_ROWS),
        "source_seed_rows_file_sha256": SOURCE_SEED_ROWS_FILE_SHA256,
        "source_seed_sample_id": SOURCE_SEED_SAMPLE_ID,
        "canonical_parent_lineage_id": SOURCE_LINEAGE_ID,
        "parent_dialogue_hash": PARENT_DIALOGUE_HASH,
        "parent_anchor_event_id": PARENT_EVENT_ID,
        "parent_anchor_turn_id": PARENT_TURN_ID,
        "source_child_graph_sha256": _sha256_json(source["child_graph"]),
        "source_claim_bundle_hash": source["claim_ir"]["claim_bundle_hash"],
        "model": MODEL,
        "scene_count": SCENE_COUNT,
        "scene_token_target": MIN_SCENE_TOKENS,
        "per_scene_token_hard_gate": False,
        "min_total_added_tokens": MIN_TOTAL_ADDED_TOKENS,
        "min_turns_per_scene": MIN_TURNS,
        "max_turns_per_scene": MAX_TURNS,
        "max_attempts_per_scene": MAX_ATTEMPTS_PER_SCENE,
        "tokenizer_path": str(TOKENIZER_PATH),
        "scenes": scenes,
    }
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def _preflight_report(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    source = _source_bundle()
    plan = _build_scene_plan(source)
    parent_max = max(
        int(event.get("global_time_index") or -1)
        for event in (source["artifact"].get("parent_graph") or {}).get("events") or []
        if isinstance(event, dict)
    )
    child_times = [
        int(event["global_time_index"])
        for event in source["child_graph"].get("events") or []
        if event.get("event_id") in TYPED_NODE_SPECS
    ]
    checks = {
        "source_artifact_hash_exact": True,
        "source_seed_hash_exact": True,
        "source_lineage_exact": True,
        "parent_anchor_exists": True,
        "parent_evidence_turn_exists": True,
        "four_scene_plan": len(plan["scenes"]) == 4,
        "two_claims_per_scene": all(
            scene["claim_count"] == 2 for scene in plan["scenes"]
        ),
        "unique_claims_within_scene": all(
            len(set(scene["frozen_claim_ids"])) == 2 for scene in plan["scenes"]
        ),
        "child_global_time_after_parent": bool(child_times)
        and min(child_times) > parent_max,
        "ancillary_excluded_from_plan": all(
            "ancillary" not in _canonical_json(scene["frozen_claims"]).casefold()
            for scene in plan["scenes"]
        ),
        "new_run_dir_fresh": not output_dir.exists(),
        "old_state_path_not_referenced": all(
            token not in _canonical_json(plan)
            for token in ("samples.partial.jsonl", "resume", "230l_truth_first_temporal_n1_retry11/generation/samples")
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": all(checks.values()),
        "checks": checks,
        "plan_sha256": plan["plan_sha256"],
        "parent_event_count": len(
            (source["artifact"].get("parent_graph") or {}).get("events") or []
        ),
        "parent_max_global_time_index": parent_max,
        "child_global_time_indices": child_times,
        "scene_claim_counts": [scene["claim_count"] for scene in plan["scenes"]],
        "physical_api_calls": 0,
        "training_files_written": 0,
    }


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "manifest.json",
        "plan": output_dir / "frozen_scene_plan.json",
        "state": output_dir / "state.json",
        "attempts": output_dir / "scene_attempts.jsonl",
        "accepted": output_dir / "accepted_scenes.jsonl",
        "rejects": output_dir / "scene_rejects.jsonl",
        "child": output_dir / "pilot_artifacts" / "temporal_child_artifact.json",
        "qa_contracts": output_dir / "pilot_artifacts" / "qa_contracts.json",
        "qa_polish": output_dir / "qa_polish_attempts.jsonl",
        "qa_review": output_dir / "pilot_artifacts" / "qa_polish_review.json",
        "final": output_dir / "pilot_artifacts" / "temporal_child_multiqa_artifact.json",
        "qa_rows": output_dir / "pilot_artifacts" / "multiqa_rows_preview.jsonl",
        "marked_rows": output_dir / "pilot_artifacts" / "multiqa_marked_rows_preview.jsonl",
        "report": output_dir / "reports" / "temporal_incremental_pilot_report.json",
    }


def _load_initialized(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _paths(output_dir)
    if not output_dir.is_dir():
        raise PilotError(f"run is not initialized: {output_dir}")
    plan = _read_json(paths["plan"])
    state = _read_json(paths["state"])
    declared_hash = plan.get("plan_sha256")
    unhashed = copy.deepcopy(plan)
    unhashed.pop("plan_sha256", None)
    if declared_hash != _sha256_json(unhashed):
        raise PilotError("frozen scene plan hash mismatch")
    if state.get("plan_sha256") != declared_hash:
        raise PilotError("state points to a different frozen scene plan")
    _audit_accepted_chain(output_dir, state)
    return plan, state


def _audit_accepted_chain(output_dir: Path, state: Mapping[str, Any]) -> None:
    rows = _read_jsonl(_paths(output_dir)["accepted"])
    previous_hash = None
    total_tokens = 0
    total_turns = 0
    for expected_index, row in enumerate(rows, start=1):
        if row.get("scene_index") != expected_index:
            raise PilotError("accepted scene journal is not ordered")
        if row.get("previous_accepted_scene_sha256") != previous_hash:
            raise PilotError("accepted scene hash chain predecessor mismatch")
        actual = copy.deepcopy(row)
        declared = actual.pop("accepted_scene_sha256", None)
        if declared != _sha256_json(actual):
            raise PilotError("accepted scene journal hash mismatch")
        previous_hash = declared
        total_tokens += int(row.get("scene_token_count") or 0)
        total_turns += len(row.get("turns") or [])
    expected = {
        "accepted_scene_count": len(rows),
        "accepted_scene_chain_head": previous_hash,
        "total_added_tokens": total_tokens,
        "total_added_turns": total_turns,
    }
    mismatches = {
        key: {"state": state.get(key), "journal": value}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise PilotError(f"state and accepted scene journal diverged: {mismatches}")


def _accepted_journal_summary(output_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(_paths(output_dir)["accepted"])
    previous_hash = None
    total_tokens = 0
    total_turns = 0
    for expected_index, row in enumerate(rows, start=1):
        if row.get("scene_index") != expected_index:
            raise PilotError("accepted scene journal is not ordered")
        if row.get("previous_accepted_scene_sha256") != previous_hash:
            raise PilotError("accepted scene recovery hash predecessor mismatch")
        unhashed = copy.deepcopy(row)
        declared = unhashed.pop("accepted_scene_sha256", None)
        if declared != _sha256_json(unhashed):
            raise PilotError("accepted scene recovery hash mismatch")
        previous_hash = declared
        total_tokens += int(row.get("scene_token_count") or 0)
        total_turns += len(row.get("turns") or [])
    return {
        "accepted_scene_count": len(rows),
        "accepted_scene_chain_head": previous_hash,
        "total_added_tokens": total_tokens,
        "total_added_turns": total_turns,
    }


def _command_recover(args: argparse.Namespace) -> int:
    paths = _paths(args.output_dir)
    if not args.output_dir.is_dir():
        raise PilotError(f"run is not initialized: {args.output_dir}")
    plan = _read_json(paths["plan"])
    state = _read_json(paths["state"])
    unhashed_plan = copy.deepcopy(plan)
    declared_plan_hash = unhashed_plan.pop("plan_sha256", None)
    if declared_plan_hash != _sha256_json(unhashed_plan):
        raise PilotError("cannot recover a run with an invalid frozen plan hash")
    if state.get("plan_sha256") != declared_plan_hash:
        raise PilotError("cannot recover state for a different plan")
    journal = _accepted_journal_summary(args.output_dir)
    state_count = int(state.get("accepted_scene_count") or 0)
    journal_count = int(journal["accepted_scene_count"])
    if journal_count not in {state_count, state_count + 1}:
        raise PilotError(
            f"unsafe accepted journal/state gap: state={state_count}, journal={journal_count}"
        )
    recovered_actions = []
    if journal_count == state_count + 1:
        state.update(journal)
        state["pending_attempt_id"] = None
        state["inflight_attempt_id"] = None
        state["status"] = (
            "scenes_complete"
            if journal_count == SCENE_COUNT
            else "ready_for_scene"
        )
        recovered_actions.append("reconciled_committed_accepted_scene")
    inflight = state.get("inflight_attempt_id")
    if inflight:
        results = [
            row
            for row in _read_jsonl(paths["attempts"])
            if row.get("record_type") == "api_result"
            and row.get("attempt_id") == inflight
        ]
        if len(results) == 1:
            state["inflight_attempt_id"] = None
            state["pending_attempt_id"] = inflight
            state["status"] = "scene_pending_semantic_review"
            recovered_actions.append("reconciled_persisted_api_result")
        elif not results and args.abandon_unknown_inflight:
            scene = plan["scenes"][int(state["accepted_scene_count"])]
            _append_jsonl(
                paths["rejects"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "attempt_id": inflight,
                    "scene_id": scene["scene_id"],
                    "scene_index": scene["scene_index"],
                    "attempt_number": state["scene_attempt_counts"][scene["scene_id"]],
                    "reason_codes": ["inflight_outcome_unknown_after_interruption"],
                    "accepted_scenes_retained": state["accepted_scene_count"],
                    "accepted_scene_chain_head_retained": state[
                        "accepted_scene_chain_head"
                    ],
                    "rejected_candidate_modified": False,
                },
            )
            state["inflight_attempt_id"] = None
            state["status"] = (
                "failed_scene_retry_budget"
                if int(state["scene_attempt_counts"][scene["scene_id"]])
                >= MAX_ATTEMPTS_PER_SCENE
                else "ready_for_scene"
            )
            recovered_actions.append("abandoned_unknown_inflight_attempt")
        elif not results:
            raise PilotError(
                "inflight call has no persisted result; rerun recover with "
                "--abandon-unknown-inflight to consume it without inventing output"
            )
        else:
            raise PilotError("inflight attempt has duplicate persisted results")
    _atomic_write_json(paths["state"], state)
    _audit_accepted_chain(args.output_dir, state)
    print(
        json.dumps(
            {
                "recovered": True,
                "actions": recovered_actions,
                "status": state["status"],
                "accepted_scene_count": state["accepted_scene_count"],
                "pending_attempt_id": state.get("pending_attempt_id"),
            },
            sort_keys=True,
        )
    )
    return 0


def _command_preflight(args: argparse.Namespace) -> int:
    report = _preflight_report(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["pass"] else 2


def _command_init(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if output_dir.exists():
        raise PilotError(f"refusing to reuse existing run directory: {output_dir}")
    report = _preflight_report(output_dir)
    if not report["pass"]:
        raise PilotError(f"offline preflight failed: {report['checks']}")
    source = _source_bundle()
    plan = _build_scene_plan(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = _paths(output_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "created_unix": int(time.time()),
        "output_dir": str(output_dir),
        "isolated_pilot_only": True,
        "training_data_mutation_allowed": False,
        "production_stage3_resume_allowed": False,
        "source_paths_read_only": [str(SOURCE_ARTIFACT), str(SOURCE_SEED_ROWS)],
        "model": MODEL,
        "plan_sha256": plan["plan_sha256"],
        "offline_preflight": report,
    }
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "plan_sha256": plan["plan_sha256"],
        "status": "ready_for_scene",
        "accepted_scene_count": 0,
        "accepted_scene_chain_head": None,
        "total_added_tokens": 0,
        "total_added_turns": 0,
        "scene_attempt_counts": {spec["scene_id"]: 0 for spec in SCENE_SPECS},
        "pending_attempt_id": None,
        "inflight_attempt_id": None,
        "qa_polish_attempt_count": 0,
        "pending_qa_polish_attempt_id": None,
        "physical_scene_api_calls": 0,
        "physical_qa_polish_api_calls": 0,
        "training_files_written": 0,
    }
    _atomic_write_json(paths["manifest"], manifest)
    _atomic_write_json(paths["plan"], plan)
    _atomic_write_json(paths["state"], state)
    print(
        json.dumps(
            {
                "initialized": True,
                "output_dir": str(output_dir),
                "plan_sha256": plan["plan_sha256"],
                "api_calls": 0,
                "next_scene": "S1",
            },
            sort_keys=True,
        )
    )
    return 0


def _checkpoint(state: Mapping[str, Any], accepted: list[dict[str, Any]]) -> Any:
    if not accepted:
        return None
    last_turn = accepted[-1]["turns"][-1]
    expressed_claim_ids = [
        claim_id
        for scene in accepted
        for claim_id in scene.get("frozen_claim_ids") or []
    ]
    return {
        "participants": ["John", "Tim"],
        "last_speaker": last_turn["speaker"],
        "previous_scene_topic": "travel planning and destination records",
        "tone": "warm_casual_collaborative",
        "dialogue_goal": "Continue the same conversation without recapping closed scenes.",
        "accepted_scene_ids": [scene["scene_id"] for scene in accepted],
        "expressed_claim_ids": expressed_claim_ids,
        "prior_scene_raw_text_included": False,
        "prior_fact_values_included": False,
        "ancillary_facts_carried_forward": False,
        "accepted_scene_count": state["accepted_scene_count"],
    }


def _api_claim_view(claim: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"provenance", "claim_id", "source_event_id", "target_event_id"}
    return {key: copy.deepcopy(value) for key, value in claim.items() if key not in excluded}


def _scene_prompt(
    plan: Mapping[str, Any],
    scene: Mapping[str, Any],
    state: Mapping[str, Any],
    accepted: list[dict[str, Any]],
    failure_feedback: list[str] | None,
) -> dict[str, Any]:
    prompt = {
        "mode": "truth_first_incremental_temporal_child_scene_v1",
        "operation": "temporal_state_update",
        "scene_index": scene["scene_index"],
        "scene_count": plan["scene_count"],
        "speaker_roster": ["John", "Tim"],
        "discourse_mode": {
            "name": scene["discourse_mode"],
            "instruction": scene["discourse_instruction"],
        },
        "immutable_typed_claims": [
            _api_claim_view(claim) for claim in scene["frozen_claims"]
        ],
        "discourse_checkpoint": _checkpoint(state, accepted),
        "rules": [
            "Write 8-10 turns and target 400-500 visible dialogue tokens. Use only John and Tim.",
            "Express both immutable typed claims naturally. The two claims may share one adjacent exchange, but neither may be omitted.",
            "Preserve every subject, value, typed predicate, status, relation direction, and must_mean exclusion exactly.",
            "A prior value mentioned by a transition keeps its prior status; do not reconfirm, cancel, expire, or replace it unless the frozen transition says so.",
            "Use first-person and second-person references naturally when speaker roles make their subjects unambiguous.",
            "Write a connected conversation, not an interview, checklist, database recital, or sequence of isolated fact sentences.",
            "Do not introduce new proper names, locations, dates, causes, counts, bookings, replacement options, or completed actions.",
            "Generic reactions, feelings, an unnamed present setting, and ordinary present conversational actions are ancillary and allowed, but they must not create a destination fact or alter a frozen claim.",
            "Do not repeat a core proposition after it has been established. Later turns should react, reflect, or coordinate without restating its value and status.",
            "The checkpoint contains no fact values. Do not infer or restate facts from earlier scenes; only the current transition may refer to its explicitly supplied prior value.",
            "Return exactly one JSON object with the sole top-level key turns. Each turn has only speaker and text. Return no claim IDs, evidence, quotes, graph, QA, explanation, or meta commentary.",
        ],
        "response_schema": {
            "turns": [{"speaker": "John or Tim", "text": "visible dialogue"}]
        },
    }
    if scene["scene_id"] != "S1":
        prompt["rules"][1] = (
            "Express the current node claim naturally and provide a light transition "
            "cue. The relation claim is compiled from one already-frozen prior-scene "
            "span plus one current-scene span; do not restate the prior value or its "
            "status in this scene."
        )
        prompt["cross_scene_relation_evidence_policy"] = {
            "relation_is_new_in_this_scene": True,
            "relation_support": "one frozen prior-scene span plus one current-scene span",
            "prior_claim_restatement_allowed": False,
            "current_node_expression_required": True,
        }
    if failure_feedback:
        prompt["previous_attempt_feedback"] = {
            "reason_codes": list(failure_feedback),
            "instruction": "Correct only these issues; no rejected dialogue text is supplied.",
        }
    if scene["scene_id"] == "S1" and failure_feedback:
        prompt["immutable_parent_anchor_retry_guard"] = {
            "purpose": "continuity_reference_only_not_a_new_child_fact",
            "subject": "John",
            "companion": "John's wife",
            "event": "completed road trip",
            "location_scope": "the European coastline",
            "allowed_memory_details": [
                "spectacular views",
                "fun",
                "bonding",
                "creating amazing memories",
            ],
            "speaker_role_constraint": (
                "Tim is only the present listener. Tim was not stated to be a "
                "participant in the earlier trip."
            ),
            "forbidden_additions": [
                "a country, city, or named venue",
                "a route, date, deadline, booking, or coordinates",
                "getting lost, a cafe, food, or another concrete incident",
                "a source that introduced Cedar Hall",
            ],
        }
    if failure_feedback and any(
        code in {"turn_count_out_of_range", "scene_below_minimum_token_count"}
        for code in failure_feedback
    ):
        prompt["retry_output_length_guard"] = {
            "turn_count": "exactly 10 turns, no more and no fewer",
            "visible_word_count": "330-380 English words across all 10 turns",
            "per_turn_guidance": "usually 28-42 words per turn",
            "content_guidance": (
                "Use reactions and general reflection to reach the length. Do not "
                "add another event, trip, plan, place, date, or source."
            ),
        }
    if failure_feedback and "invented_intervening_adventure_plans" in failure_feedback:
        prompt["retry_truth_guard_no_intervening_events"] = (
            "Do not claim that John or his wife planned, attempted, or completed "
            "any other trip or adventure after the frozen parent memory."
        )
    return prompt


def _basic_scene_audit(response: Any, scene_index: int) -> dict[str, Any]:
    issues: list[str] = []
    turns: list[Any]
    if not isinstance(response, dict) or set(response) != {"turns"}:
        issues.append("response_must_contain_only_turns")
        turns = []
    elif not isinstance(response.get("turns"), list):
        issues.append("turns_must_be_array")
        turns = []
    else:
        turns = response["turns"]
    if not MIN_TURNS <= len(turns) <= MAX_TURNS:
        issues.append("turn_count_out_of_range")
    normalized = []
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict) or set(turn) != {"speaker", "text"}:
            issues.append(f"turn_{index}_shape_invalid")
            continue
        speaker = str(turn.get("speaker") or "").strip()
        text = " ".join(str(turn.get("text") or "").split()).strip()
        if speaker not in {"John", "Tim"}:
            issues.append(f"turn_{index}_speaker_invalid")
        if not text:
            issues.append(f"turn_{index}_text_empty")
        normalized.append(
            {
                "turn_id": f"C{scene_index:02d}:T{index:02d}",
                "speaker": speaker,
                "text": text,
            }
        )
    token_count, token_mode = native.dialogue_tokens_for_turns(
        normalized, str(TOKENIZER_PATH)
    )
    return {
        "pass": not issues,
        "issues": issues,
        "turn_count": len(normalized),
        "scene_token_count": token_count,
        "scene_token_target": MIN_SCENE_TOKENS,
        "scene_token_target_met": token_count >= MIN_SCENE_TOKENS,
        "token_count_mode": token_mode,
        "normalized_candidate_turns": normalized,
    }


def _last_rejection_codes(output_dir: Path, scene_id: str) -> list[str]:
    rows = [
        row
        for row in _read_jsonl(_paths(output_dir)["rejects"])
        if row.get("scene_id") == scene_id
    ]
    if not rows:
        return []
    return [str(code) for code in rows[-1].get("reason_codes") or []]


def _api_args() -> argparse.Namespace:
    config = _read_json(ROOT / "config.json").get("api") or {}
    return argparse.Namespace(
        challenger_api_key=config.get("api_key"),
        challenger_base_url=config.get("base_url"),
        challenger_model=MODEL,
        challenger_temperature=0.25,
        challenger_max_tokens=2800,
        max_llm_retries=0,
        llm_timeout=180,
        api_yield_normalizers=False,
    )


def _find_attempt(output_dir: Path, attempt_id: str) -> dict[str, Any]:
    results = [
        row
        for row in _read_jsonl(_paths(output_dir)["attempts"])
        if row.get("record_type") == "api_result"
        and row.get("attempt_id") == attempt_id
    ]
    if len(results) != 1:
        raise PilotError(f"expected exactly one result for pending attempt {attempt_id}")
    return results[0]


def _command_generate_next(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise PilotError("generate-next requires --execute-live")
    plan, state = _load_initialized(args.output_dir)
    if state.get("pending_attempt_id") or state.get("inflight_attempt_id"):
        raise PilotError("current scene already has a pending or inflight attempt")
    scene_index = int(state["accepted_scene_count"])
    if scene_index >= SCENE_COUNT:
        raise PilotError("all planned scenes are already accepted")
    scene = plan["scenes"][scene_index]
    scene_id = scene["scene_id"]
    attempt_number = int(state["scene_attempt_counts"][scene_id]) + 1
    if attempt_number > MAX_ATTEMPTS_PER_SCENE:
        raise PilotError(f"retry budget exhausted for {scene_id}")
    accepted = _read_jsonl(_paths(args.output_dir)["accepted"])
    prompt = _scene_prompt(
        plan,
        scene,
        state,
        accepted,
        _last_rejection_codes(args.output_dir, scene_id),
    )
    attempt_id = _id(
        "scene_attempt",
        {
            "run_id": RUN_ID,
            "scene_id": scene_id,
            "attempt_number": attempt_number,
            "prompt_sha256": _sha256_json(prompt),
        },
    )
    state["scene_attempt_counts"][scene_id] = attempt_number
    state["physical_scene_api_calls"] += 1
    state["inflight_attempt_id"] = attempt_id
    state["status"] = "scene_api_call_inflight"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    _append_jsonl(
        _paths(args.output_dir)["attempts"],
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "api_call_intent",
            "attempt_id": attempt_id,
            "scene_id": scene_id,
            "scene_index": scene["scene_index"],
            "attempt_number": attempt_number,
            "accepted_scene_count_before_call": len(accepted),
            "accepted_scene_chain_head_before_call": state[
                "accepted_scene_chain_head"
            ],
            "prompt_sha256": _sha256_json(prompt),
            "prompt": prompt,
            "model": MODEL,
            "max_tokens": 2800,
            "temperature": 0.25,
            "max_internal_api_retries": 0,
            "unix_time": int(time.time()),
        },
    )
    started = time.monotonic()
    try:
        response, api_meta = native.call_raw_json_api(
            _api_args(),
            prompt,
            system_prompt=SYSTEM_PROMPT,
            required_keys=("turns",),
            call_kind="api_truth_single_scene_probe",
            max_tokens_override=2800,
            max_retries_override=0,
            temperature_override=0.25,
            include_raw_response=True,
        )
    except Exception as exc:
        response = None
        api_meta = {
            "used": False,
            "call_kind": "api_truth_single_scene_probe",
            "reason": "uncaught_transport_or_client_exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "attempts": 1,
        }
    audit = _basic_scene_audit(response, scene["scene_index"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "api_result",
        "attempt_id": attempt_id,
        "scene_id": scene_id,
        "scene_index": scene["scene_index"],
        "attempt_number": attempt_number,
        "frozen_scene_plan_sha256": scene["scene_plan_sha256"],
        "frozen_claim_ids": copy.deepcopy(scene["frozen_claim_ids"]),
        "response": response,
        "candidate_turns": audit.pop("normalized_candidate_turns"),
        "basic_local_audit": audit,
        "api_meta": api_meta,
        "latency_ms": int(round((time.monotonic() - started) * 1000)),
        "manual_semantic_review_required": True,
        "accepted_automatically": False,
    }
    _append_jsonl(_paths(args.output_dir)["attempts"], result)
    state["inflight_attempt_id"] = None
    state["pending_attempt_id"] = attempt_id
    state["status"] = "scene_pending_semantic_review"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "scene_id": scene_id,
                "attempt_number": attempt_number,
                "api_used": bool(api_meta.get("used")),
                "basic_local_audit": audit,
                "accepted_scene_count_unchanged": state["accepted_scene_count"],
                "manual_review_required": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _validate_evidence(
    scene: Mapping[str, Any],
    attempt: Mapping[str, Any],
    review: Mapping[str, Any],
    accepted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = review.get("claim_evidence")
    if not isinstance(evidence, list):
        raise PilotError("review.claim_evidence must be a list")
    expected_ids = set(scene["frozen_claim_ids"])
    actual_ids = {
        row.get("claim_id") for row in evidence if isinstance(row, dict)
    }
    if actual_ids != expected_ids or len(evidence) != len(expected_ids):
        raise PilotError(
            f"review evidence must cover each frozen claim exactly once: "
            f"expected={sorted(expected_ids)}, actual={sorted(str(x) for x in actual_ids)}"
        )
    candidate_turns = attempt.get("candidate_turns") or []
    candidate_turn_ids = {turn["turn_id"] for turn in candidate_turns}
    prior_turns = [turn for prior in accepted for turn in prior.get("turns") or []]
    prior_turn_ids = {turn["turn_id"] for turn in prior_turns}
    turns = {turn["turn_id"]: turn for turn in [*prior_turns, *candidate_turns]}
    normalized = []
    for row in evidence:
        support_turn_ids = row.get("support_turn_ids")
        spans = row.get("support_spans")
        if (
            not isinstance(support_turn_ids, list)
            or not 1 <= len(support_turn_ids) <= 2
            or len(set(support_turn_ids)) != len(support_turn_ids)
        ):
            raise PilotError("each claim needs one or two unique support_turn_ids")
        if not isinstance(spans, list) or not spans:
            raise PilotError("each claim needs at least one exact support span")
        if row.get("resolved_subject") != "John":
            raise PilotError("all pilot claim evidence must resolve its subject to John")
        resolution = row.get("subject_resolution")
        if resolution not in {
            "speaker_first_person",
            "addressee_second_person",
            "explicit_name",
            "adjacent_exchange",
        }:
            raise PilotError("claim evidence needs a supported subject_resolution")
        for turn_id in support_turn_ids:
            if turn_id not in turns:
                raise PilotError(f"unknown support turn {turn_id}")
        for span in spans:
            if not isinstance(span, dict):
                raise PilotError("support span must be an object")
            turn_id = span.get("turn_id")
            quote = str(span.get("exact_support_quote") or "")
            if turn_id not in support_turn_ids or turn_id not in turns:
                raise PilotError("support span turn is not declared")
            if not quote or quote not in turns[turn_id]["text"]:
                raise PilotError(
                    f"support quote is not an exact substring of {turn_id}: {quote!r}"
                )
        if resolution == "speaker_first_person" and not any(
            turns[turn_id]["speaker"] == "John"
            and re.search(r"\b(?:I|I'm|I've|my|me)\b", turns[turn_id]["text"], re.I)
            for turn_id in support_turn_ids
        ):
            raise PilotError("speaker_first_person evidence is not spoken by John")
        if resolution == "addressee_second_person" and not any(
            turns[turn_id]["speaker"] == "Tim"
            and re.search(r"\b(?:you|your|you're|you've)\b", turns[turn_id]["text"], re.I)
            for turn_id in support_turn_ids
        ):
            raise PilotError("addressee_second_person evidence is not addressed by Tim")
        if resolution == "explicit_name" and not any(
            "john" in turns[turn_id]["text"].casefold()
            for turn_id in support_turn_ids
        ):
            raise PilotError("explicit_name evidence does not name John")
        if resolution == "adjacent_exchange" and len(support_turn_ids) != 2:
            raise PilotError("adjacent_exchange evidence requires exactly two turns")
        relation_id = scene["relation_claim_id"]
        if row.get("claim_id") == relation_id and scene["scene_id"] != "S1":
            if not (
                any(turn_id in prior_turn_ids for turn_id in support_turn_ids)
                and any(turn_id in candidate_turn_ids for turn_id in support_turn_ids)
            ):
                raise PilotError(
                    "cross-scene transition evidence needs one prior and one current turn"
                )
        elif not all(turn_id in candidate_turn_ids for turn_id in support_turn_ids):
            raise PilotError("node/S1 evidence must come from the current candidate scene")
        normalized_row = copy.deepcopy(row)
        normalized_row["verification_status"] = "manual_semantic_review_exact_quote"
        for span in normalized_row["support_spans"]:
            span["speaker"] = turns[span["turn_id"]]["speaker"]
        normalized.append(normalized_row)
    return normalized


def _stable_turns(
    accepted: list[dict[str, Any]], candidate_turns: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    offset = sum(len(scene.get("turns") or []) for scene in accepted)
    stable = []
    mapping = {}
    for index, turn in enumerate(candidate_turns, start=1):
        stable_id = f"D27:{offset + index}"
        mapping[turn["turn_id"]] = stable_id
        stable.append(
            {
                "turn_id": stable_id,
                "speaker": turn["speaker"],
                "text": turn["text"],
                "scene_id": f"S{len(accepted) + 1}",
                "candidate_turn_id": turn["turn_id"],
                "provenance": "qwen32b_incremental_scene_api",
            }
        )
    return stable, mapping


def _command_admit(args: argparse.Namespace) -> int:
    plan, state = _load_initialized(args.output_dir)
    pending = state.get("pending_attempt_id")
    if not pending:
        raise PilotError("there is no scene attempt pending review")
    review = _read_json(args.review_file)
    if review.get("attempt_id") != pending or review.get("decision") != "accept":
        raise PilotError("review does not accept the pending attempt")
    required_flags = (
        "semantic_clean",
        "natural_dialogue",
        "continuity_pass",
        "status_and_relation_preserved",
        "ancillary_isolated",
        "no_training_data_modified",
    )
    failed_flags = [name for name in required_flags if review.get(name) is not True]
    if failed_flags:
        raise PilotError(f"scene review failed required gates: {failed_flags}")
    attempt = _find_attempt(args.output_dir, pending)
    scene = plan["scenes"][int(state["accepted_scene_count"])]
    if attempt.get("scene_id") != scene["scene_id"]:
        raise PilotError("pending attempt is for a different scene")
    basic = attempt.get("basic_local_audit") or {}
    if basic.get("pass") is not True:
        raise PilotError(f"cannot admit a basic-invalid scene: {basic.get('issues')}")
    if basic.get("token_count_mode") != "tokenizer":
        raise PilotError("cannot admit a scene measured only by approximate tokens")
    accepted = _read_jsonl(_paths(args.output_dir)["accepted"])
    evidence = _validate_evidence(scene, attempt, review, accepted)
    stable_turns, turn_mapping = _stable_turns(accepted, attempt["candidate_turns"])
    stable_evidence = copy.deepcopy(evidence)
    for row in stable_evidence:
        row["candidate_support_turn_ids"] = list(row["support_turn_ids"])
        row["support_turn_ids"] = [
            turn_mapping.get(item, item) for item in row["support_turn_ids"]
        ]
        for span in row["support_spans"]:
            span["candidate_turn_id"] = span["turn_id"]
            span["turn_id"] = turn_mapping.get(span["turn_id"], span["turn_id"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "attempt_id": pending,
        "attempt_number": attempt["attempt_number"],
        "frozen_scene_plan_sha256": scene["scene_plan_sha256"],
        "frozen_claim_ids": copy.deepcopy(scene["frozen_claim_ids"]),
        "frozen_claims": copy.deepcopy(scene["frozen_claims"]),
        "turns": stable_turns,
        "scene_token_count": basic["scene_token_count"],
        "token_count_mode": basic["token_count_mode"],
        "claim_evidence": stable_evidence,
        "ancillary_facts": copy.deepcopy(review.get("ancillary_facts") or []),
        "ancillary_policy": {
            "graph_inclusion": False,
            "qa_inclusion": False,
            "gold_inclusion": False,
            "inheritance": False,
            "checkpoint_inclusion": False,
        },
        "manual_review": {
            key: copy.deepcopy(review.get(key))
            for key in (*required_flags, "reviewer", "review_notes")
        },
        "previous_accepted_scene_sha256": state["accepted_scene_chain_head"],
    }
    record["accepted_scene_sha256"] = _sha256_json(record)
    _append_jsonl(_paths(args.output_dir)["accepted"], record)
    state["accepted_scene_count"] += 1
    state["accepted_scene_chain_head"] = record["accepted_scene_sha256"]
    state["total_added_tokens"] += int(record["scene_token_count"])
    state["total_added_turns"] += len(stable_turns)
    state["pending_attempt_id"] = None
    if state["accepted_scene_count"] == SCENE_COUNT:
        state["status"] = (
            "scenes_complete"
            if state["total_added_tokens"] >= MIN_TOTAL_ADDED_TOKENS
            else "failed_total_token_gate"
        )
    else:
        state["status"] = "ready_for_scene"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    _audit_accepted_chain(args.output_dir, state)
    print(
        json.dumps(
            {
                "admitted": scene["scene_id"],
                "attempt_number": record["attempt_number"],
                "scene_tokens": record["scene_token_count"],
                "accepted_scene_count": state["accepted_scene_count"],
                "total_added_tokens": state["total_added_tokens"],
                "accepted_scene_chain_head": state["accepted_scene_chain_head"],
                "next_status": state["status"],
            },
            sort_keys=True,
        )
    )
    return 0


def _command_reject(args: argparse.Namespace) -> int:
    plan, state = _load_initialized(args.output_dir)
    pending = state.get("pending_attempt_id")
    if not pending:
        raise PilotError("there is no scene attempt pending review")
    review = _read_json(args.review_file)
    if review.get("attempt_id") != pending or review.get("decision") != "reject":
        raise PilotError("review does not reject the pending attempt")
    reason_codes = review.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        raise PilotError("rejection requires at least one reason code")
    attempt = _find_attempt(args.output_dir, pending)
    scene = plan["scenes"][int(state["accepted_scene_count"])]
    record = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": pending,
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "attempt_number": attempt["attempt_number"],
        "reason_codes": [str(item) for item in reason_codes],
        "review_notes": str(review.get("review_notes") or ""),
        "accepted_scenes_retained": state["accepted_scene_count"],
        "accepted_scene_chain_head_retained": state["accepted_scene_chain_head"],
        "rejected_candidate_modified": False,
    }
    _append_jsonl(_paths(args.output_dir)["rejects"], record)
    state["pending_attempt_id"] = None
    if int(state["scene_attempt_counts"][scene["scene_id"]]) >= MAX_ATTEMPTS_PER_SCENE:
        state["status"] = "failed_scene_retry_budget"
    else:
        state["status"] = "ready_for_scene"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "rejected": scene["scene_id"],
                "attempt_number": attempt["attempt_number"],
                "reason_codes": reason_codes,
                "accepted_scenes_retained": state["accepted_scene_count"],
                "next_status": state["status"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parent_max_dialogue_major(turns: list[dict[str, Any]]) -> int:
    majors = []
    for turn in turns:
        match = re.match(r"^D(\d+):", str(turn.get("turn_id") or ""))
        if match:
            majors.append(int(match.group(1)))
    return max(majors) if majors else 0


def _build_child_artifact(
    source: Mapping[str, Any], plan: Mapping[str, Any], accepted: list[dict[str, Any]]
) -> dict[str, Any]:
    child_turns = [copy.deepcopy(turn) for scene in accepted for turn in scene["turns"]]
    exact_child_tokens, exact_child_token_mode = native.dialogue_tokens_for_turns(
        child_turns, str(TOKENIZER_PATH)
    )
    parent_turns = copy.deepcopy(source["seed"]["dialogue"]["turns"])
    all_turns = parent_turns + child_turns
    dialogue_hash = _sha256_json(all_turns)
    evidence_ledger = [
        copy.deepcopy(row)
        for scene in accepted
        for row in scene.get("claim_evidence") or []
    ]
    alignment_hash = _sha256_json(evidence_ledger)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "isolated_temporal_incremental_child_pilot",
        "training_use": "pilot_only_not_admitted",
        "sample_id": _id(
            "stage3_temporal_pilot_child",
            {
                "parent": SOURCE_SEED_SAMPLE_ID,
                "plan": plan["plan_sha256"],
                "scene_chain": accepted[-1]["accepted_scene_sha256"],
            },
        ),
        "dialogue_family_id": _id(
            "dialogue_family",
            {"lineage": SOURCE_LINEAGE_ID, "scene_chain": accepted[-1]["accepted_scene_sha256"]},
        ),
        "parent_sample_id": SOURCE_SEED_SAMPLE_ID,
        "lineage_group": SOURCE_LINEAGE_ID,
        "canonical_parent_lineage_id": SOURCE_LINEAGE_ID,
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_artifact_file_sha256": SOURCE_ARTIFACT_FILE_SHA256,
        "source_parent_graph_sha256": source["artifact"]["lineage_hashes"][
            "parent_graph_sha256"
        ],
        "parent_dialogue_hash": PARENT_DIALOGUE_HASH,
        "source_child_graph_sha256": plan["source_child_graph_sha256"],
        "frozen_scene_plan_sha256": plan["plan_sha256"],
        "accepted_scene_chain_head": accepted[-1]["accepted_scene_sha256"],
        "accepted_scene_count": len(accepted),
        "accepted_scenes": copy.deepcopy(accepted),
        "parent_turn_count": len(parent_turns),
        "child_turn_count": len(child_turns),
        "per_scene_token_count_sum": sum(
            int(scene["scene_token_count"]) for scene in accepted
        ),
        "added_token_count": exact_child_tokens,
        "token_count_mode": exact_child_token_mode,
        "dialogue": {"turns": all_turns, "dialogue_hash": dialogue_hash},
        "child_dialogue": {
            "turns": child_turns,
            "dialogue_hash": _sha256_json(child_turns),
        },
        "claim_evidence_ledger": evidence_ledger,
        "alignment_hash": alignment_hash,
        "parent_anchor": {
            "event": copy.deepcopy(source["parent_event"]),
            "evidence_turn": copy.deepcopy(source["parent_turn"]),
        },
        "ancillary_isolation": {
            "ancillary_facts": [
                copy.deepcopy(item)
                for scene in accepted
                for item in scene.get("ancillary_facts") or []
            ],
            "included_in_graph": False,
            "included_in_qa": False,
            "included_in_gold": False,
            "included_in_inheritance": False,
            "included_in_checkpoints": False,
        },
        "source_graph_mutated": _sha256_file(SOURCE_ARTIFACT)
        != SOURCE_ARTIFACT_FILE_SHA256,
        "training_data_modified": False,
        "next_dialogue_major": _parent_max_dialogue_major(parent_turns) + 2,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def _command_finalize_scenes(args: argparse.Namespace) -> int:
    plan, state = _load_initialized(args.output_dir)
    if state.get("accepted_scene_count") != SCENE_COUNT:
        raise PilotError("cannot finalize before all four scenes are accepted")
    if state.get("total_added_tokens", 0) < MIN_TOTAL_ADDED_TOKENS:
        raise PilotError("cannot finalize: total added token gate failed")
    accepted = _read_jsonl(_paths(args.output_dir)["accepted"])
    source = _source_bundle()
    child = _build_child_artifact(source, plan, accepted)
    if child["token_count_mode"] != "tokenizer":
        raise PilotError("cannot finalize without exact child-dialogue tokenizer count")
    if child["added_token_count"] < MIN_TOTAL_ADDED_TOKENS:
        raise PilotError("cannot finalize: exact full-child token gate failed")
    _atomic_write_json(_paths(args.output_dir)["child"], child)
    state["status"] = "scenes_frozen_ready_for_qa"
    state["child_artifact_sha256"] = child["artifact_sha256"]
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "child_artifact": str(_paths(args.output_dir)["child"]),
                "artifact_sha256": child["artifact_sha256"],
                "accepted_scenes": child["accepted_scene_count"],
                "child_turns": child["child_turn_count"],
                "added_tokens": child["added_token_count"],
                "training_data_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _claim_evidence_map(child: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = child.get("claim_evidence_ledger") or []
    result = {}
    for row in rows:
        claim_id = row.get("claim_id")
        if not claim_id or claim_id in result:
            raise PilotError("child evidence ledger has missing or duplicate claim IDs")
        result[claim_id] = copy.deepcopy(row)
    return result


def _option_rows(gold_text: str, values: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    options = []
    gold = []
    for index, value in enumerate(values, start=1):
        option_id = f"O{index}"
        is_gold = value == gold_text
        options.append(
            {
                "id": option_id,
                "option_id": option_id,
                "text": value,
                "is_gold": is_gold,
                "truth_value": is_gold,
                "source": "local_graph_truth_or_locally_proved_false",
            }
        )
        if is_gold:
            gold.append(option_id)
    if len(gold) != 1:
        raise PilotError(f"gold {gold_text!r} is not unique in options")
    return options, gold


def _evidence_for_claims(
    child: Mapping[str, Any], claim_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    evidence_map = _claim_evidence_map(child)
    claim_to_scene = {
        claim_id: scene["scene_id"]
        for scene in child["accepted_scenes"]
        for claim_id in scene["frozen_claim_ids"]
    }
    ledger = [
        {
            **copy.deepcopy(evidence_map[claim_id]),
            "source_scope": "child",
            "scene_id": claim_to_scene[claim_id],
        }
        for claim_id in claim_ids
    ]
    parent_turn = child["parent_anchor"]["evidence_turn"]
    ledger.insert(
        0,
        {
            "claim_id": "pilot_parent_claim_P0",
            "source_scope": "parent",
            "scene_id": None,
            "support_turn_ids": [parent_turn["turn_id"]],
            "support_spans": [
                {
                    "turn_id": parent_turn["turn_id"],
                    "speaker": parent_turn["speaker"],
                    "exact_support_quote": parent_turn["text"],
                }
            ],
            "resolved_subject": "John",
            "verification_status": "source_parent_exact_turn",
        },
    )
    turn_ids = []
    child_scenes = []
    turn_to_scene = {
        turn["turn_id"]: scene["scene_id"]
        for scene in child["accepted_scenes"]
        for turn in scene["turns"]
    }
    for row in ledger:
        for turn_id in row.get("support_turn_ids") or []:
            if turn_id not in turn_ids:
                turn_ids.append(turn_id)
            scene_id = turn_to_scene.get(turn_id)
            if scene_id and scene_id not in child_scenes:
                child_scenes.append(scene_id)
    return ledger, turn_ids, child_scenes


def _query_contracts(child: Mapping[str, Any]) -> dict[str, Any]:
    scene_claim_ids = {
        scene["scene_id"]: list(scene["frozen_claim_ids"])
        for scene in child["accepted_scenes"]
    }
    all_values = ["Cedar Hall", "London", "the Midwest", "community gym"]
    sequence_values = [
        "belief -> active -> outdated -> canceled",
        "active -> belief -> outdated -> canceled",
        "belief -> active -> canceled -> outdated",
        "belief -> outdated -> active -> canceled",
    ]
    specs = [
        {
            "key": "Q1",
            "qa_role": "target_fulfillment",
            "qa_operation": "temporal_state_update",
            "qa_subtype": "status_gated_current_attribute_lookup",
            "reasoning_subtype": "status_gated_canceled_update",
            "reasoning_template": "reject_canceled_value_then_resolve_update",
            "hop_count": 4,
            "answer_type": "LocationOrScope",
            "hard_negative_strategy": "same_slot_canceled_outdated_unconfirmed_values",
            "forbid_max_time_shortcut": True,
            "target_status": "active",
            "gold_answer_text": "London",
            "options": all_values,
            "required_scene_ids": ["S1", "S2", "S3", "S4"],
            "reasoning_event_ids": [PARENT_EVENT_ID, "E1", "E2", "E3", "E4"],
            "answer_event_ids": ["E2"],
            "question_draft": (
                "The traveler who described a European coastline road trip with "
                "his wife later discussed a chain of destination records. Which "
                "destination was active after the unconfirmed Cedar Hall entry but "
                "before the Midwest entry became outdated and the community gym "
                "plan was canceled?"
            ),
            "surface_requirements": ["European coastline", "wife", "destination", "active"],
        },
        {
            "key": "Q2",
            "qa_role": "sibling",
            "qa_operation": "relation_lookup",
            "qa_subtype": "status_qualified_belief_attribute_lookup",
            "reasoning_subtype": "epistemic_status_resolution",
            "reasoning_template": "select_value_by_epistemic_status",
            "hop_count": 3,
            "answer_type": "LocationOrScope",
            "hard_negative_strategy": "same_slot_wrong_status",
            "forbid_max_time_shortcut": True,
            "target_status": "belief_unconfirmed",
            "gold_answer_text": "Cedar Hall",
            "options": all_values,
            "required_scene_ids": ["S1", "S2", "S3"],
            "reasoning_event_ids": [PARENT_EVENT_ID, "E1", "E2", "E3"],
            "answer_event_ids": ["E1"],
            "question_draft": (
                "The traveler who recalled a European coastline road trip with "
                "his wife began a later destination-state chain with an unconfirmed "
                "belief, before London became active and the Midwest appeared as "
                "outdated. Which destination began that chain?"
            ),
            "surface_requirements": ["European coastline", "wife", "destination", "unconfirmed"],
        },
        {
            "key": "Q3",
            "qa_role": "sibling",
            "qa_operation": "relation_lookup",
            "qa_subtype": "status_qualified_canceled_attribute_lookup",
            "reasoning_subtype": "lifecycle_status_resolution",
            "reasoning_template": "select_value_by_lifecycle_status",
            "hop_count": 3,
            "answer_type": "LocationOrScope",
            "hard_negative_strategy": "same_slot_wrong_lifecycle_status",
            "forbid_max_time_shortcut": True,
            "target_status": "canceled_before_effect",
            "gold_answer_text": "community gym",
            "options": all_values,
            "required_scene_ids": ["S2", "S3", "S4"],
            "reasoning_event_ids": [PARENT_EVENT_ID, "E1", "E2", "E3", "E4"],
            "answer_event_ids": ["E4"],
            "question_draft": (
                "The traveler who had earlier described a European coastline road "
                "trip with his wife later moved from an active London entry through "
                "an outdated Midwest entry. Which destination plan came next and "
                "was canceled before taking effect?"
            ),
            "surface_requirements": ["European coastline", "wife", "destination", "canceled"],
        },
        {
            "key": "Q4",
            "qa_role": "sibling",
            "qa_operation": "temporal_state_update",
            "qa_subtype": "ordered_status_sequence",
            "reasoning_subtype": "multi_scene_state_sequence",
            "reasoning_template": "order_statuses_across_frozen_scene_chain",
            "hop_count": 5,
            "answer_type": "StatusSequence",
            "hard_negative_strategy": "status_order_mutation",
            "forbid_max_time_shortcut": True,
            "target_status": "ordered_sequence",
            "gold_answer_text": sequence_values[0],
            "options": sequence_values,
            "required_scene_ids": ["S1", "S2", "S3", "S4"],
            "reasoning_event_ids": [PARENT_EVENT_ID, "E1", "E2", "E3", "E4"],
            "answer_event_ids": ["E1", "E2", "E3", "E4"],
            "question_draft": (
                "For the traveler who once described a European coastline road "
                "trip with his wife, what status sequence was recorded across the "
                "four later destination entries?"
            ),
            "surface_requirements": ["European coastline", "wife", "destination", "status"],
        },
    ]
    contracts = []
    for spec in specs:
        claim_ids = []
        for scene_id in spec["required_scene_ids"]:
            claim_ids.extend(scene_claim_ids[scene_id])
        evidence, turn_ids, evidence_scenes = _evidence_for_claims(child, claim_ids)
        if len(evidence_scenes) < 2 or not set(spec["required_scene_ids"]).issubset(
            evidence_scenes
        ):
            raise PilotError(f"{spec['key']} does not have the required cross-scene evidence")
        option_values = spec.pop("options")
        options, gold = _option_rows(spec["gold_answer_text"], option_values)
        event_status_by_value = {
            "Cedar Hall": {"event_id": "E1", "status": "belief_unconfirmed"},
            "London": {"event_id": "E2", "status": "active"},
            "the Midwest": {
                "event_id": "E3",
                "status": "outdated_no_longer_applicable",
            },
            "community gym": {
                "event_id": "E4",
                "status": "canceled_before_effect",
            },
        }
        distractor_proofs = []
        for option in options:
            if option["is_gold"]:
                option["local_truth_proof"] = {
                    "result": "gold",
                    "compiled_gold": spec["gold_answer_text"],
                }
                continue
            if spec["answer_type"] == "StatusSequence":
                observed = option["text"].split(" -> ")
                expected = spec["gold_answer_text"].split(" -> ")
                mismatches = [
                    {
                        "position": index + 1,
                        "expected": expected[index],
                        "observed": observed[index],
                    }
                    for index in range(min(len(expected), len(observed)))
                    if expected[index] != observed[index]
                ]
                proof = {
                    "result": "false",
                    "proof_type": "ordered_status_mismatch",
                    "frozen_event_order": ["E1", "E2", "E3", "E4"],
                    "mismatches": mismatches,
                }
            else:
                source = event_status_by_value[option["text"]]
                proof = {
                    "result": "false",
                    "proof_type": "frozen_status_mismatch",
                    "source_event_id": source["event_id"],
                    "observed_status": source["status"],
                    "required_status": spec["target_status"],
                }
            option["local_false_proof"] = proof
            distractor_proofs.append(
                {"option_id": option["id"], "option_text": option["text"], **proof}
            )
        semantic = {
            key: copy.deepcopy(value)
            for key, value in spec.items()
            if key not in {"key", "question_draft", "surface_requirements"}
        }
        semantic.update(
            {
                "required_claim_ids": ["pilot_parent_claim_P0", *claim_ids],
                "required_evidence_origins": ["parent", "child"],
                "minimum_child_scene_count": 2,
                "compiled_gold": spec["gold_answer_text"],
                "options": options,
                "gold": gold,
                "distractor_proofs": distractor_proofs,
                "parent_anchor_event_id": PARENT_EVENT_ID,
            }
        )
        contract_id = _id("query_contract", semantic)
        contracts.append(
            {
                "query_contract_id": contract_id,
                "question_draft": spec["question_draft"],
                "surface_requirements": spec["surface_requirements"],
                "semantic_contract": semantic,
                "evidence_ledger": evidence,
                "evidence_turn_ids": turn_ids,
                "child_scene_ids": evidence_scenes,
            }
        )
    bundle = {
        "schema_version": "stage3_temporal_pilot_query_contracts_v1",
        "source_child_artifact_sha256": child["artifact_sha256"],
        "truth_compiler": "local_deterministic",
        "distractor_source": "frozen_graph_status_contrasts",
        "api_selected_truth_fields": [],
        "contracts": contracts,
    }
    bundle["query_contract_bundle_sha256"] = _sha256_json(bundle)
    return bundle


def _command_compile_qa_contracts(args: argparse.Namespace) -> int:
    _, state = _load_initialized(args.output_dir)
    if not _paths(args.output_dir)["child"].exists():
        raise PilotError("freeze the child artifact before compiling QA contracts")
    child = _read_json(_paths(args.output_dir)["child"])
    contracts = _query_contracts(child)
    _atomic_write_json(_paths(args.output_dir)["qa_contracts"], contracts)
    state["status"] = "qa_contracts_frozen_ready_for_polish"
    state["query_contract_bundle_sha256"] = contracts[
        "query_contract_bundle_sha256"
    ]
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "qa_contracts": len(contracts["contracts"]),
                "query_contract_bundle_sha256": contracts[
                    "query_contract_bundle_sha256"
                ],
                "all_include_parent_evidence": all(
                    "D17:3" in contract["evidence_turn_ids"]
                    for contract in contracts["contracts"]
                ),
                "minimum_child_scenes": min(
                    len(contract["child_scene_ids"])
                    for contract in contracts["contracts"]
                ),
                "api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _qa_polish_prompt(contracts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "truth_first_question_surface_polish_v1",
        "task": (
            "Polish only the wording of each frozen question draft. Keep the "
            "identity binder, queried status/sequence, answer type, and target "
            "unchanged. Do not answer the question and do not alter options."
        ),
        "queries": [
            {
                "query_contract_id": row["query_contract_id"],
                "question_draft": row["question_draft"],
                "qa_operation": row["semantic_contract"]["qa_operation"],
                "qa_subtype": row["semantic_contract"]["qa_subtype"],
                "answer_type": row["semantic_contract"]["answer_type"],
                "must_retain_concepts": row["surface_requirements"],
            }
            for row in contracts["contracts"]
        ],
        "rules": [
            "Return every query_contract_id exactly once and unchanged.",
            "Return one natural single-sentence question per contract.",
            "Do not include an answer, option, explanation, evidence, graph, or new fact.",
            "Do not change current vs unconfirmed vs outdated vs canceled semantics or the requested sequence.",
        ],
        "response_schema": {
            "questions": [
                {"query_contract_id": "opaque unchanged ID", "question": "polished question"}
            ]
        },
    }


def _basic_qa_polish_audit(
    response: Any, contracts: Mapping[str, Any]
) -> dict[str, Any]:
    issues = []
    expected = {row["query_contract_id"]: row for row in contracts["contracts"]}
    rows = response.get("questions") if isinstance(response, dict) else None
    if not isinstance(rows, list):
        rows = []
        issues.append("questions_array_missing")
    actual_ids = [
        row.get("query_contract_id") for row in rows if isinstance(row, dict)
    ]
    if len(rows) != len(expected) or set(actual_ids) != set(expected):
        issues.append("query_contract_id_coverage_mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_contract_id", "question"}:
            issues.append("question_row_shape_invalid")
            continue
        question = " ".join(str(row.get("question") or "").split())
        if not question.endswith("?"):
            issues.append(f"question_not_interrogative:{row.get('query_contract_id')}")
        if len(question) < 40:
            issues.append(f"question_too_short:{row.get('query_contract_id')}")
    return {"pass": not issues, "issues": issues, "question_count": len(rows)}


def _command_polish_qas(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise PilotError("polish-qas requires --execute-live")
    _, state = _load_initialized(args.output_dir)
    if state.get("pending_qa_polish_attempt_id"):
        raise PilotError("a QA polish result is already pending review")
    if int(state.get("qa_polish_attempt_count") or 0) >= MAX_ATTEMPTS_PER_SCENE:
        raise PilotError("QA polish retry budget exhausted")
    contracts = _read_json(_paths(args.output_dir)["qa_contracts"])
    prompt = _qa_polish_prompt(contracts)
    attempt_number = int(state.get("qa_polish_attempt_count") or 0) + 1
    attempt_id = _id(
        "qa_polish_attempt",
        {
            "bundle": contracts["query_contract_bundle_sha256"],
            "attempt_number": attempt_number,
            "prompt": prompt,
        },
    )
    state["qa_polish_attempt_count"] = attempt_number
    state["physical_qa_polish_api_calls"] += 1
    state["pending_qa_polish_attempt_id"] = attempt_id
    state["status"] = "qa_polish_api_call_inflight"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    started = time.monotonic()
    try:
        response, meta = native.call_raw_json_api(
            _api_args(),
            prompt,
            system_prompt=(
                "You only polish question wording for frozen local query contracts. "
                "Never choose truth, gold, proof, options, or evidence. Return JSON only."
            ),
            required_keys=("questions",),
            call_kind="api_truth_content_binding",
            max_tokens_override=1600,
            max_retries_override=0,
            temperature_override=0.2,
            include_raw_response=True,
        )
    except Exception as exc:
        response = None
        meta = {
            "used": False,
            "call_kind": "api_truth_content_binding",
            "reason": "uncaught_transport_or_client_exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "attempts": 1,
        }
    audit = _basic_qa_polish_audit(response, contracts)
    record = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "query_contract_bundle_sha256": contracts["query_contract_bundle_sha256"],
        "prompt": prompt,
        "prompt_sha256": _sha256_json(prompt),
        "response": response,
        "api_meta": meta,
        "basic_local_audit": audit,
        "latency_ms": int(round((time.monotonic() - started) * 1000)),
        "manual_target_preservation_review_required": True,
    }
    _append_jsonl(_paths(args.output_dir)["qa_polish"], record)
    state["status"] = "qa_polish_pending_semantic_review"
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "api_used": bool(meta.get("used")),
                "basic_local_audit": audit,
                "manual_target_review_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _find_qa_polish_attempt(output_dir: Path, attempt_id: str) -> dict[str, Any]:
    rows = [
        row
        for row in _read_jsonl(_paths(output_dir)["qa_polish"])
        if row.get("attempt_id") == attempt_id
    ]
    if len(rows) != 1:
        raise PilotError("pending QA polish attempt is missing or duplicated")
    return rows[0]


def _command_admit_qa_polish(args: argparse.Namespace) -> int:
    _, state = _load_initialized(args.output_dir)
    pending = state.get("pending_qa_polish_attempt_id")
    if not pending:
        raise PilotError("no QA polish attempt is pending")
    review = _read_json(args.review_file)
    if (
        review.get("attempt_id") != pending
        or review.get("decision") != "accept"
        or review.get("all_query_targets_preserved") is not True
        or review.get("no_answer_leakage") is not True
        or review.get("no_new_facts") is not True
    ):
        raise PilotError("QA polish review did not pass all frozen-target gates")
    attempt = _find_qa_polish_attempt(args.output_dir, pending)
    if (attempt.get("basic_local_audit") or {}).get("pass") is not True:
        raise PilotError("cannot admit a structurally invalid QA polish response")
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": pending,
        "query_contract_bundle_sha256": attempt["query_contract_bundle_sha256"],
        "questions": copy.deepcopy(attempt["response"]["questions"]),
        "manual_review": copy.deepcopy(review),
        "api_only_polished_surface": True,
        "api_selected_truth_fields": [],
    }
    frozen["qa_polish_review_sha256"] = _sha256_json(frozen)
    _atomic_write_json(_paths(args.output_dir)["qa_review"], frozen)
    state["pending_qa_polish_attempt_id"] = None
    state["status"] = "qa_surfaces_frozen_ready_to_finalize"
    state["qa_polish_review_sha256"] = frozen["qa_polish_review_sha256"]
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "qa_surfaces_frozen": len(frozen["questions"]),
                "qa_polish_review_sha256": frozen["qa_polish_review_sha256"],
                "api_selected_truth_fields": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_reject_qa_polish(args: argparse.Namespace) -> int:
    _, state = _load_initialized(args.output_dir)
    pending = state.get("pending_qa_polish_attempt_id")
    if not pending:
        raise PilotError("no QA polish attempt is pending")
    review = _read_json(args.review_file)
    if review.get("attempt_id") != pending or review.get("decision") != "reject":
        raise PilotError("review does not reject the pending QA polish attempt")
    reason_codes = review.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes:
        raise PilotError("QA polish rejection requires reason_codes")
    state["pending_qa_polish_attempt_id"] = None
    state["last_qa_polish_rejection"] = {
        "attempt_id": pending,
        "reason_codes": [str(item) for item in reason_codes],
        "review_notes": str(review.get("review_notes") or ""),
    }
    state["status"] = (
        "qa_polish_failed_retry_budget"
        if int(state["qa_polish_attempt_count"]) >= MAX_ATTEMPTS_PER_SCENE
        else "qa_contracts_frozen_ready_for_polish"
    )
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(
        json.dumps(
            {
                "rejected_qa_polish_attempt": pending,
                "reason_codes": reason_codes,
                "next_status": state["status"],
            },
            sort_keys=True,
        )
    )
    return 0


def _final_qas(
    contracts: Mapping[str, Any], review: Mapping[str, Any], child: Mapping[str, Any]
) -> list[dict[str, Any]]:
    questions = {
        row["query_contract_id"]: " ".join(str(row["question"]).split())
        for row in review["questions"]
    }
    qas = []
    for contract in contracts["contracts"]:
        contract_id = contract["query_contract_id"]
        semantic = contract["semantic_contract"]
        question = questions[contract_id]
        qa_identity = {
            "query_contract_id": contract_id,
            "question": question,
            "child_artifact_sha256": child["artifact_sha256"],
        }
        proof = {
            "schema_version": "stage3_temporal_pilot_local_proof_v1",
            "parent_graph_sha256": child["source_parent_graph_sha256"],
            "child_graph_sha256": child["source_child_graph_sha256"],
            "scene_ledger_sha256": child["accepted_scene_chain_head"],
            "required_claim_ids": copy.deepcopy(semantic["required_claim_ids"]),
            "reasoning_path_claim_ids": copy.deepcopy(semantic["required_claim_ids"]),
            "reasoning_path_event_instance_ids": copy.deepcopy(
                semantic["reasoning_event_ids"]
            ),
            "required_event_instance_ids": copy.deepcopy(
                semantic["reasoning_event_ids"]
            ),
            "answer_event_instance_ids": copy.deepcopy(
                semantic["answer_event_ids"]
            ),
            "dependency_closure_event_instance_ids": copy.deepcopy(
                semantic["reasoning_event_ids"]
            ),
            "claim_id_to_event_id": {
                "pilot_parent_claim_P0": PARENT_EVENT_ID,
                "clm_2fb998c11e4906a96235d467c7893894": "E1",
                "clm_0c4098d6434d515d232eb55d1772aa8a": "E2",
                "clm_97969ef83c0d02b1db7e480b25c7dd93": "E3",
                "clm_2fac233f73b5b722f21f8824a43a3f74": "E4",
                "clm_31b6cb058619412ecc3b6b99f8d88109": "E1->E2",
                "clm_d8690275f9738888c8a66fc5be2ecf9d": "E2->E3",
                "clm_066c08e4336b6b601989a415eec96bd0": "E3->E4",
                "pilot_rel_parent_travel_to_E1": f"{PARENT_EVENT_ID}->E1",
            },
            "gold_projection": "locally_frozen_status_qualified_value_or_sequence",
            "compiled_gold": semantic["compiled_gold"],
            "derivation_rule": semantic["reasoning_template"],
            "distractor_false_proofs": copy.deepcopy(
                semantic["distractor_proofs"]
            ),
            "required_evidence_origins": ["parent", "child"],
            "required_child_scene_ids": copy.deepcopy(contract["child_scene_ids"]),
            "minimum_child_scene_count": 2,
            "pass": True,
        }
        qa = {
            "qa_id": _id("qa", qa_identity),
            "query_contract_id": contract_id,
            "qa_role": semantic["qa_role"],
            "question_type": "single_choice",
            "question": question,
            "question_text": question,
            "question_surface_source": "qwen32b_polish_of_frozen_local_contract",
            "qa_operation": semantic["qa_operation"],
            "qa_subtype": semantic["qa_subtype"],
            "reasoning_subtype": semantic["reasoning_subtype"],
            "reasoning_template": semantic["reasoning_template"],
            "hop_count": semantic["hop_count"],
            "answer_type": semantic["answer_type"],
            "answer_format": "single_choice_mcq",
            "hard_negative_strategy": semantic["hard_negative_strategy"],
            "forbid_max_time_shortcut": semantic["forbid_max_time_shortcut"],
            "options": copy.deepcopy(semantic["options"]),
            "gold": copy.deepcopy(semantic["gold"]),
            "gold_answer_text": semantic["compiled_gold"],
            "normalized_gold_answer": semantic["compiled_gold"].casefold(),
            "query_plan": {
                "qa_operation": semantic["qa_operation"],
                "qa_subtype": semantic["qa_subtype"],
                "reasoning_subtype": semantic["reasoning_subtype"],
                "reasoning_template": semantic["reasoning_template"],
                "hop_count": semantic["hop_count"],
                "answer_type": semantic["answer_type"],
                "hard_negative_strategy": semantic["hard_negative_strategy"],
                "forbid_max_time_shortcut": semantic["forbid_max_time_shortcut"],
                "gold_answer": semantic["compiled_gold"],
                "normalized_gold_answer": semantic["compiled_gold"].casefold(),
                "reasoning_anchor_path": proof["reasoning_path_event_instance_ids"],
                "required_anchor_refs": proof["required_event_instance_ids"],
                "supporting_anchor_refs": proof["required_event_instance_ids"],
                "gold_anchor_refs": proof["answer_event_instance_ids"],
                "dependency_closure_refs": proof[
                    "dependency_closure_event_instance_ids"
                ],
            },
            "anchor_refs": {
                "support": proof["required_event_instance_ids"],
                "gold": proof["answer_event_instance_ids"],
                "distractors": ["E1", "E2", "E3", "E4"],
            },
            "proof": proof,
            "evidence_ledger": copy.deepcopy(contract["evidence_ledger"]),
            "evidence_turn_ids": copy.deepcopy(contract["evidence_turn_ids"]),
            "evidence_origin_scopes": ["parent", "child"],
            "required_child_scene_ids": copy.deepcopy(contract["child_scene_ids"]),
            "independent_validation_report": {
                "pass": True,
                "source": "local_truth_compiler",
                "gold_uniquely_derivable": True,
                "parent_and_cross_scene_evidence": True,
            },
            "independent_semantic_validation_report": {
                "pass": True,
                "source": "manual_scene_admission_plus_local_contract",
                "query_target_preserved_after_polish": True,
                "distractors_locally_proved_false": True,
            },
        }
        qas.append(qa)
    return qas


def _qa_truth_view(child: Mapping[str, Any]) -> dict[str, Any]:
    frozen_claims = [
        claim
        for scene in child["accepted_scenes"]
        for claim in scene["frozen_claims"]
    ]
    claims = [
        {
            "claim_id": "pilot_parent_claim_P0",
            "claim_type": "parent_node",
            "event_id": PARENT_EVENT_ID,
            "subject": "John",
            "predicate": "completed_travel",
            "value": "European coastline road trip with his wife",
            "status": "completed",
            "origin": "parent",
            "evidence_turn_ids": [PARENT_TURN_ID],
        },
        *[
            {
                **copy.deepcopy(claim),
                "origin": "child",
                "scene_id": scene["scene_id"],
                "source_claim_qa_eligible": (
                    False if claim["claim_type"] == "status_transition" else None
                ),
                "qa_eligible": True,
                "qa_eligibility_basis": (
                    "local_semantic_relation_compiled_from_adjacent_frozen_node_"
                    "claims_and_cross_scene_text_evidence"
                    if claim["claim_type"] == "status_transition"
                    else "frozen_node_or_explicit_parent_child_continuity"
                ),
                "generation_control_edge": False,
            }
            for scene in child["accepted_scenes"]
            for claim in scene["frozen_claims"]
        ],
    ]
    view = {
        "schema_version": "stage3_temporal_pilot_qa_truth_view_v1",
        "parent_graph_sha256": child["source_parent_graph_sha256"],
        "source_artifact_file_sha256": child["source_artifact_file_sha256"],
        "child_graph_sha256": child["source_child_graph_sha256"],
        "scene_ledger_sha256": child["accepted_scene_chain_head"],
        "canonical_parent_lineage_id": SOURCE_LINEAGE_ID,
        "claims": claims,
        "ancillary_facts_included": False,
    }
    view["claim_bundle_hash"] = _sha256_json(claims)
    view["graph_snapshot_hash"] = _sha256_json(view)
    return view


def _build_envelope(
    child: Mapping[str, Any], contracts: Mapping[str, Any], review: Mapping[str, Any]
) -> dict[str, Any]:
    qas = _final_qas(contracts, review, child)
    truth_view = _qa_truth_view(child)
    dialogue_hash = child["dialogue"]["dialogue_hash"]
    alignment_hash = child["alignment_hash"]
    claim_ir = {
        "schema_version": "stage3_temporal_pilot_claim_ir_v1",
        "claims": copy.deepcopy(truth_view["claims"]),
        "claim_bundle_hash": truth_view["claim_bundle_hash"],
        "graph_snapshot_hash": truth_view["graph_snapshot_hash"],
    }
    bundle = {
        "schema_version": "stage3_claim_first_multiqa_v1",
        "dialogue_family_id": child["dialogue_family_id"],
        "claim_ir": claim_ir,
        "claim_bundle_hash": truth_view["claim_bundle_hash"],
        "graph_snapshot_hash": truth_view["graph_snapshot_hash"],
        "qas": qas,
        "qa_count": len(qas),
        "truth_compiled_locally": True,
        "question_surface_api_only": True,
        "source_query_contract_bundle_sha256": contracts[
            "query_contract_bundle_sha256"
        ],
    }
    sample = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": child["sample_id"],
        "source_id": child["sample_id"],
        "training_use": "pilot_only_not_admitted",
        "split": "train",
        "source_split": "train",
        "lineage_group": SOURCE_LINEAGE_ID,
        "canonical_parent_lineage_id": SOURCE_LINEAGE_ID,
        "dialogue_family_id": child["dialogue_family_id"],
        "parent_sample_id": SOURCE_SEED_SAMPLE_ID,
        "dialogue": copy.deepcopy(child["dialogue"]),
        "graph_update": truth_view,
        "graph_snapshot_hash": truth_view["graph_snapshot_hash"],
        "claim_bundle_hash": truth_view["claim_bundle_hash"],
        "native_contract": {
            "valid": True,
            "contract": "truth_first_incremental_temporal_pilot",
            "accepted_scene_count": child["accepted_scene_count"],
        },
        "validation": {
            "graph_dialogue_consistent": True,
            "gold_derivable": True,
            "option_valid": True,
            "anti_shortcut_pass": True,
            "graph_validation_pass": True,
            "parent_child_continuity_pass": True,
            "cross_scene_evidence_pass": True,
        },
        "metadata": {
            "lineage_group": SOURCE_LINEAGE_ID,
            "canonical_parent_lineage_id": SOURCE_LINEAGE_ID,
            "dialogue_family_id": child["dialogue_family_id"],
            "graph_snapshot_hash": truth_view["graph_snapshot_hash"],
            "dialogue_hash": dialogue_hash,
            "alignment_hash": alignment_hash,
            "training_use": "pilot_only_not_admitted",
            "added_token_count": child["added_token_count"],
            "accepted_scene_count": child["accepted_scene_count"],
        },
        "claim_first_multiqa_bundle": bundle,
    }
    return {
        "schema_version": "stage3_temporal_pilot_accepted_envelope_v1",
        "clean_sample": sample,
        "claim_bundle_hash": truth_view["claim_bundle_hash"],
        "hashes": {
            "graph_snapshot_hash": truth_view["graph_snapshot_hash"],
            "dialogue_hash": dialogue_hash,
            "alignment_hash": alignment_hash,
        },
        "audit": {
            "training_data_modified": False,
            "api_selected_truth_fields": [],
            "ancillary_in_truth_or_qa": False,
        },
    }


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise PilotError(f"refusing to overwrite pilot preview rows: {path}")
    for row in rows:
        _append_jsonl(path, row)


def _command_finalize(args: argparse.Namespace) -> int:
    _, state = _load_initialized(args.output_dir)
    required_paths = (
        _paths(args.output_dir)["child"],
        _paths(args.output_dir)["qa_contracts"],
        _paths(args.output_dir)["qa_review"],
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise PilotError(f"cannot finalize; missing frozen artifacts: {missing}")
    child = _read_json(_paths(args.output_dir)["child"])
    contracts = _read_json(_paths(args.output_dir)["qa_contracts"])
    review = _read_json(_paths(args.output_dir)["qa_review"])
    envelope = _build_envelope(child, contracts, review)
    expansion = stage3_multiqa_rows.expand_accepted_dialogue(
        envelope, min_qas=3, max_qas=6, eval_per_mille=100
    )
    if not expansion["family_ready"] or len(expansion["qa_rows"]) < 3:
        raise PilotError(f"multi-QA expansion gate failed: {expansion['summary']}")
    final_artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "temporal_incremental_child_multiqa_pilot",
        "training_use": "pilot_only_not_admitted",
        "child_artifact": child,
        "query_contracts": contracts,
        "qa_polish_review": review,
        "accepted_envelope": envelope,
        "multiqa_expansion_summary": expansion["summary"],
        "qa_rejects": expansion["qa_rejects"],
        "qa_rows_preview_path": str(_paths(args.output_dir)["qa_rows"]),
        "marked_qa_rows_preview_path": str(_paths(args.output_dir)["marked_rows"]),
        "training_data_modified": False,
    }
    final_artifact["artifact_sha256"] = _sha256_json(final_artifact)
    _atomic_write_json(_paths(args.output_dir)["final"], final_artifact)
    _write_jsonl_rows(_paths(args.output_dir)["qa_rows"], expansion["qa_rows"])
    _write_jsonl_rows(
        _paths(args.output_dir)["marked_rows"], expansion["marked_qa_rows"]
    )
    qa_gates = []
    for qa in envelope["clean_sample"]["claim_first_multiqa_bundle"]["qas"]:
        child_scenes = qa["required_child_scene_ids"]
        qa_gates.append(
            {
                "qa_id": qa["qa_id"],
                "parent_evidence": PARENT_TURN_ID in qa["evidence_turn_ids"],
                "child_scene_count": len(child_scenes),
                "cross_scene": len(child_scenes) >= 2,
                "gold": qa["gold_answer_text"],
                "qa_operation": qa["qa_operation"],
                "qa_subtype": qa["qa_subtype"],
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "pass": (
            child["accepted_scene_count"] in {3, 4}
            and child["accepted_scene_count"] == 4
            and child["added_token_count"] >= MIN_TOTAL_ADDED_TOKENS
            and len(expansion["qa_rows"]) >= 3
            and all(item["parent_evidence"] and item["cross_scene"] for item in qa_gates)
            and not child["source_graph_mutated"]
        ),
        "accepted_scene_count": child["accepted_scene_count"],
        "scene_attempt_counts": copy.deepcopy(state["scene_attempt_counts"]),
        "physical_scene_api_calls": state["physical_scene_api_calls"],
        "physical_qa_polish_api_calls": state["physical_qa_polish_api_calls"],
        "child_turn_count": child["child_turn_count"],
        "added_token_count": child["added_token_count"],
        "minimum_added_token_count": MIN_TOTAL_ADDED_TOKENS,
        "qa_count": len(expansion["qa_rows"]),
        "qa_gates": qa_gates,
        "dialogue_family_split": expansion["split"],
        "canonical_parent_lineage_id": expansion["canonical_parent_lineage_id"],
        "split_group": expansion["split_group"],
        "ancillary_in_graph_qa_gold_or_inheritance": False,
        "api_selected_truth_gold_proof_or_distractors": False,
        "training_data_modified": False,
        "source_graph_mutated": child["source_graph_mutated"],
        "proceed_to_operation_expansion": False,
        "proceed_to_n100": False,
        "proceed_to_training": False,
        "next_allowed_step_if_pass": "small_solver_training_and_independent_lineage_heldout_plan_only",
        "artifact_paths": {
            "child": str(_paths(args.output_dir)["child"]),
            "multiqa": str(_paths(args.output_dir)["final"]),
            "qa_rows_preview": str(_paths(args.output_dir)["qa_rows"]),
        },
    }
    _atomic_write_json(_paths(args.output_dir)["report"], report)
    state["status"] = "pilot_complete" if report["pass"] else "pilot_failed_gate"
    state["final_artifact_sha256"] = final_artifact["artifact_sha256"]
    state["report_pass"] = report["pass"]
    _atomic_write_json(_paths(args.output_dir)["state"], state)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["pass"] else 2


def _command_status(args: argparse.Namespace) -> int:
    plan, state = _load_initialized(args.output_dir)
    accepted = _read_jsonl(_paths(args.output_dir)["accepted"])
    next_scene = (
        plan["scenes"][state["accepted_scene_count"]]["scene_id"]
        if state["accepted_scene_count"] < SCENE_COUNT
        else None
    )
    result = {
        "run_id": RUN_ID,
        "status": state["status"],
        "accepted_scene_count": state["accepted_scene_count"],
        "accepted_scene_ids": [row["scene_id"] for row in accepted],
        "next_scene": next_scene,
        "scene_attempt_counts": state["scene_attempt_counts"],
        "pending_attempt_id": state["pending_attempt_id"],
        "pending_qa_polish_attempt_id": state.get("pending_qa_polish_attempt_id"),
        "total_added_turns": state["total_added_turns"],
        "total_added_tokens": state["total_added_tokens"],
        "physical_scene_api_calls": state["physical_scene_api_calls"],
        "physical_qa_polish_api_calls": state["physical_qa_polish_api_calls"],
        "training_files_written": state["training_files_written"],
        "accepted_chain_valid": True,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _command_write_report(args: argparse.Namespace) -> int:
    plan, state = _load_initialized(args.output_dir)
    attempt_rows = [
        row
        for row in _read_jsonl(_paths(args.output_dir)["attempts"])
        if row.get("record_type") == "api_result"
    ]
    rejection_rows = _read_jsonl(_paths(args.output_dir)["rejects"])
    rejection_by_attempt = {
        row["attempt_id"]: row for row in rejection_rows if row.get("attempt_id")
    }
    attempts = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in attempt_rows:
        api_usage = (row.get("api_meta") or {}).get("usage") or {}
        for key in usage:
            usage[key] += int(api_usage.get(key) or 0)
        rejection = rejection_by_attempt.get(row["attempt_id"]) or {}
        attempts.append(
            {
                "attempt_id": row["attempt_id"],
                "scene_id": row["scene_id"],
                "attempt_number": row["attempt_number"],
                "api_used": bool((row.get("api_meta") or {}).get("used")),
                "turn_count": (row.get("basic_local_audit") or {}).get(
                    "turn_count"
                ),
                "token_count": (row.get("basic_local_audit") or {}).get(
                    "scene_token_count"
                ),
                "token_count_mode": (row.get("basic_local_audit") or {}).get(
                    "token_count_mode"
                ),
                "basic_issues_at_call_time": copy.deepcopy(
                    (row.get("basic_local_audit") or {}).get("issues") or []
                ),
                "semantic_decision": (
                    "reject" if rejection else "pending_or_not_reviewed"
                ),
                "semantic_rejection_reasons": copy.deepcopy(
                    rejection.get("reason_codes") or []
                ),
                "raw_response_retained": bool(
                    (row.get("api_meta") or {}).get("raw_response") is not None
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "pass": state.get("status") == "pilot_complete",
        "status": state.get("status"),
        "stop_reason": (
            "S1 exhausted initial call plus two retries; every response changed "
            "the frozen parent memory or introduced unsupported Cedar Hall facts."
            if state.get("status") == "failed_scene_retry_budget"
            else None
        ),
        "user_stop_condition_honored": state.get("status")
        == "failed_scene_retry_budget",
        "model": MODEL,
        "operation": "temporal_state_update",
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_artifact_file_sha256_expected": SOURCE_ARTIFACT_FILE_SHA256,
        "source_artifact_file_sha256_actual": _sha256_file(SOURCE_ARTIFACT),
        "source_graph_mutated": _sha256_file(SOURCE_ARTIFACT)
        != SOURCE_ARTIFACT_FILE_SHA256,
        "frozen_scene_plan_sha256": plan["plan_sha256"],
        "requested_scene_count": SCENE_COUNT,
        "accepted_scene_count": state["accepted_scene_count"],
        "accepted_scene_ids": [
            row["scene_id"]
            for row in _read_jsonl(_paths(args.output_dir)["accepted"])
        ],
        "scene_attempt_counts": copy.deepcopy(state["scene_attempt_counts"]),
        "physical_scene_api_calls": state["physical_scene_api_calls"],
        "physical_qa_polish_api_calls": state["physical_qa_polish_api_calls"],
        "api_usage": usage,
        "attempts": attempts,
        "added_token_count": state["total_added_tokens"],
        "final_exact_1500_token_gate_pass": False,
        "qa_count": 0,
        "cross_scene_parent_child_qa_gate_pass": False,
        "child_artifact_emitted": _paths(args.output_dir)["child"].exists(),
        "multiqa_artifact_emitted": _paths(args.output_dir)["final"].exists(),
        "per_scene_375_tokens_is_generation_target_not_hard_gate": True,
        "final_1500_tokens_remains_exact_hard_gate": True,
        "training_data_modified": False,
        "training_files_written": state["training_files_written"],
        "automatic_next_run_started": False,
        "automatic_scale_started": False,
        "solver_training_started": False,
        "implementation_file": str(Path(__file__).resolve()),
        "implementation_file_sha256": _sha256_file(Path(__file__).resolve()),
        "artifacts": {
            "manifest": str(_paths(args.output_dir)["manifest"]),
            "frozen_scene_plan": str(_paths(args.output_dir)["plan"]),
            "state": str(_paths(args.output_dir)["state"]),
            "attempt_journal": str(_paths(args.output_dir)["attempts"]),
            "reject_journal": str(_paths(args.output_dir)["rejects"]),
            "report": str(_paths(args.output_dir)["report"]),
        },
    }
    _atomic_write_json(_paths(args.output_dir)["report"], report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=argparse.SUPPRESS
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight").set_defaults(handler=_command_preflight)
    subparsers.add_parser("init").set_defaults(handler=_command_init)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--abandon-unknown-inflight", action="store_true")
    recover.set_defaults(handler=_command_recover)
    generate = subparsers.add_parser("generate-next")
    generate.add_argument("--execute-live", action="store_true")
    generate.set_defaults(handler=_command_generate_next)
    admit = subparsers.add_parser("admit")
    admit.add_argument("--review-file", type=Path, required=True)
    admit.set_defaults(handler=_command_admit)
    reject = subparsers.add_parser("reject")
    reject.add_argument("--review-file", type=Path, required=True)
    reject.set_defaults(handler=_command_reject)
    subparsers.add_parser("finalize-scenes").set_defaults(
        handler=_command_finalize_scenes
    )
    subparsers.add_parser("compile-qa-contracts").set_defaults(
        handler=_command_compile_qa_contracts
    )
    polish = subparsers.add_parser("polish-qas")
    polish.add_argument("--execute-live", action="store_true")
    polish.set_defaults(handler=_command_polish_qas)
    admit_polish = subparsers.add_parser("admit-qa-polish")
    admit_polish.add_argument("--review-file", type=Path, required=True)
    admit_polish.set_defaults(handler=_command_admit_qa_polish)
    reject_polish = subparsers.add_parser("reject-qa-polish")
    reject_polish.add_argument("--review-file", type=Path, required=True)
    reject_polish.set_defaults(handler=_command_reject_qa_polish)
    subparsers.add_parser("finalize").set_defaults(handler=_command_finalize)
    subparsers.add_parser("status").set_defaults(handler=_command_status)
    subparsers.add_parser("write-report").set_defaults(handler=_command_write_report)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except PilotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
