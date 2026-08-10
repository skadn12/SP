"""World Memory v1 initialization and guarded semantic commits."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes
from .schemas import (
    WORLD_MEMORY_SCHEMA_VERSION,
    validate_evidence_committer_output,
    validate_world_memory,
)


def _entity_id(root_id: str, display_name: str) -> str:
    suffix = sha256_bytes(display_name.strip().casefold().encode())[:12]
    return f"{root_id}:entity:{suffix}"


def initialize_world_memory(
    root: Mapping[str, Any], parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Create source-grounded memory without deriving latent facts from text."""
    root_id = str(root.get("root_id") or "")
    parent_dialogue = parent.get("dialogue") if isinstance(parent, Mapping) else None
    turns = (parent_dialogue or {}).get("turns") if isinstance(parent_dialogue, Mapping) else None
    if not root_id or not isinstance(turns, list) or not turns:
        raise ValueError("world_memory_parent_contract_invalid")
    if parent.get("sample_id") != root_id:
        raise ValueError("world_memory_parent_root_identity_mismatch")
    dialogue_hash = str((parent_dialogue or {}).get("dialogue_hash") or "")
    if dialogue_hash != (root.get("dialogue") or {}).get("dialogue_hash"):
        raise ValueError("world_memory_parent_dialogue_hash_mismatch")

    speaker_turns: dict[str, list[str]] = {}
    for turn in turns:
        speaker = str(turn.get("speaker") or "")
        turn_id = str(turn.get("turn_id") or "")
        if not speaker or not turn_id:
            raise ValueError("world_memory_parent_turn_invalid")
        speaker_turns.setdefault(speaker, []).append(turn_id)
    entities = []
    entity_by_speaker: dict[str, str] = {}
    for speaker, source_turn_ids in speaker_turns.items():
        entity_id = _entity_id(root_id, speaker)
        entity_by_speaker[speaker] = entity_id
        entities.append({
            "entity_id": entity_id,
            "display_name": speaker,
            "source_turn_ids": source_turn_ids,
            "status": "active",
            "created_round": 0,
            "last_updated_round": 0,
            "provenance": {
                "root_id": root_id,
                "dialogue_hash": dialogue_hash,
                "source": "parent_speaker_identity",
            },
            "semantic_status": "source_grounded",
        })
    event_history = []
    for index, turn in enumerate(turns, 1):
        turn_id = str(turn["turn_id"])
        event_history.append({
            "memory_id": f"{root_id}:M{index:06d}",
            "event_identity_key": f"{root_id}:parent_utterance:{turn_id}",
            "record_type": "parent_utterance",
            "subject_ids": [entity_by_speaker[str(turn["speaker"])]],
            "value": str(turn["text"]),
            "source_turn_ids": [turn_id],
            "status": "active",
            "created_round": 0,
            "last_updated_round": 0,
            "provenance": {
                "root_id": root_id,
                "dialogue_hash": dialogue_hash,
                "partition": "parent",
                "source_path": str((root.get("source") or {}).get("source_path") or ""),
                "source_record_sha256": str(
                    (root.get("source") or {}).get("source_record_sha256") or ""),
            },
            "semantic_status": "source_grounded",
            "handoff_id": None,
        })
    memory = {
        "world_memory_schema_version": WORLD_MEMORY_SCHEMA_VERSION,
        "root_id": root_id,
        "memory_version": 1,
        "entities": entities,
        "relations": [],
        "event_history": event_history,
        "current_states": [],
        "knowledge_visibility": [],
        "open_threads": [],
        "completed_threads": [],
        "memory_update_log": [],
        "parent_child_lineage": [{
            "parent_root_id": root_id,
            "parent_dialogue_sha256": dialogue_hash,
            "child_ids": [],
            "created_round": 0,
            "provenance": "authoritative_parent_input",
        }],
        "pending_memory_updates": [],
    }
    validation = validate_world_memory(memory)
    if not validation["valid"]:
        raise RuntimeError("initialized_world_memory_invalid:" +
                           ",".join(validation["errors"]))
    return memory


def stage_model_claims(
    memory: Mapping[str, Any], deltas: list[Mapping[str, Any]],
    realizer_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain Realizer claims for review without changing authoritative state."""
    result = copy.deepcopy(dict(memory))
    delta_by_id = {str(delta["proposed_delta_id"]): delta for delta in deltas}
    for claim in realizer_output.get("claimed_graph_updates") or []:
        delta_id = str(claim.get("proposed_delta_id") or "")
        delta = delta_by_id.get(delta_id)
        if delta is None:
            raise ValueError(f"realizer_claim_unknown_delta:{delta_id}")
        result["pending_memory_updates"].append({
            "proposed_delta_id": delta_id,
            "event_identity_key": delta["event_identity_key"],
            "claimed_status": claim["claimed_status"],
            "intended_post_state": delta["intended_post_state"],
            "candidate_evidence_turn_keys": list(
                claim["candidate_evidence_turn_keys"]),
            "source_scene": realizer_output["scene_key"],
            "semantic_status": "model_claimed",
            "training_eligible": False,
            "provenance": {
                "role": "natural_realizer",
                "authority": "non_authoritative_claim",
            },
        })
    return result


def stage_emergent_claims(
    memory: Mapping[str, Any], proposals: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stage emergent thread proposals without assigning authoritative IDs."""
    result = copy.deepcopy(dict(memory))
    for proposal in proposals:
        if (proposal.get("root_id") != result.get("root_id") or
                proposal.get("semantic_status") != "model_claimed" or
                proposal.get("formal_thread_id") is not None or
                proposal.get("authoritative") is not False):
            raise ValueError("emergent_proposal_not_non_authoritative")
        result["pending_memory_updates"].append({
            "proposal_id": proposal["proposal_id"],
            "event_identity_key": proposal["event_identity_key"],
            "operation": "open_thread",
            "description": proposal["description"],
            "candidate_evidence_turn_keys": list(
                proposal["candidate_evidence_turn_keys"]),
            "source_scene": proposal["source_scene"],
            "semantic_status": "model_claimed",
            "decision": None,
            "formal_thread_id": None,
            "authoritative_applied": False,
            "training_eligible": False,
            "provenance": {
                "role": "natural_realizer",
                "authority": "non_authoritative_emergent_claim",
            },
        })
    return result


def _all_memory_ids(memory: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("event_history", "current_states", "open_threads",
                  "completed_threads"):
        for record in memory.get(field) or []:
            memory_id = record.get("memory_id")
            if isinstance(memory_id, str):
                values.add(memory_id)
    return values


def _next_memory_id(memory: Mapping[str, Any]) -> str:
    root_id = str(memory["root_id"])
    existing = _all_memory_ids(memory)
    index = len(existing) + 1
    while f"{root_id}:M{index:06d}" in existing:
        index += 1
    return f"{root_id}:M{index:06d}"


def commit_confirmed_delta(
    memory: Mapping[str, Any], delta: Mapping[str, Any],
    committer_output: Mapping[str, Any], binding: Mapping[str, Any], *,
    scene_key: str, round_index: int, dry_run: bool = False,
) -> dict[str, Any]:
    """Apply only independently confirmed, structurally bound semantic truth."""
    validation = validate_evidence_committer_output(committer_output)
    if not validation["valid"]:
        raise ValueError("evidence_committer_invalid:" +
                         ",".join(validation["errors"]))
    delta_id = str(delta.get("proposed_delta_id") or "")
    if committer_output["proposed_delta_id"] != delta_id:
        raise ValueError("evidence_committer_delta_identity_mismatch")
    result = copy.deepcopy(dict(memory))
    if (committer_output["decision"] != "confirmed" or
            committer_output["semantic_status"] != "semantically_verified"):
        pending_record = {
            "proposed_delta_id": delta_id,
            "event_identity_key": delta["event_identity_key"],
            "decision": committer_output["decision"],
            "intended_post_state": delta["intended_post_state"],
            "semantic_status": committer_output["semantic_status"],
            "training_eligible": False,
            "source_scene": scene_key,
            "provenance": {"role": "evidence_committer"},
        }
        result["pending_memory_updates"].append(pending_record)
        result["memory_update_log"].append({
            "update_version": result["memory_version"],
            "proposed_delta_id": delta_id,
            "event_identity_key": delta["event_identity_key"],
            "old_state": None,
            "new_state": None,
            "supporting_turn_ids": [],
            "source_scene": scene_key,
            "superseded_memory_ids": [],
            "new_memory_id": None,
            "history_event_memory_id": None,
            "decision": committer_output["decision"],
            "semantic_status": committer_output["semantic_status"],
            "authoritative_applied": False,
            "training_eligible": False,
            "provenance": {"role": "evidence_committer", "dry_run": dry_run},
        })
        return result

    evidence_binding = next((row for row in binding.get("evidence_bindings") or []
                             if row.get("proposed_delta_id") == delta_id), None)
    if not evidence_binding or not evidence_binding.get("evidence_turn_ids"):
        raise ValueError("confirmed_delta_missing_structural_evidence_binding")
    final_turn_ids = list(evidence_binding["evidence_turn_ids"])
    event_key = str(delta["event_identity_key"])
    old_records = [record for record in result["current_states"]
                   if record["event_identity_key"] == event_key]
    result["current_states"] = [record for record in result["current_states"]
                                if record["event_identity_key"] != event_key]
    for old in old_records:
        superseded = copy.deepcopy(old)
        superseded["status"] = "superseded"
        superseded["last_updated_round"] = round_index
        superseded["provenance"] = {
            **dict(superseded["provenance"]),
            "superseded_by_delta_id": delta_id,
        }
        result["event_history"].append(superseded)

    operation = str(delta["operation"])
    handoff = next((row for row in binding.get("allocated_handoffs") or []
                    if row.get("proposed_delta_id") == delta_id), None)
    thread = next((row for row in binding.get("allocated_threads") or []
                   if row.get("proposed_delta_id") == delta_id), None)
    if operation == "open_thread" and thread is None:
        raise ValueError("confirmed_open_thread_missing_program_thread_id")
    status = "completed" if operation in {"close_thread", "resolve_conflict"} else "active"
    new_record = {
        "memory_id": _next_memory_id(result),
        "event_identity_key": event_key,
        "record_type": operation,
        "subject_ids": list(delta["subject_ids"]),
        "value": str(committer_output["confirmed_post_state"]),
        "source_turn_ids": final_turn_ids,
        "status": status,
        "created_round": round_index,
        "last_updated_round": round_index,
        "provenance": {
            "root_id": result["root_id"],
            "source_scene": scene_key,
            "proposed_delta_id": delta_id,
            "child_id": binding["child_id"],
            "evidence_text_sha256": list(evidence_binding["evidence_text_sha256"]),
            "formal_thread_id": thread.get("thread_id") if thread else None,
            "dry_run": dry_run,
        },
        "semantic_status": "semantically_verified",
        "handoff_id": handoff.get("handoff_id") if handoff else None,
    }
    result["current_states"].append(new_record)
    history_event = copy.deepcopy(new_record)
    history_event["memory_id"] = _next_memory_id(result)
    history_event["record_type"] = f"{operation}_event"
    result["event_history"].append(history_event)
    if operation == "open_thread":
        result["open_threads"].append(copy.deepcopy(new_record))
    elif operation == "close_thread":
        closed = [row for row in result["open_threads"]
                  if row["event_identity_key"] == event_key]
        result["open_threads"] = [row for row in result["open_threads"]
                                  if row["event_identity_key"] != event_key]
        result["completed_threads"].extend(closed or [copy.deepcopy(new_record)])
    result["memory_version"] += 1
    result["memory_update_log"].append({
        "update_version": result["memory_version"],
        "proposed_delta_id": delta_id,
        "event_identity_key": event_key,
        "old_state": [row["value"] for row in old_records] or None,
        "new_state": new_record["value"],
        "supporting_turn_ids": final_turn_ids,
        "source_scene": scene_key,
        "superseded_memory_ids": [row["memory_id"] for row in old_records],
        "new_memory_id": new_record["memory_id"],
        "history_event_memory_id": history_event["memory_id"],
        "semantic_status": "semantically_verified",
        "authoritative_applied": True,
        "training_eligible": True,
        "provenance": {
            "child_id": binding["child_id"],
            "binding_sha256": sha256_bytes(canonical_json(binding).encode()),
            "dry_run": dry_run,
        },
    })
    lineage = result["parent_child_lineage"][0]
    if binding["child_id"] not in lineage["child_ids"]:
        lineage["child_ids"].append(binding["child_id"])
    validated = validate_world_memory(result)
    if not validated["valid"]:
        raise RuntimeError("committed_world_memory_invalid:" +
                           ",".join(validated["errors"]))
    return result


def commit_emergent_thread(
    memory: Mapping[str, Any], proposal: Mapping[str, Any],
    committer_output: Mapping[str, Any], binding: Mapping[str, Any], *,
    round_index: int, dry_run: bool = False,
) -> dict[str, Any]:
    """Commit an emergent open thread only after independent confirmation."""
    validation = validate_evidence_committer_output(committer_output)
    if not validation["valid"]:
        raise ValueError("emergent_evidence_committer_invalid:" +
                         ",".join(validation["errors"]))
    proposal_id = str(proposal.get("proposal_id") or "")
    if committer_output.get("proposed_delta_id") != proposal_id:
        raise ValueError("emergent_committer_proposal_identity_mismatch")
    if (proposal.get("root_id") != memory.get("root_id") or
            proposal.get("operation") != "open_thread" or
            proposal.get("semantic_status") != "model_claimed" or
            proposal.get("formal_thread_id") is not None or
            proposal.get("authoritative") is not False):
        raise ValueError("emergent_proposal_contract_invalid")

    result = copy.deepcopy(dict(memory))
    scene_key = str(proposal["source_scene"])
    if (committer_output["decision"] != "confirmed" or
            committer_output["semantic_status"] != "semantically_verified"):
        pending_record = {
            "proposal_id": proposal_id,
            "event_identity_key": proposal["event_identity_key"],
            "operation": "open_thread",
            "description": proposal["description"],
            "decision": committer_output["decision"],
            "semantic_status": committer_output["semantic_status"],
            "formal_thread_id": None,
            "authoritative_applied": False,
            "training_eligible": False,
            "source_scene": scene_key,
            "provenance": {"role": "evidence_committer", "route": "emergent"},
        }
        result["pending_memory_updates"].append(pending_record)
        result["memory_update_log"].append({
            "update_version": result["memory_version"],
            "proposal_id": proposal_id,
            "event_identity_key": proposal["event_identity_key"],
            "old_state": None,
            "new_state": None,
            "supporting_turn_ids": [],
            "source_scene": scene_key,
            "superseded_memory_ids": [],
            "new_memory_id": None,
            "history_event_memory_id": None,
            "formal_thread_id": None,
            "decision": committer_output["decision"],
            "semantic_status": committer_output["semantic_status"],
            "authoritative_applied": False,
            "training_eligible": False,
            "provenance": {
                "role": "evidence_committer", "route": "emergent",
                "dry_run": dry_run,
            },
        })
        return result

    evidence = next((row for row in binding.get("emergent_evidence_bindings") or []
                     if row.get("proposal_id") == proposal_id), None)
    thread = next((row for row in binding.get("allocated_threads") or []
                   if row.get("proposal_id") == proposal_id), None)
    if not evidence or not evidence.get("evidence_turn_ids"):
        raise ValueError("confirmed_emergent_thread_missing_evidence_binding")
    if not thread or not thread.get("thread_id"):
        raise ValueError("confirmed_emergent_thread_missing_program_thread_id")

    new_record = {
        "memory_id": _next_memory_id(result),
        "event_identity_key": str(proposal["event_identity_key"]),
        "record_type": "open_thread",
        "subject_ids": list(proposal["subject_ids"]),
        "value": str(committer_output["confirmed_post_state"]),
        "source_turn_ids": list(evidence["evidence_turn_ids"]),
        "status": "active",
        "created_round": round_index,
        "last_updated_round": round_index,
        "provenance": {
            "root_id": result["root_id"],
            "source_scene": scene_key,
            "proposal_id": proposal_id,
            "child_id": binding["child_id"],
            "formal_thread_id": thread["thread_id"],
            "evidence_text_sha256": list(evidence["evidence_text_sha256"]),
            "route": "emergent_open_thread",
            "dry_run": dry_run,
        },
        "semantic_status": "semantically_verified",
        "handoff_id": None,
    }
    result["current_states"].append(new_record)
    result["open_threads"].append(copy.deepcopy(new_record))
    history_event = copy.deepcopy(new_record)
    history_event["memory_id"] = _next_memory_id(result)
    history_event["record_type"] = "open_thread_event"
    result["event_history"].append(history_event)
    result["memory_version"] += 1
    result["memory_update_log"].append({
        "update_version": result["memory_version"],
        "proposal_id": proposal_id,
        "event_identity_key": proposal["event_identity_key"],
        "old_state": None,
        "new_state": new_record["value"],
        "supporting_turn_ids": list(evidence["evidence_turn_ids"]),
        "source_scene": scene_key,
        "superseded_memory_ids": [],
        "new_memory_id": new_record["memory_id"],
        "history_event_memory_id": history_event["memory_id"],
        "formal_thread_id": thread["thread_id"],
        "decision": "confirmed",
        "semantic_status": "semantically_verified",
        "authoritative_applied": True,
        "training_eligible": True,
        "provenance": {
            "child_id": binding["child_id"],
            "binding_sha256": sha256_bytes(canonical_json(binding).encode()),
            "route": "emergent_open_thread", "dry_run": dry_run,
        },
    })
    lineage = result["parent_child_lineage"][0]
    if binding["child_id"] not in lineage["child_ids"]:
        lineage["child_ids"].append(binding["child_id"])
    validated = validate_world_memory(result)
    if not validated["valid"]:
        raise RuntimeError("committed_emergent_world_memory_invalid:" +
                           ",".join(validated["errors"]))
    return result
