"""Bind response-local references to program-owned child/session/turn IDs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes
from .schemas import (
    validate_evidence_committer_output,
    validate_natural_realizer_output,
)


LOCAL_STRUCTURAL_BINDER_VERSION = "local_structural_binder_v2"


class StructuralBindingError(ValueError):
    pass


def bind_scene(
    *, root_id: str, round_index: int, scene_goal: Mapping[str, Any],
    realizer_output: Mapping[str, Any], deltas: Sequence[Mapping[str, Any]],
    committer_outputs: Sequence[Mapping[str, Any]], handoff_start_index: int = 1,
    emergent_proposals: Sequence[Mapping[str, Any]] = (),
    emergent_committer_outputs: Sequence[Mapping[str, Any]] = (),
    thread_start_index: int = 1,
) -> dict[str, Any]:
    realizer_validation = validate_natural_realizer_output(realizer_output)
    if not realizer_validation["valid"]:
        raise StructuralBindingError("realizer_output_invalid:" +
                                     ",".join(realizer_validation["errors"]))
    scene_key = str(scene_goal.get("scene_key") or "")
    if realizer_output.get("scene_key") != scene_key:
        raise StructuralBindingError("scene_key_mismatch")
    participants = set(str(value) for value in scene_goal.get("participants") or [])
    child_id = f"{root_id}:child:R{round_index:02d}:{scene_key}"
    session_id = f"{child_id}:session"
    key_to_turn: dict[str, dict[str, Any]] = {}
    bound_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(realizer_output["dialogue_turns"], 1):
        speaker = str(turn["speaker"])
        if participants and speaker not in participants:
            raise StructuralBindingError(f"speaker_not_in_scene_participants:{speaker}")
        turn_key = str(turn["turn_key"])
        final_turn_id = f"{session_id}:T{index:03d}"
        bound = {
            "turn_id": final_turn_id,
            "session_id": session_id,
            "speaker": speaker,
            "text": str(turn["text"]),
            "partition": "child",
            "turn_index": index,
            "source_turn_key": turn_key,
            "text_sha256": sha256_bytes(str(turn["text"]).encode()),
        }
        key_to_turn[turn_key] = bound
        bound_turns.append(bound)

    delta_by_id = {str(delta["proposed_delta_id"]): delta for delta in deltas}
    if len(delta_by_id) != len(deltas):
        raise StructuralBindingError("duplicate_proposed_delta_id")
    for claim in realizer_output["claimed_graph_updates"]:
        if claim["proposed_delta_id"] not in delta_by_id:
            raise StructuralBindingError(
                f"realizer_claim_unknown_delta:{claim['proposed_delta_id']}")
        missing = [key for key in claim["candidate_evidence_turn_keys"]
                   if key not in key_to_turn]
        if missing:
            raise StructuralBindingError(
                "realizer_claim_evidence_turn_key_missing:" + ",".join(missing))
    for claim in realizer_output.get("new_open_thread_claims") or []:
        missing = [key for key in claim["candidate_evidence_turn_keys"]
                   if key not in key_to_turn]
        if missing:
            raise StructuralBindingError(
                "open_thread_evidence_turn_key_missing:" + ",".join(missing))

    evidence_bindings: list[dict[str, Any]] = []
    seen_committer_ids: set[str] = set()
    confirmed_handoff_deltas: list[str] = []
    confirmed_open_thread_deltas: list[str] = []
    for output in committer_outputs:
        validation = validate_evidence_committer_output(output)
        if not validation["valid"]:
            raise StructuralBindingError("evidence_committer_output_invalid:" +
                                         ",".join(validation["errors"]))
        delta_id = str(output["proposed_delta_id"])
        if delta_id not in delta_by_id:
            raise StructuralBindingError(f"committer_unknown_delta:{delta_id}")
        if delta_id in seen_committer_ids:
            raise StructuralBindingError(f"duplicate_committer_delta:{delta_id}")
        seen_committer_ids.add(delta_id)
        missing = [key for key in output["evidence_turn_keys"] if key not in key_to_turn]
        if missing:
            raise StructuralBindingError(
                "evidence_turn_key_missing:" + ",".join(missing))
        evidence_bindings.append({
            "proposed_delta_id": delta_id,
            "decision": output["decision"],
            "semantic_status": output["semantic_status"],
            "evidence_turn_keys": list(output["evidence_turn_keys"]),
            "evidence_turn_ids": [key_to_turn[key]["turn_id"]
                                  for key in output["evidence_turn_keys"]],
            "evidence_text_sha256": [key_to_turn[key]["text_sha256"]
                                     for key in output["evidence_turn_keys"]],
        })
        if (output["decision"] == "confirmed" and
                output["semantic_status"] == "semantically_verified"):
            if delta_by_id[delta_id]["operation"] == "add_role_handoff":
                confirmed_handoff_deltas.append(delta_id)
            elif delta_by_id[delta_id]["operation"] == "open_thread":
                confirmed_open_thread_deltas.append(delta_id)

    allocated_handoffs = []
    for index, delta_id in enumerate(confirmed_handoff_deltas, 1):
        evidence = next(row for row in evidence_bindings
                        if row["proposed_delta_id"] == delta_id)
        allocated_handoffs.append({
            "handoff_id": f"H{handoff_start_index + index - 1:04d}",
            "proposed_delta_id": delta_id,
            "description": str(delta_by_id[delta_id]["intended_post_state"]),
            "evidence_turn_ids": list(evidence["evidence_turn_ids"]),
            "source_scene": scene_key,
        })

    proposal_by_id: dict[str, Mapping[str, Any]] = {}
    for proposal in emergent_proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        if not proposal_id or proposal_id in proposal_by_id:
            raise StructuralBindingError("emergent_proposal_identity_invalid")
        if (proposal.get("root_id") != root_id or
                proposal.get("source_scene") != scene_key or
                proposal.get("operation") != "open_thread" or
                proposal.get("semantic_status") != "model_claimed" or
                proposal.get("formal_thread_id") is not None or
                proposal.get("authoritative") is not False):
            raise StructuralBindingError(
                f"emergent_proposal_contract_invalid:{proposal_id}")
        missing = [key for key in proposal.get("candidate_evidence_turn_keys") or []
                   if key not in key_to_turn]
        if missing or not proposal.get("candidate_evidence_turn_keys"):
            raise StructuralBindingError(
                "emergent_proposal_evidence_turn_key_missing:" + ",".join(missing))
        proposal_by_id[proposal_id] = proposal

    emergent_evidence_bindings: list[dict[str, Any]] = []
    confirmed_emergent_proposals: list[str] = []
    seen_emergent_outputs: set[str] = set()
    for output in emergent_committer_outputs:
        validation = validate_evidence_committer_output(output)
        if not validation["valid"]:
            raise StructuralBindingError("emergent_committer_output_invalid:" +
                                         ",".join(validation["errors"]))
        proposal_id = str(output["proposed_delta_id"])
        if proposal_id not in proposal_by_id:
            raise StructuralBindingError(
                f"emergent_committer_unknown_proposal:{proposal_id}")
        if proposal_id in seen_emergent_outputs:
            raise StructuralBindingError(
                f"duplicate_emergent_committer_proposal:{proposal_id}")
        seen_emergent_outputs.add(proposal_id)
        missing = [key for key in output["evidence_turn_keys"]
                   if key not in key_to_turn]
        if missing:
            raise StructuralBindingError(
                "emergent_evidence_turn_key_missing:" + ",".join(missing))
        row = {
            "proposal_id": proposal_id,
            "decision": output["decision"],
            "semantic_status": output["semantic_status"],
            "evidence_turn_keys": list(output["evidence_turn_keys"]),
            "evidence_turn_ids": [key_to_turn[key]["turn_id"]
                                  for key in output["evidence_turn_keys"]],
            "evidence_text_sha256": [key_to_turn[key]["text_sha256"]
                                     for key in output["evidence_turn_keys"]],
        }
        emergent_evidence_bindings.append(row)
        if (output["decision"] == "confirmed" and
                output["semantic_status"] == "semantically_verified"):
            confirmed_emergent_proposals.append(proposal_id)

    allocated_threads: list[dict[str, Any]] = []
    next_thread_index = thread_start_index
    for delta_id in confirmed_open_thread_deltas:
        evidence = next(row for row in evidence_bindings
                        if row["proposed_delta_id"] == delta_id)
        allocated_threads.append({
            "thread_id": f"thread_{next_thread_index:04d}",
            "route_type": "planned_graph_delta",
            "proposed_delta_id": delta_id,
            "proposal_id": None,
            "description": str(delta_by_id[delta_id]["intended_post_state"]),
            "evidence_turn_ids": list(evidence["evidence_turn_ids"]),
            "source_scene": scene_key,
        })
        next_thread_index += 1
    for proposal_id in confirmed_emergent_proposals:
        evidence = next(row for row in emergent_evidence_bindings
                        if row["proposal_id"] == proposal_id)
        allocated_threads.append({
            "thread_id": f"thread_{next_thread_index:04d}",
            "route_type": "emergent_open_thread",
            "proposed_delta_id": None,
            "proposal_id": proposal_id,
            "description": str(proposal_by_id[proposal_id]["description"]),
            "evidence_turn_ids": list(evidence["evidence_turn_ids"]),
            "source_scene": scene_key,
        })
        next_thread_index += 1

    turn_key_map = {key: value["turn_id"] for key, value in key_to_turn.items()}
    result = {
        "binder_version": LOCAL_STRUCTURAL_BINDER_VERSION,
        "root_id": root_id,
        "child_id": child_id,
        "session_id": session_id,
        "scene_key": scene_key,
        "turn_key_to_turn_id": turn_key_map,
        "dialogue_turns": bound_turns,
        "evidence_bindings": evidence_bindings,
        "emergent_evidence_bindings": emergent_evidence_bindings,
        "allocated_handoffs": allocated_handoffs,
        "allocated_threads": allocated_threads,
        "provenance": {
            "root_id": root_id,
            "child_id": child_id,
            "round_index": round_index,
            "scene_key": scene_key,
            "bound_dialogue_sha256": sha256_bytes(canonical_json(bound_turns).encode()),
            "program_assigned_final_ids": True,
        },
        "api_ledger": {
            "request_count": 0,
            "completed_request_count": 0,
            "ambiguous_request_count": 0,
            "mock_backend_call_count": 0,
            "paid_api_generation_count": 0,
            "local_binding_event_count": 1,
        },
    }
    return result
