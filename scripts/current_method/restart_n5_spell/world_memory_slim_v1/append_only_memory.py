"""Append-only World Memory v1 and scene-level commit gate."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .provenance_ledger import ledger_entry
from .relation_registry import derive_relation_edges


APPEND_ONLY_MEMORY_VERSION = "append_only_world_memory_v1"
SCENE_COMMIT_GATE_VERSION = "scene_commit_gate_v2_cross_scene_dependency"
STATE_CATEGORIES = frozenset({
    "goal", "belief", "attitude", "commitment", "responsibility", "availability",
    "relationship_state", "emotional_state", "knowledge_state", "custom",
})


def normalize_state_dimension(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("state_dimension_empty")
    return normalized


def initialize_append_only_memory(
    *, root_id: str, entities: Sequence[Mapping[str, Any]],
    authoritative_records: Sequence[Mapping[str, Any]] = (),
    provenance_ledger: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    records = []
    for source in authoritative_records:
        row = deepcopy(dict(source))
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        if ref:
            row.setdefault("memory_ref", ref)
            row.setdefault("logical_key", ref)
            row.setdefault("authoritative_version", 1)
            row.setdefault("origin", "source_grounded")
            row.setdefault("verification_status",
                           row.get("semantic_status") or "source_grounded")
        records.append(row)
    active: dict[str, str] = {}
    for row in records:
        logical_key = str(row.get("logical_key") or row.get("memory_ref") or row.get("memory_id") or "")
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        if logical_key and ref:
            active[logical_key] = ref
    return {
        "append_only_memory_version": APPEND_ONLY_MEMORY_VERSION,
        "root_id": root_id,
        "entities": [deepcopy(dict(row)) for row in entities],
        "authoritative_records": records,
        "memory_transitions": [],
        "active_head_index": active,
        "provenance_ledger": [deepcopy(dict(row)) for row in provenance_ledger],
        "pending_updates": [],
        "rejected_audit_log": [],
        "high_water_marks": _high_water_marks(records),
    }


def _high_water_marks(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {"memory": 0, "entity": 0, "event": 0, "relation": 0, "thread": 0,
              "handoff": 0, "global_time": 0}
    patterns = {
        "memory": re.compile(r"(?:^|:)M(\d+)$"), "entity": re.compile(r"(?:^|:)E(\d+)$"),
        "event": re.compile(r"(?:^|:)EV(\d+)$"), "relation": re.compile(r"(?:^|:)REL(\d+)$"),
        "thread": re.compile(r"(?:^|:)thread_(\d+)$"), "handoff": re.compile(r"(?:^|:)H(\d+)$"),
    }
    for row in records:
        for value in row.values():
            if not isinstance(value, str):
                continue
            for name, pattern in patterns.items():
                match = pattern.search(value)
                if match:
                    result[name] = max(result[name], int(match.group(1)))
        result["global_time"] = max(result["global_time"], int(row.get("global_time_index") or 0))
    return result


def evaluate_scene_commit(
    *, compiled_proposals: Sequence[Mapping[str, Any]],
    proposal_claims: Sequence[Mapping[str, Any]], committer_output: Mapping[str, Any],
    previously_committed_proposal_refs: Sequence[str] = (),
) -> dict[str, Any]:
    claims = {str(row.get("proposal_ref")): row for row in proposal_claims}
    reviews = {str(row.get("proposal_ref")): row for row in committer_output.get("proposal_reviews") or []}
    failures: list[str] = []
    if committer_output.get("scope_status") != "clean":
        failures.append("scope_violation")
    allowed_consistency = {"clean", "authorized_change"} if compiled_proposals else {"clean"}
    if committer_output.get("world_consistency_status") not in allowed_consistency:
        failures.append("world_consistency_blocked")
    statuses: dict[str, str] = {}
    proposal_refs = {str(row.get("proposal_ref")) for row in compiled_proposals}
    previously_committed = {str(ref) for ref in previously_committed_proposal_refs}
    dependency_refs_satisfied = proposal_refs | previously_committed
    for proposal in compiled_proposals:
        ref = str(proposal["proposal_ref"])
        claim = claims.get(ref)
        review = reviews.get(ref)
        if not claim or claim.get("realization_status") != "realized":
            statuses[ref] = "scene_blocked"
            failures.append(f"not_realized:{ref}")
            continue
        if (not review or review.get("review_decision") != "confirmed" or
                review.get("verification_status") != "semantically_verified"):
            statuses[ref] = "scene_blocked"
            failures.append(f"not_confirmed:{ref}")
            continue
        missing = (set(proposal.get("proposal_dependencies") or []) -
                   dependency_refs_satisfied)
        if missing:
            statuses[ref] = "dependency_blocked"
            failures.append(f"dependency_blocked:{ref}")
            continue
        statuses[ref] = "commit_ready"
    for proposal in compiled_proposals:
        ref = str(proposal["proposal_ref"])
        dependencies = set(proposal.get("proposal_dependencies") or [])
        if dependencies and any(
                dependency not in previously_committed and
                statuses.get(dependency) != "commit_ready"
                for dependency in dependencies):
            statuses[ref] = "dependency_blocked"
            failures.append(f"dependency_blocked:{ref}")
    atomic_groups: dict[str, list[str]] = {}
    for proposal in compiled_proposals:
        if proposal.get("atomic_group_ref"):
            atomic_groups.setdefault(str(proposal["atomic_group_ref"]), []).append(str(proposal["proposal_ref"]))
    for refs in atomic_groups.values():
        if any(statuses.get(ref) != "commit_ready" for ref in refs):
            for ref in refs:
                statuses[ref] = "atomic_blocked"
            failures.append("atomic_group_blocked:" + ",".join(refs))
    ready = not failures and all(value == "commit_ready" for value in statuses.values())
    if not ready:
        for ref, value in list(statuses.items()):
            if value == "commit_ready":
                statuses[ref] = "scene_blocked"
    return {"scene_commit_gate_version": SCENE_COMMIT_GATE_VERSION,
            "commit_ready": ready, "proposal_commit_statuses": statuses,
            "failures": failures}


def apply_scene_commit(
    memory: Mapping[str, Any], *, scene_ref: str, round_index: int, scene_index: int,
    compiled_proposals: Sequence[Mapping[str, Any]], proposal_claims: Sequence[Mapping[str, Any]],
    committer_output: Mapping[str, Any], allocation_plan: Mapping[str, Any],
    evidence_final_turn_refs: Mapping[str, Sequence[str]], evidence_text_sha256: Mapping[str, Sequence[str]],
    previously_committed_proposal_refs: Sequence[str] = (),
) -> dict[str, Any]:
    gate = evaluate_scene_commit(compiled_proposals=compiled_proposals,
                                 proposal_claims=proposal_claims,
                                 committer_output=committer_output,
                                 previously_committed_proposal_refs=
                                 previously_committed_proposal_refs)
    result = deepcopy(dict(memory))
    if not gate["commit_ready"]:
        for proposal in compiled_proposals:
            ref = str(proposal["proposal_ref"])
            pending = deepcopy(dict(proposal))
            pending["commit_status"] = gate["proposal_commit_statuses"].get(ref, "scene_blocked")
            result["pending_updates"].append(pending)
        result["rejected_audit_log"].append({"scene_ref": scene_ref, "gate": gate})
        return result

    reviews = {str(row["proposal_ref"]): row for row in committer_output["proposal_reviews"]}
    original_count = len(result["authoritative_records"])
    new_records: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for proposal in compiled_proposals:
        ref = str(proposal["proposal_ref"])
        operation = str(proposal["operation"])
        payload = _resolve_temporary_refs(
            deepcopy(dict(proposal["compiled_payload"])),
            allocation_plan.get("temporary_ref_allocations") or {})
        allocation = dict(allocation_plan["proposal_allocations"][ref])
        memory_ref = str(allocation["memory_ref"])
        target = proposal.get("target_memory_ref")
        target_record = _record_by_ref(result, str(target)) if target else None
        if target_record is not None:
            logical_key = str(target_record["logical_key"])
            active = result["active_head_index"].get(logical_key)
            if active != target:
                raise ValueError("append_only_stale_active_head")
            if int(target_record.get("authoritative_version") or 1) != int(proposal["authoritative_version"]):
                raise ValueError("append_only_stale_authoritative_version")
            if target_record.get("value") != proposal.get("pre_state"):
                raise ValueError("append_only_pre_state_mismatch")
            version = int(target_record.get("authoritative_version") or 1) + 1
        else:
            logical_key = _logical_key(proposal)
            if logical_key in result["active_head_index"]:
                raise ValueError("append_only_duplicate_active_record")
            version = 1
        record = {
            "memory_ref": memory_ref, "logical_key": logical_key,
            "record_type": _record_type(operation), "operation": operation,
            "payload": payload, "value": _record_value(payload),
            "subject_refs": _subject_refs(payload, proposal),
            "authoritative_version": version,
            "origin": "generated_extension", "verification_status": "semantically_verified",
            "proposal_ref": ref, "scene_ref": scene_ref, "round_index": round_index,
            "scene_index": scene_index,
            "event_sequence_index": int(allocation["event_sequence_index"]),
            "global_time_index": int(allocation["global_time_index"]),
            "narrative_time_expression": payload.get("narrative_time_expression"),
            "support_turn_refs": list(evidence_final_turn_refs.get(ref) or []),
            "evidence_text_sha256": list(evidence_text_sha256.get(ref) or []),
            "formal_thread_id": allocation.get("thread_ref"),
            "handoff_id": allocation.get("handoff_ref"),
            "allocated_entity_ref": allocation.get("entity_ref"),
            "allocated_event_ref": allocation.get("event_ref"),
            "allocated_relation_ref": allocation.get("relation_ref"),
        }
        if allocation.get("entity_ref"):
            record["subject_refs"] = sorted(set(record["subject_refs"]) |
                                            {str(allocation["entity_ref"])})
        new_records.append(record)
        result["active_head_index"][logical_key] = memory_ref
        if target:
            transitions.append({
                "transition_type": "supersede", "from_memory_ref": target,
                "to_memory_ref": memory_ref, "round_index": round_index,
                "scene_ref": scene_ref,
            })
        result["provenance_ledger"].append(ledger_entry(
            record_ref=memory_ref, root_source_id=str(result["root_id"]),
            origin="generated_extension", verification_status="semantically_verified",
            introduced_round=round_index, introduced_scene_ref=scene_ref,
            proposal_ref=ref, support_turn_refs=record["support_turn_refs"],
            authoritative_memory_ref=memory_ref,
            parent_record_refs=[str(target)] if target else list(proposal.get("connected_memory_refs") or []),
        ))
        if operation == "add_person":
            entity_ref = str(allocation.get("entity_ref") or "")
            if not entity_ref or any(
                    str(row.get("entity_ref") or row.get("entity_id")) == entity_ref
                    for row in result["entities"]):
                raise ValueError("append_only_entity_allocation_invalid")
            result["entities"].append({
                "entity_id": entity_ref, "entity_ref": entity_ref,
                "display_name": str(payload["display_role"]),
                "entity_type": "person", "scene_presence": payload["scene_presence"],
                "origin": "generated_extension",
                "verification_status": "semantically_verified",
                "semantic_status": "semantically_verified",
                "introduced_round": round_index, "introduced_scene_ref": scene_ref,
                "proposal_ref": ref, "authoritative_memory_ref": memory_ref,
                "support_turn_refs": list(evidence_final_turn_refs.get(ref) or []),
            })
        if operation == "add_relation":
            record["derived_relation_edges"] = derive_relation_edges(ref, payload)
    result["authoritative_records"].extend(new_records)
    result["memory_transitions"].extend(transitions)
    result["high_water_marks"] = dict(allocation_plan["resulting_high_water_marks"])
    if len(result["authoritative_records"]) != original_count + len(compiled_proposals):
        raise AssertionError("append_only_record_count_invariant")
    return result


def _record_by_ref(memory: Mapping[str, Any], ref: str) -> Mapping[str, Any] | None:
    return next((row for row in memory.get("authoritative_records") or []
                 if row.get("memory_ref") == ref), None)


def _logical_key(proposal: Mapping[str, Any]) -> str:
    operation = str(proposal["operation"])
    payload = proposal["compiled_payload"]
    if operation == "create_state":
        category = str(payload["state_category"])
        if category not in STATE_CATEGORIES:
            raise ValueError("state_category_invalid")
        dimension = normalize_state_dimension(str(payload["state_dimension"]))
        return f"state:{payload['subject_ref']}:{category}:{dimension}"
    if operation == "add_relation":
        pair = sorted((str(payload["source_ref"]), str(payload["target_ref"])))
        return f"relation:{payload['relation_type']}:{pair[0]}:{pair[1]}"
    return f"{operation}:{proposal['proposal_fingerprint']}"


def _record_type(operation: str) -> str:
    return {"open_thread": "thread", "add_event": "event", "create_state": "state",
            "add_relation": "relation", "add_person": "entity",
            "add_shared_task": "task", "add_role_handoff": "responsibility"}.get(operation, operation)


def _record_value(payload: Mapping[str, Any]) -> Any:
    for key in ("new_value", "initial_value", "description", "task_description",
                "conflict_description", "resolution", "closure", "replacement_payload"):
        if key in payload:
            return deepcopy(payload[key])
    return deepcopy(dict(payload))


def _subject_refs(payload: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in payload.items():
        if key.endswith("_ref") and isinstance(value, str):
            values.append(value)
        elif key.endswith("_refs") and isinstance(value, list):
            values.extend(str(item) for item in value)
    return sorted(set(values) | set(str(value) for value in proposal.get("connected_memory_refs") or []))


def _resolve_temporary_refs(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        if ":TMP:" in value:
            if value not in mapping:
                raise ValueError(f"binder_temporary_ref_unresolved:{value}")
            return mapping[value]
        return value
    if isinstance(value, list):
        return [_resolve_temporary_refs(item, mapping) for item in value]
    if isinstance(value, Mapping):
        return {key: _resolve_temporary_refs(item, mapping) for key, item in value.items()}
    return value
