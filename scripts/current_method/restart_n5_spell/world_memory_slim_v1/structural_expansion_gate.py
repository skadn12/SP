"""Deterministic structural gates for continuous expansion Phase A.

These gates do not claim semantic novelty.  They reject only explicit
reference, graph, normalized-duplicate, and frozen-context violations.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

from .continuous_round_plan_v1 import HARD_DEPENDENCY_TYPES
from .continuous_round_plan_v1 import PLACEMENT_LINK_TYPES
from .context_compiler_v2 import validate_context_manifest_v2
from .narrative_order import validate_narrative_order_manifest
from .placement_choice_registry import (
    resolve_placement_choice, resolve_source_anchor, validate_placement_choice,
)


STRUCTURAL_EXPANSION_GATE_VERSION = "structural_expansion_gate_v1"
DEPENDENCY_RUNTIME_VERSION = "scene_dependency_runtime_v1"
PROMOTION_CLOSURE_VERSION = "promotion_dependency_closure_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def normalize_exact_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.strip("\"'“”‘’")
    return re.sub(r"\s+", " ", value)


def _known_refs(planning_state: Mapping[str, Any]) -> set[str]:
    refs = {str(row.get("entity_ref") or "")
            for row in planning_state.get("entities") or []}
    for field in ("explicit_relations", "interaction_links", "active_events",
                  "open_threads", "tasks", "key_states", "knowledge_asymmetries",
                  "causal_links"):
        for row in planning_state.get(field) or []:
            refs.add(str(row.get("memory_ref") or
                         row.get("interaction_link_ref") or ""))
    return {value for value in refs if value}


def _longest_chain(nodes: Sequence[str], edges: Sequence[Mapping[str, Any]]) -> int:
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("dependency_type") in HARD_DEPENDENCY_TYPES:
            incoming[str(edge["to_scene_ref"])].append(str(edge["from_scene_ref"]))
    lengths: dict[str, int] = {}
    for node in nodes:
        lengths[node] = 1 + max((lengths.get(parent, 0)
                                 for parent in incoming[node]), default=0)
    return max(lengths.values(), default=0)


def validate_round_blueprint_structure(
    *, compiled_plan: Mapping[str, Any],
    source_anchor_registry: Mapping[str, Any],
    placement_choice_registry: Mapping[str, Any],
    complete_dialogue: Mapping[str, Any], planning_state: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    scenes = list(compiled_plan.get("scenes") or [])
    nodes = [str(scene.get("scene_ref") or "") for scene in scenes]
    index = {ref: position for position, ref in enumerate(nodes)}
    known = _known_refs(planning_state)
    all_edges: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    logical_keys: dict[tuple[str, str], str] = {}
    target_versions: dict[tuple[str, str], str] = {}
    development_keys: dict[tuple[str, str, str], str] = {}
    explicit_relation_refs = {str(row.get("memory_ref") or "")
                              for row in planning_state.get("explicit_relations") or []}
    interaction_refs = {str(row.get("interaction_link_ref") or "")
                        for row in planning_state.get("interaction_links") or []}
    for scene in scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        try:
            source_anchor = resolve_source_anchor(
                source_anchor_registry, str(scene.get("source_anchor_id") or ""))
            placement_choice = resolve_placement_choice(
                placement_choice_registry,
                str(scene.get("placement_choice_id") or ""))
        except ValueError as error:
            errors.append(f"{scene_ref}:source_or_placement:{error}")
            continue
        choice_validation = validate_placement_choice(
            placement_choice, complete_dialogue=complete_dialogue)
        errors.extend(f"{scene_ref}:{value}"
                      for value in choice_validation["errors"])
        if scene.get("source_anchor") != source_anchor:
            errors.append(f"{scene_ref}:source_anchor_drift")
        if scene.get("placement_choice") != placement_choice:
            errors.append(f"{scene_ref}:placement_choice_drift")
        if scene.get("placement") != placement_choice.get("placement"):
            errors.append(f"{scene_ref}:compiled_placement_drift")
        if scene.get("link_type") not in set(
                source_anchor.get("allowed_link_types") or []):
            errors.append(f"{scene_ref}:source_anchor_link_type")
        linked = scene.get("linked_existing_refs") or {}
        linked_values = {str(value) for values in linked.values()
                         for value in (values or [])}
        if not linked_values:
            errors.append(f"{scene_ref}:no_existing_world_link")
        invalid_links = sorted(value for value in linked_values if value not in known)
        if invalid_links:
            errors.append(f"{scene_ref}:unknown_existing_refs:" + ",".join(invalid_links))
        if set(str(value) for value in linked.get("relation_refs") or []) & interaction_refs:
            errors.append(f"{scene_ref}:interaction_miscast_as_explicit_relation")
        if any(str(value) not in explicit_relation_refs
               for value in linked.get("relation_refs") or []):
            errors.append(f"{scene_ref}:explicit_relation_ref_not_authoritative")
        anchor_links = {
            str(value) for field in ("linked_entity_refs", "linked_interaction_refs",
                                     "linked_relation_refs", "linked_event_refs",
                                     "linked_task_refs", "linked_thread_refs",
                                     "linked_state_refs")
            for value in source_anchor.get(field) or []
        }
        if linked_values and anchor_links and not (linked_values & anchor_links):
            errors.append(f"{scene_ref}:bridge_not_grounded_in_anchor")
        if not str(scene.get("new_development") or "").strip():
            errors.append(f"{scene_ref}:new_development_empty")
        development_key = (
            str(scene.get("source_anchor_id") or ""),
            str(scene.get("placement_choice_id") or ""),
            normalize_exact_text(str(scene.get("new_development") or "")),
        )
        if development_key in development_keys:
            errors.append(
                f"{scene_ref}:duplicate_anchor_placement_development:"
                f"{development_keys[development_key]}")
        development_keys[development_key] = scene_ref
        for edge in scene.get("dependency_edges") or []:
            source = str(edge.get("predecessor_scene_ref") or "")
            if source not in index:
                errors.append(f"{scene_ref}:dependency_unknown:{source}")
            elif index[source] >= index.get(scene_ref, -1):
                errors.append(f"{scene_ref}:dependency_not_prior:{source}")
            all_edges.append({"from_scene_ref": source, "to_scene_ref": scene_ref,
                              "dependency_type": edge.get("dependency_type")})
        if scene.get("branch_role") != "mainline":
            merge_ref = scene.get("merge_target_scene_ref")
            impacts = list(scene.get("impact_target_refs") or [])
            if not merge_ref and not impacts:
                errors.append(f"{scene_ref}:branch_without_merge_or_impact")
            if merge_ref and (merge_ref not in index or
                              index.get(str(merge_ref), -1) <= index.get(scene_ref, -1)):
                errors.append(f"{scene_ref}:branch_merge_invalid:{merge_ref}")
        for proposal in scene.get("compiled_proposals") or []:
            fingerprint = str(proposal.get("proposal_fingerprint") or "")
            if fingerprint in fingerprints:
                errors.append(
                    f"{scene_ref}:duplicate_proposal_fingerprint:{fingerprints[fingerprint]}")
            fingerprints[fingerprint] = str(proposal.get("proposal_ref") or "")
            key = (str(proposal.get("operation") or ""),
                   str(proposal.get("target_memory_ref") or
                       (proposal.get("compiled_payload") or {}).get("logical_key") or ""))
            if key[1] and key in logical_keys:
                errors.append(f"{scene_ref}:duplicate_logical_target:{logical_keys[key]}")
            logical_keys[key] = str(proposal.get("proposal_ref") or "")
            target_version = (str(proposal.get("target_memory_ref") or ""),
                              str(proposal.get("authoritative_version") or ""))
            if target_version[0] and target_version in target_versions:
                errors.append(
                    f"{scene_ref}:duplicate_target_version:{target_versions[target_version]}")
            target_versions[target_version] = str(proposal.get("proposal_ref") or "")
    # Acyclicity is guaranteed by prior-only ordering, but keep an explicit audit.
    indegree = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in all_edges:
        source, target = edge["from_scene_ref"], edge["to_scene_ref"]
        if source in indegree and target in indegree:
            indegree[target] += 1
            outgoing[source].append(target)
    queue = deque([node for node in nodes if indegree[node] == 0])
    visited = []
    while queue:
        node = queue.popleft()
        visited.append(node)
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(visited) != len(nodes):
        errors.append("scene_dependency_cycle")
    chain_length = _longest_chain(nodes, all_edges)
    if chain_length < 3:
        errors.append(f"scene_chain_too_short:{chain_length}")
    connected_nodes = {edge["from_scene_ref"] for edge in all_edges} | {
        edge["to_scene_ref"] for edge in all_edges}
    if len(nodes) > 1:
        isolated = [node for node in nodes if node not in connected_nodes]
        if isolated:
            errors.append("isolated_scenes:" + ",".join(isolated))
    result = {
        "gate_version": STRUCTURAL_EXPANSION_GATE_VERSION,
        "stage": "round_blueprint",
        "valid": not errors, "errors": errors,
        "scene_count": len(scenes), "edge_count": len(all_edges),
        "longest_hard_dependency_chain": chain_length,
        "structural_novelty_precheck_ready": not errors,
        "semantic_novelty_review_ready": False,
    }
    result["audit_sha256"] = _sha(result)
    return result


def validate_connected_jump(
    *, link_type: str, linked_existing_refs: Mapping[str, Sequence[str]],
    planning_state: Mapping[str, Any], introduces_new_entity: bool = False,
    introduces_new_event: bool = False,
) -> dict[str, Any]:
    """Validate only explicit structural attachment; never infer semantics."""
    errors: list[str] = []
    if link_type not in PLACEMENT_LINK_TYPES:
        errors.append("connected_jump_link_type_invalid")
    known = _known_refs(planning_state)
    values = {str(value) for refs in linked_existing_refs.values() for value in refs}
    if not values:
        errors.append("unrelated_topic_jump")
    unknown = sorted(value for value in values if value not in known)
    if unknown:
        errors.append("connected_jump_unknown_refs:" + ",".join(unknown))
    interaction_refs = {str(row.get("interaction_link_ref") or "")
                        for row in planning_state.get("interaction_links") or []}
    if set(linked_existing_refs.get("relation_refs") or []) & interaction_refs:
        errors.append("interaction_miscast_as_explicit_relation")
    if (introduces_new_entity or introduces_new_event) and not values:
        errors.append("new_world_element_without_existing_entry_path")
    return {"valid": not errors, "errors": errors,
            "unrelated_topic_jump": "unrelated_topic_jump" in errors,
            "semantic_review_required": True}


def validate_realized_expansion_structure(
    *, compiled_plan: Mapping[str, Any], scene_outputs: Mapping[str, Mapping[str, Any]],
    context_manifests: Mapping[str, Mapping[str, Any]],
    complete_dialogue: Mapping[str, Any],
    narrative_order_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    narrative_validation = validate_narrative_order_manifest(
        narrative_order_manifest)
    errors.extend(f"narrative:{value}" for value in narrative_validation["errors"])
    compiled_refs = [str(scene.get("scene_ref") or "")
                     for scene in compiled_plan.get("scenes") or []]
    if list(narrative_order_manifest.get("generation_order_scene_refs") or []) != compiled_refs:
        errors.append("narrative_generation_order_mismatch")
    if set(narrative_order_manifest.get("ordered_scene_refs") or []) != set(compiled_refs):
        errors.append("narrative_scene_linkage_mismatch")
    historical = {normalize_exact_text(str(row.get("text") or ""))
                  for row in complete_dialogue.get("turns") or []}
    generated: dict[str, str] = {}
    for scene in compiled_plan.get("scenes") or []:
        scene_ref = str(scene.get("scene_ref") or "")
        output = scene_outputs.get(scene_ref)
        context = context_manifests.get(scene_ref)
        if not isinstance(output, Mapping):
            errors.append(f"{scene_ref}:scene_output_missing")
            continue
        if not isinstance(context, Mapping):
            errors.append(f"{scene_ref}:context_manifest_missing")
            continue
        validation = validate_context_manifest_v2(context)
        errors.extend(f"{scene_ref}:{value}" for value in validation["errors"])
        if context.get("source_anchor_id") != scene.get("source_anchor_id"):
            errors.append(f"{scene_ref}:context_anchor_mismatch")
        if context.get("placement_choice_id") != scene.get("placement_choice_id"):
            errors.append(f"{scene_ref}:context_placement_choice_mismatch")
        predecessor_refs = list(scene.get("predecessor_scene_refs") or [])
        for predecessor in predecessor_refs:
            previous = scene_outputs.get(predecessor)
            if previous is None:
                errors.append(f"{scene_ref}:predecessor_output_missing:{predecessor}")
                continue
            rows = [row for row in context.get("p0_required_dialogue") or []
                    if row.get("scene_ref") == predecessor]
            actual = _sha([{key: row.get(key) for key in (
                "turn_ref", "speaker_ref", "text")} for row in rows])
            source = _sha([{key: row.get(key) for key in (
                "turn_ref", "speaker_ref", "text")} for row in
                           previous.get("dialogue_turns") or []])
            recorded = (context.get("predecessor_dialogue_sha256") or {}).get(
                predecessor)
            if not rows or actual != source or recorded != source:
                errors.append(f"{scene_ref}:predecessor_dialogue_sha_mismatch:{predecessor}")
        local_refs: list[str] = []
        for turn in output.get("dialogue_turns") or []:
            ref = str(turn.get("turn_ref") or "")
            local_refs.append(ref)
            if not ref.startswith(scene_ref + ":T"):
                errors.append(f"{scene_ref}:scene_local_turn_ref_invalid:{ref}")
            normalized = normalize_exact_text(str(turn.get("text") or ""))
            if not normalized:
                errors.append(f"{scene_ref}:empty_turn:{ref}")
            elif normalized in historical:
                errors.append(f"{scene_ref}:historical_exact_duplicate:{ref}")
            elif normalized in generated:
                errors.append(f"{scene_ref}:same_round_exact_duplicate:{ref}:{generated[normalized]}")
            else:
                generated[normalized] = ref
        if len(local_refs) != len(set(local_refs)):
            errors.append(f"{scene_ref}:scene_local_turn_ref_duplicate")
    result = {
        "gate_version": STRUCTURAL_EXPANSION_GATE_VERSION,
        "stage": "realized_expansion",
        "valid": not errors, "errors": errors,
        "exact_or_normalized_duplicate_count": sum(
            "duplicate" in value for value in errors),
        "structural_novelty_precheck_ready": not errors,
        "semantic_novelty_review_ready": False,
        "semantic_review_required_for_all_candidate_updates": True,
    }
    result["audit_sha256"] = _sha(result)
    return result


def simulate_dependency_runtime(
    compiled_plan: Mapping[str, Any], scene_statuses: Mapping[str, str],
) -> dict[str, Any]:
    """Dry-run hard dependency blocking without allocating any world IDs."""
    rows = []
    successful = {"generated", "reviewed_clean", "committed", "promoted"}
    for scene in compiled_plan.get("scenes") or []:
        scene_ref = str(scene.get("scene_ref") or "")
        hard = [str(edge["predecessor_scene_ref"])
                for edge in scene.get("dependency_edges") or []
                if edge.get("dependency_type") in HARD_DEPENDENCY_TYPES]
        blocked_by = [ref for ref in hard if scene_statuses.get(ref) not in successful]
        rows.append({
            "scene_ref": scene_ref,
            "input_status": scene_statuses.get(scene_ref, "not_run"),
            "hard_dependency_refs": hard,
            "blocked_by": blocked_by,
            "runtime_decision": "dependency_blocked" if blocked_by else "eligible",
        })
    return {"dependency_runtime_version": DEPENDENCY_RUNTIME_VERSION,
            "scenes": rows,
            "blocked_scene_refs": [row["scene_ref"] for row in rows
                                   if row["blocked_by"]]}


def compute_promotion_closure(
    compiled_plan: Mapping[str, Any], *, requested_promoted_scene_refs: Sequence[str],
    committed_scene_refs: Sequence[str],
) -> dict[str, Any]:
    """Require every hard predecessor of a promoted scene to be promoted too."""
    requested = set(str(value) for value in requested_promoted_scene_refs)
    committed = set(str(value) for value in committed_scene_refs)
    scene_by_ref = {str(scene["scene_ref"]): scene
                    for scene in compiled_plan.get("scenes") or []}
    missing_dependencies: dict[str, list[str]] = {}
    not_committed: list[str] = []
    for scene_ref in requested:
        scene = scene_by_ref.get(scene_ref)
        if scene is None:
            missing_dependencies[scene_ref] = ["unknown_scene"]
            continue
        hard = [str(edge["predecessor_scene_ref"])
                for edge in scene.get("dependency_edges") or []
                if edge.get("dependency_type") in HARD_DEPENDENCY_TYPES]
        missing = [ref for ref in hard if ref not in requested]
        if missing:
            missing_dependencies[scene_ref] = missing
        if scene.get("compiled_proposals") and scene_ref not in committed:
            not_committed.append(scene_ref)
    valid = not missing_dependencies and not not_committed
    return {
        "promotion_closure_version": PROMOTION_CLOSURE_VERSION,
        "valid": valid,
        "requested_promoted_scene_refs": sorted(requested),
        "missing_hard_dependencies": missing_dependencies,
        "proposal_scenes_not_committed": sorted(not_committed),
        "promotable_scene_refs": sorted(requested) if valid else [],
    }
