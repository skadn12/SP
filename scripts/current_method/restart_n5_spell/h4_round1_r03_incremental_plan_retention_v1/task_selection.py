"""Explicit same-task selection after legacy exact execution deduplication."""
from copy import deepcopy

from .planning import sha
from .plan_normalization import (
    CORE_FIELDS, VERSION as EXACT_VERSION, _differences, dependency_errors,
    merged_queue, normalize_queue, text_key,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_planner_outline_distinctness,
)

VERSION = "merged_plan_same_task_selection_v1"
SELECTION_RULE = "original_registry_entries_first_then_frozen_input_order"


def _core(entry):
    values = [entry["scene"].get(field) for field in sorted(CORE_FIELDS)]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    return tuple(text_key(value) for value in values)


def _phase(entry):
    return entry["scene"]["phase_id"]


def _reachable(graph, start, target):
    pending, seen = list(graph.get(start, [])), set()
    while pending:
        phase = pending.pop()
        if phase == target:
            return True
        if phase not in seen:
            seen.add(phase)
            pending.extend(graph.get(phase, []))
    return False


def _distinctions(left, right, graph):
    result = []
    if (_reachable(graph, _phase(left), _phase(right))
            or _reachable(graph, _phase(right), _phase(left))):
        result.append({"reason": "structured_predecessor_successor_phases"})
    # Only established structured event/track bindings count, not ND labels,
    # source_basis prose, or differences in desired participants and actions.
    for field in ("verified_track_ref", "memory_event_ref"):
        left_value = left["scene"].get(field) or left["gate_scene"].get(field)
        right_value = right["scene"].get(field) or right["gate_scene"].get(field)
        if left_value and right_value and left_value != right_value:
            result.append({"reason": "distinct_structured_event_binding", "field": field,
                           "left": left_value, "right": right_value})
    return result


def select_same_task(prepared, merged, *, source_identity=None):
    """Select one non-equivalent version without rewriting source or dependencies.

    Inputs are the verified single original plan and its valid additive patch.
    The returned queue alone is executable; the original registry remains intact.
    """
    raw = merged_queue(prepared, merged)
    exact = normalize_queue(raw, version=EXACT_VERSION)
    result = deepcopy(exact)
    result.update(
        version=VERSION, exact_normalization_version=EXACT_VERSION,
        exact_normalization=deepcopy(exact), selection_groups=[],
        unselected_variants=[], deferred_dependencies=[], preserved_distinctions=[],
        not_selected=[], deferred=[], source_identity=deepcopy(source_identity),
        input_identity={"original_plan_sha256": prepared.get("original_sha256"),
                        "source_scope": deepcopy(prepared.get("source_scope")),
                        "candidate_registry_sha256": sha(prepared["candidate_registry"]),
                        "merged_input_sha256": sha(merged)},
        selection_policy={"core_fields": sorted(CORE_FIELDS),
                          "text_normalization": "casefold then regex whitespace collapse then strip",
                          "selection_rule": SELECTION_RULE,
                          "semantic_equivalence": "NOT_INFERRED",
                          "unselected_variants_are_aliases": False,
                          "execution_dependency_rebinding": "EXACT_ALIASES_ONLY"})
    blocking = [error for error in exact["errors"]
                if error != "UNRESOLVED_EXECUTION_EQUIVALENCE"]
    if blocking:
        result.update(valid=False, errors=blocking)
        return result

    graph = {_phase(entry): [dep["phase_id"] for dep in entry["scene"].get("dependencies", [])]
             for entry in exact["queue"]}
    raw_by_phase = {_phase(entry): entry for entry in raw}
    # Do not let an identity exclusion in the legacy exact path erase an
    # explicit event-instance difference in this newly selected execution view.
    for alias in exact["aliases"]:
        left = raw_by_phase[alias["canonical_phase_id"]]
        right = raw_by_phase[alias["alias_phase_id"]]
        distinctions = _distinctions(left, right, graph)
        if distinctions:
            result["preserved_distinctions"].append({
                "left_phase_id": _phase(left), "right_phase_id": _phase(right),
                "reasons": distinctions})
            blocking.append("exact_alias_conflicts_with_structured_distinction:" + _phase(right))
    if blocking:
        result.update(valid=False, errors=blocking, queue=deepcopy(raw), after_count=len(raw),
                      execution_queue_sha256=sha(raw))
        return result

    original_phases = {"initial:" + ref for ref in prepared["candidate_registry"]}
    original_order = {"initial:" + ref: index
                      for index, ref in enumerate(prepared["candidate_registry"])}
    raw_order = {_phase(entry): index for index, entry in enumerate(raw)}
    provenance = {phase: {"origin": "original_plan" if phase in original_phases else "incremental_patch",
                           "frozen_input_index": index}
                  for phase, index in raw_order.items()}
    result["entry_provenance"] = provenance
    ordered = sorted(exact["queue"], key=lambda entry: (
        _phase(entry) not in original_phases,
        original_order.get(_phase(entry), raw_order[_phase(entry)])))
    groups, excluded = [], set()
    for entry in ordered:
        key = _core(entry)
        compatible = None
        for group in groups:
            if key is None or key != group["key"]:
                continue
            distinctions = []
            for member in group["members"]:
                reasons = _distinctions(member, entry, graph)
                if reasons:
                    distinctions.append({"left_phase_id": _phase(member),
                                         "right_phase_id": _phase(entry), "reasons": reasons})
            if distinctions:
                result["preserved_distinctions"].extend(distinctions)
            else:
                compatible = group
                break
        if compatible is None:
            groups.append({"key": key, "members": [entry]})
            continue
        preferred = compatible["members"][0]
        compatible["members"].append(entry)
        phase, selected_phase = _phase(entry), _phase(preferred)
        excluded.add(phase)
        result["unselected_variants"].append({
            "status": "NOT_SELECTED_SAME_TASK_VARIANT", "phase_id": phase,
            "candidate_id": entry["scene"]["source_candidate_id"],
            "preferred_candidate_id": preferred["scene"]["source_candidate_id"],
            "preferred_phase_id": selected_phase, "selection_rule": SELECTION_RULE,
            "provenance": deepcopy(provenance[phase]),
            "preferred_provenance": deepcopy(provenance[selected_phase]),
            "differences": _differences(preferred, entry),
            "entry_sha256": sha(entry), "preferred_entry_sha256": sha(preferred),
            "is_exact_alias": False, "dependency_reference_replacement": False})
    for group in groups:
        if len(group["members"]) < 2:
            continue
        preferred = group["members"][0]
        result["selection_groups"].append({
            "normalized_core_fields": dict(zip(sorted(CORE_FIELDS), group["key"])),
            "comparison_policy": result["selection_policy"]["text_normalization"],
            "preferred_candidate_id": preferred["scene"]["source_candidate_id"],
            "preferred_phase_id": _phase(preferred),
            "member_phase_ids": [_phase(entry) for entry in group["members"]],
            "not_selected_phase_ids": [_phase(entry) for entry in group["members"][1:]],
            "selection_rule": SELECTION_RULE, "version": VERSION,
            "input_sha256": sha(raw)})

    unavailable = set(excluded)
    queue = []
    # Exact normalization already verified topological order. Only structured
    # dependency edges propagate deferral; no text or provenance scans occur.
    for entry in exact["queue"]:
        phase = _phase(entry)
        if phase in excluded:
            continue
        blockers = [deepcopy(dep) for dep in entry["scene"].get("dependencies", [])
                    if dep["phase_id"] in unavailable]
        if blockers:
            unavailable.add(phase)
            result["deferred_dependencies"].append({
                "status": "DEFERRED_UNAVAILABLE_EXECUTION_PREDECESSOR", "phase_id": phase,
                "candidate_id": entry["scene"]["source_candidate_id"],
                "blocked_dependencies": blockers,
                "cause": "unselected_same_task_variant_or_transitively_deferred_phase",
                "dependency_reference_replacement": False, "entry_sha256": sha(entry)})
        else:
            queue.append(deepcopy(entry))
    errors = dependency_errors(queue)
    outline = validate_planner_outline_distinctness({"scenes": [entry["gate_scene"] for entry in queue]})
    errors.extend(outline["errors"])
    result.update(valid=not errors, errors=errors, queue=queue, outline_check=outline,
                  before_count=len(raw), after_count=len(queue),
                  raw_queue_sha256=sha(raw), execution_queue_sha256=sha(queue),
                  unresolved_collisions=deepcopy(outline["collisions"]),
                  not_selected=deepcopy(result["unselected_variants"]),
                  deferred=deepcopy(result["deferred_dependencies"]))
    return result
