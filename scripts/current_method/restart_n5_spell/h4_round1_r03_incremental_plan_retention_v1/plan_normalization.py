"""Conservative, versioned execution-plan deduplication; never changes raw plans."""
from copy import deepcopy
import re

from .planning import sha
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_planner_outline_distinctness,
)

VERSION = "merged_plan_exact_execution_dedup_v1"
CORE_FIELDS = {"goal", "required_transition", "after_state"}
# Local identity/provenance labels do not distinguish an executable event.
IDENTITY_FIELDS = {"memory_event_id", "memory_event_ref", "memory_event_key",
                   "memory_event_display", "selected_candidate_ref",
                   "source_candidate_id", "phase_id", "scene_ref", "session_ref",
                   "goal_origin", "increment_type"}


def text_key(value):
    return re.sub(r"\s+", " ", value.casefold()).strip()


def merged_queue(prepared, merged):
    if not merged.get("valid"):
        raise ValueError("normalization_requires_valid_patch")
    initial = []
    for ref, source in prepared["candidate_registry"].items():
        entry = deepcopy(source)
        entry["scene"].update(phase_id="initial:" + ref, source_candidate_id=ref,
                              dependencies=deepcopy(entry["scene"].get("dependencies", [])))
        initial.append(entry)
    return initial + deepcopy(merged["merged"]["added_phases"])


def _projection(value, phase_map, *, field=None):
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name == "dependencies":
                result[name] = [{"phase_id": phase_map.get(d["phase_id"], d["phase_id"]),
                                 "condition": d["condition"]} for d in item]
            elif name not in IDENTITY_FIELDS:
                result[name] = _projection(item, phase_map, field=name)
        return result
    if isinstance(value, list):
        result = [_projection(item, phase_map, field=field) for item in value]
        # Coverage is a set of stable identities; order is not speaking order.
        return sorted(result) if field in {"participants", "required_participants"} else result
    if field in CORE_FIELDS and isinstance(value, str):
        return text_key(value)
    return value


def _signature(entry, phase_map):
    return {key: _projection(entry[key], phase_map) for key in ("scene", "gate_scene")}


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = prefix + "/" + key
            if key not in left or key not in right:
                result.append({"path": path, "left_present": key in left,
                               "right_present": key in right,
                               "left": left.get(key), "right": right.get(key)})
            else:
                result.extend(_differences(left[key], right[key], path))
        return result
    return [] if left == right else [{"path": prefix, "left": left, "right": right}]


def dependency_errors(queue):
    """Dependencies require an earlier retained stage, not merely a present ID."""
    phases = [entry["scene"]["phase_id"] for entry in queue]
    errors = []
    if len(set(phases)) != len(phases):
        errors.append("duplicate_execution_phase_id")
    positions = {phase: index for index, phase in enumerate(phases)}
    graph = {phase: [] for phase in phases}
    for entry in queue:
        scene = entry["scene"]
        phase = scene["phase_id"]
        for dep in scene.get("dependencies", []):
            if (not isinstance(dep, dict) or set(dep) != {"phase_id", "condition"}
                    or not isinstance(dep["phase_id"], str)
                    or not isinstance(dep["condition"], str) or not dep["condition"].strip()):
                errors.append("invalid_dependency:" + phase)
                continue
            target = dep["phase_id"]
            if target not in positions:
                errors.append("dangling_dependency:" + phase + ":" + target)
                continue
            graph[phase].append(target)
            if target == phase:
                errors.append("self_dependency:" + phase)
            elif positions[target] >= positions[phase]:
                errors.append("dependency_not_earlier:" + phase + ":" + target)
    visiting, visited = set(), set()
    def visit(phase):
        if phase in visiting:
            return True
        if phase in visited:
            return False
        visiting.add(phase)
        cyclic = any(visit(target) for target in graph[phase])
        visiting.remove(phase)
        visited.add(phase)
        return cyclic
    if any(visit(phase) for phase in phases):
        errors.append("dependency_cycle")
    return sorted(set(errors))


def normalize_queue(entries, *, version=VERSION):
    if version != VERSION:
        raise ValueError("unknown_plan_normalization_version")
    raw = deepcopy(entries)
    initial_errors = dependency_errors(raw)
    if initial_errors:
        return {"version": VERSION, "valid": False, "errors": initial_errors,
                "queue": raw, "aliases": [], "phase_aliases": {}, "candidate_aliases": {},
                "dependency_changes": [], "raw_queue_sha256": sha(raw),
                "execution_queue_sha256": sha(raw), "before_count": len(raw), "after_count": len(raw)}
    # Preserve frozen order within origin; originals always win over additions.
    ordered = sorted(enumerate(raw), key=lambda row: (
        not row[1]["scene"]["phase_id"].startswith("initial:"), row[0]))
    kept, aliases, phase_map, candidate_map, scene_map = [], [], {}, {}, {}
    for index, entry in ordered:
        signature = _signature(entry, phase_map)
        winner = next((item for _, item in kept if _signature(item, phase_map) == signature), None)
        if winner is None:
            kept.append((index, entry))
            continue
        scene, canonical = entry["scene"], winner["scene"]
        phase_map[scene["phase_id"]] = canonical["phase_id"]
        if scene.get("scene_ref") and canonical.get("scene_ref"):
            scene_map[scene["scene_ref"]] = canonical["scene_ref"]
        alias_candidate, canonical_candidate = scene["source_candidate_id"], canonical["source_candidate_id"]
        # A duplicate phase is not necessarily a duplicate source candidate.
        # Only whole original/independent candidate entries establish this map.
        whole_candidates = all(s["phase_id"].startswith("initial:") or
                               s.get("increment_type") == "independent_new_development"
                               for s in (scene, canonical))
        if alias_candidate != canonical_candidate and whole_candidates:
            candidate_map[alias_candidate] = canonical_candidate
        aliases.append({"alias_phase_id": scene["phase_id"], "canonical_phase_id": canonical["phase_id"],
                        "alias_candidate_id": alias_candidate, "duplicate_of": canonical_candidate,
                        "frozen_input_index": index,
                        "basis": "equal_execution_fields_except_local_identity_labels",
                        "signature_sha256": sha(signature), "alias_entry_sha256": sha(entry),
                        "canonical_entry_sha256": sha(winner)})
    queue = [deepcopy(entry) for _, entry in kept]
    changes, reference_changes = [], []
    event_map = {alias.rsplit(":", 1)[0]: canonical.rsplit(":", 1)[0]
                 for alias, canonical in candidate_map.items()}
    def rewrite_references(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                mapping = (candidate_map if key in {"source_candidate_id", "selected_candidate_ref"}
                           else event_map if key in {"memory_event_id", "memory_event_ref", "memory_event_key", "memory_event_display"}
                           else {})
                if isinstance(item, str) and item in mapping:
                    value[key] = mapping[item]
                    reference_changes.append({"path": path + "/" + key, "before": item, "after": value[key]})
                else:
                    rewrite_references(item, path + "/" + key)
        elif isinstance(value, list):
            for number, item in enumerate(value):
                rewrite_references(item, path + "/" + str(number))
    for entry in queue:
        scene = entry["scene"]
        rewrite_references(entry, scene["phase_id"])
        dependencies, seen = [], set()
        for dep in scene.get("dependencies", []):
            updated = {**dep, "phase_id": phase_map.get(dep["phase_id"], dep["phase_id"])}
            key = (updated["phase_id"], updated["condition"])
            if updated != dep or key in seen:
                changes.append({"phase_id": scene["phase_id"], "before": dep, "after": updated,
                                "removed_exact_redundant_relation": key in seen})
            if key not in seen:
                dependencies.append(updated)
                seen.add(key)
        scene["dependencies"] = dependencies
    errors = dependency_errors(queue)
    outline = validate_planner_outline_distinctness({"scenes": [e["gate_scene"] for e in queue]})
    unresolved = []
    for i, left in enumerate(queue):
        for j, right in enumerate(queue[i + 1:], i + 1):
            pair = validate_planner_outline_distinctness({"scenes": [left["gate_scene"], right["gate_scene"]]})
            if not pair["valid"]:
                unresolved.append({"left_phase_id": left["scene"]["phase_id"],
                                   "right_phase_id": right["scene"]["phase_id"],
                                   "left_candidate_id": left["scene"]["source_candidate_id"],
                                   "right_candidate_id": right["scene"]["source_candidate_id"],
                                   "reason": "transition_collision_but_execution_equivalence_not_proven",
                                   "differences": _differences(_signature(left, phase_map), _signature(right, phase_map))})
    if unresolved:
        errors.append("UNRESOLVED_EXECUTION_EQUIVALENCE")
    return {"version": VERSION, "valid": not errors, "errors": errors,
            "queue": queue, "aliases": aliases, "phase_aliases": phase_map,
            "candidate_aliases": candidate_map, "scene_aliases": scene_map, "dependency_changes": changes,
            "structured_reference_changes": reference_changes,
            "unresolved_collisions": unresolved, "outline_check": outline,
            "raw_queue_sha256": sha(raw), "execution_queue_sha256": sha(queue),
            "before_count": len(raw), "after_count": len(queue),
            "comparison_policy": {"core_text_fields": sorted(CORE_FIELDS),
                "text_normalization": "casefold then regex whitespace collapse then strip",
                "other_execution_fields": "exact parsed equality; coverage identity lists compared as sets",
                "excluded_identity_fields": sorted(IDENTITY_FIELDS),
                "unique_beats_are_not_combined": True, "semantic_equivalence": "NOT_INFERRED"}}
