"""Small immutable patch layer over the existing split-plan compiler."""
from copy import deepcopy
import hashlib
import json
import re

from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    validate_planner_output, compile_planner_output, materialize_planner_output,
    _validate_transition_fields,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.profiles import EXPANDED_SINGLE_PLAN_PROFILE
from . import POLICY_ID

PROFILE = {**EXPANDED_SINGLE_PLAN_PROFILE.to_dict(), "name": POLICY_ID,
           "initial_scene_limit": 6, "cumulative_scene_limit": "derived_from_explicit_budget",
           "max_supplement_requests_per_root_round": 1,
           "outer_attempt_limit": 1, "target_only_recovery_calls": 0,
           "remote_memory_operations": False,
           "gate_context_mode": "authoritative_full_history_no_remote_memory",
           "bounded_continuation_limit": 0, "replacement_limit": 0,
           "entry_scope": "frozen_initial_plan_pre_gate_supplement"}
TARGET_DIAGNOSTIC = re.compile(r"^(?:continuation|new_development_candidate)_\d+_future_only_after_state$")


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def prepare_original(plan, *, parent, candidates, verified_tracks, max_new_candidates=8):
    raw_validation = validate_planner_output(
        plan, candidates=candidates, parent=parent, verified_tracks=verified_tracks,
        max_new_candidates=max_new_candidates, max_continuations=3, scene_quota=6,
        new_development_slots=6, require_full_scene_quota=False)
    diagnostics = list(raw_validation["errors"])
    diagnostics += [e for r in raw_validation.get("candidate_validation", []) for e in r["errors"]]
    goals = sorted(set(e for e in diagnostics if TARGET_DIAGNOSTIC.fullmatch(e)))
    hard = [e for e in raw_validation["errors"] if not TARGET_DIAGNOSTIC.fullmatch(e)]
    result = {"original": deepcopy(plan), "original_sha256": sha(plan),
              "version": "original_v1", "legacy_validation": raw_validation,
              "goal_diagnostics": goals, "hard_errors": hard,
              "valid_for_new_policy": not hard}
    if hard:
        return result
    permitted_new = [r["index"] for r in raw_validation["candidate_validation"]
                     if not r["errors"] or all(TARGET_DIAGNOSTIC.fullmatch(e) for e in r["errors"])]
    compiled = compile_planner_output(
        plan, candidates=candidates, verified_tracks=verified_tracks,
        selected_new_candidate_indices=permitted_new[:max(0, 6-len(plan["selected_continuations"]))],
        valid_new_candidate_indices=permitted_new,
        scene_quota=6, new_development_slots=6)
    registry = {}
    for scene in compiled["candidate_pool"]:
        ref = str(scene["memory_event_id"]) + ":C01"
        materialized = materialize_planner_output(
            {"scenes": [scene]}, passed_candidate_refs=[ref], scene_count=1)["scenes"][0]
        registry[ref] = {"gate_scene": deepcopy(scene), "scene": materialized,
                         "candidate_sha256": sha(scene["next_step_candidates"][0])}
    affected = []
    for diagnostic in goals:
        if diagnostic.startswith("continuation_"):
            index = int(diagnostic.split("_")[1])-1
            affected.append(compiled["scenes"][index]["memory_event_id"] + ":C01")
        else:
            index = int(diagnostic.split("_")[3])
            affected.append(f"ND{index:02d}:C01")
    result.update(compiled=compiled, candidate_registry=registry, affected_candidate_ids=affected)
    return result


def merge_patch(prepared, patch, *, allow_unconditional_continuations=False):
    """Only append phases. Never edit any original candidate or goal."""
    errors = []
    if not isinstance(patch, dict) or set(patch) != {"status", "original_plan_sha256", "phases", "reason"}:
        return {"valid": False, "errors": ["patch_shape"]}
    if patch["original_plan_sha256"] != prepared["original_sha256"]:
        errors.append("patch_original_sha_mismatch")
    if patch["status"] not in {"PATCH", "NO_SUPPLEMENT"} or not isinstance(patch["phases"], list):
        return {"valid": False, "errors": errors + ["patch_status_or_phases"]}
    if not isinstance(patch["reason"], str) or not patch["reason"].strip():
        errors.append("patch_reason_required")
    if (patch["status"] == "NO_SUPPLEMENT") != (len(patch["phases"]) == 0):
        errors.append("patch_empty_status_mismatch")
    known = {"initial:" + ref for ref in prepared["candidate_registry"]}
    phases = []
    required = {"phase_id", "source_candidate_id", "original_goal", "required_transition", "interaction_beats", "dependencies"}
    for index, row in enumerate(patch["phases"], 1):
        if not isinstance(row, dict) or set(row) != required:
            errors.append(f"patch_phase_{index}_shape")
            continue
        if any(not isinstance(row[k], str) or not row[k].strip() for k in
               ("phase_id", "source_candidate_id", "original_goal", "required_transition")):
            errors.append(f"patch_phase_{index}_text_fields")
            continue
        source = prepared["candidate_registry"].get(row["source_candidate_id"])
        if not source:
            errors.append(f"patch_phase_{index}_unknown_source")
            continue
        if row["source_candidate_id"] not in prepared["affected_candidate_ids"]:
            errors.append(f"patch_phase_{index}_unrelated_candidate")
        if not isinstance(row["phase_id"], str) or not re.fullmatch(r"phase:[A-Za-z0-9_-]+", row["phase_id"]) or row["phase_id"] in known:
            errors.append(f"patch_phase_{index}_identity")
        if row["original_goal"] != source["scene"]["goal"]:
            errors.append(f"patch_phase_{index}_original_goal_changed")
        if (not isinstance(row["dependencies"], list)
                or (not row["dependencies"] and not allow_unconditional_continuations)):
            errors.append(f"patch_phase_{index}_explicit_dependency_required")
        else:
            for dep in row["dependencies"]:
                if not isinstance(dep, dict) or set(dep) != {"phase_id", "condition"} or not isinstance(dep["phase_id"], str) or dep["phase_id"] not in known or not isinstance(dep["condition"], str) or not dep["condition"].strip():
                    errors.append(f"patch_phase_{index}_dependency")
        candidate = deepcopy(source["gate_scene"]["next_step_candidates"][0])
        candidate["required_transition"] = row["required_transition"]
        candidate["interaction_beats"] = row["interaction_beats"]
        contract_errors = []
        _validate_transition_fields(candidate, prefix=f"patch_phase_{index}",
            allowed_participants=set(source["scene"]["participants"]), errors=contract_errors,
            required_keys=set(candidate))
        # Only this exact planned-target diagnostic is nonblocking; not a
        # substring-based waiver of future/identity/shape errors.
        errors.extend(e for e in contract_errors if e != f"patch_phase_{index}_future_only_after_state")
        scene = deepcopy(source["scene"])
        scene.update(phase_id=row["phase_id"], source_candidate_id=row["source_candidate_id"],
                     dependencies=deepcopy(row["dependencies"]), required_transition=row["required_transition"],
                     new_development=row["required_transition"], interaction_beats=deepcopy(row["interaction_beats"]))
        gate_scene = deepcopy(source["gate_scene"])
        gate_scene["next_step_candidates"] = [candidate]
        phases.append({"scene": scene, "gate_scene": gate_scene})
        known.add(row["phase_id"])
    if errors:
        return {"valid": False, "errors": errors, "raw_patch": deepcopy(patch)}
    merged = {"version": "increment_v1", "original_sha256": prepared["original_sha256"],
              "original_plan": deepcopy(prepared["original"]), "added_phases": phases}
    return {"valid": True, "errors": [], "raw_patch": deepcopy(patch),
            "merged": merged, "merged_sha256": sha(merged),
            "diff": {"original_candidate_changes": [], "original_goal_changes": [],
                     "added_phase_ids": [p["scene"]["phase_id"] for p in phases]}}
