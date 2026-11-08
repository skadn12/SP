"""Root-scoped additive planning; the historical patch contract stays intact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .planning import prepare_original, merge_patch, sha, TARGET_DIAGNOSTIC
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    build_speaker_registry, materialize_planner_speaker_refs,
    _validate_new_development_candidate, compile_planner_output,
    materialize_planner_output, is_transport_speaker_label,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.incremental_round import (
    build_round_history_parent,
)

MAX_ADDITIONAL_NEW_DEVELOPMENTS = 8
PHASE_ID_PATTERN = r"phase:[A-Za-z0-9_-]+"
NEW_DEVELOPMENT_KEYS = {
    "source_basis", "new_people", "goal", "speaker_refs",
    "transition_from_previous", "before_state", "required_transition",
    "after_state", "actor_bindings", "interaction_beats",
}


def _frozen_input_values(bundle):
    sources = bundle.get("frozen_sources")
    if not isinstance(sources, dict) or set(sources) != {"history", "plan", "candidates", "tracks"}:
        raise ValueError("root_scope_frozen_sources_required")
    values = {}
    for name, identity in sources.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise ValueError("root_scope_frozen_identity:" + name)
        content = Path(identity["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise ValueError("root_scope_frozen_sha_mismatch:" + name)
        values[name] = json.loads(content)
    return values


def _original_run_identity(bundle):
    """Use the existing bundle builder's run layout to bind sibling outputs."""
    manifest = (((bundle.get("shared_feedback_lineage") or {}).get("sources") or {})
                .get("original_request_manifest") or {})
    if not isinstance(manifest.get("path"), str):
        raise ValueError("root_scope_original_request_manifest_required")
    manifest_path = Path(manifest["path"]).resolve()
    source_run = manifest_path.parent.parent.parent
    expected = {"plan": source_run / "planner/variant_selection.json",
                "candidates": source_run / "memory/event_candidates.json",
                "tracks": source_run / "event_verifier/materialized_tracks.json"}
    if (manifest_path != source_run / "planner/actual_request/request_manifest.json"
            or any(Path(bundle["frozen_sources"][key]["path"]).resolve() != path
                   for key, path in expected.items())):
        raise ValueError("root_scope_original_run_source_association")
    freeze_path = source_run / "run_freeze_manifest.json"
    content = freeze_path.read_bytes()
    freeze = json.loads(content)
    if (freeze.get("root_id") != bundle["identity"]["root_id"]
            or freeze.get("round_index") != bundle["identity"]["round_index"]
            or Path(freeze["input_path"]).resolve() != Path(bundle["frozen_sources"]["history"]["path"]).resolve()
            or freeze.get("input_sha256") != bundle["frozen_sources"]["history"]["sha256"]):
        raise ValueError("root_scope_original_run_snapshot_mismatch")
    return {"run_path": str(source_run), "freeze_manifest_path": str(freeze_path),
            "freeze_manifest_sha256": hashlib.sha256(content).hexdigest()}


def prepare_root_original(bundle):
    """Bind all real original candidates to the same verified local snapshot."""
    from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE, is_root_scope
    if not isinstance(bundle, dict) or bundle.get("policy_id") != ROOT_POLICY_ID:
        raise ValueError("root_scope_policy_required")
    is_root_scope(bundle)
    identity = bundle.get("identity") or {}
    parent = bundle.get("parent") or {}
    if (not isinstance(identity.get("root_id"), str) or not identity["root_id"]
            or identity["root_id"] != parent.get("root_id")):
        raise ValueError("root_scope_root_mismatch")
    if (type(identity.get("round_index")) is not int
            or identity["round_index"] != parent.get("target_round_index")):
        raise ValueError("root_scope_round_mismatch")
    if bundle.get("history_sha256") != sha(parent):
        raise ValueError("root_scope_history_sha_mismatch")
    if bundle.get("original_plan_sha256") != sha(bundle.get("original_plan")):
        raise ValueError("root_scope_original_plan_sha_mismatch")
    values = _frozen_input_values(bundle)
    original_run_identity = _original_run_identity(bundle)
    frozen_parent = build_round_history_parent(values["history"], round_index=identity["round_index"])
    frozen_parent.pop("prior_expanded_dialogue", None)
    if (frozen_parent != parent or values["plan"].get("value") != bundle["original_plan"]
            or values["candidates"].get("candidates") != bundle.get("candidates")
            or values["tracks"].get("verified_tracks") != bundle.get("verified_tracks")):
        raise ValueError("root_scope_bundle_not_frozen_sources")
    selection = values["plan"]
    if any(key in selection for key in ("planner_contract_version", "shared_output")):
        from .shared_planner_contract import VERSION, ROOT_SHARED_PROMPT_VERSION
        if (selection.get("planner_contract_version") != VERSION
                or "shared_output" not in selection
                or bundle.get("planner_contract_version") != VERSION
                or bundle.get("initial_planner_output") != selection["shared_output"]
                or bundle["supplement_prompt_version"] != ROOT_SHARED_PROMPT_VERSION):
            raise ValueError("root_scope_shared_initial_not_frozen_selection")
    elif "planner_contract_version" in bundle or "initial_planner_output" in bundle:
        raise ValueError("root_scope_unfrozen_initial_relations")
    candidates, tracks = bundle["candidates"], bundle["verified_tracks"]
    event_ids = [row.get("memory_event_id") for row in candidates]
    track_ids = [row.get("verified_track_ref") for row in tracks]
    if (any(not isinstance(ref, str) or not ref for ref in event_ids + track_ids)
            or len(set(event_ids)) != len(event_ids) or len(set(track_ids)) != len(track_ids)
            or any(row.get("memory_event_ref") not in event_ids for row in tracks)):
        raise ValueError("root_scope_stable_registry_identity")
    for row in candidates + tracks:
        if "root_id" in row and row["root_id"] != identity["root_id"]:
            raise ValueError("root_scope_candidate_root_mismatch")
    prepared = prepare_original(bundle["original_plan"], parent=parent,
                                candidates=candidates, verified_tracks=tracks)
    if not prepared["valid_for_new_policy"]:
        return prepared
    registry = prepared["candidate_registry"]
    prepared.update(
        eligible_source_candidate_ids=list(registry),
        source_scope={"scope_mode": ROOT_SCOPE, "root_id": identity["root_id"],
                      "round_index": identity["round_index"],
                      "history_sha256": bundle["history_sha256"],
                      "original_plan_sha256": prepared["original_sha256"],
                      "candidate_registry_sha256": sha(registry),
                      "frozen_sources": deepcopy(bundle["frozen_sources"]),
                      "original_run_identity": original_run_identity,
                      "semantic_source_alignment": "UNKNOWN"},
        # Backend materialization data is not another model-visible history.
        _root_materialization_inputs={"parent": deepcopy(parent),
                                      "candidates": deepcopy(candidates),
                                      "verified_tracks": deepcopy(tracks)})
    return prepared


def _dependency_errors(dependencies, known, prefix):
    if not isinstance(dependencies, list):
        return [prefix + "_dependencies"]
    errors = []
    for dep in dependencies:
        if (not isinstance(dep, dict) or set(dep) != {"phase_id", "condition"}
                or not isinstance(dep["phase_id"], str) or dep["phase_id"] not in known
                or not isinstance(dep["condition"], str) or not dep["condition"].strip()):
            errors.append(prefix + "_dependency")
    return errors


def _new_people_errors(raw, parent, reserved_names, prefix):
    declarations = raw.get("new_people")
    if not isinstance(declarations, list):
        return []  # The existing candidate validator reports the shape.
    registry = build_speaker_registry(parent)
    speaker_refs = {row["speaker_ref"] for row in registry["speakers"]}
    parent_names = {row["canonical_name"].casefold() for row in registry["speakers"]}
    errors = []
    for person in declarations:
        if not isinstance(person, dict):
            continue
        name = str(person.get("name") or "").strip()
        if (name in speaker_refs or re.fullmatch(r"(?:SP|NP)\d+", name)
                or is_transport_speaker_label(name)):
            errors.append(prefix + "_new_person_identity_impersonation")
        if name.casefold() in parent_names | reserved_names:
            errors.append(prefix + "_new_person_name_collision")
    return errors


def merge_root_patch(prepared, patch, *, parent=None, candidates=None, verified_tracks=None,
                     allow_unconditional_continuations=False):
    """Append existing-source phases and explicitly declared new developments."""
    from .root_scope import ROOT_SCOPE
    required_top = {"status", "original_plan_sha256", "phases", "new_developments", "reason"}
    if not isinstance(patch, dict) or set(patch) != required_top:
        return {"valid": False, "errors": ["root_patch_shape"]}
    source_scope = prepared.get("source_scope") or {}
    stored = prepared.get("_root_materialization_inputs") or {}
    material = {"parent": parent if parent is not None else stored.get("parent"),
                "candidates": candidates if candidates is not None else stored.get("candidates"),
                "verified_tracks": verified_tracks if verified_tracks is not None else stored.get("verified_tracks")}
    if (source_scope.get("scope_mode") != ROOT_SCOPE or not all(material.values())
            or material != stored or sha(material["parent"]) != source_scope.get("history_sha256")
            or material["parent"].get("root_id") != source_scope.get("root_id")
            or sha(prepared.get("candidate_registry")) != source_scope.get("candidate_registry_sha256")
            or sha(prepared.get("original")) != source_scope.get("original_plan_sha256")
            or prepared.get("original_sha256") != source_scope.get("original_plan_sha256")
            or prepared.get("eligible_source_candidate_ids") != list(prepared["candidate_registry"])):
        return {"valid": False, "errors": ["root_patch_source_scope_mismatch"]}
    parent, candidates, verified_tracks = (material[name] for name in ("parent", "candidates", "verified_tracks"))
    if (not isinstance(patch["status"], str) or patch["status"] not in {"PATCH", "NO_SUPPLEMENT"}
            or not isinstance(patch["phases"], list) or not isinstance(patch["new_developments"], list)):
        return {"valid": False, "errors": ["root_patch_status_or_items"]}
    errors = []
    if (patch["status"] == "NO_SUPPLEMENT") != (not patch["phases"] and not patch["new_developments"]):
        errors.append("root_patch_empty_status_mismatch")
    if len(patch["new_developments"]) > MAX_ADDITIONAL_NEW_DEVELOPMENTS:
        errors.append("root_patch_new_developments_max_8")

    # Only this private copy changes permission. Old diagnostics and the old
    # merger remain byte-for-byte unchanged and retain their original meaning.
    scoped = deepcopy(prepared)
    scoped["affected_candidate_ids"] = list(prepared["candidate_registry"])
    phase_patch = {key: deepcopy(patch[key]) for key in required_top - {"new_developments"}}
    phase_patch["status"] = "PATCH" if patch["phases"] else "NO_SUPPLEMENT"
    phase_result = merge_patch(scoped, phase_patch,
                               allow_unconditional_continuations=allow_unconditional_continuations)
    errors.extend(phase_result["errors"])
    known = {"initial:" + ref for ref in prepared["candidate_registry"]}
    known.update(row["phase_id"] for row in patch["phases"]
                 if isinstance(row, dict) and isinstance(row.get("phase_id"), str))
    original_new = prepared["original"]["new_development_candidates"]
    reserved_names = {str(person.get("name") or "").strip().casefold()
                      for candidate in original_new for person in candidate.get("new_people") or []}
    parent_speakers = {turn["speaker"] for turn in parent["dialogue"]["turns"]}
    materialized_new = []
    for index, row in enumerate(patch["new_developments"], 1):
        prefix = f"root_new_development_{index}"
        if not isinstance(row, dict) or set(row) != {"phase_id", "dependencies", "candidate"}:
            errors.append(prefix + "_shape")
            continue
        phase_id = row["phase_id"]
        if (not isinstance(phase_id, str) or not re.fullmatch(PHASE_ID_PATTERN, phase_id)
                or phase_id in known):
            errors.append(prefix + "_identity")
        errors.extend(_dependency_errors(row["dependencies"], known, prefix))
        raw = row["candidate"]
        if not isinstance(raw, dict) or set(raw) != NEW_DEVELOPMENT_KEYS:
            errors.append(prefix + "_candidate_shape")
            continue
        errors.extend(_new_people_errors(raw, parent, reserved_names, prefix))
        converted = materialize_planner_speaker_refs(
            {"round_goal": "root increment materialization", "selected_continuations": [],
             "new_development_candidates": [raw]}, parent=parent)
        errors.extend(prefix + ":" + error for error in converted["errors"])
        if not converted["valid"]:
            continue
        item = converted["value"]["new_development_candidates"][0]
        contract_errors = _validate_new_development_candidate(item, index=index,
                                                              parent_speakers=parent_speakers)
        errors.extend(error for error in contract_errors if not TARGET_DIAGNOSTIC.fullmatch(error))
        materialized_new.append(item)
        reserved_names.update(str(person.get("name") or "").strip().casefold()
                              for person in item.get("new_people") or [] if isinstance(person, dict))
        if isinstance(phase_id, str):
            known.add(phase_id)
    if errors:
        return {"valid": False, "errors": errors, "raw_patch": deepcopy(patch)}

    expanded_plan = deepcopy(prepared["original"])
    expanded_plan["new_development_candidates"].extend(materialized_new)
    first_new_index = len(original_new) + 1
    new_indices = list(range(first_new_index, first_new_index + len(materialized_new)))
    compiled = compile_planner_output(expanded_plan, candidates=candidates,
        verified_tracks=verified_tracks, selected_new_candidate_indices=new_indices,
        valid_new_candidate_indices=new_indices, scene_quota=3 + len(materialized_new),
        new_development_slots=len(materialized_new))
    new_scenes = {scene["memory_event_id"]: scene for scene in compiled["candidate_pool"]}
    added = deepcopy(phase_result["merged"]["added_phases"])
    for entry in added:
        entry["scene"]["goal_origin"] = "original_candidate_goal"
        entry["scene"]["increment_type"] = "existing_candidate_phase"
    declarations = []
    for ordinal, row in zip(new_indices, patch["new_developments"]):
        event_id = f"ND{ordinal:02d}"
        ref = event_id + ":C01"
        gate_scene = deepcopy(new_scenes[event_id])
        scene = materialize_planner_output({"scenes": [gate_scene]},
            passed_candidate_refs=[ref], scene_count=1)["scenes"][0]
        scene.update(phase_id=row["phase_id"], source_candidate_id=ref,
                     dependencies=deepcopy(row["dependencies"]),
                     goal_origin="new_development_goal_not_historical",
                     increment_type="independent_new_development")
        added.append({"scene": scene, "gate_scene": gate_scene})
        declarations.append({"phase_id": row["phase_id"], "assigned_candidate_id": ref,
                             "new_people": deepcopy(scene["new_people"]),
                             "goal_origin": scene["goal_origin"]})
    merged = {"version": "root_increment_v1", "scope_mode": ROOT_SCOPE,
              "original_sha256": prepared["original_sha256"],
              "original_plan": deepcopy(prepared["original"]), "added_phases": added,
              "source_scope": deepcopy(source_scope), "new_declarations": declarations,
              "version_chain": [{"version": "original_v1", "sha256": prepared["original_sha256"]},
                                {"version": "root_increment_v1", "patch_sha256": sha(patch)}]}
    return {"valid": True, "errors": [], "raw_patch": deepcopy(patch), "merged": merged,
            "merged_sha256": sha(merged),
            "diff": {"original_candidate_changes": [], "original_goal_changes": [],
                     "added_phase_ids": [entry["scene"]["phase_id"] for entry in added],
                     "new_development_candidate_ids": [row["assigned_candidate_id"] for row in declarations],
                     "new_declarations": deepcopy(declarations)}}
