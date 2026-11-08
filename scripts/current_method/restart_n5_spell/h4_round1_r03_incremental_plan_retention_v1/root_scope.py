"""Explicit policy dispatch; no new authorization or execution state here."""
from . import POLICY_ID
from .supplement_contract_alignment import select_supplement_version
from .shared_planner_contract import ROOT_SHARED_PROMPT_VERSION

ROOT_POLICY_ID = "expanded_single_plan_root_incremental_retention_v1"
ROOT_SCOPE = "root_incremental_v1"
ROOT_PROMPT_VERSION = "incremental_retention_v1.supplement.root_incremental_v1"


def supplement_format_version(bundle):
    value = bundle.get("supplement_format_version")
    if "supplement_format_version" not in bundle:
        return None
    from .supplement_format import VERSION as FORMAT_VERSION
    if (value != FORMAT_VERSION or bundle.get("policy_id") != ROOT_POLICY_ID
            or bundle.get("supplement_prompt_version") != ROOT_PROMPT_VERSION):
        raise ValueError("supplement_formatter_requires_explicit_original_root_contract")
    return value


def with_supplement_formatting(bundle):
    """Select deterministic formatting in a new bundle, without editing inputs."""
    from copy import deepcopy
    from .supplement_format import VERSION as FORMAT_VERSION
    result = deepcopy(bundle)
    result["supplement_format_version"] = FORMAT_VERSION
    is_root_scope(result)
    return result


def is_root_scope(bundle):
    supplement_format_version(bundle)
    if bundle.get("policy_id") == ROOT_POLICY_ID:
        if (bundle.get("incremental_scope") != ROOT_SCOPE
                or bundle.get("supplement_prompt_version") not in {
                    ROOT_PROMPT_VERSION, ROOT_SHARED_PROMPT_VERSION}):
            raise ValueError("explicit_root_scope_and_template_required")
        return True
    if bundle.get("policy_id") != POLICY_ID:
        raise ValueError("unknown_incremental_policy")
    if "incremental_scope" in bundle:
        raise ValueError("legacy_policy_cannot_use_root_scope")
    select_supplement_version(bundle)
    return False


def prepare_bundle(bundle):
    if is_root_scope(bundle):
        if bundle.get("supplement_prompt_version") == ROOT_SHARED_PROMPT_VERSION:
            from .shared_planning import prepare_shared_original
            return prepare_shared_original(bundle)
        if "initial_planner_output" in bundle or "planner_contract_version" in bundle:
            raise ValueError("shared_initial_requires_shared_supplement_contract")
        from .root_planning import prepare_root_original
        return prepare_root_original(bundle)
    from .planning import prepare_original
    return prepare_original(bundle["original_plan"], parent=bundle["parent"],
        candidates=bundle["candidates"], verified_tracks=bundle["verified_tracks"])


def render_for_bundle(bundle, budget, prepared):
    inputs = {"identity": bundle["identity"], "original_plan_version": "original_v1",
        "original_plan_sha256": prepared["original_sha256"], "original_plan": prepared["original"],
        "stable_candidates": prepared["candidate_registry"],
        "affected_candidate_ids": prepared["affected_candidate_ids"],
        "goal_diagnostics": prepared["goal_diagnostics"],
        "authoritative_history": bundle["parent"]["dialogue"]["turns"],
        "retained_actual_r03_dialogue": [], "budget": budget,
        "shared_feedback_view": bundle["shared_feedback_view"]}
    if is_root_scope(bundle):
        from .root_prompts import render_root_supplement
        from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry
        inputs["eligible_source_candidate_ids"] = prepared["eligible_source_candidate_ids"]
        inputs["source_scope"] = {key: value for key, value in prepared["source_scope"].items()
                                  if key not in {"frozen_sources", "original_run_identity"}}
        inputs["speaker_registry"] = build_speaker_registry(bundle["parent"])
        inputs["memory_navigation"] = bundle["candidates"]
        # Diagnostics remain observations, not a source permission list.
        inputs["diagnostic_candidate_ids"] = inputs.pop("affected_candidate_ids")
        if bundle["supplement_prompt_version"] == ROOT_SHARED_PROMPT_VERSION:
            from .shared_planning import source_continuation_rows
            from .shared_planner_prompts import render_shared_supplement
            return render_shared_supplement(
                inputs, continuation_source_templates=source_continuation_rows(prepared))
        return render_root_supplement(inputs)
    from .prompts import render_supplement
    return render_supplement(inputs, prompt_version=select_supplement_version(bundle))


def merge_for_bundle(bundle, prepared, output):
    if is_root_scope(bundle):
        if bundle["supplement_prompt_version"] == ROOT_SHARED_PROMPT_VERSION:
            from .shared_planning import merge_shared_output
            return merge_shared_output(prepared, output)
        from .root_planning import merge_root_patch
        if supplement_format_version(bundle):
            from .supplement_format import format_root_supplement
            formatted = format_root_supplement(output, prepared=prepared)
            if not formatted["valid"]:
                return {"valid": False, "errors": formatted["errors"],
                        "raw_patch": formatted["raw_output"], "formatting": formatted}
            result = merge_root_patch(prepared, formatted["formatted_output"])
            result.update(raw_patch=formatted["raw_output"],
                          formatted_patch=formatted["formatted_output"], formatting=formatted)
            return result
        return merge_root_patch(prepared, output)
    from .planning import merge_patch
    return merge_patch(prepared, output)
