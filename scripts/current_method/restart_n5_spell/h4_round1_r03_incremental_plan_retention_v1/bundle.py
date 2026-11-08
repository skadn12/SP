"""Local production inputs anchored to the original Planner's sent feedback.

This module has no transport, fixture responses, or execution-state access.
The role projection is owned by the existing shared feedback compiler.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import POLICY_ID
from .planning import sha
from ..h4_round1_r03_feedback_view_v2 import COMPILER_RUN, REROUTE_RUN
from ..h4_round1_r03_feedback_view_v2.compiler import (
    build_backend_record, expansion_view_v3,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.incremental_round import (
    build_round_history_parent,
)

VERSION = "incremental_production_bundle_v2"
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
FAILED = PROJECT / "restart-n5-runs/h4_round1_r03_expanded_single_plan_fullflow_debug_live_3"
PRIOR = PROJECT / "restart-n5-runs/h4_round1_feedback_r02_batch_single_root_v1_live_3/expanded_dialogue.json"
FEEDBACK_MARKER = "\u3010\u5f53\u524d\u89d2\u8272\u53ef\u89c1\u7684\u5171\u4eab\u53cd\u9988\u3011"
SOURCE_KEYS = {"original_request_manifest", "original_prompt", "feedback_cards", "question_bank"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read_verified(identity, name):
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("feedback_source_identity_required:" + name)
    content = Path(identity["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
        raise ValueError("feedback_source_sha_mismatch:" + name)
    return content


def feedback_from_actual_prompt(prompt):
    """Extract the unique original shared-view block, not a rerendered prompt."""
    if prompt.count(FEEDBACK_MARKER) != 1:
        raise ValueError("original_feedback_marker_not_unique")
    payload = prompt.split(FEEDBACK_MARKER, 1)[1].lstrip()
    view, _ = json.JSONDecoder().raw_decode(payload)
    validate_shared_feedback_view(view)
    return view


def validate_shared_feedback_view(view):
    """Required task material only; absent evidence or counts are not invented."""
    if not isinstance(view, dict) or not view:
        raise ValueError("shared_feedback_view_required")
    for key in ("root_id", "source_candidate_id", "source_cluster_id"):
        if not isinstance(view.get(key), str) or not view[key].strip():
            raise ValueError("shared_feedback_identity_required:" + key)
    qa = view.get("historical_qa")
    if not isinstance(qa, dict):
        raise ValueError("shared_feedback_historical_qa_required")
    for key in ("question_type", "question"):
        if not isinstance(qa.get(key), str) or not qa[key].strip():
            raise ValueError("shared_feedback_qa_field_required:" + key)
    items = qa.get("options_or_items")
    if not isinstance(items, dict) or not items or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value.strip()
        for key, value in items.items()
    ):
        raise ValueError("shared_feedback_qa_items_required")
    gold = qa.get("canonical_gold")
    if not isinstance(gold, (str, list)) or not gold:
        raise ValueError("shared_feedback_qa_gold_required")
    if "evidence_refs" in qa and (
        not isinstance(qa["evidence_refs"], list)
        or any(not isinstance(ref, str) or not ref for ref in qa["evidence_refs"])
    ):
        raise ValueError("shared_feedback_evidence_shape")
    observations = view.get("solver_observations")
    if not isinstance(observations, dict) or not observations:
        raise ValueError("shared_feedback_observations_required")
    for model, observation in observations.items():
        if not isinstance(model, str) or not isinstance(observation, dict):
            raise ValueError("shared_feedback_observation_shape")
        for key in ("observation_snapshot_id", "evaluation_snapshot", "observation_kind"):
            if not isinstance(observation.get(key), str) or not observation[key].strip():
                raise ValueError("shared_feedback_observation_identity_required:" + key)
        for condition in ("full_context", "no_context"):
            value = observation.get(condition)
            if not isinstance(value, dict) or not value:
                raise ValueError("shared_feedback_condition_required:" + condition)
    return view


def _recover_feedback(sources, parent):
    if not isinstance(sources, dict) or set(sources) != SOURCE_KEYS:
        raise ValueError("feedback_frozen_sources_required")
    contents = {name: _read_verified(identity, name) for name, identity in sources.items()}
    manifest = json.loads(contents["original_request_manifest"])
    rows = manifest.get("messages")
    if manifest.get("purpose") != "normal_planner" or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("original_planner_request_identity")
    message = rows[0]
    expected_prompt = (Path(sources["original_request_manifest"]["path"]).parent / message["content_path"]).resolve()
    if (message.get("role") != "user"
            or expected_prompt != Path(sources["original_prompt"]["path"]).resolve()
            or message.get("content_sha256") != sources["original_prompt"]["sha256"]):
        raise ValueError("original_planner_prompt_source_mismatch")
    original_view = feedback_from_actual_prompt(contents["original_prompt"].decode("utf-8"))
    root = str(parent.get("root_id") or "")
    if original_view["root_id"] != root:
        raise ValueError("shared_feedback_root_mismatch")
    cards = [json.loads(line) for line in contents["feedback_cards"].decode("utf-8").splitlines() if line.strip()]
    cards = [row for row in cards if row.get("root_id") == root
             and row.get("source_candidate_id") == original_view["source_candidate_id"]
             and row.get("source_cluster_id") == original_view["source_cluster_id"]]
    if len(cards) != 1:
        raise ValueError("original_feedback_card_not_unique")
    candidates = [json.loads(line) for line in contents["question_bank"].decode("utf-8").splitlines() if line.strip()]
    candidates = [row for row in candidates if row.get("candidate_id") == original_view["source_candidate_id"]]
    if len(candidates) != 1:
        raise ValueError("original_feedback_candidate_not_unique")
    record = build_backend_record(cards[0], {candidates[0]["candidate_id"]: candidates[0]}, {root: parent})
    view = expansion_view_v3(record)
    if view != original_view:
        raise ValueError("compiled_feedback_differs_from_original_sent_feedback")
    return view


def _feedback_identity(view):
    return {"root_id": view["root_id"], "source_candidate_id": view["source_candidate_id"],
            "source_cluster_id": view["source_cluster_id"],
            "observation_snapshot_ids": {model: row["observation_snapshot_id"]
                                         for model, row in view["solver_observations"].items()},
            "shared_feedback_sha256": sha(view)}


def validate_bundle_feedback(bundle):
    """Fail closed before request/attempt claim; recheck local source lineage."""
    if not isinstance(bundle, dict):
        raise ValueError("incremental_bundle_required")
    view = validate_shared_feedback_view(bundle.get("shared_feedback_view"))
    lineage = bundle.get("shared_feedback_lineage")
    if not isinstance(lineage, dict) or lineage.get("version") != VERSION:
        raise ValueError("shared_feedback_lineage_required")
    root = (bundle.get("identity") or {}).get("root_id")
    if view["root_id"] != root or (bundle.get("parent") or {}).get("root_id") != root:
        raise ValueError("shared_feedback_root_mismatch")
    if lineage.get("original_plan_sha256") != bundle.get("original_plan_sha256"):
        raise ValueError("shared_feedback_original_plan_mismatch")
    if bundle.get("original_plan_sha256") != sha(bundle.get("original_plan")):
        raise ValueError("shared_feedback_original_plan_sha_mismatch")
    original = _recover_feedback(lineage.get("sources"), bundle["parent"])
    if view != original:
        raise ValueError("shared_feedback_differs_from_original_source")
    identity = _feedback_identity(original)
    if any(lineage.get(key) != value for key, value in identity.items()):
        raise ValueError("shared_feedback_lineage_identity_mismatch")
    return {"valid": True, **identity, "original_sent_content_match": True,
            "shared_compiler_content_match": True, "source_files_checked": sorted(SOURCE_KEYS)}


def build_failed_bundle(*, branch_id="unassigned-incremental-v1-122711", failed_run=FAILED):
    """Build the explicit frozen-plan entry without selecting new feedback/QA."""
    failed_run = Path(failed_run).resolve()
    freeze = _read(failed_run / "run_freeze_manifest.json")
    history_path = Path(freeze["input_path"])
    parent = build_round_history_parent(_read(history_path), round_index=3)
    parent.pop("prior_expanded_dialogue", None)
    selection = _read(failed_run / "planner/variant_selection.json")
    plan = selection["value"]
    request_manifest = failed_run / "planner/actual_request/request_manifest.json"
    actual_request = _read(request_manifest)
    prompt_path = request_manifest.parent / actual_request["messages"][0]["content_path"]
    feedback_sources = {key: _file_identity(path) for key, path in {
        "original_request_manifest": request_manifest, "original_prompt": prompt_path,
        "feedback_cards": COMPILER_RUN / "r03_expansion_feedback_cards_full.jsonl",
        "question_bank": REROUTE_RUN / "provisional_question_bank_entries.jsonl",
    }.items()}
    view = _recover_feedback(feedback_sources, parent)
    bundle = {"policy_id": POLICY_ID, "bundle_version": VERSION,
              "identity": {"root_id": parent["root_id"], "round_index": 3, "branch_id": branch_id},
              "parent": parent, "history_sha256": sha(parent),
              "original_plan": plan, "original_plan_sha256": sha(plan),
              "candidates": _read(failed_run / "memory/event_candidates.json")["candidates"],
              "verified_tracks": _read(failed_run / "event_verifier/materialized_tracks.json")["verified_tracks"],
              "retained_r03_turns": [], "realizer_mode": "text_free",
              "continuity_shape": {"speakers": sorted({turn["speaker"] for turn in parent["dialogue"]["turns"]}),
                                   "turn_count": len(parent["dialogue"]["turns"])},
              "frozen_sources": {key: _file_identity(path) for key, path in {
                  "history": history_path, "plan": failed_run / "planner/variant_selection.json",
                  "candidates": failed_run / "memory/event_candidates.json",
                  "tracks": failed_run / "event_verifier/materialized_tracks.json",
              }.items()},
              "shared_feedback_view": deepcopy(view),
              "shared_feedback_lineage": {"version": VERSION, **_feedback_identity(view),
                                          "original_plan_sha256": sha(plan), "sources": feedback_sources}}
    if "planner_contract_version" in selection or "shared_output" in selection:
        from .shared_planner_contract import VERSION as SHARED_VERSION, ROOT_SHARED_PROMPT_VERSION
        from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE
        if selection.get("planner_contract_version") != SHARED_VERSION or "shared_output" not in selection:
            raise ValueError("incomplete_shared_initial_selection")
        bundle.update(policy_id=ROOT_POLICY_ID, incremental_scope=ROOT_SCOPE,
                      supplement_prompt_version=ROOT_SHARED_PROMPT_VERSION,
                      planner_contract_version=SHARED_VERSION,
                      initial_planner_output=deepcopy(selection["shared_output"]))
    validate_bundle_feedback(bundle)
    return bundle
