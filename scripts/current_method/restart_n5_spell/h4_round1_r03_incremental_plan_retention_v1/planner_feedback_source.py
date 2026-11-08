"""Read a selected Expansion feedback export without using the QA snapshot API."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..h4_round1_r03_feedback_view_v2.compiler import expansion_view_v3
from ..round0_ordering_active_feedback_exclusion_v1 import NO_FEEDBACK, VERSION as FILTER_VERSION
from ..round0_ordering_active_feedback_exclusion_v1.loader import load_export
from ..round0_ordering_active_feedback_exclusion_v1.renderers import validate_envelope
from ..round0_ordering_active_feedback_exclusion_v1.sources import file_sha, read, value_sha


VERSION = "planner_explicit_filtered_feedback_v1"
REQUIRED_VIEWS = (
    "policy.json", "candidate_active_eligibility.jsonl", "active_eligibility.jsonl",
    "active_target_weights.jsonl", "active_composition_weights.jsonl",
    "active_expansion_seed_pool.jsonl", "active_expansion_selection.jsonl",
    "active_expansion_reserves.jsonl", "active_routing_exclusions.jsonl",
    "active_expansion_feedback_cards.jsonl", "active_composition_parent_pool.jsonl",
    "active_composition_pairs.jsonl", "root_feedback_states.jsonl",
    "history_preservation.json", "fresh_historical_backend_view.jsonl",
    "active_question_bank_projection.jsonl", "active_shared_feedback_records.jsonl",
    "excluded_candidates.jsonl", "role_feedback_envelopes.jsonl",
)


def _one(rows, key, identity, *, optional=False):
    matches = [row for row in rows if row.get(key) == identity]
    if optional and not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"planner_feedback_identity_not_unique:{key}:{identity}")
    return matches[0]


def _export_views(loaded):
    manifest = Path(loaded["export_manifest"])
    if file_sha(manifest) != loaded["export_manifest_sha256"]:
        raise ValueError("planner_export_inventory_changed_after_load")
    inventory = read(manifest)
    rows = {Path(row["path"]).name: row for row in inventory}
    if len(rows) != len(inventory) or set(REQUIRED_VIEWS) - set(rows):
        raise ValueError("planner_export_required_view_missing_or_duplicate")
    views, identities = {}, {}
    for name in REQUIRED_VIEWS:
        row = rows[name]
        path = Path(row["path"]).resolve()
        if path.parent != manifest.parent or file_sha(path) != row["sha256"]:
            raise ValueError(f"planner_export_view_sha_or_path_mismatch:{name}")
        views[name] = read(path)
        identities[name] = {"path": str(path), "sha256": row["sha256"]}
    if views["policy.json"].get("version") != FILTER_VERSION:
        raise ValueError("planner_export_policy_version_mismatch")
    return views, identities


def _validate_history(loaded, root_id, identity):
    if not isinstance(identity, Mapping) or identity.get("root_id") != root_id:
        raise ValueError("planner_history_identity_root_mismatch")
    reference = loaded["reference_identity"][root_id]
    # This preparation plans a new R03 from the frozen R02 branch. It does not
    # relabel that history as a QA-ready R03 or alter old evaluation snapshots.
    if identity.get("source_prefix") != "Parent+R01+R02-feedback":
        raise ValueError("planner_history_requires_authoritative_r02_feedback_prefix")
    if any(identity.get(key) != reference.get(key) for key in (
        "source_prefix", "input_dialogue_file_sha256", "complete_dialogue_sha256",
    )):
        raise ValueError("planner_history_not_export_authoritative_reference")
    path = Path(identity["input_dialogue_path"]).resolve()
    if file_sha(path) != identity["input_dialogue_file_sha256"]:
        raise ValueError("planner_history_file_sha_mismatch")
    dialogue = read(path)
    payload = {key: value for key, value in dialogue.items() if key != "complete_dialogue_sha256"}
    if (dialogue.get("root_id") != root_id
            or dialogue.get("complete_dialogue_sha256") != identity["complete_dialogue_sha256"]
            or value_sha(payload) != identity["complete_dialogue_sha256"]):
        raise ValueError("planner_history_content_sha_mismatch")
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not turns or dialogue.get("turn_count") != len(turns):
        raise ValueError("planner_history_turn_count_mismatch")
    refs = [turn.get("turn_ref") for turn in turns]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(refs) != len(set(refs)):
        raise ValueError("planner_history_turn_identity_missing_or_duplicate")
    rounds = {part for ref in refs for part in ref.split(":") if part.startswith("R") and part[1:].isdigit()}
    if rounds != {"R01", "R02"}:
        raise ValueError("planner_history_rounds_not_r01_r02")
    for turn in turns:
        if not isinstance(turn.get("speaker"), str) or not turn["speaker"] or not isinstance(turn.get("text"), str):
            raise ValueError("planner_history_turn_content_missing")
    checked = deepcopy(dict(identity))
    checked.update({"input_dialogue_path": str(path), "turn_count": len(turns),
                    "adapter_validation": "AUTHORITATIVE_R02_HISTORY_BOUND_FOR_NEW_R03_PLANNING"})
    return checked, set(refs) | {ref[len(root_id) + 1:] for ref in refs if ref.startswith(root_id + ":")}


def _select_expansion(loaded, views, root_id):
    key = f"expansion:{root_id}"
    envelope = loaded["envelopes"].get(key)
    if envelope is None:
        raise ValueError("planner_expansion_envelope_unavailable_in_export")
    validate_envelope(envelope, set(loaded["scope"]))
    if envelope.get("role") != "expansion" or envelope.get("selection_key") != key or envelope["root_id"] != root_id:
        raise ValueError("planner_expansion_envelope_identity_mismatch")
    selected = [row for row in views["active_expansion_selection.jsonl"] if row["root_id"] == root_id]
    cards = [row for row in views["active_expansion_feedback_cards.jsonl"] if row["root_id"] == root_id]
    pool = [row for row in views["active_expansion_seed_pool.jsonl"] if row["root_id"] == root_id]
    records = envelope["records"]
    if len(records) > 1 or len(selected) != min(1, len(pool)) or len(cards) != len(selected) or len(records) != len(selected):
        raise ValueError("planner_expansion_selection_cardinality_mismatch")
    if not records:
        if envelope["status"] != NO_FEEDBACK:
            raise ValueError("planner_expansion_explicit_no_feedback_required")
        return deepcopy(envelope), []
    record, selection, card = records[0], selected[0], cards[0]
    cid, cluster = record["source_candidate_id"], record["source_cluster_id"]
    if (cid not in loaded["records"] or record != loaded["records"][cid]
            or cid in {row["candidate_id"] for row in views["excluded_candidates.jsonl"]}):
        raise ValueError("planner_selected_record_outside_active_export")
    if (selection.get("canonical_representative_candidate_id") != cid
            or selection.get("solver_prompt_fingerprint") != cluster
            or card.get("source_candidate_id") != cid or card.get("source_cluster_id") != cluster
            or card.get("filter_version") != FILTER_VERSION):
        raise ValueError("planner_card_selection_identity_mismatch")
    scope = loaded["scope"][cid]
    if (scope.get("filter_version") != FILTER_VERSION or scope.get("split") != "train"
            or scope.get("active_feedback_scope_allowed") is not True
            or (record["source_arm"] == "ROUND0_PARENT_PLUS_R01" and record["historical_qa"]["question_type"] == "ordering")):
        raise ValueError("planner_selected_source_disallowed_scope")
    candidate_eligibility = _one(views["candidate_active_eligibility.jsonl"], "candidate_id", cid)
    eligibility = _one(views["active_eligibility.jsonl"], "canonical_representative_candidate_id", cid)
    weight = _one(views["active_target_weights.jsonl"], "canonical_representative_candidate_id", cid)
    seed = _one(pool, "canonical_representative_candidate_id", cid)
    for row in (candidate_eligibility, eligibility, weight, seed):
        if row.get("root_id") != root_id or row.get("solver_prompt_fingerprint") != cluster:
            raise ValueError("planner_selected_role_view_identity_mismatch")
    if (candidate_eligibility.get("filter_version") != FILTER_VERSION
            or any(candidate_eligibility.get(field) is not True for field in (
                "scope_allowed", "independent_active_observation", "targeted_eligible"))
            or eligibility.get("active_feedback_only") is not True
            or eligibility.get("questioner_targeted_evolution_eligible") is not True
            or weight.get("active_feedback_only") is not True or weight.get("eligible") is not True
            or seed.get("targeted_eligible_frozen") is not True):
        raise ValueError("planner_selected_source_not_expansion_eligible")
    for row in (weight, selection, seed):
        if type(row.get("raw_weight")) not in (int, float) or not row["raw_weight"] > 0:
            raise ValueError("planner_selected_source_nonpositive_weight")
    for field in ("raw_weight", "normalized_weight", "priority_key", "tier", "u"):
        if selection.get(field) != seed.get(field) or card.get("selection_weight", {}).get(field) != selection.get(field):
            raise ValueError("planner_selected_weight_or_routing_mismatch")
    if (selection["raw_weight"] != weight["raw_weight"]
            or selection["normalized_weight"] != weight["within_root_normalized_weight"]
            or selection.get("selection_disposition") != "PRIMARY"
            or seed.get("selection_disposition") != "PRIMARY"
            or selection.get("branch_compatibility") != seed.get("branch_compatibility")
            or card.get("branch_compatibility") != seed.get("branch_compatibility")
            or seed.get("branch_compatibility") not in {"CURRENT_FEEDBACK_BRANCH", "SHARED_PREFIX_COMPATIBLE"}):
        raise ValueError("planner_selected_weight_or_routing_mismatch")
    if any(row.get("candidate_id") == cid for row in views["active_routing_exclusions.jsonl"]):
        raise ValueError("planner_selected_source_routing_excluded")
    if (card.get("source_question") != record["historical_qa"]["question"]
            or card.get("source_question_type") != record["historical_qa"]["question_type"]
            or card.get("source_evidence_refs") != record["historical_qa"]["evidence_refs"]
            or card.get("source_sha256", {}).get("candidate") != record["backend_source"]["candidate_record_sha256"]
            or card.get("source_sha256", {}).get("cluster") != record["backend_source"]["cluster_projection_sha256"]):
        raise ValueError("planner_selected_card_content_mismatch")
    return deepcopy(envelope), [{
        "source_candidate_id": cid, "source_cluster_id": cluster,
        "candidate_scope": deepcopy(scope), "candidate_eligibility": deepcopy(candidate_eligibility),
        "role_eligibility": deepcopy(eligibility), "target_weight": deepcopy(weight),
        "routing_and_selection": deepcopy(selection), "expansion_card": deepcopy(card),
        "shared_record_sha256": value_sha(record),
        "historical_qa_evaluation_identity": {
            "context_signature": loaded["bank"][cid].get("context_signature"),
            "source_snapshot": deepcopy(scope.get("source_snapshot")),
            "candidate_record_sha256": record["backend_source"]["candidate_record_sha256"],
            "observation_snapshot_ids": {model: obs["observation_snapshot_id"] for model, obs in record["performance"].items()},
            "scores_reinterpreted_against_planning_history": False,
        },
    }]


def load_planner_feedback(
    *, export_manifest: str | Path, root_id: str,
    planning_history_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume the export's one already selected Expansion card for this root."""
    loaded = load_export(export_manifest)
    views, artifact_identities = _export_views(loaded)
    if root_id not in loaded["reference_identity"] or f"expansion:{root_id}" not in loaded["envelopes"]:
        raise ValueError("planner_expansion_envelope_unavailable_in_export")
    history, history_refs = _validate_history(loaded, root_id, planning_history_identity)
    envelope, selection = _select_expansion(loaded, views, root_id)
    for record in envelope["records"]:
        if any(ref not in history_refs and f"{root_id}:{ref}" not in history_refs for ref in record["historical_qa"]["evidence_refs"]):
            raise ValueError("planner_selected_evidence_absent_from_authoritative_history")
    projections = [expansion_view_v3(record) for record in envelope["records"]]
    view = {"filter_version": FILTER_VERSION, "root_id": root_id,
            "status": envelope["status"], "reason": envelope["reason"], "feedback": projections}
    return {
        "adapter_version": VERSION, "feedback_version": FILTER_VERSION,
        "status": envelope["status"], "reason": envelope["reason"], "role": "expansion",
        "root_id": root_id, "selection_key": envelope["selection_key"],
        "feedback_envelope": envelope, "shared_feedback_view": view,
        "planning_history_identity": history,
        "export_lineage": {
            "export_manifest": loaded["export_manifest"],
            "export_manifest_sha256": loaded["export_manifest_sha256"],
            "source_manifest": loaded["source_manifest_path"],
            "source_manifest_sha256": loaded["source_manifest_sha256"],
            "filter_version": FILTER_VERSION, "loaded_artifacts": artifact_identities,
            "candidate_scope_count": len(loaded["scope"]),
            "active_shared_record_count": len(loaded["records"]),
            "selected_feedback_count": len(envelope["records"]),
            "selection_recomputed": False, "original_bank_used_for_selection": False,
            "history_preservation": deepcopy(views["history_preservation.json"]),
            "source_mapping": selection,
        },
        "field_transport_validated": True, "gold_semantics_validated": False,
        "qa_snapshot_binding_performed": False, "external_calls": 0,
    }
