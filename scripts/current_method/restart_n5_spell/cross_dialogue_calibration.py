from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .core import canonical_json, sha256_bytes, sha256_file
from .input_adapter import dialogue_content_hash, read_json


CROSS_DIALOGUE_CALIBRATION_VERSION = "cross_dialogue_rejection_sampling_v1"
CALIBRATION_FEEDBACK_VERSION = "semantic_leakage_rejection_sampling_feedback_v1"
CALIBRATION_LINEAGES = ("conv-26", "conv-30", "conv-41", "conv-43", "conv-44")
CALIBRATION_TARGET_LETTERS = (
    ("A", "B", "C"),
    ("B", "C", "D"),
    ("C", "D", "A"),
    ("D", "A", "B"),
    ("A", "C", "D"),
)


def verify_semantic_repair_feedback_source(source_root: Path) -> dict[str, Any]:
    """Attest the immutable 0/3 repair result before using it as quality feedback."""
    source_root = source_root.resolve()
    manifest_path = source_root / "config/run_manifest.json"
    provenance_path = source_root / "reports/post_run_provenance.json"
    summary_path = source_root / "frozen/semantic_repair_feedback_summary.json"
    audit_path = source_root / "reports/semantic_repair_series_audit.json"
    if any(not path.is_file() for path in (
            manifest_path, provenance_path, summary_path, audit_path)):
        raise RuntimeError("calibration_feedback_source_missing")
    manifest = read_json(manifest_path)
    provenance = read_json(provenance_path)
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    if (provenance.get("run_manifest_sha256") != sha256_file(manifest_path) or
            provenance.get("authoritative_release_provenance") is not True or
            provenance.get("provenance_mixed") is not False or
            manifest.get("feedback_summary_sha256") != sha256_file(summary_path) or
            manifest.get("series_audit_sha256") != sha256_file(audit_path)):
        raise RuntimeError("calibration_feedback_source_not_authoritative")
    expected = {"source_failure_type":"bad_ground_semantic_leakage",
        "source_item_count":3, "repair_attempted":3, "repair_json_valid":3,
        "repair_minimal_accepted":3, "repair_passed_grounding":0,
        "repair_reject_easy":0, "repair_reject_all_wrong":0,
        "repair_retain_frontier":0}
    if summary != expected or audit.get("first_retain_frontier_found") is not False:
        raise RuntimeError("calibration_feedback_result_mismatch")
    items = audit.get("items")
    if not isinstance(items, list) or len(items) != 3:
        raise RuntimeError("calibration_feedback_result_mismatch")
    item_attestations = []
    for item in items:
        run_root = Path(str(item.get("run_root") or ""))
        run_manifest = run_root / "config/run_manifest.json"
        run_report = run_root / "reports/semantic_repair_full_audit.json"
        run_provenance = run_root / "reports/post_run_provenance.json"
        if (not run_manifest.is_file() or not run_report.is_file() or
                not run_provenance.is_file() or
                sha256_file(run_manifest) != item.get("run_manifest_sha256") or
                sha256_file(run_report) != item.get("repair_report_sha256") or
                read_json(run_report).get("final_decision") !=
                "bad_ground_after_semantic_repair"):
            raise RuntimeError("calibration_feedback_item_integrity_failed")
        child_provenance = read_json(run_provenance)
        if (child_provenance.get("authoritative_release_provenance") is not True or
                child_provenance.get("provenance_mixed") is not False):
            raise RuntimeError("calibration_feedback_item_integrity_failed")
        item_attestations.append({"qa_id":item["qa_id"],
            "run_root":str(run_root), "manifest_sha256":sha256_file(run_manifest),
            "report_sha256":sha256_file(run_report),
            "provenance_sha256":sha256_file(run_provenance)})
    feedback = [{"qa_id":"cross_dialogue_semantic_leakage_summary",
        "final_decision":"bad_ground_semantic_leakage",
        "no_context_rewards":[1, 1, 1],
        "diagnosis":(
            "Previous correct options remained identifiable without dialogue in every "
            "tested option permutation. Generate fresh items rather than repairing "
            "previous items; use dialogue-internal alternatives with comparable plausibility.") }]
    return {"feedback_version":CALIBRATION_FEEDBACK_VERSION,
        "factual_source":False, "feedback":feedback,
        "source_root":str(source_root),
        "source_manifest_sha256":sha256_file(manifest_path),
        "source_provenance_sha256":sha256_file(provenance_path),
        "source_summary_sha256":sha256_file(summary_path),
        "source_audit_sha256":sha256_file(audit_path),
        "source_item_attestations":item_attestations}


def freeze_parent_dialogue(lineage: str, parent: dict[str, Any]) -> dict[str, Any]:
    """Build a comparable parent-only context and verify its frozen source row."""
    if lineage not in CALIBRATION_LINEAGES:
        raise ValueError("calibration_lineage_not_selected")
    turns = (parent.get("dialogue") or {}).get("turns")
    snapshot = parent.get("source_snapshot") or {}
    if not isinstance(turns, list) or not turns:
        raise RuntimeError("calibration_parent_dialogue_missing")
    source_path = Path(str(snapshot.get("source_path") or ""))
    line_number = snapshot.get("source_line_number")
    if not source_path.is_file() or not isinstance(line_number, int) or line_number < 1:
        raise RuntimeError("calibration_parent_source_missing")
    lines = source_path.read_text(encoding="utf-8").splitlines()
    if line_number > len(lines):
        raise RuntimeError("calibration_parent_source_missing")
    source_row = json.loads(lines[line_number - 1])
    row_sha = sha256_bytes(canonical_json(source_row).encode())
    content_sha = dialogue_content_hash(turns)
    if (row_sha != snapshot.get("row_sha256") or
            content_sha != snapshot.get("dialogue_content_sha256") or
            parent.get("split") != "train"):
        raise RuntimeError("calibration_parent_source_integrity_failed")
    frozen_turns = []
    seen_ids: set[str] = set()
    for turn in turns:
        turn_id = str(turn.get("turn_id") or "")
        if (not turn_id or turn_id in seen_ids or not isinstance(turn.get("speaker"), str) or
                not isinstance(turn.get("text"), str)):
            raise RuntimeError("calibration_parent_turn_integrity_failed")
        seen_ids.add(turn_id)
        frozen_turns.append({"turn_id":turn_id,
            "session_id":turn_id.split(":", 1)[0],
            "speaker":turn["speaker"], "text":turn["text"]})
    context = {"schema_version":"cross_dialogue_parent_only_context_v1",
        "lineage":lineage, "parent_only":True, "realizer_sessions":0,
        "combined_dialogue":frozen_turns}
    return {"context":context, "source_attestation":{
        "lineage":lineage, "parent_sample_id":parent.get("sample_id"),
        "parent_source_id":parent.get("source_id"), "parent_split":parent.get("split"),
        "source_path":str(source_path), "source_line_number":line_number,
        "source_row_sha256":row_sha, "dialogue_content_sha256":content_sha,
        "parent_identity_sha256":snapshot.get("parent_identity_sha256"),
        "reconstructed_graph_sha256":snapshot.get("reconstructed_graph_sha256"),
        "turn_count":len(frozen_turns)}}


def calibration_plan() -> list[dict[str, Any]]:
    return [{"dialogue_index":index, "lineage":lineage,
             "assigned_target_letters":list(CALIBRATION_TARGET_LETTERS[index]),
             "questioner_batches":1, "mcqs_per_batch":3,
             "semantic_repair_enabled":False}
            for index, lineage in enumerate(CALIBRATION_LINEAGES)]


def summarize_calibration(dialogues: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = [item for dialogue in dialogues for item in dialogue.get("items", [])]
    decisions: dict[str, int] = {}
    diagnoses: dict[str, int] = {}
    for item in candidates:
        decision = str(item.get("final_decision") or "not_solver_evaluated")
        decisions[decision] = decisions.get(decision, 0) + 1
        diagnostic = item.get("no_context_permutation_diagnostic")
        if diagnostic:
            diagnoses[str(diagnostic)] = diagnoses.get(str(diagnostic), 0) + 1
    retained = decisions.get("retain_frontier", 0)
    if retained >= 2:
        route_decision = "training_free_route_feasible"
    elif retained == 0 and diagnoses.get("bad_ground_semantic_leakage", 0) > len(candidates) / 2:
        route_decision = "single_stage_api_questioner_not_supported"
    else:
        route_decision = "calibration_inconclusive"
    return {"calibration_version":CROSS_DIALOGUE_CALIBRATION_VERSION,
        "dialogue_count":len(dialogues), "raw_candidate_count":sum(
            int(dialogue.get("raw_candidate_count", 0)) for dialogue in dialogues),
        "minimal_accepted_count":sum(int(dialogue.get("minimal_accepted_count", 0))
                                     for dialogue in dialogues),
        "solver_evaluated_count":sum(item.get("solver_evaluated") is True
                                     for item in candidates),
        "final_decisions":decisions,
        "no_context_permutation_diagnostics":diagnoses,
        "retain_frontier_count":retained, "route_decision":route_decision}
