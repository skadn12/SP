"""Build deterministic, structural R03 question-capacity measurements.

This module deliberately does not infer semantic facts.  State/event counts
are labelled candidate signals and lexical comparisons are only mechanical
repeat checks; no QA, Solver, auditor, or model is imported.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    AUTHORITATIVE_RUNS,
    CLOSURE_RUN,
    LIFECYCLE_RUN,
    R03_RUN_PREFIX,
    REROUTE_RUN,
    RUNS,
    QUARANTINE_DISPOSITION,
    SUCCESS_DISPOSITION,
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    return digest_bytes(canonical(value).encode("utf-8"))


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}:{line_no}")
            rows.append(row)
    return rows


def verify_source(path: Path, source_hashes: dict[str, str]) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_source_sidecar:{sidecar}")
    observed = digest_file(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"source_sha_mismatch:{path}")
    source_hashes[str(path)] = observed
    return observed


def artifact_bytes(name: str, value: Any) -> bytes:
    if name.endswith(".jsonl"):
        return b"".join((canonical(row) + "\n").encode("utf-8") for row in value)
    if name.endswith(".md"):
        text = str(value)
        return (text if text.endswith("\n") else text + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_artifact(output: Path, name: str, value: Any) -> str:
    path = output / name
    data = artifact_bytes(name, value)
    path.write_bytes(data)
    digest = digest_bytes(data)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) > 1}


def _load_candidate_questions(path: Path, candidate_ids: set[str]) -> list[dict[str, Any]]:
    """Read candidate text from the frozen provisional bank, not generated R03."""
    rows = read_jsonl(path)
    result = []
    for row in rows:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else row
        candidate_id = str(row.get("candidate_id") or candidate.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            continue
        result.append({
            "candidate_id": candidate_id,
            "question": candidate.get("question"),
            "question_type": candidate.get("question_type"),
            "normalized_stem_signature": row.get("normalized_stem_signature"),
            "evidence_signature": row.get("evidence_signature"),
        })
    return result


def _load_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    source_hashes: dict[str, str] = {}
    closure_report_path = CLOSURE_RUN / "final_r03_local_batch_closure_report.json"
    root_manifest_path = CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl"
    call_closure_path = CLOSURE_RUN / "r03_local_batch_call_closure.json"
    append_audit_path = CLOSURE_RUN / "r03_local_batch_append_only_audit.json"
    baseline_path = LIFECYCLE_RUN / "novelty_capacity_baseline.json"
    lifecycle_scope_path = LIFECYCLE_RUN / "active_train_scope_manifest.jsonl"
    lifecycle_clusters_path = LIFECYCLE_RUN / "global_solver_prompt_cluster_manifest.jsonl"
    provisional_path = REROUTE_RUN / "provisional_question_bank_entries.jsonl"
    reroute_family_path = REROUTE_RUN / "question_family_manifest.jsonl"
    fixed_paths = [
        closure_report_path, root_manifest_path, call_closure_path, append_audit_path,
        baseline_path, lifecycle_scope_path, lifecycle_clusters_path,
        provisional_path, reroute_family_path,
    ]
    for path in fixed_paths:
        verify_source(path, source_hashes)

    closure_report = read_json(closure_report_path)
    root_manifest = read_jsonl(root_manifest_path)
    call_closure = read_json(call_closure_path)
    append_audit = read_json(append_audit_path)
    baseline = read_json(baseline_path)
    lifecycle_scope = read_jsonl(lifecycle_scope_path)
    lifecycle_clusters = read_jsonl(lifecycle_clusters_path)
    reroute_families = read_jsonl(reroute_family_path)
    candidate_ids = {str(row["candidate_id"]) for row in lifecycle_scope}
    provisional_questions = _load_candidate_questions(provisional_path, candidate_ids)
    family_by_id = {str(row["candidate_id"]): row for row in reroute_families}
    for question in provisional_questions:
        family = family_by_id.get(question["candidate_id"]) or {}
        question["normalized_stem_signature"] = family.get("normalized_stem_signature")
    if len(provisional_questions) != len(candidate_ids):
        raise RuntimeError(
            f"frozen_candidate_question_coverage_mismatch:{len(provisional_questions)}:{len(candidate_ids)}")

    # Paths embedded in the authoritative closure are part of the frozen input
    # set and are verified before any measurement is made.
    for row in root_manifest:
        for key in ("authoritative_run_root", "source_memory_snapshot_path"):
            value = row.get(key)
            if key == "authoritative_run_root":
                continue
            if value:
                verify_source(Path(str(value)), source_hashes)
        artifacts = row.get("dialogue_artifacts") or {}
        for key in ("r03_delta_path", "r03_delta_turns_path", "cumulative_dialogue_path"):
            value = artifacts.get(key)
            if value:
                verify_source(Path(str(value)), source_hashes)
        run_root = Path(str(row["authoritative_run_root"]))
        for rel in ("input_dialogue_manifest.json", "lineage_manifest.json", "r03_delta.json", "r03_delta_turns.jsonl", "parent_r01_r02_r03_cumulative_dialogue.json", "planner/parsed_plan.json"):
            path = run_root / rel
            if path.is_file():
                verify_source(path, source_hashes)

    return ({
        "closure_report": closure_report,
        "root_manifest": root_manifest,
        "call_closure": call_closure,
        "append_audit": append_audit,
        "baseline": baseline,
        "lifecycle_scope": lifecycle_scope,
        "lifecycle_clusters": lifecycle_clusters,
        "reroute_families": reroute_families,
        "provisional_questions": provisional_questions,
    }, source_hashes)


def load_inputs() -> dict[str, Any]:
    inputs, source_hashes = _load_inputs()
    inputs["source_hashes"] = source_hashes
    return inputs


def _prior_segments(cumulative: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns = list(cumulative.get("turns") or [])
    parent_count = int(cumulative.get("original_parent_turn_count") or cumulative.get("parent_turn_count") or 0)
    segments = [{
        "segment_id": "PARENT",
        "round_index": 0,
        "first_turn_ref": turns[0].get("turn_ref") if turns else None,
        "last_turn_ref": turns[parent_count - 1].get("turn_ref") if parent_count and len(turns) >= parent_count else None,
        "turn_count": parent_count,
    }]
    for boundary in cumulative.get("round_boundaries") or []:
        index = int(boundary.get("round_index") or 0)
        if index >= 3:
            continue
        segments.append({
            "segment_id": f"R{index:02d}",
            "round_index": index,
            "first_turn_ref": boundary.get("first_turn_ref"),
            "last_turn_ref": boundary.get("last_turn_ref"),
            "turn_count": int(boundary.get("last_turn_index_1_based") or 0) - int(boundary.get("first_turn_index_1_based") or 0) + 1,
        })
    if len(segments) != 3:
        raise RuntimeError(f"prior_segment_count_not_three:{len(segments)}")
    return segments


def _scene_descriptor(scene: Mapping[str, Any]) -> str:
    return str(scene.get("new_development") or scene.get("required_transition") or scene.get("after_state") or scene.get("goal") or "").strip()


def _scene_rows(root_row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_root = Path(str(root_row["authoritative_run_root"]))
    if root_row.get("disposition") != SUCCESS_DISPOSITION:
        return [], [], [], {"root_id": root_row["root_id"], "run_number": root_row["authoritative_run_number"], "disposition": QUARANTINE_DISPOSITION, "scene_count": 0}
    delta = read_json(run_root / "r03_delta.json")
    plan = read_json(run_root / "planner" / "parsed_plan.json")
    cumulative = read_json(run_root / "parent_r01_r02_r03_cumulative_dialogue.json")
    scenes = []
    evidence = []
    delta_scenes = list(delta.get("scenes") or [])
    plan_scenes = list(plan.get("scenes") or [])
    if len(delta_scenes) != len(plan_scenes):
        raise RuntimeError(f"planner_delta_scene_count_mismatch:{root_row['root_id']}")
    for delta_scene, plan_scene in zip(delta_scenes, plan_scenes):
        # The promoted delta intentionally contains only dialogue.  Planner
        # state fields are joined by frozen scene order for offline metrics.
        if delta_scene.get("memory_event_id") != plan_scene.get("memory_event_id"):
            raise RuntimeError(
                f"planner_delta_memory_event_mismatch:{root_row['root_id']}:{delta_scene.get('scene_ref')}")
        scene = dict(plan_scene)
        scene.update(delta_scene)
        scene_ref = str(scene.get("scene_ref") or "")
        turns = list(scene.get("dialogue_turns") or [])
        refs = [str(turn.get("turn_ref") or "") for turn in turns]
        descriptor = _scene_descriptor(scene)
        state_change_candidate = bool(
            str(scene.get("before_state") or "").strip()
            and str(scene.get("after_state") or "").strip()
            and normalize_text(scene.get("before_state")) != normalize_text(scene.get("after_state"))
        )
        new_people = scene.get("new_people") or []
        people = []
        for person in new_people:
            if isinstance(person, dict):
                people.append(str(person.get("name") or person.get("person_ref") or ""))
            else:
                people.append(str(person))
        row = {
            "root_id": root_row["root_id"],
            "authoritative_run_number": root_row["authoritative_run_number"],
            "scene_ref": scene_ref,
            "session_ref": scene.get("session_ref"),
            "memory_event_id": scene.get("memory_event_id"),
            "turn_refs": refs,
            "turn_count": len(turns),
            "first_turn_ref": refs[0] if refs else None,
            "last_turn_ref": refs[-1] if refs else None,
            "speaker_count": len({str(turn.get("speaker") or "") for turn in turns if turn.get("speaker")}),
            "descriptor": descriptor,
            "descriptor_sha256": digest_value(descriptor),
            "development_mode": scene.get("development_mode"),
            "state_change_candidate": state_change_candidate,
            "new_people": sorted(set(person for person in people if person)),
            "semantic_state_event_verified": False,
            "evidence_ref_semantics": "STRUCTURAL_R03_TURN_REF_ONLY",
        }
        scenes.append(row)
        for turn in turns:
            evidence.append({
                "root_id": root_row["root_id"],
                "scene_ref": scene_ref,
                "session_ref": scene.get("session_ref"),
                "turn_ref": turn.get("turn_ref"),
                "speaker": turn.get("speaker"),
                "origin": "generated_extension",
                "structurally_valid": bool(re.fullmatch(r"R03:S\d{2}:T\d{3}", str(turn.get("turn_ref") or ""))),
            })

    joint = []
    for scene in scenes:
        for prior in _prior_segments(cumulative):
            payload = {
                "root_id": root_row["root_id"],
                "prior_segment_id": prior["segment_id"],
                "prior_first_turn_ref": prior["first_turn_ref"],
                "prior_last_turn_ref": prior["last_turn_ref"],
                "r03_scene_ref": scene["scene_ref"],
                "r03_first_turn_ref": scene["first_turn_ref"],
                "r03_last_turn_ref": scene["last_turn_ref"],
            }
            joint.append({
                "root_id": root_row["root_id"],
                "prior_segment_id": prior["segment_id"],
                "r03_scene_ref": scene["scene_ref"],
                "memory_event_id": scene.get("memory_event_id"),
                "bundle_signature": digest_value(payload),
                "evidence_ref_count": prior["turn_count"] + scene["turn_count"],
                "semantic_bundle_verified": False,
            })
    summary = {
        "root_id": root_row["root_id"],
        "run_number": root_row["authoritative_run_number"],
        "disposition": SUCCESS_DISPOSITION,
        "scene_count": len(scenes),
        "r03_turn_count": sum(row["turn_count"] for row in scenes),
        "structural_evidence_ref_count": len(evidence),
        "independent_scene_bundle_count": len({row["descriptor_sha256"] for row in scenes}),
        "state_change_candidate_count": sum(row["state_change_candidate"] for row in scenes),
        "new_people_candidate_count": sum(len(row["new_people"]) for row in scenes),
        "joint_cross_round_bundle_candidate_count": len(joint),
        "ordering_pair_candidate_count": len(scenes) * (len(scenes) - 1) // 2,
        "ordering_triple_candidate_count": 1 if len(scenes) >= 3 else 0,
        "semantic_state_event_verified_count": 0,
    }
    return scenes, evidence, joint, summary


def _lexical_repeat_audit(scene_rows: list[dict[str, Any]], old_questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old = []
    old_stems = set()
    for row in old_questions:
        question = str(row.get("question") or "")
        if not question:
            continue
        old.append((row, tokens(question), normalize_text(question)))
        if row.get("normalized_stem_signature"):
            old_stems.add(str(row["normalized_stem_signature"]))
    audits = []
    for scene in scene_rows:
        descriptor = scene.get("descriptor") or ""
        descriptor_tokens = tokens(descriptor)
        best = {"candidate_id": None, "similarity": 0.0, "question": None}
        for row, question_tokens, normalized_question in old:
            if not descriptor_tokens or not question_tokens:
                score = 0.0
            else:
                score = SequenceMatcher(None, normalize_text(descriptor), normalized_question).ratio()
            if score > best["similarity"]:
                best = {"candidate_id": row["candidate_id"], "similarity": round(score, 6), "question": row["question"]}
        # This is intentionally a high, mechanical threshold.  It is not a
        # semantic-family classifier and never changes lifecycle labels.
        repeated = float(best["similarity"]) >= 0.80
        audits.append({
            "root_id": scene["root_id"],
            "scene_ref": scene["scene_ref"],
            "descriptor_sha256": scene["descriptor_sha256"],
            "max_sequence_similarity": best["similarity"],
            "nearest_old_candidate_id": best["candidate_id"],
            "nearest_old_question": best["question"],
            "mechanical_lexical_repeat": repeated,
            "semantic_family_status": "NOT_AUDITED",
        })
    repeated_count = sum(bool(row["mechanical_lexical_repeat"]) for row in audits)
    return audits, {
        "old_active_candidate_count": len(old_questions),
        "old_normalized_stem_family_count": len(old_stems),
        "new_scene_descriptor_count": len(audits),
        "mechanical_lexical_repeat_count": repeated_count,
        "mechanical_lexical_repeat_rate": (repeated_count / len(audits)) if audits else 0.0,
        "threshold_sequence_similarity": 0.80,
        "semantic_family_claimed": False,
        "semantic_auditor_called": False,
    }


def _threshold_contract() -> dict[str, Any]:
    return {
        "version": "r03_capacity_threshold_contract_v1",
        "unit": "successful_authoritative_R03_scene_and_structural_bundle",
        "hard_gates": {
            "closure_valid": True,
            "success_root_count_min": 7,
            "independent_scene_bundle_count_min": 12,
            "joint_cross_round_bundle_candidate_count_min": 30,
            "estimated_new_stem_lower_bound_min": 12,
            "mechanical_lexical_repeat_rate_max": 0.80,
            "forbidden_calls_all_zero": True,
        },
        "estimate_contract": {
            "question_types": ["single_choice", "multi_select", "ordering"],
            "lower_bound": "one prospective stem per independent state/event scene",
            "upper_bound": "one prospective stem per scene and question type",
            "not_a_generation": True,
        },
        "semantic_limits": [
            "state_change_candidate_count is structural text comparison only",
            "evidence refs are generated R03 turn refs, not semantic evidence validation",
            "semantic_family_status remains NOT_AUDITED",
            "no keyword or semantic auditor is used",
        ],
    }


def build_all(inputs: Mapping[str, Any]) -> dict[str, Any]:
    closure = inputs["closure_report"]
    root_manifest = list(inputs["root_manifest"])
    if not closure.get("valid") or closure.get("status") != "R03_LOCAL_READONLY_DIALOGUE_BATCH_COMPLETE":
        raise RuntimeError("r03_local_batch_closure_not_valid")
    if closure.get("qa_generation_started") or closure.get("capacity_preflight_started"):
        raise RuntimeError("capacity_preflight_started_flag_inconsistent")
    by_root = {str(row["root_id"]): row for row in root_manifest}
    expected_roots = {root for root, _ in AUTHORITATIVE_RUNS}
    if set(by_root) != expected_roots or len(root_manifest) != len(expected_roots):
        raise RuntimeError("authoritative_root_set_mismatch")

    all_scene_rows: list[dict[str, Any]] = []
    all_evidence_rows: list[dict[str, Any]] = []
    all_joint_rows: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []
    for root_id, run_number in AUTHORITATIVE_RUNS:
        row = by_root[root_id]
        if int(row["authoritative_run_number"]) != run_number:
            raise RuntimeError(f"authoritative_run_number_mismatch:{root_id}")
        scenes, evidence, joint, summary = _scene_rows(row)
        all_scene_rows.extend(scenes)
        all_evidence_rows.extend(evidence)
        all_joint_rows.extend(joint)
        root_summaries.append(summary)

    success_roots = [row for row in root_summaries if row["disposition"] == SUCCESS_DISPOSITION]
    quarantine_roots = [row for row in root_manifest if row.get("disposition") == QUARANTINE_DISPOSITION]
    valid_evidence = [row for row in all_evidence_rows if row["structurally_valid"]]
    independent_bundles = {row["descriptor_sha256"] for row in all_scene_rows}
    state_candidates = sum(bool(row["state_change_candidate"]) for row in all_scene_rows)
    new_people_candidates = sum(len(row["new_people"]) for row in all_scene_rows)
    estimated_lower = len(independent_bundles)
    estimated_upper = estimated_lower * 3
    old_audits, lexical_audit = _lexical_repeat_audit(all_scene_rows, inputs["provisional_questions"])
    thresholds = _threshold_contract()
    call_closure = inputs["call_closure"]
    calls = {
        "memorax_search_calls": int(call_closure.get("memorax_search_calls") or 0),
        "memorax_add_calls": int(call_closure.get("memorax_add_calls") or 0),
        "memorax_status_calls": int(call_closure.get("memorax_status_calls") or 0),
        "memorax_mutation_calls": int(call_closure.get("memorax_mutation_calls") or 0),
        "qa_calls": int(call_closure.get("qa_writer_calls") or 0),
        "solver_calls": int(call_closure.get("solver_calls") or 0),
        "semantic_auditor_calls": int(call_closure.get("semantic_auditor_calls") or 0),
        "training_calls": int(call_closure.get("training_calls") or 0),
    }
    forbidden_zero = all(int(value or 0) == 0 for value in calls.values())
    gate_results = {
        "closure_valid": bool(closure.get("valid")),
        "success_root_count": len(success_roots) >= thresholds["hard_gates"]["success_root_count_min"],
        "independent_scene_bundle_count": len(independent_bundles) >= thresholds["hard_gates"]["independent_scene_bundle_count_min"],
        "joint_cross_round_bundle_candidate_count": len(all_joint_rows) >= thresholds["hard_gates"]["joint_cross_round_bundle_candidate_count_min"],
        "estimated_new_stem_lower_bound": estimated_lower >= thresholds["hard_gates"]["estimated_new_stem_lower_bound_min"],
        "mechanical_lexical_repeat_rate": lexical_audit["mechanical_lexical_repeat_rate"] <= thresholds["hard_gates"]["mechanical_lexical_repeat_rate_max"],
        "forbidden_calls_all_zero": forbidden_zero,
        "all_success_scene_refs_structurally_valid": len(valid_evidence) == len(all_evidence_rows) and len(all_evidence_rows) == sum(int(row.get("r03_turn_count") or 0) for row in root_summaries),
    }
    ready = all(gate_results.values())
    status = "R03_NOVEL_QUESTION_CAPACITY_PREFLIGHT_PASS" if ready else "R03_NOVEL_QUESTION_CAPACITY_PREFLIGHT_INSUFFICIENT"
    next_status = (
        "READY_FOR_TARGETED_EVOLUTION_MULTI_QA_AND_FRESH_EXPLORATION"
        if ready else "CONTINUE_R04_LOCAL_READONLY_DIALOGUE"
    )
    report = {
        "version": "h4_round1_r03_novel_question_capacity_preflight_v1",
        "status": status,
        "next_status": next_status,
        "capacity_preflight_started": True,
        "qa_generation_started": False,
        "solver_started": False,
        "semantic_auditor_called": False,
        "training_started": False,
        "memory_writes": 0,
        "root_count": len(root_manifest),
        "success_root_count": len(success_roots),
        "quarantine_root_count": len(quarantine_roots),
        "success_root_ids": [row["root_id"] for row in success_roots],
        "quarantine_root_ids": [row["root_id"] for row in quarantine_roots],
        "r03_turn_count": sum(int(row.get("r03_turn_count") or 0) for row in root_summaries),
        "new_r03_evidence_ref_count": len(all_evidence_rows),
        "structurally_valid_r03_evidence_ref_count": len(valid_evidence),
        "independent_evidence_bundle_count": len(independent_bundles),
        "state_event_change_candidate_count": state_candidates,
        "semantic_state_event_verified_count": 0,
        "new_people_candidate_count": new_people_candidates,
        "joint_cross_round_bundle_candidate_count": len(all_joint_rows),
        "joint_question_space_candidate_count": len(all_joint_rows) + sum(int(row.get("ordering_pair_candidate_count") or 0) + int(row.get("ordering_triple_candidate_count") or 0) for row in root_summaries),
        "estimated_new_prompt_stem_lower_bound": estimated_lower,
        "estimated_new_prompt_stem_upper_bound": estimated_upper,
        "mechanical_lexical_repeat_rate": lexical_audit["mechanical_lexical_repeat_rate"],
        "semantic_family_claimed": False,
        "gate_results": gate_results,
        "forbidden_call_counts": calls,
        "valid": ready,
    }
    md = "\n".join([
        "# h4_round1_r03_novel_question_capacity_preflight_v1",
        "",
        f"Status: `{status}`",
        f"Next: `{next_status}`",
        "",
        f"- Authoritative R03 roots: `{len(success_roots)}/{len(root_manifest)}` success; `{len(quarantine_roots)}` quarantine.",
        f"- Added structural R03 turn refs: `{len(all_evidence_rows)}`; valid format: `{len(valid_evidence)}`.",
        f"- Independent scene bundles: `{len(independent_bundles)}`; Parent/R01/R02 joint bundle candidates: `{len(all_joint_rows)}`.",
        f"- State/event change candidates: `{state_candidates}`; semantic verification: `0` (not performed by contract).",
        f"- Estimated new prompt stems: lower `{estimated_lower}`, upper `{estimated_upper}`; mechanical lexical repeat rate: `{lexical_audit['mechanical_lexical_repeat_rate']:.6f}`.",
        "- QA Writer, Solver, semantic auditor, training, and Memorax calls: `0`.",
        "- This report is a structural capacity estimate; it does not create or approve questions.",
        "",
    ])
    return {
        "artifacts": {
            "capacity_threshold_contract.json": thresholds,
            "r03_capacity_input_closure.json": {
                "closure_report": closure,
                "call_closure": inputs["call_closure"],
                "append_audit": inputs["append_audit"],
                "source_prefix": "authoritative R03 local-only runs live_6..live_13",
            },
            "r03_scene_evidence_manifest.jsonl": all_scene_rows,
            "r03_structural_evidence_ref_manifest.jsonl": all_evidence_rows,
            "r03_joint_evidence_bundle_manifest.jsonl": all_joint_rows,
            "r03_root_capacity_summary.json": {"roots": root_summaries, "total": report},
            "r03_estimated_prompt_stem_manifest.jsonl": [
                {
                    "root_id": row["root_id"],
                    "scene_ref": row["scene_ref"],
                    "question_type": question_type,
                    "prospective_stem_signature": digest_value({"scene_descriptor_sha256": row["descriptor_sha256"], "question_type": question_type}),
                    "estimate_only": True,
                    "semantic_family_status": "NOT_AUDITED",
                }
                for row in all_scene_rows
                for question_type in ("single_choice", "multi_select", "ordering")
            ],
            "r03_old_family_repeat_audit.json": lexical_audit,
            "r03_old_family_repeat_manifest.jsonl": old_audits,
            "final_r03_novel_question_capacity_preflight.json": report,
            "final_r03_novel_question_capacity_preflight.md": md,
        },
        "source_hashes": inputs["source_hashes"],
        "ready": ready,
    }
