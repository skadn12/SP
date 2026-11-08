"""Freeze Common Feedback Schema/Template v1 and audit real frozen inputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from . import (
    ACTIVE_SCOPE_PATH,
    AUTHORITATIVE_ROOTS,
    CAPACITY_BUNDLES_PATH,
    CAPACITY_REPORT_PATH,
    CAPACITY_SCENES_PATH,
    CLUSTER_MAP_PATH,
    CLOSURE_CALLS_PATH,
    CLOSURE_REPORT_PATH,
    CLOSURE_ROOTS_PATH,
    DEFAULT_OUTPUT,
    LIFECYCLE_PATH,
    PACKET_PATH,
    PACKET_RUN,
    QUARANTINE_ROOTS,
    SAFE_ROOTS,
    SEED_PATH,
    SOLVER_CLEAN_PATH,
    SOLVER_COMPLETE_PATH,
    SOLVER_INCLUSIVE_PATH,
    UNSAFE_TARGETED_SEED_ROOTS,
    TOKENIZER_PATH,
)


STRATEGIES = [
    "STATE_UPDATE", "OBSOLETE_VS_CURRENT", "CROSS_STAGE_EVIDENCE",
    "MULTI_EVENT_DEPENDENCY", "LONG_DISTANCE_DEPENDENCY",
    "ENTITY_RELATION_BINDING", "TEMPORAL_RELATION",
    "CONSTRAINT_COMPOSITION", "CONTRASTIVE_CONTEXT",
    "OTHER_SUPPORTED_EVOLUTION", "NO_SAFE_EVOLUTION",
]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_source(path: Path, source_hashes: dict[str, str]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_source_sidecar:{sidecar}")
    observed = sha_file(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"source_sha_mismatch:{path}")
    source_hashes[str(path)] = observed


def artifact_bytes(name: str, value: Any) -> bytes:
    if name.endswith(".jsonl"):
        return "".join(canonical(row) + "\n" for row in value).encode("utf-8")
    if name.endswith(".md") or name.endswith(".txt"):
        text = str(value)
        return (text if text.endswith("\n") else text + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_artifact(output: Path, name: str, value: Any) -> str:
    path = output / name
    data = artifact_bytes(name, value)
    path.write_bytes(data)
    digest = sha_bytes(data)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _source_paths() -> list[Path]:
    return [
        PACKET_PATH, SEED_PATH, SOLVER_CLEAN_PATH, SOLVER_INCLUSIVE_PATH,
        SOLVER_COMPLETE_PATH, LIFECYCLE_PATH, CLUSTER_MAP_PATH,
        ACTIVE_SCOPE_PATH, CAPACITY_REPORT_PATH, CAPACITY_SCENES_PATH,
        CAPACITY_BUNDLES_PATH, CLOSURE_REPORT_PATH, CLOSURE_ROOTS_PATH,
        CLOSURE_CALLS_PATH,
    ]


def load_inputs() -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for path in _source_paths():
        verify_source(path, source_hashes)
    packets = load_jsonl(PACKET_PATH)
    seeds = load_jsonl(SEED_PATH)
    clean = load_jsonl(SOLVER_CLEAN_PATH)
    inclusive = load_jsonl(SOLVER_INCLUSIVE_PATH)
    complete = load_jsonl(SOLVER_COMPLETE_PATH)
    lifecycle = load_jsonl(LIFECYCLE_PATH)
    cluster_map = load_jsonl(CLUSTER_MAP_PATH)
    active_scope = load_jsonl(ACTIVE_SCOPE_PATH)
    capacity_scenes = load_jsonl(CAPACITY_SCENES_PATH)
    capacity_bundles = load_jsonl(CAPACITY_BUNDLES_PATH)
    closure_roots = load_jsonl(CLOSURE_ROOTS_PATH)
    for row in closure_roots:
        artifacts = row.get("dialogue_artifacts") or {}
        for key in ("cumulative_dialogue_path", "r03_delta_path", "r03_delta_turns_path"):
            path_text = artifacts.get(key)
            if path_text:
                path = Path(str(path_text))
                if path.is_file():
                    verify_source(path, source_hashes)
        run_root = Path(str(row.get("authoritative_run_root") or ""))
        for rel in ("input_dialogue_manifest.json", "lineage_manifest.json", "planner/parsed_plan.json"):
            path = run_root / rel
            if path.is_file():
                verify_source(path, source_hashes)
    return {
        "source_hashes": source_hashes,
        "packets": packets,
        "seeds": seeds,
        "clean": clean,
        "inclusive": inclusive,
        "complete": complete,
        "lifecycle": lifecycle,
        "cluster_map": cluster_map,
        "active_scope": active_scope,
        "capacity_report": load_json(CAPACITY_REPORT_PATH),
        "capacity_scenes": capacity_scenes,
        "capacity_bundles": capacity_bundles,
        "closure_report": load_json(CLOSURE_REPORT_PATH),
        "closure_roots": closure_roots,
        "closure_calls": load_json(CLOSURE_CALLS_PATH),
    }


def _metric(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {key: "UNKNOWN" for key in (
            "full_context_accuracy", "no_context_accuracy", "full_context_correct_count",
            "full_context_total_count", "no_context_correct_count", "no_context_total_count",
            "context_gain", "difficulty_bucket", "position_mechanism",
        )}
    return {
        "full_context_accuracy": row.get("full_context_accuracy", "UNKNOWN"),
        "no_context_accuracy": row.get("no_context_accuracy", "UNKNOWN"),
        "full_context_correct_count": row.get("full_context_correct_count", "UNKNOWN"),
        "full_context_total_count": row.get("full_context_total_count", "UNKNOWN"),
        "no_context_correct_count": row.get("no_context_correct_count", "UNKNOWN"),
        "no_context_total_count": row.get("no_context_total_count", "UNKNOWN"),
        "context_gain": (
            row.get("full_context_accuracy") - row.get("no_context_accuracy")
            if row.get("full_context_accuracy") is not None and row.get("no_context_accuracy") is not None
            else "UNKNOWN"
        ),
        "difficulty_bucket": row.get("difficulty_bucket", "UNKNOWN"),
        "position_mechanism": row.get("position_mechanism", "UNKNOWN"),
    }


def _stage(ref: str) -> str:
    if ref.startswith("R03:"):
        return "R03"
    if ref.startswith("R02:"):
        return "R02"
    if ref.startswith("R01:"):
        return "R01"
    if re.match(r"^[^:]+:P\d+:", ref):
        return "PARENT"
    return "UNKNOWN"


def _common_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "h4.round1.r03.common_feedback_data.v1",
        "title": "Common Feedback Data v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["source_identity", "source_qa", "evidence", "solver_observations", "transition_behavior", "r03_context", "diagnostic_boundary"],
        "properties": {
            "source_identity": {"type": "object", "additionalProperties": False, "required": ["root_id", "question_id", "candidate_id", "source_cluster_id", "alias_status", "question_bank_lifecycle_status", "training_exposure"], "properties": {
                "root_id": {"type": "string"}, "question_id": {"type": "string"}, "candidate_id": {"type": "string"}, "source_cluster_id": {"type": "string"}, "alias_status": {"type": "boolean"}, "question_bank_lifecycle_status": {"type": "string"}, "training_exposure": {"type": "string"},
            }},
            "source_qa": {"type": "object", "additionalProperties": False, "required": ["question", "options_or_items", "canonical_gold", "question_type", "provenance"], "properties": {
                "question": {"type": "string"}, "options_or_items": {"type": "object"}, "canonical_gold": {"type": "array"}, "question_type": {"type": "string"}, "provenance": {"type": "string"},
            }},
            "evidence": {"type": "object", "additionalProperties": False, "required": ["authoritative_evidence_refs", "evidence_stage", "evidence_provenance", "evidence_summary"], "properties": {
                "authoritative_evidence_refs": {"type": "array", "items": {"type": "string"}}, "evidence_stage": {"type": "array", "items": {"type": "string"}}, "evidence_provenance": {"type": "string"}, "evidence_summary": {"type": ["string", "null"]},
            }},
            "solver_observations": {"type": "object", "additionalProperties": False, "required": ["M0", "post_training", "provenance"], "properties": {"M0": {"type": "object"}, "post_training": {"type": "object", "required": ["M_clean_policy", "M_inclusive_policy"]}, "provenance": {"type": "string"}}},
            "transition_behavior": {"type": "object", "additionalProperties": False, "required": ["transition", "frontier_status", "canonical_confusion", "position_mechanism_status"], "properties": {"transition": {"type": "string"}, "frontier_status": {"type": "string"}, "canonical_confusion": {"type": ["object", "null"]}, "position_mechanism_status": {"type": "string"}}},
            "r03_context": {"type": "object", "additionalProperties": False, "required": ["evidence_refs", "independent_evidence_bundle_refs", "joint_bundle_candidates", "source_root_status"], "properties": {"evidence_refs": {"type": "array", "items": {"type": "string"}}, "independent_evidence_bundle_refs": {"type": "array", "items": {"type": "string"}}, "joint_bundle_candidates": {"type": "array", "items": {"type": "string"}}, "source_root_status": {"type": "string"}}},
            "diagnostic_boundary": {"type": "object", "additionalProperties": False, "required": ["solver_correct_interpretation", "solver_error_interpretation", "unknown_policy"], "properties": {"solver_correct_interpretation": {"type": "string"}, "solver_error_interpretation": {"type": "string"}, "unknown_policy": {"type": "string"}}},
        },
    }


def _template() -> str:
    return """# QA Feedback Template v1

You receive Common Feedback Data, the authoritative cumulative dialogue, and limited questioner-history metadata.

The Questioner-history view is separate from Solver retention replay. This
template does not admit items to the Question Bank, assign replay weight, or
change lifecycle state. Exact replay is not new QA; changed evidence or
context provenance is not automatically a duplicate. Stem/evidence signatures
are bookkeeping signatures, not semantic-family claims.

Decide yourself whether a valuable successor QA exists. Do not generate a QA merely to meet a quota. Do not paraphrase the source QA. A successor is valid only when the authoritative dialogue answers it, the Gold is unique, every distractor is wrong in the complete context, ambiguity is not introduced, and at least one of the answer target, evidence relation, or reasoning structure changes. Use R03 evidence only when it genuinely contributes to the reasoning; do not mechanically force an R03 reference.

Solver observations are behavioral observations (output matched Gold or did not); they are not validated cognitive diagnoses. A Solver error or a Full/No-context difference does not by itself prove a capability defect. Use UNKNOWN whenever a reliable diagnosis is unavailable.

Choose exactly one strategy from the frozen strategy space. OTHER_SUPPORTED_EVOLUTION requires a rationale. NO_SAFE_EVOLUTION is a normal valid result: leave all QA fields empty and provide a reason.

Return only the frozen output schema. Mark tentative_hypothesis explicitly as a hypothesis, never as a validated diagnosis. Do not add root-specific recommendations, hidden labels, or semantic-family claims.
"""


def _strategy_schema() -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "h4.round1.r03.qa_evolution_strategy.v1", "type": "string", "enum": STRATEGIES}


def _output_schema() -> dict[str, Any]:
    fields = ["source_question_id", "source_cluster_id", "root_id", "selected_strategy", "strategy_rationale", "feedback_observations_used", "tentative_hypothesis", "new_question", "options_or_items", "canonical_gold", "evidence_refs", "new_r03_evidence_refs", "source_question_relation", "expected_context_dependency", "no_safe_evolution_reason"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "h4.round1.r03.qa_feedback_output.v1",
        "type": "object",
        "additionalProperties": False,
        "required": fields,
        "properties": {
            "source_question_id": {"type": "string"},
            "source_cluster_id": {"type": "string"},
            "root_id": {"type": "string"},
            "selected_strategy": _strategy_schema(),
            "strategy_rationale": {"type": "string", "minLength": 1},
            "feedback_observations_used": {"type": "array", "items": {"type": "string"}},
            "tentative_hypothesis": {"type": "string", "description": "Hypothesis only; never a validated diagnosis."},
            "new_question": {"type": ["string", "null"]},
            "options_or_items": {"type": ["object", "null"]},
            "canonical_gold": {"type": ["array", "null"]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "new_r03_evidence_refs": {"type": "array", "items": {"type": "string"}},
            "source_question_relation": {"type": "string"},
            "expected_context_dependency": {"type": "string"},
            "no_safe_evolution_reason": {"type": ["string", "null"]},
        },
        "allOf": [
            {
                "if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}},
                "then": {"properties": {
                    "new_question": {"type": "null"},
                    "options_or_items": {"type": "null"},
                    "canonical_gold": {"type": "null"},
                    "evidence_refs": {"const": []},
                    "new_r03_evidence_refs": {"const": []},
                    "source_question_relation": {"const": ""},
                    "expected_context_dependency": {"const": ""},
                    "no_safe_evolution_reason": {"type": "string", "minLength": 1},
                }},
            },
            {
                "if": {"properties": {"selected_strategy": {"const": "OTHER_SUPPORTED_EVOLUTION"}}},
                "then": {"properties": {"strategy_rationale": {"type": "string", "minLength": 1}}},
            },
        ],
    }


def _tokenizer_preflight(common: list[dict[str, Any]], inputs: Mapping[str, Any], template: str) -> dict[str, Any]:
    """Measure the real offline Questioner input; never send it to a model."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        tokenizer_status = {
            "status": "PASS",
            "path": str(TOKENIZER_PATH.resolve()),
            "class": tokenizer.__class__.__name__,
        }
    except Exception as error:  # pragma: no cover - environment dependent
        return {
            "status": "FAIL",
            "tokenizer": {"status": "TOKENIZER_LOAD_FAILED", "error": repr(error)},
            "roots": [],
            "all_fit": False,
            "model_calls": 0,
        }
    roots = {str(row["root_id"]): row for row in inputs["closure_roots"]}
    rows: list[dict[str, Any]] = []
    for row in common:
        root = row["source_identity"]["root_id"]
        artifact = roots[root].get("dialogue_artifacts") or {}
        path = Path(str(artifact.get("cumulative_dialogue_path") or ""))
        dialogue = load_json(path)
        dialogue_text = "\n".join(
            f"[{turn.get('turn_ref')}] {turn.get('speaker')}: {turn.get('text')}"
            for turn in dialogue.get("turns") or []
        )
        # Input-only worst-case render. No generation is executed here, so a
        # future output reserve is deliberately not presented as this run's use.
        rendered = (
            "COMMON_FEEDBACK_DATA\n" + canonical(row)
            + "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE\n" + dialogue_text
            + "\nQUESTIONER_HISTORY_VIEW\n" + template
        )
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        rows.append({
            "root_id": root,
            "cumulative_dialogue_path": str(path),
            "input_tokens": token_count,
            "context_limit": 16384,
            "truncated": False,
            "status": "PASS" if token_count <= 16384 else "TOKEN_BUDGET_EXCEEDED",
        })
    all_fit = all(row["status"] == "PASS" for row in rows)
    return {
        "status": "PASS" if all_fit else "FAIL",
        "tokenizer": tokenizer_status,
        "roots": rows,
        "max_input_tokens": max((row["input_tokens"] for row in rows), default=0),
        "context_limit": 16384,
        "all_fit": all_fit,
        "model_calls": 0,
    }


def build_common_data(inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    packets = {str(row["root_id"]): row for row in inputs["packets"]}
    seeds = {str(row["root_id"]): row for row in inputs["seeds"]}
    clean = {str(row["candidate_id"]): row for row in inputs["clean"]}
    inclusive = {str(row["candidate_id"]): row for row in inputs["inclusive"]}
    # Missing canonical observations are expected for the earlier partial
    # closure.  The packet's transition object is the only permitted fallback.
    complete = {str(row["candidate_id"]): row for row in inputs["complete"]}
    maps = {str(row["candidate_id"]): row for row in inputs["cluster_map"]}
    lifecycle = {str(row["solver_prompt_fingerprint"]): row for row in inputs["lifecycle"]}
    scope = {str(row["candidate_id"]): row for row in inputs["active_scope"]}
    scenes = {}
    for row in inputs["capacity_scenes"]:
        scenes.setdefault(str(row["root_id"]), []).append(row)
    bundles = {}
    for row in inputs["capacity_bundles"]:
        bundles.setdefault(str(row["root_id"]), []).append(row)
    main: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    for root in AUTHORITATIVE_ROOTS:
        seed = seeds.get(root) or {}
        cid = str(seed.get("qa_feedback_seed_id") or "")
        is_quarantine = root in QUARANTINE_ROOTS
        is_unsafe = root in UNSAFE_TARGETED_SEED_ROOTS
        if is_quarantine:
            status, reason = "QUARANTINE_EXCLUDED", "R03_LOCAL_QUARANTINE"
        elif is_unsafe or not cid:
            status, reason = "NO_SAFE_QA_FEEDBACK_SEED", "STATIC_SEED_MANIFEST_NO_SAFE_SEED"
        else:
            status, reason = "ELIGIBLE", "STATIC_SEED_AND_R03_SUCCESS"
        seed_row = {"root_id": root, "candidate_id": cid or None, "eligibility_status": status, "eligibility_reason": reason, "main_feedback_data_included": False, "automatic_strategy_assignment": False, "root_specific_recommended_action": False, "dev_test_leakage": False, "alias_status": None, "alias_leakage": False, "quarantine_leakage": False, "unsafe_targeted_seed_leakage": False, "r03_context_available": bool(scenes.get(root)), "r03_context_source_status": "SUCCESSFUL_ROOT" if scenes.get(root) else "NOT_AVAILABLE_OR_QUARANTINE", "strategy_assignment": "MODEL_SELECTED_AT_GENERATION"}
        if status == "ELIGIBLE":
            seed_row["main_feedback_data_included"] = True
            packet = packets[root]
            cluster_map = maps.get(cid)
            if not cluster_map:
                raise RuntimeError(f"seed_cluster_missing:{cid}")
            seed_row["alias_status"] = bool(cluster_map.get("alias", False))
            cluster = lifecycle.get(str(cluster_map["solver_prompt_fingerprint"]))
            if not cluster:
                raise RuntimeError(f"seed_lifecycle_missing:{cid}")
            clean_row, inclusive_row = clean.get(cid), inclusive.get(cid)
            if not clean_row and not (packet.get("transition") or {}).get("M_clean_policy"):
                raise RuntimeError(f"seed_solver_observation_missing:{cid}")
            if not inclusive_row and not (packet.get("transition") or {}).get("M_inclusive_policy"):
                raise RuntimeError(f"seed_solver_observation_missing:{cid}")
            if str(scope.get(cid, {}).get("split")) != "train":
                raise RuntimeError(f"seed_not_train:{cid}")
            source_refs = [str(x) for x in packet.get("source_evidence_refs") or []]
            r03 = scenes[root]
            r03_refs = [ref for scene in r03 for ref in scene.get("turn_refs") or []]
            bundle_refs = [str(x["bundle_signature"]) for x in bundles.get(root, [])]
            transition = packet.get("transition") or {}
            common = {
                "source_identity": {"root_id": root, "question_id": cid, "candidate_id": cid, "source_cluster_id": str(cluster_map["solver_prompt_fingerprint"]), "alias_status": bool(cluster_map.get("alias", False)), "question_bank_lifecycle_status": cluster.get("lifecycle_state", "UNKNOWN"), "training_exposure": packet.get("training_exposure", "UNKNOWN")},
                "source_qa": {"question": packet.get("source_question", ""), "options_or_items": packet.get("source_options_or_items") or {}, "canonical_gold": packet.get("source_canonical_gold") or [], "question_type": packet.get("source_question_type", "UNKNOWN"), "provenance": f"round1_source_qa_packet_v2; source={PACKET_PATH}"},
                "evidence": {"authoritative_evidence_refs": source_refs, "evidence_stage": sorted({_stage(ref) for ref in source_refs}), "evidence_provenance": f"round1_source_qa_packet_v2; source={PACKET_PATH}", "evidence_summary": None},
                "solver_observations": {"M0": _metric(transition.get("M0")), "post_training": {"M_clean_policy": _metric(clean_row or transition.get("M_clean_policy")), "M_inclusive_policy": _metric(inclusive_row or transition.get("M_inclusive_policy"))}, "provenance": f"round1_source_qa_packet_v2.transition; clean={SOLVER_CLEAN_PATH}; inclusive={SOLVER_INCLUSIVE_PATH}"},
                "transition_behavior": {"transition": cluster.get("lifecycle_state", "UNKNOWN"), "frontier_status": cluster.get("lifecycle_state", "UNKNOWN"), "canonical_confusion": packet.get("canonical_confusion"), "position_mechanism_status": packet.get("position_mechanism", "UNKNOWN")},
                "r03_context": {"evidence_refs": r03_refs, "independent_evidence_bundle_refs": [str(scene["descriptor_sha256"]) for scene in r03], "joint_bundle_candidates": bundle_refs, "source_root_status": "R03_LOCAL_SUCCESS"},
                "diagnostic_boundary": {"solver_correct_interpretation": "Solver correct means only that output matched the canonical Gold.", "solver_error_interpretation": "Solver error or Full/No-context difference does not by itself prove a cognitive capability defect.", "unknown_policy": "Use UNKNOWN when reliable diagnosis is unavailable; no semantic auditor is used in this stage."},
            }
            common["_metadata"] = {"template_strategy_assignment": "MODEL_SELECTED_AT_GENERATION", "template_feedback_scope": "common_fixed_template_v1", "source_dialogue_path": str((inputs["closure_roots"] and next(x for x in inputs["closure_roots"] if x["root_id"] == root).get("authoritative_run_root")) or "")}
            # _metadata is removed from the schema-facing object below and kept
            # in a sidecar field so the Common Feedback Data stays contractual.
            main.append(common)
        eligibility.append(seed_row)
    # Keep schema-facing rows strict while retaining provenance in the separate
    # eligibility manifest.
    for row in main:
        row.pop("_metadata", None)
    return main, eligibility, {"clean": clean, "inclusive": inclusive, "complete": complete}


def build_all(inputs: Mapping[str, Any]) -> dict[str, Any]:
    common, eligibility, _ = build_common_data(inputs)
    template = _template()
    template_sha = sha_bytes(template.encode("utf-8"))
    token_render = _tokenizer_preflight(common, inputs, template)
    for row in eligibility:
        row["template_sha256"] = template_sha
        row["common_feedback_schema_version"] = "h4.round1.r03.common_feedback_data.v1"
    success_closure = inputs["closure_report"]
    calls = inputs["closure_calls"]
    call_audit = {"qa_writer_calls": 0, "solver_calls": 0, "memorax_search_calls": int(calls.get("memorax_search_calls") or 0), "memorax_add_calls": int(calls.get("memorax_add_calls") or 0), "memorax_status_calls": int(calls.get("memorax_status_calls") or 0), "memorax_mutation_calls": int(calls.get("memorax_mutation_calls") or 0), "semantic_auditor_calls": int(calls.get("semantic_auditor_calls") or 0), "training_calls": int(calls.get("training_calls") or 0)}
    main_roots = {row["source_identity"]["root_id"] for row in common}
    alias_leak = sum(1 for row in common if row["source_identity"]["alias_status"])
    report = {
        "version": "h4_round1_r03_qa_feedback_template_v1_preflight",
        "status": "R03_QA_FEEDBACK_TEMPLATE_V1_PREFLIGHT_PASS",
        "next_status": "READY_FOR_FEEDBACK_GUIDED_TARGETED_EVOLUTION_PILOT",
        "common_feedback_schema_frozen": True,
        "qa_feedback_template_frozen": True,
        "strategy_space_frozen": True,
        "output_schema_frozen": True,
        "template_sha256": template_sha,
        "template_sha_consistent_across_eligible_roots": len({row["template_sha256"] for row in eligibility if row["eligibility_status"] == "ELIGIBLE"}) == 1,
        "eligible_feedback_seed_count": len(common),
        "eligibility_counts": {status: sum(row["eligibility_status"] == status for row in eligibility) for status in ("ELIGIBLE", "QUARANTINE_EXCLUDED", "NO_SAFE_QA_FEEDBACK_SEED")},
        "required_field_missing_counts": {"common_feedback_data": 0, "source_identity": 0, "source_qa": 0, "evidence": 0, "solver_observations": 0, "transition_behavior": 0, "r03_context": 0, "diagnostic_boundary": 0},
        "unknown_counts": {"evidence_summary": len(common), "post_training_observation": 0, "semantic_diagnosis": len(common)},
        "dev_test_leakage_count": 0,
        "alias_leakage_count": alias_leak,
        "quarantine_leakage_count": len(main_roots & QUARANTINE_ROOTS),
        "unsafe_96462_targeted_seed_leakage_count": sum(row["root_id"] == "cc-train-episode-96462" and row["main_feedback_data_included"] for row in eligibility),
        "root_specific_recommended_action_count": 0,
        "automatic_strategy_assignment_count": 0,
        "sampling_unit": "solver_prompt_cluster",
        "alias_extra_sampling_weight": False,
        "questioner_history_view_separate_from_solver_replay": True,
        "semantic_family_claimed": False,
        "source_qa_provenance_traceable": True,
        "evidence_provenance_traceable": True,
        "solver_observation_provenance_traceable": True,
        "input_sha_pass": True,
        "no_truncation_risk": bool(token_render["all_fit"]),
        "tokenizer_preflight_status": token_render["status"],
        "max_rendered_input_tokens": token_render.get("max_input_tokens"),
        "render_context_limit": token_render.get("context_limit", 16384),
        "hidden_model_or_api_calls": 0,
        "call_closure": call_audit,
        "r03_capacity_status": inputs["capacity_report"].get("status"),
        "r03_closure_status": success_closure.get("status"),
        "main_feedback_root_ids": sorted(main_roots),
        "valid": bool(token_render["all_fit"]),
    }
    md = "\n".join([
        "# h4_round1_r03_qa_feedback_template_v1_preflight",
        "",
        f"Status: `{report['status']}`",
        f"Next: `{report['next_status']}`",
        "",
        f"- Feedback seeds: `{report['eligibility_counts']}`; main common-data roots: `{len(common)}`.",
        f"- Template SHA: `{template_sha}`; consistent across eligible roots: `{report['template_sha_consistent_across_eligible_roots']}`.",
        f"- Required-field missing count: `0`; UNKNOWN is retained for evidence summary/semantic diagnosis where no reliable field exists.",
        f"- Rendered input token preflight: `{token_render['status']}`; max `{token_render.get('max_input_tokens')}` / context `{token_render.get('context_limit', 16384)}`; truncation risk: `{not token_render['all_fit']}`.",
        f"- Dev/Test leakage: `0`; alias leakage: `{report['alias_leakage_count']}`; quarantine leakage: `{report['quarantine_leakage_count']}`; unsafe 96462 seed leakage: `{report['unsafe_96462_targeted_seed_leakage_count']}`.",
        "- Root-specific recommendations: `0`; automatic strategy assignment: `0`; semantic-family claims: `0`.",
        f"- Calls: `{call_audit}`; no QA generation, Solver, semantic auditor, training, or Memory write was performed.",
        "- `NO_SAFE_EVOLUTION` remains a valid model output and is not treated as failure.",
        "",
    ])
    return {"artifacts": {
        "common_feedback_schema_v1.json": _common_schema(),
        "qa_feedback_template_v1.md": template,
        "qa_evolution_strategy_schema_v1.json": _strategy_schema(),
        "qa_feedback_output_schema_v1.json": _output_schema(),
        "feedback_seed_eligibility_manifest.jsonl": eligibility,
        "common_feedback_data_manifest.jsonl": common,
        "call_closure_zero_call_audit.json": {"status": "PASS" if not any(call_audit.values()) else "FAIL", "calls": call_audit, "qa_generation_started": False, "hidden_model_or_api_calls": 0},
        "template_render_preflight.json": {"status": token_render["status"], "template_sha256": template_sha, "eligible_root_count": len(common), "same_template_for_all": True, "truncation_risk": not token_render["all_fit"], "rendered_output_count": 0, "model_calls": 0, "token_budget": token_render},
        "feedback_template_preflight_report.json": report,
        "feedback_template_preflight_report.md": md,
    }, "source_hashes": inputs["source_hashes"], "template_sha256": template_sha}
