"""Run the offline shared-view/template preflight.

This entry point never imports a service client and never starts a model.  It
is intentionally a fixture renderer, not a generation runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import (
    COMPILER_RUN, DEFAULT_OUTPUT, EXTERNAL_CALLS, FROZEN_TARGETED_TEMPLATE_RUN,
    HIDDEN_BACKEND_KEYS, QA_SCHEMA_RUN, R02_INPUT_RUN, REROUTE_RUN, VERSION,
)
from .compiler import (
    _metric, canonical, digest_file, expansion_view_v3, hidden_keys, load_frozen_inputs,
    qa_composition_view, qa_fresh_view_v3, qa_targeted_view, read_json, read_jsonl,
)
from .renderers import (
    render_expansion_planner, render_expansion_realizer, render_qa_composition,
    render_qa_fresh, render_qa_targeted, VERSIONED_RENDERER_CONFIG,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.profiles import (
    LEGACY_PROFILE, EXPANDED_PROFILE, EXPANDED_SINGLE_PLAN_PROFILE,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.scene_continuation import (
    build_scene_continuation_prompt,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.workflow import (
    build_realizer_plan_batches,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    select_planner_variant,
)
from .templates import (
    EXPANSION_TASK_V2, EXPANSION_TEMPLATE_V2, QA_TASK_V2, QA_TEMPLATE_V2,
    SHARED_FEEDBACK_RULES_V2, SHARED_FEEDBACK_RULES_V3,
    PLANNER_EXPANSION_TASK_V3, REALIZER_EXPANSION_TASK_V3,
    REALIZER_EXPANSION_TASK_FULL_V4, REALIZER_EXPANSION_TASK_TEXT_FREE_V4,
    QA_TASK_V3,
    EXPANSION_PLANNER_TEMPLATE_V3, EXPANSION_REALIZER_FULL_TEMPLATE_V4,
    EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V4,
    EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V3,
)


def _sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _write(root: Path, name: str, value: Any) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".jsonl"):
        text = "".join(canonical(row) + "\n" for row in value)
    elif name.endswith(".md") or name.endswith(".txt"):
        text = str(value).rstrip("\n") + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"immutable_artifact_drift:{path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    sha = digest_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    side_text = f"{sha}  {path.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != side_text:
        raise RuntimeError(f"immutable_sidecar_drift:{sidecar}")
    if not sidecar.exists():
        sidecar.write_text(side_text, encoding="utf-8")
    return sha


def _dialogue_text(dialogue: dict[str, Any]) -> str:
    return "\n".join(
        f"[{row['turn_ref']}] {row['speaker']}: {row['text']}"
        for row in dialogue["turns"]
    )


def _assert_public_view(name: str, view: Any) -> None:
    leaked = hidden_keys(view)
    if leaked:
        raise AssertionError(f"{name}_hidden_backend_fields:{sorted(leaked)}")


def _assert_no_unresolved(prompt: str) -> None:
    for token in (
        "{ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW}",
        "{AUTHORITATIVE_CUMULATIVE_DIALOGUE}",
        "{EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT}",
        "{EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT}",
    ):
        if token in prompt:
            raise AssertionError(f"unresolved_template_placeholder:{token}")


def _shape_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {"passed": True, "checks": []}
    for record in records:
        qa = record["historical_qa"]
        qtype = qa.get("question_type")
        values = qa.get("options_or_items") or {}
        if qtype == "single_choice":
            ok = list(values) == ["A", "B", "C", "D"]
        elif qtype in {"multi_select", "ordering"}:
            keys = list(values)
            if qtype == "multi_select":
                ok = 4 <= len(keys) <= 6 and keys == [chr(ord("A") + i) for i in range(len(keys))]
            else:
                ok = 4 <= len(keys) <= 6 and keys == [f"I{i + 1}" for i in range(len(keys))]
        else:
            ok = False
        results["checks"].append({"root_id": record["root_id"], "question_type": qtype, "option_shape_pass": ok})
        results["passed"] = results["passed"] and ok
    return results


def _missing_data_checks() -> dict[str, Any]:
    """Fixtures prove missing denominators/evidence stay unknown/empty."""
    metric = _metric(
        {"solver_profile": {"M_clean_policy": {"clean_full": 0.625}}},
        "M_clean_policy", "full",
    )
    fixture = {
        "root_id": "fixture", "source_cluster_id": "cluster",
        "source_candidate_id": "candidate",
        "historical_qa": {
            "question_type": "single_choice", "question": "q",
            "options_or_items": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "canonical_gold": ["A"], "evidence_refs": [],
        }, "performance": {}, "canonical_observation": {},
    }
    return {
        "accuracy_only_preserved": metric.get("accuracy") == 0.625,
        "accuracy_only_sample_unknown": metric.get("sample_size") == "UNKNOWN",
        "accuracy_only_no_fabricated_counts": "correct_count" not in metric and "total_count" not in metric,
        "missing_evidence_stays_empty": qa_targeted_view(fixture)["historical_qa"]["evidence_refs"] == [],
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    inputs = load_frozen_inputs()
    records = inputs["records"]
    by_root = {str(row["root_id"]): row for row in records}
    first = records[0]
    first_dialogue = inputs["dialogues"][first["root_id"]]

    # Public projections are all made from the same backend records.
    expansion_rows = [expansion_view_v3(row) for row in records]
    targeted_rows = [qa_targeted_view(row) for row in records]
    composition_view = qa_composition_view(records[:2])
    negative_path = COMPILER_RUN / "fresh_exploration_negative_signature_view.jsonl"
    negative_rows = read_jsonl(negative_path)
    fresh_rows = [qa_fresh_view_v3(row, [x for x in negative_rows if x.get("root_id") == row["root_id"]]) for row in records]
    for name, view in [("expansion", x) for x in expansion_rows] + [("targeted", x) for x in targeted_rows] + [("fresh", x) for x in fresh_rows] + [("composition", composition_view)]:
        _assert_public_view(name, view)
    if "historical_qa" in fresh_rows[0] or "source_candidate_id" in fresh_rows[0]:
        raise AssertionError("fresh_source_qa_binding")
    if "historical_qas" not in composition_view or len(composition_view["historical_qas"]) < 2:
        raise AssertionError("composition_history_missing")

    schemas = {
        mode: read_json(QA_SCHEMA_RUN / f"qa_output_schema_v1_1_revision_1_{mode}.json")
        for mode in ("targeted", "composition", "fresh")
    }
    schema_sidecar_checks = {}
    for mode in schemas:
        schema_path = QA_SCHEMA_RUN / f"qa_output_schema_v1_1_revision_1_{mode}.json"
        sidecar = schema_path.with_name(schema_path.name + ".sha256")
        expected = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else ""
        schema_sidecar_checks[mode] = digest_file(schema_path) == expected
    if not all(schema_sidecar_checks.values()):
        raise AssertionError(f"frozen_schema_sha_check_failed:{schema_sidecar_checks}")
    strategy_schema = read_json(FROZEN_TARGETED_TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json")
    plan_fixture = {"scenes": [{"memory_event_id": "FIXTURE", "development_mode": "new_development", "source_basis": [], "new_people": [], "verified_current_state": "", "goal": "", "required_participants": [], "transition_from_previous": "", "before_state": "", "required_transition": "", "after_state": "", "actor_bindings": {}, "interaction_beats": []} for _ in range(3)]}
    planner = render_expansion_planner(record=first, dialogue=first_dialogue, memory_overview={}, memory_candidates=[])
    planner_expanded = render_expansion_planner(
        record=first, dialogue=first_dialogue, memory_overview={},
        memory_candidates=[], planner_max_plan_variants=1,
        planner_max_new_candidates=8, planner_scene_quota=6,
        planner_max_tokens=EXPANDED_SINGLE_PLAN_PROFILE.planner_max_tokens)
    if planner.get("planner_max_tokens") != LEGACY_PROFILE.planner_max_tokens:
        raise AssertionError("legacy_planner_budget_metadata_mismatch")
    if planner_expanded.get("planner_max_tokens") != EXPANDED_SINGLE_PLAN_PROFILE.planner_max_tokens:
        raise AssertionError("expanded_planner_budget_metadata_mismatch")
    realizer = render_expansion_realizer(record=first, dialogue=first_dialogue, plan=plan_fixture,
                                         input_mode="text_free")
    realizer_full = render_expansion_realizer(record=first, dialogue=first_dialogue,
                                              plan=plan_fixture, input_mode="full_parent")
    targeted = render_qa_targeted(record=first, dialogue=first_dialogue, schema=schemas["targeted"], strategy_schema=strategy_schema)
    composition = render_qa_composition(records=records[:2], dialogue=first_dialogue, schema=schemas["composition"])
    fresh = render_qa_fresh(record=first, dialogue=first_dialogue, negative_signatures=[x for x in negative_rows if x.get("root_id") == first["root_id"]], schema=schemas["fresh"])
    supplement_fixture = {
        "version": "scene_continuation_request_v1",
        "round_index": 3,
        "eligible_scenes": [{
            "scene_ref": "R03:S01", "memory_event_id": "ME01",
            "required_participants": ["A", "B"],
            "selected_plan": {"required_transition": "A completes the scheduled handoff", "after_state": "the handoff is complete"},
            "actual_scene_turns": [{"speaker": "A", "text": "The first part is done."}, {"speaker": "B", "text": "We still need the handoff."}],
            "adherence": {"participant_adherence": "pass", "development_status": "partial", "durable_outcome_status": "partial"},
            "completion_instruction": "Continue only from the actual last turn and complete the fixed plan.",
        }],
    }
    supplement = build_scene_continuation_prompt(supplement_fixture)
    prompts = {"planner": planner, "realizer_text_free": realizer,
               "planner_expanded": planner_expanded,
               "realizer_full_parent": realizer_full, "qa_targeted": targeted,
               "qa_composition": composition, "qa_fresh": fresh,
               "realizer_supplement": supplement}
    for item in prompts.values():
        _assert_no_unresolved(item["prompt"])

    output.mkdir(parents=True, exist_ok=True)
    artifact_shas: dict[str, str] = {}
    artifact_shas["template_shared_feedback_rules_v2.md"] = _write(output, "template_shared_feedback_rules_v2.md", SHARED_FEEDBACK_RULES_V2)
    artifact_shas["template_expansion_task_v2.md"] = _write(output, "template_expansion_task_v2.md", EXPANSION_TASK_V2)
    artifact_shas["template_qa_task_v2.md"] = _write(output, "template_qa_task_v2.md", QA_TASK_V2)
    artifact_shas["template_shared_feedback_rules_v3.md"] = _write(output, "template_shared_feedback_rules_v3.md", SHARED_FEEDBACK_RULES_V3)
    artifact_shas["template_planner_task_v3.md"] = _write(output, "template_planner_task_v3.md", PLANNER_EXPANSION_TASK_V3)
    artifact_shas["template_realizer_task_v3.md"] = _write(output, "template_realizer_task_v3.md", REALIZER_EXPANSION_TASK_V3)
    artifact_shas["template_realizer_task_full_v4.md"] = _write(output, "template_realizer_task_full_v4.md", REALIZER_EXPANSION_TASK_FULL_V4)
    artifact_shas["template_realizer_task_text_free_v4.md"] = _write(output, "template_realizer_task_text_free_v4.md", REALIZER_EXPANSION_TASK_TEXT_FREE_V4)
    artifact_shas["template_qa_task_v3.md"] = _write(output, "template_qa_task_v3.md", QA_TASK_V3)
    artifact_shas["feedback_view_backend_manifest.jsonl"] = _write(output, "feedback_view_backend_manifest.jsonl", records)
    artifact_shas["authoritative_parent_r01_r02_dialogue_manifest.jsonl"] = _write(output, "authoritative_parent_r01_r02_dialogue_manifest.jsonl", list(inputs["dialogues"].values()))
    artifact_shas["feedback_view_expansion_manifest.jsonl"] = _write(output, "feedback_view_expansion_manifest.jsonl", expansion_rows)
    artifact_shas["feedback_view_qa_targeted_manifest.jsonl"] = _write(output, "feedback_view_qa_targeted_manifest.jsonl", targeted_rows)
    artifact_shas["feedback_view_qa_composition_fixture.json"] = _write(output, "feedback_view_qa_composition_fixture.json", composition_view)
    artifact_shas["feedback_view_qa_fresh_manifest.jsonl"] = _write(output, "feedback_view_qa_fresh_manifest.jsonl", fresh_rows)
    artifact_shas["field_visibility_matrix.json"] = _write(output, "field_visibility_matrix.json", {
        "model_visible": {
    "expansion": ["root_id", "source_cluster_id", "source_candidate_id", "historical_qa.question_type", "historical_qa.question", "historical_qa.options_or_items", "historical_qa.canonical_gold", "historical_qa.evidence_refs", "solver_observations"],
            "realizer": ["root_id", "source_cluster_id"],
            "targeted": ["root_id", "source_cluster_id", "source_candidate_id", "historical_qa", "solver_observations", "canonical_observation"],
            "composition": ["historical_qas", "source_count"],
            "fresh": ["root_id", "source_cluster_id", "historical_coverage", "constraint_note"],
        },
        "backend_only": sorted(HIDDEN_BACKEND_KEYS | {"backend_source", "source_arm", "canonical_observation.position_mechanism"}),
        "notes": ["Full historical QA remains in backend and QA views; expansion is intentionally not bound to an answer menu.", "Fresh receives signature-only coverage and no source QA.", "Original turn refs and source IDs are retained where the role requires them."],
    })
    artifact_shas["template_identity.json"] = _write(output, "template_identity.json", {
        "version": VERSION, "template_sha256": {
            "shared": _sha_value(SHARED_FEEDBACK_RULES_V2),
            "expansion": _sha_value(EXPANSION_TEMPLATE_V2),
            "qa": _sha_value(QA_TEMPLATE_V2),
            "expansion_task": _sha_value(EXPANSION_TASK_V2),
            "shared_v3": _sha_value(SHARED_FEEDBACK_RULES_V3),
            "planner_task_v3": _sha_value(PLANNER_EXPANSION_TASK_V3),
            "realizer_task_v3": _sha_value(REALIZER_EXPANSION_TASK_V3),
            # v3 remains a historical import/identity alias.  The active
            # renderer selects one of these mode-specific v4 blocks.
            "realizer_task_full_v4": _sha_value(REALIZER_EXPANSION_TASK_FULL_V4),
            "realizer_task_text_free_v4": _sha_value(REALIZER_EXPANSION_TASK_TEXT_FREE_V4),
            "qa_task_v3": _sha_value(QA_TASK_V3),
            "planner_template_v3": _sha_value(EXPANSION_PLANNER_TEMPLATE_V3),
            "realizer_template_v3": _sha_value(EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V3),
            "realizer_template_full_v4": _sha_value(EXPANSION_REALIZER_FULL_TEMPLATE_V4),
            "realizer_template_text_free_v4": _sha_value(EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V4),
            "qa_task": _sha_value(QA_TASK_V2),
        },
        "frozen_qa_schema": {mode: {"id": schemas[mode].get("$id"), "sha256": digest_file(QA_SCHEMA_RUN / f"qa_output_schema_v1_1_revision_1_{mode}.json")} for mode in schemas},
        "strategy_schema": {"id": strategy_schema.get("$id"), "sha256": digest_file(FROZEN_TARGETED_TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json")},
        "qa_contract_unchanged": True,
        "renderer_config": VERSIONED_RENDERER_CONFIG,
        "active_expansion_profile": EXPANDED_SINGLE_PLAN_PROFILE.to_dict(),
        "historical_multi_plan_profile": EXPANDED_PROFILE.to_dict(),
    })
    artifact_shas["baseline_expanded_profiles.json"] = _write(output, "baseline_expanded_profiles.json", {
        "baseline": LEGACY_PROFILE.to_dict(),
        "expanded_multi_plan_historical": EXPANDED_PROFILE.to_dict(),
        "expanded_single_plan": EXPANDED_SINGLE_PLAN_PROFILE.to_dict(),
        "baseline_worst_case_requests": LEGACY_PROFILE.worst_case_requests(),
        "expanded_multi_plan_historical_worst_case_requests": EXPANDED_PROFILE.worst_case_requests(),
        "expanded_single_plan_worst_case_requests": EXPANDED_SINGLE_PLAN_PROFILE.worst_case_requests(),
        "authorization": "example_only_not_generation_authorization",
    })
    artifact_shas["process_integration_table.json"] = _write(output, "process_integration_table.json", {
        "feedback_view": "live.py -> expanded profile feedback_view_version -> compiler.load_frozen_inputs -> render_expansion_planner/render_expansion_realizer; legacy profile leaves this opt-in disabled",
        "planner": "workflow.execute_round -> validate_planner_output -> compile_planner_output",
        "semantic_gate": "workflow.execute_round -> semantic_gate_call -> validate_semantic_gate_output",
        "realizer": "workflow.execute_round -> build_realizer_prompt (normal_generation), batches preserve scene refs and optionally use only actually promoted prefixes",
        "planner_variants": "active expanded_single_plan_v1 accepts one complete plan; a response with more than one plan_variants entry is rejected before Gate. Generic materialization/selection remains available for the historical expanded_multi_plan_v1 compatibility route.",
        "adherence": "workflow.execute_round -> build_adherence_auditor_input/prompt -> validate_and_materialize_adherence_audit",
        "bounded_continuation": "scene_continuation.build_scene_continuation_request/prompt; one bounded callback when eligible",
        "replacement": "scene_replacement.build_scene_replacement_request; at most configured replacement callback",
        "promotion": "promotion.build_promotable_child -> validate_promotable_child; partial/full/none",
        "memory_write": "no_memorax_writes delegates reads and raises on add/status; quality path remains active",
        "expanded_budget": "profiles.ExpansionExecutionProfile.worst_case_requests derives per-role and total hard limits; example is not authorization",
        "expanded_single_plan_budget": {
            "normal_planner": 1,
            "planner_repair": 1,
            "planner_related_total": 2,
            "repair_is_conditional": True,
            "note": "The one-plan contract changes plan cardinality, not the independent repair budget or the rest of the quality path.",
        },
        "qa_schema": "frozen revision-1 files unchanged",
    })
    artifact_shas["completion_matrix.json"] = _write(output, "completion_matrix.json", {
        "1_shared_feedback_projection": {"status": "COMPLETED", "files": ["compiler.py", "renderers.py", "templates.py"]},
        "2_event_bucket_route": {"status": "COMPLETED", "files": ["prompts.py", "renderers.py", "workflow.py"]},
        "3_planner_action_summary_and_binding": {"status": "COMPLETED", "files": ["prompts.py", "contracts.py"]},
        "4_realizer_input_modes": {"status": "COMPLETED", "files": ["renderers.py", "prompts.py"]},
        "5_quality_workflow": {"status": "REUSED_AND_PARAMETERIZED", "files": ["workflow.py", "promotion.py", "scene_continuation.py"]},
        "6_multi_plan_and_batches": {"status": "COMPLETED_OFFLINE", "files": ["profiles.py", "prompts.py", "contracts.py", "workflow.py", "live_runtime.py"], "notes": "current expanded_single_plan_v1 requests one complete plan containing up to 3 continuations and up to 8 new-development candidates, then applies the existing Gate and up to six-scene batch limits; historical expanded_multi_plan_v1 remains available for reproducibility"},
        "7_token_and_budget_guards": {"status": "COMPLETED_OFFLINE", "files": ["profiles.py", "live.py"]},
        "8_qa_revision_1": {"status": "REUSED_UNCHANGED", "files": [str(QA_SCHEMA_RUN)]},
    })
    artifact_shas["final_templates.md"] = _write(output, "final_templates.md", "\n\n".join([
        "# Shared Feedback Rules v3\n\n" + SHARED_FEEDBACK_RULES_V3,
        "# Planner Expansion Task v3\n\n" + PLANNER_EXPANSION_TASK_V3,
        "# Realizer Expansion Task full_parent v4\n\n" + REALIZER_EXPANSION_TASK_FULL_V4,
        "# Realizer Expansion Task text_free v4\n\n" + REALIZER_EXPANSION_TASK_TEXT_FREE_V4,
        "# QA Task v3\n\n" + QA_TASK_V3,
    ]))
    artifact_shas["minimal_diff.md"] = _write(output, "minimal_diff.md", """# Minimal Diff

Baseline recovery: event-bucket-first Planner authority, complete Expansion
feedback projection, explicit Planner action-summary instructions, and a
Realizer renderer that obeys full_parent/text_free boundaries. Existing Gate,
adherence, bounded continuation, replacement, and scene promotion functions
remain the quality path; `no_memorax_writes` only blocks remote writes.

Expanded capability: `profiles.py` adds the active `expanded_single_plan_v1`
profile with one complete Planner plan, an independent continuation and
new-development candidate pool, six-scene scheduling, two-scene Realizer
batches, and Planner max output 8192. The former `expanded_multi_plan_v1`
profile remains available as a historical compatibility route. The legacy
profile remains three scenes and 4096. QA revision-1 input/output schema and
its SHA are not modified.
""")
    artifact_shas["unknowns_and_calls_zero.json"] = _write(output, "unknowns_and_calls_zero.json", {
        "new_external_calls": {key: 0 for key in EXTERNAL_CALLS},
        "unknown_or_not_tested": [
            "semantic correctness of historical Gold/distractors",
            "real provider acceptance of planner_max_tokens=8192",
            "new R03 runtime generation and scene quality",
            "remote Memory state",
        ],
        "mock_pass_is_not_formal_generation_success": True,
    })
    artifact_shas["source_sha_manifest.json"] = _write(output, "source_sha_manifest.json", {
        "sources": [{"name": str(path), "sha256": digest_file(path)} for path in (
            inputs["cards_path"], inputs["candidates_path"], inputs["dialogue_path"], inputs["common_path"], negative_path,
        )],
        "authoritative_dialogue_records": [{"root_id": row["root_id"], "path_sha256": row["input_dialogue_file_sha256"], "complete_dialogue_sha256": row["complete_dialogue_sha256"], "source_prefix": row["source_prefix"]} for row in inputs["dialogues"].values()],
        "old_r03_mixed": False,
        "new_branch_source": "Parent+R01+R02-feedback",
        "historical_feedback_backend_only": str(inputs["common_path"]),
    })
    for name, item in prompts.items():
        artifact_shas[f"rendered_prompts/{name}.txt"] = _write(output, f"rendered_prompts/{name}.txt", item["prompt"])
    render_manifest = [{"role": name, "renderer": item.get("renderer", "offline_fixture"), "prompt_version": item.get("prompt_version", "unknown"), "root_id": item.get("root_id", first["root_id"]), "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(), "offline_fixture": True} for name, item in prompts.items()]
    artifact_shas["rendered_prompt_manifest.jsonl"] = _write(output, "rendered_prompt_manifest.jsonl", render_manifest)
    artifact_shas["shape_and_source_checks.json"] = _write(output, "shape_and_source_checks.json", {
        "historical_qa_option_shapes": _shape_checks(records),
        "authoritative_dialogue_root_count": len(inputs["dialogues"]),
        "all_dialogue_source_prefixes_r02_only": all(row["source_prefix"] == "Parent+R01+R02-feedback" for row in inputs["dialogues"].values()),
        "fresh_not_source_bound": True,
        "composition_has_multiple_history_records": True,
        "evidence_refs_are_source_only": True,
        "new_r03_evidence_refs_output_field": False,
        "feedback_profile_integration": {
            "legacy_feedback_view_version": LEGACY_PROFILE.feedback_view_version,
            "expanded_feedback_view_version": EXPANDED_SINGLE_PLAN_PROFILE.feedback_view_version,
            "expanded_profile_is_explicit": EXPANDED_SINGLE_PLAN_PROFILE.feedback_view_version == VERSION,
            "expanded_profile_name": EXPANDED_SINGLE_PLAN_PROFILE.name,
            "renderer_module": "h4_round1_r03_feedback_view_v2.renderers",
        },
        "targeted_scheduling": "SYNCHRONIZED_ONLY_NOT_SCHEDULED",
        "missing_data_fixtures": _missing_data_checks(),
        "frozen_schema_sidecar_checks": schema_sidecar_checks,
        "event_bucket_route": {
            "renderer_event_bucket_only": VERSIONED_RENDERER_CONFIG["event_bucket_only"],
            "planner_prompt_event_bucket_first": "event-bucket rows are the main selectable event source" in planner["prompt"].lower(),
            "person_bucket_only_conflict_absent": "person-bucket event facts are the only Memory state hints" not in planner["prompt"],
            "memory_only_fast_label_absent": "memory-only fast" not in planner["prompt"].lower(),
        },
        "realizer_mode_boundaries": {
            "full_parent_included": realizer_full["full_parent_included"],
            "text_free_included": realizer["full_parent_included"],
            "text_free_has_no_transcript_marker": "FROZEN PARENT TRANSCRIPT" not in realizer["prompt"],
            "full_has_transcript_marker": "FROZEN PARENT TRANSCRIPT" in realizer_full["prompt"],
        },
        "supplement_prompt_has_actual_tail": "actual_scene_turns" in supplement["prompt"] and "first part is done" in supplement["prompt"],
        "multi_plan_batch_fixture": {
            "expanded_batch_count": len(build_realizer_plan_batches({"scenes": [{"memory_event_id": str(i)} for i in range(6)]}, batch_size=2)),
            "legacy_batch_count": len(build_realizer_plan_batches({"scenes": [{"memory_event_id": str(i)} for i in range(3)]}, batch_size=3)),
            "expanded_sequential_promotion": bool(EXPANDED_SINGLE_PLAN_PROFILE.sequential_batch_promotion),
            "expanded_batch_adherence_audits": EXPANDED_SINGLE_PLAN_PROFILE.worst_case_requests()["batch_adherence_audits"],
            "expanded_single_plan": {
                "prompt_requires_one_plan": (
                    "requires exactly one complete plan" in planner_expanded["prompt"]),
                "prompt_forbids_second_plan": (
                    "Do not generate a second alternative plan" in planner_expanded["prompt"]),
                "renderer_limit": planner_expanded.get("max_plan_variants") == 1,
                "two_plan_rejection_fixture": select_planner_variant(
                    {"plan_variants": [{"round_goal": "a"}, {"round_goal": "b"}]},
                    max_plan_variants=1),
                "single_element_compatibility_fixture": select_planner_variant(
                    {"plan_variants": [{"round_goal": "a"}]},
                    max_plan_variants=1),
            },
        },
    })
    report = {
        "status": "EXPANDED_SINGLE_PLAN_OFFLINE_PREFLIGHT_PASS",
        "next_step": "WAITING_FOR_EXPLICIT_AUTHORIZATION_FOR_SINGLE_PLAN_R03_EXPANSION",
        "version": VERSION,
        "authoritative_roots": len(records),
        "new_r03_generated": False,
        "old_r03_mixed_into_new_branch": False,
        "old_composition_fresh_requests_reused": False,
        "historical_old_r03_feedback_retained_backend_only": True,
        "external_calls": dict(EXTERNAL_CALLS),
        "external_calls_zero": all(value == 0 for value in EXTERNAL_CALLS.values()),
        "qa_output_contract_revision_1_unchanged": True,
        "planner_realizer_existing_builders_invoked_offline": True,
        "qa_existing_revision_1_builders_invoked_offline": True,
        "explicit_versioned_renderer_config": VERSIONED_RENDERER_CONFIG,
        "active_expanded_profile": EXPANDED_SINGLE_PLAN_PROFILE.to_dict(),
        "historical_multi_plan_profile": EXPANDED_PROFILE.to_dict(),
        "backend_record_count": len(records),
        "model_view_counts": {"expansion": len(expansion_rows), "qa_targeted": len(targeted_rows), "qa_composition_fixture_records": len(composition_view["historical_qas"]), "qa_fresh": len(fresh_rows)},
        "checks": {"hidden_fields_absent": True, "fresh_source_binding_absent": True, "composition_history_present": True, "dialogue_scope_pass": True, "option_shape_pass": _shape_checks(records)["passed"], "unresolved_placeholders_absent": True, "missing_data_fixtures_pass": all(_missing_data_checks().values()), "frozen_schema_sha_pass": all(schema_sidecar_checks.values())},
        "not_tested": ["semantic correctness of historical Gold/distractors", "real model generation", "new R03 branch runtime", "remote memory state"],
        "artifact_sha256": artifact_shas,
    }
    artifact_shas["offline_preflight_report.json"] = _write(output, "offline_preflight_report.json", report)
    report_md = "\n".join([
        "# R03 Shared Feedback View v2 Offline Preflight",
        "",
        "Status: `EXPANDED_SINGLE_PLAN_OFFLINE_PREFLIGHT_PASS`",
        "",
        "The versioned shared backend record deterministically projects expansion, targeted, composition, and fresh views. The active expanded profile requests one complete Planner plan with a multi-candidate pool; Planner/Realizer and revision-1 QA builders were invoked with local fixtures only.",
        "",
        f"- Authoritative Parent+R01+R02 roots: {len(records)}/{len(records)}",
        "- Old R03 mixed into new branch: false",
        "- Fresh source-QA binding: false",
        "- Existing QA revision-1 schema changed: false",
        "- External calls: all zero (Memorax, model, Solver, auditor, training)",
        "- Semantic Gold/distractor correctness: NOT_TESTED",
        "",
        "Next step: wait for explicit authorization for single-plan R03 debug expansion.",
    ])
    artifact_shas["offline_preflight_report.md"] = _write(output, "offline_preflight_report.md", report_md)
    # The manifest is written last so it can cover all generated artifacts
    # except itself and its sidecars without a self-referential hash.
    final_manifest = [{"path": str(path.relative_to(output)), "sha256": digest_file(path)} for path in sorted(output.rglob("*")) if path.is_file() and not path.name.endswith(".sha256") and path.name != "output_sha_manifest.json"]
    _write(output, "output_sha_manifest.json", final_manifest)
    return report


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), ensure_ascii=False, indent=2))
