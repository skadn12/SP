from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_namespace_repair_v11 import (
    apply_planner_namespace_repairs_v11,
    compile_planner_namespace_repair_spec_v11,
    validate_planner_namespace_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.production_batch_v11 import (
    expansion_qa_admission,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_prompt,
    compile_round_validator_model_output,
    materialize_round_validator_optional_model_fields,
    normalize_exact_duplicate_scene_reviews,
    normalize_round_validator_output,
    run_round_validator_with_retry,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.single_item_qa_canary import (
    deterministic_answer_placement,
    freeze_core_qa,
    materialize_single_item_mcq,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.mcq_option_diagnostics import (
    diagnose_option_shape,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    _shape_retry_suffix,
    build_shape_balanced_distractor_prompt,
    evidence_alignment_audit,
    validate_distractors_with_shape_gate,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


BATCH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_4"
)
BATCH_SHA = "1b1498933530267b80a2191856f1f9357b0a0be2cd99e83bb7498a9226d5aae8"
VALIDATOR_RUN = BATCH / "roots/cc-train-episode-216742/01_expansion"
PLANNER_RUN = BATCH / "roots/cc-train-episode-41625/01_expansion"
PILOT5 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_5"
)
PILOT5_SHA = "ffee918c1a006abfa5d857327c898dcbaaf9c750632469c048431f38123c0de6"
PILOT6 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_6"
)
PILOT6_SHA = "69eacae6b8d2c6814d2c1051f3bd92472b69999f40b1c77bcdff8fd30ac7ef37"
PILOT7 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_7"
)
PILOT7_SHA = "1f07bdcbbc4ed0084a4e6bfb705f545e4e110e36f9e24d65910b5e20236b42fb"
PILOT8 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_8"
)
PILOT8_SHA = "1f8ecee44a7bb01a902f5814069f06a8c8c4fd78a3da74275247a541a8cc89fb"
PILOT9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_9"
)
PILOT9_SHA = "ae962a56e10d37f0015150ad292d4e2e776994054f28dc5c1428ed408e9e5d27"
PILOT10 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_10"
)
PILOT10_SHA = "c1bdc4fc9ac5b3cb170fc5400af4767be29bfcff573af35467d9cf5bfc2e5f20"
PILOT11 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_11"
)
PILOT11_SHA = "931245ca56736f004c868ba4b2c73d11fc2480ea0a204a5c37c22fccbc5b3b5e"
PILOT12 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_12"
)
PILOT12_SHA = "bfe3b2f81c5ba510e4499c0c97178d5b1c1523a37872d0fc1ab65f8f79a3533c"
PILOT13 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_13"
)
PILOT13_SHA = "550a26d8429a3a7ace4b4599b79d9dfe01c24d3527a3727b5f2934944420aee0"
PILOT14 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_14"
)
PILOT14_SHA = "6dc03f350d587ec81328ba22b92c6756124829594d6a3d551a36c1b38c04d421"
PILOT15 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_15"
)
PILOT15_SHA = "899a175d2a07672da461ec24128c6878975b2cb341bb0f05273cd73e4b1bb90b"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sparse_repair_fixture(plan):
    rows = []
    for target in plan["scene_reference_targets"]:
        row = {"repair_kind": "scene_reference",
               "scene_alias": target["scene_alias"]}
        if target["repair_current_scene_supporting_turn_refs"]:
            row["current_scene_supporting_turn_refs"] = target[
                "allowed_current_scene_turn_refs"][:2]
        dependency = target["original_dependency_context_turn_refs"]
        if target["repair_dependency_context_turn_refs"]:
            dependency = target["allowed_dependency_turn_refs"][:2]
            row["dependency_context_turn_refs"] = dependency
        if target["repair_uses_scene_refs"]:
            row["uses_scene_refs"] = sorted({
                ref.rsplit(":T", 1)[0]
                for ref in dependency
            })
        if target.get("repair_historical_supporting_turn_refs"):
            row["historical_supporting_turn_refs"] = target[
                "allowed_historical_turn_refs"][:2]
        rows.append(row)
    for target in plan["targets"]:
        row = {"repair_kind": "action_realization",
               "action_ref": target["action_ref"]}
        evidence = target["original_evidence_turn_refs"]
        if target["evidence_turn_refs_requires_repair"]:
            evidence = target["allowed_evidence_turn_refs"][:2]
            row["evidence_turn_refs"] = evidence
        if target.get("progression_evidence_turn_refs_requires_repair"):
            row["progression_evidence_turn_refs"] = evidence[:1]
        rows.append(row)
    return {"repairs": rows}


class DialogueFirstBatchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator_input = load(VALIDATOR_RUN / "round_validator_input.json")
        cls.validator_raw = load(
            VALIDATOR_RUN / "attempts/round_validator_attempt_01.json"
        )["parsed_response"]
        cls.validator_scenes = cls.validator_input["scenes"]
        cls.review_world_view = load(VALIDATOR_RUN / "review_world_view.json")
        cls.local_gates = load(VALIDATOR_RUN / "local_scene_gate_results.json")

        cls.outline = load(
            PLANNER_RUN / "attempts/planner_v11_attempt_01.json"
        )["parsed_response"]
        cls.planner_validation = load(
            PLANNER_RUN / "planner/attempt_01_validation.json"
        )
        cls.planning = load(PLANNER_RUN / "planning_state_view.json")
        cls.availability = load(PLANNER_RUN / "action_availability_view.json")
        cls.anchors = load(PLANNER_RUN / "source_anchor_registry.json")
        cls.placements = load(PLANNER_RUN / "placement_choice_registry.json")
        cls.namespace_spec = compile_planner_namespace_repair_spec_v11(
            outline=cls.outline,
            errors=cls.planner_validation["errors"],
            validation=cls.planner_validation["validation"],
            action_contracts=action_contract_registry(),
            action_availability_view=cls.availability,
            authoritative_role_table=cls.planning["entities"],
        )

    def test_historical_batch_is_unchanged(self):
        self.assertEqual(BATCH_SHA, hash_tree(BATCH)["tree_sha256"])

    def test_dialogue_only_policy_keeps_reviews_and_drops_memory_candidates(self):
        calls = []

        def backend(_prompt, attempt):
            calls.append(attempt)
            return deepcopy(self.validator_raw)

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=self.validator_input,
            realized_scenes=self.validator_scenes,
            review_world_view=self.review_world_view,
            local_gate_results=self.local_gates,
            durable_update_policy="dialogue_only",
        )
        self.assertTrue(result["completed"])
        self.assertEqual([1], calls)
        self.assertEqual([], result["valid_output"]["durable_updates"])
        self.assertEqual(7, len(result["valid_output"]["scene_reviews"]))
        self.assertTrue(all(
            row["decision"] == "accept"
            for row in result["valid_output"]["scene_reviews"]
        ))
        audit = result["durable_update_projection_audit"]
        self.assertGreater(audit["candidate_count"], 0)
        self.assertFalse(audit["semantic_status_upgrade_performed"])
        self.assertFalse(audit["authoritative_memory_write_allowed"])

    def test_dialogue_only_policy_does_not_hide_scene_contract_errors(self):
        raw = deepcopy(self.validator_raw)
        raw["scene_reviews"] = raw["scene_reviews"][:-1]

        def backend(_prompt, _attempt):
            return deepcopy(raw)

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=self.validator_input,
            realized_scenes=self.validator_scenes,
            review_world_view=self.review_world_view,
            local_gate_results=self.local_gates,
            durable_update_policy="dialogue_only",
        )
        self.assertFalse(result["completed"])
        self.assertTrue(any(
            "scene_review_coverage" in error
            for attempt in result["attempts"]
            for error in attempt["validation"]["errors"]
        ))

    def _connection_choice_response(self):
        action_spec = self.namespace_spec["action_repair_spec"]
        choice_id = next(
            row["connection_choice_id"]
            for row in action_spec["authoritative_connection_choices"]
            if row["action_type"] == "introduce_connected_event"
        )
        original = self.outline["scenes"][3]["primary_action"]["arguments"]
        arguments = {
            key: deepcopy(value) for key, value in original.items()
            if key not in {"event_alias", "connected_existing"}
        }
        arguments["connection_choice_id"] = choice_id
        return {
            "action_repairs": [{
                "action_ref": "S04.primary_action",
                "replacement": {
                    "action_mode": "new",
                    "action_type": "introduce_connected_event",
                    "arguments": arguments,
                },
            }],
            "link_type_repairs": [{
                "scene_ref": "S07",
                "replacement_choice_id": self.namespace_spec[
                    "link_type_choices"][0]["choice_id"],
            }],
        }

    def test_program_connection_choice_repairs_real_pilot4_outline(self):
        response = self._connection_choice_response()
        patch_validation = validate_planner_namespace_repair_response_v11(
            response, spec=self.namespace_spec,
        )
        self.assertTrue(patch_validation["valid"], patch_validation["errors"])
        applied = apply_planner_namespace_repairs_v11(
            frozen_outline=self.outline,
            response=response,
            spec=self.namespace_spec,
        )
        self.assertTrue(applied["valid"], applied["errors"])
        repaired = applied["reconstructed_outline"]
        event = repaired["scenes"][3]["primary_action"]["arguments"]
        self.assertEqual("event_s04_primary", event["event_alias"])
        self.assertEqual("person", event["connected_existing"]["ref_type"])
        outline_validation = validate_planner_outline_v11(
            repaired,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            authoritative_entity_refs={
                row["entity_ref"] for row in self.planning["entities"]
            },
            authoritative_entities=self.planning["entities"],
            action_availability_view=self.availability,
        )
        self.assertTrue(outline_validation["valid"], outline_validation["errors"])
        compiled = compile_planner_outline_v11(
            repaired,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            planning_state=self.planning,
            action_availability_view=self.availability,
        )
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            planning_state=self.planning,
            complete_dialogue=load(PLANNER_RUN / "complete_dialogue_view.json"),
        )
        self.assertTrue(structural["valid"], structural["errors"])

    def test_unknown_program_connection_choice_is_rejected(self):
        response = self._connection_choice_response()
        response["action_repairs"][0]["replacement"]["arguments"][
            "connection_choice_id"] = "CCE9999"
        validation = validate_planner_namespace_repair_response_v11(
            response, spec=self.namespace_spec,
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "program_connection_choice_unknown" in error
            for error in validation["errors"]
        ))

    def test_memory_growth_is_not_a_qa_admission_gate(self):
        admission = expansion_qa_admission({
            "dialogue_ready": True,
            "qa_source_ready": True,
            "memory_growth_ready": False,
            "fresh_e2e_ready": False,
        })
        self.assertTrue(admission["valid"])
        self.assertFalse(admission["memory_growth_required"])
        self.assertFalse(admission["memory_growth_ready"])

    def test_pilot5_is_unchanged(self):
        self.assertEqual(PILOT5_SHA, hash_tree(PILOT5)["tree_sha256"])

    def test_pilot6_is_unchanged(self):
        self.assertEqual(PILOT6_SHA, hash_tree(PILOT6)["tree_sha256"])

    def test_pilot7_is_unchanged(self):
        self.assertEqual(PILOT7_SHA, hash_tree(PILOT7)["tree_sha256"])

    def test_pilot8_is_unchanged(self):
        self.assertEqual(PILOT8_SHA, hash_tree(PILOT8)["tree_sha256"])

    def test_pilot9_is_unchanged(self):
        self.assertEqual(PILOT9_SHA, hash_tree(PILOT9)["tree_sha256"])

    def test_pilot10_is_unchanged(self):
        self.assertEqual(PILOT10_SHA, hash_tree(PILOT10)["tree_sha256"])

    def test_pilot11_is_unchanged(self):
        self.assertEqual(PILOT11_SHA, hash_tree(PILOT11)["tree_sha256"])

    def test_pilot12_is_unchanged(self):
        self.assertEqual(PILOT12_SHA, hash_tree(PILOT12)["tree_sha256"])

    def test_pilot13_is_unchanged(self):
        self.assertEqual(PILOT13_SHA, hash_tree(PILOT13)["tree_sha256"])

    def test_pilot14_is_unchanged(self):
        self.assertEqual(PILOT14_SHA, hash_tree(PILOT14)["tree_sha256"])

    def test_pilot15_is_unchanged(self):
        self.assertEqual(PILOT15_SHA, hash_tree(PILOT15)["tree_sha256"])

    def test_shape_retry_does_not_echo_failed_distractors(self):
        marker = "DO_NOT_COPY_THIS_FAILED_OPTION"
        suffix = _shape_retry_suffix(
            {"distractors": [marker, "bad two", "bad three"]},
            {"reasons": ["gold_plus_three_small_mutations"]},
        )
        self.assertNotIn(marker, suffix)
        self.assertIn("counterfactual components", suffix)
        self.assertIn("Do not return complete distractors", suffix)
        self.assertTrue(suffix.endswith(
            'OUTPUT ONLY {"counterfactual_parent_component":"...",'
            '"counterfactual_child_component":"..."}'))

    def test_batch_distractor_prompt_requires_independent_counterfactuals(self):
        root = PILOT14 / "roots/cc-train-episode-41625"
        source = load(root / "02_questioner/frozen/qa_source_view.json")
        core = load(root / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(root / "02_questioner/frozen/span_catalog.json")
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=source["combined_dialogue"],
            span_catalog=catalog,
            base_qa=core,
        )
        self.assertIn("FACTORIAL COUNTERFACTUAL COMPONENT CONTRACT", prompt)
        self.assertIn("false parent component", prompt)
        self.assertIn("false-parent/false-child", prompt)
        self.assertNotIn("three incorrect cells of this 2x2 contrast", prompt)

    def test_pilot17_independent_counterfactual_fixture_passes_shape_gate(self):
        root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "slim_continuous_expansion_v11_end_to_end_batch_pilot_17/roots/"
            "cc-train-episode-216742")
        core = load(root / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(root / "02_questioner/frozen/span_catalog.json")
        result = validate_distractors_with_shape_gate(
            {"distractors": [
                "Initially: They said the cracked window was only cosmetic and could "
                "wait until after their writing deadline. Later: They decided to sell "
                "the car, use public transit, and redirect the repair money toward rent.",
                "Initially: They blamed the repair shop for the delay and insisted "
                "maintenance was someone else's responsibility. Later: They asked a "
                "family member to manage every appointment while they focused entirely "
                "on finishing a manuscript.",
                "Initially: They admitted the damage was dangerous but planned to keep "
                "driving until the weather improved. Later: They booked a mechanic for "
                "the same afternoon and cancelled their weekend plans so the window "
                "could be replaced immediately.",
            ]},
            core_qa=core,
            span_catalog=catalog,
            placement=deterministic_answer_placement(
                "cc-train-episode-216742-round-1", 0),
        )
        self.assertEqual("distractor_accept", result["decision"])
        self.assertFalse(result["option_shape"]["single_mutation_star"])
        self.assertFalse(result["option_shape"]["correct_unique_medoid_leakage"])

    def test_factorial_distractors_pass_materializer_shape_gate(self):
        root = PILOT14 / "roots/cc-train-episode-41625"
        core = load(root / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(root / "02_questioner/frozen/span_catalog.json")
        initial_gold = (
            "Initially: I was thinking a music festival. I have a lot of musician "
            "friends who would love to perform, and I think it could be really fun "
            "and successful.")
        initial_false = (
            "Initially: I was planning a community theater production. I know "
            "several actor friends who would enjoy rehearsing, and I thought it "
            "could be engaging and popular.")
        later_gold = (
            "Later: That's fantastic, Classmates B. I'm confident we've covered "
            "all the bases. Musician C, let's make sure we're all aligned on the "
            "schedule and venue details.")
        later_false = (
            "Later: That's disappointing, Classmates B. I'm worried we've uncovered "
            "several major problems. Organizer D, let's make sure we revise the "
            "budget and performer list.")
        self.assertEqual(initial_gold + " " + later_gold,
                         core["correct_answer_text"])
        result = validate_distractors_with_shape_gate(
            {"distractors": [
                initial_false + " " + later_false,
                initial_false + " " + later_gold,
                initial_gold + " " + later_false,
            ]},
            core_qa=core,
            span_catalog=catalog,
            placement=deterministic_answer_placement(
                "cc-train-episode-41625-round-1", 0),
        )
        self.assertEqual("distractor_accept", result["decision"])
        self.assertFalse(result["option_shape"]["answer_shape_leakage_risk"])

    def test_pilot13_leaky_distractors_trigger_shape_retry(self):
        root = PILOT13 / "roots/cc-train-episode-41625"
        record = next((root / "04_mcq_materializer/raw/distractor_writer").glob(
            "*.json"))
        raw = json.loads(load(record)["raw_response"])
        core = load(root / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(root / "02_questioner/frozen/span_catalog.json")
        result = validate_distractors_with_shape_gate(
            raw,
            core_qa=core,
            span_catalog=catalog,
            placement=deterministic_answer_placement(
                "cc-train-episode-41625-round-1", 0),
        )
        self.assertEqual("distractor_shape_invalid", result["decision"])
        self.assertTrue(result["option_shape"]["answer_shape_leakage_risk"])
        self.assertIn("correct_answer_unique_option_medoid", result["reasons"])

    def test_pilot12_legacy_two_field_question_is_rejected_by_owned_evidence_schema(self):
        run = PILOT12 / "roots/cc-train-episode-41625/02_questioner"
        record = next((run / "raw/question_answer_writer").glob("*.json"))
        stored = load(record)
        raw = json.loads(stored["raw_response"])
        view = load(run / "frozen/qa_source_view.json")
        catalog = load(run / "frozen/span_catalog.json")
        historical = validate_cross_boundary_core_object(
            raw,
            span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
        )
        self.assertEqual("cross_boundary_qa_schema_invalid",
                         historical["decision"])
        self.assertIn("cross_boundary_qa_model_top_level_schema",
                      historical["reasons"])
        validation = validate_cross_boundary_core_object(
            {
                "question": (
                    "How did Classmates A's earlier concern change in the "
                    "later planning discussion?"),
                "correct_answer_text": (
                    "Classmates A moved from concern to discussing a concrete "
                    "later plan."),
                "answer_support_span_ids": raw["answer_support_span_ids"],
            },
            span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
        )
        self.assertEqual("cross_boundary_qa_accept", validation["decision"])
        frozen = freeze_core_qa(
            validation["core_qa"],
            lineage_id="cc-train-episode-41625-round-1",
            sample_index=0,
            span_catalog=catalog,
        )
        self.assertEqual(validation["correct_answer_text"],
                         frozen["correct_answer_text"])
        self.assertEqual(2, len(frozen["answer_support_span_ids"]))

    def test_pilot12_repair_prompt_hides_invalid_evidence_ref(self):
        run = PILOT12 / "roots/cc-train-episode-216742/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = load(run / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        normalized = normalize_exact_duplicate_scene_reviews(raw)["output"]
        compilation = compile_round_validator_model_output(
            normalized, realized_scenes=validator_input["scenes"])
        self.assertFalse(compilation["valid"])
        plan = build_round_validator_contract_repair_plan(
            model_output=normalized,
            realized_scenes=validator_input["scenes"],
            compilation_errors=compilation["errors"],
            historical_anchor_refs={
                row["source_anchor_id"]: row["source_refs"]
                for row in validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan,
            frozen_input_sha256=validator_input[
                "frozen_round_validator_input_sha256"],
        )
        self.assertNotIn('"R01:S07:T006"', prompt)
        self.assertIn('"R01:S07:T005"', prompt)
        repaired = apply_round_validator_contract_repair(
            base_model_output=normalized,
            repair_output=sparse_repair_fixture(plan),
            repair_plan=plan,
        )
        self.assertTrue(repaired["valid"], repaired["errors"])
        compiled = compile_round_validator_model_output(
            repaired["reconstructed_output"],
            realized_scenes=validator_input["scenes"],
        )
        self.assertTrue(compiled["valid"], compiled["errors"])
        validation = validate_round_validator_output(
            compiled["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertTrue(validation["valid"], validation["errors"])

    def test_pilot11_exact_duplicate_and_new_progression_are_inert(self):
        run = PILOT11 / "roots/cc-train-episode-216742/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = load(run / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        compilation = compile_round_validator_model_output(
            raw, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        audit = compilation["compatibility_normalization"]
        self.assertEqual(
            [{"scene_alias": "s7", "duplicate_index": 7}],
            audit["removed_exact_duplicate_scene_reviews"],
        )
        self.assertTrue(audit["ignored_new_action_progression_fields"])
        self.assertFalse(audit["semantic_decision_changed"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertTrue(validation["valid"], validation["errors"])

        conflicting = deepcopy(raw)
        conflicting["scene_reviews"][-1]["decision"] = "reject"
        conflict_compilation = compile_round_validator_model_output(
            conflicting, realized_scenes=validator_input["scenes"])
        self.assertTrue(conflict_compilation["valid"],
                        conflict_compilation["errors"])
        conflict_validation = validate_round_validator_output(
            conflict_compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertFalse(conflict_validation["valid"])
        self.assertTrue(any("duplicate_scene_alias" in error
                            for error in conflict_validation["errors"]))

    def test_pilot11_dependency_source_error_uses_compact_repair(self):
        run = PILOT11 / "roots/cc-train-episode-41625/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = load(run / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        normalized = normalize_exact_duplicate_scene_reviews(raw)["output"]
        compilation = compile_round_validator_model_output(
            normalized, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertFalse(validation["valid"])
        self.assertIn("scene_review:1:dependency_context_source_not_declared",
                      validation["errors"])
        base = materialize_round_validator_optional_model_fields(normalized)
        base["durable_updates"] = []
        plan = build_round_validator_contract_repair_plan(
            model_output=base,
            realized_scenes=validator_input["scenes"],
            compilation_errors=validation["errors"],
            historical_anchor_refs={
                row["source_anchor_id"]: row["source_refs"]
                for row in validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        repair = apply_round_validator_contract_repair(
            base_model_output=base,
            repair_output=sparse_repair_fixture(plan),
            repair_plan=plan,
        )
        self.assertTrue(repair["valid"], repair["errors"])
        repaired_compilation = compile_round_validator_model_output(
            repair["reconstructed_output"],
            realized_scenes=validator_input["scenes"],
        )
        repaired_validation = validate_round_validator_output(
            repaired_compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertTrue(repaired_validation["valid"],
                        repaired_validation["errors"])

    def test_pilot10_planner_names_are_canonicalized_before_action_validation(self):
        run = PILOT10 / "roots/cc-train-episode-41625/01_expansion"
        outline = load(run / "attempts/planner_v11_attempt_01.json")[
            "parsed_response"]
        planning = load(run / "planning_state_view.json")
        anchors = load(run / "source_anchor_registry.json")
        placements = load(run / "placement_choice_registry.json")
        availability = load(run / "action_availability_view.json")
        validation = validate_planner_outline_v11(
            outline,
            source_anchor_registry=anchors,
            placement_choice_registry=placements,
            authoritative_entity_refs={
                row["entity_ref"] for row in planning["entities"]},
            authoritative_entities=planning["entities"],
            action_availability_view=availability,
        )
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            outline,
            source_anchor_registry=anchors,
            placement_choice_registry=placements,
            planning_state=planning,
            action_availability_view=availability,
        )
        self.assertEqual(
            planning["entities"][0]["entity_ref"],
            compiled["scenes"][0]["primary_action"][
                "resolved_arguments"]["target"],
        )
        self.assertEqual(
            "R01:P01",
            compiled["scenes"][3]["secondary_action"][
                "resolved_arguments"]["target"],
        )
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=anchors,
            placement_choice_registry=placements,
            planning_state=planning,
            complete_dialogue=load(run / "complete_dialogue_view.json"),
        )
        self.assertTrue(structural["valid"], structural["errors"])

    def test_pilot10_validator_omitted_dependency_refs_use_compact_repair(self):
        run = PILOT10 / "roots/cc-train-episode-216742/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = deepcopy(load(
            run / "attempts/round_validator_attempt_01.json"
        )["parsed_response"])
        self.assertTrue(all(
            "dependency_context_turn_refs" not in row
            for row in raw["scene_reviews"]
        ))
        compilation = compile_round_validator_model_output(
            raw, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        self.assertTrue(all(
            row["dependency_context_turn_refs"] == []
            for row in compilation["compiled_output"]["scene_reviews"]
        ))
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(all(
            "progression" in error for error in validation["errors"]
        ), validation["errors"])
        projected = materialize_round_validator_optional_model_fields(raw)
        projected["durable_updates"] = []
        plan = build_round_validator_contract_repair_plan(
            model_output=projected,
            realized_scenes=validator_input["scenes"],
            compilation_errors=validation["errors"],
            historical_anchor_refs={
                row["source_anchor_id"]: row["source_refs"]
                for row in validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        repair = apply_round_validator_contract_repair(
            base_model_output=projected,
            repair_output=sparse_repair_fixture(plan),
            repair_plan=plan,
        )
        self.assertTrue(repair["valid"], repair["errors"])
        repaired_compilation = compile_round_validator_model_output(
            repair["reconstructed_output"],
            realized_scenes=validator_input["scenes"],
        )
        self.assertTrue(repaired_compilation["valid"],
                        repaired_compilation["errors"])
        repaired_validation = validate_round_validator_output(
            repaired_compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertTrue(repaired_validation["valid"],
                        repaired_validation["errors"])

    def test_dialogue_only_validator_prompt_forbids_memory_candidates(self):
        run = PILOT7 / "roots/cc-train-episode-216742/01_expansion"
        prompt = build_round_validator_prompt(
            load(run / "round_validator_input.json"),
            durable_update_policy="dialogue_only",
        )
        self.assertIn("DIALOGUE-ONLY MODE", prompt)
        self.assertIn("Return durable_updates=[] exactly", prompt)
        self.assertIn('"rows":[]', prompt)
        self.assertNotIn('"candidate_update_key":"DU001"', prompt)
        self.assertIn(
            '"status":null,"evidence_turn_refs":["!"],'
            '"progression_evidence_turn_refs":[],"reason":""',
            prompt,
        )
        self.assertIn(
            '"status":null,"evidence_turn_refs":["!"],'
            '"progression_evidence_turn_refs":["!"],"reason":""',
            prompt,
        )

    def test_pilot7_missing_review_template_requires_explicit_evidence(self):
        run = PILOT7 / "roots/cc-train-episode-216742/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = deepcopy(load(
            run / "attempts/round_validator_attempt_01.json"
        )["parsed_response"])
        raw["durable_updates"] = []
        compilation = compile_round_validator_model_output(
            raw, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        errors = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )["errors"]
        plan = build_round_validator_contract_repair_plan(
            model_output=raw,
            realized_scenes=validator_input["scenes"],
            compilation_errors=errors,
            historical_anchor_refs={
                row["source_anchor_id"]: row["source_refs"]
                for row in validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(4, len(plan["missing_scene_review_targets"]))
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan,
            frozen_input_sha256=validator_input[
                "frozen_round_validator_input_sha256"],
        )
        self.assertIn('"current_scene_supporting_turn_refs":["!"]', prompt)
        self.assertIn('"status":"<STATUS>"', prompt)
        self.assertIn('"reason":"<NONEMPTY_REASON>"', prompt)

    def test_pilot7_mcq_materialization_and_option_diagnostic_replay(self):
        root = PILOT7 / "roots/cc-train-episode-41625"
        gold = load(root / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(root / "02_questioner/frozen/span_catalog.json")
        response = load(
            root / "04_mcq_materializer/raw/distractor_writer/"
            "cc-train-episode-41625-round1-distractors-a00.json"
        )
        materialization = materialize_single_item_mcq(
            core_qa=gold,
            distractors=response["validation"]["distractors"],
            placement=deterministic_answer_placement(
                "cc-train-episode-41625-round-1", 0),
            span_catalog=catalog,
        )
        mcq = materialization["materialized_qa"]
        diagnostic = diagnose_option_shape(
            options=mcq["options"],
            reference_answer_letter=mcq["reference_answer_letter"],
        )
        self.assertEqual(set("ABCD"), set(mcq["options"]))
        self.assertIn("answer_shape_leakage_risk", diagnostic)

    def test_mcq_sidecar_preserves_evidence_mismatch_marker(self):
        audit = evidence_alignment_audit({
            "evidence_mismatch_policy": "mark_only",
            "evidence_mismatch_marked": True,
            "evidence_mismatch_reasons": ["answer_not_supported_by_evidence"],
            "semantic_entailment_independently_verified": False,
        })
        self.assertEqual("mismatch_marked", audit["evidence_alignment_status"])
        self.assertTrue(audit["evidence_mismatch_marked"])
        self.assertEqual(["answer_not_supported_by_evidence"],
                         audit["evidence_mismatch_reasons"])

    def test_pilot8_duplicate_reason_is_strictly_rejected(self):
        raw = (PILOT8 / "roots/cc-train-episode-216742/01_expansion/raw/"
               "backend_responses/round_validator_attempt_01.txt").read_text(
                   encoding="utf-8")
        normalized = normalize_round_validator_output(raw)
        self.assertFalse(normalized["parse_valid"])
        self.assertTrue(any(
            "duplicate_json_object_key:reason" in error
            for error in normalized["errors"]
        ))

    def test_pilot8_uniform_action_review_shadow_closes_contract(self):
        run = PILOT8 / "roots/cc-train-episode-41625/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = deepcopy(load(
            run / "attempts/round_validator_attempt_02.json"
        )["parsed_response"])
        scenes = {row["scene_alias"]: row for row in validator_input["scenes"]}
        for review in raw["scene_reviews"]:
            if "current_scene_support_refs" in review:
                review["current_scene_supporting_turn_refs"] = review.pop(
                    "current_scene_support_refs")
            scene = scenes[review["scene_alias"]]
            for role in ("primary", "secondary"):
                action = scene.get(f"{role}_action")
                realization = review.get(f"{role}_action_realization")
                if not isinstance(action, dict) or not isinstance(realization, dict):
                    continue
                if action.get("action_mode") == "continuation":
                    realization.setdefault(
                        "progression_evidence_turn_refs",
                        realization["evidence_turn_refs"][:1],
                    )
                else:
                    realization.setdefault("progression_evidence_turn_refs", [])
        compilation = compile_round_validator_model_output(
            raw, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertTrue(validation["valid"], validation["errors"])

    def test_pilot9_validator_repair_collects_latent_continuation_errors(self):
        run = PILOT9 / "roots/cc-train-episode-41625/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = load(
            run / "attempts/round_validator_attempt_01.json"
        )["parsed_response"]
        projected = deepcopy(raw)
        projected["durable_updates"] = []
        compilation = compile_round_validator_model_output(
            projected, realized_scenes=validator_input["scenes"])
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
        )
        self.assertFalse(validation["valid"])
        plan = build_round_validator_contract_repair_plan(
            model_output=projected,
            realized_scenes=validator_input["scenes"],
            compilation_errors=validation["errors"],
            historical_anchor_refs={
                row["source_anchor_id"]: row["source_refs"]
                for row in validator_input["historical_anchor_index"]
            },
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        progression_targets = {
            row["action_ref"] for row in plan["targets"]
            if row.get("progression_evidence_turn_refs_requires_repair")
        }
        self.assertEqual({
            "S02.primary_action", "S03.primary_action", "S04.primary_action",
            "S05.primary_action", "S07.primary_action",
        }, progression_targets)
        repair = sparse_repair_fixture(plan)

        def backend(_prompt, attempt):
            return deepcopy(raw if attempt == 1 else repair)

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=validator_input,
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
            durable_update_policy="dialogue_only",
        )
        self.assertTrue(result["completed"], result["attempts"])

    def test_pilot9_questioner_answer_is_preserved_not_materialized_locally(self):
        run = PILOT9 / "roots/cc-train-episode-216742/02_questioner"
        stored = load(
            run / "raw/question_answer_writer/"
            "cc-train-episode-216742-round1-cross-boundary-qa-a00.json"
        )
        raw = json.loads(stored["raw_response"])
        view = load(run / "frozen/qa_source_view.json")
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=load(run / "frozen/span_catalog.json"),
            partition_by_turn=view["partition_by_turn"],
        )
        self.assertEqual("cross_boundary_qa_accept", result["decision"])
        self.assertFalse(result["gold_materialized_locally"])
        self.assertTrue(result["questioner_selected_evidence"])
        self.assertEqual(raw["correct_answer_text"],
                         result["core_qa"]["correct_answer_text"])

    def test_dialogue_only_retry_plan_does_not_reintroduce_durable_targets(self):
        run = PILOT5 / "roots/cc-train-episode-216742/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        responses = [
            load(run / "attempts/round_validator_attempt_01.json")[
                "parsed_response"],
            load(run / "attempts/round_validator_attempt_02.json")[
                "parsed_response"],
        ]
        responses[1] = {"repairs": responses[1]["repairs"][:2]}

        def backend(_prompt, attempt):
            return deepcopy(responses[attempt - 1])

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=validator_input,
            realized_scenes=validator_input["scenes"],
            review_world_view=load(run / "review_world_view.json"),
            local_gate_results=load(run / "local_scene_gate_results.json"),
            durable_update_policy="dialogue_only",
        )
        self.assertTrue(result["completed"])
        self.assertEqual(2, result["physical_calls"])
        self.assertEqual([], result["valid_output"]["durable_updates"])
        self.assertEqual([], result["attempts"][1]["validation"]["errors"])

    def test_missing_scene_review_retry_prompt_is_compact(self):
        run = PILOT5 / "roots/cc-train-episode-41625/01_expansion"
        validator_input = load(run / "round_validator_input.json")
        raw = deepcopy(load(
            run / "attempts/round_validator_attempt_01.json"
        )["parsed_response"])
        raw["durable_updates"] = []
        compilation = compile_round_validator_model_output(
            raw, realized_scenes=validator_input["scenes"])
        self.assertFalse(compilation["valid"])
        historical = {
            row["source_anchor_id"]: row["source_refs"]
            for row in validator_input["historical_anchor_index"]
        }
        plan = build_round_validator_contract_repair_plan(
            model_output=raw,
            realized_scenes=validator_input["scenes"],
            compilation_errors=compilation["errors"],
            historical_anchor_refs=historical,
            eligible_durable_action_refs=[],
        )
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(5, len(plan["missing_scene_review_targets"]))
        self.assertFalse(plan["durable_update_targets"])
        self.assertFalse(plan["durable_removal_targets"])
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan,
            frozen_input_sha256=validator_input[
                "frozen_round_validator_input_sha256"],
        )
        self.assertLess(len(prompt), 30_000)
        self.assertNotIn("linked_existing_refs", prompt)
        self.assertNotIn("interaction_beats", prompt)

    def test_pilot6_sparse_repair_closes_both_real_validator_failures(self):
        for root_id in ("cc-train-episode-216742", "cc-train-episode-41625"):
            with self.subTest(root_id=root_id):
                run = PILOT6 / "roots" / root_id / "01_expansion"
                validator_input = load(run / "round_validator_input.json")
                raw = load(
                    run / "attempts/round_validator_attempt_01.json"
                )["parsed_response"]
                projected = deepcopy(raw)
                projected["durable_updates"] = []
                compilation = compile_round_validator_model_output(
                    projected, realized_scenes=validator_input["scenes"])
                if compilation["valid"]:
                    errors = validate_round_validator_output(
                        compilation["compiled_output"],
                        realized_scenes=validator_input["scenes"],
                        review_world_view=load(run / "review_world_view.json"),
                        local_gate_results=load(
                            run / "local_scene_gate_results.json"),
                    )["errors"]
                else:
                    errors = compilation["errors"]
                plan = build_round_validator_contract_repair_plan(
                    model_output=projected,
                    realized_scenes=validator_input["scenes"],
                    compilation_errors=errors,
                    historical_anchor_refs={
                        row["source_anchor_id"]: row["source_refs"]
                        for row in validator_input["historical_anchor_index"]
                    },
                    eligible_durable_action_refs=[],
                )
                repair = sparse_repair_fixture(plan)

                def backend(_prompt, attempt):
                    return deepcopy(raw if attempt == 1 else repair)

                result = run_round_validator_with_retry(
                    backend_call=backend,
                    validator_input=validator_input,
                    realized_scenes=validator_input["scenes"],
                    review_world_view=load(run / "review_world_view.json"),
                    local_gate_results=load(run / "local_scene_gate_results.json"),
                    durable_update_policy="dialogue_only",
                )
                self.assertTrue(result["completed"], result["attempts"])
                self.assertEqual([], result["attempts"][1]["validation"]["errors"])
                self.assertEqual([], result["valid_output"]["durable_updates"])


if __name__ == "__main__":
    unittest.main()
