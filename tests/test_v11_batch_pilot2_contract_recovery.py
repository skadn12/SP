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
    build_planner_namespace_only_retry_prompt_v11,
    compile_planner_namespace_repair_spec_v11,
    validate_planner_namespace_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_input,
    build_round_validator_prompt,
    compile_round_validator_model_output,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    validate_round_validator_contract_repair_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


BATCH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_2"
)
BATCH_SHA = "c124d65f9a4eaea15cc2df28892f4f74ebf1fd0d60a7160e00b9706c9ae959d3"
PLANNER_RUN = BATCH / "roots/cc-train-episode-216742/01_expansion"
VALIDATOR_RUN = BATCH / "roots/cc-train-episode-41625/01_expansion"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot2ContractRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner_outline = load(
            PLANNER_RUN / "attempts/planner_v11_attempt_01.json"
        )["parsed_response"]
        cls.planner_attempt = load(
            PLANNER_RUN / "planner/attempt_01_validation.json"
        )
        cls.planning = load(PLANNER_RUN / "planning_state_view.json")
        cls.availability = load(PLANNER_RUN / "action_availability_view.json")
        cls.anchors = load(PLANNER_RUN / "source_anchor_registry.json")
        cls.placements = load(PLANNER_RUN / "placement_choice_registry.json")
        cls.namespace_spec = compile_planner_namespace_repair_spec_v11(
            outline=cls.planner_outline,
            errors=cls.planner_attempt["errors"],
            validation=cls.planner_attempt["validation"],
            action_contracts=action_contract_registry(),
            action_availability_view=cls.availability,
            authoritative_role_table=cls.planning["entities"],
        )

        cls.validator_input = load(VALIDATOR_RUN / "round_validator_input.json")
        cls.realized_scenes = cls.validator_input["scenes"]
        cls.validator_raw = load(
            VALIDATOR_RUN / "attempts/round_validator_attempt_01.json"
        )["parsed_response"]
        cls.validator_errors = load(
            VALIDATOR_RUN / "round_validator_run.json"
        )["attempts"][0]["validation"]["errors"]
        anchor_refs = {
            row["source_anchor_id"]: row["source_refs"]
            for row in cls.validator_input["historical_anchor_index"]
        }
        eligible = [
            action["action_ref"]
            for scene in cls.realized_scenes
            for role in ("primary", "secondary")
            if isinstance((action := scene.get(f"{role}_action")), dict)
        ]
        cls.validator_plan = build_round_validator_contract_repair_plan(
            model_output=cls.validator_raw,
            realized_scenes=cls.realized_scenes,
            compilation_errors=cls.validator_errors,
            historical_anchor_refs=anchor_refs,
            eligible_durable_action_refs=eligible,
        )

    def planner_repair_fixture(self):
        refs = {
            row["display_name"]: row["entity_ref"]
            for row in self.planning["entities"]
        }
        replacement_by_name = {
            "Classmates A": refs["Classmates A"],
            "Classmates B": refs["Classmates B"],
            "Friend C": "local_1",
        }
        repairs = []
        for context in self.namespace_spec["action_repair_spec"][
                "target_scene_contexts"]:
            action = deepcopy(context["original_action"])
            action["arguments"]["target"] = replacement_by_name[
                action["arguments"]["target"]]
            repairs.append({
                "action_ref": context["action_ref"],
                "replacement": action,
            })
        link_choice = next(
            row["choice_id"] for row in self.namespace_spec["link_type_choices"]
            if row["link_type"] == "consequence_followup"
        )
        return {
            "action_repairs": repairs,
            "link_type_repairs": [{
                "scene_ref": "S07", "replacement_choice_id": link_choice,
            }],
        }

    def validator_repair_fixture(self):
        rows = []
        for target in self.validator_plan["scene_reference_targets"]:
            current = (
                target["allowed_current_scene_turn_refs"][:1]
                if target["repair_current_scene_supporting_turn_refs"] else
                target["original_current_scene_supporting_turn_refs"]
            )
            dependency = (
                target["allowed_dependency_turn_refs"][:1]
                if target["repair_dependency_context_turn_refs"] else
                target["original_dependency_context_turn_refs"]
            )
            uses = (
                sorted({ref.rsplit(":T", 1)[0] for ref in dependency})
                if target["repair_uses_scene_refs"] else
                target["original_uses_scene_refs"]
            )
            historical = (
                target["allowed_historical_turn_refs"][:1]
                if target["repair_historical_supporting_turn_refs"] else
                target["original_historical_supporting_turn_refs"]
            )
            rows.append({
                "repair_kind": "scene_reference",
                "scene_alias": target["scene_alias"],
                "current_scene_supporting_turn_refs": current,
                "dependency_context_turn_refs": dependency,
                "historical_supporting_turn_refs": historical,
                "uses_scene_refs": uses,
            })
        for target in self.validator_plan["targets"]:
            evidence = (
                target["allowed_evidence_turn_refs"][:1]
                if target["evidence_turn_refs_requires_repair"] else
                target["original_evidence_turn_refs"]
            )
            rows.append({
                "repair_kind": "action_realization",
                "action_ref": target["action_ref"],
                "evidence_turn_refs": evidence,
                "progression_evidence_turn_refs": evidence[:1],
            })
        for target in self.validator_plan["durable_update_targets"]:
            rows.append({
                "repair_kind": "durable_update_evidence",
                "candidate_update_key": target["candidate_update_key"],
                "evidence_turn_refs": target["allowed_evidence_turn_refs"][:1],
            })
        for target in self.validator_plan["durable_removal_targets"]:
            rows.append({
                "repair_kind": "durable_update_nullification",
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        for target in self.validator_plan["verified_connection_removal_targets"]:
            rows.append({
                "repair_kind": "verified_connection_nullification",
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        return {"repairs": rows}

    def test_planner_reference_repair_freezes_action_semantics(self):
        constraints = self.namespace_spec["action_repair_spec"][
            "reference_only_constraints"]
        self.assertEqual(len(constraints), 4)
        self.assertTrue(all(
            row["required_action_type"] == "develop_character_or_task_state"
            and row["mutable_argument_paths"] == ["target"]
            for row in constraints
        ))
        prompt = build_planner_namespace_only_retry_prompt_v11(
            self.namespace_spec)
        self.assertIn("REFERENCE-ONLY FREEZE", prompt)
        self.assertIn("Classmates A", prompt)
        self.assertIn("cc-train-episode-216742:E:17ca03a05c10", prompt)

    def test_historical_namespace_patch_is_rejected_for_frozen_semantic_drift(self):
        historical = load(
            PLANNER_RUN / "attempts/planner_v11_attempt_02.json"
        )["parsed_response"]
        validation = validate_planner_namespace_repair_response_v11(
            historical, spec=self.namespace_spec)
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "reference_only_frozen_argument_changed" in error
            for error in validation["errors"]
        ))

    def test_planner_shadow_repair_passes_compile_and_structural_gate(self):
        applied = apply_planner_namespace_repairs_v11(
            frozen_outline=self.planner_outline,
            response=self.planner_repair_fixture(),
            spec=self.namespace_spec,
        )
        self.assertTrue(applied["valid"], applied["errors"])
        outline = applied["reconstructed_outline"]
        validation = validate_planner_outline_v11(
            outline,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            action_availability_view=self.availability,
            authoritative_entity_refs={
                row["entity_ref"] for row in self.planning["entities"]
            },
            authoritative_entities=self.planning["entities"],
        )
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            outline,
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

    def test_validator_raw_and_historical_repair_remain_invalid(self):
        self.assertTrue(any(
            "continuation_model_schema" in error
            for error in self.validator_errors
        ))
        historical = load(
            VALIDATOR_RUN / "attempts/round_validator_attempt_02.json"
        )["parsed_response"]
        validation = validate_round_validator_contract_repair_output(
            historical, repair_plan=self.validator_plan)
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "scene_reference_repair" in error
            for error in validation["errors"]
        ))

    def test_validator_repair_preserves_scene_and_drops_unsafe_update(self):
        self.assertTrue(self.validator_plan["valid"], self.validator_plan["errors"])
        self.assertEqual(
            [row["candidate_update_key"]
             for row in self.validator_plan["durable_removal_targets"]],
            ["DU001"],
        )
        self.assertEqual(self.validator_plan["verified_connection_targets"], [])
        fixture = self.validator_repair_fixture()
        applied = apply_round_validator_contract_repair(
            base_model_output=self.validator_raw,
            repair_output=fixture,
            repair_plan=self.validator_plan,
        )
        self.assertTrue(applied["valid"], applied["errors"])
        reconstructed = applied["reconstructed_output"]
        self.assertNotIn(
            "DU001", {row["candidate_update_key"]
                      for row in reconstructed["durable_updates"]})
        compilation = compile_round_validator_model_output(
            reconstructed, realized_scenes=self.realized_scenes)
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=self.realized_scenes,
            review_world_view=load(VALIDATOR_RUN / "review_world_view.json"),
            local_gate_results=load(VALIDATOR_RUN / "local_scene_gate_results.json"),
        )
        self.assertTrue(validation["valid"], validation["errors"])

    def test_new_validator_input_exposes_disjoint_reference_scopes(self):
        projected = build_round_validator_input(
            round_blueprint=load(VALIDATOR_RUN / "planner/compiled_round_plan.json"),
            review_world_view=load(VALIDATOR_RUN / "review_world_view.json"),
            realized_scenes=self.realized_scenes,
            scene_dependency_dag=load(VALIDATOR_RUN / "scene_dependency_dag.json"),
            structural_gate_results=load(
                VALIDATOR_RUN / "local_scene_gate_results.json"),
        )
        for scene in projected["scenes"]:
            scope = scene["review_reference_scope"]
            self.assertEqual(
                scope["historical_source_anchor_id"],
                scene["source_anchor_id"],
            )
            self.assertEqual(scope["current_scene_ref"], scene["scene_ref"])
            self.assertTrue(all(
                ref.startswith("R01:")
                for ref in scope["dependency_scene_refs"]
            ))
        prompt = build_round_validator_prompt(projected)
        self.assertIn("review_reference_scope", prompt)
        self.assertIn("Never copy an R01 ref into historical support", prompt)

    def test_batch_history_is_unchanged(self):
        self.assertEqual(hash_tree(BATCH)["tree_sha256"], BATCH_SHA)


if __name__ == "__main__":
    unittest.main()
