from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_action_repair_v11 import (
    apply_planner_action_repairs_v11,
    compile_planner_action_repair_spec_v11,
    validate_planner_action_repair_response_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11,
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    run_round_validator_with_retry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    build_round_validator_contract_repair_plan,
    validate_round_validator_contract_repair_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


BATCH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_3"
)
BATCH_SHA = "660e3c13285d1d6b5b11dd71a8ab1065134774c0a1a2eaa10f1d0e8f8d4ce059"
PLANNER_RUN = BATCH / "roots/cc-train-episode-41625/01_expansion"
VALIDATOR_RUN = BATCH / "roots/cc-train-episode-216742/01_expansion"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot3ContractRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner_outline = load(
            PLANNER_RUN / "attempts/planner_v11_attempt_01.json"
        )["parsed_response"]
        cls.planner_validation = load(
            PLANNER_RUN / "planner/attempt_01_validation.json"
        )
        cls.planning = load(PLANNER_RUN / "planning_state_view.json")
        cls.availability = load(PLANNER_RUN / "action_availability_view.json")
        cls.anchors = load(PLANNER_RUN / "source_anchor_registry.json")
        cls.placements = load(PLANNER_RUN / "placement_choice_registry.json")
        cls.action_spec = compile_planner_action_repair_spec_v11(
            outline=cls.planner_outline,
            errors=cls.planner_validation["errors"],
            validation=cls.planner_validation["validation"],
            action_contracts=action_contract_registry(),
            action_availability_view=cls.availability,
            authoritative_role_table=cls.planning["entities"],
        )
        cls.historical_action_patch = load(
            PLANNER_RUN / "attempts/planner_v11_attempt_03.json"
        )["parsed_response"]

        cls.validator_input = load(VALIDATOR_RUN / "round_validator_input.json")
        cls.realized_scenes = cls.validator_input["scenes"]
        cls.validator_raw = load(
            VALIDATOR_RUN / "attempts/round_validator_attempt_01.json"
        )["parsed_response"]
        cls.validator_attempts = load(VALIDATOR_RUN / "round_validator_run.json")
        cls.validator_errors = cls.validator_attempts["attempts"][0][
            "validation"
        ]["errors"]
        historical_anchor_refs = {
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
            historical_anchor_refs=historical_anchor_refs,
            eligible_durable_action_refs=eligible,
        )

    def corrected_action_patch(self):
        patch = deepcopy(self.historical_action_patch)
        replacements = {
            row["action_ref"]: row for row in patch["action_repairs"]
        }
        replacements["S04.primary_action"]["replacement"] = {
            "action_mode": "continuation",
            "continuation_of": "S03.primary_action",
            "progression": {
                "stage": "development",
                "new_step": (
                    "Turn the initial festival proposal into a detailed "
                    "submission package for council review."
                ),
            },
        }
        replacements["S06.primary_action"]["replacement"] = {
            "action_mode": "continuation",
            "continuation_of": "S02.primary_action",
            "progression": {
                "stage": "consolidation",
                "new_step": (
                    "Move Council Official C from an introduced contact into "
                    "an active review role for the revised proposal."
                ),
            },
        }
        return patch

    def validator_repair_fixture(self):
        rows = []
        for target in self.validator_plan["scene_reference_targets"]:
            current = (
                target["allowed_current_scene_turn_refs"][:1]
                if target["repair_current_scene_supporting_turn_refs"]
                else target["original_current_scene_supporting_turn_refs"]
            )
            dependency = target["original_dependency_context_turn_refs"]
            uses = target["original_uses_scene_refs"]
            if target["repair_dependency_context_turn_refs"]:
                allowed_owners = set(uses or [])
                dependency = [
                    ref for ref in target["allowed_dependency_turn_refs"]
                    if ref.rsplit(":T", 1)[0] in allowed_owners
                ][:1]
                if not dependency:
                    dependency = target["allowed_dependency_turn_refs"][:1]
            if target["repair_uses_scene_refs"]:
                uses = sorted({ref.rsplit(":T", 1)[0] for ref in dependency})
            historical = (
                target["allowed_historical_turn_refs"][:1]
                if target["repair_historical_supporting_turn_refs"]
                else target["original_historical_supporting_turn_refs"]
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
                if target["evidence_turn_refs_requires_repair"]
                else target["original_evidence_turn_refs"]
            )
            row = {
                "repair_kind": "action_realization",
                "action_ref": target["action_ref"],
                "evidence_turn_refs": evidence,
            }
            if target["action_mode"] == "continuation":
                row["progression_evidence_turn_refs"] = evidence[:1]
            rows.append(row)
        for target in self.validator_plan["durable_removal_targets"]:
            rows.append({
                "repair_kind": "durable_update_nullification",
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        return {"repairs": rows}

    def test_planner_spec_reserves_untouched_identity_aliases(self):
        reserved = {
            (row["alias"], row["owner_action_ref"])
            for row in self.action_spec["reserved_narrative_aliases"]
        }
        self.assertEqual(reserved, {
            ("local_1", "S02.primary_action"),
            ("music_festival_proposal", "S03.primary_action"),
        })

    def test_historical_action_patch_cannot_bypass_program_connection_choices(self):
        validation = validate_planner_action_repair_response_v11(
            self.historical_action_patch, spec=self.action_spec
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "program_connection_choice_unknown" in error
            for error in validation["errors"]
        ))

    def test_planner_continuation_shadow_passes_complete_local_chain(self):
        response = self.corrected_action_patch()
        validation = validate_planner_action_repair_response_v11(
            response, spec=self.action_spec
        )
        self.assertTrue(validation["valid"], validation["errors"])
        applied = apply_planner_action_repairs_v11(
            frozen_outline=self.planner_outline,
            response=response,
            spec=self.action_spec,
        )
        self.assertTrue(applied["valid"], applied["errors"])
        outline = applied["reconstructed_outline"]
        outline_validation = validate_planner_outline_v11(
            outline,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            authoritative_entity_refs={
                row["entity_ref"] for row in self.planning["entities"]
            },
            action_availability_view=self.availability,
            authoritative_entities=self.planning["entities"],
        )
        self.assertTrue(outline_validation["valid"], outline_validation["errors"])
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

    def test_validator_plan_splits_new_and_continuation_repair_shapes(self):
        self.assertTrue(self.validator_plan["valid"], self.validator_plan["errors"])
        by_ref = {row["action_ref"]: row for row in self.validator_plan["targets"]}
        self.assertEqual(by_ref["S03.primary_action"]["action_mode"], "continuation")
        self.assertTrue(by_ref["S03.primary_action"][
            "progression_evidence_turn_refs_requires_repair"
        ])
        self.assertEqual(by_ref["S03.secondary_action"]["action_mode"], "new")
        self.assertFalse(by_ref["S03.secondary_action"][
            "progression_evidence_turn_refs_requires_repair"
        ])
        fixture = self.validator_repair_fixture()
        by_action = {
            row["action_ref"]: row for row in fixture["repairs"]
            if row["repair_kind"] == "action_realization"
        }
        self.assertIn("progression_evidence_turn_refs", by_action["S03.primary_action"])
        self.assertNotIn(
            "progression_evidence_turn_refs", by_action["S03.secondary_action"]
        )

    def test_new_action_repair_rejects_progression_evidence(self):
        fixture = self.validator_repair_fixture()
        row = next(
            item for item in fixture["repairs"]
            if item.get("action_ref") == "S03.secondary_action"
        )
        row["progression_evidence_turn_refs"] = row["evidence_turn_refs"][:1]
        validation = validate_round_validator_contract_repair_output(
            fixture, repair_plan=self.validator_plan
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            error.endswith(":schema")
            for error in validation["errors"]
        ))

    def test_missing_semantic_status_removes_all_durable_candidates(self):
        removals = self.validator_plan["durable_removal_targets"]
        self.assertEqual(
            [row["candidate_update_key"] for row in removals],
            [f"DU{index:03d}" for index in range(1, 9)],
        )
        self.assertTrue(all(
            row["reason"] == "verification_status_invalid_or_missing"
            for row in removals
        ))
        self.assertEqual(self.validator_plan["durable_update_targets"], [])
        self.assertEqual(self.validator_plan["verified_connection_targets"], [])

    def test_actual_first_response_routes_to_compact_contract_repair(self):
        fixture = self.validator_repair_fixture()
        calls = []

        def backend(prompt, attempt):
            calls.append((prompt, attempt))
            return self.validator_raw if attempt == 1 else fixture

        result = run_round_validator_with_retry(
            backend_call=backend,
            validator_input=self.validator_input,
            realized_scenes=self.realized_scenes,
            review_world_view=load(VALIDATOR_RUN / "review_world_view.json"),
            local_gate_results=load(
                VALIDATOR_RUN / "local_scene_gate_results.json"
            ),
        )
        self.assertTrue(result["completed"], result["attempts"])
        self.assertEqual(result["physical_calls"], 2)
        self.assertEqual(
            [row["request_mode"] for row in result["attempts"]],
            ["full_review", "contract_only_repair"],
        )
        self.assertTrue(result["contract_only_repair_used"])
        self.assertEqual(result["valid_output"]["durable_updates"], [])

    def test_historical_second_validator_response_remains_invalid(self):
        second = load(
            VALIDATOR_RUN / "attempts/round_validator_attempt_02.json"
        )["parsed_response"]
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
            compile_round_validator_model_output,
            validate_round_validator_output,
        )
        compilation = compile_round_validator_model_output(
            second, realized_scenes=self.realized_scenes
        )
        self.assertTrue(compilation["valid"], compilation["errors"])
        validation = validate_round_validator_output(
            compilation["compiled_output"],
            realized_scenes=self.realized_scenes,
            review_world_view=load(VALIDATOR_RUN / "review_world_view.json"),
            local_gate_results=load(
                VALIDATOR_RUN / "local_scene_gate_results.json"
            ),
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(any(
            "evidence_turn_refs" in error for error in validation["errors"]
        ))

    def test_batch_history_is_unchanged(self):
        self.assertEqual(hash_tree(BATCH)["tree_sha256"], BATCH_SHA)


if __name__ == "__main__":
    unittest.main()
