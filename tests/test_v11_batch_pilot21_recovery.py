from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.core_qa_admission import admit_core_qa
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    validate_cross_boundary_core_object,
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
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    validate_round_blueprint_structure,
)


PILOT21 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_21"
)
PILOT21_SHA = "835a3c104c393a625e2f4af09a6be22202c9d9555cecbf2ad2a238b417ade38e"


class BatchPilot21RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expansion = PILOT21 / "roots/cc-train-episode-216742/01_expansion"
        cls.outline = cls.load("attempts/planner_v11_attempt_01.json")[
            "parsed_response"]
        cls.validation = cls.load("planner/attempt_01_validation.json")[
            "validation"]
        cls.availability = cls.load("action_availability_view.json")
        cls.planning = cls.load("planning_state_view.json")
        cls.anchors = cls.load("source_anchor_registry.json")
        cls.placements = cls.load("placement_choice_registry.json")
        cls.contracts = cls.load("shared_narrative_action_contracts.json")
        cls.roles = cls.planning["entities"]
        cls.spec = compile_planner_namespace_repair_spec_v11(
            outline=cls.outline,
            errors=cls.validation["errors"],
            validation=cls.validation,
            action_contracts=cls.contracts,
            action_availability_view=cls.availability,
            authoritative_role_table=cls.roles,
        )

    @classmethod
    def load(cls, relative: str):
        return json.loads((cls.expansion / relative).read_text(encoding="utf-8"))

    def valid_namespace_response(self) -> dict:
        action_spec = self.spec["action_repair_spec"]
        constraint = action_spec["reference_only_constraints"][0]
        connection_choice = constraint["connection_choice_ids"][0]
        link_choice = next(
            row["choice_id"] for row in self.spec["link_type_choices"]
            if row["link_type"] == "relationship_followup")
        return {
            "action_repairs": [{
                "action_ref": "S04.primary_action",
                "replacement": {
                    "action_mode": "new",
                    "action_type": "introduce_connected_person",
                    "arguments": {
                        "connection_choice_id": connection_choice,
                        **constraint["frozen_arguments"],
                    },
                },
            }],
            "link_type_repairs": [{
                "scene_ref": "S07",
                "replacement_choice_id": link_choice,
            }],
        }

    def test_connected_person_reference_repair_freezes_alias_and_semantics(self):
        action_spec = self.spec["action_repair_spec"]
        constraint = action_spec["reference_only_constraints"][0]
        self.assertEqual("S04.primary_action", constraint["action_ref"])
        self.assertEqual("local_1", constraint["frozen_narrative_alias"])
        self.assertEqual(["connection_target_ref"],
                         constraint["mutable_argument_paths"])
        self.assertEqual(["CCP0001", "CCP0002"],
                         constraint["connection_choice_ids"])
        prompt = build_planner_namespace_only_retry_prompt_v11(self.spec)
        output_tail = prompt.rsplit("OUTPUT ONLY THIS EXACT SHAPE", 1)[1]
        self.assertIn('"connection_choice_id"', output_tail)
        self.assertNotIn('"connection_target_ref"', output_tail)
        self.assertNotIn('"participant_alias"', output_tail)

    def test_historical_patch_remains_rejected_but_choice_patch_passes_chain(self):
        historical = self.load("attempts/planner_v11_attempt_02.json")[
            "parsed_response"]
        historical_validation = validate_planner_namespace_repair_response_v11(
            historical, spec=self.spec)
        self.assertFalse(historical_validation["valid"])

        result = apply_planner_namespace_repairs_v11(
            frozen_outline=self.outline,
            response=self.valid_namespace_response(),
            spec=self.spec,
        )
        self.assertTrue(result["valid"], result["errors"])
        repaired = result["reconstructed_outline"]
        arguments = repaired["scenes"][3]["primary_action"]["arguments"]
        self.assertEqual("local_1", arguments["participant_alias"])
        self.assertIn(arguments["connection_target_ref"], {
            row["entity_ref"] for row in self.roles})

        validation = validate_planner_outline_v11(
            repaired,
            source_anchor_registry=self.anchors,
            placement_choice_registry=self.placements,
            authoritative_entity_refs={row["entity_ref"] for row in self.roles},
            authoritative_entities=self.roles,
            action_availability_view=self.availability,
        )
        self.assertTrue(validation["valid"], validation["errors"])
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
            complete_dialogue=self.load("complete_dialogue_view.json"),
        )
        self.assertTrue(structural["valid"], structural["errors"])

    def test_questioner_rejects_unsupported_named_speaker_before_mcq(self):
        qa_root = (PILOT21 / "roots/cc-train-episode-41625/qa_candidates/"
                   "candidate_01/02_questioner")
        view = json.loads((qa_root / "frozen/qa_source_view.json").read_text())
        catalog = json.loads((qa_root / "frozen/span_catalog.json").read_text())
        old = json.loads((qa_root / "qa/core_qa.json").read_text())
        raw = {
            "question": old["question"],
            "answer_support_span_ids": old["answer_support_span_ids"],
        }
        result = validate_cross_boundary_core_object(
            raw,
            span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=view["lineage_pair_registry"],
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn(
            "cross_boundary_qa_question_speaker_not_grounded:Classmates B",
            result["reasons"],
        )

        repaired = {
            "question": (
                "What broader shift is reflected when comparing the speaker's "
                "earlier and later remarks?"),
            "answer_support_span_ids": next(
                [parent_id, child_id]
                for row in view["lineage_pair_registry"]["rows"]
                for parent_id in row["parent_span_ids"]
                for child_id in row["child_span_ids"]
                if next(span for span in catalog["spans"]
                        if span["span_id"] == parent_id)["speaker"] ==
                   next(span for span in catalog["spans"]
                        if span["span_id"] == child_id)["speaker"]),
        }
        accepted = validate_cross_boundary_core_object(
            repaired,
            span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=view["lineage_pair_registry"],
            evidence_mismatch_policy="mark_only",
        )
        self.assertEqual("cross_boundary_qa_accept", accepted["decision"])
        mechanical = admit_core_qa(
            accepted["core_qa"], span_catalog=catalog,
            qa_id="pilot21_shadow_repaired")
        self.assertEqual("core_qa_accept", mechanical["decision"])

    def test_pilot21_history_is_unchanged(self):
        self.assertEqual(PILOT21_SHA, hash_tree(PILOT21)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
