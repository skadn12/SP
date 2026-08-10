import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 import (
    _round_turn_budget,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11, normalize_realizer_dialogue_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_role_runtime_v11 import (
    ALLOWED_LIVE_ROLES_V11,
)


class LiveB2ContractTests(unittest.TestCase):
    def test_roles_exclude_committer_and_extractor(self):
        self.assertEqual(ALLOWED_LIVE_ROLES_V11, {
            "planner_v11", "realizer_v11", "round_expansion_validator_v1"})
        self.assertFalse(any("committer" in value for value in ALLOWED_LIVE_ROLES_V11))
        self.assertFalse(any("extractor" in value for value in ALLOWED_LIVE_ROLES_V11))

    def test_planner_prompt_uses_two_real_entities_and_long_budget(self):
        planning = {"entities": [
            {"entity_ref": "root:E1", "display_name": "A"},
            {"entity_ref": "root:E2", "display_name": "B"},
        ]}
        anchors = {"root_id": "root", "anchors": [{
            "anchor_id": "PA0001", "anchor_type": "tail_anchor",
            "source_kind": "historical_dialogue", "available_placements": ["after"],
            "boundary": {"after_turn_ref": "root:T1", "before_turn_ref": "root:T1"},
            "linked_entity_refs": ["root:E1", "root:E2"],
            "exact_excerpt": [{"turn_ref": "root:T1", "text": "x"}],
        }]}
        source_anchors = {"root_id": "root", "source_anchors": [{
            "source_anchor_id": "PA0001", "anchor_type": "tail_anchor",
            "source_kind": "historical_dialogue", "source_turn_refs": ["root:T1"],
            "linked_entity_refs": ["root:E1", "root:E2"],
            "allowed_link_types": ["direct_continuation"],
            "exact_excerpt": [{"turn_ref": "root:T1", "text": "x"}],
        }]}
        placement_choices = {"root_id": "root", "choices": [{
            "placement_choice_id": "PC_TAIL_AFTER", "placement": "after",
            "after_turn_ref": "root:T1", "before_turn_ref": None,
            "allowed_time_semantics": ["current_tail_continuation"],
            "knowledge_boundary": {}, "narrative_order_key": "00000000.250",
            "preceding_known_turn_refs": ["root:T1"],
            "following_constraint_turn_refs": [],
        }]}
        prompt, frozen = build_planner_prompt_v11(
            root_id="root", planning_state=planning,
            source_anchor_registry=source_anchors,
            placement_choice_registry=placement_choices,
            planner_action_view={"actions": []})
        self.assertIn('"minimum":5', prompt)
        self.assertIn('"minimum":30', prompt)
        self.assertIn("root:E1", prompt)
        self.assertIn("root:E2", prompt)
        self.assertNotIn("EXISTING_ENTITY_REF", prompt)
        self.assertEqual(len(frozen), 64)

    def test_realizer_refs_are_program_owned(self):
        result = normalize_realizer_dialogue_v11(
            {"dialogue_turns": [
                {"speaker_ref": "root:E1", "text": "one"},
                {"speaker_ref": "root:E2", "text": "two"},
            ]}, scene_ref="R01:S01", allowed_speaker_refs={"root:E1", "root:E2"})
        self.assertTrue(result["valid"])
        self.assertEqual([row["turn_ref"] for row in result["scene_output"]["dialogue_turns"]],
                         ["R01:S01:T001", "R01:S01:T002"])
        invalid = normalize_realizer_dialogue_v11(
            {"dialogue_turns": [{"turn_ref": "R01:S01:T999",
                                  "speaker_ref": "root:E1", "text": "x"}]},
            scene_ref="R01:S01", allowed_speaker_refs={"root:E1"})
        self.assertFalse(invalid["valid"])

    def test_round_turn_allocation_keeps_target_in_30_45_with_scene_slack(self):
        scene = {"speaker_participants": [{}, {}], "interaction_beats": ["a", "b", "c"],
                 "compiled_proposals": []}
        for count in (5, 6, 7):
            budget = _round_turn_budget(scene, count)
            self.assertGreaterEqual(count * budget["target"], 30)
            self.assertLessEqual(count * budget["target"], 45)
            self.assertLess(budget["minimum"], budget["preferred_minimum"])
            self.assertGreater(budget["maximum"], budget["preferred_maximum"])
            self.assertEqual(
                budget["single_scene_limit_policy"], "flexible_safety_bounds")


if __name__ == "__main__":
    unittest.main()
