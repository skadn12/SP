from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view,
    validate_action_against_availability,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_action_repair_v11 import (
    build_planner_action_only_retry_prompt_v11,
    compile_planner_action_repair_spec_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_32"
)
RUN_SHA = "a92387aa0840f78137364f5110b2f329ba8814823b8e6131e5939326e2410128"


class Canary32PlannerActionReferenceAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planning = json.loads((RUN / "planning_state_view.json").read_text())
        cls.outline = json.loads((
            RUN / "attempts/planner_v11_attempt_02.json").read_text()
        )["parsed_response"]
        cls.anchors = json.loads((RUN / "source_anchor_registry.json").read_text())
        cls.placements = json.loads((RUN / "placement_choice_registry.json").read_text())
        cls.availability = compile_action_availability_view(
            planning_state=cls.planning)
        cls.validation = validate_planner_outline_v11(
            cls.outline,
            source_anchor_registry=cls.anchors,
            placement_choice_registry=cls.placements,
            authoritative_entity_refs={
                str(row["entity_ref"]) for row in cls.planning["entities"]
            },
            authoritative_entities=cls.planning["entities"],
            action_availability_view=cls.availability,
        )

    def test_history_is_read_only(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)

    def test_full_outline_aggregates_continuation_and_connection_errors(self):
        errors = self.validation["errors"]
        self.assertIn(
            "action:S02.primary_action:continuation_slot_format:s1.primary_action",
            errors,
        )
        self.assertIn(
            "action:S04.secondary_action.arguments.connected_existing.ref:"
            "unknown_world_ref:local_1",
            errors,
        )
        self.assertTrue(any(
            error.startswith(
                "action:S04.secondary_action.arguments.connected_existing.ref:"
                "unknown_typed_ref:participant_refs:local_1"
            ) for error in errors
        ), errors)

    def test_round_alias_can_participate_but_cannot_be_existing_world_edge(self):
        action = self.outline["scenes"][3]["secondary_action"]
        errors = validate_action_against_availability(
            action, path="action:S04.secondary_action",
            availability_view=self.availability,
            additional_participant_refs={"local_1"},
        )
        self.assertFalse(any("participant_refs" in error and error.endswith(
            ":local_1") and "connected_existing" not in error for error in errors), errors)
        self.assertIn(
            "action:S04.secondary_action.arguments.connected_existing.ref:"
            "unknown_typed_ref:participant_refs:local_1",
            errors,
        )

    def test_action_repair_targets_all_observable_bad_slots(self):
        spec = compile_planner_action_repair_spec_v11(
            outline=self.outline, errors=self.validation["errors"],
            validation=self.validation,
            action_contracts=action_contract_registry(),
            action_availability_view=self.availability,
        )
        self.assertIn("S04.secondary_action", spec["target_action_refs"])
        self.assertIn("S02.primary_action", spec["target_action_refs"])
        prompt = build_planner_action_only_retry_prompt_v11(spec)
        self.assertIn(
            "Connected-person and connected-event repairs are the exception",
            prompt,
        )
        self.assertIn("connection_choice_id", prompt)
        self.assertNotIn(
            "Participant-valued arguments must copy exact authoritative refs or an "
            "allowed round participant alias",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
