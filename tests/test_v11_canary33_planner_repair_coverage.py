from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.growth_profile_v11 import (
    frozen_person_event_growth_profile,
    validate_planner_growth_profile_v11,
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
    "v11_live_canary_33"
)
RUN_SHA = "4140779f9b0bc66c5c9ec4cdb2762d31ecda591e14e278586f947848634132fd"


class Canary33PlannerRepairCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planning = json.loads((RUN / "planning_state_view.json").read_text())
        cls.outline = json.loads((
            RUN / "attempts/planner_v11_attempt_02.json").read_text()
        )["parsed_response"]
        cls.availability = compile_action_availability_view(
            planning_state=cls.planning)
        cls.validation = validate_planner_outline_v11(
            cls.outline,
            source_anchor_registry=json.loads((
                RUN / "source_anchor_registry.json").read_text()),
            placement_choice_registry=json.loads((
                RUN / "placement_choice_registry.json").read_text()),
            authoritative_entity_refs={
                str(row["entity_ref"]) for row in cls.planning["entities"]
            },
            authoritative_entities=cls.planning["entities"],
            action_availability_view=cls.availability,
        )

    def test_history_is_unchanged(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)

    def test_schema_errors_and_growth_error_are_simultaneously_observable(self):
        self.assertIn(
            "action:S02.secondary_action:unknown_action_type:"
            "create_information_asymmetry",
            self.validation["errors"],
        )
        growth = validate_planner_growth_profile_v11(
            self.outline, profile=frozen_person_event_growth_profile())
        self.assertIn("growth_profile_missing_occurred_event", growth["errors"])

    def test_action_repair_prompt_exposes_only_authoritative_connection_pairs(self):
        action_errors = [error for error in self.validation["errors"]
                         if error.startswith("action:")]
        spec = compile_planner_action_repair_spec_v11(
            outline=self.outline, errors=action_errors,
            validation=self.validation,
            action_contracts=action_contract_registry(),
            action_availability_view=self.availability,
        )
        options = spec["authoritative_connection_options"]
        self.assertTrue(options["introduce_connected_person"])
        self.assertTrue(options["introduce_connected_event"])
        self.assertFalse(any(
            str(value).startswith("local_")
            for rows in options.values() for row in rows for value in row.values()
        ))
        prompt = build_planner_action_only_retry_prompt_v11(spec)
        self.assertIn("PROGRAM CONNECTION CHOICES", prompt)
        self.assertIn("connection_choice_id", prompt)
        self.assertNotIn('"connection_target_ref":"local_event_1"', prompt)

    def test_historical_third_response_remains_strictly_invalid(self):
        attempt = json.loads((
            RUN / "planner/attempt_03_validation.json").read_text())
        self.assertIn(
            "planner_action_repair:item:2:replacement.arguments."
            "connection_target_ref:unknown_world_ref:local_event_1",
            attempt["errors"],
        )


if __name__ == "__main__":
    unittest.main()
