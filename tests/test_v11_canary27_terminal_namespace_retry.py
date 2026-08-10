import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_v11
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    action_contract_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_namespace_repair_v11 import (
    apply_planner_namespace_repairs_v11,
    compile_planner_namespace_repair_spec_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY27 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_27"
)
EXPECTED_SHA = "b203b85e1b93443ccf77a076f25396f87f63211b2f9c2a573a6d6b1d819f6b07"


class Canary27TerminalNamespaceRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attempt = json.loads((
            CANARY27 / "attempts/planner_v11_attempt_03.json"
        ).read_text(encoding="utf-8"))
        cls.audit = json.loads((
            CANARY27 / "planner/attempt_03_validation.json"
        ).read_text(encoding="utf-8"))
        cls.availability = json.loads((
            CANARY27 / "action_availability_view.json"
        ).read_text(encoding="utf-8"))

    def test_history_is_read_only(self):
        self.assertEqual(hash_tree(CANARY27)["tree_sha256"], EXPECTED_SHA)

    def test_terminal_retry_is_narrow_and_bounded(self):
        self.assertEqual(live_canary_v11.PLANNER_BASE_MAX_ATTEMPTS, 3)
        self.assertEqual(
            live_canary_v11.PLANNER_TERMINAL_NAMESPACE_REPAIR_ATTEMPTS, 1)
        self.assertEqual(live_canary_v11.PLANNER_ABSOLUTE_MAX_ATTEMPTS, 4)
        self.assertTrue(live_canary_v11._terminal_planner_namespace_retry_eligible(
            attempt=3, request_mode="full_outline",
            outline=self.attempt["parsed_response"],
            errors=self.audit["errors"], parse_errors=[]))
        self.assertFalse(live_canary_v11._terminal_planner_namespace_retry_eligible(
            attempt=2, request_mode="full_outline",
            outline=self.attempt["parsed_response"],
            errors=self.audit["errors"], parse_errors=[]))
        self.assertFalse(live_canary_v11._terminal_planner_namespace_retry_eligible(
            attempt=3, request_mode="action_only",
            outline=self.attempt["parsed_response"],
            errors=self.audit["errors"], parse_errors=[]))

    def test_choice_id_repair_preserves_frozen_outline(self):
        spec = compile_planner_namespace_repair_spec_v11(
            outline=self.attempt["parsed_response"], errors=self.audit["errors"],
            validation=self.audit["validation"],
            action_contracts=action_contract_registry(),
            action_availability_view=self.availability)
        result = apply_planner_namespace_repairs_v11(
            frozen_outline=self.attempt["parsed_response"], spec=spec,
            response={
                "action_repairs": [],
                "link_type_repairs": [{
                    "scene_ref": "S07", "replacement_choice_id": "LT13",
                }],
            })
        self.assertTrue(result["valid"], result["errors"])
        repaired = result["reconstructed_outline"]
        self.assertEqual(repaired["scenes"][6]["link_type"], "relationship_followup")
        before = json.loads(json.dumps(self.attempt["parsed_response"]))
        before["scenes"][6]["link_type"] = "relationship_followup"
        self.assertEqual(repaired, before)


if __name__ == "__main__":
    unittest.main()
