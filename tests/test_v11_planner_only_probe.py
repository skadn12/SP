import ast
import hashlib
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    FAILED_PROMPT_SHA, build_planner_probe_input,
)


class PlannerOnlyProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = build_planner_probe_input(DEFAULT_INPUT)

    def test_historical_probe_keeps_the_byte_identical_failed_prompt(self):
        historical = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_"
            "planner_transport_probe_1/planner_prompt.txt"
        )
        self.assertEqual(
            hashlib.sha256(historical.read_bytes()).hexdigest(), FAILED_PROMPT_SHA
        )

    def test_new_contract_intentionally_invalidates_one_shot_probe_preflight(self):
        self.assertNotEqual(self.values["prompt_sha256"], FAILED_PROMPT_SHA)

    def test_original_parent_only(self):
        complete = self.values["complete_dialogue"]
        self.assertEqual(complete["turn_count"], 60)
        self.assertEqual(complete["rounds"], [])
        self.assertTrue(all(row["source_partition"] == "original_parent"
                            for row in complete["turns"]))

    def test_probe_source_has_no_downstream_runtime_import(self):
        path = Path(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/"
            "planner_only_probe_v11.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        forbidden = ("realizer", "round_expansion_validator", "evidence_scope",
                     "binder", "questioner", "solver", "training")
        self.assertFalse(any(any(value in module for value in forbidden)
                             for module in modules))


if __name__ == "__main__":
    unittest.main()
