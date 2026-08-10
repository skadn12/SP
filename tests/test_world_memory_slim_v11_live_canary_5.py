import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_5
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    implementation_manifest_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_5_TREE_SHA = (
    "6432259bad24a08bd32d03dccdca2116e236861012c9b3b26d497294e5798c12"
)


class LiveCanary5FreezeTests(unittest.TestCase):
    def test_unique_fresh_run_root_identity(self):
        self.assertEqual(
            live_canary_5.RUN_ROOT.name,
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_5",
        )

    def test_completed_failed_run_is_immutable(self):
        self.assertTrue(live_canary_5.RUN_ROOT.is_dir())
        self.assertEqual(hash_tree(live_canary_5.RUN_ROOT)["tree_sha256"],
                         CANARY_5_TREE_SHA)

    def test_history_baselines_are_current(self):
        self.assertTrue(live_canary_5._history_valid({
            "historical_manifests": live_canary_5.historical_manifests_canary_5()
        }))

    def test_runner_and_action_repair_are_in_implementation_manifest(self):
        manifest = implementation_manifest_v11()
        self.assertTrue(manifest["valid"])
        paths = {row["relative_path"] for row in manifest["files"]}
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_5.py",
            paths,
        )
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/"
            "planner_action_repair_v11.py",
            paths,
        )
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/"
            "round_validator_contract_repair_v11.py",
            paths,
        )
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/"
            "round_validator_transport_v11.py",
            paths,
        )


if __name__ == "__main__":
    unittest.main()
