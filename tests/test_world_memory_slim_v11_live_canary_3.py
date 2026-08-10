import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_3
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    implementation_manifest_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_3_SHA = "974035b8ddb29f8fdc5aad9c8edeffb7109a4d81582d40cec702162eb593cf38"


class LiveCanary3FreezeTests(unittest.TestCase):
    def test_completed_canary_root_is_read_only_baseline(self):
        self.assertEqual(
            live_canary_3.RUN_ROOT.name,
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_3",
        )
        self.assertTrue(live_canary_3.RUN_ROOT.is_dir())
        self.assertEqual(hash_tree(live_canary_3.RUN_ROOT)["tree_sha256"],
                         CANARY_3_SHA)

    def test_history_baselines_are_current(self):
        self.assertTrue(live_canary_3._history_valid({
            "historical_manifests": live_canary_3.historical_manifests_canary_3()
        }))

    def test_runner_is_in_implementation_manifest(self):
        manifest = implementation_manifest_v11()
        self.assertTrue(manifest["valid"])
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_3.py",
            {row["relative_path"] for row in manifest["files"]},
        )


if __name__ == "__main__":
    unittest.main()
