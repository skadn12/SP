import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_4
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    implementation_manifest_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY_4_SHA = "a07b0f9a09968141f0a72250fe2fb516227783fa495c78440ef6fbc6f9649c70"


class LiveCanary4FreezeTests(unittest.TestCase):
    def test_completed_unique_run_root_is_read_only_baseline(self):
        self.assertEqual(
            live_canary_4.RUN_ROOT.name,
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_4",
        )
        self.assertTrue(live_canary_4.RUN_ROOT.is_dir())
        self.assertEqual(hash_tree(live_canary_4.RUN_ROOT)["tree_sha256"], CANARY_4_SHA)

    def test_history_baselines_are_current(self):
        self.assertTrue(live_canary_4._history_valid({
            "historical_manifests": live_canary_4.historical_manifests_canary_4()
        }))

    def test_runner_is_in_implementation_manifest(self):
        manifest = implementation_manifest_v11()
        self.assertTrue(manifest["valid"])
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_4.py",
            {row["relative_path"] for row in manifest["files"]},
        )


if __name__ == "__main__":
    unittest.main()
