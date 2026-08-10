import json
import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_14
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import implementation_manifest_v11


class LiveCanary14FreezeTests(unittest.TestCase):
    def test_unique_run_root_identity(self):
        self.assertEqual(live_canary_14.RUN_ROOT.name,
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_14")

    def test_history_baselines_are_current(self):
        self.assertTrue(live_canary_14._history_valid({
            "historical_manifests": live_canary_14.historical_manifests_canary_14()}))

    def test_runner_is_in_implementation_manifest(self):
        manifest = implementation_manifest_v11()
        self.assertTrue(manifest["valid"])
        self.assertIn(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_14.py",
            {row["relative_path"] for row in manifest["files"]})

    def test_existing_run_if_any_has_exact_identity(self):
        if live_canary_14.RUN_ROOT.exists():
            freeze = json.loads((live_canary_14.RUN_ROOT /
                                 "run_freeze_manifest.json").read_text())
            self.assertEqual(freeze["run_root"], str(live_canary_14.RUN_ROOT))


if __name__ == "__main__":
    unittest.main()
