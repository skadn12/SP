import json
import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_canary_7
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    implementation_manifest_v11,
)


class LiveCanary7FreezeTests(unittest.TestCase):
    def test_unique_run_root_identity(self):
        self.assertEqual(
            live_canary_7.RUN_ROOT.name,
            "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_7",
        )

    def test_history_baselines_are_current(self):
        self.assertTrue(live_canary_7._history_valid({
            "historical_manifests": live_canary_7.historical_manifests_canary_7()
        }))

    def test_runner_and_recovery_are_in_implementation_manifest(self):
        manifest = implementation_manifest_v11()
        self.assertTrue(manifest["valid"])
        paths = {row["relative_path"] for row in manifest["files"]}
        for relative in (
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/live_canary_7.py",
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_validator_contract_repair_v11.py",
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/round_validator_transport_v11.py",
        ):
            self.assertIn(relative, paths)

    def test_existing_run_if_any_has_exact_identity(self):
        if live_canary_7.RUN_ROOT.exists():
            freeze = json.loads((live_canary_7.RUN_ROOT /
                                 "run_freeze_manifest.json").read_text())
            self.assertEqual(freeze["run_root"], str(live_canary_7.RUN_ROOT))


if __name__ == "__main__":
    unittest.main()
