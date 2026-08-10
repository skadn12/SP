import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.simple_append_v1.additional_round_live import (
    DEFAULT_API_CONFIG,
    DEFAULT_PRIOR_ROUND_ROOT,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    _highest_round_index,
    preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.contracts import (
    normalize_round_realizer_output,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.multi_round_live import _load


class SimpleAppendAdditionalRoundTests(unittest.TestCase):
    def test_highest_round_is_three(self):
        expanded = _load(DEFAULT_PRIOR_ROUND_ROOT / "expanded_dialogue.json")
        self.assertEqual(3, _highest_round_index(expanded))

    def test_round4_preflight_is_zero_api_and_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "round4"
            report = preflight(
                prior_round_root=DEFAULT_PRIOR_ROUND_ROOT,
                run_root=run_root, round_index=4,
                api_config=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL)
            self.assertEqual("pass", report["decision"], report)
            self.assertEqual(0, report["api_calls"])
            self.assertFalse(run_root.exists())

    def test_non_next_round_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = preflight(
                prior_round_root=DEFAULT_PRIOR_ROUND_ROOT,
                run_root=Path(temporary) / "wrong", round_index=5,
                api_config=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL)
            self.assertEqual("fail", report["decision"])
            self.assertIn("round_index_is_next", report["failed_checks"])

    def test_failed_round4_is_admitted_offline_with_one_cross_scene_warning(self):
        failed_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_1/round_04")
        compiled = _load(failed_root / "planner/compiled_round_plan.json")
        realized = _load(failed_root / "realizer/round_realizer_output.json")
        prior = _load(DEFAULT_PRIOR_ROUND_ROOT / "expanded_dialogue.json")
        replay = normalize_round_realizer_output(
            realized, compiled_plan=compiled,
            parent_utterances=[row["text"] for row in prior["turns"]])
        self.assertTrue(replay["valid"], replay)
        self.assertEqual(49, replay["turn_count"])
        self.assertEqual(1, replay["cross_scene_exact_duplicate_count"])
        self.assertEqual({
            "first_turn_ref": "R04:S02:T015",
            "duplicate_turn_ref": "R04:S03:T005",
        }, replay["cross_scene_exact_duplicates"][0])

    def test_second_failed_round4_is_admitted_with_one_parent_copy_warning(self):
        failed_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_2/round_04")
        compiled = _load(failed_root / "planner/compiled_round_plan.json")
        realized = _load(failed_root / "realizer/round_realizer_output.json")
        prior = _load(DEFAULT_PRIOR_ROUND_ROOT / "expanded_dialogue.json")
        replay = normalize_round_realizer_output(
            realized, compiled_plan=compiled,
            parent_utterances=[row["text"] for row in prior["turns"]],
            parent_turn_refs=[row["turn_ref"] for row in prior["turns"]])
        self.assertTrue(replay["valid"], replay)
        self.assertEqual(48, replay["turn_count"])
        self.assertEqual(1, replay["parent_exact_copy_count"])
        self.assertEqual({
            "duplicate_turn_ref": "R04:S02:T009",
            "source_turn_refs": ["R03:S03:T005"],
        }, replay["parent_exact_copies"][0])


if __name__ == "__main__":
    unittest.main()
