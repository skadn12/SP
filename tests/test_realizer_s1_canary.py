from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from scripts.current_method.restart_n5_spell.realizer_s1_canary import (
    DEFAULT_SOURCE_RUN, audit_historical_s1_failures, load_frozen_s1_source,
)
from scripts.current_method.restart_n5_spell.stateful_graph_orchestrator import (
    build_stateful_realizer_retry_suffix,
    stateful_realizer_retry_error_feedback,
)


TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


@unittest.skipUnless(DEFAULT_SOURCE_RUN.is_dir(), "authoritative source run absent")
class RealizerS1CanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_frozen_s1_source(DEFAULT_SOURCE_RUN)
        cls.contract = cls.source["s1_contract"]
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER, local_files_only=True)
        cls.raws = []
        for attempt in range(1, 4):
            cls.raws.append(json.loads((DEFAULT_SOURCE_RUN / (
                f"realizer/raw/stateful-realizer-s01-a{attempt:02d}.json"
            )).read_text(encoding="utf-8")))

    def test_frozen_source_validates_and_has_expected_shape(self) -> None:
        self.assertTrue(self.source["graph_validation"]["valid"])
        self.assertEqual(sorted(self.source["arc_canonical_sha256"]),
                         ["E1", "E2", "E3", "E4"])
        self.assertEqual(self.contract["scene_id"], "S1")
        self.assertEqual(len(self.source["parent_turns"]), 60)

    def test_length_feedback_has_actual_and_minimum(self) -> None:
        parsed = json.loads(self.raws[0]["raw_response"])
        feedback = stateful_realizer_retry_error_feedback(
            contract=self.contract,
            errors=["stateful_scene_token_count_out_of_range"],
            parsed=parsed, actual_tokens=162, actual_turns=7)
        self.assertEqual(feedback, [{
            "code": "stateful_scene_token_count_out_of_range",
            "actual_tokens": 162, "minimum_tokens": 256,
            "maximum_tokens": 576,
        }])

    def test_retry_contains_speaker_and_terminal_marker_contract(self) -> None:
        parsed = json.loads(self.raws[2]["raw_response"])
        suffix = build_stateful_realizer_retry_suffix(
            contract=self.contract,
            errors=["stage_evidence_speaker_mismatch",
                    "terminal_evidence_marker_missing"],
            parsed=parsed, actual_tokens=239, actual_turns=7)
        self.assertIn("Co-workers A", suffix)
        self.assertIn("I paid back the $100 I owed to Friend C.", suffix)
        self.assertIn("actual_evidence_speaker", suffix)
        self.assertIn("Friend C", suffix)
        self.assertIn("Each assigned stage has its own evidence turn", suffix)
        self.assertIn("do not need to be in the final dialogue turn", suffix)
        self.assertNotIn(
            "final turn must contain the exact terminal marker", suffix)

    def test_length_only_retry_preserves_event_and_evidence(self) -> None:
        parsed = json.loads(self.raws[0]["raw_response"])
        suffix = build_stateful_realizer_retry_suffix(
            contract=self.contract,
            errors=["stateful_scene_token_count_out_of_range"],
            parsed=parsed, actual_tokens=162, actual_turns=7)
        self.assertIn("do not rewrite the event or evidence", suffix)
        self.assertIn("Preserve the authoritative participants", suffix)
        self.assertNotIn("change the scene event", suffix)

    def test_speaker_error_is_explicit(self) -> None:
        parsed = json.loads(self.raws[2]["raw_response"])
        feedback = stateful_realizer_retry_error_feedback(
            contract=self.contract,
            errors=["stage_evidence_speaker_mismatch"], parsed=parsed,
            actual_tokens=239, actual_turns=7)
        self.assertEqual(feedback[0]["actual_evidence_speaker"], "Friend C")
        self.assertEqual(feedback[0]["required_speakers"], ["Co-workers A"])

    def test_terminal_marker_error_is_explicit(self) -> None:
        parsed = json.loads(self.raws[2]["raw_response"])
        feedback = stateful_realizer_retry_error_feedback(
            contract=self.contract,
            errors=["terminal_evidence_marker_missing"], parsed=parsed,
            actual_tokens=239, actual_turns=7)
        self.assertEqual(feedback[0]["required_terminal_markers"],
                         ["I paid back the $100 I owed to Friend C."])

    def test_multiple_errors_are_all_reported_once(self) -> None:
        parsed = json.loads(self.raws[2]["raw_response"])
        suffix = build_stateful_realizer_retry_suffix(
            contract=self.contract,
            errors=["stage_evidence_speaker_mismatch",
                    "stateful_scene_token_count_out_of_range",
                    "terminal_evidence_marker_missing"],
            parsed=parsed, actual_tokens=239, actual_turns=7)
        self.assertIn("stage_evidence_speaker_mismatch", suffix)
        self.assertIn("stateful_scene_token_count_out_of_range", suffix)
        self.assertIn("terminal_evidence_marker_missing", suffix)
        self.assertIn('"actual_tokens": 239', suffix)
        self.assertIn('"minimum_tokens": 256', suffix)

    def test_retry_builder_does_not_mutate_or_patch_output(self) -> None:
        parsed = json.loads(self.raws[2]["raw_response"])
        before = copy.deepcopy(parsed)
        suffix = build_stateful_realizer_retry_suffix(
            contract=self.contract,
            errors=["terminal_evidence_marker_missing"], parsed=parsed,
            actual_tokens=239, actual_turns=7)
        self.assertEqual(parsed, before)
        self.assertIsInstance(suffix, str)
        # The prior response is diagnostic context only; the function returns no
        # candidate object and therefore cannot append a marker or stitch raws.
        self.assertNotIsInstance(suffix, dict)

    def test_historical_three_raw_regression(self) -> None:
        audit = audit_historical_s1_failures(
            source_run=DEFAULT_SOURCE_RUN, contract=self.contract,
            tokenizer=self.tokenizer)
        self.assertEqual([row["actual_dialogue_tokens"]
                          for row in audit["attempts"]], [162, 235, 239])
        self.assertEqual([row["actual_turn_count"]
                          for row in audit["attempts"]], [7, 7, 7])
        self.assertEqual([row["finish_reason"]
                          for row in audit["attempts"]],
                         ["stop", "stop", "stop"])
        self.assertTrue(audit["attempts"][0]["evidence_speaker_correct"])
        self.assertTrue(audit["attempts"][1]["terminal_marker_present"])
        self.assertFalse(audit["attempts"][2]["evidence_speaker_correct"])
        self.assertFalse(audit["attempts"][2]["terminal_marker_present"])


if __name__ == "__main__":
    unittest.main()
