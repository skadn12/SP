import json
from pathlib import Path
import unittest

from duplicate_check import inspect_duplicates


def turn(index, *, text=None, speaker="Emily", prefix="history"):
    return {
        "speaker": speaker,
        "text": text or f"For event number {index}, the reservation is confirmed for Friday afternoon at five.",
        "turn_ref": f"{prefix}:T{index:03d}",
    }


class DuplicateCheckTests(unittest.TestCase):
    def test_historical_103866_nine_turn_copy_rejected(self):
        base = Path(__file__).resolve().parent.parent / "h4_r02_source_unique_v2/runs"
        a = json.loads((base / "recovery_01/roots/cc-train-episode-103866/A/S01.json").read_text())["turns"]
        b = json.loads((base / "recovery_b_budget_01/roots/cc-train-episode-103866/B/S01.json").read_text())["turns"]
        # Existing artifacts are only read. Attach canonical references in memory.
        a = [dict(t, turn_ref=f"R02:A:S01:T{i:03d}") for i, t in enumerate(a, 1)]
        b = [dict(t, turn_ref=f"R02:B:S01:T{i:03d}") for i, t in enumerate(b, 1)]
        report = inspect_duplicates(b, {"A": a})
        self.assertEqual(report["status"], "REJECT")
        self.assertEqual(report["duplicate_turn_count"], 9)
        self.assertEqual(report["duplicate_fraction"], 9 / 16)
        self.assertEqual(report["max_consecutive_span_length"], 9)
        span = max(report["consecutive_spans"], key=lambda s: s["length"])
        self.assertEqual((span["new_start_ref"], span["new_end_ref"]),
                         ("R02:B:S01:T008", "R02:B:S01:T016"))
        self.assertEqual(span["source_start_ref"], "R02:A:S01:T008")

    def test_changing_ids_does_not_hide_copying(self):
        reference = [turn(i) for i in range(1, 4)]
        new = [dict(t, turn_ref=f"R02:B:T{i}") for i, t in enumerate(reference)]
        report = inspect_duplicates(new, {"A": reference})
        self.assertEqual(report["status"], "REJECT")
        self.assertEqual(report["duplicate_turn_count"], 3)

    def test_short_greetings_are_diagnostic_only(self):
        short = [turn(i, text=text) for i, text in enumerate(["Hi!", "Hello.", "Thanks!", "Bye!"])]
        report = inspect_duplicates(short, {"history": short})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["duplicate_turn_count"], 4)
        self.assertEqual(report["repeated_substantive_turn_count"], 0)

    def test_two_substantive_matches_do_not_reject_scene(self):
        history = [turn(i) for i in range(1, 3)]
        self.assertEqual(inspect_duplicates(history, {"history": history})["status"], "PASS")

    def test_nonconsecutive_half_of_substantive_turns_rejected(self):
        history = [turn(i) for i in range(1, 5)]
        new = []
        for i, repeated in enumerate(history):
            new.extend([repeated, turn(i + 10)])
        report = inspect_duplicates(new, {"history": history})
        self.assertEqual(report["status"], "REJECT")
        self.assertEqual(report["substantive_duplicate_fraction"], 0.5)
        self.assertEqual(report["max_consecutive_span_length"], 1)
        self.assertEqual(report["reasons"], ["SUBSTANTIVE_DUPLICATE_FRACTION_EXCEEDED"])

    def test_normal_new_scene_passes(self):
        history = [turn(i) for i in range(1, 8)]
        report = inspect_duplicates([turn(i) for i in range(10, 26)], {"history": history})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["matches"], [])

    def test_whitespace_normalization(self):
        history = [turn(i) for i in range(1, 4)]
        new = [dict(t, speaker="  Emily \n", text="\n " + t["text"].replace(" ", "\t  ") + "  ") for t in history]
        report = inspect_duplicates(new, {"history": history})
        self.assertEqual(report["status"], "REJECT")
        self.assertEqual(report["duplicate_turn_count"], 3)

    def test_multiple_source_matches_count_new_turn_once(self):
        history = [turn(i) for i in range(1, 4)]
        report = inspect_duplicates(history, {"history": history, "A": history})
        self.assertEqual(report["duplicate_turn_count"], 3)
        self.assertEqual(report["substantive_duplicate_fraction"], 1.0)
        self.assertEqual(len(report["matches"][0]["sources"]), 2)

    def test_thresholds_are_configurable_and_recorded(self):
        history = [turn(i) for i in range(1, 4)]
        report = inspect_duplicates(history, {"history": history}, min_consecutive=4)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["policy"]["min_consecutive"], 4)

    def test_speakers_are_part_of_exact_identity(self):
        history = [turn(i) for i in range(1, 4)]
        new = [dict(t, speaker="James") for t in history]
        self.assertEqual(inspect_duplicates(new, {"history": history})["duplicate_turn_count"], 0)

    def test_invalid_input_does_not_silently_pass(self):
        for data in ([{"speaker": "Emily"}], [{"speaker": "", "text": "hello"}], "not-list"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                inspect_duplicates(data, {})
        for threshold in (0, -1, 2, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                inspect_duplicates([], {}, max_duplicate_fraction=threshold)

    def test_empty_reference_has_no_false_match(self):
        report = inspect_duplicates([turn(1)], {})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["duplicate_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
