"""Date annotations preserve utterances and never silently change legacy prompts."""
from copy import deepcopy
import unittest

from .sources import render_dialogue
from .repair_trial import expected_schedule, validate_schedule, grouped_metrics


class DatedRenderingTests(unittest.TestCase):
    def fixture(self):
        return {"dialogue_rendering": "conversation_dates_v1",
                "turns": [{"turn_ref": "t1", "speaker": "A", "text": "Tomorrow I will send it."},
                          {"turn_ref": "t2", "speaker": "B", "text": "Yesterday I revised it."}],
                "conversation_date_segments": [{"segment_id": "S01", "conversation_date": "2024-04-08",
                    "first_turn_index_1_based": 1, "last_turn_index_1_based": 2,
                    "first_turn_ref": "t1", "last_turn_ref": "t2"}]}

    def test_exact_header_and_unchanged_turns(self):
        value = self.fixture(); original = deepcopy(value)
        self.assertEqual(render_dialogue(value), "[Conversation date: 2024-04-08 | Session: S01]\n"
            "t1 | A | Tomorrow I will send it.\nt2 | B | Yesterday I revised it.")
        self.assertEqual(value, original)
        value.pop("dialogue_rendering")
        self.assertEqual(render_dialogue(value), "t1 | A | Tomorrow I will send it.\nt2 | B | Yesterday I revised it.")

    def test_incomplete_or_wrong_segment_rejected(self):
        for key, changed in (("first_turn_index_1_based", 2), ("last_turn_index_1_based", 1),
                             ("last_turn_ref", "invented"), ("conversation_date", "2024-02-31")):
            with self.subTest(key=key):
                value = self.fixture(); value["conversation_date_segments"][0][key] = changed
                with self.assertRaises(ValueError): render_dialogue(value)

    def test_schedule_cannot_reorder_drop_or_duplicate_slots(self):
        schedule = expected_schedule()
        validate_schedule(schedule)
        self.assertEqual(len(schedule), 12)
        for changed in (list(reversed(schedule)), schedule[:-1], schedule[:-1] + [schedule[0]]):
            with self.assertRaisesRegex(ValueError, "SCHEDULE_DRIFT"): validate_schedule(changed)

    def test_cost_unknown_and_bad_response_do_not_disappear_from_report(self):
        schedule = expected_schedule()
        slot = schedule[1]
        metrics = grouped_metrics(schedule, [], [{**slot, "raw": {"candidates": 12}}],
            [{**slot, "stage": "primary", "http_attempts": 1, "usage": None}],
            [{**slot, "status": "STOPPED", "admission_summary": {
                "abstained_minimums": {"ordering": 1}}}], logical_calls=[{**slot, "stage": "primary"}])
        row = next(x for x in metrics if x["root_id"] == slot["root_id"] and x["arm_id"] == slot["arm_id"])
        self.assertEqual(row["http_attempts"], 1)
        self.assertEqual(row["legal_ordering_abstention_slots"], 1)
        self.assertEqual(row["usage"]["input_tokens"]["complete_total"], "UNKNOWN")
        self.assertEqual(row["supported_unique_ordering"], "UNKNOWN")
        self.assertEqual(row["cost_per_supported_question"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
