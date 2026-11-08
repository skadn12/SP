"""Offline input projection tests; no backend or production ledger is constructed."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .input_views import project_supplement_inputs, reconstruct_supplement_inputs

OLD_REQUEST = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_incremental_plan_retention_v1_offline_preflight_live_4/"
    "offline_failed_plan_path/requests/001_supplement.json")


def historical_inputs():
    request = json.loads(OLD_REQUEST.read_text())
    return json.loads(request["messages"][0]["content"].split("\n\nINPUT DATA\n", 1)[1])


class InputProjectionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = historical_inputs()

    def test_all_original_fields_and_turns_reconstruct_exactly(self):
        before = deepcopy(self.inputs)
        projected, mapping = project_supplement_inputs(self.inputs)
        self.assertEqual(self.inputs, before)
        self.assertEqual(reconstruct_supplement_inputs(projected, mapping), before)
        self.assertTrue(mapping["reconstruction_exact"])
        self.assertEqual(mapping["source_sha256"], mapping["reconstruction_sha256"])
        self.assertEqual(mapping["original_turn_count"], 143)
        self.assertEqual(mapping["original_candidate_count"], 11)

    def test_original_plan_remains_exact_single_body(self):
        projected, mapping = project_supplement_inputs(self.inputs)
        self.assertEqual(projected["original_plan"], self.inputs["original_plan"])
        self.assertEqual(len(projected["original_plan"]["selected_continuations"]), 3)
        self.assertEqual(len(projected["original_plan"]["new_development_candidates"]), 8)
        encoded = json.dumps(projected, sort_keys=True, ensure_ascii=False)
        self.assertEqual(encoded.count(json.dumps(projected["original_plan"], sort_keys=True, ensure_ascii=False)), 1)
        self.assertGreater(len(mapping["reference_replacements"]), 0)
        self.assertEqual(projected["retained_actual_r03_dialogue"], [])

    def test_registry_unique_bindings_and_state_differences_survive(self):
        candidate = next(iter(self.inputs["stable_candidates"]))
        original = self.inputs["stable_candidates"][candidate]
        original["scene"]["before_state"] = "UNIQUE fixture state not shared with original plan or Gate."
        original["scene"]["fixture_dependency"] = {"phase_id": "initial:unique", "condition": "A prerequisite not yet satisfied."}
        original["gate_scene"]["fixture_binding"] = {"actor": "James", "action": "Distinct Gate binding."}
        projected, mapping = project_supplement_inputs(self.inputs)
        restored = reconstruct_supplement_inputs(projected, mapping)
        self.assertEqual(restored["stable_candidates"][candidate], original)
        self.assertIn("UNIQUE fixture state", json.dumps(projected))
        self.assertIn("A prerequisite not yet satisfied.", json.dumps(projected))

    def test_stable_plan_association_uses_track_or_compiler_nd_identity(self):
        projected, _ = project_supplement_inputs(self.inputs)
        for candidate, row in projected["stable_candidates"].items():
            association = row["original_plan_input_pointer"]
            if candidate.startswith("ND"):
                self.assertEqual(association, "/original_plan/new_development_candidates/" + str(int(candidate[2:4]) - 1))
            else:
                index = int(association.rsplit("/", 1)[1])
                track = self.inputs["stable_candidates"][candidate]["gate_scene"]["verified_track_ref"]
                self.assertEqual(self.inputs["original_plan"]["selected_continuations"][index]["verified_track_ref"], track)

    def test_every_history_string_order_and_extra_fact_metadata_survives(self):
        turn = self.inputs["authoritative_history"][4]
        turn.update(scene_ref="R02:S01", actual_time="Tuesday 13:00", unknown_future_field={"quoted_fact": "Retain this verbatim."})
        turn["text"] += "\nA second line containing [brackets] and {JSON-like wording}."
        projected, mapping = project_supplement_inputs(self.inputs)
        restored = reconstruct_supplement_inputs(projected, mapping)
        self.assertEqual(restored["authoritative_history"], self.inputs["authoritative_history"])
        model_values = [value for group in projected["authoritative_history"]["groups"] for row in group["turns"]
                        for column, value in zip(group["turn_columns"], row) if column == "text"]
        self.assertEqual(model_values, [row["text"] for row in self.inputs["authoritative_history"]])
        self.assertIn("Tuesday 13:00", json.dumps(projected))

    def test_actual_history_duplicate_lines_are_not_deduplicated(self):
        self.inputs["authoritative_history"][1]["text"] = self.inputs["authoritative_history"][0]["text"]
        projected, _ = project_supplement_inputs(self.inputs)
        lines = [row[group["turn_columns"].index("text")] for group in projected["authoritative_history"]["groups"] for row in group["turns"]]
        self.assertEqual(lines[0], lines[1])
        self.assertEqual(len(lines), 143)

    def test_backend_paths_stay_out_of_model_but_can_be_reconstructed(self):
        self.inputs["frozen_sources"] = {"history": {"path": "/BACKEND_ONLY_TEST/input.json", "sha256": "a" * 64}}
        projected, mapping = project_supplement_inputs(self.inputs)
        self.assertNotIn("BACKEND_ONLY_TEST", json.dumps(projected))
        self.assertEqual(reconstruct_supplement_inputs(projected, mapping), self.inputs)

    def test_feedback_identity_and_observation_are_untouched(self):
        self.inputs["shared_feedback_view"] = {"root_id": "fixture", "candidate_id": "fixture:source",
            "historical_qa": {"items": ["a", "b", "c", "d"], "canonical_gold": ["a", "b", "c", "d"]},
            "observations": [{"accuracy": .5, "sample_count": "UNKNOWN"}]}
        projected, _ = project_supplement_inputs(self.inputs)
        self.assertEqual(projected["shared_feedback_view"], self.inputs["shared_feedback_view"])

    def test_deterministic_projection_and_explicit_reserved_field_rejection(self):
        self.assertEqual(project_supplement_inputs(self.inputs), project_supplement_inputs(self.inputs))
        self.inputs["original_plan"]["injected"] = {"$input_ref": "/untrusted"}
        with self.assertRaisesRegex(ValueError, "reserved_reference"):
            project_supplement_inputs(self.inputs)

    def test_mismatched_plan_track_is_not_silently_guessed(self):
        candidate = next(key for key in self.inputs["stable_candidates"] if key.startswith("ME"))
        self.inputs["stable_candidates"][candidate]["gate_scene"]["verified_track_ref"] = "VT_UNKNOWN"
        with self.assertRaisesRegex(ValueError, "track_association"):
            project_supplement_inputs(self.inputs)


if __name__ == "__main__":
    unittest.main()
