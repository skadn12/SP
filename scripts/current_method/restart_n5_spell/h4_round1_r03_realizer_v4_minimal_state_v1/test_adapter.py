import json
import tempfile
import unittest
from pathlib import Path

from .adapter import (build_provenance, freeze_manifest, leakage_scan,
                      minimal_current_state, project_scene, render_prompt,
                      structured_actual_progress)


class AdapterTests(unittest.TestCase):
    def scene(self):
        return {
            "selected_candidate_ref": "ME02:C01", "verified_track_ref": "VT02",
            "memory_event_key": "attending_religious_cultural_workshop",
            "memory_event_display": "attending religious cultural workshop",
            "goal": "Attend the workshop and discuss the content afterward.",
            "required_transition": "The participants attend and discuss what they learned.",
            "actor_bindings": [{"actor": "Taylor", "binding": "attends"}],
            "required_participants": ["Taylor", "Casey"],
            "interaction_beats": ["Arrive at the workshop.", "Join the interactive session.", "Discuss afterward."],
            "dependencies": [],
            "before_state": "Taylor registered both names for a workshop next Saturday.",
            "memory_event_content": ["Taylor registered both names for a workshop next Saturday."],
            "supporting_sources": [{"contents": ["Taylor registered both names for a workshop next Saturday."]}],
            "verified_current_state": "Taylor registered both names for a workshop next Saturday.",
        }

    def test_redacted_fields_are_absent_from_payload_and_provenance_keeps_them(self):
        scene = self.scene(); visible = project_scene(scene)
        encoded = json.dumps(visible, ensure_ascii=False)
        for field in ("before_state", "memory_event_content", "supporting_sources", "verified_current_state", "retained_actual_prefix"):
            self.assertNotIn(field, visible)
        self.assertNotIn("registered both names", encoded)
        provenance = build_provenance(scene, prior_dialogue=[{"text": "old"}])
        self.assertIn("before_state", provenance["original_scene"])
        self.assertTrue(provenance["excluded_content_sha256"])

    def test_status_and_provenance_are_explicit(self):
        progress = structured_actual_progress(
            completed_actions=[{"fact": "The meeting happened", "provenance": "R03:S01:T08"}],
            unresolved_targets=["The final result remains unknown"],
            source_turn_refs=["R03:S01:T08"], source_sha256="abc", extraction_method="manual")
        self.assertEqual(progress["completed_actions"][0]["status"], "observed")
        self.assertEqual(progress["unresolved_targets"][0]["status"], "unknown")
        self.assertEqual(progress["extraction_method"], "manual")
        state = minimal_current_state(self.scene())
        self.assertEqual(state["planned_but_not_observed"][0]["status"], "planned")
        self.assertEqual(state["unknowns"][0]["status"], "unknown")

    def test_full_prefix_is_not_rendered_and_relation_survives(self):
        scene = self.scene(); scene["dependencies"] = [{"phase_id": "initial:ME01:C01",
            "predecessor_candidate_ref": "ME01:C01", "relation_kind": "natural_continuity", "condition": None}]
        progress = structured_actual_progress(completed_actions=[{"fact": "A prior action occurred", "provenance": "R03:S01:T09"}], source_turn_refs=["R03:S01:T09"])
        prompt = render_prompt([scene], actual_progress_by_ref={"ME02:C01": progress})
        self.assertNotIn("retained_actual_prefix", prompt["payload"])
        self.assertNotIn("old dialogue", prompt["prompt"])
        self.assertEqual(prompt["payload"]["planned_phases_minimal_state"][0]["dependencies"][0]["relation_kind"], "natural_continuity")
        self.assertEqual(prompt["payload"]["planned_phases_minimal_state"][0]["structured_actual_progress"]["source_turn_refs"], ["R03:S01:T09"])
        self.assertIn('"scenes"', prompt["prompt"])
        self.assertIn('"dialogue_turns"', prompt["prompt"])
        self.assertIn("Never return a singular `scene` object", prompt["prompt"])
        self.assertIn("exactly 14", prompt["prompt"])
        self.assertIn("never exceed 18", prompt["prompt"])

    def test_content_scan_detects_long_leak_but_allows_minimal_payload(self):
        scene = self.scene(); visible = project_scene(scene)
        self.assertTrue(leakage_scan(visible, {"before_state": scene["before_state"], "memory": scene["memory_event_content"][0]})["valid"])
        leaked = dict(visible); leaked["accidental"] = scene["before_state"]
        self.assertFalse(leakage_scan(leaked, {"before_state": scene["before_state"]})["valid"])

    def test_manifest_is_recursive_and_stable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "nested").mkdir(); (root / "nested/a.txt").write_text("x")
            manifest = freeze_manifest(root)
            self.assertEqual(manifest["file_count"], 1)
            self.assertEqual(len(manifest["files"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
