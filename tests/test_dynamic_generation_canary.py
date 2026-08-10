from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.dynamic_generation_canary import (
    api_usage, audit_child_generation, audit_child_naturalness,
)


class DynamicGenerationCanaryTests(unittest.TestCase):
    def _parent(self):
        return [{
            "turn_id": "P1", "session_id": "parent_session",
            "speaker": "Alex", "text": "An existing parent fact.",
            "partition": "parent",
        }]

    def _contract(self, participants=None, anchors=True):
        return {
            "scene_id": "S1",
            "required_participants": participants or ["Alex", "Casey"],
            "parent_anchor_evidence": [{"turn_id": "P1"}] if anchors else [],
        }

    def _session(self):
        return {
            "graph_scene_id": "S1",
            "session_id": "session_01",
            "dialogue_turns": [{
                "turn_id": "C0001", "session_id": "session_01",
                "speaker": "Casey", "text": "I completed the review.",
            }],
        }

    def test_child_audit_tracks_identity_continuity_and_evidence(self):
        audit = audit_child_generation(
            parent=self._parent(), outline={"scene_plan": [{}]},
            contracts=[self._contract()], sessions=[self._session()],
            final_ledger={"stage_states": {"E1:stage_01": {
                "status": "completed", "evidence_turn_index": 0,
            }}},
        )
        self.assertTrue(audit["parent_child_turn_ids_disjoint"])
        self.assertTrue(audit["parent_child_session_ids_disjoint"])
        self.assertTrue(audit["child_turn_ids_unique"])
        self.assertTrue(audit["child_session_ids_unique"])
        self.assertTrue(audit["child_session_ids_in_expected_order"])
        self.assertTrue(audit["scene_parent_continuity"])
        self.assertEqual(0, audit["isolated_new_character_count"])
        self.assertEqual(1.0, audit["planner_event_evidence_coverage"])
        self.assertEqual(0, audit["unauthorized_speaker_count"])
        self.assertEqual(0, audit["unauthorized_durable_fact_count"])
        self.assertEqual(0, audit["residual_nonterminal_stage_count"])

    def test_naturalness_audit_records_terminal_and_utterance_duplicates(self):
        contract = {
            **self._contract(),
            "assigned_event_stages": [{
                "stage_id": "E1:stage_01",
                "completion_evidence_phrases": ["The review is complete."],
            }],
        }
        session = {
            **self._session(),
            "dialogue_turns": [
                {"turn_id": "C0001", "session_id": "session_01",
                 "speaker": "Casey", "text": "The review is complete."},
                {"turn_id": "C0002", "session_id": "session_01",
                 "speaker": "Alex", "text": "The review is complete."},
            ],
            "authoritative_stage_updates": [{
                "stage_id": "E1:stage_01", "evidence_turn_index": 0,
            }],
        }
        audit = audit_child_naturalness(
            sessions=[session], contracts=[contract])
        self.assertEqual(1, audit["exact_repeated_utterance_count"])
        self.assertEqual(1, audit["duplicate_marker_turn_count"])
        self.assertEqual(0, audit["duplicate_terminal_evidence_count"])
        self.assertEqual(1.0, audit["unique_evidence_turn_rate"])
        self.assertTrue(audit["soft_quality_only"])
        self.assertEqual("pending", audit["scene_manual_review"][0][
            "manual_review_status"])

    def test_child_audit_rejects_unapproved_durable_candidate_and_speaker(self):
        session = self._session()
        session["dialogue_turns"][0]["speaker"] = "Unknown"
        session["durable_assertion_candidates"] = ["invented fact"]
        audit = audit_child_generation(
            parent=self._parent(), outline={"scene_plan": [{}]},
            contracts=[self._contract()], sessions=[session],
            final_ledger={"stage_states": {"E1:stage_01": {
                "status": "assigned", "evidence_turn_index": None,
            }}},
        )
        self.assertEqual(1, audit["unauthorized_speaker_count"])
        self.assertEqual(1, audit["unauthorized_durable_fact_count"])
        self.assertEqual(1, audit["residual_nonterminal_stage_count"])

    def test_child_audit_detects_isolated_new_character(self):
        audit = audit_child_generation(
            parent=self._parent(), outline={"scene_plan": [{}]},
            contracts=[self._contract(participants=["Casey"], anchors=False)],
            sessions=[self._session()], final_ledger={"stage_states": {}})
        self.assertEqual(["Casey"], audit["isolated_new_characters"])
        self.assertFalse(audit["scene_parent_continuity"])

    def test_child_audit_handles_planner_failure_before_any_session(self):
        audit = audit_child_generation(
            parent=self._parent(), outline=None, contracts=[], sessions=[],
            final_ledger=None)
        self.assertEqual(0, audit["child_turn_count"])
        self.assertEqual(0.0, audit["planner_event_evidence_coverage"])
        self.assertEqual(0, audit["exact_repeated_or_template_scene_count"])

    def test_api_usage_counts_only_completed_physical_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "planner/ledger/requests"
            requests.mkdir(parents=True)
            (requests / "one.json").write_text(json.dumps({
                "status": "completed", "intent": "lineage_stateful_outline",
                "attempt": 1, "response_metadata": {"usage": {
                    "prompt_tokens": 10, "completion_tokens": 5,
                    "total_tokens": 15}},
            }))
            (requests / "two.json").write_text(json.dumps({
                "status": "reserved", "intent": "stateful_realize_scene",
                "attempt": 1,
            }))
            usage = api_usage(root)
            self.assertEqual(1, usage["physical_api_requests"])
            self.assertEqual(1, usage["ambiguous_request_count"])
            self.assertEqual(15, usage["total_tokens"])

    def test_module_has_no_questioner_or_solver_backend(self):
        source = Path(
            "scripts/current_method/restart_n5_spell/"
            "dynamic_generation_canary.py").read_text()
        self.assertNotIn("LiveQuestionerBackend", source)
        self.assertNotIn("VLLMSolverBackend", source)
        self.assertNotIn("QuestionerRequest", source)


if __name__ == "__main__":
    unittest.main()
