"""Offline contract tests; fixture snapshots are not production extractions."""
from copy import deepcopy
import json
import unittest

import latest_state as state


class LatestStateTests(unittest.TestCase):
    def setUp(self):
        self.root = "cc-train-episode-210046"
        self.route = "dataset_new_event"
        # Exact excerpts from v6's frozen 210046 input. Later reported arrival
        # must be available, rather than inferring arrival from 'heading into'.
        self.history = [
            {"turn_ref": "R01:S01:T015", "speaker": "Emily",
             "text": "I'm just heading into the office now. The email had all the details, so I think I've got everything covered."},
            {"turn_ref": "R01:S01:T016", "speaker": "David",
             "text": "Perfect, Emily. I'll be waiting in the lobby to greet you. Let me know if you need anything before you arrive."},
            {"turn_ref": "R01:S01:T021", "speaker": "Emily",
             "text": "I'm officially here! Let's make this work."},
            {"turn_ref": "R01:S01:T022", "speaker": "David",
             "text": "That's the spirit. Let's get started."},
            {"turn_ref": "R01:S03:T007", "speaker": "Emily",
             "text": "Thanks, I really appreciate your support. I'll make the donation as soon as I can."},
            {"turn_ref": "R01:S03:T011", "speaker": "Emily",
             "text": "I've made the donation, David. I sent you the receipt as well. I hope it helps."},
        ]
        self.snapshot = {
            "root_id": self.root, "route": self.route, "scene_id": "S01",
            "facts": [
                {"fact_id": "f1", "entity": "Emily", "event": "arrival at the office",
                 "status": "completed", "statement": "Emily reports she is officially here after heading into the office.",
                 "evidence": [{"turn_ref": "R01:S01:T015", "quote": "I'm just heading into the office now."},
                              {"turn_ref": "R01:S01:T021", "quote": "I'm officially here!"}]},
                {"fact_id": "f2", "entity": "Emily", "event": "donation",
                 "status": "completed", "statement": "Emily reports making the donation and sending David the receipt.",
                 "evidence": [{"turn_ref": "R01:S03:T011",
                               "quote": "I've made the donation, David. I sent you the receipt as well."}]},
            ],
            "conflicts": [], "unknowns": ["The date of a next trip is not established."],
        }
        self.card = {
            "card_id": "card_fixture", "card_sha256": "fixture", "guidance_level": "scaffold",
            "source_id": "PRIVATE SOURCE ID", "source_text": "PRIVATE ORIGINAL TEXT",
            "actors": [{"actor_id": "a1", "role": "customer", "source_name": "PRIVATE NAME"}],
            "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request a transfer", "state": "proposed",
                       "evidence": [{"quote": "PRIVATE SOURCE QUOTE"}]}],
            "dependencies": [], "confusion_axes": ["pickup versus destination"], "open_ends": [],
        }
        self.scene = {"scene_id": "S01", "state_transition": {
            "basis_fact_ids": ["f1"], "new_event": "Arrange a separate later transfer after arrival.",
            "temporal_placement": "After the reported arrival, for a distinct later trip; date remains unspecified.",
        }}

    def validate(self, raw=None):
        return state.validate_state(self.snapshot if raw is None else raw, self.root, self.route, "S01", self.history)

    def codes(self, raw=None):
        with self.assertRaises(state.StateValidationError) as caught:
            self.validate(raw)
        return {error["code"] for error in caught.exception.errors}

    def test_valid_late_completed_evidence_is_preserved_without_mutation(self):
        original = deepcopy((self.snapshot, self.history))
        parsed = self.validate(json.dumps(self.snapshot))
        self.assertEqual(parsed, self.snapshot)
        self.assertEqual((self.snapshot, self.history), original)
        self.assertEqual(parsed["facts"][0]["evidence"][-1]["turn_ref"], "R01:S01:T021")
        # The latest topical turn concerns another event; the prior completed
        # milestone is retained, not truncated to the final few history turns.
        self.assertEqual(parsed["facts"][1]["status"], "completed")

    def test_dict_result_is_detached(self):
        parsed = self.validate()
        parsed["facts"][0]["statement"] = "caller modification"
        self.assertNotEqual(parsed, self.snapshot)

    def test_exact_quotes_required(self):
        for quote in ("I'm already here!", "I'm officially … here!", " ", None):
            with self.subTest(quote=quote):
                snapshot = deepcopy(self.snapshot)
                snapshot["facts"][0]["evidence"][1]["quote"] = quote
                self.assertTrue(self.codes(snapshot) & {"state_evidence_quote_not_exact", "state_evidence_quote_invalid"})

    def test_foreign_and_bogus_turn_refs_rejected(self):
        for ref in ("R02:A:S01:T001", "missing", 1, []):
            with self.subTest(ref=ref):
                snapshot = deepcopy(self.snapshot)
                snapshot["facts"][0]["evidence"][1]["turn_ref"] = ref
                self.assertIn("state_evidence_reference_invalid", self.codes(snapshot))

    def test_duplicate_json_keys_at_any_depth_rejected(self):
        raw = json.dumps(self.snapshot)
        for corrupted in (raw.replace('"root_id":', '"root_id": "other", "root_id":', 1),
                          raw.replace('"status": "completed"', '"status": "planned", "status": "completed"', 1)):
            self.assertIn("state_duplicate_json_key", self.codes(corrupted))

    def test_plain_json_only(self):
        for raw in ("not json", "[]", "null", "```json\n{}\n```", 3, '{"facts": NaN}'):
            with self.subTest(raw=raw):
                self.assertTrue(self.codes(raw) & {"state_response_invalid_json", "state_response_not_object"})

    def test_exact_fields_and_identity(self):
        snapshot = deepcopy(self.snapshot)
        snapshot.update({"root_id": "other", "route": "old_event_continuation", "scene_id": "S02", "invented": True})
        codes = self.codes(snapshot)
        self.assertIn("state_fields_invalid", codes)
        self.assertIn("state_identity_mismatch", codes)
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"][0]["evidence"][0]["source"] = "other"
        self.assertIn("state_fields_invalid", self.codes(snapshot))

    def test_status_whitelist_and_unknowns_are_separate(self):
        for status in ("unknown", "arrived", "done", [], None):
            with self.subTest(status=status):
                snapshot = deepcopy(self.snapshot)
                snapshot["facts"][0]["status"] = status
                self.assertIn("state_status_invalid", self.codes(snapshot))
        for status in sorted(state.STATUSES):
            snapshot = deepcopy(self.snapshot)
            snapshot["facts"][0]["status"] = status
            self.assertEqual(self.validate(snapshot)["facts"][0]["status"], status)
        self.assertEqual(self.validate()["unknowns"], self.snapshot["unknowns"])

    def test_citation_validator_does_not_claim_semantic_entailment(self):
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"][0]["statement"] = "This intentionally unsupported statement is for a contract-only test."
        # Semantic correctness remains Qwen's responsibility, then the reviewer;
        # a mechanically valid exact citation is not a correctness certificate.
        self.assertEqual(self.validate(snapshot), snapshot)

    def test_conflicting_reports_and_unknowns_survive_unchanged(self):
        self.history.append({"turn_ref": "R01:fixture:conflict", "speaker": "David", "text": "The donation has not been sent."})
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"].append({"fact_id": "f3", "entity": "David", "event": "donation",
                                  "status": "uncertain", "statement": "David reports the donation has not been sent.",
                                  "evidence": [{"turn_ref": "R01:fixture:conflict", "quote": "The donation has not been sent."}]})
        snapshot["conflicts"] = [{"fact_ids": ["f2", "f3"], "reason": "The two reported statuses conflict; no resolution is supplied."}]
        snapshot["unknowns"].append("Whether the two reports refer to the same transfer remains unresolved.")
        self.assertEqual(self.validate(snapshot), snapshot)

    def test_conflict_requires_two_distinct_actual_fact_ids(self):
        for refs in ([], ["f1"], ["f1", "f1"], ["f1", "f9"], ["f1", []]):
            with self.subTest(refs=refs):
                snapshot = deepcopy(self.snapshot)
                snapshot["conflicts"] = [{"fact_ids": refs, "reason": "fixture conflict"}]
                self.assertIn("state_conflict_fact_ids_invalid", self.codes(snapshot))

    def test_facts_count_ids_evidence_count_and_text(self):
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"] = []
        self.assertIn("state_facts_count_invalid", self.codes(snapshot))
        snapshot["facts"] = [deepcopy(self.snapshot["facts"][0]) for _ in range(7)]
        self.assertIn("state_facts_count_invalid", self.codes(snapshot))
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"][1]["fact_id"] = "f1"
        snapshot["facts"][0]["evidence"] = []
        snapshot["facts"][0]["entity"] = ""
        codes = self.codes(snapshot)
        self.assertTrue({"state_fact_id_invalid", "state_evidence_count_invalid", "state_text_required"} <= codes)

    def test_conflict_and_unknown_count_bounds(self):
        snapshot = deepcopy(self.snapshot)
        snapshot["conflicts"] = [{"fact_ids": ["f1", "f2"], "reason": "fixture"}] * 4
        snapshot["unknowns"] = ["fixture"] * 5
        self.assertTrue({"state_conflicts_count_invalid", "state_unknowns_invalid"} <= self.codes(snapshot))
        for unknowns in (None, {}, [""], [False]):
            snapshot["unknowns"] = unknowns
            self.assertIn("state_unknowns_invalid", self.codes(snapshot))

    def test_duplicate_history_references_and_missing_history_rejected(self):
        self.history.append(deepcopy(self.history[0]))
        self.assertIn("state_history_duplicate_reference", self.codes())
        self.history.clear()
        self.assertIn("state_history_required", self.codes())

    def test_request_preserves_full_history_and_model_config(self):
        history = deepcopy(self.history)
        history[2:2] = [{"turn_ref": f"fixture:{i}", "speaker": "Emily", "text": f"Unrelated fixture mention {i}."}
                        for i in range(200)]
        original = deepcopy((history, self.card))
        payload = state.build_state_request(self.root, self.route, "S01", history, [self.card])
        self.assertEqual(payload["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(payload["temperature"], .25)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        context = json.loads(payload["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["cumulative_history"], history)
        self.assertEqual(set(context), {"root_id", "route", "scene_id", "cumulative_history", "dataset_cards"})
        self.assertEqual((history, self.card), original)

    def test_request_only_abstract_card_guidance_and_no_feedback(self):
        payload = state.build_state_request(self.root, self.route, "S01", self.history, [self.card])
        context = json.loads(payload["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])
        encoded = json.dumps(context)
        for forbidden in ("PRIVATE", "source_id", "source_name", "feedback", "memory", "other_branch"):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(context["dataset_cards"][0]["source_is_root_evidence"])

    def test_A_request_has_no_card_and_invalid_inputs_fail(self):
        payload = state.build_state_request(self.root, "old_event_continuation", "S01", self.history, [])
        context = json.loads(payload["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["dataset_cards"], [])
        for route, history, cards in (("wrong", self.history, []),
                                      (self.route, [], [self.card]),
                                      (self.route, self.history, []),
                                      ("old_event_continuation", self.history, [self.card])):
            with self.subTest(route=route, cards=cards):
                with self.assertRaises(ValueError):
                    state.build_state_request(self.root, route, "S01", history, cards)

    def test_repair_preserves_all_errors_and_raw_without_editing(self):
        snapshot = deepcopy(self.snapshot)
        snapshot["facts"][0]["evidence"] = []
        snapshot["facts"][1]["status"] = "unknown"
        original = deepcopy(snapshot)
        try:
            self.validate(snapshot)
        except state.StateValidationError as error:
            repair = state.build_state_repair(error, snapshot)
            self.assertEqual(repair["errors"], error.errors)
        self.assertEqual(repair["previous_raw"], snapshot)
        self.assertEqual(snapshot, original)
        payload = state.build_state_request(self.root, self.route, "S01", self.history, [self.card], repair=repair)
        context = json.loads(payload["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["targeted_repair"], repair)

    def test_transition_accepts_only_real_distinct_refs_and_explicit_fields(self):
        original = deepcopy((self.scene, self.snapshot))
        self.assertTrue(state.validate_transition(self.scene, self.snapshot))
        self.assertEqual((self.scene, self.snapshot), original)
        for refs in ([], ["f1", "f1"], ["f3"], [1], None):
            scene = deepcopy(self.scene)
            scene["state_transition"]["basis_fact_ids"] = refs
            with self.subTest(refs=refs), self.assertRaises(state.StateValidationError) as caught:
                state.validate_transition(scene, self.snapshot)
            self.assertIn("state_transition_fact_ids_invalid", {e["code"] for e in caught.exception.errors})
        for scene in ({}, {"state_transition": None}, {"state_transition": {"basis_fact_ids": ["f1"], "new_event": "", "extra": 1}}):
            with self.subTest(scene=scene), self.assertRaises(state.StateValidationError):
                state.validate_transition(scene, self.snapshot)

    def test_prompt_requires_latest_full_history_and_explicit_conflicts(self):
        for marker in ("ENTIRE cumulative_history", "NOT automatically event time", "unresolved conflict",
                       "not a destination", "does\nnot prove arrival", "unknowns, never", "NOT an event"):
            self.assertIn(marker, state.STATE_PROMPT)
        for marker in ("basis_fact_ids", "temporal_placement", "not a\ncertified summary", "completed earlier one"):
            self.assertIn(marker, state.PLAN_STATE_PROMPT)


if __name__ == "__main__":
    unittest.main()
