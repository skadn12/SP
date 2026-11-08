"""Card-first prompts keep fallible purpose drafts without promoting them.

Recorded v30 data exercise request wiring; scripted semantic findings exercise
existing rejection contracts. Neither proves a fresh Qwen semantic judgment.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import event_state as scan
import latest_state as state
import planner_contract
import purpose_contract as purpose
import realization_review as review
import runner
import test_purpose_review_v25 as purpose_fixtures
import test_seed_review_v24 as seed_fixtures


ORIGIN = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v30/runs/inherited_a2_b3_01"


def recorded_requests(prefix):
    directory = next((ORIGIN / "requests").glob(prefix + "-*"))
    return json.loads((directory / "REQUEST.json").read_text()), directory


def context_of(request, marker):
    content = request["messages"][0]["content"] if isinstance(request, dict) else request
    return json.JSONDecoder().raw_decode(content.split(marker, 1)[1])[0]


class ProvisionalPurposeV31Tests(unittest.TestCase):
    def test_three_recorded_drift_drafts_keep_original_card_and_schema(self):
        cards = json.loads((ORIGIN / "dataset_cards_result.json").read_text())["accepted_cards"]
        cards = {card["card_id"]: card for card in cards}
        for prefix in ("004", "012", "017"):
            old_request, directory = recorded_requests(prefix)
            before = (directory / "RESPONSE.json").read_bytes()
            draft = json.loads(json.loads(before)["text"])
            card = cards[draft["dataset_card_id"]]
            old_context = context_of(old_request, "\nPURPOSE_INPUT_JSON\n")
            request = purpose.build_request(draft["root_id"], draft["scene_id"], card,
                old_context["allowed_speakers"], runner.MODEL)
            with self.subTest(prefix=prefix):
                # New instructions change authority, not source data, schema,
                # model, request count, or a recorded false model response.
                self.assertEqual(context_of(request, "\nPURPOSE_INPUT_JSON\n"), old_context)
                self.assertEqual(purpose.validate_purpose(draft, draft["root_id"], draft["route"],
                    draft["scene_id"], card), draft)
                self.assertEqual(set(draft), set(purpose.FIELDS))
                self.assertEqual(request["model"], old_request["model"])
                self.assertEqual(request["max_tokens"], old_request["max_tokens"])
                self.assertEqual((directory / "RESPONSE.json").read_bytes(), before)
                text = request["messages"][0]["content"].split("PURPOSE_OUTPUT_SCHEMA_JSON", 1)[0]
                self.assertIn("core interaction takes priority", text)
                self.assertIn("or a binding goal", text)
                self.assertIn("concrete new occasion", text)
                self.assertIn("A seed does not require its entire source workflow", text)
                self.assertIn("do not invent earlier conversations", text)

    def test_planner_and_realizer_receive_draft_as_provisional_with_no_rewrite(self):
        request, _ = recorded_requests("007")
        data = context_of(request, "\nINPUT_JSON\n")
        before = deepcopy(data)
        card = data["dataset_cards"][0]
        scene = {"scene_id": data["target_scene_id"], "goal": "A new card-faithful occasion",
            "developments": ["Offer thanks for current assistance.", "Acknowledge the thanks."],
            "dataset_card_id": card["card_id"], "source_anchors": [],
            "card_development_links": [{"card_step_ids": [card["steps"][0]["step_id"]],
                "dependency_indices": [], "development_indices": [1, 2]}]}
        for role in ("planner", "realizer"):
            with patch.object(runner, "branch_history_ok", return_value=data["cumulative_history"]), \
                 patch.object(runner, "history_proof", return_value={}), \
                 patch.object(runner.base, "read", return_value={"target_solver_identity_sha256": "test-id"}), \
                 patch.object(runner.base.feedback, "append_feedback_to_request",
                    side_effect=lambda payload, *_args, **_kwargs: (payload, {})):
                payload, proof = runner.make_request(Path("/unused-test-run"), data["root_id"],
                    data["route"], role, data["cumulative_history"], [card], data["target_scene_id"],
                    scene=scene if role == "realizer" else None, latest_state=data["latest_state"],
                    purpose_input=data["purpose_input"])
            with self.subTest(role=role):
                text = payload["messages"][0]["content"]
                sent = context_of(payload, "\nINPUT_JSON\n")
                self.assertEqual(sent["purpose_input"], before["purpose_input"])
                self.assertEqual(sent["dataset_cards"], before["dataset_cards"])
                self.assertEqual(proof["purpose_input_sha256"], runner.base.digest(before["purpose_input"]))
                self.assertIn("PROVISIONAL PURPOSE", text)
                self.assertIn("core interaction takes priority", text)
                self.assertNotIn("BOUND PURPOSE", text)
                self.assertNotIn("Preserve its concrete trigger and goal", text)
                if role == "planner":
                    self.assertIn("disagreement with the draft alone is not a failure", text)
                else:
                    self.assertIn("Do not reintroduce purpose draft details absent from scene_plan", text)
        self.assertEqual(data, before)

    def test_state_scanner_reviewer_share_priority_and_preserve_frozen_context(self):
        request, _ = recorded_requests("007")
        data = context_of(request, "\nINPUT_JSON\n")
        card = data["dataset_cards"][0]
        draft = deepcopy(data["purpose_input"])
        scene = {"scene_id": draft["scene_id"], "dataset_card_id": card["card_id"],
            "goal": "Thank a helper on a newly introduced occasion.", "source_anchors": [],
            "developments": ["Offer thanks for the current help.", "Acknowledge the thanks."],
            "card_development_links": [{"card_step_ids": [card["steps"][0]["step_id"]],
                "dependency_indices": [], "development_indices": [1, 2]}]}
        realization = {"root_id": draft["root_id"], "route": draft["route"],
            "scene_id": draft["scene_id"], "status": "scene", "turns": [
                {"speaker": "Avery", "text": "Thank you for helping just now."},
                {"speaker": "Blake", "text": "I'm glad I could help."}]}
        state_request = state.build_state_request(draft["root_id"], draft["route"], draft["scene_id"],
            data["cumulative_history"], [card], purpose_input=draft)
        scan_request = scan.build_request(draft["root_id"], draft["route"], scene,
            data["cumulative_history"], [card], data["latest_state"], purpose_input=draft)
        review_request = review.build_review_prompt(draft["root_id"], draft["route"], scene,
            data["cumulative_history"], realization, dataset_card=card, purpose_input=draft)
        for output, marker, card_key in (
            (state_request, "\nSTATE_INPUT_JSON\n", "dataset_cards"),
            (scan_request, "\nEVENT_STATE_INPUT_JSON\n", "dataset_cards"),
            (review_request, "\nREVIEW INPUT DATA:\n", "dataset_card_guidance"),
        ):
            text = output["messages"][0]["content"] if isinstance(output, dict) else output
            sent = context_of(output, marker)
            with self.subTest(marker=marker):
                self.assertIn("core interaction takes priority over purpose_input", text)
                self.assertNotIn("purpose_input is the assigned future objective", text)
                self.assertEqual(sent["purpose_input"], draft)
                guidance = sent[card_key][0] if card_key == "dataset_cards" else sent[card_key]
                self.assertEqual(guidance, planner_contract.mapping_context(card)["card_guidance"])
        self.assertIn("Disagreement with the draft alone is not assigned-event drift", scan_request["messages"][0]["content"])
        self.assertIn("assigned_event_mismatch", scan_request["messages"][0]["content"])
        self.assertIn("No semantic pass follows from the purpose hash or copied wording", review_request)

    def test_existing_assignment_negative_still_binds_without_new_check_fields(self):
        # Explicit synthetic Qwen judgment of actual card drift still requires
        # replanning even if both a copied goal label and route tuple say pass.
        suite = purpose_fixtures.PurposeReviewV25Tests()
        suite.setUp()
        fixture = suite.scan_fixture()
        checked = fixture.validate()
        self.assertTrue(scan.requires_replan(checked))
        finding = scan.binding_plan_checks(checked)[0]
        self.assertEqual(finding["conflict"]["type"], "assigned_event_mismatch")
        self.assertEqual(finding["constraint_ids"], [])
        self.assertTrue(scan.route_fulfilled(checked["route_fulfillment"], fixture.route))

    def test_card_fidelity_negative_not_overridden_by_purpose_wording(self):
        fixture = seed_fixtures.SeedReviewV24Tests()
        fixture.setUp()
        raw = deepcopy(fixture.raw)
        raw["card_checks"][0].update(realized=False, evidence=[],
            reason="SYNTHETIC: a logistics conversation abandons the card's recovery interaction.")
        checked = fixture.validate(raw)
        self.assertTrue(checked["schema_valid"], checked["schema_failures"])
        self.assertEqual(checked["status"], "REJECT")
        self.assertTrue(checked["parsed"]["route_fulfillment"]["satisfied"])
        self.assertFalse(checked["semantic_certified"])


if __name__ == "__main__":
    unittest.main()
