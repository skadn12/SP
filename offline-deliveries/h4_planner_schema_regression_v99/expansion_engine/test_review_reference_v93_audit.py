"""Offline tamper checks for final reviewer reference binding; no model calls."""
from copy import deepcopy
import json
import unittest

import audit_run as audit
import runner as r
from test_runner_semantic_integration import CARD, HISTORY, ROOT, SCENE, B


class FinalReviewReferenceAuditTests(unittest.TestCase):
    def event(self, repair=None):
        payload = r.event_contract.build_request(ROOT, B, SCENE, HISTORY, [CARD], {"facts": []}, repair=repair)
        context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
        return payload, context

    def scene(self, repair=None):
        realization = {"root_id": ROOT, "route": B, "scene_id": "S01", "status": "scene",
                       "turns": [{"speaker": "Elena", "text": "Please give me the new lunch venue's details."},
                                 {"speaker": "Milan", "text": "The venue is open from noon at 18 Birch Street."}],
                       "feedback_topic_ids": [], "no_scene_reason": ""}
        payload, _ = r.review_request(ROOT, B, SCENE, HISTORY, realization,
                                      [{"turn_ref": "R02:A:S01:T001", "speaker": "Milan", "text": "Other branch private comparison."}],
                                      card=CARD, repair=repair)
        context = audit.decode_context(payload, "\nREVIEW INPUT DATA:\n")
        return payload, context, realization

    def check(self, role, payload, context):
        return audit.review_instruction_tail(payload, role, context,
                                             cards=[CARD] if role == "event_state_extractor" else None)

    def test_event_tail_after_input_and_targeted_repair_is_exact(self):
        for repair in (None, {"previous_raw": "bad original quote...", "errors": ["bad reference"]}):
            payload, context = self.event(repair)
            before = deepcopy((payload, context))
            tail = self.check("event_state_extractor", payload, context)
            self.assertEqual(tail, r.event_contract.final_review_instruction(SCENE, [CARD]))
            self.assertEqual((payload, context), before)
            self.assertEqual(context.get("targeted_repair"), repair)

    def test_scene_tail_after_input_and_format_repair_uses_same_turn_metadata(self):
        for repair in (None, {"previous_response": "bad original response...", "failure_codes": ["bad citation"]}):
            payload, context, raw = self.scene(repair)
            before = deepcopy((payload, context, raw))
            tail = self.check("scene_reviewer", payload, context)
            self.assertEqual(tail, r.realization_review.final_review_instruction(SCENE, raw))
            self.assertEqual(tail, r.realization_review.final_review_instruction(SCENE, context["new_scene"]))
            self.assertNotIn("Other branch private comparison.", tail)
            self.assertEqual((payload, context, raw), before)

    def test_missing_changed_or_postfixed_tail_rejects_even_if_payload_hash_rebound(self):
        event_payload, event_context = self.event()
        scene_payload, scene_context, _ = self.scene()
        for role, original, context in (("event_state_extractor", event_payload, event_context),
                                        ("scene_reviewer", scene_payload, scene_context)):
            tail = self.check(role, original, context)
            for mutation in ("remove", "change", "after"):
                payload = deepcopy(original)
                text = payload["messages"][0]["content"]
                if mutation == "remove":
                    text = text[:-len(tail)]
                elif mutation == "change":
                    text = text[:-len(tail)] + tail.replace("development", "wrong_reference", 1)
                else:
                    text += "\nUnbound instruction after current review."
                payload["messages"][0]["content"] = text
                # Rehashing an altered request cannot satisfy semantic input/tail binding.
                self.assertTrue(r.base.digest(payload))
                with self.subTest(role=role, mutation=mutation), self.assertRaises(ValueError):
                    self.check(role, payload, context)

    def test_tail_for_other_plan_card_or_scene_cannot_bind_original_context(self):
        payload, context = self.event()
        actual_tail = self.check("event_state_extractor", payload, context)
        scene, card = deepcopy(SCENE), deepcopy(CARD)
        scene["developments"][0] = "Wrong unrelated development."
        card["steps"][0]["action"] = "Wrong unrelated action."
        for wrong_tail in (r.event_contract.final_review_instruction(scene, [CARD]),
                           r.event_contract.final_review_instruction(SCENE, [card])):
            mutated = deepcopy(payload)
            mutated["messages"][0]["content"] = payload["messages"][0]["content"][:-len(actual_tail)] + wrong_tail
            with self.assertRaises(ValueError):
                self.check("event_state_extractor", mutated, context)
        payload, context, raw = self.scene()
        actual_tail = self.check("scene_reviewer", payload, context)
        wrong_raw = deepcopy(raw)
        wrong_raw["turns"][0]["speaker"] = "Sidney"
        wrong_raw["turns"][1]["text"] = "A different venue at 99 Wrong Street."
        payload["messages"][0]["content"] = payload["messages"][0]["content"][:-len(actual_tail)] + r.realization_review.final_review_instruction(SCENE, wrong_raw)
        with self.assertRaises(ValueError):
            self.check("scene_reviewer", payload, context)

    def test_repair_or_arbitrary_prose_between_data_and_tail_cannot_escape_audit(self):
        event_payload, event_context = self.event({"previous_raw": "bad"})
        scene_payload, scene_context, _ = self.scene({"previous_response": "bad"})
        for role, original, context in (("event_state_extractor", event_payload, event_context),
                                        ("scene_reviewer", scene_payload, scene_context)):
            tail = self.check(role, original, context)
            mutated = deepcopy(original)
            mutated["messages"][0]["content"] = original["messages"][0]["content"][:-len(tail)] + "\nUnbound prose." + tail
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.check(role, mutated, context)

    def test_tail_binding_never_rewrites_original_response_or_imports_comparison_into_generator(self):
        payload, context, _ = self.scene({"previous_response": "KEEP RAW RESPONSE EXACTLY"})
        text_before = payload["messages"][0]["content"]
        self.check("scene_reviewer", payload, context)
        self.assertEqual(payload["messages"][0]["content"], text_before)
        self.assertIn("KEEP RAW RESPONSE EXACTLY", text_before)
        event_payload, event_context = self.event()
        self.assertNotIn("other_branch_added_turns_for_review_only", event_context)
        self.assertNotIn("Other branch private comparison.", event_payload["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
