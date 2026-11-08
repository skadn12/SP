"""Offline v94 evidence-boundary regressions; authored verdicts are not model evals.

Tests use the production validator, reference-tail audit and runner reviewer
retry path. Archived bytes are read-only and model responses are never repaired
by the program. FakeCaller performs no network calls.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import audit_run as audit
import runner as r
from test_event_state_v92 import review_fixture
from test_runner_semantic_integration import CARD, HISTORY, ROOT, SCENE, B, FakeCaller


class EvidenceAuditV94Tests(unittest.TestCase):
    def setUp(self):
        self.card = deepcopy(CARD)
        self.scene = deepcopy(SCENE)
        # Put the relation at its before-step development while its after-step
        # remains at a second development, as in real archived failures.
        self.scene["card_development_links"][0]["dependency_indices"] = [1]
        self.scene["card_development_links"][1]["dependency_indices"] = []
        self.scene["goal"] = "Elena and Milan arrange a new lunch venue."
        self.scene["state_transition"] = {
            "basis_fact_ids": [], "new_event": "Elena and Milan discuss a new venue.",
            "temporal_placement": "Tomorrow, after their earlier discussion.",
        }
        self.response = review_fixture(self.scene, HISTORY, [self.card], root=ROOT)

    def dev(self, index):
        return {"development_index": index, "quote": self.scene["developments"][index - 1]}

    def metadata(self, field_id):
        text = self.fields()[field_id]
        return {"plan_field_id": field_id, "quote": text}

    def fields(self):
        return {row["plan_field_id"]: row["text"]
                for row in r.event_contract.plan_field_inventory(self.scene)}

    def dependency(self, response=None):
        return (response if response is not None else self.response)["card_checks"][0]["dependency_checks"][0]

    def validate(self, response=None):
        return r.event_contract.validate(response if response is not None else self.response,
                                         ROOT, B, self.scene, HISTORY, cards=[self.card])

    def assert_invalid(self, response):
        with self.assertRaises(r.event_contract.EventStateValidationError) as caught:
            self.validate(response)
        return caught.exception.errors

    def test_dependency_context_quotes_do_not_expand_original_mapping_or_change_verdict(self):
        self.dependency()["plan_evidence"] = [self.dev(1), self.dev(2)]
        self.dependency().update(verdict="revise", reason="Both actions occur but the required response relation is absent.")
        before = deepcopy((self.response, self.scene, self.card))
        raw = json.dumps(self.response, ensure_ascii=False)
        result = self.validate(raw)
        self.assertEqual(result, self.response)
        self.assertEqual(self.dependency(result)["mapped_development_indices"], [1])
        self.assertEqual(before, (self.response, self.scene, self.card))
        self.assertEqual(raw, json.dumps(self.response, ensure_ascii=False))
        self.assertTrue(r.event_contract.requires_replan(result))
        self.assertEqual(r.event_contract.binding_plan_checks(result)[0]["check_kind"], "card_dependency")

    def test_endpoint_only_evidence_does_not_replace_original_dependency_map_evidence(self):
        self.dependency()["plan_evidence"] = [self.dev(2)]
        errors = self.assert_invalid(self.response)
        self.assertIn("event_state_dependency_original_mapping_not_cited", [row["code"] for row in errors])
        self.assertEqual(self.dependency()["mapped_development_indices"], [1])

    def test_dependency_original_map_field_cannot_be_rewritten_to_union(self):
        self.dependency().update(mapped_development_indices=[1, 2], plan_evidence=[self.dev(1), self.dev(2)])
        errors = self.assert_invalid(self.response)
        self.assertIn("event_state_card_mapped_developments_mismatch", [row["code"] for row in errors])

    def test_step_cannot_borrow_other_step_or_plan_metadata_evidence(self):
        for span in (self.dev(2), self.metadata("goal"), self.metadata("actors:1")):
            response = deepcopy(self.response)
            response["card_checks"][0]["step_checks"][0]["plan_evidence"] = [span]
            with self.subTest(span=span):
                self.assert_invalid(response)

    def test_metadata_is_exactly_allowlisted_and_only_for_consistency(self):
        fields = self.fields()
        self.assertEqual(set(fields), {"goal", "actors:1", "actors:2",
                                       "state_transition:new_event", "state_transition:temporal_placement"})
        self.response["consistency_checks"][0]["plan_evidence"] = [self.metadata("state_transition:new_event")]
        self.response["consistency_checks"][0].update(verdict="revise", reason="The transition attributes this event to the wrong established people.")
        result = self.validate()
        self.assertTrue(r.event_contract.requires_replan(result))
        self.assertEqual(r.event_contract.binding_plan_checks(result)[0]["check_kind"], "plan_consistency")
        for field_id in ("state_transition:basis_fact_ids", "state_transition:new_event.text", "__class__",
                         "cumulative_history:1", "other_branch_added_turns_for_review_only:1", "actors:0"):
            response = deepcopy(self.response)
            response["consistency_checks"][0]["plan_evidence"] = [{"plan_field_id": field_id, "quote": "Elena"}]
            with self.subTest(field_id=field_id):
                errors = self.assert_invalid(response)
                self.assertIn("event_state_plan_evidence_field_invalid", [row["code"] for row in errors])
        response = deepcopy(self.response)
        response["plan_checks"][0]["plan_evidence"] = [self.metadata("goal")]
        self.assert_invalid(response)
        response = deepcopy(self.response)
        self.dependency(response)["plan_evidence"] = [self.dev(1), self.metadata("goal")]
        self.assert_invalid(response)

    def test_metadata_quote_path_mismatch_and_mixed_index_shape_rejected(self):
        for span, code in (({"plan_field_id": "goal", "quote": self.scene["state_transition"]["temporal_placement"]},
                            "event_state_plan_evidence_quote_invalid"),
                           ({"plan_field_id": "goal", "quote": self.scene["goal"], "development_index": 1},
                            "event_state_fields_invalid"),
                           ({"plan_field_id": "goal", "quote": self.scene["goal"][:5] + "..."},
                            "event_state_plan_evidence_quote_invalid"),
                           ({"plan_field_id": "goal", "quote": ""}, "event_state_plan_evidence_quote_invalid")):
            response = deepcopy(self.response)
            response["consistency_checks"][0]["plan_evidence"] = [span]
            with self.subTest(span=span):
                errors = self.assert_invalid(response)
                self.assertIn(code, [row["code"] for row in errors])

    def test_metadata_inventory_does_not_coerce_or_import_arbitrary_fields(self):
        self.scene["state_transition"]["basis_fact_ids"] = ["foreign quote"]
        self.scene["state_transition"]["other_branch"] = "OTHER_BRANCH_SENTINEL"
        self.scene["source_anchors"] = [{"quote": "SOURCE_QUOTE_SENTINEL"}]
        self.scene["goal"] = {"text": "DO_NOT_COERCE_OBJECT"}
        fields = self.fields()
        self.assertNotIn("goal", fields)
        self.assertNotIn("state_transition:basis_fact_ids", fields)
        self.assertNotIn("state_transition:other_branch", fields)
        self.assertNotIn("OTHER_BRANCH_SENTINEL", json.dumps(fields))
        self.assertNotIn("SOURCE_QUOTE_SENTINEL", json.dumps(fields))

    def test_audit_tail_binds_context_scope_and_metadata_literal_text(self):
        payload = r.event_contract.build_request(ROOT, B, self.scene, HISTORY, [self.card], {"facts": []})
        context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
        before = deepcopy((payload, context, self.scene, self.card))
        tail = audit.review_instruction_tail(payload, "event_state_extractor", context, cards=[self.card])
        self.assertEqual(before, (payload, context, self.scene, self.card))
        table = json.loads(tail.split("ACTUAL REVIEW REFERENCE TABLE\n", 1)[1])
        self.assertEqual(table["actual_card_targets"][0]["dependency_targets"][0]["mapped_development_indices"], [1])
        self.assertTrue(any(field["plan_field_id"] == "state_transition:new_event" for field in table["plan_fields"]))
        wrong_scene = deepcopy(self.scene)
        wrong_scene["state_transition"]["new_event"] = "A modified relationship premise."
        altered = deepcopy(payload)
        altered["messages"][0]["content"] = payload["messages"][0]["content"][:-len(tail)] + r.event_contract.final_review_instruction(wrong_scene, [self.card])
        with self.assertRaises(ValueError):
            audit.review_instruction_tail(altered, "event_state_extractor", context, cards=[self.card])

    def test_runner_invalid_metadata_review_retries_without_rewriting_original_response(self):
        bad, good = deepcopy(self.response), deepcopy(self.response)
        bad["consistency_checks"][0]["plan_evidence"] = [{"plan_field_id": "state_transition:invented", "quote": "new venue"}]
        with tempfile.TemporaryDirectory() as tmp:
            r.base.write(Path(tmp) / "sources" / f"{ROOT}.json", {"turns": HISTORY})
            caller = FakeCaller(tmp, [("event_state_extractor", bad), ("event_state_extractor", good)])
            result = r.extract_event_state(caller, Path(tmp), ROOT, B, HISTORY, [self.card], self.scene,
                                           {"facts": []}, plan_attempt=1)
            self.assertEqual(result, good)
            self.assertEqual(r.base.read(caller.calls[0]["folder"] / "VALIDATION.json")["status"], "REJECT")
            context = audit.decode_context(caller.calls[1]["payload"], "\nEVENT_STATE_INPUT_JSON\n")
            self.assertEqual(context["targeted_repair"]["previous_raw"], json.dumps(bad))
            self.assertEqual(bad["consistency_checks"][0]["plan_evidence"][0]["plan_field_id"], "state_transition:invented")
            audit.review_instruction_tail(caller.calls[1]["payload"], "event_state_extractor", context, cards=[self.card])

    def test_actual_v93_dependency_context_review_replays_without_archive_mutation(self):
        deliveries = Path(__file__).resolve().parents[2]
        folder = (deliveries / "h4_review_reference_minfix_v93/expansion_engine/runs/four_other_roots_B_scene01"
                  / "requests/009-cc-train-episode-14304-B-S01-event-state-1-2")
        files = [folder / name for name in ("REQUEST.json", "RESPONSE.json", "VALIDATION.json")]
        hashes = {str(path): sha256(path.read_bytes()).hexdigest() for path in files}
        request = json.loads(files[0].read_text())
        context = audit.decode_context(request, "\nEVENT_STATE_INPUT_JSON\n")
        raw = json.loads(files[1].read_text())["text"]
        with self.assertRaises(r.event_contract.EventStateValidationError) as caught:
            r.event_contract.validate(raw, context["root_id"], B, context["scene_plan"],
                                      context["cumulative_history"], cards=context["dataset_cards"])
        self.assertTrue(any("function_check" in row.get("missing", []) or
                            row.get("path", "").endswith("function_check")
                            for row in caught.exception.errors))
        self.assertEqual([row["mapped_development_indices"] for row in json.loads(raw)["card_checks"][0]["dependency_checks"]],
                         [[1], [2], [4]])
        self.assertEqual({str(path): sha256(path.read_bytes()).hexdigest() for path in files}, hashes)
        # This proves only that previously legitimate context references are
        # recordable. The preserved compatible verdicts are not certified true.


if __name__ == "__main__":
    unittest.main()
