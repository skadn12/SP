"""Offline contract tests; synthetic records are not source-reviewed pilot data."""

from __future__ import annotations

import copy
import json
import unittest

import b_arm


class BArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = {"root_id": "fixture", "source_sha256": "source-hash", "turns": []}
        for index in range(1, 13):
            self.surface["turns"].append({"turn_ref": f"R01:S01:T{index:03d}",
                                          "speaker": "Ada", "text": f"At checkpoint {index}, the plan was stated."})
        for index, name in enumerate("ABCD", 13):
            self.surface["turns"].append({"turn_ref": f"R01:S01:T{index:03d}",
                                          "speaker": "Ada", "text": f"I completed event {name}."})
        for index, clause in enumerate(("A happened before B.", "B happened before C.",
                                        "C happened before D."), 17):
            self.surface["turns"].append({"turn_ref": f"R01:S01:T{index:03d}",
                                          "speaker": "Ada", "text": clause})
        self.by_ref = {row["turn_ref"]: row for row in self.surface["turns"]}

    def quote(self, index: int) -> dict:
        row = self.by_ref[f"R01:S01:T{index:03d}"]
        return {"ref": row["turn_ref"], "speaker": row["speaker"], "quote": row["text"]}

    def relation(self, index: int) -> dict:
        return {"relation_id": f"R{index:03d}", "relation_type": "request_withdrawal",
                "actor": "Ada", "object_id": f"plan-{index}", "version": "",
                "action": "request", "scope": f"plan {index}",
                "state_before": "proposed", "state_after": "cancelled",
                "occurrence_time": "", "report_time": "",
                "checkpoints": [self.quote(index), self.quote(index + 6)]}

    def occurrence(self, index: int) -> dict:
        return {"event_id": f"E{index:03d}", "actor": "Ada", "object_id": f"event-{index}",
                "version": "", "action": f"step {'ABCD'[index - 1]}", "status": "completed",
                "occurrence_time": "", "report_time": "",
                "evidence": self.quote(index + 12)}

    def edge(self, index: int) -> dict:
        return {"edge_id": f"O{index:03d}", "before_event_id": f"E{index:03d}",
                "after_event_id": f"E{index + 1:03d}", "temporal_evidence": self.quote(index + 16)}

    def payload(self) -> dict:
        return {"relations": [self.relation(i) for i in range(1, 7)],
                "occurrences": [self.occurrence(i) for i in range(1, 5)],
                "order_edges": [self.edge(i) for i in range(1, 4)], "insufficiency_reason": ""}

    def reviewed(self) -> tuple[dict, dict]:
        validated = b_arm.validate_extraction(json.dumps(self.payload()), self.surface)
        review = b_arm.review_template(validated)
        review["reviewer"] = "fixture reviewer"
        for assessment in review["assessments"]:
            assessment.update(decision="supported", reason="Checked fixture evidence and strict scope.")
        return validated, review

    def test_prompt_projects_only_visible_turns(self) -> None:
        self.surface["turns"][0]["hidden_event_key"] = "do-not-disclose"
        prompt = b_arm.extraction_prompt(self.surface)
        self.assertNotIn("hidden_event_key", prompt)
        self.assertNotIn("do-not-disclose", prompt)
        self.assertIn("R01:S01:T001", prompt)

    def test_invalid_record_does_not_discard_valid_siblings(self) -> None:
        payload = self.payload()
        payload["relations"][0]["checkpoints"][1]["quote"] = "invented quote"
        payload["occurrences"][0]["status"] = "scheduled"
        validated = b_arm.validate_extraction(json.dumps(payload), self.surface)
        self.assertEqual(len(validated["records"]["relations"]), 5)
        self.assertEqual(len(validated["records"]["occurrences"]), 3)
        self.assertEqual(len(validated["records"]["order_edges"]), 2)
        invalid = [row["reason"] for row in validated["assessments"] if row["status"] == "mechanically_invalid"]
        self.assertIn("evidence_quote_not_exact", invalid)
        self.assertIn("occurrence_not_observed", invalid)
        self.assertIn("edge_event_missing_or_invalid", invalid)

    def test_review_is_complete_and_hash_bound(self) -> None:
        validated, review = self.reviewed()
        accepted = b_arm.select_accepted(validated, review)
        self.assertEqual(len(accepted["relations"]), 6)
        edited = copy.deepcopy(validated)
        edited["records"]["relations"][0]["state_after"] = "completed"
        with self.assertRaisesRegex(ValueError, "review_binding_invalid"):
            b_arm.select_accepted(edited, review)
        incomplete = copy.deepcopy(review)
        incomplete["assessments"].pop()
        with self.assertRaisesRegex(ValueError, "review_incomplete"):
            b_arm.select_accepted(validated, incomplete)

    def test_pool_requires_reviewed_direct_edges_for_unique_order(self) -> None:
        validated, review = self.reviewed()
        accepted = b_arm.select_accepted(validated, review)
        pool = b_arm.build_pool(accepted, self.surface)
        self.assertEqual(len(pool["pairs"]), 6)
        self.assertEqual(len(pool["ordering_chains"]), 1)
        self.assertEqual(pool["ordering_chains"][0]["event_ids"], ["E001", "E002", "E003", "E004"])
        accepted["order_edges"].pop()
        no_chain = b_arm.build_pool(accepted, self.surface)
        self.assertEqual(no_chain["ordering_chains"], [])
        self.assertEqual(no_chain["ordering_abstain_reason"], "no_four_event_unique_explicit_chain")

    def test_selection_uses_only_ids_and_program_derives_gold(self) -> None:
        validated, review = self.reviewed()
        pool = b_arm.build_pool(b_arm.select_accepted(validated, review), self.surface)
        raw = json.dumps({"questions": [{"selected_options": [
            {"pair_id": f"P{i:03d}", "orientation": "as_stated" if i in (1, 3) else "contradicted"}
            for i in range(1, 6)]}], "abstain_reason": "Only one independent composition."})
        result = b_arm.render_selection(raw, pool)
        question = result["questions"][0]
        self.assertEqual(question["gold"], ["A", "C"])
        self.assertEqual(question["options"]["B"],
                         "For plan 2, the status of Ada's request on plan-2 changed from cancelled to proposed across the two checkpoints.")
        self.assertNotIn("did not", " ".join(question["options"].values()))
        self.assertEqual(validated["rendered_relation_pairs"][1]["contradicted_claim"], question["options"]["B"])
        self.assertFalse(question["training_promoted"])
        order = b_arm.render_selection(json.dumps({"selected_chain_id": "C001", "abstain_reason": ""}),
                                       pool, "ordering")["questions"][0]
        self.assertNotEqual(order["gold"], ["I1", "I2", "I3", "I4"])
        self.assertEqual([order["display_event_ids"][int(label[1:]) - 1] for label in order["gold"]],
                         ["E001", "E002", "E003", "E004"])
        self.assertEqual(len(order["adjacent_edge_evidence"]), 3)
        for label in ("I1", "I2", "I3", "I4"):
            event_id = order["event_evidence"][label]["event_id"]
            self.assertEqual(event_id, order["display_event_ids"][int(label[1:]) - 1])
            ref = order["event_evidence"][label]["evidence"]["ref"]
            quote = order["event_evidence"][label]["evidence"]["quote"]
            self.assertIn(quote, self.by_ref[ref]["text"])
        self.assertEqual(validated["rendered_occurrences"][0]["item_text"],
                         "Ada completed step A for event-1.")

    def test_persistent_state_uses_same_wording_for_contradicted_alternative(self) -> None:
        row = self.relation(1)
        row["state_after"] = row["state_before"]
        supported, contradicted = b_arm._relation_claims(row)
        self.assertIn("remained proposed across", supported)
        self.assertIn("remained offered across", contradicted)
        self.assertNotIn("did not", supported + contradicted)
        self.assertEqual(b_arm._relation_claims(row), (supported, contradicted))

    def test_global_temporal_cycle_abstains(self) -> None:
        validated, review = self.reviewed()
        accepted = b_arm.select_accepted(validated, review)
        accepted["order_edges"].append({"edge_id": "O004", "before_event_id": "E004",
                                        "after_event_id": "E001", "temporal_evidence": self.quote(17)})
        pool = b_arm.build_pool(accepted, self.surface)
        self.assertEqual(pool["ordering_chains"], [])

    def test_unreviewed_claim_or_duplicate_selection_fails(self) -> None:
        validated, review = self.reviewed()
        pool = b_arm.build_pool(b_arm.select_accepted(validated, review), self.surface)
        selection = {"questions": [{"selected_options": [
            {"pair_id": "P001", "orientation": "as_stated"},
            {"pair_id": "P001", "orientation": "contradicted"},
            {"pair_id": "P002", "orientation": "as_stated"},
            {"pair_id": "P003", "orientation": "contradicted"}]}], "abstain_reason": "One only."}
        with self.assertRaisesRegex(ValueError, "pair_or_orientation_invalid"):
            b_arm.render_selection(json.dumps(selection), pool)
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            b_arm.validate_extraction('{"relations":[],"relations":[],"occurrences":[],"order_edges":[],"insufficiency_reason":""}',
                                      self.surface)

    def test_model_authored_option_text_is_rejected(self) -> None:
        payload = self.payload()
        payload["relations"][0]["contradicted_claim"] = "A convenient false sentence."
        payload["occurrences"][0]["claim"] = "A convenient ordering item."
        validated = b_arm.validate_extraction(json.dumps(payload), self.surface)
        self.assertEqual(validated["assessments"][0]["status"], "mechanically_invalid")
        self.assertEqual(validated["assessments"][0]["reason"], "record_shape_invalid")
        occurrence_audit = next(row for row in validated["assessments"] if row["kind"] == "occurrences")
        self.assertEqual(occurrence_audit["status"], "mechanically_invalid")


if __name__ == "__main__":
    unittest.main()
