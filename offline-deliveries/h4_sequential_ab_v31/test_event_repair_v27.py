"""Lossless, model-selected local repair of the actual v26 failure category."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as scan
import test_event_v19 as event_fixture


class TimelinePatchTests(unittest.TestCase):
    def setUp(self):
        fixture = event_fixture.EventV19Tests()
        fixture.setUp()
        self.fixture = fixture
        self.previous = deepcopy(fixture.result)
        self.previous["timeline"].append({"constraint_id": "c2", "status": "uncertain",
            "evidence_basis": "inferred", "statement": "A new software problem is proposed.",
            "evidence": [], "implication": "This is a proposed new development.", "uncertainty": None})

    def repair(self, previous=None):
        previous = self.previous if previous is None else previous
        fixture = self.fixture
        with self.assertRaises(scan.EventStateValidationError) as caught:
            scan.validate(previous, fixture.root, fixture.route, fixture.scene, fixture.history)
        return scan.build_repair(caught.exception, previous, fixture.history)

    def patch(self, choice="planned_development", **extra):
        return {"timeline_repairs": [{"constraint_id": "c2", "choice": choice, **extra}],
                "plan_check_repairs": []}

    def merge(self, patch=None, previous=None, repair=None):
        previous = self.previous if previous is None else previous
        repair = self.repair(previous) if repair is None else repair
        return scan.merge_event_repair(previous, self.patch() if patch is None else patch,
                                       repair, self.fixture.history)

    def validate(self, result):
        fixture = self.fixture
        return scan.validate(result, fixture.root, fixture.route, fixture.scene, fixture.history)

    def test_planned_development_relocated_verbatim_without_claim_mutation(self):
        original = deepcopy(self.previous)
        patch = self.patch()
        before_patch = deepcopy(patch)
        repair = self.repair()
        merged = self.merge(patch, repair=repair)
        self.assertEqual(self.previous, original)
        self.assertEqual(patch, before_patch)
        self.assertEqual(merged["timeline"], original["timeline"][:1])
        self.assertEqual(merged["nonhistorical_entries"], [{"source_constraint_id": "c2",
            "kind": "planned_development", "scope": "scene_plan", "entry": original["timeline"][1]}])
        self.assertEqual(merged["plan_checks"], original["plan_checks"])
        self.assertEqual(merged["route_fulfillment"], original["route_fulfillment"])
        self.validate(merged)
        self.assertTrue(scan.validate_repair_preservation(original, merged, repair))
        receipt = scan.event_repair_provenance(original, patch, repair, merged)
        self.assertEqual(receipt["constraint_id_remap"], {"c1": "c1"})
        self.assertEqual(receipt["classification_source"], "explicit_model_patch_choice")
        self.assertFalse(receipt["semantic_certified"])

    def test_not_found_choice_is_model_declared_not_inferred_from_wording(self):
        self.previous["timeline"][1]["statement"] = "No earlier software problem was mentioned."
        for choice in ("not_found", "planned_development"):
            with self.subTest(choice=choice):
                merged = self.merge(self.patch(choice))
                self.assertEqual(merged["nonhistorical_entries"][0]["kind"], choice)
                self.assertEqual(merged["nonhistorical_entries"][0]["entry"], self.previous["timeline"][1])
                self.validate(merged)

    def test_historical_claim_requires_explicit_model_selected_existing_references(self):
        patch = self.patch("historical_claim", evidence=[{"turn_ref": "R02:A:S01:T001"}])
        merged = self.merge(patch)
        self.assertNotIn("nonhistorical_entries", merged)
        self.assertEqual(merged["timeline"][1]["evidence"], patch["timeline_repairs"][0]["evidence"])
        self.assertEqual(merged["timeline"][1]["statement"], self.previous["timeline"][1]["statement"])
        # Literal citation admission is deliberately not semantic entailment.
        self.validate(merged)
        for evidence in ([], ["UNKNOWN"], [{"turn_ref": "P01:T001", "quote": "invented"}],
                         ["P01:T001", "P01:T001"]):
            with self.subTest(evidence=evidence), self.assertRaises(scan.EventStateValidationError):
                self.merge(self.patch("historical_claim", evidence=evidence))

    def test_mixed_namespaces_remain_verbatim_without_normalization(self):
        self.fixture.history.append({"turn_ref": "fixture:P02:T001", "text": "Another report."})
        self.previous["segments"].append({"segment_id": "fixture:P02", "none_reason": "Other report."})
        for ref in ("P01:T001", "R02:A:S01:T001", "fixture:P02:T001"):
            with self.subTest(ref=ref):
                merged = self.merge(self.patch("historical_claim", evidence=[ref]))
                self.assertEqual(merged["timeline"][1]["evidence"], [ref])
                self.validate(merged)
        with self.assertRaises(scan.EventStateValidationError):
            self.merge(self.patch("historical_claim", evidence=["fixture:R02:A:S01:T001"]))

    def test_empty_evidence_not_automatically_relocated_and_mixed_faults_use_full_repair(self):
        previous = deepcopy(self.previous)
        repair = self.repair()
        self.assertEqual(repair["repair_mode"], "timeline_entry_patch")
        self.assertNotIn("nonhistorical_entries", previous)
        error = scan.EventStateValidationError(repair["errors"] + [
            {"code": "other_fault", "path": "route_fulfillment", "actual": None}])
        other = scan.build_repair(error, previous, self.fixture.history)
        self.assertNotEqual(other["repair_mode"], "timeline_entry_patch")

    def test_exact_patch_coverage_and_no_extra_semantic_fields(self):
        mutations = [
            {"timeline_repairs": [], "plan_check_repairs": []},
            {"timeline_repairs": [self.patch()["timeline_repairs"][0]] * 2, "plan_check_repairs": []},
            {"timeline_repairs": [{"constraint_id": "c1", "choice": "not_found"}], "plan_check_repairs": []},
            {**self.patch(), "route_fulfillment": {"satisfied": True}},
            {"timeline_repairs": [{"constraint_id": "c2", "choice": "planned_development",
                                    "statement": "Changed"}], "plan_check_repairs": []},
            {"timeline_repairs": [{"constraint_id": "c2", "choice": "delete"}], "plan_check_repairs": []},
        ]
        for patch in mutations:
            with self.subTest(patch=patch), self.assertRaises(scan.EventStateValidationError):
                self.merge(patch)
        # Some Qwen responses redundantly include an empty evidence array for a
        # nonhistorical choice. It carries no citation and is accepted losslessly.
        merged = self.merge(self.patch("not_found", evidence=[]))
        self.assertEqual(merged["nonhistorical_entries"][0]["kind"], "not_found")
        with self.assertRaises(scan.EventStateValidationError):
            self.merge(self.patch("not_found", evidence=["P01:T001"]))

    def test_dependent_reference_removal_requires_explicit_patch_and_preserves_negative(self):
        previous = deepcopy(self.previous)
        row = previous["plan_checks"][0]
        row.update(verdict="revise", comparison_kind="historical_assertion", constraint_ids=["c1", "c2"],
            conflict={"type": "unsupported_prior_assertion", "historical_claim": "Claim was not established.",
                      "proposed_claim": "Claims prior software study.", "correction": "Describe a new development."},
            reason="Preserve this negative finding.")
        repair = self.repair(previous)
        with self.assertRaises(scan.EventStateValidationError):
            self.merge(previous=previous, repair=repair)
        patch = self.patch()
        patch["plan_check_repairs"] = [{"development_index": 1, "constraint_ids": ["c1"]}]
        merged = self.merge(patch, previous, repair)
        expected = deepcopy(row)
        expected["constraint_ids"] = ["c1"]
        self.assertEqual(merged["plan_checks"][0], expected)
        self.assertTrue(scan.requires_replan(self.validate(merged)))
        for refs in ([], ["c2"], ["c1", "c1"]):
            invalid = deepcopy(patch)
            invalid["plan_check_repairs"][0]["constraint_ids"] = refs
            with self.subTest(refs=refs), self.assertRaises(scan.EventStateValidationError):
                self.merge(invalid, previous, repair)
        invalid = deepcopy(patch)
        invalid["plan_check_repairs"][0]["verdict"] = "compatible"
        with self.assertRaises(scan.EventStateValidationError):
            self.merge(invalid, previous, repair)

    def test_middle_row_relocation_sequential_rename_and_negative_refs_preserved(self):
        previous = deepcopy(self.previous)
        third = deepcopy(previous["timeline"][0])
        third["constraint_id"] = "c3"
        previous["timeline"].append(third)
        previous["plan_checks"][0]["constraint_ids"] = ["c1", "c2", "c3"]
        previous["plan_checks"][1]["constraint_ids"] = ["c3"]
        patch = self.patch()
        patch["plan_check_repairs"] = [{"development_index": 1, "constraint_ids": ["c1", "c3"]}]
        repair = self.repair(previous)
        merged = self.merge(patch, previous, repair)
        self.assertEqual([row["constraint_id"] for row in merged["timeline"]], ["c1", "c2"])
        self.assertEqual(merged["plan_checks"][0]["constraint_ids"], ["c1", "c2"])
        self.assertEqual(merged["plan_checks"][1]["constraint_ids"], ["c2"])
        self.assertEqual(merged["nonhistorical_entries"][0]["entry"], previous["timeline"][1])
        self.assertTrue(scan.validate_repair_preservation(previous, merged, repair))
        self.validate(merged)

    def test_preservation_rejects_lost_claim_changed_observation_negative_or_route(self):
        self.previous["scan_observations"] = [{"observation_id": "o1", "kind": "not_found_in_supplied_history",
            "scope": "cumulative_history", "statement": "Limited original observation."}]
        repair = self.repair()
        merged = self.merge(repair=repair)
        for field in ("claim", "verdict", "reason", "route", "observation", "entry"):
            changed = deepcopy(merged)
            if field == "claim": changed["timeline"][0]["statement"] = "Changed"
            elif field == "verdict": changed["plan_checks"][0]["verdict"] = "revise"
            elif field == "reason": changed["plan_checks"][0]["reason"] = "Changed"
            elif field == "route": changed["route_fulfillment"]["reason"] = "Changed"
            elif field == "observation": changed["scan_observations"] = []
            else: changed["nonhistorical_entries"][0]["entry"]["implication"] = "Changed"
            with self.subTest(field=field), self.assertRaises(scan.EventStateValidationError):
                scan.validate_repair_preservation(self.previous, changed, repair)

    def test_prompt_trusted_local_controls_outside_input_and_only_patch_output(self):
        fixture = self.fixture
        request = scan.build_request(fixture.root, fixture.route, fixture.scene, fixture.history,
                                     [], {}, repair=self.repair())
        trusted, untrusted = request["messages"][0]["content"].split("\nEVENT_STATE_INPUT_JSON\n")
        self.assertIn("TRUSTED LOCAL REPAIR MODE: timeline_entry_patch", trusted)
        self.assertIn("Do not output a full scan", trusted)
        self.assertIn("THREE SOURCE CATEGORIES", trusted)
        context = json.loads(untrusted)
        self.assertEqual(context["cumulative_history"], fixture.history)
        self.assertEqual(context["targeted_repair"]["faulty_timeline_entries"], self.previous["timeline"][1:])

    def test_real_v26_s02_s03_failures_have_lossless_model_patch_path(self):
        run = Path(__file__).parent.parent / "h4_sequential_ab_v26" / "runs" / "inherited_a2_b3_01"
        history = json.loads((run / "roots/cc-train-episode-104814/B/INPUT.json").read_text())["turns"]
        for slot, request_number, plan_number, choice in (("S02", "010", "009", "not_found"),
                                                       ("S03", "016", "015", "planned_development")):
            with self.subTest(slot=slot):
                directory = next((run / "requests").glob(request_number + "-*"))
                raw = json.loads((directory / "RESPONSE.json").read_text())["text"]
                previous = json.loads(raw)
                plan_directory = next((run / "requests").glob(plan_number + "-*"))
                scene = json.loads(json.loads((plan_directory / "RESPONSE.json").read_text())["text"])["scenes"][0]
                with self.assertRaises(scan.EventStateValidationError) as caught:
                    scan.validate(raw, previous["root_id"], previous["route"], scene, history)
                repair = scan.build_repair(caught.exception, raw, history)
                target = repair["faulty_timeline_entries"][0]
                removed = target["constraint_id"]
                patch = {"timeline_repairs": [{"constraint_id": removed, "choice": choice}],
                    "plan_check_repairs": [{"development_index": check["development_index"],
                        "constraint_ids": [ref for ref in check["constraint_ids"] if ref != removed]}
                        for check in previous["plan_checks"] if removed in check["constraint_ids"]]}
                merged = scan.merge_event_repair(raw, patch, repair, history)
                self.assertEqual(merged["nonhistorical_entries"][0]["entry"], target)
                self.assertEqual(merged["route_fulfillment"], previous["route_fulfillment"])
                self.assertTrue(scan.validate_repair_preservation(raw, merged, repair))
                checked = scan.validate(merged, previous["root_id"], previous["route"], scene, history)
                self.assertEqual(len(checked["timeline"]), len(previous["timeline"]) - 1)
                # This is offline interface validation, never a live semantic acceptance.


if __name__ == "__main__":
    unittest.main()
