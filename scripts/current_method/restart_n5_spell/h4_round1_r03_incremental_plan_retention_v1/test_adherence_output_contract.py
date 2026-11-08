from copy import deepcopy
import json
import unittest

from .adherence_output_contract import normalize_references, output_schema, render_contract, validate_response
from . import test_adherence_scopes
from .retention import ADHERENCE_VERSION, assess_retention


class InputContractTests(unittest.TestCase):
    def setUp(self):
        case = test_adherence_scopes.EvidenceScopeTests()
        case.setUp()
        self.case = case
        self.schema = output_schema(parent=case.parent, child=case.child, requested_conditions=case.conditions)

    def check(self, audit):
        return validate_response(audit, schema=self.schema, scene_order=["R03:S01"])

    def test_valid_scoped_response_and_required_fields(self):
        self.assertTrue(self.check(self.case.audit)["valid"])
        for field in self.case.row:
            audit = deepcopy(self.case.audit)
            del audit["scene_audits"][0][field]
            self.assertFalse(self.check(audit)["valid"], field)

    def test_empty_conditions_cannot_become_goal_conditions(self):
        self.case.row["condition_results"] = [{"condition": "An invented criterion", "status": "established", "turn_refs": [self.case.current]}]
        self.assertFalse(self.check(self.case.audit)["valid"])
        normalized = normalize_references(json.dumps(self.case.audit))
        self.assertEqual(normalized["formatted_output"], self.case.audit)
        result = assess_retention(parent=self.case.parent, child=self.case.child, plan=self.case.plan,
            raw_audit=normalized["formatted_output"], requested_conditions=self.case.conditions, audit_version=ADHERENCE_VERSION)
        self.assertFalse(result["decisions"][0]["retained"])
        self.assertTrue(result["decisions"][0]["semantic_reaudit_required"])

    def test_only_correct_reference_sources_are_permitted(self):
        for field, invalid in (("history_evidence_refs", self.case.current), ("goal_evidence_refs", "invented")):
            audit = deepcopy(self.case.audit)
            audit["scene_audits"][0][field] = [invalid]
            self.assertFalse(self.check(audit)["valid"])
        self.case.row["actual_states"][0]["turn_refs"] = [self.case.historical]
        self.assertFalse(self.check(self.case.audit)["valid"])

    def test_order_and_extra_fields_are_rejected(self):
        audit = deepcopy(self.case.audit)
        audit["scene_audits"][0]["scene_ref"] = "R03:S02"
        self.assertFalse(self.check(audit)["valid"])
        audit = deepcopy(self.case.audit)
        audit["extra"] = "value"
        self.assertFalse(self.check(audit)["valid"])

    def test_deduplication_preserves_judgments_and_conditions(self):
        for field in ("goal_evidence_refs", "history_evidence_refs"):
            self.case.row[field] *= 2
        self.case.row["condition_results"] = [{"condition": "Invented", "status": "UNKNOWN", "turn_refs": []}]
        original = deepcopy(self.case.audit)
        result = normalize_references(json.dumps(original))
        self.assertEqual(result["raw_output"], original)
        self.assertEqual(len(result["transformations"]), 2)
        self.assertEqual(self.case.audit, original)
        after = result["formatted_output"]["scene_audits"][0]
        for field in set(after) - {"goal_evidence_refs", "history_evidence_refs"}:
            self.assertEqual(after[field], self.case.row[field])
        del after["reason"]
        again = normalize_references(result["formatted_output"])
        self.assertNotIn("reason", again["formatted_output"]["scene_audits"][0])

    def test_requested_condition_schema_uses_exact_supplied_text(self):
        self.case.conditions["R03:S01"] = ["An exact prerequisite"]
        self.schema = output_schema(parent=self.case.parent, child=self.case.child, requested_conditions=self.case.conditions)
        self.case.row["condition_results"] = [{"condition": "An exact prerequisite", "status": "established", "turn_refs": [self.case.historical]}]
        self.assertTrue(self.check(self.case.audit)["valid"])
        self.case.row["condition_results"][0]["condition"] += " changed"
        self.assertFalse(self.check(self.case.audit)["valid"])

    def test_renderer_keeps_full_dialogue_and_targets(self):
        self.case.plan["scenes"][0]["goal"] = "Discuss the route"
        generated = json.dumps({"scenes": [{"dialogue_turns": [{"speaker": "A", "text": "A new route is arranged today."}]}]})
        built = render_contract(parent=self.case.parent, plan=self.case.plan, child=self.case.child,
            requested_conditions=self.case.conditions, generated_response=generated)
        self.assertEqual(built["response_format"]["json_schema"]["schema"], self.schema)
        self.assertEqual(built["audit_version"], ADHERENCE_VERSION)
        self.assertIn('"condition_results_must_equal_empty_list":true', built["messages"][-1]["content"])
        data = json.loads(built["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
        self.assertEqual(data["planned_targets"], self.case.plan)
        self.assertEqual(data["authoritative_history"], self.case.parent["dialogue"]["turns"])

    def test_duplicate_keys_and_nonfinite_response_rejected(self):
        for raw in ('{"scene_audits":[],"scene_audits":[]}', '{"scene_audits":NaN}', '{"scene_audits":1e999}'):
            with self.assertRaises(ValueError):
                normalize_references(raw)

    def test_empty_history_has_no_allowed_historical_evidence(self):
        self.case.parent["dialogue"]["turns"] = []
        self.schema = output_schema(parent=self.case.parent, child=self.case.child, requested_conditions=self.case.conditions)
        self.assertFalse(self.check(self.case.audit)["valid"])
        self.case.row["history_evidence_refs"] = []
        self.assertTrue(self.check(self.case.audit)["valid"])


if __name__ == "__main__":
    unittest.main()
