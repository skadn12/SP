"""Offline regressions for the opt-in initial Planner cap-9 policy.

These tests exercise only schema/compiler/template and budget code.  They never
invoke a model or network transport; cap-8 remains the default contract.
"""

from copy import deepcopy
import unittest

from . import planner_relations, planner_templates
from .planner_candidate_policy import VERSION, candidate_limit, gate_budget
from .test_root_planning import root_fixture_bundle
from .test_shared_planning import shared_initial
from .root_planning import prepare_root_original
from .run import offline_network_guard
from .fixtures import example_budget


class PlannerCap9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the frozen local fixture and its canonical speaker conversion;
        # this is strictly an offline compiler test.
        guard = offline_network_guard()
        guard.__enter__()
        cls.addClassCleanup(guard.__exit__, None, None, None)
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_root_original(cls.bundle)

    def _initial(self, count):
        value = shared_initial(self.prepared["original"], self.bundle["parent"])
        while len(value["new_development_candidates"]) < count:
            row = deepcopy(value["new_development_candidates"][-1])
            index = len(value["new_development_candidates"]) + 1
            # Synthetic boundary fixture only; no semantic quality claim.
            row["goal"] += f" (cap9 candidate {index})"
            row["transition_from_previous"] += f" (candidate {index})"
            row["before_state"] += f" (candidate {index})"
            row["required_transition"] += f" (candidate {index})"
            row["after_state"] += f" (candidate {index})"
            value["new_development_candidates"].append(row)
        return value

    def test_cap8_and_cap9_compile_but_cap10_is_rejected(self):
        for count, cap in ((8, 8), (9, 9)):
            with self.subTest(count=count, cap=cap):
                raw = self._initial(count)
                result = planner_relations.compile_initial_output(
                    raw, parent=self.bundle["parent"], candidates=self.bundle["candidates"],
                    verified_tracks=self.bundle["verified_tracks"], max_new_candidates=cap)
                self.assertTrue(result["valid"], result.get("errors"))
        ten = planner_relations.compile_initial_output(
            self._initial(10), parent=self.bundle["parent"], candidates=self.bundle["candidates"],
            verified_tracks=self.bundle["verified_tracks"], max_new_candidates=9)
        self.assertFalse(ten["valid"])
        self.assertTrue(any("too long" in str(error) or "max" in str(error)
                            for error in ten.get("errors", [])))

    def test_legacy_default_cap8_still_rejects_ninth(self):
        result = planner_relations.compile_initial_output(
            self._initial(9), parent=self.bundle["parent"], candidates=self.bundle["candidates"],
            verified_tracks=self.bundle["verified_tracks"])
        self.assertFalse(result["valid"])

    def test_nd09_backward_reference_and_self_reference(self):
        raw = self._initial(9)
        raw["new_development_candidates"][8]["dependencies"] = [{
            "source_candidate_ref": None,
            "output_ref": "/new_development_candidates/7",
            "relation_kind": planner_relations.NATURAL,
            "condition": None,
        }]
        valid = planner_relations.validate_initial_output(raw, max_new_candidates=9,
                                                          contract_version=planner_relations.CAP9_VERSION)
        self.assertTrue(valid["valid"], valid["errors"])
        before = deepcopy(raw)
        compiled = planner_relations.compile_initial_output(raw, parent=self.bundle["parent"],
            candidates=self.bundle["candidates"], verified_tracks=self.bundle["verified_tracks"], max_new_candidates=9)
        self.assertTrue(compiled["valid"], compiled["errors"])
        self.assertEqual(raw, before)
        self.assertIn("ND09:C01", compiled["prepared"]["candidate_registry"])
        self.assertEqual(compiled["relation_sidecar"]["relations"][-1]["predecessor_phase_id"], "initial:ND08:C01")
        self.assertLessEqual(len(compiled["prepared"]["compiled"]["scenes"]), 6)
        raw["new_development_candidates"][8]["dependencies"][0]["output_ref"] = \
            "/new_development_candidates/8"
        invalid = planner_relations.validate_initial_output(raw, max_new_candidates=9,
                                                            contract_version=planner_relations.CAP9_VERSION)
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("self_output_dependency_forbidden" in error
                            for error in invalid["errors"]))

    def test_soft_and_hard_relation_contracts_are_unchanged(self):
        raw = self._initial(9)
        row = raw["new_development_candidates"][8]
        row["dependencies"] = [{"source_candidate_ref": None,
                                 "output_ref": "/new_development_candidates/7",
                                 "relation_kind": planner_relations.NATURAL,
                                 "condition": None}]
        self.assertTrue(planner_relations.validate_initial_output(
            raw, max_new_candidates=9, contract_version=planner_relations.CAP9_VERSION)["valid"])
        row["dependencies"][0].update(
            relation_kind=planner_relations.HARD,
            condition="The earlier candidate must establish the actual result.")
        self.assertTrue(planner_relations.validate_initial_output(
            raw, max_new_candidates=9, contract_version=planner_relations.CAP9_VERSION)["valid"])
        row["dependencies"][0]["condition"] = None
        self.assertFalse(planner_relations.validate_initial_output(
            raw, max_new_candidates=9, contract_version=planner_relations.CAP9_VERSION)["valid"])

    def test_cap9_template_targets_eight_with_hard_nine(self):
        base = {
            "feedback_view": {"root_id": self.bundle["identity"]["root_id"], "items": []},
            "dialogue": {"root_id": self.bundle["identity"]["root_id"],
                         "turns": [{**row, "turn_ref": row.get("turn_id", "")}
                                   for row in self.bundle["parent"]["dialogue"]["turns"]]},
            "memory_overview": {}, "memory_candidates": self.bundle["candidates"],
            "verified_tracks": self.bundle["verified_tracks"],
        }
        cap8 = planner_templates.render_initial_planner(**base, max_new_candidates=8)["prompt"]
        cap9 = planner_templates.render_initial_planner(**base, max_new_candidates=9)["prompt"]
        self.assertIn("at most eight new-development candidates", cap8)
        self.assertIn("never output a ninth", cap8)
        self.assertIn("Target eight new-development candidates", cap9)
        self.assertIn("hard maximum is nine", cap9)
        self.assertIn("never output ten", cap9)
        self.assertNotIn("never output a ninth", cap9)
        self.assertEqual(planner_templates.CAP9_VERSION,
                         "planner_minimal_relations_filtered_feedback_target8_cap9_v1")
        self.assertEqual(planner_relations.CAP9_VERSION,
                         "shared_planner_relations_v2_cap9")

    def test_policy_requires_initial_only_and_gate_budget_is_twelve(self):
        self.assertEqual(candidate_limit({}), 8)
        self.assertEqual(candidate_limit({"candidate_policy_version": VERSION,
                                           "append_enabled": False}), 9)
        with self.assertRaises(ValueError):
            candidate_limit({"candidate_policy_version": VERSION, "append_enabled": True})
        budget = gate_budget(example_budget(), max_new_candidates=9)
        self.assertEqual(budget["role_requests"]["gate"], 12)
        self.assertEqual(budget["total_requests"], 13)
        self.assertEqual(budget["max_generated_tokens"],
                         budget["max_output_tokens"]["planner"] +
                         12 * budget["max_output_tokens"]["gate"])


if __name__ == "__main__":
    unittest.main()
