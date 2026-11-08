"""Shared Planner schema regressions, without model or network requests."""
from copy import deepcopy
import json
import unittest

from .shared_planner_contract import (
    COLLECTIONS, ROOT_SHARED_PROMPT_VERSION, VERSION, iter_candidates,
    output_examples, output_schema, render_output_contract, strip_relations, validate_output,
    parse_output,
)


class SharedPlannerContractTests(unittest.TestCase):
    def setUp(self):
        self.value = output_examples()[0]

    def assert_valid(self, value):
        result = validate_output(value)
        self.assertTrue(result["valid"], result["errors"])

    def assert_invalid(self, value):
        result = validate_output(value)
        self.assertFalse(result["valid"], value)
        self.assertTrue(result["errors"])

    def test_parser_rejects_duplicate_keys_and_non_json_constants(self):
        self.assertEqual(parse_output(json.dumps(self.value)), self.value)
        for text in ('{"round_goal":"first","round_goal":"second"}',
                     '{"row":{"source_candidate_ref":"ND01:C01","source_candidate_ref":null}}',
                     '{"round_goal":NaN}', '{"round_goal":Infinity}',
                     '```json\n{}\n```', '{} {}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_output(text)

    def test_append_example_uses_same_schema_and_explicit_prerequisites(self):
        append = output_examples()[2]
        self.assert_valid(append)
        row = append['selected_continuations'][0]
        self.assertEqual(row['source_candidate_ref'], 'ND08:C01')
        self.assertIsNone(row['verified_track_ref'])
        self.assertTrue(row['dependencies'])
        rendered = render_output_contract()
        self.assertNotIn('APPEND-ONLY OUTPUT EXAMPLE', rendered)
        self.assertNotIn(row['goal'], rendered)

    def test_schema_and_examples_are_independent_valid_json(self):
        self.assertEqual(VERSION, "shared_planner_output_v2")
        self.assertEqual(ROOT_SHARED_PROMPT_VERSION,
                         "incremental_retention_v1.supplement.shared_planner_v2")
        for example in output_examples():
            self.assert_valid(json.loads(json.dumps(example)))
        self.assertEqual(output_examples()[1]["selected_continuations"], [])
        self.assertEqual(output_examples()[1]["new_development_candidates"], [])
        changed = output_schema()
        changed["properties"].clear()
        self.assertEqual(set(output_schema()["properties"]),
                         {"round_goal", *COLLECTIONS})
        changed_example = output_examples()[0]
        changed_example["selected_continuations"][0]["goal"] = "edited"
        self.assertNotEqual(output_examples()[0], changed_example)

    def test_rendered_schema_has_no_example_values_or_output_instances(self):
        rendered = render_output_contract()
        decoder = json.JSONDecoder()
        schema, _ = decoder.raw_decode(rendered.split("JSON SCHEMA\n", 1)[1])
        self.assertEqual(schema, output_schema())
        self.assertNotIn('"examples"', rendered)
        self.assertNotIn('OUTPUT EXAMPLE', rendered)
        self.assertNotIn('workshop', rendered.lower())
        self.assertNotIn('ND08:C01', rendered)
        self.assertNotIn('VT01', rendered)
        self.assertNotIn('Elena', rendered)
        for collection in COLLECTIONS:
            for key in ('goal', 'required_transition', 'after_state', 'interaction_beats'):
                value = output_examples()[0][collection][0][key]
                for text in value if isinstance(value, list) else [value]:
                    self.assertNotIn(text, rendered)
        self.assertIn('"description"', rendered)

    def test_previous_patch_shape_and_phase_fields_are_rejected(self):
        self.assert_invalid({"status": "PATCH", "original_plan_sha256": "a" * 64,
                             "phases": [], "new_developments": [], "reason": "example"})
        for field in ("phase_id", "candidate", "status", "original_plan_sha256",
                      "source_candidate_id", "original_goal"):
            with self.subTest(field=field):
                value = deepcopy(self.value)
                value["new_development_candidates"][0][field] = "phase:ND08:C01"
                self.assert_invalid(value)
        value = deepcopy(self.value)
        value["new_development_candidates"][0] = {
            "candidate": value["new_development_candidates"][0], "dependencies": []}
        self.assert_invalid(value)

    def test_missing_extra_or_misplaced_candidate_fields_are_rejected(self):
        for collection in COLLECTIONS:
            for field in self.value[collection][0]:
                with self.subTest(collection=collection, field=field):
                    value = deepcopy(self.value)
                    value[field] = value[collection][0].pop(field)
                    self.assert_invalid(value)
        for object_path in ((), ("selected_continuations", 0),
                            ("new_development_candidates", 0),
                            ("new_development_candidates", 0, "actor_bindings", 0)):
            value = deepcopy(self.value)
            target = value
            for part in object_path:
                target = target[part]
            target["unexpected"] = "extra"
            self.assert_invalid(value)

    def test_relations_are_required_on_both_candidate_types(self):
        for collection in COLLECTIONS:
            for field in ("source_candidate_ref", "dependencies"):
                value = deepcopy(self.value)
                del value[collection][0][field]
                self.assert_invalid(value)
        value = deepcopy(self.value)
        continuation = value["selected_continuations"][0]
        continuation["source_candidate_ref"] = "ND08:C01"
        continuation["verified_track_ref"] = None
        self.assert_valid(value)

    def test_dependency_ref_union_and_canonical_output_pointers(self):
        row = self.value["new_development_candidates"][0]
        for source, output in (("ND08:C01", None), (None, "/selected_continuations/0"),
                               (None, "/new_development_candidates/10")):
            row["dependencies"] = [{"source_candidate_ref": source, "output_ref": output,
                                    "condition": "An explicit prerequisite is established."}]
            self.assert_valid(self.value)
        for source, output in ((None, None), ("ND08:C01", "/selected_continuations/0"),
                               ("", None), ("  ", None),
                               (None, "phase:ND08:C01"),
                               (None, "/selected_continuations/00"),
                               (None, "/selected_continuations/-1"),
                               (None, "/selected_continuations/1.0"),
                               (None, "/selected_continuations/0\n"),
                               (None, "/selected_continuations/0/goal"),
                               (None, "/other/0")):
            with self.subTest(source=source, output=output):
                row["dependencies"] = [{"source_candidate_ref": source, "output_ref": output,
                                        "condition": "An explicit prerequisite is established."}]
                self.assert_invalid(self.value)

    def test_dependency_fields_and_condition_are_strict(self):
        dependency = {"source_candidate_ref": "ND08:C01", "output_ref": None,
                      "condition": "An explicit prerequisite is established."}
        for field in dependency:
            value = deepcopy(self.value)
            bad = deepcopy(dependency)
            del bad[field]
            value["new_development_candidates"][0]["dependencies"] = [bad]
            self.assert_invalid(value)
        for extra in ({"phase_id": "phase:followup_1"}, {"condition": " "}):
            value = deepcopy(self.value)
            value["new_development_candidates"][0]["dependencies"] = [{**dependency, **extra}]
            self.assert_invalid(value)

    def test_existing_structure_limits_are_preserved(self):
        for collection in COLLECTIONS:
            for field, bad_items in (
                    ("interaction_beats", ["one", "two"]),
                    ("interaction_beats", ["action"] * 8),
                    ("new_people", [{"name": "Elena", "relationship_basis": "Neighbor."}] * 3),
                    ("speaker_refs", []), ("speaker_refs", ["SP01", "SP01"]),
                    ("actor_bindings", [])):
                value = deepcopy(self.value)
                value[collection][0][field] = bad_items
                self.assert_invalid(value)
        for count in (0, 5):
            value = deepcopy(self.value)
            value["new_development_candidates"][0]["source_basis"] = ["Existing fact."] * count
            self.assert_invalid(value)
        value = deepcopy(self.value)
        value["new_development_candidates"] *= 9
        self.assert_invalid(value)
        value = deepcopy(self.value)
        value["selected_continuations"] *= 9
        self.assert_valid(value)

    def test_projection_and_candidate_order_preserve_input(self):
        before = deepcopy(self.value)
        entries = list(iter_candidates(self.value))
        self.assertEqual([(pointer, collection) for pointer, _, collection in entries],
                         [(f"/{collection}/0", collection) for collection in COLLECTIONS])
        legacy = strip_relations(self.value)
        for collection in COLLECTIONS:
            expected = {key: value for key, value in self.value[collection][0].items()
                        if key not in {"source_candidate_ref", "dependencies"}}
            self.assertEqual(legacy[collection][0], expected)
        legacy["selected_continuations"][0]["goal"] = "edited"
        self.assertEqual(self.value, before)


if __name__ == "__main__":
    unittest.main()
