"""Offline relation compilation and actual Planner-message regressions."""
from copy import deepcopy
import json
import unittest

from . import planner_relations as relations
from . import planner_templates as templates
from . import shared_planner_contract as old_contract
from . import shared_planning as old_planning
from .planning import sha
from .root_planning import prepare_root_original
from .run import offline_network_guard
from .test_root_planning import root_fixture_bundle, independent_new_patch
from .test_shared_planning import shared_initial
from ..h4_round1_r03_feedback_view_v2.templates import PLANNER_DEVELOPMENT_GUIDANCE_V1


def relation(*, output=None, source=None, kind=relations.NATURAL):
    return {"source_candidate_ref": source, "output_ref": output, "relation_kind": kind,
            "condition": "The earlier scene establishes the specific date." if kind == relations.HARD else None}


class PlannerRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = offline_network_guard()
        guard.__enter__()
        cls.addClassCleanup(guard.__exit__, None, None, None)
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_root_original(cls.bundle)

    def initial(self):
        return shared_initial(self.prepared["original"], self.bundle["parent"])

    def compile(self, raw):
        return relations.compile_initial_output(raw, parent=self.bundle["parent"],
            candidates=self.bundle["candidates"], verified_tracks=self.bundle["verified_tracks"])

    def append(self):
        row = deepcopy(old_planning.source_continuation_rows(self.prepared)[0])
        row.update(required_transition="Emily records the agreed date on a new workshop card.",
                   interaction_beats=["Emily proposes a new workshop date.",
                                      "James checks the date.", "Emily records the agreed date."])
        row["dependencies"] = [relation(source=row["source_candidate_ref"])]
        new = independent_new_patch(self.bundle)["new_developments"][0]["candidate"]
        new.update(source_candidate_ref=None,
                   dependencies=[relation(output="/selected_continuations/0", kind=relations.HARD)])
        return {"round_goal": self.prepared["original"]["round_goal"],
                "selected_continuations": [row], "new_development_candidates": [new]}

    def test_typed_schema_and_every_example_parse(self):
        for example in relations.output_examples():
            self.assertTrue(relations.validate_output(json.loads(json.dumps(example)))["valid"])
        raw = self.initial()
        raw["selected_continuations"][1]["dependencies"] = [relation(output="/selected_continuations/0")]
        self.assertTrue(relations.validate_output(raw)["valid"])
        raw["selected_continuations"][1]["dependencies"][0]["condition"] = "A false invented requirement."
        self.assertFalse(relations.validate_output(raw)["valid"])

    def test_initial_rejects_append_candidate_refs_and_mixed_dependency_refs(self):
        raw = self.initial()
        raw["selected_continuations"][1]["dependencies"] = [relation(output="/selected_continuations/0")]
        self.assertTrue(relations.validate_initial_output(raw)["valid"])
        non_null = deepcopy(raw)
        non_null["selected_continuations"][0]["source_candidate_ref"] = "ME01"
        self.assertFalse(relations.validate_initial_output(non_null)["valid"])
        mixed = deepcopy(raw)
        mixed["selected_continuations"][1]["dependencies"][0]["source_candidate_ref"] = "ME01"
        self.assertFalse(relations.validate_initial_output(mixed)["valid"])

    def test_initial_rejects_top_level_output_ref_and_accepts_typed_relations(self):
        raw = self.initial()
        raw["selected_continuations"][1]["dependencies"] = [relation(output="/selected_continuations/0")]
        raw["selected_continuations"][1]["dependencies"][0]["condition"] = None
        self.assertTrue(relations.validate_initial_output(raw)["valid"])
        invalid = deepcopy(raw)
        invalid["selected_continuations"][0]["output_ref"] = "/selected_continuations/0"
        self.assertFalse(relations.validate_initial_output(invalid)["valid"])

    def test_initial_soft_and_hard_survive_only_hard_guards_execution(self):
        raw = self.initial()
        raw["selected_continuations"][1]["dependencies"] = [relation(output="/selected_continuations/0")]
        raw["new_development_candidates"][0]["dependencies"] = [relation(
            output="/selected_continuations/1", kind=relations.HARD)]
        before = sha(raw), sha(self.prepared)
        result = self.compile(raw)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["value"], self.prepared["original"])
        self.assertEqual(result["shared_output"], raw)
        sidecar = result["relation_sidecar"]
        self.assertEqual([row["relation_kind"] for row in sidecar["relations"]],
                         [relations.NATURAL, relations.HARD])
        registry = result["prepared"]["candidate_registry"]
        soft_ref = sidecar["output_candidate_map"]["/selected_continuations/1"]
        hard_ref = sidecar["output_candidate_map"]["/new_development_candidates/0"]
        self.assertEqual(registry[soft_ref]["scene"]["dependencies"], [])
        self.assertEqual(len(registry[hard_ref]["scene"]["dependencies"]), 1)
        self.assertEqual((sha(raw), sha(self.prepared)), before)

    def test_empty_is_undeclared_and_formal_ids_are_explicit(self):
        result = self.compile(self.initial())
        self.assertTrue(result["valid"], result["errors"])
        sidecar = result["relation_sidecar"]
        self.assertEqual(sidecar["relations"], [])
        self.assertTrue(all(node["relations_status"] == "undeclared" for node in sidecar["nodes"]))
        phase = next(iter(sidecar["output_phase_map"].values()))
        mapped = relations.map_formal_scene_ids(sidecar, {phase: "R03:S09"})
        self.assertEqual(mapped["nodes"][0]["scene_ref"], "R03:S09")
        self.assertIsNone(sidecar["nodes"][0]["scene_ref"])
        with self.assertRaisesRegex(ValueError, "scene_mapping_invalid"):
            relations.map_formal_scene_ids(sidecar, {"unknown": "R03:S01"})

    def test_unknown_self_forward_cycle_reject_for_both_relation_kinds(self):
        for kind in (relations.NATURAL, relations.HARD):
            for pointer in ("/selected_continuations/0", "/selected_continuations/1", "/selected_continuations/99"):
                with self.subTest(kind=kind, pointer=pointer):
                    raw = self.initial()
                    raw["selected_continuations"][0]["dependencies"] = [relation(output=pointer, kind=kind)]
                    raw["selected_continuations"][1]["dependencies"] = [relation(
                        output="/selected_continuations/0", kind=kind)]
                    self.assertFalse(self.compile(raw)["valid"])

    def test_unavailable_natural_predecessor_rejects(self):
        raw = self.initial()
        raw["new_development_candidates"][0]["required_transition"] = "Emily discusses familiar progress."
        raw["new_development_candidates"][1]["dependencies"] = [relation(output="/new_development_candidates/0")]
        result = self.compile(raw)
        self.assertFalse(result["valid"])
        self.assertTrue(any("unavailable" in error for error in result["errors"]))

    def test_append_soft_source_keeps_original_and_actual_candidate_mapping(self):
        raw = self.append()
        before = sha(self.prepared), sha(raw)
        result = relations.merge_output(self.prepared, raw)
        self.assertTrue(result["valid"], result["errors"])
        scenes = [entry["scene"] for entry in result["merged"]["added_phases"]]
        self.assertEqual(scenes[0]["dependencies"], [])
        self.assertEqual(scenes[1]["dependencies"][0]["phase_id"], scenes[0]["phase_id"])
        self.assertEqual(result["relation_sidecar"]["output_candidate_map"]["/new_development_candidates/0"],
                         scenes[1]["source_candidate_id"])
        self.assertEqual(result["merged"]["original_plan"], self.prepared["original"])
        self.assertEqual((sha(self.prepared), sha(raw)), before)
        self.assertEqual(result["shared_output"], raw)

    def test_append_empty_allowed_but_original_identity_still_enforced(self):
        raw = self.append()
        raw["selected_continuations"][0]["dependencies"] = []
        self.assertTrue(relations.merge_output(self.prepared, raw)["valid"])
        for field in ("goal", "after_state", "transition_from_previous"):
            altered = deepcopy(raw)
            altered["selected_continuations"][0][field] += " changed"
            self.assertFalse(relations.merge_output(self.prepared, altered)["valid"])

    def test_scene_mapping_requires_every_referenced_source_endpoint(self):
        for kind in (relations.NATURAL, relations.HARD):
            with self.subTest(kind=kind):
                raw = self.append()
                owner = raw["selected_continuations"][0]
                owner["dependencies"] = [relation(source=owner["source_candidate_ref"], kind=kind)]
                result = relations.merge_output(self.prepared, raw)
                self.assertTrue(result["valid"], result["errors"])
                sidecar = result["relation_sidecar"]
                mapping = {node["phase_id"]: f"R03:S{index:02d}"
                           for index, node in enumerate(sidecar["nodes"], 2)}
                before = sha(sidecar)
                partial = relations.map_formal_scene_ids(sidecar, mapping)
                self.assertTrue(all(node["scene_ref"] is not None for node in partial["nodes"]))
                self.assertEqual(partial["scene_mapping_status"], "partial")
                self.assertIsNone(partial["relations"][0]["predecessor_scene_ref"])
                predecessor = sidecar["relations"][0]["predecessor_phase_id"]
                complete = relations.map_formal_scene_ids(sidecar, {predecessor: "R03:S01", **mapping})
                self.assertEqual(complete["scene_mapping_status"], "complete")
                self.assertEqual(complete["relations"][0]["predecessor_scene_ref"], "R03:S01")
                self.assertEqual(sha(sidecar), before)

    def test_legacy_explicit_version_remains_readable_and_strict(self):
        raw = self.initial()
        old = relations.compile_versioned_output(raw, version=old_contract.VERSION,
            parent=self.bundle["parent"], candidates=self.bundle["candidates"],
            verified_tracks=self.bundle["verified_tracks"])
        self.assertTrue(old["valid"], old["errors"])
        self.assertNotIn("relation_sidecar", old)
        append = self.append()
        append["selected_continuations"][0]["dependencies"] = []
        append["new_development_candidates"] = []
        self.assertFalse(old_planning.merge_shared_output(self.prepared, append)["valid"])

    def inputs(self):
        return {"original_plan": self.prepared["original"],
                "stable_candidates": self.prepared["candidate_registry"],
                "authoritative_history": self.bundle["parent"]["dialogue"]["turns"],
                "shared_feedback_view": deepcopy(self.bundle["shared_feedback_view"]),
                "memory_navigation": self.bundle["candidates"],
                "source_scope": self.prepared["source_scope"]}

    def test_actual_initial_and_append_messages_have_identical_contract_and_guidance(self):
        dialogue = {"root_id": self.bundle["parent"]["root_id"], "turns": [
            {"turn_ref": row["turn_id"], "speaker": row["speaker"], "text": row["text"]}
            for row in self.bundle["parent"]["dialogue"]["turns"]]}
        initial = templates.render_initial_planner(feedback_view=self.bundle["shared_feedback_view"],
            dialogue=dialogue, memory_candidates=self.bundle["candidates"],
            verified_tracks=self.bundle["verified_tracks"])
        append = templates.render_supplement(self.inputs(),
            continuation_source_templates=old_planning.source_continuation_rows(self.prepared))
        for built in (initial, append):
            text = built["messages"][0]["content"]
            self.assertEqual(text, built["prompt"])
            self.assertEqual(text.count(PLANNER_DEVELOPMENT_GUIDANCE_V1), 1)
            if built["planner_contract_identity"]["execution_mode"] == "initial":
                self.assertEqual(text.count(templates.INITIAL_RELATION_DETAIL_RULES), 1)
                self.assertEqual(text.count(relations.render_output_contract("initial")), 1)
                self.assertNotIn("Use existing\nsource_candidate_ref or simple output_ref pointers", text)
                self.assertIn("A selected_continuation continues its verified historical track", text)
                self.assertIn("never self-reference", text)
            else:
                self.assertEqual(text.count(templates.RELATION_DETAIL_RULES), 1)
                self.assertEqual(text.count(relations.render_output_contract("append_only")), 1)
            self.assertNotIn("must actually happen", text)
            self.assertNotIn("THIS scene", text)
            self.assertNotIn("scene should enact", text)
            example, _ = json.JSONDecoder().raw_decode(text.split(
                "JSON FIELD EXAMPLE (syntax only; derive actual content from the supplied input)\n", 1)[1])
            self.assertTrue(relations.validate_output(example)["valid"])
            self.assertEqual(built["feedback_view"], self.bundle["shared_feedback_view"])

    def test_missing_or_changed_inherited_feedback_is_rejected(self):
        inputs = self.inputs()
        inputs.pop("shared_feedback_view")
        with self.assertRaisesRegex(ValueError, "inherited_feedback_required"):
            templates.render_supplement(inputs, continuation_source_templates=[])
        with self.assertRaisesRegex(ValueError, "differs_from_initial"):
            templates.render_supplement(self.inputs(), continuation_source_templates=[],
                                        feedback_view={"status": "no_feedback", "reason": "fixture"})
        inputs = self.inputs()
        inputs["shared_feedback_view"] = {"status": "no_feedback", "reason": "explicit export status"}
        built = templates.render_supplement(inputs, continuation_source_templates=[])
        self.assertEqual(built["feedback_view"], inputs["shared_feedback_view"])

    def test_final_messages_preserve_supplied_feedback_and_dynamic_wording(self):
        view = {"status": "eligible", "records": [{"historical_qa": {
            "question": "A deliberately supplied feedback question.",
            "options_or_items": {"A": "First", "B": "Second"}, "canonical_gold": "B"},
            "solver_observations": {"answer": "A", "score": 0}}]}
        dialogue = {"root_id": self.bundle["parent"]["root_id"], "turns": [
            {"turn_ref": "P01:T001", "speaker": "Emily", "text": "THIS scene is what James called it."},
            {"turn_ref": "P01:T002", "speaker": "James", "text": "I remember that phrase."}]}
        built = templates.render_initial_planner(feedback_view=view, dialogue=dialogue)
        text = built["messages"][0]["content"]
        actual, _ = json.JSONDecoder().raw_decode(text.split("【当前角色可见的共享反馈】\n", 1)[1])
        self.assertEqual(actual, view)
        self.assertIn(dialogue["turns"][0]["text"], text)
        inputs = self.inputs()
        inputs["shared_feedback_view"] = view
        append = templates.render_supplement(inputs, continuation_source_templates=[])
        data = json.loads(append["messages"][0]["content"].split("\n\nINPUT DATA\n", 1)[1])
        self.assertEqual(data["shared_feedback_view"], view)


if __name__ == "__main__":
    unittest.main()
