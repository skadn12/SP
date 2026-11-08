"""Offline Planner guidance and JSON examples through the active renderers."""
from copy import deepcopy
import json
import unittest

from .fixtures import example_budget
from .prompts import TARGET_RULES, render_initial_planner
from .root_scope import ROOT_PROMPT_VERSION, prepare_bundle, render_for_bundle
from .run import offline_network_guard
from .test_root_planning import root_fixture_bundle
from ..h4_round1_r03_feedback_view_v2.renderers import render_expansion_planner
from ..h4_round1_r03_feedback_view_v2.templates import PLANNER_DEVELOPMENT_GUIDANCE_V1


def planner_example(prompt):
    example_and_suffix = prompt.split("Output only this JSON object.", 1)[1].split("\n", 1)[1]
    return json.JSONDecoder().raw_decode(example_and_suffix.lstrip())[0]


class PromptRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = offline_network_guard()
        cls.guard.__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_bundle(cls.bundle)
        view = cls.bundle["shared_feedback_view"]
        cls.record = {**deepcopy(view), "performance": deepcopy(view["solver_observations"])}
        cls.dialogue = {"root_id": cls.bundle["parent"]["root_id"], "turns": [
            {"turn_ref": row["turn_id"], "speaker": row["speaker"], "text": row["text"]}
            for row in cls.bundle["parent"]["dialogue"]["turns"]]}

    def planner_kwargs(self, *, strict=True, variants=1):
        return {"record": self.record, "dialogue": self.dialogue,
                "memory_candidates": self.bundle["candidates"],
                "verified_tracks": self.bundle["verified_tracks"],
                "strict_speaker_refs": strict, "planner_max_plan_variants": variants}

    def test_five_guidance_directions_are_present_once_in_each_actual_entry(self):
        paragraphs = PLANNER_DEVELOPMENT_GUIDANCE_V1.split("\n\n")[1:]
        self.assertEqual(len(paragraphs), 5)
        required_fragments = (
            "为后续 QA 增加可用事实和关系",
            "不需要逐题处理反馈",
            "可以继续发展原有事件，也可以引入相关的新人物",
            "允许出现相似、容易混淆但能够通过上下文区分的事实",
            "而不是无法消解的矛盾、关键事实缺失或故意写不清楚",
        )
        for paragraph, required in zip(paragraphs, required_fragments):
            self.assertIn(required, paragraph)
        built = {
            "expansion_planner": render_expansion_planner(**self.planner_kwargs()),
            "initial_planner": render_initial_planner(**self.planner_kwargs()),
            "root_supplement": render_for_bundle(self.bundle, example_budget(), self.prepared),
        }
        for name, result in built.items():
            with self.subTest(entry=name):
                self.assertEqual(result["prompt"].count(PLANNER_DEVELOPMENT_GUIDANCE_V1), 1)
                for paragraph in paragraphs:
                    self.assertEqual(result["prompt"].count(paragraph), 1)
        self.assertEqual(built["expansion_planner"]["prompt_version"],
                         "h4.r03.expansion.planner.shared_feedback.v4")
        for name in ("initial_planner", "root_supplement"):
            self.assertEqual(built[name]["prompt"].count(TARGET_RULES), 1)
        identity = built["root_supplement"]["supplement_contract_identity"]
        self.assertEqual(identity["prompt_version"], ROOT_PROMPT_VERSION)
        self.assertEqual(identity["output_contract"], "root_incremental_patch_v1")
        self.assertEqual(identity["development_guidance_version"], "planner_development_guidance_v1")

    def assert_single_plan_example(self, example, *, strict):
        self.assertEqual(set(example), {"round_goal", "selected_continuations", "new_development_candidates"})
        entries = example["selected_continuations"] + example["new_development_candidates"]
        self.assertEqual(len(entries), 2)
        speaker_field, actor_field = ("speaker_refs", "actor_ref") if strict else ("participants", "actor")
        forbidden_speaker, forbidden_actor = ("participants", "actor") if strict else ("speaker_refs", "actor_ref")
        for entry in entries:
            self.assertIn(speaker_field, entry)
            self.assertNotIn(forbidden_speaker, entry)
            self.assertIsInstance(entry["actor_bindings"], list)
            self.assertEqual(len(entry["actor_bindings"]), 1)
            self.assertEqual(set(entry["actor_bindings"][0]), {actor_field, "binding"})
            self.assertNotIn(forbidden_actor, entry["actor_bindings"][0])
            self.assertEqual(len(entry["interaction_beats"]), 3)
            if strict:
                self.assertEqual(entry["speaker_refs"], ["SP01", "SP02"])
                self.assertEqual(entry["actor_bindings"][0]["actor_ref"], "SP01")

    def test_expansion_examples_are_valid_json_for_single_double_and_speaker_modes(self):
        for variants in (1, 2):
            for strict in (True, False):
                with self.subTest(variants=variants, strict=strict):
                    built = render_expansion_planner(**self.planner_kwargs(strict=strict, variants=variants))
                    example = planner_example(built["prompt"])
                    if variants == 1:
                        plans = [example]
                        self.assertNotIn("plan_variants", example)
                    else:
                        self.assertEqual(set(example), {"plan_variants"})
                        plans = example["plan_variants"]
                        self.assertEqual(len(plans), 2)
                    for plan in plans:
                        self.assert_single_plan_example(plan, strict=strict)
                    self.assertEqual(built["max_plan_variants"], variants)
                    for token in ("{ACTOR_BINDING}", "SPEAKER_FIELD", "SPEAKER_EXAMPLE"):
                        self.assertNotIn(token, built["prompt"])

    def test_initial_planner_forces_single_plan_and_preserves_target_policy(self):
        for strict in (True, False):
            with self.subTest(strict=strict):
                built = render_initial_planner(planner_contract_version="legacy_initial_v1",
                                               **self.planner_kwargs(strict=strict, variants=2))
                example = planner_example(built["prompt"])
                self.assert_single_plan_example(example, strict=strict)
                self.assertNotIn("plan_variants", example)
                self.assertIn(TARGET_RULES, built["prompt"])
                self.assertIn("The original goal and planned after_state/durable_outcome are desired targets", built["prompt"])
                self.assertNotIn("The transition and resulting state must be a\nbounded mini-arc", built["prompt"])
                for entry in example["selected_continuations"] + example["new_development_candidates"]:
                    self.assertEqual(entry["after_state"], "Original desired final target; not an observed fact")


if __name__ == "__main__":
    unittest.main()
