"""Final rendered Planner guidance and parseable examples, without requests."""
import json
from pathlib import Path
import unittest

from .bundle import build_failed_bundle
from .prompts import render_initial_planner, TARGET_RULES
from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE, ROOT_PROMPT_VERSION, prepare_bundle, render_for_bundle
from .run import offline_network_guard
from .fixtures import example_budget
from ..h4_round1_r03_feedback_view_v2.compiler import load_frozen_inputs
from ..h4_round1_r03_feedback_view_v2.templates import PLANNER_DEVELOPMENT_GUIDANCE_V1
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import build_planner_prompt
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry


class PlannerGuidanceRestorationTests(unittest.TestCase):
    def test_initial_and_incremental_share_guidance_in_actual_messages(self):
        with offline_network_guard():
            bundle = build_failed_bundle()
            frozen = load_frozen_inputs()
            record = next(row for row in frozen['records'] if row['root_id'] == bundle['identity']['root_id'])
            initial = render_initial_planner(record=record, dialogue=frozen['dialogues'][record['root_id']],
                memory_candidates=bundle['candidates'], verified_tracks=bundle['verified_tracks'])
            bundle.update(policy_id=ROOT_POLICY_ID, incremental_scope=ROOT_SCOPE,
                          supplement_prompt_version=ROOT_PROMPT_VERSION)
            incremental = render_for_bundle(bundle, example_budget(), prepare_bundle(bundle))
            for prompt in (initial, incremental):
                text = prompt['messages'][0]['content']
                self.assertEqual(text.count(PLANNER_DEVELOPMENT_GUIDANCE_V1), 1)
                self.assertIn(TARGET_RULES, text)
                self.assertIn('third-person', text)
            self.assertNotIn('The resulting state already true before scene end', initial['prompt'])

    def test_final_planner_examples_are_json_in_all_speaker_and_plan_modes(self):
        with offline_network_guard():
            bundle = build_failed_bundle()
            registry = build_speaker_registry(bundle['parent'])
            for strict in (False, True):
                for variants in (1, 2):
                    with self.subTest(strict=strict, variants=variants):
                        rendered = build_planner_prompt(parent=bundle['parent'], memory_overview={},
                            candidates=bundle['candidates'], verified_tracks=bundle['verified_tracks'],
                            speaker_registry=registry, strict_speaker_refs=strict,
                            max_plan_variants=variants, max_continuations=3,
                            max_new_development_candidates=8, scene_quota=6)
                        text = rendered['prompt']
                        start = text.index('Output only this JSON object.')
                        example = json.JSONDecoder().raw_decode(text[text.index('\n', start):].lstrip())[0]
                        plans = example.get('plan_variants', [example])
                        self.assertEqual(len(plans), variants)
                        for plan in plans:
                            self.assertEqual(set(plan), {'round_goal', 'selected_continuations', 'new_development_candidates'})
                            field = 'actor_ref' if strict else 'actor'
                            for row in plan['selected_continuations'] + plan['new_development_candidates']:
                                self.assertEqual(set(row['actor_bindings'][0]), {field, 'binding'})


if __name__ == '__main__':
    unittest.main()
