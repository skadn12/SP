"""Actual initial and append-only requests share one rendered output contract."""
from copy import deepcopy
import json
import unittest

from .fixtures import example_budget
from .input_views import reconstruct_supplement_inputs
from .planning import sha
from .prompts import TARGET_RULES, render_initial_planner
from .root_scope import prepare_bundle, render_for_bundle
from .run import offline_network_guard
from .shared_planner_contract import (
    VERSION, RENDER_VERSION, ROOT_SHARED_PROMPT_VERSION, output_schema,
    render_output_contract, validate_output,
)
from .test_root_planning import root_fixture_bundle
from ..h4_round1_r03_feedback_view_v2.templates import PLANNER_DEVELOPMENT_GUIDANCE_V1


class SharedPlannerPromptTests(unittest.TestCase):
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

    def initial(self, **overrides):
        overrides.setdefault('planner_contract_version', VERSION)
        return render_initial_planner(record=self.record, dialogue=self.dialogue,
            memory_candidates=self.bundle["candidates"],
            verified_tracks=self.bundle["verified_tracks"], **overrides)

    def supplement(self):
        bundle = deepcopy(self.bundle)
        bundle["supplement_prompt_version"] = ROOT_SHARED_PROMPT_VERSION
        return render_for_bundle(bundle, example_budget(), self.prepared)

    def test_explicit_shared_initial_entry_selects_shared_version_and_strict_speakers(self):
        for strict in (True, False):
            with self.subTest(requested_strict_speakers=strict):
                built = self.initial(strict_speaker_refs=strict, planner_max_plan_variants=2)
                self.assertEqual(built["prompt_version"],
                                 "incremental_retention_v1.initial_planner." + VERSION)
                self.assertEqual(built["planner_contract_identity"]["execution_mode"], "initial")
                self.assertIn("Every source_candidate_ref must be null", built["prompt"])
                self.assertNotIn('"participants":', built["prompt"])
                self.assertNotIn('"actor":', built["prompt"])
                self.assertNotIn('"plan_variants":', built["prompt"])

    def test_actual_initial_and_append_output_blocks_and_schema_identity_match(self):
        outputs = (self.initial(), self.supplement())
        contract = render_output_contract()
        for built in outputs:
            with self.subTest(version=built["prompt_version"]):
                self.assertEqual(built["prompt"].count(contract), 1)
                self.assertEqual(built["messages"], [{"role": "user", "content": built["prompt"]}])
                identity = built["planner_contract_identity"]
                self.assertEqual(identity["output_contract"], VERSION)
                self.assertEqual(identity["schema_sha256"], sha(output_schema()))
                self.assertEqual(identity["render_version"], RENDER_VERSION)
                self.assertFalse(identity["strict_decoding_added"])
        self.assertEqual(outputs[1]["planner_contract_identity"]["execution_mode"], "append_only")
        self.assertEqual(outputs[1]["supplement_contract_identity"]["output_contract"], VERSION)
        self.assertEqual(outputs[1]["prompt_version"], ROOT_SHARED_PROMPT_VERSION)

    def test_both_actual_messages_send_schema_without_concrete_story_examples(self):
        decoder = json.JSONDecoder()
        for built in (self.initial(), self.supplement()):
            self.assertNotIn('OUTPUT EXAMPLE', built['prompt'])
            self.assertNotIn('"examples"', built['prompt'])
            self.assertNotIn('workshop card', built['prompt'].lower())
            self.assertNotIn('workshop label', built['prompt'].lower())
            self.assertIn('otherwise declare the applicable dependency', built['prompt'])
            self.assertIn('exact canonical person names', built['prompt'])
            schema, _ = decoder.raw_decode(built["prompt"].split("JSON SCHEMA\n", 1)[1])
            self.assertEqual(schema, output_schema())

    def test_restored_guidance_and_target_policy_remain_once(self):
        paragraphs = PLANNER_DEVELOPMENT_GUIDANCE_V1.split("\n\n")[1:]
        self.assertEqual(len(paragraphs), 5)
        for built in (self.initial(), self.supplement()):
            with self.subTest(version=built["prompt_version"]):
                self.assertEqual(built["prompt"].count(PLANNER_DEVELOPMENT_GUIDANCE_V1), 1)
                self.assertEqual(built["prompt"].count(TARGET_RULES), 1)
                for paragraph in paragraphs:
                    self.assertEqual(built["prompt"].count(paragraph), 1)

    def test_append_instructions_have_no_legacy_phase_output_contract(self):
        built = self.supplement()
        instructions = built["prompt"].split("CONTINUATION SOURCE TEMPLATES\n", 1)[0]
        for legacy in ('"status":"PATCH"', "NO_SUPPLEMENT", '"phases":',
                       '"new_developments":', '"candidate":', "phase:[A-Za-z0-9_-]+",
                       "phase:followup_1", "original_goal must copy", "original_plan_sha256"):
            self.assertNotIn(legacy, instructions)
        self.assertIn("Both output arrays contain only\nnew additions", instructions)
        self.assertIn("round_goal exactly from original_plan.round_goal", instructions)
        self.assertIn("No forward, self or unknown references", instructions)
        self.assertIn("actual retained dialogue must establish", instructions)

    def test_source_templates_are_exact_and_all_original_sources_are_available(self):
        from .shared_planning import source_continuation_rows
        built = self.supplement()
        decoder = json.JSONDecoder()
        templates, _ = decoder.raw_decode(built["prompt"].split("CONTINUATION SOURCE TEMPLATES\n", 1)[1])
        self.assertEqual(templates, source_continuation_rows(self.prepared))
        refs = [row["source_candidate_ref"] for row in templates]
        self.assertEqual(refs, self.prepared["eligible_source_candidate_ids"])
        self.assertEqual(built["supplement_contract_identity"]["allowed_source_view"], templates)
        self.assertEqual(built["supplement_contract_identity"]["allowed_source_view_sha256"], sha(templates))
        for row in templates:
            source = self.prepared["candidate_registry"][row["source_candidate_ref"]]
            self.assertEqual(row["goal"], source["scene"]["goal"])
            if row["source_candidate_ref"].startswith("ND"):
                self.assertIsNone(row["verified_track_ref"])

    def test_complete_transcript_appears_once_and_projection_is_lossless(self):
        initial = self.initial()
        transcript = "\n".join(f"{row['turn_ref']} | {row['speaker']} | {row['text']}"
                               for row in self.dialogue["turns"])
        self.assertEqual(initial["prompt"].count("AUTHORITATIVE PARENT TRANSCRIPT\n"), 1)
        self.assertEqual(initial["prompt"].count(transcript), 1)
        supplement = self.supplement()
        self.assertEqual(supplement["prompt"].count("\n\nINPUT DATA\n"), 1)
        data = json.loads(supplement["prompt"].split("\n\nINPUT DATA\n", 1)[1])
        mapping = supplement["input_projection_mapping"]
        self.assertTrue(mapping["reconstruction_exact"])
        restored = reconstruct_supplement_inputs(data, mapping)
        self.assertEqual(restored["authoritative_history"], self.bundle["parent"]["dialogue"]["turns"])
        self.assertEqual(data["authoritative_history"]["turn_count"], len(self.dialogue["turns"]))
        self.assertNotIn("prior_expanded_dialogue", data)
        self.assertEqual(data["retained_actual_r03_dialogue"], [])


if __name__ == "__main__":
    unittest.main()
