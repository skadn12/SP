from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    build_cross_boundary_questioner_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    durable_candidate_action_eligible,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    materialize_round_validator_optional_model_fields,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
    validate_round_validator_contract_repair_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
)


PILOT37 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_37"
)
PILOT37_SHA = "c9f987e209eaea6c150ca6561bd9e98fc2647e3b675ea5384b4d8aca5f455796"
EXPANSION = PILOT37 / "roots/cc-train-episode-216742/01_expansion"
CANDIDATES = PILOT37 / "roots/cc-train-episode-61555/qa_candidates"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot37RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator_input = load(EXPANSION / "round_validator_input.json")
        cls.base = load(EXPANSION / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.base = materialize_round_validator_optional_model_fields(cls.base)
        cls.base["durable_updates"] = []
        cls.realized = [load(path) for path in sorted(
            (EXPANSION / "scenes").glob("*/realized_scene.json"))]
        run = load(EXPANSION / "round_validator_run.json")
        historical = {
            row["source_anchor_id"]: row["source_refs"]
            for row in cls.validator_input["historical_anchor_index"]
        }
        eligible = [
            action["action_ref"]
            for scene in cls.validator_input["scenes"]
            for role in ("primary", "secondary")
            if isinstance((action := scene.get(f"{role}_action")), dict)
            and durable_candidate_action_eligible(action)
        ]
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.base,
            realized_scenes=cls.realized,
            compilation_errors=run["attempts"][0]["validation"]["errors"],
            historical_anchor_refs=historical,
            historical_turns=cls.validator_input["historical_turns"],
            eligible_durable_action_refs=eligible,
        )
        cls.old_repair = load(
            EXPANSION / "attempts/round_validator_attempt_02.json")[
                "parsed_response"]

    def test_not_realized_continuation_template_requires_empty_progression(self):
        target = self.plan["targets"][0]
        self.assertEqual(target["action_ref"], "S04.primary_action")
        self.assertEqual(target["status"], "not_realized")
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"],
        )
        final = prompt.rsplit("OUTPUT ONLY THIS OBJECT", 1)[-1]
        self.assertIn('"action_ref":"S04.primary_action"', final)
        self.assertIn('"progression_evidence_turn_refs":[]', final)

    def test_literal_placeholder_remains_invalid_but_empty_fixture_passes(self):
        old = validate_round_validator_contract_repair_output(
            self.old_repair, repair_plan=self.plan)
        self.assertFalse(old["valid"])
        self.assertIn("contract_repair:0:evidence_not_candidate_subset",
                      old["errors"])
        corrected = deepcopy(self.old_repair)
        corrected["repairs"][0]["progression_evidence_turn_refs"] = []
        valid = validate_round_validator_contract_repair_output(
            corrected, repair_plan=self.plan)
        self.assertTrue(valid["valid"], valid["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=self.base, repair_output=corrected,
            repair_plan=self.plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        final = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=load(EXPANSION / "review_world_view.json"),
            local_gate_results=load(EXPANSION / "local_scene_gate_results.json"),
        )
        self.assertTrue(final["valid"], final["errors"])

    def test_questioner_prompt_requires_evaluatively_neutral_stem(self):
        frozen = CANDIDATES / "candidate_04/02_questioner/frozen"
        view = load(frozen / "qa_source_view.json")
        prompt = build_cross_boundary_questioner_prompt(
            root_id="cc-train-episode-61555",
            span_catalog=load(frozen / "span_catalog.json"),
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=view["lineage_pair_registry"],
            frozen_support_pair=view["frozen_support_pair"],
        )
        self.assertIn("evaluatively neutral comparison stem", prompt)
        self.assertIn("personal-versus-professional focus", prompt)

    def test_distractor_prompt_requires_evaluative_and_question_label_parity(self):
        candidate = CANDIDATES / "candidate_04"
        frozen = candidate / "02_questioner/frozen"
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=load(frozen / "qa_source_view.json")[
                "combined_dialogue"],
            span_catalog=load(frozen / "span_catalog.json"),
            base_qa=load(candidate / "03_semantic_audit/qa/admitted_gold.json"),
        )
        self.assertIn("EVALUATIVE PARITY", prompt)
        self.assertIn("QUESTION-LABEL PARITY", prompt)
        self.assertIn("social desirability", prompt)

    def test_historical_solver_rejections_remain_strict(self):
        expected = {3: "reject_all_wrong", 4: "grounding_inconclusive",
                    5: "grounding_inconclusive"}
        rates = {}
        for index, decision in expected.items():
            result = load(CANDIDATES / f"candidate_{index:02d}/05_spell_filter/"
                          "filter_results.json")[0]
            self.assertEqual(result["final_decision"], decision)
            if "balanced_grounding" in result:
                rates[index] = result["balanced_grounding"][
                    "semantic_correct_rate"]
        self.assertEqual(rates[4], 11 / 12)
        self.assertEqual(rates[5], 9 / 12)

    def test_pilot37_remains_read_only(self):
        self.assertEqual(PILOT37_SHA, hash_tree(PILOT37)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
