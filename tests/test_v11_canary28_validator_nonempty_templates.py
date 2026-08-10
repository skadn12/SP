import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    build_round_validator_prompt,
    compile_round_validator_model_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    build_round_validator_contract_repair_plan,
    build_round_validator_contract_repair_prompt,
)


CANARY28 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_28"
)
EXPECTED_SHA = "8f853b89b4ea46b803a23aaf8b9e69e2a3b99bf15ce3c06d7b8a7c3fc566647f"


class Canary28ValidatorNonemptyTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator_input = json.loads((
            CANARY28 / "round_validator_input.json"
        ).read_text(encoding="utf-8"))
        cls.base = json.loads((
            CANARY28 / "attempts/round_validator_attempt_01.json"
        ).read_text(encoding="utf-8"))["parsed_response"]
        cls.compilation = compile_round_validator_model_output(
            cls.base, realized_scenes=cls.validator_input["scenes"])
        cls.plan = build_round_validator_contract_repair_plan(
            model_output=cls.base,
            realized_scenes=cls.validator_input["scenes"],
            compilation_errors=cls.compilation["errors"])

    def test_history_is_read_only(self):
        self.assertEqual(hash_tree(CANARY28)["tree_sha256"], EXPECTED_SHA)

    def test_real_failure_was_template_shaped_empty_evidence(self):
        self.assertEqual(set(self.base), {
            "round_decision", "round_reason_codes", "scene_reviews",
        })
        self.assertEqual(len(self.base["scene_reviews"]), 9)
        self.assertEqual(self.compilation["errors"], [
            "validator_model_top_level_schema",
        ])
        self.assertTrue(self.plan["valid"], self.plan["errors"])

    def test_primary_template_marks_required_selections(self):
        prompt = build_round_validator_prompt(self.validator_input)
        template_text = prompt.split(
            "OUTPUT DATA TEMPLATE — replace values; preserve keys and scene order\n", 1
        )[1].split("\n\nKeep three arrays", 1)[0]
        template = json.loads(template_text)
        for review in template["scene_reviews"]:
            self.assertEqual(review["current_scene_supporting_turn_refs"], ["!"])
            self.assertEqual(review["historical_supporting_turn_refs"], ["!"])
            for role in ("primary_action_realization", "secondary_action_realization"):
                action = review[role]
                if action is not None:
                    self.assertEqual(action["evidence_turn_refs"], ["!"])

    def test_repair_template_never_suggests_empty_required_refs(self):
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=self.plan,
            frozen_input_sha256=self.validator_input[
                "frozen_round_validator_input_sha256"])
        template = json.loads(prompt.split("OUTPUT ONLY THIS OBJECT\n", 1)[1])
        rows = template["repairs"]
        for row in rows:
            kind = row["repair_kind"]
            if kind == "scene_reference":
                self.assertEqual(row["current_scene_supporting_turn_refs"], ["!"])
            elif kind == "action_realization":
                if row.get("evidence_turn_refs") == ["!"]:
                    self.assertNotEqual(row["evidence_turn_refs"], [])
                if row.get("progression_evidence_turn_refs") == ["!"]:
                    self.assertNotEqual(row["progression_evidence_turn_refs"], [])
            elif kind == "durable_update_evidence":
                self.assertEqual(row["evidence_turn_refs"], ["!"])


if __name__ == "__main__":
    unittest.main()
