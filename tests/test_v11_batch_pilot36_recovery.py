from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

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


PILOT36 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_36"
)
PILOT36_SHA = "e5752e84c753190cc1a2686c5a8adc2d3bd26ddf299451774aed500f81892bd1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def recovery_fixture(root_id: str):
    expansion = PILOT36 / "roots" / root_id / "01_expansion"
    validator_input = load(expansion / "round_validator_input.json")
    base = load(expansion / "attempts/round_validator_attempt_01.json")[
        "parsed_response"]
    base = materialize_round_validator_optional_model_fields(base)
    base["durable_updates"] = []
    realized = [load(path) for path in sorted(
        (expansion / "scenes").glob("*/realized_scene.json"))]
    run = load(expansion / "round_validator_run.json")
    historical_anchor_refs = {
        row["source_anchor_id"]: row["source_refs"]
        for row in validator_input["historical_anchor_index"]
    }
    eligible = [
        action["action_ref"]
        for scene in validator_input["scenes"]
        for role in ("primary", "secondary")
        if isinstance((action := scene.get(f"{role}_action")), dict)
        and durable_candidate_action_eligible(action)
    ]
    plan = build_round_validator_contract_repair_plan(
        model_output=base,
        realized_scenes=realized,
        compilation_errors=run["attempts"][0]["validation"]["errors"],
        historical_anchor_refs=historical_anchor_refs,
        historical_turns=validator_input["historical_turns"],
        eligible_durable_action_refs=eligible,
    )
    repair = load(expansion / "attempts/round_validator_attempt_02.json")[
        "parsed_response"]
    return expansion, validator_input, base, realized, plan, repair


class BatchPilot36RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root1 = recovery_fixture("cc-train-episode-61555")
        cls.root2 = recovery_fixture("cc-train-episode-216742")

    def test_repair_plan_exposes_numbered_historical_turn_text(self):
        _, validator_input, _, _, plan, _ = self.root1
        self.assertTrue(plan["valid"], plan["errors"])
        target = plan["scene_reference_targets"][0]
        options = target["allowed_historical_turn_options"]
        self.assertEqual([row["selection_index"] for row in options],
                         list(range(1, len(options) + 1)))
        self.assertTrue(all(row["turn_ref"] and row["text"] for row in options))
        prompt = build_round_validator_contract_repair_prompt(
            repair_plan=plan,
            frozen_input_sha256=validator_input[
                "frozen_round_validator_input_sha256"],
        )
        self.assertIn("historical_supporting_turn_indices", prompt)
        self.assertIn(options[0]["text"], prompt)
        self.assertIn("main Validator contract permits empty historical support",
                      prompt)

    def test_empty_historical_repair_matches_primary_validator_contract(self):
        expansion, _, base, realized, plan, repair = self.root1
        self.assertTrue(any(
            row.get("repair_kind") == "scene_reference" and
            row.get("historical_supporting_turn_refs") == []
            for row in repair["repairs"]))
        validation = validate_round_validator_contract_repair_output(
            repair, repair_plan=plan)
        self.assertTrue(validation["valid"], validation["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=repair, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=realized)
        final = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=realized,
            review_world_view=load(expansion / "review_world_view.json"),
            local_gate_results=load(expansion / "local_scene_gate_results.json"),
        )
        self.assertTrue(final["valid"], final["errors"])

    def test_historical_index_selection_is_deterministic_and_strict(self):
        expansion, _, base, realized, plan, old_repair = self.root2
        old_validation = validate_round_validator_contract_repair_output(
            old_repair, repair_plan=plan)
        self.assertFalse(old_validation["valid"])
        self.assertIn("scene_reference_repair:2:historical_not_allowed_subset",
                      old_validation["errors"])

        indexed = deepcopy(old_repair)
        for row in indexed["repairs"]:
            if row.get("repair_kind") != "scene_reference":
                continue
            row.pop("historical_supporting_turn_refs", None)
            row["historical_supporting_turn_indices"] = [1]
        validation = validate_round_validator_contract_repair_output(
            indexed, repair_plan=plan)
        self.assertTrue(validation["valid"], validation["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=indexed, repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=realized)
        final = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=realized,
            review_world_view=load(expansion / "review_world_view.json"),
            local_gate_results=load(expansion / "local_scene_gate_results.json"),
        )
        self.assertTrue(final["valid"], final["errors"])

        out_of_range = deepcopy(indexed)
        first = next(row for row in out_of_range["repairs"]
                     if row.get("repair_kind") == "scene_reference")
        first["historical_supporting_turn_indices"] = [999]
        invalid = validate_round_validator_contract_repair_output(
            out_of_range, repair_plan=plan)
        self.assertFalse(invalid["valid"])
        self.assertIn("scene_reference_repair:0:historical_selection_out_of_range",
                      invalid["errors"])

    def test_mixed_historical_representations_are_rejected(self):
        _, _, _, _, plan, repair = self.root2
        mixed = deepcopy(repair)
        first = next(row for row in mixed["repairs"]
                     if row.get("repair_kind") == "scene_reference")
        first["historical_supporting_turn_indices"] = [1]
        invalid = validate_round_validator_contract_repair_output(
            mixed, repair_plan=plan)
        self.assertFalse(invalid["valid"])
        self.assertIn(
            "scene_reference_repair:0:historical_selection_mixed_representation",
            invalid["errors"],
        )

    def test_pilot36_remains_read_only(self):
        self.assertEqual(PILOT36_SHA, hash_tree(PILOT36)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
