import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    materialize_dialogue_only_top_level_defaults,
    normalize_exact_duplicate_scene_reviews,
    rehome_dialogue_only_trailing_empty_arrays,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.scene_decision_closure import (
    compute_scene_decision_closure,
)


PILOT39 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_39")
PILOT39_SHA = "10ab4dcd67afc786407caa0c56fabecbe436bf43a55599f9943a93e700cb810b"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Pilot39RecoveryTests(unittest.TestCase):
    def test_legacy_duplicate_status_is_losslessly_renamed(self):
        expansion = PILOT39 / "roots/cc-train-episode-61555/01_expansion"
        failure = load(expansion / "failure.json")
        raw = load(expansion / "attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        normalized = normalize_exact_duplicate_scene_reviews(raw)
        self.assertEqual(6, len(normalized["renamed_scene_fields"]))
        for review in normalized["output"]["scene_reviews"]:
            self.assertNotIn("duplicate_status", review)
            self.assertIn("semantic_duplicate_status", review)
        output = rehome_dialogue_only_trailing_empty_arrays(
            normalized["output"],
            expected_scene_count=len(failure["realized"]))["output"]
        output = materialize_dialogue_only_top_level_defaults(output)["output"]
        compiled = compile_round_validator_model_output(
            output, realized_scenes=failure["realized"])
        self.assertTrue(compiled["valid"], compiled["errors"])
        compiled["compiled_output"]["durable_updates"] = []
        validation = validate_round_validator_output(
            compiled["compiled_output"],
            realized_scenes=failure["realized"],
            review_world_view=failure["review_world_view"],
            local_gate_results=failure["local_results"],
        )
        self.assertTrue(validation["valid"], validation["errors"])

    def test_partial_primary_progress_keeps_review_accepted_scene(self):
        expansion = PILOT39 / "roots/cc-train-episode-216742/01_expansion"
        realized = load(expansion / "round_validator_input.json")["scenes"]
        closure = compute_scene_decision_closure(
            realized_scenes=realized,
            local_gate_results=load(expansion / "local_scene_gate_results.json"),
            validator_output=load(expansion / "round_validator_output.json"),
            scene_dependency_dag=load(expansion / "scene_dependency_dag.json"),
        )
        self.assertEqual(5, len(closure["accepted_scene_refs"]))
        self.assertEqual(["R01:S06"], closure["rejected_scene_refs"])
        rows = {row["scene_alias"]: row for row in closure["scenes"]}
        self.assertEqual("validator_accept", rows["s3"]["closure_reason"])
        self.assertEqual("validator_accept", rows["s5"]["closure_reason"])
        self.assertEqual("continuation_source_unavailable",
                         rows["s6"]["closure_reason"])

    def test_not_realized_primary_still_blocks_scene(self):
        expansion = PILOT39 / "roots/cc-train-episode-216742/01_expansion"
        realized = load(expansion / "round_validator_input.json")["scenes"]
        validator = load(expansion / "round_validator_output.json")
        validator["scene_reviews"][2]["primary_action_realization"][
            "status"] = "not_realized"
        validator["scene_reviews"][2]["primary_action_realization"][
            "evidence_turn_refs"] = []
        closure = compute_scene_decision_closure(
            realized_scenes=realized,
            local_gate_results=load(expansion / "local_scene_gate_results.json"),
            validator_output=validator,
            scene_dependency_dag=load(expansion / "scene_dependency_dag.json"),
        )
        rows = {row["scene_alias"]: row for row in closure["scenes"]}
        self.assertEqual("primary_action_not_realized",
                         rows["s3"]["closure_reason"])
        self.assertNotIn("R01:S03", closure["accepted_scene_refs"])

    def test_pilot39_remains_read_only(self):
        self.assertEqual(PILOT39_SHA, hash_tree(PILOT39)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
