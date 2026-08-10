import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.complete_dialogue_view import (
    compose_complete_dialogue_view,
    validate_complete_dialogue_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    _shape_retry_suffix,
    build_shape_balanced_distractor_prompt,
)


PILOT38 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_38")
PILOT38_SHA = "f691adf4d8bdf7faffa174ecf634bd1f69f2d025bcf606740c5450ef424ee911"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Pilot38RecoveryTests(unittest.TestCase):
    def test_promotable_subset_projects_full_round_narrative_manifest(self):
        expansion = PILOT38 / "roots/cc-train-episode-61555/01_expansion"
        parent_path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
            "conversation_chronicles_v1/roots/cc-train-episode-61555/"
            "unified_root.json")
        parent_wrapper = load(parent_path)
        parent = parent_wrapper.get("parent") or parent_wrapper
        child = load(expansion / "child_dialogue_promotable.json")
        manifest = load(expansion / "narrative_order_manifest.json")
        speaker_map = load(expansion / "source_initialization.json")[
            "speaker_to_entity_ref"]
        complete = compose_complete_dialogue_view(
            original_parent=parent,
            promoted_rounds=[{"round_index": 1, "child": child,
                              "narrative_order_manifest": manifest}],
            speaker_to_entity_ref=speaker_map,
        )
        promoted = child["scenes"]
        expected = [row["scene_ref"] for row in promoted]
        self.assertEqual(complete["generation_order_scene_refs"], expected)
        self.assertEqual(complete["narrative_order_scene_refs"], expected)
        self.assertTrue(validate_complete_dialogue_view(complete)["valid"])

    def test_manifest_still_rejects_unknown_or_missing_child_scene(self):
        parent = {
            "sample_id": "r", "dialogue": {"turns": [{
                "turn_id": "P:T1", "session_id": "P", "speaker": "A",
                "text": "Parent.",
            }]},
        }
        child = {"root_id": "r", "child_id": "c", "scenes": [{
            "scene_ref": "R01:S01", "dialogue_turns": [{
                "turn_ref": "R01:S01:T001", "speaker_ref": "E1",
                "text": "Child.",
            }],
        }]}
        manifest = {
            "generation_order_scene_refs": ["R01:S02"],
            "ordered_scene_refs": ["R01:S02"],
            "scenes": [{"scene_ref": "R01:S02",
                        "intended_narrative_position": 2.0}],
        }
        with self.assertRaisesRegex(
                ValueError, "complete_dialogue_narrative_scene_set_mismatch"):
            compose_complete_dialogue_view(
                original_parent=parent,
                promoted_rounds=[{"round_index": 1, "child": child,
                                  "narrative_order_manifest": manifest}],
                speaker_to_entity_ref={"A": "E1"},
            )

    def test_distractor_prompt_and_retry_expose_exact_length_ranges(self):
        candidate = PILOT38 / (
            "roots/cc-train-episode-216742/qa_candidates/candidate_04")
        gold = load(candidate / "03_semantic_audit/qa/admitted_gold.json")
        source = load(candidate / "02_questioner/frozen/qa_source_view.json")
        catalog = load(candidate / "02_questioner/frozen/span_catalog.json")
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=source["combined_dialogue"],
            span_catalog=catalog, base_qa=gold)
        self.assertIn('"maximum":14', prompt)
        self.assertIn('"target":11', prompt)
        raw = load(candidate / (
            "04_mcq_materializer/raw/distractor_writer/"
            "cc-train-episode-216742-round1-distractors-04-a00.json"))
        suffix = _shape_retry_suffix(
            json.loads(raw["raw_response"]), raw["validation"])
        self.assertIn("8 to 14 word tokens (target 11)", suffix)
        self.assertIn("Do not repeat this previous invalid", suffix)
        self.assertIn("neglecting to fix their car's broken window", suffix)

    def test_pilot38_remains_read_only(self):
        self.assertEqual(PILOT38_SHA, hash_tree(PILOT38)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
