import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.expanded_dialogue_assembly import (
    assemble_expanded_dialogue, complete_dialogue_manifest,
    expanded_dialogue_markdown, validate_expanded_dialogue,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.materialize_round1_complete_dialogue import (
    EXPECTED_CANARY_14_TREE_SHA, run,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


PARENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json"
)
CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_14"
)


class ExpandedDialogueAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(PARENT.read_text())
        cls.child_path = CANARY / "child_dialogue_promotable.json"
        cls.child = json.loads(cls.child_path.read_text())
        cls.narrative = json.loads((CANARY / "narrative_order_manifest.json").read_text())
        cls.source = json.loads((CANARY / "source_initialization.json").read_text())

    def assemble(self):
        return assemble_expanded_dialogue(
            original_parent=self.parent, promotable_child=self.child,
            narrative_order_manifest=self.narrative,
            speaker_to_entity_ref=self.source["speaker_to_entity_ref"],
            parent_path=PARENT, parent_sha256=sha256_file(PARENT),
            child_path=self.child_path, child_sha256=sha256_file(self.child_path))

    def test_materializes_parent_plus_child_as_98_unique_turns(self):
        expanded = self.assemble()
        validation = validate_expanded_dialogue(
            expanded, original_parent=self.parent, promotable_child=self.child)
        self.assertTrue(validation["valid"], validation)
        self.assertEqual((60, 38, 98), (
            validation["parent_turn_count"], validation["child_turn_count"],
            validation["turn_count"]))
        self.assertEqual(98, validation["unique_turn_ref_count"])

    def test_boundary_and_origins_are_explicit(self):
        expanded = self.assemble()
        self.assertEqual(61, expanded["boundary"]["first_child_narrative_index"])
        self.assertEqual("R01:S01:T001", expanded["boundary"]["first_child_turn_ref"])
        self.assertEqual("cc-train-episode-61555:P05:T016",
                         expanded["turns"][59]["turn_ref"])
        self.assertEqual("R01:S01:T001", expanded["turns"][60]["turn_ref"])
        self.assertTrue(all(row["origin"] == "source_grounded"
                            for row in expanded["turns"][:60]))
        self.assertTrue(all(row["origin"] == "generated_extension"
                            for row in expanded["turns"][60:]))

    def test_source_text_and_refs_are_byte_for_byte_preserved(self):
        parent_before, child_before = deepcopy(self.parent), deepcopy(self.child)
        expanded = self.assemble()
        self.assertEqual(parent_before, self.parent)
        self.assertEqual(child_before, self.child)
        parent_source = {row["turn_id"]: row["text"]
                         for row in self.parent["dialogue"]["turns"]}
        child_source = {row["turn_ref"]: row["text"]
                        for scene in self.child["scenes"]
                        for row in scene["dialogue_turns"]}
        self.assertEqual(parent_source, {row["turn_ref"]: row["text"]
            for row in expanded["turns"] if row["origin"] == "source_grounded"})
        self.assertEqual(child_source, {row["turn_ref"]: row["text"]
            for row in expanded["turns"] if row["origin"] == "generated_extension"})

    def test_tampered_parent_or_child_text_is_rejected(self):
        expanded = self.assemble()
        expanded["turns"][0]["text"] = "tampered"
        validation = validate_expanded_dialogue(
            expanded, original_parent=self.parent, promotable_child=self.child)
        self.assertFalse(validation["valid"])
        self.assertIn("expanded_dialogue_parent_text_changed", validation["errors"])

    def test_markdown_has_visible_child_boundary(self):
        markdown = expanded_dialogue_markdown(self.assemble())
        self.assertIn("# CHILD START — ROUND 1", markdown)
        self.assertIn("`cc-train-episode-61555:P05:T016`", markdown)
        self.assertIn("`R01:S01:T001`", markdown)

    def test_manifest_declares_qa_source_but_does_not_run_qa(self):
        expanded = self.assemble()
        validation = validate_expanded_dialogue(
            expanded, original_parent=self.parent, promotable_child=self.child)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, markdown_path = root / "expanded.json", root / "expanded.md"
            json_path.write_text(json.dumps(expanded), encoding="utf-8")
            markdown_path.write_text(expanded_dialogue_markdown(expanded), encoding="utf-8")
            manifest = complete_dialogue_manifest(
                expanded_dialogue=expanded, validation=validation,
                expanded_json_path=json_path, expanded_json_sha256=sha256_file(json_path),
                expanded_markdown_path=markdown_path,
                expanded_markdown_sha256=sha256_file(markdown_path))
        self.assertTrue(manifest["readiness"]["complete_dialogue_materialized"])
        self.assertTrue(manifest["readiness"]["qa_source_ready"])
        self.assertFalse(manifest["readiness"]["qa_pipeline_ready"])
        self.assertFalse(manifest["memory_is_qa_text_source"])

    def test_standalone_materializer_writes_only_new_output(self):
        parent_sha, canary_sha = sha256_file(PARENT), hash_tree(CANARY)["tree_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "derived"
            report = run(parent_path=PARENT, canary_root=CANARY, output_root=output)
            self.assertEqual("completed", report["status"])
            self.assertTrue((output / "expanded_dialogue_round_1.json").is_file())
            self.assertTrue((output / "expanded_dialogue_round_1.md").is_file())
            self.assertTrue((output / "complete_dialogue_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                run(parent_path=PARENT, canary_root=CANARY, output_root=output)
        self.assertEqual(parent_sha, sha256_file(PARENT))
        self.assertEqual(canary_sha, hash_tree(CANARY)["tree_sha256"])

    def test_canary14_history_baseline_is_unchanged(self):
        self.assertEqual(EXPECTED_CANARY_14_TREE_SHA, hash_tree(CANARY)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
