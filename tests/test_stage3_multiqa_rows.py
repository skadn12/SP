from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method import native_api_graph_dialogue_expansion as native
from scripts.current_method import stage3_solver_feedback_mining
from scripts.current_method import stage3_multiqa_rows as rows


def _qa(
    index: int,
    *,
    proof_pass: bool = True,
    semantic_pass: bool = True,
    query_operation: str = "relation_lookup",
) -> dict:
    qa_id = f"qa-{index}"
    gold_text = f"answer-{index}"
    event_a = f"graph:E{index:02d}a"
    event_b = f"graph:E{index:02d}b"
    return {
        "schema_version": "stage3_claim_first_multiqa_v1",
        "qa_id": qa_id,
        "dialogue_family_id": "family-test",
        "qa_operation": "relation_lookup",
        "qa_subtype": "entity_constrained_relation",
        "reasoning_subtype": "entity_constrained_relation",
        "reasoning_template": "bind_entity_then_relation",
        "hop_count": 2,
        "hard_negative_strategy": "same_relation_wrong_entity",
        "forbid_max_time_shortcut": False,
        "answer_type": "AttributeValue",
        "question_text_draft": f"Which value belongs to subject {index}?",
        "options": [
            {
                "id": "A",
                "text": gold_text,
                "source_event_instance_id": event_b,
            },
            {
                "id": "B",
                "text": f"distractor-{index}",
                "source_event_instance_id": event_a,
            },
        ],
        "gold": ["A"],
        "gold_answer_text": gold_text,
        "proof": {
            "proof_id": f"proof-{index}",
            "required_event_instance_ids": [event_a, event_b],
            "reasoning_path_event_instance_ids": [event_a, event_b],
            "answer_event_instance_ids": [event_b],
            "dependency_closure_event_instance_ids": [event_a, event_b],
        },
        "query_plan": {
            "schema_version": "stage3_claim_query_plan_compat_v1",
            "qa_operation": query_operation,
            "reasoning_subtype": "entity_constrained_relation",
            "reasoning_template": "bind_entity_then_relation",
            "hop_count": 2,
            "hard_negative_strategy": "same_relation_wrong_entity",
            "reasoning_anchor_path": [event_a, event_b],
            "required_anchor_refs": [event_a, event_b],
            "supporting_anchor_refs": [event_a, event_b],
            "gold_anchor_refs": [event_b],
            "distractor_anchor_refs": [event_a],
            "dependency_closure_refs": [event_a, event_b],
            "gold_answer": gold_text,
            "normalized_gold_answer": gold_text,
            "forbid_max_time_shortcut": False,
        },
        "evidence_ledger": [
            {"claim_id": f"claim-{index}-a", "turn_id": f"T{index:02d}"},
            {"claim_id": f"claim-{index}-b", "turn_id": f"T{index + 1:02d}"},
        ],
        "independent_validation_report": {
            "schema_version": "stage3_claim_validator_v1",
            "pass": proof_pass,
        },
        "independent_semantic_validation_report": {
            "schema_version": "stage3_claim_semantic_validator_v1",
            "pass": semantic_pass,
        },
    }


def _envelope(qas: list[dict], *, family_id: str = "family-test", lineage: str = "conv-root") -> dict:
    graph_hash = "g" * 64
    dialogue_hash = "d" * 64
    alignment_hash = "a" * 64
    claim_hash = "c" * 64
    normalized_qas = copy.deepcopy(qas)
    for qa in normalized_qas:
        qa["dialogue_family_id"] = family_id
    clean_sample = {
        "sample_id": f"dialogue-{family_id}",
        "source_id": "parent-source",
        "parent_sample_id": "parent-sample",
        "lineage_group": lineage,
        "dialogue_family_id": family_id,
        "dialogue_family_split": "train",
        "expanded_dialogue_id": f"dialogue-{family_id}-turns",
        "expanded_graph_id": f"dialogue-{family_id}-graph",
        "dialogue": {
            "dialogue_hash": dialogue_hash,
            "turns": [
                {"turn_id": "T01", "speaker": "Ava", "text": "First fact."},
                {"turn_id": "T02", "speaker": "Noah", "text": "Second fact."},
            ],
        },
        "graph_update": {"events": [{"event_instance_id": "graph:E01a"}]},
        "metadata": {
            "graph_snapshot_hash": graph_hash,
            "dialogue_hash": dialogue_hash,
            "alignment_hash": alignment_hash,
            "lineage_group": lineage,
        },
        "native_contract": {"valid": True},
        "validation": {
            "graph_dialogue_consistent": True,
            "gold_derivable": True,
            "option_valid": True,
            "anti_shortcut_pass": True,
            "graph_validation_pass": True,
        },
        "claim_bundle_hash": claim_hash,
        "claim_first_multiqa_bundle": {
            "schema_version": "stage3_claim_first_multiqa_v1",
            "dialogue_family_id": family_id,
            "dialogue_family_split": "train",
            "graph_snapshot_hash": graph_hash,
            "claim_bundle_hash": claim_hash,
            "claim_ir": {
                "claim_bundle_hash": claim_hash,
                "claims": [{"claim_id": "claim-frozen"}],
            },
            "qa_count": len(normalized_qas),
            "qas": normalized_qas,
            "dialogue_retained": True,
            "proof_ready": True,
        },
    }
    return {
        "clean_sample": clean_sample,
        "current_event_plan": {"graph_snapshot_hash": graph_hash, "events": []},
        "clean_dialogue": copy.deepcopy(clean_sample["dialogue"]),
        "current_alignment": {"graph:E01a": {"turn_id": "T01"}},
        "hashes": {
            "graph_snapshot_hash": graph_hash,
            "dialogue_hash": dialogue_hash,
            "alignment_hash": alignment_hash,
        },
        "claim_bundle_hash": claim_hash,
        "claim_first_multiqa_bundle": copy.deepcopy(
            clean_sample["claim_first_multiqa_bundle"]
        ),
        "dialogue_family_id": family_id,
    }


class MultiQARowExpansionTests(unittest.TestCase):
    def test_bad_sibling_is_rejected_without_deleting_three_good_rows(self) -> None:
        source = _envelope(
            [_qa(1), _qa(2), _qa(3, semantic_pass=False), _qa(4)]
        )
        before = copy.deepcopy(source)

        result = rows.expand_accepted_dialogue(source)

        self.assertTrue(result["family_ready"])
        self.assertEqual(len(result["qa_rows"]), 3)
        self.assertEqual(len(result["marked_qa_rows"]), 3)
        self.assertEqual(len(result["qa_rejects"]), 1)
        self.assertEqual(result["qa_rejects"][0]["qa_id"], "qa-3")
        self.assertTrue(result["qa_rejects"][0]["dialogue_retained"])
        self.assertTrue(result["qa_rejects"][0]["sibling_qas_retained"])
        self.assertEqual(source, before)

    def test_rows_have_unique_ids_and_synchronized_qa_target_aliases(self) -> None:
        result = rows.expand_accepted_dialogue(
            _envelope([_qa(1), _qa(2), _qa(3), _qa(4)])
        )
        row_ids = [row["sample_id"] for row in result["qa_rows"]]
        self.assertEqual(len(row_ids), len(set(row_ids)))
        for row in result["qa_rows"]:
            qa = row["qa"]
            target = row["stage64_target"]
            query = row["query_plan"]
            self.assertEqual(row["qa_id"], qa["qa_id"])
            self.assertEqual(row["question"], qa["question"])
            self.assertEqual(row["question_text"], qa["question_text"])
            self.assertEqual(row["options"], qa["options"])
            self.assertEqual(row["gold"], qa["gold"])
            self.assertEqual(row["gold_answer"], qa["gold_answer_text"])
            self.assertEqual(row["gold_answers"], [qa["gold_answer_text"]])
            self.assertEqual(row["gold_option"], qa["gold"][0])
            for field in rows.TARGET_FIELDS:
                self.assertEqual(target[field], qa[field], field)
                self.assertEqual(query[field], qa[field], field)
            self.assertEqual(target["preferred_answer_type"], qa["answer_type"])
            self.assertEqual(query["answer_type"], qa["answer_type"])
            self.assertEqual(query["gold_answer"], qa["gold_answer_text"])
            self.assertEqual(qa["immutable_stage64_target"], target)
            self.assertEqual(qa["operation_specific_target"], target)
            self.assertEqual(
                row["claim_first_multiqa_bundle"]["qas"], [qa]
            )
            self.assertEqual(
                row["claim_first_multiqa_bundle"]["active_qa_id"], qa["qa_id"]
            )

    def test_native_dataset_expansion_keeps_dialogue_envelope_and_emits_four_rows(self) -> None:
        envelope = _envelope([_qa(1), _qa(2), _qa(3), _qa(4)])
        dialogue = copy.deepcopy(envelope["clean_sample"])
        before = copy.deepcopy(dialogue)

        expanded = native.expand_truth_first_multiqa_dataset(
            [dialogue],
            [envelope],
            min_qas=3,
            max_qas=6,
            eval_per_mille=100,
        )

        self.assertEqual(dialogue, before)
        self.assertEqual(expanded["dialogue_count"], 1)
        self.assertEqual(expanded["qa_row_count"], 4)
        self.assertEqual(expanded["marked_qa_row_count"], 4)
        self.assertTrue(expanded["all_families_ready"])
        self.assertEqual(
            {row["source_dialogue_sample_id"] for row in expanded["qa_rows"]},
            {dialogue["sample_id"]},
        )
        self.assertEqual(
            {row["split"] for row in expanded["qa_rows"]},
            {expanded["family_summaries"][0]["split"]},
        )
        for row in expanded["qa_rows"]:
            frozen = stage3_solver_feedback_mining.stage64_target_freeze_payload(
                row
            )
            self.assertEqual(
                frozen["stage64_target"], row["stage64_target"]
            )
            self.assertEqual(row["attack_skill"], row["qa_operation"])
            self.assertEqual(
                row["attack_plan"]["attack_skill"], row["qa_operation"]
            )

    def test_primary_row_preserves_the_original_immutable_target_and_identity(self) -> None:
        qas = [_qa(1), _qa(2), _qa(3), _qa(4)]
        qas[0]["qa_role"] = "primary"
        source = _envelope(qas)
        clean = source["clean_sample"]
        primary = qas[0]
        target = {
            "qa_operation": primary["qa_operation"],
            "preferred_answer_type": primary["answer_type"],
            "reasoning_subtype": primary["reasoning_subtype"],
            "reasoning_template": primary["reasoning_template"],
            "hop_count": primary["hop_count"],
            "hard_negative_strategy": primary["hard_negative_strategy"],
            "forbid_max_time_shortcut": primary["forbid_max_time_shortcut"],
            "target_basis": "immutable_source_unit_test",
        }
        target_index = 7
        plan_sha256 = "f" * 64
        identity = native._stage64_target_identity(
            target_index, target, plan_sha256
        )
        clean.update(
            {
                "stage64_target": copy.deepcopy(target),
                "target_index": target_index,
                "stage64_target_plan_sha256": plan_sha256,
                "stage64_target_identity": identity,
                "stage64_target_id": identity,
            }
        )
        source["clean_sample"] = copy.deepcopy(clean)

        expanded = native.expand_truth_first_multiqa_dataset(
            [clean], [source], min_qas=3, max_qas=6
        )
        primary_row = next(
            row for row in expanded["qa_rows"] if row["qa"]["qa_role"] == "primary"
        )

        self.assertEqual(primary_row["stage64_target"], target)
        self.assertEqual(primary_row["target_index"], target_index)
        self.assertEqual(primary_row["stage64_target_plan_sha256"], plan_sha256)
        self.assertEqual(primary_row["stage64_target_identity"], identity)
        self.assertTrue(primary_row["source_dialogue_target_preserved"])
        stage3_solver_feedback_mining.stage64_target_freeze_payload(primary_row)

    def test_generation_writer_keeps_resume_rows_and_writes_independent_qa_artifacts(self) -> None:
        envelope = _envelope([_qa(1), _qa(2), _qa(3), _qa(4)])
        dialogue = copy.deepcopy(envelope["clean_sample"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            native._write_generation_files(
                output,
                [dialogue],
                [envelope],
                [],
                [],
                {"generated_count": 1},
                truth_first_mode=True,
                claim_first_min_qas=3,
                claim_first_max_qas=6,
                claim_first_eval_per_mille=100,
            )
            dialogue_rows = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            qa_rows = [
                json.loads(line)
                for line in (output / "claim_first_qa_rows.jsonl")
                .read_text()
                .splitlines()
            ]
            marked_qa_rows = [
                json.loads(line)
                for line in (output / "claim_first_marked_qa_rows.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(len(dialogue_rows), 1)
        self.assertEqual(dialogue_rows[0]["sample_id"], dialogue["sample_id"])
        self.assertEqual(len(qa_rows), 4)
        self.assertEqual(len(marked_qa_rows), 4)

    def test_marked_rows_preserve_frozen_graph_dialogue_alignment_and_claim_hashes(self) -> None:
        source = _envelope([_qa(1), _qa(2), _qa(3)])
        result = rows.expand_accepted_dialogue(source)
        expected_hashes = {
            **source["hashes"],
            "claim_bundle_hash": source["claim_bundle_hash"],
        }
        for clean, marked in zip(result["qa_rows"], result["marked_qa_rows"]):
            self.assertEqual(clean["frozen_source_hashes"], expected_hashes)
            self.assertEqual(marked["hashes"], source["hashes"])
            self.assertEqual(marked["claim_bundle_hash"], source["claim_bundle_hash"])
            self.assertEqual(marked["current_event_plan"], source["current_event_plan"])
            self.assertEqual(marked["clean_dialogue"], source["clean_dialogue"])
            self.assertEqual(marked["current_alignment"], source["current_alignment"])
            self.assertEqual(marked["clean_sample"], clean)
            self.assertEqual(
                marked["qa_row_bundle"]["frozen_source_hashes"], expected_hashes
            )

    def test_query_plan_conflict_rejects_only_that_sibling(self) -> None:
        result = rows.expand_accepted_dialogue(
            _envelope(
                [
                    _qa(1),
                    _qa(2),
                    _qa(3, query_operation="comparison"),
                    _qa(4),
                ]
            )
        )
        self.assertTrue(result["family_ready"])
        self.assertEqual(
            [row["qa_id"] for row in result["qa_rows"]],
            ["qa-1", "qa-2", "qa-4"],
        )
        self.assertEqual(result["qa_rejects"][0]["qa_id"], "qa-3")
        self.assertEqual(
            result["qa_rejects"][0]["reason_codes"], ["qa_row_render_failed"]
        )

    def test_below_minimum_keeps_survivors_but_emits_no_dataset_rows(self) -> None:
        result = rows.expand_accepted_dialogue(
            _envelope([_qa(1), _qa(2), _qa(3, proof_pass=False)])
        )
        self.assertFalse(result["family_ready"])
        self.assertEqual(len(result["retained_accepted_qas"]), 2)
        self.assertEqual(result["qa_rows"], [])
        self.assertEqual(result["marked_qa_rows"], [])
        self.assertTrue(result["summary"]["dialogue_retained"])
        self.assertTrue(result["summary"]["valid_siblings_retained"])

    def test_duplicate_qa_ids_are_all_rejected_without_arbitrary_selection(self) -> None:
        duplicate_a = _qa(8)
        duplicate_b = _qa(8)
        duplicate_b["question_text_draft"] = "A different surface with the same id?"
        result = rows.expand_accepted_dialogue(
            _envelope([duplicate_a, _qa(1), duplicate_b, _qa(2), _qa(3)])
        )
        self.assertTrue(result["family_ready"])
        self.assertEqual(len(result["qa_rows"]), 3)
        duplicate_rejects = [
            row for row in result["qa_rejects"] if row.get("qa_id") == "qa-8"
        ]
        self.assertEqual(len(duplicate_rejects), 2)
        self.assertTrue(
            all(row["reason_codes"] == ["duplicate_qa_id"] for row in duplicate_rejects)
        )

    def test_same_parent_lineage_forces_same_split_group_across_families(self) -> None:
        first = rows.expand_accepted_dialogue(
            _envelope([_qa(1), _qa(2), _qa(3)], family_id="family-a", lineage="conv-7")
        )
        second = rows.expand_accepted_dialogue(
            _envelope([_qa(4), _qa(5), _qa(6)], family_id="family-b", lineage="conv-7")
        )
        other = rows.expand_accepted_dialogue(
            _envelope([_qa(7), _qa(8), _qa(9)], family_id="family-c", lineage="conv-8")
        )
        self.assertNotEqual(first["dialogue_family_id"], second["dialogue_family_id"])
        self.assertEqual(first["split_group"], second["split_group"])
        self.assertEqual(first["split"], second["split"])
        self.assertNotEqual(first["split_group"], other["split_group"])
        self.assertTrue(
            all(row["split_group"] == first["split_group"] for row in first["qa_rows"])
        )

    def test_unaccepted_dialogue_cannot_be_expanded(self) -> None:
        source = _envelope([_qa(1), _qa(2), _qa(3)])
        source["clean_sample"]["native_contract"]["valid"] = False
        with self.assertRaises(rows.MultiQARowExpansionError):
            rows.expand_accepted_dialogue(source)

    def test_conflicting_frozen_hashes_fail_the_whole_envelope(self) -> None:
        source = _envelope([_qa(1), _qa(2), _qa(3)])
        source["clean_sample"]["metadata"]["dialogue_hash"] = "x" * 64
        with self.assertRaises(rows.MultiQARowExpansionError):
            rows.expand_accepted_dialogue(source)

    def test_expansion_is_deterministic_and_json_serializable(self) -> None:
        source = _envelope([_qa(3), _qa(1), _qa(2)])
        first = rows.expand_accepted_dialogue(source)
        second = rows.expand_accepted_dialogue(source)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["qa_id"] for row in first["qa_rows"]],
            ["qa-1", "qa-2", "qa-3"],
        )
        json.dumps(first, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
