from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import audit_stage3_fastlane_salvage as salvage
from scripts.current_method import native_api_graph_dialogue_expansion as native


ROOT = Path(__file__).resolve().parents[1]
RUN175 = (
    ROOT
    / "base_restart_current_method_validation"
    / "175_stage3_backfilled_smoke_n10"
)


class Stage3ClaimFirstNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths = salvage.source_paths(RUN175)
        raw_rows = salvage.read_jsonl(paths["raw"])
        cls.parents = salvage.parent_index(salvage.read_jsonl(paths["parents"]))
        cls.raw_record = next(
            row for row in raw_rows if row.get("api_stage64_final_qa_attempts")
        )
        cls.replay = salvage.replay_candidate(
            cls.raw_record,
            RUN175,
            cls.parents,
        )
        cls.seed_row = cls.parents[
            native.normalize_space(cls.raw_record.get("seed_sample_id"))
        ]
        cls.canonical_targets = native._build_stage64_canonical_targets(1)
        cls.stage64_target = copy.deepcopy(
            cls.canonical_targets[0]["stage64_target"]
        )
        cls.pre_claim_ir, cls.precompiled = native.precompile_claim_first_multiqa(
            cls.replay["current_event_plan"],
            cls.seed_row,
            min_qas=3,
            max_qas=6,
            immutable_stage64_target=cls.stage64_target,
        )

    def _bind(self) -> dict:
        return native.bind_validate_claim_first_multiqa(
            current_event_plan=self.replay["current_event_plan"],
            current_alignment=self.replay["current_alignment"],
            final_dialogue=self.replay["final_dialogue"],
            pre_claim_ir=copy.deepcopy(self.pre_claim_ir),
            precompiled=copy.deepcopy(self.precompiled),
            min_qas=3,
        )

    @staticmethod
    def _hashes() -> dict[str, str]:
        return {
            "graph_snapshot_hash": "graph-hash",
            "dialogue_hash": "dialogue-hash",
            "alignment_hash": "alignment-hash",
        }

    def _target_sample(self, bound: dict | None = None) -> dict:
        bundle = copy.deepcopy(bound if bound is not None else self._bind())
        primary = native.select_claim_first_primary_qa(
            bundle,
            self.stage64_target,
        )
        rendered = native.render_claim_first_primary_qa(
            primary,
            bundle["claim_ir"],
            self._hashes(),
        )
        expected = self.canonical_targets[0]
        return {
            "sample_id": "claim-first-target-compatible",
            "generation_slot_id": 0,
            "target_index": 0,
            "stage64_target_plan_sha256": expected["target_plan_sha256"],
            "stage64_target_identity": expected["target_identity"],
            "stage64_target": copy.deepcopy(self.stage64_target),
            "qa_operation": rendered["qa_operation"],
            "current_event_plan": copy.deepcopy(
                self.replay["current_event_plan"]
            ),
            "dialogue": copy.deepcopy(self.replay["final_dialogue"]),
            "alignment": copy.deepcopy(self.replay["current_alignment"]),
            "qa": rendered,
            "claim_first_multiqa_bundle": bundle,
            **self._hashes(),
        }

    def test_precompile_freezes_three_to_six_qas_before_support_binding(self) -> None:
        compiled = self.precompiled

        self.assertGreaterEqual(compiled["qa_count"], 3)
        self.assertLessEqual(compiled["qa_count"], 6)
        self.assertEqual(compiled["support_binding_status"], "pending")
        self.assertEqual(compiled["api_calls_made"], 0)
        self.assertTrue(compiled["semantic_contract_frozen"])
        self.assertTrue(
            all(not qa.get("evidence_ledger") for qa in compiled["qas"])
        )
        self.assertTrue(
            all(
                qa["frozen_claim_bundle_hash"]
                == self.pre_claim_ir["claim_bundle_hash"]
                for qa in compiled["qas"]
            )
        )
        self.assertEqual(
            compiled["immutable_stage64_target"], self.stage64_target
        )
        target_id = compiled["target_fulfillment_qa_id"]
        target_qas = [
            qa
            for qa in compiled["qas"]
            if qa.get("fulfills_immutable_stage64_target") is True
        ]
        self.assertEqual([qa["qa_id"] for qa in target_qas], [target_id])
        self.assertEqual(target_qas[0]["qa_operation"], "temporal_state_update")
        self.assertEqual(
            target_qas[0]["query_plan"]["claim_id_to_event_instance_id"],
            target_qas[0]["claim_event_map"],
        )
        self.assertEqual(
            target_qas[0]["query_plan"]["event_instance_id_to_claim_id"],
            target_qas[0]["event_claim_map"],
        )

    def test_post_dialogue_bind_keeps_hash_and_validates_each_qa_independently(self) -> None:
        original = native.stage3_claim_validator.validation_report
        with mock.patch.object(
            native.stage3_claim_validator,
            "validation_report",
            wraps=original,
        ) as validator:
            bound = self._bind()

        self.assertTrue(bound["claim_bundle_hash_stable"])
        self.assertEqual(
            bound["pre_dialogue_claim_bundle_hash"],
            bound["post_dialogue_claim_bundle_hash"],
        )
        self.assertEqual(
            validator.call_count,
            bound["support_bound_qa_count"],
        )
        self.assertEqual(bound["qa_count"], bound["validator_pass_qa_count"])
        self.assertEqual(
            set(bound["independent_validation_reports"]),
            {qa["qa_id"] for qa in bound["qas"]},
        )
        self.assertTrue(
            all(
                qa["independent_validation_report"]["pass"] is True
                for qa in bound["qas"]
            )
        )
        self.assertTrue(bound["proof_ready"])
        self.assertEqual(bound["api_qa_calls_made"], 0)

        changed = copy.deepcopy(self.pre_claim_ir)
        changed["claim_bundle_hash"] = "tampered-pre-dialogue-hash"
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "claim_bundle_hash changed",
        ):
            native.bind_validate_claim_first_multiqa(
                current_event_plan=self.replay["current_event_plan"],
                current_alignment=self.replay["current_alignment"],
                final_dialogue=self.replay["final_dialogue"],
                pre_claim_ir=changed,
                precompiled=copy.deepcopy(self.precompiled),
                min_qas=3,
            )

    def test_primary_renders_the_legacy_single_choice_envelope(self) -> None:
        bound = self._bind()
        primary = native.select_claim_first_primary_qa(
            bound, self.stage64_target
        )
        rendered = native.render_claim_first_primary_qa(
            primary,
            bound["claim_ir"],
            self._hashes(),
        )

        self.assertEqual(rendered["question_type"], "single_choice")
        self.assertEqual(rendered["question"], rendered["question_text"])
        self.assertEqual(rendered["gold_source"], "claim_first_local_truth_compiler")
        self.assertEqual(rendered["question_surface_source"], "local_frozen_skeleton")
        self.assertEqual(len(rendered["options"]), 4)
        self.assertEqual([option["id"] for option in rendered["options"]], list("ABCD"))
        self.assertEqual(
            [option["id"] for option in rendered["options"] if option["is_gold"]],
            rendered["gold"],
        )
        self.assertEqual(rendered["qa_id"], primary["qa_id"])
        self.assertEqual(rendered["proof_id"], primary["proof"]["proof_id"])
        self.assertEqual(
            rendered["claim_bundle_hash"], bound["claim_bundle_hash"]
        )
        self.assertEqual(
            rendered["depends_on_graph_snapshot_hash"], "graph-hash"
        )
        self.assertEqual(rendered["depends_on_dialogue_hash"], "dialogue-hash")
        self.assertEqual(rendered["depends_on_alignment_hash"], "alignment-hash")
        self.assertFalse(rendered["generated_by_api_stage64_schema"])

        # Rendering is a compatibility projection, not a second compiler.  It
        # must retain every immutable target field and the complete lossless
        # claim -> event -> query-plan mapping.
        for field in (
            "qa_operation",
            "reasoning_subtype",
            "reasoning_template",
            "hop_count",
            "hard_negative_strategy",
            "forbid_max_time_shortcut",
            "answer_type",
            "query_plan",
            "claim_event_map",
            "event_claim_map",
        ):
            self.assertEqual(rendered[field], primary[field], field)
        self.assertEqual(
            rendered["answer_type"],
            self.stage64_target["preferred_answer_type"],
        )
        self.assertTrue(rendered["forbid_max_time_shortcut"])
        for rendered_option, frozen_option in zip(
            rendered["options"], primary["options"], strict=True
        ):
            self.assertEqual(
                rendered_option["source_event_instance_id"],
                frozen_option["source_event_instance_id"],
            )
            self.assertEqual(
                rendered_option["anchor_ref"], frozen_option["anchor_ref"]
            )

    def test_one_validator_failure_keeps_valid_siblings_and_dialogue(self) -> None:
        original = native.stage3_claim_validator.validation_report
        target_id = self.precompiled["target_fulfillment_qa_id"]
        rejected_sibling_id = next(
            qa["qa_id"]
            for qa in self.precompiled["qas"]
            if qa["qa_id"] != target_id
        )
        rejected_sibling_proof_id = next(
            qa["proof"]["proof_id"]
            for qa in self.precompiled["qas"]
            if qa["qa_id"] == rejected_sibling_id
        )

        def reject_first(value: dict) -> dict:
            proof = value.get("proof") if isinstance(value.get("proof"), dict) else {}
            if proof.get("proof_id") == rejected_sibling_proof_id:
                return {
                    "schema_version": "test_injected_validator_v1",
                    "pass": False,
                    "issue_count": 1,
                    "issue_counts_by_code": {"injected_single_qa_failure": 1},
                    "issues": [{"code": "injected_single_qa_failure"}],
                }
            return original(value)

        with mock.patch.object(
            native.stage3_claim_validator,
            "validation_report",
            side_effect=reject_first,
        ):
            bound = self._bind()

        self.assertEqual(bound["validator_reject_qa_count"], 1)
        self.assertEqual(
            bound["validator_pass_qa_count"],
            bound["support_bound_qa_count"] - 1,
        )
        self.assertGreaterEqual(bound["qa_count"], 1)
        self.assertTrue(bound["dialogue_retained_when_sibling_qa_rejected"])
        rejected_ids = {row["qa_id"] for row in bound["qa_rejects"]}
        self.assertTrue(rejected_ids)
        self.assertTrue(
            all(qa["qa_id"] not in rejected_ids for qa in bound["qas"])
        )
        self.assertIn(target_id, {qa["qa_id"] for qa in bound["qas"]})

    def test_semantic_failure_removes_only_sibling_and_target_survives(self) -> None:
        target_id = self.precompiled["target_fulfillment_qa_id"]
        rejected_sibling_id = next(
            qa["qa_id"]
            for qa in self.precompiled["qas"]
            if qa["qa_id"] != target_id
        )
        original = native.stage3_claim_semantic_validator.validate_qa_semantics

        def reject_one_sibling(
            claim_ir: dict, qa: dict, dialogue: dict | None = None
        ) -> dict:
            if qa.get("qa_id") == rejected_sibling_id:
                return {
                    "schema_version": "test_injected_semantic_gate_v1",
                    "qa_id": rejected_sibling_id,
                    "pass": False,
                    "issue_count": 1,
                    "issue_counts_by_code": {"injected_semantic_failure": 1},
                    "issues": [{"code": "injected_semantic_failure"}],
                }
            return original(claim_ir, qa, dialogue)

        with mock.patch.object(
            native.stage3_claim_semantic_validator,
            "validate_qa_semantics",
            side_effect=reject_one_sibling,
        ):
            bound = self._bind()

        accepted_ids = {qa["qa_id"] for qa in bound["qas"]}
        self.assertNotIn(rejected_sibling_id, accepted_ids)
        self.assertIn(target_id, accepted_ids)
        self.assertTrue(bound["target_fulfillment_pass"])
        self.assertTrue(bound["proof_ready"])
        semantic_rejects = [
            row
            for row in bound["qa_rejects"]
            if row.get("stage") == "independent_semantic_validation"
        ]
        self.assertEqual(
            [row["qa_id"] for row in semantic_rejects],
            [rejected_sibling_id],
        )
        self.assertTrue(bound["dialogue_retained_when_sibling_qa_rejected"])

    def test_target_semantic_failure_makes_bundle_not_proof_ready(self) -> None:
        target_id = self.precompiled["target_fulfillment_qa_id"]
        original = native.stage3_claim_semantic_validator.validate_qa_semantics

        def reject_target(
            claim_ir: dict, qa: dict, dialogue: dict | None = None
        ) -> dict:
            if qa.get("qa_id") == target_id:
                return {
                    "schema_version": "test_injected_semantic_gate_v1",
                    "qa_id": target_id,
                    "pass": False,
                    "issue_count": 1,
                    "issue_counts_by_code": {"injected_target_failure": 1},
                    "issues": [{"code": "injected_target_failure"}],
                }
            return original(claim_ir, qa, dialogue)

        with mock.patch.object(
            native.stage3_claim_semantic_validator,
            "validate_qa_semantics",
            side_effect=reject_target,
        ):
            bound = self._bind()

        self.assertNotIn(target_id, {qa["qa_id"] for qa in bound["qas"]})
        self.assertFalse(bound["target_fulfillment_pass"])
        self.assertFalse(bound["proof_ready"])
        with self.assertRaisesRegex(
            native.NativeGraphDialogueReject,
            "immutable-target QA",
        ):
            native.select_claim_first_primary_qa(bound, self.stage64_target)

    def test_claim_first_primary_fully_matches_immutable_target_and_rejects_drift(self) -> None:
        bound = self._bind()
        primary = native.select_claim_first_primary_qa(
            bound, self.stage64_target
        )
        self.assertEqual(primary["qa_id"], bound["target_fulfillment_qa_id"])
        field_pairs = {
            "qa_operation": "qa_operation",
            "reasoning_subtype": "reasoning_subtype",
            "reasoning_template": "reasoning_template",
            "hop_count": "hop_count",
            "hard_negative_strategy": "hard_negative_strategy",
            "answer_type": "preferred_answer_type",
        }
        for primary_field, target_field in field_pairs.items():
            self.assertEqual(
                primary[primary_field],
                self.stage64_target[target_field],
                primary_field,
            )
        self.assertEqual(
            primary["forbid_max_time_shortcut"],
            self.stage64_target["forbid_max_time_shortcut"],
        )
        sample = self._target_sample(bound)

        self.assertEqual(
            native._validate_stage64_sample_target_provenance(
                sample, self.canonical_targets
            ),
            (0, 0),
        )

        mutations = {
            "target_plan_hash": lambda row: row.__setitem__(
                "stage64_target_plan_sha256", "wrong-plan-hash"
            ),
            "target_identity": lambda row: row.__setitem__(
                "stage64_target_identity", "wrong-target-identity"
            ),
            "target_payload": lambda row: row["stage64_target"].__setitem__(
                "target_basis", "tampered-target-payload"
            ),
            "claim_bundle_hash": lambda row: row[
                "claim_first_multiqa_bundle"
            ].__setitem__("post_dialogue_claim_bundle_hash", "wrong-claim-hash"),
            "primary_operation": lambda row: row["qa"].__setitem__(
                "qa_operation", "relation_lookup"
            ),
            "reasoning_subtype": lambda row: row["qa"].__setitem__(
                "reasoning_subtype", "drifted-reasoning-subtype"
            ),
            "reasoning_template": lambda row: row["qa"].__setitem__(
                "reasoning_template", "drifted-reasoning-template"
            ),
            "hop_count": lambda row: row["qa"].__setitem__("hop_count", 3),
            "hard_negative_strategy": lambda row: row["qa"].__setitem__(
                "hard_negative_strategy", "same_slot_outdated_value"
            ),
            "forbid_max_time_shortcut": lambda row: row["qa"].__setitem__(
                "forbid_max_time_shortcut", False
            ),
            "query_plan": lambda row: row["qa"]["query_plan"].__setitem__(
                "resolved_gold_event_instance_ids", ["wrong-event-id"]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(sample)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    native._validate_stage64_sample_target_provenance(
                        changed, self.canonical_targets
                    )

    def test_target_primary_passes_existing_stage64_semantic_quality(self) -> None:
        sample = self._target_sample()
        self.assertEqual(native.stage64_quality.semantic_quality_issues(sample), [])

    def test_local_claim_first_path_never_calls_legacy_final_qa_or_repair_api(self) -> None:
        with (
            mock.patch.object(native, "call_api_stage64_final_qa") as final_qa_api,
            mock.patch.object(native, "call_api_stage64_qa_repair") as repair_api,
        ):
            pre_claim_ir, precompiled = native.precompile_claim_first_multiqa(
                self.replay["current_event_plan"],
                self.seed_row,
                min_qas=3,
                max_qas=6,
                immutable_stage64_target=self.stage64_target,
            )
            bound = native.bind_validate_claim_first_multiqa(
                current_event_plan=self.replay["current_event_plan"],
                current_alignment=self.replay["current_alignment"],
                final_dialogue=self.replay["final_dialogue"],
                pre_claim_ir=pre_claim_ir,
                precompiled=precompiled,
                min_qas=3,
            )
            primary = native.select_claim_first_primary_qa(
                bound, self.stage64_target
            )
            rendered = native.render_claim_first_primary_qa(
                primary, bound["claim_ir"], self._hashes()
            )

        final_qa_api.assert_not_called()
        repair_api.assert_not_called()
        self.assertEqual(bound["api_qa_calls_made"], 0)
        self.assertFalse(rendered["generated_by_api_stage64_schema"])

    def test_legacy_target_provenance_still_requires_operation_and_reasoning_identity(self) -> None:
        canonical_targets = native._build_stage64_canonical_targets(1)
        expected = canonical_targets[0]
        target = expected["stage64_target"]
        reasoning = {
            field: copy.deepcopy(target[field])
            for field in native.STAGE64_REASONING_TARGET_FIELDS
        }
        sample = {
            "sample_id": "legacy-provenance",
            "generation_slot_id": 0,
            "target_index": 0,
            "stage64_target_plan_sha256": expected["target_plan_sha256"],
            "stage64_target_identity": expected["target_identity"],
            "stage64_target": copy.deepcopy(target),
            "qa_operation": target["qa_operation"],
            **copy.deepcopy(reasoning),
            "qa": {
                "qa_operation": target["qa_operation"],
                **copy.deepcopy(reasoning),
                "query_plan": copy.deepcopy(reasoning),
            },
        }

        self.assertEqual(
            native._validate_stage64_sample_target_provenance(
                sample, canonical_targets
            ),
            (0, 0),
        )

        changed_operation = copy.deepcopy(sample)
        changed_operation["qa_operation"] = "relation_lookup"
        changed_operation["qa"]["qa_operation"] = "relation_lookup"
        with self.assertRaisesRegex(RuntimeError, "operation does not match"):
            native._validate_stage64_sample_target_provenance(
                changed_operation, canonical_targets
            )

        changed_reasoning = copy.deepcopy(sample)
        changed_reasoning["reasoning_template"] = "drifted-template"
        with self.assertRaisesRegex(RuntimeError, "reasoning identity"):
            native._validate_stage64_sample_target_provenance(
                changed_reasoning, canonical_targets
            )


if __name__ == "__main__":
    unittest.main()
