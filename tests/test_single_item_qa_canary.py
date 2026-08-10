from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.core_qa_admission import (
    CORE_QA_JSON_SCHEMA,
    core_qa_schema_sha256,
)
from scripts.current_method.restart_n5_spell.single_item_qa_canary import (
    SINGLE_ITEM_QA_JSON_SCHEMA,
    build_compact_dialogue_view,
    build_single_item_questioner_prompt,
    classify_candidate_disposition,
    deterministic_answer_placement,
    freeze_core_qa,
    frozen_core_qa_metadata,
    materialize_single_item_mcq,
    single_item_qa_protocol_metadata,
    single_item_qa_schema_sha256,
    validate_single_item_core_object,
)


def _catalog() -> dict:
    return {
        "span_catalog_version": "dialogue_span_catalog_v1",
        "spans": [
            {
                "span_id": "P001#S0001",
                "turn_id": "P001",
                "session_id": "parent",
                "speaker": "Mara",
                "exact_text": "Mara promised to submit the river report.",
            },
            {
                "span_id": "C001#S0001",
                "turn_id": "C001",
                "session_id": "child",
                "speaker": "Eli",
                "exact_text": "Eli confirmed that the river report arrived on Tuesday.",
            },
        ],
    }


def _core() -> dict:
    return {
        "question": "What report did Mara promise to submit?",
        "correct_answer_text": "The river report.",
        "answer_support_span_ids": ["P001#S0001"],
    }


class SingleItemQaCanaryTests(unittest.TestCase):
    def test_schema_exposes_only_three_model_fields(self) -> None:
        expected = {
            "question", "correct_answer_text", "answer_support_span_ids"
        }
        self.assertEqual(set(SINGLE_ITEM_QA_JSON_SCHEMA["properties"]), expected)
        self.assertFalse(SINGLE_ITEM_QA_JSON_SCHEMA["additionalProperties"])
        self.assertEqual(
            SINGLE_ITEM_QA_JSON_SCHEMA["properties"]
            ["answer_support_span_ids"]["minItems"], 1)
        self.assertEqual(
            SINGLE_ITEM_QA_JSON_SCHEMA["properties"]
            ["answer_support_span_ids"]["maxItems"], 6)
        self.assertIs(SINGLE_ITEM_QA_JSON_SCHEMA, CORE_QA_JSON_SCHEMA)
        self.assertEqual(single_item_qa_schema_sha256(),
                         core_qa_schema_sha256())

    def test_compact_prompt_contains_each_span_text_once(self) -> None:
        view = build_compact_dialogue_view(_catalog())
        prompt = build_single_item_questioner_prompt(
            span_catalog=_catalog(), lineage_id="conv-26")
        for span in _catalog()["spans"]:
            self.assertEqual(view.count(span["exact_text"]), 1)
            self.assertEqual(prompt.count(span["exact_text"]), 1)
        self.assertIn("Return one strict JSON object only", prompt)
        self.assertIn("no Markdown fence", prompt)

    def test_structural_validator_accepts_exact_object(self) -> None:
        result = validate_single_item_core_object(_core(), span_catalog=_catalog())
        self.assertEqual(result["decision"], "single_item_questioner_accept")
        self.assertEqual(result["core_qa"], _core())

    def test_structural_validator_rejects_model_local_fields(self) -> None:
        for field, value in (
            ("qa_id", "model-id"),
            ("distractors", ["a", "b", "c"]),
            ("reasoning_steps", ["step"]),
        ):
            raw = {**_core(), field: value}
            result = validate_single_item_core_object(raw, span_catalog=_catalog())
            self.assertEqual(
                result["decision"], "single_item_questioner_schema_invalid")
            self.assertIn("core_qa_field_set_invalid", result["reasons"])

    def test_support_count_duplicates_and_unknown_ids_rejected(self) -> None:
        invalid_supports = (
            [],
            ["P001#S0001"] * 2,
            [f"unknown-{index}" for index in range(7)],
            ["missing#S0001"],
        )
        for support in invalid_supports:
            result = validate_single_item_core_object(
                {**_core(), "answer_support_span_ids": support},
                span_catalog=_catalog(),
            )
            self.assertEqual(
                result["decision"], "single_item_questioner_schema_invalid")

    def test_program_freezes_stable_qa_id_after_acceptance(self) -> None:
        first = freeze_core_qa(
            _core(), lineage_id="conv-26", sample_index=0,
            span_catalog=_catalog())
        second = freeze_core_qa(
            _core(), lineage_id="conv-26", sample_index=0,
            span_catalog=_catalog())
        self.assertEqual(first, second)
        self.assertEqual(first["qa_id"], "conv_26_single_item_001")
        self.assertEqual(first["reasoning_steps"], [])
        self.assertNotIn("qa_id", _core())
        with self.assertRaises(ValueError):
            freeze_core_qa(
                {**_core(), "qa_id": "forbidden"}, lineage_id="conv-26",
                sample_index=0, span_catalog=_catalog())

    def test_frozen_id_and_protocol_metadata_are_deterministic(self) -> None:
        frozen = freeze_core_qa(
            _core(), lineage_id="conv-41", sample_index=2,
            span_catalog=_catalog())
        first = frozen_core_qa_metadata(
            frozen_core_qa=frozen, lineage_id="conv-41", sample_index=2,
            span_catalog=_catalog())
        second = frozen_core_qa_metadata(
            frozen_core_qa=frozen, lineage_id="conv-41", sample_index=2,
            span_catalog=_catalog())
        self.assertEqual(first, second)
        metadata = single_item_qa_protocol_metadata()
        self.assertEqual(metadata["max_completion_tokens"], 1024)
        self.assertEqual(metadata["qa_id_source"], "program")
        self.assertEqual(metadata["distractor_semantic_attempt_cap"], 1)

    def test_answer_placement_is_content_independent_and_generic(self) -> None:
        for lineage in ("conv-26", "conv-41", "conv-44", "future-lineage"):
            first = deterministic_answer_placement(lineage, 0)
            second = deterministic_answer_placement(lineage, 0)
            self.assertEqual(first, second)
            self.assertIn(first["target_letter"], "ABCD")
            self.assertFalse(first["content_dependent"])
        self.assertNotEqual(
            deterministic_answer_placement("conv-26", 0)["placement_seed"],
            deterministic_answer_placement("conv-26", 1)["placement_seed"],
        )

    def test_materialization_reuses_minimal_admission_contract(self) -> None:
        core = freeze_core_qa(
            _core(), lineage_id="conv-26", sample_index=0,
            span_catalog=_catalog())
        placement = deterministic_answer_placement("conv-26", 0)
        result = materialize_single_item_mcq(
            core_qa=core,
            distractors=[
                "The bridge survey.",
                "The school budget.",
                "The transit proposal.",
            ],
            placement=placement,
            span_catalog=_catalog(),
        )
        self.assertEqual(
            result["admission"]["spell_style_minimal_v1"]["decision"],
            "accept",
        )
        self.assertTrue(result["admission"]["solver_eligible"])
        self.assertEqual(result["admission"]["item_index"], 0)
        self.assertEqual(len(result["reasoning_contracts"]), 1)
        self.assertEqual(
            result["materialized_qa"]["reference_answer_letter"],
            placement["target_letter"],
        )
        self.assertFalse(result["distractor_semantic_retry_allowed"])

    def test_materialization_never_trims_fourth_distractor(self) -> None:
        core = freeze_core_qa(
            _core(), lineage_id="conv-26", sample_index=0,
            span_catalog=_catalog())
        with self.assertRaises(ValueError):
            materialize_single_item_mcq(
                core_qa=core,
                distractors=["one", "two", "three", "four"],
                placement=deterministic_answer_placement("conv-26", 0),
                span_catalog=_catalog(),
            )

    def test_bad_ground_discards_entire_qa_and_resamples(self) -> None:
        disposition = classify_candidate_disposition("bad_ground")
        self.assertTrue(disposition["discard_whole_qa"])
        self.assertTrue(disposition["resample_required"])
        self.assertFalse(disposition["same_qa_distractor_retry_allowed"])
        self.assertEqual(
            disposition["disposition"], "discard_whole_qa_and_resample")


if __name__ == "__main__":
    unittest.main()
