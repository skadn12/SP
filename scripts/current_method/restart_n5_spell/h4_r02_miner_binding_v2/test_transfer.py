"""Offline behavioral tests for deterministic extraction and v2 boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents, ROLE_KEYS
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract, contract_has_three_disjoint_groups
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import (
    FIXED_POLICY as V1_FIXED_POLICY, model_cards as v1_model_cards,
    model_history as v1_model_history, intent_item_schema as v1_intent_item_schema,
)
from .transfer import (
    TransferHooksV2, binding_schema, miner_output_schema, validate_bindings,
    derive_intent_evidence, derive_all_intent_evidence, EvidenceDerivationError,
    compile_free_brief, compile_guided_brief, CHANGE_FIELDS, DELIVERY_STATUS,
    FIXED_POLICY, DERIVATION,
)


ROOT = "cc-train-episode-198480"


def expanded(count=12):
    return {"root_id": ROOT, "turns": [{"turn_ref": f"{ROOT}:P01:T{i:03d}",
        "session_ref": "scene_a" if i <= 6 else "scene_b", "speaker": "S" + str(i % 2),
        "text": f"literal line {i}  \n"} for i in range(1, count + 1)]}


def intent(identifier="own_007", groups=None, target="test original distinction"):
    return {"intent_id": identifier, "intent_type": "state_transition",
        "reasoning_chain": groups or {"before_state": [5, 1], "trigger": [3], "after_state": [8, 6]},
        "discriminative_variable": target,
        "candidate_question_types": ["single_choice", "multi_select", "ordering"]}


def proposal(identifier="own_007", source="FB-198480-I003-C-Q03"):
    return {"intent_id": identifier, "status": "PROPOSED_TRANSFER", "source_feedback_ids": [source],
        "scope": "Scope is a nonempty explanation, not mechanically verified.",
        "change_from_source": {"old_question_focus": "Original old distinction.",
            "feedback_basis": "M0 full_context and no_context behavior motivates a testable relation.",
            "added_distinction": "Current original distinction adds a relation."}}


def no_transfer(identifier="own_007"):
    return {"intent_id": identifier, "status": "NO_SUPPORTED_TRANSFER", "source_feedback_ids": [],
        "scope": "", "change_from_source": {field: "" for field in CHANGE_FIELDS}}


def cards():
    return {"observation_cards": [{"root_id": ROOT, "feedback_id": "FB-198480-I003-C-Q03",
        "source_quality_status": "ACTIVE_GUIDANCE_ELIGIBLE", "source_question_type": "single_choice",
        "source_evidence_refs": [f"{ROOT}:P01:T012"],
        "solver_observations": {"M0": {"full_context": {"correct": 10, "total": 15},
            "no_context": {"correct": 3, "total": 15}}}, "historical_gold": "B"}]}


def canonical(raw, transcript):
    result = canonicalize_intents({"intents": raw["intents"]}, expanded_dialogue=transcript)
    contracts = [build_reasoning_contract(row) for row in result["intents"] if row["valid"]]
    return result, [row for row in contracts if contract_has_three_disjoint_groups(row)]


def checked(raw, transcript=None, active=None):
    transcript = transcript or expanded()
    result, contracts = canonical(raw, transcript)
    return validate_bindings(raw, result, contracts, expanded=transcript, root_id=ROOT,
        cards=cards() if active is None else active), result, contracts


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.net = patch("urllib.request.urlopen", side_effect=AssertionError("TEST_NETWORK_FORBIDDEN"))
        self.net.start()
        self.addCleanup(self.net.stop)

    def test_exact_order_text_group_union_and_hash_identity(self):
        transcript = expanded()
        raw = {"intents": [intent()], "feedback_bindings": [proposal()]}
        _, result, contracts = checked(raw, transcript)
        row = derive_intent_evidence(result["intents"][0], contracts[0], expanded=transcript,
            transcript_sha256="frozen-exact-byte-hash")
        indices = [1, 3, 5, 6, 8]
        self.assertEqual(row["derived_current_evidence"], [{key: transcript["turns"][i - 1][key]
            for key in ("turn_ref", "speaker", "text")} for i in indices])
        self.assertEqual(row["derived_evidence_groups"], result["intents"][0]["reasoning_chain"])
        self.assertEqual(row["derived_target_distinction"], result["intents"][0]["discriminative_variable"])
        self.assertEqual(row["transcript_sha256"], "frozen-exact-byte-hash")
        self.assertEqual(row["transcript_sha256_basis"], "frozen_file_bytes")
        self.assertEqual(row["derivation"], DERIVATION)
        self.assertEqual(row["delivery_status"], DELIVERY_STATUS)
        self.assertEqual(row["diagnostic_counts_only"]["scene_or_session_count"], 2)

    def test_target_and_explanation_never_search_or_add_historical_source_refs(self):
        transcript = expanded()
        transcript["turns"][11]["text"] = "Unique desired target happens here, outside the declared chain."
        raw = {"intents": [intent(target=transcript["turns"][11]["text"])], "feedback_bindings": [proposal()]}
        raw["feedback_bindings"][0]["change_from_source"]["added_distinction"] = transcript["turns"][11]["text"]
        validation, result, contracts = checked(raw, transcript)
        row = derive_intent_evidence(result["intents"][0], contracts[0], expanded=transcript)
        self.assertEqual(validation["valid_proposed_binding_count"], 1)
        self.assertNotIn(transcript["turns"][11]["turn_ref"], [item["turn_ref"] for item in row["derived_current_evidence"]])
        before = deepcopy(row["derived_current_evidence"])
        result["intents"][0]["discriminative_variable"] = "Completely unrelated alternate target"
        contracts[0]["discriminative_variable"] = result["intents"][0]["discriminative_variable"]
        self.assertEqual(derive_intent_evidence(result["intents"][0], contracts[0], expanded=transcript)["derived_current_evidence"], before)

    def test_wide_chain_preserved_without_cap(self):
        transcript = expanded(150)
        raw = {"intents": [intent(groups={"before_state": [1], "trigger": [2], "after_state": list(range(3, 151))})], "feedback_bindings": [proposal()]}
        _, result, contracts = checked(raw, transcript)
        row = derive_intent_evidence(result["intents"][0], contracts[0], expanded=transcript)
        self.assertEqual(len(row["derived_current_evidence"]), 150)
        self.assertEqual(len(row["derived_evidence_groups"]["after_state"]), 148)

    def test_invalid_item_does_not_drop_later_noncontinuous_id_or_shift_binding(self):
        raw = {"intents": [intent("x_42"), intent("bad", groups={"before_state": [1], "trigger": [1], "after_state": [9]}), intent("arbitrary_900")],
            "feedback_bindings": [proposal("arbitrary_900"), proposal("bad"), no_transfer("x_42")]}
        validation, result, contracts = checked(raw)
        derived = derive_all_intent_evidence(result, contracts, expanded=expanded(), root_id=ROOT)
        self.assertFalse(result["valid"])
        self.assertEqual(derived["usable_intent_count"], 2)
        self.assertEqual(derived["items"][1]["error"], "ORIGINAL_INTENT_INVALID")
        self.assertEqual(validation["by_intent"]["arbitrary_900"]["binding"]["intent_id"], "arbitrary_900")
        self.assertEqual(validation["by_intent"]["x_42"]["binding_state"], "NO_SUPPORTED_TRANSFER")
        self.assertIn("INTENT_FILTERED", validation["bindings"][1]["errors"])

    def test_invalid_binding_does_not_cancel_valid_evidence(self):
        raw = {"intents": [intent()], "feedback_bindings": [proposal()]}
        raw["feedback_bindings"][0]["target_distinction"] = "old forbidden field"
        validation, result, contracts = checked(raw)
        self.assertEqual(validation["by_intent"]["own_007"]["binding_state"], "BINDING_INVALID")
        self.assertEqual(derive_all_intent_evidence(result, contracts, expanded=expanded(), root_id=ROOT)["usable_intent_count"], 1)

    def test_missing_ref_and_ambiguous_ref_stop_item_derivation(self):
        raw = {"intents": [intent()], "feedback_bindings": [proposal()]}
        _, result, contracts = checked(raw)
        for mode in ("missing", "duplicate"):
            with self.subTest(mode=mode):
                transcript = expanded()
                if mode == "missing":
                    transcript["turns"].pop(0)
                else:
                    transcript["turns"].append(deepcopy(transcript["turns"][0]))
                with self.assertRaisesRegex(EvidenceDerivationError, "INTEGRITY_ERROR"):
                    derive_intent_evidence(result["intents"][0], contracts[0], expanded=transcript)
                sidecar = derive_all_intent_evidence(result, contracts, expanded=transcript, root_id=ROOT)
                self.assertEqual(len(sidecar["integrity_errors"]), 1)
                self.assertEqual(sidecar["by_intent"], {})

    def test_contract_chain_cannot_be_replaced(self):
        raw = {"intents": [intent()], "feedback_bindings": [proposal()]}
        _, result, contracts = checked(raw)
        contracts[0]["required_evidence_groups"]["after_state"] = [expanded()["turns"][11]["turn_ref"]]
        with self.assertRaisesRegex(EvidenceDerivationError, "CHAIN_MISMATCH"):
            derive_intent_evidence(result["intents"][0], contracts[0], expanded=expanded())

    def test_legal_v2_binding_and_schema_match_original_intent(self):
        validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [proposal()]})
        self.assertEqual(validation["valid_proposed_binding_count"], 1)
        self.assertEqual(set(binding_schema()["properties"]), {"intent_id", "status", "source_feedback_ids", "scope", "change_from_source"})
        self.assertEqual(miner_output_schema()["properties"]["intents"]["items"], v1_intent_item_schema())

    def test_removed_fields_string_change_missing_subfield_and_extra_fields_are_schema_invalid(self):
        mutations = [lambda b: b.update(evidence_refs=[]), lambda b: b.update(target_distinction="forbidden"),
            lambda b: b.update(change_from_source="old v1 string"),
            lambda b: b["change_from_source"].pop("old_question_focus"),
            lambda b: b["change_from_source"].update(extra="not allowed"),
            lambda b: b.update(extra="not allowed"), lambda b: b.update(scope="  "),
            lambda b: b["change_from_source"].update(feedback_basis=1),
            lambda b: b.update(source_feedback_ids=[{"unhashable": "invalid"}])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                binding = proposal()
                mutation(binding)
                validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [binding]})
                self.assertIn("BINDING_SCHEMA_INVALID", validation["bindings"][0]["errors"])
                self.assertEqual(validation["valid_proposed_binding_count"], 0)

    def test_explanation_truth_is_not_keyword_validated(self):
        binding = proposal()
        binding["change_from_source"]["feedback_basis"] = "This nonempty claim intentionally has no model or context key."
        validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [binding]})
        self.assertEqual(validation["valid_proposed_binding_count"], 1)
        self.assertNotIn("semantic_pass", json.dumps(validation))

    def test_no_supported_transfer_missing_and_invalid_remain_distinct_without_active_source(self):
        for binding_rows, expected in [([no_transfer()], "NO_SUPPORTED_TRANSFER"), ([], "BINDING_MISSING"),
                ([proposal()], "BINDING_INVALID")]:
            with self.subTest(expected=expected):
                validation, _, _ = checked({"intents": [intent()], "feedback_bindings": binding_rows}, active={"observation_cards": []})
                self.assertEqual(validation["feedback_status"], "NO_ACTIVE_FEEDBACK")
                self.assertEqual(validation["by_intent"]["own_007"]["binding_state"], expected)
                self.assertEqual(validation["valid_no_supported_transfer_count"], int(expected == "NO_SUPPORTED_TRANSFER"))

    def test_no_transfer_fields_must_all_be_empty(self):
        binding = no_transfer()
        binding["change_from_source"]["feedback_basis"] = "invented observation"
        validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [binding]})
        self.assertIn("NO_SUPPORTED_TRANSFER_NONEMPTY_FIELDS", validation["bindings"][0]["errors"])

    def test_inactive_cross_root_unknown_sources_blocked(self):
        alternatives = [cards(), cards(), cards()]
        alternatives[0]["observation_cards"][0]["source_quality_status"] = "EXCLUDED_ACTIVE_GUIDANCE"
        alternatives[1]["observation_cards"][0]["root_id"] = "cc-train-episode-122711"
        alternatives[2]["observation_cards"][0]["feedback_id"] = "different-existing-source"
        for active in alternatives:
            validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [proposal()]}, active=active)
            self.assertIn("SOURCE_NOT_ELIGIBLE_FOR_ROOT", validation["bindings"][0]["errors"])

    def test_duplicate_missing_unknown_and_implicit_id_association(self):
        raw = {"intents": [intent()], "feedback_bindings": [proposal(), proposal()]}
        validation, _, _ = checked(raw)
        self.assertTrue(all("DUPLICATE_BINDING_INTENT_ID" in row["errors"] for row in validation["bindings"]))
        raw["feedback_bindings"] = [proposal("unknown")]
        validation, _, _ = checked(raw)
        self.assertIn("UNKNOWN_INTENT_ID", validation["bindings"][0]["errors"])
        self.assertEqual(validation["by_intent"]["own_007"]["binding_state"], "BINDING_MISSING")
        implicit = intent()
        implicit.pop("intent_id")
        validation, result, _ = checked({"intents": [implicit], "feedback_bindings": [proposal("I001")]})
        self.assertTrue(result["intents"][0]["valid"])
        self.assertIn("INTENT_ID_NOT_EXPLICIT", validation["bindings"][0]["errors"])
        raw = {"intents": [intent(), intent()], "feedback_bindings": [proposal()]}
        validation, _, _ = checked(raw)
        self.assertIn("DUPLICATE_RAW_INTENT_ID", validation["bindings"][0]["errors"])

    def test_envelope_failures_and_nonobject_binding_preserved(self):
        for raw, code in [({"intents": [intent()], "feedback_bindings": [proposal()], "unknown": True}, "UNKNOWN_ENVELOPE_FIELDS:unknown"),
                ({"intents": [intent()]}, "BINDINGS_MISSING"),
                ({"intents": [intent()], "feedback_bindings": {}}, "BINDINGS_NOT_ARRAY")]:
            validation, _, _ = checked(raw)
            self.assertIn(code, validation["envelope_errors"])
            self.assertEqual(validation["valid_proposed_binding_count"], 0)
        validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [42]})
        self.assertEqual(validation["bindings"][0]["binding"], 42)
        validation = validate_bindings(None, {"intents": []}, [], expanded=expanded(), root_id=ROOT, cards=cards())
        self.assertIn("ENVELOPE_NOT_OBJECT_OR_JSON_PARSE_FAILED", validation["envelope_errors"])

    def test_explicit_id_cannot_bind_to_a_valid_implicit_id_collision(self):
        implicit = intent()
        implicit.pop("intent_id")
        raw = {"intents": [implicit, intent("I001"), intent("distinct_93")],
            "feedback_bindings": [proposal("I001"), no_transfer("distinct_93")]}
        validation, result, contracts = checked(raw)
        self.assertTrue(result["intents"][0]["valid"])
        self.assertFalse(result["intents"][1]["valid"])
        self.assertIn("AMBIGUOUS_NORMALIZED_INTENT_ID", validation["bindings"][0]["errors"])
        self.assertEqual(validation["by_intent"]["I001"]["binding_state"], "BINDING_INVALID")
        self.assertEqual(validation["by_intent"]["distinct_93"]["binding_state"], "NO_SUPPORTED_TRANSFER")
        # Canonicalizer-valid evidence survives; it is never silently reassigned
        # to the invalid explicit row or used to validate that row's binding.
        derived = derive_all_intent_evidence(result, contracts, expanded=expanded(), root_id=ROOT)
        self.assertEqual(derived["usable_intent_count"], 2)
        self.assertEqual(derived["items"][1]["error"], "ORIGINAL_INTENT_INVALID")

    def test_malformed_json_value_types_never_crash_binding_validation(self):
        invalid_values = [None, True, 7, [], {}, ["x"], {"x": "y"}]
        for field in ("intent_id", "status", "source_feedback_ids", "scope", "change_from_source"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    binding = proposal()
                    binding[field] = value
                    validation, _, _ = checked({"intents": [intent()], "feedback_bindings": [binding]})
                    self.assertFalse(validation["bindings"][0]["mechanical_valid"])

    def test_prompt_common_policy_observation_granularity_and_negative_isolation(self):
        root = "cc-train-episode-197635"
        history = {"records": [{"root_id": root, "source_candidate_id": root + ":old", "stem": "Old tracking only",
            "historical_gold": "SECRET_GOLD", "solver_observations": {"secret": "SECRET_SCORE"}, "accuracy": "SECRET_ACCURACY"}]}
        active = cards()
        active["observation_cards"][0]["root_id"] = root
        active["observation_cards"][0]["source_quality_status"] = "EXCLUDED_ACTIVE_GUIDANCE"
        active["observation_cards"][0]["historical_gold"] = "SECRET_GOLD"
        transcript = expanded()
        transcript["root_id"] = root
        with tempfile.TemporaryDirectory() as temp:
            hook = TransferHooksV2(active, history)
            original = build_intent_prompt(transcript)
            spec = hook.miner_prompt(original, expanded=transcript, root_id=root, root_dir=Path(temp))
        prompt = spec["prompt"]
        self.assertIn("NO_ACTIVE_FEEDBACK", prompt)
        self.assertNotIn("SECRET_", prompt)
        self.assertNotIn("workload sustainability", prompt)
        payload = json.loads(prompt.split("ROOT OBSERVATIONS AND HISTORICAL TRACKING:\n", 1)[1])
        self.assertEqual(payload["observation_cards"], [])
        self.assertEqual(payload["historical_tracking"], v1_model_history(history, root_id=root))
        self.assertEqual(FIXED_POLICY, V1_FIXED_POLICY)
        with tempfile.TemporaryDirectory() as temp:
            positive = TransferHooksV2(cards(), {})
            spec = positive.miner_prompt(build_intent_prompt(expanded()), expanded=expanded(), root_id=ROOT, root_dir=Path(temp))
        payload = json.loads(spec["prompt"].split("ROOT OBSERVATIONS AND HISTORICAL TRACKING:\n", 1)[1])
        self.assertEqual(payload["observation_cards"], v1_model_cards(cards(), root_id=ROOT))

    def test_free_projection_cannot_read_binding_scope_or_guided_evidence(self):
        label = {"intent_id": "id", "intent_type": "state_transition", "scope": "SECRET_SCOPE",
            "change_from_source": "SECRET_CHANGE", "derived_current_evidence": "SECRET_EVIDENCE"}
        preview = compile_free_brief(root_id=ROOT, intent_label=label, cards=cards(), history_only={}, expanded=expanded())
        self.assertNotIn("SECRET_", json.dumps(preview))
        self.assertEqual(preview["complete_transcript"], expanded())
        with self.assertRaises(TypeError):
            compile_free_brief(root_id=ROOT, intent_label=label, cards=cards(), history_only={}, expanded=expanded(), binding=proposal())

    def test_hooks_preserve_raw_invalid_binding_and_all_item_previews_without_writer(self):
        raw = {"intents": [intent("valid_17"), intent("broken", groups={"before_state": [1], "trigger": [1], "after_state": [2]}), intent("normal_23")],
            "feedback_bindings": [proposal("valid_17"), proposal("broken"), no_transfer("normal_23")]}
        raw["feedback_bindings"][0]["evidence_refs"] = ["forbidden"]
        before = deepcopy(raw)
        result, contracts = canonical(raw, expanded())
        with tempfile.TemporaryDirectory() as temp:
            hook = TransferHooksV2(cards(), {})
            hook.after_intents(raw, result, contracts, expanded=expanded(), root_id=ROOT, root_dir=Path(temp))
            names = {path.name for path in Path(temp).iterdir()}
            self.assertEqual(names, {"feedback_bindings.json", "binding_validation.json", "derived_intent_evidence.json", "guided_brief_preview.json"})
            rows = json.loads((Path(temp) / "guided_brief_preview.json").read_text())["items"]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["binding_state"] for row in rows], ["BINDING_INVALID", "ORIGINAL_INTENT_INVALID", "NO_SUPPORTED_TRANSFER"])
            self.assertIsNotNone(rows[0]["derived_intent_evidence"])
            self.assertIsNone(rows[1]["derived_intent_evidence"])
            self.assertTrue(all(row["delivery_status"] == DELIVERY_STATUS for row in rows))
            self.assertFalse(hasattr(hook, "writer_suffix"))
        self.assertEqual(raw, before)

    def test_guided_preview_uses_full_original_card_and_derived_chain(self):
        raw = {"intents": [intent()], "feedback_bindings": [proposal()]}
        validation, result, contracts = checked(raw)
        derived = derive_intent_evidence(result["intents"][0], contracts[0], expanded=expanded())
        preview = compile_guided_brief(root_id=ROOT, intent=result["intents"][0], contract=contracts[0],
            validation=validation, derived=derived, cards=cards())
        self.assertEqual(preview["observation_cards"], v1_model_cards(cards(), root_id=ROOT))
        self.assertEqual(preview["derived_intent_evidence"], derived)
        self.assertEqual(preview["original_intent"], result["intents"][0])
        self.assertEqual(preview["original_contract"], contracts[0])
        self.assertEqual(preview["binding"], raw["feedback_bindings"][0])
        self.assertEqual(preview["delivery_status"], DELIVERY_STATUS)


if __name__ == "__main__":
    unittest.main()
