from __future__ import annotations

import copy
import unittest

from scripts.current_method import stage3_claim_validator as validator


def relation_claim(
    claim_id: str,
    *,
    subject: str,
    subject_event_id: str,
    predicate: str,
    object_value: object,
    object_event_id: str,
    time_index: int,
    polarity: str = "positive",
    status: str = "active",
    condition: object = None,
    scope: object = None,
    edge_role: str = "semantic",
    qa_eligible: bool = True,
) -> dict:
    support = {
        "subject": subject,
        "predicate": predicate,
        "objects": [object_value],
        "polarity": polarity,
        "status": status,
        "time": {
            "time_index": time_index,
            "global_time_index": time_index,
        },
        "condition": condition,
        "scope": scope,
    }
    return {
        "claim_id": claim_id,
        "claim_type": "relation",
        "subject_event_id": subject_event_id,
        "predicate": predicate,
        "object_event_ids": [object_event_id],
        "edge_role": edge_role,
        "qa_eligible": qa_eligible,
        "polarity": polarity,
        "status": status,
        "time_index": time_index,
        "global_time_index": time_index,
        "condition": condition,
        "scope": scope,
        "source_event_ids": [subject_event_id, object_event_id],
        "canonical_support_claim": support,
        "truth_signature": copy.deepcopy(support),
    }


def golden_bundle() -> dict:
    claim_1 = relation_claim(
        "C1",
        subject="Alice",
        subject_event_id="E:alice",
        predicate="grant_access",
        object_value="Bob",
        object_event_id="E:bob",
        time_index=1,
        condition="audit passed",
    )
    claim_2 = relation_claim(
        "C2",
        subject="Alice",
        subject_event_id="E:alice",
        predicate="grant_access",
        object_value="Carol",
        object_event_id="E:carol",
        time_index=2,
        condition="audit passed",
    )
    quote_1 = "At time 1, Alice granted Bob access if the audit passed."
    quote_2 = "At time 2, Alice now granted Carol access if the audit passed."
    return {
        "claims": [claim_1, claim_2],
        "proof": {
            "proof_id": "P1",
            "operation": "status_transition",
            "required_claim_ids": ["C1", "C2"],
            "reasoning_path_claim_ids": ["C1", "C2"],
            "gold": "Carol",
            "answer_type": "entity",
            "derivation": {
                "rule": "latest_active",
                "answer_claim_ids": ["C2"],
            },
            "gold_operand_claim_ids": ["C2"],
            "dependency_closure_claim_ids": ["C1"],
        },
        "support_ledger": {
            "schema_version": "claim_support_ledger_v1",
            "entries": [
                {
                    "claim_id": "C1",
                    "turn_id": "D1:1",
                    "exact_support_quote": quote_1,
                },
                {
                    "claim_id": "C2",
                    "turn_id": "D1:2",
                    "exact_support_quote": quote_2,
                },
            ],
        },
        "dialogue": {
            "turns": [
                {"turn_id": "D1:1", "speaker": "Alice", "text": quote_1},
                {"turn_id": "D1:2", "speaker": "Alice", "text": quote_2},
            ]
        },
    }


def issue_codes(bundle: dict) -> set[str]:
    return {issue["code"] for issue in validator.validate_claim_bundle(bundle)}


def replace_bound_quote(bundle: dict, index: int, quote: str) -> None:
    bundle["dialogue"]["turns"][index]["text"] = quote
    bundle["support_ledger"]["entries"][index]["exact_support_quote"] = quote


class ClaimValidatorGoldenTest(unittest.TestCase):
    def test_golden_bundle_passes_and_input_is_not_mutated(self) -> None:
        bundle = golden_bundle()
        before = copy.deepcopy(bundle)

        self.assertEqual(validator.validate_claim_bundle(bundle), [])
        self.assertTrue(validator.is_valid_claim_bundle(bundle))
        self.assertEqual(bundle, before)
        self.assertEqual(
            validator.validation_report(bundle),
            {
                "schema_version": validator.SCHEMA_VERSION,
                "pass": True,
                "issue_count": 0,
                "issue_counts_by_code": {},
                "issues": [],
            },
        )

    def test_exact_substring_can_be_smaller_than_the_turn(self) -> None:
        bundle = golden_bundle()
        quote = bundle["support_ledger"]["entries"][1]["exact_support_quote"]
        bundle["dialogue"]["turns"][1]["text"] = f"For the record: {quote} Please remember that."

        self.assertEqual(validator.validate_claim_bundle(bundle), [])

    def test_negative_polarity_and_canceled_status_are_explicitly_supported(self) -> None:
        negative = relation_claim(
            "C-neg",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="grant_access",
            object_value="Bob",
            object_event_id="E:bob",
            time_index=1,
            polarity="negative",
        )
        negative_quote = "At time 1, Alice did not grant Bob access."
        canceled = relation_claim(
            "C-canceled",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="grant_access",
            object_value="Bob",
            object_event_id="E:bob",
            time_index=2,
            status="canceled",
        )
        canceled_quote = (
            "At time 2, Alice granted Bob access, but it was canceled before taking effect."
        )

        self.assertEqual(
            validator._validate_quote_semantics(
                negative, negative_quote, path="$.negative"
            ),
            [],
        )
        self.assertEqual(
            validator._validate_quote_semantics(
                canceled, canceled_quote, path="$.canceled"
            ),
            [],
        )

    def test_visibility_scope_tokens_are_required(self) -> None:
        claim = relation_claim(
            "C-vis",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="can_see",
            object_value="budget file",
            object_event_id="E:file",
            time_index=4,
            scope="finance team only",
        )
        good = "At time 4, Alice can see the budget file for the finance team only."
        bad = "At time 4, Alice can see the budget file."

        self.assertEqual(
            validator._validate_quote_semantics(claim, good, path="$.quote"), []
        )
        self.assertIn(
            "support_scope_missing",
            {
                issue["code"]
                for issue in validator._validate_quote_semantics(
                    claim, bad, path="$.quote"
                )
            },
        )

    def test_subject_equal_to_object_does_not_create_false_direction_error(self) -> None:
        claim = relation_claim(
            "C-name",
            subject="Max",
            subject_event_id="E:max",
            predicate="name",
            object_value="Max",
            object_event_id="E:max-name",
            time_index=1,
        )

        issues = validator._validate_quote_semantics(
            claim,
            "Max's Name is Max",
            path="$.quote",
        )

        self.assertNotIn(
            "support_direction_mismatch", {issue["code"] for issue in issues}
        )
        self.assertEqual(issues, [])

    def test_status_words_inside_frozen_object_do_not_change_claim_status(self) -> None:
        value = (
            "is adopted by Nate, who is excited about the new addition to his "
            "family but is unsure about training him."
        )
        claim = relation_claim(
            "C-relationship",
            subject="Max",
            subject_event_id="E:max",
            predicate="relationship",
            object_value=value,
            object_event_id="E:relationship",
            time_index=3,
            status="active",
        )
        quote = (
            "Max's Relationship is Is adopted by Nate, who is excited about the "
            "new addition to his family but is unsure about training him"
        )

        issues = validator._validate_quote_semantics(claim, quote, path="$.quote")

        self.assertNotIn("support_status_conflict", {item["code"] for item in issues})
        self.assertEqual(issues, [])

    def test_scope_limited_authorized_people_quote_is_recognized(self) -> None:
        claim = relation_claim(
            "C-private",
            subject="Maria",
            subject_event_id="E:maria",
            predicate="pet_adoption_status",
            object_value=0,
            object_event_id="E:pet-status",
            time_index=6,
            status="scope_limited",
        )
        quote = (
            "Only authorized people may know that Maria's Pet adoption status is 0"
        )

        self.assertEqual(
            validator._validate_quote_semantics(claim, quote, path="$.quote"), []
        )


class ClaimValidatorMutationTest(unittest.TestCase):
    def test_mutation_suite_kills_every_semantic_mutant(self) -> None:
        mutants: dict[str, tuple[dict, set[str]]] = {}

        direction = golden_bundle()
        replace_bound_quote(
            direction,
            1,
            "At time 2, Carol now granted Alice access if the audit passed.",
        )
        mutants["subject_object_direction"] = (
            direction,
            {"support_direction_mismatch"},
        )

        negation = golden_bundle()
        replace_bound_quote(
            negation,
            1,
            "At time 2, Alice did not grant Carol access if the audit passed.",
        )
        mutants["negation"] = (negation, {"support_unexpected_negation"})

        status = golden_bundle()
        replace_bound_quote(
            status,
            1,
            (
                "At time 2, Alice now granted Carol access if the audit passed, "
                "but that grant was canceled."
            ),
        )
        mutants["status"] = (status, {"support_status_conflict"})

        time = golden_bundle()
        replace_bound_quote(
            time,
            1,
            "At time 3, Alice now granted Carol access if the audit passed.",
        )
        mutants["time"] = (time, {"support_time_missing"})

        condition = golden_bundle()
        replace_bound_quote(
            condition,
            1,
            "At time 2, Alice now granted Carol access if the budget passed.",
        )
        mutants["condition"] = (condition, {"support_condition_missing"})

        semantic_object = golden_bundle()
        replace_bound_quote(
            semantic_object,
            1,
            "At time 2, Alice now granted Dana access if the audit passed.",
        )
        mutants["semantic_object"] = (
            semantic_object,
            {"support_object_missing"},
        )

        exact_quote = golden_bundle()
        exact_quote["support_ledger"]["entries"][1][
            "exact_support_quote"
        ] = "At time 2, alice now granted Carol access if the audit passed."
        mutants["exact_quote"] = (
            exact_quote,
            {"support_quote_not_exact_substring"},
        )

        later_override = golden_bundle()
        later_override["dialogue"]["turns"].append(
            {
                "turn_id": "D1:3",
                "speaker": "Alice",
                "text": (
                    "At time 3, Alice now granted Dana access instead if the audit passed."
                ),
            }
        )
        mutants["later_override"] = (
            later_override,
            {"answer_claim_later_unmodeled_override"},
        )

        killed = 0
        observed: dict[str, list[str]] = {}
        for name, (mutant, expected_codes) in mutants.items():
            with self.subTest(mutation=name):
                codes = issue_codes(mutant)
                observed[name] = sorted(codes)
                self.assertTrue(expected_codes & codes, codes)
                killed += bool(codes)

        report = {
            "total": len(mutants),
            "killed": killed,
            "survived": len(mutants) - killed,
            "kill_rate": killed / len(mutants),
        }
        self.assertEqual(
            report,
            {"total": 8, "killed": 8, "survived": 0, "kill_rate": 1.0},
            observed,
        )

    def test_truth_signature_mutation_is_killed_before_quote_validation(self) -> None:
        bundle = golden_bundle()
        bundle["claims"][1]["truth_signature"]["objects"] = ["Dana"]

        self.assertIn("truth_signature_mismatch", issue_codes(bundle))

    def test_frozen_answer_claim_cannot_be_an_overridden_earlier_state(self) -> None:
        bundle = golden_bundle()
        bundle["proof"]["gold"] = "Bob"
        bundle["proof"]["derivation"]["answer_claim_ids"] = ["C1"]
        bundle["proof"]["gold_operand_claim_ids"] = ["C1"]

        codes = issue_codes(bundle)

        self.assertIn("answer_claim_later_overridden", codes)


class ClaimValidatorProofAndNegativeTest(unittest.TestCase):
    def test_numeric_gold_answer_text_matches_numeric_claim_value(self) -> None:
        claim = relation_claim(
            "C-number",
            subject="Maria",
            subject_event_id="E:maria",
            predicate="score",
            object_value=25,
            object_event_id="E:score",
            time_index=1,
        )
        proof = {
            "proof_id": "P-number",
            "operation": "relation_lookup",
            "required_claim_ids": ["C-number"],
            "reasoning_path_claim_ids": ["C-number"],
            "dependency_closure_claim_ids": ["C-number"],
            "gold_operand_claim_ids": [],
            "gold": ["B"],
            "gold_answer_text": "25",
            "answer_type": "AttributeValue",
            "derivation": {
                "rule": "read_named_entity_attribute",
                "answer_claim_ids": ["C-number"],
            },
        }

        self.assertEqual(validator.validate_proof(proof, [claim]), [])

    def test_control_and_noneligible_claims_cannot_enter_reasoning_path(self) -> None:
        bundle = golden_bundle()
        bundle["claims"][0]["edge_role"] = "control"
        bundle["claims"][0]["qa_eligible"] = False

        codes = issue_codes(bundle)

        self.assertIn("reasoning_path_control_claim", codes)
        self.assertIn("reasoning_path_not_qa_eligible", codes)

    def test_unknown_proof_ref_and_missing_required_ledger_are_rejected(self) -> None:
        bundle = golden_bundle()
        bundle["proof"]["required_claim_ids"].append("C-missing")
        bundle["support_ledger"]["entries"] = bundle["support_ledger"]["entries"][:1]

        codes = issue_codes(bundle)

        self.assertIn("proof_unknown_claim", codes)
        self.assertIn("missing_required_claim_ledger", codes)

    def test_quote_must_be_a_case_sensitive_substring_of_the_cited_turn(self) -> None:
        bundle = golden_bundle()
        bundle["support_ledger"]["entries"][1][
            "exact_support_quote"
        ] = bundle["support_ledger"]["entries"][1][
            "exact_support_quote"
        ].replace("Alice", "alice")

        self.assertIn("support_quote_not_exact_substring", issue_codes(bundle))

    def test_semantic_tokens_are_checked_even_when_quote_is_exact(self) -> None:
        bundle = golden_bundle()
        replace_bound_quote(
            bundle,
            1,
            "At time 2, Alice now granted Dana access if the audit passed.",
        )

        self.assertIn("support_object_missing", issue_codes(bundle))

    def test_later_claim_time_must_follow_earlier_support_turn(self) -> None:
        bundle = golden_bundle()
        first = bundle["support_ledger"]["entries"][0]
        second = bundle["support_ledger"]["entries"][1]
        first["turn_id"], second["turn_id"] = second["turn_id"], first["turn_id"]
        first["exact_support_quote"], second["exact_support_quote"] = (
            second["exact_support_quote"],
            first["exact_support_quote"],
        )

        self.assertIn("support_time_order_mismatch", issue_codes(bundle))

    def test_status_conditioned_proof_allows_recalled_out_of_time_order(self) -> None:
        bundle = golden_bundle()
        bundle["proof"]["derivation"] = {
            "rule": "select_unique_status_in_entity_slot",
            "status": "active",
            "answer_claim_ids": ["C2"],
        }
        # The later state is recounted first, but each exact quote remains
        # correctly bound to its own frozen claim.
        bundle["dialogue"]["turns"] = list(
            reversed(bundle["dialogue"]["turns"])
        )

        codes = issue_codes(bundle)

        self.assertNotIn("support_time_order_mismatch", codes)
        self.assertNotIn("answer_claim_later_overridden", codes)
        self.assertNotIn("answer_claim_later_unmodeled_override", codes)

    def test_frozen_later_claim_is_not_reported_as_unmodeled_override(self) -> None:
        bundle = golden_bundle()
        later = relation_claim(
            "C3",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="grant_access",
            object_value="Dana",
            object_event_id="E:dana",
            time_index=3,
            condition="audit passed",
        )
        quote = "At time 3, Alice now granted Dana access if the audit passed."
        bundle["claims"].append(later)
        bundle["support_ledger"]["entries"].append(
            {
                "claim_id": "C3",
                "turn_id": "D1:3",
                "exact_support_quote": quote,
            }
        )
        bundle["dialogue"]["turns"].append(
            {"turn_id": "D1:3", "speaker": "Alice", "text": quote}
        )

        codes = issue_codes(bundle)

        self.assertIn("answer_claim_later_overridden", codes)
        self.assertNotIn("answer_claim_later_unmodeled_override", codes)

    def test_unreferenced_effectively_ineligible_relation_may_remain_unbound(self) -> None:
        claim = relation_claim(
            "C-unused-relation",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="depends_on",
            object_value="review",
            object_event_id="E:review",
            time_index=1,
            qa_eligible=False,
        )
        # A stale producer hint must not override effective QA eligibility.
        claim["answer_bearing"] = True
        ledger = [
            {
                "claim_id": "C-unused-relation",
                "turn_id": None,
                "exact_support_quote": None,
            }
        ]
        dialogue = {"turns": [{"turn_id": "D1:1", "text": "Unrelated chat."}]}

        issues = validator.validate_claim_artifacts(
            [claim], [], ledger, dialogue
        )

        self.assertNotIn("required_claim_unbound", {item["code"] for item in issues})
        self.assertEqual(issues, [])

    def test_absolute_difference_is_independently_recomputed(self) -> None:
        score_a = relation_claim(
            "C-score-a",
            subject="Alice",
            subject_event_id="E:alice",
            predicate="score",
            object_value=12,
            object_event_id="E:score-12",
            time_index=1,
        )
        score_b = relation_claim(
            "C-score-b",
            subject="Bob",
            subject_event_id="E:bob",
            predicate="score",
            object_value=7,
            object_event_id="E:score-7",
            time_index=2,
        )
        quote_a = "At time 1, Alice recorded score 12."
        quote_b = "At time 2, Bob recorded score 7."
        bundle = {
            "claims": [score_a, score_b],
            "proof": {
                "proof_id": "P-diff",
                "operation": "comparison",
                "required_claim_ids": ["C-score-a", "C-score-b"],
                "reasoning_path_claim_ids": ["C-score-a", "C-score-b"],
                "gold_operand_claim_ids": ["C-score-a", "C-score-b"],
                "dependency_closure_claim_ids": [],
                "gold": 5,
                "answer_type": "number",
                "derivation": {
                    "rule": "absolute_difference",
                    "operand_claim_ids": ["C-score-a", "C-score-b"],
                    "answer_claim_ids": [],
                    "computed_value": 5,
                },
            },
            "support_ledger": [
                {
                    "claim_id": "C-score-a",
                    "turn_id": "D1:1",
                    "exact_support_quote": quote_a,
                },
                {
                    "claim_id": "C-score-b",
                    "turn_id": "D1:2",
                    "exact_support_quote": quote_b,
                },
            ],
            "dialogue": {
                "turns": [
                    {"turn_id": "D1:1", "text": quote_a},
                    {"turn_id": "D1:2", "text": quote_b},
                ]
            },
        }

        self.assertEqual(validator.validate_claim_bundle(bundle), [])

        bad_computed = copy.deepcopy(bundle)
        bad_computed["proof"]["derivation"]["computed_value"] = 6
        self.assertIn(
            "derivation_computed_value_mismatch", issue_codes(bad_computed)
        )

        bad_gold = copy.deepcopy(bundle)
        bad_gold["proof"]["gold"] = 6
        self.assertIn("proof_gold_mismatch", issue_codes(bad_gold))

    def test_count_distinct_is_independently_recomputed(self) -> None:
        claims = []
        for index, (member, included) in enumerate(
            (("Maya", True), ("Noah", False), ("Ava", True), ("Liam", False)),
            start=1,
        ):
            claim = relation_claim(
                f"C-member-{index}",
                subject=member,
                subject_event_id=f"E-member-{index}",
                predicate="member_of" if included else "not_member_of",
                object_value="weekend roster",
                object_event_id="set:weekend-roster",
                time_index=index,
                polarity="positive" if included else "negative",
                condition="submitted a complete draft",
            )
            claim.update(
                {
                    "claim_type": "membership",
                    "included": included,
                    "set_id": "weekend roster",
                }
            )
            claim["canonical_support_claim"].update(
                {
                    "included": included,
                    "set_id": "weekend roster",
                    "entity": "Writers Guild",
                    "slot": "eligible_member",
                }
            )
            claim["truth_signature"] = copy.deepcopy(
                claim["canonical_support_claim"]
            )
            claims.append(claim)
        claim_ids = [claim["claim_id"] for claim in claims]
        included_ids = [
            claim["claim_id"] for claim in claims if claim["included"]
        ]
        proof = {
            "proof_id": "P-count",
            "operation": "count",
            "required_claim_ids": claim_ids,
            "reasoning_path_claim_ids": claim_ids,
            "gold_operand_claim_ids": claim_ids,
            "dependency_closure_claim_ids": claim_ids,
            "gold": 2,
            "gold_answer_text": "2",
            "answer_type": "Number",
            "derivation": {
                "rule": "count_distinct",
                "operand_claim_ids": claim_ids,
                "included_claim_ids": included_ids,
                "answer_claim_ids": [],
                "computed_value": "2",
                "set_id": "weekend roster",
                "filter_condition": "submitted a complete draft",
            },
        }
        self.assertEqual(validator.validate_proof(proof, claims), [])

        reordered = copy.deepcopy(proof)
        reordered_ids = [
            claim_ids[index] for index in (1, 3, 0, 2)
        ]
        reordered["required_claim_ids"] = reordered_ids
        reordered["reasoning_path_claim_ids"] = reordered_ids
        reordered["gold_operand_claim_ids"] = reordered_ids
        reordered["dependency_closure_claim_ids"] = reordered_ids
        self.assertEqual(validator.validate_proof(reordered, claims), [])

        bad_ids = copy.deepcopy(proof)
        bad_ids["derivation"]["included_claim_ids"] = claim_ids[:2]
        self.assertIn(
            "count_included_claim_ids_mismatch",
            {issue["code"] for issue in validator.validate_proof(bad_ids, claims)},
        )
        bad_gold = copy.deepcopy(proof)
        bad_gold["gold_answer_text"] = "3"
        self.assertIn(
            "proof_gold_mismatch",
            {issue["code"] for issue in validator.validate_proof(bad_gold, claims)},
        )

    def test_plain_json_only_and_no_compiler_import(self) -> None:
        bundle = golden_bundle()
        bundle["not_json"] = {"a", "set"}

        self.assertEqual(
            issue_codes(bundle),
            {"bundle_not_plain_json"},
        )
        self.assertNotIn("stage3_claim_compiler", validator.__dict__)
        self.assertNotIn("stage3_claim_ir", validator.__dict__)


if __name__ == "__main__":
    unittest.main()
