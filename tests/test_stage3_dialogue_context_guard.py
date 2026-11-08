from __future__ import annotations

import copy
import unittest

from scripts.current_method import stage3_dialogue_context_guard as guard


def claim(
    claim_id: str,
    subject: str,
    predicate: str,
    objects: list[object],
    *,
    polarity: str = "positive",
    status: str = "active",
    condition: str | None = None,
    scope: dict[str, object] | None = None,
    claim_type: str | None = None,
    canonical_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    canonical = {
        "subject": subject,
        "predicate": predicate,
        "objects": objects,
        "polarity": polarity,
        "status": status,
        "condition": condition,
        "scope": scope,
    }
    canonical.update(canonical_extra or {})
    row = {
        "claim_id": claim_id,
        "canonical_support_claim": canonical,
    }
    if claim_type is not None:
        row["claim_type"] = claim_type
    return row


def plan() -> dict[str, object]:
    return {
        "schema_version": "test_discourse_plan",
        "turn_slots": [
            {
                "turn_slot_id": "TS001",
                "lead_sentence_count": 1,
                "follow_sentence_count": 1,
            },
            {
                "turn_slot_id": "TS002",
                "lead_sentence_count": 1,
                "follow_sentence_count": 1,
            },
        ],
        "semantic_claims": [
            claim("clm_destination", "the workshop", "destination", ["Cedar Hall"]),
            claim(
                "clm_canceled",
                "the workshop",
                "destination",
                ["West Annex"],
                status="canceled",
            ),
            claim("clm_cause", "Alice", "caused", ["the schedule delay"]),
            claim(
                "clm_scope",
                "Noah",
                "view",
                ["the red notebook"],
                status="scope_limited",
                condition="after final approval",
                scope={
                    "observer": "Noah",
                    "permission": "view",
                    "condition": "after final approval",
                },
            ),
        ],
    }


def valid_turns() -> list[dict[str, object]]:
    return [
        {
            "turn_slot_id": "TS001",
            "lead_in_sentences": [
                "The planning group carefully reviews practical concerns before choosing what everyone should discuss next."
            ],
            "follow_up_sentences": [
                "Several participants compare their priorities while keeping the conversation focused on useful follow-up questions."
            ],
        },
        {
            "turn_slot_id": "TS002",
            "lead_in_sentences": [
                "A thoughtful reply connects the previous concern with details that still need collective consideration."
            ],
            "follow_up_sentences": [
                "The exchange closes with concrete next steps and a clear invitation for further discussion."
            ],
        },
    ]


class DialogueContextGuardTest(unittest.TestCase):
    def report_for_sentence(
        self,
        sentence: str,
        *,
        claims: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        frozen_plan = plan()
        if claims is not None:
            frozen_plan["semantic_claims"] = claims
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [sentence]
        return guard.validate_api_authored_context(frozen_plan, turns)

    def test_valid_context_passes_without_mutating_inputs(self) -> None:
        frozen_plan = plan()
        turns = valid_turns()
        before_plan = copy.deepcopy(frozen_plan)
        before_turns = copy.deepcopy(turns)

        report = guard.validate_api_authored_context(frozen_plan, turns)

        self.assertTrue(report["pass"], report["issues"])
        self.assertEqual(report["turn_count"], 2)
        self.assertEqual(report["sentence_count"], 4)
        self.assertEqual(frozen_plan, before_plan)
        self.assertEqual(turns, before_turns)

    def test_turn_schema_allows_only_three_api_owned_fields(self) -> None:
        turns = valid_turns()
        turns[0]["support"] = []
        turns[1].pop("follow_up_sentences")

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertIn("unexpected_flat_turn_keys", report["issue_counts_by_code"])
        self.assertIn("missing_flat_turn_keys", report["issue_counts_by_code"])

    def test_internal_claim_id_leak_is_rejected(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "The speakers explicitly consult clm_deadbeef before continuing their otherwise ordinary planning conversation."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn("internal_identifier_leak", report["issue_counts_by_code"])

    def test_internal_identifier_separator_families_are_rejected(self) -> None:
        identifiers = ("clm:deadbeef", "evt_deadbeef", "TFD:123")
        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                turns = valid_turns()
                turns[0]["lead_in_sentences"] = [
                    f"The planning group carefully consults {identifier} while reviewing practical concerns for the workshop."
                ]

                report = guard.validate_api_authored_context(plan(), turns)

                self.assertFalse(report["pass"])
                self.assertIn(
                    "internal_identifier_leak", report["issue_counts_by_code"]
                )

    def test_frozen_positive_claim_cannot_be_explicitly_negated(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "Cedar Hall was never the workshop destination, despite what everyone had discussed during planning."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "frozen_claim_polarity_contradiction", report["issue_counts_by_code"]
        )

    def test_outer_negation_of_frozen_fact_is_rejected(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "It is not true that Cedar Hall was the workshop destination according to the planning group."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "frozen_claim_surface_contradiction", report["issue_counts_by_code"]
        )

    def test_canceled_claim_cannot_be_described_as_current(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "The workshop destination at West Annex is currently active and definitely remains valid for everyone."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "frozen_claim_status_contradiction", report["issue_counts_by_code"]
        )

    def test_active_claim_rejects_all_generic_nonactive_status_families(self) -> None:
        sentences = {
            "canceled": "The workshop destination at Cedar Hall was canceled before the group finished its careful discussion.",
            "expired": "The workshop destination at Cedar Hall expired before the group finished its careful discussion.",
            "outdated": "The workshop destination at Cedar Hall became outdated before the group finished its careful discussion.",
            "reported": "The workshop destination at Cedar Hall remains an unconfirmed reported detail during this careful discussion.",
        }
        for status_family, sentence in sentences.items():
            with self.subTest(status_family=status_family):
                turns = valid_turns()
                turns[0]["lead_in_sentences"] = [sentence]
                report = guard.validate_api_authored_context(plan(), turns)
                self.assertIn(
                    "frozen_claim_status_contradiction",
                    report["issue_counts_by_code"],
                )

    def test_all_nonactive_status_families_reject_explicit_active_assertion(self) -> None:
        for status in ("canceled", "expired", "outdated", "report"):
            with self.subTest(status=status):
                frozen_plan = plan()
                frozen_plan["semantic_claims"][0]["canonical_support_claim"][
                    "status"
                ] = status
                turns = valid_turns()
                turns[0]["lead_in_sentences"] = [
                    "The workshop destination at Cedar Hall is currently active and definitely confirmed for the whole group."
                ]
                report = guard.validate_api_authored_context(frozen_plan, turns)
                self.assertIn(
                    "frozen_claim_status_contradiction",
                    report["issue_counts_by_code"],
                )

    def test_negative_frozen_claim_rejects_unhedged_positive_assertion(self) -> None:
        frozen_plan = plan()
        frozen_plan["semantic_claims"][0]["canonical_support_claim"][
            "polarity"
        ] = "negative"
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "The workshop destination is Cedar Hall, according to the direct statement everyone accepted during planning."
        ]

        report = guard.validate_api_authored_context(frozen_plan, turns)

        self.assertIn(
            "frozen_claim_polarity_contradiction", report["issue_counts_by_code"]
        )

    def test_same_subject_predicate_with_replacement_object_is_rejected(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "The workshop destination is Riverside Studio according to the latest detailed planning discussion today."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "api_authored_answer_bearing_relation",
            report["issue_counts_by_code"],
        )

    def test_typed_action_and_relation_alternate_objects_are_rejected(self) -> None:
        cases = (
            (
                "Noah may view the blue notebook after final approval during the planning meeting.",
                claim(
                    "clm_view",
                    "Noah",
                    "view",
                    ["the red notebook"],
                    claim_type="visibility",
                ),
            ),
            (
                "Alice caused the budget delay during the detailed planning meeting yesterday.",
                claim(
                    "clm_cause",
                    "Alice",
                    "caused",
                    ["the schedule delay"],
                    claim_type="cause",
                ),
            ),
            (
                "Maya belongs to the second roster according to the detailed planning discussion today.",
                claim(
                    "clm_member",
                    "Maya",
                    "member_of",
                    ["the first roster"],
                    claim_type="membership",
                ),
            ),
        )
        for sentence, frozen_claim in cases:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(sentence, claims=[frozen_claim])
                self.assertIn(
                    "api_authored_answer_bearing_relation",
                    report["issue_counts_by_code"],
                )

    def test_subject_and_predicate_without_typed_object_are_not_answer_bearing(self) -> None:
        sentences = (
            "The workshop destination continues to prompt a careful planning discussion among everyone today.",
            "The workshop destination is still under discussion among the planning group today.",
            "The workshop destination remains undecided while the planning group reviews practical concerns.",
            "The workshop destination is a question that the planning group will revisit tomorrow.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(sentence)
                self.assertNotIn(
                    "api_authored_answer_bearing_relation",
                    report["issue_counts_by_code"],
                )

    def test_run230f_scheduled_event_subject_predicate_family_false_positive(self) -> None:
        report = self.report_for_sentence(
            "I hope the travel club can recover from this setback and reschedule the event.",
            claims=[
                claim(
                    "clm_schedule",
                    "travel club",
                    "scheduled_event",
                    ["faced a setback"],
                )
            ],
        )

        self.assertTrue(report["pass"], report["issues"])
        self.assertNotIn(
            "api_authored_answer_bearing_relation",
            report["issue_counts_by_code"],
        )

    def test_transition_to_value_is_answer_bearing_without_full_from_value(self) -> None:
        transition = claim(
            "clm_transition",
            "Tim's writing project",
            "transitions_to",
            ["Cedar Hall", "city park"],
            claim_type="status_transition",
            canonical_extra={
                "slot": "destination",
                "from_status": "belief",
                "to_status": "active",
            },
        )
        report = self.report_for_sentence(
            "City park has become the clear focus for Tim's writing project now.",
            claims=[transition],
        )

        self.assertIn(
            "api_authored_answer_bearing_relation",
            report["issue_counts_by_code"],
        )
        issue = next(
            row
            for row in report["issues"]
            if row["code"] == "api_authored_answer_bearing_relation"
        )
        self.assertEqual(issue["details"]["match_kinds"], ["transition_to_value"])

    def test_same_sentence_answer_bearing_matches_are_aggregated(self) -> None:
        claims = [
            claim("clm_cedar", "Tim's writing project", "destination", ["Cedar Hall"]),
            claim("clm_city", "Tim's writing project", "destination", ["city park"]),
            claim(
                "clm_europe",
                "Tim's writing project",
                "destination",
                ["northern Europe"],
                status="outdated",
            ),
            claim(
                "clm_members",
                "Tim's writing project",
                "destination",
                ["members only"],
                status="canceled",
            ),
        ]
        report = self.report_for_sentence(
            "The city park is now the confirmed destination for Tim's writing project.",
            claims=claims,
        )

        self.assertEqual(
            report["issue_counts_by_code"]["api_authored_answer_bearing_relation"],
            1,
        )
        issue = next(
            row
            for row in report["issues"]
            if row["code"] == "api_authored_answer_bearing_relation"
        )
        self.assertEqual(issue["details"]["matched_claim_ids"], ["clm_city"])

    def test_object_first_and_passive_same_direction_only_report_answer_bearing(self) -> None:
        cases = (
            (
                "Cedar Hall is the destination for the workshop according to the final planning note.",
                [claim("clm_destination", "the workshop", "destination", ["Cedar Hall"])],
            ),
            (
                "The schedule delay was caused by Alice during the detailed planning discussion yesterday.",
                [claim("clm_cause", "Alice", "caused", ["the schedule delay"])],
            ),
        )
        for sentence, claims in cases:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(sentence, claims=claims)
                self.assertIn(
                    "api_authored_answer_bearing_relation",
                    report["issue_counts_by_code"],
                )
                self.assertNotIn(
                    "frozen_claim_direction_contradiction",
                    report["issue_counts_by_code"],
                )

    def test_active_voice_and_passive_role_reversals_are_rejected(self) -> None:
        cause_claim = claim("clm_cause", "Alice", "caused", ["the schedule delay"])
        sentences = (
            "The schedule delay caused Alice during the detailed planning discussion yesterday.",
            "Alice was caused by the schedule delay during the detailed planning discussion yesterday.",
        )
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(sentence, claims=[cause_claim])
                self.assertIn(
                    "frozen_claim_direction_contradiction",
                    report["issue_counts_by_code"],
                )

    def test_evaluative_stance_does_not_change_active_fact_status(self) -> None:
        cases = (
            (
                "I think having John as the assigned partner for A Dance with Dragons is a solid choice.",
                claim(
                    "clm_partner",
                    "A Dance with Dragons",
                    "assigned_partner",
                    ["John"],
                ),
            ),
            (
                "Canceling the members only destination might have been the right decision for Tim's writing project.",
                claim(
                    "clm_members",
                    "Tim's writing project",
                    "destination",
                    ["members only"],
                    status="canceled",
                ),
            ),
        )
        for sentence, frozen_claim in cases:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(
                    sentence, claims=[frozen_claim]
                )
                self.assertIn(
                    "api_authored_answer_bearing_relation",
                    report["issue_counts_by_code"],
                )
                self.assertNotIn(
                    "frozen_claim_status_contradiction",
                    report["issue_counts_by_code"],
                )

    def test_nonactive_realizations_do_not_reverse_fact_or_status(self) -> None:
        cases = (
            (
                "The West Annex destination was canceled, so it is not part of the current plan for the workshop.",
                claim(
                    "clm_canceled",
                    "the workshop",
                    "destination",
                    ["West Annex"],
                    status="canceled",
                ),
            ),
            (
                "Northern Europe was once the destination for the workshop, but that entry is no longer valid.",
                claim(
                    "clm_outdated",
                    "the workshop",
                    "destination",
                    ["Northern Europe"],
                    status="outdated",
                ),
            ),
        )
        forbidden = {
            "frozen_claim_status_contradiction",
            "frozen_claim_polarity_contradiction",
            "frozen_claim_surface_contradiction",
            "frozen_claim_direction_contradiction",
        }
        for sentence, frozen_claim in cases:
            with self.subTest(sentence=sentence):
                report = self.report_for_sentence(sentence, claims=[frozen_claim])
                self.assertIn(
                    "api_authored_answer_bearing_relation",
                    report["issue_counts_by_code"],
                )
                self.assertTrue(
                    forbidden.isdisjoint(report["issue_counts_by_code"]), report
                )

    def test_epistemic_prefix_still_conflicts_with_active_status(self) -> None:
        report = self.report_for_sentence(
            "Reportedly, Cedar Hall is the destination for the workshop according to an uncertain source today.",
            claims=[claim("clm_destination", "the workshop", "destination", ["Cedar Hall"])],
        )

        self.assertIn(
            "frozen_claim_status_contradiction", report["issue_counts_by_code"]
        )

    def test_narrative_when_is_not_a_claim_condition(self) -> None:
        report = self.report_for_sentence(
            "I remember when we first discussed the workshop, the destination was Cedar Hall, but the early conversation remained uncertain.",
            claims=[
                claim(
                    "clm_destination",
                    "the workshop",
                    "destination",
                    ["Cedar Hall"],
                    status="belief",
                )
            ],
        )

        self.assertIn(
            "api_authored_answer_bearing_relation",
            report["issue_counts_by_code"],
        )
        self.assertNotIn(
            "frozen_claim_unexpected_condition", report["issue_counts_by_code"]
        )

    def test_real_wrong_conditions_remain_rejected(self) -> None:
        no_condition = self.report_for_sentence(
            "If the committee approves tomorrow, the workshop destination is Cedar Hall for the whole group.",
            claims=[claim("clm_destination", "the workshop", "destination", ["Cedar Hall"])],
        )
        self.assertIn(
            "frozen_claim_unexpected_condition",
            no_condition["issue_counts_by_code"],
        )

        wrong_condition = self.report_for_sentence(
            "Noah may view the red notebook if the manager signs tomorrow during the planning meeting.",
            claims=[
                claim(
                    "clm_scope",
                    "Noah",
                    "view",
                    ["the red notebook"],
                    status="scope_limited",
                    condition="after final approval",
                    scope={
                        "observer": "Noah",
                        "permission": "view",
                        "condition": "after final approval",
                    },
                )
            ],
        )
        self.assertIn(
            "frozen_claim_condition_mismatch",
            wrong_condition["issue_counts_by_code"],
        )

    def test_scoped_condition_cannot_be_made_unconditional(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "Noah can view the red notebook unconditionally, without any conditions or access restrictions whatsoever."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "frozen_claim_condition_contradiction", report["issue_counts_by_code"]
        )
        self.assertIn(
            "frozen_claim_scope_contradiction", report["issue_counts_by_code"]
        )

    def test_directed_relation_cannot_be_reversed(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = [
            "The schedule delay caused Alice during the discussion, reversing the relationship everyone had considered."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertFalse(report["pass"])
        self.assertIn(
            "frozen_claim_direction_contradiction", report["issue_counts_by_code"]
        )

    def test_exact_and_near_sentence_duplicates_are_rejected_globally(self) -> None:
        turns = valid_turns()
        sentence = turns[0]["lead_in_sentences"][0]
        turns[0]["follow_up_sentences"] = [sentence]
        turns[1]["lead_in_sentences"] = [
            "The planning group carefully reviews practical concerns before selecting what everyone should discuss next."
        ]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertIn(
            "exact_context_sentence_duplicate", report["issue_counts_by_code"]
        )
        self.assertIn(
            "near_context_sentence_duplicate", report["issue_counts_by_code"]
        )

    def test_short_generic_sentence_is_not_substantive(self) -> None:
        turns = valid_turns()
        turns[0]["lead_in_sentences"] = ["That makes sense."]

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertIn(
            "context_sentence_not_substantive", report["issue_counts_by_code"]
        )
        self.assertIn("generic_context_filler", report["issue_counts_by_code"])

    def test_wrong_slot_order_and_sentence_counts_are_rejected(self) -> None:
        turns = valid_turns()
        turns.reverse()
        turns[0]["lead_in_sentences"] = []

        report = guard.validate_api_authored_context(plan(), turns)

        self.assertEqual(report["issue_counts_by_code"]["turn_slot_order_mismatch"], 2)
        self.assertIn(
            "context_sentence_count_mismatch", report["issue_counts_by_code"]
        )


if __name__ == "__main__":
    unittest.main()
