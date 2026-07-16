from __future__ import annotations

import copy
import unittest

from scripts.current_method import rebuild_stage2_canonical_graphs as subject


class Stage2CanonicalGraphRebuildTests(unittest.TestCase):
    def test_request_excludes_legacy_qa_events_with_polluted_evidence(self) -> None:
        row = {
            "sample_id": "s1",
            "_source_line_number": 3,
            "dialogue": {
                "turns": [
                    {"turn_id": "t1", "speaker": "Maya", "text": "Maya moved to Berlin last spring."},
                    {"turn_id": "t2", "speaker": "Analyst", "text": "Maya raw_qa: Q: Where? A: Berlin"},
                ]
            },
            "graph_update": {
                "graph_snapshot_hash": "g1",
                "events": [
                    {
                        "event_instance_id": "e1",
                        "entity": "Maya",
                        "slot": "raw_qa_location",
                        "value": "Q: Where did Maya move? A: Berlin",
                        "status": "active",
                        "time_index": 1,
                        "turn_ids": ["t1"],
                    },
                    {
                        "event_instance_id": "e2",
                        "entity": "Maya",
                        "slot": "raw_qa_location",
                        "value": "Q: Where? A: Berlin",
                        "status": "outdated",
                        "time_index": 2,
                        "turn_ids": ["t2"],
                    },
                ],
            },
        }
        request = subject.extraction_request_for_row(row)
        self.assertIsNotNone(request)
        self.assertEqual([item["source_event_id"] for item in request["events"]], ["e1"])
        self.assertEqual(request["skipped_events"][0]["reason"], "legacy_qa_evidence_turn")
        self.assertNotIn("Q:", subject.build_prompt(request))
        self.assertNotIn("raw_qa", subject.build_prompt(request))

    def test_validate_extracted_event_requires_exact_evidence_value(self) -> None:
        source = {
            "source_event_id": "e1",
            "source_event_order": 1,
            "evidence": [
                {
                    "turn_id": "t1",
                    "speaker": "Maya",
                    "text": "Maya moved to Berlin last spring.",
                }
            ],
        }
        valid = {
            "source_event_id": "e1",
            "entity": "Maya",
            "slot": "moved_residence_city",
            "value": "Berlin",
            "status": "active",
            "supporting_span": "Maya moved to Berlin",
            "relation_supporting_span": "moved to Berlin",
            "status_supporting_span": "",
            "supported": True,
            "rejection_reason": "",
        }
        event, errors = subject.validate_extracted_event(source, valid, request_id="r1")
        self.assertEqual(errors, [])
        self.assertEqual(event["value"], "Berlin")
        invented = dict(valid, value="Paris")
        event, errors = subject.validate_extracted_event(source, invented, request_id="r1")
        self.assertIsNone(event)
        self.assertIn("value_not_exact_evidence_span", errors)

    def test_nonactive_status_requires_explicit_evidence_span(self) -> None:
        source = {
            "source_event_id": "e2",
            "source_event_order": 2,
            "evidence": [
                {
                    "turn_id": "t2",
                    "speaker": "Maya",
                    "text": "Maya's Berlin plan is an old version and no longer applies.",
                }
            ],
        }
        extracted = {
            "source_event_id": "e2",
            "entity": "Maya",
            "slot": "plan_relocation",
            "value": "Berlin plan",
            "status": "outdated",
            "supporting_span": "Maya's Berlin plan",
            "relation_supporting_span": "Berlin plan",
            "status_supporting_span": "old version",
            "supported": True,
            "rejection_reason": "",
        }
        event, errors = subject.validate_extracted_event(source, extracted, request_id="r2")
        self.assertEqual(errors, [])
        self.assertEqual(event["status"], "outdated")
        contradictory_source = copy.deepcopy(source)
        contradictory_source["evidence"][0]["text"] += " It has also expired."
        event, errors = subject.validate_extracted_event(
            contradictory_source, extracted, request_id="r2"
        )
        self.assertEqual(errors, [])
        self.assertEqual(event["status"], "expired")

        truly_contradictory_source = copy.deepcopy(contradictory_source)
        truly_contradictory_source["evidence"][0]["text"] += " It was also called off."
        event, errors = subject.validate_extracted_event(
            truly_contradictory_source, extracted, request_id="r2"
        )
        self.assertIsNone(event)
        self.assertIn("ambiguous_evidence_status", errors)

    def test_outer_modal_is_preserved_in_slot(self) -> None:
        source = {
            "source_event_id": "modal-1",
            "source_event_order": 1,
            "evidence": [
                {
                    "turn_id": "t1",
                    "speaker": "Analyst",
                    "text": "John is considering adopting a rescue dog to help his family.",
                }
            ],
        }
        extracted = {
            "source_event_id": "modal-1",
            "entity": "John",
            "slot": "adopted_pet",
            "value": "adopting a rescue dog",
            "supporting_span": "adopting a rescue dog",
            "supported": True,
        }

        event, errors = subject.validate_extracted_event(
            source, extracted, request_id="modal-request"
        )

        self.assertEqual(errors, [])
        self.assertEqual(event["slot"], "considering")
        self.assertEqual(
            event["provenance"]["relation_repair_source"],
            "local_outer_modal_slot",
        )

    def test_direct_object_and_count_clauses_override_wrong_model_roles(self) -> None:
        direct_source = {
            "source_event_id": "direct-1",
            "source_event_order": 1,
            "evidence": [
                {
                    "turn_id": "t1",
                    "speaker": "Analyst",
                    "text": "Nate got custom controller decorations for his gaming party attendees.",
                }
            ],
        }
        wrong_direct = {
            "source_event_id": "direct-1",
            "entity": "Nate",
            "slot": "playing_game",
            "value": "gaming party attendees",
            "supporting_span": "Nate got custom controller decorations for his gaming party attendees.",
            "supported": True,
        }
        event, errors = subject.validate_extracted_event(
            direct_source, wrong_direct, request_id="direct-request"
        )
        self.assertEqual(errors, [])
        self.assertEqual((event["slot"], event["value"]), ("got_item", "custom controller decorations"))

        count_source = {
            "source_event_id": "count-1",
            "source_event_order": 2,
            "evidence": [
                {
                    "turn_id": "t2",
                    "speaker": "Reviewer",
                    "text": "Joanna's gaming party in June 2022 had 7 attendees.",
                }
            ],
        }
        wrong_count = {
            "source_event_id": "count-1",
            "entity": "Joanna's gaming party in June 2022",
            "slot": "playing_game",
            "value": "gaming party",
            "supporting_span": "Joanna's gaming party in June 2022 had 7 attendees.",
            "supported": True,
        }
        event, errors = subject.validate_extracted_event(
            count_source, wrong_count, request_id="count-request"
        )
        self.assertEqual(errors, [])
        self.assertEqual((event["slot"], event["value"]), ("attendee_count", "7 attendees"))

    def test_false_destination_keeps_discriminating_qualifier(self) -> None:
        source = {
            "source_event_id": "false-1",
            "source_event_order": 1,
            "evidence": [
                {
                    "turn_id": "t1",
                    "speaker": "Reviewer",
                    "text": (
                        "Maria donated her old car to a luxury store in December 2023. "
                        "The speaker corrects this as not true"
                    ),
                }
            ],
        }
        extracted = {
            "source_event_id": "false-1",
            "entity": "Maria",
            "slot": "donated_item",
            "value": "her old car",
            "supporting_span": "Maria donated her old car to a luxury store in December 2023.",
            "supported": True,
        }

        event, errors = subject.validate_extracted_event(
            source, extracted, request_id="false-request"
        )

        self.assertEqual(errors, [])
        self.assertEqual(event["status"], "false")
        self.assertEqual(event["slot"], "donated_to")
        self.assertEqual(event["value"], "a luxury store in December 2023")

    def test_dialogue_fallback_is_disabled_by_default(self) -> None:
        row = {
            "sample_id": "no-linked-events",
            "dialogue": {
                "turns": [
                    {
                        "turn_id": "t1",
                        "speaker": "Andrew",
                        "text": "I officially started my new role last Monday and enjoy the work.",
                    }
                ]
            },
            "graph_update": {"events": []},
        }

        request = subject.extraction_request_for_row(row)

        self.assertEqual(request["events"], [])
        self.assertEqual(request["fallback_event_count"], 0)

    def test_terminal_dependency_links_only_matching_prior_active_fact(self) -> None:
        events = [
            {
                "event_id": "active-party",
                "entity": "Nate",
                "slot": "hosted_gaming_party",
                "value": "hosted a gaming party in June 2022",
                "status": "active",
                "time_index": 1,
            },
            {
                "event_id": "active-unrelated",
                "entity": "Nate",
                "slot": "book_preference",
                "value": "science fiction novels",
                "status": "active",
                "time_index": 2,
            },
            {
                "event_id": "expired-party",
                "entity": "Nate",
                "slot": "gaming_party",
                "value": "gaming party in June 2022",
                "status": "expired",
                "time_index": 3,
                "provenance": {},
            },
            {
                "event_id": "expired-unmatched",
                "entity": "Nate",
                "slot": "travel_plan",
                "value": "trip to Oslo",
                "status": "expired",
                "time_index": 4,
                "provenance": {},
            },
            {
                "event_id": "active-hiking",
                "entity": "Andrew",
                "slot": "going_hiking",
                "value": "going hiking",
                "status": "active",
                "time_index": 5,
            },
            {
                "event_id": "canceled-hiking",
                "entity": "Andrew",
                "slot": "hiking_plans",
                "value": "hiking plans with Audrey in August",
                "status": "canceled",
                "time_index": 6,
                "provenance": {},
            },
            {
                "event_id": "active-owned-party",
                "entity": "Joanna's gaming party in June 2022",
                "slot": "attendee_count",
                "value": "7 attendees",
                "status": "active",
                "time_index": 7,
            },
            {
                "event_id": "expired-owned-party",
                "entity": "Joanna",
                "slot": "gaming_party",
                "value": "gaming party in June 2022",
                "status": "expired",
                "time_index": 8,
                "provenance": {},
            },
            {
                "event_id": "active-colleague-invite",
                "entity": "John's colleague Rob",
                "slot": "invited_to",
                "value": "a beginner's yoga class",
                "status": "active",
                "time_index": 9,
            },
            {
                "event_id": "delayed-john-participation",
                "entity": "John",
                "slot": "participation",
                "value": "participation in the beginner's yoga class",
                "status": "delayed_not_active",
                "time_index": 10,
                "provenance": {},
            },
        ]

        linked = subject.link_terminal_dependencies(events)
        by_id = {event["event_id"]: event for event in linked}

        self.assertEqual(by_id["expired-party"]["depends_on"], ["active-party"])
        self.assertNotIn("depends_on", by_id["expired-unmatched"])
        self.assertEqual(by_id["canceled-hiking"]["depends_on"], ["active-hiking"])
        self.assertEqual(
            by_id["expired-owned-party"]["depends_on"],
            ["active-owned-party"],
        )
        self.assertNotIn("depends_on", by_id["delayed-john-participation"])

    def test_reviewed_semantic_exclusion_is_applied_before_admission(self) -> None:
        requests = [
            {
                "request_id": "old-request-id",
                "source_sample_id": "sample-1",
                "source_line_number": 7,
                "events": [
                    {"source_event_id": "keep", "source_event_order": 1},
                    {"source_event_id": "drop", "source_event_order": 2},
                ],
                "skipped_events": [],
            }
        ]
        exclusions = [
            {
                "schema_version": subject.SEMANTIC_EXCLUSION_SCHEMA_VERSION,
                "source_line_number": 7,
                "source_event_id": "drop",
                "reason": "predicate_slot_mismatch",
                "detail": "reviewed mismatch",
                "decision": "quarantine",
            }
        ]

        applied = subject.apply_semantic_audit_exclusions(requests, exclusions)

        self.assertEqual(
            [event["source_event_id"] for event in requests[0]["events"]],
            ["keep"],
        )
        self.assertNotEqual(requests[0]["request_id"], "old-request-id")
        self.assertEqual(
            requests[0]["skipped_events"][0]["reason"],
            "independent_semantic_audit_exclusion",
        )
        self.assertEqual(
            applied[0]["disposition"],
            "quarantined_before_canonical_admission",
        )

    def test_wrong_entity_status_value_and_typed_slot_fail_closed(self) -> None:
        source = {
            "source_event_id": "e3",
            "source_event_order": 3,
            "evidence": [
                {
                    "turn_id": "t3",
                    "speaker": "Joanna",
                    "text": (
                        "Joanna planned a cooking show for July 8, but the proposal "
                        "was called off."
                    ),
                }
            ],
        }
        base = {
            "source_event_id": "e3",
            "entity": "Joanna",
            "slot": "planned_cooking_show_date",
            "value": "July 8",
            "status": "canceled",
            "supporting_span": "Joanna planned a cooking show for July 8",
            "relation_supporting_span": "planned a cooking show for July 8",
            "status_supporting_span": "called off",
            "supported": True,
            "rejection_reason": "",
        }
        event, errors = subject.validate_extracted_event(source, base, request_id="r3")
        self.assertEqual(errors, [])
        self.assertEqual(event["time_index"], 3)

        wrong_entity = dict(base, entity="Nate")
        event, errors = subject.validate_extracted_event(
            source, wrong_entity, request_id="r3"
        )
        self.assertEqual(errors, [])
        self.assertEqual(event["entity"], "Joanna")
        self.assertEqual(event["provenance"]["model_entity"], "Nate")

        status_as_value = dict(base, value="called off")
        event, errors = subject.validate_extracted_event(
            source, status_as_value, request_id="r3"
        )
        self.assertIsNone(event)
        self.assertIn("status_language_used_as_value", errors)
        self.assertIn("value_overlaps_status_supporting_span", errors)

        wrong_date_value = dict(base, value="cooking show")
        event, errors = subject.validate_extracted_event(
            source, wrong_date_value, request_id="r3"
        )
        self.assertIsNone(event)
        self.assertIn("slot_relation_not_supported", errors)

    def test_completed_fact_cannot_receive_canceled_suffix(self) -> None:
        source = {
            "source_event_id": "e4",
            "source_event_order": 4,
            "evidence": [
                {
                    "turn_id": "t4",
                    "speaker": "Audrey",
                    "text": (
                        "Audrey joined the church last year. That plan was called off."
                    ),
                }
            ],
        }
        extracted = {
            "source_event_id": "e4",
            "entity": "Audrey",
            "slot": "joined_church_date",
            "value": "last year",
            "status": "canceled",
            "supporting_span": "Audrey joined the church last year",
            "relation_supporting_span": "joined the church last year",
            "status_supporting_span": "called off",
            "supported": True,
            "rejection_reason": "",
        }
        event, errors = subject.validate_extracted_event(
            source, extracted, request_id="r4"
        )
        self.assertIsNone(event)
        self.assertIn("nonactive_plan_status_without_prospective_fact", errors)

        direct_source = copy.deepcopy(source)
        direct_source["evidence"][0]["text"] = "Andrew's fishing trip was canceled."
        direct = dict(
            extracted,
            source_event_id="e4",
            entity="Andrew",
            slot="fishing_trip",
            value="fishing trip",
            supporting_span="Andrew's fishing trip was canceled.",
        )
        event, errors = subject.validate_extracted_event(
            direct_source, direct, request_id="r4"
        )
        self.assertEqual(errors, [])
        self.assertEqual(event["status"], "canceled")


if __name__ == "__main__":
    unittest.main()
