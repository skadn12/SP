from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.current_method import native_api_graph_dialogue_expansion as subject


def stage3_policy(operation: str = "relation_lookup", difficulty: str | None = None) -> dict:
    policy = {
        "native_curriculum_mode": subject.EASY_V2_MODE,
        "planned_cell": {
            "question_type": "single_choice",
            "attack_skill": "conditional_plan",
        },
        "stage64_target": {
            "qa_operation": operation,
            "preferred_answer_type": "ObjectOrCommonNoun",
        },
    }
    if difficulty:
        policy["item_difficulty_target"] = difficulty
    return policy


def seed_row() -> dict:
    return {
        "sample_id": "stage2-parent-1",
        "dialogue": {
            "turns": [
                {"turn_id": "D1:1", "speaker": "Maya", "text": "I left my notebook by the lamp."},
                {"turn_id": "D1:2", "speaker": "Noah", "text": "I will remember that."},
            ]
        },
        "graph_update": {
            "events": [
                {
                    "event_id": "legacy-1",
                    "entity": "Maya",
                    "slot": "raw_qa_single_hop",
                    "value": "Q: What did Maya leave by the lamp? A: notebook",
                    "status": "active",
                },
                {
                    "event_id": "clean-1",
                    "entity": "Maya",
                    "slot": "item_beside_lamp",
                    "value": "notebook",
                    "status": "active",
                    "evidence_turn_ids": ["D1:1"],
                },
            ]
        },
        "qa": {"question": "legacy question", "gold": ["A"]},
    }


def raw_events(count: int) -> list[dict]:
    values = ["3", "Yes", "Paris", "yoga", "violin", "July 8", "notebook"]
    return [
        {
            "time_index": index + 1,
            "event_type": "state_change",
            "entity": f"entity-{index}",
            "slot": f"semantic_slot_{index}",
            "value": values[index],
            "status": "active",
            "required_for_dialogue": True,
        }
        for index in range(count)
    ]


def run106_temporal_two_chain_events() -> list[dict]:
    """Reduced lossless fixture from run106's repeated 4+4 temporal failure."""

    return [
        {"event_id": "E1", "entity": "John", "slot": "political_aspiration", "value": "local politics", "status": "active", "time_index": 1, "previous_value": None},
        {"event_id": "E2", "entity": "John", "slot": "political_aspiration", "value": "military service", "status": "expired", "time_index": 2, "previous_value": "local politics"},
        {"event_id": "E3", "entity": "John", "slot": "political_aspiration", "value": "community advocacy", "status": "outdated", "time_index": 3, "previous_value": "military service"},
        {"event_id": "E4", "entity": "John", "slot": "political_aspiration", "value": "local politics", "status": "restored", "time_index": 4, "previous_value": "community advocacy"},
        {"event_id": "E5", "entity": "Maria", "slot": "volunteer_activity", "value": "homeless shelter", "status": "active", "time_index": 5, "previous_value": None},
        {"event_id": "E6", "entity": "Maria", "slot": "volunteer_activity", "value": "yoga instructor", "status": "expired", "time_index": 6, "previous_value": "homeless shelter"},
        {"event_id": "E7", "entity": "Maria", "slot": "volunteer_activity", "value": "mental health support", "status": "outdated", "time_index": 7, "previous_value": "yoga instructor"},
        {"event_id": "E8", "entity": "Maria", "slot": "volunteer_activity", "value": "homeless shelter", "status": "restored", "time_index": 8, "previous_value": "mental health support"},
    ]


def exact_temporal_generation_events() -> list[dict]:
    return [
        {"event_id": "E1", "entity": "John", "slot": "political_aspiration", "value": "local politics", "status": "outdated", "time_index": 1},
        {"event_id": "E2", "entity": "John", "slot": "political_aspiration", "value": "military service", "status": "expired", "time_index": 2, "previous_value": "local politics", "depends_on": ["E1"]},
        {"event_id": "E3", "entity": "John", "slot": "political_aspiration", "value": "community advocacy", "status": "canceled", "time_index": 3, "previous_value": "military service", "depends_on": ["E2"]},
        {"event_id": "E4", "entity": "John", "slot": "political_aspiration", "value": "municipal planning", "status": "active", "time_index": 4, "previous_value": "community advocacy", "depends_on": ["E3"]},
        {"event_id": "E5", "entity": "Maria", "slot": "volunteer_activity", "value": "homeless shelter", "status": "active", "time_index": 5},
        {"event_id": "E6", "entity": "Tim", "slot": "project_interest", "value": "film archive", "status": "outdated", "time_index": 6},
        {"event_id": "E7", "entity": "Li", "slot": "hotel_name", "value": "Orchid", "status": "active", "time_index": 7},
        {"event_id": "E8", "entity": "Omar", "slot": "train_number", "value": "T18", "status": "canceled", "time_index": 8},
    ]


def exact_temporal_anti_shortcut_generation_events() -> list[dict]:
    events = exact_temporal_generation_events()
    events[1]["status"] = "active"
    events[3]["status"] = "canceled"
    return events


def temporal_graph_payload(events: list[dict]) -> dict:
    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "stage3:S0000:semantic_graph_v2",
        "parent_graph_id": "stage3_parent:stage2-parent-1",
        "operation": "temporal_state_update",
        "question_type": "single_choice",
        "attack_skill": "conditional_plan",
        "events": copy.deepcopy(events),
        "false_candidates": [],
        "metadata": {},
    }


def comparison_graph_payload(events: list[dict]) -> dict:
    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "stage3:S0000:semantic_graph_v2",
        "parent_graph_id": "stage3_parent:stage2-parent-1",
        "operation": "comparison",
        "question_type": "single_choice",
        "attack_skill": "conditional_plan",
        "events": copy.deepcopy(events),
        "false_candidates": [],
        "metadata": {},
    }


def count_graph_payload(events: list[dict]) -> dict:
    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "stage3:S0000:semantic_graph_v2",
        "parent_graph_id": "stage3_parent:stage2-parent-1",
        "operation": "count",
        "question_type": "single_choice",
        "attack_skill": "conditional_plan",
        "events": copy.deepcopy(events),
        "false_candidates": [],
        "metadata": {},
    }


def exact_count_generation_events() -> list[dict]:
    values = (
        "strong",
        "moderate",
        "weak",
        "emerging",
        "very strong",
        "average",
        "very weak",
        "non-existent",
    )
    return [
        {
            "event_id": f"E{index}",
            "entity": "Andrew",
            "slot": "bond_strength_with_toby",
            "value": value,
            "status": "active",
            "time_index": 1,
            "set_membership": {
                "set_id": "bond_strength_with_toby_set",
                "condition": "bond strength is at least emerging",
                "included": index <= 4,
            },
        }
        for index, value in enumerate(values, start=1)
    ]


def run115_duplicate_count_member_events() -> list[dict]:
    """Lossless semantic fields from run115's final repeated count failure."""

    events = exact_count_generation_events()
    events[3]["value"] = "non-existent"
    return events


def run109_screenplay_second_pair_events() -> list[dict]:
    """Lossless semantic fields from run109's repeated screenplay response."""

    return [
        {"event_id": "E1", "entity": "Joanna", "slot": "drama_idea_progress", "value": 45, "status": "active", "time_index": 1},
        {"event_id": "E2", "entity": "Nate", "slot": "drama_idea_progress", "value": 30, "status": "active", "time_index": 1},
        {"event_id": "E3", "entity": "Joanna", "slot": "mood_level", "value": "low", "status": "active", "time_index": 1},
        {"event_id": "E4", "entity": "Nate", "slot": "mood_level", "value": "moderate", "status": "active", "time_index": 1},
        {"event_id": "E5", "entity": "Joanna", "slot": "drama_idea_progress", "value": 45, "status": "expired", "time_index": 2},
        {"event_id": "E6", "entity": "Nate", "slot": "drama_idea_progress", "value": 30, "status": "delayed_not_active", "time_index": 2},
        {"event_id": "E7", "entity": "Joanna", "slot": "screenplay_ideas", "value": 2, "status": "active", "time_index": 1},
        {"event_id": "E8", "entity": "Nate", "slot": "screenplay_ideas", "value": 3, "status": "active", "time_index": 1},
    ]


def run109_art_show_date_second_pair_events() -> list[dict]:
    """Lossless semantic fields from run109's repeated art-show response."""

    return [
        {"event_id": "E1", "entity": "Caroline", "slot": "art_show_attendance", "value": 250, "status": "active", "time_index": 1},
        {"event_id": "E2", "entity": "Melanie", "slot": "art_show_attendance", "value": 180, "status": "active", "time_index": 1},
        {"event_id": "E3", "entity": "Caroline", "slot": "art_show_theme", "value": "LGBTQ Unity", "status": "active", "time_index": 1},
        {"event_id": "E4", "entity": "Caroline", "slot": "art_show_date", "value": 15, "status": "active", "time_index": 1},
        {"event_id": "E5", "entity": "Melanie", "slot": "art_show_theme", "value": "LGBTQ Advocacy", "status": "active", "time_index": 1},
        {"event_id": "E6", "entity": "Melanie", "slot": "art_show_date", "value": 20, "status": "active", "time_index": 1},
        {"event_id": "E7", "entity": "Caroline", "slot": "art_show_location", "value": "Gallery A", "status": "active", "time_index": 1},
        {"event_id": "E8", "entity": "Melanie", "slot": "art_show_location", "value": "Community Center B", "status": "active", "time_index": 1},
    ]


def run109_target1_core_valid_events() -> list[dict]:
    """Semantic fields from run109 S0001, accepted by the general core gate."""

    return [
        {"event_id": "E1", "entity": "Caroline", "slot": "art_show_attendance", "value": 150, "status": "active", "time_index": 1},
        {"event_id": "E2", "entity": "Melanie", "slot": "art_show_attendance", "value": 120, "status": "active", "time_index": 1},
        {"event_id": "E3", "entity": "Caroline", "slot": "painting_theme", "value": "Unity and strength", "status": "active", "time_index": 1},
        {"event_id": "E4", "entity": "Caroline", "slot": "art_show_date", "value": 15, "status": "active", "time_index": 1},
        {"event_id": "E5", "entity": "Melanie", "slot": "art_show_date", "value": "March 15", "status": "active", "time_index": 1},
        {"event_id": "E6", "entity": "Caroline", "slot": "painting_style", "value": "Abstract expressionism", "status": "active", "time_index": 1},
        {"event_id": "E7", "entity": "Melanie", "slot": "painting_style", "value": "Realism", "status": "active", "time_index": 1},
        {"event_id": "E8", "entity": "Caroline", "slot": "art_show_attendance", "value": 150, "status": "expired", "time_index": 1},
    ]


class Stage3GraphGenerationTest(unittest.TestCase):
    @staticmethod
    def _dispersion_fixture(indices: list[int], appended_turn_count: int = 12) -> tuple[dict, dict, dict]:
        parent_turns = [
            {"turn_id": "D1:1", "speaker": "Maya", "text": "We should review the plan."},
            {"turn_id": "D1:2", "speaker": "Noah", "text": "Let's take it carefully."},
        ]
        appended_turns = [
            {
                "turn_id": f"D2:{index + 1}",
                "speaker": "Maya" if index % 2 == 0 else "Noah",
                "text": f"Natural appended discussion turn {index + 1}.",
            }
            for index in range(appended_turn_count)
        ]
        event_ids = [f"event-{index + 1}" for index in range(len(indices))]
        plan = {
            "events": [
                {"event_instance_id": event_id, "required_for_dialogue": True}
                for event_id in event_ids
            ]
        }
        alignment = {
            event_id: {"turn_ids": [f"D2:{turn_index + 1}"]}
            for event_id, turn_index in zip(event_ids, indices)
        }
        dialogue = {"turns": parent_turns + appended_turns}
        return plan, dialogue, alignment

    @staticmethod
    def _local_stage64_qa_repair_kwargs(slot: str = "slot-a") -> dict:
        hashes = {
            "graph_snapshot_hash": f"graph-{slot}",
            "dialogue_hash": f"dialogue-{slot}",
            "alignment_hash": f"alignment-{slot}",
        }
        return {
            "current_event_plan": {
                "operation": "comparison",
                "question_type": "single_choice",
                "graph_snapshot_hash": hashes["graph_snapshot_hash"],
                "events": [
                    {
                        "event_instance_id": f"{slot}:E1",
                        "entity": slot,
                        "slot": "score",
                        "value": 3,
                    }
                ],
            },
            "graph_update": {
                "operation": "comparison",
                "events": [{"event_instance_id": f"{slot}:E1"}],
            },
            "current_alignment": {
                f"{slot}:E1": {"turn_ids": ["D2:1"]}
            },
            "initial_qa_proposal": {
                "slot": slot,
                "revision": 0,
                "query_plan": {"qa_operation": "comparison"},
            },
            "question_type": "single_choice",
            "rng": random.Random(11),
            "hashes": hashes,
            "final_dialogue": {
                "dialogue_hash": hashes["dialogue_hash"],
                "turns": [
                    {
                        "turn_id": "D2:1",
                        "speaker": "Maya",
                        "text": f"Frozen dialogue for {slot}.",
                    }
                ],
            },
            "policy": {
                "qa_operation_answer_type_diversification": True,
                "stage64_target": {
                    "qa_operation": "comparison",
                    "preferred_answer_type": "Number",
                },
            },
            "seed_row": {"sample_id": f"parent-{slot}"},
            "appended_dialogue_turns": [
                {
                    "turn_id": "D2:1",
                    "speaker": "Maya",
                    "text": f"Frozen dialogue for {slot}.",
                }
            ],
            "api_args": SimpleNamespace(),
            "max_repair_attempts": 3,
            "item_difficulty_target": "easy",
        }

    def test_meta_leak_allows_domain_current_state_of_only_in_dialogue(self) -> None:
        turns = [
            {
                "speaker": "Nate",
                "text": "What's the current state of quantum computing?",
            },
            {
                "speaker": "Joanna",
                "text": "The current state of quantum computing is the NISQ era.",
            },
        ]

        top_level_ok, top_level_details = subject.assert_no_meta_leakage(
            {"turns": turns}
        )
        self.assertTrue(top_level_ok)
        self.assertEqual(top_level_details["meta_leak_count"], 0)

        clean_row_ok, clean_row_details = subject.assert_no_meta_leakage(
            {
                "dialogue": {"turns": turns},
                "expanded_dialogue_text": " ".join(turn["text"] for turn in turns),
            }
        )
        self.assertTrue(clean_row_ok)
        self.assertEqual(clean_row_details["meta_leak_count"], 0)

        qa_ok, qa_details = subject.assert_no_meta_leakage(
            {"qa": {"question": "What is the current state of quantum computing?"}}
        )
        self.assertFalse(qa_ok)
        self.assertEqual(qa_details["banned_term_counts"]["current_state"], 1)

    def test_meta_leak_scans_top_level_turns_and_rejects_internal_current_state(self) -> None:
        forbidden = (
            "The current state is active.",
            "Review the current state after all updates.",
            "This is the active current state.",
            "The current_state value is active.",
            "The current state of the graph slot is active.",
            "The current state of E4 is active.",
            "The current state of E4's status is active.",
            "The current state of graph_update is active.",
            "The current state of semantic_graph_v17 is active.",
            "The current state of Stage64 is active.",
        )

        for text in forbidden:
            with self.subTest(text=text):
                ok, details = subject.assert_no_meta_leakage(
                    {"turns": [{"speaker": "Nate", "text": text}]}
                )
                self.assertFalse(ok)
                self.assertGreater(details["meta_leak_count"], 0)

    def test_stage64_evidence_dispersion_rejects_consecutive_tail_fact_table(self) -> None:
        plan, dialogue, alignment = self._dispersion_fixture([8, 9, 10, 11])

        diagnostics = subject.stage64_evidence_dispersion_diagnostics(
            plan,
            dialogue,
            alignment,
            parent_turn_count=2,
        )

        self.assertFalse(diagnostics["pass"])
        self.assertEqual(diagnostics["evidence_span_turns"], 4)
        self.assertEqual(diagnostics["max_consecutive_evidence_run"], 4)
        self.assertEqual(diagnostics["tail_concentration"], 0.75)
        self.assertIn("evidence_span_too_small", diagnostics["failures"])
        self.assertIn("evidence_consecutive_run_too_long", diagnostics["failures"])
        self.assertIn("evidence_tail_concentration_too_high", diagnostics["failures"])

    def test_stage64_evidence_dispersion_accepts_early_middle_late_evidence(self) -> None:
        plan, dialogue, alignment = self._dispersion_fixture([0, 3, 6, 8])

        diagnostics = subject.stage64_evidence_dispersion_diagnostics(
            plan,
            dialogue,
            alignment,
            parent_turn_count=2,
        )

        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["evidence_span_turns"], 9)
        self.assertEqual(diagnostics["max_consecutive_evidence_run"], 1)
        self.assertEqual(diagnostics["tail_concentration"], 0.0)

    def test_stage64_evidence_dispersion_accepts_six_events_on_five_turns(self) -> None:
        plan, dialogue, alignment = self._dispersion_fixture([0, 2, 4, 6, 8, 8])

        diagnostics = subject.stage64_evidence_dispersion_diagnostics(
            plan,
            dialogue,
            alignment,
            parent_turn_count=2,
        )

        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["distinct_evidence_turn_count"], 5)
        self.assertEqual(diagnostics["minimum_distinct_evidence_turn_count"], 5)
        self.assertEqual(diagnostics["maximum_required_events_on_one_turn"], 2)
        self.assertEqual(diagnostics["required_event_count_by_evidence_turn_index"][8], 2)

    def test_stage64_evidence_dispersion_rejects_three_events_on_one_turn(self) -> None:
        plan, dialogue, alignment = self._dispersion_fixture([0, 2, 4, 8, 8, 8])

        diagnostics = subject.stage64_evidence_dispersion_diagnostics(
            plan,
            dialogue,
            alignment,
            parent_turn_count=2,
        )

        self.assertFalse(diagnostics["pass"])
        self.assertEqual(diagnostics["maximum_required_events_on_one_turn"], 3)
        self.assertIn("evidence_distinct_turn_coverage_too_low", diagnostics["failures"])
        self.assertIn("evidence_turn_required_event_overload", diagnostics["failures"])

    def test_stage64_evidence_layout_plan_layers_six_events(self) -> None:
        checklist = [
            {"fact_id": f"F{index + 1:02d}", "marker": f"MARKER-{index + 1}"}
            for index in range(6)
        ]

        plan = subject._stage64_evidence_layout_plan(checklist)

        self.assertEqual(plan["minimum_distinct_evidence_turns"], 5)
        self.assertEqual(plan["maximum_shared_turn_reduction"], 1)
        self.assertEqual(plan["maximum_required_events_per_turn"], 2)
        self.assertEqual([band["name"] for band in plan["placement_bands"]], ["early", "middle", "later"])
        self.assertEqual([len(band["marker_refs"]) for band in plan["placement_bands"]], [2, 2, 2])

    @staticmethod
    def _formal_segment_plan(event_count: int = 6) -> dict:
        events = []
        event_to_anchor = {}
        anchor_to_event = {}
        for index in range(event_count):
            event_id = f"event-{index}"
            marker = f"MARKER-{index}"
            value = f"route-{index}"
            events.append(
                {
                    "event_instance_id": event_id,
                    "entity": f"Person {index}",
                    "slot": "delivery route",
                    "value": value,
                    "dialogue_value": value,
                    "dialogue_support_claim": f"Person {index} selected {value} as the delivery route.",
                    "status": "active",
                    "required_for_dialogue": True,
                    "alignment_anchor": marker,
                }
            )
            event_to_anchor[event_id] = marker
            anchor_to_event[marker] = event_id
        return {
            "question_type": "single_choice",
            "attack_skill": "conditional_plan",
            "events": events,
            "graph_snapshot_hash": "frozen-graph-hash",
            "current_event_instance_to_anchor": event_to_anchor,
            "current_anchor_to_event_instance_id": anchor_to_event,
        }

    @staticmethod
    def _formal_segment_payload(prompt: dict) -> dict:
        segment_index = prompt["segment"]["segment_index"]
        requested_turns = int(prompt["segment"]["requested_turns"])
        turns = []
        for item_index, item in enumerate(prompt["segment"]["assigned_anchor_realization_checklist"]):
            marker = item["marker"]
            claim = item["natural_support_claim_to_include"]
            turns.append(
                {
                    "speaker": f"Person {item_index}",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                }
            )
        neutral_templates = (
            "Logistics begins with warehouse access and two delivery windows. The coordinators compare loading times before choosing an afternoon handoff.",
            "Budget review separates equipment rental from travel reimbursement. A revised ceiling leaves enough reserve for one unexpected purchase.",
            "Scheduling focuses on the room change after lunch. Everyone records the new arrival order so the transition remains orderly.",
            "Documentation now needs a concise ownership table. Each participant takes responsibility for checking one section before circulation.",
            "Staffing analysis identifies a gap during the evening shift. The group assigns a backup contact who can cover that specific interval.",
            "Transport planning compares the express route with the quieter local road. Weather risk makes the local option more dependable today.",
            "Catering discussion distinguishes dietary requests from ordinary meal counts. A separate vendor will handle the allergy-safe portions.",
            "Security review checks badge collection at both entrances. The west desk receives an updated roster before visitors begin arriving.",
            "Accessibility planning adds captions to the recorded briefing. Printed large-type instructions will also be available near registration.",
            "Outreach work targets neighborhood groups that missed the first notice. A shorter invitation will be sent through their usual contacts.",
            "Procurement compares repairable supplies with disposable alternatives. The durable choice costs more initially but reduces later orders.",
            "Contingency planning reserves a smaller backup room across the hall. That space can open quickly if the main equipment becomes unavailable.",
            "Training starts with a demonstration rather than a long lecture. New participants then practice the procedure with a supervised example.",
            "Reporting separates verified totals from provisional estimates. The final memo will label both categories so readers cannot confuse them.",
            "Handoff preparation lists the decisions that require confirmation tomorrow. Unresolved questions remain with their named owners overnight.",
            "Follow-up planning sets a brief review after the first milestone. The team will compare observed results before changing the next phase.",
        )
        segment_contexts = (
            "This exchange concerns a coastal conference where wind delays the outdoor installation and volunteers must protect borrowed equipment.",
            "The discussion concerns a neighborhood library renovation with limited elevator access and a quiet-hours agreement for nearby residents.",
            "These decisions support a winter clinic whose vaccine deliveries require cold storage and staggered appointments for several age groups.",
            "The participants are preparing a student exhibition where fragile sculptures, lighting tests, and public tours follow different timetables.",
            "This planning round covers a community garden expansion involving soil deliveries, irrigation inspections, and shared tool reservations.",
            "The current scenario is a regional workshop whose interpreters, laboratory stations, and safety briefings must be coordinated separately.",
            "The group is organizing a river cleanup with changing tide conditions, multiple launch points, and strict waste-sorting procedures.",
            "This conversation supports a small theater production where costume fittings, sound rehearsals, and ticketing volunteers have distinct needs.",
            "The team is arranging a mobile archive visit that depends on secure display cases, humidity checks, and timed school-group entry.",
            "This phase concerns an emergency shelter drill with pet intake, medical screening, and overnight supply tracking handled by different crews.",
            "The participants are coordinating a science fair whose electrical inspections, judging routes, and family demonstrations cannot overlap.",
            "The setting is a multilingual town meeting where childcare, remote participation, and written comment collection require separate preparation.",
        )
        while len(turns) < requested_turns:
            neutral_text = neutral_templates[len(turns) % len(neutral_templates)]
            segment_context = segment_contexts[segment_index % len(segment_contexts)]
            turns.append(
                {
                    "speaker": f"Coordinator {len(turns) % 3}",
                    "text": (
                        f"{neutral_text} {segment_context}"
                    ),
                    "alignment_anchors": [],
                    "support_quotes": {},
                }
            )
        return {"appended_dialogue_turns_segment": turns}

    def test_formal_segment_marker_schedule_reserves_middle_and_later_evidence(self) -> None:
        checklist = [
            {"fact_id": f"F{index}", "marker": f"M{index}"}
            for index in range(6)
        ]
        layout = subject._stage64_evidence_layout_plan(checklist)

        early = subject.formal_segment_marker_items(
            layout,
            set(),
            cumulative_tokens=0,
            target_tokens=1000,
            segment_index=0,
            max_segments=6,
        )
        middle = subject.formal_segment_marker_items(
            layout,
            {item["marker"] for item in early},
            cumulative_tokens=250,
            target_tokens=1000,
            segment_index=1,
            max_segments=6,
        )
        later = subject.formal_segment_marker_items(
            layout,
            {item["marker"] for item in early + middle},
            cumulative_tokens=550,
            target_tokens=1000,
            segment_index=2,
            max_segments=6,
        )

        self.assertEqual([item["marker"] for item in early], ["M0", "M1"])
        self.assertEqual([item["marker"] for item in middle], ["M2", "M3"])
        self.assertEqual([item["marker"] for item in later], ["M4", "M5"])

    def test_formal_segment_marker_scheduler_distributes_eight_under_cap(self) -> None:
        checklist = [
            {"fact_id": f"F{index}", "marker": f"M{index}"}
            for index in range(8)
        ]
        layout = subject._stage64_evidence_layout_plan(checklist)
        emitted = set()
        schedules = []

        for segment_index, cumulative_tokens in enumerate((0, 250, 500, 750)):
            schedule = subject.formal_segment_marker_schedule(
                layout,
                emitted,
                cumulative_tokens=cumulative_tokens,
                target_tokens=1000,
                segment_index=segment_index,
                max_segments=4,
            )
            self.assertTrue(schedule["pass"], schedule)
            self.assertTrue(schedule["capacity_proof_pass"], schedule)
            self.assertLessEqual(schedule["assigned_count"], 2)
            schedules.append(schedule)
            emitted.update(schedule["assigned_markers"])

        assignments = [
            marker
            for schedule in schedules
            for marker in schedule["assigned_markers"]
        ]
        self.assertEqual(assignments, [f"M{index}" for index in range(8)])
        self.assertEqual(len(assignments), len(set(assignments)))
        self.assertEqual(
            [schedule["assigned_by_band"] for schedule in schedules],
            [
                {"early": ["M0", "M1"], "middle": [], "later": []},
                {"early": ["M2"], "middle": ["M3"], "later": []},
                {"early": [], "middle": ["M4", "M5"], "later": []},
                {"early": [], "middle": [], "later": ["M6", "M7"]},
            ],
        )
        self.assertNotIn("M2", schedules[-1]["assigned_markers"])
        self.assertEqual(schedules[-1]["pending_after_count"], 0)

        capacity = subject.formal_segment_marker_capacity_diagnostics(
            layout,
            max_segments=4,
        )
        summary = subject.formal_segment_marker_scheduler_summary(
            capacity,
            schedules,
            emitted,
        )
        self.assertTrue(summary["all_markers_assigned_once_under_cap"])
        self.assertEqual(summary["marker_assignment_segment_count"], 4)
        self.assertEqual(summary["max_assigned_markers_per_segment"], 2)
        self.assertEqual(
            summary["band_assignment_segments"],
            {"early": [0, 1], "middle": [1, 2], "later": [3]},
        )

    @staticmethod
    def _run114_comparison_evidence_layout_fixture() -> tuple[dict, list[dict], dict]:
        prefix = "current_method_stage3_v9_segmented_comparison_smoke:S0000:"
        event_specs = (
            ("E01", "Joanna", "screenplay_chapter_count", 5, 1),
            ("E02", "Nate", "screenplay_chapter_count", 3, 1),
            ("E03", "Joanna", "screenplay_genre", "drama", 2),
            ("E04", "Nate", "screenplay_format", "comedy", 3),
            ("E05", "Joanna", "screenplay_last_updated", 12, 4),
            ("E06", "Nate", "screenplay_review_count", 6, 5),
            ("E07", "Joanna", "screenplay_draft_status", 1, 6),
            ("E08", "Nate", "screenplay_submission_count", 0, 7),
        )
        events_by_short_id = {}
        marker_by_event_instance_id = {}
        for short_id, entity, slot, value, time_index in event_specs:
            event_instance_id = f"{prefix}{short_id}"
            marker = f"RUN114-M{short_id[1:]}"
            events_by_short_id[short_id] = {
                "event_id": short_id,
                "event_instance_id": event_instance_id,
                "entity": entity,
                "slot": slot,
                "value": value,
                "status": "active",
                "time_index": time_index,
                "required_for_dialogue": True,
                "marker": marker,
            }
            marker_by_event_instance_id[event_instance_id] = marker

        # Primary events are deliberately not first, proving that placement binds to
        # durable event_instance_id values rather than event or checklist positions.
        shuffled_ids = ("E05", "E03", "E01", "E07", "E04", "E02", "E08", "E06")
        events = [copy.deepcopy(events_by_short_id[event_id]) for event_id in shuffled_ids]
        plan = {
            "operation": "comparison",
            "events": events,
            "graph_snapshot_hash": "run114-comparison-layout",
        }
        checklist = [
            {
                "event_instance_id": event["event_instance_id"],
                "fact_id": f"F{index + 1:02d}",
                "marker": event["marker"],
            }
            for index, event in enumerate(events)
        ]
        return plan, checklist, marker_by_event_instance_id

    def test_run114_comparison_primary_evidence_tracks_difficulty_bands(self) -> None:
        plan, checklist, marker_by_event_id = self._run114_comparison_evidence_layout_fixture()
        primary_ids = [
            "current_method_stage3_v9_segmented_comparison_smoke:S0000:E01",
            "current_method_stage3_v9_segmented_comparison_smoke:S0000:E02",
        ]
        primary_markers = [marker_by_event_id[event_id] for event_id in primary_ids]
        expectations = {
            "easy": ("later", [3, 3]),
            "boundary": ("middle", [1, 1]),
            "harder_boundary": ("early", [0, 0]),
        }

        for difficulty, (expected_band, expected_segments) in expectations.items():
            with self.subTest(difficulty=difficulty):
                layout = subject._stage64_evidence_layout_plan(
                    checklist,
                    current_event_plan=plan,
                    qa_operation="comparison",
                    item_difficulty_target=difficulty,
                )
                placement = layout["difficulty_evidence_placement"]
                self.assertTrue(placement["mapping_pass"])
                self.assertEqual(placement["primary_event_instance_ids"], primary_ids)
                self.assertEqual(placement["primary_markers"], primary_markers)
                self.assertEqual(placement["target_band"], expected_band)

                context_markers = set(marker_by_event_id.values()) - set(primary_markers)
                planned_context_bands = {
                    band["name"]
                    for band in layout["placement_bands"]
                    if context_markers.intersection(
                        item["marker"] for item in band["marker_refs"]
                    )
                }
                self.assertEqual(planned_context_bands, {"early", "middle", "later"})

                capacity = subject.formal_segment_marker_capacity_diagnostics(
                    layout,
                    max_segments=4,
                )
                self.assertTrue(capacity["pass"], capacity)
                emitted = set()
                schedules = []
                for segment_index, cumulative_tokens in enumerate((0, 300, 600, 900)):
                    schedule = subject.formal_segment_marker_schedule(
                        layout,
                        emitted,
                        cumulative_tokens=cumulative_tokens,
                        target_tokens=1000,
                        segment_index=segment_index,
                        max_segments=4,
                    )
                    self.assertTrue(schedule["pass"], schedule)
                    self.assertTrue(schedule["capacity_proof_pass"], schedule)
                    self.assertLessEqual(schedule["assigned_count"], 2)
                    schedules.append(schedule)
                    emitted.update(schedule["assigned_markers"])

                assignments = [
                    marker
                    for schedule in schedules
                    for marker in schedule["assigned_markers"]
                ]
                self.assertEqual(len(assignments), 8)
                self.assertEqual(len(assignments), len(set(assignments)))
                self.assertEqual(set(assignments), set(marker_by_event_id.values()))
                summary = subject.formal_segment_marker_scheduler_summary(
                    capacity,
                    schedules,
                    emitted,
                )
                self.assertTrue(summary["all_markers_assigned_once_under_cap"])
                self.assertTrue(summary["difficulty_evidence_placement_pass"])
                self.assertEqual(
                    summary["requested_item_difficulty_target"], difficulty
                )
                self.assertEqual(summary["primary_markers"], primary_markers)
                self.assertEqual(summary["target_primary_evidence_band"], expected_band)
                self.assertEqual(
                    summary["realized_primary_evidence_bands"],
                    [expected_band, expected_band],
                )
                self.assertEqual(
                    [
                        row["segment_index"]
                        for row in summary["primary_marker_assignment_rows"]
                    ],
                    expected_segments,
                )
                if difficulty == "easy":
                    self.assertNotIn(0, expected_segments)
                    self.assertEqual(expected_segments, [3, 3])

    def test_comparison_evidence_layout_fails_closed_on_missing_primary_mapping(self) -> None:
        plan, checklist, _marker_by_event_id = self._run114_comparison_evidence_layout_fixture()
        checklist = copy.deepcopy(checklist)
        checklist = [
            item
            for item in checklist
            if not item["event_instance_id"].endswith(":E02")
        ]
        layout = subject._stage64_evidence_layout_plan(
            checklist,
            current_event_plan=plan,
            qa_operation="comparison",
            item_difficulty_target="easy",
        )
        capacity = subject.formal_segment_marker_capacity_diagnostics(
            layout,
            max_segments=4,
        )

        self.assertFalse(layout["difficulty_evidence_placement"]["mapping_pass"])
        self.assertFalse(capacity["pass"])
        self.assertEqual(
            capacity["failure_reason"],
            "invalid_primary_evidence_placement_contract",
        )

    def test_formal_segment_marker_capacity_fails_before_api(self) -> None:
        plan = self._formal_segment_plan(event_count=8)
        layout = subject._stage64_evidence_layout_plan(
            [
                {"fact_id": f"F{index}", "marker": f"MARKER-{index}"}
                for index in range(8)
            ]
        )
        capacity = subject.formal_segment_marker_capacity_diagnostics(
            layout,
            max_segments=3,
        )
        self.assertFalse(capacity["pass"])
        self.assertEqual(capacity["minimum_marker_segments_required"], 4)
        self.assertEqual(capacity["failure_reason"], "insufficient_marker_segment_capacity")

        with mock.patch.object(subject, "call_raw_json_api") as api_call:
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=3,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=4000,
            )

        self.assertIsNone(proposal)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["error_type"], "insufficient_formal_marker_segment_capacity")
        self.assertEqual(meta["failure_reason"], "insufficient_marker_segment_capacity")
        self.assertEqual(meta["segment_api_call_count"], 0)
        api_call.assert_not_called()

    def test_formal_segment_fact_plan_omits_unassigned_values_and_support_claims(self) -> None:
        plan = self._formal_segment_plan()
        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(plan)
        dialogue_plan["events"][0]["dependency_facts_to_express"] = [
            "Person 5 selected route-5 as the delivery route."
        ]
        dialogue_plan["events"][0]["scope_fact_to_express"] = {
            "observer": "Person 5"
        }

        segment_plan = subject._formal_segment_fact_plan(dialogue_plan, {"MARKER-0"})

        self.assertEqual(len(segment_plan["assigned_events"]), 1)
        self.assertEqual(segment_plan["assigned_events"][0]["marker"], "MARKER-0")
        self.assertEqual(segment_plan["frozen_unassigned_event_count"], 5)
        serialized = repr(segment_plan)
        self.assertNotIn("route-1", serialized)
        self.assertNotIn("Person 1 selected", serialized)
        self.assertNotIn("dependency_facts_to_express", segment_plan["assigned_events"][0])
        self.assertNotIn("scope_fact_to_express", segment_plan["assigned_events"][0])
        self.assertTrue(
            segment_plan["assigned_events"][0][
                "reasoning_contract_embedded_in_exact_claim"
            ]
        )
        self.assertTrue(segment_plan["unassigned_event_text_omitted"])

    def test_accepted_prefix_audit_does_not_replay_embedded_dependency_fields(self) -> None:
        marker = "CURRENT-MARKER"
        dependency_claim = (
            "Regarding Maria, the membership status is pending and remains under review."
        )
        exact_claim = (
            "Regarding Maria, the destination is Lisbon. " + dependency_claim
        )
        fact_item = {
            "marker": marker,
            "event_instance_id": "event-current",
            "dialogue_support_claim": exact_claim,
            "dependency_facts_to_express": [dependency_claim],
        }
        checklist_item = {
            "marker": marker,
            "event_instance_id": "event-current",
            "natural_support_claim_to_include": exact_claim,
            "dependency_facts_to_express": [dependency_claim],
        }
        sanitized_fact = subject._formal_segment_claim_only_prompt_item(fact_item)
        sanitized_checklist = subject._formal_segment_claim_only_prompt_item(
            checklist_item
        )
        prompt = {
            "frozen_graph": {
                "fact_plan": {
                    "assigned_events": [copy.deepcopy(sanitized_fact)],
                }
            },
            "segment": {
                "assigned_anchor_realization_checklist": [
                    copy.deepcopy(sanitized_checklist)
                ],
                "marker_to_exact_natural_support_claim": {marker: exact_claim},
                "required_marked_turn_shapes": [
                    {
                        "marker": marker,
                        "text_must_start_with": exact_claim,
                        "support_quotes_must_equal": {marker: exact_claim},
                    }
                ],
            },
        }
        provenance = [
            {
                "marker": marker,
                "event_instance_id": "event-current",
                "validated_claim": exact_claim,
                "claim_source": "canonical_entity_slot_value_status",
            }
        ]
        exemptions = subject.formal_current_exact_claim_audit_exempt_spans(
            prompt, provenance, {marker}
        )
        audit = subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
            prompt,
            [{"speaker": "John", "text": dependency_claim}],
            local_exempt_spans=exemptions,
        )

        self.assertNotIn("dependency_facts_to_express", repr(prompt))
        self.assertTrue(audit["pass"], audit)

    def test_dialogue_plan_and_canonical_claim_preserve_reasoning_contract_surfaces(
        self,
    ) -> None:
        plan = {
            "operation": "comparison",
            "events": [
                {
                    "event_instance_id": "E1",
                    "dialogue_fact_id": "F1",
                    "alignment_anchor": "MARKER1",
                    "entity": "Atlas team",
                    "slot": "resource status",
                    "value": "approved",
                    "dialogue_support_claim": "The Atlas resource was approved.",
                    "status": "active",
                    "time_index": 1,
                },
                {
                    "event_instance_id": "E2",
                    "dialogue_fact_id": "F2",
                    "alignment_anchor": "MARKER2",
                    "entity": "Atlas team",
                    "slot": "review candidate",
                    "value": 7,
                    "dialogue_support_claim": "A vague numeric note.",
                    "status": "active",
                    "time_index": 2,
                    "depends_on": ["E1", "unknown-parent-id"],
                    "scope": {
                        "observer": "Morgan",
                        "permission": "review access",
                        "condition": "Morgan is on duty",
                    },
                    "set_membership": {
                        "set_id": "atlas-review",
                        "condition": "the review is complete",
                        "included": False,
                    },
                },
            ],
            "current_event_instance_to_anchor": {
                "E1": "MARKER1",
                "E2": "MARKER2",
            },
        }

        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(plan)
        prompt_event = dialogue_plan["events"][1]
        self.assertEqual(prompt_event["depends_on"], ["E1"])
        self.assertEqual(prompt_event["scope"], plan["events"][1]["scope"])
        self.assertEqual(
            prompt_event["set_membership"]["condition"],
            "the review is complete",
        )
        self.assertIn("Morgan", prompt_event["scope_fact_to_express"])
        self.assertIn("Review access", prompt_event["scope_fact_to_express"])
        self.assertIn(
            "does not qualify",
            prompt_event["membership_fact_to_express"],
        )
        self.assertIn(
            "follows from",
            prompt_event["dependency_facts_to_express"][0],
        )
        self.assertNotIn("unknown-parent-id", repr(prompt_event))

        validated, diagnostics = subject.formal_segment_exact_claim_preflight(
            plan,
            dialogue_plan,
        )
        self.assertIsNotNone(validated)
        provenance = diagnostics["claim_provenance"][1]
        self.assertEqual(
            provenance["claim_source"],
            "canonical_entity_slot_value_status",
        )
        canonical = provenance["validated_claim"]
        canonical_casefold = canonical.casefold()
        for surface in (
            "condition that the review is complete",
            "for morgan",
            "review access",
            "morgan is on duty",
            "follows from atlas team's resource status being approved",
        ):
            self.assertIn(surface, canonical_casefold)
        self.assertNotIn("E1", canonical)
        self.assertNotIn("set_membership", canonical)
        self.assertNotIn("depends_on", canonical)

    def test_formal_exact_claim_preflight_repairs_run101_lexical_failures(self) -> None:
        plan = self._formal_segment_plan()
        plan["events"][0].update(
            {
                "entity": "hike_plans",
                "slot": "participant",
                "value": "Screenplay Writing",
                "dialogue_value": "Screenplay Writing",
                "dialogue_support_claim": (
                    "Nate mentioned thinking of a screenplay, but it is unrelated to "
                    "the hike participants."
                ),
                "set_membership": {"set_id": "hike_participants", "included": False},
            }
        )
        plan["events"][1].update(
            {
                "entity": "hike_plans",
                "slot": "participant",
                "value": "Trails",
                "dialogue_value": "Trails",
                "dialogue_support_claim": (
                    "Joanna mentioned the trails being special and writing a drama, but "
                    "the trails themselves are not participants."
                ),
                "set_membership": {"set_id": "hike_participants", "included": True},
            }
        )
        self.assertFalse(
            subject._event_lexical_supported(
                plan["events"][0], plan["events"][0]["dialogue_support_claim"]
            )
        )
        self.assertFalse(
            subject._event_lexical_supported(
                plan["events"][1], plan["events"][1]["dialogue_support_claim"]
            )
        )
        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(plan)

        validated_plan, diagnostics = subject.formal_segment_exact_claim_preflight(
            plan,
            dialogue_plan,
        )

        self.assertIsNotNone(validated_plan)
        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["required_event_count"], 6)
        self.assertEqual(diagnostics["original_claim_accept_count"], 4)
        self.assertEqual(diagnostics["canonical_claim_rewrite_count"], 2)
        changed = [
            item for item in diagnostics["claim_provenance"] if item["claim_changed"]
        ]
        self.assertEqual(
            {item["claim_source"] for item in changed},
            {"canonical_entity_slot_value_status"},
        )
        self.assertEqual(
            {item["validated_claim"] for item in changed},
            {
                "Regarding hike plans, Screenplay Writing is not included in the participant list",
                "Regarding hike plans, Trails is included in the participant list",
            },
        )
        source_by_id = {
            item["event_instance_id"]: item for item in plan["events"]
        }
        for item in changed:
            self.assertTrue(item["canonical_meta_clean"])
            self.assertTrue(item["canonical_lexical_supported"])
            self.assertTrue(
                subject._event_lexical_supported(
                    source_by_id[item["event_instance_id"]],
                    item["validated_claim"],
                )
            )
            meta_clean, _ = subject.assert_no_meta_leakage(
                [{"text": item["validated_claim"]}]
            )
            self.assertTrue(meta_clean)
        self.assertEqual(
            dialogue_plan["events"][0]["dialogue_support_claim"],
            "Nate mentioned thinking of a screenplay, but it is unrelated to the hike participants",
        )

    def test_event_lexical_support_keeps_short_numeric_facts_only_in_event_gate(self) -> None:
        ingredient_event = {
            "entity": "Joanna's Chocolate Cake",
            "slot": "ingredient_count",
            "value": 4,
            "dialogue_value": "4",
            "dialogue_support_claim": "Joanna's cake uses four main ingredients",
            "status": "active",
        }
        duration_event = {
            "entity": "John",
            "slot": "team_membership_duration",
            "value": 3,
            "dialogue_value": "3",
            "dialogue_support_claim": (
                "John has been with the Minnesota Wolves for 3 seasons"
            ),
            "status": "active",
        }

        self.assertNotIn("4", subject.token_set("Joanna used 4 ingredients"))
        self.assertIn(
            "4",
            subject._event_lexical_token_set("Joanna used 4 ingredients"),
        )
        self.assertFalse(
            subject._event_lexical_supported(
                ingredient_event,
                ingredient_event["dialogue_support_claim"],
            )
        )
        self.assertTrue(
            subject._event_lexical_supported(
                ingredient_event,
                "Regarding Joanna's Chocolate Cake, the ingredient count is 4",
            )
        )
        self.assertFalse(
            subject._event_lexical_supported(
                ingredient_event,
                "Regarding Joanna's Chocolate Cake, the ingredient count is 5",
            )
        )
        for unsupported_surface in (
            "Regarding Joanna's Chocolate Cake, the ingredient count is missing",
            "Regarding Joanna's Chocolate Cake, the ingredient count is 14",
            "Regarding Joanna's Chocolate Cake, the ingredient count is 4.5",
            "Regarding Joanna's Chocolate Cake, the ingredient count is 4,000",
            "Regarding Joanna's Chocolate Cake, the ingredient count is E4",
            "Regarding Joanna's Chocolate Cake, the ingredient count is +4",
            "Regarding Joanna's Chocolate Cake, the ingredient count is -4",
        ):
            self.assertFalse(
                subject._event_lexical_supported(
                    ingredient_event,
                    unsupported_surface,
                ),
                unsupported_surface,
            )
        self.assertFalse(
            subject._event_lexical_supported(
                ingredient_event,
                "Regarding Joanna's Chocolate Cake, the unrelated score is 4",
            )
        )
        self.assertTrue(
            subject._event_lexical_supported(
                duration_event,
                duration_event["dialogue_support_claim"],
            )
        )
        self.assertTrue(
            subject._event_lexical_supported(
                duration_event,
                "Regarding John, the team membership duration is 3",
            )
        )
        self.assertTrue(
            subject._event_lexical_supported(
                duration_event,
                "Regarding John, the team membership duration is 3, as confirmed",
            )
        )
        for unsupported_surface in (
            "Regarding John, the team membership duration is 13",
            "Regarding John, the team membership duration is 3.5",
            "Regarding John, the team membership duration is 3,000",
            "Regarding John, the team membership duration is E3",
            "Regarding John, the team membership duration is -3",
        ):
            self.assertFalse(
                subject._event_lexical_supported(duration_event, unsupported_surface),
                unsupported_surface,
            )

        for integer_value, supported, unsupported in (
            (0, "Maya's lap count is 0.", "Maya's lap count is 10."),
            (-3, "Maya's score change is -3.", "Maya's score change is 3."),
        ):
            boundary_event = {
                "entity": "Maya",
                "slot": "numeric_result",
                "value": integer_value,
                "dialogue_value": str(integer_value),
                "dialogue_support_claim": supported,
                "status": "active",
            }
            self.assertTrue(
                subject._event_lexical_supported(boundary_event, supported),
                supported,
            )
            self.assertFalse(
                subject._event_lexical_supported(boundary_event, unsupported),
                unsupported,
            )

        nonnumeric_event = {
            "entity": "Maya",
            "slot": "delivery_route",
            "value": "Orchid Road",
            "dialogue_value": "Orchid Road",
            "dialogue_support_claim": "Maya selected Orchid Road",
            "status": "active",
        }
        self.assertTrue(
            subject._event_lexical_supported(
                nonnumeric_event,
                "Maya selected Orchid Road",
            )
        )

    def test_formal_exact_claim_preflight_repairs_run108_numeric_claims(self) -> None:
        plan = self._formal_segment_plan(event_count=2)
        plan["operation"] = "comparison"
        plan["events"][0].update(
            {
                "entity": "Joanna's Chocolate Cake",
                "slot": "ingredient_count",
                "value": 4,
                "dialogue_value": "4",
                "dialogue_support_claim": "Joanna's cake uses four main ingredients",
            }
        )
        plan["events"][1].update(
            {
                "entity": "John",
                "slot": "team_membership_duration",
                "value": 3,
                "dialogue_value": "3",
                "dialogue_support_claim": (
                    "John has been with the Minnesota Wolves for 3 seasons"
                ),
            }
        )
        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(plan)

        validated_plan, diagnostics = subject.formal_segment_exact_claim_preflight(
            plan,
            dialogue_plan,
        )

        self.assertIsNotNone(validated_plan)
        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["required_event_count"], 2)
        self.assertEqual(diagnostics["original_claim_accept_count"], 0)
        self.assertEqual(diagnostics["canonical_claim_rewrite_count"], 2)
        provenance = diagnostics["claim_provenance"]
        self.assertEqual(
            [item["claim_source"] for item in provenance],
            [
                "canonical_entity_slot_value_status",
                "canonical_entity_slot_value_status",
            ],
        )
        self.assertEqual(
            [item["validated_claim"] for item in provenance],
            [
                "Regarding Joanna's Chocolate Cake, the ingredient count is 4",
                "Regarding John, the team membership duration is 3",
            ],
        )
        self.assertTrue(provenance[0]["claim_changed"])
        self.assertTrue(provenance[0]["canonical_lexical_supported"])
        self.assertTrue(provenance[1]["claim_changed"])
        self.assertTrue(provenance[1]["original_lexical_supported"])
        self.assertTrue(provenance[1]["canonical_lexical_supported"])

    def test_noncomparison_numeric_preflight_preserves_supported_original_claim(self) -> None:
        plan = self._formal_segment_plan(event_count=1)
        plan["operation"] = "count"
        original_claim = "John has been with the Minnesota Wolves for 3 seasons"
        plan["events"][0].update(
            {
                "entity": "John",
                "slot": "team_membership_duration",
                "value": 3,
                "dialogue_value": "3",
                "dialogue_support_claim": original_claim,
            }
        )

        validated_plan, diagnostics = subject.formal_segment_exact_claim_preflight(
            plan,
            subject.sanitize_event_plan_for_dialogue_api(plan),
        )

        self.assertIsNotNone(validated_plan)
        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["original_claim_accept_count"], 1)
        self.assertEqual(diagnostics["canonical_claim_rewrite_count"], 0)
        self.assertEqual(
            diagnostics["claim_provenance"][0]["validated_claim"],
            original_claim,
        )
        self.assertEqual(
            diagnostics["claim_provenance"][0]["claim_source"],
            "original_dialogue_support_claim",
        )

    def test_formal_segment_prompt_uses_preflight_validated_claim_map(self) -> None:
        plan = self._formal_segment_plan()
        original_claim = (
            "Nate mentioned thinking of a screenplay, but it is unrelated to the hike participants."
        )
        plan["events"][0].update(
            {
                "entity": "hike_plans",
                "slot": "participant",
                "value": "Screenplay Writing",
                "dialogue_value": "Screenplay Writing",
                "dialogue_support_claim": original_claim,
                "set_membership": {"set_id": "hike_participants", "included": False},
            }
        )
        prompts = []
        system_prompts = []

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            system_prompts.append(_kwargs.get("system_prompt"))
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertEqual(meta["exact_claim_preflight"]["canonical_claim_rewrite_count"], 1)
        marker = plan["events"][0]["alignment_anchor"]
        validated_claim = prompts[0]["segment"]["marker_to_exact_natural_support_claim"][marker]
        self.assertNotEqual(validated_claim, original_claim)
        self.assertEqual(
            validated_claim,
            "Regarding hike plans, Screenplay Writing is not included in the participant list",
        )
        checklist_item = next(
            item
            for item in prompts[0]["segment"]["assigned_anchor_realization_checklist"]
            if item["marker"] == marker
        )
        self.assertEqual(checklist_item["natural_support_claim_to_include"], validated_claim)
        assigned_event = next(
            event
            for event in prompts[0]["frozen_graph"]["fact_plan"]["assigned_events"]
            if event["marker"] == marker
        )
        self.assertEqual(assigned_event["dialogue_support_claim"], validated_claim)
        requirements = " ".join(prompts[0]["requirements"])
        self.assertIn("entire text equal", requirements)
        self.assertIn("cannot exist only in support_quotes", requirements)
        self.assertIn("same digit characters", requirements)
        self.assertIn("named subject must not be the turn's speaker", requirements)
        self.assertIn("character-for-character", system_prompts[0])
        self.assertIn("add no prefix, suffix", system_prompts[0])
        self.assertIn("never spell it as a number word", system_prompts[0])
        self.assertIn("named subject cannot be speaker", system_prompts[0])

    def test_formal_exact_claim_preflight_fails_closed_before_api(self) -> None:
        plan = self._formal_segment_plan()
        plan["events"][0].update(
            {
                "entity": "X",
                "slot": "id",
                "value": "1",
                "dialogue_value": "1",
                "dialogue_support_claim": "X id 1",
            }
        )

        with mock.patch.object(subject, "call_raw_json_api") as api_call:
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(challenger_max_tokens=8192, max_llm_retries=0),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNone(proposal)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["error_type"], "formal_segment_exact_claim_preflight_failed")
        self.assertEqual(meta["segment_api_call_count"], 0)
        self.assertEqual(meta["attempts"], 0)
        self.assertFalse(meta["exact_claim_preflight"]["pass"])
        self.assertEqual(meta["exact_claim_preflight"]["failure_count"], 1)
        self.assertEqual(
            meta["exact_claim_preflight"]["failures"][0]["reason"],
            "no_meta_clean_lexically_supported_claim",
        )
        api_call.assert_not_called()

    def test_formal_segment_validation_rejects_early_qa_and_prefix_rewrite(self) -> None:
        plan = self._formal_segment_plan()
        marker = "MARKER-0"
        claim = plan["events"][0]["dialogue_support_claim"]
        base_payload = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Person 0",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                }
            ]
        }

        with_qa = copy.deepcopy(base_payload)
        with_qa["qa_proposal"] = {"question": "too early"}
        qa_result = subject.validate_formal_dialogue_segment(
            with_qa,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
        )
        repeated_result = subject.validate_formal_dialogue_segment(
            base_payload,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[{"speaker": "Person 0", "text": claim}],
        )

        self.assertEqual(qa_result["reason"], "qa_returned_before_final_alignment")
        self.assertEqual(repeated_result["reason"], "frozen_prefix_turn_repeated")

    def test_formal_segment_machine_enforces_shape_count_and_marker_cardinality(self) -> None:
        plan = self._formal_segment_plan(event_count=1)
        marker = "MARKER-0"
        claim = plan["events"][0]["dialogue_support_claim"]
        valid_payload = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Coordinator",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                },
                {
                    "speaker": "Reviewer",
                    "text": "Budget review identifies one separate follow-up action.",
                    "alignment_anchors": [],
                    "support_quotes": {},
                },
            ]
        }

        def validate(payload: dict, *, expected_turn_count: int = 2) -> dict:
            return subject.validate_formal_dialogue_segment(
                payload,
                expected_turn_count=expected_turn_count,
                assigned_markers={marker},
                anchor_to_event=plan["current_anchor_to_event_instance_id"],
                current_event_plan=plan,
                frozen_prefix_turns=[],
                marker_to_exact_claim={marker: claim},
            )

        self.assertTrue(validate(valid_payload)["pass"])

        wrong_count = validate(valid_payload, expected_turn_count=3)
        self.assertEqual(wrong_count["reason"], "segment_turn_count_mismatch")
        self.assertEqual(wrong_count["actual_turn_count"], 2)

        extra_top_level = copy.deepcopy(valid_payload)
        extra_top_level["event_alignment_proposal_segment"] = []
        self.assertEqual(
            validate(extra_top_level)["reason"], "segment_top_level_schema_mismatch"
        )

        repeated_marker = copy.deepcopy(valid_payload)
        repeated_marker["appended_dialogue_turns_segment"][1].update(
            {
                "text": claim,
                "alignment_anchors": [marker],
                "support_quotes": {marker: claim},
            }
        )
        repeated_result = validate(repeated_marker)
        self.assertEqual(
            repeated_result["reason"], "assigned_marker_occurrence_count_mismatch"
        )
        self.assertEqual(repeated_result["marker_occurrence_counts"], {marker: 2})

        dirty_neutral = copy.deepcopy(valid_payload)
        dirty_neutral["appended_dialogue_turns_segment"][1]["support_quotes"] = {
            marker: claim
        }
        self.assertEqual(
            validate(dirty_neutral)["reason"], "neutral_turn_has_support_sidecar"
        )

        marked_tail = copy.deepcopy(valid_payload)
        marked_tail["appended_dialogue_turns_segment"][0]["text"] = (
            f"{claim} An extra explanation follows."
        )
        marked_tail_result = validate(marked_tail)
        self.assertEqual(
            marked_tail_result["reason"], "marked_turn_text_not_exact_claim"
        )

        leaked_neutral = copy.deepcopy(valid_payload)
        leaked_neutral["appended_dialogue_turns_segment"][1]["text"] = (
            "Person 0 later repeated that route-0 was the selected option."
        )
        leaked_neutral_result = validate(leaked_neutral)
        self.assertEqual(
            leaked_neutral_result["reason"],
            "neutral_graph_fact_surface_exposure",
        )

        leaked_status = copy.deepcopy(valid_payload)
        leaked_status["appended_dialogue_turns_segment"][1]["text"] = (
            "Person 0 says the delivery route remains active."
        )
        leaked_status_result = validate(leaked_status)
        self.assertEqual(
            leaked_status_result["reason"],
            "neutral_graph_fact_surface_exposure",
        )
        self.assertEqual(
            leaked_status_result["graph_fact_isolation_audit"]["violations"][0][
                "matched_surface"
            ],
            "entity+slot+status",
        )

    def test_run127_missing_neutral_sidecars_are_completed_absent_keys_only(self) -> None:
        plan = self._formal_segment_plan(event_count=1)
        marker = "MARKER-0"
        claim = plan["events"][0]["dialogue_support_claim"]
        payload = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Coordinator",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                },
                {
                    "speaker": "Reviewer",
                    "text": "A separate budget review will continue tomorrow morning.",
                },
            ]
        }
        original = copy.deepcopy(payload)

        result = subject.validate_formal_dialogue_segment(
            payload,
            expected_turn_count=2,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: claim},
        )

        self.assertTrue(result["pass"])
        self.assertEqual(payload, original)
        self.assertEqual(result["marked_turns"][1]["alignment_anchors"], [])
        self.assertEqual(result["marked_turns"][1]["support_quotes"], {})
        diagnostics = result["neutral_sidecar_normalization"]
        self.assertEqual(
            diagnostics["schema_version"],
            subject.FORMAL_SEGMENT_NEUTRAL_SIDECAR_NORMALIZER_VERSION,
        )
        self.assertTrue(diagnostics["used"])
        self.assertEqual(diagnostics["repair_count"], 1)
        self.assertEqual(
            diagnostics["repairs"][0]["added_keys"],
            ["alignment_anchors", "support_quotes"],
        )
        self.assertTrue(diagnostics["absent_keys_only"])
        self.assertFalse(diagnostics["alignment_evidence_inferred"])
        self.assertEqual(
            diagnostics["speaker_text_sha256_before"],
            diagnostics["speaker_text_sha256_after"],
        )

    def test_run127_neutral_sidecar_completion_keeps_ambiguous_evidence_fail_closed(self) -> None:
        plan = self._formal_segment_plan(event_count=1)
        marker = "MARKER-0"
        claim = plan["events"][0]["dialogue_support_claim"]

        missing_quote = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Coordinator",
                    "text": claim,
                    "alignment_anchors": [marker],
                }
            ]
        }
        missing_quote_result = subject.validate_formal_dialogue_segment(
            missing_quote,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: claim},
        )
        self.assertFalse(missing_quote_result["pass"])
        self.assertEqual(
            missing_quote_result["reason"], "segment_sidecar_type_mismatch"
        )
        self.assertEqual(
            missing_quote_result["neutral_sidecar_normalization"]["skipped"][0][
                "reason"
            ],
            "present_sidecar_not_exact_empty_container",
        )

        explicit_null = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Coordinator",
                    "text": claim,
                    "alignment_anchors": None,
                }
            ]
        }
        explicit_null_result = subject.validate_formal_dialogue_segment(
            explicit_null,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: claim},
        )
        self.assertFalse(explicit_null_result["pass"])
        self.assertEqual(
            explicit_null_result["reason"], "segment_sidecar_type_mismatch"
        )

        unmarked_claim = {
            "appended_dialogue_turns_segment": [
                {"speaker": "Coordinator", "text": claim}
            ]
        }
        unmarked_claim_result = subject.validate_formal_dialogue_segment(
            unmarked_claim,
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: claim},
        )
        self.assertFalse(unmarked_claim_result["pass"])
        self.assertEqual(
            unmarked_claim_result["reason"],
            "assigned_marker_occurrence_count_mismatch",
        )

    def test_run112_segment_internal_exact_duplicate_fixture_is_rejected(self) -> None:
        repeated = (
            "I admire your dedication, John. It's clear that you're committed to making a "
            "positive change. I hope your efforts lead to great outcomes. I think working "
            "together on community projects would be a great way to support each other and "
            "make an impact. I'm confident we can achieve a lot if we team up again."
        )
        payload = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Maria",
                    "text": repeated,
                    "alignment_anchors": [],
                    "support_quotes": {},
                },
                {
                    "speaker": "John",
                    "text": repeated,
                    "alignment_anchors": [],
                    "support_quotes": {},
                },
            ]
        }

        result = subject.validate_formal_dialogue_segment(
            payload,
            assigned_markers=set(),
            anchor_to_event={},
            current_event_plan={"events": []},
            frozen_prefix_turns=[],
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "segment_exact_duplicate_turn")
        self.assertEqual(result["turn_indices"], [1])
        audit = result["turn_duplicate_audit"]
        self.assertEqual(
            audit["schema_version"],
            subject.FORMAL_SEGMENT_TURN_DUPLICATE_AUDIT_VERSION,
        )
        self.assertEqual(audit["exact_duplicate_pair_count"], 1)
        self.assertEqual(audit["exact_duplicate_pairs"][0]["left_index"], 1)
        self.assertEqual(audit["exact_duplicate_pairs"][0]["right_index"], 0)

    def test_run112_high_similarity_fixture_is_rejected_but_distinct_long_turns_pass(self) -> None:
        first = (
            "I think that's a great perspective. Maria's volunteering hours per week are 6, "
            "and I'm glad she's finding strength in her relationships and her work. It's "
            "important to have a balance between personal and community life. I hope to find "
            "that balance as well, and I think working together can help us both achieve that."
        )
        second = (
            "I think we both have different paths, but the same goal. Maria's volunteering "
            "hours per week are 6, and I'm glad she's finding strength in her relationships "
            "and her work. It's important to have a balance between personal and community "
            "life. I hope to find that balance as well, and I think working together can help "
            "us both achieve that."
        )
        near = subject.audit_formal_dialogue_segment_turn_duplicates(
            [
                {"speaker": "Maria", "text": first, "alignment_anchors": []},
                {"speaker": "John", "text": second, "alignment_anchors": []},
            ],
            [],
        )
        distinct = subject.audit_formal_dialogue_segment_turn_duplicates(
            [
                {
                    "speaker": "Maria",
                    "text": (
                        "The neighborhood clinic scheduled a Saturday workshop on emergency "
                        "preparedness. Nurses will demonstrate first-aid supplies, then each "
                        "family will practice a written contact plan before taking materials home."
                    ),
                    "alignment_anchors": [],
                },
                {
                    "speaker": "John",
                    "text": (
                        "Our school garden team mapped the shaded beds behind the science wing. "
                        "Students chose native herbs for those plots, measured the soil, and "
                        "assigned a rotating watering schedule for the rest of the semester."
                    ),
                    "alignment_anchors": [],
                },
            ],
            [],
        )

        self.assertFalse(near["pass"])
        self.assertEqual(near["near_duplicate_pair_count"], 1)
        pair = near["near_duplicate_pairs"][0]
        self.assertGreaterEqual(
            pair["sequence_similarity"],
            subject.FORMAL_SEGMENT_NEAR_DUPLICATE_SEQUENCE_THRESHOLD,
        )
        self.assertGreaterEqual(
            pair["token_jaccard"],
            subject.FORMAL_SEGMENT_NEAR_DUPLICATE_JACCARD_THRESHOLD,
        )
        self.assertTrue(distinct["pass"])

    def test_phrase_containment_catches_copied_turn_expansion(self) -> None:
        frozen = (
            "I have been thinking about how our neighborhood program can support families this "
            "autumn. The planning committee agreed to review volunteer schedules before Friday. "
            "We should keep the discussion focused on practical steps for the next meeting."
        )
        expanded = frozen + (
            " I also drafted a longer sign-up form and prepared detailed notes for everyone who "
            "could not attend."
        )

        audit = subject.audit_formal_dialogue_segment_turn_duplicates(
            [{"speaker": "A", "text": expanded, "alignment_anchors": []}],
            [{"speaker": "B", "text": frozen}],
        )

        self.assertFalse(audit["pass"])
        self.assertEqual(audit["phrase_containment_pair_count"], 1)
        pair = audit["phrase_containment_pairs"][0]
        self.assertEqual(pair["right_scope"], "frozen_prefix")
        self.assertGreaterEqual(
            pair["shared_word_ngram_count"],
            subject.FORMAL_SEGMENT_PHRASE_MIN_SHARED_NGRAMS,
        )
        self.assertGreaterEqual(
            pair["word_ngram_containment"],
            subject.FORMAL_SEGMENT_PHRASE_CONTAINMENT_THRESHOLD,
        )

    def test_validated_claim_prefix_is_stripped_without_exempting_copied_body(self) -> None:
        claim_a = "Regarding Nora, the confirmed weekly total is 6."
        claim_b = "Regarding Luis, the confirmed weekly total is 7."
        copied_body = (
            "The committee reviewed the same planning notes during the afternoon meeting and "
            "agreed that every volunteer should receive a written schedule before Friday. "
            "They also assigned two coordinators to answer follow-up questions from families."
        )
        marker_claims = {"MARKER-A": claim_a, "MARKER-B": claim_b}
        copied = subject.audit_formal_dialogue_segment_turn_duplicates(
            [
                {
                    "speaker": "Maya",
                    "text": f"{claim_a} {copied_body}",
                    "alignment_anchors": ["MARKER-A"],
                },
                {
                    "speaker": "Noah",
                    "text": f"{claim_b} {copied_body}",
                    "alignment_anchors": ["MARKER-B"],
                },
            ],
            [],
            marker_claims,
        )
        claim_only = subject.audit_formal_dialogue_segment_turn_duplicates(
            [
                {
                    "speaker": "Maya",
                    "text": claim_a,
                    "alignment_anchors": ["MARKER-A"],
                },
                {
                    "speaker": "Noah",
                    "text": claim_b,
                    "alignment_anchors": ["MARKER-B"],
                },
            ],
            [],
            marker_claims,
        )

        self.assertFalse(copied["pass"])
        self.assertTrue(
            copied["phrase_containment_pair_count"]
            or copied["near_duplicate_pair_count"]
        )
        violation = (
            copied["phrase_containment_pairs"] or copied["near_duplicate_pairs"]
        )[0]
        self.assertTrue(violation["left_canonical_support_prefix_stripped"])
        self.assertTrue(violation["right_canonical_support_prefix_stripped"])
        self.assertTrue(claim_only["pass"])

    def test_run102_paraphrased_marked_turn_is_rejected_without_dialogue_mutation(self) -> None:
        marker = "TZ4NAS8LEZD"
        event_id = "event-new-york"
        exact_claim = "John is considering New York City as a destination for the team trip"
        turn_text = (
            "I've been looking into New York City as a possible destination for the team trip. "
            "It's a vibrant city with so much to offer, from iconic landmarks to world-class "
            "dining and entertainment. I think it would be a great place for the team to "
            "experience something new and exciting."
        )
        plan = {
            "events": [
                {
                    "event_instance_id": event_id,
                    "entity": "team_trip",
                    "slot": "destination",
                    "value": "New York City",
                    "dialogue_value": "New York City",
                    "dialogue_support_claim": exact_claim,
                    "status": "active",
                    "required_for_dialogue": True,
                }
            ]
        }
        proposal = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "John",
                    "text": turn_text,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: exact_claim},
                }
            ]
        }
        original = copy.deepcopy(proposal)

        result = subject.validate_formal_dialogue_segment(
            proposal,
            assigned_markers={marker},
            anchor_to_event={marker: event_id},
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: exact_claim},
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "marked_turn_text_not_exact_claim")
        self.assertEqual(proposal, original)
        audit = result["graph_fact_isolation_audit"]
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["violation_count"], 1)

    def test_run108_sidecar_only_number_word_without_expired_status_is_rejected(self) -> None:
        marker = "MARKER-CAROLINE-COUNT"
        event_id = "event-caroline-count-expired"
        exact_claim = (
            "Regarding Caroline, the support group attendance was 5, but that information has expired"
        )
        plan = {
            "events": [
                {
                    "event_instance_id": event_id,
                    "entity": "Caroline",
                    "slot": "support_group_attendance",
                    "value": 5,
                    "dialogue_value": 5,
                    "dialogue_support_claim": exact_claim,
                    "status": "expired",
                    "required_for_dialogue": True,
                }
            ]
        }
        proposal = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Caroline",
                    "text": (
                        "I went to the support group five times, and each meeting helped me "
                        "feel less alone during a difficult period."
                    ),
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: exact_claim},
                }
            ]
        }

        result = subject.validate_formal_dialogue_segment(
            proposal,
            assigned_markers={marker},
            anchor_to_event={marker: event_id},
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: exact_claim},
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "unsupported_segment_alignment")
        self.assertFalse(result["sidecar_quote_normalization"]["used"])
        self.assertEqual(
            result["sidecar_quote_normalization"]["skipped"][0]["reason"],
            "speaker_aware_event_semantics_mismatch",
        )

    def test_first_person_sidecar_quote_rejects_wrong_speaker(self) -> None:
        marker = "MARKER-JOANNA"
        event_id = "event-joanna-hike"
        exact_claim = "Joanna is planning a hike with friends this weekend"
        proposal = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "Tim",
                    "text": (
                        "I've been planning a hike with friends this weekend, and I think the "
                        "trail will give all of us a welcome change of scenery."
                    ),
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: exact_claim},
                }
            ]
        }
        plan = {
            "events": [
                {
                    "event_instance_id": event_id,
                    "entity": "hike_plans",
                    "slot": "participants",
                    "value": "Joanna",
                    "dialogue_value": "Joanna",
                    "dialogue_support_claim": exact_claim,
                    "status": "active",
                    "required_for_dialogue": True,
                }
            ]
        }

        result = subject.validate_formal_dialogue_segment(
            proposal,
            assigned_markers={marker},
            anchor_to_event={marker: event_id},
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: exact_claim},
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "unsupported_segment_alignment")
        diagnostics = result["sidecar_quote_normalization"]
        self.assertFalse(diagnostics["used"])
        self.assertEqual(
            diagnostics["skipped"][0]["reason"],
            "speaker_aware_event_semantics_mismatch",
        )

    def test_first_person_sidecar_quote_rejects_wrong_event_content(self) -> None:
        marker = "MARKER-NYC"
        event_id = "event-new-york"
        exact_claim = "John is considering New York City as a destination for the team trip"
        proposal = {
            "appended_dialogue_turns_segment": [
                {
                    "speaker": "John",
                    "text": (
                        "I've confirmed the Paris catering budget, and the revised invoice "
                        "will be ready for the finance meeting tomorrow."
                    ),
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: exact_claim},
                }
            ]
        }
        plan = {
            "events": [
                {
                    "event_instance_id": event_id,
                    "entity": "team_trip",
                    "slot": "destination",
                    "value": "New York City",
                    "dialogue_value": "New York City",
                    "dialogue_support_claim": exact_claim,
                    "status": "active",
                    "required_for_dialogue": True,
                }
            ]
        }

        result = subject.validate_formal_dialogue_segment(
            proposal,
            assigned_markers={marker},
            anchor_to_event={marker: event_id},
            current_event_plan=plan,
            frozen_prefix_turns=[],
            marker_to_exact_claim={marker: exact_claim},
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "unsupported_segment_alignment")
        diagnostics = result["sidecar_quote_normalization"]
        self.assertFalse(diagnostics["used"])
        self.assertEqual(
            diagnostics["skipped"][0]["reason"],
            "speaker_aware_event_semantics_mismatch",
        )

    def test_formal_global_alignment_requires_exact_quote_and_event_semantics(self) -> None:
        marker = "MARKER-ROUTE"
        event_id = "event-route"
        event = {
            "event_instance_id": event_id,
            "entity": "Person 0",
            "slot": "delivery route",
            "value": "route-0",
            "dialogue_value": "route-0",
            "dialogue_support_claim": "Person 0 selected route-0 as the delivery route.",
            "status": "active",
            "required_for_dialogue": True,
        }
        plan = {"events": [event], "graph_snapshot_hash": "graph-hash"}
        semantic_turn = {
            "turn_id": "D2:1",
            "speaker": "Person 0",
            "text": "Person 0 selected route-0 as the delivery route for tomorrow.",
        }
        dialogue = {"turns": [semantic_turn], "dialogue_hash": "dialogue-hash"}
        fuzzy_proposal = {
            "proposals": [
                {
                    "alignment_anchor": marker,
                    "turn_index": 0,
                    "support_quote": "The delivery route route-0 was selected by Person 0.",
                }
            ]
        }

        relaxed = subject.verify_current_graph_dialogue_alignment(
            plan,
            dialogue,
            fuzzy_proposal,
            {marker: event_id},
        )
        strict = subject.verify_current_graph_dialogue_alignment(
            plan,
            dialogue,
            fuzzy_proposal,
            {marker: event_id},
            require_exact_support_quote=True,
        )

        self.assertTrue(relaxed["coverage_pass"])
        self.assertFalse(strict["coverage_pass"])
        self.assertEqual(strict["unsupported_alignment_count"], 1)

        unrelated_text = "The catering invoice is ready for tomorrow's finance meeting."
        unrelated_dialogue = {
            "turns": [
                {
                    "turn_id": "D2:1",
                    "speaker": "Person 0",
                    "text": unrelated_text,
                }
            ],
            "dialogue_hash": "unrelated-dialogue-hash",
        }
        exact_but_unrelated = {
            "proposals": [
                {
                    "alignment_anchor": marker,
                    "turn_index": 0,
                    "support_quote": unrelated_text,
                }
            ]
        }
        semantic_guard = subject.verify_current_graph_dialogue_alignment(
            plan,
            unrelated_dialogue,
            exact_but_unrelated,
            {marker: event_id},
            require_exact_support_quote=True,
        )
        self.assertFalse(semantic_guard["coverage_pass"])
        self.assertEqual(semantic_guard["unsupported_alignment_count"], 1)

    def test_formal_segment_retry_contract_maps_only_failed_markers_to_exact_claims(self) -> None:
        assigned_items = [
            {
                "marker": "MARKER-0",
                "natural_support_claim_to_include": "Person 0 selected route-0 as the delivery route.",
            },
            {
                "marker": "MARKER-1",
                "natural_support_claim_to_include": "Person 1 selected route-1 as the delivery route.",
            },
        ]

        missing = subject._formal_segment_retry_contract(
            {
                "reason": "missing_segment_anchor",
                "missing_markers": ["MARKER-1"],
            },
            assigned_items,
        )
        unsupported = subject._formal_segment_retry_contract(
            {
                "reason": "unsupported_segment_alignment",
                "unsupported_markers": ["MARKER-0"],
            },
            assigned_items,
        )

        self.assertEqual(
            missing["failed_marker_to_exact_natural_support_claim"],
            {"MARKER-1": "Person 1 selected route-1 as the delivery route."},
        )
        self.assertEqual(
            unsupported["failed_marker_to_exact_natural_support_claim"],
            {"MARKER-0": "Person 0 selected route-0 as the delivery route."},
        )
        self.assertTrue(missing["regenerate_entire_uncommitted_segment"])
        self.assertIn("distinct NEW turn", " ".join(unsupported["required_repairs"]))
        for contract in (missing, unsupported):
            repairs = " ".join(contract["required_repairs"])
            self.assertIn("entire text equal", repairs)
            self.assertIn("do not add a prefix", repairs)
            self.assertIn("same digit characters", repairs)
            self.assertIn("named subject", repairs)
            self.assertIn("never let the named subject speak", repairs)
        self.assertIn("Never use a parent-context", " ".join(unsupported["required_repairs"]))

    def test_formal_segment_bare_list_normalizer_wraps_valid_turns_losslessly(self) -> None:
        turns = [
            {
                "speaker": "Maya",
                "text": "The delivery route remains the east corridor.",
                "alignment_anchors": ["MARKER-0"],
                "support_quotes": {
                    "MARKER-0": "The delivery route remains the east corridor."
                },
            },
            {
                "speaker": "Noah",
                "text": "I will coordinate the timing with the front desk.",
                "alignment_anchors": [],
                "support_quotes": {},
            },
        ]

        normalized, diagnostics = (
            subject.normalize_formal_dialogue_segment_bare_list_payload(turns)
        )

        self.assertEqual(normalized, {"appended_dialogue_turns_segment": turns})
        self.assertIsNot(normalized["appended_dialogue_turns_segment"], turns)
        self.assertTrue(
            diagnostics["formal_dialogue_segment_bare_list_normalizer_used"]
        )
        self.assertFalse(
            diagnostics["formal_dialogue_segment_bare_list_normalizer_failure"]
        )
        self.assertTrue(
            diagnostics["formal_dialogue_segment_bare_list_content_preserved"]
        )
        self.assertEqual(
            diagnostics["formal_dialogue_segment_bare_list_original_sha256"],
            diagnostics["formal_dialogue_segment_bare_list_wrapped_value_sha256"],
        )

    def test_formal_segment_bare_list_normalizer_rejects_nonturns_and_wrong_types(self) -> None:
        cases = (
            ([{"speaker": "Maya", "text": "Valid."}, "not a turn"], "turn_1_not_object"),
            ([{"speaker": "Maya", "text": 7}], "turn_0_invalid_text"),
            (
                [{"speaker": "Maya", "text": "Valid.", "alignment_anchors": "MARKER-0"}],
                "turn_0_invalid_alignment_anchors",
            ),
            (
                [{"speaker": "Maya", "text": "Valid.", "support_quotes": []}],
                "turn_0_invalid_support_quotes",
            ),
        )
        for candidate, reason in cases:
            with self.subTest(reason=reason):
                normalized, diagnostics = (
                    subject.normalize_formal_dialogue_segment_bare_list_payload(candidate)
                )
                self.assertIs(normalized, candidate)
                self.assertFalse(
                    diagnostics["formal_dialogue_segment_bare_list_normalizer_used"]
                )
                self.assertTrue(
                    diagnostics["formal_dialogue_segment_bare_list_normalizer_failure"]
                )
                self.assertEqual(
                    diagnostics["formal_dialogue_segment_bare_list_reject_reason"],
                    reason,
                )

    def test_formal_segment_raw_api_accepts_valid_bare_list_with_provenance(self) -> None:
        turns = [
            {
                "speaker": "Maya",
                "text": "The delivery route remains the east corridor.",
                "alignment_anchors": ["MARKER-0"],
                "support_quotes": {
                    "MARKER-0": "The delivery route remains the east corridor."
                },
            }
        ]
        captured_requests = []

        def respond(_url, request_payload, _api_key, _timeout):
            captured_requests.append(copy.deepcopy(request_payload))
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps(turns)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }

        args = SimpleNamespace(
            challenger_api_key="test-key",
            challenger_base_url="https://example.invalid/v1",
            challenger_model="test-model",
            challenger_temperature=0.25,
            challenger_max_tokens=4096,
            max_llm_retries=0,
            llm_timeout=1,
            api_yield_normalizers=True,
        )
        with mock.patch.object(subject.old_gate, "post_json_with_curl", side_effect=respond):
            payload, meta = subject.call_raw_json_api(
                args,
                {"task": "test"},
                system_prompt="Return JSON only.",
                required_keys=("appended_dialogue_turns_segment",),
                call_kind="api_formal_dialogue_segment",
                max_retries_override=0,
            )

        self.assertEqual(payload, {"appended_dialogue_turns_segment": turns})
        self.assertTrue(meta["used"])
        self.assertEqual(captured_requests[0]["response_format"], {"type": "json_object"})
        diagnostics = meta["normalizer_diagnostics"]
        self.assertTrue(
            diagnostics["formal_dialogue_segment_bare_list_normalizer_used"]
        )
        self.assertTrue(
            diagnostics["formal_dialogue_segment_bare_list_content_preserved"]
        )
        self.assertEqual(
            diagnostics["formal_dialogue_segment_bare_list_normalizer_source"],
            "bare_json_list",
        )

    def test_raw_json_api_temperature_override_is_explicit_and_audited(self) -> None:
        captured_requests = []

        def respond(_url, request_payload, _api_key, _timeout):
            captured_requests.append(copy.deepcopy(request_payload))
            return {
                "choices": [
                    {
                        "message": {"content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ]
            }

        args = SimpleNamespace(
            challenger_api_key="test-key",
            challenger_base_url="https://example.invalid/v1",
            challenger_model="test-model",
            challenger_temperature=0.25,
            challenger_max_tokens=4096,
            max_llm_retries=0,
            llm_timeout=1,
            api_yield_normalizers=False,
        )
        default_request = subject.build_json_request(
            args,
            {"task": "default"},
            "Return JSON only.",
            None,
        )
        self.assertEqual(default_request["temperature"], 0.25)
        with mock.patch.object(
            subject.old_gate,
            "post_json_with_curl",
            side_effect=respond,
        ):
            payload, meta = subject.call_raw_json_api(
                args,
                {"task": "formal diversity retry"},
                system_prompt="Return JSON only.",
                required_keys=("ok",),
                call_kind="api_formal_dialogue_segment",
                max_retries_override=0,
                temperature_override=0.55,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(captured_requests[0]["temperature"], 0.55)
        self.assertEqual(meta["temperature_override"], 0.55)
        self.assertEqual(meta["effective_temperature"], 0.55)

    def test_formal_segment_raw_api_never_replays_invalid_preview_internally(self) -> None:
        captured_requests = []

        def respond(_url, request_payload, _api_key, _timeout):
            captured_requests.append(copy.deepcopy(request_payload))
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Europe and the parent values 7 10 3 5 1 2 appeared in an invalid response"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        args = SimpleNamespace(
            challenger_api_key="test-key",
            challenger_base_url="https://example.invalid/v1",
            challenger_model="test-model",
            challenger_temperature=0.25,
            challenger_max_tokens=4096,
            max_llm_retries=3,
            llm_timeout=1,
            api_yield_normalizers=False,
        )
        with mock.patch.object(subject.old_gate, "post_json_with_curl", side_effect=respond):
            payload, meta = subject.call_raw_json_api(
                args,
                {"task": "formal segment test"},
                system_prompt="Return JSON only.",
                required_keys=("appended_dialogue_turns_segment",),
                call_kind="api_formal_dialogue_segment",
                max_retries_override=0,
            )

        self.assertIsNone(payload)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["attempts"], 1)
        self.assertEqual(meta["max_retries_override"], 0)
        self.assertEqual(len(captured_requests), 1)
        self.assertNotIn(
            "previous_response_error",
            json.loads(captured_requests[0]["messages"][1]["content"]),
        )

    def test_formal_segmented_dialogue_builds_verified_prefix_chain_and_global_alignment(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
                "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            }

        def fixed_tokens(turns, _tokenizer_path=None):
            return len(turns) * 100, "test"

        args = SimpleNamespace(
            solver_tokenizer=None,
            tokenizer_path=None,
            base_model=None,
            challenger_max_tokens=8192,
            max_llm_retries=0,
            formal_dialogue_max_segments=5,
            formal_dialogue_segment_target_tokens=400,
        )
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=fixed_tokens,
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                args,
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertTrue(meta["used"])
        self.assertTrue(meta["prefix_hash_chain_verified"])
        self.assertTrue(meta["qa_deferred_until_final_alignment"])
        self.assertEqual(
            meta["parent_context_provenance"]["audited_prompt_count"],
            meta["segment_api_call_count"],
        )
        self.assertTrue(
            meta["parent_context_provenance"][
                "internal_retry_disabled_for_parent_audit"
            ]
        )
        self.assertEqual(meta["segment_count"], 3)
        self.assertEqual(meta["api_native_appended_tokens"], 3600)
        self.assertEqual(len(set(meta["retry_generation_ids"])), 3)
        self.assertTrue(
            meta["accepted_prefix_context_provenance"][
                "accepted_prefix_prompt_audit_verified"
            ]
        )
        self.assertEqual(
            meta["accepted_prefix_context_provenance"][
                "accepted_prefix_verbatim_exposure_count"
            ],
            0,
        )
        self.assertTrue(
            meta["marker_scheduler_summary"]["all_markers_assigned_once_under_cap"]
        )
        self.assertEqual(meta["marker_scheduler_summary"]["marker_cap_per_segment"], 2)
        self.assertEqual(len(meta["marker_schedule_records"]), 3)
        for schedule, prompt in zip(meta["marker_schedule_records"], prompts):
            self.assertIn("pending_before_by_band", schedule)
            self.assertIn("pending_after_by_band", schedule)
            self.assertLessEqual(schedule["assigned_count"], 2)
            self.assertEqual(
                prompt["segment"]["marker_schedule_contract"]["assigned_count"],
                schedule["assigned_count"],
            )
            self.assertGreaterEqual(
                prompt["segment"]["requested_turns"],
                subject.FORMAL_DIALOGUE_SEGMENT_MIN_TURNS,
            )
            self.assertLessEqual(
                prompt["segment"]["requested_turns"],
                subject.FORMAL_DIALOGUE_SEGMENT_MAX_TURNS,
            )
            array_schema = prompt["output_schema"][
                "appended_dialogue_turns_segment"
            ]
            turn_schema = array_schema["item_schema"]
            self.assertEqual(
                set(prompt["output_schema"]),
                {"appended_dialogue_turns_segment"},
            )
            self.assertEqual(array_schema["value_type"], "array")
            self.assertEqual(
                array_schema["exact_item_count"],
                prompt["segment"]["requested_turns"],
            )
            self.assertNotIsInstance(array_schema, list)
            self.assertEqual(turn_schema["alignment_anchors"], [])
            self.assertEqual(turn_schema["support_quotes"], {})
            self.assertEqual(
                prompt["segment"]["neutral_turn_fields"],
                {"alignment_anchors": [], "support_quotes": {}},
            )
            marked_shapes = prompt["segment"]["required_marked_turn_shapes"]
            self.assertEqual(
                {shape["marker"] for shape in marked_shapes},
                set(prompt["segment"]["allowed_markers"]),
            )
            for shape in marked_shapes:
                marker = shape["marker"]
                exact_claim = prompt["segment"][
                    "marker_to_exact_natural_support_claim"
                ][marker]
                self.assertEqual(shape["alignment_anchors_must_equal"], [marker])
                self.assertEqual(
                    shape["support_quotes_must_equal"], {marker: exact_claim}
                )
                self.assertEqual(shape["text_must_equal"], exact_claim)
                self.assertTrue(shape["must_appear_in_exactly_one_turn"])
            self.assertNotIn(
                "allowed marker; empty on neutral turns",
                json.dumps(prompt["output_schema"], ensure_ascii=False),
            )
            prompt_json = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
            pending_after = {
                marker
                for markers in schedule["pending_after_by_band"].values()
                for marker in markers
            }
            for future_marker in pending_after:
                self.assertNotIn(future_marker, prompt_json)
            self.assertFalse(
                any(
                    "pending" in key or "assigned_markers" in key
                    for key in prompt["segment"]["marker_schedule_contract"]
                )
            )
        self.assertNotIn("qa_proposal", proposal)
        accepted_texts = []
        accepted_records = [
            record
            for record in meta["segment_records"]
            if (record.get("validation") or {}).get("pass") is True
        ]
        for prompt, record in zip(prompts, accepted_records):
            frozen_prefix = prompt["frozen_prefix"]
            self.assertNotIn("frozen_prefix_tail", frozen_prefix)
            self.assertFalse(frozen_prefix["accepted_prefix_turn_text_included"])
            manifest = frozen_prefix["accepted_prefix_context"]
            self.assertEqual(
                manifest["schema_version"],
                subject.FORMAL_ACCEPTED_PREFIX_CONTEXT_SCHEMA_VERSION,
            )
            self.assertFalse(
                manifest["accepted_prefix_digest"][
                    "accepted_turn_text_included"
                ]
            )
            serialized_prompt = json.dumps(
                prompt,
                ensure_ascii=False,
                sort_keys=True,
            )
            for accepted_text in accepted_texts:
                self.assertNotIn(accepted_text, serialized_prompt)
            accepted_texts.extend(
                turn["text"]
                for turn in record["validation"]["clean_turns"]
                if turn.get("text")
            )
        self.assertEqual(
            prompts[1]["frozen_prefix"]["prefix_hash"],
            meta["segment_records"][0]["prefix_hash_after"],
        )
        self.assertEqual(
            prompts[2]["frozen_prefix"]["prefix_hash"],
            meta["segment_records"][1]["prefix_hash_after"],
        )
        self.assertTrue(
            subject.verify_formal_dialogue_prefix_hash_chain(
                prompts[0]["frozen_prefix"]["prefix_hash"],
                meta["segment_records"],
                meta["final_prefix_hash"],
            )
        )
        tampered_records = copy.deepcopy(meta["segment_records"])
        tampered_records[0]["prefix_hash_after"] = "tampered"
        self.assertFalse(
            subject.verify_formal_dialogue_prefix_hash_chain(
                prompts[0]["frozen_prefix"]["prefix_hash"],
                tampered_records,
                meta["final_prefix_hash"],
            )
        )

        parent_turns = seed_row()["dialogue"]["turns"]
        frozen = subject.freeze_append_only_dialogue(
            parent_turns,
            subject.strip_alignment_anchors_from_dialogue(proposal["appended_dialogue_turns"]),
            0,
        )
        annotations = subject.parse_anchor_annotations(
            proposal["appended_dialogue_turns"],
            plan["current_anchor_to_event_instance_id"],
            [],
            turn_index_offset=len(parent_turns),
        )
        alignment = subject.verify_current_graph_dialogue_alignment(
            plan,
            frozen,
            annotations,
            plan["current_anchor_to_event_instance_id"],
        )
        dispersion = subject.stage64_evidence_dispersion_diagnostics(
            plan,
            frozen,
            alignment["current_alignment"],
            parent_turn_count=len(parent_turns),
        )
        self.assertTrue(alignment["coverage_pass"])
        self.assertTrue(dispersion["pass"])

    def test_formal_dialogue_structural_retry_remains_api_authored(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []

        def neutral_payload(prompt: dict) -> dict:
            count = int(prompt["segment"]["minimum_neutral_turns"])
            turns = [
                {
                    "speaker": f"Coordinator {index % 3}",
                    "text": (
                        f"Briefing {index} uses a distinct agenda, "
                        f"separate logistics, and a different follow-up discussion."
                    ),
                    "alignment_anchors": [],
                    "support_quotes": {},
                }
                for index in range(count)
            ]
            turns[0]["alignment_anchors"] = ["API-MARKER"]
            turns[0]["support_quotes"] = {"API-MARKER": "neutral metadata"}
            return {"appended_dialogue_turns_segment": turns}

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            if len(prompts) == 1:
                return {
                    "appended_dialogue_turns_segment": [
                        {
                            "speaker": "Coordinator",
                            "text": "This incomplete response deliberately misses the requested turn count.",
                            "alignment_anchors": [],
                            "support_quotes": {},
                        }
                    ]
                }, {"used": True, "attempts": 1, "usage": {}}
            if prompt["segment"].get("local_evidence_compiler", {}).get("enabled"):
                return neutral_payload(prompt), {"used": True, "attempts": 1, "usage": {}}
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
                "usage": {},
            }

        args = SimpleNamespace(
            solver_tokenizer=None,
            tokenizer_path=None,
            base_model=None,
            challenger_max_tokens=8192,
            max_llm_retries=0,
            formal_dialogue_max_segments=5,
            formal_dialogue_segment_target_tokens=400,
        )
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                args,
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        compiler_prompts = [
            prompt
            for prompt in prompts
            if prompt["segment"].get("local_evidence_compiler", {}).get("enabled")
        ]
        self.assertEqual(compiler_prompts, [])
        self.assertTrue(
            all(
                prompt["segment"].get("local_evidence_compiler", {}).get("enabled")
                is not True
                for prompt in prompts
            )
        )
        self.assertTrue(
            any(
                "segment_alignment_repair_contract" in prompt
                for prompt in prompts[1:]
            )
        )
        compiler_records = [
            record
            for record in meta["segment_records"]
            if (record.get("validation") or {}).get("local_evidence_compiler")
        ]
        self.assertEqual(compiler_records, [])

    def test_formal_segment_prompt_excludes_run116_parent_text_but_keeps_structure(self) -> None:
        plan = self._formal_segment_plan()
        parent = seed_row()
        parent["dialogue"]["turns"] = [
            {
                "turn_id": "D1:1",
                "speaker": "Maya",
                "text": (
                    "Europe was discussed alongside the ordered values 7, 10, 3, 5, 1, and 2 "
                    "during a long planning exchange that must never be copied into a new round."
                ),
            },
            {
                "turn_id": "D1:2",
                "speaker": "Noah",
                "text": (
                    "The archived itinerary contains another deliberately long parent sentence "
                    "whose wording belongs only to the preserved historical dialogue."
                ),
            },
        ]
        parent_context = {
            "selected_relevant_parent_turns": copy.deepcopy(
                parent["dialogue"]["turns"]
            ),
            "active_superseded_state_summary": {
                "active_or_latest_context": [
                    "Europe 7 10 3 5 1 2 is a parent-only active value"
                ],
                "superseded_or_nonsettled_context": [
                    "The old Europe schedule was superseded"
                ],
            },
            "diagnostics": {"parent_context_turns": 2},
        }
        prompts = []

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                parent,
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context=parent_context,
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertGreaterEqual(len(prompts), 2)
        for prompt in prompts:
            serialized = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("Europe", serialized)
            self.assertNotIn("7, 10, 3, 5, 1, and 2", serialized)
            self.assertNotIn("selected_relevant_parent_turns", serialized)
            audit = subject.audit_formal_parent_prompt_verbatim_exposure(
                prompt,
                parent["dialogue"]["turns"],
            )
            self.assertTrue(audit["pass"])
            self.assertEqual(audit["parent_verbatim_exposure_count"], 0)
            compact = prompt["compact_parent_context"]
            self.assertEqual(
                compact["schema_version"],
                subject.FORMAL_PARENT_CONTEXT_SCHEMA_VERSION,
            )
            self.assertEqual(
                compact["speaker_structure"]["speaker_set"], ["Maya", "Noah"]
            )
            self.assertEqual(
                compact["active_superseded_state_summary"]["active_or_latest"][
                    "item_count"
                ],
                1,
            )
        first_prefix = prompts[0]["frozen_prefix"]["accepted_prefix_context"]
        later_prefix = prompts[1]["frozen_prefix"]["accepted_prefix_context"]
        self.assertEqual(first_prefix["accepted_prefix_digest"]["full_turn_count"], 0)
        self.assertGreater(
            later_prefix["accepted_prefix_digest"]["full_turn_count"],
            0,
        )
        self.assertFalse(
            later_prefix["accepted_prefix_digest"]["accepted_turn_text_included"]
        )
        self.assertNotIn("frozen_prefix_tail", prompts[1]["frozen_prefix"])
        provenance = meta["parent_context_provenance"]
        self.assertTrue(provenance["parent_prompt_audit_verified"])
        self.assertEqual(provenance["parent_verbatim_exposure_count"], 0)
        self.assertEqual(
            provenance["schema_version"],
            subject.FORMAL_PARENT_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(
            provenance["prompt_audit_schema_version"],
            subject.FORMAL_PARENT_PROMPT_AUDIT_VERSION,
        )
        self.assertTrue(provenance["audit_sha256"])
        self.assertEqual(provenance["validated_parent_turn_count"], 2)
        self.assertEqual(
            provenance["audited_prompt_count"], meta["segment_api_call_count"]
        )

    def test_formal_segment_parent_prompt_exposure_hard_gates_before_api(self) -> None:
        plan = self._formal_segment_plan()
        parent = seed_row()
        parent_text = (
            "This exact long historical sentence is private to the preserved parent dialogue."
        )
        parent["dialogue"]["turns"][0]["text"] = parent_text
        unsafe_manifest = {
            "schema_version": subject.FORMAL_PARENT_CONTEXT_SCHEMA_VERSION,
            "manifest_sha256": "unsafe-test-manifest",
            "leaked_text": parent_text,
        }

        with mock.patch.object(
            subject,
            "build_formal_nonverbatim_parent_context",
            return_value=unsafe_manifest,
        ), mock.patch.object(subject, "call_raw_json_api") as api_call:
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                parent,
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNone(proposal)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["error_type"], "formal_parent_verbatim_prompt_exposure")
        self.assertGreater(
            meta["parent_context_provenance"]["parent_verbatim_exposure_count"],
            0,
        )
        api_call.assert_not_called()

    def test_run117_local_status_spans_are_provenance_exempt_without_exempting_fact_body(self) -> None:
        current_plan = {
            "events": [
                {
                    "event_instance_id": f"E{index}",
                    "dialogue_fact_id": f"F{index}",
                    "alignment_anchor": f"MARKER{index}",
                    "entity": name,
                    "slot": "project_note",
                    "value": value,
                    "dialogue_support_claim": claim,
                    "status": status,
                    "time_index": index,
                }
                for index, (name, value, claim, status) in enumerate(
                    (
                        ("Andrew", "analysis", "Andrew discussed a new analysis role", "outdated"),
                        ("Nate", "screenplay", "Nate discussed a screenplay idea", "belief"),
                        ("Joanna", "feedback", "Joanna shared a feedback update", "report"),
                    ),
                    start=1,
                )
            ],
            "current_event_instance_to_anchor": {
                "E1": "MARKER1",
                "E2": "MARKER2",
                "E3": "MARKER3",
            },
        }
        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(current_plan)
        spans = subject.formal_local_parent_audit_exempt_spans(dialogue_plan)
        prompt = {
            "claims": [
                event["dialogue_support_claim"] for event in dialogue_plan["events"]
            ],
            "status_meaning": dialogue_plan["events"][0]["status_meaning"],
        }
        parent_turns = [
            {
                "turn_id": f"D1:{index}",
                "speaker": "Parent",
                "text": text,
            }
            for index, text in enumerate(
                (
                    subject._status_natural_suffix("outdated"),
                    subject._status_natural_suffix("belief"),
                    subject._status_natural_suffix("report"),
                    subject._formal_status_meaning("outdated"),
                ),
                start=1,
            )
        ]

        audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            prompt,
            parent_turns,
            local_exempt_spans=spans,
        )

        self.assertTrue(audit["pass"])
        self.assertEqual(audit["parent_verbatim_exposure_count"], 0)
        self.assertGreaterEqual(audit["local_exempted_span_count"], 4)
        self.assertIn(
            "locally_appended_status_natural_suffix",
            audit["local_exemption_reasons"],
        )
        self.assertIn(
            "local_canonical_status_meaning", audit["local_exemption_reasons"]
        )
        self.assertTrue(audit["local_exempted_span_sha256"])

        copied_parent_fact = (
            "Andrew discussed a new analysis role before the department meeting"
        )
        unsafe_prompt = copy.deepcopy(prompt)
        unsafe_prompt["claims"][0] = subject.normalize_space(
            f"{copied_parent_fact}. {subject._status_natural_suffix('outdated')}"
        )
        unsafe_parent = [
            {"turn_id": "D2:1", "speaker": "Parent", "text": copied_parent_fact}
        ]
        unsafe_audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            unsafe_prompt,
            unsafe_parent,
            local_exempt_spans=spans,
        )
        self.assertFalse(unsafe_audit["pass"])
        self.assertGreater(unsafe_audit["parent_verbatim_exposure_count"], 0)

    def test_canonical_claim_status_suffix_is_locally_exempt_but_core_remains_audited(self) -> None:
        current_plan = {
            "operation": "temporal_state_update",
            "graph_snapshot_hash": "canonical-audit",
            "events": [
                {
                    "event_instance_id": "E-canonical",
                    "dialogue_fact_id": "F-canonical",
                    "alignment_anchor": "CANONICALMARKER",
                    "entity": "Andrew",
                    "slot": "project_note",
                    "value": "financial analysis",
                    "dialogue_value": "financial analysis",
                    "dialogue_support_claim": "A vague note without the required fact",
                    "status": "outdated",
                    "time_index": 1,
                }
            ],
            "current_event_instance_to_anchor": {
                "E-canonical": "CANONICALMARKER"
            },
        }
        initial = subject.sanitize_event_plan_for_dialogue_api(current_plan)
        validated, diagnostics = subject.formal_segment_exact_claim_preflight(
            current_plan,
            initial,
        )

        self.assertIsNotNone(validated)
        self.assertEqual(diagnostics["canonical_claim_rewrite_count"], 1)
        event = validated["events"][0]
        self.assertEqual(
            event["local_claim_transform_provenance"]["source"],
            "local_canonical_status_suffix",
        )
        spans = subject.formal_local_parent_audit_exempt_spans(validated)
        suffix = subject._status_natural_suffix("outdated")
        suffix_audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            {"claim": event["dialogue_support_claim"]},
            [{"turn_id": "D1:1", "speaker": "Parent", "text": suffix}],
            local_exempt_spans=spans,
        )
        self.assertTrue(suffix_audit["pass"])
        self.assertIn(
            "locally_appended_canonical_status_suffix",
            suffix_audit["local_exemption_reasons"],
        )

        copied_core = "Regarding Andrew, the project note is financial analysis."
        core_audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            {"claim": event["dialogue_support_claim"]},
            [{"turn_id": "D1:2", "speaker": "Parent", "text": copied_core}],
            local_exempt_spans=spans,
        )
        self.assertFalse(core_audit["pass"])
        self.assertGreater(core_audit["parent_verbatim_exposure_count"], 0)

    def test_run118_internal_status_surface_rewrites_are_provenance_exempt(self) -> None:
        synthetic_status_text = {
            "belief": "This is a belief, not an active current state.",
            "report": "This is only a report, not an active current state.",
            "expired": "This is expired and not an active current state.",
            "canceled": "This is canceled and not an active current state.",
            "outdated": "This is outdated and not an active current state.",
            "false": "This is false and not an active current state.",
            "delayed_not_active": "This is delayed and not active yet.",
            "scope_limited": (
                "This is scope limited, not a generally visible active state."
            ),
        }
        events = []
        for index, (status, status_text) in enumerate(
            synthetic_status_text.items(), start=1
        ):
            raw_claim = f"Project {index} has setting value {index}. {status_text}"
            events.append(
                {
                    "event_instance_id": f"E{index}",
                    "dialogue_fact_id": f"F{index}",
                    "alignment_anchor": f"MARKER{index}",
                    "entity": f"Project {index}",
                    "slot": "setting",
                    "value": f"value {index}",
                    "status": status,
                    "time_index": index,
                    "raw_support_claim": raw_claim,
                    "dialogue_support_claim": subject.sanitize_internal_text(raw_claim),
                }
            )
        current_plan = {
            "events": events,
            "current_event_instance_to_anchor": {
                event["event_instance_id"]: event["alignment_anchor"]
                for event in events
            },
        }

        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(current_plan)
        for event in dialogue_plan["events"]:
            provenance = event["local_claim_transform_provenance"]
            self.assertEqual(
                provenance["source"],
                "local_internal_status_surface_replacement",
            )
            self.assertTrue(provenance["surface_replacement_verified"])
            self.assertTrue(provenance["suffix_appended"])

        spans = subject.formal_local_parent_audit_exempt_spans(dialogue_plan)
        prompt = {
            "claims": [
                event["dialogue_support_claim"] for event in dialogue_plan["events"]
            ]
        }
        parent_turns = [
            {
                "turn_id": f"D1:{index}",
                "speaker": "Parent",
                "text": subject._status_natural_suffix(status),
            }
            for index, status in enumerate(synthetic_status_text, start=1)
        ]
        audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            prompt,
            parent_turns,
            local_exempt_spans=spans,
        )

        self.assertTrue(audit["pass"])
        self.assertEqual(audit["parent_verbatim_exposure_count"], 0)
        self.assertIn(
            "locally_rewritten_status_surface", audit["local_exemption_reasons"]
        )

        copied_core = "Project 1 has setting value 1 before the later update"
        unsafe_prompt = copy.deepcopy(prompt)
        unsafe_prompt["claims"][0] = subject.normalize_space(
            f"{copied_core}. {subject._status_natural_suffix('belief')}"
        )
        unsafe_audit = subject.audit_formal_parent_prompt_verbatim_exposure(
            unsafe_prompt,
            [{"turn_id": "D2:1", "speaker": "Parent", "text": copied_core}],
            local_exempt_spans=spans,
        )
        self.assertFalse(unsafe_audit["pass"])

    def test_nonactive_status_suffix_is_not_skipped_by_broad_aliases(self) -> None:
        cases = (
            ("report", "ReportB distribution remains internal"),
            ("scope_limited", "ReportB distribution is Internal Only"),
            ("outdated", "The old service address is Pine Street"),
        )
        for status, claim in cases:
            with self.subTest(status=status):
                rendered = subject._naturalize_claim_for_status(claim, status)
                self.assertIn(
                    subject.normalize_text(subject._status_natural_suffix(status)),
                    subject.normalize_text(rendered),
                )

    def test_nonactive_own_status_precedes_dependency_surface(self) -> None:
        predecessor = {
            "event_instance_id": "E3",
            "entity": "ProjectX",
            "slot": "access scope",
            "value": "Public Access",
            "status": "outdated",
        }
        event = {
            "event_instance_id": "E4",
            "entity": "ProjectX",
            "slot": "access scope",
            "value": "Restricted to Executives",
            "status": "canceled",
            "depends_on": ["E3"],
        }
        claim = subject._formal_canonical_exact_claim(
            event,
            event_by_id={"E3": predecessor, "E4": event},
        )
        self.assertLess(claim.index("Restricted to Executives"), claim.index("withdrawn"))
        self.assertLess(claim.index("withdrawn"), claim.index("Public Access"))
        self.assertIn("effect. This refers", claim)

    def test_formal_segment_copy_output_still_rejected_against_full_parent(self) -> None:
        plan = self._formal_segment_plan(event_count=1)
        marker = plan["events"][0]["alignment_anchor"]
        claim = plan["events"][0]["dialogue_support_claim"]
        copied_parent_text = (
            "This complete historical planning turn has enough detail to be detected as an exact copy."
        )
        result = subject.validate_formal_dialogue_segment(
            {
                "appended_dialogue_turns_segment": [
                    {
                        "speaker": "Coordinator",
                        "text": claim,
                        "alignment_anchors": [marker],
                        "support_quotes": {marker: claim},
                    },
                    {
                        "speaker": "Maya",
                        "text": copied_parent_text,
                        "alignment_anchors": [],
                        "support_quotes": {},
                    },
                ]
            },
            assigned_markers={marker},
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[
                {"speaker": "Maya", "text": copied_parent_text}
            ],
            accepted_appended_turns=[],
            marker_to_exact_claim={marker: claim},
        )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "frozen_prefix_turn_repeated")

    def test_marker_free_formal_segment_accepts_bounded_extra_neutral_turn(self) -> None:
        """One extra neutral turn is retained; it is never silently truncated."""
        plan = self._formal_segment_plan(event_count=1)
        prompt = {
            "segment": {
                "segment_index": 0,
                "requested_turns": 17,
                "assigned_anchor_realization_checklist": [],
            }
        }
        proposal = self._formal_segment_payload(prompt)
        proposal["appended_dialogue_turns_segment"][-1]["text"] = (
            "A separate editorial group records a distinct review sequence for an unrelated later briefing. "
            "Its participants compare draft order, room setup, and approval timing without revisiting earlier decisions."
        )
        result = subject.validate_formal_dialogue_segment(
            proposal,
            expected_turn_count=(16, 17),
            assigned_markers=set(),
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            accepted_appended_turns=[],
            marker_to_exact_claim={},
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["actual_turn_count"], 17)

        proposal["appended_dialogue_turns_segment"].append(
            {
                "speaker": "Coordinator 9",
                "text": "A final independent note records an unrelated timetable for a later discussion.",
                "alignment_anchors": [],
                "support_quotes": {},
            }
        )
        rejected = subject.validate_formal_dialogue_segment(
            proposal,
            expected_turn_count=(16, 17),
            assigned_markers=set(),
            anchor_to_event=plan["current_anchor_to_event_instance_id"],
            current_event_plan=plan,
            frozen_prefix_turns=[],
            accepted_appended_turns=[],
            marker_to_exact_claim={},
        )
        self.assertFalse(rejected["pass"])
        self.assertEqual(rejected["reason"], "segment_turn_count_out_of_range")

    def test_fixed_stage3_graph_artifact_binds_parent_target_and_anchor_ledger(self) -> None:
        target = {"qa_operation": "comparison", "preferred_answer_type": "Number"}
        seed = {
            "sample_id": "parent-1",
            "semantic_parent_graph": {"graph_id": "parent-graph"},
        }
        artifact = {
            "schema_version": subject.FIXED_STAGE3_GRAPH_ARTIFACT_SCHEMA_VERSION,
            "seed_sample_id": "parent-1",
            "stage64_target_identity": "target-1",
            "stage64_target": copy.deepcopy(target),
            "parent_graph": {"graph_id": "parent-graph"},
            "child_graph_delta": {"parent_graph_id": "parent-graph"},
            "current_event_plan": {
                "events": [{"event_instance_id": "child:E1"}],
                "current_event_instance_to_anchor": {"child:E1": "ANCHOR-1"},
                "current_anchor_to_event_instance_id": {"ANCHOR-1": "child:E1"},
            },
            "anchor_ledger": {},
            "artifact_id": "stage3_materialized_test",
            "source_graph_meta": {
                "semantic_graph_v2_quality": {
                    "pass": True,
                    "qa_operation": "comparison",
                    "gate_version": subject.GRAPH_CACHE_SCHEMA_VERSION,
                    "issues": [],
                },
                "parent_child_continuity_audit": {
                    "pass": True,
                    "continuity_score": 1.0,
                },
                "usage": {"total_tokens": 999},
            },
        }
        self.assertEqual(
            subject.fixed_stage3_graph_artifact_issues(
                artifact,
                seed_row=seed,
                stage64_target_identity="target-1",
                stage64_target=target,
            ),
            [],
        )
        reuse_meta = subject.fixed_stage3_graph_artifact_reuse_graph_meta(
            artifact,
            expected_operation="comparison",
            artifact_source="materialized_reuse",
            artifact_path="/tmp/materialized.json",
        )
        self.assertTrue(reuse_meta["semantic_graph_v2_quality"]["pass"])
        self.assertEqual(
            reuse_meta["semantic_graph_v2_quality"]["gate_version"],
            subject.GRAPH_CACHE_SCHEMA_VERSION,
        )
        self.assertEqual(
            reuse_meta["parent_child_continuity_audit"]["continuity_score"],
            1.0,
        )
        self.assertNotIn("usage", reuse_meta)
        self.assertFalse(reuse_meta["api_called"])

        source_gate_mutations = (
            (
                lambda value: value.pop("semantic_graph_v2_quality"),
                "fixed_graph_source_semantic_gate_missing",
            ),
            (
                lambda value: value["semantic_graph_v2_quality"].update(
                    {"pass": False}
                ),
                "fixed_graph_source_semantic_gate_failed",
            ),
            (
                lambda value: value["semantic_graph_v2_quality"].update(
                    {"gate_version": "semantic_graph_v1"}
                ),
                "fixed_graph_source_semantic_gate_version_mismatch",
            ),
            (
                lambda value: value["semantic_graph_v2_quality"].update(
                    {"qa_operation": "count"}
                ),
                "fixed_graph_source_semantic_gate_operation_mismatch",
            ),
        )
        for mutate, expected_code in source_gate_mutations:
            with self.subTest(expected_code=expected_code):
                changed = copy.deepcopy(artifact)
                mutate(changed["source_graph_meta"])
                codes = {
                    issue["code"]
                    for issue in subject.fixed_stage3_graph_source_gate_issues(
                        changed,
                        expected_operation="comparison",
                    )
                }
                self.assertIn(expected_code, codes)
                with self.assertRaises(subject.NativeGraphDialogueReject):
                    subject.fixed_stage3_graph_artifact_reuse_graph_meta(
                        changed,
                        expected_operation="comparison",
                        artifact_source="materialized_reuse",
                        artifact_path="/tmp/materialized.json",
                    )
        artifact["current_event_plan"]["current_anchor_to_event_instance_id"] = {
            "ANCHOR-1": "different-event"
        }
        issues = subject.fixed_stage3_graph_artifact_issues(
            artifact,
            seed_row=seed,
            stage64_target_identity="target-1",
            stage64_target=target,
        )
        self.assertIn("fixed_graph_anchor_coverage_mismatch", [x["code"] for x in issues])

    def test_materialized_stage3_graph_artifact_roundtrips_with_target_and_parent(self) -> None:
        """A batch retry must recover the exact G3, not regenerate it."""
        import tempfile

        target = {"qa_operation": "comparison", "preferred_answer_type": "Number"}
        seed = {
            "sample_id": "parent-1",
            "semantic_parent_graph": {
                "graph_id": "parent-graph",
                "events": [
                    {"event_id": "parent:E1", "evidence_turn_ids": ["D1:1"]}
                ],
            },
        }
        plan = {
            "graph_id": "child-graph",
            "events": [{"event_instance_id": "child:E1"}],
            "current_event_instance_to_anchor": {"child:E1": "ANCHOR-1"},
            "current_anchor_to_event_instance_id": {"ANCHOR-1": "child:E1"},
        }
        artifact = subject.build_stage3_graph_artifact(
            seed_row=seed,
            current_event_plan=plan,
            child_graph_delta={"graph_id": "child-graph", "parent_graph_id": "parent-graph"},
            stage64_target=target,
            stage64_target_identity="a" * 64,
            source_graph_meta={
                "used": True,
                "semantic_graph_v2_quality": {
                    "pass": True,
                    "qa_operation": "comparison",
                    "gate_version": subject.GRAPH_CACHE_SCHEMA_VERSION,
                    "issues": [],
                },
            },
        )
        self.assertEqual(
            subject.fixed_stage3_graph_artifact_issues(
                artifact,
                seed_row=seed,
                stage64_target_identity="a" * 64,
                stage64_target=target,
            ),
            [],
        )
        self.assertEqual(
            artifact["anchor_ledger"]["parent_event_evidence_turn_ids"],
            {"parent:E1": ["D1:1"]},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = subject.stage3_graph_artifact_path(
                temp_dir,
                target_identity="a" * 64,
            )
            subject.write_json_atomic(path, artifact)
            reloaded = subject.load_stage3_graph_artifact(path)
        self.assertEqual(reloaded["artifact_id"], artifact["artifact_id"])

    def test_formal_invalid_anchor_retry_contract_lists_only_assigned_markers(self) -> None:
        assigned = [
            {
                "marker": "MARKER-0",
                "natural_support_claim_to_include": "Person 0 selected route-0.",
            },
            {
                "marker": "MARKER-1",
                "natural_support_claim_to_include": "Person 1 selected route-1.",
            },
        ]
        contract = subject._formal_segment_retry_contract(
            {
                "reason": "invalid_segment_anchor",
                "invalid_markers": ["Europe", "7", "10", "3", "5", "1", "2"],
            },
            assigned,
        )

        self.assertEqual(
            contract["allowed_exact_marker_strings"], ["MARKER-0", "MARKER-1"]
        )
        self.assertTrue(contract["silent_anchor_normalization_forbidden"])
        repairs = " ".join(contract["required_repairs"])
        self.assertIn("alignment_anchors=[]", repairs)
        self.assertIn("support_quotes={}", repairs)
        self.assertIn("geographic regions, numeric tokens", repairs)
        self.assertIn("Do not silently normalize", repairs)
        self.assertEqual(contract["invalid_or_unexpected_marker_count"], 7)
        self.assertFalse(contract["invalid_marker_text_replayed"])
        self.assertNotIn("Europe", json.dumps(contract, ensure_ascii=False))

    def test_formal_invalid_anchor_retry_prompt_exposes_exact_allowed_markers(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        request_kwargs = []

        def respond(_args, prompt, **kwargs):
            prompts.append(copy.deepcopy(prompt))
            request_kwargs.append(copy.deepcopy(kwargs))
            payload = self._formal_segment_payload(prompt)
            if len(prompts) == 1:
                neutral = next(
                    turn
                    for turn in payload["appended_dialogue_turns_segment"]
                    if not turn["alignment_anchors"]
                )
                neutral["alignment_anchors"] = [
                    "Europe",
                    "7",
                    "10",
                    "3",
                    "5",
                    "1",
                    "2",
                ]
            return payload, {
                "used": True,
                "attempts": 1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertGreaterEqual(len(prompts), 2)
        retry = prompts[1]["segment_alignment_repair_contract"]
        expected_markers = sorted(prompts[0]["segment"]["allowed_markers"])
        self.assertEqual(retry["reason"], "invalid_segment_anchor")
        self.assertEqual(retry["allowed_exact_marker_strings"], expected_markers)
        self.assertEqual(retry["invalid_or_unexpected_marker_count"], 7)
        self.assertFalse(retry["invalid_marker_text_replayed"])
        self.assertTrue(retry["silent_anchor_normalization_forbidden"])
        self.assertIsNone(request_kwargs[0]["temperature_override"])
        self.assertEqual(
            request_kwargs[1]["temperature_override"],
            subject.FORMAL_DIALOGUE_DIVERSITY_RETRY_TEMPERATURE,
        )
        self.assertEqual(
            prompts[1]["retry_variation_contract"]["structural_retry_reason"],
            "invalid_segment_anchor",
        )
        self.assertIsNone(
            meta["segment_records"][1]["meta"]["diversity_retry_reason"]
        )
        self.assertEqual(meta["retry_temperature_override_count"], 1)
        self.assertEqual(meta["structural_temperature_override_count"], 1)
        self.assertEqual(meta["diversity_temperature_override_count"], 0)
        generation_summary = subject._empty_summary()
        subject._record_api_meta_retry_diagnostics(generation_summary, meta)
        self.assertEqual(
            generation_summary[
                "formal_dialogue_structural_temperature_override_count"
            ],
            1,
        )
        self.assertEqual(
            generation_summary[
                "formal_dialogue_structural_retry_reason_counts"
            ],
            {"invalid_segment_anchor": 1},
        )
        self.assertEqual(
            generation_summary[
                "formal_dialogue_diversity_temperature_override_count"
            ],
            0,
        )
        retry_prompt_json = json.dumps(prompts[1], ensure_ascii=False, sort_keys=True)
        self.assertNotIn("Europe", retry_prompt_json)
        self.assertNotIn('"invalid_markers"', retry_prompt_json)

    def test_formal_segment_duplicate_retries_entire_segment_before_token_commit(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        request_kwargs = []
        first_call = True
        repeated = (
            "The volunteer group reviewed the weekend schedule and assigned every open shift. "
            "They checked the contact sheet, confirmed the meeting point, and prepared written "
            "instructions for families who would arrive after the orientation session."
        )

        def respond(_args, prompt, **_kwargs):
            nonlocal first_call
            prompts.append(copy.deepcopy(prompt))
            request_kwargs.append(copy.deepcopy(_kwargs))
            payload = self._formal_segment_payload(prompt)
            if first_call:
                first_call = False
                payload["appended_dialogue_turns_segment"][-2]["text"] = repeated
                payload["appended_dialogue_turns_segment"][-1]["text"] = repeated
            return payload, {"used": True, "attempts": 1}

        def fixed_tokens(turns, _tokenizer_path=None):
            return len(turns) * 100, "test"

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=fixed_tokens,
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=1,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertEqual(meta["segment_api_call_count"], 4)
        self.assertEqual(meta["segment_count"], 3)
        self.assertEqual(meta["api_native_appended_turn_count"], 36)
        self.assertEqual(meta["api_native_appended_tokens"], 3600)
        self.assertFalse(meta["segment_records"][0]["validation"]["pass"])
        self.assertEqual(
            meta["segment_records"][0]["validation"]["reason"],
            "segment_exact_duplicate_turn",
        )
        self.assertTrue(meta["segment_records"][1]["validation"]["pass"])
        self.assertEqual(
            prompts[1]["previous_segment_validation_error"]["reason"],
            "segment_exact_duplicate_turn",
        )
        retry = prompts[1]["segment_alignment_repair_contract"]
        self.assertTrue(retry["regenerate_entire_uncommitted_segment"])
        self.assertIn("silently delete", " ".join(retry["required_repairs"]))
        self.assertIsNone(request_kwargs[0]["temperature_override"])
        self.assertEqual(
            request_kwargs[1]["temperature_override"],
            subject.FORMAL_DIALOGUE_DIVERSITY_RETRY_TEMPERATURE,
        )
        self.assertNotEqual(
            prompts[0]["retry_variation_contract"]["retry_generation_id"],
            prompts[1]["retry_variation_contract"]["retry_generation_id"],
        )
        forbidden = prompts[1]["retry_variation_contract"][
            "forbidden_opening_prefixes"
        ]
        self.assertIn("the volunteer group", forbidden)
        self.assertIn("the volunteer group reviewed", forbidden)
        self.assertNotIn(repeated, json.dumps(prompts[1], ensure_ascii=False))
        self.assertEqual(
            prompts[0]["segment"]["turn_duplicate_contract"]["schema_version"],
            subject.FORMAL_SEGMENT_TURN_DUPLICATE_AUDIT_VERSION,
        )
        requirements = " ".join(prompts[0]["requirements"])
        self.assertIn("prior accepted appended turn", requirements)
        self.assertIn("genuinely new semantic information", requirements)
        audit_summary = meta["turn_duplicate_audit_summary"]
        self.assertTrue(audit_summary["pass"])
        self.assertEqual(audit_summary["accepted_segment_audit_count"], 3)
        self.assertEqual(audit_summary["audited_segment_turn_count"], 36)
        self.assertEqual(audit_summary["exact_duplicate_pair_count"], 0)
        self.assertEqual(
            proposal["segmented_dialogue_protocol"]["turn_duplicate_audit_summary"],
            audit_summary,
        )

    def test_formal_opening_diversity_audit_replays_run114_and_run105(self) -> None:
        root = Path(__file__).resolve().parents[1]
        fixtures = {
            "114_stage3_v9_segmented_e2e_comparison_n1_seed20260815": False,
            "105_stage3_v9_segmented_e2e_micro_n1_tail4_retry_seed20260805": True,
        }
        for run_name, expected_all_pass in fixtures.items():
            raw_path = (
                root
                / "base_restart_current_method_validation"
                / run_name
                / "generation"
                / "graph_conditioned_api_dialogue_raw.jsonl"
            )
            if not raw_path.exists():
                self.skipTest(f"local replay fixture is unavailable: {raw_path}")
            raw_rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            accepted_row = next(row for row in reversed(raw_rows) if row.get("accepted"))
            meta = accepted_row["api_dialogue_attempts"][-1]["meta"]
            claim_map = {
                row["marker"]: row["validated_claim"]
                for row in meta["exact_claim_preflight"]["claim_provenance"]
                if row.get("marker") and row.get("validated_claim")
            }
            accepted_turns = []
            audit_results = []
            for record in meta["segment_records"]:
                validation = record.get("validation") or {}
                if validation.get("pass") is not True:
                    continue
                segment_turns = validation["marked_turns"]
                audit = subject.audit_formal_dialogue_segment_opening_diversity(
                    segment_turns,
                    accepted_turns,
                    claim_map,
                )
                audit_results.append(audit)
                accepted_turns.extend(segment_turns)

            self.assertTrue(audit_results)
            self.assertEqual(
                all(audit["pass"] for audit in audit_results),
                expected_all_pass,
                run_name,
            )
            if run_name.startswith("114_"):
                final_rejections = audit_results[-1]["rejected_prefixes"]
                self.assertTrue(
                    any(
                        row["normalized_prefix"] == "i m going to"
                        and row["count"] == 37
                        for row in final_rejections
                    )
                )

    def test_formal_opening_diversity_strips_claim_and_short_body_is_diagnostic(self) -> None:
        marker = "MARKER-CLAIM"
        claim = "Regarding Joanna, the screenplay chapter count is 5"
        varied_bodies = [
            "A careful outline now separates the middle scenes from the ending.",
            "During rehearsal, the group can compare pacing across the two acts.",
            "Several notes from yesterday suggest moving the reveal much earlier.",
            "Because the venue closes early, tonight's review needs a firm agenda.",
            "Nate offered a fresh transition that connects the family subplot cleanly.",
            "Before lunch, everyone should annotate one scene without rewriting it.",
            "The producer requested a short table read before any budget discussion.",
            "Once those revisions settle, the team can revisit the final sequence.",
        ]
        claim_turns = [
            {
                "speaker": "Editor",
                "text": f"{claim}. {body}",
                "alignment_anchors": [marker],
            }
            for body in varied_bodies
        ]
        audit = subject.audit_formal_dialogue_segment_opening_diversity(
            claim_turns,
            [],
            {marker: claim},
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(
            audit["segment_scope"]["canonical_support_prefix_stripped_count"],
            8,
        )

        claim_only = [
            {
                "speaker": "Editor",
                "text": claim,
                "alignment_anchors": [marker],
            }
            for _ in range(4)
        ]
        short_audit = subject.audit_formal_dialogue_segment_opening_diversity(
            claim_only,
            [],
            {marker: claim},
        )
        self.assertTrue(short_audit["pass"])
        self.assertEqual(short_audit["segment_scope"]["short_or_empty_body_count"], 4)
        self.assertTrue(
            short_audit["segment_scope"]["short_or_empty_body_is_diagnostic_only"]
        )

    def test_formal_opening_diversity_retries_entire_segment_before_token_commit(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        request_kwargs = []
        first_call = True

        def respond(_args, prompt, **_kwargs):
            nonlocal first_call
            prompts.append(copy.deepcopy(prompt))
            request_kwargs.append(copy.deepcopy(_kwargs))
            payload = self._formal_segment_payload(prompt)
            if first_call:
                first_call = False
                for turn_index, turn in enumerate(
                    payload["appended_dialogue_turns_segment"]
                ):
                    repeated_body = (
                        "I'm going to advance a distinct agenda item numbered "
                        f"{turn_index} before the group changes topics."
                    )
                    if not turn["alignment_anchors"]:
                        turn["text"] = repeated_body
            return payload, {"used": True, "attempts": 1}

        def fixed_tokens(turns, _tokenizer_path=None):
            return len(turns) * 100, "test"

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=fixed_tokens,
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=1,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertEqual(meta["segment_api_call_count"], 4)
        self.assertEqual(meta["segment_count"], 3)
        self.assertEqual(meta["api_native_appended_turn_count"], 36)
        self.assertEqual(meta["api_native_appended_tokens"], 3600)
        self.assertEqual(
            meta["segment_records"][0]["validation"]["reason"],
            "segment_opening_prefix_concentration",
        )
        self.assertTrue(meta["segment_records"][1]["validation"]["pass"])
        self.assertEqual(
            prompts[1]["previous_segment_validation_error"]["reason"],
            "segment_opening_prefix_concentration",
        )
        retry = prompts[1]["segment_alignment_repair_contract"]
        self.assertTrue(retry["regenerate_entire_uncommitted_segment"])
        self.assertIn("I'm going to", " ".join(retry["required_repairs"]))
        self.assertIsNone(request_kwargs[0]["temperature_override"])
        self.assertEqual(
            request_kwargs[1]["temperature_override"],
            subject.FORMAL_DIALOGUE_DIVERSITY_RETRY_TEMPERATURE,
        )
        self.assertIn(
            "i m going",
            prompts[1]["retry_variation_contract"][
                "forbidden_opening_prefixes"
            ],
        )
        self.assertIn(
            "i m going to",
            prompts[1]["retry_variation_contract"][
                "forbidden_opening_prefixes"
            ],
        )
        prompt_contract = prompts[0]["segment"]["opening_diversity_contract"]
        self.assertEqual(
            prompt_contract["schema_version"],
            subject.FORMAL_SEGMENT_OPENING_DIVERSITY_AUDIT_VERSION,
        )
        self.assertTrue(prompt_contract["token_length_credit_before_pass_forbidden"])
        summary = meta["opening_diversity_audit_summary"]
        self.assertTrue(summary["pass"])
        self.assertEqual(summary["accepted_segment_audit_count"], 3)
        self.assertEqual(summary["audited_segment_turn_count"], 36)
        self.assertEqual(summary["accepted_concentrated_prefix_count"], 0)
        self.assertEqual(
            proposal["segmented_dialogue_protocol"][
                "opening_diversity_audit_summary"
            ],
            summary,
        )
        generation_summary = subject._empty_summary()
        subject._record_api_meta_retry_diagnostics(generation_summary, meta)
        self.assertEqual(
            generation_summary[
                "formal_segment_opening_concentration_reject_count"
            ],
            1,
        )
        self.assertEqual(
            generation_summary[
                "formal_segment_opening_segment_scope_reject_count"
            ],
            1,
        )
        self.assertGreaterEqual(
            generation_summary["formal_segment_opening_rejected_prefix_count"],
            2,
        )
        self.assertEqual(
            generation_summary[
                "formal_dialogue_diversity_temperature_override_count"
            ],
            1,
        )
        self.assertGreaterEqual(
            generation_summary[
                "formal_dialogue_forbidden_opening_prefix_count_max"
            ],
            2,
        )

    def test_formal_retry_accumulates_short_openings_across_failure_types(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        request_kwargs = []

        def respond(_args, prompt, **kwargs):
            prompts.append(copy.deepcopy(prompt))
            request_kwargs.append(copy.deepcopy(kwargs))
            payload = self._formal_segment_payload(prompt)
            if len(prompts) == 1:
                for index, turn in enumerate(
                    payload["appended_dialogue_turns_segment"]
                ):
                    turn["text"] = (
                        "I will also prepare a distinct planning note numbered "
                        f"{index} before the next topic begins."
                    )
                    turn["alignment_anchors"] = []
                    turn["support_quotes"] = {}
                for item_index, item in enumerate(
                    prompt["segment"]["assigned_anchor_realization_checklist"]
                ):
                    marker = item["marker"]
                    claim = item["natural_support_claim_to_include"]
                    payload["appended_dialogue_turns_segment"][item_index].update(
                        {
                            "text": claim,
                            "alignment_anchors": [marker],
                            "support_quotes": {marker: claim},
                        }
                    )
            elif len(prompts) == 2:
                payload["appended_dialogue_turns_segment"].pop()
            return payload, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            side_effect=respond,
        ), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_temperature=0.25,
                    challenger_max_tokens=8192,
                    max_llm_retries=1,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertEqual(
            [
                prompts[index].get("previous_segment_validation_error", {}).get(
                    "reason"
                )
                for index in (1, 2)
            ],
            [
                "segment_opening_prefix_concentration",
                "segment_turn_count_mismatch",
            ],
        )
        self.assertEqual(
            request_kwargs[1]["temperature_override"],
            subject.FORMAL_DIALOGUE_DIVERSITY_RETRY_TEMPERATURE,
        )
        self.assertEqual(
            request_kwargs[2]["temperature_override"],
            subject.FORMAL_DIALOGUE_DIVERSITY_RETRY_TEMPERATURE,
        )
        self.assertIn("segment_alignment_repair_contract", prompts[2])
        self.assertFalse(
            prompts[2]["segment"].get("local_evidence_compiler", {}).get("enabled")
        )
        self.assertEqual(
            prompts[2]["retry_variation_contract"]["structural_retry_reason"],
            "segment_turn_count_mismatch",
        )
        self.assertIsNone(
            meta["segment_records"][2]["meta"]["diversity_retry_reason"]
        )
        self.assertEqual(meta["retry_temperature_override_count"], 2)
        self.assertEqual(meta["diversity_temperature_override_count"], 1)
        self.assertEqual(meta["structural_temperature_override_count"], 1)
        for retry_prompt in prompts[1:3]:
            forbidden = retry_prompt["retry_variation_contract"][
                "forbidden_opening_prefixes"
            ]
            self.assertIn("i will also", forbidden)
            self.assertLessEqual(
                len(forbidden),
                subject.FORMAL_DIALOGUE_FORBIDDEN_OPENING_PREFIX_LIMIT,
            )

    def test_formal_prefix_manifest_blocks_run123_run125_tail_copy_regression(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        run_tail_texts = [
            "A compliance review now separates the data policy questions from the ordinary scheduling notes so the group can proceed without conflating them.",
            "Before the next meeting, a risk matrix will distinguish operational concerns from the unrelated appointment details discussed elsewhere.",
            "Because the communication protocol still needs approval, the project lead will review its scope without restating any earlier decision.",
            "Once the final memo is ready, the coordinator will circulate only newly verified logistics and leave every previous statement unchanged.",
        ]

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            payload = self._formal_segment_payload(prompt)
            if prompt["segment"]["segment_index"] == 0:
                for turn, text in zip(
                    payload["appended_dialogue_turns_segment"][-4:],
                    run_tail_texts,
                ):
                    turn["text"] = text
            return payload, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            side_effect=respond,
        ), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        second_prompt_json = json.dumps(prompts[1], ensure_ascii=False, sort_keys=True)
        for text in run_tail_texts:
            self.assertNotIn(text, second_prompt_json)
        accepted_context = prompts[1]["frozen_prefix"]["accepted_prefix_context"]
        self.assertGreater(
            accepted_context["accepted_prefix_digest"]["full_turn_count"],
            0,
        )
        self.assertFalse(
            accepted_context["accepted_prefix_digest"]["accepted_turn_text_included"]
        )
        self.assertNotIn("frozen_prefix_tail", prompts[1]["frozen_prefix"])
        self.assertTrue(meta["prefix_hash_chain_verified"])

    def test_run126_accepted_prefix_audit_exempts_only_registered_current_claim_paths(self) -> None:
        marker = "E6US7CBT5J"
        prior_claims = [
            "Regarding ProjectX, the access scope is Internal Team Only. The speaker remains unsure about this personal impression",
            "Regarding ProjectX, the access scope is Restricted to Division A. A later decision canceled that plan",
        ]
        current_claim = (
            "Regarding ProjectX, the access scope is Limited to External Partners. "
            "It is scheduled but not active yet"
        )
        accepted_turns = [
            {
                "speaker": "Coordinator",
                "text": f"{claim}. Additional accepted body text remains hidden.",
            }
            for claim in prior_claims
        ]
        prompt = {
            "frozen_graph": {
                "fact_plan": {
                    "assigned_events": [
                        {
                            "marker": marker,
                            "dialogue_support_claim": current_claim,
                        }
                    ]
                }
            },
            "segment": {
                "assigned_anchor_realization_checklist": [
                    {
                        "marker": marker,
                        "natural_support_claim_to_include": current_claim,
                    }
                ],
                "marker_to_exact_natural_support_claim": {
                    marker: current_claim
                },
                "required_marked_turn_shapes": [
                    {
                        "marker": marker,
                        "text_must_start_with": current_claim,
                        "support_quotes_must_equal": {marker: current_claim},
                    }
                ],
            },
        }
        provenance = [
            {
                "marker": marker,
                "event_instance_id": "event-current",
                "validated_claim": current_claim,
                "claim_source": "canonical_entity_slot_value_status",
            },
            {
                "marker": "PRIOR-MARKER",
                "event_instance_id": "event-prior",
                "validated_claim": prior_claims[0],
                "claim_source": "canonical_entity_slot_value_status",
            },
        ]

        without_exemption = (
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                prompt,
                accepted_turns,
            )
        )
        self.assertFalse(without_exemption["pass"])
        self.assertEqual(
            set(without_exemption["exposed_prompt_paths"]),
            {
                "$.frozen_graph.fact_plan.assigned_events[0].dialogue_support_claim",
                "$.segment.assigned_anchor_realization_checklist[0].natural_support_claim_to_include",
                f"$.segment.marker_to_exact_natural_support_claim.{marker}",
                f"$.segment.required_marked_turn_shapes[0].support_quotes_must_equal.{marker}",
                "$.segment.required_marked_turn_shapes[0].text_must_start_with",
            },
        )

        exemptions = subject.formal_current_exact_claim_audit_exempt_spans(
            prompt,
            provenance,
            {marker},
        )
        self.assertEqual(len(exemptions), 5)
        self.assertTrue(
            all(
                exemption["schema_version"]
                == subject.FORMAL_CURRENT_EXACT_CLAIM_EXEMPTION_VERSION
                and exemption["marker"] == marker
                and exemption["span_text"] == current_claim
                and len(exemption["allowed_prompt_paths"]) == 1
                for exemption in exemptions
            )
        )
        audit = subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
            prompt,
            accepted_turns,
            local_exempt_spans=exemptions,
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["accepted_prefix_verbatim_exposure_count"], 0)
        self.assertEqual(
            audit["registered_path_scoped_exact_claim_exempt_span_count"],
            5,
        )
        self.assertEqual(
            set(audit["local_exempted_prompt_paths"]),
            set(without_exemption["exposed_prompt_paths"]),
        )
        self.assertEqual(
            audit["local_exemption_reasons"],
            ["current_graph_registered_exact_claim"],
        )

    def test_current_claim_exemption_rejects_raw_tail_arbitrary_path_and_tamper(self) -> None:
        marker = "CURRENT-MARKER"
        prior_claim = (
            "Regarding ProjectX, the access scope is Internal Team Only. "
            "This accepted sentence has a private continuation."
        )
        current_claim = (
            "Regarding ProjectX, the access scope is Limited to External Partners. "
            "It is scheduled but not active yet"
        )
        accepted_turns = [{"speaker": "A", "text": prior_claim}]
        prompt = {
            "frozen_graph": {
                "fact_plan": {
                    "assigned_events": [
                        {
                            "marker": marker,
                            "dialogue_support_claim": current_claim,
                        }
                    ]
                }
            },
            "segment": {
                "assigned_anchor_realization_checklist": [],
                "marker_to_exact_natural_support_claim": {},
                "required_marked_turn_shapes": [],
            },
        }
        provenance = [
            {
                "marker": marker,
                "event_instance_id": "event-current",
                "validated_claim": current_claim,
                "claim_source": "canonical_entity_slot_value_status",
            }
        ]
        exemptions = subject.formal_current_exact_claim_audit_exempt_spans(
            prompt,
            provenance,
            {marker},
        )
        self.assertEqual(len(exemptions), 1)
        self.assertTrue(
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                prompt,
                accepted_turns,
                local_exempt_spans=exemptions,
            )["pass"]
        )

        raw_tail_prompt = copy.deepcopy(prompt)
        raw_tail_prompt["frozen_prefix_tail"] = [{"text": prior_claim}]
        raw_tail_audit = (
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                raw_tail_prompt,
                accepted_turns,
                local_exempt_spans=(
                    subject.formal_current_exact_claim_audit_exempt_spans(
                        raw_tail_prompt,
                        provenance,
                        {marker},
                    )
                ),
            )
        )
        self.assertFalse(raw_tail_audit["pass"])
        self.assertIn(
            "$.frozen_prefix_tail[0].text",
            raw_tail_audit["exposed_prompt_paths"],
        )

        arbitrary_prompt = copy.deepcopy(prompt)
        arbitrary_prompt["unregistered_field"] = current_claim
        arbitrary_audit = (
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                arbitrary_prompt,
                accepted_turns,
                local_exempt_spans=(
                    subject.formal_current_exact_claim_audit_exempt_spans(
                        arbitrary_prompt,
                        provenance,
                        {marker},
                    )
                ),
            )
        )
        self.assertFalse(arbitrary_audit["pass"])
        self.assertIn("$.unregistered_field", arbitrary_audit["exposed_prompt_paths"])

        bad_hash = copy.deepcopy(exemptions)
        bad_hash[0]["registered_claim_sha256"] = "0" * 64
        self.assertFalse(
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                prompt,
                accepted_turns,
                local_exempt_spans=bad_hash,
            )["pass"]
        )
        bad_path = copy.deepcopy(exemptions)
        bad_path[0]["allowed_prompt_paths"] = ["$.unregistered_field"]
        self.assertFalse(
            subject.audit_formal_accepted_prefix_prompt_verbatim_exposure(
                prompt,
                accepted_turns,
                local_exempt_spans=bad_path,
            )["pass"]
        )

    def test_run126_common_entity_slot_claims_pass_full_segment_state_machine(self) -> None:
        plan = self._formal_segment_plan(event_count=4)
        values = (
            "Internal Team Only",
            "Restricted to Division A",
            "Limited to External Partners",
            "Public Access Granted",
        )
        for event, value in zip(plan["events"], values):
            event.update(
                {
                    "entity": "ProjectX",
                    "slot": "access scope",
                    "value": value,
                    "dialogue_value": value,
                    "dialogue_support_claim": (
                        f"Regarding ProjectX, the access scope is {value}"
                    ),
                }
            )
        prompts = []

        def respond(_args, prompt, **_kwargs):
            prompts.append(copy.deepcopy(prompt))
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            side_effect=respond,
        ), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        later_audits = [
            (record.get("meta") or {}).get("accepted_prefix_prompt_audit") or {}
            for record in meta["segment_records"]
            if int(record.get("segment_index") or 0) > 0
        ]
        self.assertTrue(later_audits)
        self.assertTrue(all(audit.get("pass") is True for audit in later_audits))
        self.assertTrue(
            any(
                int(
                    audit.get(
                        "registered_path_scoped_exact_claim_exempt_span_count"
                    )
                    or 0
                )
                > 0
                and int(
                    audit.get(
                        "local_exempted_accepted_prefix_substring_count"
                    )
                    or 0
                )
                > 0
                for audit in later_audits
            )
        )
        self.assertTrue(meta["prefix_hash_chain_verified"])

    def test_formal_late_segment_retry_remains_api_authored(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        segment_two_calls = 0

        def respond(_args, prompt, **_kwargs):
            nonlocal segment_two_calls
            prompts.append(copy.deepcopy(prompt))
            payload = self._formal_segment_payload(prompt)
            if prompt["segment"]["segment_index"] == 2:
                segment_two_calls += 1
                if segment_two_calls <= 3:
                    payload["appended_dialogue_turns_segment"].pop()
            return payload, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            side_effect=respond,
        ), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=lambda turns, _tokenizer_path=None: (len(turns) * 100, "test"),
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=5,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertIsNotNone(proposal)
        self.assertTrue(meta["used"])
        self.assertEqual(segment_two_calls, 4)
        self.assertEqual(meta["segment_api_call_count"], len(prompts))
        self.assertEqual(
            [
                prompt["segment"]["segment_validation_attempt_limit"]
                for prompt in prompts
                if prompt["segment"]["segment_index"] == 2
            ],
            [4, 4, 4, 4],
        )
        self.assertFalse(
            prompts[-1]["segment"].get("local_evidence_compiler", {}).get("enabled")
        )
        self.assertTrue(meta["prefix_hash_chain_verified"])
        generation_summary = subject._empty_summary()
        subject._record_api_meta_retry_diagnostics(generation_summary, meta)
        self.assertEqual(
            generation_summary["formal_dialogue_late_local_retry_count"],
            0,
        )

    def test_formal_outer_retry_context_is_compact_and_changes_generation_id(self) -> None:
        secret = (
            "This complete failed generated turn must never be replayed into an outer retry prompt."
        )
        raw_context = {
            "previous_error": "dialogue API did not return a valid JSON proposal",
            "dialogue_meta": {
                "error_type": "formal_dialogue_segment_failed",
                "failed_segment_index": 0,
                "segment_api_call_count": 11,
                "last_failure": {
                    "reason": "segment_exact_duplicate_turn",
                    "segment_index": 0,
                    "segment_attempt": 3,
                    "retry_generation_id": "old-retry-id",
                    "forbidden_opening_prefixes": [
                        "i think it",
                        "i think it is",
                    ],
                },
                "segment_records": [{"payload": {"text": secret}}],
            },
        }
        compact = subject.compact_formal_dialogue_retry_context(raw_context)
        self.assertEqual(
            compact["schema_version"],
            subject.FORMAL_DIALOGUE_COMPACT_RETRY_CONTEXT_VERSION,
        )
        self.assertFalse(compact["raw_failure_text_included"])
        self.assertNotIn(secret, json.dumps(compact, ensure_ascii=False))

        first_prompts = []
        second_prompts = []

        def run(dialogue_attempt_index, prior, captured):
            def respond(_args, prompt, **_kwargs):
                captured.append(copy.deepcopy(prompt))
                return self._formal_segment_payload(prompt), {
                    "used": True,
                    "attempts": 1,
                }

            with mock.patch.object(
                subject,
                "call_raw_json_api",
                side_effect=respond,
            ), mock.patch.object(
                subject,
                "dialogue_tokens_for_turns",
                side_effect=lambda turns, _tokenizer_path=None: (
                    len(turns) * 100,
                    "test",
                ),
            ):
                return subject.call_api_formal_segmented_dialogue_expansion(
                    self._formal_segment_plan(),
                    seed_row(),
                    {"qa_operation_answer_type_diversification": True},
                    SimpleNamespace(
                        seed=20260821,
                        challenger_temperature=0.25,
                        challenger_max_tokens=8192,
                        max_llm_retries=0,
                        formal_dialogue_max_segments=5,
                        formal_dialogue_segment_target_tokens=400,
                    ),
                    parent_context={"diagnostics": {}},
                    parent_dialogue_source="dialogue.turns",
                    parent_turn_count=2,
                    min_added_turns=8,
                    min_added_tokens=600,
                    dialogue_attempt_index=dialogue_attempt_index,
                    prior_failure_context=prior,
                )

        first_proposal, first_meta = run(1, None, first_prompts)
        second_proposal, second_meta = run(2, compact, second_prompts)
        self.assertIsNotNone(first_proposal)
        self.assertIsNotNone(second_proposal)
        self.assertTrue(first_meta["used"])
        self.assertTrue(second_meta["used"])
        self.assertEqual(
            first_prompts[0]["frozen_prefix"]["prefix_hash"],
            second_prompts[0]["frozen_prefix"]["prefix_hash"],
        )
        self.assertNotEqual(
            first_prompts[0]["retry_variation_contract"]["retry_generation_id"],
            second_prompts[0]["retry_variation_contract"]["retry_generation_id"],
        )
        self.assertEqual(second_prompts[0]["outer_dialogue_retry_context"], compact)
        self.assertNotIn(secret, json.dumps(second_prompts[0], ensure_ascii=False))
        self.assertEqual(
            second_prompts[0]["retry_variation_contract"][
                "forbidden_opening_prefixes"
            ],
            ["i think it", "i think it is"],
        )

    def test_formal_segmented_dialogue_retry_receives_exact_claim_repair_contract(self) -> None:
        plan = self._formal_segment_plan()
        prompts = []
        system_prompts = []
        first_call = True

        def respond(_args, prompt, **kwargs):
            nonlocal first_call
            prompts.append(copy.deepcopy(prompt))
            system_prompts.append(kwargs.get("system_prompt"))
            if first_call:
                first_call = False
                invalid_payload = self._formal_segment_payload(prompt)
                for turn in invalid_payload["appended_dialogue_turns_segment"]:
                    markers = turn.get("alignment_anchors") or []
                    if not markers:
                        continue
                    marker = markers[0]
                    turn["text"] = (
                        "We discussed the routing decision for "
                        f"{marker} without repeating its exact wording."
                    )
                    turn["support_quotes"] = {
                        marker: "I left my notebook by the lamp."
                    }
                return invalid_payload, {"used": True, "attempts": 1}
            return self._formal_segment_payload(prompt), {
                "used": True,
                "attempts": 1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }

        def fixed_tokens(turns, _tokenizer_path=None):
            return len(turns) * 100, "test"

        args = SimpleNamespace(
            solver_tokenizer=None,
            tokenizer_path=None,
            base_model=None,
            challenger_max_tokens=8192,
            max_llm_retries=1,
            formal_dialogue_max_segments=5,
            formal_dialogue_segment_target_tokens=400,
        )
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=fixed_tokens,
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                args,
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=600,
            )

        self.assertTrue(meta["used"])
        self.assertIsNotNone(proposal)
        first_mapping = prompts[0]["segment"]["marker_to_exact_natural_support_claim"]
        self.assertTrue(first_mapping)
        for marker, claim in first_mapping.items():
            self.assertEqual(
                claim,
                next(
                    item["natural_support_claim_to_include"]
                    for item in prompts[0]["segment"]["assigned_anchor_realization_checklist"]
                    if item["marker"] == marker
                ),
            )
        self.assertTrue(
            prompts[0]["segment"]["exact_claim_realization_contract"]
            ["new_turn_text_must_equal_exact_claim"]
        )
        self.assertEqual(
            prompts[1]["previous_segment_validation_error"]["reason"],
            "unsupported_segment_alignment",
        )
        failed_markers = prompts[1]["previous_segment_validation_error"]["unsupported_markers"]
        repair = prompts[1]["segment_alignment_repair_contract"]
        self.assertEqual(repair["failed_markers"], failed_markers)
        for failed_marker in failed_markers:
            self.assertEqual(
                repair["failed_marker_to_exact_natural_support_claim"][failed_marker],
                first_mapping[failed_marker],
            )
        self.assertIn("verbatim", system_prompts[0])
        self.assertIn("Never use parent", system_prompts[0])

    def test_formal_segment_retry_contract_handles_prefix_copy_and_missing_payload(self) -> None:
        prefix_contract = subject._formal_segment_retry_contract(
            {
                "reason": "frozen_prefix_turn_repeated",
                "turn_indices": [3, 7],
            },
            [],
        )
        payload_contract = subject._formal_segment_retry_contract(
            {"reason": "segment_payload_missing"},
            [],
        )

        self.assertTrue(prefix_contract["regenerate_entire_uncommitted_segment"])
        self.assertEqual(prefix_contract["repeated_turn_indices"], [3, 7])
        self.assertIn("completely new segment", " ".join(prefix_contract["required_repairs"]))
        self.assertTrue(payload_contract["regenerate_entire_uncommitted_segment"])
        self.assertIn(
            "appended_dialogue_turns_segment",
            " ".join(payload_contract["required_repairs"]),
        )
        self.assertEqual(subject.FORMAL_DIALOGUE_PREFIX_TAIL_TURNS, 4)
        self.assertEqual(subject.FORMAL_DIALOGUE_MIN_SEGMENT_VALIDATION_ATTEMPTS, 3)
        self.assertEqual(subject.FORMAL_DIALOGUE_LATE_RETRY_START_SEGMENT_INDEX, 2)
        self.assertEqual(subject.FORMAL_DIALOGUE_LATE_SEGMENT_VALIDATION_ATTEMPTS, 4)

    def test_formal_segmented_dialogue_fails_closed_without_local_padding(self) -> None:
        plan = self._formal_segment_plan()

        def respond(_args, prompt, **_kwargs):
            return self._formal_segment_payload(prompt), {"used": True, "attempts": 1}

        def very_short_tokens(turns, _tokenizer_path=None):
            return len(turns) * 10, "test"

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond), mock.patch.object(
            subject,
            "dialogue_tokens_for_turns",
            side_effect=very_short_tokens,
        ):
            proposal, meta = subject.call_api_formal_segmented_dialogue_expansion(
                plan,
                seed_row(),
                {"qa_operation_answer_type_diversification": True},
                SimpleNamespace(
                    challenger_max_tokens=8192,
                    max_llm_retries=0,
                    formal_dialogue_max_segments=3,
                    formal_dialogue_segment_target_tokens=400,
                ),
                parent_context={"diagnostics": {}},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=4000,
            )

        self.assertIsNone(proposal)
        self.assertFalse(meta["used"])
        self.assertEqual(meta["error_type"], "formal_dialogue_segment_failed")
        self.assertEqual(meta["failed_segment_index"], 0)
        self.assertEqual(meta["last_failure"]["reason"], "segment_token_shortfall")
        self.assertEqual(meta["last_failure"]["segment_token_count"], 120)
        self.assertEqual(meta["last_failure"]["minimum_segment_tokens"], 192)

    def test_formal_accept_summary_exposes_segment_hash_and_final_qa_counts(self) -> None:
        summary = subject._empty_summary()
        sample = {
            "metadata": {
                "formal_mining_api_native_dialogue_only": True,
                "dialogue_local_padding_used": False,
            },
            "native_contract": {
                "formal_mining_segmented_dialogue": True,
                "formal_mining_prefix_hash_chain_verified": True,
                "formal_mining_accepted_prefix_prompt_audit_verified": True,
                "formal_mining_accepted_prefix_verbatim_exposure_count": 0,
                "formal_mining_qa_generated_after_frozen_alignment": True,
                "formal_mining_marker_scheduler_verified": True,
                "formal_mining_turn_duplicate_audit_verified": True,
                "formal_mining_turn_duplicate_audited_turn_count": 16,
                "formal_mining_turn_duplicate_exact_pair_count": 0,
                "formal_mining_turn_duplicate_near_pair_count": 0,
                "formal_mining_turn_duplicate_phrase_pair_count": 0,
                "formal_mining_opening_diversity_audit_verified": True,
                "formal_mining_opening_diversity_audited_turn_count": 16,
                "formal_mining_opening_diversity_short_body_count": 2,
                "formal_mining_opening_diversity_concentrated_prefix_count": 0,
                "formal_mining_marker_assignment_segment_count": 4,
                "formal_mining_marker_max_assigned_per_segment": 2,
            },
            "qa": {},
        }
        subject._update_summary_for_accept(
            summary,
            sample,
            {"events": []},
            {"coverage_pass": True},
        )

        self.assertEqual(summary["formal_mining_segmented_dialogue_accept_count"], 1)
        self.assertEqual(summary["formal_mining_prefix_hash_chain_verified_count"], 1)
        self.assertEqual(
            summary[
                "formal_mining_accepted_prefix_prompt_audit_verified_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "formal_mining_accepted_prefix_verbatim_exposure_count_sum"
            ],
            0,
        )
        self.assertEqual(summary["formal_mining_final_qa_after_frozen_alignment_count"], 1)
        self.assertEqual(summary["formal_mining_marker_scheduler_verified_count"], 1)
        self.assertEqual(summary["formal_mining_turn_duplicate_audit_verified_count"], 1)
        self.assertEqual(summary["formal_mining_turn_duplicate_audited_turn_count_sum"], 16)
        self.assertEqual(
            summary["formal_mining_opening_diversity_audit_verified_count"], 1
        )
        self.assertEqual(
            summary["formal_mining_opening_diversity_audited_turn_count_sum"], 16
        )
        self.assertEqual(
            summary["formal_mining_opening_diversity_short_body_count_sum"], 2
        )
        self.assertEqual(
            summary["formal_mining_opening_diversity_audit_version"],
            subject.FORMAL_SEGMENT_OPENING_DIVERSITY_AUDIT_VERSION,
        )
        self.assertEqual(
            summary["formal_mining_turn_duplicate_audit_version"],
            subject.FORMAL_SEGMENT_TURN_DUPLICATE_AUDIT_VERSION,
        )
        self.assertEqual(summary["formal_mining_marker_assignment_segment_count_sum"], 4)
        self.assertEqual(summary["formal_mining_marker_max_assigned_per_segment"], 2)

    def test_formal_stage64_candidate_rejects_deterministic_scaffold_fallback(self) -> None:
        events = []
        marked_turns = []
        anchor_to_event = {}
        for index in range(4):
            marker = f"MARKER-{index}"
            event_id = f"event-{index}"
            claim = f"Maya selected route {index} for the delivery."
            events.append(
                {
                    "event_instance_id": event_id,
                    "entity": "Maya",
                    "slot": f"delivery_route_{index}",
                    "value": f"route {index}",
                    "status": "active",
                    "required_for_dialogue": True,
                }
            )
            anchor_to_event[marker] = event_id
            marked_turns.extend(
                [
                    {
                        "speaker": "Maya",
                        "text": claim,
                        "alignment_anchors": [marker],
                        "support_quotes": {marker: claim},
                    },
                    {"speaker": "Noah", "text": f"We can discuss detail {index} before moving on."},
                ]
            )

        candidate = subject.validate_appended_dialogue_candidate(
            appended_dialogue_marked=marked_turns,
            event_alignment_proposal=[],
            current_event_plan={"events": events, "graph_snapshot_hash": "graph"},
            anchor_to_event=anchor_to_event,
            parent_dialogue_turns=[],
            known_anchors=list(anchor_to_event),
            sample_index=0,
            min_added_turns=1,
            min_added_tokens=0,
            require_evidence_dispersion=False,
            deterministic_scaffold_used=True,
            formal_mining=True,
        )

        self.assertFalse(candidate["pass"])
        self.assertFalse(candidate["formal_mining_eligible"])
        self.assertEqual(candidate["reason"], "deterministic_scaffold_not_mining_eligible")

    def test_formal_stage64_length_guard_never_locally_pads_api_dialogue(self) -> None:
        marker = "MARKER-ONLY"
        event_id = "event-only"
        claim = "Maya selected the north route for the delivery."
        candidate = subject.validate_appended_dialogue_candidate(
            appended_dialogue_marked=[
                {
                    "speaker": "Maya",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                }
            ],
            event_alignment_proposal=[],
            current_event_plan={
                "events": [
                    {
                        "event_instance_id": event_id,
                        "entity": "Maya",
                        "slot": "delivery_route",
                        "value": "north route",
                        "status": "active",
                        "required_for_dialogue": True,
                    }
                ],
                "graph_snapshot_hash": "graph",
            },
            anchor_to_event={marker: event_id},
            parent_dialogue_turns=[],
            known_anchors=[marker],
            sample_index=0,
            min_added_turns=8,
            min_added_tokens=4000,
            fallback_min_added_tokens=32,
            emergency_min_added_tokens=16,
            formal_mining=True,
        )

        self.assertFalse(candidate["pass"])
        self.assertEqual(candidate["reason"], "length")
        self.assertEqual(len(candidate["final_dialogue"]["turns"]), 1)
        metadata = candidate["length_metadata"]
        self.assertEqual(metadata["api_appended_turn_count"], 1)
        self.assertFalse(metadata["dialogue_generic_padding_used"])
        self.assertFalse(metadata["dialogue_length_padding_used"])
        self.assertFalse(metadata["dialogue_local_padding_used"])
        self.assertTrue(metadata["formal_mining_api_native_dialogue_only"])
        self.assertFalse(metadata.get("length_guard_fallback_used", False))

        rejected_shortfall = subject.validate_appended_dialogue_candidate(
            appended_dialogue_marked=[
                {
                    "speaker": "Maya",
                    "text": claim,
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: claim},
                }
            ],
            event_alignment_proposal=[],
            current_event_plan={
                "events": [
                    {
                        "event_instance_id": event_id,
                        "entity": "Maya",
                        "slot": "delivery_route",
                        "value": "north route",
                        "status": "active",
                        "required_for_dialogue": True,
                    }
                ],
                "graph_snapshot_hash": "graph",
            },
            anchor_to_event={marker: event_id},
            parent_dialogue_turns=[],
            known_anchors=[marker],
            sample_index=0,
            min_added_turns=1,
            min_added_tokens=4000,
            formal_mining=True,
            allow_length_shortfall=True,
        )
        self.assertFalse(rejected_shortfall["pass"])
        self.assertEqual(rejected_shortfall["reason"], "length")
        rejected_metadata = rejected_shortfall["length_metadata"]
        self.assertFalse(rejected_metadata["length_guard_pass"])
        self.assertFalse(rejected_metadata.get("length_guard_target_pass", False))
        self.assertFalse(rejected_metadata.get("length_guard_exemption_used", False))

    def test_formal_stage64_length_shortfall_policy_is_fail_closed_at_4000(self) -> None:
        self.assertTrue(
            subject._formal_length_shortfall_must_reject(True, 4000)
        )
        self.assertTrue(
            subject._formal_length_shortfall_must_reject(True, 4096)
        )
        self.assertFalse(
            subject._formal_length_shortfall_must_reject(False, 4000)
        )
        self.assertFalse(
            subject._formal_length_shortfall_must_reject(True, 3999)
        )

    def test_long_form_fact_chunks_stop_after_all_required_events(self) -> None:
        checklist = [
            {"event_instance_id": f"event-{index}"}
            for index in range(8)
        ]

        selected = [
            subject._long_form_chunk_fact_event_ids(
                checklist,
                chunk_index=chunk_index,
                chunk_count=31,
            )
            for chunk_index in range(10)
        ]

        self.assertEqual(
            selected[:8],
            [[f"event-{index}"] for index in range(8)],
        )
        self.assertEqual(selected[8:], [[], []])

    def test_visible_claim_transport_retry_budget_has_one_retry_minimum(self) -> None:
        self.assertEqual(
            subject._visible_claim_transport_retry_budget(
                SimpleNamespace(max_llm_retries=0)
            ),
            1,
        )
        self.assertEqual(
            subject._visible_claim_transport_retry_budget(
                SimpleNamespace(max_llm_retries=4)
            ),
            4,
        )
        self.assertEqual(
            subject._visible_claim_transport_retry_budget(SimpleNamespace()),
            1,
        )

    def test_visible_claim_append_repair_passes_bounded_transport_retry(self) -> None:
        plan = {
            "events": [
                {
                    "event_instance_id": "event-1",
                    "dialogue_support_claim": "Maya selected the north route.",
                }
            ]
        }
        captured: dict = {}

        def fake_call(_args, _prompt, **kwargs):
            captured.update(kwargs)
            return {
                "appended_dialogue_turns": [
                    {
                        "speaker": "Maya",
                        "text": "Maya selected the north route.",
                        "alignment_anchors": [],
                        "support_quotes": {},
                    }
                ]
            }, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=fake_call):
            proposal, meta = subject.call_api_visible_claim_append_repair(
                plan,
                ["event-1"],
                SimpleNamespace(max_llm_retries=0),
                total_turn_count=1,
            )

        self.assertIsInstance(proposal, dict)
        self.assertTrue(meta["used"])
        self.assertEqual(captured["call_kind"], "api_visible_claim_append_repair")
        self.assertEqual(captured["max_retries_override"], 1)
        self.assertEqual(captured["temperature_override"], 0.0)

    def test_formal_visible_claim_alignment_never_inserts_local_text(self) -> None:
        marker = "opaque-anchor"
        event_id = "event-visible-claim"
        claim = "Maya selected the north route for the delivery."
        candidate = subject.validate_appended_dialogue_candidate(
            appended_dialogue_marked=[
                {
                    "speaker": "Maya",
                    "text": claim,
                    "alignment_anchors": [],
                    "support_quotes": {},
                }
            ],
            event_alignment_proposal=[],
            current_event_plan={
                "events": [
                    {
                        "event_instance_id": event_id,
                        "alignment_anchor": marker,
                        "entity": "Maya",
                        "slot": "delivery_route",
                        "value": "north route",
                        "status": "active",
                        "dialogue_support_claim": claim,
                        "required_for_dialogue": True,
                    }
                ],
                "graph_snapshot_hash": "graph",
            },
            anchor_to_event={marker: event_id},
            parent_dialogue_turns=[],
            known_anchors=[marker],
            sample_index=0,
            min_added_turns=1,
            min_added_tokens=0,
            formal_mining=True,
        )

        self.assertTrue(candidate["pass"])
        self.assertEqual(
            candidate["alignment_result"]["current_alignment"][event_id][
                "support_quotes"
            ],
            [claim],
        )
        self.assertFalse(candidate["length_metadata"]["dialogue_local_padding_used"])

    def test_stage64_dialogue_prompt_omits_solver_visible_scaffold(self) -> None:
        event_plan = {
            "question_type": "single_choice",
            "events": [
                {
                    "event_instance_id": "event-1",
                    "fact_id": "fact-1",
                    "alignment_anchor": "MARKER-ONE",
                    "entity": "Maya",
                    "slot": "delivery_route",
                    "value": "north road",
                    "dialogue_value": "north road",
                    "dialogue_support_claim": "Maya chose the north road for the delivery.",
                    "status": "active",
                    "required_for_dialogue": True,
                }
            ],
        }
        policy = stage3_policy("relation_lookup", "harder_boundary")
        policy["qa_operation_answer_type_diversification"] = True
        captured = {}

        def fake_call(_args, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"appended_dialogue_turns": [], "event_alignment_proposal": [], "qa_proposal": {}}, {"used": True}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=fake_call):
            subject.call_api_graph_conditioned_dialogue_expansion(
                event_plan,
                seed_row(),
                {},
                policy,
                SimpleNamespace(
                    solver_tokenizer=None,
                    tokenizer_path=None,
                    base_model=None,
                    challenger_max_tokens=2048,
                    parent_context_tail_turns=12,
                ),
                parent_context={
                    "parent_dialogue_digest": {},
                    "selected_relevant_parent_turns": [],
                    "active_superseded_state_summary": {},
                    "diagnostics": {},
                },
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=14,
                min_added_tokens=512,
            )

        prompt = captured["prompt"]
        self.assertNotIn("sanitized_dialogue_fact_plan", prompt)
        self.assertNotIn("natural_dialogue_scaffold", prompt)
        self.assertEqual(
            prompt["required_facts"],
            [
                {
                    "evidence_id": "MARKER-ONE",
                    "natural_fact": "Maya chose the north road for the delivery",
                }
            ],
        )
        self.assertTrue(prompt["append_contract"]["parent_is_immutable"])
        self.assertTrue(prompt["append_contract"]["local_text_insertion_forbidden"])
        requirements = " ".join(prompt["requirements"])
        self.assertIn("Naturally state every required_fact", requirements)
        self.assertIn("early, middle, and later", requirements)
        self.assertNotIn("fact-table scaffold", requirements)

    def test_length_continuation_reuses_only_generated_tail_not_graph(self) -> None:
        captured: dict = {}

        def fake_call(_args, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {
                "appended_dialogue_turns": [
                    {
                        "speaker": "Noah",
                        "text": "I will document the next steps carefully so the team can follow the same plan tomorrow.",
                    }
                ]
            }, {"used": True, "attempts": 1}

        prefix = [
            {
                "speaker": "Maya",
                "text": "The volunteers should use the north entrance after lunch.",
                "alignment_anchors": ["CURRENT-ANCHOR"],
                "support_quotes": {"CURRENT-ANCHOR": "north entrance"},
            }
        ]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=fake_call):
            payload, meta = subject.call_api_graph_conditioned_dialogue_continuation(
                prefix,
                SimpleNamespace(challenger_max_tokens=2048),
                required_additional_tokens=1266,
                parent_context={
                    "parent_dialogue_digest": {"topic": "volunteer routing"},
                    "selected_relevant_parent_turns": [],
                    "diagnostics": {},
                },
            )

        self.assertEqual(len(payload["appended_dialogue_turns"]), 1)
        self.assertEqual(meta["continuation_required_additional_tokens"], 1266)
        self.assertEqual(meta["continuation_target_additional_tokens"], 800)
        self.assertEqual(meta["continuation_minimum_turns"], 16)
        self.assertEqual(captured["kwargs"]["call_kind"], "api_dialogue_length_continuation")
        self.assertEqual(captured["kwargs"]["required_keys"], ("appended_dialogue_turns",))
        prompt = captured["prompt"]
        self.assertNotIn("existing_appended_dialogue_tail", prompt)
        self.assertTrue(
            prompt["existing_appended_dialogue_prefix"]["text_omitted_to_prevent_replay"]
        )
        self.assertNotIn("sanitized_dialogue_fact_plan", prompt)
        self.assertTrue(prompt["continuation_length_contract"]["graph_regeneration_forbidden"])
        self.assertTrue(prompt["continuation_length_contract"]["prefix_rewrite_forbidden"])

    def test_semantic_graph_plan_uses_natural_dialogue_claims(self) -> None:
        policy = stage3_policy("relation_lookup")
        policy["qa_operation_answer_type_diversification"] = True
        plan = subject.normalize_current_event_plan(
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": "stage3:S0000:semantic_graph_v2",
                "parent_graph_id": "stage3_parent:stage2-parent-1",
                "events": [
                    {
                        "event_id": "E1",
                        "entity": "Maya",
                        "slot": "access_scope",
                        "value": "Full Access",
                        "status": "active",
                        "time_index": 1,
                    },
                    {
                        "event_id": "E2",
                        "entity": "Noah",
                        "slot": "destination",
                        "value": "Harbor Office",
                        "status": "report",
                        "time_index": 2,
                    },
                    {
                        "event_id": "E3",
                        "entity": "Iris",
                        "slot": "appointment_date",
                        "value": "2026-08-03",
                        "status": "scope_limited",
                        "time_index": 3,
                    },
                    {
                        "event_id": "E4",
                        "entity": "Maya",
                        "slot": "membership_status",
                        "value": "Pending Approval",
                        "status": "delayed_not_active",
                        "time_index": 4,
                    },
                ],
            },
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=policy,
            sample_index=0,
            round_id="stage3",
        )

        claims = [event["dialogue_support_claim"] for event in plan["events"]]
        self.assertTrue(all("Mark this fact" not in claim for claim in claims))
        self.assertTrue(all("access_scope" not in claim for claim in claims))
        self.assertIn("Maya's Access level is Full Access", claims[0])
        self.assertIn("reported", claims[1])
        self.assertIn("Only authorized people", claims[2])
        self.assertIn("delayed and not active yet", claims[3])

    def test_membership_and_cause_support_claims_preserve_relation_truth(self) -> None:
        count_events = []
        for index, (member, included) in enumerate(
            (("Maya", True), ("Noah", False), ("Ava", True), ("Liam", False)),
            start=1,
        ):
            count_events.append(
                {
                    "event_id": f"E{index}",
                    "entity": "Writers Guild",
                    "slot": "candidate_member",
                    "value": member,
                    "status": "active",
                    "time_index": index,
                    "set_membership": {
                        "set_id": "weekend workshop roster",
                        "condition": "submitted a complete draft",
                        "included": included,
                    },
                }
            )
        count_plan = subject.normalize_current_event_plan(
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": "count-graph",
                "operation": "count",
                "events": count_events,
            },
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("count"),
            sample_index=0,
            round_id="stage3",
        )
        count_claims = [
            event["dialogue_support_claim"] for event in count_plan["events"]
        ]
        normalized_count_claims = [claim.casefold() for claim in count_claims]
        self.assertTrue(
            all("writers guild" in claim for claim in normalized_count_claims)
        )
        self.assertTrue(
            all("weekend workshop roster" in claim for claim in normalized_count_claims)
        )
        self.assertTrue(
            all("submitted a complete draft" in claim for claim in normalized_count_claims)
        )
        self.assertIn("is a member of", count_claims[0])
        self.assertIn("is not a member of", count_claims[1])

        cause_plan = subject.normalize_current_event_plan(
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": "cause-graph",
                "operation": "cause_consequence",
                "events": [
                    {
                        "event_id": "E1",
                        "entity": "Storm alert",
                        "slot": "trigger",
                        "value": "severe weather",
                        "status": "active",
                        "time_index": 1,
                    },
                    {
                        "event_id": "E2",
                        "entity": "Workshop",
                        "slot": "replacement_host",
                        "value": "Maya",
                        "status": "active",
                        "time_index": 2,
                        "depends_on": ["E1"],
                    },
                    {
                        "event_id": "E3",
                        "entity": "Train delay",
                        "slot": "traveler",
                        "value": "Ava",
                        "status": "active",
                        "time_index": 3,
                    },
                    {
                        "event_id": "E4",
                        "entity": "Panel",
                        "slot": "speaker",
                        "value": "Noah",
                        "status": "active",
                        "time_index": 4,
                    },
                ],
            },
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("cause_consequence"),
            sample_index=0,
            round_id="stage3",
        )
        effect_claim = cause_plan["events"][1]["dialogue_support_claim"]
        normalized_effect_claim = effect_claim.casefold()
        for surface in (
            "Storm alert",
            "severe weather",
            "causes",
            "Workshop",
            "Replacement host",
            "Maya",
        ):
            self.assertIn(surface.casefold(), normalized_effect_claim)

    def test_visibility_normalizer_preserves_typed_permissions_claims_and_times(self) -> None:
        scopes = [
            {"observer": "Nate", "permission": "view", "condition": "after 6 PM"},
            {"observer": "Joanna", "permission": "view", "condition": "after 6 PM"},
            {"observer": "Nate", "permission": "edit", "condition": "after 6 PM"},
        ]
        events = [
            {
                "event_id": "E1",
                "entity": "Nate's gaming room",
                "slot": "access_scope",
                "value": "private",
                "status": "active",
                "time_index": 1,
            }
        ]
        for index, scope in enumerate(scopes, start=2):
            events.append(
                {
                    "event_id": f"E{index}",
                    "entity": "Nate's gaming room",
                    "slot": "access_scope",
                    "value": "private",
                    "status": "active",
                    "time_index": index,
                    "global_time": f"2026-07-13T{index + 16:02d}:00:00Z",
                    "local_time": f"{index + 16:02d}:00",
                    "scope": copy.deepcopy(scope),
                }
            )

        plan = subject.normalize_current_event_plan(
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": "stage3:S0000:semantic_graph_v2",
                "parent_graph_id": "stage3-parent",
                "operation": "visibility",
                "events": events,
            },
            seed_row(),
            question_type="single_choice",
            attack_skill="permission_visibility_scope",
            policy=stage3_policy("visibility"),
            sample_index=0,
            round_id="stage3",
        )

        permissions = plan["events"][1:]
        self.assertTrue(
            all(
                event["event_type"] == subject.VISIBILITY_PERMISSION_EVENT_TYPE
                for event in permissions
            )
        )
        self.assertEqual(
            [
                {
                    field: event[field]
                    for field in subject.VISIBILITY_TYPED_FIELDS
                }
                for event in permissions
            ],
            [
                {
                    "owner": "Nate",
                    "resource": "gaming room",
                    "observer": scope["observer"],
                    "action": scope["permission"],
                    "condition": scope["condition"],
                    "access_class": "private",
                    "status": "active",
                }
                for scope in scopes
            ],
        )
        self.assertEqual(
            [event["scope"] for event in permissions],
            scopes,
        )
        claims = [event["dialogue_support_claim"] for event in permissions]
        self.assertEqual(len(set(claims)), len(permissions))
        for event, claim in zip(permissions, claims):
            normalized_claim = subject._visibility_gate_text_key(claim)
            for field in subject.VISIBILITY_TYPED_FIELDS[:-1]:
                self.assertIn(
                    subject._visibility_gate_text_key(event[field]),
                    normalized_claim,
                )

        gate = subject.normalized_visibility_pre_dialogue_gate(plan)
        self.assertTrue(gate["pass"], gate)
        self.assertEqual(gate["required_count"], 3)
        self.assertEqual(gate["unique_typed_signature_count"], 3)
        self.assertEqual(gate["unique_support_claim_count"], 3)

        dialogue_plan = subject.sanitize_event_plan_for_dialogue_api(plan)
        self.assertEqual(dialogue_plan["events"][1]["owner"], "Nate")
        self.assertEqual(
            dialogue_plan["events"][1]["global_time"],
            "2026-07-13T18:00:00Z",
        )
        compiled = subject.compile_graph_update_from_current_event_plan(
            plan,
            {},
            {
                "graph_snapshot_hash": "graph",
                "dialogue_hash": "dialogue",
                "alignment_hash": "alignment",
            },
        )
        compiled_permission = compiled["events"][1]
        self.assertEqual(compiled_permission["owner"], "Nate")
        self.assertEqual(compiled_permission["resource"], "gaming room")
        self.assertEqual(
            compiled_permission["global_time"], "2026-07-13T18:00:00Z"
        )
        self.assertEqual(compiled_permission["local_time"], "18:00")

    def test_visibility_pre_dialogue_gate_rejects_claim_and_value_collisions(self) -> None:
        base_events = [
            {
                "event_id": "E1",
                "entity": "Nate's gaming room",
                "slot": "access_scope",
                "value": "private",
                "status": "active",
                "time_index": 1,
            },
            *[
                {
                    "event_id": f"E{index}",
                    "entity": "Nate's gaming room",
                    "slot": "access_scope",
                    "value": "private",
                    "status": "active",
                    "time_index": index,
                    "scope": scope,
                }
                for index, scope in enumerate(
                    [
                        {"observer": "Nate", "permission": "view", "condition": "after 6 PM"},
                        {"observer": "Joanna", "permission": "view", "condition": "after 6 PM"},
                        {"observer": "Nate", "permission": "edit", "condition": "after 6 PM"},
                    ],
                    start=2,
                )
            ],
        ]
        plan = subject.normalize_current_event_plan(
            {"events": base_events},
            seed_row(),
            question_type="single_choice",
            attack_skill="permission_visibility_scope",
            policy=stage3_policy("visibility"),
            sample_index=0,
            round_id="stage3",
        )

        claim_collapse = copy.deepcopy(plan)
        claim_collapse["events"][2]["dialogue_support_claim"] = (
            claim_collapse["events"][1]["dialogue_support_claim"]
        )
        claim_collapse["events"][2]["alignment_hints"]["support_claim"] = (
            claim_collapse["events"][1]["dialogue_support_claim"]
        )
        claim_gate = subject.normalized_visibility_pre_dialogue_gate(
            claim_collapse
        )
        self.assertFalse(claim_gate["pass"])
        self.assertEqual(claim_gate["required_count"], 3)
        self.assertEqual(claim_gate["unique_typed_signature_count"], 3)
        self.assertEqual(claim_gate["unique_support_claim_count"], 2)
        self.assertIn(
            "normalized_visibility_cardinality_collapse",
            {issue["code"] for issue in claim_gate["issues"]},
        )
        with mock.patch.object(subject, "call_raw_json_api") as dialogue_api:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "collision gate failed before dialogue API",
            ):
                subject.call_api_graph_conditioned_dialogue_expansion(
                    claim_collapse,
                    seed_row(),
                    {},
                    stage3_policy("visibility"),
                    SimpleNamespace(),
                    parent_context={},
                    parent_dialogue_source="dialogue.turns",
                    parent_turn_count=2,
                    min_added_turns=8,
                    min_added_tokens=1024,
                )
        dialogue_api.assert_not_called()

        value_collision = copy.deepcopy(plan)
        first = value_collision["events"][1]
        second = value_collision["events"][2]
        for field in ("owner", "resource", "observer", "action", "condition", "status"):
            second[field] = first[field]
        second["scope"] = copy.deepcopy(first["scope"])
        second["value"] = "public"
        second["access_class"] = "public"
        second["dialogue_support_claim"] = subject._visibility_permission_support_claim(
            second
        )
        second["alignment_hints"]["support_claim"] = second[
            "dialogue_support_claim"
        ]
        value_gate = subject.normalized_visibility_pre_dialogue_gate(
            value_collision
        )
        self.assertFalse(value_gate["pass"])
        self.assertIn(
            "visibility_canonical_slot_access_class_collision",
            {issue["code"] for issue in value_gate["issues"]},
        )

        owner_missing = copy.deepcopy(plan)
        owner_missing["events"][1]["entity"] = "Archive"
        owner_missing["events"][1]["owner"] = ""
        owner_gate = subject.normalized_visibility_pre_dialogue_gate(
            owner_missing
        )
        self.assertFalse(owner_gate["pass"])
        typed_field_issue = next(
            issue
            for issue in owner_gate["issues"]
            if issue["code"] == "visibility_typed_fields_missing"
        )
        self.assertIn("owner", typed_field_issue["details"]["missing_fields"])

    def test_long_form_quote_normalizer_repairs_only_semantically_supported_turn(self) -> None:
        marker = "CURRENT-ANCHOR"
        event_id = "stage3:S0000:E01"
        plan = {
            "current_event_instance_to_anchor": {event_id: marker},
            "events": [
                {
                    "event_instance_id": event_id,
                    "alignment_anchor": marker,
                    "entity": "Maya",
                    "slot": "access_scope",
                    "value": "Full Access",
                    "dialogue_value": "Full Access",
                    "dialogue_support_claim": "Maya's access level is Full Access.",
                    "status": "active",
                }
            ],
        }
        proposal = {
            "appended_dialogue_turns": [
                {
                    "speaker": "Noah",
                    "text": "Maya has Full Access to the office, so the team can ask her for help.",
                    "alignment_anchors": [marker],
                    "support_quotes": {marker: "Maya's access level is Full Access."},
                }
            ],
            "qa_proposal": {},
        }

        normalized, diagnostics = subject.normalize_long_form_sidecar_support_quotes(
            proposal,
            anchor_to_event={marker: event_id},
            current_event_plan=plan,
        )

        self.assertTrue(diagnostics["used"])
        self.assertTrue(diagnostics["dialogue_content_preserved"])
        self.assertEqual(
            normalized["appended_dialogue_turns"][0]["support_quotes"][marker],
            proposal["appended_dialogue_turns"][0]["text"],
        )

    def test_long_form_evidence_compiler_places_each_fixed_fact_once(self) -> None:
        first_id, second_id = "stage3:S0000:E01", "stage3:S0000:E02"
        first_marker, second_marker = "MARKER-ONE", "MARKER-TWO"
        plan = {
            "current_event_instance_to_anchor": {
                first_id: first_marker,
                second_id: second_marker,
            },
            "current_anchor_to_event_instance_id": {
                first_marker: first_id,
                second_marker: second_id,
            },
            "events": [
                {
                    "event_instance_id": first_id,
                    "alignment_anchor": first_marker,
                    "entity": "Maya",
                    "dialogue_support_claim": "Maya's access level is Full Access.",
                    "required_for_dialogue": True,
                },
                {
                    "event_instance_id": second_id,
                    "alignment_anchor": second_marker,
                    "entity": "Noah",
                    "dialogue_support_claim": "Noah's destination was reported as Harbor Office, but nobody has confirmed it.",
                    "required_for_dialogue": True,
                },
            ],
        }
        evidence, diagnostics = subject.compile_long_form_local_evidence_turns(
            plan,
            [{"speaker": "Maya", "text": "Can we review the schedule?"}],
        )
        combined = subject.interleave_formal_local_evidence_turns(
            [
                {"speaker": "Maya", "text": "We can prepare the shared agenda."},
                {"speaker": "Noah", "text": "I will circulate the notes this evening."},
                {"speaker": "Maya", "text": "That gives everyone time to respond."},
            ],
            evidence,
        )

        self.assertTrue(diagnostics["pass"])
        self.assertEqual(diagnostics["compiled_marker_count"], 2)
        markers = [
            marker
            for turn in combined
            for marker in turn.get("alignment_anchors", [])
        ]
        self.assertEqual(markers.count(first_marker), 1)
        self.assertEqual(markers.count(second_marker), 1)
        for turn in combined:
            for marker, quote in (turn.get("support_quotes") or {}).items():
                self.assertEqual(quote, turn["text"])
                self.assertIn(marker, (turn.get("alignment_anchors") or []))

    def test_length_truncated_dialogue_recovers_complete_turn_prefix_only(self) -> None:
        first = {"speaker": "Maya", "text": "The schedule gives everyone enough time to prepare."}
        second = {"speaker": "Noah", "text": "I can send the final reminder after the meeting."}
        raw = (
            '{"appended_dialogue_turns": ['
            + json.dumps(first)
            + ","
            + json.dumps(second)
            + ',{"speaker":"Maya","text":"unfinished'
        )
        proposal, diagnostics = subject.recover_truncated_appended_dialogue_turns(
            {
                "errors": [
                    {
                        "finish_reason": "length",
                        "raw_response": raw,
                    }
                ]
            }
        )

        self.assertTrue(diagnostics["used"])
        self.assertTrue(diagnostics["discarded_incomplete_tail"])
        self.assertEqual(diagnostics["recovered_turn_count"], 2)
        self.assertEqual(proposal["appended_dialogue_turns"], [first, second])

    def test_compile_graph_update_preserves_semantic_optional_fields(self) -> None:
        plan = {
            "graph_snapshot_id": "stage3:child",
            "events": [
                {
                    "event_instance_id": "event-1",
                    "event_type": "state_change",
                    "entity": "Atlas project",
                    "slot": "active_member",
                    "value": "Maya",
                    "status": "active",
                    "time_index": 3,
                    "previous_value": "Noah",
                    "depends_on": ["event-0"],
                    "scope": {"team": "Atlas"},
                    "set_membership": {"set_id": "atlas_active_members", "included": True},
                    "evidence_turn_ids": ["untrusted:turn"],
                    "provenance": {"source": "api"},
                    "unsafe_internal_field": "drop-me",
                }
            ],
        }
        hashes = {
            "graph_snapshot_hash": "graph",
            "dialogue_hash": "dialogue",
            "alignment_hash": "alignment",
        }

        compiled = subject.compile_graph_update_from_current_event_plan(
            plan,
            {"event-1": {"turn_ids": ["D1:3"]}},
            hashes,
        )

        event = compiled["events"][0]
        self.assertEqual(event["time_index"], 3)
        self.assertEqual(event["previous_value"], "Noah")
        self.assertEqual(event["depends_on"], ["event-0"])
        self.assertEqual(event["scope"], {"team": "Atlas"})
        self.assertEqual(event["set_membership"], {"set_id": "atlas_active_members", "included": True})
        self.assertEqual(event["provenance"], {"source": "api"})
        self.assertEqual(event["evidence_turn_ids"], ["D1:3"])
        self.assertNotIn("unsafe_internal_field", event)

        clean = subject.build_clean_sample(
            sample_index=0,
            seed_row={"sample_id": "stage2-parent", "dialogue": {"turns": []}},
            current_event_plan=plan,
            final_dialogue={"turns": [], "dialogue_hash": "dialogue"},
            graph_update=compiled,
            qa={"question_type": "state_reconstruction", "question": "What changed?"},
            hashes=hashes,
            length_metadata={},
        )
        self.assertEqual(clean["graph_update"]["events"][0]["set_membership"]["set_id"], "atlas_active_members")

    def test_compile_graph_update_routes_events_by_status_before_event_type(self) -> None:
        statuses = [
            ("outdated", "state_change"),
            ("false", "restore"),
            ("canceled", "state_change"),
            ("delayed_not_active", "state_change"),
            ("active", "cancel"),
            ("restored", "restore"),
        ]
        plan = {
            "graph_snapshot_id": "stage3:status-first",
            "events": [
                {
                    "event_instance_id": f"event-{index}",
                    "event_type": event_type,
                    "entity": "Atlas project",
                    "slot": f"status_slot_{index}",
                    "value": f"value-{index}",
                    "status": status,
                    "time_index": index,
                }
                for index, (status, event_type) in enumerate(statuses)
            ],
        }
        hashes = {
            "graph_snapshot_hash": "graph",
            "dialogue_hash": "dialogue",
            "alignment_hash": "alignment",
        }
        alignment = {
            f"event-{index}": {"turn_ids": [f"D1:{index + 1}"]}
            for index in range(len(statuses))
        }

        compiled = subject.compile_graph_update_from_current_event_plan(plan, alignment, hashes)

        self.assertEqual(
            [row["event_instance_id"] for row in compiled["state_changes"]],
            ["event-4", "event-5"],
        )
        self.assertEqual(
            [row["event_instance_id"] for row in compiled["invalidations"]],
            ["event-0", "event-1", "event-2", "event-3"],
        )
        self.assertEqual(
            [row["event_instance_id"] for row in compiled["cancellations"]],
            ["event-2"],
        )
        self.assertEqual(
            [row["event_instance_id"] for row in compiled["delayed_activations"]],
            ["event-3"],
        )
        self.assertEqual(
            [row["event_instance_id"] for row in compiled["restorations"]],
            ["event-5"],
        )

    def test_graph_cache_key_changes_with_reasoning_contract_schema_v17(self) -> None:
        args = SimpleNamespace(
            min_length_delta_tokens=512,
            query_conditioned_graph_expansion=False,
        )
        policy = stage3_policy("temporal_state_update")
        current_key = subject.graph_cache_key_for_attempt(
            seed_row(),
            policy["planned_cell"],
            policy,
            args,
            sample_index=0,
            round_id="stage3",
        )
        with mock.patch.object(subject, "GRAPH_CACHE_SCHEMA_VERSION", "semantic_graph_v16"):
            old_key = subject.graph_cache_key_for_attempt(
                seed_row(),
                policy["planned_cell"],
                policy,
                args,
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(subject.GRAPH_CACHE_SCHEMA_VERSION, "semantic_graph_v17")
        self.assertNotEqual(current_key, old_key)

    def test_graph_cache_key_is_bound_to_parent_dialogue_and_semantic_graph(self) -> None:
        args = SimpleNamespace(
            min_length_delta_tokens=4000,
            query_conditioned_graph_expansion=False,
        )
        policy = stage3_policy("comparison")
        parent = seed_row()
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "parent-graph",
            "parent_graph_id": "source",
            "events": [],
        }
        dialogue_changed = copy.deepcopy(parent)
        dialogue_changed["dialogue"]["turns"][0]["text"] = "A cleaned parent turn."
        graph_changed = copy.deepcopy(parent)
        graph_changed["semantic_parent_graph"]["graph_id"] = "parent-graph-cleaned"

        def key(row: dict) -> str:
            return subject.graph_cache_key_for_attempt(
                row,
                policy["planned_cell"],
                policy,
                args,
                sample_index=0,
                round_id="stage3",
            )

        original_key = key(parent)
        self.assertNotEqual(original_key, key(dialogue_changed))
        self.assertNotEqual(original_key, key(graph_changed))

    def test_semantic_graph_quality_forwards_target_for_every_operation(self) -> None:
        for operation in subject.STAGE64_QA_OPERATIONS:
            with self.subTest(operation=operation):
                target = {"qa_operation": operation, "sentinel": operation}
                with (
                    mock.patch.object(
                        subject.semantic_graph_v2,
                        "validate_graph_delta",
                        return_value=[],
                    ) as core_gate,
                    mock.patch.object(
                        subject.stage64_quality,
                        "graph_plan_quality_issues",
                        return_value=[],
                    ) as quality_gate,
                    mock.patch.object(
                        subject,
                        "_stage64_reasoning_graph_target_issues",
                        return_value=[],
                    ) as reasoning_gate,
                ):
                    self.assertEqual(
                        subject.semantic_graph_plan_quality_issues(
                            {"events": []},
                            operation,
                            target,
                        ),
                        [],
                    )
                self.assertIs(
                    core_gate.call_args.kwargs["reasoning_target"],
                    target,
                )
                self.assertIs(quality_gate.call_args.args[2], target)
                self.assertIs(reasoning_gate.call_args.args[1], target)

    def test_graph_cache_loader_rejects_v11_entries_after_status_contract_v12(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cache_key = "old-count-cache"
            (cache_dir / f"{cache_key}.json").write_text(
                json.dumps(
                    {
                        "cache_schema_version": "semantic_graph_v11",
                        "current_event_plan": {"events": []},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(subject.load_graph_cache_entry(cache_dir, cache_key))

    def test_stage64_difficulty_contracts_are_distinct_and_do_not_enable_legacy_planner(self) -> None:
        boundary_target, boundary = subject._stage64_prompt_difficulty_contract("boundary", "relation_lookup")
        harder_target, harder = subject._stage64_prompt_difficulty_contract("harder_boundary", "relation_lookup")
        hard_target, hard = subject._stage64_prompt_difficulty_contract("hard", "relation_lookup")

        self.assertEqual((boundary_target, harder_target, hard_target), ("boundary", "harder_boundary", "hard"))
        self.assertNotEqual(boundary["graph"], harder["graph"])
        self.assertNotEqual(harder["qa"], hard["qa"])
        self.assertIn("minimal-pair", harder["qa"])
        self.assertIn("at least two", harder["operation_qa"])
        self.assertTrue(all("boundary_aware_qa_planner" not in contract for contract in (boundary, harder, hard)))

    def test_solver_visible_gate_rejects_only_explicit_meta_question_shortcuts(self) -> None:
        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "generic question"):
            subject._assert_no_stage64_solver_visible_artifact(
                "question",
                "Which event confirms Nate's access to the trails?",
            )
        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "solver-visible artifact"):
            subject._assert_no_stage64_solver_visible_artifact(
                "question",
                "What is Nate's current access status after all updates?",
            )

        subject._assert_no_stage64_solver_visible_artifact(
            "question",
            "After Joanna corrected the route, which trails could Nate access?",
        )

    def test_clean_sample_preserves_stage3_lineage_group(self) -> None:
        parent = {
            "sample_id": "stage3-parent-1",
            "lineage_group": "conv-41",
            "seed_source": {"lineage_group": "conv-41"},
        }
        sample = subject.build_clean_sample(
            sample_index=0,
            seed_row=parent,
            current_event_plan={"graph_snapshot_hash": "graph", "attack_skill": "bridge", "metadata": {}},
            final_dialogue={
                "turns": [{"turn_id": "D1:1", "speaker": "Ada", "text": "The key is here."}],
                "dialogue_hash": "dialogue",
            },
            graph_update={"events": []},
            qa={
                "question_type": "single_choice",
                "question": "Which statement is current?",
                "options": [
                    {"id": "A", "text": "The key is here.", "is_gold": True},
                    {"id": "B", "text": "The key is elsewhere.", "is_gold": False},
                ],
                "gold": ["A"],
                "evidence_turn_ids": ["D1:1"],
                "graph_validation_pass": True,
            },
            hashes={
                "graph_snapshot_hash": "graph",
                "dialogue_hash": "dialogue",
                "alignment_hash": "alignment",
            },
            length_metadata={},
        )

        self.assertEqual(sample["lineage_group"], "conv-41")
        self.assertEqual(sample["metadata"]["lineage_group"], "conv-41")
        self.assertEqual(sample["parent_sample_id"], "stage3-parent-1")

    def test_semantic_mode_keeps_short_values_without_padding(self) -> None:
        plan = subject.normalize_current_event_plan(
            {"events": raw_events(5)},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy(),
            sample_index=0,
            round_id="stage3",
        )

        self.assertEqual(plan["schema_version"], "semantic_graph_v2")
        self.assertEqual([event["value"] for event in plan["events"]], ["3", "Yes", "Paris", "yoga", "violin"])
        self.assertTrue(all(event["status"] == "active" for event in plan["events"]))
        self.assertTrue(plan["metadata"]["legacy_fallback_disabled"])

    def test_semantic_comparison_normalization_preserves_shared_time_index(self) -> None:
        events = raw_events(6)
        events[0].update(
            {
                "event_id": "E1",
                "entity": "Caroline",
                "slot": "support_group_attendance",
                "value": 11,
                "time_index": 1,
            }
        )
        events[1].update(
            {
                "event_id": "E2",
                "entity": "Melanie",
                "slot": "support_group_attendance",
                "value": 8,
                "time_index": 1,
            }
        )

        plan = subject.normalize_current_event_plan(
            {"events": events},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("comparison"),
            sample_index=0,
            round_id="stage3",
        )

        self.assertEqual([event["time_index"] for event in plan["events"][:2]], [1, 1])
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "comparison"), [])

    def test_semantic_bridge_normalization_preserves_same_surface_non_latin_events(self) -> None:
        events = [
            {
                "event_id": " E1 ",
                "entity": "李",
                "slot": "偏好",
                "value": "茶",
                "status": "active",
                "time_index": 1,
            },
            {
                "event_id": "E2",
                "entity": "李",
                "slot": "偏好",
                "value": "茶",
                "status": "active",
                "time_index": 2,
            },
            {
                "event_id": "E3",
                "entity": "工作室",
                "slot": "准入依据",
                "value": "两份独立记录",
                "status": "active",
                "time_index": 3,
                "depends_on": [" E1 ", "E2"],
            },
            {
                "event_id": "E4",
                "entity": "王",
                "slot": "偏好",
                "value": "咖啡",
                "status": "active",
                "time_index": 4,
            },
            {
                "event_id": "E5",
                "entity": "陈",
                "slot": "城市",
                "value": "苏州",
                "status": "active",
                "time_index": 5,
            },
            {
                "event_id": "E6",
                "entity": "赵",
                "slot": "乐器",
                "value": "钢琴",
                "status": "active",
                "time_index": 6,
            },
        ]

        plan = subject.normalize_current_event_plan(
            {"events": events},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("bridge"),
            sample_index=0,
            round_id="stage3",
        )

        self.assertEqual(len(plan["events"]), 6)
        by_origin = {event["origin_event_id"]: event for event in plan["events"]}
        self.assertEqual(
            set(by_origin["E3"]["depends_on"]),
            {by_origin["E1"]["event_id"], by_origin["E2"]["event_id"]},
        )
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "bridge"), [])

    def test_semantic_mode_rejects_too_few_events_instead_of_fallback(self) -> None:
        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "enough graph events"):
            subject.normalize_current_event_plan(
                {"events": raw_events(3)},
                seed_row(),
                question_type="single_choice",
                attack_skill="conditional_plan",
                policy=stage3_policy(),
                sample_index=0,
                round_id="stage3",
            )

    def test_stage3_seed_prompt_omits_legacy_graph_and_qa(self) -> None:
        row = seed_row()
        row["graph_update"]["events"][1]["alignment_hints"] = {
            "support_claim": "I left my notebook by the lamp."
        }
        row["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "parent-graph-1",
            "parent_graph_id": "stage2-graph-1",
            "events": row["graph_update"]["events"],
        }
        payload = subject.compact_seed_for_prompt(row, semantic_graph_v2=True)

        self.assertNotIn("graph_update", payload)
        self.assertNotIn("qa_gold", payload)
        self.assertNotIn("dialogue_turns", payload)
        self.assertTrue(payload["parent_dialogue_text_omitted"])
        self.assertFalse(payload["parent_dialogue_content"]["parent_turn_text_included"])
        self.assertEqual(len(payload["parent_dialogue_content"]["content_sha256"]), 64)
        self.assertEqual(payload["dialogue_structure"]["turn_count"], 2)
        self.assertEqual(payload["dialogue_structure"]["speaker_turn_counts"], {"Maya": 1, "Noah": 1})
        self.assertEqual(payload["dialogue_structure"]["recent_speaker_order"], ["Maya", "Noah"])
        self.assertNotIn("turn_speaker_sequence", payload["dialogue_structure"])
        self.assertTrue(payload["legacy_graph_omitted"])
        self.assertEqual(payload["semantic_parent_graph"]["graph_id"], "parent-graph-1")
        self.assertEqual(payload["semantic_parent_graph"]["parent_graph_id"], "stage2-graph-1")
        parent_event = payload["semantic_parent_graph"]["events"][0]
        self.assertEqual(parent_event["event_id"], "clean-1")
        self.assertEqual(parent_event["entity_anchor"], "Maya")
        self.assertTrue(parent_event["fact_content_omitted"])
        self.assertNotIn("slot", parent_event)
        self.assertNotIn("value", parent_event)
        self.assertNotIn("evidence_turn_ids", parent_event)
        self.assertNotIn("alignment_hints", parent_event)

    def test_stage3_seed_prompt_preserves_ten_canonical_parent_events(self) -> None:
        row = seed_row()
        row["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "parent-graph-ten-events",
            "events": [
                {
                    "event_id": f"P{index}",
                    "entity": f"Person {index}",
                    "slot": f"canonical_slot_{index}",
                    "value": f"value-{index}",
                    "status": "active",
                    "time_index": index,
                }
                for index in range(1, 11)
            ],
        }

        payload = subject.compact_seed_for_prompt(row, semantic_graph_v2=True)
        parent_events = payload["semantic_parent_graph"]["events"]

        self.assertEqual(len(parent_events), 10)
        self.assertEqual(parent_events[-1]["event_id"], "P10")

    def test_chained_qa_context_prefers_canonical_parent_v3_events(self) -> None:
        row = seed_row()
        row["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "stage3-parent-v3-graph",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Maya",
                    "slot": "item_beside_lamp",
                    "value": "notebook",
                    "status": "active",
                    "time_index": 1,
                    "evidence_turn_ids": ["D1:1"],
                },
                {
                    "event_id": "P2",
                    "entity": "Noah",
                    "slot": "reminder_state",
                    "value": "recorded",
                    "status": "active",
                    "time_index": 2,
                    "evidence_turn_ids": ["D1:2"],
                },
            ],
        }

        events = subject._parent_context_events(row)

        self.assertEqual([event["event_id"] for event in events], ["P1", "P2"])
        self.assertTrue(all(event["_context_source"] == "parent" for event in events))
        self.assertNotIn("legacy-1", {event["event_id"] for event in events})

    def test_stage3_seed_prompt_omits_parent_fact_atoms_but_keeps_structural_anchors(self) -> None:
        row = seed_row()
        long_value = "Andrew used to be a Data Scientist before moving into financial analysis"
        row["dialogue"]["turns"][0]["text"] = long_value
        row["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "parent-graph-long-value",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Andrew",
                    "slot": "job_history",
                    "value": long_value,
                    "previous_value": long_value,
                    "status": "outdated",
                    "depends_on": ["parent-event-with-a-long-stable-identifier-0001"],
                    "alignment_hints": {"support_claim": long_value},
                }
            ],
        }

        payload = subject.compact_seed_for_prompt(row, semantic_graph_v2=True)
        event = payload["semantic_parent_graph"]["events"][0]

        self.assertEqual(event["event_id"], "P1")
        self.assertEqual(event["entity_anchor"], "Andrew")
        self.assertTrue(event["fact_content_omitted"])
        self.assertEqual(
            event["depends_on_event_ids"],
            ["parent-event-with-a-long-stable-identifier-0001"],
        )
        self.assertNotIn("slot", event)
        self.assertNotIn("value", event)
        self.assertNotIn(long_value, json.dumps(payload, ensure_ascii=False))
        self.assertFalse(payload["semantic_parent_event_atoms_readable"])
        self.assertTrue(payload["semantic_parent_entity_anchors_readable"])
        self.assertTrue(payload["parent_dialogue_text_omitted"])
        audit = subject.audit_graph_parent_prompt_verbatim_exposure(
            {"seed": payload}, row
        )
        self.assertTrue(audit["pass"])
        self.assertGreater(audit["atomic_structured_exemption_count"], 0)

    def test_stage3_legacy_alignment_prompt_is_counts_only(self) -> None:
        legacy = {
            "legacy_alignment": {"raw_qa": {"expected_text": "Q: secret? A: answer"}},
            "legacy_high_confidence_count": 2,
            "legacy_low_confidence_count": 1,
            "legacy_missing_ref_total": 3,
        }

        payload = subject.legacy_alignment_for_graph_prompt(legacy, semantic_graph_v2=True)

        self.assertNotIn("legacy_alignment", payload)
        self.assertEqual(payload["legacy_high_confidence_count"], 2)
        self.assertTrue(payload["legacy_content_omitted"])

    def test_graph_parent_prompt_audit_rejects_parent_utterance_fields_and_substrings(self) -> None:
        row = seed_row()
        parent_sentence = (
            "I left my notebook by the lamp before the evening planning meeting."
        )
        row["dialogue"]["turns"][0]["text"] = parent_sentence
        seed = subject.compact_seed_for_prompt(row, semantic_graph_v2=True)
        seed["dialogue_turns"] = [{"text": parent_sentence}]
        prompt = {
            "seed": seed,
            "hidden_query_conditioned_objective": {
                "copied_context": parent_sentence
            },
        }

        audit = subject.audit_graph_parent_prompt_verbatim_exposure(prompt, row)

        self.assertFalse(audit["pass"])
        self.assertIn("$.seed.dialogue_turns", audit["forbidden_seed_lexical_paths"])
        self.assertGreater(audit["parent_verbatim_exposure_count"], 0)
        self.assertFalse(audit["raw_parent_text_recorded"])

    def test_temporal_comparison_count_graph_prompts_are_parent_text_free(self) -> None:
        comparison_events = [
            {"event_id": "E1", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1},
            {"event_id": "E2", "entity": "Bob", "slot": "score", "value": 13, "status": "active", "time_index": 1},
            {"event_id": "E3", "entity": "Alice", "slot": "score", "value": 9, "status": "outdated", "time_index": 2},
            {"event_id": "E4", "entity": "Bob", "slot": "score", "value": 11, "status": "outdated", "time_index": 2},
            {"event_id": "E5", "entity": "Evan", "slot": "team", "value": "Atlas", "status": "active", "time_index": 4},
            {"event_id": "E6", "entity": "Faye", "slot": "route", "value": "West", "status": "active", "time_index": 5},
            {"event_id": "E7", "entity": "Gina", "slot": "shelf", "value": "Upper", "status": "active", "time_index": 6},
            {"event_id": "E8", "entity": "Hana", "slot": "badge", "value": "Silver", "status": "active", "time_index": 7},
        ]
        cases = {
            "temporal_state_update": temporal_graph_payload(
                exact_temporal_generation_events()
            ),
            "comparison": comparison_graph_payload(comparison_events),
            "count": count_graph_payload(exact_count_generation_events()),
        }
        for operation, api_payload in cases.items():
            with self.subTest(operation=operation):
                captured: list[dict] = []

                def respond(_args, prompt, **_kwargs):
                    captured.append(prompt)
                    return copy.deepcopy(api_payload), {"used": True, "attempts": 1}

                with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
                    _plan, meta, _raw = subject.call_api_graph_expansion(
                        seed_row(),
                        {},
                        stage3_policy(operation),
                        SimpleNamespace(
                            query_conditioned_graph_expansion=False,
                            graph_expansion_max_tokens=4096,
                            graph_semantic_retries=0,
                        ),
                        random.Random(117),
                        sample_index=0,
                        round_id="stage3",
                    )

                self.assertEqual(len(captured), 1)
                prompt_json = json.dumps(captured[0], ensure_ascii=False)
                self.assertNotIn("I left my notebook by the lamp.", prompt_json)
                self.assertNotIn("dialogue_turns", captured[0]["seed"])
                self.assertEqual(captured[0]["seed"]["dialogue_structure"]["turn_count"], 2)
                self.assertIn("Create a novel child graph delta", " ".join(captured[0]["requirements"]))
                self.assertTrue(meta["graph_parent_prompt_audit"]["pass"])
                self.assertTrue(meta["graph_parent_novelty_gate"]["pass"])

    def test_normalization_failure_preserves_raw_api_graph(self) -> None:
        raw_payload = {"events": raw_events(3)}
        api_args = SimpleNamespace(
            query_conditioned_graph_expansion=False,
            graph_expansion_max_tokens=2048,
            graph_semantic_retries=0,
        )
        meta = {"used": True, "call_kind": "api_graph_expansion", "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", return_value=(raw_payload, meta)):
            with self.assertRaises(subject.NativeGraphDialogueReject) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy(),
                    api_args,
                    random.Random(7),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(caught.exception.api_graph_payload, raw_payload)
        self.assertEqual(caught.exception.api_graph_meta, meta)

    def test_api_graph_rejects_disconnected_parent_graph_id(self) -> None:
        parent = seed_row()
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "expected-parent",
            "parent_graph_id": "stage2-root",
            "events": [],
        }
        raw_payload = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "child-graph",
            "parent_graph_id": "wrong-parent",
            "operation": "relation_lookup",
            "events": [
                {
                    "event_id": f"event-{index}",
                    "entity": "Maya" if index < 3 else "Noah",
                    "slot": "item_location",
                    "value": value,
                    "status": "active" if index == 0 else "outdated",
                    "time_index": index,
                }
                for index, value in enumerate(("desk", "lamp", "shelf", "drawer", "cabinet"))
            ],
        }
        api_args = SimpleNamespace(
            query_conditioned_graph_expansion=False,
            graph_expansion_max_tokens=2048,
            graph_semantic_retries=0,
        )
        meta = {"used": True, "call_kind": "api_graph_expansion", "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", return_value=(raw_payload, meta)):
            with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "parent_graph_id_mismatch"):
                subject.call_api_graph_expansion(
                    parent,
                    {},
                    stage3_policy(),
                    api_args,
                    random.Random(11),
                    sample_index=0,
                    round_id="stage3",
                )

    def test_invalid_json_failure_preserves_full_unparsed_graph_response(self) -> None:
        raw_response = '{"schema_version":"semantic_graph_v2","events":['
        meta = {
            "used": False,
            "error_type": "invalid_json",
            "errors": [
                {
                    "error_type": "invalid_json",
                    "finish_reason": "length",
                    "raw_response": raw_response,
                }
            ],
        }
        with mock.patch.object(subject, "call_raw_json_api", return_value=(None, meta)):
            with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "invalid_json") as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy(),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=0,
                    ),
                    random.Random(13),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(
            caught.exception.api_graph_payload,
            {
                "parse_status": "unparsed_api_graph_response",
                "raw_responses": [raw_response],
            },
        )
        self.assertEqual(caught.exception.api_graph_meta, meta)

    def test_semantic_retry_sends_validator_feedback_and_accepts_correction(self) -> None:
        expected_parent = "stage3_parent:stage2-parent-1"
        expected_child = "stage3:S0000:semantic_graph_v2"

        def bridge_payload(with_dependencies: bool) -> dict:
            events = [
                {
                    "event_id": f"E{index}",
                    "entity": f"entity-{index}",
                    "slot": f"slot-{index}",
                    "value": f"value-{index}",
                    "status": "active",
                    "time_index": index,
                }
                for index in range(1, 9)
            ]
            if with_dependencies:
                events[0].update(
                    {
                        "entity": "Workshop roster",
                        "slot": "selected_person",
                        "value": "Maya",
                    }
                )
                events[1].update(
                    {
                        "entity": "Maya",
                        "slot": "festival_partner",
                        "value": "Noah",
                    }
                )
                events[2].update(
                    {
                        "entity": "Bridge closure",
                        "slot": "linked_record",
                        "value": "complete",
                        "depends_on": ["E1", "E2"],
                    }
                )
                events[3].update(
                    {
                        "entity": "Bridge closure",
                        "slot": "festival_partner",
                        "value": "Ava",
                    }
                )
                events[4].update(
                    {"entity": "Studio", "slot": "festival_partner", "value": "Liam"}
                )
                events[5].update(
                    {"entity": "Panel", "slot": "festival_partner", "value": "Zoe"}
                )
                events[6].update(
                    {"entity": "Route", "slot": "direction", "value": "west route"}
                )
                events[7].update(
                    {"entity": "Room", "slot": "setting", "value": "quiet room"}
                )
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": expected_child,
                "parent_graph_id": expected_parent,
                "operation": "bridge",
                "events": events,
            }

        first = bridge_payload(False)
        first["task"] = "echoed input specification"
        corrected = bridge_payload(True)
        api_args = SimpleNamespace(
            query_conditioned_graph_expansion=False,
            graph_expansion_max_tokens=4096,
            graph_semantic_retries=1,
        )
        calls: list[dict] = []
        system_prompts: list[str] = []

        def respond(_args, prompt, **kwargs):
            calls.append(prompt)
            system_prompts.append(kwargs.get("system_prompt", ""))
            payload = first if len(calls) == 1 else corrected
            return payload, {"used": True, "attempts": 1}

        policy = stage3_policy("bridge")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1,
            {"bridge": 1.0},
            {"bridge": {"Person": 1.0}},
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                policy,
                api_args,
                random.Random(17),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        self.assertEqual(len(meta["semantic_attempt_history"]), 1)
        self.assertIn("previous_semantic_validation_error", calls[1])
        self.assertEqual(
            calls[1]["invalid_graph_to_correct"],
            subject.sanitize_invalid_graph_for_semantic_retry(first),
        )
        self.assertNotIn("task", calls[1]["invalid_graph_to_correct"])
        self.assertTrue(meta["graph_parent_prompt_audit"]["pass"])
        bridge_layout = calls[0]["bridge_generation_layout_contract"]
        self.assertEqual(bridge_layout["total_event_count"], 8)
        self.assertEqual(
            bridge_layout["required_dependency_map"]["E3"],
            ["E1", "E2"],
        )
        self.assertEqual(
            bridge_layout["typed_hard_negative_array_indexes"],
            [3, 4, 5],
        )
        output_events = calls[0]["output_schema"]["events"]
        self.assertEqual(
            [event["event_id"] for event in output_events],
            [f"E{index}" for index in range(1, 9)],
        )
        self.assertEqual(output_events[2]["depends_on"], ["E1", "E2"])
        self.assertIn("proper name", output_events[1]["value"])
        self.assertTrue(
            all("proper name" in output_events[index]["value"] for index in (3, 4, 5))
        )
        self.assertIn(
            "bridge_generation_layout_contract",
            system_prompts[0],
        )
        self.assertTrue(any(issue["code"] == "bridge_missing_two_hop_dependency" for issue in calls[1]["previous_semantic_validation_error"]))
        self.assertTrue(any(issue["code"] == "forbidden_graph_output_key" for issue in calls[1]["previous_semantic_validation_error"]))
        bridge_constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(bridge_constraints["required_total_event_count"], 8)
        self.assertEqual(
            bridge_constraints["required_total_event_count_range"], [8, 8]
        )
        self.assertEqual(
            bridge_constraints["required_dependency_map"]["E3"],
            ["E1", "E2"],
        )
        self.assertIn(
            "events[0].value == events[1].entity",
            bridge_constraints["required_field_equalities"],
        )
        self.assertIn(
            "proper name",
            bridge_constraints["required_role_schema"]["E2"]["value_shape"],
        )
        self.assertTrue(bridge_constraints["forbidden_additional_dependency_edges"])
        self.assertIn("remove depends_on from every other event", calls[1]["correction_instruction"])
        self.assertIn("not a sentence or relation clause", calls[1]["correction_instruction"])
        self.assertEqual(len(plan["events"]), 8)

    def test_cause_retry_supplies_unique_edge_and_parent_novelty_constraints(self) -> None:
        expected_parent = "stage3_parent:stage2-parent-1"
        expected_child = "stage3:S0000:semantic_graph_v2"

        def cause_payload(extra_edges: bool) -> dict:
            events = [
                {"event_id": "E1", "entity": "Iris", "slot": "input_state", "value": "sealed", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Iris", "slot": "result_state", "value": "opened", "status": "active", "time_index": 2, "depends_on": ["E1"]},
                {"event_id": "E3", "entity": "Omar", "slot": "result_state", "value": "quiet", "status": "active", "time_index": 2},
                {"event_id": "E4", "entity": "Omar", "slot": "lamp_state", "value": "dim", "status": "active", "time_index": 4},
                {"event_id": "E5", "entity": "Pia", "slot": "door_state", "value": "closed", "status": "active", "time_index": 5},
                {"event_id": "E6", "entity": "Pia", "slot": "alarm_state", "value": "silent", "status": "active", "time_index": 6},
            ]
            if extra_edges:
                events[4]["depends_on"] = ["E3"]
                events[5]["depends_on"] = ["E4"]
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": expected_child,
                "parent_graph_id": expected_parent,
                "operation": "cause_consequence",
                "events": events,
            }

        first = cause_payload(True)
        corrected = cause_payload(False)
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return (first if len(calls) == 1 else corrected), {"used": True, "attempts": 1}

        policy = stage3_policy("cause_consequence")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1, {"cause_consequence": 1.0}, None
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                policy,
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(41),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_dependency_map"]["E2"], ["E1"])
        self.assertEqual(constraints["required_dependency_map"]["E6"], [])
        self.assertTrue(constraints["forbidden_additional_dependency_edges"])
        self.assertIn("remove depends_on from every event after the effect", calls[1]["correction_instruction"])

    def test_run117_parent_support_claim_is_rejected_then_sanitized_for_semantic_retry(self) -> None:
        parent = seed_row()
        copied_parent_sentence = (
            "Andrew used to be a Data Scientist before transitioning to a Financial Analyst."
        )
        parent["dialogue"]["turns"][0]["text"] = copied_parent_sentence
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "run117-parent-graph",
            "parent_graph_id": "stage2-root",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Andrew",
                    "slot": "job_title",
                    "value": "Data Scientist",
                    "status": "outdated",
                    "time_index": 1,
                    "alignment_hints": {"support_claim": copied_parent_sentence},
                }
            ],
        }
        first = temporal_graph_payload(exact_temporal_generation_events())
        corrected = copy.deepcopy(first)
        for payload in (first, corrected):
            payload["parent_graph_id"] = "run117-parent-graph"
        first["events"][0]["alignment_hints"] = {
            "support_claim": copied_parent_sentence
        }
        corrected["events"][0]["alignment_hints"] = {
            "support_claim": "John previously considered local politics as his main direction."
        }
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(first if len(calls) == 1 else corrected), {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, raw = subject.call_api_graph_expansion(
                parent,
                {},
                stage3_policy("temporal_state_update"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(1170),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(len(calls), 2)
        first_issue_codes = {
            issue["code"] for issue in meta["semantic_attempt_history"][0]["issues"]
        }
        self.assertIn("parent_support_claim_verbatim_reuse", first_issue_codes)
        self.assertNotIn(copied_parent_sentence, json.dumps(calls[0], ensure_ascii=False))
        self.assertNotIn(copied_parent_sentence, json.dumps(calls[1], ensure_ascii=False))
        self.assertNotIn(
            "alignment_hints", calls[1]["invalid_graph_to_correct"]["events"][0]
        )
        self.assertTrue(meta["graph_parent_prompt_audit"]["pass"])
        self.assertTrue(meta["graph_parent_novelty_gate"]["pass"])

    def test_run117_reconstructed_parent_atom_is_rejected_but_new_temporal_transition_is_allowed(self) -> None:
        parent = seed_row()
        parent_sentence = "Nate's gaming room has red and purple lighting for late sessions."
        parent["dialogue"]["turns"][0]["text"] = parent_sentence
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "run117-count-parent",
            "events": [
                {
                    "event_id": "P-count",
                    "entity": "Nate",
                    "slot": "gaming_room_lighting",
                    "value": "has red and purple lighting",
                    "status": "active",
                    "time_index": 1,
                }
            ],
        }
        copied = {
            "events": [
                {
                    "event_id": "E1",
                    "entity": "Nate",
                    "slot": "gaming_room_lighting",
                    "value": "has red and purple lighting",
                    "status": "active",
                    "time_index": 1,
                    "alignment_hints": {"support_claim": parent_sentence},
                }
            ]
        }

        copied_codes = {
            issue["code"]
            for issue in subject.graph_parent_novelty_issues(
                copied, parent, operation="count"
            )
        }
        self.assertEqual(
            copied_codes,
            {
                "parent_event_exact_semantic_reuse",
                "parent_dialogue_entity_value_semantic_reuse",
                "parent_support_claim_verbatim_reuse",
            },
        )

        slot_alias_copy = {
            "events": [
                {
                    "event_id": "E5",
                    "entity": "Nate",
                    "slot": "room_appearance",
                    "value": "red and purple lighting",
                    "status": "active",
                    "time_index": 5,
                    "alignment_hints": {
                        "support_claim": "Nate selected red and purple lighting."
                    },
                }
            ]
        }
        slot_alias_issues = subject.graph_parent_novelty_issues(
            slot_alias_copy,
            parent,
            operation="temporal_state_update",
        )
        self.assertEqual(
            {issue["code"] for issue in slot_alias_issues},
            {"parent_dialogue_entity_value_semantic_reuse"},
        )
        slot_alias_details = slot_alias_issues[0]["details"]
        self.assertEqual(slot_alias_details["child_event_id"], "E5")
        self.assertEqual(slot_alias_details["parent_turn_ids"], ["D1:1"])
        self.assertFalse(slot_alias_details["raw_parent_text_recorded"])
        self.assertNotIn(
            parent_sentence,
            json.dumps(slot_alias_issues, ensure_ascii=False),
        )

        temporal_update = copy.deepcopy(copied)
        temporal_update["events"][0]["value"] = "has amber and cyan lighting"
        temporal_update["events"][0]["status"] = "outdated"
        temporal_update["events"][0]["alignment_hints"]["support_claim"] = (
            "Nate previously tried amber and cyan lighting in that room."
        )
        self.assertEqual(
            subject.graph_parent_novelty_issues(
                temporal_update,
                parent,
                operation="temporal_state_update",
            ),
            [],
        )

    def test_run219_parent_dialogue_entity_value_collision_retries_before_dialogue(self) -> None:
        parent = seed_row()
        copied_parent_sentence = (
            "Nate's gaming room has red and purple lighting and he won a "
            "Street Fighter tournament."
        )
        parent["dialogue"]["turns"][0]["text"] = copied_parent_sentence
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "run219-parent-graph",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Nate",
                    "slot": "memory_fact",
                    "value": "won a local Street Fighter tournament last month",
                    "status": "active",
                    "time_index": 1,
                }
            ],
        }
        first = temporal_graph_payload(exact_temporal_generation_events())
        first["parent_graph_id"] = "run219-parent-graph"
        first["events"][4].update(
            {
                "entity": "Nate",
                "slot": "gaming_tournament",
                "value": "Street Fighter",
                "status": "active",
                "alignment_hints": {
                    "support_claim": "Nate's gaming tournament is Street Fighter."
                },
            }
        )
        corrected = copy.deepcopy(first)
        corrected["events"][4].update(
            {
                "entity": "Omar",
                "slot": "chess_tournament",
                "value": "rapid chess",
                "alignment_hints": {
                    "support_claim": "Omar entered a rapid chess tournament."
                },
            }
        )
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(first if len(calls) == 1 else corrected), {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, raw = subject.call_api_graph_expansion(
                parent,
                {},
                stage3_policy("temporal_state_update"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(2190),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(len(calls), 2)
        issue_codes = {
            issue["code"] for issue in meta["semantic_attempt_history"][0]["issues"]
        }
        self.assertIn("parent_dialogue_entity_value_semantic_reuse", issue_codes)
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["parent_collision_event_ids"], ["E5"])
        self.assertIn("E5", calls[1]["correction_instruction"])
        self.assertNotIn(copied_parent_sentence, json.dumps(calls[1], ensure_ascii=False))
        self.assertTrue(meta["graph_parent_prompt_audit"]["pass"])
        self.assertTrue(meta["graph_parent_novelty_gate"]["pass"])

    def test_semantic_retry_identical_invalid_payload_breaks_before_third_call(self) -> None:
        payload = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "stage3:S0000:semantic_graph_v2",
            "parent_graph_id": "stage3_parent:stage2-parent-1",
            "operation": "bridge",
            "events": [
                {"event_id": f"E{index}", "entity": f"entity-{index}", "slot": f"slot-{index}", "value": f"value-{index}", "status": "active", "time_index": index}
                for index in range(1, 7)
            ],
        }
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return copy.deepcopy(payload), {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "retry circuit breaker",
            ) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy("bridge"),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=3,
                    ),
                    random.Random(18),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(len(calls), 2)
        meta = caught.exception.api_graph_meta
        self.assertEqual(meta["error_type"], "semantic_retry_repeated_payload")
        self.assertEqual(len(meta["semantic_attempt_history"]), 2)
        diagnostic = meta["semantic_retry_circuit_breaker"]
        self.assertTrue(diagnostic["triggered"])
        self.assertEqual(diagnostic["observed_semantic_attempts"], [1, 2])
        self.assertEqual(diagnostic["prevented_next_semantic_attempt"], 3)
        self.assertEqual(caught.exception.api_graph_payload, payload)

    def test_semantic_retry_issue_view_removes_parent_value_prose(self) -> None:
        parent = seed_row()
        copied_parent_value = (
            "is considering starting a community project regarding infrastructure "
            "to improve the neighborhood."
        )
        parent["dialogue"]["turns"][0]["text"] = copied_parent_value
        issues = [
            {
                "code": "temporal_broken_previous_value_chain",
                "path": "$.events",
                "message": "dynamic prose must not be needed by the retry prompt",
                "details": {
                    "entity": "John",
                    "slot": "community service",
                    "broken_links": [
                        {
                            "event_id": "E4",
                            "predecessor_event_id": "E3",
                            "expected_previous_value": "another parent value",
                            "actual_previous_value": copied_parent_value,
                        }
                    ],
                },
            }
        ]
        original = copy.deepcopy(issues)

        compact = subject.sanitize_semantic_retry_prompt_issues(issues)

        serialized = json.dumps(compact, ensure_ascii=False)
        self.assertNotIn(copied_parent_value, serialized)
        self.assertNotIn("actual_previous_value", serialized)
        self.assertNotIn("expected_previous_value", serialized)
        self.assertEqual(compact[0]["code"], issues[0]["code"])
        self.assertEqual(compact[0]["path"], issues[0]["path"])
        broken_link = compact[0]["details"]["broken_links"][0]
        self.assertEqual(broken_link["event_id"], "E4")
        self.assertEqual(broken_link["predecessor_event_id"], "E3")
        audit = subject.audit_graph_parent_prompt_verbatim_exposure(
            {"previous_semantic_validation_error": compact},
            parent,
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["parent_verbatim_exposure_count"], 0)
        self.assertEqual(issues, original)

    def test_run124_unsupported_status_retry_is_path_specific_and_fail_closed(self) -> None:
        invalid_events = exact_temporal_generation_events()
        invalid_events[7]["status"] = "non-active"
        corrected_events = copy.deepcopy(invalid_events)
        corrected_events[7]["status"] = "report"
        payloads = [
            temporal_graph_payload(invalid_events),
            temporal_graph_payload(corrected_events),
        ]
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(payloads[len(calls) - 1]), {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("temporal_state_update"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(1241),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(meta["semantic_retry_count"], 1)
        self.assertEqual(raw["events"][7]["status"], "report")
        # Unsupported status literals are omitted from the machine correction
        # payload; the path-specific correction contract carries the value and
        # whitelist separately, preventing the model from copying it again.
        self.assertNotIn("status", calls[1]["invalid_graph_to_correct"]["events"][7])
        correction = calls[1]["unsupported_status_correction"]
        self.assertEqual(
            correction["invalid_paths"],
            [
                {
                    "path": "$.events[7].status",
                    "invalid_status_literal": "non-active",
                    "supported_status_literals": sorted(
                        subject.semantic_graph_v2.SUPPORTED_STATUSES
                    ),
                }
            ],
        )
        self.assertEqual(
            correction["non_active_status_literals"],
            sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
        )
        self.assertEqual(
            correction["forbidden_category_labels_as_status"],
            ["non-active"],
        )
        self.assertIn(
            "unsupported_status path",
            calls[1]["correction_instruction"],
        )
        first_issue_codes = {
            issue["code"] for issue in meta["semantic_attempt_history"][0]["issues"]
        }
        self.assertEqual(first_issue_codes, {"unsupported_status"})

    def test_run124_machine_equivalent_status_retries_ignore_support_prose(self) -> None:
        first_events = exact_temporal_generation_events()
        first_events[7]["status"] = "non-active"
        first_events[7]["alignment_hints"] = {
            "support_claim": "The client feedback remains unreviewed."
        }
        second_events = copy.deepcopy(first_events)
        second_events[7]["alignment_hints"] = {
            "support_claim": "No review has yet been recorded for the client feedback."
        }
        payloads = [
            temporal_graph_payload(first_events),
            temporal_graph_payload(second_events),
        ]
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(payloads[min(len(calls) - 1, 1)]), {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "retry circuit breaker",
            ) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy("temporal_state_update"),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=3,
                    ),
                    random.Random(1242),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(len(calls), 2)
        self.assertNotEqual(
            subject._semantic_retry_canonical_sha256(payloads[0]),
            subject._semantic_retry_canonical_sha256(payloads[1]),
        )
        self.assertEqual(
            subject._semantic_retry_canonical_sha256(
                subject._semantic_retry_machine_payload(payloads[0])
            ),
            subject._semantic_retry_canonical_sha256(
                subject._semantic_retry_machine_payload(payloads[1])
            ),
        )
        diagnostic = caught.exception.api_graph_meta[
            "semantic_retry_circuit_breaker"
        ]
        self.assertEqual(diagnostic["payload_hash_basis"], "sanitized_machine_fields")
        self.assertEqual(diagnostic["observed_semantic_attempts"], [1, 2])
        self.assertEqual(diagnostic["prevented_next_semantic_attempt"], 3)

    def test_semantic_retry_different_invalid_payload_can_reach_valid_correction(self) -> None:
        def bridge_payload(value_suffix: str, valid: bool = False) -> dict:
            events = [
                {"event_id": f"E{index}", "entity": f"entity-{index}", "slot": f"slot-{index}", "value": f"value-{index}-{value_suffix}", "status": "active", "time_index": index}
                for index in range(1, 7)
            ]
            if valid:
                events[0]["value"] = events[1]["entity"]
                events[1]["value"] = "Noah"
                events[2].update(
                    {
                        "entity": "Bridge closure",
                        "slot": "linked_record",
                        "value": "complete",
                        "depends_on": ["E1", "E2"],
                    }
                )
                events[3]["slot"] = "linked_record"
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": "stage3:S0000:semantic_graph_v2",
                "parent_graph_id": "stage3_parent:stage2-parent-1",
                "operation": "bridge",
                "events": events,
            }

        payloads = [bridge_payload("first"), bridge_payload("second"), bridge_payload("valid", valid=True)]
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return copy.deepcopy(payloads[len(calls) - 1]), {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("bridge"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=2,
                ),
                random.Random(181),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(raw, payloads[-1])
        self.assertEqual(meta["semantic_retry_count"], 2)
        self.assertNotIn("semantic_retry_circuit_breaker", meta)

    def test_harder_boundary_reaches_semantic_graph_prompt(self) -> None:
        expected_child = "stage3:S0000:semantic_graph_v2"
        payload = {
            "schema_version": "semantic_graph_v2",
            "graph_id": expected_child,
            "parent_graph_id": "stage3_parent:stage2-parent-1",
            "operation": "relation_lookup",
            "events": [
                {"event_id": "E1", "entity": "Maya record", "slot": "entity_identity", "value": "Maya", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "desk_item", "value": "red notebook", "status": "active", "time_index": 2, "depends_on": ["E1"]},
                {"event_id": "E3", "entity": "Maya", "slot": "desk_item", "value": "blue notebook", "status": "outdated", "time_index": 3},
                {"event_id": "E4", "entity": "Noah", "slot": "desk_item", "value": "red folder", "status": "active", "time_index": 4},
                {"event_id": "E5", "entity": "Maya", "slot": "shelf_item", "value": "brass key", "status": "active", "time_index": 5},
                {"event_id": "E6", "entity": "Li", "slot": "locker_item", "value": "silver badge", "status": "active", "time_index": 6},
                {"event_id": "E7", "entity": "Omar", "slot": "train_route", "value": "West", "status": "active", "time_index": 7},
                {"event_id": "E8", "entity": "Pia", "slot": "meeting_room", "value": "Cedar", "status": "active", "time_index": 8},
            ],
        }
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return payload, {"used": True, "attempts": 1}

        policy = stage3_policy("relation_lookup", "harder_boundary")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1,
            {"relation_lookup": 1.0},
            None,
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_graph_expansion(
                seed_row(),
                {},
                policy,
                SimpleNamespace(query_conditioned_graph_expansion=False, graph_expansion_max_tokens=4096, graph_semantic_retries=0),
                random.Random(23),
                sample_index=0,
                round_id="stage3",
            )

        prompt = captured[0]
        self.assertEqual(prompt["item_difficulty_target"], "harder_boundary")
        self.assertIn("minimal-pair", prompt["stage64_item_difficulty_contract"]["graph"])
        self.assertIn("without invoking or imitating a legacy QA planner", " ".join(prompt["requirements"]))
        self.assertNotIn("boundary_aware_qa_planner", prompt)
        reasoning_target = prompt["stage64_reasoning_target_contract"]
        self.assertEqual(
            reasoning_target["reasoning_subtype"],
            "entity_constrained_relation",
        )
        self.assertIn(
            "reasoning target is immutable",
            " ".join(prompt["requirements"]),
        )
        self.assertIn("exactly one dependency edge", " ".join(prompt["requirements"]))
        self.assertIn("preferred_answer_type='Person'", " ".join(prompt["requirements"]))

    def test_relation_target_rejects_extra_dependency_edges(self) -> None:
        target = subject._build_stage64_canonical_targets(
            1, {"relation_lookup": 1.0}, None
        )[0]["stage64_target"]
        payload = {
            "schema_version": "semantic_graph_v2",
            "operation": "relation_lookup",
            "events": [
                {"event_id": "E1", "entity": "Maya", "slot": "identity", "value": "Maya", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "desk_item", "value": "red notebook", "status": "active", "time_index": 2, "depends_on": ["E1"]},
                {"event_id": "E3", "entity": "Maya", "slot": "desk_item", "value": "blue notebook", "status": "outdated", "time_index": 3, "depends_on": ["E2"]},
                {"event_id": "E4", "entity": "Noah", "slot": "desk_item", "value": "red folder", "status": "active", "time_index": 4},
            ],
        }
        issues = subject._stage64_reasoning_graph_target_issues(payload, target)
        codes = {issue["code"] for issue in issues}
        self.assertIn("relation_reasoning_target_missing_binder_path", codes)

    def test_bridge_person_target_rejects_proposition_and_mixed_options(self) -> None:
        target = subject._build_stage64_canonical_targets(
            1,
            {"bridge": 1.0},
            {"bridge": {"Person": 1.0}},
        )[0]["stage64_target"]
        payload = {
            "schema_version": "semantic_graph_v2",
            "operation": "bridge",
            "events": [
                {"event_id": "E1", "entity": "Workshop roster", "slot": "selected_person", "value": "Maya", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "festival_partner", "value": "Noah", "status": "active", "time_index": 2},
                {"event_id": "E3", "entity": "Bridge closure", "slot": "linked_record", "value": "complete", "status": "active", "time_index": 3, "depends_on": ["E1", "E2"]},
                {"event_id": "E4", "entity": "Bridge closure", "slot": "festival_partner", "value": "Ava", "status": "active", "time_index": 4},
                {"event_id": "E5", "entity": "Studio", "slot": "festival_partner", "value": "Liam", "status": "active", "time_index": 5},
                {"event_id": "E6", "entity": "Panel", "slot": "festival_partner", "value": "Zoe", "status": "active", "time_index": 6},
                {"event_id": "E7", "entity": "Route", "slot": "direction", "value": "west route", "status": "active", "time_index": 7},
                {"event_id": "E8", "entity": "Room", "slot": "setting", "value": "quiet room", "status": "active", "time_index": 8},
            ],
        }
        valid_codes = {
            issue["code"]
            for issue in subject._stage64_reasoning_graph_target_issues(
                payload, target
            )
        }
        self.assertNotIn(
            "bridge_reasoning_target_answer_type_mismatch",
            valid_codes,
        )
        self.assertNotIn(
            "bridge_reasoning_target_insufficient_typed_distractors",
            valid_codes,
        )

        proposition = copy.deepcopy(payload)
        proposition["events"][1]["value"] = (
            "Joanna watched the trilogy based on Nate's recommendation"
        )
        proposition_codes = {
            issue["code"]
            for issue in subject._stage64_reasoning_graph_target_issues(
                proposition, target
            )
        }
        self.assertIn(
            "bridge_reasoning_target_answer_type_mismatch",
            proposition_codes,
        )

        mixed = copy.deepcopy(payload)
        mixed["events"][3]["value"] = "pending review"
        mixed_codes = {
            issue["code"]
            for issue in subject._stage64_reasoning_graph_target_issues(
                mixed, target
            )
        }
        self.assertIn(
            "bridge_reasoning_target_insufficient_typed_distractors",
            mixed_codes,
        )

    def test_relation_retry_rebuilds_negative_roles_and_parent_collisions(self) -> None:
        parent = seed_row()
        copied_claim = "The archive ledger says Maya is the bound record for this lookup."
        parent["dialogue"]["turns"][0]["text"] = copied_claim
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "relation-parent",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Archive record",
                    "slot": "identity",
                    "value": "Maya",
                    "status": "active",
                    "time_index": 1,
                }
            ],
        }

        def payload(events: list[dict]) -> dict:
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": "stage3:S0000:semantic_graph_v2",
                "parent_graph_id": "relation-parent",
                "operation": "relation_lookup",
                "events": events,
            }

        invalid = payload(
            [
                {"event_id": "E1", "entity": "Archive record", "slot": "identity", "value": "Maya", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "desk_item", "value": "red notebook", "status": "active", "time_index": 2, "depends_on": ["E1"]},
                {"event_id": "E3", "entity": "Archive record", "slot": "identity", "value": "Noah", "status": "outdated", "time_index": 3},
                {"event_id": "E4", "entity": "Noah", "slot": "shelf_item", "value": "red folder", "status": "active", "time_index": 4},
                {"event_id": "E5", "entity": "Iris", "slot": "office", "value": "North", "status": "active", "time_index": 5},
                {"event_id": "E6", "entity": "Omar", "slot": "route", "value": "West", "status": "active", "time_index": 6, "alignment_hints": {"support_claim": copied_claim}},
                {"event_id": "E7", "entity": "Pia", "slot": "badge", "value": "Silver", "status": "active", "time_index": 7},
                {"event_id": "E8", "entity": "Quinn", "slot": "room", "value": "Quiet", "status": "active", "time_index": 8},
            ]
        )
        corrected = payload(
            [
                {"event_id": "E1", "entity": "Maya record", "slot": "resolved_identity", "value": "Maya", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Maya", "slot": "desk_item", "value": "red notebook", "status": "active", "time_index": 2, "depends_on": ["E1"]},
                {"event_id": "E3", "entity": "Maya", "slot": "desk_item", "value": "blue notebook", "status": "outdated", "time_index": 3},
                {"event_id": "E4", "entity": "Noah", "slot": "desk_item", "value": "red folder", "status": "active", "time_index": 4},
                {"event_id": "E5", "entity": "Iris", "slot": "office", "value": "North", "status": "active", "time_index": 5},
                {"event_id": "E6", "entity": "Omar", "slot": "route", "value": "East", "status": "active", "time_index": 6, "alignment_hints": {"support_claim": "Omar selected the east route for the next delivery."}},
                {"event_id": "E7", "entity": "Pia", "slot": "badge", "value": "Silver", "status": "active", "time_index": 7},
                {"event_id": "E8", "entity": "Quinn", "slot": "room", "value": "Quiet", "status": "active", "time_index": 8},
            ]
        )
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(invalid if len(calls) == 1 else corrected), {
                "used": True,
                "attempts": 1,
            }

        policy = stage3_policy("relation_lookup")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1, {"relation_lookup": 1.0}, None
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                parent,
                {},
                policy,
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(1731),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        layout = calls[0]["relation_generation_layout_contract"]
        self.assertEqual(layout["binder_array_index"], 0)
        self.assertIn("events[2].entity equals events[1].entity", layout["same_entity_negative"]["entity_rule"])
        first_relation_issue = next(
            issue
            for issue in meta["semantic_attempt_history"][0]["issues"]
            if issue["code"] == "relation_reasoning_target_missing_binder_path"
        )
        self.assertEqual(first_relation_issue["details"]["actual_edges"], [("E2", "E1")])
        self.assertIn(
            "same_entity_negative_entity_matches_relation",
            first_relation_issue["details"]["failed_constraints"],
        )
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_total_event_count"], 8)
        self.assertTrue(
            constraints["required_binder_binding"]["alternate_binding_form_forbidden"]
        )
        self.assertEqual(constraints["parent_collision_event_ids"], ["E1", "E6"])
        self.assertIn(
            "events[2].entity == events[1].entity",
            constraints["required_field_equalities"],
        )
        self.assertIn("cascade those changes into E3 and E4", calls[1]["correction_instruction"])
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "relation_lookup"), [])

    def test_visibility_retry_lists_every_duplicate_scope_and_parent_collision(self) -> None:
        parent = seed_row()
        copied_claim = "The archive currently stores the budget draft as its active document."
        parent["dialogue"]["turns"][0]["text"] = copied_claim
        parent["semantic_parent_graph"] = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "visibility-parent",
            "events": [
                {
                    "event_id": "P1",
                    "entity": "Archive",
                    "slot": "document",
                    "value": "budget draft",
                    "status": "active",
                    "time_index": 1,
                }
            ],
        }
        required_scope = {
            "observer": "Alice",
            "permission": "view",
            "condition": "after approval",
        }

        def scoped_event(event_id: str, index: int, scope: dict, **updates) -> dict:
            event = {
                "event_id": event_id,
                "entity": "Archive",
                "slot": "document_access",
                "value": "budget draft",
                "status": "active",
                "time_index": index,
                "owner": "Archive",
                "resource": "document",
                "scope": copy.deepcopy(scope),
            }
            event.update(updates)
            return event

        def payload(events: list[dict]) -> dict:
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": "stage3:S0000:semantic_graph_v2",
                "parent_graph_id": "visibility-parent",
                "operation": "visibility",
                "events": events,
            }

        invalid = payload(
            [
                {"event_id": "E1", "entity": "Archive", "slot": "document", "value": "budget draft", "status": "active", "time_index": 1, "alignment_hints": {"support_claim": copied_claim}},
                scoped_event("E2", 2, required_scope, depends_on=["E1"]),
                scoped_event("E3", 3, required_scope),
                scoped_event("E4", 4, required_scope),
                scoped_event("E5", 5, {**required_scope, "condition": "before approval"}),
                scoped_event("E6", 6, required_scope),
                scoped_event("E7", 7, {**required_scope, "permission": "edit"}),
                scoped_event("E8", 8, {**required_scope, "observer": "Review team"}),
            ]
        )
        corrected = payload(
            [
                {"event_id": "E1", "entity": "Archive", "slot": "document", "value": "revised budget", "status": "active", "time_index": 1, "alignment_hints": {"support_claim": "The archive now stores the revised budget document."}},
                scoped_event("E2", 2, required_scope, depends_on=["E1"]),
                scoped_event("E3", 3, {**required_scope, "observer": "Review team"}),
                scoped_event("E4", 4, {**required_scope, "permission": "edit"}),
                scoped_event("E5", 5, {**required_scope, "condition": "before approval"}),
                scoped_event("E6", 6, {**required_scope, "observer": "Finance team"}),
                scoped_event("E7", 7, {**required_scope, "permission": "download"}),
                scoped_event("E8", 8, {**required_scope, "condition": "after publication"}),
            ]
        )
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(copy.deepcopy(prompt))
            return copy.deepcopy(invalid if len(calls) == 1 else corrected), {
                "used": True,
                "attempts": 1,
            }

        policy = stage3_policy("visibility")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1, {"visibility": 1.0}, None
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                parent,
                {},
                policy,
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(1732),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        layout = calls[0]["visibility_generation_layout_contract"]
        self.assertEqual(layout["wrong_observer_array_index"], 2)
        self.assertIn("OMIT scope", layout["resource"]["scope_rule"])
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_total_event_count"], 8)
        self.assertEqual(
            constraints["detected_exact_scope_match_event_ids"],
            ["E2", "E3", "E4", "E6"],
        )
        self.assertEqual(
            constraints["detected_duplicate_scope_event_ids"],
            ["E3", "E4", "E6"],
        )
        self.assertEqual(constraints["parent_collision_event_ids"], ["E1"])
        self.assertEqual(constraints["required_dependency_map"]["E8"], [])
        first_visibility_issue = next(
            issue
            for issue in meta["semantic_attempt_history"][0]["issues"]
            if issue["code"] == "visibility_reasoning_target_nonunique_composition"
        )
        self.assertEqual(first_visibility_issue["details"]["exact_scope_match_count"], 4)
        self.assertEqual(
            first_visibility_issue["details"]["exact_scope_match_event_ids"],
            ["E2", "E3", "E4", "E6"],
        )
        self.assertIn("Rewrite every ID in detected_duplicate_scope_event_ids", calls[1]["correction_instruction"])
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "visibility"), [])

    def test_temporal_graph_prompt_requires_earlier_current_and_later_invalid(self) -> None:
        payload = temporal_graph_payload(
            exact_temporal_anti_shortcut_generation_events()
        )
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return payload, {"used": True, "attempts": 1}

        policy = stage3_policy("temporal_state_update")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_graph_expansion(
                seed_row(),
                {},
                policy,
                SimpleNamespace(query_conditioned_graph_expansion=False, graph_expansion_max_tokens=4096, graph_semantic_retries=0),
                random.Random(31),
                sample_index=0,
                round_id="stage3",
            )

        requirements = " ".join(captured[0]["requirements"])
        self.assertIn("Return exactly 8 events", requirements)
        self.assertIn("events[0:4] are the only update chain", requirements)
        self.assertIn("events[1].status must be exactly 'active'", requirements)
        self.assertIn("maximum-time gold is a hard failure", requirements)
        self.assertIn("events[4:8] are four context singletons", requirements)
        self.assertIn("OMIT previous_value", requirements)
        self.assertIn("category label 'non-active' is forbidden", requirements)
        self.assertIn("greatest time_index alone is a forbidden shortcut", requirements)
        self.assertIn("['canceled']", requirements)
        self.assertEqual(
            captured[0]["stage64_reasoning_target_contract"][
                "hard_negative_strategy"
            ],
            "same_slot_canceled_value",
        )
        status_contract = captured[0]["status_literal_contract"]
        self.assertEqual(
            status_contract["allowed_status_literals"],
            sorted(subject.semantic_graph_v2.SUPPORTED_STATUSES),
        )
        self.assertEqual(
            status_contract["non_active_status_literals"],
            sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
        )
        self.assertEqual(
            status_contract["forbidden_category_labels_as_status"],
            ["non-active"],
        )
        layout = captured[0]["temporal_generation_layout_contract"]
        self.assertEqual(layout["primary_chain_array_indexes"], [0, 1, 2, 3])
        self.assertEqual(layout["context_singleton_array_indexes"], [4, 5, 6, 7])
        self.assertEqual(
            layout["required_dependency_map"],
            {
                "E1": [],
                "E2": ["E1"],
                "E3": ["E2"],
                "E4": ["E3"],
                "E5": [],
                "E6": [],
                "E7": [],
                "E8": [],
            },
        )
        self.assertIn(
            "E5,E6,E7,E8 each omit depends_on or use []",
            layout["machine_checks"],
        )
        for index in ("0", "2"):
            self.assertEqual(
                layout["primary_chain"]["status_by_array_index"][index][
                    "allowed_status_literals"
                ],
                sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
            )
        self.assertEqual(
            layout["primary_chain"]["status_by_array_index"]["1"][
                "allowed_status_literals"
            ],
            ["active"],
        )
        self.assertEqual(
            layout["primary_chain"]["status_by_array_index"]["3"][
                "allowed_status_literals"
            ],
            ["canceled"],
        )
        self.assertTrue(layout["primary_chain"]["max_time_shortcut_forbidden"])
        self.assertEqual(layout["primary_chain"]["current_gold_array_index"], 1)
        temporal_event_schemas = captured[0]["output_schema"]["events"]
        self.assertEqual(len(temporal_event_schemas), 8)
        event_schema = temporal_event_schemas[0]
        self.assertIn("depends_on", event_schema)
        self.assertIn("depends_on_by_array_index", layout["primary_chain"])
        self.assertIn("immediately preceding event_id", requirements)
        self.assertNotIn("previous_value", event_schema)
        self.assertNotIn("event_type", event_schema)
        self.assertEqual(
            event_schema["status"]["allowed_values"],
            sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
        )
        self.assertTrue(event_schema["status"]["must_equal_one_allowed_value"])
        self.assertEqual(
            [event["event_id"] for event in temporal_event_schemas],
            [f"E{index}" for index in range(1, 9)],
        )
        self.assertEqual(
            [event["depends_on"] for event in temporal_event_schemas],
            [[], ["E1"], ["E2"], ["E3"], [], [], [], []],
        )
        self.assertEqual(
            temporal_event_schemas[1]["status"]["allowed_values"], ["active"]
        )
        self.assertEqual(
            temporal_event_schemas[2]["status"]["allowed_values"],
            sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
        )
        self.assertEqual(
            temporal_event_schemas[3]["status"]["allowed_values"], ["canceled"]
        )

    def test_temporal_dialogue_prompt_defers_qa_until_alignment_is_frozen(self) -> None:
        events = [
            {"event_id": "E1", "entity": "Maya", "slot": "meeting_city", "value": "Paris", "status": "outdated", "time_index": 1},
            {"event_id": "E2", "entity": "Maya", "slot": "meeting_city", "value": "Lyon", "status": "canceled", "time_index": 2, "previous_value": "Paris"},
            {"event_id": "E3", "entity": "Maya", "slot": "meeting_city", "value": "Berlin", "status": "active", "time_index": 3, "previous_value": "Lyon"},
            {"event_id": "E4", "entity": "Noah", "slot": "meeting_city", "value": "Rome", "status": "outdated", "time_index": 4},
        ]
        policy = stage3_policy("temporal_state_update")
        policy["qa_operation_answer_type_diversification"] = True
        plan = subject.normalize_current_event_plan(
            {"events": events},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=policy,
            sample_index=0,
            round_id="stage3",
        )
        plan = subject.assign_current_alignment_anchors(plan, random.Random(32))
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"event_alignment_proposal": [], "qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_graph_conditioned_dialogue_expansion(
                plan,
                seed_row(),
                {},
                policy,
                SimpleNamespace(
                    solver_tokenizer=None,
                    tokenizer_path=None,
                    base_model=None,
                    challenger_max_tokens=4096,
                    parent_context_tail_turns=12,
                ),
                parent_context={},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=1024,
            )

        requirements = " ".join(captured[0]["requirements"])
        self.assertIn("Do not return qa_proposal", requirements)
        self.assertIn("separate QA-only call", requirements)
        self.assertNotIn("qa_proposal", captured[0]["output_schema"])

    def test_hard_target_dialogue_prompt_defers_qa_until_alignment_is_frozen(self) -> None:
        plan = subject.normalize_current_event_plan(
            {"events": raw_events(5)},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("relation_lookup", "hard"),
            sample_index=0,
            round_id="stage3",
        )
        plan = subject.assign_current_alignment_anchors(plan, random.Random(29))
        policy = stage3_policy("relation_lookup", "hard")
        policy["qa_operation_answer_type_diversification"] = True
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"event_alignment_proposal": [], "qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_graph_conditioned_dialogue_expansion(
                plan,
                seed_row(),
                {},
                policy,
                SimpleNamespace(
                    solver_tokenizer=None,
                    tokenizer_path=None,
                    base_model=None,
                    challenger_max_tokens=4096,
                    parent_context_tail_turns=12,
                ),
                parent_context={},
                parent_dialogue_source="dialogue.turns",
                parent_turn_count=2,
                min_added_turns=8,
                min_added_tokens=1024,
            )

        prompt = captured[0]
        self.assertEqual(prompt["item_difficulty_target"], "hard")
        requirements = " ".join(prompt["requirements"])
        self.assertIn("Do not return qa_proposal", requirements)
        self.assertIn("separate QA-only call", requirements)
        self.assertNotIn("qa_proposal", prompt["output_schema"])

    def test_comparison_retry_supplies_machine_checkable_pair_constraints(self) -> None:
        expected_parent = "stage3_parent:stage2-parent-1"
        expected_child = "stage3:S0000:semantic_graph_v2"

        def comparison_payload(right_entity: str) -> dict:
            return {
                "schema_version": "semantic_graph_v2",
                "graph_id": expected_child,
                "parent_graph_id": expected_parent,
                "operation": "comparison",
                "events": [
                    {"event_id": "E1", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1},
                    {"event_id": "E2", "entity": right_entity, "slot": "score", "value": 12, "status": "active", "time_index": 1},
                    {"event_id": "E3", "entity": "Alice", "slot": "score", "value": 9, "status": "outdated", "time_index": 2},
                    {"event_id": "E4", "entity": right_entity, "slot": "score", "value": 12, "status": "outdated", "time_index": 2},
                    {"event_id": "E5", "entity": "Bob", "slot": "office", "value": "North", "status": "active", "time_index": 4},
                    {"event_id": "E6", "entity": "Evan", "slot": "team", "value": "Atlas", "status": "active", "time_index": 5},
                    {"event_id": "E7", "entity": "Faye", "slot": "route", "value": "West", "status": "active", "time_index": 6},
                    {"event_id": "E8", "entity": "Gina", "slot": "project", "value": "Orchid", "status": "active", "time_index": 7},
                ],
            }

        first = comparison_payload("Alice")
        corrected = comparison_payload("Bob")
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return (first if len(calls) == 1 else corrected), {"used": True, "attempts": 1}

        policy = stage3_policy("comparison")
        policy["stage64_target"] = subject._build_stage64_canonical_targets(
            1, {"comparison": 1.0}, None
        )[0]["stage64_target"]
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                policy,
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(19),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_total_event_count"], 8)
        self.assertEqual(constraints["primary_pair_event_ids"], ["E1", "E2"])
        self.assertEqual(constraints["contrast_pair_event_ids"], ["E3", "E4"])
        self.assertEqual(constraints["context_event_ids"], ["E5", "E6", "E7", "E8"])
        self.assertEqual(constraints["required_right_entity"], "Bob")
        self.assertEqual(constraints["required_shared_slot"], "score")
        self.assertEqual(constraints["required_shared_time_index"], 1)
        self.assertEqual(constraints["primary_key"], {"normalized_slot": "score", "time_index": 1})
        self.assertEqual(
            constraints["required_contrast_difference_relation"],
            "P=abs(events[0].value-events[1].value); C=abs(events[2].value-events[3].value); abs(P-C)==1 and C!=P",
        )
        self.assertIn(
            "context (normalized slot, time_index) keys are pairwise distinct",
            constraints["machine_checks"],
        )
        self.assertIn(
            "P=abs(events[0].value-events[1].value), C=abs(events[2].value-events[3].value), abs(P-C)==1, and C!=P",
            constraints["machine_checks"],
        )
        self.assertIn("last four are context roles", calls[1]["correction_instruction"])
        self.assertIn("abs(P-C)==1", calls[1]["correction_instruction"])
        graph_requirements = " ".join(calls[0]["requirements"])
        self.assertIn("Return exactly 8 events", graph_requirements)
        self.assertIn("sole permitted repeated non-primary", graph_requirements)
        self.assertIn("explicitly include that event's own decimal integer value", graph_requirements)
        self.assertIn("mixed units such as years/seasons are forbidden", graph_requirements)
        layout = calls[0]["comparison_generation_layout_contract"]
        self.assertEqual(layout["primary_pair_array_indexes"], [0, 1])
        self.assertEqual(layout["contrast_array_indexes"], [2, 3])
        self.assertEqual(layout["context_array_indexes"], [4, 5, 6, 7])
        self.assertIn("abs(P-C)==1", layout["two_plus_two_contrast"]["numeric_relation"])
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "comparison"), [])

    def test_run109_second_numeric_pair_fixtures_fail_native_comparison_layout(self) -> None:
        fixtures = [
            run109_screenplay_second_pair_events(),
            run109_art_show_date_second_pair_events(),
        ]
        for events in fixtures:
            with self.subTest(primary_slot=events[0]["slot"]):
                payload = comparison_graph_payload(events)
                original = copy.deepcopy(payload)

                layout_issues = subject.comparison_raw_generation_layout_issues(payload)
                layout_codes = {issue["code"] for issue in layout_issues}
                core_codes = {
                    issue["code"]
                    for issue in subject.semantic_graph_v2.validate_graph_delta(
                        payload,
                        operation="comparison",
                    )
                }

                self.assertIn("comparison_generation_layout_context_key_collision", layout_codes)
                self.assertIn("comparison_missing_unique_numeric_pair", core_codes)
                self.assertEqual(payload, original)

    def test_run109_target1_core_valid_raw_is_rejected_only_by_tighter_native_layout(self) -> None:
        payload = comparison_graph_payload(run109_target1_core_valid_events())

        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(payload, operation="comparison"),
            [],
        )
        layout_codes = {
            issue["code"]
            for issue in subject.comparison_raw_generation_layout_issues(payload)
        }
        self.assertIn("comparison_generation_layout_context_key_collision", layout_codes)
        self.assertIn(
            "comparison_generation_layout_context_primary_key_collision",
            layout_codes,
        )

    def test_run109_target1_time_key_legalized_fixture_passes_both_gates(self) -> None:
        events = run109_target1_core_valid_events()
        # Preserve every event, slot, status, and value; only separate the three
        # repeated context keys along their time dimension.
        events[4]["time_index"] = 2
        events[6]["time_index"] = 2
        events[7]["time_index"] = 2
        payload = comparison_graph_payload(events)

        self.assertEqual(subject.comparison_raw_generation_layout_issues(payload), [])
        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(payload, operation="comparison"),
            [],
        )

    def test_native_comparison_layout_is_narrower_than_general_six_event_core(self) -> None:
        events = run109_target1_core_valid_events()
        events[4]["time_index"] = 2
        six_event_payload = comparison_graph_payload(events[:6])

        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(
                six_event_payload,
                operation="comparison",
            ),
            [],
        )
        self.assertEqual(
            [
                issue["code"]
                for issue in subject.comparison_raw_generation_layout_issues(
                    six_event_payload
                )
            ],
            ["comparison_generation_layout_event_count"],
        )

    def test_run109_art_show_second_pair_is_hard_rejected_by_api_graph_gate(self) -> None:
        payload = comparison_graph_payload(run109_art_show_date_second_pair_events())
        original = copy.deepcopy(payload)
        with mock.patch.object(
            subject,
            "call_raw_json_api",
            return_value=(payload, {"used": True, "attempts": 1}),
        ):
            with self.assertRaises(subject.NativeGraphDialogueReject) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy("comparison"),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=0,
                    ),
                    random.Random(191),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(caught.exception.api_graph_payload, original)
        self.assertIn("comparison_generation_layout_context_key_collision", str(caught.exception))

    def test_run115_duplicate_count_member_fixture_fails_native_layout_without_mutation(self) -> None:
        payload = count_graph_payload(run115_duplicate_count_member_events())
        original = copy.deepcopy(payload)

        layout_codes = {
            issue["code"]
            for issue in subject.count_raw_generation_layout_issues(payload)
        }
        core_codes = {
            issue["code"]
            for issue in subject.semantic_graph_v2.validate_graph_delta(
                payload,
                operation="count",
            )
        }

        self.assertIn("count_generation_layout_duplicate_member_values", layout_codes)
        self.assertIn("count_missing_candidate_set", core_codes)
        self.assertEqual(payload, original)

    def test_exact_count_generation_layout_passes_native_and_general_gates(self) -> None:
        payload = count_graph_payload(exact_count_generation_events())

        self.assertEqual(subject.count_raw_generation_layout_issues(payload), [])
        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(payload, operation="count"),
            [],
        )

    def test_count_layout_rejects_balanced_but_wrong_array_truth_roles(self) -> None:
        events = exact_count_generation_events()
        wrong_roles = [True, False, True, False, True, False, True, False]
        for event, included in zip(events, wrong_roles):
            event["set_membership"]["included"] = included
        payload = count_graph_payload(events)
        original = copy.deepcopy(payload)

        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(payload, operation="count"),
            [],
        )
        self.assertIn(
            "count_generation_layout_inclusion_roles",
            {
                issue["code"]
                for issue in subject.count_raw_generation_layout_issues(payload)
            },
        )
        self.assertEqual(payload, original)

    def test_run115_duplicate_count_graph_is_hard_rejected_by_api_graph_gate(self) -> None:
        payload = count_graph_payload(run115_duplicate_count_member_events())
        original = copy.deepcopy(payload)

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            return_value=(payload, {"used": True, "attempts": 1}),
        ):
            with self.assertRaises(subject.NativeGraphDialogueReject) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy("count"),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=0,
                    ),
                    random.Random(1150),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(caught.exception.api_graph_payload, original)
        self.assertIn("count_generation_layout_duplicate_member_values", str(caught.exception))

    def test_native_count_layout_is_narrower_than_general_six_member_core(self) -> None:
        payload = count_graph_payload(exact_count_generation_events()[:6])

        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(payload, operation="count"),
            [],
        )
        self.assertEqual(
            [
                issue["code"]
                for issue in subject.count_raw_generation_layout_issues(payload)
            ],
            ["count_generation_layout_event_count"],
        )

    def test_count_retry_supplies_fixed_four_true_four_false_machine_checks(self) -> None:
        invalid = count_graph_payload(run115_duplicate_count_member_events())
        corrected = count_graph_payload(exact_count_generation_events())
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return (invalid if len(calls) == 1 else corrected), {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("count"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(115),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(raw, corrected)
        self.assertEqual(meta["semantic_retry_count"], 1)
        initial_requirements = " ".join(calls[0]["requirements"])
        self.assertIn("Return exactly 8 candidate-member events", initial_requirements)
        self.assertIn("events[0:4].set_membership.included to true", initial_requirements)
        layout = calls[0]["count_generation_layout_contract"]
        self.assertEqual(layout["included_member_array_indexes"], [0, 1, 2, 3])
        self.assertEqual(layout["excluded_member_array_indexes"], [4, 5, 6, 7])
        self.assertEqual(
            layout["membership_by_array_index"],
            [True, True, True, True, False, False, False, False],
        )
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_total_event_count"], 8)
        self.assertEqual(constraints["included_member_event_ids"], ["E1", "E2", "E3", "E4"])
        self.assertEqual(constraints["excluded_member_event_ids"], ["E5", "E6", "E7", "E8"])
        self.assertEqual(
            constraints["required_included_by_event_id"],
            {
                "E1": True,
                "E2": True,
                "E3": True,
                "E4": True,
                "E5": False,
                "E6": False,
                "E7": False,
                "E8": False,
            },
        )
        self.assertIn(
            "all eight normalized value fields are pairwise distinct",
            constraints["machine_checks"],
        )
        self.assertIn("replacing any value duplicated", calls[1]["correction_instruction"])
        self.assertEqual(subject.semantic_graph_plan_quality_issues(plan, "count"), [])

    def test_comparison_integer_string_normalizer_is_lossless_and_narrow(self) -> None:
        payload = {
            "operation": "comparison",
            "events": [
                {"event_id": "E1", "value": " 12 "},
                {"event_id": "E2", "value": "-7"},
                {"event_id": "E3", "value": "12.5"},
                {"event_id": "E4", "value": "12 people"},
                {"event_id": "E5", "value": 4},
            ],
        }
        before = copy.deepcopy(payload)

        normalized, diagnostics = subject.normalize_comparison_integer_strings(payload)

        self.assertEqual([event["value"] for event in normalized["events"]], [12, -7, "12.5", "12 people", 4])
        self.assertEqual(payload, before)
        self.assertTrue(diagnostics["comparison_integer_string_normalizer_used"])
        self.assertEqual(diagnostics["comparison_integer_string_normalized_count"], 2)

    def test_comparison_integer_strings_are_normalized_before_semantic_validation(self) -> None:
        payload = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "stage3:S0000:semantic_graph_v2",
            "parent_graph_id": "stage3_parent:stage2-parent-1",
            "operation": "comparison",
            "events": [
                {"event_id": "E1", "entity": "Alice", "slot": "score", "value": "10", "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Bob", "slot": "score", "value": "13", "status": "active", "time_index": 1},
                {"event_id": "E3", "entity": "Alice", "slot": "score", "value": 9, "status": "outdated", "time_index": 2},
                {"event_id": "E4", "entity": "Bob", "slot": "score", "value": 11, "status": "outdated", "time_index": 2},
                {"event_id": "E5", "entity": "Evan", "slot": "team", "value": "Atlas", "status": "active", "time_index": 4},
                {"event_id": "E6", "entity": "Faye", "slot": "route", "value": "West", "status": "active", "time_index": 5},
                {"event_id": "E7", "entity": "Gina", "slot": "shelf", "value": "Upper", "status": "active", "time_index": 6},
                {"event_id": "E8", "entity": "Hana", "slot": "badge", "value": "Silver", "status": "active", "time_index": 7},
            ],
        }
        with mock.patch.object(
            subject,
            "call_raw_json_api",
            return_value=(payload, {"used": True, "attempts": 1}),
        ):
            plan, meta, raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("comparison"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=0,
                ),
                random.Random(43),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual([raw["events"][0]["value"], raw["events"][1]["value"]], [10, 13])
        self.assertEqual([plan["events"][0]["value"], plan["events"][1]["value"]], [10, 13])
        self.assertEqual(
            meta["normalizer_diagnostics"]["comparison_integer_string_normalized_count"],
            2,
        )

    def test_comparison_exact_duplicate_events_are_removed_before_validation(self) -> None:
        payload = {
            "schema_version": "semantic_graph_v2",
            "graph_id": "stage3:S0000:semantic_graph_v2",
            "parent_graph_id": "stage3_parent:stage2-parent-1",
            "operation": "comparison",
            "events": [
                {"event_id": "E1", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1, "required_for_dialogue": True},
                {"event_id": "E2", "entity": "Bob", "slot": "score", "value": 13, "status": "active", "time_index": 1, "required_for_dialogue": True},
                {"event_id": "E3", "entity": "Cara", "slot": "city", "value": "Paris", "status": "outdated", "time_index": 0, "required_for_dialogue": True},
                {"event_id": "E4", "entity": "Dina", "slot": "office", "value": "North", "status": "outdated", "time_index": 0, "required_for_dialogue": True},
                {"event_id": "E5", "entity": "Evan", "slot": "team", "value": "Atlas", "status": "active", "time_index": 4, "required_for_dialogue": True},
                {"event_id": "E6", "entity": "Faye", "slot": "route", "value": "West", "status": "active", "time_index": 5, "required_for_dialogue": True},
                {"event_id": "E7", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1, "required_for_dialogue": True},
                {"event_id": "E8", "entity": "Bob", "slot": "score", "value": 13, "status": "active", "time_index": 1, "required_for_dialogue": True},
            ],
        }
        before = copy.deepcopy(payload)

        normalized, diagnostics = subject.normalize_comparison_exact_duplicate_events(payload)

        self.assertEqual(payload, before)
        self.assertEqual([event["event_id"] for event in normalized["events"]], ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertTrue(diagnostics["comparison_exact_duplicate_normalizer_used"])
        self.assertEqual(diagnostics["comparison_exact_duplicate_removed_event_ids"], ["E7", "E8"])
        self.assertEqual(
            diagnostics["comparison_exact_duplicate_retained_by_removed_id"],
            {"E7": "E1", "E8": "E2"},
        )
        self.assertEqual(diagnostics["comparison_exact_duplicate_original_event_count"], 8)
        self.assertEqual(diagnostics["comparison_exact_duplicate_result_event_count"], 6)
        self.assertNotEqual(
            diagnostics["comparison_exact_duplicate_original_sha256"],
            diagnostics["comparison_exact_duplicate_result_sha256"],
        )
        self.assertEqual(subject.semantic_graph_v2.validate_graph_delta(normalized, operation="comparison"), [])

    def test_comparison_near_duplicate_event_is_not_removed(self) -> None:
        payload = {
            "operation": "comparison",
            "events": [
                {"event_id": f"E{index}", "entity": f"entity-{index}", "slot": "score", "value": index, "status": "active", "time_index": index}
                for index in range(1, 8)
            ]
            + [
                {"event_id": "E8", "entity": "entity-1", "slot": "score", "value": 1, "status": "outdated", "time_index": 1}
            ],
        }

        normalized, diagnostics = subject.normalize_comparison_exact_duplicate_events(payload)

        self.assertIs(normalized, payload)
        self.assertFalse(diagnostics["comparison_exact_duplicate_normalizer_used"])
        self.assertEqual(len(normalized["events"]), 8)

    def test_comparison_duplicate_id_is_not_repaired_by_exact_dedup(self) -> None:
        payload = {
            "operation": "comparison",
            "events": [
                {"event_id": f"E{index}", "entity": f"entity-{index}", "slot": f"slot-{index}", "value": index, "status": "active", "time_index": index}
                for index in range(1, 8)
            ],
        }
        payload["events"].append(copy.deepcopy(payload["events"][0]))

        normalized, diagnostics = subject.normalize_comparison_exact_duplicate_events(payload)

        self.assertIs(normalized, payload)
        self.assertEqual(diagnostics["comparison_exact_duplicate_noop_reason"], "duplicate_event_id")
        issue_codes = {
            issue["code"]
            for issue in subject.semantic_graph_v2.validate_graph_delta(payload, operation="comparison")
        }
        self.assertIn("duplicate_event_id", issue_codes)

    def test_comparison_out_of_range_graph_is_not_repaired_by_exact_dedup(self) -> None:
        payload = {
            "operation": "comparison",
            "events": [
                {"event_id": f"E{index}", "entity": f"entity-{index}", "slot": f"slot-{index}", "value": index, "status": "active", "time_index": index}
                for index in range(1, 9)
            ],
        }
        duplicate = copy.deepcopy(payload["events"][0])
        duplicate["event_id"] = "E9"
        payload["events"].append(duplicate)

        normalized, diagnostics = subject.normalize_comparison_exact_duplicate_events(payload)

        self.assertIs(normalized, payload)
        self.assertEqual(diagnostics["comparison_exact_duplicate_noop_reason"], "raw_event_count_out_of_range")
        self.assertEqual(len(normalized["events"]), 9)

    def test_comparison_dependency_referenced_duplicate_event_is_not_removed(self) -> None:
        payload = {
            "operation": "comparison",
            "events": [
                {"event_id": "E1", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1},
                {"event_id": "E2", "entity": "Bob", "slot": "score", "value": 13, "status": "active", "time_index": 1},
                {"event_id": "E3", "entity": "Cara", "slot": "city", "value": "Paris", "status": "active", "time_index": 2},
                {"event_id": "E4", "entity": "Dina", "slot": "office", "value": "North", "status": "active", "time_index": 3},
                {"event_id": "E5", "entity": "Evan", "slot": "team", "value": "Atlas", "status": "active", "time_index": 4},
                {"event_id": "E6", "entity": "Faye", "slot": "route", "value": "West", "status": "active", "time_index": 5},
                {"event_id": "E7", "entity": "Alice", "slot": "score", "value": 10, "status": "active", "time_index": 1},
                {"event_id": "E8", "entity": "Gina", "slot": "audit", "value": "done", "status": "active", "time_index": 6, "depends_on": ["E7"]},
            ],
        }

        normalized, diagnostics = subject.normalize_comparison_exact_duplicate_events(payload)

        self.assertIs(normalized, payload)
        self.assertFalse(diagnostics["comparison_exact_duplicate_normalizer_used"])
        self.assertEqual(len(normalized["events"]), 8)

    def test_run106_temporal_two_chain_is_reported_and_hard_rejected_without_mutation(self) -> None:
        payload = temporal_graph_payload(run106_temporal_two_chain_events())
        original = copy.deepcopy(payload)

        layout_issues = subject.temporal_raw_generation_layout_issues(payload)
        layout_codes = {issue["code"] for issue in layout_issues}
        self.assertIn("temporal_generation_layout_primary_value_cycle", layout_codes)
        self.assertIn("temporal_generation_layout_earlier_primary_active", layout_codes)
        self.assertIn("temporal_generation_layout_terminal_not_active", layout_codes)
        self.assertIn("temporal_generation_layout_initial_previous_value_present", layout_codes)
        self.assertIn("temporal_generation_layout_context_not_singleton", layout_codes)
        self.assertIn("temporal_generation_layout_context_previous_value_present", layout_codes)
        core_codes = {
            issue["code"]
            for issue in subject.semantic_graph_v2.validate_graph_delta(
                payload,
                operation="temporal_state_update",
            )
        }
        self.assertIn("temporal_multiple_update_chains", core_codes)
        self.assertEqual(payload, original)

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            return_value=(payload, {"used": True, "attempts": 1}),
        ):
            with self.assertRaises(subject.NativeGraphDialogueReject) as caught:
                subject.call_api_graph_expansion(
                    seed_row(),
                    {},
                    stage3_policy("temporal_state_update"),
                    SimpleNamespace(
                        query_conditioned_graph_expansion=False,
                        graph_expansion_max_tokens=4096,
                        graph_semantic_retries=0,
                    ),
                    random.Random(40),
                    sample_index=0,
                    round_id="stage3",
                )

        self.assertEqual(caught.exception.api_graph_payload, original)
        self.assertIn("temporal_generation_layout_primary_value_cycle", str(caught.exception))

    def test_exact_temporal_generation_layout_passes_native_and_general_gates(self) -> None:
        payload = temporal_graph_payload(exact_temporal_generation_events())

        self.assertEqual(subject.temporal_raw_generation_layout_issues(payload), [])
        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(
                payload,
                operation="temporal_state_update",
            ),
            [],
        )

    def test_temporal_raw_layout_rejects_date_chain_for_location_target_before_dialogue(
        self,
    ) -> None:
        events = exact_temporal_anti_shortcut_generation_events()
        for index, event in enumerate(events[:4]):
            event["slot"] = "appointment_date"
            event["value"] = f"2026-04-{10 + index:02d}"
            if index:
                event["previous_value"] = f"2026-04-{9 + index:02d}"
        payload = temporal_graph_payload(events)
        location_target = {
            "qa_operation": "temporal_state_update",
            "preferred_answer_type": "LocationOrScope",
        }
        date_target = {
            "qa_operation": "temporal_state_update",
            "preferred_answer_type": "DateOrTime",
        }

        location_codes = {
            issue["code"]
            for issue in subject.temporal_raw_generation_layout_issues(
                payload,
                location_target,
            )
        }
        self.assertIn(
            "temporal_generation_layout_primary_answer_type_mismatch",
            location_codes,
        )
        date_codes = {
            issue["code"]
            for issue in subject.temporal_raw_generation_layout_issues(
                payload,
                date_target,
            )
        }
        self.assertNotIn(
            "temporal_generation_layout_primary_answer_type_mismatch",
            date_codes,
        )

    def test_native_temporal_layout_is_narrower_than_general_semantic_core(self) -> None:
        payload = temporal_graph_payload(exact_temporal_generation_events()[:6])

        self.assertEqual(
            subject.semantic_graph_v2.validate_graph_delta(
                payload,
                operation="temporal_state_update",
            ),
            [],
        )
        layout_issues = subject.temporal_raw_generation_layout_issues(payload)
        self.assertEqual(
            [issue["code"] for issue in layout_issues],
            ["temporal_generation_layout_event_count"],
        )

    def test_native_temporal_layout_rejects_any_dependency_outside_exact_map(self) -> None:
        events = exact_temporal_generation_events()
        events[4]["depends_on"] = ["E1"]
        codes = {
            issue["code"]
            for issue in subject.temporal_raw_generation_layout_issues(
                temporal_graph_payload(events)
            )
        }
        self.assertIn("temporal_generation_layout_dependency_map", codes)

    def test_temporal_dependency_compiler_fills_only_omitted_frozen_edges(self) -> None:
        events = exact_temporal_generation_events()
        for event in events[1:4]:
            event.pop("depends_on")
        provider_payload = temporal_graph_payload(events)
        provider_sha = subject.canonical_json_sha256(provider_payload)

        compiled, audit = subject.materialize_temporal_dependency_contract(
            provider_payload
        )

        self.assertTrue(audit["pass"])
        self.assertTrue(audit["applied"])
        self.assertEqual(audit["materialized_event_ids"], ["E2", "E3", "E4"])
        self.assertEqual(audit["explicit_conflict_event_ids"], [])
        self.assertEqual(audit["provider_payload_sha256"], provider_sha)
        self.assertEqual(
            [event.get("depends_on", []) for event in compiled["events"]],
            [[], ["E1"], ["E2"], ["E3"], [], [], [], []],
        )
        self.assertEqual(
            subject.temporal_raw_generation_layout_issues(compiled), []
        )
        # The provider object is immutable and remains independently auditable.
        self.assertNotIn("depends_on", provider_payload["events"][1])

    def test_temporal_dependency_compiler_never_overwrites_explicit_conflict(self) -> None:
        events = exact_temporal_generation_events()
        events[2]["depends_on"] = ["E1"]
        provider_payload = temporal_graph_payload(events)

        compiled, audit = subject.materialize_temporal_dependency_contract(
            provider_payload
        )

        self.assertFalse(audit["pass"])
        self.assertFalse(audit["applied"])
        self.assertEqual(audit["explicit_conflict_event_ids"], ["E3"])
        self.assertEqual(compiled, provider_payload)
        self.assertIn(
            "temporal_generation_layout_dependency_map",
            {
                issue["code"]
                for issue in subject.temporal_raw_generation_layout_issues(
                    compiled
                )
            },
        )

    def test_temporal_dependency_compiler_does_not_hide_duplicate_values(self) -> None:
        events = exact_temporal_generation_events()
        events[1]["value"] = events[0]["value"]
        events[2]["previous_value"] = events[1]["value"]
        for event in events[1:4]:
            event.pop("depends_on")

        compiled, audit = subject.materialize_temporal_dependency_contract(
            temporal_graph_payload(events)
        )

        self.assertTrue(audit["applied"])
        codes = {
            issue["code"]
            for issue in subject.temporal_raw_generation_layout_issues(compiled)
        }
        self.assertIn("temporal_generation_layout_primary_value_cycle", codes)
        self.assertNotIn("temporal_generation_layout_dependency_map", codes)

    def test_graph_call_audits_compiled_temporal_dependencies_before_validation(self) -> None:
        events = exact_temporal_generation_events()
        for event in events[1:4]:
            event.pop("depends_on")
        provider_payload = temporal_graph_payload(events)

        with mock.patch.object(
            subject,
            "call_raw_json_api",
            return_value=(copy.deepcopy(provider_payload), {"used": True, "attempts": 1}),
        ) as semantic_api:
            _plan, meta, compiled_payload = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("temporal_state_update"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=0,
                ),
                random.Random(217),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(semantic_api.call_count, 1)
        audit = meta["temporal_dependency_compiler"]
        self.assertTrue(audit["pass"])
        self.assertTrue(audit["applied"])
        self.assertEqual(audit["materialized_event_ids"], ["E2", "E3", "E4"])
        self.assertEqual(
            [event.get("depends_on", []) for event in compiled_payload["events"]],
            [[], ["E1"], ["E2"], ["E3"], [], [], [], []],
        )
        provider_copy = meta[
            "provider_payload_before_temporal_dependency_compilation"
        ]
        self.assertNotIn("depends_on", provider_copy["events"][1])

    def test_temporal_retry_supplies_exact_single_chain_constraints(self) -> None:
        expected_parent = "stage3_parent:stage2-parent-1"
        expected_child = "stage3:S0000:semantic_graph_v2"
        invalid_events = run106_temporal_two_chain_events()
        corrected_events = exact_temporal_generation_events()
        payloads = [
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": expected_child,
                "parent_graph_id": expected_parent,
                "operation": "temporal_state_update",
                "events": invalid_events,
            },
            {
                "schema_version": "semantic_graph_v2",
                "graph_id": expected_child,
                "parent_graph_id": expected_parent,
                "operation": "temporal_state_update",
                "events": corrected_events,
            },
        ]
        calls: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            calls.append(prompt)
            return payloads[len(calls) - 1], {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            _plan, meta, _raw = subject.call_api_graph_expansion(
                seed_row(),
                {},
                stage3_policy("temporal_state_update"),
                SimpleNamespace(
                    query_conditioned_graph_expansion=False,
                    graph_expansion_max_tokens=4096,
                    graph_semantic_retries=1,
                ),
                random.Random(41),
                sample_index=0,
                round_id="stage3",
            )

        self.assertEqual(meta["semantic_retry_count"], 1)
        constraints = calls[1]["semantic_correction_constraints"]
        self.assertEqual(constraints["required_total_event_count"], 8)
        self.assertEqual(constraints["required_total_event_count_range"], [8, 8])
        self.assertEqual(constraints["required_chain_count"], 1)
        self.assertEqual(constraints["required_chain_event_count_range"], [4, 4])
        self.assertEqual(constraints["primary_chain_event_ids"], ["E1", "E2", "E3", "E4"])
        self.assertEqual(constraints["secondary_context_event_ids"], ["E5", "E6", "E7", "E8"])
        self.assertEqual(
            constraints["required_dependency_map"],
            {
                "E1": [],
                "E2": ["E1"],
                "E3": ["E2"],
                "E4": ["E3"],
                "E5": [],
                "E6": [],
                "E7": [],
                "E8": [],
            },
        )
        self.assertIn(
            "no dependency edge exists beyond required_dependency_map",
            constraints["machine_checks"],
        )
        self.assertEqual(
            constraints["previous_value_forbidden_event_ids"],
            ["E1", "E5", "E6", "E7", "E8"],
        )
        for event_id in ("E1", "E2", "E3"):
            self.assertEqual(
                constraints["required_status_literals_by_event_id"][event_id],
                sorted(subject.semantic_graph_v2.NON_ACTIVE_STATUSES),
            )
        self.assertEqual(
            constraints["required_status_literals_by_event_id"]["E4"],
            ["active"],
        )
        self.assertNotIn("required_status_by_event_id", constraints)
        self.assertTrue(
            constraints["primary_value_uniqueness_contract"][
                "canceled_strategy_value_must_differ_from_current_gold"
            ]
        )
        self.assertTrue(
            constraints["primary_value_uniqueness_contract"][
                "strategy_label_is_not_a_restore_instruction"
            ]
        )
        self.assertEqual(
            constraints["forbidden_category_labels_as_status"],
            ["non-active"],
        )
        diagnostics = constraints["detected_candidate_chain_diagnostics"]
        self.assertEqual([item["event_ids"] for item in diagnostics], [["E1", "E2", "E3", "E4"], ["E5", "E6", "E7", "E8"]])
        self.assertIn("duplicate_or_cycle_value", diagnostics[0]["machine_failures"])
        self.assertIn("current_state_count_not_one", diagnostics[0]["machine_failures"])
        self.assertIn("earlier_event_active", diagnostics[0]["machine_failures"])
        self.assertIn("restored_status_forbidden_by_native_layout", diagnostics[0]["machine_failures"])
        self.assertIn("previous_value", calls[1]["correction_instruction"])
        self.assertIn("Do not preserve a restore cycle", calls[1]["correction_instruction"])
        self.assertIn(
            "canceled event is an alternative value",
            calls[1]["correction_instruction"],
        )
        self.assertIn(
            "never the category label 'non-active'",
            calls[1]["correction_instruction"],
        )
        self.assertIn(
            "E2=['E1'], E3=['E2'], E4=['E3']",
            calls[1]["correction_instruction"],
        )
        self.assertIn(
            "E5=E6=E7=E8=[]",
            calls[1]["correction_instruction"],
        )

    def test_temporal_qa_repair_prompt_matches_same_slot_quality_gate(self) -> None:
        plan = subject.normalize_current_event_plan(
            {
                "events": exact_temporal_generation_events()
            },
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=stage3_policy("temporal_state_update"),
            sample_index=0,
            round_id="stage3",
        )
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_qa_repair(
                plan,
                {},
                {},
                {"qa_operation": "temporal_state_update", "preferred_answer_type": "LocationOrScope"},
                "temporal_distractors_not_same_slot_updates",
                [],
                SimpleNamespace(),
            )

        requirements = " ".join(captured[0]["requirements"])
        self.assertIn("gold plus at least two distractors", requirements)
        self.assertIn("exact same entity and exact same slot", requirements)
        self.assertIn("every earlier same-slot source must be non-active", requirements)
        self.assertIn("three semantically distinct graph values", requirements)
        self.assertIn("gold source_event_instance_id must be that unique current event", requirements)
        self.assertIn("Never add parenthetical status annotations", requirements)
        self.assertEqual(captured[0]["target"]["required_qa_subtype"], "latest_state_update")
        self.assertEqual(
            captured[0]["output_schema"]["qa_proposal"]["qa_subtype"],
            "latest_state_update",
        )
        temporal_contract = captured[0]["temporal_state_update_graph_contract"]
        chain_ids = temporal_contract["event_instance_ids"]
        self.assertEqual(len(chain_ids), 4)
        self.assertEqual(
            captured[0]["output_schema"]["qa_proposal"]["gold"], ["D"]
        )
        self.assertEqual(
            [
                option["text"]
                for option in captured[0]["output_schema"]["qa_proposal"]["options"]
            ],
            temporal_contract["value_surfaces"],
        )
        for option, event_id in zip(
            captured[0]["output_schema"]["qa_proposal"]["options"], chain_ids
        ):
            self.assertEqual(option["source_event_instance_id"], event_id)
            self.assertEqual(option["anchor_ref"], event_id)
            self.assertIn("truth_value", option)
        query_plan = captured[0]["output_schema"]["qa_proposal"]["query_plan"]
        self.assertEqual(query_plan["reasoning_anchor_path"], chain_ids)
        self.assertEqual(
            query_plan["gold_anchor_refs"],
            [temporal_contract["current_event_instance_id"]],
        )
        self.assertNotIn("temporal_state_update_graph_contract", query_plan)

    def test_temporal_qa_prompt_binds_reasoning_identity_and_forbids_max_time_only(self) -> None:
        target = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        policy = stage3_policy("temporal_state_update")
        policy["stage64_target"] = copy.deepcopy(target)
        plan = subject.normalize_current_event_plan(
            {"events": exact_temporal_anti_shortcut_generation_events()},
            seed_row(),
            question_type="single_choice",
            attack_skill="conditional_plan",
            policy=policy,
            sample_index=0,
            round_id="stage3",
        )
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_qa_repair(
                plan,
                {},
                {},
                target,
                "initial_final_qa_generation_after_frozen_alignment",
                [],
                SimpleNamespace(),
                mode="final_generation",
            )

        prompt = captured[0]
        schema = prompt["output_schema"]["qa_proposal"]
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(prompt["target"][field], target[field])
            self.assertEqual(schema[field], target[field])
            self.assertEqual(schema["query_plan"][field], target[field])
        self.assertIn("canceled", schema["question"].lower())
        self.assertIn("max-time-only shortcut", " ".join(prompt["requirements"]))

    def test_formal_final_qa_is_generated_only_with_frozen_dependencies(self) -> None:
        captured = []
        hashes = {
            "graph_snapshot_hash": "graph-hash",
            "dialogue_hash": "dialogue-hash",
            "alignment_hash": "alignment-hash",
        }
        appended = [{"speaker": "Maya", "text": "Maya selected the north route for delivery."}]
        original = copy.deepcopy(appended)

        def respond(_args, prompt, **_kwargs):
            captured.append(copy.deepcopy(prompt))
            return {"qa_proposal": {"question": "Which route did Maya select?"}}, {
                "used": True,
                "attempts": 1,
            }

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            proposal, meta = subject.call_api_stage64_final_qa(
                self._formal_segment_plan(),
                {},
                {"qa_operation": "relation_lookup", "preferred_answer_type": "ObjectOrCommonNoun"},
                appended,
                hashes,
                SimpleNamespace(),
                item_difficulty_target="harder_boundary",
            )

        self.assertTrue(meta["used"])
        self.assertEqual(meta["call_kind"], "api_stage64_final_qa")
        self.assertEqual(captured[0]["frozen_dependencies"], hashes)
        self.assertIn("frozen dialogue alignment", captured[0]["task"])
        self.assertEqual(appended, original)
        self.assertIn("qa_proposal", proposal)

        missing, missing_meta = subject.call_api_stage64_final_qa(
            self._formal_segment_plan(),
            {},
            {"qa_operation": "relation_lookup"},
            appended,
            {"graph_snapshot_hash": "graph-only"},
            SimpleNamespace(),
        )
        self.assertIsNone(missing)
        self.assertEqual(missing_meta["error_type"], "unfrozen_final_qa_dependencies")

    def test_final_qa_event_candidates_preserve_comparison_integer_values_and_time(self) -> None:
        events = [
            {
                "event_id": "E1",
                "event_instance_id": "stage3:E1",
                "entity": "Alice",
                "slot": "attendance count",
                "value": 7,
                "status": "active",
                "time_index": 3,
                "depends_on": ["E0"],
                "scope": {"visible_to": ["Alice", "Bob"]},
                "required_for_dialogue": True,
            },
            {
                "event_id": "E2",
                "event_instance_id": "stage3:E2",
                "entity": "Bob",
                "slot": "attendance count",
                "value": 4,
                "status": "active",
                "time_index": 3,
                "previous_value": 2,
                "required_for_dialogue": True,
            },
        ]
        plan = {
            "events": events,
            "current_event_instance_to_anchor": {"stage3:E1": "M1", "stage3:E2": "M2"},
        }

        candidates = subject._stage64_event_candidates_for_prompt(plan, {})

        self.assertEqual([row["value"] for row in candidates], [7, 4])
        self.assertTrue(all(isinstance(row["value"], int) for row in candidates))
        self.assertEqual([row["time_index"] for row in candidates], [3, 3])
        self.assertEqual(candidates[0]["depends_on"], ["E0"])
        self.assertEqual(candidates[0]["scope"], {"visible_to": ["Alice", "Bob"]})
        self.assertEqual(candidates[1]["previous_value"], 2)

    def test_bridge_qa_repair_requires_nonconflicting_canonical_subtype(self) -> None:
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_qa_repair(
                {"events": []},
                {},
                {"qa_operation": "bridge", "qa_subtype": "relation_lookup"},
                {"qa_operation": "bridge", "preferred_answer_type": "Event"},
                "qa_subtype_conflicts_with_operation",
                [],
                SimpleNamespace(),
            )

        requirements = " ".join(captured[0]["requirements"])
        self.assertIn("exactly to 'multi_hop_dependency'", requirements)
        self.assertEqual(captured[0]["target"]["required_qa_subtype"], "multi_hop_dependency")

    def test_comparison_qa_repair_requires_explicit_two_anchor_comparison(self) -> None:
        captured: list[dict] = []
        plan, _graph_update, alignment, previous = self._run111_comparison_duplicate_fixture(
            use_primary_pair=False
        )

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_qa_repair(
                plan,
                alignment,
                previous,
                {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                "qa_operation_question_mismatch",
                [],
                SimpleNamespace(),
            )

        requirements = " ".join(captured[0]["requirements"])
        self.assertIn("literal word 'compare' or 'difference'", requirements)
        self.assertIn("exactly those same two event_candidates", requirements)
        self.assertIn("Gold must equal abs(left.value-right.value)", requirements)
        self.assertEqual(
            captured[0]["output_schema"]["qa_proposal"]["question"],
            "question naming 'Jon' and 'Gina', explicitly naming the 'business idea score' semantic slot, and asking for their absolute difference",
        )

    def test_comparison_qa_repair_missing_unique_contract_fails_before_api(self) -> None:
        with mock.patch.object(subject, "call_raw_json_api") as api:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "requires one frozen unique comparison_graph_contract",
            ):
                subject.call_api_stage64_qa_repair(
                    {"events": []},
                    {},
                    {"qa_operation": "comparison"},
                    {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                    "qa_operation_question_mismatch",
                    [],
                    SimpleNamespace(),
                )
        api.assert_not_called()

    @staticmethod
    def _run111_comparison_duplicate_fixture(
        *,
        use_primary_pair: bool,
        gold_text: str = "2",
    ) -> tuple[dict, dict, dict, dict]:
        prefix = "current_method_stage3_v9_segmented_comparison_smoke:S0000:"
        event_specs = (
            ("E01", "Jon", "business_idea_score", 1, 1),
            ("E02", "Gina", "business_idea_score", 3, 1),
            ("E03", "Jon", "employment_change_count", 1, 2),
            ("E04", "Gina", "store_planning_sessions", 5, 3),
            ("E05", "John", "kickboxing_sessions_per_week", 4, 4),
            ("E06", "Maria", "aerial_yoga_sessions_per_week", 2, 5),
            ("E07", "Jon", "confidence_level", 6, 6),
            ("E08", "Gina", "community_event_count", 8, 7),
        )
        events = []
        alignment = {}
        for index, (short_id, entity, slot, value, time_index) in enumerate(
            event_specs,
            start=1,
        ):
            event_id = f"{prefix}{short_id}"
            events.append(
                {
                    "event_id": short_id,
                    "event_instance_id": event_id,
                    "event_type": "state_change",
                    "entity": entity,
                    "slot": slot,
                    "value": value,
                    "status": "active",
                    "time_index": time_index,
                    "required_for_dialogue": True,
                }
            )
            alignment[event_id] = {"turn_ids": [f"D2:{index}"]}
        current_event_plan = {
            "operation": "comparison",
            "events": copy.deepcopy(events),
            "graph_snapshot_hash": "run111-graph",
        }
        graph_update = {
            "graph_snapshot_id": "run111-comparison",
            "operation": "comparison",
            "events": copy.deepcopy(events),
        }
        primary_refs = [f"{prefix}E01", f"{prefix}E02"]
        wrong_context_refs = [f"{prefix}E05", f"{prefix}E06"]
        selected_refs = primary_refs if use_primary_pair else wrong_context_refs
        selected_entities = ("Jon", "Gina") if use_primary_pair else ("John", "Maria")
        selected_slot = (
            "business idea score" if use_primary_pair else "weekly activity scores"
        )
        question = (
            f"What is the absolute difference between {selected_entities[0]} and "
            f"{selected_entities[1]} in their {selected_slot}?"
        )
        proposal = {
            "question_type": "single_choice",
            "question": question,
            "question_text": question,
            "qa_operation": "comparison",
            "qa_subtype": "entity_comparison",
            "answer_type": "Number",
            "answer_format": "single_choice_mcq",
            "gold": ["A"],
            "gold_answer_text": gold_text,
            "options": [
                {
                    "id": label,
                    "text": text,
                    "answer_type": "Number",
                    "anchor_kind": "event",
                    "anchor_ref": selected_refs[index % 2],
                    "source_event_instance_id": selected_refs[index % 2],
                    "truth_value": index == 0,
                }
                for index, (label, text) in enumerate(
                    zip("ABCD", (gold_text, gold_text, "1", "1"))
                )
            ],
            "anchor_refs": {
                "support": selected_refs,
                "gold": selected_refs,
                "distractors": selected_refs,
            },
            "query_plan": {
                "answer_source": "graph_anchor",
                "qa_operation": "comparison",
                "qa_subtype": "entity_comparison",
                "answer_type": "Number",
                "reasoning_anchor_path": selected_refs,
                "required_anchor_refs": selected_refs,
                "supporting_anchor_refs": selected_refs,
                "gold_anchor_refs": selected_refs,
                "distractor_anchor_refs": selected_refs,
                "gold_answer": gold_text,
                "normalized_gold_answer": gold_text,
            },
        }
        return current_event_plan, graph_update, alignment, proposal

    @classmethod
    def _run123_missing_top_level_schema_fixture(
        cls,
        *,
        gold_text: str = "2",
    ) -> tuple[dict, dict, dict, dict]:
        plan, graph_update, alignment, proposal = (
            cls._run111_comparison_duplicate_fixture(
                use_primary_pair=True,
                gold_text=gold_text,
            )
        )
        plan["question_type"] = "single_choice"
        for field in (
            "qa_operation",
            "qa_subtype",
            "question_type",
            "question_text",
        ):
            proposal.pop(field)
        return plan, graph_update, alignment, proposal

    @staticmethod
    def _run125_missing_query_plan_fixture() -> tuple[dict, dict, dict, dict]:
        prefix = "current_method_stage3_v12_frozenqa_tcc_smoke:S0001:"
        event_specs = (
            ("E01", "Melanie", "project_budget", 250000, 1),
            ("E02", "Caroline", "project_budget", 320000, 1),
            ("E03", "Melanie", "team_size", 12, 2),
            ("E04", "Caroline", "team_size", 15, 3),
            ("E05", "Melanie", "deadline", 120, 4),
            ("E06", "Caroline", "deadline", 100, 5),
            ("E07", "Melanie", "approval_status", 1, 6),
            ("E08", "Caroline", "approval_status", 0, 7),
        )
        events = []
        alignment = {}
        for index, (short_id, entity, slot, value, time_index) in enumerate(
            event_specs,
            start=1,
        ):
            event_id = f"{prefix}{short_id}"
            events.append(
                {
                    "event_id": short_id,
                    "event_instance_id": event_id,
                    "entity": entity,
                    "slot": slot,
                    "status": "active",
                    "time_index": time_index,
                    "value": value,
                    "required_for_dialogue": True,
                }
            )
            alignment[event_id] = {"turn_ids": [f"D2:{index}"]}
        plan = {
            "operation": "comparison",
            "question_type": "single_choice",
            "events": copy.deepcopy(events),
        }
        graph_update = {
            "operation": "comparison",
            "graph_snapshot_id": "run125-comparison",
            "events": copy.deepcopy(events),
        }
        primary_ids = [f"{prefix}E01", f"{prefix}E02"]
        question = (
            "What is the absolute difference between Melanie's and Caroline's "
            "project budgets?"
        )
        proposal = {
            "anchor_refs": {
                "distractors": primary_ids,
                "gold": primary_ids,
                "support": primary_ids,
            },
            "answer_format": "single_choice_mcq",
            "answer_type": "Number",
            "gold": ["A"],
            "gold_answer_text": "70000",
            "options": [
                {
                    "id": label,
                    "text": text,
                    "answer_type": "Number",
                    "source_event_instance_id": primary_ids[index % 2],
                    "anchor_ref": primary_ids[index % 2],
                    "anchor_kind": "event",
                    "truth_value": index == 0,
                }
                for index, (label, text) in enumerate(
                    zip("ABCD", ("70000", "69999", "70001", "70002"))
                )
            ],
            "qa_operation": "comparison",
            "qa_subtype": "entity_comparison",
            "question": question,
            "question_text": question,
        }
        return plan, graph_update, alignment, proposal

    def test_run125_missing_query_plan_completes_locally_without_repair_api(
        self,
    ) -> None:
        plan, graph_update, alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        originals = copy.deepcopy((plan, graph_update, alignment, proposal))
        hashes = {
            "graph_snapshot_hash": "run125-graph",
            "dialogue_hash": "run125-dialogue",
            "alignment_hash": "run125-alignment",
        }
        audit: dict = {}
        with mock.patch.object(subject, "call_api_stage64_qa_repair") as repair:
            qa, chosen = subject.build_stage64_qa_with_local_repairs(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                initial_qa_proposal=proposal,
                question_type="single_choice",
                rng=random.Random(125),
                hashes=hashes,
                final_dialogue={"turns": []},
                policy={
                    "qa_operation_answer_type_diversification": True,
                    "stage64_target": {
                        "qa_operation": "comparison",
                        "preferred_answer_type": "Number",
                    },
                },
                seed_row={},
                appended_dialogue_turns=[],
                api_args=SimpleNamespace(),
                audit=audit,
            )

        repair.assert_not_called()
        self.assertEqual((plan, graph_update, alignment, proposal), originals)
        self.assertEqual(chosen, proposal)
        self.assertNotIn("query_plan", proposal)
        self.assertEqual(qa["question_type"], "single_choice")
        self.assertEqual(qa["query_plan"]["gold_answer"], "70000")
        self.assertEqual(
            qa["query_plan"]["reasoning_anchor_path"],
            [
                "current_method_stage3_v12_frozenqa_tcc_smoke:S0001:E01",
                "current_method_stage3_v12_frozenqa_tcc_smoke:S0001:E02",
            ],
        )
        completion = qa["stage64_query_plan_completion_normalizer"]
        self.assertTrue(
            completion["stage64_query_plan_completion_normalizer_used"]
        )
        self.assertEqual(
            completion["contract_sha256"],
            "5ec5a17cdc726275a5c4b13261cb2c82fe96054e9d8bb562fb4f106999c7503f",
        )
        self.assertEqual(completion["completed_fields"], ["query_plan"])
        self.assertTrue(completion["only_query_plan_added"])
        self.assertNotEqual(
            completion["pre_qa_proposal_sha256"],
            completion["post_qa_proposal_sha256"],
        )
        self.assertEqual(audit["repair_attempt_count"], 0)
        self.assertTrue(audit["success"])
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {
                    "qa": qa,
                    "graph_update": graph_update,
                    "current_event_plan": plan,
                }
            ),
            [],
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "comparison",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_query_plan_completion_normalizer_used_count"],
            1,
        )
        self.assertEqual(
            summary["stage64_query_plan_completion_normalizer_by_operation"],
            {"comparison": 1},
        )
        self.assertEqual(
            summary[
                "stage64_query_plan_completion_normalizer_completed_field_counts"
            ],
            {"query_plan": 1},
        )
        self.assertEqual(
            summary["stage64_query_plan_completion_question_validated_count"],
            1,
        )
        self.assertEqual(
            summary["stage64_query_plan_completion_options_validated_count"],
            1,
        )
        self.assertEqual(
            summary["stage64_query_plan_completion_gold_validated_count"],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_query_plan_completion_anchor_refs_validated_count"
            ],
            1,
        )

    def test_missing_top_level_identity_bootstraps_absent_query_plan_from_frozen_target(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        proposal.pop("qa_operation")
        proposal.pop("qa_subtype")
        original = copy.deepcopy(proposal)

        normalized, diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "comparison"},
            )
        )

        self.assertEqual(proposal, original)
        self.assertEqual(normalized["qa_operation"], "comparison")
        self.assertEqual(normalized["qa_subtype"], "entity_comparison")
        self.assertEqual(
            normalized["query_plan"]["qa_operation"], "comparison"
        )
        self.assertEqual(
            normalized["query_plan"]["qa_subtype"], "entity_comparison"
        )
        self.assertEqual(
            diagnostics["completed_top_level_fields"],
            ["qa_operation", "qa_subtype"],
        )
        self.assertEqual(
            diagnostics["completed_fields"],
            ["qa_proposal.qa_operation", "qa_proposal.qa_subtype", "query_plan"],
        )
        self.assertFalse(diagnostics["only_query_plan_added"])

    def test_missing_top_level_identity_bootstrap_refuses_unfrozen_operation(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        proposal.pop("qa_operation")
        proposal.pop("qa_subtype")
        plan["operation"] = "count"
        graph_update["operation"] = "count"

        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "refuses to guess missing API fields",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "comparison"},
            )

    def test_missing_query_plan_completion_accepts_unique_count_and_temporal_contracts(
        self,
    ) -> None:
        count_plan, count_graph, _count_alignment, count_proposal = (
            self._run116_count_contract_fixture()
        )
        count_plan["operation"] = "count"
        count_proposal.pop("query_plan")

        temporal_plan, temporal_graph, _temporal_alignment, temporal_proposal = (
            self._run121_temporal_reference_drift_fixture(
                already_correct=True
            )
        )
        temporal_proposal.pop("query_plan")
        for label, option in zip("ABCD", temporal_proposal["options"]):
            option["truth_value"] = label == "D"

        cases = (
            (
                "count",
                count_plan,
                count_graph,
                count_proposal,
                "4",
                [event["event_instance_id"] for event in count_plan["events"]],
            ),
            (
                "temporal_state_update",
                temporal_plan,
                temporal_graph,
                temporal_proposal,
                "Public",
                [
                    event["event_instance_id"]
                    for event in temporal_plan["events"][:4]
                ],
            ),
        )
        for operation, plan, graph, proposal, answer, reasoning_ids in cases:
            with self.subTest(operation=operation):
                original = copy.deepcopy(proposal)
                normalized, diagnostics = (
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph,
                        proposal,
                        {"qa_operation": operation},
                    )
                )
                self.assertEqual(proposal, original)
                self.assertTrue(
                    diagnostics[
                        "stage64_query_plan_completion_normalizer_used"
                    ]
                )
                self.assertEqual(diagnostics["operation"], operation)
                self.assertEqual(
                    normalized["query_plan"]["reasoning_anchor_path"],
                    reasoning_ids,
                )
                self.assertEqual(
                    normalized["query_plan"]["gold_answer"], answer
                )
                self.assertEqual(
                    normalized["query_plan"]["dependency_closure_refs"],
                    reasoning_ids,
                )
                self.assertEqual(
                    normalized["query_plan"]["answer_projection"],
                    subject.STAGE64_ANSWER_PROJECTION_BY_OPERATION[operation],
                )
                self.assertFalse(
                    any(
                        field.startswith("resolved_")
                        for field in normalized["query_plan"]
                    )
                )

    def test_missing_query_plan_completion_refuses_null_and_completes_partial_plan(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        explicit_null = copy.deepcopy(proposal)
        explicit_null["query_plan"] = None
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "explicit non-object query_plan",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                explicit_null,
                {"qa_operation": "comparison"},
            )

        for name, explicit_query_plan in (
            ("empty", {}),
            ("partial", {"answer_source": "graph_anchor"}),
        ):
            with self.subTest(name=name):
                candidate = copy.deepcopy(proposal)
                candidate["query_plan"] = explicit_query_plan
                original = copy.deepcopy(candidate)
                normalized, diagnostics = (
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph_update,
                        candidate,
                        {"qa_operation": "comparison"},
                    )
                )
                self.assertEqual(candidate, original)
                self.assertEqual(
                    normalized["query_plan"]["answer_source"],
                    "graph_anchor",
                )
                self.assertEqual(
                    normalized["query_plan"]["gold_answer"],
                    "70000",
                )
                self.assertTrue(
                    diagnostics["only_missing_query_plan_fields_added"]
                )
                if name == "partial":
                    self.assertNotIn(
                        "answer_source",
                        diagnostics["completed_query_plan_fields"],
                        msg="an explicit correct field must be preserved, not completed",
                    )

    def test_partial_query_plan_rejects_conflicts_and_explicit_resolved_drift(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        complete, _diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "comparison"},
            )
        )
        for field, wrong_value in (
            ("answer_source", "direct"),
            ("qa_operation", "count"),
            ("qa_subtype", "filtered_count"),
            ("answer_type", "Event"),
            ("gold_answer", "1"),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(complete)
                candidate["query_plan"][field] = wrong_value
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    f"query_plan.{field}",
                ):
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph_update,
                        candidate,
                        {"qa_operation": "comparison"},
                    )

        resolved_conflict = copy.deepcopy(complete)
        resolved_conflict["query_plan"][
            "resolved_reasoning_event_instance_ids"
        ] = [plan["events"][2]["event_instance_id"]]
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "resolved_reasoning_event_instance_ids",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                resolved_conflict,
                {"qa_operation": "comparison"},
            )

    def test_partial_temporal_query_plan_requires_chronological_reasoning_path(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run121_temporal_reference_drift_fixture(already_correct=True)
        )
        candidate = copy.deepcopy(proposal)
        candidate["query_plan"].pop("supporting_anchor_refs")
        candidate["query_plan"]["reasoning_anchor_path"] = list(
            reversed(candidate["query_plan"]["reasoning_anchor_path"])
        )
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "query_plan.reasoning_anchor_path",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                candidate,
                {"qa_operation": "temporal_state_update"},
            )

    def test_query_plan_completion_derives_safe_top_level_schema_fields(self) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run116_count_contract_fixture()
        )
        plan["operation"] = "count"
        graph_update["operation"] = "count"
        candidate = copy.deepcopy(proposal)
        candidate.pop("query_plan")
        candidate.pop("answer_type")
        candidate.pop("answer_format")

        normalized, diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                candidate,
                {"qa_operation": "count"},
            )
        )

        self.assertNotIn("answer_type", candidate)
        self.assertEqual(normalized["answer_type"], "Number")
        self.assertEqual(normalized["answer_format"], "single_choice_mcq")
        self.assertEqual(
            diagnostics["completed_top_level_fields"],
            ["answer_type", "answer_format"],
        )

    def test_generic_partial_query_plan_completes_only_required_supporting_refs(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        ids = list(proposal["anchor_refs"]["support"])
        plan["operation"] = "bridge"
        graph_update["operation"] = "bridge"
        proposal["qa_operation"] = "bridge"
        proposal["qa_subtype"] = "multi_hop_dependency"
        proposal["query_plan"] = {
            "answer_source": "graph_anchor",
            "qa_operation": "bridge",
            "qa_subtype": "multi_hop_dependency",
            "answer_type": "Number",
            "reasoning_anchor_path": ids,
            "gold_anchor_refs": ids,
            "distractor_anchor_refs": ids,
            "gold_answer": "70000",
            "normalized_gold_answer": "70000",
        }
        original = copy.deepcopy(proposal)

        normalized, diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "bridge"},
            )
        )

        self.assertEqual(proposal, original)
        self.assertEqual(normalized["query_plan"]["required_anchor_refs"], ids)
        self.assertEqual(normalized["query_plan"]["supporting_anchor_refs"], ids)
        self.assertEqual(
            diagnostics["contract_schema_version"],
            "generic_partial_reference_completion_v1",
        )

        absent = copy.deepcopy(proposal)
        absent.pop("query_plan")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "does not support operation 'bridge'",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                absent,
                {"qa_operation": "bridge"},
            )

    def test_query_plan_preserves_equivalent_normalized_gold_surface(self) -> None:
        self.assertTrue(
            subject._stage64_query_plan_completion_scalar_matches(
                "normalized_gold_answer",
                "director's office",
                "Director's Office",
            )
        )
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        complete, _diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "comparison"},
            )
        )
        candidate = copy.deepcopy(complete)
        candidate["query_plan"]["normalized_gold_answer"] = "70000"
        normalized, diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                candidate,
                {"qa_operation": "comparison"},
            )
        )
        self.assertIs(normalized, candidate)
        self.assertEqual(
            diagnostics["noop_reason"],
            "query_plan_already_present",
        )

    def test_query_plan_completion_is_noop_for_complete_plan_and_rejects_unsupported_or_mismatched_operation(
        self,
    ) -> None:
        complete_plan, complete_graph, _alignment, complete_proposal = (
            self._run116_count_contract_fixture()
        )
        complete_plan["operation"] = "count"
        original = copy.deepcopy(complete_proposal)
        normalized, diagnostics = (
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                complete_plan,
                complete_graph,
                complete_proposal,
                {"qa_operation": "count"},
            )
        )
        self.assertIs(normalized, complete_proposal)
        self.assertEqual(complete_proposal, original)
        self.assertFalse(
            diagnostics["stage64_query_plan_completion_normalizer_used"]
        )
        self.assertEqual(
            diagnostics["noop_reason"], "query_plan_already_present"
        )

        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )
        unsupported = copy.deepcopy(proposal)
        unsupported["qa_operation"] = "visibility"
        unsupported["qa_subtype"] = "permission_scope"
        unsupported_plan = copy.deepcopy(plan)
        unsupported_graph = copy.deepcopy(graph_update)
        unsupported_plan["operation"] = "visibility"
        unsupported_graph["operation"] = "visibility"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "does not support operation 'visibility'",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                unsupported_plan,
                unsupported_graph,
                unsupported,
                {"qa_operation": "visibility"},
            )

        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "operation evidence mismatch",
        ):
            subject.normalize_stage64_missing_query_plan_from_unique_contract(
                plan,
                graph_update,
                proposal,
                {"qa_operation": "count"},
            )

    def test_missing_query_plan_completion_requires_one_unique_operation_contract(
        self,
    ) -> None:
        comparison_plan, comparison_graph, _alignment, comparison_proposal = (
            self._run125_missing_query_plan_fixture()
        )
        comparison_plan["events"][3]["time_index"] = 2
        comparison_graph["events"][3]["time_index"] = 2

        count_plan, count_graph, _alignment, count_proposal = (
            self._run116_count_contract_fixture()
        )
        count_plan["operation"] = "count"
        count_plan["events"][0]["entity"] = "SomeoneElse"
        count_proposal.pop("query_plan")

        temporal_plan, temporal_graph, _alignment, temporal_proposal = (
            self._run121_temporal_reference_drift_fixture(
                already_correct=True
            )
        )
        temporal_proposal.pop("query_plan")
        for label, option in zip("ABCD", temporal_proposal["options"]):
            option["truth_value"] = label == "D"
        second_values = ("One", "Two", "Three", "Four")
        second_statuses = ("outdated", "canceled", "expired", "active")
        for event_group in (
            temporal_plan["events"],
            temporal_graph["events"],
        ):
            for index, event in enumerate(event_group[4:8]):
                event["entity"] = "SecondProject"
                event["slot"] = "delivery_mode"
                event["value"] = second_values[index]
                event["status"] = second_statuses[index]
                if index == 0:
                    event.pop("previous_value", None)
                else:
                    event["previous_value"] = second_values[index - 1]

        for operation, plan, graph, proposal, error in (
            (
                "comparison",
                comparison_plan,
                comparison_graph,
                comparison_proposal,
                "comparison contract is not unique",
            ),
            (
                "count",
                count_plan,
                count_graph,
                count_proposal,
                "count contract is not unique",
            ),
            (
                "temporal_state_update",
                temporal_plan,
                temporal_graph,
                temporal_proposal,
                "temporal contract is not unique",
            ),
        ):
            with self.subTest(operation=operation):
                original = copy.deepcopy(proposal)
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    error,
                ):
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph,
                        proposal,
                        {"qa_operation": operation},
                    )
                self.assertEqual(proposal, original)

    def test_missing_query_plan_completion_refuses_wrong_comparison_api_fields(
        self,
    ) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run125_missing_query_plan_fixture()
        )

        def wrong_question(candidate: dict) -> None:
            question = candidate["question"].replace("Caroline", "Alice")
            candidate["question"] = question
            candidate["question_text"] = question

        def wrong_gold(candidate: dict) -> None:
            candidate["options"][0]["text"] = "69998"
            candidate["gold_answer_text"] = "69998"

        def wrong_truth(candidate: dict) -> None:
            candidate["options"][1]["truth_value"] = True

        def wrong_option_source(candidate: dict) -> None:
            context_id = (
                "current_method_stage3_v12_frozenqa_tcc_smoke:S0001:E03"
            )
            candidate["options"][1]["source_event_instance_id"] = context_id
            candidate["options"][1]["anchor_ref"] = context_id

        def wrong_anchors(candidate: dict) -> None:
            candidate["anchor_refs"]["support"] = candidate["anchor_refs"][
                "support"
            ][:1]

        cases = (
            ("question", wrong_question, "question does not identify"),
            ("gold", wrong_gold, "options/gold do not match"),
            ("truth", wrong_truth, "gold label and truth_value"),
            ("option_source", wrong_option_source, "options/gold do not match"),
            ("anchors", wrong_anchors, "anchor_refs do not match"),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                candidate = copy.deepcopy(proposal)
                mutate(candidate)
                original = copy.deepcopy(candidate)
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    expected_error,
                ):
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph_update,
                        candidate,
                        {"qa_operation": "comparison"},
                    )
                self.assertEqual(candidate, original)

    def test_missing_query_plan_completion_refuses_wrong_count_and_temporal_gold(
        self,
    ) -> None:
        count_plan, count_graph, _alignment, count_proposal = (
            self._run116_count_contract_fixture()
        )
        count_plan["operation"] = "count"
        count_proposal.pop("query_plan")
        count_proposal["options"][0]["text"] = "3"
        count_proposal["options"][1]["text"] = "4"
        count_proposal["gold_answer_text"] = "3"

        temporal_plan, temporal_graph, _alignment, temporal_proposal = (
            self._run121_temporal_reference_drift_fixture(
                already_correct=True
            )
        )
        temporal_proposal.pop("query_plan")
        temporal_proposal["gold"] = ["A"]
        temporal_proposal["gold_answer_text"] = "Internal"
        for label, option in zip("ABCD", temporal_proposal["options"]):
            option["truth_value"] = label == "A"

        for operation, plan, graph, proposal, expected_error in (
            (
                "count",
                count_plan,
                count_graph,
                count_proposal,
                "count options/gold do not match",
            ),
            (
                "temporal_state_update",
                temporal_plan,
                temporal_graph,
                temporal_proposal,
                "temporal gold is not the unique current state",
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    expected_error,
                ):
                    subject.normalize_stage64_missing_query_plan_from_unique_contract(
                        plan,
                        graph,
                        proposal,
                        {"qa_operation": operation},
                    )

    def test_run123_schema_completion_recovers_only_four_missing_redundant_fields(
        self,
    ) -> None:
        plan, graph_update, alignment, proposal = (
            self._run123_missing_top_level_schema_fixture()
        )
        original = copy.deepcopy(proposal)
        target = {
            "qa_operation": "comparison",
            "preferred_answer_type": "Number",
        }

        normalized, diagnostics = subject.normalize_stage64_qa_schema_completion(
            plan,
            graph_update,
            proposal,
            target,
        )

        self.assertEqual(proposal, original)
        self.assertTrue(
            diagnostics["stage64_qa_schema_completion_normalizer_used"]
        )
        self.assertEqual(
            diagnostics["completed_fields"],
            [
                "qa_operation",
                "qa_subtype",
                "question_type",
                "question_text",
            ],
        )
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            subject._stage64_json_sha256(original),
        )
        self.assertEqual(
            diagnostics["post_qa_proposal_sha256"],
            subject._stage64_json_sha256(normalized),
        )
        self.assertNotEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )
        self.assertEqual(normalized["qa_operation"], "comparison")
        self.assertEqual(normalized["qa_subtype"], "entity_comparison")
        self.assertEqual(normalized["question_type"], "single_choice")
        self.assertEqual(normalized["question_text"], normalized["question"])
        self.assertEqual(
            diagnostics["completion_evidence"]["current_event_plan_operation"],
            "comparison",
        )
        self.assertTrue(diagnostics["completion_evidence_sha256"])

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run123-graph",
                "dialogue_hash": "run123-dialogue",
                "alignment_hash": "run123-alignment",
            },
            stage64_target=target,
        )

        self.assertEqual(proposal, original)
        self.assertEqual(qa["qa_operation"], "comparison")
        self.assertEqual(qa["qa_subtype"], "entity_comparison")
        self.assertEqual(qa["question_type"], "single_choice")
        self.assertEqual(qa["question_text"], qa["question"])
        self.assertEqual(
            qa["stage64_qa_schema_completion_normalizer"]["completed_fields"],
            diagnostics["completed_fields"],
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "comparison",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_qa_schema_completion_normalizer_used_count"],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_normalizer_completed_field_count"
            ],
            4,
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_normalizer_completed_field_counts"
            ],
            {
                "qa_operation": 1,
                "qa_subtype": 1,
                "question_type": 1,
                "question_text": 1,
            },
        )

    def test_run123_schema_completion_requires_all_operation_evidence_to_match(
        self,
    ) -> None:
        for mismatch_source in ("target", "plan", "graph_update", "query_plan"):
            with self.subTest(mismatch_source=mismatch_source):
                plan, graph_update, _alignment, proposal = (
                    self._run123_missing_top_level_schema_fixture()
                )
                target = {"qa_operation": "comparison"}
                if mismatch_source == "target":
                    target["qa_operation"] = "count"
                elif mismatch_source == "plan":
                    plan["operation"] = "count"
                elif mismatch_source == "graph_update":
                    graph_update["operation"] = "count"
                else:
                    proposal["query_plan"]["qa_operation"] = "count"
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    "operation evidence mismatch",
                ):
                    subject.normalize_stage64_qa_schema_completion(
                        plan,
                        graph_update,
                        proposal,
                        target,
                    )

    def test_run123_schema_completion_refuses_missing_or_conflicting_evidence(
        self,
    ) -> None:
        cases = (
            (
                "subtype",
                lambda plan, graph, proposal: proposal["query_plan"].__setitem__(
                    "qa_subtype", "filtered_count"
                ),
                "subtype evidence mismatch",
            ),
            (
                "question_type",
                lambda plan, graph, proposal: plan.__setitem__(
                    "question_type", "ordering"
                ),
                "question_type evidence mismatch",
            ),
            (
                "answer_format",
                lambda plan, graph, proposal: proposal.__setitem__(
                    "answer_format", "free_text"
                ),
                "answer_format evidence mismatch",
            ),
            (
                "question",
                lambda plan, graph, proposal: proposal.pop("question"),
                "requires one nonempty question string",
            ),
            (
                "explicit_null",
                lambda plan, graph, proposal: proposal.__setitem__(
                    "qa_operation", None
                ),
                "refuses to overwrite explicit qa_operation",
            ),
            (
                "conflicting_question_text",
                lambda plan, graph, proposal: proposal.__setitem__(
                    "question_text", "A different question"
                ),
                "refuses conflicting question/question_text aliases",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                plan, graph_update, _alignment, proposal = (
                    self._run123_missing_top_level_schema_fixture()
                )
                mutate(plan, graph_update, proposal)
                plan_before = copy.deepcopy(plan)
                graph_update_before = copy.deepcopy(graph_update)
                proposal_before = copy.deepcopy(proposal)
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    expected_error,
                ):
                    subject.normalize_stage64_qa_schema_completion(
                        plan,
                        graph_update,
                        proposal,
                        {"qa_operation": "comparison"},
                    )
                self.assertEqual(plan, plan_before)
                self.assertEqual(graph_update, graph_update_before)
                self.assertEqual(proposal, proposal_before)

    def test_stage64_schema_completion_is_noop_for_complete_schema(self) -> None:
        plan, graph_update, _alignment, proposal = (
            self._run111_comparison_duplicate_fixture(use_primary_pair=True)
        )

        normalized, diagnostics = subject.normalize_stage64_qa_schema_completion(
            plan,
            graph_update,
            proposal,
            {"qa_operation": "comparison"},
        )

        self.assertIs(normalized, proposal)
        self.assertFalse(
            diagnostics["stage64_qa_schema_completion_normalizer_used"]
        )
        self.assertEqual(
            diagnostics["noop_reason"], "schema_fields_already_present"
        )
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )

    def test_run123_schema_completion_does_not_bypass_wrong_gold_gate(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run123_missing_top_level_schema_fixture(gold_text="1")
        )

        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "refused wrong gold option",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=proposal,
                hashes={
                    "graph_snapshot_hash": "run123-graph",
                    "dialogue_hash": "run123-dialogue",
                    "alignment_hash": "run123-alignment",
                },
                stage64_target={
                    "qa_operation": "comparison",
                    "preferred_answer_type": "Number",
                },
            )

    def test_stage64_builder_refuses_conflicting_complete_question_text(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run111_comparison_duplicate_fixture(use_primary_pair=True)
        )
        proposal["question_text"] = proposal["question"] + " "

        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "refuses conflicting question/question_text aliases",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=proposal,
                hashes={
                    "graph_snapshot_hash": "complete-graph",
                    "dialogue_hash": "complete-dialogue",
                    "alignment_hash": "complete-alignment",
                },
                stage64_target={"qa_operation": "comparison"},
            )

    def test_run124_comparison_initial_completes_missing_question_without_graph_update_operation(
        self,
    ) -> None:
        plan, graph_update, alignment, proposal = (
            self._run111_comparison_duplicate_fixture(use_primary_pair=True)
        )
        plan["question_type"] = "single_choice"
        graph_update.pop("operation")
        proposal.pop("question")
        original = copy.deepcopy(proposal)

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run124-comparison-graph",
                "dialogue_hash": "run124-comparison-dialogue",
                "alignment_hash": "run124-comparison-alignment",
            },
            stage64_target={
                "qa_operation": "comparison",
                "preferred_answer_type": "Number",
            },
        )

        self.assertEqual(proposal, original)
        self.assertEqual(qa["question"], original["question_text"])
        self.assertEqual(qa["question_text"], qa["question"])
        diagnostics = qa["stage64_qa_schema_completion_normalizer"]
        self.assertEqual(diagnostics["completed_fields"], ["question"])
        self.assertEqual(diagnostics["overwritten_fields"], [])
        self.assertEqual(diagnostics["question_alias_completed_field"], "question")
        self.assertFalse(
            diagnostics["completion_evidence"]["graph_update_operation_present"]
        )
        self.assertEqual(
            diagnostics["completion_evidence"]["graph_update_operation"], ""
        )

    def test_run124_temporal_repair_completes_question_and_repairs_qtype_confusion(
        self,
    ) -> None:
        plan, graph_update, alignment, proposal = (
            self._run121_temporal_reference_drift_fixture(already_correct=True)
        )
        plan["question_type"] = "single_choice"
        graph_update.pop("operation")
        proposal.pop("question")
        proposal["question_type"] = proposal["answer_type"]
        original = copy.deepcopy(proposal)

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run124-temporal-graph",
                "dialogue_hash": "run124-temporal-dialogue",
                "alignment_hash": "run124-temporal-alignment",
            },
            stage64_target={
                "qa_operation": "temporal_state_update",
                "preferred_answer_type": "LocationOrScope",
            },
        )

        self.assertEqual(proposal, original)
        self.assertEqual(qa["question_type"], "single_choice")
        self.assertEqual(qa["answer_type"], "LocationOrScope")
        self.assertEqual(qa["question"], original["question_text"])
        self.assertEqual(qa["question_text"], qa["question"])
        diagnostics = qa["stage64_qa_schema_completion_normalizer"]
        self.assertEqual(diagnostics["completed_fields"], ["question"])
        self.assertEqual(diagnostics["overwritten_fields"], ["question_type"])
        self.assertEqual(diagnostics["question_alias_completed_field"], "question")
        self.assertTrue(
            diagnostics["question_type_answer_type_confusion_repaired"]
        )
        self.assertEqual(diagnostics["question_type_before"], "LocationOrScope")
        self.assertEqual(diagnostics["question_type_after"], "single_choice")
        self.assertTrue(
            diagnostics["completion_evidence"]["four_object_options"]
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "temporal_state_update",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_qa_schema_completion_normalizer_used_count"], 1
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_normalizer_completed_field_counts"
            ],
            {"question": 1},
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_normalizer_overwritten_field_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_normalizer_overwritten_field_counts"
            ],
            {"question_type": 1},
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_question_alias_completed_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_qa_schema_completion_question_type_answer_type_confusion_repaired_count"
            ],
            1,
        )

    def test_run124_qtype_confusion_repair_is_fail_closed(self) -> None:
        cases = (
            (
                "unrelated_explicit_qtype",
                lambda plan, graph, proposal: proposal.__setitem__(
                    "question_type", "ordering"
                ),
                "refuses explicit question_type conflict",
            ),
            (
                "wrong_plan_qtype",
                lambda plan, graph, proposal: plan.__setitem__(
                    "question_type", "ordering"
                ),
                "question_type evidence mismatch",
            ),
            (
                "wrong_answer_format",
                lambda plan, graph, proposal: proposal.__setitem__(
                    "answer_format", "free_text"
                ),
                "answer_format evidence mismatch",
            ),
            (
                "three_options",
                lambda plan, graph, proposal: proposal["options"].pop(),
                "confusion lacks four object options",
            ),
            (
                "non_object_option",
                lambda plan, graph, proposal: proposal["options"].__setitem__(
                    0, "not-an-object"
                ),
                "confusion lacks four object options",
            ),
            (
                "wrong_query_operation",
                lambda plan, graph, proposal: proposal["query_plan"].__setitem__(
                    "qa_operation", "comparison"
                ),
                "operation evidence mismatch",
            ),
            (
                "wrong_query_subtype",
                lambda plan, graph, proposal: proposal["query_plan"].__setitem__(
                    "qa_subtype", "entity_comparison"
                ),
                "subtype evidence mismatch",
            ),
            (
                "present_blank_graph_operation",
                lambda plan, graph, proposal: graph.__setitem__("operation", ""),
                "operation evidence mismatch",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                plan, graph_update, _alignment, proposal = (
                    self._run121_temporal_reference_drift_fixture(
                        already_correct=True
                    )
                )
                plan["question_type"] = "single_choice"
                graph_update.pop("operation")
                proposal.pop("question")
                proposal["question_type"] = proposal["answer_type"]
                mutate(plan, graph_update, proposal)
                plan_before = copy.deepcopy(plan)
                graph_update_before = copy.deepcopy(graph_update)
                proposal_before = copy.deepcopy(proposal)
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    expected_error,
                ):
                    subject.normalize_stage64_qa_schema_completion(
                        plan,
                        graph_update,
                        proposal,
                        {"qa_operation": "temporal_state_update"},
                    )
                self.assertEqual(plan, plan_before)
                self.assertEqual(graph_update, graph_update_before)
                self.assertEqual(proposal, proposal_before)

    def test_run124_question_alias_completion_rejects_null_blank_and_conflict(
        self,
    ) -> None:
        cases = (
            ("missing_question_null_text", "question", None),
            ("missing_question_blank_text", "question", "  "),
            ("missing_question_text_null_question", "question_text", None),
            ("missing_question_text_blank_question", "question_text", ""),
        )
        for name, missing_field, remaining_value in cases:
            with self.subTest(name=name):
                plan, graph_update, _alignment, proposal = (
                    self._run111_comparison_duplicate_fixture(
                        use_primary_pair=True
                    )
                )
                plan["question_type"] = "single_choice"
                proposal.pop(missing_field)
                remaining_field = (
                    "question_text" if missing_field == "question" else "question"
                )
                proposal[remaining_field] = remaining_value
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    "null or blank",
                ):
                    subject.normalize_stage64_qa_schema_completion(
                        plan,
                        graph_update,
                        proposal,
                        {"qa_operation": "comparison"},
                    )

    def test_derive_unique_comparison_contract_is_read_only_and_durable(self) -> None:
        plan, _graph_update, _alignment, _proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=True
        )
        original = copy.deepcopy(plan)

        contract = subject.derive_unique_comparison_contract(plan)

        self.assertEqual(plan, original)
        self.assertIsNotNone(contract)
        self.assertEqual(
            contract["primary_event_instance_ids"],
            [
                "current_method_stage3_v9_segmented_comparison_smoke:S0000:E01",
                "current_method_stage3_v9_segmented_comparison_smoke:S0000:E02",
            ],
        )
        self.assertEqual(contract["entities"], ["Jon", "Gina"])
        self.assertEqual(contract["normalized_slot"], "business idea score")
        self.assertEqual(contract["time_index"], 1)
        self.assertEqual(contract["values"], [1, 3])
        self.assertEqual(contract["expected_absolute_difference"], 2)
        unhashed = {key: value for key, value in contract.items() if key != "contract_sha256"}
        self.assertEqual(contract["contract_sha256"], subject._stage64_json_sha256(unhashed))

        ambiguous = copy.deepcopy(plan)
        ambiguous["events"][4]["slot"] = "context_pair"
        ambiguous["events"][4]["time_index"] = 9
        ambiguous["events"][5]["slot"] = "context_pair"
        ambiguous["events"][5]["time_index"] = 9
        self.assertIsNone(subject.derive_unique_comparison_contract(ambiguous))

        missing_durable_id = copy.deepcopy(plan)
        missing_durable_id["events"][0].pop("event_instance_id")
        self.assertIsNone(subject.derive_unique_comparison_contract(missing_durable_id))

    def test_run111_comparison_prompt_binds_exact_frozen_pair_for_final_and_repair(self) -> None:
        plan, _graph_update, alignment, wrong_proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=False
        )
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        hashes = {
            "graph_snapshot_hash": "run111-graph",
            "dialogue_hash": "run111-dialogue",
            "alignment_hash": "run111-alignment",
        }
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_final_qa(
                plan,
                alignment,
                {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                [],
                hashes,
                SimpleNamespace(),
            )
            subject.call_api_stage64_qa_repair(
                plan,
                alignment,
                wrong_proposal,
                {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                "stage64 API QA exact duplicate option text",
                [],
                SimpleNamespace(),
            )

        self.assertEqual(len(captured), 2)
        pair_ids = [
            "current_method_stage3_v9_segmented_comparison_smoke:S0000:E01",
            "current_method_stage3_v9_segmented_comparison_smoke:S0000:E02",
        ]
        for prompt in captured:
            contract = prompt["comparison_graph_contract"]
            self.assertEqual(contract["primary_event_instance_ids"], pair_ids)
            self.assertEqual(contract["entities"], ["Jon", "Gina"])
            self.assertEqual(contract["expected_absolute_difference"], 2)
            self.assertEqual(contract["normalized_slot"], "business idea score")
            requirements = " ".join(prompt["requirements"])
            self.assertIn(
                "question_type is the MCQ schema type and must never equal or repeat",
                requirements,
            )
            self.assertIn(
                "question and question_text must be byte-identical",
                requirements,
            )
            self.assertIn("Jon", requirements)
            self.assertIn("Gina", requirements)
            self.assertIn("business idea score", requirements)
            self.assertIn("sole primary pair's semantic slot", requirements)
            self.assertIn("must each contain exactly these two primary", requirements)
            self.assertIn("must each equal the exact derived absolute difference 2", requirements)
            self.assertIn("at least one value lower than 2", requirements)
            self.assertIn("at least one value higher than 2", requirements)
            self.assertIn("Context event IDs must never substitute", requirements)
            self.assertIn("previous_qa_proposal selected a context pair", requirements)
            schema = prompt["output_schema"]["qa_proposal"]
            self.assertIn("business idea score", schema["question"])
            self.assertEqual(schema["gold_answer_text"], "2")
            self.assertEqual(
                [option["text"] for option in schema["options"]],
                ["2", "1", "3", "4"],
            )
            self.assertEqual(schema["query_plan"]["reasoning_anchor_path"], pair_ids)
            self.assertEqual(schema["query_plan"]["required_anchor_refs"], pair_ids)
            self.assertEqual(schema["query_plan"]["supporting_anchor_refs"], pair_ids)
            self.assertEqual(schema["query_plan"]["gold_anchor_refs"], pair_ids)
            self.assertEqual(schema["anchor_refs"]["support"], pair_ids)
            self.assertEqual(schema["anchor_refs"]["gold"], pair_ids)
            self.assertNotIn("comparison_graph_contract", schema)

    def test_comparison_prompt_requires_natural_time_qualifier_when_slot_repeats(self) -> None:
        plan, _graph_update, alignment, _proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=True
        )
        plan["events"][2]["entity"] = "Jon"
        plan["events"][2]["slot"] = "business_idea_score"
        plan["events"][2]["time_index"] = 2
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_final_qa(
                plan,
                alignment,
                {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                [],
                {
                    "graph_snapshot_hash": "run111-graph",
                    "dialogue_hash": "run111-dialogue",
                    "alignment_hash": "run111-alignment",
                },
                SimpleNamespace(),
            )

        prompt = captured[0]
        contract = prompt["comparison_graph_contract"]
        self.assertTrue(contract["requires_natural_time_qualifier"])
        self.assertEqual(
            contract["time_disambiguation_dimensions"][0]["entity"], "Jon"
        )
        requirements = " ".join(prompt["requirements"])
        self.assertIn("natural, solver-visible timing qualifier", requirements)
        self.assertIn("Never expose 'time_index'", requirements)
        self.assertIn(
            "natural dialogue-grounded time qualifier",
            prompt["output_schema"]["qa_proposal"]["question"],
        )

    def test_run111_comparison_duplicate_normalizer_preserves_wrong_pair_for_reject(self) -> None:
        plan, graph_update, alignment, proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=False
        )
        original = copy.deepcopy(proposal)

        normalized, diagnostics = subject.normalize_stage64_comparison_number_duplicate_options(
            plan,
            proposal,
        )

        self.assertEqual(proposal, original)
        self.assertTrue(
            diagnostics["stage64_comparison_number_duplicate_option_normalizer_used"]
        )
        self.assertEqual([option["text"] for option in normalized["options"]], ["2", "1", "3", "4"])
        self.assertEqual(normalized["query_plan"], original["query_plan"])
        self.assertEqual(normalized["anchor_refs"], original["anchor_refs"])
        self.assertEqual(normalized["question"], original["question"])
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "comparison_reference_group_not_exact_pair",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=proposal,
                hashes={
                    "graph_snapshot_hash": "run111-graph",
                    "dialogue_hash": "run111-dialogue",
                    "alignment_hash": "run111-alignment",
                },
                stage64_target={"qa_operation": "comparison", "preferred_answer_type": "Number"},
            )

    def test_run111_comparison_duplicate_normalizer_passes_correct_pair(self) -> None:
        plan, graph_update, alignment, proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=True
        )
        original = copy.deepcopy(proposal)

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run111-graph",
                "dialogue_hash": "run111-dialogue",
                "alignment_hash": "run111-alignment",
            },
            stage64_target={"qa_operation": "comparison", "preferred_answer_type": "Number"},
        )

        self.assertEqual(proposal, original)
        self.assertEqual([option["text"] for option in qa["options"]], ["2", "1", "3", "4"])
        normalizer = qa["stage64_comparison_number_duplicate_option_normalizer"]
        self.assertTrue(normalizer["stage64_comparison_number_duplicate_option_normalizer_used"])
        self.assertEqual(normalizer["expected_absolute_difference"], 2)
        self.assertEqual(normalizer["rewritten_option_labels"], ["B", "C", "D"])
        self.assertEqual(
            normalizer["comparison_pair_event_instance_ids"],
            [
                "current_method_stage3_v9_segmented_comparison_smoke:S0000:E01",
                "current_method_stage3_v9_segmented_comparison_smoke:S0000:E02",
            ],
        )
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {"qa": qa, "graph_update": graph_update, "current_event_plan": plan}
            ),
            [],
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "comparison",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_comparison_number_duplicate_option_normalizer_used_count"],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_comparison_number_duplicate_option_normalizer_rewritten_option_count"
            ],
            3,
        )

    def test_run111_comparison_duplicate_normalizer_wrong_gold_fails_closed(self) -> None:
        plan, _graph_update, _alignment, proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=True,
            gold_text="1",
        )

        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "refused wrong gold option"):
            subject.normalize_stage64_comparison_number_duplicate_options(plan, proposal)

        _plan, _graph_update, _alignment, wrong_declared = (
            self._run111_comparison_duplicate_fixture(use_primary_pair=True)
        )
        wrong_declared["query_plan"]["gold_answer"] = "7"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "refused wrong declared gold answer",
        ):
            subject.normalize_stage64_comparison_number_duplicate_options(
                _plan,
                wrong_declared,
            )

    def test_comparison_duplicate_normalizer_is_noop_for_unique_options(self) -> None:
        plan, _graph_update, _alignment, proposal = self._run111_comparison_duplicate_fixture(
            use_primary_pair=True
        )
        for option, text in zip(proposal["options"], ("2", "1", "3", "4")):
            option["text"] = text

        normalized, diagnostics = subject.normalize_stage64_comparison_number_duplicate_options(
            plan,
            proposal,
        )

        self.assertIs(normalized, proposal)
        self.assertFalse(
            diagnostics["stage64_comparison_number_duplicate_option_normalizer_used"]
        )
        self.assertEqual(diagnostics["noop_reason"], "option_texts_already_unique")

    @staticmethod
    def _run121_temporal_reference_drift_fixture(
        *,
        already_correct: bool = False,
    ) -> tuple[dict, dict, dict, dict]:
        prefix = "current_method_stage3_v11_segmented_tcc_smoke:S0000:"
        event_specs = (
            ("E01", "ProjectX", "access_scope", "Internal", "outdated", 1, None),
            ("E02", "ProjectX", "access_scope", "TeamA", "canceled", 2, "Internal"),
            ("E03", "ProjectX", "access_scope", "DepartmentY", "expired", 3, "TeamA"),
            ("E04", "ProjectX", "access_scope", "Public", "active", 4, "DepartmentY"),
            ("E05", "Budget2024", "approval_status", "Pending", "report", 5, None),
            ("E06", "TeamA", "project_participation", "ProjectX", "belief", 6, None),
            ("E07", "DepartmentY", "access_rights", "Read-only", "scope_limited", 7, None),
            ("E08", "Public", "access_rights", "Full", "belief", 8, None),
        )
        events = []
        alignment = {}
        for index, (short_id, entity, slot, value, status, time_index, previous) in enumerate(
            event_specs,
            start=1,
        ):
            event_id = f"{prefix}{short_id}"
            event = {
                "event_id": short_id,
                "event_instance_id": event_id,
                "event_type": "state_change",
                "entity": entity,
                "slot": slot,
                "value": value,
                "status": status,
                "time_index": time_index,
                "required_for_dialogue": True,
            }
            if previous is not None:
                event["previous_value"] = previous
            events.append(event)
            alignment[event_id] = {"turn_ids": [f"D2:{index}"]}

        current_event_plan = {
            "operation": "temporal_state_update",
            "events": copy.deepcopy(events),
            "graph_snapshot_hash": "run121-graph",
        }
        graph_update = {
            "graph_snapshot_id": "run121-temporal",
            "operation": "temporal_state_update",
            "events": copy.deepcopy(events),
        }
        chain_ids = [f"{prefix}E0{index}" for index in range(1, 5)]
        wrong_source_ids = [f"{prefix}E07", f"{prefix}E08", f"{prefix}E03"]
        source_ids = chain_ids[:3] if already_correct else wrong_source_ids
        option_texts = (
            ("Internal", "TeamA", "DepartmentY")
            if already_correct
            else ("Read-only", "Full", "DepartmentY")
        )
        options = [
            {
                "id": label,
                "text": text,
                "answer_type": "LocationOrScope",
                "source_event_instance_id": source_id,
                "anchor_ref": source_id,
                "anchor_kind": "event",
                "truth_value": label == "D",
            }
            for label, text, source_id in zip("ABC", option_texts, source_ids)
        ]
        options.append(
            {
                "id": "D",
                "text": "Public",
                "answer_type": "LocationOrScope",
                "source_event_instance_id": chain_ids[3],
                "anchor_ref": chain_ids[3],
                "anchor_kind": "event",
                "truth_value": True,
            }
        )
        support_ids = chain_ids if already_correct else [
            chain_ids[3],
            *wrong_source_ids,
        ]
        distractor_ids = chain_ids[:3] if already_correct else wrong_source_ids
        proposal = {
            "question_type": "single_choice",
            "question": "What is the current access scope for ProjectX?",
            "question_text": "What is the current access scope for ProjectX?",
            "qa_operation": "temporal_state_update",
            "qa_subtype": "latest_state_update",
            "answer_type": "LocationOrScope",
            "answer_format": "single_choice_mcq",
            "gold": ["D"],
            "gold_answer_text": "Public",
            "options": options,
            "anchor_refs": {
                "support": support_ids,
                "gold": [chain_ids[3]],
                "distractors": distractor_ids,
            },
            "query_plan": {
                "answer_source": "graph_anchor",
                "qa_operation": "temporal_state_update",
                "qa_subtype": "latest_state_update",
                "answer_type": "LocationOrScope",
                "reasoning_anchor_path": support_ids,
                "required_anchor_refs": support_ids,
                "supporting_anchor_refs": support_ids,
                "gold_anchor_refs": [chain_ids[3]],
                "distractor_anchor_refs": distractor_ids,
                "gold_answer": "Public",
                "normalized_gold_answer": "public",
            },
        }
        return current_event_plan, graph_update, alignment, proposal

    @classmethod
    def _run122_temporal_gold_label_drift_fixture(
        cls,
        *,
        current_option_label: str = "D",
        reference_drift: bool = True,
    ) -> tuple[dict, dict, dict, dict]:
        plan, graph_update, alignment, proposal = (
            cls._run121_temporal_reference_drift_fixture(
                already_correct=not reference_drift
            )
        )
        chain_events = plan["events"][:4]
        current_event = chain_events[-1]
        prior_events = iter(chain_events[:3])
        if current_option_label != "D" or not reference_drift:
            for label, option in zip("ABCD", proposal["options"]):
                event = current_event if label == current_option_label else next(prior_events)
                event_id = event["event_instance_id"]
                option["text"] = event["value"]
                option["source_event_instance_id"] = event_id
                option["anchor_ref"] = event_id
        for label, option in zip("ABCD", proposal["options"]):
            option["answer_type"] = "AttributeValue"
            option["truth_value"] = label == current_option_label
        proposal["answer_type"] = "AttributeValue"
        proposal["query_plan"]["answer_type"] = "AttributeValue"
        proposal["gold"] = ["A" if current_option_label != "A" else "B"]
        if reference_drift:
            current_id = current_event["event_instance_id"]
            distractor_ids = [
                plan["events"][6]["event_instance_id"],
                plan["events"][7]["event_instance_id"],
                plan["events"][2]["event_instance_id"],
            ]
            proposal["anchor_refs"]["support"] = [current_id]
            proposal["anchor_refs"]["distractors"] = distractor_ids
            proposal["query_plan"]["reasoning_anchor_path"] = [current_id]
            proposal["query_plan"]["required_anchor_refs"] = [current_id]
            proposal["query_plan"]["supporting_anchor_refs"] = [current_id]
            proposal["query_plan"]["distractor_anchor_refs"] = distractor_ids
        return plan, graph_update, alignment, proposal

    @classmethod
    def _run123_temporal_missing_gold_answer_text_fixture(
        cls,
    ) -> tuple[dict, dict, dict, dict]:
        plan, graph_update, alignment, proposal = (
            cls._run121_temporal_reference_drift_fixture(already_correct=True)
        )
        values = (
            "InternalTeamOnly",
            "CrossDepartmental",
            "RestrictedToLeadership",
            "PublicAccess",
        )
        for event_group in (plan["events"], graph_update["events"]):
            for index, (event, value) in enumerate(zip(event_group[:4], values)):
                event["entity"] = "ProjectAlpha"
                event["value"] = value
                if index == 0:
                    event.pop("previous_value", None)
                else:
                    event["previous_value"] = values[index - 1]
        chain_ids = [event["event_instance_id"] for event in plan["events"][:4]]
        question = "What is the current access scope for ProjectAlpha?"
        proposal["question"] = question
        proposal["question_text"] = question
        proposal["answer_type"] = "LocationOrScope"
        proposal["gold"] = ["D"]
        proposal.pop("gold_answer_text", None)
        for option, text, source_id, truth in zip(
            proposal["options"],
            ("RestrictedToLeadership", "CrossDepartmental", "PublicAccess", "PublicAccess"),
            chain_ids,
            (False, False, False, True),
        ):
            option["text"] = text
            option["source_event_instance_id"] = source_id
            option["anchor_ref"] = source_id
            option["answer_type"] = "LocationOrScope"
            option["truth_value"] = truth
        proposal["anchor_refs"] = {
            "support": chain_ids,
            "gold": [chain_ids[3]],
            "distractors": chain_ids[:3],
        }
        proposal["query_plan"].update(
            {
                "answer_type": "LocationOrScope",
                "reasoning_anchor_path": chain_ids,
                "required_anchor_refs": chain_ids,
                "supporting_anchor_refs": chain_ids,
                "gold_anchor_refs": [chain_ids[3]],
                "distractor_anchor_refs": chain_ids[:3],
                "gold_answer": "PublicAccess",
                "normalized_gold_answer": "publicaccess",
            }
        )
        return plan, graph_update, alignment, proposal

    def test_run121_temporal_normalizer_repairs_prior_options_and_all_refs(self) -> None:
        plan, _graph_update, _alignment, proposal = (
            self._run121_temporal_reference_drift_fixture()
        )
        original = copy.deepcopy(proposal)

        normalized, diagnostics = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                proposal,
            )
        )

        prefix = "current_method_stage3_v11_segmented_tcc_smoke:S0000:"
        chain_ids = [f"{prefix}E0{index}" for index in range(1, 5)]
        self.assertEqual(proposal, original)
        self.assertTrue(
            diagnostics[
                "stage64_temporal_state_update_option_reference_normalizer_used"
            ]
        )
        self.assertEqual(diagnostics["rewritten_option_labels"], ["A", "B", "C"])
        self.assertEqual(diagnostics["changed_option_labels"], ["A", "B"])
        self.assertEqual(
            [option["text"] for option in normalized["options"]],
            ["Internal", "TeamA", "DepartmentY", "Public"],
        )
        self.assertEqual(
            [option["source_event_instance_id"] for option in normalized["options"]],
            chain_ids,
        )
        self.assertEqual(normalized["options"][3], original["options"][3])
        for field in (
            "question",
            "question_text",
            "question_type",
            "qa_operation",
            "qa_subtype",
            "answer_type",
            "answer_format",
            "gold",
            "gold_answer_text",
        ):
            self.assertEqual(normalized[field], original[field])
        self.assertEqual(normalized["anchor_refs"]["support"], chain_ids)
        self.assertEqual(normalized["anchor_refs"]["gold"], [chain_ids[3]])
        self.assertEqual(normalized["anchor_refs"]["distractors"], chain_ids[:3])
        for field in (
            "reasoning_anchor_path",
            "required_anchor_refs",
            "supporting_anchor_refs",
        ):
            self.assertEqual(normalized["query_plan"][field], chain_ids)
        self.assertEqual(normalized["query_plan"]["gold_anchor_refs"], [chain_ids[3]])
        self.assertEqual(
            normalized["query_plan"]["distractor_anchor_refs"], chain_ids[:3]
        )
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            subject._stage64_json_sha256(original),
        )
        self.assertEqual(
            diagnostics["post_qa_proposal_sha256"],
            subject._stage64_json_sha256(normalized),
        )
        self.assertNotEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )
        self.assertEqual(len(diagnostics["synchronized_reference_fields"]), 8)

    def test_run121_temporal_normalizer_wrong_gold_and_ambiguous_chain_fail_closed(self) -> None:
        plan, _graph_update, _alignment, proposal = (
            self._run121_temporal_reference_drift_fixture()
        )
        wrong_gold = copy.deepcopy(proposal)
        wrong_gold["gold"] = ["C"]
        wrong_gold["options"][2]["truth_value"] = True
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "inconsistent truth_value",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                wrong_gold,
            )

        wrong_declared = copy.deepcopy(proposal)
        wrong_declared["query_plan"]["gold_answer"] = "Internal"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "refused wrong declared gold answer",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                wrong_declared,
            )

        ambiguous = copy.deepcopy(plan)
        for index, event in enumerate(ambiguous["events"][4:], start=1):
            event["entity"] = "ProjectY"
            event["slot"] = "deployment_scope"
            event["value"] = f"Scope{index}"
            event["status"] = "active" if index == 4 else "outdated"
            event["time_index"] = index
            if index > 1:
                event["previous_value"] = f"Scope{index - 1}"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "requires one unique validated four-event update chain",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                ambiguous,
                proposal,
            )

    def test_run121_temporal_normalizer_already_correct_is_noop(self) -> None:
        plan, _graph_update, _alignment, proposal = (
            self._run121_temporal_reference_drift_fixture(already_correct=True)
        )

        normalized, diagnostics = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                proposal,
            )
        )

        self.assertIs(normalized, proposal)
        self.assertFalse(
            diagnostics[
                "stage64_temporal_state_update_option_reference_normalizer_used"
            ]
        )
        self.assertEqual(
            diagnostics["noop_reason"],
            "temporal_options_and_references_already_correct",
        )
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )

    def test_run121_temporal_normalizer_reenters_builder_and_semantic_gate(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run121_temporal_reference_drift_fixture()
        )

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run121-graph",
                "dialogue_hash": "run121-dialogue",
                "alignment_hash": "run121-alignment",
            },
            stage64_target={
                "qa_operation": "temporal_state_update",
                "preferred_answer_type": "LocationOrScope",
            },
        )

        self.assertEqual(
            [option["text"] for option in qa["options"]],
            ["Internal", "TeamA", "DepartmentY", "Public"],
        )
        self.assertEqual(qa["gold"], ["D"])
        normalizer = qa[
            "stage64_temporal_state_update_option_reference_normalizer"
        ]
        self.assertTrue(
            normalizer[
                "stage64_temporal_state_update_option_reference_normalizer_used"
            ]
        )
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {
                    "qa": qa,
                    "graph_update": graph_update,
                    "current_event_plan": plan,
                }
            ),
            [],
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "temporal_state_update",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary[
                "stage64_temporal_state_update_option_reference_normalizer_used_count"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_temporal_state_update_option_reference_normalizer_rewritten_option_count"
            ],
            3,
        )
        self.assertEqual(
            summary[
                "stage64_temporal_state_update_option_reference_normalizer_synchronized_reference_field_count"
            ],
            8,
        )

    def test_reasoning_identity_completion_injects_only_missing_target_fields(self) -> None:
        target = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        original = {"query_plan": {}, "unrelated": {"keep": [1, 2, 3]}}

        normalized, diagnostics = (
            subject.normalize_stage64_reasoning_identity_completion(
                original,
                target,
            )
        )

        self.assertEqual(original, {"query_plan": {}, "unrelated": {"keep": [1, 2, 3]}})
        self.assertTrue(
            diagnostics[
                "stage64_reasoning_identity_completion_normalizer_used"
            ]
        )
        self.assertEqual(
            diagnostics["completed_top_level_fields"],
            list(subject.STAGE64_REASONING_TARGET_FIELDS),
        )
        self.assertEqual(
            diagnostics["completed_query_plan_fields"],
            list(subject.STAGE64_REASONING_TARGET_FIELDS),
        )
        self.assertNotEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(normalized[field], target[field])
            self.assertEqual(normalized["query_plan"][field], target[field])
        self.assertEqual(normalized["unrelated"], original["unrelated"])

        no_query_plan, no_query_plan_diagnostics = (
            subject.normalize_stage64_reasoning_identity_completion({}, target)
        )
        self.assertNotIn("query_plan", no_query_plan)
        self.assertEqual(
            no_query_plan_diagnostics["completed_query_plan_fields"],
            [],
        )
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(no_query_plan[field], target[field])

    def test_reasoning_identity_completion_refuses_explicit_conflicts(self) -> None:
        target = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        complete = {
            **{
                field: copy.deepcopy(target[field])
                for field in subject.STAGE64_REASONING_TARGET_FIELDS
            },
            "query_plan": {
                field: copy.deepcopy(target[field])
                for field in subject.STAGE64_REASONING_TARGET_FIELDS
            },
        }
        conflicts = []
        top_level = copy.deepcopy(complete)
        top_level["reasoning_template"] = "direct_lookup"
        conflicts.append(top_level)
        query_plan = copy.deepcopy(complete)
        query_plan["query_plan"]["hard_negative_strategy"] = "random_negative"
        conflicts.append(query_plan)
        wrong_type = copy.deepcopy(complete)
        wrong_type["hop_count"] = float(target["hop_count"])
        conflicts.append(wrong_type)
        for proposal in conflicts:
            with self.subTest(proposal=proposal), self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "reasoning target drift",
            ):
                subject.normalize_stage64_reasoning_identity_completion(
                    proposal,
                    target,
                )

    def test_canonical_reasoning_identity_survives_full_temporal_qa_builder(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run121_temporal_reference_drift_fixture(already_correct=True)
        )
        target = subject._build_stage64_canonical_targets(
            1,
            {"temporal_state_update": 1.0},
            None,
        )[0]["stage64_target"]
        chain_ids = [event["event_instance_id"] for event in plan["events"][:4]]
        for container in (plan, graph_update):
            container["events"][1]["status"] = "active"
            container["events"][3]["status"] = "canceled"
        for index in range(1, 4):
            plan["events"][index]["depends_on"] = [chain_ids[index - 1]]
            graph_update["events"][index]["depends_on"] = [
                chain_ids[index - 1]
            ]
        for index, option in enumerate(proposal["options"]):
            option["truth_value"] = index == 1
        proposal["gold"] = ["B"]
        proposal["gold_answer_text"] = "TeamA"
        proposal["anchor_refs"] = {
            "support": chain_ids,
            "gold": [chain_ids[1]],
            "distractors": [chain_ids[0], chain_ids[2], chain_ids[3]],
        }
        proposal["query_plan"].update(
            {
                "reasoning_anchor_path": chain_ids,
                "required_anchor_refs": chain_ids,
                "supporting_anchor_refs": chain_ids,
                "gold_anchor_refs": [chain_ids[1]],
                "distractor_anchor_refs": [
                    chain_ids[0],
                    chain_ids[2],
                    chain_ids[3],
                ],
                "gold_answer": "TeamA",
                "normalized_gold_answer": "teama",
                "dependency_closure_refs": chain_ids,
                "answer_projection": "current_value",
            }
        )
        identity = {
            field: target[field] for field in subject.STAGE64_REASONING_TARGET_FIELDS
        }
        proposal.update(identity)
        proposal["query_plan"].update(identity)
        proposal["question"] = (
            "After disregarding the canceled proposal, what is the current access "
            "scope for ProjectX?"
        )
        proposal["question_text"] = proposal["question"]

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run121-graph",
                "dialogue_hash": "run121-dialogue",
                "alignment_hash": "run121-alignment",
            },
            stage64_target=target,
        )

        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(qa[field], target[field])
            self.assertEqual(qa["query_plan"][field], target[field])
        self.assertTrue(qa["forbid_max_time_shortcut"])
        self.assertEqual(qa["gold"], ["B"])
        self.assertLess(
            plan["events"][1]["time_index"],
            plan["events"][3]["time_index"],
        )
        self.assertIn(
            chain_ids[3],
            qa["query_plan"]["resolved_reasoning_event_instance_ids"],
        )
        self.assertIn(
            chain_ids[3],
            qa["query_plan"]["resolved_distractor_event_instance_ids"],
        )

        missing_closure = copy.deepcopy(proposal)
        missing_closure["query_plan"].pop("dependency_closure_refs")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "dependency_closure_refs",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=missing_closure,
                hashes={
                    "graph_snapshot_hash": "run121-graph",
                    "dialogue_hash": "run121-dialogue",
                    "alignment_hash": "run121-alignment",
                },
                stage64_target=target,
            )

        wrong_projection = copy.deepcopy(proposal)
        wrong_projection["query_plan"]["answer_projection"] = "direct_copy"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "answer_projection",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=wrong_projection,
                hashes={
                    "graph_snapshot_hash": "run121-graph",
                    "dialogue_hash": "run121-dialogue",
                    "alignment_hash": "run121-alignment",
                },
                stage64_target=target,
            )

        broken_dependency_plan = copy.deepcopy(plan)
        broken_dependency_plan["events"][2].pop("depends_on")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "graph structure not realized",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=broken_dependency_plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=proposal,
                hashes={
                    "graph_snapshot_hash": "run121-graph",
                    "dialogue_hash": "run121-dialogue",
                    "alignment_hash": "run121-alignment",
                },
                stage64_target=target,
            )

        omitted_identity = copy.deepcopy(proposal)
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            omitted_identity.pop(field)
            omitted_identity["query_plan"].pop(field)
        completed_qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=omitted_identity,
            hashes={
                "graph_snapshot_hash": "run121-graph",
                "dialogue_hash": "run121-dialogue",
                "alignment_hash": "run121-alignment",
            },
            stage64_target=target,
        )
        for field in subject.STAGE64_REASONING_TARGET_FIELDS:
            self.assertEqual(completed_qa[field], target[field])
            self.assertEqual(completed_qa["query_plan"][field], target[field])
        completion = completed_qa[
            "stage64_reasoning_identity_completion_normalizer"
        ]
        self.assertEqual(
            completion["completed_top_level_fields"],
            list(subject.STAGE64_REASONING_TARGET_FIELDS),
        )
        self.assertEqual(
            completion["completed_query_plan_fields"],
            list(subject.STAGE64_REASONING_TARGET_FIELDS),
        )

        drifted = copy.deepcopy(proposal)
        drifted["query_plan"]["reasoning_template"] = "direct_lookup"
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "reasoning target drift",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=drifted,
                hashes={
                    "graph_snapshot_hash": "run121-graph",
                    "dialogue_hash": "run121-dialogue",
                    "alignment_hash": "run121-alignment",
                },
                stage64_target=target,
            )

    def test_run122_temporal_gold_label_reconciliation_repairs_exact_payload(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run122_temporal_gold_label_drift_fixture()
        )
        original = copy.deepcopy(proposal)

        normalized, diagnostics = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                proposal,
            )
        )

        chain_ids = [event["event_instance_id"] for event in plan["events"][:4]]
        self.assertEqual(proposal, original)
        self.assertEqual(normalized["gold"], ["D"])
        self.assertEqual(
            [option["text"] for option in normalized["options"]],
            ["Internal", "TeamA", "DepartmentY", "Public"],
        )
        self.assertEqual(
            [option["source_event_instance_id"] for option in normalized["options"]],
            chain_ids,
        )
        self.assertEqual(
            [option["truth_value"] for option in normalized["options"]],
            [False, False, False, True],
        )
        self.assertTrue(diagnostics["gold_label_reconciled"])
        self.assertEqual(diagnostics["gold_label_before"], "A")
        self.assertEqual(diagnostics["gold_label_after"], "D")
        self.assertEqual(diagnostics["gold_label"], "D")
        self.assertEqual(
            diagnostics["schema_version"],
            subject.STAGE64_TEMPORAL_STATE_UPDATE_OPTION_REFERENCE_NORMALIZER_VERSION,
        )
        self.assertTrue(diagnostics["gold_label_reconciliation_evidence_sha256"])
        self.assertEqual(
            len(diagnostics["gold_label_reconciliation_evidence_sha256"]),
            64,
        )
        self.assertIn(
            "unique_true_option_matches_current",
            diagnostics["gold_label_reconciliation_evidence_fields"],
        )
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            subject._stage64_json_sha256(original),
        )
        self.assertEqual(
            diagnostics["post_qa_proposal_sha256"],
            subject._stage64_json_sha256(normalized),
        )

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run122-graph",
                "dialogue_hash": "run122-dialogue",
                "alignment_hash": "run122-alignment",
            },
            stage64_target={
                "qa_operation": "temporal_state_update",
                "preferred_answer_type": "AttributeValue",
            },
        )
        self.assertEqual(qa["gold"], ["D"])
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {
                    "qa": qa,
                    "graph_update": graph_update,
                    "current_event_plan": plan,
                }
            ),
            [],
        )
        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "temporal_state_update",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_temporal_state_update_gold_label_reconciled_count"],
            1,
        )

    def test_temporal_gold_label_reconciliation_is_position_invariant(self) -> None:
        for current_label in "ABCD":
            with self.subTest(current_label=current_label):
                plan, _graph_update, _alignment, proposal = (
                    self._run122_temporal_gold_label_drift_fixture(
                        current_option_label=current_label,
                        reference_drift=False,
                    )
                )
                expected = copy.deepcopy(proposal)
                expected["gold"] = [current_label]

                normalized, diagnostics = (
                    subject.normalize_stage64_temporal_state_update_options_and_references(
                        plan,
                        proposal,
                    )
                )

                self.assertEqual(normalized, expected)
                self.assertTrue(diagnostics["gold_label_reconciled"])
                self.assertEqual(diagnostics["gold_label_after"], current_label)
                self.assertEqual(diagnostics["rewritten_option_labels"], [])
                self.assertEqual(diagnostics["synchronized_reference_fields"], [])

    def test_temporal_gold_label_reconciliation_requires_complete_consistent_signals(self) -> None:
        plan, _graph_update, _alignment, proposal = (
            self._run122_temporal_gold_label_drift_fixture()
        )

        for invalid_label in ("", "AB", "E"):
            with self.subTest(invalid_label=invalid_label):
                invalid_gold = copy.deepcopy(proposal)
                invalid_gold["gold"] = [invalid_label]
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    "requires one A-D gold label",
                ):
                    subject.normalize_stage64_temporal_state_update_options_and_references(
                        plan,
                        invalid_gold,
                    )

        missing_truth = copy.deepcopy(proposal)
        missing_truth["options"][0].pop("truth_value")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "requires boolean truth_value",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                missing_truth,
            )

        double_true = copy.deepcopy(proposal)
        double_true["options"][0]["truth_value"] = True
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "inconsistent truth_value",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                double_true,
            )

        wrong_gold_ref = copy.deepcopy(proposal)
        wrong_gold_ref["anchor_refs"]["gold"] = [
            plan["events"][2]["event_instance_id"]
        ]
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "wrong declared gold source in anchor_refs.gold",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                wrong_gold_ref,
            )

        wrong_current_source = copy.deepcopy(proposal)
        wrong_current_source["options"][3]["source_event_instance_id"] = (
            plan["events"][2]["event_instance_id"]
        )
        wrong_current_source["options"][3]["anchor_ref"] = (
            plan["events"][2]["event_instance_id"]
        )
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "requires one unique current option text and source",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                wrong_current_source,
            )

    def test_run123_temporal_missing_gold_answer_text_is_safely_completed(self) -> None:
        plan, graph_update, alignment, proposal = (
            self._run123_temporal_missing_gold_answer_text_fixture()
        )
        original = copy.deepcopy(proposal)

        normalized, diagnostics = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                proposal,
            )
        )

        self.assertEqual(proposal, original)
        self.assertNotIn("gold_answer_text", proposal)
        self.assertEqual(normalized["gold_answer_text"], "PublicAccess")
        self.assertEqual(normalized["gold"], ["D"])
        self.assertEqual(
            [option["text"] for option in normalized["options"]],
            [
                "InternalTeamOnly",
                "CrossDepartmental",
                "RestrictedToLeadership",
                "PublicAccess",
            ],
        )
        self.assertTrue(diagnostics["gold_answer_text_completed"])
        self.assertFalse(diagnostics["gold_label_reconciled"])
        self.assertEqual(
            diagnostics["gold_answer_text_completion_source"],
            "unique_current_option.text",
        )
        self.assertEqual(
            len(diagnostics["gold_answer_text_completion_evidence_sha256"]),
            64,
        )
        self.assertIn(
            "raw_gold_label_matches_current_option",
            diagnostics["gold_answer_text_completion_evidence_fields"],
        )

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run123-graph",
                "dialogue_hash": "run123-dialogue",
                "alignment_hash": "run123-alignment",
            },
            stage64_target={
                "qa_operation": "temporal_state_update",
                "preferred_answer_type": "LocationOrScope",
            },
        )
        self.assertEqual(qa["gold"], ["D"])
        self.assertEqual(qa["gold_answer_text"], "PublicAccess")
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {
                    "qa": qa,
                    "graph_update": graph_update,
                    "current_event_plan": plan,
                }
            ),
            [],
        )
        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "temporal_state_update",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary[
                "stage64_temporal_state_update_gold_answer_text_completed_count"
            ],
            1,
        )

    def test_temporal_gold_answer_text_completion_is_strict_and_not_stacked(self) -> None:
        plan, _graph_update, _alignment, proposal = (
            self._run123_temporal_missing_gold_answer_text_fixture()
        )

        simultaneous_label_fix = copy.deepcopy(proposal)
        simultaneous_label_fix["gold"] = ["A"]
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "simultaneous gold label reconciliation and gold_answer_text completion",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                simultaneous_label_fix,
            )

        missing_truth = copy.deepcopy(proposal)
        missing_truth["options"][0].pop("truth_value")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "requires boolean truth_value",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                missing_truth,
            )

        double_true = copy.deepcopy(proposal)
        double_true["options"][0]["truth_value"] = True
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "inconsistent truth_value",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                double_true,
            )

        missing_query_gold = copy.deepcopy(proposal)
        missing_query_gold["query_plan"].pop("gold_answer")
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "safe completion evidence was incomplete",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                missing_query_gold,
            )

        wrong_gold_ref = copy.deepcopy(proposal)
        wrong_gold_ref["query_plan"]["gold_anchor_refs"] = [
            plan["events"][2]["event_instance_id"]
        ]
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "wrong declared gold source in query_plan.gold_anchor_refs",
        ):
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                wrong_gold_ref,
            )

        for invalid_surface in ("", "InternalTeamOnly"):
            with self.subTest(invalid_surface=invalid_surface):
                wrong_surface = copy.deepcopy(proposal)
                wrong_surface["gold_answer_text"] = invalid_surface
                with self.assertRaisesRegex(
                    subject.NativeGraphDialogueReject,
                    "refused wrong declared gold answer",
                ):
                    subject.normalize_stage64_temporal_state_update_options_and_references(
                        plan,
                        wrong_surface,
                    )

        already_present = copy.deepcopy(proposal)
        already_present["gold_answer_text"] = "PublicAccess"
        _normalized, diagnostics = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                already_present,
            )
        )
        self.assertFalse(diagnostics["gold_answer_text_completed"])

        explicit_none = copy.deepcopy(proposal)
        explicit_none["gold_answer_text"] = None
        normalized_none, diagnostics_none = (
            subject.normalize_stage64_temporal_state_update_options_and_references(
                plan,
                explicit_none,
            )
        )
        self.assertEqual(normalized_none["gold_answer_text"], "PublicAccess")
        self.assertTrue(diagnostics_none["gold_answer_text_completed"])

    def test_stage64_generic_qa_schema_has_no_gold_a_truth_a_bias(self) -> None:
        temporal_plan, _graph_update, temporal_alignment, temporal_proposal = (
            self._run122_temporal_gold_label_drift_fixture()
        )
        comparison_plan, _graph_update, comparison_alignment, comparison_proposal = (
            self._run111_comparison_duplicate_fixture(use_primary_pair=True)
        )
        count_plan, _graph_update, count_alignment, count_proposal = (
            self._run116_count_contract_fixture()
        )
        prompts: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            prompts.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            for plan, alignment, proposal, target in (
                (
                    temporal_plan,
                    temporal_alignment,
                    temporal_proposal,
                    {
                        "qa_operation": "temporal_state_update",
                        "preferred_answer_type": "AttributeValue",
                    },
                ),
                (
                    comparison_plan,
                    comparison_alignment,
                    comparison_proposal,
                    {"qa_operation": "comparison", "preferred_answer_type": "Number"},
                ),
                (
                    count_plan,
                    count_alignment,
                    count_proposal,
                    {"qa_operation": "count", "preferred_answer_type": "Number"},
                ),
            ):
                subject.call_api_stage64_qa_repair(
                    plan,
                    alignment,
                    proposal,
                    target,
                    "inconsistent truth_value declaration",
                    [],
                    SimpleNamespace(),
                )

        temporal_schema = prompts[0]["output_schema"]["qa_proposal"]
        self.assertEqual(
            temporal_schema["gold"],
            ["D"],
        )
        self.assertIs(temporal_schema["options"][0]["truth_value"], False)
        self.assertIs(temporal_schema["options"][3]["truth_value"], True)
        self.assertIn(
            "sole option whose truth_value is the JSON boolean true",
            " ".join(prompts[0]["requirements"]),
        )
        self.assertEqual(
            temporal_schema["gold_answer_text"],
            "Public",
        )
        self.assertIn(
            "gold_answer_text is mandatory",
            " ".join(prompts[0]["requirements"]),
        )
        for deterministic_prompt in prompts[1:]:
            deterministic_schema = deterministic_prompt["output_schema"]["qa_proposal"]
            self.assertEqual(deterministic_schema["gold"], ["A"])
            self.assertIs(deterministic_schema["options"][0]["truth_value"], True)
        for prompt in prompts:
            query_schema = prompt["output_schema"]["qa_proposal"]["query_plan"]
            self.assertIn("dependency_closure_refs", query_schema)
            self.assertIn("answer_projection", query_schema)
        count_query_schema = prompts[2]["output_schema"]["qa_proposal"][
            "query_plan"
        ]
        self.assertEqual(
            count_query_schema["filter_predicate"],
            {
                "condition": "the screenplay milestone counts as completed",
                "included": True,
            },
        )
        self.assertEqual(
            count_query_schema["answer_projection"],
            "included_member_count",
        )

    def test_count_qa_repair_requires_distinct_numeric_options_and_member_anchors(self) -> None:
        plan, _graph_update, alignment, proposal = self._run116_count_contract_fixture()
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_qa_repair(
                plan,
                alignment,
                proposal,
                {"qa_operation": "count", "preferred_answer_type": "Number"},
                "insufficient_reasoning_anchors",
                [],
                SimpleNamespace(),
                item_difficulty_target="harder_boundary",
            )

        prompt = captured[0]
        requirements = " ".join(prompt["requirements"])
        self.assertIn("at least six distinct member events", requirements)
        self.assertIn("four distinct numeric options", requirements)
        self.assertEqual(prompt["target"]["item_difficulty_target"], "harder_boundary")
        self.assertEqual(prompt["target"]["required_answer_types"], ["Number"])

    @staticmethod
    def _run105_count_duplicate_fixture(*, included_ids: set[str]) -> tuple[dict, dict, dict, dict]:
        prefix = "current_method_stage3_v9_segmented_micro:S0000:"
        cities = (
            "New York City",
            "Los Angeles",
            "Chicago",
            "Seattle",
            "Boston",
            "San Francisco",
            "Austin",
            "Denver",
        )
        events = []
        alignment = {}
        for index, city in enumerate(cities, start=1):
            short_id = f"E{index:02d}"
            event_id = f"{prefix}{short_id}"
            events.append(
                {
                    "event_id": short_id,
                    "event_instance_id": event_id,
                    "event_type": "state_change",
                    "entity": "team_trip",
                    "slot": "destination",
                    "value": city,
                    "status": "active",
                    "time_index": 1,
                    "required_for_dialogue": True,
                    "set_membership": {
                        "set_id": "team_trip_candidate_destinations",
                        "condition": "the destination remains under active consideration",
                        "included": short_id in included_ids,
                    },
                }
            )
            alignment[event_id] = {"turn_ids": [f"D2:{index}"]}
        current_event_plan = {
            "events": copy.deepcopy(events),
            "graph_snapshot_hash": "run105-graph",
        }
        graph_update = {
            "graph_snapshot_id": "run105-count",
            "events": copy.deepcopy(events),
        }
        refs = [f"{prefix}{short_id}" for short_id in ("E05", "E06", "E07", "E08", "E01", "E02")]
        gold_refs = [f"{prefix}E05", f"{prefix}E06"]
        distractor_refs = [f"{prefix}{short_id}" for short_id in ("E05", "E06", "E01", "E02")]
        proposal = {
            "question_type": "single_choice",
            "question": "How many cities are currently being considered as potential destinations for the team trip?",
            "question_text": "How many cities are currently being considered as potential destinations for the team trip?",
            "qa_operation": "count",
            "qa_subtype": "filtered_count",
            "answer_type": "Number",
            "answer_format": "single_choice_mcq",
            "gold": ["A"],
            "gold_answer_text": "2",
            "options": [
                {
                    "id": label,
                    "text": text,
                    "answer_type": "Number",
                    "anchor_kind": "event",
                    "anchor_ref": ref,
                    "source_event_instance_id": ref,
                    "truth_value": truth,
                }
                for label, text, ref, truth in zip(
                    "ABCD",
                    ("2", "2", "1", "1"),
                    distractor_refs,
                    (True, True, False, False),
                )
            ],
            "anchor_refs": {
                "support": refs,
                "gold": gold_refs,
                "distractors": distractor_refs,
            },
            "query_plan": {
                "answer_source": "graph_anchor",
                "qa_operation": "count",
                "qa_subtype": "filtered_count",
                "answer_type": "Number",
                "reasoning_anchor_path": refs,
                "required_anchor_refs": refs,
                "supporting_anchor_refs": refs,
                "gold_anchor_refs": gold_refs,
                "distractor_anchor_refs": distractor_refs,
                "filter_predicate": {
                    "condition": "the destination remains under active consideration",
                    "included": True,
                },
                "gold_answer": "2",
                "normalized_gold_answer": "2",
            },
        }
        return current_event_plan, graph_update, alignment, proposal

    @classmethod
    def _run116_count_contract_fixture(cls) -> tuple[dict, dict, dict, dict]:
        plan, graph_update, alignment, proposal = cls._run105_count_duplicate_fixture(
            included_ids={"E01", "E02", "E03", "E04"}
        )
        values = (
            "finished her screenplay",
            "working on her screenplay",
            "paused her screenplay",
            "restarted her screenplay",
            "abandoned her screenplay",
            "submitted her screenplay",
            "rewrote her screenplay",
            "never completed her screenplay",
        )
        for event_group in (plan["events"], graph_update["events"]):
            for event, value in zip(event_group, values):
                event["entity"] = "Joanna"
                event["slot"] = "screenplay_completion"
                event["value"] = value
                event["set_membership"]["set_id"] = "screenplay_completion_set"
                event["set_membership"]["condition"] = (
                    "the screenplay milestone counts as completed"
                )
        all_ids = [event["event_instance_id"] for event in plan["events"]]
        included_ids = all_ids[:4]
        excluded_ids = all_ids[4:]
        question = (
            "How many candidates in Joanna's screenplay completion set are included "
            "under the screenplay completion condition?"
        )
        proposal.update(
            {
                "question": question,
                "question_text": question,
                "gold": ["A"],
                "gold_answer_text": "4",
            }
        )
        for option, text, source_id, truth in zip(
            proposal["options"],
            ("4", "3", "5", "6"),
            all_ids[:4],
            (True, False, False, False),
        ):
            option.update(
                {
                    "text": text,
                    "source_event_instance_id": source_id,
                    "anchor_ref": source_id,
                    "truth_value": truth,
                }
            )
        proposal["anchor_refs"] = {
            "support": all_ids,
            "gold": included_ids,
            "distractors": excluded_ids,
        }
        proposal["query_plan"].update(
            {
                "reasoning_anchor_path": all_ids,
                "required_anchor_refs": all_ids,
                "supporting_anchor_refs": all_ids,
                "gold_anchor_refs": included_ids,
                "distractor_anchor_refs": excluded_ids,
                "filter_predicate": {
                    "condition": "the screenplay milestone counts as completed",
                    "included": True,
                },
                "gold_answer": "4",
                "normalized_gold_answer": "4",
            }
        )
        return plan, graph_update, alignment, proposal

    def test_derive_unique_count_contract_is_read_only_and_durable(self) -> None:
        plan, _graph_update, _alignment, _proposal = self._run116_count_contract_fixture()
        original = copy.deepcopy(plan)

        contract = subject.derive_unique_count_contract(plan)

        self.assertEqual(plan, original)
        self.assertEqual(contract["schema_version"], subject.STAGE64_COUNT_GRAPH_CONTRACT_VERSION)
        self.assertEqual(contract["set_id"], "screenplay_completion_set")
        self.assertEqual(contract["entity"], "Joanna")
        self.assertEqual(contract["slot"], "screenplay_completion")
        self.assertEqual(
            contract["condition"],
            "the screenplay milestone counts as completed",
        )
        self.assertEqual(contract["member_count"], 8)
        self.assertEqual(contract["expected_included_count"], 4)
        self.assertEqual(contract["event_instance_ids"], [event["event_instance_id"] for event in plan["events"]])
        self.assertEqual(contract["included_event_instance_ids"], contract["event_instance_ids"][:4])
        self.assertEqual(contract["excluded_event_instance_ids"], contract["event_instance_ids"][4:])
        unhashed = copy.deepcopy(contract)
        contract_hash = unhashed.pop("contract_sha256")
        self.assertEqual(contract_hash, subject._stage64_json_sha256(unhashed))

        missing_durable_id = copy.deepcopy(plan)
        missing_durable_id["events"][0].pop("event_instance_id")
        self.assertIsNone(subject.derive_unique_count_contract(missing_durable_id))

        mixed_raw_dimension = copy.deepcopy(plan)
        mixed_raw_dimension["events"][0]["entity"] = "JOANNA"
        self.assertIsNone(subject.derive_unique_count_contract(mixed_raw_dimension))

    def test_run116_count_prompt_binds_exact_frozen_set_for_final_and_repair(self) -> None:
        plan, _graph_update, alignment, proposal = self._run116_count_contract_fixture()
        captured: list[dict] = []

        def respond(_args, prompt, **_kwargs):
            captured.append(prompt)
            return {"qa_proposal": {}}, {"used": True, "attempts": 1}

        hashes = {
            "graph_snapshot_hash": "run116-graph",
            "dialogue_hash": "run116-dialogue",
            "alignment_hash": "run116-alignment",
        }
        wrong_previous = copy.deepcopy(proposal)
        wrong_previous["question"] = "How many items are there based on the conversation?"
        wrong_previous["gold_answer_text"] = "1"
        with mock.patch.object(subject, "call_raw_json_api", side_effect=respond):
            subject.call_api_stage64_final_qa(
                plan,
                alignment,
                {"qa_operation": "count", "preferred_answer_type": "Number"},
                [],
                hashes,
                SimpleNamespace(),
            )
            subject.call_api_stage64_qa_repair(
                plan,
                alignment,
                wrong_previous,
                {"qa_operation": "count", "preferred_answer_type": "Number"},
                "count_gold_not_exact_included_count",
                [],
                SimpleNamespace(),
            )

        self.assertEqual(len(captured), 2)
        all_ids = [event["event_instance_id"] for event in plan["events"]]
        for prompt in captured:
            contract = prompt["count_graph_contract"]
            self.assertEqual(contract["event_instance_ids"], all_ids)
            self.assertEqual(contract["included_event_instance_ids"], all_ids[:4])
            self.assertEqual(contract["excluded_event_instance_ids"], all_ids[4:])
            self.assertEqual(contract["expected_included_count"], 4)
            requirements = " ".join(prompt["requirements"])
            self.assertIn("Joanna", requirements)
            self.assertIn("screenplay_completion", requirements)
            self.assertIn(
                "the screenplay milestone counts as completed",
                requirements,
            )
            self.assertIn("must each contain exactly all count-set", requirements)
            self.assertIn("must each equal the exact graph-derived included count 4", requirements)
            self.assertIn("based on the conversation", requirements)
            schema = prompt["output_schema"]["qa_proposal"]
            self.assertIn("Joanna", schema["question"])
            self.assertIn("screenplay_completion", schema["question"])
            self.assertEqual([option["text"] for option in schema["options"]], ["4", "3", "5", "6"])
            self.assertEqual(schema["gold_answer_text"], "4")
            self.assertEqual(schema["query_plan"]["reasoning_anchor_path"], all_ids)
            self.assertEqual(schema["query_plan"]["required_anchor_refs"], all_ids)
            self.assertEqual(schema["query_plan"]["supporting_anchor_refs"], all_ids)
            self.assertEqual(schema["query_plan"]["gold_anchor_refs"], all_ids[:4])
            self.assertEqual(
                schema["query_plan"]["filter_predicate"],
                {
                    "condition": "the screenplay milestone counts as completed",
                    "included": True,
                },
            )
            self.assertEqual(schema["anchor_refs"]["support"], all_ids)
            self.assertEqual(schema["anchor_refs"]["gold"], all_ids[:4])
            self.assertNotIn("count_graph_contract", schema)

    def test_count_final_qa_fails_before_api_when_contract_is_missing(self) -> None:
        with mock.patch.object(subject, "call_raw_json_api") as api_call:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "requires one frozen unique count_graph_contract",
            ):
                subject.call_api_stage64_qa_repair(
                    {"events": []},
                    {},
                    None,
                    {"qa_operation": "count", "preferred_answer_type": "Number"},
                    "initial_final_qa_generation_after_frozen_alignment",
                    [],
                    SimpleNamespace(),
                    mode="final_generation",
                )
        api_call.assert_not_called()

    def test_run116_wrong_gold_and_generic_question_are_not_locally_rewritten(self) -> None:
        plan, graph_update, alignment, correct = self._run116_count_contract_fixture()
        hashes = {
            "graph_snapshot_hash": "run116-graph",
            "dialogue_hash": "run116-dialogue",
            "alignment_hash": "run116-alignment",
        }
        for generic_question in (
            "How many items are included based on the conversation?",
            "How many items are included according to the information provided?",
        ):
            generic = copy.deepcopy(correct)
            generic["question"] = generic_question
            generic["question_text"] = generic_question
            with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "generic question"):
                subject._build_stage64_api_qa_from_proposal(
                    current_event_plan=plan,
                    graph_update=graph_update,
                    current_alignment=alignment,
                    qa_proposal=generic,
                    hashes=hashes,
                    stage64_target={"qa_operation": "count", "preferred_answer_type": "Number"},
                )
            self.assertEqual(generic["question"], generic_question)

        leaked_contract = copy.deepcopy(correct)
        leaked_contract["question"] = "How many count_graph_contract members are included for Joanna?"
        leaked_contract["question_text"] = leaked_contract["question"]
        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "solver-visible artifact"):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=leaked_contract,
                hashes=hashes,
                stage64_target={"qa_operation": "count", "preferred_answer_type": "Number"},
            )

        wrong_gold = copy.deepcopy(correct)
        wrong_gold["gold_answer_text"] = "1"
        wrong_gold["query_plan"]["gold_answer"] = "1"
        wrong_gold["query_plan"]["normalized_gold_answer"] = "1"
        for option, text in zip(wrong_gold["options"], ("1", "2", "3", "4")):
            option["text"] = text
        with self.assertRaisesRegex(
            subject.NativeGraphDialogueReject,
            "count_gold_not_exact_included_count",
        ):
            subject._build_stage64_api_qa_from_proposal(
                current_event_plan=plan,
                graph_update=graph_update,
                current_alignment=alignment,
                qa_proposal=wrong_gold,
                hashes=hashes,
                stage64_target={"qa_operation": "count", "preferred_answer_type": "Number"},
            )
        self.assertEqual(wrong_gold["gold_answer_text"], "1")

    def test_count_number_duplicate_normalizer_repairs_only_non_gold_option_text(self) -> None:
        plan, _graph_update, _alignment, proposal = self._run105_count_duplicate_fixture(
            included_ids={"E05", "E06"}
        )
        original = copy.deepcopy(proposal)

        normalized, diagnostics = subject.normalize_stage64_count_number_duplicate_options(
            plan,
            proposal,
        )

        self.assertEqual(proposal, original)
        self.assertTrue(
            diagnostics["stage64_count_number_duplicate_option_normalizer_used"]
        )
        self.assertEqual(diagnostics["rewritten_option_labels"], ["B", "C", "D"])
        self.assertEqual(diagnostics["derived_included_count"], 2)
        self.assertEqual(diagnostics["derived_set_id"], "team_trip_candidate_destinations")
        self.assertEqual(diagnostics["gold_label"], "A")
        self.assertEqual(
            diagnostics["pre_qa_proposal_sha256"],
            subject._stage64_json_sha256(original),
        )
        self.assertEqual(
            diagnostics["post_qa_proposal_sha256"],
            subject._stage64_json_sha256(normalized),
        )
        self.assertNotEqual(
            diagnostics["pre_qa_proposal_sha256"],
            diagnostics["post_qa_proposal_sha256"],
        )
        self.assertEqual([option["text"] for option in normalized["options"]], ["2", "1", "3", "4"])
        self.assertEqual(normalized["options"][0], original["options"][0])
        for before, after in zip(original["options"][1:], normalized["options"][1:]):
            self.assertEqual(
                {key: value for key, value in before.items() if key != "text"},
                {key: value for key, value in after.items() if key != "text"},
            )
        for field in ("question", "question_text", "gold", "anchor_refs", "query_plan"):
            self.assertEqual(normalized[field], original[field])

    def test_count_number_duplicate_normalizer_run105_wrong_gold_fails_closed(self) -> None:
        # The real run105 graph has six included=true and two included=false
        # members.  Its API gold "2" is therefore wrong under formal count
        # semantics and must never be normalized into an accepted item.
        plan, _graph_update, _alignment, proposal = self._run105_count_duplicate_fixture(
            included_ids={"E01", "E02", "E03", "E04", "E05", "E06"}
        )

        with self.assertRaisesRegex(subject.NativeGraphDialogueReject, "refused wrong gold option"):
            subject.normalize_stage64_count_number_duplicate_options(plan, proposal)

    def test_count_number_duplicate_normalizer_is_noop_outside_exact_trigger(self) -> None:
        plan, _graph_update, _alignment, proposal = self._run105_count_duplicate_fixture(
            included_ids={"E05", "E06"}
        )
        unique = copy.deepcopy(proposal)
        for option, text in zip(unique["options"], ("2", "1", "3", "4")):
            option["text"] = text
        normalized, diagnostics = subject.normalize_stage64_count_number_duplicate_options(plan, unique)
        self.assertIs(normalized, unique)
        self.assertFalse(
            diagnostics["stage64_count_number_duplicate_option_normalizer_used"]
        )
        self.assertEqual(diagnostics["noop_reason"], "option_texts_already_unique")

        non_count = copy.deepcopy(proposal)
        non_count["qa_operation"] = "comparison"
        normalized, diagnostics = subject.normalize_stage64_count_number_duplicate_options(
            plan,
            non_count,
        )
        self.assertIs(normalized, non_count)
        self.assertEqual(diagnostics["noop_reason"], "not_count_operation")

        non_number = copy.deepcopy(proposal)
        non_number["answer_type"] = "Event"
        normalized, diagnostics = subject.normalize_stage64_count_number_duplicate_options(
            plan,
            non_number,
        )
        self.assertIs(normalized, non_number)
        self.assertEqual(diagnostics["noop_reason"], "not_number_answer_type")

    def test_count_number_duplicate_normalizer_reenters_builder_and_semantic_gate(self) -> None:
        plan, graph_update, alignment, proposal = self._run105_count_duplicate_fixture(
            included_ids={"E05", "E06"}
        )

        qa = subject._build_stage64_api_qa_from_proposal(
            current_event_plan=plan,
            graph_update=graph_update,
            current_alignment=alignment,
            qa_proposal=proposal,
            hashes={
                "graph_snapshot_hash": "run105-graph",
                "dialogue_hash": "run105-dialogue",
                "alignment_hash": "run105-alignment",
            },
            stage64_target={"qa_operation": "count", "preferred_answer_type": "Number"},
        )

        self.assertEqual([option["text"] for option in qa["options"]], ["2", "1", "3", "4"])
        self.assertEqual(qa["gold"], ["A"])
        self.assertTrue(qa["stage64_api_qa_schema_pass"])
        self.assertTrue(
            qa["stage64_count_number_duplicate_option_normalizer"]
            ["stage64_count_number_duplicate_option_normalizer_used"]
        )
        self.assertEqual(
            subject.stage64_quality.semantic_quality_issues(
                {
                    "qa": qa,
                    "graph_update": graph_update,
                    "current_event_plan": plan,
                }
            ),
            [],
        )

        summary = subject._empty_summary()
        subject._update_summary_for_accept(
            summary,
            {
                "qa": qa,
                "attack_skill": "count",
                "metadata": {},
                "native_contract": {},
                "validation": {"anti_shortcut_pass": True},
            },
            plan,
            {
                "required_event_count": 8,
                "aligned_required_event_count": 8,
                "missing_required_event_instance_ids": [],
                "coverage_pass": True,
            },
        )
        self.assertEqual(
            summary["stage64_count_number_duplicate_option_normalizer_used_count"],
            1,
        )
        self.assertEqual(
            summary[
                "stage64_count_number_duplicate_option_normalizer_rewritten_option_count"
            ],
            3,
        )

    def test_count_reference_completion_requires_one_compatible_candidate_set(self) -> None:
        events = [
            {
                "event_id": f"E{index}",
                "event_instance_id": f"E{index}",
                "entity": "Atlas project",
                "slot": "active_member",
                "value": name,
                "set_membership": {
                    "set_id": "atlas_active_members",
                    "included": index <= 3,
                },
            }
            for index, name in enumerate(("Maya", "Noah", "Li", "Omar", "Zoe", "Iris"), start=1)
        ]
        partial = {
            "resolved_reasoning_event_instance_ids": ["E1", "E2"],
            "resolved_required_event_instance_ids": ["E1"],
            "resolved_supporting_event_instance_ids": ["E2"],
            "resolved_gold_event_instance_ids": ["E1"],
            "resolved_distractor_event_instance_ids": ["E4"],
        }
        original = copy.deepcopy(partial)
        completed = subject._stage64_count_reference_completion({"events": events}, partial)

        self.assertEqual(completed["event_instance_ids"], [f"E{index}" for index in range(1, 7)])
        self.assertEqual(completed["included_count"], 3)
        self.assertEqual(partial, original)

        unrelated = copy.deepcopy(partial)
        unrelated["resolved_distractor_event_instance_ids"] = ["unrelated-event"]
        self.assertIsNone(
            subject._stage64_count_reference_completion({"events": events}, unrelated)
        )

    def test_stage64_local_qa_repair_initial_success_is_noop(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        original = copy.deepcopy(kwargs)
        audit: dict = {}

        def build(*args, **_kwargs):
            args[3]["builder_mutation"] = True
            return {"question_type": "single_choice", "gold": ["A"]}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(subject, "call_api_stage64_qa_repair") as repair:
            qa, chosen = subject.build_stage64_qa_with_local_repairs(
                **kwargs,
                audit=audit,
            )

        repair.assert_not_called()
        self.assertEqual(qa["gold"], ["A"])
        self.assertEqual(chosen, original["initial_qa_proposal"])
        for key in (
            "current_event_plan",
            "graph_update",
            "current_alignment",
            "initial_qa_proposal",
            "hashes",
            "final_dialogue",
            "policy",
            "seed_row",
            "appended_dialogue_turns",
        ):
            self.assertEqual(kwargs[key], original[key])
        self.assertTrue(audit["success"])
        self.assertEqual(audit["repair_attempt_count"], 0)
        self.assertEqual(audit["chosen_candidate_index"], 0)
        self.assertTrue(audit["frozen_context_verified"])

        summary = subject._empty_summary()
        raw_record: dict = {}
        subject._record_stage64_qa_local_repair_audit(
            summary,
            raw_record,
            audit,
        )
        self.assertEqual(summary["stage64_qa_repair_attempt_count"], 0)
        self.assertEqual(summary["stage64_qa_repair_success_count"], 0)
        self.assertEqual(summary["stage64_qa_repair_candidate_count"], 1)
        self.assertEqual(len(raw_record["stage64_qa_candidate_history"]), 1)

    def test_stage64_local_qa_repair_same_sha_fails_after_one_no_progress_call(
        self,
    ) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        audit: dict = {}

        def build(*_args, **_kwargs):
            raise subject.NativeGraphDialogueReject(
                "stage64 API QA requires query_plan object"
            )

        def unchanged_repair(*args, **_kwargs):
            return {
                "qa_proposal": copy.deepcopy(args[2])
            }, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=unchanged_repair,
        ) as repair:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "repair made no progress: proposal SHA unchanged",
            ):
                subject.build_stage64_qa_with_local_repairs(
                    **kwargs,
                    audit=audit,
                )

        self.assertEqual(repair.call_count, 1)
        self.assertEqual(audit["repair_attempt_count"], 1)
        self.assertEqual(audit["no_progress_same_sha_count"], 1)
        self.assertTrue(audit["nonrepairable_failure"])
        self.assertFalse(audit["exhausted"])
        self.assertEqual(len(audit["candidate_history"]), 1)
        self.assertEqual(len(audit["repair_attempts"]), 1)
        repair_record = audit["repair_attempts"][0]
        self.assertTrue(repair_record["no_progress_same_sha"])
        self.assertFalse(repair_record["candidate_replaced"])
        self.assertEqual(repair_record["candidate_index"], 0)
        self.assertEqual(
            repair_record["source_candidate_sha256"],
            repair_record["proposal_sha256"],
        )

        summary = subject._empty_summary()
        raw_record: dict = {}
        subject._record_stage64_qa_local_repair_audit(
            summary,
            raw_record,
            audit,
        )
        self.assertEqual(
            summary["stage64_qa_repair_no_progress_same_sha_count"], 1
        )
        self.assertEqual(summary["stage64_qa_repair_attempt_count"], 1)
        self.assertEqual(summary["stage64_qa_repair_failed_count"], 1)
        self.assertEqual(summary["stage64_qa_repair_candidate_count"], 1)
        self.assertTrue(
            raw_record["api_stage64_qa_repair_attempts"][0][
                "no_progress_same_sha"
            ]
        )

    def test_stage64_local_qa_repair_succeeds_after_multiple_repairs(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        original = copy.deepcopy(kwargs)
        audit: dict = {}
        repair_calls: list[dict] = []

        def build(*args, **_kwargs):
            proposal = args[3]
            if int(proposal.get("revision") or 0) < 2:
                raise subject.NativeGraphDialogueReject(
                    "stage64 API QA options require another repair"
                )
            return {
                "question_type": "single_choice",
                "gold": ["A"],
                "revision": proposal["revision"],
            }

        def repair(*args, **call_kwargs):
            previous = copy.deepcopy(args[2])
            repair_calls.append(
                {
                    "previous": previous,
                    "frozen_dependencies": copy.deepcopy(
                        call_kwargs.get("frozen_dependencies")
                    ),
                }
            )
            args[0]["mutated_inside_repair"] = True
            args[5].append({"text": "mutation on deep copy"})
            repaired = copy.deepcopy(previous)
            repaired["revision"] = int(previous.get("revision") or 0) + 1
            return {"qa_proposal": repaired}, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=repair,
        ):
            qa, chosen = subject.build_stage64_qa_with_local_repairs(
                **kwargs,
                audit=audit,
            )

        self.assertEqual(qa["revision"], 2)
        self.assertEqual(chosen["revision"], 2)
        for key in (
            "current_event_plan",
            "graph_update",
            "current_alignment",
            "initial_qa_proposal",
            "hashes",
            "final_dialogue",
            "policy",
            "seed_row",
            "appended_dialogue_turns",
        ):
            self.assertEqual(kwargs[key], original[key])
        self.assertEqual(len(repair_calls), 2)
        self.assertTrue(
            all(
                call["frozen_dependencies"] == kwargs["hashes"]
                for call in repair_calls
            )
        )
        self.assertEqual(audit["repair_attempt_count"], 2)
        self.assertEqual(audit["candidate_build_failure_count"], 2)
        self.assertEqual(len(audit["candidate_history"]), 3)
        self.assertTrue(audit["frozen_context_verified"])
        self.assertEqual(
            audit["frozen_context_sha256"],
            audit["frozen_context_post_sha256"],
        )

        summary = subject._empty_summary()
        raw_record: dict = {}
        subject._record_stage64_qa_local_repair_audit(
            summary,
            raw_record,
            audit,
        )
        self.assertEqual(summary["stage64_qa_repair_attempt_count"], 2)
        self.assertEqual(summary["stage64_qa_repair_success_count"], 1)
        self.assertEqual(
            summary["stage64_qa_repair_saved_full_regeneration_count"],
            1,
        )
        self.assertEqual(len(raw_record["api_stage64_qa_repair_attempts"]), 2)
        self.assertTrue(
            all(
                row["frozen_dependencies"] == kwargs["hashes"]
                for row in raw_record["api_stage64_qa_repair_attempts"]
            )
        )

    def test_stage64_local_qa_repair_malformed_payload_keeps_last_candidate(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        audit: dict = {}
        previous_candidates: list[dict] = []
        repair_number = 0

        def build(*args, **_kwargs):
            proposal = args[3]
            if proposal.get("revision") != 1:
                raise subject.NativeGraphDialogueReject(
                    "stage64 API QA qa_proposal is incomplete"
                )
            return {"question_type": "single_choice", "gold": ["B"]}

        def repair(*args, **_kwargs):
            nonlocal repair_number
            repair_number += 1
            previous_candidates.append(copy.deepcopy(args[2]))
            if repair_number == 1:
                return None, {"used": False, "error_type": "parse_failure"}
            repaired = copy.deepcopy(args[2])
            repaired["revision"] = 1
            return {"qa_proposal": repaired}, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=repair,
        ):
            _qa, chosen = subject.build_stage64_qa_with_local_repairs(
                **kwargs,
                audit=audit,
            )

        self.assertEqual(chosen["revision"], 1)
        self.assertEqual(previous_candidates[0], previous_candidates[1])
        self.assertEqual(previous_candidates[0]["revision"], 0)
        self.assertFalse(audit["repair_attempts"][0]["candidate_replaced"])
        self.assertTrue(audit["repair_attempts"][1]["candidate_replaced"])
        self.assertEqual(audit["api_failure_count"], 1)
        self.assertEqual(len(audit["candidate_history"]), 2)

    def test_stage64_local_qa_repair_recovers_run124_schema_regression(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        initial = kwargs["initial_qa_proposal"]
        initial.update(
            {
                "question_text": "What is the score difference?",
                "answer_type": "Number",
            }
        )
        audit: dict = {}
        responses = [
            {
                "qa_proposal": {
                    "slot": "slot-a",
                    "revision": 1,
                    "question": "What is the score difference?",
                    "question_text": "What is the score difference?",
                    "answer_type": "Number",
                }
            },
            {
                "qa_proposal": {
                    "slot": "slot-a",
                    "revision": 2,
                    "question": "What is the score difference?",
                    "question_text": "What is the score difference?",
                    "answer_type": "Number",
                    "query_plan": {"qa_operation": "comparison"},
                }
            },
        ]

        def build(*args, **_kwargs):
            proposal = args[3]
            if not proposal.get("question"):
                raise subject.NativeGraphDialogueReject(
                    "stage64 API QA question_text must exactly equal question"
                )
            if not isinstance(proposal.get("query_plan"), dict):
                raise subject.NativeGraphDialogueReject(
                    "stage64 API QA schema completion normalizer requires query_plan object as independent evidence"
                )
            return {"question_type": "single_choice", "gold": ["A"]}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=[
                (responses[0], {"used": True, "attempts": 1}),
                (responses[1], {"used": True, "attempts": 1}),
            ],
        ):
            _qa, chosen = subject.build_stage64_qa_with_local_repairs(
                **kwargs,
                audit=audit,
            )

        self.assertIn("query_plan", audit["candidate_history"][0]["proposal"])
        self.assertNotIn("query_plan", audit["candidate_history"][1]["proposal"])
        self.assertIn("query_plan", chosen)
        self.assertEqual(audit["chosen_candidate_index"], 2)
        self.assertEqual(audit["repair_attempt_count"], 2)

    def test_stage64_local_qa_repair_nonrepairable_fails_without_api(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        audit: dict = {}
        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=subject.NativeGraphDialogueReject(
                "frozen graph invariant failed"
            ),
        ), mock.patch.object(subject, "call_api_stage64_qa_repair") as repair:
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "frozen graph invariant failed",
            ):
                subject.build_stage64_qa_with_local_repairs(
                    **kwargs,
                    audit=audit,
                )

        repair.assert_not_called()
        self.assertTrue(audit["nonrepairable_failure"])
        self.assertFalse(audit["repair_triggered"])
        self.assertEqual(audit["repair_attempt_count"], 0)
        self.assertFalse(audit["exhausted"])

    def test_stage64_local_qa_repair_exhaustion_records_one_failed_sample(self) -> None:
        kwargs = self._local_stage64_qa_repair_kwargs()
        kwargs["max_repair_attempts"] = 2
        audit: dict = {}
        repair_number = 0

        def repair(*args, **_kwargs):
            nonlocal repair_number
            repair_number += 1
            proposal = copy.deepcopy(args[2])
            proposal["revision"] = repair_number
            return {"qa_proposal": proposal}, {"used": True, "attempts": 1}

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=subject.NativeGraphDialogueReject(
                "stage64 API QA options remain invalid"
            ),
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=repair,
        ):
            with self.assertRaisesRegex(
                subject.NativeGraphDialogueReject,
                "options remain invalid",
            ):
                subject.build_stage64_qa_with_local_repairs(
                    **kwargs,
                    audit=audit,
                )

        self.assertEqual(repair_number, 2)
        self.assertEqual(audit["repair_attempt_count"], 2)
        self.assertEqual(audit["candidate_build_failure_count"], 3)
        self.assertEqual(len(audit["candidate_history"]), 3)
        self.assertTrue(audit["exhausted"])
        self.assertTrue(audit["frozen_context_verified"])

        summary = subject._empty_summary()
        raw_record: dict = {}
        subject._record_stage64_qa_local_repair_audit(
            summary,
            raw_record,
            audit,
        )
        self.assertEqual(summary["stage64_qa_repair_failed_count"], 1)
        self.assertEqual(summary["stage64_qa_repair_exhausted_count"], 1)
        self.assertEqual(summary["stage64_qa_repair_attempt_count"], 2)
        self.assertEqual(len(raw_record["api_stage64_qa_repair_attempts"]), 2)
        self.assertTrue(
            all(
                row["frozen_context_sha256"]
                == audit["frozen_context_sha256"]
                for row in raw_record["api_stage64_qa_repair_attempts"]
            )
        )

    def test_stage64_local_qa_repair_isolates_concurrent_worker_histories(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(2)

        def build(*args, **_kwargs):
            proposal = args[3]
            if proposal.get("revision") != 1:
                raise subject.NativeGraphDialogueReject(
                    "stage64 API QA concurrent candidate needs repair"
                )
            return {
                "question_type": "single_choice",
                "gold": ["A"],
                "slot": proposal["slot"],
            }

        def repair(*args, **_kwargs):
            previous = copy.deepcopy(args[2])
            barrier.wait(timeout=5)
            previous["revision"] = 1
            return {"qa_proposal": previous}, {"used": True, "attempts": 1}

        def run(slot: str) -> tuple[dict, dict, dict]:
            kwargs = self._local_stage64_qa_repair_kwargs(slot)
            audit: dict = {}
            qa, chosen = subject.build_stage64_qa_with_local_repairs(
                **kwargs,
                audit=audit,
            )
            return qa, chosen, audit

        with mock.patch.object(
            subject,
            "build_qa_from_current_graph_and_alignment",
            side_effect=build,
        ), mock.patch.object(
            subject,
            "call_api_stage64_qa_repair",
            side_effect=repair,
        ), ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, ("slot-a", "slot-b")))

        self.assertEqual({qa["slot"] for qa, _chosen, _audit in results}, {"slot-a", "slot-b"})
        for qa, chosen, audit in results:
            slot = qa["slot"]
            self.assertEqual(chosen["slot"], slot)
            self.assertTrue(audit["frozen_context_verified"])
            self.assertEqual(audit["repair_attempt_count"], 1)
            self.assertTrue(
                all(
                    row["proposal"]["slot"] == slot
                    for row in audit["candidate_history"]
                )
            )
            self.assertEqual(
                audit["frozen_dependencies"]["graph_snapshot_hash"],
                f"graph-{slot}",
            )

    def test_stage64_qa_repair_cli_defaults_to_three_and_accepts_override(self) -> None:
        from scripts import run_auto_graph_expansion_round as runner

        with mock.patch("sys.argv", ["run_auto_graph_expansion_round.py"]):
            defaults = runner.parse_args()
        with mock.patch(
            "sys.argv",
            [
                "run_auto_graph_expansion_round.py",
                "--stage64-qa-repair-attempts",
                "5",
            ],
        ):
            overridden = runner.parse_args()

        self.assertEqual(defaults.stage64_qa_repair_attempts, 3)
        self.assertEqual(overridden.stage64_qa_repair_attempts, 5)
        generated_args = runner.generation_args(overridden, {}, {})
        self.assertEqual(generated_args.stage64_qa_repair_attempts, 5)

    def test_worker_boolean_summary_flags_are_or_merged_not_counted(self) -> None:
        summary = {"challenger_api_used": False}

        subject._merge_local_summary(summary, {"challenger_api_used": True})
        subject._merge_local_summary(summary, {"challenger_api_used": True})
        subject._merge_local_summary(summary, {"challenger_api_used": True})

        self.assertIs(summary["challenger_api_used"], True)


if __name__ == "__main__":
    unittest.main()
