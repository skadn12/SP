from __future__ import annotations

import copy
import math
import unittest

from scripts.current_method import stage3_claim_ir
from scripts.current_method import stage3_truth_first_discourse as discourse
from tests.stage3_truth_test_helpers import context_sentence, flat_context_body


def golden_graph() -> dict:
    def event(
        event_id: str,
        entity: str,
        slot: str,
        value: object,
        time_index: int,
        *,
        required: bool = True,
    ) -> dict:
        return {
            "event_id": event_id,
            "entity": entity,
            "slot": slot,
            "value": value,
            "status": "active",
            "time_index": time_index,
            "global_time_index": time_index,
            "required_for_dialogue": required,
            "qa_eligible": required,
        }

    return {
        "schema_version": "semantic_graph_v2",
        "graph_id": "truth-first-discourse-golden",
        "operation": "cause_consequence",
        "events": [
            event("E1", "Maya", "destination", "Cedar Hall", 1),
            event("E2", "Noah", "assigned_partner", "Maya", 2),
            event("E3", "Maya", "affiliated_group", "Cedar Guild", 3),
            event("E4", "Cedar Guild", "scheduled_event", "Spring Workshop", 4),
            event("E5", "Spring Workshop", "event_location", "Cedar Hall", 5),
            event(
                "E6",
                "Spring Workshop",
                "booking_outcome",
                "gallery booking",
                6,
                required=False,
            ),
        ],
        "relations": [
            {
                "claim_type": "cause",
                "subject_event_id": "E5",
                "object_event_ids": ["E6"],
                "predicate": "causes",
                "edge_role": "semantic",
                "qa_eligible": True,
                "polarity": "positive",
                "status": "active",
                "time_index": 6,
                "global_time_index": 6,
                "canonical_support_claim": {
                    "subject": "Spring Workshop",
                    "predicate": "causes",
                    "objects": ["Spring Workshop"],
                    "polarity": "positive",
                    "status": "active",
                    "time": {
                        "cause_time_index": 5,
                        "time_index": 6,
                        "global_time_index": 6,
                    },
                    "condition": None,
                    "scope": None,
                    "cause_event": {
                        "event_id": "E5",
                        "entity": "Spring Workshop",
                        "slot": "event_location",
                        "value": "Cedar Hall",
                        "status": "active",
                        "time_index": 5,
                        "global_time_index": 5,
                    },
                    "effect_event": {
                        "event_id": "E6",
                        "entity": "Spring Workshop",
                        "slot": "booking_outcome",
                        "value": "gallery booking",
                        "status": "active",
                        "time_index": 6,
                        "global_time_index": 6,
                    },
                },
            }
        ],
    }


def golden_parent_context() -> dict:
    return {
        "speaker_roster": ["Maya", "Noah"],
        "topic": "Cedar workshop planning",
        "setting": "Cedar Hall planning table",
        "retrieval": {
            "selected_parent_event_ids": ["P11", "P12"],
            "selected_parent_event_count": 2,
        },
    }


def golden_bundle(graph: dict) -> dict:
    return stage3_claim_ir.compile_claim_ir(graph)


def golden_artifacts() -> tuple[dict, dict, dict, dict, dict]:
    graph = golden_graph()
    bundle = golden_bundle(graph)
    parent = golden_parent_context()
    plan = discourse.build_discourse_plan(graph, bundle, parent)
    quote_by_claim = {
        support["claim_id"]: support["exact_support_quote"]
        for slot in plan["turn_slots"]
        for support in slot["frozen_support"]
    }
    turn_number = 1
    response_acts: list[dict] = []
    for act in plan["acts"]:
        response_beats: list[dict] = []
        for beat in act["beats"]:
            turns: list[dict] = []
            for claim_id in beat["required_claim_ids"]:
                quote = quote_by_claim[claim_id]
                turns.append(
                    {
                        "turn_id": f"D1:{turn_number}",
                        "speaker": plan["speaker_roster"][(turn_number - 1) % 2],
                        "text": (
                            f"While we review the Cedar workshop plan, {quote} "
                            "That detail keeps the Cedar Hall discussion moving."
                        ),
                        "support": [
                            {
                                "claim_id": claim_id,
                                "exact_support_quote": quote,
                            }
                        ],
                    }
                )
                turn_number += 1
            response_beats.append({"beat_id": beat["beat_id"], "turns": turns})
        response_acts.append({"act_id": act["act_id"], "beats": response_beats})
    response = {
        "schema_version": discourse.API_RESPONSE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "frozen_input_hashes": copy.deepcopy(plan["frozen_input_hashes"]),
        "acts": response_acts,
    }
    return graph, bundle, parent, plan, response


def validate(
    graph: dict, bundle: dict, parent: dict, plan: dict, response: dict
) -> dict:
    return discourse.validate_api_response(
        plan,
        response,
        frozen_graph=graph,
        frozen_claim_bundle=bundle,
        parent_context=parent,
    )


def codes(report: dict) -> set[str]:
    return {item["code"] for item in report["issues"]}


def all_support_rows(response: dict) -> list[dict]:
    return [
        support
        for act in response["acts"]
        for beat in act["beats"]
        for turn in beat["turns"]
        for support in turn["support"]
    ]


def all_turns(response: dict) -> list[dict]:
    return [
        turn
        for act in response["acts"]
        for beat in act["beats"]
        for turn in beat["turns"]
    ]


def flat_body(plan: dict) -> dict:
    return flat_context_body(
        plan["turn_slots"], plan["topic_spine"]["topic"]
    )


def segment_body(brief: dict) -> dict:
    topic = str(brief["topic_spine"]["topic"])
    turns = []
    for slot in brief["turn_slots"]:
        context = []
        sentence_cursor = int(slot["order_index"]) * 32
        minimum_words = int(
            slot.get("minimum_context_words")
            or slot.get("target_words")
            or 24
        )
        minimum_sentences = max(
            2, int(slot.get("target_context_sentence_count") or 0)
        )
        while (
            len(context) < minimum_sentences
            or sum(len(sentence.split()) for sentence in context) < minimum_words
        ):
            context.append(
                context_sentence(sentence_cursor + len(context), topic)
            )
        turns.append(
            {
                "turn_slot_id": slot["turn_slot_id"],
                "context_sentences": context,
            }
        )
    return {"turns": turns}


def nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(nested_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(nested_keys(child))
        return result
    return set()


def nested_scalars(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            scalar
            for child in value.values()
            for scalar in nested_scalars(child)
        ]
    if isinstance(value, list):
        return [scalar for child in value for scalar in nested_scalars(child)]
    return [str(value)] if isinstance(value, (str, int, float, bool)) else []


class TruthFirstDiscoursePlanTest(unittest.TestCase):
    def test_plan_is_deterministic_bounded_and_does_not_mutate_inputs(self) -> None:
        graph = golden_graph()
        bundle = golden_bundle(graph)
        parent = golden_parent_context()
        before = copy.deepcopy((graph, bundle, parent))

        first = discourse.build_discourse_plan(graph, bundle, parent)
        second = discourse.build_discourse_plan(graph, bundle, parent)

        self.assertEqual(first, second)
        self.assertEqual((graph, bundle, parent), before)
        self.assertLessEqual(len(first["acts"]), 2)
        self.assertTrue(
            all(len(act["beats"]) <= 2 for act in first["acts"])
        )
        planned = [
            claim_id
            for act in first["acts"]
            for beat in act["beats"]
            for claim_id in beat["required_claim_ids"]
        ]
        self.assertCountEqual(planned, first["required_claim_ids"])
        self.assertEqual(len(planned), len(set(planned)))

    def test_plan_hash_covers_semantics_and_parent_context(self) -> None:
        graph, bundle, parent, plan, _response = golden_artifacts()

        self.assertEqual(
            plan["frozen_input_hashes"],
            {
                "graph_sha256": discourse.canonical_sha256(graph),
                "claim_bundle_sha256": discourse.canonical_sha256(bundle),
                "parent_context_sha256": discourse.canonical_sha256(parent),
            },
        )
        unhashed = copy.deepcopy(plan)
        claimed = unhashed.pop("plan_sha256")
        self.assertEqual(claimed, discourse.canonical_sha256(unhashed))

    def test_api_contract_has_no_truth_mutation_output_surface(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()

        contract = discourse.build_api_contract(plan)
        response_schema_keys = nested_keys(contract["response_schema"])

        self.assertTrue(
            discourse._FORBIDDEN_RESPONSE_KEYS.isdisjoint(  # noqa: SLF001
                {str(key).casefold() for key in response_schema_keys}
            )
        )
        self.assertNotIn("graph", contract["read_only_brief"])
        self.assertNotIn("proof", contract["read_only_brief"])
        self.assertNotIn("gold", contract["read_only_brief"])
        self.assertNotIn("reasoning_path", contract["read_only_brief"])
        self.assertNotIn("claim_obligations", contract["read_only_brief"])
        self.assertEqual(
            contract["read_only_brief"]["required_turn_row_count"],
            len(contract["read_only_brief"]["turn_slots"]),
        )
        self.assertNotIn("frozen_support", response_schema_keys)
        self.assertNotIn("canonical_fact", response_schema_keys)
        self.assertIn("context_sentences", response_schema_keys)
        response_schema = contract["response_schema"]
        self.assertEqual(response_schema["top_level_required_keys"], ["turns"])
        self.assertFalse(response_schema["top_level_additional_keys_allowed"])
        turns_schema = response_schema["turns"]
        exposed_slots = contract["read_only_brief"]["turn_slots"]
        self.assertEqual(turns_schema["item_count_exact"], len(exposed_slots))
        self.assertEqual(
            turns_schema["turn_slot_ids_exact_order"],
            [slot["turn_slot_id"] for slot in exposed_slots],
        )
        context_schema = turns_schema["item_schema"]["context_sentences"]
        self.assertGreaterEqual(context_schema["minimum_items"], 2)
        self.assertGreaterEqual(
            context_schema["maximum_items"], context_schema["minimum_items"]
        )
        self.assertNotIn("schema_version", contract["response_schema"])
        self.assertNotIn("plan_sha256", contract["response_schema"])
        self.assertTrue(
            any(
                "never return a one-sentence row" in rule
                for rule in contract["generation_rules"]
            )
        )

    def test_api_contract_recursively_redacts_child_slot_value_and_status(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        self.assertIn("destination", plan["topic_spine"]["anchor_terms"])
        self.assertIn("cedar", plan["topic_spine"]["anchor_terms"])

        contract = discourse.build_api_contract(plan)
        normalized_scalars = " ".join(nested_scalars(contract)).casefold()

        self.assertNotIn("destination", normalized_scalars)
        self.assertNotIn("cedar hall", normalized_scalars)
        self.assertNotIn("active", normalized_scalars)
        safe_spine = contract["read_only_brief"]["topic_spine"]
        self.assertEqual(
            safe_spine["redaction"],
            "parent_topic_minus_child_truth_surfaces",
        )
        reserved = discourse._reserved_truth_tokens(  # noqa: SLF001
            plan["semantic_claims"]
        )
        self.assertTrue(
            reserved.isdisjoint(
                {
                    token.casefold()
                    for token in safe_spine["anchor_terms"]
                }
            )
        )

    def test_long_plan_allocates_exact_flat_turn_slots(self) -> None:
        graph = golden_graph()
        bundle = golden_bundle(graph)
        parent = golden_parent_context()
        plan = discourse.build_discourse_plan(
            graph,
            bundle,
            parent,
            minimum_turns=24,
            target_dialogue_tokens=4000,
        )
        slots = plan["turn_slots"]
        natural_turn_count = math.ceil(
            4000
            * discourse.CONTEXT_WORD_BUDGET_FACTOR
            / discourse.NATURAL_CONTEXT_TARGET_WORDS_PER_TURN
        )
        self.assertEqual(len(slots), natural_turn_count)
        self.assertEqual(
            len({slot["turn_slot_id"] for slot in slots}), natural_turn_count
        )
        self.assertEqual(len({slot["turn_id"] for slot in slots}), natural_turn_count)
        self.assertTrue(
            all(
                slot["target_words"]
                == discourse.NATURAL_CONTEXT_TARGET_WORDS_PER_TURN
                for slot in slots
            )
        )
        self.assertTrue(
            all(
                2 <= slot["target_context_sentence_count"] <= 3
                for slot in slots
            )
        )
        self.assertTrue(
            all(
                slot["lead_sentence_count"] + slot["follow_sentence_count"]
                == slot["target_context_sentence_count"]
                for slot in slots
            )
        )
        assigned = [claim_id for slot in slots for claim_id in slot["required_claim_ids"]]
        self.assertCountEqual(assigned, plan["required_claim_ids"])
        filler_moves = {
            slot["discourse_move"]
            for slot in slots
            if not slot["contains_local_support"]
        }
        self.assertGreater(len(filler_moves), 1)
        contract = discourse.build_api_contract(plan)
        exposed_slots = contract["read_only_brief"]["turn_slots"]
        self.assertTrue(all("frozen_support" not in slot for slot in exposed_slots))
        self.assertTrue(
            all("required_claim_ids" not in slot for slot in exposed_slots)
        )
        self.assertTrue(
            all(
                "Faithfully express" not in slot["purpose"]
                for slot in exposed_slots
            )
        )
        per_beat = {}
        for slot in slots:
            key = (slot["act_id"], slot["beat_id"])
            per_beat[key] = per_beat.get(key, 0) + 1
        self.assertEqual(len(per_beat), 4)
        self.assertTrue(
            all(
                count <= discourse.MAX_TURNS_PER_BEAT
                for count in per_beat.values()
            )
        )
        segment_ids = discourse.discourse_segment_ids(plan)
        self.assertEqual(segment_ids, ["A1B1", "A1B2", "A2B1", "A2B2"])
        segmented_slots = [
            slot
            for segment_id in segment_ids
            for slot in discourse.segment_turn_slots(plan, segment_id)
        ]
        self.assertEqual(
            [slot["turn_slot_id"] for slot in segmented_slots],
            [slot["turn_slot_id"] for slot in slots],
        )

    def test_long_plan_capacity_does_not_depend_on_claim_count(self) -> None:
        graph = golden_graph()
        source_bundle = golden_bundle(graph)
        one_claim_bundle = {
            "claims": [copy.deepcopy(source_bundle["claims"][0])]
        }

        plan = discourse.build_discourse_plan(
            graph,
            one_claim_bundle,
            golden_parent_context(),
            minimum_turns=24,
            target_dialogue_tokens=4000,
        )

        self.assertEqual(len(plan["required_claim_ids"]), 1)
        self.assertEqual(discourse.discourse_segment_ids(plan), [
            "A1B1",
            "A1B2",
            "A2B1",
            "A2B2",
        ])
        self.assertEqual(len(plan["turn_slots"]), 115)
        self.assertEqual(
            sum(
                len(beat["required_claim_ids"])
                for act in plan["acts"]
                for beat in act["beats"]
            ),
            1,
        )

    def test_missing_roster_and_noncanonical_claim_fail_before_generation(self) -> None:
        graph = golden_graph()
        bundle = golden_bundle(graph)
        with self.assertRaisesRegex(discourse.DiscourseContractError, "speaker roster"):
            discourse.build_discourse_plan(graph, bundle, {})

        broken = copy.deepcopy(bundle)
        broken["claims"][0]["truth_signature"]["objects"] = ["wrong"]
        with self.assertRaisesRegex(discourse.DiscourseContractError, "not canonical"):
            discourse.build_discourse_plan(graph, broken, golden_parent_context())


class TruthFirstDiscourseValidationTest(unittest.TestCase):
    def test_golden_continuous_response_passes(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()

        report = validate(graph, bundle, parent, plan, response)

        self.assertEqual(
            report,
            {
                "schema_version": discourse.VALIDATION_SCHEMA_VERSION,
                "pass": True,
                "issue_count": 0,
                "issue_counts_by_code": {},
                "issues": [],
            },
        )

    def test_flat_creative_body_is_compiled_without_changing_text_or_support(self) -> None:
        graph, bundle, parent, plan, _response = golden_artifacts()
        raw = flat_body(plan)
        before = copy.deepcopy(raw)
        normalized, audit = discourse.normalize_api_response_envelope(plan, raw)

        self.assertEqual(raw, before)
        self.assertTrue(audit["pass"], audit["issues"])
        self.assertEqual(audit["body_format"], "flat_turn_slots")
        self.assertEqual(
            audit["injected_fields"],
            ["schema_version", "plan_sha256", "frozen_input_hashes"],
        )
        self.assertEqual(
            audit["creative_body_sha256_before"],
            audit["creative_body_sha256_after"],
        )
        self.assertTrue(validate(graph, bundle, parent, plan, normalized)["pass"])

    def test_single_context_array_is_partitioned_without_text_edits(self) -> None:
        graph, bundle, parent, plan, _response = golden_artifacts()
        legacy = flat_body(plan)
        compact = {
            "turns": [
                {
                    "turn_slot_id": row["turn_slot_id"],
                    "context_sentences": (
                        row["lead_in_sentences"] + row["follow_up_sentences"]
                    ),
                }
                for row in legacy["turns"]
            ]
        }
        before = copy.deepcopy(compact)

        normalized, audit = discourse.normalize_api_response_envelope(plan, compact)

        self.assertTrue(audit["pass"], audit["issues"])
        self.assertEqual(compact, before)
        self.assertEqual(
            audit["creative_body_sha256_before"],
            audit["creative_body_sha256_after"],
        )
        self.assertTrue(
            all(
                row["sentence_sequence_sha256_before"]
                == row["sentence_sequence_sha256_after"]
                and row["sentence_text_modified"] is False
                and row["sentence_order_modified"] is False
                and row["sentences_dropped"] == 0
                for row in audit["representation_normalization"]
            )
        )
        self.assertTrue(validate(graph, bundle, parent, plan, normalized)["pass"])

    def test_every_legacy_left_right_partition_is_safely_rebalanced(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        baseline = flat_body(plan)
        sequence = (
            baseline["turns"][0]["lead_in_sentences"]
            + baseline["turns"][0]["follow_up_sentences"]
        )
        for cut in range(len(sequence) + 1):
            with self.subTest(cut=cut):
                raw = copy.deepcopy(baseline)
                raw["turns"][0]["lead_in_sentences"] = sequence[:cut]
                raw["turns"][0]["follow_up_sentences"] = sequence[cut:]
                _normalized, audit = discourse.normalize_api_response_envelope(
                    plan, raw
                )
                self.assertTrue(audit["pass"], audit["issues"])
                row_audit = audit["representation_normalization"][0]
                self.assertEqual(
                    row_audit["sentence_sequence_sha256_before"],
                    row_audit["sentence_sequence_sha256_after"],
                )
                self.assertEqual(row_audit["sentences_dropped"], 0)

    def test_representation_leniency_never_hides_duplicate_or_excess_text(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        duplicate = flat_body(plan)
        repeated = duplicate["turns"][0]["lead_in_sentences"][0]
        duplicate["turns"][0]["follow_up_sentences"].append(repeated)

        _normalized, duplicate_audit = discourse.normalize_api_response_envelope(
            plan, duplicate
        )

        self.assertFalse(duplicate_audit["pass"])
        self.assertIn(
            "exact_context_sentence_duplicate",
            {issue["code"] for issue in duplicate_audit["issues"]},
        )
        self.assertTrue(
            all(
                row["sentences_dropped"] == 0
                for row in duplicate_audit["representation_normalization"]
            )
        )

        excessive = flat_body(plan)
        excessive["turns"][0]["follow_up_sentences"].extend(
            [
                "Careful listeners compare the meeting atmosphere with earlier conversations while avoiding every hidden factual conclusion.",
                "A patient pause gives both speakers room to consider their priorities before the discussion moves forward.",
            ]
        )
        _normalized, excessive_audit = discourse.normalize_api_response_envelope(
            plan, excessive
        )
        # Sentence-array cardinality is representation, not truth.  Extra
        # distinct substantive context remains visible and is never dropped;
        # the exact tokenizer length gate still owns the overall 4k budget.
        self.assertTrue(excessive_audit["pass"], excessive_audit["issues"])
        self.assertFalse(
            excessive_audit["representation_normalization"][0][
                "sentence_count_within_target_range"
            ]
        )
        self.assertEqual(
            excessive_audit["representation_normalization"][0][
                "sentences_dropped"
            ],
            0,
        )

    def test_multisentence_array_element_is_split_without_visible_text_change(self) -> None:
        graph, bundle, parent, plan, _response = golden_artifacts()
        baseline = flat_body(plan)
        compact = {
            "turns": [
                {
                    "turn_slot_id": row["turn_slot_id"],
                    "context_sentences": [
                        " ".join(
                            row["lead_in_sentences"]
                            + row["follow_up_sentences"]
                        )
                    ],
                }
                for row in baseline["turns"]
            ]
        }

        normalized, audit = discourse.normalize_api_response_envelope(
            plan, compact
        )

        self.assertTrue(audit["pass"], audit["issues"])
        self.assertTrue(
            all(
                row["sentence_grouping_normalized"]
                and row["sentence_text_modified"] is False
                and row["visible_text_sha256_before"]
                == row["visible_text_sha256_after"]
                and row["sentences_dropped"] == 0
                for row in audit["representation_normalization"]
            )
        )
        self.assertTrue(validate(graph, bundle, parent, plan, normalized)["pass"])

    def test_sentence_splitter_preserves_abbreviation_decimal_and_quoted_boundaries(self) -> None:
        text = (
            'Dr. Maya reviewed the 3.5 mile route with the planning team. '
            '"Are we ready?" Noah asked while everyone considered the schedule. '
            'They agreed to continue the workshop discussion carefully.'
        )

        parts = discourse._split_context_fragment(text)  # noqa: SLF001

        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0].startswith("Dr. Maya"))
        self.assertIn("3.5 mile", parts[0])
        self.assertEqual(" ".join(parts), text)

    def test_segment_validation_is_cumulative_and_enforces_word_budget(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        act_ids = discourse.discourse_act_ids(plan)
        self.assertEqual(len(act_ids), 2)
        bodies = []
        accepted = []
        covered_slot_ids = []
        for act_id in act_ids:
            contract = discourse.build_api_contract(
                plan,
                act_id=act_id,
                accepted_prefix_tail=[],
            )
            exposed = contract["read_only_brief"]
            body = segment_body(exposed)
            report = discourse.validate_api_response_segment(
                plan,
                body,
                act_id=act_id,
                accepted_prefix_turns=accepted,
            )
            self.assertTrue(report["pass"], report["issues"])
            accepted.extend(copy.deepcopy(body["turns"]))
            covered_slot_ids.extend(
                row["turn_slot_id"] for row in body["turns"]
            )
            bodies.append(body)
        self.assertEqual(
            covered_slot_ids,
            [slot["turn_slot_id"] for slot in plan["turn_slots"]],
        )

        legacy_first = {"turns": []}
        for row in bodies[0]["turns"]:
            cut = max(1, len(row["context_sentences"]) // 2)
            legacy_first["turns"].append(
                {
                    "turn_slot_id": row["turn_slot_id"],
                    "lead_in_sentences": row["context_sentences"][:cut],
                    "follow_up_sentences": row["context_sentences"][cut:],
                }
            )
        legacy_report = discourse.validate_api_response_segment(
            plan, legacy_first, act_id=act_ids[0]
        )
        self.assertFalse(legacy_report["pass"])
        self.assertIn(
            "segment_compact_turn_format_required",
            {issue["code"] for issue in legacy_report["issues"]},
        )

        short = copy.deepcopy(bodies[0])
        for row in short["turns"]:
            row["context_sentences"] = [
                "The planning discussion remains focused for everyone involved.",
                "Each participant considers the planning topic with care.",
            ]
        short_report = discourse.validate_api_response_segment(
            plan, short, act_id=act_ids[0]
        )
        self.assertFalse(short_report["pass"])
        self.assertIn(
            "segment_context_word_shortfall",
            {issue["code"] for issue in short_report["issues"]},
        )

        duplicate_second = copy.deepcopy(bodies[1])
        duplicate_second["turns"][0]["context_sentences"][0] = copy.deepcopy(
            bodies[0]["turns"][0]["context_sentences"][0]
        )
        duplicate_report = discourse.validate_api_response_segment(
            plan,
            duplicate_second,
            act_id=act_ids[1],
            accepted_prefix_turns=bodies[0]["turns"],
        )
        self.assertFalse(duplicate_report["pass"])
        self.assertTrue(
            {
                "exact_context_sentence_duplicate",
                "near_context_sentence_duplicate",
            }
            & {issue["code"] for issue in duplicate_report["issues"]}
        )

    def test_beat_segment_validation_covers_production_path(self) -> None:
        graph = golden_graph()
        bundle = golden_bundle(graph)
        parent = golden_parent_context()
        plan = discourse.build_discourse_plan(
            graph,
            bundle,
            parent,
            minimum_turns=6,
            target_dialogue_tokens=512,
        )
        accepted = []
        covered_slot_ids = []
        segment_ids = discourse.discourse_segment_ids(plan)
        self.assertEqual(len(segment_ids), 4)

        for segment_id in segment_ids:
            contract = discourse.build_api_contract(
                plan,
                segment_id=segment_id,
                accepted_prefix_tail=accepted[-discourse.ACCEPTED_PREFIX_TAIL_TURNS :],
            )
            scope = contract["read_only_brief"]["generation_scope"]
            self.assertEqual(scope["segment_id"], segment_id)
            self.assertTrue(scope["segment_purpose"])
            body = segment_body(contract["read_only_brief"])
            report = discourse.validate_api_response_segment(
                plan,
                body,
                segment_id=segment_id,
                accepted_prefix_turns=accepted,
            )
            self.assertTrue(report["pass"], report["issues"])
            accepted.extend(copy.deepcopy(body["turns"]))
            covered_slot_ids.extend(
                row["turn_slot_id"] for row in body["turns"]
            )

        self.assertEqual(
            covered_slot_ids,
            [slot["turn_slot_id"] for slot in plan["turn_slots"]],
        )

    def test_segment_overflow_normalizer_is_lossless_and_fail_closed(self) -> None:
        graph = golden_graph()
        bundle = golden_bundle(graph)
        parent = golden_parent_context()
        plan = discourse.build_discourse_plan(
            graph,
            bundle,
            parent,
            minimum_turns=6,
            target_dialogue_tokens=512,
        )
        segment_ids = discourse.discourse_segment_ids(plan)
        first_contract = discourse.build_api_contract(
            plan, segment_id=segment_ids[0]
        )
        first_body = segment_body(first_contract["read_only_brief"])
        second_contract = discourse.build_api_contract(
            plan, segment_id=segment_ids[1]
        )
        second_body = segment_body(second_contract["read_only_brief"])
        third_contract = discourse.build_api_contract(
            plan, segment_id=segment_ids[2]
        )
        third_body = segment_body(third_contract["read_only_brief"])

        future_overflow = copy.deepcopy(second_body)
        future_overflow["turns"].extend(
            copy.deepcopy(third_body["turns"][:2])
        )
        before = copy.deepcopy(future_overflow)
        normalized, audit = discourse.normalize_api_response_segment_overflow(
            plan,
            future_overflow,
            segment_id=segment_ids[1],
        )
        self.assertEqual(future_overflow, before)
        self.assertTrue(audit["used"], audit)
        self.assertEqual(audit["overflow_suffix_count"], 2)
        self.assertEqual(
            audit["overflow_suffix_pattern"], "contiguous_future_slot_ids"
        )
        self.assertTrue(audit["retained_prefix_content_preserved"])
        self.assertEqual(
            audit["retained_prefix_sha256_before"],
            audit["retained_prefix_sha256_after"],
        )
        self.assertEqual(normalized, second_body)
        report = discourse.validate_api_response_segment(
            plan,
            normalized,
            segment_id=segment_ids[1],
            accepted_prefix_turns=first_body["turns"],
        )
        self.assertTrue(report["pass"], report["issues"])

        exact_copy_overflow = copy.deepcopy(second_body)
        exact_copy_overflow["turns"].extend(
            copy.deepcopy(second_body["turns"][:2])
        )
        exact_normalized, exact_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                exact_copy_overflow,
                segment_id=segment_ids[1],
            )
        )
        self.assertTrue(exact_audit["used"], exact_audit)
        self.assertEqual(
            exact_audit["overflow_suffix_pattern"],
            "exact_retained_prefix_row_copies",
        )
        self.assertEqual(exact_normalized, second_body)

        wrong_prefix = copy.deepcopy(future_overflow)
        wrong_prefix["turns"][0], wrong_prefix["turns"][1] = (
            wrong_prefix["turns"][1],
            wrong_prefix["turns"][0],
        )
        wrong_normalized, wrong_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                wrong_prefix,
                segment_id=segment_ids[1],
            )
        )
        self.assertFalse(wrong_audit["used"])
        self.assertTrue(wrong_audit["failure"])
        self.assertEqual(
            wrong_audit["reason"], "required_prefix_slot_ids_mismatch"
        )
        self.assertEqual(wrong_normalized, wrong_prefix)

        arbitrary_suffix = copy.deepcopy(second_body)
        arbitrary_row = copy.deepcopy(second_body["turns"][0])
        arbitrary_row["turn_slot_id"] = "NOT-A-FROZEN-SLOT"
        arbitrary_suffix["turns"].append(arbitrary_row)
        _arbitrary_normalized, arbitrary_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                arbitrary_suffix,
                segment_id=segment_ids[1],
            )
        )
        self.assertFalse(arbitrary_audit["used"])
        self.assertTrue(arbitrary_audit["failure"])

        retained_duplicate = copy.deepcopy(future_overflow)
        retained_duplicate["turns"][1]["context_sentences"] = copy.deepcopy(
            retained_duplicate["turns"][0]["context_sentences"]
        )
        duplicate_normalized, duplicate_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                retained_duplicate,
                segment_id=segment_ids[1],
            )
        )
        self.assertTrue(duplicate_audit["used"])
        duplicate_report = discourse.validate_api_response_segment(
            plan,
            duplicate_normalized,
            segment_id=segment_ids[1],
            accepted_prefix_turns=first_body["turns"],
        )
        self.assertFalse(duplicate_report["pass"])
        self.assertIn(
            "exact_context_sentence_duplicate",
            duplicate_report["issue_counts_by_code"],
        )

        retained_short = copy.deepcopy(future_overflow)
        for row in retained_short["turns"][: len(second_body["turns"])]:
            row["context_sentences"] = [
                "The planning conversation remains focused on practical details today.",
                "Everyone considers the planning topic with appropriate care now.",
            ]
        short_normalized, short_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                retained_short,
                segment_id=segment_ids[1],
            )
        )
        self.assertTrue(short_audit["used"])
        short_report = discourse.validate_api_response_segment(
            plan,
            short_normalized,
            segment_id=segment_ids[1],
            accepted_prefix_turns=first_body["turns"],
        )
        self.assertFalse(short_report["pass"])
        self.assertIn(
            "segment_context_word_shortfall",
            short_report["issue_counts_by_code"],
        )

        no_overflow, no_overflow_audit = (
            discourse.normalize_api_response_segment_overflow(
                plan,
                second_body,
                segment_id=segment_ids[1],
            )
        )
        self.assertFalse(no_overflow_audit["used"])
        self.assertFalse(no_overflow_audit["failure"])
        self.assertEqual(no_overflow, second_body)

    def test_flat_creative_body_cannot_repeat_frozen_support(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        raw = flat_body(plan)
        raw["turns"][0]["lead_in_sentences"][0] = (
            plan["turn_slots"][0]["frozen_support"][0]["exact_support_quote"]
        )

        _normalized, audit = discourse.normalize_api_response_envelope(plan, raw)

        self.assertFalse(audit["pass"])
        self.assertIn(
            "api_repeated_frozen_support",
            {issue["code"] for issue in audit["issues"]},
        )

    def test_malformed_flat_schema_issues_are_not_reported_twice(self) -> None:
        _graph, _bundle, _parent, plan, _response = golden_artifacts()
        cases = []

        empty = {"turns": []}
        cases.append((empty, "flat_turn_count_mismatch"))

        extra_turn_key = flat_body(plan)
        extra_turn_key["turns"][0]["support"] = []
        cases.append((extra_turn_key, "unexpected_flat_turn_keys"))

        wrong_order = flat_body(plan)
        wrong_order["turns"][0]["turn_slot_id"] = "TS999"
        cases.append((wrong_order, "turn_slot_order_mismatch"))

        for raw, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                _normalized, audit = discourse.normalize_api_response_envelope(
                    plan, raw
                )
                issue_codes = [issue["code"] for issue in audit["issues"]]
                self.assertFalse(audit["pass"])
                self.assertEqual(issue_codes.count(expected_code), 1, audit["issues"])

    def test_response_normalizer_never_overwrites_or_deletes_explicit_fields(self) -> None:
        graph, bundle, parent, plan, _response = golden_artifacts()
        raw = flat_body(plan)
        raw.update(
            {
                "schema_version": None,
                "plan_sha256": "wrong-plan",
                "frozen_input_hashes": {"graph_sha256": "partial"},
            }
        )
        normalized, audit = discourse.normalize_api_response_envelope(plan, raw)
        self.assertEqual(audit["injected_fields"], [])
        self.assertIsNone(normalized["schema_version"])
        self.assertEqual(normalized["plan_sha256"], "wrong-plan")
        self.assertEqual(normalized["frozen_input_hashes"], {"graph_sha256": "partial"})
        result_codes = codes(validate(graph, bundle, parent, plan, normalized))
        self.assertTrue(
            {
                "response_schema_mismatch",
                "response_plan_hash_mismatch",
                "response_frozen_hashes_mismatch",
            }.issubset(result_codes)
        )

        extra = flat_body(plan)
        extra["mode"] = "do-not-delete"
        normalized_extra, extra_audit = discourse.normalize_api_response_envelope(
            plan, extra
        )
        self.assertFalse(extra_audit["pass"])
        self.assertEqual(extra["mode"], "do-not-delete")
        self.assertEqual(normalized_extra["mode"], "do-not-delete")

    def test_nested_acts_only_body_gets_envelope_but_no_structure_repair(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        acts_only = {"acts": copy.deepcopy(response["acts"])}
        _live_normalized, live_audit = (
            discourse.normalize_api_response_envelope(plan, acts_only)
        )
        self.assertFalse(live_audit["pass"])
        self.assertIn(
            "flat_turn_body_required_for_live",
            {issue["code"] for issue in live_audit["issues"]},
        )
        normalized, audit = discourse.normalize_api_response_envelope(
            plan, acts_only, allow_nested_replay=True
        )
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["body_format"], "nested_acts")
        self.assertEqual(
            discourse.canonical_sha256(acts_only["acts"]),
            discourse.canonical_sha256(normalized["acts"]),
        )
        self.assertTrue(validate(graph, bundle, parent, plan, normalized)["pass"])

        malformed = {"acts": copy.deepcopy(response["acts"][:1])}
        normalized_bad, _audit = discourse.normalize_api_response_envelope(
            plan, malformed, allow_nested_replay=True
        )
        self.assertIn(
            "act_count_mismatch",
            codes(validate(graph, bundle, parent, plan, normalized_bad)),
        )
        self.assertTrue(
            discourse.is_valid_api_response(
                plan,
                response,
                frozen_graph=graph,
                frozen_claim_bundle=bundle,
                parent_context=parent,
            )
        )

    def test_changed_frozen_graph_claim_bundle_or_parent_is_rejected(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        cases = []
        changed_graph = copy.deepcopy(graph)
        changed_graph["events"][0]["value"] = "South Hall"
        cases.append((changed_graph, bundle, parent))
        changed_bundle = copy.deepcopy(bundle)
        changed_bundle["claims"][0]["status"] = "canceled"
        cases.append((graph, changed_bundle, parent))
        changed_parent = copy.deepcopy(parent)
        changed_parent["topic"] = "unrelated topic"
        cases.append((graph, bundle, changed_parent))

        for graph_case, bundle_case, parent_case in cases:
            with self.subTest():
                self.assertIn(
                    "frozen_input_hash_mismatch",
                    codes(validate(graph_case, bundle_case, parent_case, plan, response)),
                )

    def test_response_hashes_and_plan_hash_are_enforced(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        broken_response = copy.deepcopy(response)
        broken_response["plan_sha256"] = "0" * 64
        broken_response["frozen_input_hashes"]["graph_sha256"] = "1" * 64

        report = validate(graph, bundle, parent, plan, broken_response)

        self.assertIn("response_plan_hash_mismatch", codes(report))
        self.assertIn("response_frozen_hashes_mismatch", codes(report))

    def test_speaker_roster_is_closed(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        all_turns(response)[0]["speaker"] = "Narrator"

        self.assertIn(
            "speaker_outside_roster",
            codes(validate(graph, bundle, parent, plan, response)),
        )

    def test_act_and_beat_order_are_fixed(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        response["acts"].reverse()

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("act_order_mismatch", codes(report))

        graph, bundle, parent, plan, response = golden_artifacts()
        response["acts"][0]["beats"].reverse()
        self.assertIn(
            "beat_order_mismatch",
            codes(validate(graph, bundle, parent, plan, response)),
        )

    def test_exact_quote_must_be_a_visible_substring(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        support = all_support_rows(response)[0]
        support["exact_support_quote"] = support["exact_support_quote"].replace(
            "Cedar Hall", "South Hall"
        )

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("exact_quote_not_substring", codes(report))
        self.assertIn("typed_support_object_role_missing", codes(report))

    def test_claim_coverage_is_exactly_once(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        turns = all_turns(response)
        duplicate = copy.deepcopy(turns[0]["support"][0])
        turns[0]["support"].append(duplicate)

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("duplicate_claim_coverage", codes(report))

        graph, bundle, parent, plan, response = golden_artifacts()
        all_turns(response)[0]["support"] = []
        self.assertIn(
            "missing_claim_coverage",
            codes(validate(graph, bundle, parent, plan, response)),
        )

    def test_visible_quote_may_occur_only_once_globally(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        turns = all_turns(response)
        quote = turns[0]["support"][0]["exact_support_quote"]
        turns[1]["text"] += f" As a reminder, {quote}"

        self.assertIn(
            "support_quote_occurrence_mismatch",
            codes(validate(graph, bundle, parent, plan, response)),
        )

    def test_semantic_direction_is_checked_by_independent_claim_validator(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        turn = all_turns(response)[-1]
        old_quote = turn["support"][0]["exact_support_quote"]
        wrong_quote = "At time 6, gallery booking directly caused Spring Workshop."
        turn["text"] = turn["text"].replace(old_quote, wrong_quote)
        turn["support"][0]["exact_support_quote"] = wrong_quote

        report = validate(graph, bundle, parent, plan, response)

        self.assertTrue(
            {
                "typed_support_cause_direction_mismatch",
                "typed_support_typed_role_missing",
                "typed_support_predicate_missing",
            }
            & codes(report)
        )

    def test_fact_sentence_cannot_be_an_isolated_turn(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        turn = all_turns(response)[0]
        turn["text"] = turn["support"][0]["exact_support_quote"]

        self.assertIn(
            "isolated_fact_turn",
            codes(validate(graph, bundle, parent, plan, response)),
        )

    def test_meta_filler_and_topic_drift_are_rejected(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        beat = response["acts"][0]["beats"][0]
        beat["turns"].extend(
            [
                {
                    "turn_id": "D1:90",
                    "speaker": "Maya",
                    "text": "The graph schema says this gold answer should pass.",
                    "support": [],
                },
                {
                    "turn_id": "D1:91",
                    "speaker": "Noah",
                    "text": "Clouds drift above the distant ocean without any local concern.",
                    "support": [],
                },
            ]
        )

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("visible_internal_meta", codes(report))
        self.assertIn("unrelated_filler_turn", codes(report))
        self.assertIn("topic_window_drift", codes(report))

    def test_generic_acknowledgement_is_rejected(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        response["acts"][0]["beats"][0]["turns"].append(
            {
                "turn_id": "D1:92",
                "speaker": "Maya",
                "text": "That makes sense.",
                "support": [],
            }
        )

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("generic_filler_turn", codes(report))
        self.assertIn("unrelated_filler_turn", codes(report))

    def test_truth_mutation_fields_are_rejected_at_any_depth(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        all_turns(response)[0]["proof"] = {"gold": "Cedar Hall"}

        report = validate(graph, bundle, parent, plan, response)

        self.assertIn("forbidden_truth_mutation_surface", codes(report))
        self.assertIn("unexpected_response_keys", codes(report))

    def test_claim_cannot_be_moved_to_another_beat(self) -> None:
        graph, bundle, parent, plan, response = golden_artifacts()
        beats = [beat for act in response["acts"] for beat in act["beats"]]
        source_turn = beats[0]["turns"].pop(0)
        beats[-1]["turns"].append(source_turn)

        self.assertIn(
            "claim_assigned_to_wrong_beat",
            codes(validate(graph, bundle, parent, plan, response)),
        )


if __name__ == "__main__":
    unittest.main()
