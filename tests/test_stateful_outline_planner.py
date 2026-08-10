import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.stateful_graph_expansion import (
    compile_stateful_scene_contracts,
    cross_arc_stage_conflict_gate,
    terminal_evidence_is_future_or_conditional,
)
from scripts.current_method.restart_n5_spell.stateful_outline_planner import (
    STATEFUL_OUTLINE_SCHEMA_VERSION,
    StatefulOutlineArcRetryValidator, StatefulOutlineE1RetryValidator,
    arc_completion_evidence_feedback,
    arc_e1_retry_error_feedback,
    build_stateful_outline_e1_repair_suffix,
    build_stateful_outline_e1_retry_freeze_contract,
    build_stateful_outline_arc_prompt, build_stateful_outline_overview_prompt,
    build_stateful_outline_overview_anchor_repair_suffix,
    build_stateful_outline_overview_interface_repair_suffix,
    build_cross_arc_stage_conflict_freeze_contract,
    validate_cross_arc_stage_conflict_freeze,
    build_stateful_outline_prompt, normalize_stateful_outline,
    overview_anchor_overflow_feedback,
    overview_participant_error_feedback,
    completion_evidence_interface_nonterminal,
    canonical_completion_marker_errors,
    structured_completion_contract_errors,
    validate_stateful_outline_e1_retry_freeze,
    validate_raw_stateful_outline, validate_stateful_outline_arc,
    validate_stateful_outline_overview,
)
from scripts.current_method.restart_n5_spell.run_stateful_expansion_qa_solver_canary import (
    _arc_scene_indices,
)


def _turns():
    return [{"turn_id": f"D1:{index}", "session_id": "D1",
             "speaker": "John" if index % 2 else "Maria",
             "text": f"Established parent fact {index}."}
            for index in range(1, 9)]


def _raw():
    arcs = []
    for arc_index in range(1, 5):
        stages = []
        for stage_index in range(1, 5):
            stages.append({
                "description": f"John makes explicit decision {arc_index}.{stage_index}.",
                "object_id": f"object_{arc_index}_{stage_index}",
                "state_key": f"state_{(arc_index - 1) * 4 + stage_index}",
                "event_identity_key": (
                    f"event_{arc_index}_{stage_index}"),
                "stage_role": "base_fact_completion",
                "predecessor_stage_ref": None,
                "pre_state": "pending", "target_post_state": "approved",
                "allowed_post_states": ["approved", "rejected"],
                "terminal_post_states": ["approved", "rejected"],
                "completion_event_type": "explicit_decision",
                "mandatory_resolution": True,
                "authorized_result_speakers": ["John"],
                "completion_status": "achieved",
                "completion_temporality": "current_or_past",
                "completion_conditionality": "unconditional",
                "completion_evidence_phrases": ["I approve this item"],
                "outcome_handoffs": [],
            })
        arcs.append({"summary": f"Arc {arc_index}",
                     "parent_anchor_aliases": [f"T{arc_index:04d}"],
                     "stages": stages})
    return {"title": "Stateful continuation",
            "expansion_objective": "Produce explicit state transitions.",
            "allowed_new_characters": [],
            "allowed_locations": ["office", "library", "community room"],
            "time_window": "two months", "event_arcs": arcs,
            "scenes": [{"purpose": f"Resolve scene {index}.",
                        "participants": ["John", "Maria"],
                        "parent_anchor_aliases": ["T0001"] if index == 1 else [],
                        "location": "office",
                        "scene_role": "advance_parent_thread" if index <= 5 else
                                      "create_new_state_chain"}
                       for index in range(1, 11)]}


class StatefulOutlinePlannerTests(unittest.TestCase):
    def test_valid_raw_plan_normalizes_and_compiles(self):
        raw = _raw()
        self.assertEqual("stateful_outline_accept",
                         validate_raw_stateful_outline(
                             raw, parent_turns=_turns())["decision"])
        outline = normalize_stateful_outline(raw, parent_turns=_turns())
        self.assertEqual("stateful_graph_outline_v4",
                         outline["schema_version"])
        self.assertEqual("structured_canonical_v1",
                         outline["completion_contract_version"])
        self.assertEqual("stateful_graph_outline_v4",
                         STATEFUL_OUTLINE_SCHEMA_VERSION)
        contracts = compile_stateful_scene_contracts(outline)
        self.assertEqual(10, len(contracts))
        self.assertEqual(16, sum(len(row["assigned_event_stage_ids"])
                                 for row in contracts))
        self.assertTrue(all(row["parent_anchor_evidence"] for row in contracts))

    def test_duplicate_state_instance_is_rejected(self):
        raw = _raw()
        raw["event_arcs"][0]["stages"][1]["object_id"] = "object_1_1"
        raw["event_arcs"][0]["stages"][1]["state_key"] = "state_1"
        result = validate_raw_stateful_outline(raw, parent_turns=_turns())
        self.assertIn("duplicate_state_instance", result["reasons"])

    def test_cross_arc_duplicate_base_completion_is_rejected(self):
        arcs = deepcopy(_raw()["event_arcs"])
        arcs[1]["stages"][0]["event_identity_key"] = (
            arcs[0]["stages"][0]["event_identity_key"])
        gate = cross_arc_stage_conflict_gate(arcs)
        self.assertFalse(gate["valid"])
        self.assertIn("duplicate_completion_obligation", gate["errors"])
        self.assertIn("E2:stage_01", gate["conflicting_stage_refs"])

    def test_cross_arc_derived_state_requires_exact_predecessor(self):
        arcs = deepcopy(_raw()["event_arcs"])
        base = arcs[0]["stages"][0]
        derived = arcs[1]["stages"][0]
        derived["event_identity_key"] = base["event_identity_key"]
        derived["stage_role"] = "derived_state"
        derived["predecessor_stage_ref"] = "E1:stage_01"
        derived["pre_state"] = base["target_post_state"]
        derived["target_post_state"] = "responsibility_reflected"
        derived["allowed_post_states"] = ["responsibility_reflected"]
        derived["terminal_post_states"] = ["responsibility_reflected"]
        derived["completion_evidence_phrases"] = [
            "I feel proud that I handled the situation responsibly"]
        self.assertTrue(cross_arc_stage_conflict_gate(arcs)["valid"])
        derived["predecessor_stage_ref"] = "E1:stage_02"
        gate = cross_arc_stage_conflict_gate(arcs)
        self.assertIn("derived_state_predecessor_invalid", gate["errors"])

    def test_cross_arc_repeated_base_marker_is_rejected(self):
        arcs = deepcopy(_raw()["event_arcs"])
        base = arcs[0]["stages"][0]
        derived = arcs[1]["stages"][0]
        derived.update({
            "event_identity_key": base["event_identity_key"],
            "stage_role": "derived_state",
            "predecessor_stage_ref": "E1:stage_01",
            "pre_state": base["target_post_state"],
            "target_post_state": "reflected",
            "allowed_post_states": ["reflected"],
            "terminal_post_states": ["reflected"],
            "completion_evidence_phrases":
                list(base["completion_evidence_phrases"]),
        })
        gate = cross_arc_stage_conflict_gate(arcs)
        self.assertIn("repeated_base_fact_marker", gate["errors"])

    def test_cross_arc_repair_freezes_nonconflicting_stages(self):
        arcs = deepcopy(_raw()["event_arcs"])
        arcs[1]["stages"][0]["event_identity_key"] = (
            arcs[0]["stages"][0]["event_identity_key"])
        gate = cross_arc_stage_conflict_gate(arcs)
        contract = build_cross_arc_stage_conflict_freeze_contract(
            arcs=arcs, conflict_gate=gate, arc_id="E2")
        candidate = deepcopy(arcs[1])
        candidate["stages"][0]["event_identity_key"] = "new_distinct_event"
        self.assertTrue(validate_cross_arc_stage_conflict_freeze(
            candidate, contract)["valid"])
        candidate["stages"][1]["description"] = "Unauthorized rewrite"
        frozen = validate_cross_arc_stage_conflict_freeze(candidate, contract)
        self.assertFalse(frozen["valid"])
        self.assertIn("retry_modified_frozen_stage", frozen["errors"])

    def test_scene_must_include_authorized_result_speaker(self):
        raw = _raw()
        raw["scenes"][0]["participants"] = ["Maria", "Asha"]
        raw["allowed_new_characters"] = [{"name": "Asha", "role": "reviewer"}]
        result = validate_raw_stateful_outline(raw, parent_turns=_turns())
        self.assertIn("scene_missing_stage_result_speaker", result["reasons"])

    def test_description_wording_does_not_override_typed_transition(self):
        raw = _raw()
        raw["event_arcs"][0]["stages"][0]["description"] = (
            "The item aligns with the general goal.")
        result = validate_raw_stateful_outline(raw, parent_turns=_turns())
        self.assertEqual("stateful_outline_accept", result["decision"])

    def test_conditional_completion_phrase_is_rejected_before_realizer(self):
        raw = _raw()
        raw["event_arcs"][0]["stages"][0][
            "completion_evidence_phrases"] = [
                "We can finish this if we work together."]
        result = validate_raw_stateful_outline(raw, parent_turns=_turns())
        self.assertIn("completion_evidence_phrase_nonterminal",
                      result["reasons"])

    def test_third_person_completion_marker_is_rejected_before_realizer(self):
        raw = _raw()
        raw["event_arcs"][0]["stages"][0][
            "completion_evidence_phrases"] = [
                "John approved this version."]
        result = validate_raw_stateful_outline(raw, parent_turns=_turns())
        self.assertIn("completion_evidence_phrase_not_speakable",
                      result["reasons"])

    def test_prompt_contains_parent_only_and_no_question_task(self):
        prompt = build_stateful_outline_prompt(parent_turns=_turns())
        self.assertIn("FROZEN PARENT DIALOGUE", prompt)
        self.assertIn("Do not write dialogue, questions, options, answers", prompt)
        self.assertNotIn("historical child", prompt.casefold())

    def test_chunked_overview_and_arcs_reassemble_exact_plan(self):
        raw = _raw()
        model_overview = {key: deepcopy(value) for key, value in raw.items()
                          if key != "event_arcs"}
        for scene in model_overview["scenes"]:
            scene.pop("scene_role")
        checked = validate_stateful_outline_overview(
            model_overview, parent_turns=_turns())
        self.assertEqual("stateful_outline_overview_accept",
                         checked["decision"])
        overview = checked["overview"]
        self.assertEqual(
            ["advance_parent_thread"] * 5 + ["create_new_state_chain"] * 5,
            [scene["scene_role"] for scene in overview["scenes"]])
        accepted_arcs = []
        for index, source_arc in enumerate(raw["event_arcs"], 1):
            arc = deepcopy(source_arc)
            for stage in arc["stages"]:
                stage.pop("outcome_handoffs")
            result = validate_stateful_outline_arc(
                arc, parent_turns=_turns(), overview=overview,
                assigned_scene_indices=_arc_scene_indices(index))
            self.assertEqual("stateful_outline_arc_accept", result["decision"])
            self.assertTrue(all(stage["outcome_handoffs"] == []
                                for stage in result["arc"]["stages"]))
            accepted_arcs.append(result["arc"])
        assembled = {**overview, "event_arcs": accepted_arcs}
        self.assertEqual("stateful_outline_accept",
                         validate_raw_stateful_outline(
                             assembled, parent_turns=_turns())["decision"])

    def test_chunked_arc_wrong_result_speaker_is_rejected(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        arc = deepcopy(raw["event_arcs"][0])
        for stage in arc["stages"]:
            stage.pop("outcome_handoffs")
        arc["stages"][0]["authorized_result_speakers"] = ["Maria"]
        overview["scenes"][0]["participants"] = ["John", "Maria"]
        # Maria is a known person and participant, so move her out of the scene.
        overview["scenes"][0]["participants"] = ["John", "John Jr"]
        overview["allowed_new_characters"] = [
            {"name": "John Jr", "role": "assistant"}]
        result = validate_stateful_outline_arc(
            arc, parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertIn("scene_missing_stage_result_speaker", result["reasons"])

    def test_chunked_prompts_do_not_request_dialogue_or_qa(self):
        raw = _raw()
        overview = {key: value for key, value in raw.items()
                    if key != "event_arcs"}
        overview_prompt = build_stateful_outline_overview_prompt(
            parent_turns=_turns())
        arc_prompt = build_stateful_outline_arc_prompt(
            parent_turns=_turns(), overview=overview, arc_index=1,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertIn("exactly ten scenes", overview_prompt)
        self.assertIn("DO NOT output scene_role", overview_prompt)
        self.assertIn("exactly four typed stages", arc_prompt)
        self.assertIn("no dialogue, QA, options, answers", arc_prompt)

    def test_e1_prompt_distinguishes_terminal_from_in_progress(self):
        raw = _raw()
        overview = {key: value for key, value in raw.items()
                    if key != "event_arcs"}
        prompt = build_stateful_outline_arc_prompt(
            parent_turns=_turns(), overview=overview, arc_index=1,
            assigned_scene_indices=_arc_scene_indices(1))
        for word in ("Planning", "agreement", "assignment", "starting",
                     "working on", "preparation"):
            self.assertIn(word, prompt)
        for word in ("submitted", "delivered", "approved", "booked",
                     "received", "finished"):
            self.assertIn(word, prompt)
        self.assertIn("structured completion fields", build_stateful_outline_e1_repair_suffix(
            previous_arc=raw["event_arcs"][0], validation={
                "reasons": ["completion_evidence_phrase_nonterminal"]}))
        self.assertIn("independently understandable", prompt)
        self.assertIn("I confirmed the venue reservation", prompt)
        self.assertIn("It's finally done", prompt)

    def test_ambiguous_openings_fail_frozen_speakable_validator(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        arc = deepcopy(raw["event_arcs"][0])
        for stage in arc["stages"]:
            stage.pop("outcome_handoffs")
        for phrase in ("It's finally done.", "That's been approved.",
                       "This is ready now.", "They did it."):
            candidate = deepcopy(arc)
            candidate["stages"][0]["completion_evidence_phrases"] = [phrase]
            result = validate_stateful_outline_arc(
                candidate, parent_turns=_turns(), overview=overview,
                assigned_scene_indices=_arc_scene_indices(1))
            self.assertIn("completion_evidence_phrase_not_speakable",
                          result["reasons"], phrase)

    def test_explicit_object_phrases_pass_frozen_speakable_validator(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        arc = deepcopy(raw["event_arcs"][0])
        for stage in arc["stages"]:
            stage.pop("outcome_handoffs")
        phrases = (
            "I confirmed the venue reservation.",
            "I submitted the completed report this morning.",
            "I received approval for the revised schedule.",
            "We received the replacement equipment.",
        )
        for stage, phrase in zip(arc["stages"], phrases):
            stage["completion_evidence_phrases"] = [phrase]
        result = validate_stateful_outline_arc(
            arc, parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertEqual("stateful_outline_arc_accept", result["decision"])

    def test_interface_marks_nonterminal_completion_language(self):
        for phrase in (
                "I started the review", "I was assigned the review",
                "I planned the review", "I agreed to review it",
                "I am working on the review", "I am preparing the review"):
            self.assertTrue(completion_evidence_interface_nonterminal(phrase),
                            phrase)
        for phrase in (
                "I submitted the report", "I delivered the package",
                "I approved the request", "I booked the venue",
                "I received the signed copy", "I finished the review"):
            self.assertFalse(completion_evidence_interface_nonterminal(phrase),
                             phrase)

    def test_modal_scope_allows_completed_matrix_with_embedded_modal(self):
        for phrase in (
                "I tried to understand what my dad might be going through.",
                "I am unsure how this will affect our relationship.",
                "I considered what could have caused this."):
            self.assertFalse(
                terminal_evidence_is_future_or_conditional(phrase), phrase)
        for phrase in (
                "I might try to understand.",
                "I will consider it.",
                "I could decide later.",
                "Once I understand, I can respond.",
                "If I decide, I will respond."):
            self.assertTrue(
                terminal_evidence_is_future_or_conditional(phrase), phrase)

    def test_structured_completion_contract_and_marker_cross_checks(self):
        stage = deepcopy(_raw()["event_arcs"][0]["stages"][0])
        self.assertEqual([], structured_completion_contract_errors(stage))
        for field, value, expected_error in (
                ("completion_status", "not_achieved",
                 "mandatory_terminal_completion_not_achieved"),
                ("completion_temporality", "future",
                 "completion_contract_inconsistent"),
                ("completion_conditionality", "conditional",
                 "completion_contract_inconsistent")):
            candidate = deepcopy(stage)
            candidate[field] = value
            self.assertIn(expected_error,
                          structured_completion_contract_errors(candidate))
        candidate = deepcopy(stage)
        candidate["completion_temporality"] = "tomorrow"
        self.assertIn("completion_temporality_invalid",
                      structured_completion_contract_errors(candidate))

    def test_canonical_marker_uses_only_obvious_prefix_rejection(self):
        for marker in (
                "I tried to understand what my dad might be going through.",
                "I still stand by the idea when it improves my life.",
                "I have reflected on what happened with my dad.",
                "I still stand by my spending values."):
            self.assertEqual([], canonical_completion_marker_errors(marker),
                             marker)
        for marker in (
                "I might reflect on it.",
                "I will make a decision.",
                "If I understand it, I will act.",
                "When I finish, I will report back.",
                "Once I decide, I will tell you."):
            self.assertIn("completion_evidence_phrase_nonterminal",
                          canonical_completion_marker_errors(marker), marker)

    def test_old_v2_stage_is_not_silently_accepted_as_v3(self):
        raw = _raw()
        arc = deepcopy(raw["event_arcs"][0])
        for stage in arc["stages"]:
            stage.pop("outcome_handoffs")
            stage.pop("completion_status")
            stage.pop("completion_temporality")
            stage.pop("completion_conditionality")
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        checked = validate_stateful_outline_arc(
            arc, parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertIn("stateful_stage_fields", checked["reasons"])

    def test_program_freeze_allows_only_invalid_stage_field(self):
        original = deepcopy(_raw()["event_arcs"][0])
        for stage in original["stages"]:
            stage.pop("outcome_handoffs")
        original["stages"][3]["completion_evidence_phrases"] = [
            "I might try to understand."]
        validation = {"reasons": [
            "completion_evidence_phrase_nonterminal"]}
        freeze = build_stateful_outline_e1_retry_freeze_contract(
            original, validation)
        self.assertEqual(
            ["E1.stage_01", "E1.stage_02", "E1.stage_03"],
            freeze["valid_stage_ids"])
        self.assertEqual(["E1.stage_04"], freeze["mutable_stage_ids"])
        repaired = deepcopy(original)
        repaired["stages"][3]["completion_evidence_phrases"] = [
            "I considered what could have caused this."]
        self.assertTrue(validate_stateful_outline_e1_retry_freeze(
            repaired, freeze)["valid"])
        repaired["stages"][1]["completion_evidence_phrases"] = [
            "I am unsure how this will affect us."]
        checked = validate_stateful_outline_e1_retry_freeze(repaired, freeze)
        self.assertFalse(checked["valid"])
        self.assertIn("retry_modified_frozen_stage", checked["errors"])
        self.assertEqual("E1.stage_02", checked["diagnostics"][0][
            "stage_id"])

    def test_stateful_e1_retry_validator_enforces_freeze(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        original = deepcopy(raw["event_arcs"][0])
        for stage in original["stages"]:
            stage.pop("outcome_handoffs")
        original["stages"][3]["completion_evidence_phrases"] = [
            "I will consider it."]
        validator = StatefulOutlineE1RetryValidator(
            parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        first = validator(original)
        self.assertIn("completion_evidence_phrase_nonterminal",
                      first["reasons"])
        repaired = deepcopy(original)
        repaired["stages"][3]["completion_evidence_phrases"] = [
            "I considered what could have caused this."]
        repaired["stages"][0]["description"] = "Unauthorized rewrite."
        second = validator(repaired)
        self.assertIn("retry_modified_frozen_stage", second["reasons"])

    def test_stateful_e1_retry_validator_accepts_only_mutable_repair(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        original = deepcopy(raw["event_arcs"][0])
        for stage in original["stages"]:
            stage.pop("outcome_handoffs")
        original["stages"][3]["completion_evidence_phrases"] = [
            "I will consider it."]
        validator = StatefulOutlineE1RetryValidator(
            parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertEqual(
            "stateful_outline_arc_schema_invalid",
            validator(original)["decision"])
        repaired = deepcopy(original)
        repaired["stages"][3]["completion_evidence_phrases"] = [
            "I considered what could have caused this."]
        checked = validator(repaired)
        self.assertEqual("stateful_outline_arc_accept", checked["decision"])
        self.assertTrue(checked["retry_freeze_validation"]["valid"])

    def test_program_freeze_applies_to_non_e1_arcs(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        original = deepcopy(raw["event_arcs"][1])
        for stage in original["stages"]:
            stage.pop("outcome_handoffs")
        original["stages"][0]["completion_temporality"] = "future"
        validator = StatefulOutlineArcRetryValidator(
            parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(2), arc_index=2)
        first = validator(original)
        self.assertIn(
            "mandatory_terminal_completion_not_current_or_past",
            first["reasons"])
        self.assertEqual(["E2.stage_01"], first["retry_freeze_contract"][
            "mutable_stage_ids"])
        repaired = deepcopy(original)
        repaired["stages"][0]["completion_temporality"] = "current_or_past"
        repaired["stages"][1]["description"] = "Unauthorized rewrite."
        second = validator(repaired)
        self.assertIn("retry_modified_frozen_stage", second["reasons"])

    def test_fresh_e2e_failed_outputs_modal_and_freeze_regression(self):
        root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_fresh_e2e_canary_v1")
        parent = json.loads((root / "frozen/parent_contract.json").read_text())[
            "dialogue"]["turns"]
        overview = json.loads((root /
            "planner/stateful_outline_overview.json").read_text())
        wrappers = [json.loads((root /
            f"planner/raw/stateful_outline_arc/stateful-outline-E1-a0{i}.json"
            ).read_text()) for i in (0, 1)]
        normal, retry = [json.loads(row["raw_response"])
                         for row in wrappers]
        # Explicit test-only conversion proves the v1 semantics under v3.
        # Production validation intentionally rejects unversioned v2 stages.
        for arc in (normal, retry):
            for stage_index, stage in enumerate(arc["stages"], 1):
                stage["completion_status"] = "achieved"
                stage["completion_temporality"] = "current_or_past"
                stage["completion_conditionality"] = "unconditional"
                stage["event_identity_key"] = f"legacy_v1_event_{stage_index}"
                stage["stage_role"] = "base_fact_completion"
                stage["predecessor_stage_ref"] = None
            arc["stages"][0]["completion_evidence_phrases"] = [
                "I still stand by my spending values."]
            arc["stages"][2]["completion_evidence_phrases"] = [
                "I now recognize the value of experiences."]
            arc["stages"][3]["completion_evidence_phrases"] = [
                "I recognize the value of small workplace gestures."]
        normal["stages"][3]["completion_evidence_phrases"] = [
            "I have reflected on what happened with my dad."]
        retry["stages"][3]["completion_evidence_phrases"] = [
            "I have reflected on what happened with my dad."]
        normal_checked = validate_stateful_outline_arc(
            normal, parent_turns=parent, overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertEqual("stateful_outline_arc_accept",
                         normal_checked["decision"])
        freeze = build_stateful_outline_e1_retry_freeze_contract(
            normal, wrappers[0]["validation"])
        retry_checked = validate_stateful_outline_e1_retry_freeze(
            retry, freeze)
        self.assertFalse(retry_checked["valid"])
        self.assertIn("retry_modified_frozen_stage", retry_checked["errors"])

    def test_fresh_e2e_v2_explicit_contract_conversion_and_freeze_regression(self):
        root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_fresh_e2e_canary_v2")
        parent = json.loads((root / "frozen/parent_contract.json").read_text())[
            "dialogue"]["turns"]
        overview = json.loads((root /
            "planner/stateful_outline_overview.json").read_text())
        wrappers = [json.loads((root /
            f"planner/raw/stateful_outline_arc/stateful-outline-E1-a0{i}.json"
            ).read_text()) for i in (0, 1)]
        normal_legacy, retry_legacy = [json.loads(row["raw_response"])
                                       for row in wrappers]
        legacy_checked = validate_stateful_outline_arc(
            normal_legacy, parent_turns=parent, overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertIn("stateful_stage_fields", legacy_checked["reasons"])
        normal, retry = deepcopy(normal_legacy), deepcopy(retry_legacy)
        for arc in (normal, retry):
            for stage_index, stage in enumerate(arc["stages"], 1):
                stage["completion_status"] = "achieved"
                stage["completion_temporality"] = "current_or_past"
                stage["completion_conditionality"] = "unconditional"
                stage["event_identity_key"] = f"legacy_v2_event_{stage_index}"
                stage["stage_role"] = "base_fact_completion"
                stage["predecessor_stage_ref"] = None
            arc["stages"][0]["completion_evidence_phrases"] = [
                "I still stand by my spending values."]
            arc["stages"][2]["completion_evidence_phrases"] = [
                "I now recognize the value of experiences."]
            arc["stages"][3]["completion_evidence_phrases"] = [
                "I recognize the value of small workplace gestures."]
        normal_checked = validate_stateful_outline_arc(
            normal, parent_turns=parent, overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertEqual(["event_arc_parent_anchors"],
                         normal_checked["reasons"])
        freeze = build_stateful_outline_e1_retry_freeze_contract(
            normal, normal_checked)
        retry_checked = validate_stateful_outline_e1_retry_freeze(
            retry, freeze)
        self.assertFalse(retry_checked["valid"])
        self.assertIn("retry_modified_frozen_stage", retry_checked["errors"])

    def test_e1_retry_preserves_legal_anchors_and_reports_actual_phrase(self):
        arc = deepcopy(_raw()["event_arcs"][0])
        arc["parent_anchor_aliases"] = ["T0001", "T0002", "T0005"]
        arc["stages"][2]["completion_evidence_phrases"] = [
            "I could do something small to help."]
        validation = {"reasons": [
            "completion_evidence_phrase_nonterminal"]}
        feedback = arc_completion_evidence_feedback(arc)
        self.assertEqual(3, feedback[0]["stage_index"])
        suffix = build_stateful_outline_e1_repair_suffix(
            previous_arc=arc, validation=validation)
        self.assertIn("could do", suffix)
        self.assertIn("at most 4 arc anchors", suffix)
        self.assertIn("Preserve this exact list and order", suffix)
        self.assertIn("all already-valid fields", suffix)
        self.assertIn("PREVIOUS E1 ARC JSON", suffix)
        self.assertIn("Program-enforced retry freeze contract", suffix)
        self.assertIn("frozen_stage_sha256", suffix)
        self.assertEqual(["T0001", "T0002", "T0005"],
                         arc["parent_anchor_aliases"])

    def test_retry_aggregates_anchor_and_speakable_errors(self):
        arc = deepcopy(_raw()["event_arcs"][0])
        arc["parent_anchor_aliases"] = [
            "T0001", "T0002", "T0003", "T0004", "T0005", "T0006"]
        arc["stages"][2]["completion_evidence_phrases"] = [
            "It's nice to know that my actions are noticed."]
        frozen = deepcopy(arc)
        validation = {"reasons": [
            "completion_evidence_phrase_not_speakable",
            "event_arc_parent_anchors",
        ]}
        feedback = arc_e1_retry_error_feedback(arc, validation)
        self.assertEqual(
            ["anchor_count_exceeded",
             "completion_evidence_phrase_not_speakable"],
            [row["code"] for row in feedback])
        suffix = build_stateful_outline_e1_repair_suffix(
            previous_arc=arc, validation=validation)
        self.assertIn('"actual_count": 6', suffix)
        self.assertIn('"maximum_count": 4', suffix)
        self.assertIn("stage_index", suffix)
        self.assertIn("It's nice to know", suffix)
        self.assertIn("every already-valid evidence phrase", suffix)
        self.assertEqual(frozen, arc)

    def test_retry_aggregates_nonterminal_with_other_errors(self):
        arc = deepcopy(_raw()["event_arcs"][0])
        arc["parent_anchor_aliases"] = [
            "T0001", "T0002", "T0003", "T0004", "T0005"]
        arc["stages"][1]["completion_evidence_phrases"] = [
            "I could submit the completed report tomorrow."]
        validation = {"reasons": [
            "completion_evidence_phrase_nonterminal",
            "event_arc_parent_anchors",
        ]}
        feedback = arc_e1_retry_error_feedback(arc, validation)
        self.assertEqual(
            ["anchor_count_exceeded",
             "completion_evidence_phrase_nonterminal"],
            [row["code"] for row in feedback])

    def test_arc_anchor_cap_is_hard_and_not_locally_truncated(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        arc = deepcopy(raw["event_arcs"][0])
        for stage in arc["stages"]:
            stage.pop("outcome_handoffs")
        for count in range(1, 5):
            candidate = deepcopy(arc)
            candidate["parent_anchor_aliases"] = [
                f"T{index:04d}" for index in range(1, count + 1)]
            result = validate_stateful_outline_arc(
                candidate, parent_turns=_turns(), overview=overview,
                assigned_scene_indices=_arc_scene_indices(1))
            self.assertEqual("stateful_outline_arc_accept", result["decision"])
        candidate = deepcopy(arc)
        candidate["parent_anchor_aliases"] = [
            f"T{index:04d}" for index in range(1, 6)]
        result = validate_stateful_outline_arc(
            candidate, parent_turns=_turns(), overview=overview,
            assigned_scene_indices=_arc_scene_indices(1))
        self.assertIn("event_arc_parent_anchors", result["reasons"])
        self.assertEqual(5, len(candidate["parent_anchor_aliases"]))

    def test_prior_e1_outputs_regression(self):
        normal = deepcopy(_raw()["event_arcs"][0])
        normal["parent_anchor_aliases"] = ["T0001", "T0002", "T0005"]
        normal["stages"][2]["completion_evidence_phrases"] = [
            "I could do something small to show my appreciation."]
        retry = deepcopy(normal)
        retry["parent_anchor_aliases"] = [
            f"T{index:04d}" for index in range(1, 13)]
        retry["stages"][0]["completion_evidence_phrases"] = [
            "I've decided to focus more on balancing my spending."]
        self.assertEqual(3, len(normal["parent_anchor_aliases"]))
        self.assertEqual(12, len(retry["parent_anchor_aliases"]))
        self.assertEqual(1, len(arc_completion_evidence_feedback(normal)))
        suffix = build_stateful_outline_e1_repair_suffix(
            previous_arc=normal, validation={"reasons": [
                "completion_evidence_phrase_nonterminal"]})
        self.assertIn('["T0001", "T0002", "T0005"]', suffix)
        self.assertNotIn("T0012\". Preserve this exact list", suffix)

    def test_actual_prior_e1_raw_files_regression(self):
        raw_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_generation_canary_"
            "anchor_repair_v1/planner/raw/stateful_outline_arc")
        normal_wrapper = json.loads(
            (raw_root / "stateful-outline-E1-a00.json").read_text())
        retry_wrapper = json.loads(
            (raw_root / "stateful-outline-E1-a01.json").read_text())
        normal = json.loads(normal_wrapper["raw_response"])
        retry = json.loads(retry_wrapper["raw_response"])
        self.assertEqual(
            ["completion_evidence_phrase_nonterminal"],
            normal_wrapper["validation"]["reasons"])
        self.assertEqual(3, len(normal["parent_anchor_aliases"]))
        self.assertEqual([], arc_completion_evidence_feedback(normal))
        self.assertEqual(
            ["completion_evidence_phrase_nonterminal",
             "event_arc_parent_anchors"],
            retry_wrapper["validation"]["reasons"])
        self.assertEqual(12, len(retry["parent_anchor_aliases"]))
        suffix = build_stateful_outline_e1_repair_suffix(
            previous_arc=normal, validation=normal_wrapper["validation"])
        self.assertIn("could do", suffix)
        self.assertIn("Preserve this exact list and order", suffix)
        self.assertIn('["T0001", "T0002", "T0005"]', suffix)

    def test_actual_latest_e1_raw_files_multi_error_regression(self):
        raw_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_generation_canary_"
            "e1_interface_v1/planner/raw/stateful_outline_arc")
        normal_wrapper = json.loads(
            (raw_root / "stateful-outline-E1-a00.json").read_text())
        retry_wrapper = json.loads(
            (raw_root / "stateful-outline-E1-a01.json").read_text())
        normal = json.loads(normal_wrapper["raw_response"])
        retry = json.loads(retry_wrapper["raw_response"])
        self.assertEqual(6, len(normal["parent_anchor_aliases"]))
        self.assertEqual(4, len(retry["parent_anchor_aliases"]))
        feedback = arc_e1_retry_error_feedback(
            normal, normal_wrapper["validation"])
        self.assertEqual(
            ["anchor_count_exceeded",
             "completion_evidence_phrase_not_speakable"],
            [row["code"] for row in feedback])
        self.assertEqual(3, feedback[1]["stage_index"])
        suffix = build_stateful_outline_e1_repair_suffix(
            previous_arc=normal, validation=normal_wrapper["validation"])
        self.assertIn("It's nice to know", suffix)
        self.assertIn('"actual_count": 6', suffix)

    def test_overview_prompt_and_repair_enforce_three_anchor_cap(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        for scene in overview["scenes"]:
            scene.pop("scene_role")
        overview["scenes"][0]["parent_anchor_aliases"] = [
            "T0001", "T0002", "T0003", "T0004"]
        checked = validate_stateful_outline_overview(
            overview, parent_turns=_turns())
        self.assertIn("scene_schema", checked["reasons"])
        feedback = overview_anchor_overflow_feedback(overview)
        self.assertEqual([{"scene_index": 1, "anchor_count": 4}], feedback)
        suffix = build_stateful_outline_overview_anchor_repair_suffix(
            previous_overview=overview)
        self.assertIsNotNone(suffix)
        self.assertIn("scene 1 currently has 4 anchors", suffix)
        self.assertIn("0-3 aliases", suffix)
        self.assertIn("PREVIOUS OVERVIEW JSON", suffix)
        self.assertEqual(4, len(overview["scenes"][0][
            "parent_anchor_aliases"]))
        prompt = build_stateful_outline_overview_prompt(parent_turns=_turns())
        self.assertIn("0-3 aliases", prompt)

    def test_overview_prompt_freezes_exact_allowed_participants(self):
        prompt = build_stateful_outline_overview_prompt(parent_turns=_turns())
        self.assertIn("ALLOWED_PARTICIPANTS", prompt)
        self.assertIn('["John", "Maria"]', prompt)
        self.assertIn("character-for-character", prompt)
        self.assertIn("Do not change singular/plural", prompt)
        self.assertIn("2-4 distinct exact names", prompt)

    def test_overview_participant_feedback_uses_exact_identity(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        for scene in overview["scenes"]:
            scene.pop("scene_role")
        overview["scenes"][0]["participants"] = ["john"]
        frozen = deepcopy(overview)
        feedback = overview_participant_error_feedback(
            overview, parent_turns=_turns())
        self.assertEqual(
            ["participant_count_invalid", "participant_name_not_allowed"],
            [row["code"] for row in feedback])
        self.assertEqual(["John", "Maria"], feedback[1]["allowed_names"])
        self.assertEqual(frozen, overview)

    def test_overview_retry_reports_all_participant_errors_only(self):
        raw = _raw()
        overview = {key: deepcopy(value) for key, value in raw.items()
                    if key != "event_arcs"}
        for scene in overview["scenes"]:
            scene.pop("scene_role")
        overview["scenes"][0]["participants"] = ["john"]
        frozen = deepcopy(overview)
        suffix = build_stateful_outline_overview_interface_repair_suffix(
            previous_overview=overview,
            validation={"reasons": ["scene_schema"]},
            parent_turns=_turns())
        self.assertIsNotNone(suffix)
        self.assertIn("participant_count_invalid", suffix)
        self.assertIn("participant_name_not_allowed", suffix)
        self.assertIn('"actual_count": 1', suffix)
        self.assertIn('allowed_participants exact existing names: ["John", "Maria"]', suffix)
        self.assertIn('Fields permitted to change in this retry: ["participants"]', suffix)
        self.assertIn("every already-legal parent_anchor_aliases", suffix)
        self.assertEqual(frozen, overview)

    def test_actual_overview_raw_files_participant_regression(self):
        raw_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_generation_canary_"
            "e1_speakable_v1/planner/raw/stateful_outline_overview")
        normal_wrapper = json.loads(
            (raw_root / "stateful-outline-overview-a00.json").read_text())
        retry_wrapper = json.loads(
            (raw_root / "stateful-outline-overview-a01.json").read_text())
        normal = json.loads(normal_wrapper["raw_response"])
        retry = json.loads(retry_wrapper["raw_response"])
        normal_feedback = overview_participant_error_feedback(
            normal, parent_turns=[
                {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"}])
        retry_feedback = overview_participant_error_feedback(
            retry, parent_turns=[
                {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"}])
        self.assertEqual(10, sum(
            row["code"] == "participant_name_not_allowed"
            for row in normal_feedback))
        self.assertTrue(any(
            row["code"] == "participant_count_invalid" and
            row["scene_index"] == 1 and row["actual_count"] == 1
            for row in retry_feedback))
        suffix = build_stateful_outline_overview_interface_repair_suffix(
            previous_overview=retry,
            validation=retry_wrapper["validation"],
            parent_turns=[
                {"speaker": "Co-workers A"}, {"speaker": "Co-workers B"}])
        self.assertIn("Co-worker A", suffix)
        self.assertIn("Co-workers A", suffix)
        self.assertIn('"actual_count": 1', suffix)
        self.assertIn('Fields permitted to change in this retry: ["participants"]', suffix)


if __name__ == "__main__":
    unittest.main()
