import tempfile
import unittest
import json
import shutil
import copy
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import (
    classify_reward_group, collect_valid_rollouts, context_preflight, grounding_decision,
    local_reward, option_audit, parse_mcq_response, parse_boxed_mcq_answer,
    classify_frontier_pre_audit, full_context_gate,
)
from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.run_canary import (
    DEFAULTS, create_source_snapshot, initialize_run, render_solver_prompt, select_solver_qa,
)
from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json
from scripts.current_method.restart_n5_spell.ledger import AmbiguousRequestError, RequestLedger
from scripts.current_method.restart_n5_spell.live_backends import (
    ChallengerAPIConfig, LiveRealizerBackend, OpenAICompatibleChallengerClient,
)
from scripts.current_method.restart_n5_spell.backends import RealizerRequest
from scripts.current_method.restart_n5_spell.orchestrator import CanaryOrchestrator
from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell import input_adapter as input_adapter_module
from scripts.current_method.restart_n5_spell import validation as validation_module
from scripts.current_method.restart_n5_spell import run_canary as run_canary_module
from scripts.current_method.restart_n5_spell.prompts import (
    REALIZER_OUTPUT_SCHEMA, build_questioner_prompt, build_realizer_prompt, prompt_template_hashes,
)
from scripts.current_method.restart_n5_spell.validation import (
    can_stop, evidence_audit, parse_questioner_batches, validate_config,
    validate_dialogue, validate_lineage_splits, validate_memory, validate_qa,
    validate_realizer_identity_contract, validate_realizer_output_schema,
    derive_local_character_ledger, audit_continuity_hooks,
    materialize_questioner_item, validate_new_hook_descriptions, validate_questioner_smoke_batch,
    validate_durable_candidate_indices,
)
from scripts.current_method.restart_n5_spell.questioner_contract import (
    SPAN_CATALOG_VERSION, REASONING_CONTRACT_VERSION, QUESTIONER_SCHEMA_VERSION,
    build_span_catalog, default_reasoning_contracts, derive_local_difficulty,
    reasoning_contract_sha256, span_catalog_sha256,
)


OPTIONS = {"A": "red", "B": "blue", "C": "green", "D": "gold"}


class RewardTests(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(classify_reward_group([1]*5)["decision"], "reject_easy")
        self.assertEqual(classify_reward_group([0]*5)["decision"], "reject_all_wrong")
        for rewards in ([1,0,1,0,0], [1,1,1,1,0]):
            result = classify_reward_group(list(rewards))
            self.assertEqual(result["decision"], "retain_frontier")
            self.assertGreater(result["std"], 0)

    def test_parser_and_valid_only_retry(self):
        self.assertEqual(local_reward(parse_boxed_mcq_answer("<box>B</box>", OPTIONS), "blue"), 1)
        self.assertEqual(local_reward(parse_boxed_mcq_answer("<box>A</box>", OPTIONS), "blue"), 0)
        self.assertIsNone(local_reward(parse_boxed_mcq_answer("A or B", OPTIONS), "blue"))
        seq = iter(["nonsense", "<box>B</box>"])
        result = collect_valid_rollouts(lambda *_: next(seq), OPTIONS, "blue", 1)
        self.assertEqual(result["verified_rewards"], [1])
        self.assertEqual(result["parse_invalid_generations"], 1)
        self.assertEqual(result["retry_generations"], 1)

    def test_exhausted_is_not_zero(self):
        result = collect_valid_rollouts(lambda *_: "?", OPTIONS, "blue", 1)
        self.assertEqual(result["verified_rewards"], [])
        self.assertEqual(result["exhausted_target_slots"], 1)
        self.assertEqual(classify_reward_group([1,1,1,1])["decision"], "reject_responder_format")

    def test_boxed_parser_fullmatch(self):
        valid = {"<box>A</box>":"A", " <box>C</box> ":"C", "<box> B </box>":"B"}
        for raw, letter in valid.items():
            self.assertEqual(parse_boxed_mcq_answer(raw, OPTIONS)["parsed_letter"], letter)
        for raw in ("<box>E</box>", "<box>A</box><box>B</box>", "Answer: <box>A</box>",
                    "<box>A</box> explanation", "A", "Answer: B",
                    "C. Maria was inspired...", "a charity event", "<BOX>A</BOX>",
                    "<box>a</box>", "<box>A"):
            self.assertEqual(parse_boxed_mcq_answer(raw, OPTIONS)["parse_status"], "parse_invalid")


class GroundingTests(unittest.TestCase):
    def test_policy(self):
        for values in ([0,0,0], [1,0,0]):
            self.assertEqual(grounding_decision(values)["decision"], "pass_grounding")
        self.assertEqual(grounding_decision([1,1,1])["decision"], "bad_ground")
        self.assertTrue(grounding_decision([1,1,0])["needs_extension"])
        self.assertEqual(grounding_decision([1,1,0], [1,0])["decision"], "bad_ground")
        self.assertEqual(grounding_decision([1,1,0], [0,0])["decision"], "pass_grounding")
        self.assertEqual(grounding_decision([1, 0])["decision"], "bad_no_context_eval")

    def test_adaptive_full_context_gate_and_preliminary_decisions(self):
        for initial in ([0,0,0], [1,0,0]):
            ground = grounding_decision(list(initial))
            self.assertTrue(full_context_gate(ground)["run_full_context"])
        bad = grounding_decision([1,1,1])
        self.assertFalse(full_context_gate(bad)["run_full_context"])
        self.assertEqual(full_context_gate({"decision":"pass_grounding"}, [1]*4)["decision"],
                         "reject_responder_format")
        self.assertEqual(classify_frontier_pre_audit([1]*5)["decision"], "reject_easy")
        self.assertEqual(classify_frontier_pre_audit([0]*5)["decision"], "reject_all_wrong")
        self.assertEqual(classify_frontier_pre_audit([1,0,1,0,1])["decision"],
                         "retain_frontier_pre_audit")


class OptionTests(unittest.TestCase):
    def order(self, rewards, letter="A", context="same"):
        return {"valid_rewards": rewards, "gold_letter": letter, "context_hash": context}

    def test_thresholds(self):
        self.assertEqual(option_audit([self.order([1]*4), self.order([1]*5)])["decision"], "option_audit_inconclusive")
        self.assertEqual(option_audit([self.order([1,0,1,0,1]), self.order([1,0,1,1,0])])["decision"], "pass")
        sensitive = option_audit([self.order([1]*4+[0], "A"), self.order([1,0,0,0,0], "B")])
        self.assertEqual(sensitive["decision"], "bad_option_bias")
        report_only = option_audit([self.order([1,1,1,0,0], "A"), self.order([1,0,0,0,0], "B")])
        self.assertEqual(report_only["decision"], "option_order_sensitive")

    def test_semantic_identity(self):
        shuffled = {"A": "gold", "B": "red", "C": "blue", "D": "green"}
        self.assertEqual(parse_boxed_mcq_answer("<box>C</box>", shuffled)["parsed_option_text"], "blue")


class SchemaTests(unittest.TestCase):
    def valid(self):
        return {"question":"q", "options":dict(OPTIONS), "reference_answer_letter":"B",
                "reference_answer_text":"blue", "correct_answer_explanation":"because",
                "distractor_explanations":{"A":"x","B":"correct","C":"y","D":"z"},
                "distractor_values_from_evidence":["red","green","gold"],
                "distractor_supporting_quotes":[
                    [{"turn_id":"T1","exact_quote":"red"}],
                    [{"turn_id":"T1","exact_quote":"green"}],
                    [{"turn_id":"T2","exact_quote":"gold"}]],
                "evidence_turn_ids":["T1","T2"], "reasoning_steps":["a","b"],
                "reasoning_type":"combine", "difficulty_level":"L1",
                "answer_support_type":"direct_span",
                "answer_evidence_values":[
                    {"turn_id":"T1","value":"blue","exact_quote":"blue","role":"answer_value"},
                    {"turn_id":"T2","value":"gold","exact_quote":"gold","role":"supporting_fact"}],
                "final_reasoning_conclusion":"blue"}

    def test_qa_gates(self):
        self.assertEqual(validate_qa(self.valid(), {"T1","T2"}), [])
        q = self.valid(); q["options"]["D"] = "red"
        self.assertIn("bad_option_schema", validate_qa(q, {"T1","T2"}))
        q = self.valid(); q["evidence_turn_ids"] = ["missing"]
        self.assertIn("bad_evidence", validate_qa(q, {"T1"}))
        q = self.valid(); q["reference_answer_text"] = "green"
        self.assertIn("bad_reference_mapping", validate_qa(q, {"T1","T2"}))
        q = self.valid(); del q["distractor_explanations"]["A"]
        self.assertIn("bad_format", validate_qa(q, {"T1","T2"}))

    def test_qa_evidence_consistency(self):
        self.assertEqual(validate_qa(self.valid(), {"T1","T2"},
            {"T1":"The folder was blue, not red or green.","T2":"gold"}), [])
        conflict = self.valid()
        conflict.update({"reference_answer_text":"friends", "final_reasoning_conclusion":"friends",
                         "answer_evidence_values":[
                             {"turn_id":"T1","value":"aunt","exact_quote":"aunt",
                              "role":"answer_value"}]})
        conflict["options"]["B"] = "friends"
        self.assertIn("bad_answer_evidence_binding",
                      validate_qa(conflict, {"T1"}, {"T1":"Her aunt inspired her."}))
        missing = self.valid()
        self.assertIn("bad_supporting_quote", validate_qa(missing, {"T1","T2"}, {"T1":"red","T2":"gold"}))
        multi = self.valid()
        multi.update({"answer_support_type":"multi_hop", "answer_evidence_values":[
                      {"turn_id":"T1","value":"blue","exact_quote":"blue clue","role":"answer_value"},
                      {"turn_id":"T2","value":"gold","exact_quote":"gold","role":"supporting_fact"}],
                      "final_reasoning_conclusion":"blue"})
        self.assertNotIn("bad_reference_evidence_consistency",
                         validate_qa(multi, {"T1","T2"},
                                     {"T1":"blue clue appears with red and green","T2":"gold"}))

    def test_multihop_answer_must_include_evidence_value(self):
        qa = self.valid()
        qa.update({"answer_support_type":"multi_hop", "answer_evidence_values":[
                   {"turn_id":"T1","value":"education","exact_quote":"education",
                    "role":"answer_value"}],
                   "options":{"A":"healthcare","B":"roads","C":"parks","D":"schools"},
                   "reference_answer_letter":"A", "reference_answer_text":"healthcare",
                   "final_reasoning_conclusion":"healthcare"})
        self.assertIn("bad_answer_evidence_binding",
                      validate_qa(qa, {"T1"}, {"T1":"education"}))

    def test_evidence_session_and_difficulty_contracts(self):
        qa = self.valid()
        qa["evidence_session_ids"] = ["S2"]
        errors = validate_qa(qa, {"T1","T2"}, {"T1":"blue","T2":"x"}, {"T1":"S1","T2":"S1"})
        self.assertIn("bad_evidence_session_mapping", errors)
        qa["evidence_session_ids"] = ["S1"]
        qa["difficulty_level"] = "L2"
        self.assertIn("bad_difficulty_evidence_contract",
                      validate_qa(qa, {"T1","T2"}, {"T1":"blue","T2":"x"}, {"T1":"S1","T2":"S1"}))

    def test_materialized_questioner_output_is_solver_source(self):
        generated = self.smoke_item("q1", "B")
        catalog, contracts = self.catalog(), self.contracts()
        materialized, self_errors = materialize_questioner_item(generated, "B",
            span_catalog=catalog, reasoning_contract=contracts[1])
        self.assertEqual(self_errors, [])
        raw, qa, admission = select_solver_qa([generated], {"items":[{
            "item_index":1, "qa_id":"q1", "schema_decision":"accept", "target_letter":"B",
            "materialized_qa":materialized}]}, "q1",
            frozen_turns=self.turn_text(), turn_session_ids=self.turn_sessions(),
            span_catalog=catalog, reasoning_contracts=contracts)
        self.assertIs(raw, generated)
        self.assertEqual(qa["options"]["B"], generated["correct_answer_text"])
        self.assertEqual(qa["canonical_answer_text"], generated["correct_answer_text"])
        self.assertEqual(admission["schema_decision"], "accept")

    def test_questioner_batching(self):
        ok = parse_questioner_batches([[1,2,3] for _ in range(4)])
        self.assertFalse(ok["questioner_candidate_shortfall"])
        for bad in (2, 4):
            batches = [[1,2,3] for _ in range(4)]; batches[1] = list(range(bad))
            result = parse_questioner_batches(batches)
            self.assertTrue(result["questioner_candidate_shortfall"])
            self.assertIn(1, result["batch_count_invalid"])

    @staticmethod
    def dialogue():
        return [
            {"turn_id":"T1","session_id":"S1","speaker":"A","text":"education red priorities"},
            {"turn_id":"T2","session_id":"S1","speaker":"B","text":"infrastructure green priorities"},
            {"turn_id":"T3","session_id":"S1","speaker":"A","text":"parks gold priorities"},
            {"turn_id":"T4","session_id":"S2","speaker":"B","text":"health roads priorities"},
            {"turn_id":"T5","session_id":"S2","speaker":"A","text":"housing music priorities"},
            {"turn_id":"T6","session_id":"S2","speaker":"B","text":"art libraries priorities"},
        ]

    @classmethod
    def catalog(cls): return build_span_catalog(cls.dialogue())

    @classmethod
    def turn_text(cls): return {x["turn_id"]:x["text"] for x in cls.dialogue()}

    @classmethod
    def turn_sessions(cls): return {x["turn_id"]:x["session_id"] for x in cls.dialogue()}

    @staticmethod
    def contracts(): return default_reasoning_contracts(("A", "B", "C"))

    def smoke_item(self, qa_id, letter="A", question=None):
        choices = {
            "A":(["T1#S0001","T2#S0001"],["education","infrastructure"]),
            "B":(["T1#S0001","T4#S0001"],["education","health"]),
            "C":(["T2#S0001","T5#S0001"],["infrastructure","housing"]),
            "D":(["T3#S0001","T6#S0001"],["parks","art"]),
        }
        answer_spans, values = choices[letter]
        correct = f"{values[0]} and {values[1]} for {qa_id}"
        return {"qa_id":qa_id, "question":question or f"What conclusion follows in {qa_id}?",
            "correct_answer_text":correct,
            "distractors":[f"red for {qa_id}",f"green for {qa_id}",f"gold for {qa_id}"],
            "answer_support_span_ids":answer_spans,
            "answer_evidence_values":[
                {"span_id":answer_spans[0],"value":values[0]},
                {"span_id":answer_spans[1],"value":values[1]}],
            "distractor_support":[
                {"distractor_index":0,"span_ids":["T1#S0001"],"evidence_values":["red"]},
                {"distractor_index":1,"span_ids":["T2#S0001"],"evidence_values":["green"]},
                {"distractor_index":2,"span_ids":["T3#S0001"],"evidence_values":["gold"]}],
            "correct_answer_explanation":"The selected spans jointly support it.",
            "distractor_explanations":["path one", "path two", "path three"],
            "reasoning_steps":["Combine fact one.","Apply fact two."]}

    def test_questioner_smoke_strict_batch_and_independent_items(self):
        batch = [self.smoke_item("q1","A"), self.smoke_item("q2","B"), self.smoke_item("q3","C")]
        result = validate_questioner_smoke_batch(batch, ["A","B","C"], set(self.turn_text()),
            {"S1","S2"}, self.turn_text(), self.turn_sessions(),
            span_catalog=self.catalog(), reasoning_contracts=self.contracts())
        self.assertEqual(result["schema_accepted_count"], 3)
        for size in (2, 4):
            bad = validate_questioner_smoke_batch(batch[:size] if size == 2 else batch + [self.smoke_item("q4")],
                ["A","B","C"], set(self.turn_text()), {"S1","S2"}, self.turn_text(),
                self.turn_sessions(), span_catalog=self.catalog(), reasoning_contracts=self.contracts())
            self.assertEqual(bad["batch_decision"], "questioner_batch_count_invalid")
        broken = [dict(x) for x in batch]
        broken[1] = {**broken[1], "reference_answer_letter":"A"}
        independent = validate_questioner_smoke_batch(broken, ["A","B","C"], set(self.turn_text()),
            {"S1","S2"}, self.turn_text(), self.turn_sessions(), span_catalog=self.catalog(),
            reasoning_contracts=self.contracts())
        self.assertEqual([x["schema_decision"] for x in independent["items"]], ["accept","reject","accept"])
        self.assertIn("model_supplied_local_binding_fields", independent["items"][1]["reject_reasons"])

    def test_local_option_binding(self):
        for target in "ABCD":
            generated = self.smoke_item("q1", target)
            contract = {**self.contracts()[0], "target_letter":target}
            qa, errors = materialize_questioner_item(generated, target,
                span_catalog=self.catalog(), reasoning_contract=contract)
            self.assertEqual(errors, [])
            self.assertEqual(qa["options"][target], generated["correct_answer_text"])
            self.assertEqual(qa["reference_answer_letter"], target)
            self.assertEqual(qa["reference_answer_text"], qa["options"][target])

    def test_questioner_smoke_evidence_and_duplicate_gates(self):
        batch = [self.smoke_item("q1","A"), self.smoke_item("q2","B"), self.smoke_item("q3","C")]
        batch[1]["answer_support_span_ids"][0] = "missing#S0001"
        batch[1]["answer_evidence_values"][0]["span_id"] = "missing#S0001"
        batch[2]["question"] = batch[0]["question"]
        result = validate_questioner_smoke_batch(batch, ["A","B","C"], set(self.turn_text()),
            {"S1","S2"}, self.turn_text(), self.turn_sessions(), span_catalog=self.catalog(),
            reasoning_contracts=self.contracts())
        self.assertIn("bad_span_id", result["items"][1]["reject_reasons"])
        self.assertIn("duplicate_or_option_permutation", result["items"][2]["reject_reasons"])

    def test_distractor_candidates_require_exact_dialogue_support(self):
        generated = self.smoke_item("q1")
        contract = {**self.contracts()[0], "target_letter":"B"}
        qa, errors = materialize_questioner_item(generated, "B",
            span_catalog=self.catalog(), reasoning_contract=contract)
        self.assertEqual(errors, [])
        self.assertNotIn("bad_distractor_support", validate_qa(qa, set(self.turn_text()),
            self.turn_text(), self.turn_sessions()))
        generated["distractor_support"][1]["evidence_values"] = ["purple"]
        qa, errors = materialize_questioner_item(generated, "B",
            span_catalog=self.catalog(), reasoning_contract=contract)
        self.assertIsNone(qa)
        self.assertIn("bad_distractor_support", errors)

    def test_generated_schema_rejects_scalar_lists_and_non_string_ids(self):
        generated = self.smoke_item("q1")
        generated["reasoning_steps"] = "not a list"
        generated["qa_id"] = 7
        qa, errors = materialize_questioner_item(generated, "A",
            span_catalog=self.catalog(), reasoning_contract=self.contracts()[0])
        self.assertIsNone(qa)
        self.assertIn("questioner_generated_schema_invalid", errors)


class DialogueMemoryTests(unittest.TestCase):
    def test_accepted_session_integrity_index_blocks_resume_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_path = root / "accepted/session_01.json"
            atomic_json(session_path, {"session_id":"session_01", "dialogue_turns":[]})
            orchestrator = object.__new__(CanaryOrchestrator)
            orchestrator.run_root = root
            orchestrator.manifest = {"run_id":"test"}; orchestrator.manifest_sha = "manifest"
            atomic_json(root / "accepted/index.json", {"schema_version":"accepted_session_integrity_v2",
                "run_id":"test", "run_manifest_sha256":"manifest",
                "session_sha256":{"session_01.json":sha256_file(session_path)}})
            self.assertEqual(len(orchestrator._accepted_sessions()), 1)
            atomic_json(session_path, {"session_id":"session_01", "dialogue_turns":[{"text":"tampered"}]})
            with self.assertRaises(RuntimeError):
                orchestrator._accepted_sessions()

    def test_dynamic_continuity_hook_audit(self):
        self.assertTrue(audit_continuity_hooks({"H0001","H0002"}, {"H0001","H0002"}, set())["continuity_complete"])
        unknown = audit_continuity_hooks({"H0001","H0002"}, {"H0003"}, set())
        self.assertFalse(unknown["no_unknown_hook_reference"])
        self.assertFalse(unknown["continuity_complete"])
        overlap = audit_continuity_hooks({"H0001"}, {"H0001"}, {"H0001"})
        self.assertFalse(overlap["continued_resolved_disjoint"])
        self.assertFalse(audit_continuity_hooks({"H0042"}, set(), set())["continuity_complete"])
        self.assertTrue(audit_continuity_hooks({"H0042","H9000"}, {"H0042"}, set())["continuity_complete"])

    def test_active_hook_state_carries_unresolved_and_removes_resolved(self):
        sessions = [{"new_unresolved_hooks":[{"hook_id":"H0001","description":"one"}],
                     "resolved_hook_ids":[]},
                    {"continued_unresolved_hook_ids":["H0001"], "resolved_hook_ids":[],
                     "new_unresolved_hooks":[{"hook_id":"H0002","description":"two"}]},
                    {"resolved_hook_ids":["H0001"], "new_unresolved_hooks":[]}]
        self.assertEqual(CanaryOrchestrator._active_hooks(sessions),
                         [{"hook_id":"H0002","description":"two"}])
        self.assertEqual(validate_new_hook_descriptions(
            [{"description":"one"}], [{"hook_id":"H0001","description":"one"}]),
            ["duplicate_unresolved_hook_description"])
        self.assertEqual(validate_new_hook_descriptions(
            [{"description":"new"}, {"description":"new"}], []),
            ["duplicate_unresolved_hook_description"])

    def test_durable_memory_uses_realizer_candidates_without_sentinels(self):
        sessions = [{"session_id":"session_01", "dialogue_turns":[
            {"turn_id":"C0001","speaker":"John","text":"The brief is drafting."}],
            "durable_assertion_candidates":[{"subject":"brief","predicate":"status",
                "object_or_state":"drafting","supporting_turn_indices":[0]}]},
            {"session_id":"session_02", "dialogue_turns":[
                {"turn_id":"C0002","speaker":"John","text":"The brief is submitted."}],
             "durable_assertion_candidates":[{"subject":"brief","predicate":"status",
                "object_or_state":"submitted","supporting_turn_indices":[0]}]}]
        memory = CanaryOrchestrator._durable_memory(None, sessions)
        self.assertEqual(memory["accepted_count"], 2)
        self.assertEqual(memory["assertions"][1]["supersedes_assertion_ids"], ["mem0001"])
        self.assertEqual(memory["assertions"][0]["status"], "superseded")

    @staticmethod
    def _realizer_payload(speaker="John", text="We met in the public library meeting room."):
        return {"dialogue_turns":[{"speaker":speaker,"text":text}],
            "used_parent_anchor_ids":["P1"], "advanced_event_arc_ids":["E1"],
            "continued_unresolved_hook_ids":[], "resolved_hook_ids":[],
            "new_unresolved_hooks":[{"description":"Review the remaining notes."}],
            "durable_assertion_candidates":[], "local_scene_facts":[]}

    def test_realizer_v4_schema_removes_model_entity_metadata(self):
        payload = self._realizer_payload()
        self.assertNotIn("new_named_entities", REALIZER_OUTPUT_SCHEMA)
        self.assertEqual(validate_realizer_output_schema(payload), [])
        deprecated = {**payload, "new_named_entities":["public library meeting room"]}
        errors = validate_realizer_output_schema(deprecated)
        self.assertIn("deprecated_new_named_entities", errors)
        self.assertIn("realizer_extra_fields", errors)

    def test_local_character_ledger_uses_speakers_not_free_text(self):
        outline = {"source_parent":{"core_speakers":["John","Maria"]},
            "allowed_new_characters":[{"name":"Priya Chen","introduction_scene":"S2"}]}
        text = "The city advisory board for education met in the public library meeting room."
        payload = self._realizer_payload("John", text)
        ledger = derive_local_character_ledger(payload, outline, [])
        self.assertEqual(ledger["speaker_names"], ["John"])
        self.assertEqual(ledger["newly_introduced_character_names"], [])
        self.assertNotIn("public library meeting room", ledger["all_seen_character_names"])
        self.assertNotIn("city advisory board for education", ledger["all_seen_character_names"])
        session1 = {"dialogue_turns":[{"speaker":"John","text":"Earlier."}]}
        priya = self._realizer_payload("Priya Chen", "I can join now.")
        self.assertEqual(derive_local_character_ledger(priya, outline, [session1])[
            "newly_introduced_character_names"], ["Priya Chen"])

    def test_dialogue(self):
        parent = [{"turn_id":"P1", "text":"p"}]
        child = [{"turn_id":"P1", "text":"changed"}]
        self.assertIn("duplicate_turn_id", validate_dialogue(parent, child, 1999, []))
        self.assertIn("child_token_delta_below_2000", validate_dialogue(parent, [], 1999, []))
        sessions = [{"used_parent_anchor_ids":["A"]}, {"continued_unresolved_hook_ids":[]}]
        self.assertIn("session_missing_unresolved_hook", validate_dialogue(parent, [], 2000, sessions))
        self.assertTrue(can_stop(2000, True))

    def test_memory(self):
        base = {"assertion_id":"n", "subject":"John", "predicate":"status", "object_or_state":"applied",
                "supporting_turn_ids":["T1"], "supporting_text_spans":[{"turn_id":"T1","exact_span":"applied"}],
                "status":"active", "supersedes_assertion_ids":[]}
        self.assertEqual(validate_memory(base, {"T1":"John applied."}), [])
        bad = dict(base); bad["supporting_text_spans"]=[{"turn_id":"T1","exact_span":"won"}]
        self.assertIn("exact_span_not_found", validate_memory(bad, {"T1":"John applied."}))
        self.assertIn("supporting_turn_not_frozen", validate_memory(base, {}))
        prior = [{"assertion_id":"old", "subject":"John", "predicate":"status", "object_or_state":"drafting", "status":"active"}]
        self.assertIn("missing_supersession", validate_memory(base, {"T1":"John applied."}, prior))

    def test_evidence(self):
        q = {"evidence_turn_ids":["T1","T2"], "reasoning_steps":["a","b"]}
        self.assertEqual(evidence_audit(q, [1,1,0], [0,0,1])["decision"], "certified")
        self.assertEqual(evidence_audit(q, [0,0,0], [1,1,1])["decision"], "downgrade_l1")
        self.assertEqual(evidence_audit({}, [0,0,0], [1,1,1])["decision"], "bad_evidence_relevance")

    def test_realizer_introduction_timing_and_id_namespaces(self):
        outline = {"source_parent":{"core_speakers":["John","Maria"]},
            "allowed_new_characters":[{"name":"Priya Chen","introduction_scene":"S2"}],
            "allowed_locations":["library"], "planned_event_arcs":[{"event_id":"E1"}],
            "parent_anchors":[{"anchor_id":"P1"}]}
        base = {"dialogue_turns":[{"speaker":"John","text":"x"}],
            "advanced_event_arc_ids":["E1"], "used_parent_anchor_ids":["P1"],
            "continued_unresolved_hook_ids":[], "resolved_hook_ids":[]}
        early = {**base, "dialogue_turns":[{"speaker":"Priya Chen","text":"x"}]}
        self.assertIn("unauthorized_early_entity_introduction",
                      validate_realizer_identity_contract(early, outline, 1, []))
        self.assertNotIn("unauthorized_early_entity_introduction",
                         validate_realizer_identity_contract(early, outline, 2, []))
        bad_hook = {**base, "continued_unresolved_hook_ids":["P1"]}
        errors = validate_realizer_identity_contract(bad_hook, outline, 1, [])
        self.assertIn("unauthorized_unresolved_hook_id", errors)
        self.assertIn("session_1_existing_hook_ids_must_be_empty", errors)
        valid_hook = {**base, "continued_unresolved_hook_ids":["H0001"]}
        self.assertEqual(validate_realizer_identity_contract(
            valid_hook, outline, 2, [{"hook_id":"H0001","description":"next"}]), [])
        wrong_arc = {**base, "advanced_event_arc_ids":["P1"]}
        self.assertIn("unauthorized_event_arc",
                      validate_realizer_identity_contract(wrong_arc, outline, 1, []))
        wrong_anchor = {**base, "used_parent_anchor_ids":["E1"]}
        self.assertIn("unauthorized_parent_anchor_id",
                      validate_realizer_identity_contract(wrong_anchor, outline, 1, []))


class SafetyTests(unittest.TestCase):
    def test_config_context_and_lineage(self):
        validate_config(DEFAULTS)
        bad = dict(DEFAULTS); bad["max_verifier_generations"] = 1
        with self.assertRaises(ValueError): validate_config(bad)
        bad_length = dict(DEFAULTS); bad_length["solver_effective_max_model_len"] = 40960
        with self.assertRaises(ValueError): validate_config(bad_length)
        result = context_preflight(10, 90, 10, 10, 100)
        self.assertEqual(result["decision"], "reject_context_overflow")
        self.assertFalse(result["truncated"])
        with self.assertRaises(ValueError):
            validate_lineage_splits([{"conversation_id":"x","split":"train"},{"conversation_id":"x","split":"dev"}])

    def test_nonempty_run_refused(self):
        root = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/test-safety-temp")
        root.mkdir(parents=True, exist_ok=True); (root/"x").write_text("x")
        try:
            with self.assertRaises(FileExistsError): initialize_run(root, False)
        finally:
            (root/"x").unlink(); root.rmdir()

    def test_request_ledger_resume_ambiguity_and_budget(self):
        import shutil
        root = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/test-ledger-temp")
        shutil.rmtree(root, ignore_errors=True)
        calls = []
        try:
            ledger = RequestLedger(root, "manifest-hash", "mock")
            def backend():
                calls.append(1); return BackendResponse("raw", "mock")
            kwargs = dict(request_id="done", intent="questioner_batch", attempt=1,
                          prompt_sha256="prompt", call=backend,
                          budget_class="questioner", budget_limit=2)
            self.assertNotIn("reused", ledger.invoke(**kwargs).metadata)
            reused = ledger.invoke(**kwargs)
            self.assertEqual(reused.raw_text, "raw")
            self.assertTrue(reused.metadata["reused"])
            self.assertEqual(len(calls), 1)
            with self.assertRaises(RuntimeError):
                ledger.invoke(**{**kwargs, "prompt_sha256":"changed"})
            ledger.record_decision("done", "accepted")
            ledger.record_decision("done", "accepted")
            self.assertEqual(ledger.read("done")["final_decision"], "accepted")
            with self.assertRaises(RuntimeError): ledger.record_decision("done", "changed")
            atomic_json(ledger.state_dir/"ambiguous.json", {"request_id":"ambiguous", "status":"reserved",
                "budget_class":"questioner"})
            with self.assertRaises(AmbiguousRequestError):
                ledger.invoke(request_id="ambiguous", intent="questioner_batch", attempt=1,
                    prompt_sha256="p", call=backend, budget_class="questioner", budget_limit=2)
            with self.assertRaises(RuntimeError):
                ledger.invoke(request_id="over", intent="questioner_batch", attempt=1,
                    prompt_sha256="p", call=backend, budget_class="questioner", budget_limit=2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_parent_dialogue_hash_is_recomputed(self):
        turns = [{"speaker":"John", "text":" Hello   world "}]
        first = dialogue_content_hash(turns)
        self.assertEqual(first, dialogue_content_hash([{"speaker":"john","text":"Hello world"}]))
        self.assertNotEqual(first, dialogue_content_hash([{"speaker":"John","text":"changed"}]))

    def test_live_realizer_adapter_performs_one_raw_call(self):
        from unittest.mock import patch
        import json
        config = ChallengerAPIConfig("https://example.invalid/v1", "secret", "model", realizer_max_tokens=123)
        client = OpenAICompatibleChallengerClient(config)
        calls = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self):
                return json.dumps({"choices":[{"message":{"content":"{\"dialogue_turns\":[]}"},
                    "finish_reason":"stop"}], "usage":{"completion_tokens":7}}).encode()
        def fake_urlopen(request, timeout):
            calls.append((request, timeout)); return Response()
        backend = LiveRealizerBackend(client)
        request = RealizerRequest("r", 1, 1, "prompt", "hash")
        with patch("urllib.request.urlopen", fake_urlopen):
            response = backend.generate_session(request)
        self.assertEqual(len(calls), 1)
        self.assertEqual(response.raw_text, '{"dialogue_turns":[]}')
        self.assertEqual(response.metadata["usage"]["completion_tokens"], 7)
        sent = json.loads(calls[0][0].data.decode())
        self.assertEqual(sent["max_tokens"], 123)
        self.assertEqual(sent["response_format"], {"type":"json_object"})


class RemainingAuditRegressionTests(unittest.TestCase):
    """Regression cases added before their corresponding audit fixes."""

    @staticmethod
    def generated_item(qa_id="q1", letter="B"):
        return SchemaTests().smoke_item(qa_id, letter)

    @staticmethod
    def qa_context():
        helper=SchemaTests()
        return helper.catalog(), helper.contracts(), helper.turn_text(), helper.turn_sessions()

    def test_solver_source_is_materialized_only(self):
        raw = {**self.generated_item(), "options":dict(OPTIONS),
               "reference_answer_letter":"B", "reference_answer_text":"blue",
               "canonical_answer_text":"blue"}
        report = {"items":[{"qa_id":"q1", "schema_decision":"accept",
                            "target_letter":"B"}]}
        with self.assertRaisesRegex(RuntimeError, "solver_qa_not_materialized"):
            select_solver_qa([raw], report, "q1")

    def test_solver_source_rejects_admission_and_integrity_mismatch(self):
        generated = self.generated_item()
        catalog,contracts,turns,sessions=self.qa_context()
        materialized, errors = materialize_questioner_item(generated, "B",
            span_catalog=catalog, reasoning_contract=contracts[1])
        self.assertEqual(errors, [])
        with self.assertRaisesRegex(RuntimeError, "solver_qa_not_admitted"):
            select_solver_qa([generated], {"items":[{"qa_id":"q1", "schema_decision":"reject",
                "target_letter":"B", "materialized_qa":materialized}]}, "q1")
        mismatched = {**materialized, "qa_id":"other"}
        with self.assertRaisesRegex(RuntimeError, "solver_qa_integrity_mismatch"):
            select_solver_qa([generated], {"items":[{"qa_id":"q1", "schema_decision":"accept",
                "target_letter":"B", "materialized_qa":mismatched}]}, "q1")

    def test_solver_source_revalidates_tampered_materialized_qa(self):
        generated = self.generated_item()
        catalog,contracts,turns,sessions=self.qa_context()
        materialized, _ = materialize_questioner_item(generated, "B",
            span_catalog=catalog, reasoning_contract=contracts[1])
        materialized["evidence_turn_ids"] = ["missing"]
        report = {"items":[{"item_index":1,"qa_id":"q1", "schema_decision":"accept", "target_letter":"B",
                            "materialized_qa":materialized}]}
        with self.assertRaisesRegex(RuntimeError, "solver_qa_integrity_mismatch"):
            select_solver_qa([generated], report, "q1",
                frozen_turns=turns, turn_session_ids=sessions,
                span_catalog=catalog, reasoning_contracts=contracts)

    def test_ledger_metadata_is_part_of_completed_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); calls = []
            ledger = RequestLedger(root, "manifest", "mock")
            def backend():
                calls.append(1); return BackendResponse("raw", "model", {"usage":{"total_tokens":3}})
            base = dict(request_id="q", intent="questioner_batch", attempt=1,
                prompt_sha256="p", call=backend, budget_class="questioner", budget_limit=4,
                request_metadata={"assigned_gold_letters":["A","B","C"], "seed":7,
                                  "target_slot":0, "retry_index":0},
                backend_identity={"model":"m", "config_sha256":"cfg"})
            ledger.invoke(**base)
            reordered = {**base, "request_metadata":{"retry_index":0, "target_slot":0,
                "seed":7, "assigned_gold_letters":["A","B","C"]}}
            reused = ledger.invoke(**reordered)
            self.assertTrue(reused.metadata["reused"]); self.assertEqual(len(calls), 1)
            for changed in ({"assigned_gold_letters":["C","B","A"], "seed":7,
                             "target_slot":0, "retry_index":0},
                            {"assigned_gold_letters":["A","B","C"], "seed":8,
                             "target_slot":0, "retry_index":0},
                            {"assigned_gold_letters":["A","B","C"], "seed":7,
                             "target_slot":1, "retry_index":1}):
                with self.assertRaisesRegex(RuntimeError, "request contract drift"):
                    ledger.invoke(**{**base, "request_metadata":changed})
            with self.assertRaisesRegex(RuntimeError, "request contract drift"):
                ledger.invoke(**{**base, "backend_identity":{"model":"other", "config_sha256":"cfg"}})
            self.assertEqual(reused.metadata["usage"], {"total_tokens":3})

    def test_ledger_reuse_rejects_raw_tamper_and_reserved_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); ledger=RequestLedger(root,"manifest","mock")
            kwargs=dict(request_id="x", intent="solver_full_context", attempt=1,
                prompt_sha256="p", call=lambda:BackendResponse("<box>A</box>","m"),
                budget_class="local_solver", budget_limit=2,
                request_metadata={"qa_id":"q","target_slot":0,"retry_index":0,"seed":1},
                backend_identity={"model":"m"})
            ledger.invoke(**kwargs)
            Path(ledger.read("x")["raw_response_path"]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw response hash drift"):
                ledger.invoke(**kwargs)
            atomic_json(ledger.state_dir/"reserved.json", {"request_id":"reserved","status":"reserved"})
            with self.assertRaises(AmbiguousRequestError):
                ledger.invoke(**{**kwargs,"request_id":"reserved"})

    def test_generic_evidence_values_are_rejected(self):
        for value in ("the", "is", "During session", "the person", "this event"):
            qa = SchemaTests().valid()
            qa.pop("answer_evidence_values")
            qa["reference_answer_text"] = f"{value} blue"
            qa["options"]["B"] = qa["reference_answer_text"]
            qa["final_reasoning_conclusion"] = qa["reference_answer_text"]
            qa["answer_value_from_evidence"] = value
            qa["answer_supporting_quotes"] = [
                {"turn_id":"T1", "exact_quote":f"{value} blue"},
                {"turn_id":"T2", "exact_quote":"gold"}]
            errors = validate_qa(qa, {"T1","T2"})
            self.assertIn("bad_generic_evidence_value", errors, value)

    def test_structured_multihop_values_are_distinct_and_bound(self):
        checker = getattr(validation_module, "validate_answer_evidence_values", None)
        self.assertTrue(callable(checker))
        frozen = {"T1":"education repairs were documented",
                  "T2":"infrastructure safety was documented"}
        good = [{"turn_id":"T1", "value":"education", "exact_quote":frozen["T1"],
                 "role":"answer_value"},
                {"turn_id":"T2", "value":"infrastructure", "exact_quote":frozen["T2"],
                 "role":"answer_value"}]
        self.assertEqual(checker(good, ["T1","T2"], frozen,
                                 "education and infrastructure", "multi_hop"), [])
        duplicate = [good[0], {**good[0], "turn_id":"T2", "exact_quote":"education again"}]
        self.assertIn("bad_duplicate_information_point",
            checker(duplicate, ["T1","T2"], {**frozen,"T2":"education again"},
                    "education", "multi_hop"))
        substring = [{"turn_id":"T1","value":"cat","exact_quote":"education",
                      "role":"answer_value"}]
        self.assertIn("bad_answer_evidence_binding",
            checker(substring,["T1"],{"T1":"education"},"education","direct_span"))

    def test_source_run_integrity_import_contract(self):
        verifier = getattr(input_adapter_module, "verify_source_run_sessions", None)
        self.assertTrue(callable(verifier))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "accepted/session_01.json"
            atomic_json(session, {"session_id":"session_01", "dialogue_turns":[]})
            manifest = root / "config/run_manifest.json"
            atomic_json(manifest, {"run_id":"source"})
            atomic_json(root / "accepted/index.json", {"schema_version":"accepted_session_integrity_v2",
                "run_id":"source", "run_manifest_sha256":sha256_file(manifest),
                "session_sha256":{"session_01.json":sha256_file(session)}})
            atomic_json(root / "reports/post_run_provenance.json", {
                "authoritative_release_provenance":True, "provenance_mixed":False,
                "run_manifest_sha256":sha256_file(manifest)})
            result = verifier(root, ["session_01.json"])
            self.assertEqual(result["source_run_id"], "source")
            atomic_json(session, {"session_id":"tampered", "dialogue_turns":[]})
            with self.assertRaisesRegex(RuntimeError, "source_session_sha_mismatch"):
                verifier(root, ["session_01.json"])
            session.unlink(); (root / "accepted/index.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "source_run_integrity_failed"):
                verifier(root, ["session_01.json"])

    def test_source_run_rejects_non_authoritative_provenance(self):
        verifier = getattr(input_adapter_module, "verify_source_run_sessions", None)
        self.assertTrue(callable(verifier))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"accepted").mkdir(parents=True)
            session = root/"accepted/session_01.json"; atomic_json(session,{"session_id":"s"})
            manifest=root/"config/run_manifest.json"; atomic_json(manifest,{"run_id":"r"})
            atomic_json(root/"accepted/index.json", {"run_id":"r",
                "run_manifest_sha256":sha256_file(manifest),
                "session_sha256":{"session_01.json":sha256_file(session)}})
            for authoritative, mixed in ((False,False),(True,True)):
                atomic_json(root/"reports/post_run_provenance.json", {
                    "authoritative_release_provenance":authoritative, "provenance_mixed":mixed,
                    "run_manifest_sha256":sha256_file(manifest)})
                with self.assertRaisesRegex(RuntimeError, "source_run_not_authoritative"):
                    verifier(root,["session_01.json"])

    def test_dry_run_preflight_uses_effective_16384(self):
        builder = getattr(run_canary_module, "build_input_dry_run_audit", None)
        self.assertTrue(callable(builder))
        audit = builder(parent_turns=237, parent_tokens=16808, lineage="conv-41",
                        reserved_generation_tokens=64, checkpoint_max_model_len=40960,
                        effective_max_model_len=16384, spell_commit="x", spell_dirty=False)
        self.assertEqual(audit["context_preflight"]["solver_max_model_len"], 16384)
        self.assertEqual(audit["context_preflight"]["decision"], "reject_context_overflow")

    def test_hook_transition_blocks_required_continuation_dead_end(self):
        checker = getattr(validation_module, "validate_hook_transition", None)
        self.assertTrue(callable(checker))
        s1 = checker(set(), set(), set(), ["H1"], can_stop_after=False)
        self.assertEqual(s1["active_after"], ["H1"])
        s2 = checker({"H1"}, {"H1"}, set(), ["H2"], can_stop_after=False)
        self.assertEqual(s2["active_after"], ["H1","H2"])
        s3 = checker({"H1","H2"}, {"H2"}, {"H1"}, [], can_stop_after=False)
        self.assertEqual(s3["active_after"], ["H2"])
        blocked = checker({"H2"}, set(), {"H2"}, [], can_stop_after=False)
        self.assertIn("no_unresolved_hook_for_required_continuation", blocked["errors"])
        allowed = checker({"H2"}, set(), {"H2"}, [], can_stop_after=True)
        self.assertEqual(allowed["errors"], [])

    def test_event_arc_advance_is_assigned_and_not_closed(self):
        checker = getattr(validation_module, "validate_event_arc_advancement", None)
        self.assertTrue(callable(checker))
        turns = [{"speaker":"John", "text":"The application plan for E1 was discussed."}]
        self.assertIn("unassigned_event_arc_advance",
            checker(["E1","E2"], {"E1"}, set(), turns, {"E1":"application plan"}))
        self.assertIn("already_closed_event_arc",
            checker(["E1"], {"E1"}, {"E1"}, turns, {"E1":"application plan"}))
        self.assertEqual(checker(["E1"], {"E1"}, set(), turns,
                                 {"E1":"application plan"}), [])
        self.assertIn("missing_event_arc_dialogue_support",
            checker(["E1"], {"E1"}, set(), [{"speaker":"John","text":"Unrelated weather."}],
                    {"E1":"application plan"}))

    def test_durable_assertion_requires_supported_subject_and_predicate(self):
        bad = {"assertion_id":"m", "subject":"Alice owns the city", "predicate":"status",
               "object_or_state":"drafting", "supporting_turn_ids":["T1"],
               "supporting_text_spans":[{"turn_id":"T1", "exact_span":"brief is drafting"}],
               "status":"active", "supersedes_assertion_ids":[]}
        self.assertIn("unsupported_assertion_subject",
            validate_memory(bad, {"T1":"The brief is drafting."}, known_subjects={"brief"},
                            predicate_whitelist={"status"}))
        good = {**bad, "subject":"brief"}
        self.assertEqual(validate_memory(good, {"T1":"The brief is drafting."},
            known_subjects={"brief"}, predicate_whitelist={"status"}), [])
        wrong_predicate = {**good, "predicate":"owns the city"}
        self.assertIn("unsupported_assertion_predicate",
            validate_memory(wrong_predicate, {"T1":"The brief is drafting."},
                            known_subjects={"brief"}, predicate_whitelist={"status"}))
        self.assertIn("bad_supporting_turn_indices",
            validate_durable_candidate_indices([{"supporting_turn_indices":[2]}], 1))

    def test_durable_supersession_key_is_subject_and_predicate(self):
        sessions = [{"session_id":"session_01", "dialogue_turns":[
            {"turn_id":"C0001","speaker":"John","text":"The brief status is drafting."}],
            "durable_assertion_candidates":[{"subject":"brief","predicate":"status",
                "object_or_state":"drafting","supporting_turn_indices":[0]}]},
            {"session_id":"session_02", "dialogue_turns":[
                {"turn_id":"C0002","speaker":"John","text":"The report status is submitted."}],
             "durable_assertion_candidates":[{"subject":"report","predicate":"status",
                "object_or_state":"submitted","supporting_turn_indices":[0]}]}]
        orchestrator=object.__new__(CanaryOrchestrator)
        orchestrator.data={"outline":{"source_parent":{"core_speakers":["John"]},
            "planned_event_arcs":[{"summary":"brief report"}]}}
        memory=orchestrator._durable_memory(sessions)
        self.assertEqual(memory["accepted_count"],2)
        self.assertEqual(memory["assertions"][1]["supersedes_assertion_ids"],[])

    def test_evidence_audit_shortfall_is_inconclusive(self):
        qa = {"difficulty_level":"L2", "evidence_turn_ids":["T1","T2"],
              "reasoning_steps":["a","b","c"]}
        complete = {"requested_valid_rollouts":3, "obtained_valid_rollouts":3,
                    "verified_rewards":[1,1,0], "raw_generations":3,
                    "parse_invalid_generations":0, "retry_generations":0,
                    "exhausted_target_slots":0}
        short = {**complete, "obtained_valid_rollouts":2, "verified_rewards":[1,1],
                 "raw_generations":4, "parse_invalid_generations":2,
                 "retry_generations":1, "exhausted_target_slots":1}
        for left, right in ((short,complete),(complete,short)):
            result = evidence_audit(qa, left, right)
            self.assertEqual(result["decision"], "evidence_audit_inconclusive")
            self.assertEqual(result["difficulty_level_after"], "L2")
        self.assertEqual(evidence_audit(qa, complete, {**complete,"verified_rewards":[0,0,1]})[
            "decision"], "certified")

    def test_candidate_shortfall_distinguishes_raw_and_accepted(self):
        assessor = getattr(validation_module, "assess_candidate_shortfall", None)
        self.assertTrue(callable(assessor))
        accepted_zero = assessor([3,3,3,3], 0, minimum_solver_eligible_candidates=1)
        self.assertFalse(accepted_zero["raw_candidate_count_shortfall"])
        self.assertTrue(accepted_zero["accepted_candidate_shortfall"])
        self.assertFalse(accepted_zero["completed"])
        raw_short = assessor([3,2,3,3], 8, minimum_solver_eligible_candidates=1)
        self.assertTrue(raw_short["raw_candidate_count_shortfall"])
        accepted = assessor([3,3,3,3], 1, minimum_solver_eligible_candidates=1)
        self.assertTrue(accepted["completed"])

    def test_source_snapshot_contains_bodies_and_manifest(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            result=create_source_snapshot(project, Path(temp))
            snapshot=Path(result["source_snapshot_root"])
            self.assertTrue((snapshot/"files/scripts/current_method/restart_n5_spell/orchestrator.py").is_file())
            self.assertEqual((snapshot/"files/tests/test_restart_n5_spell.py").read_bytes(),
                             (project/"tests/test_restart_n5_spell.py").read_bytes())
            manifest=json.loads((snapshot/"snapshot_manifest.json").read_text())
            self.assertFalse(manifest["git_commit_created"])
            self.assertEqual(result["source_snapshot_manifest_sha256"],
                             sha256_file(snapshot/"snapshot_manifest.json"))

    def test_live_like_offline_materialization_pipeline(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False): return str(text).split()
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False):
                return "\n".join(x["content"] for x in messages) + "\n<think>\n\n</think>\n\n"
        class Realizer:
            model_identifier="offline-live-like-realizer"
            topics={1:"application",2:"resident",3:"verification",4:"presentation"}
            def generate_session(self, request):
                topic=self.topics[request.session_index]
                text=(f"John discussed the {topic} work, education, infrastructure, and the brief status "
                      + ("drafting. " if request.session_index == 1 else
                         "submitted. " if request.session_index == 4 else "active. ")) * 90
                hooks={1:([],[],[{"description":"continue application review"}]),
                    2:(["H0001"],[],[{"description":"continue resident review"}]),
                    3:(["H0002"],["H0001"],[]), 4:([],["H0002"],[])}[request.session_index]
                assertions=[]
                if request.session_index in (1,4):
                    assertions=[{"subject":"brief","predicate":"status",
                        "object_or_state":"drafting" if request.session_index == 1 else "submitted",
                        "supporting_turn_indices":[0]}]
                payload={"dialogue_turns":[{"speaker":"John","text":text}],
                    "used_parent_anchor_ids":["P1"] if request.session_index == 1 else [],
                    "advanced_event_arc_ids":[f"E{request.session_index}"],
                    "continued_unresolved_hook_ids":hooks[0], "resolved_hook_ids":hooks[1],
                    "new_unresolved_hooks":hooks[2], "durable_assertion_candidates":assertions,
                    "local_scene_facts":[f"{topic} work occurred"]}
                return BackendResponse(json.dumps(payload),self.model_identifier)
        class Questioner:
            model_identifier="offline-live-like-questioner"
            def generate_batch(self, request):
                child=[x for x in request.span_catalog["spans"] if x["turn_id"].startswith("C")]
                rows=[]
                for index in range(3):
                    qa_id=f"batch{request.batch_index}_item{index+1}"
                    correct=f"application and resident priorities {qa_id}"
                    rows.append({"qa_id":qa_id,"question":f"Which paired priorities resolve {qa_id}?",
                        "correct_answer_text":correct,
                        "distractors":[f"verification alternative {qa_id}",
                            f"presentation alternative {qa_id}",f"brief alternative {qa_id}"],
                        "answer_support_span_ids":[child[0]["span_id"],child[1]["span_id"]],
                        "answer_evidence_values":[
                            {"span_id":child[0]["span_id"],"value":"application"},
                            {"span_id":child[1]["span_id"],"value":"resident"}],
                        "distractor_support":[
                            {"distractor_index":0,"span_ids":[child[2]["span_id"]],
                             "evidence_values":["verification"]},
                            {"distractor_index":1,"span_ids":[child[3]["span_id"]],
                             "evidence_values":["presentation"]},
                            {"distractor_index":2,"span_ids":[child[0]["span_id"]],
                             "evidence_values":["brief"]}],
                        "correct_answer_explanation":"The two answer values are directly supported.",
                        "distractor_explanations":["uses the verification candidate",
                            "uses the presentation candidate","uses the brief candidate"],
                        "reasoning_steps":["Read the application value.","Combine the resident value."],
                    })
                return BackendResponse(json.dumps(rows),self.model_identifier)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest={"run_id":"offline-live-like","tokenizer_path":"fake",
                "solver_max_model_len":16384}
            atomic_json(root/"config/run_manifest.json",manifest)
            data={"parent":{"dialogue":{"turns":[{"turn_id":"D1:1","speaker":"Maria",
                    "text":"Maria asked John about education and infrastructure."}]}},
                "outline":{"source_parent":{"core_speakers":["John","Maria"]},
                    "allowed_new_characters":[],"allowed_locations":["library"],
                    "parent_anchors":[{"anchor_id":"P1","text":"community plans"}],
                    "planned_event_arcs":[
                        {"event_id":"E1","summary":"application","stages":["application"]},
                        {"event_id":"E2","summary":"resident","stages":["resident"]},
                        {"event_id":"E3","summary":"verification","stages":["verification"]},
                        {"event_id":"E4","summary":"presentation","stages":["presentation"]}],
                    "scene_plan":[{"scene_id":f"S{i}","purpose":"advance assigned work"}
                                  for i in range(1,5)],"forbidden_conflicts":[]}}
            config={**DEFAULTS,"mode":"mock_e2e","max_realizer_generations":8,
                    "max_questioner_generations":4}
            orchestrator=CanaryOrchestrator(run_root=root,manifest=manifest,config=config,
                data=data,tokenizer=Tokenizer(),realizer=Realizer(),questioner=Questioner(),
                solver=None,backend_mode="mock")
            sessions=orchestrator.realize()
            self.assertEqual(len(sessions),4); self.assertEqual(orchestrator._active_hooks(sessions),[])
            child,memory=orchestrator.freeze(sessions)
            self.assertEqual(memory["accepted_count"],2)
            candidates,summary=orchestrator.question(child)
            self.assertEqual(summary["questioner_solver_eligible_count"],12)
            raw_wrapper=json.loads((root/"raw/questioner_batch_00.json").read_text())
            raw_batch=json.loads(raw_wrapper["raw_response"])
            turns=data["parent"]["dialogue"]["turns"]+child
            turn_text={x["turn_id"]:x["text"] for x in turns}
            turn_sessions={x["turn_id"]:str(x.get("session_id") or x["turn_id"].split(":",1)[0])
                           for x in turns}
            admission=validate_questioner_smoke_batch(raw_batch,["A","B","C"],set(turn_text),
                set(turn_sessions.values()),turn_text,turn_sessions,
                span_catalog=build_span_catalog(turns),
                reasoning_contracts=default_reasoning_contracts(("A","B","C")))
            _,qa,_=select_solver_qa(raw_batch,admission,raw_batch[0]["qa_id"],
                frozen_turns=turn_text,turn_session_ids=turn_sessions,
                span_catalog=build_span_catalog(turns),
                reasoning_contracts=default_reasoning_contracts(("A","B","C")))
            self.assertTrue(qa["local_option_binding"])
            prompt=render_solver_prompt(Tokenizer(),qa,[])
            self.assertIn("<box>A</box>",prompt)
            self.assertNotIn("reference_answer_text", raw_batch[0])


class PromptTests(unittest.TestCase):
    def realizer_args(self, session=1):
        return dict(session_index=session, narrative_beats=[{"event_id":"E1","beat":"advance"}],
            recent_sessions=[{"session_id":"S0","dialogue_turns":[{"speaker":"A","text":"prior"}]}],
            continuity_summary="earlier facts", active_plans=["plan"], pending_decisions=["decision"],
            superseded_information=["old state"], character_knowledge={"A":["known"]},
            visibility_boundaries=["A did not see X"], unresolved_hooks=["hook"] if session > 1 else [],
            allowed_people=["A","B"], session_location="library", forbidden_conflicts=["no contradiction"],
            allowed_event_arc_ids=["E1","E2"],
            parent_anchors=[{"anchor_id":"P1","text":"opening"}])

    def test_realizer_dynamic_protocol(self):
        prompt = build_realizer_prompt(**self.realizer_args())
        for expected in ("Session 1", "supporting_turn_indices", "Do not output turn_id",
                         "narrative_beats_to_advance", "superseded_information", "strict JSON",
                         "allowed_event_arc_ids", "allowed_parent_anchor_ids", "allowed_speaker_names",
                         "fixed_session_location_do_not_echo_as_metadata", "resolved_hook_ids", "description only"):
            self.assertIn(expected, prompt)
        self.assertIn("continued_unresolved_hook_ids and resolved_hook_ids MUST both be []", prompt)
        self.assertIn("parent anchors use P*", prompt)
        prompt2 = build_realizer_prompt(**self.realizer_args(2))
        self.assertIn("Session 2+", prompt2)
        bad = self.realizer_args(2); bad["unresolved_hooks"] = []
        with self.assertRaises(ValueError): build_realizer_prompt(**bad)

    def test_questioner_strict_three_item_protocol(self):
        dialogue = [{"session_id":"S1","turn_id":"T1","speaker":"A","text":"fact"},
                    {"session_id":"S2","turn_id":"T2","speaker":"B","text":"fact two"}]
        history = [{"question":"old", "canonical_answer":"history-only-secret", "reasoning_type":"old",
                    "evidence_session_pattern":["S0"], "verified_reward_mean":.4, "round":0,
                    "forbidden_extra":"must-not-appear"}]
        prompt = build_questioner_prompt(frozen_dialogue=dialogue, valid_turn_ids=["T1","T2"],
            valid_session_ids=["S1","S2"], assigned_gold_letters=["A","B","C"],
            recent_frontier_history=history, batch_index=1)
        for expected in ("exactly three", "correct_answer_text", "three unlabeled distractors",
                         "span_catalog", "reasoning_contracts_by_array_position",
                         "answer_support_span_ids", "sole factual source"):
            self.assertIn(expected, prompt)
        for forbidden in ("valid_turn_id_whitelist", "evidence_turn_ids\":",
                          "difficulty_level\":", "reasoning_type\": \"string"):
            self.assertNotIn(forbidden, prompt)
        self.assertNotIn('"assigned_gold_letters_by_array_index"', prompt)
        self.assertNotIn("forbidden_extra", prompt)
        with self.assertRaises(ValueError):
            build_questioner_prompt(frozen_dialogue=dialogue, valid_turn_ids=["T1"], valid_session_ids=["S1"],
                assigned_gold_letters=["A","B"], recent_frontier_history=[])

    def test_template_hashes_are_stable_and_distinct(self):
        first = prompt_template_hashes(); second = prompt_template_hashes()
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"realizer","questioner","responder"})
        self.assertEqual(len(set(first.values())), 3)
        self.assertTrue(all(len(value) == 64 for value in first.values()))


class QuestionerV7RedTests(unittest.TestCase):
    """Added first: these cases intentionally fail against the frozen v6 checkpoint."""

    def test_v7_contract_module_and_versions_exist(self):
        import importlib
        module = importlib.import_module(
            "scripts.current_method.restart_n5_spell.questioner_contract")
        self.assertEqual(module.SPAN_CATALOG_VERSION, "dialogue_span_catalog_v1")
        self.assertEqual(module.REASONING_CONTRACT_VERSION, "questioner_reasoning_contract_v1")
        self.assertEqual(module.QUESTIONER_SCHEMA_VERSION, "questioner_unlabeled_span_ids_v1")

    def test_prompt_version_is_v7(self):
        from scripts.current_method.restart_n5_spell.prompts import QUESTIONER_PROMPT_VERSION
        self.assertEqual(
            QUESTIONER_PROMPT_VERSION,
            "spell_dialogue_mcq_3x_v7_span_contract_local_derivation")

    def test_v7_raw_schema_has_span_ids_and_no_model_local_fields(self):
        from scripts.current_method.restart_n5_spell.prompts import QUESTIONER_GENERATED_ITEM_SCHEMA
        self.assertIn("answer_support_span_ids", QUESTIONER_GENERATED_ITEM_SCHEMA)
        self.assertIn("distractor_support", QUESTIONER_GENERATED_ITEM_SCHEMA)
        for forbidden in ("exact_quote", "evidence_turn_ids", "evidence_session_ids",
                          "reasoning_type", "difficulty_level", "options",
                          "reference_answer_letter", "reference_answer_text"):
            self.assertNotIn(forbidden, QUESTIONER_GENERATED_ITEM_SCHEMA)

    def test_v7_materializer_requires_catalog_and_contract(self):
        raw = {"qa_id":"q1", "question":"Which two values?",
            "correct_answer_text":"education and infrastructure",
            "distractors":["parks and roads", "health and housing", "art and music"],
            "answer_support_span_ids":["T1#S0001", "T2#S0001"],
            "answer_evidence_values":[
                {"span_id":"T1#S0001", "value":"education"},
                {"span_id":"T2#S0001", "value":"infrastructure"}],
            "distractor_support":[
                {"distractor_index":0,"span_ids":["T1#S0001"],"evidence_values":["parks"]},
                {"distractor_index":1,"span_ids":["T4#S0001"],"evidence_values":["health"]},
                {"distractor_index":2,"span_ids":["T5#S0001"],"evidence_values":["art"]}],
            "correct_answer_explanation":"Two turns state the two values.",
            "distractor_explanations":["different pair", "different pair", "different pair"],
            "reasoning_steps":["Read the first turn.", "Combine the second turn."]}
        catalog = build_span_catalog([
            {"turn_id":"T1","session_id":"S1","speaker":"A",
             "text":"education red parks"},
            {"turn_id":"T2","session_id":"S1","speaker":"B",
             "text":"infrastructure green health"},
            {"turn_id":"T3","session_id":"S1","speaker":"A","text":"gold art music"},
            {"turn_id":"T4","session_id":"S2","speaker":"B","text":"health roads"},
            {"turn_id":"T5","session_id":"S2","speaker":"A","text":"art libraries"},
        ])
        contract = default_reasoning_contracts(("A","B","C"))[0]
        qa, errors = materialize_questioner_item(
            raw, "A", span_catalog=catalog, reasoning_contract=contract)
        self.assertIsNotNone(qa)
        self.assertEqual(errors, [])


class QuestionerV7ContractTests(unittest.TestCase):
    def setUp(self):
        self.helper = SchemaTests()
        self.catalog = self.helper.catalog()
        self.contracts = self.helper.contracts()
        self.turns = self.helper.turn_text()
        self.sessions = self.helper.turn_sessions()

    def materialize(self, raw, index=0):
        return materialize_questioner_item(raw, self.contracts[index]["target_letter"],
            span_catalog=self.catalog, reasoning_contract=self.contracts[index])

    def test_span_catalog_is_deterministic_unique_and_hashed(self):
        again = build_span_catalog(self.helper.dialogue())
        self.assertEqual(self.catalog, again)
        self.assertEqual(len({x["span_id"] for x in self.catalog["spans"]}), 6)
        self.assertEqual(len(span_catalog_sha256(self.catalog)), 64)
        duplicate = self.helper.dialogue() + [dict(self.helper.dialogue()[0])]
        with self.assertRaisesRegex(ValueError, "span_catalog_invalid_turn"):
            build_span_catalog(duplicate)

    def test_local_exact_quote_restoration_ignores_explanation_paraphrase(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["correct_answer_explanation"] = "A lightly paraphrased explanation is model prose."
        qa, errors = self.materialize(raw)
        self.assertEqual(errors, [])
        self.assertEqual(qa["answer_supporting_quotes"][0]["exact_quote"],
                         self.turns["T1"])
        self.assertNotEqual(qa["answer_supporting_quotes"][0]["exact_quote"],
                            raw["correct_answer_explanation"])

    def test_unknown_answer_span_is_rejected(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["answer_support_span_ids"][0] = "UNKNOWN#S0001"
        raw["answer_evidence_values"][0]["span_id"] = "UNKNOWN#S0001"
        qa, errors = self.materialize(raw)
        self.assertIsNone(qa); self.assertIn("bad_span_id", errors)

    def test_model_supplied_exact_quote_is_rejected(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["answer_evidence_values"][0]["exact_quote"] = "education red priorities"
        qa, errors = self.materialize(raw)
        self.assertIsNone(qa)
        self.assertIn("model_supplied_local_binding_fields", errors)

    def test_model_supplied_reasoning_and_difficulty_are_rejected(self):
        for field, value in (("reasoning_type", "freeform"), ("difficulty_level", "L1")):
            raw = self.helper.smoke_item("q1", "A"); raw[field] = value
            qa, errors = self.materialize(raw)
            self.assertIsNone(qa); self.assertIn("model_supplied_local_binding_fields", errors)

    def test_contract_requires_two_distinct_turns(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["answer_support_span_ids"] = ["T1#S0001"]
        raw["answer_evidence_values"] = [{"span_id":"T1#S0001","value":"education"}]
        raw["correct_answer_text"] = "education for q1"
        qa, errors = self.materialize(raw)
        self.assertIsNone(qa)
        self.assertIn("bad_reasoning_contract", errors)
        self.assertIn("bad_difficulty_evidence_contract", errors)

    def test_cross_session_contract_rejects_one_session(self):
        raw = self.helper.smoke_item("q2", "B")
        raw["answer_support_span_ids"] = ["T1#S0001", "T2#S0001"]
        raw["answer_evidence_values"] = [
            {"span_id":"T1#S0001","value":"education"},
            {"span_id":"T2#S0001","value":"infrastructure"}]
        raw["correct_answer_text"] = "education and infrastructure for q2"
        qa, errors = self.materialize(raw, 1)
        self.assertIsNone(qa); self.assertIn("bad_reasoning_contract", errors)

    def test_duplicate_and_generic_evidence_values_are_rejected(self):
        duplicate = self.helper.smoke_item("q1", "A")
        duplicate["answer_evidence_values"][1]["value"] = "education"
        duplicate["correct_answer_text"] = "education for q1"
        self.assertIn("bad_duplicate_information_point", self.materialize(duplicate)[1])
        generic = self.helper.smoke_item("q1", "A")
        generic["answer_evidence_values"][0]["value"] = "the"
        generic["correct_answer_text"] = "the and infrastructure for q1"
        self.assertIn("bad_generic_evidence_value", self.materialize(generic)[1])

    def test_every_answer_span_requires_an_evidence_value(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["answer_evidence_values"] = raw["answer_evidence_values"][:1]
        self.assertIn("bad_answer_evidence_binding", self.materialize(raw)[1])

    def test_distractor_unknown_span_and_value_binding_are_rejected(self):
        unknown = self.helper.smoke_item("q1", "A")
        unknown["distractor_support"][0]["span_ids"] = ["UNKNOWN#S0001"]
        errors = self.materialize(unknown)[1]
        self.assertIn("bad_span_id", errors); self.assertIn("bad_distractor_support", errors)
        absent_span = self.helper.smoke_item("q1", "A")
        absent_span["distractor_support"][0]["evidence_values"] = ["purple"]
        self.assertIn("bad_distractor_support", self.materialize(absent_span)[1])
        absent_option = self.helper.smoke_item("q1", "A")
        absent_option["distractor_support"][0]["evidence_values"] = ["education"]
        self.assertIn("bad_distractor_support", self.materialize(absent_option)[1])

    def test_unhashable_distractor_index_is_rejected_without_crashing(self):
        raw = self.helper.smoke_item("q1", "A")
        raw["distractor_support"][0]["distractor_index"] = [0]
        qa, errors = self.materialize(raw)
        self.assertIsNone(qa)
        self.assertIn("bad_distractor_support", errors)

    def test_local_ids_reasoning_difficulty_and_target_are_derived(self):
        expected = ((0,"A","multi_turn_combination","L1",["T1","T2"],["S1"]),
                    (1,"B","cross_session_relation","L2",["T1","T4"],["S1","S2"]),
                    (2,"C","temporal_or_causal","L2",["T2","T5"],["S1","S2"]))
        for index,target,reasoning,difficulty,turn_ids,session_ids in expected:
            raw = self.helper.smoke_item(f"q{index+1}", target)
            qa, errors = self.materialize(raw, index)
            self.assertEqual(errors, [])
            self.assertEqual(qa["reference_answer_letter"], target)
            self.assertEqual(qa["reference_answer_text"], qa["options"][target])
            self.assertEqual(qa["evidence_turn_ids"], turn_ids)
            self.assertEqual(qa["evidence_session_ids"], session_ids)
            self.assertEqual(qa["reasoning_type"], reasoning)
            self.assertEqual(qa["difficulty_level"], difficulty)

    def test_l3_is_derived_only_from_higher_structure(self):
        self.assertEqual(derive_local_difficulty(distinct_spans=3, distinct_turns=3,
            distinct_sessions=2, distinct_values=3, reasoning_steps=3,
            reasoning_type="temporal_or_causal"), "L3")
        self.assertEqual(derive_local_difficulty(distinct_spans=3, distinct_turns=3,
            distinct_sessions=2, distinct_values=3, reasoning_steps=3,
            reasoning_type="multi_turn_combination"), "L2")

    def test_three_positions_have_unique_local_contracts(self):
        self.assertEqual([x["item_index"] for x in self.contracts], [0,1,2])
        self.assertEqual(len({x["reasoning_type"] for x in self.contracts}), 3)
        self.assertEqual(len(reasoning_contract_sha256(self.contracts)), 64)

    def test_independent_item_failure_and_live_like_selector(self):
        batch = [self.helper.smoke_item("q1","A"), self.helper.smoke_item("q2","B"),
                 self.helper.smoke_item("q3","C")]
        batch[2]["answer_support_span_ids"][0] = "BAD#S0001"
        batch[2]["answer_evidence_values"][0]["span_id"] = "BAD#S0001"
        admission = validate_questioner_smoke_batch(batch,["A","B","C"],set(self.turns),
            {"S1","S2"},self.turns,self.sessions,span_catalog=self.catalog,
            reasoning_contracts=self.contracts)
        self.assertEqual(admission["schema_accepted_count"], 2)
        self.assertEqual([x["schema_decision"] for x in admission["items"]],
                         ["accept","accept","reject"])
        self.assertIn("bad_span_id", admission["items"][2]["reject_reasons"])
        for qa_id in ("q1","q2"):
            _, qa, _ = select_solver_qa(batch, admission, qa_id, frozen_turns=self.turns,
                turn_session_ids=self.sessions, span_catalog=self.catalog,
                reasoning_contracts=self.contracts)
            self.assertTrue(qa["local_option_binding"])
            self.assertTrue(all(quote["exact_quote"] in self.turns[quote["turn_id"]]
                                for quote in qa["answer_supporting_quotes"]))

    def test_three_accepted_materialized_qas_render_boxed_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False):
                return "\n".join(x["content"] for x in messages)
        batch=[self.helper.smoke_item("q1","A"),self.helper.smoke_item("q2","B"),
               self.helper.smoke_item("q3","C")]
        admission=validate_questioner_smoke_batch(batch,["A","B","C"],set(self.turns),
            {"S1","S2"},self.turns,self.sessions,span_catalog=self.catalog,
            reasoning_contracts=self.contracts)
        self.assertEqual(admission["schema_accepted_count"],3)
        for raw in batch:
            _,qa,_=select_solver_qa(batch,admission,raw["qa_id"],frozen_turns=self.turns,
                turn_session_ids=self.sessions,span_catalog=self.catalog,
                reasoning_contracts=self.contracts)
            self.assertIn("<box>A</box>", render_solver_prompt(Tokenizer(),qa,[]))

    def test_prompt_exposes_only_compact_span_view_and_local_contracts(self):
        prompt=build_questioner_prompt(frozen_dialogue=self.helper.dialogue(),
            valid_turn_ids=list(self.turns),valid_session_ids=["S1","S2"],
            assigned_gold_letters=["A","B","C"],recent_frontier_history=[],
            span_catalog=self.catalog,reasoning_contracts=self.contracts)
        self.assertIn('"span_id": "T1#S0001"',prompt)
        self.assertNotIn('"turn_id":',prompt); self.assertNotIn('"session_id":',prompt)
        self.assertNotIn('"target_letter":',prompt)


if __name__ == "__main__": unittest.main()
