import tempfile
import unittest
import json
import shutil
import copy
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import (
    canonical_json,
    classify_reward_group, collect_valid_rollouts, context_preflight, grounding_decision,
    local_reward, option_audit, parse_mcq_response, parse_boxed_mcq_answer,
    classify_frontier_pre_audit, classify_no_context_permutation_bias, full_context_gate,
    sha256_bytes,
)
from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.run_canary import (
    DEFAULTS, create_source_snapshot, initialize_run, render_solver_prompt,
    select_minimal_solver_qa, select_solver_qa,
)
from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json
from scripts.current_method.restart_n5_spell.ledger import AmbiguousRequestError, RequestLedger
from scripts.current_method.restart_n5_spell.live_backends import (
    ChallengerAPIConfig, LiveRealizerBackend, LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
)
from scripts.current_method.restart_n5_spell.backends import RealizerRequest
from scripts.current_method.restart_n5_spell.orchestrator import CanaryOrchestrator
from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
from scripts.current_method.restart_n5_spell import input_adapter as input_adapter_module
from scripts.current_method.restart_n5_spell import validation as validation_module
from scripts.current_method.restart_n5_spell import run_canary as run_canary_module
from scripts.current_method.restart_n5_spell import run_cross_dialogue_calibration as cross_calibration_runner
from scripts.current_method.restart_n5_spell.prompts import (
    REALIZER_OUTPUT_SCHEMA, build_graph_realizer_prompt,
    build_graph_realizer_retry_prompt, build_questioner_feedback_prompt,
    build_questioner_prompt, build_questioner_span_ids_only_feedback_prompt, build_realizer_prompt,
    build_semantic_leakage_repair_prompt,
    feedback_questioner_template_hash, graph_realizer_template_hash, prompt_template_hashes,
    semantic_leakage_repair_template_hash,
    SEMANTIC_LEAKAGE_REPAIR_JSON_SCHEMA,
    SEMANTIC_LEAKAGE_REPAIR_PROMPT_VERSION,
    span_ids_only_questioner_template_hash,
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
from scripts.current_method.restart_n5_spell.semantic_repair import (
    admit_semantic_repair_item, build_semantic_repair_feedback_summary,
    semantic_repair_eligibility,
)
from scripts.current_method.restart_n5_spell.cross_dialogue_calibration import (
    CALIBRATION_LINEAGES, CALIBRATION_TARGET_LETTERS, calibration_plan,
    freeze_parent_dialogue, summarize_calibration,
    verify_semantic_repair_feedback_source,
)
from scripts.current_method.restart_n5_spell.graph_expansion import (
    GRAPH_SCENE_CONTRACT_VERSION, compile_graph_scene_contracts,
    graph_completion_report, normalize_graph_realizer_managed_ids,
    validate_graph_scene_output,
)
from scripts.current_method.restart_n5_spell.evidence_view_ablation import (
    BALANCED_VIEW, CHILD_ONLY_VIEW, BALANCED_PARENT_SELECTION,
    build_questioner_evidence_view, evidence_view_sha256,
    selected_evidence_diagnostics, uniform_endpoint_indices,
)
from scripts.current_method.restart_n5_spell.three_scope_canary import (
    CHILD_LOCAL, PARENT_CHILD_JOINT, PARENT_LOCAL, SCOPED_BATCH_JSON_SCHEMA,
    SCOPE_ASSIGNMENTS, batch_distractor_schema_valid, build_three_scope_prompt,
    classify_joint_necessity, compile_arc_lineage_views,
    run_scoped_questioner_batch, strip_scope_field, validate_scope_item,
)
from scripts.current_method.restart_n5_spell.lineage_relation_miner import (
    RELATION_MINER_JSON_SCHEMA, build_relation_miner_prompt,
    relation_schema_sha256, validate_relation_candidate,
)
from scripts.current_method.restart_n5_spell.lineage_qa_writer import (
    QA_WRITER_JSON_SCHEMA, build_qa_writer_prompt,
    validate_question_answer_object,
)
from scripts.current_method.restart_n5_spell.lineage_distractor_writer import (
    DISTRACTOR_WRITER_JSON_SCHEMA, build_distractor_writer_prompt,
    validate_attempt_feedback, validate_distractor_object,
)
from scripts.current_method.restart_n5_spell.lineage_three_stage_canary import (
    ApiBudget, build_relation_diagnostic_contexts,
    classify_joint_necessity_v1, deterministic_answer_placement,
    materialize_three_stage_mcq, run_single_object_task,
    should_stop_after_first_retain_frontier,
)
from scripts.current_method.restart_n5_spell.lineage_full_dialogue_prompts import (
    FULL_DIALOGUE_INPUT_VERSION,
    build_full_dialogue_distractor_writer_prompt,
    build_full_dialogue_qa_writer_prompt,
    build_full_dialogue_relation_miner_prompt,
    full_dialogue_distractor_prompt_sha256,
    full_dialogue_qa_prompt_sha256, full_dialogue_relation_prompt_sha256,
)
from scripts.current_method.restart_n5_spell.run_lineage_three_stage_canary import (
    DEFAULT_SOURCE_ROOT as THREE_STAGE_SOURCE_ROOT,
    _attest_source as attest_three_stage_source,
    _request_view as three_stage_request_view,
    _route_conclusion as three_stage_route_conclusion,
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

    def test_solver_format_retry_instructions_are_format_only(self):
        from scripts.current_method.restart_n5_spell.prompts import (
            SOLVER_PARSE_RETRY_SUFFIX,
            SOLVER_TRAILING_FORMAT_INSTRUCTION,
        )
        self.assertIn("<box>A</box>", SOLVER_TRAILING_FORMAT_INSTRUCTION)
        self.assertIn("Do not include the option text",
                      SOLVER_TRAILING_FORMAT_INSTRUCTION)
        self.assertIn("same unchanged question", SOLVER_PARSE_RETRY_SUFFIX)
        self.assertNotIn("previous response:",
                         SOLVER_PARSE_RETRY_SUFFIX.casefold())


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

    def test_no_context_permutation_bias_diagnosis(self):
        semantic = classify_no_context_permutation_bias([
            {"order_id":"original", "gold_letter":"A",
             "valid_answer_letters":["A"] * 3, "valid_rewards":[1] * 3},
            {"order_id":"perm1", "gold_letter":"B",
             "valid_answer_letters":["B"] * 3, "valid_rewards":[1] * 3},
            {"order_id":"perm2", "gold_letter":"C",
             "valid_answer_letters":["C"] * 3, "valid_rewards":[1] * 3}])
        self.assertEqual(semantic["decision"], "bad_ground_semantic_leakage")
        position = classify_no_context_permutation_bias([
            {"order_id":"original", "gold_letter":"A",
             "valid_answer_letters":["A"] * 3, "valid_rewards":[1] * 3},
            {"order_id":"perm1", "gold_letter":"B",
             "valid_answer_letters":["A"] * 3, "valid_rewards":[0] * 3},
            {"order_id":"perm2", "gold_letter":"C",
             "valid_answer_letters":["A"] * 3, "valid_rewards":[0] * 3}])
        self.assertEqual(position["decision"], "bad_ground_position_bias")
        mixed = classify_no_context_permutation_bias([
            {"order_id":"original", "gold_letter":"A",
             "valid_answer_letters":["A"] * 3, "valid_rewards":[1] * 3},
            {"order_id":"perm1", "gold_letter":"B",
             "valid_answer_letters":["B", "A", "B"], "valid_rewards":[1,0,1]},
            {"order_id":"perm2", "gold_letter":"C",
             "valid_answer_letters":["A", "C", "D"], "valid_rewards":[0,1,0]}])
        self.assertEqual(mixed["decision"], "bad_ground_mixed_bias")
        incomplete = [dict(row) for row in mixed["orders"]]
        incomplete[1]["valid_rewards"] = [1, 0]
        self.assertEqual(classify_no_context_permutation_bias(incomplete)["decision"],
                         "bad_ground_permutation_inconclusive")


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


class EvidenceViewAblationTests(unittest.TestCase):
    @staticmethod
    def turns(prefix, count, *, child=False):
        return [{"turn_id":f"{prefix}{index:02d}",
                 "session_id":f"session_{index:02d}" if child else f"D{index}",
                 "speaker":"A" if index % 2 else "B",
                 "text":f"immutable {prefix} text {index}"}
                for index in range(1, count + 1)]

    @staticmethod
    def outline():
        return GraphExpansionTests.outline()

    def test_child_only_hides_parent_text_and_spans(self):
        parent = self.turns("P", 6)
        child = self.turns("C", 4, child=True)
        view = build_questioner_evidence_view(mode=CHILD_ONLY_VIEW,
            parent_turns=parent, child_turns=child)
        self.assertEqual(view["selected_parent_turn_count"], 0)
        self.assertEqual(view["selected_child_turn_count"], 4)
        self.assertEqual(view["combined_dialogue"], child)
        catalog = build_span_catalog(view["combined_dialogue"])
        self.assertTrue(all(span["span_id"].startswith("C")
                            for span in catalog["spans"]))
        self.assertTrue(all("immutable P" not in span["exact_text"]
                            for span in catalog["spans"]))

    def test_balanced_view_is_exact_and_preserves_source_order(self):
        parent = self.turns("P", 6)
        child = self.turns("C", 4, child=True)
        view = build_questioner_evidence_view(mode=BALANCED_VIEW,
            parent_turns=parent, child_turns=child)
        self.assertEqual(view["selection"]["algorithm"], BALANCED_PARENT_SELECTION)
        self.assertEqual(view["selection"]["parent_indices"], [0, 1, 3, 5])
        self.assertEqual(view["selected_parent_turn_count"], 4)
        self.assertEqual(view["selected_child_turn_count"], 4)
        self.assertEqual([turn["turn_id"] for turn in view["combined_dialogue"]],
                         ["P01", "P02", "P04", "P06",
                          "C01", "C02", "C03", "C04"])
        without_hash = {key:value for key,value in view.items() if key != "view_sha256"}
        self.assertEqual(view["view_sha256"], evidence_view_sha256(without_hash))

    def test_uniform_selection_is_endpoint_inclusive_and_unique(self):
        indices = uniform_endpoint_indices(237, 134)
        self.assertEqual(len(indices), 134)
        self.assertEqual(len(set(indices)), 134)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 236)
        self.assertEqual(tuple(sorted(indices)), indices)

    def test_selected_evidence_diagnostics_are_soft_counts(self):
        parent = self.turns("P", 2)
        child = self.turns("C", 10, child=True)
        catalog = build_span_catalog(parent + child)
        contracts = compile_graph_scene_contracts(self.outline())
        diagnostics = selected_evidence_diagnostics(
            span_ids=["P01#S0001", "C03#S0001", "C07#S0001"],
            catalog=catalog, scene_contracts=contracts)
        self.assertEqual(diagnostics["selected_parent_span_count"], 1)
        self.assertEqual(diagnostics["selected_child_span_count"], 2)
        self.assertEqual(diagnostics["selected_session_count"], 3)
        self.assertEqual(diagnostics["selected_turn_count"], 3)
        self.assertEqual(diagnostics["selected_event_arc_ids"], ["E1", "E2", "E3"])
        self.assertEqual(diagnostics["selected_event_arc_count"], 3)
        self.assertEqual(diagnostics["selected_event_stage_count"], 5)

    def test_invalid_view_or_claimed_span_is_rejected_deterministically(self):
        parent = self.turns("P", 2)
        child = self.turns("C", 2, child=True)
        with self.assertRaisesRegex(ValueError, "questioner_evidence_view_mode_invalid"):
            build_questioner_evidence_view(mode="full", parent_turns=parent,
                                           child_turns=child)
        with self.assertRaisesRegex(ValueError, "evidence_view_selected_span_missing"):
            selected_evidence_diagnostics(span_ids=["missing"],
                catalog=build_span_catalog(parent + child),
                scene_contracts=compile_graph_scene_contracts(self.outline()))


class ThreeScopeCanaryTests(unittest.TestCase):
    @staticmethod
    def fixture():
        outline = copy.deepcopy(GraphExpansionTests.outline())
        outline["parent_anchors"] = [
            {"anchor_id":f"P{index}", "turn_id":f"P{index}",
             "text":f"parent anchor {index}"} for index in range(1, 5)]
        for index, arc in enumerate(outline["planned_event_arcs"], start=1):
            arc["parent_anchor_ids"] = [f"P{index}"]
        parent = [{"turn_id":f"P{index}", "session_id":f"D{index}",
                   "speaker":"A" if index % 2 else "B",
                   "text":f"immutable parent anchor {index}"}
                  for index in range(1, 5)]
        contracts = compile_graph_scene_contracts(outline)
        child = []
        for contract in contracts:
            session = int(contract["scene_index"])
            child.append({"turn_id":f"C{session:02d}:01",
                          "session_id":f"session_{session:02d}",
                          "speaker":"A",
                          "text":f"immutable child stage scene {session}"})
        views = compile_arc_lineage_views(outline=outline, parent_turns=parent,
            child_turns=child, scene_contracts=contracts)
        return outline, contracts, views

    @staticmethod
    def item(scope, spans, *, qa_id="q"):
        return {"qa_id":qa_id, "qa_scope":scope, "question":"Which state followed?",
            "correct_answer_text":"state alpha",
            "distractors":["state beta", "state gamma", "state delta"],
            "answer_support_span_ids":list(spans),
            "reasoning_steps":["Compare the selected frozen facts."]}

    def test_schema_enforces_exactly_three_distractors_and_no_extras(self):
        item_schema = SCOPED_BATCH_JSON_SCHEMA["items"]
        distractors = item_schema["properties"]["distractors"]
        self.assertEqual((distractors["minItems"], distractors["maxItems"]), (3, 3))
        self.assertFalse(item_schema["additionalProperties"])
        self.assertTrue(batch_distractor_schema_valid([
            self.item(scope, ["span"]) for scope in SCOPE_ASSIGNMENTS]))
        bad = [self.item(scope, ["span"]) for scope in SCOPE_ASSIGNMENTS]
        bad[1]["distractors"].append("state epsilon")
        self.assertFalse(batch_distractor_schema_valid(bad))

    def test_four_lineage_views_cover_parent_anchor_and_all_arc_stages(self):
        _, _, views = self.fixture()
        self.assertEqual([view["event_arc_id"] for view in views],
                         ["E1", "E2", "E3", "E4"])
        for index, view in enumerate(views, start=1):
            self.assertGreater(view["selected_parent_turn_count"], 0)
            self.assertGreater(view["selected_child_turn_count"], 0)
            self.assertEqual(view["covered_event_stage_ids"],
                [f"E{index}:stage_{stage:02d}" for stage in range(1, 5)])
            self.assertEqual(view["parent_anchor_turn_ids"], [f"P{index}"])
            ids = [turn["turn_id"] for turn in view["combined_dialogue"]]
            self.assertEqual(ids[:view["selected_parent_turn_count"]],
                [turn["turn_id"] for turn in view["combined_dialogue"]
                 if turn["turn_id"].startswith("P")])

    def test_scope_gate_is_separate_and_deterministic(self):
        _, _, views = self.fixture(); view = views[0]
        parent = view["selected_parent_span_ids"][0]
        child = view["selected_child_span_ids"][0]
        rows = [self.item(PARENT_LOCAL, [parent]),
                self.item(CHILD_LOCAL, [child]),
                self.item(PARENT_CHILD_JOINT, [parent, child])]
        for index, row in enumerate(rows):
            self.assertEqual(validate_scope_item(row, item_index=index,
                lineage_view=view)["decision"], "qa_scope_followed")
        wrong = self.item(PARENT_CHILD_JOINT, [child])
        result = validate_scope_item(wrong, item_index=2, lineage_view=view)
        self.assertEqual(result["decision"], "qa_scope_not_followed")
        self.assertIn("joint_support_missing_partition", result["reasons"])
        self.assertNotIn("qa_scope", strip_scope_field(rows[0]))

    def test_prompt_assigns_three_scopes_without_fixed_hop_contract(self):
        _, _, views = self.fixture()
        prompt = build_three_scope_prompt(lineage_view=views[0],
            assigned_gold_letters=("A", "B", "C"), solver_feedback=[])
        for scope in SCOPE_ASSIGNMENTS:
            self.assertIn(scope, prompt)
        self.assertIn("exactly three distinct distractors", prompt)
        self.assertIn("removing either partition must make the answer underdetermined", prompt)
        self.assertIn("Do not impose a fixed number of spans", prompt)

    class Backend:
        model_identifier = "three-scope-fixture"
        def __init__(self, payloads):
            self.payloads = iter(payloads); self.requests = []
        def generate_batch(self, request):
            self.requests.append(request)
            return BackendResponse(next(self.payloads), self.model_identifier,
                                   {"finish_reason":"stop"})

    def run_batch(self, payloads):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        atomic_json(root / "config/run_manifest.json", {"run_id":"three-scope-test"})
        _, _, views = self.fixture(); view = views[0]
        backend = self.Backend(payloads)
        ledger = RequestLedger(root, sha256_file(root / "config/run_manifest.json"), "mock")
        result = run_scoped_questioner_batch(ledger=ledger, backend=backend,
            semantic_prompt="semantic three-scope prompt", event_arc_id="E1",
            assigned_gold_letters=("A", "B", "C"), lineage_view=view,
            prompt_protocol="three-scope-test", backend_identity={"backend":"mock"})
        return result, backend, ledger, view

    def test_schema_only_retry_regenerates_complete_batch_once(self):
        _, _, views = self.fixture(); view = views[0]
        parent = view["selected_parent_span_ids"][0]
        child = view["selected_child_span_ids"][0]
        valid = [self.item(PARENT_LOCAL, [parent], qa_id="q1"),
                 self.item(CHILD_LOCAL, [child], qa_id="q2"),
                 self.item(PARENT_CHILD_JOINT, [parent, child], qa_id="q3")]
        invalid = copy.deepcopy(valid); invalid[0]["distractors"].append("fourth")
        result, backend, ledger, _ = self.run_batch(
            [json.dumps(invalid), json.dumps(valid)])
        self.assertEqual(result["physical_questioner_calls"], 2)
        self.assertTrue(result["schema_retry_used"])
        self.assertEqual(result["final_decision"],
                         "continue_to_scope_and_minimal_admission")
        self.assertNotIn("fourth", backend.requests[1].prompt)
        retry = ledger.read("questioner-E1-a01")
        self.assertEqual(retry["request_metadata"]["retry_reason"],
                         "distractor_count_invalid")
        self.assertEqual(retry["request_metadata"]["retry_mode"],
                         "full_batch_regeneration")

        exhausted, exhausted_backend, _, _ = self.run_batch(
            [json.dumps(invalid), json.dumps(invalid)])
        self.assertEqual(len(exhausted_backend.requests), 2)
        self.assertEqual(exhausted["final_decision"],
                         "questioner_schema_invalid_exhausted")

    def test_joint_necessity_classification(self):
        joint = classify_joint_necessity(no_context_rewards=[0, 0, 1],
            parent_only_rewards=[0, 1, 0], child_only_rewards=[1, 0, 0],
            full_context_rewards=[1, 1, 1, 1, 1])
        self.assertEqual(joint["decision"], "lineage_jointly_required")
        self.assertEqual(classify_joint_necessity(no_context_rewards=[0, 0, 0],
            parent_only_rewards=[1, 1, 1], child_only_rewards=[0, 0, 0],
            full_context_rewards=[1] * 5)["decision"], "parent_only_sufficient")
        self.assertEqual(classify_joint_necessity(no_context_rewards=[0, 0, 0],
            parent_only_rewards=[0, 0, 0], child_only_rewards=[1, 1, 1],
            full_context_rewards=[1] * 5)["decision"], "child_only_sufficient")


class LineageThreeStageCanaryTests(unittest.TestCase):
    @staticmethod
    def view():
        turns = [
            {"turn_id":"P1", "session_id":"parent", "speaker":"A",
             "text":"A planned to submit the original blue brief."},
            {"turn_id":"P2", "session_id":"parent", "speaker":"B",
             "text":"B expected the original review on Monday."},
            {"turn_id":"C1", "session_id":"session_01", "speaker":"A",
             "text":"A revised the brief to green after the review."},
            {"turn_id":"C2", "session_id":"session_02", "speaker":"B",
             "text":"B approved the revised green brief on Friday."},
        ]
        catalog = build_span_catalog(turns)
        body = {"event_arc_id":"E1", "covered_event_stage_ids":[
            "E1:stage_01", "E1:stage_02"],
            "selected_parent_span_ids":["P1#S0001", "P2#S0001"],
            "selected_child_span_ids":["C1#S0001", "C2#S0001"],
            "combined_dialogue":turns, "span_catalog":catalog}
        body["view_sha256"] = sha256_bytes(canonical_json(body).encode())
        return body

    @classmethod
    def relation_raw(cls, relation_id="E1_R1"):
        return {"relation_id":relation_id,
            "parent_condition_span_ids":["P1#S0001"],
            "child_consequence_span_ids":["C1#S0001"],
            "relation_type":"state_revision",
            "relation_description":"The planned blue brief became green after review.",
            "joint_dependency_explanation":"Parent establishes blue; child establishes revision."}

    @classmethod
    def accepted_relation(cls):
        result = validate_relation_candidate(cls.relation_raw(), arc_id="E1",
            candidate_index=0, lineage_view=cls.view())
        return result["relation_bundle"], result["relation_bundle_sha256"]

    @classmethod
    def qa_raw(cls):
        return {"qa_id":"E1_R1_QA",
            "question":"How did the brief differ after the review?",
            "correct_answer_text":"It changed from blue to green.",
            "answer_support_span_ids":["P1#S0001", "C1#S0001"],
            "reasoning_steps":["Read the initial plan.", "Compare the revision."]}

    def test_relation_schema_and_canonical_sha_are_strict_and_stable(self):
        self.assertFalse(RELATION_MINER_JSON_SCHEMA["additionalProperties"])
        self.assertEqual(RELATION_MINER_JSON_SCHEMA["properties"][
            "parent_condition_span_ids"]["maxItems"], 4)
        self.assertEqual(RELATION_MINER_JSON_SCHEMA["properties"][
            "child_consequence_span_ids"]["maxItems"], 5)
        first = validate_relation_candidate(self.relation_raw(), arc_id="E1",
            candidate_index=0, lineage_view=self.view())
        reordered = self.relation_raw()
        second = validate_relation_candidate(reordered, arc_id="E1",
            candidate_index=0, lineage_view=self.view())
        self.assertEqual(first["decision"], "relation_accept")
        self.assertEqual(first["relation_bundle_sha256"],
                         second["relation_bundle_sha256"])
        self.assertEqual(len(relation_schema_sha256()), 64)
        extra = {**self.relation_raw(), "question":"forbidden"}
        self.assertEqual(validate_relation_candidate(extra, arc_id="E1",
            candidate_index=0, lineage_view=self.view())["decision"],
            "relation_schema_invalid")

    def test_relation_partition_and_lineage_span_gates(self):
        parent_in_child = self.relation_raw()
        parent_in_child["child_consequence_span_ids"] = ["P2#S0001"]
        self.assertEqual(validate_relation_candidate(parent_in_child, arc_id="E1",
            candidate_index=0, lineage_view=self.view())["decision"],
            "relation_scope_invalid")
        child_in_parent = self.relation_raw()
        child_in_parent["parent_condition_span_ids"] = ["C2#S0001"]
        self.assertEqual(validate_relation_candidate(child_in_parent, arc_id="E1",
            candidate_index=0, lineage_view=self.view())["decision"],
            "relation_scope_invalid")
        outside = self.relation_raw()
        outside["child_consequence_span_ids"] = ["OUTSIDE#S0001"]
        result = validate_relation_candidate(outside, arc_id="E1",
            candidate_index=0, lineage_view=self.view())
        self.assertIn("relation_span_invalid", result["reasons"])

    def test_duplicate_relation_support_set_is_rejected(self):
        first, _ = self.accepted_relation()
        duplicate = self.relation_raw("E1_R2")
        result = validate_relation_candidate(duplicate, arc_id="E1",
            candidate_index=1, lineage_view=self.view(),
            prior_support_sets=[first["canonical_support_span_ids"]])
        self.assertEqual(result["decision"], "duplicate_relation_candidate")

    def test_qa_support_must_equal_relation_union_and_drift_is_rejected(self):
        relation, relation_sha = self.accepted_relation()
        accepted = validate_question_answer_object(self.qa_raw(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=self.view())
        self.assertEqual(accepted["decision"], "qa_accept")
        self.assertFalse(QA_WRITER_JSON_SCHEMA["additionalProperties"])
        missing = self.qa_raw(); missing["answer_support_span_ids"] = ["P1#S0001"]
        self.assertIn("qa_support_mismatch", validate_question_answer_object(missing,
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=self.view())["reasons"])
        drift = self.qa_raw(); drift["answer_support_span_ids"] = [
            "P1#S0001", "C1#S0001", "C2#S0001"]
        result = validate_question_answer_object(drift, relation_bundle=relation,
            relation_bundle_sha256=relation_sha, lineage_view=self.view())
        self.assertIn("qa_relation_drift", result["reasons"])

    def test_distractor_schema_never_trims_four_and_rejects_duplicates(self):
        self.assertEqual(DISTRACTOR_WRITER_JSON_SCHEMA["properties"][
            "distractors"]["minItems"], 3)
        self.assertEqual(DISTRACTOR_WRITER_JSON_SCHEMA["properties"][
            "distractors"]["maxItems"], 3)
        valid = {"distractors":["blue to red", "green to blue", "red to green"]}
        self.assertEqual(validate_distractor_object(valid,
            correct_answer_text="blue to green")["decision"], "distractor_accept")
        four = {"distractors":[*valid["distractors"], "fourth"]}
        result = validate_distractor_object(four, correct_answer_text="blue to green")
        self.assertEqual(result["decision"], "distractor_schema_invalid")
        self.assertIsNone(result["distractors"])
        duplicate = {"distractors":["same", " same ", "different"]}
        self.assertIn("distractor_duplicate", validate_distractor_object(duplicate,
            correct_answer_text="answer")["reasons"])

    def test_deterministic_answer_placement_covers_letters_without_content(self):
        attempt0 = [deterministic_answer_placement(arc, index, 0)["target_letter"]
                    for arc in ("E1", "E4") for index in (0, 1)]
        self.assertEqual(attempt0, ["A", "B", "C", "D"])
        for arc in ("E1", "E4"):
            for index in (0, 1):
                first = deterministic_answer_placement(arc, index, 1)
                second = deterministic_answer_placement(arc, index, 1)
                self.assertEqual(first, second)
                self.assertFalse(first["content_dependent"])

    def test_attempt1_accepts_only_fixed_structured_feedback(self):
        feedback = {"previous_distractors":["one", "two", "three"],
            "previous_final_decision":"bad_ground",
            "previous_grounding_diagnosis":"bad_ground_semantic_leakage"}
        self.assertEqual(validate_attempt_feedback(1, feedback), feedback)
        with self.assertRaisesRegex(ValueError, "feedback_label_invalid"):
            validate_attempt_feedback(1, {**feedback,
                "previous_grounding_diagnosis":"new free-form diagnosis"})
        relation, relation_sha = self.accepted_relation()
        qa_result = validate_question_answer_object(self.qa_raw(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=self.view())
        qa = qa_result["question_answer_object"]
        prompt = build_distractor_writer_prompt(lineage_view=self.view(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            question_answer_object=qa,
            question_answer_sha256=qa_result["question_answer_object_sha256"],
            attempt_index=1, structured_feedback=feedback)
        self.assertIn('"previous_grounding_diagnosis": "bad_ground_semantic_leakage"', prompt)

    class Backend:
        model_identifier = "three-stage-fixture"
        def __init__(self, responses):
            self.responses = iter(responses); self.requests = []
            config = type("Config", (), {"model":"fixture", "base_url":"fixture",
                "temperature":0.25})()
            self.client = type("Client", (), {"config":config})()
            self.max_tokens = 128
        def generate_batch(self, request):
            self.requests.append(request)
            return BackendResponse(next(self.responses), self.model_identifier,
                                   {"usage":{"prompt_tokens":10,
                                              "completion_tokens":2,
                                              "total_tokens":12}})

    def test_each_logical_request_has_at_most_one_format_retry(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        atomic_json(root / "config/run_manifest.json", {"run_id":"three-stage"})
        ledger = RequestLedger(root, sha256_file(root / "config/run_manifest.json"),
                               "mock")
        backend = self.Backend(["{", json.dumps(self.relation_raw())])
        result = run_single_object_task(run_root=root, ledger=ledger,
            backend=backend, budget=ApiBudget(), stage="relation_miner",
            logical_request_id="relation-E1_R1", semantic_prompt="semantic secret",
            lineage_view=self.view(), validator=lambda raw:validate_relation_candidate(
                raw, arc_id="E1", candidate_index=0, lineage_view=self.view()),
            schema_retry_decisions=("relation_schema_invalid",))
        self.assertEqual(result["physical_request_count"], 2)
        self.assertTrue(result["accepted"])
        self.assertNotIn("{", backend.requests[1].prompt.split("semantic secret", 1)[-1])
        self.assertEqual(ledger.read("relation-E1_R1-a01")["request_metadata"][
            "retry_mode"], "full_object_regeneration")

    def test_global_api_budget_and_retain_frontier_stop(self):
        budget = ApiBudget()
        for _ in range(4): budget.reserve("relation_miner", is_retry=False)
        with self.assertRaisesRegex(RuntimeError, "relation_miner_normal_budget_exhausted"):
            budget.reserve("relation_miner", is_retry=False)
        budget = ApiBudget()
        for _ in range(8): budget.reserve("relation_miner", is_retry=True)
        with self.assertRaisesRegex(RuntimeError, "global_format_retry_budget_exhausted"):
            budget.reserve("qa_writer", is_retry=True)
        self.assertTrue(should_stop_after_first_retain_frontier("retain_frontier"))
        self.assertFalse(should_stop_after_first_retain_frontier("reject_easy"))

    def test_diagnostic_contexts_and_joint_necessity_thresholds(self):
        contexts = build_relation_diagnostic_contexts(lineage_view=self.view())
        self.assertEqual([turn["turn_id"] for turn in contexts["parent_only"]],
                         ["P1", "P2"])
        self.assertEqual([turn["turn_id"] for turn in contexts["child_only"]],
                         ["C1", "C2"])
        self.assertEqual(len(contexts["full_lineage"]), 4)
        cases = [
            ([1,1,0],[0,0,0],[1,1,1],"parent_only_sufficient"),
            ([0,0,0],[1,0,1],[1,1,1],"child_only_sufficient"),
            ([1,0,0],[0,1,0],[1,1,0],"lineage_jointly_required"),
            ([0,0,0],[0,0,0],[1,0,0],"lineage_not_answerable"),
        ]
        for parent, child, lineage, expected in cases:
            self.assertEqual(classify_joint_necessity_v1(parent_rewards=parent,
                child_rewards=child, lineage_rewards=lineage)["decision"], expected)

    def test_adaptive_bad_ground_still_runs_permutation_diagnosis(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        relation, relation_sha = self.accepted_relation()
        qa_result = validate_question_answer_object(self.qa_raw(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=self.view())
        qa = materialize_three_stage_mcq(
            question_answer=qa_result["question_answer_object"],
            distractors=["blue to red", "green to blue", "red to green"],
            target_letter="A", span_catalog=self.view()["span_catalog"])[
                "materialized_qa"]
        calls = []
        def fake_rollouts(candidate, mode, order_id, requested, options, slot_start=0):
            calls.append(order_id)
            gold = next(letter for letter, text in options.items()
                        if text == candidate["canonical_answer_text"])
            return {"requested_valid_rollouts":3, "obtained_valid_rollouts":3,
                "verified_rewards":[1, 0, 1], "raw_generations":3,
                "parse_invalid_generations":0, "retry_generations":0,
                "exhausted_target_slots":0,
                "rollouts":[{"parsed_letter":gold, "local_reward":reward}
                            for reward in (1, 0, 1)]}
        orchestrator._rollouts = fake_rollouts
        initial = fake_rollouts(qa, "no_context", "source", 3, qa["options"])
        result = orchestrator.diagnose_adaptive_bad_ground_permutations(qa, initial)
        self.assertEqual(calls[-2:], ["perm1", "perm2"])
        self.assertIn(result["diagnostic_decision"], {
            "bad_ground_mixed_bias", "bad_ground_semantic_leakage",
            "bad_ground_position_bias"})

    def test_materialization_reuses_minimal_gate(self):
        relation, relation_sha = self.accepted_relation()
        qa_result = validate_question_answer_object(self.qa_raw(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=self.view())
        result = materialize_three_stage_mcq(
            question_answer=qa_result["question_answer_object"],
            distractors=["blue to red", "green to blue", "red to green"],
            target_letter="D", span_catalog=self.view()["span_catalog"])
        self.assertEqual(result["minimal_decision"], "accept")
        self.assertEqual(result["materialized_qa"]["reference_answer_letter"], "D")
        self.assertEqual(result["materialized_qa"]["reference_answer_text"],
                         self.qa_raw()["correct_answer_text"])

    def test_fixed_route_conclusions(self):
        supported = {"retain_frontier":1, "materialized_mcqs":4}
        self.assertEqual(three_stage_route_conclusion(supported, []),
                         "three_stage_fixed_api_challenger_supported")
        no_joint = {"retain_frontier":0, "materialized_mcqs":4}
        self.assertEqual(three_stage_route_conclusion(no_joint, []),
                         "fixed_api_joint_question_generation_not_supported")
        inconclusive = {"retain_frontier":0, "materialized_mcqs":3}
        self.assertEqual(three_stage_route_conclusion(inconclusive, []),
                         "three_stage_canary_inconclusive")

    def test_authoritative_source_attestation_is_read_only(self):
        manifest = THREE_STAGE_SOURCE_ROOT / "config/run_manifest.json"
        before = sha256_file(manifest)
        source = attest_three_stage_source(THREE_STAGE_SOURCE_ROOT)
        self.assertEqual(source["public"]["parent_turn_count"], 237)
        self.assertEqual(source["public"]["child_turn_count"], 134)
        self.assertEqual(before, sha256_file(manifest))

    def test_full_dialogue_prompt_uses_ordered_verbatim_turns_not_outline(self):
        view = self.view()
        full_dialogue = [
            *view["combined_dialogue"],
            {"turn_id":"X1", "session_id":"unrelated", "speaker":"C",
             "text":"This complete-dialogue turn is outside the E1 lineage."}]
        relation_prompt = build_full_dialogue_relation_miner_prompt(
            arc_id="E1", candidate_index=0, lineage_view=view,
            full_dialogue=full_dialogue, span_stage_membership={},
            prior_relation=None)
        self.assertIn(FULL_DIALOGUE_INPUT_VERSION, relation_prompt)
        self.assertIn('"full_dialogue_turn_count":5', relation_prompt)
        self.assertIn('"turn_id":"X1"', relation_prompt)
        self.assertIn("This complete-dialogue turn is outside", relation_prompt)
        self.assertNotIn("event_arc_summary", relation_prompt)
        self.assertNotIn("expansion_outline", relation_prompt)
        self.assertNotIn("graph_gold", relation_prompt)

    def test_full_dialogue_qa_and_distractor_prompts_preserve_frozen_relation(self):
        view = self.view(); relation, relation_sha = self.accepted_relation()
        qa_result = validate_question_answer_object(self.qa_raw(),
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            lineage_view=view)
        qa = qa_result["question_answer_object"]
        full_dialogue = [*view["combined_dialogue"],
            {"turn_id":"X1", "session_id":"other", "speaker":"C",
             "text":"A fifth verbatim turn."}]
        qa_prompt = build_full_dialogue_qa_writer_prompt(lineage_view=view,
            full_dialogue=full_dialogue, relation_bundle=relation,
            relation_bundle_sha256=relation_sha)
        distractor_prompt = build_full_dialogue_distractor_writer_prompt(
            lineage_view=view, full_dialogue=full_dialogue,
            relation_bundle=relation, relation_bundle_sha256=relation_sha,
            question_answer_object=qa,
            question_answer_sha256=qa_result["question_answer_object_sha256"],
            attempt_index=0, structured_feedback=None)
        for prompt in (qa_prompt, distractor_prompt):
            self.assertIn('"full_dialogue_turn_count":5', prompt)
            self.assertIn(relation_sha, prompt)
            self.assertIn('"turn_id":"X1"', prompt)
        self.assertEqual(len(full_dialogue_relation_prompt_sha256()), 64)
        self.assertEqual(len(full_dialogue_qa_prompt_sha256()), 64)
        self.assertEqual(len(full_dialogue_distractor_prompt_sha256()), 64)

    def test_full_dialogue_request_view_changes_visibility_not_support_catalog(self):
        view = self.view()
        full = {"combined_dialogue":[*view["combined_dialogue"],
            {"turn_id":"X1", "session_id":"other", "speaker":"C",
             "text":"Outside lineage."}]}
        request_view = three_stage_request_view(
            view, full, FULL_DIALOGUE_INPUT_VERSION)
        self.assertEqual(len(request_view["combined_dialogue"]), 5)
        self.assertEqual(request_view["span_catalog"], view["span_catalog"])
        self.assertEqual(request_view["selected_parent_span_ids"],
                         view["selected_parent_span_ids"])


class GraphExpansionTests(unittest.TestCase):
    @staticmethod
    def outline():
        arcs = []
        for arc_index in range(1, 5):
            arcs.append({"event_id":f"E{arc_index}",
                "summary":f"event arc {arc_index} progression",
                "stages":[f"arc{arc_index} stage{stage}" for stage in range(1, 5)]})
        scenes = []
        for scene_index in range(1, 11):
            scenes.append({"scene_id":f"S{scene_index}",
                "target_child_tokens":420, "acceptable_token_range":[360, 480],
                "target_turns":[8, 14], "participants":["A", "B"],
                "purpose":f"scene {scene_index} purpose",
                "required_parent_anchor_ids":["P1"] if scene_index == 1 else [],
                "handoff_requirement":"continue the current plan"})
        return {"target":{"child_token_growth":4200, "scene_count":10,
                    "tokens_per_scene":420, "minimum_accepted_child_tokens":4000,
                    "maximum_new_named_characters":2},
            "source_parent":{"core_speakers":["A", "B"]},
            "allowed_new_characters":[
                {"name":"C", "introduction_scene":"S2"},
                {"name":"D", "introduction_scene":"S5"}],
            "allowed_locations":["first", "second", "third"],
            "time_window":"five weeks", "planned_event_arcs":arcs,
            "scene_plan":scenes, "parent_anchors":[{"anchor_id":"P1", "text":"anchor"}],
            "forbidden_conflicts":[]}

    def test_scene_compiler_covers_full_graph(self):
        contracts = compile_graph_scene_contracts(self.outline())
        self.assertEqual(len(contracts), 10)
        self.assertEqual({stage for contract in contracts
                          for stage in contract["assigned_event_stage_ids"]},
                         {f"E{arc}:stage_{stage:02d}" for arc in range(1, 5)
                          for stage in range(1, 5)})
        self.assertEqual(len({stage for contract in contracts
                              for stage in contract["assigned_event_stage_ids"]}), 16)
        self.assertIn("C", contracts[1]["required_participants"])
        self.assertIn("D", contracts[4]["required_participants"])
        self.assertEqual({contract["session_location"] for contract in contracts},
                         {"first", "second", "third"})
        self.assertTrue(all(contract["target_dialogue_token_range"] == [360, 480]
                            for contract in contracts))
        self.assertTrue(all(contract["hard_dialogue_token_range"] == [220, 650]
                            for contract in contracts))
        self.assertEqual(contracts[0]["new_unresolved_hook_policy"], "exactly_one")
        self.assertTrue(all(contract["new_unresolved_hook_policy"] == "forbidden"
                            for contract in contracts[1:]))

    def test_scene_output_contract_is_hard_gate(self):
        contract = compile_graph_scene_contracts(self.outline())[1]
        turns = [{"speaker":speaker, "text":"substantive scene text"}
                 for speaker in ("A", "B", "C", "A", "B", "C", "A", "B")]
        payload = {"dialogue_turns":turns, "used_parent_anchor_ids":[],
            "advanced_event_arc_ids":contract["assigned_event_arc_ids"],
            "new_unresolved_hooks":[]}
        self.assertEqual(validate_graph_scene_output(payload, contract, 420), [])
        missing = copy.deepcopy(payload); missing["dialogue_turns"] = [
            {"speaker":"A", "text":"x"} for _ in range(8)]
        self.assertIn("graph_required_participant_missing",
                      validate_graph_scene_output(missing, contract, 420))
        wrong_arc = copy.deepcopy(payload); wrong_arc["advanced_event_arc_ids"] = ["E4"]
        self.assertIn("graph_assigned_event_arcs_incomplete",
                      validate_graph_scene_output(wrong_arc, contract, 420))
        self.assertIn("graph_scene_turn_count_out_of_range",
                      validate_graph_scene_output({**payload, "dialogue_turns":turns[:2]},
                                                  contract, 420))
        self.assertIn("graph_scene_token_count_out_of_range",
                      validate_graph_scene_output(payload, contract, 200))
        unexpected_hook = copy.deepcopy(payload)
        unexpected_hook["new_unresolved_hooks"] = [{"description":"duplicate"}]
        self.assertIn("graph_scene_hook_policy_violation",
                      validate_graph_scene_output(unexpected_hook, contract, 420))

    def test_graph_realizer_managed_ids_are_removed_without_text_changes(self):
        raw = {"dialogue_turns": [
            {"speaker":"A", "text":"First.", "turn_id":"C0001"},
            {"speaker":"B", "text":"Second.", "turn_id":"C0002",
             "session_id":"session_01"}],
            "session_id":"model-session"}
        normalized, report = normalize_graph_realizer_managed_ids(raw)
        self.assertEqual({"speaker":"A", "text":"First."},
                         normalized["dialogue_turns"][0])
        self.assertEqual({"speaker":"B", "text":"Second."},
                         normalized["dialogue_turns"][1])
        self.assertNotIn("session_id", normalized)
        self.assertEqual("normalized_managed_ids", report["decision"])
        self.assertFalse(report["semantic_text_modified"])
        self.assertIn("turn_id", raw["dialogue_turns"][0])

    def test_graph_realizer_normalization_preserves_unknown_extra_fields(self):
        raw = {"dialogue_turns": [{"speaker":"A", "text":"First.",
                                    "turn_id":"C0001", "emotion":"calm"}]}
        normalized, _ = normalize_graph_realizer_managed_ids(raw)
        self.assertNotIn("turn_id", normalized["dialogue_turns"][0])
        self.assertEqual("calm", normalized["dialogue_turns"][0]["emotion"])
        self.assertIn("bad_dialogue_turn_schema",
                      validate_realizer_output_schema({
                          **{key:[] for key in REALIZER_OUTPUT_SCHEMA
                             if key != "dialogue_turns"},
                          "dialogue_turns":normalized["dialogue_turns"]}))

    def test_completion_requires_scenes_stages_and_minimum_tokens(self):
        outline = self.outline(); contracts = compile_graph_scene_contracts(outline)
        sessions = [{"graph_scene_id":contract["scene_id"],
                     "completed_event_stage_ids":contract["assigned_event_stage_ids"]}
                    for contract in contracts]
        self.assertTrue(graph_completion_report(outline, contracts, sessions, 4100)["complete"])
        self.assertFalse(graph_completion_report(outline, contracts, sessions, 3999)["complete"])
        self.assertFalse(graph_completion_report(outline, contracts, sessions[:-1], 4100)["complete"])

    def test_orchestrator_uses_scene_specific_beats_and_locations(self):
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.config = {"mode":"live_graph_expansion"}
        orchestrator.data = {"outline":self.outline()}
        contracts = compile_graph_scene_contracts(self.outline())
        beats = orchestrator._assigned_narrative_beats(7, [])
        self.assertEqual({beat["event_id"] for beat in beats},
                         set(contracts[6]["assigned_event_arc_ids"]))
        self.assertEqual(orchestrator._session_location(5), "third")

    def test_graph_prompt_freezes_scene_execution_contract(self):
        contract = compile_graph_scene_contracts(self.outline())[1]
        kwargs = dict(session_index=2, narrative_beats=[{"event_id":"E1"}],
            recent_sessions=[], continuity_summary={}, active_plans=[], pending_decisions=[],
            superseded_information=[], character_knowledge={}, visibility_boundaries=[],
            unresolved_hooks=[{"hook_id":"H0001"}], allowed_people=["A", "B", "C"],
            session_location=contract["session_location"], allowed_event_arc_ids=["E1"],
            forbidden_conflicts=[], parent_anchors=[{"anchor_id":"P1"}])
        prompt = build_graph_realizer_prompt(graph_scene_contract=contract, **kwargs)
        for expected in (GRAPH_SCENE_CONTRACT_VERSION, "required_participant",
                         "target_dialogue_token_range", "hard_dialogue_token_range",
                         "assigned_event_stage_ids",
                         "Produce 8 to 14 dialogue turns", "Do not copy or paraphrase",
                         "Follow new_unresolved_hook_policy exactly",
                         "REQUIRED SPEAKER ROSTER FOR THIS RESPONSE: A, B, C"):
            self.assertIn(expected, prompt)
        self.assertEqual(len(graph_realizer_template_hash()), 64)

    def test_graph_retry_prompt_uses_specific_labels_without_raw_output(self):
        contract = compile_graph_scene_contracts(self.outline())[1]
        prompt = build_graph_realizer_retry_prompt(
            base_prompt="BASE", previous_errors=[
                "bad_dialogue_turn_schema",
                "graph_required_participant_missing",
                "graph_required_parent_anchor_missing",
                "graph_scene_token_count_out_of_range"],
            retry_index=1, graph_scene_contract=contract,
            dialogue_token_count=219, missing_required_participants=["C"],
            missing_required_parent_anchor_ids=["P1"],
            accepted_child_tokens=900)
        self.assertIn("only speaker and text", prompt)
        self.assertIn("220-650", prompt)
        self.assertIn("Missing roster: C", prompt)
        self.assertIn("Missing parent anchors: P1", prompt)
        self.assertIn('"previous_raw_response_included": false', prompt)
        self.assertNotIn("malformed model text", prompt)

    def test_graph_mode_budget_is_frozen(self):
        config = {**DEFAULTS, "mode":"live_graph_expansion",
            "max_total_paid_api_generations":30, "max_realizer_generations":30,
            "max_session_attempts_per_session":3, "max_questioner_generations":0,
            "max_post_freeze_memory_generations":0, "max_verifier_generations":0}
        validate_config(config)
        bad = dict(config); bad["max_realizer_generations"] = 31
        with self.assertRaises(ValueError): validate_config(bad)

    def test_full_graph_mock_executes_all_ten_scenes(self):
        outline = self.outline()
        contracts = compile_graph_scene_contracts(outline)

        class LineTokenizer:
            def encode(self, text, add_special_tokens=False):
                lines = 0 if not text else text.count("\n") + 1
                return list(range(lines * 52))

        class GraphRealizer:
            model_identifier = "mock-graph-realizer"

            def generate_session(self, request):
                contract = contracts[request.session_index - 1]
                participants = list(contract["required_participants"])
                support = " ".join(
                    stage["event_summary"] + " " + stage["stage"]
                    for stage in contract["assigned_event_stages"])
                turns = [{"speaker":participants[position % len(participants)],
                          "text":f"{support} substantive scene line {position}."}
                         for position in range(8)]
                payload = {"dialogue_turns":turns,
                    "used_parent_anchor_ids":contract["required_parent_anchor_ids"],
                    "advanced_event_arc_ids":contract["assigned_event_arc_ids"],
                    "continued_unresolved_hook_ids":[] if request.session_index == 1 else ["H0001"],
                    "resolved_hook_ids":[],
                    "new_unresolved_hooks":([{"description":"continue the graph plan"}]
                                              if request.session_index == 1 else []),
                    "durable_assertion_candidates":[], "local_scene_facts":[support]}
                return BackendResponse(json.dumps(payload), self.model_identifier)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "config/run_manifest.json", {"run_id":"graph-mock"})
            config = {**DEFAULTS, "mode":"live_graph_expansion",
                "max_total_paid_api_generations":30, "max_realizer_generations":30,
                "max_session_attempts_per_session":3, "max_questioner_generations":0,
                "max_post_freeze_memory_generations":0, "max_verifier_generations":0,
                "max_sessions":10}
            data = {"outline":outline, "parent":{"dialogue":{"turns":[
                {"turn_id":"PARENT1", "speaker":"A", "text":"anchor"}]}}}
            orchestrator = CanaryOrchestrator(run_root=root,
                manifest={"run_id":"graph-mock", "tokenizer_path":"mock"}, config=config,
                data=data, tokenizer=LineTokenizer(), realizer=GraphRealizer(),
                questioner=None, solver=None, backend_mode="mock")
            sessions = orchestrator.realize()
            self.assertEqual(len(sessions), 10)
            child = [turn for session in sessions for turn in session["dialogue_turns"]]
            completion = graph_completion_report(
                outline, contracts, sessions, orchestrator._tokens(child))
            self.assertTrue(completion["complete"])
            self.assertEqual({session["session_location"] for session in sessions},
                             {"first", "second", "third"})
            self.assertEqual({name for session in sessions
                              for name in session["newly_introduced_character_names"]},
                             {"C", "D"})
            resumed = orchestrator.realize()
            self.assertEqual(len(resumed), 10)
            self.assertEqual(orchestrator.ledger.summary()["mock_backend_call_count"], 10)


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

    def test_solver_feedback_layer_preserves_v7_and_uses_failures_only_as_quality_signals(self):
        dialogue = [{"session_id":"S1","turn_id":"T1","speaker":"A","text":"fact one"},
                    {"session_id":"S2","turn_id":"T2","speaker":"B","text":"fact two"}]
        feedback = [{"qa_id":"batch0_item1", "final_decision":"bad_ground",
                     "no_context_rewards":[1,0,1,1,1],
                     "diagnosis":"answer was guessable without dialogue"},
                    {"qa_id":"batch0_item2", "final_decision":"reject_easy",
                     "no_context_rewards":[0,0,0], "full_context_rewards":[1,1,1,1,1],
                     "diagnosis":"answer was a direct lookup with full dialogue"}]
        prompt = build_questioner_feedback_prompt(frozen_dialogue=dialogue,
            valid_turn_ids=["T1","T2"], valid_session_ids=["S1","S2"],
            assigned_gold_letters=["A","B","C"], recent_frontier_history=[],
            solver_feedback=feedback)
        self.assertIn("spell_dialogue_mcq_3x_v7_span_ids_solver_feedback_v1", prompt)
        self.assertIn("bad_ground", prompt); self.assertIn("reject_easy", prompt)
        self.assertIn("not a factual source", prompt)
        self.assertIn("Do not optimize for evidence-count quotas", prompt)
        self.assertIn("state changes, temporal order, causal dependencies", prompt)
        self.assertNotIn("question was", prompt)
        self.assertEqual(len(feedback_questioner_template_hash()), 64)

    def test_v8_span_ids_only_feedback_schema_has_no_evidence_value_contract(self):
        dialogue = [{"session_id":"S1","turn_id":"T1","speaker":"A","text":"fact one"},
                    {"session_id":"S2","turn_id":"T2","speaker":"B","text":"fact two"}]
        prompt = build_questioner_span_ids_only_feedback_prompt(frozen_dialogue=dialogue,
            valid_turn_ids=["T1","T2"], valid_session_ids=["S1","S2"],
            assigned_gold_letters=["A","B","C"], solver_feedback=[{
                "qa_id":"old", "final_decision":"bad_ground",
                "diagnosis":"answer was guessable without dialogue"}])
        self.assertIn("spell_dialogue_mcq_3x_v8_span_ids_only_solver_feedback_v1", prompt)
        self.assertIn('"additionalProperties": false', prompt)
        self.assertIn('"answer_support_span_ids"', prompt)
        self.assertIn("Do not output answer_evidence_values", prompt)
        self.assertIn("Do not output distractor_support", prompt)
        self.assertNotIn('"reasoning_contracts_by_array_position"', prompt)
        self.assertEqual(len(span_ids_only_questioner_template_hash()), 64)


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


class SpellStyleMinimalRedTests(unittest.TestCase):
    """These tests are added before the minimal admission implementation."""

    @staticmethod
    def dialogue():
        return [
            {"turn_id":"T1", "session_id":"S1", "speaker":"Maria",
             "text":"My aunt inspired me to volunteer."},
            {"turn_id":"T2", "session_id":"S1", "speaker":"Maria",
             "text":"Her aunt believed in volunteering."},
            {"turn_id":"T3", "session_id":"S2", "speaker":"John",
             "text":"Grandma discussed the film camera and charity event."},
        ]

    @staticmethod
    def raw():
        return {"qa_id":"minimal_q1", "question":"Who inspired Maria?",
            "correct_answer_text":"aunt",
            "distractors":["grandma", "film camera", "charity event"],
            "answer_support_span_ids":["T1#S0001", "T2#S0001"],
            "answer_evidence_values":[
                {"span_id":"T1#S0001", "value":"aunt"},
                {"span_id":"T2#S0001", "value":"aunt"}],
            "distractor_support":[
                {"distractor_index":0,"span_ids":["T3#S0001"],"evidence_values":["Grandma"]},
                {"distractor_index":1,"span_ids":["T3#S0001"],"evidence_values":["film camera"]},
                {"distractor_index":2,"span_ids":["T3#S0001"],"evidence_values":["charity event"]}],
            "correct_answer_explanation":"The selected spans name the aunt.",
            "distractor_explanations":["different person", "different object", "different event"],
            "reasoning_steps":["Read the selected spans."]}

    def api(self):
        from scripts.current_method.restart_n5_spell.minimal_admission import (
            build_answer_evidence_contexts, materialize_spell_style_minimal,
            normalize_minimal_text_v1,
        )
        return build_answer_evidence_contexts, materialize_spell_style_minimal, normalize_minimal_text_v1

    def test_strict_rejects_duplicate_value_but_minimal_accepts(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import (
            build_span_catalog, default_reasoning_contracts)
        from scripts.current_method.restart_n5_spell.validation import materialize_questioner_item
        _, minimal, _ = self.api()
        catalog = build_span_catalog(self.dialogue())
        contract = default_reasoning_contracts(("A","B","C"))[0]
        self.assertIn("bad_duplicate_information_point",
            materialize_questioner_item(self.raw(), "A", span_catalog=catalog,
                                        reasoning_contract=contract)[1])
        qa, errors, diagnostics = minimal(self.raw(), "A", span_catalog=catalog,
            strict_v7_reasons=["bad_duplicate_information_point", "bad_reasoning_contract"])
        self.assertEqual(errors, [])
        self.assertEqual(qa["reference_answer_text"], qa["options"]["A"])
        self.assertEqual(diagnostics["distinct_answer_value_count"], 1)

    def test_same_session_short_or_missing_reasoning_is_soft(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        for mutation in (lambda item: item,
                         lambda item: item.pop("reasoning_steps")):
            raw = self.raw(); mutation(raw)
            qa, errors, diagnostics = minimal(raw, "B", span_catalog=catalog)
            self.assertEqual(errors, [])
            self.assertEqual(qa["evidence_session_ids"], ["S1"])
            self.assertIn(diagnostics["reasoning_step_count"], (0, 1))

    def test_literal_span_claims_are_hard_but_answer_semantics_are_soft(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        not_in_answer = self.raw(); not_in_answer["correct_answer_text"] = "relative"
        qa, errors, _ = minimal(not_in_answer, "A", span_catalog=catalog)
        self.assertEqual(errors, [])
        not_in_span = self.raw(); not_in_span["answer_evidence_values"][0]["value"] = "friends"
        self.assertIn("bad_answer_evidence_binding",
                      minimal(not_in_span, "A", span_catalog=catalog)[1])
        unknown = self.raw(); unknown["answer_support_span_ids"] = ["UNKNOWN#S0001"]
        self.assertIn("bad_span_id", minimal(unknown, "A", span_catalog=catalog)[1])

    def test_optional_support_is_optional_but_malformed_is_rejected(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        absent = self.raw()
        for key in ("answer_evidence_values", "distractor_support", "correct_answer_explanation",
                    "distractor_explanations", "reasoning_steps"):
            absent.pop(key)
        qa, errors, diagnostics = minimal(absent, "C", span_catalog=catalog)
        self.assertEqual(errors, [])
        self.assertEqual(qa["reasoning_steps"], [])
        self.assertTrue(all(value == "missing" for value in diagnostics["optional_field_status"].values()))
        malformed = self.raw(); malformed["distractor_support"][0]["distractor_index"] = [0]
        self.assertIn("bad_distractor_support", minimal(malformed, "A", span_catalog=catalog)[1])

    def test_reserved_fields_and_normalized_option_duplicates_are_rejected(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, normalize = self.api(); catalog = build_span_catalog(self.dialogue())
        self.assertEqual(normalize("  A\u3000  B  "), "a b")
        reserved = self.raw(); reserved["options"] = {"A":"aunt"}
        self.assertIn("model_supplied_local_binding_fields",
                      minimal(reserved, "A", span_catalog=catalog)[1])
        duplicate = self.raw(); duplicate["distractors"][0] = "  AUNT "
        self.assertIn("bad_unlabeled_options", minimal(duplicate, "A", span_catalog=catalog)[1])

    def test_no_fuzzy_or_synonym_matching(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        raw = self.raw(); raw["answer_evidence_values"][0]["value"] = "aunts"
        self.assertIn("bad_answer_evidence_binding", minimal(raw, "A", span_catalog=catalog)[1])

    def test_evidence_context_uses_only_answer_support_turns(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        contexts, minimal, _ = self.api(); dialogue = self.dialogue(); catalog = build_span_catalog(dialogue)
        qa, errors, _ = minimal(self.raw(), "A", span_catalog=catalog)
        self.assertEqual(errors, [])
        result = contexts(qa, span_catalog=catalog, frozen_dialogue=dialogue)
        self.assertEqual([x["turn_id"] for x in result["evidence_only"]], ["T1", "T2"])
        self.assertEqual([x["turn_id"] for x in result["evidence_masked"]], ["T3"])

    def test_dual_batch_keeps_strict_and_minimal_decisions_independent(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import (
            build_span_catalog, default_reasoning_contracts)
        from scripts.current_method.restart_n5_spell.validation import validate_dual_admission_batch
        catalog = build_span_catalog(self.dialogue())
        raw = [self.raw(), {**self.raw(), "qa_id":"minimal_q2"},
               {**self.raw(), "qa_id":"minimal_q3", "answer_support_span_ids":["BAD#S0001"]}]
        result = validate_dual_admission_batch(raw, ["A","B","C"], span_catalog=catalog,
            reasoning_contracts=default_reasoning_contracts(("A","B","C")))
        self.assertEqual(result["strict_v7_accepted_count"], 0)
        self.assertEqual(result["minimal_accepted_count"], 2)
        self.assertEqual([x["spell_style_minimal_v1"]["decision"] for x in result["items"]],
                         ["accept", "accept", "reject"])

    def test_distractor_literal_claim_does_not_require_distractor_semantics(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        raw = self.raw()
        raw["distractors"][0] = "an unrelated but unique answer"
        qa, errors, diagnostics = minimal(raw, "A", span_catalog=catalog)
        self.assertEqual(errors, [])
        self.assertFalse(diagnostics["distractor_value_literal_checks"][0][
            "literal_in_distractor"])
        self.assertEqual(qa["options"]["B"], "an unrelated but unique answer")

    def test_placeholders_and_duplicate_distractors_are_rejected(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        placeholder = self.raw(); placeholder["question"] = "TODO insert answer"
        self.assertIn("minimal_placeholder_or_raw_artifact",
                      minimal(placeholder, "A", span_catalog=catalog)[1])
        duplicate = self.raw(); duplicate["distractors"][2] = " FILM   CAMERA "
        self.assertIn("bad_unlabeled_options", minimal(duplicate, "A", span_catalog=catalog)[1])

    def test_minimal_solver_selector_is_materialized_only(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import (
            build_span_catalog, default_reasoning_contracts)
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        contracts = default_reasoning_contracts(("A", "B", "C"))
        raw = self.raw()
        materialized, errors, diagnostics = minimal(raw, "A", span_catalog=catalog,
            strict_v7_reasons=["bad_reasoning_contract"], reasoning_contract=contracts[0])
        self.assertEqual(errors, [])
        admission = {"qa_id":raw["qa_id"], "item_index":0, "target_letter":"A",
            "strict_v7":{"decision":"reject", "reasons":["bad_reasoning_contract"]},
            "spell_style_minimal_v1":{"decision":"accept", "reasons":[],
                "diagnostics":diagnostics}, "solver_eligible":True}
        selected_raw, selected, _ = select_minimal_solver_qa([raw], admission, raw["qa_id"],
            materialized_qa=materialized, span_catalog=catalog, reasoning_contracts=contracts)
        self.assertIs(selected_raw, raw)
        self.assertEqual(selected, materialized)
        with self.assertRaisesRegex(RuntimeError, "solver_qa_not_materialized"):
            select_minimal_solver_qa([raw], admission, raw["qa_id"], materialized_qa=None,
                span_catalog=catalog, reasoning_contracts=contracts)
        tampered = copy.deepcopy(materialized); tampered["options"]["A"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "solver_qa_integrity_mismatch"):
            select_minimal_solver_qa([raw], admission, raw["qa_id"],
                materialized_qa=tampered, span_catalog=catalog, reasoning_contracts=contracts)

    def test_minimal_solver_selector_uses_admitted_item_index_with_later_duplicate_id(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import (
            build_span_catalog, default_reasoning_contracts)
        _, minimal, _ = self.api(); catalog = build_span_catalog(self.dialogue())
        contracts = default_reasoning_contracts(("A", "B", "C"))
        raw = self.raw()
        later_duplicate = copy.deepcopy(raw)
        later_duplicate["question"] = "A different question with the same invalid ID?"
        third = copy.deepcopy(raw); third["qa_id"] = "minimal_q3"
        materialized, errors, diagnostics = minimal(raw, "A", span_catalog=catalog,
            strict_v7_reasons=["bad_reasoning_contract"], reasoning_contract=contracts[0])
        self.assertEqual(errors, [])
        admission = {"qa_id":raw["qa_id"], "item_index":0, "target_letter":"A",
            "strict_v7":{"decision":"reject", "reasons":["bad_reasoning_contract"]},
            "spell_style_minimal_v1":{"decision":"accept", "reasons":[],
                "diagnostics":diagnostics}, "solver_eligible":True}
        selected_raw, selected, _ = select_minimal_solver_qa(
            [raw, later_duplicate, third], admission, raw["qa_id"],
            materialized_qa=materialized, span_catalog=catalog,
            reasoning_contracts=contracts)
        self.assertIs(selected_raw, raw)
        self.assertEqual(selected, materialized)

    def test_span_ids_only_profile_accepts_core_fields_and_rejects_removed_fields(self):
        from scripts.current_method.restart_n5_spell.questioner_contract import build_span_catalog
        from scripts.current_method.restart_n5_spell.minimal_admission import (
            SPAN_IDS_ONLY_SCHEMA_PROFILE, materialize_spell_style_minimal)
        catalog = build_span_catalog(self.dialogue())
        raw = self.raw()
        for key in ("answer_evidence_values", "distractor_support",
                    "correct_answer_explanation", "distractor_explanations"):
            raw.pop(key)
        qa, errors, _ = materialize_spell_style_minimal(raw, "A", span_catalog=catalog,
            schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
        self.assertEqual(errors, []); self.assertEqual(qa["reasoning_steps"], raw["reasoning_steps"])
        stale = copy.deepcopy(raw); stale["answer_evidence_values"] = []
        self.assertIn("span_ids_only_additional_properties",
            materialize_spell_style_minimal(stale, "A", span_catalog=catalog,
                schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)[1])


class MinimalSolverFinalMergeTests(unittest.TestCase):
    @staticmethod
    def qa():
        return {"qa_id":"minimal_solver_q1", "question":"q",
            "options":{"A":"correct", "B":"b", "C":"c", "D":"d"},
            "reference_answer_letter":"A", "reference_answer_text":"correct",
            "canonical_answer_text":"correct", "admission_mode":"spell_style_minimal_v1",
            "difficulty_level":"unclassified", "evidence_span_ids":["T1#S0001"],
            "evidence_turn_ids":["T1"], "evidence_session_ids":["S1"],
            "reasoning_steps":[]}

    @staticmethod
    def rollout(rewards, context="context"):
        return {"requested_valid_rollouts":len(rewards), "obtained_valid_rollouts":len(rewards),
            "verified_rewards":list(rewards), "raw_generations":len(rewards),
            "parse_invalid_generations":0, "retry_generations":0,
            "exhausted_target_slots":0, "rollouts":[], "context_sha256":context,
            "context_turn_ids":[]}

    def run_audit(self, responses):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        calls = []
        def fake_rollouts(qa, mode, order_id, valid_k, options, slot_start=0):
            calls.append((mode, order_id, valid_k, slot_start))
            value = responses[(mode, order_id, slot_start)]
            return self.rollout(value, "full" if mode in ("full_context", "option_permutation") else mode)
        orchestrator._rollouts = fake_rollouts
        return orchestrator.audit_minimal_candidate(self.qa()), calls

    def test_complete_audits_retain_and_reuse_original_k5(self):
        result, calls = self.run_audit({
            ("no_context", "original", 0):[0,0,1],
            ("full_context", "original", 0):[1,0,1,0,1],
            ("evidence_only", "original", 0):[1,1,0],
            ("evidence_masked", "original", 0):[0,0,1],
            ("option_permutation", "perm1", 0):[1,0,1,0,1],
            ("option_permutation", "perm2", 0):[0,1,0,1,1],
        })
        self.assertEqual(result["final_decision"], "retain_frontier")
        self.assertTrue(result["original_full_context_reused_for_option_audit"])
        self.assertEqual(calls.count(("full_context", "original", 5, 0)), 1)

    def test_bad_ground_never_runs_full_context(self):
        result, calls = self.run_audit({("no_context", "original", 0):[1,1,1]})
        self.assertEqual(result["final_decision"], "bad_ground")
        self.assertFalse(any(mode == "full_context" for mode, *_ in calls))

    def test_final_merge_priorities_and_inconclusive(self):
        base = {
            ("no_context", "original", 0):[0,0,1],
            ("full_context", "original", 0):[1,0,1,0,1],
            ("evidence_only", "original", 0):[0,0,0],
            ("evidence_masked", "original", 0):[1,1,1],
            ("option_permutation", "perm1", 0):[1,1,1,1,0],
            ("option_permutation", "perm2", 0):[0,0,0,0,1],
        }
        result, _ = self.run_audit(base)
        self.assertEqual(result["evidence_audit"]["decision"], "bad_evidence_relevance")
        self.assertEqual(result["option_audit"]["decision"], "bad_option_bias")
        self.assertEqual(result["final_decision"], "bad_evidence_relevance")
        incomplete = dict(base)
        incomplete[("evidence_only", "original", 0)] = [1, 0]
        incomplete[("option_permutation", "perm1", 0)] = [1,0,1,0,1]
        incomplete[("option_permutation", "perm2", 0)] = [0,1,0,1,1]
        result, _ = self.run_audit(incomplete)
        self.assertEqual(result["final_decision"], "frontier_audit_inconclusive")


class NoContextPermutationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def qa():
        return {"qa_id":"diagnostic_q1", "question":"q",
            "options":{"A":"correct", "B":"b", "C":"c", "D":"d"},
            "reference_answer_letter":"A", "reference_answer_text":"correct",
            "canonical_answer_text":"correct", "admission_mode":"spell_style_minimal_v1",
            "evidence_span_ids":["T1#S0001"]}

    @staticmethod
    def original():
        return {"requested_valid_rollouts":3, "obtained_valid_rollouts":3,
            "verified_rewards":[1,1,1], "raw_generations":3,
            "parse_invalid_generations":0, "retry_generations":0,
            "exhausted_target_slots":0,
            "rollouts":[{"parsed_letter":"A", "local_reward":1} for _ in range(3)]}

    def test_original_k3_is_reused_and_only_two_no_context_orders_run(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        calls = []
        qa = self.qa()
        def fake_rollouts(candidate, mode, order_id, requested, options, slot_start=0):
            calls.append((mode, order_id, requested, slot_start, dict(options)))
            gold = next(letter for letter, value in options.items() if value == "correct")
            return {"requested_valid_rollouts":3, "obtained_valid_rollouts":3,
                "verified_rewards":[1,1,1], "raw_generations":3,
                "parse_invalid_generations":0, "retry_generations":0,
                "exhausted_target_slots":0,
                "rollouts":[{"parsed_letter":gold, "local_reward":1} for _ in range(3)]}
        orchestrator._rollouts = fake_rollouts
        result = orchestrator.diagnose_no_context_permutations(qa, self.original())
        self.assertEqual([(mode, order, requested, slot) for mode, order, requested, slot, _ in calls],
                         [("no_context","perm1",3,0), ("no_context","perm2",3,0)])
        self.assertTrue(result["original_no_context_reused"])
        self.assertEqual(result["diagnostic_decision"], "bad_ground_semantic_leakage")
        self.assertFalse(result["authoritative_grounding_decision_changed"])
        self.assertFalse(result["full_context_ran"])

    def test_fixed_letter_is_position_bias(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        def fake_rollouts(candidate, mode, order_id, requested, options, slot_start=0):
            gold = next(letter for letter, value in options.items() if value == "correct")
            reward = 1 if gold == "A" else 0
            return {"requested_valid_rollouts":3, "obtained_valid_rollouts":3,
                "verified_rewards":[reward] * 3, "raw_generations":3,
                "parse_invalid_generations":0, "retry_generations":0,
                "exhausted_target_slots":0,
                "rollouts":[{"parsed_letter":"A", "local_reward":reward} for _ in range(3)]}
        orchestrator._rollouts = fake_rollouts
        result = orchestrator.diagnose_no_context_permutations(self.qa(), self.original())
        self.assertEqual(result["diagnostic_decision"], "bad_ground_position_bias")

    def test_incomplete_permutation_is_inconclusive(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        def fake_rollouts(candidate, mode, order_id, requested, options, slot_start=0):
            return {"requested_valid_rollouts":3, "obtained_valid_rollouts":2,
                "verified_rewards":[1,1], "raw_generations":4,
                "parse_invalid_generations":2, "retry_generations":1,
                "exhausted_target_slots":1,
                "rollouts":[{"parsed_letter":"B", "local_reward":1} for _ in range(2)]}
        orchestrator._rollouts = fake_rollouts
        result = orchestrator.diagnose_no_context_permutations(self.qa(), self.original())
        self.assertEqual(result["diagnostic_decision"],
                         "bad_ground_permutation_inconclusive")


class QuestionerFormatRetryTests(unittest.TestCase):
    class Backend:
        model_identifier = "format-retry-fixture"
        def __init__(self, responses):
            self.responses = iter(responses); self.requests = []
        def generate_batch(self, request):
            self.requests.append(request)
            return BackendResponse(next(self.responses), self.model_identifier,
                                   {"finish_reason":"stop"})

    def run_smoke(self, responses):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        atomic_json(root / "config/run_manifest.json", {"run_id":"retry-test"})
        backend = self.Backend(responses)
        orchestrator = CanaryOrchestrator(run_root=root, manifest={"run_id":"retry-test"},
            config={"max_questioner_generations":2, "max_total_paid_api_generations":2,
                    "allow_local_difficulty_downgrade":False}, data={}, tokenizer=None,
            realizer=None, questioner=backend, solver=None, backend_mode="mock")
        dialogue = [{"turn_id":"T1","session_id":"S1","speaker":"A","text":"fact"}]
        result = orchestrator.question_smoke(dialogue, ("T1",), ("S1",), ("A","B","C"),
            rendered_prompt="semantic prompt", prompt_protocol="feedback-v1",
            max_format_retries=1, solver_feedback_sha256="feedback-sha")
        return result, backend, orchestrator

    def test_json_invalid_gets_one_full_batch_format_retry(self):
        result, backend, orchestrator = self.run_smoke(["MALFORMED_SECRET", "[]"])
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(result["physical_questioner_calls"], 2)
        self.assertEqual(result["attempts"][0]["decision"], "retry_format_only")
        self.assertEqual(result["attempts"][1]["parse_status"], "json_parsed")
        self.assertNotIn("MALFORMED_SECRET", backend.requests[1].prompt)
        first = orchestrator.ledger.read("questioner-b00-a00")
        second = orchestrator.ledger.read("questioner-b00-a01")
        self.assertEqual(first["final_decision"], "retry_format_only")
        self.assertEqual(second["request_metadata"]["retry_of_request_id"],
                         "questioner-b00-a00")
        self.assertEqual(second["request_metadata"]["retry_reason"], "json_invalid")

    def test_second_json_invalid_is_exhausted_without_third_call(self):
        result, backend, _ = self.run_smoke(["{", "["])
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(result["batch_decision"], "questioner_json_invalid_exhausted")
        self.assertEqual(result["attempts"][-1]["decision"],
                         "questioner_json_invalid_exhausted")

    def test_parseable_wrong_batch_count_does_not_retry(self):
        result, backend, _ = self.run_smoke(["[]"])
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(result["parse_status"], "json_parsed")
        self.assertEqual(result["batch_decision"], "questioner_batch_count_invalid")


class SemanticLeakageRepairTests(unittest.TestCase):
    V8_GENERATION_PROMPT_SHA = (
        "0d01bf9e66a40eb243dd4efaf7cd192786e1fd173f52a0e3b1de910451c2a087"
    )

    @staticmethod
    def dialogue():
        return [
            {"turn_id":"T1", "session_id":"S1", "speaker":"Maria",
             "text":"Maria chose the library meeting after the charity dinner."},
            {"turn_id":"T2", "session_id":"S2", "speaker":"John",
             "text":"John chose the park meeting after the camera workshop."},
        ]

    @classmethod
    def catalog(cls):
        return build_span_catalog(cls.dialogue())

    @staticmethod
    def raw(qa_id="source_q_repair1"):
        return {"qa_id":qa_id,
            "question":"Which meeting did Maria choose after the charity dinner?",
            "correct_answer_text":"the library meeting",
            "distractors":["the park meeting", "the camera workshop", "the charity dinner"],
            "answer_support_span_ids":["T1#S0001"],
            "reasoning_steps":["Match Maria's later choice to the named event."]}

    def contract(self):
        return default_reasoning_contracts(("A", "B", "C"))[0]

    def prompt(self):
        return build_semantic_leakage_repair_prompt(
            frozen_dialogue=self.dialogue(), span_catalog=self.catalog(),
            original_item={**self.raw("source_q"), "qa_id":"source_q"},
            repaired_qa_id="source_q_repair1", target_answer_letter="A",
            original_no_context={"verified_rewards":[1, 1, 1]},
            permutation_diagnostic={"decision":"bad_ground_semantic_leakage",
                "orders":[{"order_id":"original", "gold_letter":"A",
                           "valid_answer_letters":["A", "A", "A"],
                           "valid_rewards":[1, 1, 1]}]})

    def test_fixed_prompt_and_single_object_schema(self):
        self.assertEqual(SEMANTIC_LEAKAGE_REPAIR_PROMPT_VERSION,
                         "spell_dialogue_mcq_single_v8_semantic_leakage_repair_v1")
        self.assertFalse(SEMANTIC_LEAKAGE_REPAIR_JSON_SCHEMA["additionalProperties"])
        self.assertEqual(set(SEMANTIC_LEAKAGE_REPAIR_JSON_SCHEMA["properties"]), {
            "qa_id", "question", "correct_answer_text", "distractors",
            "answer_support_span_ids", "reasoning_steps"})
        prompt = self.prompt()
        for text in ("semantic answer leakage", "not letter-position bias",
                     "same answer type", "parallel grammatical form",
                     "similar length", "Regenerate one complete item from scratch"):
            self.assertIn(text, prompt)
        for forbidden in ("answer_evidence_values", "distractor_support", "repair_notes"):
            self.assertIn(forbidden, prompt)
        self.assertEqual(len(semantic_leakage_repair_template_hash()), 64)
        self.assertEqual(span_ids_only_questioner_template_hash(),
                         self.V8_GENERATION_PROMPT_SHA)

    def test_only_authoritative_semantic_bad_ground_is_eligible_once(self):
        base = {"original_final_decision":"bad_ground",
            "diagnostic_decision":"bad_ground_semantic_leakage",
            "repair_attempt_count":0, "source_authoritative":True,
            "source_provenance_mixed":False}
        self.assertTrue(semantic_repair_eligibility(**base)["eligible"])
        variations = (
            {"diagnostic_decision":"bad_ground_position_bias"},
            {"diagnostic_decision":"bad_ground_mixed_bias"},
            {"diagnostic_decision":"bad_ground_permutation_inconclusive"},
            {"original_final_decision":"reject_easy"},
            {"repair_attempt_count":1},
            {"source_authoritative":False},
            {"source_provenance_mixed":True},
        )
        for change in variations:
            with self.subTest(change=change):
                self.assertFalse(semantic_repair_eligibility(
                    **{**base, **change})["eligible"])

    def test_hybrid_budget_contract_is_frozen(self):
        config = {**DEFAULTS, "mode":"hybrid_semantic_leakage_repair",
            "max_total_paid_api_generations":2, "max_semantic_repair_generations":2,
            "max_realizer_generations":0, "max_questioner_generations":0,
            "max_verifier_generations":0, "semantic_repair_max_attempts":1,
            "semantic_repair_max_format_retries":1,
            "solver_effective_max_model_len":16384}
        validate_config(config)
        config["max_semantic_repair_generations"] = 3
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_repair_reuses_exact_minimal_gate_and_local_binding(self):
        from scripts.current_method.restart_n5_spell.minimal_admission import (
            SPAN_IDS_ONLY_SCHEMA_PROFILE, materialize_spell_style_minimal)
        raw = self.raw()
        admission = admit_semantic_repair_item(raw,
            expected_qa_id=raw["qa_id"], target_letter="B",
            span_catalog=self.catalog(), reasoning_contract=self.contract())
        direct, direct_errors, _ = materialize_spell_style_minimal(raw, "B",
            span_catalog=self.catalog(), reasoning_contract=self.contract(),
            schema_profile=SPAN_IDS_ONLY_SCHEMA_PROFILE)
        self.assertEqual(admission["minimal_decision"], "accept")
        self.assertEqual(admission["materialized_qa"], direct)
        self.assertEqual(direct_errors, [])
        self.assertEqual(direct["reference_answer_letter"], "B")
        self.assertEqual(direct["reference_answer_text"], raw["correct_answer_text"])
        self.assertEqual(direct["parent_qa_id"] if "parent_qa_id" in direct else None, None)

        for extra in ("options", "answer_evidence_values", "distractor_support",
                      "difficulty_level", "repair_notes"):
            stale = {**raw, extra:{} if extra == "options" else []}
            rejected = admit_semantic_repair_item(stale,
                expected_qa_id=raw["qa_id"], target_letter="B",
                span_catalog=self.catalog(), reasoning_contract=self.contract())
            self.assertEqual(rejected["minimal_decision"], "reject", extra)
        bad_span = {**raw, "answer_support_span_ids":["missing"]}
        self.assertIn("bad_span_id", admit_semantic_repair_item(bad_span,
            expected_qa_id=raw["qa_id"], target_letter="A",
            span_catalog=self.catalog(), reasoning_contract=self.contract())[
                "minimal_reasons"])
        self.assertIn("repair_qa_id_mismatch", admit_semantic_repair_item(raw,
            expected_qa_id="other_repair1", target_letter="A",
            span_catalog=self.catalog(), reasoning_contract=self.contract())[
                "minimal_reasons"])

    class Backend:
        model_identifier = "semantic-repair-fixture"
        def __init__(self, responses):
            self.responses = iter(responses)
            self.requests = []
        def generate_batch(self, request):
            self.requests.append(request)
            return BackendResponse(next(self.responses), self.model_identifier,
                                   {"finish_reason":"stop", "usage":{}})

    def run_repair(self, responses):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        manifest = {"run_id":"semantic-repair-test"}
        atomic_json(root / "config/run_manifest.json", manifest)
        backend = self.Backend(responses)
        orchestrator = CanaryOrchestrator(run_root=root, manifest=manifest,
            config={"max_semantic_repair_generations":2,
                    "max_total_paid_api_generations":2}, data={}, tokenizer=None,
            realizer=None, questioner=backend, solver=None, backend_mode="mock")
        result = orchestrator.semantic_leakage_repair(
            parent_qa_id="source_q", repaired_qa_id="source_q_repair1",
            target_letter="A", semantic_prompt=self.prompt(),
            span_catalog=self.catalog(), reasoning_contract=self.contract(),
            source_contract={"source_qa_sha256":"qa-sha",
                             "source_diagnostic_sha256":"diagnostic-sha"})
        return result, backend, orchestrator

    def test_json_invalid_gets_at_most_one_format_only_retry(self):
        result, backend, orchestrator = self.run_repair([
            "MALFORMED_REPAIR_SECRET", json.dumps(self.raw())])
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(result["physical_api_request_count"], 2)
        self.assertTrue(result["format_retry_used"])
        self.assertNotIn("MALFORMED_REPAIR_SECRET", backend.requests[1].prompt)
        self.assertEqual(result["admission"]["minimal_decision"], "accept")
        retry = orchestrator.ledger.read("semantic-repair-source_q-a01")
        self.assertEqual(retry["request_metadata"]["retry_reason"], "json_invalid")
        self.assertEqual(retry["request_metadata"]["retry_mode"],
                         "full_item_regeneration")

        exhausted, exhausted_backend, _ = self.run_repair(["{", "["])
        self.assertEqual(len(exhausted_backend.requests), 2)
        self.assertEqual(exhausted["final_decision"],
                         "semantic_leakage_repair_json_invalid_exhausted")

    def test_parseable_schema_failure_does_not_retry_or_run_solver(self):
        result, backend, _ = self.run_repair([json.dumps([self.raw()])])
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(result["final_decision"], "semantic_repair_minimal_reject")
        self.assertNotIn("materialized_qa_path", result)

    def test_live_repair_backend_requests_json_object(self):
        class Client:
            config = type("Config", (), {"questioner_max_tokens":123})()
            def __init__(self): self.kwargs = None
            def complete(self, **kwargs):
                self.kwargs = kwargs
                return BackendResponse("{}", "fixture")
        client = Client()
        backend = LiveSemanticRepairBackend(client)
        request = type("Request", (), {"prompt":"repair"})()
        backend.generate_batch(request)
        self.assertTrue(client.kwargs["json_object"])
        self.assertEqual(client.kwargs["max_tokens"], 123)
        limited = LiveSemanticRepairBackend(client, 77)
        limited.generate_batch(request)
        self.assertEqual(client.kwargs["max_tokens"], 77)

    def test_repair_final_decisions_are_namespaced_and_never_repair_twice(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.run_root = Path(temporary.name)
        qa = MinimalSolverFinalMergeTests.qa()
        cases = (("bad_ground", "bad_ground_after_semantic_repair"),
                 ("reject_easy", "reject_easy_after_semantic_repair"),
                 ("reject_all_wrong", "reject_all_wrong_after_semantic_repair"),
                 ("retain_frontier", "retain_frontier"))
        for base, expected in cases:
            with self.subTest(base=base):
                orchestrator.audit_minimal_candidate = lambda candidate, base=base: {
                    "qa_id":candidate["qa_id"], "final_decision":base,
                    "grounding_decision":"bad_ground" if base == "bad_ground" else
                    "pass_grounding"}
                result = orchestrator.audit_semantic_repair_candidate(qa, "source_q")
                self.assertEqual(result["final_decision"], expected)
                self.assertFalse(result["second_semantic_repair_allowed"])
                self.assertEqual(result["origin"], "semantic_leakage_repair")

    def test_feedback_summary_is_deterministic(self):
        summary = build_semantic_repair_feedback_summary([
            {"repair_json_valid":True, "repair_minimal_accepted":True,
             "repair_passed_grounding":False,
             "final_decision":"bad_ground_after_semantic_repair"},
            {"repair_json_valid":True, "repair_minimal_accepted":True,
             "repair_passed_grounding":True,
             "final_decision":"reject_easy_after_semantic_repair"},
            {"repair_json_valid":True, "repair_minimal_accepted":True,
             "repair_passed_grounding":True, "final_decision":"retain_frontier"},
        ])
        self.assertEqual(summary["repair_attempted"], 3)
        self.assertEqual(summary["repair_passed_grounding"], 2)
        self.assertEqual(summary["repair_reject_easy"], 1)
        self.assertEqual(summary["repair_retain_frontier"], 1)


class CrossDialogueCalibrationTests(unittest.TestCase):
    def test_plan_is_exactly_five_fresh_batches_without_repair(self):
        plan = calibration_plan()
        self.assertEqual(tuple(item["lineage"] for item in plan), CALIBRATION_LINEAGES)
        self.assertEqual(tuple(tuple(item["assigned_target_letters"]) for item in plan),
                         CALIBRATION_TARGET_LETTERS)
        self.assertEqual(sum(item["mcqs_per_batch"] for item in plan), 15)
        self.assertTrue(all(item["questioner_batches"] == 1 for item in plan))
        self.assertTrue(all(item["semantic_repair_enabled"] is False for item in plan))

    def test_parent_only_freeze_attests_source_row_and_content(self):
        from scripts.current_method.restart_n5_spell.core import canonical_json, sha256_bytes
        from scripts.current_method.restart_n5_spell.input_adapter import dialogue_content_hash
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "source.jsonl"
        source_row = {"sample_id":"source", "payload":{"stable":True}}
        source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
        turns = [{"turn_id":"D1:1", "speaker":"A", "text":"First fact."},
                 {"turn_id":"D2:1", "speaker":"B", "text":"Second fact."}]
        parent = {"sample_id":"parent", "source_id":"source", "split":"train",
            "dialogue":{"turns":turns}, "source_snapshot":{
                "source_path":str(source), "source_line_number":1,
                "row_sha256":sha256_bytes(canonical_json(source_row).encode()),
                "dialogue_content_sha256":dialogue_content_hash(turns),
                "parent_identity_sha256":"parent-sha",
                "reconstructed_graph_sha256":"graph-sha"}}
        frozen = freeze_parent_dialogue("conv-26", parent)
        self.assertTrue(frozen["context"]["parent_only"])
        self.assertEqual([x["session_id"] for x in frozen["context"]["combined_dialogue"]],
                         ["D1", "D2"])
        self.assertEqual(frozen["source_attestation"]["source_row_sha256"],
                         parent["source_snapshot"]["row_sha256"])
        parent["dialogue"]["turns"][0]["text"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "source_integrity_failed"):
            freeze_parent_dialogue("conv-26", parent)

    def feedback_fixture(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        summary = {"source_failure_type":"bad_ground_semantic_leakage",
            "source_item_count":3, "repair_attempted":3, "repair_json_valid":3,
            "repair_minimal_accepted":3, "repair_passed_grounding":0,
            "repair_reject_easy":0, "repair_reject_all_wrong":0,
            "repair_retain_frontier":0}
        items = []
        for index in range(3):
            child = root / f"child{index}"
            atomic_json(child / "config/run_manifest.json", {"run_id":f"child{index}"})
            atomic_json(child / "reports/semantic_repair_full_audit.json",
                        {"final_decision":"bad_ground_after_semantic_repair"})
            atomic_json(child / "reports/post_run_provenance.json",
                        {"authoritative_release_provenance":True,
                         "provenance_mixed":False})
            items.append({"qa_id":f"q{index}", "run_root":str(child),
                "run_manifest_sha256":sha256_file(child / "config/run_manifest.json"),
                "repair_report_sha256":sha256_file(
                    child / "reports/semantic_repair_full_audit.json")})
        audit = {"items":items, "first_retain_frontier_found":False}
        atomic_json(root / "frozen/semantic_repair_feedback_summary.json", summary)
        atomic_json(root / "reports/semantic_repair_series_audit.json", audit)
        manifest = {"feedback_summary_sha256":sha256_file(
                root / "frozen/semantic_repair_feedback_summary.json"),
            "series_audit_sha256":sha256_file(
                root / "reports/semantic_repair_series_audit.json")}
        atomic_json(root / "config/run_manifest.json", manifest)
        atomic_json(root / "reports/post_run_provenance.json", {
            "run_manifest_sha256":sha256_file(root / "config/run_manifest.json"),
            "authoritative_release_provenance":True, "provenance_mixed":False})
        return root

    def test_feedback_memory_is_bound_to_authoritative_failed_repair(self):
        root = self.feedback_fixture()
        result = verify_semantic_repair_feedback_source(root)
        self.assertFalse(result["factual_source"])
        self.assertEqual(result["feedback"][0]["final_decision"],
                         "bad_ground_semantic_leakage")
        self.assertIn("Generate fresh items rather than repairing",
                      result["feedback"][0]["diagnosis"])
        summary_path = root / "frozen/semantic_repair_feedback_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["repair_passed_grounding"] = 1
        atomic_json(summary_path, summary)
        with self.assertRaisesRegex(RuntimeError, "not_authoritative"):
            verify_semantic_repair_feedback_source(root)

    def test_calibration_summary_uses_frontier_and_leakage_thresholds(self):
        dialogues = [{"raw_candidate_count":3, "minimal_accepted_count":3,
            "items":[{"solver_evaluated":True, "final_decision":"retain_frontier"},
                     {"solver_evaluated":True, "final_decision":"retain_frontier"},
                     {"solver_evaluated":True, "final_decision":"reject_easy"}]}]
        self.assertEqual(summarize_calibration(dialogues)["route_decision"],
                         "training_free_route_feasible")
        leakage = [{"raw_candidate_count":3, "minimal_accepted_count":3,
            "items":[{"solver_evaluated":True, "final_decision":"bad_ground",
                       "no_context_permutation_diagnostic":
                           "bad_ground_semantic_leakage"} for _ in range(3)]}
                   for _ in range(5)]
        summary = summarize_calibration(leakage)
        self.assertEqual(summary["retain_frontier_count"], 0)
        self.assertEqual(summary["route_decision"],
                         "single_stage_api_questioner_not_supported")

    def test_runner_reuses_v8_and_contains_no_semantic_repair_backend(self):
        source = (Path(__file__).resolve().parents[1] /
            "scripts/current_method/restart_n5_spell/run_cross_dialogue_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertIn("build_questioner_span_ids_only_feedback_prompt", source)
        self.assertIn("audit_minimal_candidate", source)
        self.assertIn("diagnose_no_context_permutations", source)
        self.assertNotIn("LiveSemanticRepairBackend", source)
        self.assertNotIn("semantic_leakage_repair(", source)
        self.assertIn("adaptive_grounding_boundary_result", source)
        self.assertEqual(span_ids_only_questioner_template_hash(),
            SemanticLeakageRepairTests.V8_GENERATION_PROMPT_SHA)

    def test_live_like_questioner_batch_reaches_materialized_minimal_qas(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        turns = [{"turn_id":"T1", "session_id":"S1", "speaker":"A",
                  "text":"Alpha chose red after lunch."},
                 {"turn_id":"T2", "session_id":"S2", "speaker":"B",
                  "text":"Beta chose blue before dinner."}]
        catalog = build_span_catalog(turns)
        contracts = default_reasoning_contracts(("A", "B", "C"))
        raw = []
        for index, span_id in enumerate(("T1#S0001", "T2#S0001", "T1#S0001")):
            raw.append({"qa_id":f"calibration_q{index}",
                "question":f"Which choice applies in case {index}?",
                "correct_answer_text":f"correct {index}",
                "distractors":[f"wrong {index} a", f"wrong {index} b",
                               f"wrong {index} c"],
                "answer_support_span_ids":[span_id]})
        backend = QuestionerFormatRetryTests.Backend([json.dumps(raw)])
        config = {**DEFAULTS, "mode":"cross_dialogue_questioner_batch",
            "max_total_paid_api_generations":2, "max_realizer_generations":0,
            "max_questioner_generations":2, "max_verifier_generations":0,
            "questioner_generations":1, "questioner_mcqs_per_generation":3,
            "questioner_total_candidates":3,
            "allow_local_difficulty_downgrade":False}
        manifest = {"run_id":"calibration-questioner-fixture"}
        atomic_json(root / "config/run_manifest.json", manifest)
        result = cross_calibration_runner._run_questioner(root=root,
            manifest=manifest, config=config,
            artifacts={"context":{"lineage":"fixture", "combined_dialogue":turns},
                "catalog":catalog, "contracts":contracts,
                "letters":("A", "B", "C"), "prompt":"v8 fixture prompt"},
            backend=backend, tokenizer=None, feedback_sha="feedback-sha")
        self.assertEqual(result["report"]["raw_candidate_count"], 3)
        self.assertEqual(result["report"]["minimal_accepted_count"], 3)
        self.assertEqual(set(result["materialized"]), {0, 1, 2})
        self.assertTrue(all(qa["admission_mode"] == "spell_style_minimal_v1"
                            for qa in result["materialized"].values()))
        self.assertEqual(len(backend.requests), 1)


if __name__ == "__main__": unittest.main()
