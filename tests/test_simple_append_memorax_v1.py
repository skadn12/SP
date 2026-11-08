import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from scripts.current_method.restart_n5_spell.simple_append_v1 import (
    build_tail_context,
    compile_round_plan,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1 import (
    MemoraxClient,
    MemoraxConfig,
    MemoraxTransportError,
    build_add_intent,
    build_bucket_search_payload,
    build_memory_augmented_mcq_prompt,
    build_memory_augmented_planner_prompt,
    build_memory_augmented_realizer_prompt,
    compile_memory_overview,
    memorax_user_id,
    normalize_bucket_snapshot,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.preflight import (
    run as run_preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.qa_live import (
    DEFAULT_MEMORY_RECOVERY as QA_MEMORY_RECOVERY,
    DEFAULT_SOURCE_RUN as QA_SOURCE_RUN,
    _quarantine_invalid_writer_questions,
    materialize_position_first_writer_output,
    _writer_request_summary,
    _highest_round_index,
    preflight as run_qa_preflight,
)
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.qa_admission import (
    classify_qa_admission_tier,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.qa_sources import (
    resolve_qa_artifacts,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    normalize_answer_dimension,
    normalize_focus_writer_batch,
    partition_mcq_candidates,
    validate_option_contract,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.batch_preflight import (
    INPUT_MANIFEST_VERSION,
    load_input_manifest,
    run as run_batch_preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.batch_live import (
    _checkpoint,
    batch_exit_code,
    classify_batch_status,
    run as run_batch_live,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.generation_registry import (
    GenerationAlreadyRegistered,
    build_generation_identity,
    finalize_generation,
    mark_memory_status,
    memorax_namespace_sha,
    reserve_generation,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    DEFAULT_API_CONFIG,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
)


PARENT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/"
    "unified_root.json")
EXPANDED_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_simple_append_v1_round_04_live_3/"
    "round_04/expanded_dialogue.json")


def bucket_response():
    return {"success": True, "data": {"buckets": {
        "person": {
            "alice": {"display_key": "Alice", "content": ["Alice met Bob."]},
            "bob": {"display_key": "Bob", "content": ["Bob met Alice."]},
        },
        "location": {"shanghai": {"display_key": "Shanghai", "content": [
            "The meeting is in Shanghai."]}},
        "event": {"meeting": {"display_key": "Meeting", "content": [
            "Alice and Bob review a plan."]}},
        "item": {"camera": {"display_key": "Camera", "content": [
            {"attribute": "model", "value": "A7"}]}},
        "dreaming": {},
    }}}


class SimpleAppendMemoraxV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        cls.expanded = json.loads(EXPANDED_PATH.read_text(encoding="utf-8"))
        cls.tail = build_tail_context(cls.parent, session_count=2)
        cls.root_id = cls.parent["root_id"]
        cls.snapshot = normalize_bucket_snapshot(
            bucket_response(), root_id=cls.root_id,
            user_id=memorax_user_id(cls.root_id))
        cls.overview = compile_memory_overview(
            cls.snapshot,
            recent_dialogue_text="Alice revisits the meeting with Bob.")
        speakers = cls.tail["parent_speakers"][:2]
        cls.plan = {
            "planner_schema_version": PLANNER_SCHEMA_VERSION,
            "round_goal": "Continue the story.",
            "scenes": [{
                "goal": f"Goal {index}",
                "participants": speakers,
                "new_development": f"Development {index}",
                "interaction_beats": ["One", "Two", "Three"],
                "transition_from_previous": "Continue naturally.",
                "desired_outcome": f"Outcome {index}",
            } for index in range(1, 4)],
        }
        cls.compiled = compile_round_plan(cls.plan, tail_context=cls.tail)

    def test_user_id_is_stable_and_root_isolated(self):
        self.assertEqual("gxy-selfplay:cc-train-episode-61555",
                         memorax_user_id(self.root_id))
        self.assertNotEqual(memorax_user_id(self.root_id),
                            memorax_user_id("cc-train-episode-91716"))

    def test_add_intent_contains_only_dialogue_and_speaker_prefixes(self):
        intent = build_add_intent(
            root_id=self.root_id, round_index=0,
            source_dialogue_sha256=self.parent["dialogue"]["dialogue_hash"],
            turns=self.parent["dialogue"]["turns"])
        self.assertEqual(60, len(intent["turn_refs"]))
        self.assertEqual("default", intent["payload"]["mode"])
        self.assertFalse(intent["payload"]["return_raw_text"])
        self.assertTrue(all(row["content"].startswith("[Speaker: ")
                            for row in intent["payload"]["messages"]))
        serialized = json.dumps(intent)
        for forbidden in ("question", "gold_answer", "evidence_turn_ids",
                          "MEMORAX_API_KEY", "Authorization"):
            self.assertNotIn(forbidden, serialized)

    def test_add_intent_rejects_duplicate_turn_ref(self):
        turn = self.parent["dialogue"]["turns"][0]
        with self.assertRaisesRegex(ValueError, "turn_ref_duplicate"):
            build_add_intent(
                root_id=self.root_id, round_index=1,
                source_dialogue_sha256=self.parent["dialogue"]["dialogue_hash"],
                turns=[turn, turn])

    def test_bucket_query_requests_all_five_types(self):
        value = build_bucket_search_payload(root_id=self.root_id)
        filters = value["payload"]["bucket_filters"]
        self.assertEqual("all", filters["entity_names"])
        self.assertEqual("all", filters["location_names"])
        self.assertEqual("all", filters["event_names"])
        self.assertEqual("all", filters["item_names"])
        self.assertEqual("all", filters["dreaming_names"])

    def test_bucket_normalizer_preserves_item_object_shape(self):
        item = self.snapshot["buckets"]["item"]["camera"]["content"]
        self.assertEqual([{"attribute": "model", "value": "A7"}], item)
        self.assertEqual(0, self.snapshot["bucket_counts"]["dreaming"])

    def test_memory_overview_is_bounded_and_non_authoritative(self):
        self.assertFalse(self.overview["memory_is_authoritative"])
        self.assertFalse(self.overview["memory_is_qa_fact_source"])
        self.assertLess(self.overview["token_estimate"], 1000)
        self.assertEqual("alice", self.overview["people"][0]["canonical_key"])

    def test_prompt_wrappers_preserve_schemas_and_add_memory(self):
        planner = build_memory_augmented_planner_prompt(
            self.tail, self.overview)
        realizer = build_memory_augmented_realizer_prompt(
            compiled_plan=self.compiled, tail_context=self.tail,
            memory_overview=self.overview)
        mcq = build_memory_augmented_mcq_prompt(
            self.expanded, self.overview, focus_round_index=4)
        for result in (planner, realizer, mcq):
            self.assertIn("NON-AUTHORITATIVE MEMORAX MEMORY", result["prompt"])
            self.assertFalse(result["memory_is_qa_fact_source"])
        self.assertIn("simple_append_planner_v1", planner["prompt"])
        self.assertIn("simple_append_round_realizer_v2", realizer["prompt"])
        self.assertIn("simple_append_mcq_focus_writer_v4", mcq["prompt"])
        self.assertIn("If a Memory hint cannot be located there", mcq["prompt"])
        self.assertIn("Create 8 to 12 grounded", mcq["prompt"])
        self.assertIn("Do not exceed 12 questions", mcq["prompt"])
        self.assertIn("minimal counterfactual", mcq["prompt"])
        self.assertIn("exactly two current_round_only", mcq["prompt"])
        self.assertIn("TRANSCRIPT-GROUNDED DISTRACTOR CONTRACT", mcq["prompt"])
        self.assertIn("same local story arc", mcq["prompt"])
        self.assertIn("omit the candidate instead of inventing them", mcq["prompt"])
        self.assertEqual("compact_dialogue_rows_v1",
                         mcq["dialogue_serialization"])
        self.assertEqual("memorax_qa_topic_index_v1",
                         mcq["memory_prompt_view"])
        self.assertFalse(mcq["memory_overview_full_embedded"])
        self.assertIn("COMPACT DIALOGUE ROW CONTRACT", mcq["prompt"])
        self.assertNotIn(json.dumps(
            self.overview, ensure_ascii=False, separators=(",", ":")),
            mcq["prompt"])
        self.assertIn("answer_dimension", mcq["prompt"])
        self.assertIn("distractor_source_positions", mcq["prompt"])
        self.assertNotIn('"distractor_sources"', mcq["prompt"])

        compact_rows = [
            json.loads(line) for line in mcq["prompt"].splitlines()
            if line.startswith('["D')
        ]
        self.assertEqual(len(self.expanded["turns"]), len(compact_rows))
        compact_by_ref = {row[0]: row[3] for row in compact_rows}
        self.assertEqual(len(compact_rows), len(compact_by_ref))
        self.assertEqual({
            f"D{index:03d}": row["text"]
            for index, row in enumerate(self.expanded["turns"], 1)
        }, compact_by_ref)

    def _option_contract_question(self):
        return {
            "coverage_role": "current_round_only",
            "question_type": "causal",
            "question": "Why did Alice change the plan?",
            "options": {
                "A": "The meeting was moved.",
                "B": "The camera was unavailable.",
                "C": "Bob requested a review.",
                "D": "The room was already booked.",
            },
            "correct_option": "A",
            "evidence_turn_refs": ["D001"],
            "answer_dimension": "cause",
            "distractor_sources": {
                "B": ["D002"], "C": ["D003"], "D": ["D004"],
            },
        }

    def test_option_contract_requires_three_grounded_distractor_sources(self):
        row = self._option_contract_question()
        result = validate_option_contract(
            row, expanded_dialogue=self.expanded)
        self.assertTrue(result["valid"], result)
        value = {
            "mcq_focus_writer_schema_version": "simple_append_mcq_focus_writer_v4",
            "questions": [row, {**row, "question": "Why did Alice revise the plan?"}],
        }
        envelope = normalize_focus_writer_batch(
            value, minimum_current_round_only=2,
            minimum_current_round_joint_history=0, minimum_parent_child=0,
            minimum_general=0, require_option_contract=True)
        self.assertTrue(envelope["valid"], envelope)
        partition = partition_mcq_candidates(
            envelope["batch"], expanded_dialogue=self.expanded,
            minimum_count=1, min_child_supported=0, min_joint_parent_child=0)
        self.assertEqual(2, partition["solver_ready_count"])
        self.assertTrue(all(row["option_contract"]["valid"]
                            for row in partition["items"]))

    def test_option_contract_rejects_missing_or_invalid_provenance(self):
        row = self._option_contract_question()
        del row["distractor_sources"]["C"]
        result = validate_option_contract(
            row, expanded_dialogue=self.expanded)
        self.assertFalse(result["valid"])
        self.assertIn("distractor_sources_keys", result["errors"])
        row = self._option_contract_question()
        row["distractor_sources"]["B"] = ["D999"]
        result = validate_option_contract(
            row, expanded_dialogue=self.expanded)
        self.assertFalse(result["valid"])
        self.assertIn("distractor_source_invalid:B", result["errors"])

    def test_answer_dimension_reasoning_aliases_are_deterministic(self):
        self.assertEqual("state", normalize_answer_dimension("revision"))
        self.assertEqual("constraint", normalize_answer_dimension("dependency"))
        self.assertEqual("information", normalize_answer_dimension("visibility"))
        self.assertEqual("order", normalize_answer_dimension("temporal_sequence"))
        self.assertEqual("comparison", normalize_answer_dimension("synthesis"))
        self.assertEqual("future_slot", normalize_answer_dimension("future_slot"))

    def test_memory_option_contract_requires_enriched_writer_fields(self):
        row = self._option_contract_question()
        row.pop("answer_dimension")
        row.pop("distractor_sources")
        value = {
            "mcq_focus_writer_schema_version": "simple_append_mcq_focus_writer_v4",
            "questions": [row] * 2,
        }
        envelope = normalize_focus_writer_batch(
            value, minimum_current_round_only=0,
            minimum_current_round_joint_history=0, minimum_parent_child=0,
            minimum_general=0, require_option_contract=True)
        self.assertFalse(envelope["valid"])
        self.assertIn("question_01:mcq_focus_writer_schema", envelope["errors"])

    def test_position_first_materialization_binds_gold_and_positions(self):
        raw = {
            "mcq_focus_writer_schema_version": "simple_append_mcq_focus_writer_v4",
            "questions": [{
                "coverage_role": "current_round_only",
                "question_type": "causal",
                "question": "Why did Alice change the plan?",
                "options": {
                    "A": "The meeting was moved.",
                    "B": "The camera was unavailable.",
                    "C": "Bob requested a review.",
                    "D": "The room was already booked.",
                },
                "gold_text": "The meeting was moved.",
                "answer_dimension": "cause",
                "evidence_positions": [1, 2],
                "distractor_source_positions": [[3], [4], [5]],
            }],
        }
        result = materialize_position_first_writer_output(
            raw, expanded_dialogue=self.expanded)
        self.assertEqual([], result["quarantined"])
        row = result["value"]["questions"][0]
        self.assertEqual("A", row["correct_option"])
        self.assertEqual(["D001", "D002"], row["evidence_turn_refs"])
        self.assertEqual({"B": ["D003"], "C": ["D004"], "D": ["D005"]},
                         row["distractor_sources"])

    def test_position_first_materialization_rejects_overlong_or_invalid_positions(self):
        raw = {
            "mcq_focus_writer_schema_version": "simple_append_mcq_focus_writer_v4",
            "questions": [{
                "coverage_role": "current_round_only",
                "question_type": "causal",
                "question": "Why?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "gold_text": "a",
                "answer_dimension": "cause",
                "evidence_positions": list(range(1, 10)),
                "distractor_source_positions": [[0], [2], [3]],
            }],
        }
        result = materialize_position_first_writer_output(
            raw, expanded_dialogue=self.expanded)
        errors = result["quarantined"][0]["errors"]
        self.assertIn("evidence_positions_count_or_duplicate", errors)
        self.assertIn("distractor_source_position_out_of_bounds:1", errors)

    def test_prompt_rejects_cross_root_memory(self):
        bad = dict(self.overview)
        bad["root_id"] = "cc-train-episode-other"
        with self.assertRaisesRegex(ValueError, "root_mismatch"):
            build_memory_augmented_planner_prompt(self.tail, bad)

    def test_client_requires_explicit_plaintext_override(self):
        with self.assertRaisesRegex(ValueError, "plaintext_http_forbidden"):
            MemoraxConfig(base_url="http://example.test", api_key="secret")
        config = MemoraxConfig(
            base_url="http://example.test", api_key="secret",
            allow_insecure_http=True)
        self.assertFalse(config.redacted()["api_key_recorded"])
        self.assertNotIn("secret", repr(config))

    def test_client_sends_memory_platform_without_recording_token(self):
        seen = {}

        def handler(request):
            seen["platform"] = request.headers.get("X-Platform")
            seen["authorization"] = request.headers.get("Authorization")
            return httpx.Response(200, json={
                "success": True, "data": {"buckets": {
                    "person": {}, "location": {}, "event": {},
                    "item": {}, "dreaming": {}}}})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://example.test", api_key="secret"),
            transport=httpx.MockTransport(handler))
        result = client.search_buckets(
            build_bucket_search_payload(root_id=self.root_id)["payload"],
            request_id="fixture-search")
        self.assertTrue(result["success"])
        self.assertEqual("memory", seen["platform"])
        self.assertEqual("Token secret", seen["authorization"])
        self.assertFalse(client.calls[0]["authorization_header_recorded"])
        self.assertNotIn("secret", json.dumps(client.calls))

    def test_failed_transport_is_counted_without_recording_secret(self):
        def handler(request):
            raise httpx.ReadTimeout("fixture timeout", request=request)

        client = MemoraxClient(
            MemoraxConfig(base_url="https://example.test", api_key="secret"),
            transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(MemoraxTransportError, "ReadTimeout"):
            client.search_buckets(
                build_bucket_search_payload(root_id=self.root_id)["payload"],
                request_id="fixture-timeout")
        self.assertEqual(1, len(client.calls))
        self.assertTrue(client.calls[0]["request_attempted"])
        self.assertIsNone(client.calls[0]["http_status"])
        self.assertEqual("ReadTimeout", client.calls[0]["transport_error"])
        self.assertNotIn("secret", json.dumps(client.calls))

    def test_qa_writer_reserved_request_is_ambiguous_not_zero_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = RequestLedger(Path(directory), "manifest", "live")
            ledger.state_dir.mkdir(parents=True, exist_ok=True)
            (ledger.state_dir / "writer.json").write_text(json.dumps({
                "request_id": "writer",
                "budget_class": "mcq_writer",
                "status": "reserved",
                "paid_generation_count": 0,
            }), encoding="utf-8")
            summary = _writer_request_summary(ledger)
        self.assertEqual(1, summary["request_attempt_count"])
        self.assertEqual(0, summary["completed_response_count"])
        self.assertEqual(1, summary["ambiguous_request_count"])
        self.assertEqual(0, summary["paid_generation_count"])

    def test_batch_status_recognizes_qa_pending_recovery(self):
        self.assertEqual("pending_recovery", classify_batch_status([
            {"status": "qa_pending_recovery"},
            {"status": "qa_pending_recovery"},
        ]))
        self.assertEqual("partial", classify_batch_status([
            {"status": "completed"},
            {"status": "qa_pending_recovery"},
        ]))

    def test_client_reads_bare_async_status_contract(self):
        def handler(request):
            self.assertEqual("memory", request.headers.get("X-Platform"))
            self.assertEqual(
                "/v1/memories/add/status/task_fixture_004",
                request.url.path)
            return httpx.Response(200, json={
                "status": "processing", "memory": {}, "error": None})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://example.test", api_key="secret"),
            transport=httpx.MockTransport(handler))
        value = client.add_status(
            "task_fixture_004", request_id="fixture-status")
        self.assertEqual("processing", value["status"])
        self.assertEqual(1, len(client.calls))
        self.assertEqual(200, client.calls[0]["http_status"])
        self.assertNotIn("secret", json.dumps(client.calls))

    def test_zero_network_preflight_passes(self):
        report = run_preflight()
        self.assertEqual("pass", report["status"])
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(all(report["checks"].values()), report)

    def test_recovered_memory_qa_preflight_is_zero_api_and_root_is_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            absent_root = Path(directory) / "fresh-qa-run"
            report = run_qa_preflight(
                source_run=QA_SOURCE_RUN,
                memory_recovery=QA_MEMORY_RECOVERY,
                run_root=absent_root,
                api_config=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL,
            )
        self.assertEqual("pass", report["decision"], report)
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(report["checks"]["saved_prompt_memory_matches"])
        self.assertTrue(report["details"]["upstream_prompt_contract_changed"])
        self.assertTrue(report["checks"]["complete_dialogue_is_qa_fact_source"])
        self.assertTrue(report["checks"]["memory_not_qa_fact_source"])
        self.assertEqual(109, report["details"]["turn_count"])

    def test_recovered_memory_qa_preflight_rejects_existing_run_root(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_qa_preflight(
                source_run=QA_SOURCE_RUN,
                memory_recovery=QA_MEMORY_RECOVERY,
                run_root=Path(directory),
                api_config=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL,
            )
        self.assertEqual("fail", report["decision"])
        self.assertEqual(["run_root_absent"], report["failed_checks"])
        self.assertEqual(0, report["api_calls"])

    def test_qa_focus_round_uses_latest_round_in_dialogue(self):
        expanded = {
            "turns": [
                {"scene_ref": "R01:S01"},
                {"scene_ref": "R02:S03"},
                {"scene_ref": "R02:S01"},
            ]
        }
        self.assertEqual(2, _highest_round_index(expanded))
        self.assertEqual(1, max(1, _highest_round_index({"turns": []})))

    def test_solver_frontier_without_semantic_audit_is_training_candidate(self):
        result = classify_qa_admission_tier({
            "admitted": True,
            "semantic_evidence_audit_executed": False,
        })
        self.assertEqual(
            "training_candidate_unaudited_evidence", result["admission_tier"])
        self.assertEqual(
            "evidence_not_audited", result["evidence_semantic_status"])
        self.assertTrue(result["observation_ready"])
        self.assertTrue(result["training_ready"])

    def test_evidence_mismatch_is_marked_without_blocking_frontier(self):
        result = classify_qa_admission_tier({
            "admitted": True,
            "semantic_evidence_audit_executed": True,
            "answer_semantically_verified": True,
            "submitted_evidence_sufficient": False,
            "evidence_mismatch": True,
        })
        self.assertEqual(
            "training_candidate_evidence_mismatch_marked",
            result["admission_tier"])
        self.assertEqual(
            "evidence_mismatch_marked", result["evidence_semantic_status"])
        self.assertTrue(result["training_ready"])

    def test_executed_answer_audit_failure_is_quarantined(self):
        result = classify_qa_admission_tier({
            "admitted": True,
            "semantic_evidence_audit_executed": True,
            "answer_semantically_verified": False,
            "submitted_evidence_sufficient": True,
        })
        self.assertEqual("semantic_quarantine", result["admission_tier"])
        self.assertFalse(result["training_ready"])

    def test_only_verified_answer_and_evidence_become_training_candidate(self):
        result = classify_qa_admission_tier({
            "admitted": True,
            "semantic_evidence_audit_executed": True,
            "answer_semantically_verified": True,
            "submitted_evidence_sufficient": True,
            "evidence_mismatch": False,
        })
        self.assertEqual("training_candidate", result["admission_tier"])
        self.assertTrue(result["observation_ready"])
        self.assertTrue(result["training_ready"])

    def test_qa_source_adapter_accepts_verified_async_recovery(self):
        result = resolve_qa_artifacts(
            source_run=QA_SOURCE_RUN, memory_artifacts=QA_MEMORY_RECOVERY)
        self.assertEqual("async_recovery", result["source_kind"])
        self.assertTrue(result["readiness"]["memory_write_verified"])
        self.assertTrue(result["readiness"]["qa_prompt_ready"])

    def test_qa_source_adapter_accepts_normal_successful_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory").mkdir()
            (root / "qa").mkdir()
            (root / "reports").mkdir()
            (root / "expanded_dialogue.json").write_bytes(
                (QA_SOURCE_RUN / "expanded_dialogue.json").read_bytes())
            (root / "memory/post_generation_overview.json").write_bytes(
                (QA_MEMORY_RECOVERY /
                 "memory/post_generation_overview.json").read_bytes())
            (root / "qa/mcq_prompt_manifest.json").write_bytes(
                (QA_MEMORY_RECOVERY / "qa/mcq_prompt_manifest.json").read_bytes())
            (root / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json").write_text(
                json.dumps({
                    "status": "completed",
                    "failure_stage": None,
                    "generation_ready": True,
                    "qa_prompt_ready": True,
                    "integrity": {
                        "input_unchanged": True,
                        "source_manifest_unchanged": True,
                        "llm_config_unchanged": True,
                    },
                }), encoding="utf-8")
            result = resolve_qa_artifacts(
                source_run=root, memory_artifacts=root)
        self.assertEqual("normal_single_root", result["source_kind"])
        self.assertTrue(result["readiness"]["memory_write_verified"])

    def test_qa_source_adapter_accepts_incremental_new_turn_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory").mkdir()
            (root / "reports").mkdir()
            (root / "expanded_dialogue.json").write_text(
                json.dumps({"root_id": "fixture-root"}), encoding="utf-8")
            (root / "expanded_dialogue_validation.json").write_text(
                json.dumps({
                    "valid": True,
                    "prior_turn_count": 150,
                    "new_turn_count": 42,
                    "turn_count": 192,
                }), encoding="utf-8")
            (root / "memory/post_generation_overview.json").write_text(
                json.dumps({"root_id": "fixture-root"}), encoding="utf-8")
            (root / "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json").write_text(
                json.dumps({
                    "status": "completed",
                    "generation_ready": True,
                    "memory_write_verified": True,
                }), encoding="utf-8")
            result = resolve_qa_artifacts(
                source_run=root, memory_artifacts=root)
        self.assertEqual("memory_event_continuation", result["source_kind"])
        self.assertTrue(result["readiness"]["qa_prompt_ready"])

    def test_qa_writer_quarantines_one_malformed_question(self):
        valid = {
            "coverage_role": "general",
            "question_type": "causal",
            "question": "Why did A call B?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "correct_option": "A",
            "evidence_turn_refs": ["D001"],
        }
        malformed = {
            key: item for key, item in valid.items()
            if key not in {"correct_option", "evidence_turn_refs"}
        }
        malformed["options"] = {
            **valid["options"],
            "correct_option": "A",
            "evidence_turn_refs": ["D001"],
        }
        value = {
            "mcq_focus_writer_schema_version":
                "simple_append_mcq_focus_writer_v4",
            "questions": [valid, malformed, {**valid, "question": "Second?"}],
        }
        result = _quarantine_invalid_writer_questions(value)
        self.assertTrue(result["applied"])
        self.assertEqual(2, result["retained_question_count"])
        self.assertEqual(1, result["quarantined_question_count"])
        self.assertEqual(2, result["quarantined"][0]["writer_index"])

    def test_qa_source_adapter_accepts_generation_with_pending_memory_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory").mkdir()
            (root / "qa").mkdir()
            (root / "reports").mkdir()
            (root / "expanded_dialogue.json").write_bytes(
                (QA_SOURCE_RUN / "expanded_dialogue.json").read_bytes())
            (root / "memory/post_generation_overview.json").write_bytes(
                (QA_MEMORY_RECOVERY /
                 "memory/post_generation_overview.json").read_bytes())
            (root / "qa/mcq_prompt_manifest.json").write_bytes(
                (QA_MEMORY_RECOVERY / "qa/mcq_prompt_manifest.json").read_bytes())
            (root / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json").write_text(
                json.dumps({
                    "status": "completed",
                    "failure_stage": None,
                    "generation_ready": True,
                    "qa_prompt_ready": True,
                    "memory_write_verified": False,
                    "memory_context_degraded": True,
                    "integrity": {
                        "input_unchanged": True,
                        "source_manifest_unchanged": True,
                        "llm_config_unchanged": True,
                    },
                }), encoding="utf-8")
            result = resolve_qa_artifacts(source_run=root, memory_artifacts=root)
        self.assertFalse(result["readiness"]["memory_write_verified"])
        self.assertTrue(result["readiness"]["memory_context_degraded"])
        self.assertTrue(result["readiness"]["qa_prompt_ready"])

    def test_two_root_batch_preflight_is_zero_api_and_creates_no_run(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "absent-batch"
            report = run_batch_preflight(batch_root=batch_root)
            self.assertFalse(batch_root.exists())
        self.assertEqual("passed", report["status"], report)
        self.assertEqual(2, report["totals"]["root_count"])
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(report["checks"]["memorax_user_ids_unique"])
        self.assertTrue(all(row["ready"] for row in report["roots"]))
        self.assertEqual(0, report["totals"]["memorax_status_max"])
        self.assertTrue(all(
            row["parent_memory_mode"] == "initialize_async_pending"
            for row in report["roots"]))

    def test_five_root_batch_preflight_is_zero_api_and_creates_no_run(self):
        roots = tuple(
            f"cc-train-episode-{episode}" for episode in
            (216742, 34406, 41409, 41625, 91716))
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "absent-five-root-batch"
            report = run_batch_preflight(
                batch_root=batch_root, root_ids=roots)
            self.assertFalse(batch_root.exists())
        self.assertEqual("passed", report["status"], report)
        self.assertEqual(5, report["totals"]["root_count"])
        self.assertEqual(15, report["totals"]["paid_llm_request_budget"])
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(
            report["checks"]["root_count_within_supported_range"])
        self.assertTrue(report["checks"]["memorax_user_ids_unique"])
        self.assertTrue(all(row["ready"] for row in report["roots"]))

    def test_frozen_input_manifest_supports_compatibility_paths_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "compat.json"
            input_path.write_text(json.dumps({"root_id": "fixture-root"}))
            manifest_path = root / "inputs.json"
            manifest_path.write_text(json.dumps({
                "manifest_version": INPUT_MANIFEST_VERSION,
                "roots": {"fixture-root": str(input_path)},
            }))
            loaded = load_input_manifest(manifest_path)
        self.assertEqual(input_path.resolve(), loaded["fixture-root"])

    def test_batch_preflight_accepts_explicit_input_path_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "absent-manifest-batch"
            report = run_batch_preflight(
                batch_root=batch_root,
                root_ids=("cc-train-episode-61555",),
                input_paths={"cc-train-episode-61555": PARENT_PATH},
            )
            self.assertFalse(batch_root.exists())
        self.assertEqual("passed", report["status"], report)
        self.assertEqual("simple_append_memorax_input_manifest_v1",
                         report["input_manifest_version"])
        self.assertEqual(0, report["api_calls"])

    def test_batch_status_describes_pipeline_outcome_not_process_exit(self):
        self.assertEqual("completed", classify_batch_status([
            {"status": "completed"}, {"status": "completed"}]))
        self.assertEqual("partial", classify_batch_status([
            {"status": "completed"}, {"status": "failed"}]))
        self.assertEqual("pending_recovery", classify_batch_status([
            {"status": "pending_recovery"}, {"status": "failed"}]))
        self.assertEqual("failed", classify_batch_status([
            {"status": "qa_failed"}, {"status": "failed"}]))

    def test_batch_exit_code_fails_for_any_incomplete_batch(self):
        self.assertEqual(0, batch_exit_code({"status": "completed"}))
        for status in ("partial", "pending_recovery", "failed"):
            self.assertEqual(1, batch_exit_code({"status": status}))

    def test_solver_load_failure_precedes_run_root_and_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory) / "must-remain-absent"
            with patch(
                "scripts.current_method.restart_n5_spell."
                "simple_append_memorax_v1.batch_live.run_preflight",
                return_value={"status": "passed"},
            ), patch(
                "scripts.current_method.restart_n5_spell."
                "simple_append_memorax_v1.batch_live."
                "build_local_solver_runtime",
                side_effect=RuntimeError("fixture_solver_load_failed"),
            ), patch(
                "scripts.current_method.restart_n5_spell."
                "simple_append_memorax_v1.batch_live._execute_batch",
            ) as execute:
                with self.assertRaisesRegex(
                        RuntimeError, "fixture_solver_load_failed"):
                    run_batch_live(
                        batch_root=batch_root, root_ids=("fixture-root",),
                        generation_registry=Path(directory) / "registry.json")
            self.assertFalse(batch_root.exists())
            execute.assert_not_called()

    def test_solver_is_closed_after_batch_execution(self):
        backend = Mock()
        summary = {"status": "completed"}
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.current_method.restart_n5_spell."
            "simple_append_memorax_v1.batch_live.run_preflight",
            return_value={
                "status": "passed",
                "roots": [{
                    "root_id": "fixture-root",
                    "input_sha256": "a" * 64,
                }],
            },
        ), patch(
            "scripts.current_method.restart_n5_spell."
            "simple_append_memorax_v1.batch_live."
            "build_local_solver_runtime",
            return_value=(Mock(), backend),
        ), patch(
            "scripts.current_method.restart_n5_spell."
            "simple_append_memorax_v1.batch_live._execute_batch",
            return_value=summary,
        ):
            result = run_batch_live(
                batch_root=Path(directory) / "fixture-batch",
                root_ids=("fixture-root",),
                generation_registry=Path(directory) / "registry.json")
        self.assertEqual(summary, result)
        backend.close.assert_called_once_with()

    def test_generation_registry_rejects_same_identity_before_second_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            identity = build_generation_identity(
                root_id="fixture-root", parent_sha256="a" * 64,
                round_index=1, generation_implementation_sha256="b" * 64,
                api_config_sha256="c" * 64,
                generation_configuration={"runner": "fixture"},
                memorax_namespace_sha256="d" * 64)
            reserve_generation(
                registry_path=registry, identities={"fixture-root": identity},
                run_root=root / "run-1")
            with self.assertRaises(GenerationAlreadyRegistered):
                reserve_generation(
                    registry_path=registry,
                    identities={"fixture-root": identity},
                    run_root=root / "run-2")
            changed = build_generation_identity(
                root_id="fixture-root", parent_sha256="a" * 64,
                round_index=1, generation_implementation_sha256="e" * 64,
                api_config_sha256="f" * 64,
                generation_configuration={"runner": "qa-only-change"},
                memorax_namespace_sha256="d" * 64)
            with self.assertRaises(GenerationAlreadyRegistered):
                reserve_generation(
                    registry_path=registry,
                    identities={"fixture-root": changed},
                    run_root=root / "run-3")

    def test_generation_registry_keeps_project_namespaces_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.env"
            second = root / "second.env"
            first.write_text("MEMORAX_API_KEY=project-a\n", encoding="utf-8")
            second.write_text("MEMORAX_API_KEY=project-b\n", encoding="utf-8")
            first.chmod(0o600)
            second.chmod(0o600)
            self.assertNotEqual(
                memorax_namespace_sha(first), memorax_namespace_sha(second))

    def test_generation_registry_finalizes_pending_memory_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            identity = build_generation_identity(
                root_id="fixture-root", parent_sha256="a" * 64,
                round_index=1, generation_implementation_sha256="b" * 64,
                api_config_sha256="c" * 64,
                generation_configuration={"runner": "fixture"},
                memorax_namespace_sha256="d" * 64)
            reserve_generation(
                registry_path=registry, identities={"fixture-root": identity},
                run_root=root / "run-1")
            finalize_generation(
                registry_path=registry, identities={"fixture-root": identity},
                rows=[{
                    "root_id": "fixture-root",
                    "status": "completed",
                    "expansion": {
                        "generation_ready": True,
                        "pending_memory_writes": [{"task_id": "task"}],
                        "memory_write_verified": False,
                        "run_root": str(root / "run-1"),
                    },
                }])
            value = json.loads(registry.read_text(encoding="utf-8"))
            entry = next(iter(value["entries"].values()))
            self.assertEqual(
                "generation_complete_memory_pending", entry["state"])
            self.assertFalse(entry["automatic_regeneration_allowed"])
            mark_memory_status(
                registry_path=registry,
                status_by_root={
                    "fixture-root": {
                        "parent_add": "success",
                        "child_add": "success",
                    },
                })
            value = json.loads(registry.read_text(encoding="utf-8"))
            entry = next(iter(value["entries"].values()))
            self.assertEqual(
                "generation_complete_memory_verified", entry["state"])
            self.assertTrue(entry["memory_write_verified"])
            self.assertFalse(entry["memory_recovery_required"])

    def test_batch_checkpoint_persists_current_root_and_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint(
                batch_root=root,
                rows=[{"root_id": "root-a", "status": "completed"}],
                root_count=3,
                current_root_id="root-b",
            )
            value = json.loads((root / "batch_status.json").read_text())
        self.assertEqual("running", value["status"])
        self.assertEqual(1, value["processed_root_count"])
        self.assertEqual(1, value["completed_root_count"])
        self.assertEqual("root-b", value["current_root_id"])


if __name__ == "__main__":
    unittest.main()
