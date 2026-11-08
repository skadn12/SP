"""Offline tests for stage sequencing and source-bound reporting, no models."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import types
import unittest
from unittest.mock import patch

import run_collection as collection
import finalize_report as report


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="v62-orchestration-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        blocker = patch("socket.socket", side_effect=AssertionError("No model calls in offline tests"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_focus_can_reuse_real_scene_but_cannot_invent_missing_scene(self):
        gen, _ = collection.modules()
        path = self.home / "focus.json"
        config = {**collection.CONFIG, "focus_topics": str(path)}
        value = {"root_id": collection.ROOT, "chunks": [{"index": index, "slots_by_type": {kind: [
            {"preferred_scene_refs": ["CANDIDATE:A:S01"], "focus": f"Focus {index} / {kind} / {slot}"}
            for slot in range(2)] for kind in ("multi_select",)} } for index in range(1, 4)]}
        path.write_text(json.dumps(value))
        turns = [{"turn_ref": "CANDIDATE:A:S01:T001", "scene_ref": "CANDIDATE:A:S01"}]
        with patch.object(collection, "CONFIG", config):
            chunks = collection.focus_chunks(gen, turns)
            self.assertEqual(len(chunks), 6)
            self.assertEqual(sum(len(c["candidate_ids"]) for c in chunks), 12)
            self.assertEqual(collection.QUESTION_TYPES, ("multi_select",))
            self.assertEqual((collection.MAX_HTTP, collection.MAX_SOLVER), (15, 216))
            value["chunks"][0]["slots_by_type"]["multi_select"][0]["preferred_scene_refs"] = ["CANDIDATE:B:S03"]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "scene_not_in_full_source"):
                collection.focus_chunks(gen, turns)

    def test_evaluation_cannot_begin_or_invoke_solver_without_frozen_review(self):
        with patch.object(collection, "verify_plan", return_value={"chunks": []}), \
             patch.object(collection, "completion", return_value={}), \
             patch.object(collection, "verify_review", side_effect=FileNotFoundError("audit missing")), \
             patch.object(collection, "begin") as begin, patch.object(collection, "command") as command:
            with self.assertRaises(FileNotFoundError):
                collection.evaluate()
            begin.assert_not_called()
            command.assert_not_called()

    def fixture_review(self):
        run = self.home / "generation"
        run.mkdir()
        question = {"candidate_id": "q1", "root_id": collection.ROOT,
                    "question_type": "multi_select", "source_sha256": "a" * 64}
        (run / "MECHANICAL_POOL.json").write_text(json.dumps({"questions": [question]}))
        (self.home / "generation.COMPLETE.json").write_text("{}")
        audit = {"solver_scores_used": False, "questions": [{**question, "core_status": "CORE_USABLE",
            "mechanical_pool_sha256": collection.binding(run / "MECHANICAL_POOL.json")["sha256"]}]}
        (self.home / "INDEPENDENT_CONTENT_AUDIT.json").write_text(json.dumps(audit))
        return run, question, audit

    def test_review_requires_exact_candidate_set_and_source_hash(self):
        run, question, audit = self.fixture_review()
        with patch.object(collection, "REPORT", self.home), \
             patch.object(collection, "completion", return_value={}), \
             patch.object(collection, "verify_plan", return_value={"chunks": [{"generation_run": str(run)}]}):
            self.assertEqual(collection.reviewed_content()["reviewed_questions"], 1)
            audit["questions"][0]["source_sha256"] = "b" * 64
            (self.home / "INDEPENDENT_CONTENT_AUDIT.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "source_binding"):
                collection.reviewed_content()
            audit["questions"] = []
            (self.home / "INDEPENDENT_CONTENT_AUDIT.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "complete_mechanical_pool"):
                collection.reviewed_content()

    def test_review_binds_exact_mechanical_question_content(self):
        run, question, audit = self.fixture_review()
        with patch.object(collection, "REPORT", self.home), \
             patch.object(collection, "completion", return_value={}), \
             patch.object(collection, "verify_plan", return_value={"chunks": [{"generation_run": str(run)}]}):
            question["question"] = "Changed proposition text"
            (run / "MECHANICAL_POOL.json").write_text(json.dumps({"questions": [question]}))
            with self.assertRaisesRegex(ValueError, "mechanical_pool_binding"):
                collection.reviewed_content()

    def test_empty_mechanical_review_and_empty_question_exports(self):
        run, question, audit = self.fixture_review()
        (run / "MECHANICAL_POOL.json").write_text(json.dumps({"questions": []}))
        audit["questions"] = []
        (self.home / "INDEPENDENT_CONTENT_AUDIT.json").write_text(json.dumps(audit))
        with patch.object(collection, "REPORT", self.home), \
             patch.object(collection, "completion", return_value={}), \
             patch.object(collection, "verify_plan", return_value={"chunks": [{"generation_run": str(run)}]}):
            self.assertEqual(collection.reviewed_content()["reviewed_questions"], 0)
        with patch.object(report, "HERE", self.home), patch.object(report, "OUTPUTS", []):
            views = report.export_questions("EMPTY", [], "Empty selected", {})
            self.assertEqual(json.loads((self.home / "EMPTY_CANDIDATES.json").read_text())["questions"], [])
            self.assertEqual(json.loads((self.home / "EMPTY_DISPLAY.json").read_text())["questions"], [])
            self.assertIsNone(report.distribution([])["all_six_rate_percent"])

    def test_expansion_summary_accepts_actual_v51_report_schema(self):
        history = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v51/runs/root138343_a3b3_01/CANDIDATE_HISTORY.json")
        summary, turns = report.expansion_summary({"root_id": "cc-train-episode-138343", "history": {"path": str(history)}})
        self.assertEqual(summary["formal_original_plus_r01"], {"turns": 106, "body_tokens": 2116})
        self.assertEqual(summary["retained_by_route"]["A"], {"scenes": 3, "turns": 38, "body_tokens": 1321})
        self.assertEqual(summary["retained_by_route"]["B"], {"scenes": 3, "turns": 37, "body_tokens": 1413})
        self.assertEqual(summary["full_provisional_cumulative"], {"turns": 181, "body_tokens": 4850})
        self.assertEqual(len(turns), 181)


    def test_target_v57_expansion_metrics_exclude_rejected_and_quarantined_scenes(self):
        summary, turns = report.expansion_summary({"root_id": collection.ROOT,
            "history": {"path": str(collection.HISTORY)}})
        self.assertEqual(summary["formal_original_plus_r01"], {"turns": 92, "body_tokens": 2171})
        self.assertEqual(summary["retained_by_route"]["A"], {"scenes": 2, "turns": 29, "body_tokens": 1086})
        self.assertEqual(summary["retained_by_route"]["B"], {"scenes": 2, "turns": 28, "body_tokens": 970})
        self.assertEqual(summary["full_provisional_cumulative"], {"turns": 149, "body_tokens": 4227})
        self.assertEqual(len(turns), 149)
        self.assertEqual(summary["retained_scenes_by_route"], {"A": ["CANDIDATE:A:S01", "CANDIDATE:A:S02"],
            "B": ["CANDIDATE:B:S01", "CANDIDATE:B:S03"]})
        self.assertFalse(summary["new_expansion_in_this_experiment"])

    def test_configured_baseline_uses_actual_v61_50_question_pool(self):
        baseline = json.loads(Path(collection.CONFIG["baseline_usable_pool"]).read_text())["questions"]
        counts = report.distribution(baseline)
        expected = collection.CONFIG["expected_baseline_summary"]
        self.assertEqual(len(baseline), expected["questions"])
        self.assertEqual(counts["multi_select"], expected["multi_select"])
        self.assertEqual(counts["questions_by_type"]["ordering"], expected["ordering"])
        self.assertEqual({str(k): v for k, v in counts["multi_correct_count_histogram"].items()},
                         expected["multi_correct_count_histogram"])
        self.assertEqual(counts["correct_options"], expected["correct_options"])
        self.assertEqual(counts["total_options"], expected["total_options"])

    def test_frozen_review_cannot_be_changed_after_freeze(self):
        value = {"audit": {"path": "audit", "sha256": "a" * 64}}
        (self.home / "CONTENT_REVIEW.COMPLETE.json").write_text(json.dumps({
            "version": collection.VERSION, "before_any_solver": True, **value}))
        with patch.object(collection, "REPORT", self.home), patch.object(collection, "reviewed_content", return_value=value):
            self.assertEqual(collection.verify_review()["audit"], value["audit"])
        changed = deepcopy(value)
        changed["audit"]["sha256"] = "b" * 64
        with patch.object(collection, "REPORT", self.home), patch.object(collection, "reviewed_content", return_value=changed):
            with self.assertRaisesRegex(ValueError, "frozen_content_review_changed"):
                collection.verify_review()

    def test_full_conversation_retains_every_turn_text(self):
        turns = [{"turn_ref": "P:T1", "speaker": "A", "text": "First\nsecond."},
                 {"turn_ref": "CANDIDATE:B:S01:T001", "speaker": "B", "text": "Quoted `value` & more."}]
        before = deepcopy(turns)
        rendered = report.conversation_markdown("fixture", turns)
        self.assertEqual(turns, before)
        self.assertTrue(all(turn["text"] in rendered for turn in turns))
        self.assertLess(rendered.index(turns[0]["text"]), rendered.index(turns[1]["text"]))


class ComparisonAndPriorTests(unittest.TestCase):
    def test_original_mixed_responses_and_aggregates_replay_independently(self):
        import comparison
        run = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v55_qa/runs/root135466_collect20_01_c01_evaluation/solver')
        manifest = json.loads((run / 'manifest.json').read_text())
        gold = json.loads((run / 'gold.json').read_text())
        counts = {'multi_select': 0, 'ordering': 0}
        values, requests = {}, {}
        for request in manifest['requests']:
            path = run / 'responses' / (request['request_id'] + '.json')
            response = json.loads(path.read_text())
            response['response_path'] = str(path)
            fn = comparison.independent_mc_score if request['question_type'] == 'multi_select' else comparison.independent_order_score
            actual = fn(response, request, gold[request['candidate_id']])
            self.assertEqual(actual['partial'], response['score']['partial_score_fraction'])
            counts[request['question_type']] += 1
            values[request['request_id']] = actual
            requests[request['request_id']] = request
        self.assertEqual(counts, {'multi_select': 18, 'ordering': 24})
        comparison.verify_result_aggregates(run, values, requests)

    def test_empty_comparison_is_explicit_not_divide_by_zero(self):
        import comparison
        value = comparison.compare([])
        self.assertEqual(value['base']['calls'], 0)
        self.assertIsNone(value['base']['exact_rate'])
        self.assertIsNone(value['trained_minus_base_exact_percentage_points'])
        text = comparison.markdown({'primary': value, 'all_ready_diagnostic': value,
            'trained_selected_core_usable_secondary': value, 'primary_by_type': comparison.compare_by_type([]), 'questions': []})
        self.assertIn('0/0', text)

    def test_empty_base_chunk_needs_explicit_skipped_marker(self):
        import run_paired_base
        with tempfile.TemporaryDirectory() as temp:
            trained, base = Path(temp) / 'trained', Path(temp) / 'base'
            trained.mkdir(); base.mkdir()
            (trained / 'SKIPPED.json').write_text(json.dumps({'candidate_count': 0}))
            with self.assertRaises(FileNotFoundError):
                run_paired_base.paired_chunk('c01', trained, base)
            (base / 'SKIPPED.json').write_text(json.dumps({'candidate_count': 0}))
            self.assertEqual(run_paired_base.paired_chunk('c01', trained, base, complete=True)['base_actual_calls'], 0)

    def test_copy_snapshot_uses_frozen_changed_copied_files_schema(self):
        snapshot = json.loads((collection.CODE / "COPY_BINDINGS.json").read_text())
        self.assertEqual(set(snapshot["changed_copied_files"]), {"generate.py"})
        self.assertEqual(collection.verify_copy_snapshot()["path"],
                         str(collection.CODE / "COPY_BINDINGS.json"))

    def test_target_root_mechanical_pools_are_in_new_public_prior(self):
        paths = collection.historical_paths()
        v57 = [p for p in paths if "h4_sequential_ab_v57_qa" in str(p)]
        v58 = [p for p in paths if "h4_sequential_ab_v58_ordering_qa" in str(p)]
        self.assertEqual((len(v57), len(v58), len(paths)), (6, 4, 13))
        self.assertTrue(all(p.name == "MECHANICAL_POOL.json" for p in paths))
        self.assertEqual(collection.CONFIG["baseline_usable_pool"].split("/")[-2], "h4_multiselect_probe_v61")
        for path in paths:
            self.assertTrue(any(q["root_id"] == collection.ROOT for q in json.loads(path.read_text())["questions"]))

    def test_public_prior_hides_gold_and_scores(self):
        q = {'candidate_id': 'x', 'root_id': collection.ROOT, 'question_type': 'ordering',
             'question': 'When?', 'items': {f'I{i}': f'Event {i}' for i in range(1, 7)},
             'gold': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6'], 'score': 0.7, 'item_evidence': {'private': True}}
        self.assertEqual(set(collection.public_question(q)), {'candidate_id', 'root_id', 'question_type', 'question', 'items'})

    def test_mc_false_positive_penalty_and_all_six_gold(self):
        import comparison
        request = {'variant': {'display_to_content': dict(zip('ABCDEF', range(6)))}}
        response = {'raw_output': '<box>A,B,C,D,E,F</box>', 'response_path': 'fixture',
                    'score': {'exact_match': False, 'partial_score_fraction': '0', 'tp': 2, 'fp': 4, 'fn': 0}}
        self.assertEqual(comparison.independent_mc_score(response, request, [0, 1])['partial'], '0')
        response['score'] = {'exact_match': True, 'partial_score_fraction': '1', 'tp': 6, 'fp': 0, 'fn': 0}
        self.assertEqual(comparison.independent_mc_score(response, request, list(range(6)))['partial'], '1')


class MethodComparisonTests(unittest.TestCase):
    def test_same_topics_and_fixed_prior_for_both_arms(self):
        gen, _ = collection.modules()
        turns = json.loads(collection.HISTORY.read_text())["turns"]
        chunks = collection.focus_chunks(gen, turns)
        self.assertEqual([c["chunk_name"] for c in chunks],
            ["direct_c01", "staged_c01", "staged_c02", "direct_c02", "direct_c03", "staged_c03"])
        self.assertEqual(len({collection.cumulative_prior_path(c["index"]) for c in chunks}), 1)
        self.assertEqual(len({qid for c in chunks for qid in c["candidate_ids"]}), 12)
        for n in range(1, 4):
            pair = [list(c["slot_focus_by_candidate_id"].values()) for c in chunks if c["topic_chunk"] == n]
            self.assertEqual(pair[0], pair[1])

    def test_planner_usage_missing_is_unknown_not_free(self):
        rows = [{"remote_http_attempt": True, "token_usage_status": "known", "reported_input_tokens": 100,
                 "reported_output_tokens": 20},
                {"remote_http_attempt": True, "token_usage_status": "unknown", "reported_input_tokens": None,
                 "reported_output_tokens": None}]
        result = report.cost_summary(rows, 2)
        self.assertEqual(result["http_attempts"], 2)
        self.assertEqual(result["input_tokens"], "unknown")
        self.assertEqual(result["known_input_token_subtotal"], 100)
        self.assertEqual(result["http_per_content_usable_question"], "1")
        self.assertEqual(result["input_tokens_per_content_usable_question"], "unknown")
        self.assertIsNone(report.cost_summary(rows, 0)["http_per_content_usable_question"])

    def test_exact_dedup_is_same_root_and_keeps_truth_metadata(self):
        baseline = json.loads(Path(collection.CONFIG["baseline_usable_pool"]).read_text())["questions"]
        q = deepcopy(next(q for q in baseline if q["question_type"] == "multi_select"))
        other = deepcopy(q); other["candidate_id"] = "other-arm"
        kept, duplicated = report.deduplicate_exact([q, other])
        self.assertEqual(kept, [q])
        self.assertEqual(duplicated[0]["candidate_id"], "other-arm")
        self.assertFalse(duplicated[0]["gold_content_disagreement"])
        other["root_id"] = "different-root"
        self.assertEqual(len(report.deduplicate_exact([q, other])[0]), 2)


class GenerationAuditIntegrationTests(unittest.TestCase):
    def test_staged_success_repair_and_failed_planner_audit_offline(self):
        import audit_provenance as audit
        import test_generation_method_v62 as fixtures
        for mode in ("success", "repair", "bad_plan", "failed_http"):
            with self.subTest(mode=mode):
                fixture = fixtures.GenerationMethodTests()
                fixture.setUp()
                try:
                    flags = {mode: True} if mode != "success" else {}
                    run, generated, _ = fixture.exercise(**flags)
                    for index, folder in enumerate(sorted((run / "requests").iterdir()), 1):
                        proof_path = folder / "PROOF.json"
                        proof = json.loads(proof_path.read_text())
                        proof["capacity"] = {"status": "PASS", "required_context_tokens": 1000}
                        proof_path.write_text(json.dumps(proof))
                        attempt_path = folder / "HTTP_ATTEMPT.json"
                        attempt = json.loads(attempt_path.read_text()); attempt['transport_retries'] = 0
                        attempt_path.write_text(json.dumps(attempt))
                        failed = (folder / 'ERROR.json').exists()
                        (folder / 'HTTP_RESULT.json').write_text(json.dumps({
                            "http_completed_at_utc": f"2026-10-12T12:0{index}:01+00:00",
                            "outcome": "error" if failed else "response"}))
                        (folder / 'THROTTLE.json').write_text(json.dumps({
                            "previous_completion": None, "http_started_at_utc": f"2026-10-12T12:0{index}:00+00:00"}))
                    bound = audit.bind(fixture.fixture.history)
                    plan = {'roots': [{'root_id': fixture.root, 'chunk': mode, 'history': bound, 'prior': bound,
                            'generation_run': str(run)}]}
                    checked = audit.generation_audit(plan)
                    self.assertEqual(checked['counts']['http_requests'], generated['http_attempts'])
                    drafts, requests, errors = report.request_details(run, fixture.root, 1)
                    self.assertEqual(len(drafts), 0 if mode in ('bad_plan', 'failed_http') else (3 if mode == 'repair' else 2))
                    self.assertEqual(sum(r['role'] == 'evidence_planner' for r in requests), 1)
                    self.assertEqual(report.cost_summary(requests)['usage_complete'], mode != 'failed_http')
                finally:
                    fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
