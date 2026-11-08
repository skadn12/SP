"""Offline regression coverage for actual v56 prompts and MC-only exports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_and_pool as pipeline
import generate
import prompt_layout
import test_generate_adapter as fixtures


class MultiselectPromptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_actual_payload_keeps_source_feedback_and_soft_preference(self):
        run = self.fixture.prepare()
        root = self.fixture.root
        manifest = generate.verify(run)
        context = generate.initial_context(run, root, 'multi_select')
        payload, receipt = generate.make_payload(run, root, 'multi_select', context)
        text = payload['messages'][0]['content']
        self.assertEqual(payload['max_tokens'], 4096)
        self.assertEqual(manifest['question_types'], ['multi_select'])
        self.assertEqual(manifest['budget'], {'candidate_slots': 2,
            'initial_generation_requests': 1, 'mechanical_repair_requests_max': 1,
            'ordering_review_requests_max': 0, 'http_attempts_max': 2,
            'solver_full_calls_max': 36})
        self.assertEqual(fixtures.input_json(payload, '\nINPUT_JSON\n')['source_turns'], self.fixture.turns)
        self.assertIn('preferentially seek 2, 3 or 4 supports', text)
        self.assertIn('soft sampling preference, not a quota or validity gate', text)
        self.assertIn('5 or 6 supports remain valid', text)
        self.assertNotIn('Exactly 2, 3 or 4 options must be supports', text)
        self.assertIn('Never relabel a true option or omit a supported Gold label', text)
        self.assertIn('Unmentioned, unconfirmed or', text)
        self.assertIn('they are not false options', text)
        suffix = (run / 'feedback' / root / 'writer__M_clean_policy_feedback.txt').read_text()
        intermediate, kind, tail = prompt_layout.split_task_tail(text, expected_kind='multi_select')
        self.assertEqual(kind, 'multi_select')
        self.assertTrue(intermediate.endswith(suffix))
        self.assertIn('never rejects five or six supports', tail)
        self.assertIn('retain fully evidenced five- or six-support', tail)
        self.assertNotIn('or report abstention if the', tail)
        prompt_layout.verify_request(payload, receipt, suffix, root, generate.TARGET,
            generate.base.digest((run / 'feedback/source_lock.json').read_bytes()),
            expected_kind='multi_select')

    def test_budget_rejects_other_slot_counts_before_live_calls(self):
        for roots, slots in ((1, 1), (1, 4), (2, 2)):
            with self.subTest(roots=roots, slots=slots), self.assertRaises(ValueError):
                generate.generation_budget(roots, slots)


class MultiselectDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v56-offline-delivery-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        blocker = patch('socket.socket', side_effect=AssertionError('No model calls in regression tests'))
        blocker.start()
        self.addCleanup(blocker.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def exercise(self, *, candidate, selected):
        gen, solver, target = (self.home / name for name in ('generation', 'solver', 'delivery'))
        question = {'candidate_id': 'q1', 'root_id': 'fixture', 'question_type': 'multi_select',
            'question': 'Which statements are supported?', 'options': {k: 'Fact ' + k for k in 'ABCDEF'},
            'gold': list('ABCDEF'), 'source_path': '/synthetic/source.json', 'source_sha256': 'a' * 64}
        original = deepcopy(question)
        questions = [question] if candidate else []
        calls = 18 if candidate else 0
        result = {'candidate_id': 'q1', 'selected_by_policy': selected,
            'actual_full_calls': 18, 'expected_full_calls': 18,
            'full_partial_mean': '1/2' if selected else '1'}
        self.put(gen / 'REPORT.json', {'ready_for_solver': len(questions),
            'roots': [{'root_id': 'fixture', 'question_type': 'multi_select', 'candidate_slots': 2,
                'mechanical_pass': len(questions), 'ready_for_solver': len(questions),
                'abstentions': [], 'mechanical_rejected': [], 'review_rejected': []}],
            'candidate_slots': 2, 'feedback_in_every_request': True, 'http_attempts': 1})
        self.put(gen / 'SOLVER_POOL.json', {'questions': questions,
            'provenance': {'requested_model': 'synthetic', 'ordering_review_owner': 'synthetic'}})
        self.put(solver / 'manifest.json', {'pool_sha256': pipeline.delivery.sha(gen / 'SOLVER_POOL.json'),
            'request_count': calls})
        self.put(solver / 'COMPLETE.json', {'actual_call_count': calls})
        self.put(solver / 'solver_identity.json', {'synthetic_test_only': True})
        self.put(solver / 'RESULTS.json', {'call_set_errors': [], 'questions': [result] if candidate else [],
            'solver_model': 'synthetic', 'model_path': '/synthetic/model', 'policy_version': 'synthetic'})
        report = pipeline.delivery.deliver(gen, solver, target)
        self.assertEqual(Path(pipeline.delivery.__file__).resolve(), Path(__file__).with_name('deliver.py').resolve())
        self.assertNotIn('ordering', report['per_root']['fixture'])
        self.assertEqual(report['selected_ordering'], 0)
        self.assertEqual(report['full_solver_calls'], calls)
        self.assertEqual(report['selected_total'], int(candidate and selected))
        self.assertIn('| 0 / 0 |', (target / 'REPORT.md').read_text())
        delivered = pipeline.delivery.read(target / 'SELECTED_QUESTIONS.json')['questions']
        if selected:
            self.assertEqual(delivered[0]['gold'], original['gold'])
            self.assertEqual(delivered[0]['options'], original['options'])
        else:
            self.assertEqual(delivered, [])

    def test_mc_only_nonempty_selection(self):
        self.exercise(candidate=True, selected=True)

    def test_mc_only_empty_selection_from_evaluated_candidate(self):
        self.exercise(candidate=True, selected=False)

    def test_empty_candidate_pool(self):
        self.exercise(candidate=False, selected=False)


if __name__ == '__main__':
    unittest.main()
