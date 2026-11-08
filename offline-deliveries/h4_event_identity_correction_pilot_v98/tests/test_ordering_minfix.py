"""Meaningful regressions for v96 observed output/prior failures; no model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HOME = Path(__file__).resolve().parents[1]
ENGINE = HOME / 'qa_engine'
OLD = HOME.parent / 'h4_v95_ready_mixed_qa_v96'
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ENGINE))
import actor_reference
import batch_scope as scope
import ordering_plan
import ordering_prior
import ordering_generation

builder_spec = importlib.util.spec_from_file_location('v98_minfix_builder', HOME / 'evidence/build_fresh_shard.py')
builder = importlib.util.module_from_spec(builder_spec); builder_spec.loader.exec_module(builder)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fixture_plan(home):
    roots = ['cc-train-episode-' + str(n) for n in (11, 22, 33, 44)]
    run = home / 'expansion_engine/runs/four_roots_B_scene01'
    for name, value in [('MANIFEST.json', {'roots': roots}),
                        ('REPORT.json', {'routes': [{'root_id': r, 'route': 'dataset_new_event', 'accepted_scenes': 0} for r in roots]}),
                        ('FINAL_AUDIT.json', {'status': 'PASS', 'failures': [], 'semantic_certified': False})]:
        save(run / name, value)
    return {'version': scope.PLAN_VERSION, 'candidate_root_budget': 4, 'candidate_roots': roots,
            'source_split': 'train', 'training_allowed': False, 'formal_training_admission': False,
            'question_types': ['multi_select', 'ordering'], 'max_generation_http_total': 12,
            'max_solver_calls_per_model_total': 456,
            'upstream_expansion_run': {name: scope.bind(run / name) for name in ('MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json')}}


class ReadySubsetScopeTests(unittest.TestCase):
    def test_four_preselected_roots_allow_all_ordered_ready_subsets_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); plan = fixture_plan(home); roots = plan['candidate_roots']
            with patch.object(scope, 'PILOT', home):
                scope.validate_plan(plan)
                for selected in (roots, roots[:1], roots[1:3], [roots[0], roots[-1]]):
                    scope.validate_roots(selected, plan)
                scope.validate_roots([], plan, allow_empty=True)
                for selected in ([], roots[::-1], roots[:1] * 2, ['cc-train-episode-99']):
                    with self.assertRaises(ValueError): scope.validate_roots(selected, plan)

    def test_upstream_binding_and_budget_tampering_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); plan = fixture_plan(home)
            with patch.object(scope, 'PILOT', home):
                for changes in ({'max_generation_http_total': 13}, {'max_solver_calls_per_model_total': 457},
                                {'candidate_root_budget': 3}, {'training_allowed': True}):
                    with self.assertRaises(ValueError): scope.validate_plan({**plan, **changes})
                save(scope.upstream_run(plan) / 'REPORT.json', {'routes': []})
                with self.assertRaisesRegex(ValueError, 'bindings_changed'): scope.validate_plan(plan)

    def test_empty_ready_builder_records_skip_without_creating_qa_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); plan = fixture_plan(home)
            save(home / 'PILOT_PLAN.json', plan); save(home / 'SCENE_PLANS.json', {'plans': []})
            spec = {'version': 'v98_ready_subset_shard_spec_v1', 'shard_id': 'none', 'roots': [],
                    'plan_path': str(home / 'PILOT_PLAN.json'), 'scene_plans': str(home / 'SCENE_PLANS.json')}
            save(home / 'SPEC.json', spec)
            with patch.object(scope, 'PILOT', home), patch.object(scope, 'PLAN_PATH', home / 'PILOT_PLAN.json'), patch.object(builder, 'PILOT', home):
                builder.build(home / 'SPEC.json')
            receipt = read(home / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json')
            self.assertEqual((receipt['qa_http_calls'], receipt['solver_calls']), (0, 0))
            self.assertFalse((home / 'shards').exists())


class MaterializerFailureCoverageTests(unittest.TestCase):
    def test_all_four_failed_model_outcomes_are_recorded_without_historical_source_fallback(self):
        from types import SimpleNamespace
        import hashlib
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); plan = fixture_plan(home); run = home / 'expansion_engine/runs/four_roots_B_scene01'
            save(home / 'PILOT_PLAN.json', plan)
            save(run / 'MANIFEST.json', {'roots': plan['candidate_roots'], 'files': []})
            save(run / 'SOURCE_RESERVATIONS.json', {})
            save(run / 'UNIQUE_EVENT_ALLOCATION.json', {'assignments': {r: [] for r in plan['candidate_roots']}})
            save(run / 'dataset_cards_result.json', {})
            save(home / 'expansion_engine/SOURCE_REGISTRY.json', {})
            (run / 'requests').mkdir()
            for root in plan['candidate_roots']:
                source = {'root_id': root, 'turns': [{'turn_ref':'P01:T001', 'speaker':'Alex', 'text':'Original history.'}]}
                folder = run / 'roots' / root / 'B'
                save(run / 'sources' / (root + '.json'), source)
                save(folder / 'cumulative_dialogue.json', source)
                save(folder / 'RESULT.json', {'root_id':root, 'route':'dataset_new_event',
                    'added_turns':0, 'accepted_scenes':0, 'failures':[{'reason':'actual_model_rejection'}],
                    'output_history_sha256':digest(source['turns'])})
            path = HOME / 'evidence/materialize_expansion.py'
            with patch.object(scope, 'upstream_run', return_value=run), patch.object(scope, 'PLAN_PATH', home / 'PILOT_PLAN.json'):
                spec = importlib.util.spec_from_file_location('v98_materialize_allfail_fixture', path)
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                fake = SimpleNamespace(ROUTES=['other','dataset_new_event'], base=SimpleNamespace(digest=digest))
                with patch.object(module, 'V98', home), patch.object(module, 'runtime', return_value=fake):
                    result = module.materialize_once()
            self.assertEqual(result['completed_root_count'], 4)
            self.assertFalse((home / 'sources').exists())
            receipt = read(home / 'evidence/EXPANSION_MATERIALIZATION_COMPLETE.json')
            self.assertEqual((receipt['accepted_roots'], receipt['failed_roots']), (0, 4))
            self.assertTrue(all(r['source'] is None for r in receipt['roots']))
            self.assertEqual(len(list((home / 'evidence/expansion_failures').glob('*.json'))), 4)


class PriorRegressionTests(unittest.TestCase):
    def test_explicit_empty_extras_preserve_mandatory_baseline(self):
        self.assertEqual(ordering_prior.configured_paths({'prior_pool_path': '/tmp/base', 'prior_pool_paths': []}), ['/tmp/base'])
        self.assertEqual(ordering_prior.configured_paths({'prior_pool_path': '/tmp/base', 'prior_pool_paths': ['/tmp/base', '/tmp/extra']}), ['/tmp/base', '/tmp/extra'])

    def test_actual_v96_history_complete_public_ordering_text_without_gold_or_evidence(self):
        for number, count in ((25652, 36), (19708, 35)):
            root = f'cc-train-episode-{number}'
            path = OLD / f'shards/shard_01/ordering/root{number}_ordering/feedback/{root}/candidate_history.json'
            history = read(path); priors, receipt, files = ordering_prior.collect(root, path, [])
            expected = [h for h in history if h['historical_question']['question_type'] == 'ordering']
            self.assertEqual(receipt['historical_candidate_count'], count)
            self.assertEqual([q['candidate_id'] for q in priors], [h['candidate_id'] for h in expected])
            self.assertGreater(len(priors), 0)
            self.assertIn(4, [len(q['items']) for q in priors])
            for actual, old in zip(priors, expected):
                self.assertEqual(actual['items'], old['historical_question']['items'])
                self.assertEqual(actual['question'], old['historical_question']['question'])
                self.assertFalse({'gold', 'evidence_refs', 'evidence_groups', 'option_evidence'} & actual.keys())

    def test_extra_archives_and_adapter_priors_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); root = 'cc-train-episode-11'
            q = {'candidate_id': 'old', 'root_id': root, 'question_type': 'ordering', 'question': 'Arrange',
                 'items': {'I1': 'Alex arrives.', 'I2': 'Morgan leaves.'}, 'gold': ['I2', 'I1'], 'evidence': ['secret']}
            save(home / 'H.json', [{'candidate_id': 'old', 'root_id': root, 'historical_question': q}])
            save(home / 'P.json', {'questions': [{**q, 'candidate_id': 'extra'}]})
            compact = [{'candidate_id': 'selected', 'question_type': 'ordering', 'question': 'Arrange selected',
                        'facts': ['Morgan calls.', 'Alex answers.'], 'source_path': str(home / 'P.json'),
                        'source_sha256': scope.bind(home / 'P.json')['sha256']}]
            priors, receipt, files = ordering_prior.collect(root, home / 'H.json', [str(home / 'P.json')],
                                                          compact_rows=compact, compact_files=[home / 'P.json'])
            self.assertEqual([q['candidate_id'] for q in priors], ['old', 'selected', 'extra'])
            self.assertEqual(receipt['supplied_public_prior_rows'], 3)


class SingleEventWriterRegressionTests(unittest.TestCase):
    def fixture(self):
        run = OLD / 'shards/shard_01/ordering/root25652_ordering'
        source = read(run / 'SOURCE.json'); plan = read(run / 'ANCHOR_PLAN.json')['questions'][0]
        gold = ordering_plan.gold_questions(source, {'questions': [plan]})[0]
        records = []
        for anchor in plan['anchors'].values():
            # Test fixture grammar only; semantic certification remains external.
            records.append({'anchor_id': anchor['turn_ref'], 'support_quote': anchor['quote'],
                            'predicate': 'reports a source-grounded event involving Elena.'})
        return source, plan, gold, {'candidate_id': plan['candidate_id'], 'root_id': plan['root_id'], 'records': records}

    def test_support_excerpt_does_not_control_fixed_full_turn_gold(self):
        source, plan, gold, raw = self.fixture(); original = deepcopy(raw)
        projected = ordering_plan.project(raw, plan, gold)
        self.assertEqual(projected['gold'], gold['gold'])
        self.assertEqual(raw, original)
        for label, anchor in plan['anchors'].items():
            self.assertEqual(projected['item_evidence'][label]['quote'], anchor['quote'])
        # A shortened excerpt remains inside the same original full turn.
        first = raw['records'][0]; first['support_quote'] = plan['anchors'][next(iter(plan['anchors']))]['quote'][:12]
        self.assertEqual(ordering_plan.project(raw, plan, gold)['gold'], gold['gold'])

    def test_wrong_turn_quote_and_names_only_violations_reject_without_repair(self):
        _, plan, gold, raw = self.fixture()
        bad = deepcopy(raw); bad['records'][0]['support_quote'] = 'Not present in this frozen turn.'
        with self.assertRaisesRegex(ValueError, 'support_quote_not_exact'): ordering_plan.project(bad, plan, gold)
        bad = deepcopy(raw); bad['records'][0]['predicate'] = 'says she appreciates his help.'
        before = deepcopy(bad)
        with self.assertRaisesRegex(ValueError, 'forbidden_gendered_pronoun'): ordering_plan.project(bad, plan, gold)
        self.assertEqual(bad, before)

    def test_actual_v96_19708_original_draft_remains_rejected(self):
        run = OLD / 'shards/shard_01/ordering/root19708_ordering'
        response_path = next(run.glob('requests/*/RESPONSE.json'))
        raw = json.loads(read(response_path)['text'])
        for slot in raw['questions']:
            self.assertTrue(any(actor_reference.PATTERN.search(r['predicate']) for r in slot['records']))
        self.assertEqual(len(read(run / 'REJECTIONS.json')), 2)

    def test_private_semantics_and_solver_algorithms_remain_identical_after_namespace_change(self):
        for name in ('fresh_parent.py', 'ordering_review.py', 'actor_reference.py', 'mixed_evaluation.py', 'prior_rendering.py'):
            self.assertEqual((ENGINE / name).read_text().replace('v98', 'v96').replace('V98', 'V96'), (OLD / 'engine' / name).read_text(), name)
        old_spec = importlib.util.spec_from_file_location('v96_frozen_ordering_plan_test', OLD / 'engine/ordering_plan.py')
        old = importlib.util.module_from_spec(old_spec); old_spec.loader.exec_module(old)
        source, plan, _, _ = self.fixture()
        self.assertEqual(ordering_plan.gold_questions(source, {'questions': [plan]}), old.gold_questions(source, {'questions': [plan]}))
        self.assertEqual(ordering_plan.writer_plan(source, {'root_id': plan['root_id'], 'source_sha256': 'x', 'questions': [plan]})['questions'],
                         old.writer_plan(source, {'root_id': plan['root_id'], 'source_sha256': 'x', 'questions': [plan]})['questions'])

    def test_installer_includes_prior_adapter_and_installed_scope_finds_delivery_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); shutil.copytree(ENGINE, home / 'qa_engine')
            code = 'from install_shard import install; install(r"' + str(home / 'shards/shard_01') + '")'
            result = subprocess.run([PYTHON, '-B', '-c', code], cwd=home / 'qa_engine', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            shard = home / 'shards/shard_01'
            result = subprocess.run([PYTHON, '-B', '-c', 'import batch_scope; print(batch_scope.PILOT)'], cwd=shard, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), str(home))
            self.assertTrue((shard / 'ordering_prior.py').is_file())


if __name__ == '__main__': unittest.main(verbosity=2)
