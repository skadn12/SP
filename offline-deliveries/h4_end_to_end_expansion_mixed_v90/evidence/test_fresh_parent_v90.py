"""Offline integrity regressions for the fresh-parent admission branch."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PILOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PILOT / 'engine'))
import fresh_parent as fresh


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fixture(home, number):
    root = 'cc-train-episode-' + str(number)
    source_path = home / (root + '_SOURCE.json')
    source = {'root_id': root, 'turns': [
        {'turn_ref': root + ':R02:S01:T' + str(i).zfill(3), 'speaker': 'Alex',
         'text': f'I completed task {i} on Monday.', 'origin': 'generated_extension', 'scene_ref': root + ':R02:S01'}
        for i in range(1, 7)]}
    save(source_path, source)
    expansion_path = home / (root + '_EXPANSION.json')
    save(expansion_path, {'root_id': root, 'output_source': fresh.bind(source_path),
                          'purpose': 'synthetic offline validator regression fixture; no production provenance claim'})
    public_path = home / (root + '_PUBLIC.json')
    public = {'candidate_id': root + '-v90-fixture-parent', 'root_id': root,
              'question_type': 'multi_select', 'question': 'Which tasks did Alex complete on Monday?',
              'options': {label: f'Alex completed task {i} on Monday.' for i, label in enumerate('ABCDEF', 1)},
              'source_path': str(source_path), 'source_sha256': fresh.bind(source_path)['sha256']}
    save(public_path, public)
    review_path = home / (root + '_SOURCE_REVIEW.json')
    review = {'schema_version': 'dialogue_consistency_v1', 'root_id': root,
              'source': fresh.bind(source_path), 'review_scope': 'source_consistency_only',
              'reviewer': 'synthetic_test_fixture', 'limitations': 'Fixture assertions exercise record validation only.',
              'created_at_utc': fresh.now(), 'full_source_read': True, 'current_question_gold_read': False,
              'current_solver_scores_read': False, 'historical_question_gold_known': False, 'fresh_blind_review': True,
              'turn_refs_read': [t['turn_ref'] for t in source['turns']],
              'checks_completed': {key: True for key in sorted(fresh.CHECKS)}, 'decision': 'READY', 'findings': [],
              'units': [{'unit_id': root + ':R02:S01', 'origin': 'generated_extension',
                         'turn_refs': [t['turn_ref'] for t in source['turns']], 'reason': 'Synthetic fixture.',
                         'decision': 'PASS', 'finding_ids': []}]}
    save(review_path, review)
    judgment_path = home / (root + '_SOURCE_JUDGMENTS.json')
    judgment = {'version': fresh.PARENT_REVIEW_VERSION, 'root_id': root, 'candidate_id': public['candidate_id'],
        'created_at_utc': fresh.now(), 'reviewer': 'synthetic_test_fixture', 'review_mode': 'fresh_public_source_review',
        'limitations': 'Fixture assertions exercise record validation only.', 'public_question': public,
        'public_question_path': str(public_path), 'public_question_sha256': fresh.bind(public_path)['sha256'],
        'source_path': str(source_path), 'source_sha256': public['source_sha256'], 'full_source_read': True,
        'full_source_turn_count': len(source['turns']), 'turn_refs_read': [t['turn_ref'] for t in source['turns']],
        'stem_valid': True, 'known_old_parent_gold': False, 'fresh_blind_review': True,
        'public_parent_authored_by_reviewer': False, 'proposed_all_supported_intent_known': True,
        'historical_question_gold_known': False, 'current_parent_private_gold_read_before_judgments_saved': False,
        'current_parent_private_evidence_read_before_judgments_saved': False, 'current_solver_results_read': False,
        'content_status': 'CORE_USABLE', 'judgments': [
            {'option': label, 'proposition': public['options'][label], 'source_judgment': 'supported', 'relation': 'supports',
             'entailed': True, 'scope_valid': True, 'fact_distinct': True, 'reason': 'Source turn explicitly states the task.',
             'required_evidence': [{key: source['turns'][i][key] for key in ('turn_ref', 'speaker')} |
                                   {'quote': source['turns'][i]['text']}]}
            for i, label in enumerate('ABCDEF')]}
    save(judgment_path, judgment)
    return {'root_id': root, 'public_parent': str(public_path), 'parent_judgments': str(judgment_path),
            'source_review': str(review_path), 'public_prior_paths': [], 'expanded_source_provenance': str(expansion_path)}


class FreshParentIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v90_fresh_parent_')
        self.home = Path(self.temp.name)
        self.row = fixture(self.home, 900000001)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return fresh.validate_judgments(self.row['public_parent'], self.row['parent_judgments'],
                                         self.row['expanded_source_provenance'])

    def change_judgment(self, edit):
        value = fresh.read(self.row['parent_judgments']); edit(value); save(Path(self.row['parent_judgments']), value)

    def test_fresh_and_honest_nonblind_authoring_both_validate(self):
        self.validate()
        self.change_judgment(lambda j: j.update(fresh_blind_review=False, public_parent_authored_by_reviewer=True,
                                               historical_question_gold_known=True))
        self.validate()

    def test_known_old_parent_and_authored_blind_claims_rejected(self):
        original = fresh.read(self.row['parent_judgments'])
        for patch in ({'known_old_parent_gold': True}, {'public_parent_authored_by_reviewer': True}):
            save(Path(self.row['parent_judgments']), {**original, **patch})
            with self.assertRaises(ValueError):
                self.validate()

    def test_unknown_missing_scope_and_inexact_evidence_rejected(self):
        original = fresh.read(self.row['parent_judgments'])
        edits = [lambda j: j['judgments'][0].update(source_judgment='unknown'),
                 lambda j: j['judgments'][0].update(scope_valid=False),
                 lambda j: j['judgments'][0]['required_evidence'][0].update(quote='not in the source')]
        for edit in edits:
            value = deepcopy(original); edit(value); save(Path(self.row['parent_judgments']), value)
            with self.assertRaises(ValueError):
                self.validate()

    def test_incomplete_source_read_and_private_exposure_rejected(self):
        original = fresh.read(self.row['parent_judgments'])
        for edit in (lambda j: j['turn_refs_read'].pop(),
                     lambda j: j.update(current_parent_private_gold_read_before_judgments_saved=True),
                     lambda j: j.update(current_solver_results_read=True)):
            value = deepcopy(original); edit(value); save(Path(self.row['parent_judgments']), value)
            with self.assertRaises(ValueError):
                self.validate()

    def test_private_fields_cannot_enter_public_draft(self):
        public = fresh.read(self.row['public_parent']); public['gold'] = list('ABCDEF')
        save(Path(self.row['public_parent']), public)
        with self.assertRaisesRegex(ValueError, 'no_private_gold'):
            self.validate()

    def test_expansion_provenance_cannot_point_at_another_source(self):
        other = fixture(self.home, 900000002)
        self.row['expanded_source_provenance'] = other['expanded_source_provenance']
        with self.assertRaisesRegex(ValueError, 'expanded_source_provenance_identity'):
            self.validate()

    def test_future_parent_judgment_rejected(self):
        self.change_judgment(lambda j: j.update(created_at_utc=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()))
        with self.assertRaisesRegex(ValueError, 'future'):
            self.validate()

    def test_source_review_truthful_exposure_and_all_checks(self):
        public = fresh.read(self.row['public_parent'])
        row = {'root_id': self.row['root_id'], 'source': fresh.bind(public['source_path']), 'review_path': self.row['source_review']}
        self.assertEqual(fresh.validate_source_review(row)['decision'], 'READY')
        review = fresh.read(self.row['source_review']); review['checks_completed'].pop('time')
        save(Path(self.row['source_review']), review)
        with self.assertRaisesRegex(ValueError, 'all_six_checks'):
            fresh.validate_source_review(row)

    def test_blocking_finding_cannot_be_reported_ready(self):
        public = fresh.read(self.row['public_parent']); source = fresh.read(public['source_path'])
        review = fresh.read(self.row['source_review'])
        review['findings'] = [{'finding_id': 'F01', 'severity': 'BLOCKING', 'kind': 'time',
            'reason': 'Synthetic blocking finding.', 'proposed_action': 'Hold this fixture.',
            'unit_ids': [source['turns'][0]['scene_ref']],
            'evidence': [{'turn_ref': source['turns'][0]['turn_ref'], 'quote': source['turns'][0]['text']}]}]
        review['units'][0].update(decision='HOLD', finding_ids=['F01'])
        save(Path(self.row['source_review']), review)
        row = {'root_id': self.row['root_id'], 'source': fresh.bind(public['source_path']), 'review_path': self.row['source_review']}
        with self.assertRaisesRegex(ValueError, 'source_root_decision'):
            fresh.validate_source_review(row)
        review['decision'] = 'HOLD'; save(Path(self.row['source_review']), review)
        self.assertEqual(fresh.validate_source_review(row)['decision'], 'HOLD')

    def test_two_root_builder_freezes_all_fresh_inputs_and_v79_accepts_parent(self):
        test_pilot = self.home / 'pilot'; test_pilot.mkdir()
        shutil.copytree(PILOT / 'engine', test_pilot / 'engine')
        second = fixture(self.home, 900000002)
        multi_prior = self.home / 'MULTI_PRIOR.json'
        ordering_prior = self.home / 'ORDERING_PRIOR.json'
        save(multi_prior, {'questions': []})
        save(ordering_prior, {'questions': []})
        self.row['public_prior_paths'] = [str(multi_prior)]
        self.row['public_ordering_prior_paths'] = [str(ordering_prior)]
        plan_path = self.home / 'PILOT_PLAN.json'
        save(plan_path, {'version': 'v90_four_root_end_to_end_expansion_mixed_v1', 'candidate_root_budget': 4,
             'source_split': 'train', 'training_allowed': False, 'formal_training_admission': False,
             'exclude_repeated_roots': [], 'exclude_known_hold_roots': []})
        plans_path = self.home / 'PLANS.json'; plans = []
        for row in (self.row, second):
            source = fresh.read(fresh.read(row['public_parent'])['source_path'])
            for number in (1, 2):
                plans.append({'root_id': row['root_id'], 'slot_number': number,
                    'scene_id': source['turns'][0]['scene_ref'],
                    'anchors': [{'turn_ref': t['turn_ref'], 'speaker': t['speaker'], 'quote': t['text']} for t in source['turns']]})
        save(plans_path, {'plans': plans})
        spec_path = self.home / 'SPEC.json'; save(spec_path, {'shard_id': 'offline_fixture', 'roots': [self.row, second],
                                                         'scene_plans': str(plans_path), 'plan_path': str(plan_path)})
        spec = importlib.util.spec_from_file_location('v90_builder_fixture', PILOT / 'evidence/build_fresh_shard.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        builder.PILOT = test_pilot
        builder.build(spec_path)
        shard = test_pilot / 'shards/offline_fixture'; frozen = fresh.read(shard / 'SHARD_INPUTS.COMPLETE.json')
        paths = {r['path'] for r in frozen['files']}
        self.assertTrue({self.row['public_parent'], self.row['expanded_source_provenance']} <= paths)
        self.assertTrue({str(multi_prior), str(ordering_prior)} <= paths)
        config = fresh.read(shard / 'CONFIG.json')
        self.assertEqual(config['multiselect_config']['slots'][0]['prior_pool_paths'], [str(multi_prior)])
        self.assertEqual(config['ordering_slots'][0]['prior_pool_paths'], [str(ordering_prior)])
        self.assertEqual(config['ordering_slots'][1]['prior_pool_paths'], [])
        code = """import json, sys
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_output_contract_multiselect_v79')
import asset_generation as a
config=json.loads(Path('CONFIG.json').read_text())
for slot in config['multiselect_config']['slots']:
    parent,source,path,inputs=a.parent_source(slot)
    public={'root_id':parent['root_id'],'source_sha256':parent['source_sha256'],'public_stem':parent['question'],
            'parent_facts':[{'parent_label':k,'proposition':v} for k,v in parent['options'].items()]}
    bindings=a.review_schema.validate_parent_reuse(slot,public,a.aware_time(a.v15.now(),'test'),Path('fixture_review.json'))
    assert len(bindings)==2
print('V79_PARENT_AND_REUSE_PASS')
"""
        result = subprocess.run([PYTHON, '-B', '-c', code], cwd=shard, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('V79_PARENT_AND_REUSE_PASS', result.stdout)
        result = subprocess.run([PYTHON, '-B', '-c', "import batch; print(batch.config()['version'])"],
                                cwd=shard, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('v90_full_turn_mixed_shard_v1', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
