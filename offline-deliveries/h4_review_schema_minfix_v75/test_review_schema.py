"""Fail-closed disclosure, public binding and historical reuse checks, offline."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import review_schema as schema


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def disclosures(stage='round'):
    j = {'review_schema_version': schema.SCHEMA_VERSION,
         'parent_all_true_provenance_known': True, 'parent_gold_known_before_review': True,
         'fresh_blind_parent_review': False, 'current_solver_scores_read': False,
         'solver_scores_used': False}
    if stage == 'round':
        j.update(full_source_read=True, new_variant_private_artifacts_read_before_save=False)
    elif stage == 'final':
        j.update(child_gold_read_before_save=False, current_pool_read_before_save=False)
    else:
        j.update(derived_from_rounds=True, source_first_per_round=True,
                 new_variant_private_artifacts_read_before_save=True,
                 created_at_utc=None, review_mode='derived_from_saved_round_source_judgments',
                 aggregate_is_new_independent_blind_review=False)
    return j


class SchemaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        block = patch('socket.socket', side_effect=AssertionError('Offline test: no models'))
        block.start(); self.addCleanup(block.stop)

    def fixture(self):
        source = {'root_id': 'root1', 'turns': [{'turn_ref': 'T1', 'speaker': 'Alex', 'text': 'Alex proposed Tuesday.'}]}
        save(self.run / 'SOURCE.json', source)
        public = {'root_id': 'root1', 'family_id': 'family1', 'round_id': 'initial',
                  'source_sha256': schema.sha(self.run / 'SOURCE.json'), 'public_stem': 'What was proposed?',
                  'parent_facts': [{'fact_id': 'F01', 'parent_label': 'A', 'proposition': 'Alex proposed Tuesday.'}],
                  'versions': [{'version_id': 'V01', 'proposition': 'Alex proposed Wednesday.'}]}
        folder = self.run / 'rounds' / 'initial'; save(folder / 'PUBLIC_ASSETS.json', public)
        j = {**disclosures(), 'reviewer': 'synthetic-reviewer', 'root_id': 'root1', 'round_id': 'initial',
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'stem_valid': True,
             'source_sha256': public['source_sha256'], 'public_assets_sha256': schema.sha(folder / 'PUBLIC_ASSETS.json'),
             'complete_source_turns_read': 1}
        option = {'scope_valid': True, 'fact_distinct': True, 'reason': 'Synthetic fixed date.',
                  'citations': [{'turn_ref': 'T1', 'speaker': 'Alex', 'quote': 'proposed Tuesday'}]}
        j['parents'] = [{**option, **public['parent_facts'][0], 'relation': 'supports'}]
        j['versions'] = [{**option, **public['versions'][0], 'relation': 'contradicts'}]
        save(folder / 'SOURCE_JUDGMENTS.json', j)
        save(self.run / 'MANIFEST.json', {'root_id': 'root1', 'slot': {}})
        return folder, public, j

    def test_required_canonical_booleans_for_every_stage_without_mutation(self):
        for stage in ('round', 'final', 'derived'):
            original = disclosures(stage); before = deepcopy(original)
            schema.validate_disclosures(original, 'synthetic.json', stage)
            self.assertEqual(original, before)
            for key in schema.COMMON_DISCLOSURES:
                for invalid in ('missing', None, 'false', 'true', 0, 1, [], {}):
                    j = deepcopy(original)
                    if invalid == 'missing':
                        del j[key]
                    else:
                        j[key] = invalid
                    with self.subTest(stage=stage, key=key, invalid=invalid), self.assertRaisesRegex(ValueError, key):
                        schema.validate_disclosures(j, 'synthetic.json', stage)

    def test_seven_original_missing_occurrences_have_no_alias_fallback(self):
        occurrences = [('root194259', 'initial', 'parent_gold_known_before_review'),
                       ('root231899', 'initial', 'current_solver_scores_read'),
                       ('root231899', 'supplement', 'current_solver_scores_read'),
                       ('root138343', 'initial', 'current_solver_scores_read'),
                       ('root138343', 'supplement', 'current_solver_scores_read'),
                       ('root130221', 'initial', 'parent_gold_known_before_review'),
                       ('root130221', 'supplement', 'parent_gold_known_before_review')]
        for root, rid, key in occurrences:
            j = disclosures(); value = j.pop(key)
            j['parent_gold_known_before_current_review' if key.startswith('parent') else 'solver_scores_used'] = value
            with self.subTest(root=root, round=rid), self.assertRaisesRegex(ValueError, key + ':required'):
                schema.validate_disclosures(j, f'{root}/{rid}/SOURCE_JUDGMENTS.json')

    def test_known_gold_cannot_be_fresh_blind_and_current_exposure_blocks(self):
        for stage, key in (('round', 'fresh_blind_parent_review'), ('round', 'current_solver_scores_read'),
                           ('round', 'new_variant_private_artifacts_read_before_save'),
                           ('final', 'child_gold_read_before_save'), ('final', 'current_pool_read_before_save')):
            j = disclosures(stage); j[key] = True
            with self.subTest(stage=stage, key=key), self.assertRaisesRegex(ValueError, key):
                schema.validate_disclosures(j, 'synthetic.json', stage)

    def test_derived_record_cannot_claim_new_review_or_skip_aggregate_disclosure(self):
        j = disclosures('derived')
        for key, value in [('created_at_utc', datetime.now(timezone.utc).isoformat()),
                           ('review_mode', 'fresh_blind'), ('new_variant_private_artifacts_read_before_save', False),
                           ('aggregate_is_new_independent_blind_review', True)]:
            bad = {**j, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError): schema.validate_disclosures(bad, 'derived.json', 'derived')
        bad = deepcopy(j); del bad['aggregate_is_new_independent_blind_review']
        with self.assertRaisesRegex(ValueError, 'aggregate_is_new'): schema.validate_disclosures(bad, 'derived.json', 'derived')
        for invalid in (None, 0, '', True):
            bad = {**disclosures(), 'derived_from_rounds': invalid}
            with self.assertRaisesRegex(ValueError, 'derived_from_rounds'): schema.validate_disclosures(bad, 'round.json')

    def test_valid_saved_round_and_complete_source_requirements(self):
        folder, _, j = self.fixture()
        self.assertEqual(schema.validate_round(self.run, 'initial'), j)
        for invalid in (0, True, '1', None):
            save(folder / 'SOURCE_JUDGMENTS.json', {**j, 'complete_source_turns_read': invalid})
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'complete_source'):
                schema.validate_round(self.run, 'initial')

    def test_unsaved_alias_and_private_files_are_not_read_at_source_gate(self):
        folder, _, j = self.fixture(); del j['current_solver_scores_read']
        save(folder / 'SOURCE_JUDGMENTS.json', j)
        real = schema.read; accesses = []
        def public_only(path):
            accesses.append(Path(path).name)
            self.assertNotIn(Path(path).name, ('ASSETS.json', 'VERSION_MAP.json', 'POOL.json'))
            return real(path)
        with patch.object(schema, 'read', side_effect=public_only), self.assertRaisesRegex(ValueError, 'current_solver_scores_read'):
            schema.validate_round(self.run, 'initial')
        self.assertEqual(accesses, ['SOURCE_JUDGMENTS.json'])

    def test_changed_source_public_coverage_text_and_exact_citations_block(self):
        folder, public, j = self.fixture()
        for key, value in [('source_sha256', 'bad'), ('public_assets_sha256', 'bad'), ('root_id', 'other'), ('round_id', 'supplement')]:
            save(folder / 'SOURCE_JUDGMENTS.json', {**j, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): schema.validate_round(self.run, 'initial')
        for mutation in ('coverage', 'proposition', 'citation'):
            bad = deepcopy(j)
            if mutation == 'coverage': bad['versions'] = []
            elif mutation == 'proposition': bad['versions'][0]['proposition'] = 'Another question.'
            else: bad['versions'][0]['citations'][0]['quote'] = 'not present'
            save(folder / 'SOURCE_JUDGMENTS.json', bad)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): schema.validate_round(self.run, 'initial')
        save(folder / 'SOURCE_JUDGMENTS.json', j)
        save(folder / 'PUBLIC_ASSETS.json', {**public, 'public_stem': 'Changed stem'})
        with self.assertRaisesRegex(ValueError, 'binding'): schema.validate_round(self.run, 'initial')

    def test_future_naive_and_invalid_explicit_private_times_block(self):
        folder, _, j = self.fixture()
        for when in ((datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), '2020-01-01T00:00:00', None):
            save(folder / 'SOURCE_JUDGMENTS.json', {**j, 'created_at_utc': when})
            with self.subTest(when=when), self.assertRaises(ValueError): schema.validate_round(self.run, 'initial')
        save(folder / 'SOURCE_JUDGMENTS.json', j)
        for before in ('', False, 0):
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'private_started_at'):
                schema.validate_round(self.run, 'initial', before_private_at=before)
        with self.assertRaisesRegex(ValueError, 'not_saved'): schema.validate_round(self.run, 'initial', judgment={**j, 'reviewer': 'unsaved'})

    def test_historical_reuse_binding_identity_and_time_are_required(self):
        _, public, j = self.fixture()
        created = datetime.now(timezone.utc)
        old = {'created_at_utc': (created - timedelta(days=1)).isoformat(), 'questions': [
            {'candidate_id': 'parent1', 'root_id': public['root_id'], 'source_sha256': public['source_sha256'],
             'question': public['public_stem'], 'options': {'A': {'proposition': public['parent_facts'][0]['proposition']}}}]}
        old_path = self.run / 'legacy_j.json'; save(old_path, old)
        prov = {'created_at_utc': old['created_at_utc'], 'parent_candidate_id': 'parent1',
                'source': schema.bind(self.run / 'SOURCE.json')}
        prov_path = self.run / 'legacy_p.json'; save(prov_path, prov)
        slot = {'parent_candidate_id': 'parent1', 'parent_review_path': str(old_path), 'parent_review_sha256': schema.sha(old_path),
                'parent_provenance_path': str(prov_path), 'parent_provenance_sha256': schema.sha(prov_path)}
        self.assertEqual(len(schema.validate_parent_reuse(slot, public, created, 'round.json')), 2)
        for field, value in [('parent_review_sha256', 'bad'), ('parent_candidate_id', 'other')]:
            with self.subTest(field=field), self.assertRaises(ValueError): schema.validate_parent_reuse({**slot, field: value}, public, created, 'round.json')
        for change in ('wrong_root', 'future', 'changed_proposition'):
            bad = deepcopy(old)
            if change == 'wrong_root': bad['questions'][0]['root_id'] = 'other'
            elif change == 'future': bad['created_at_utc'] = (created + timedelta(days=1)).isoformat()
            else: bad['questions'][0]['options']['A']['proposition'] = 'Changed.'
            save(old_path, bad); changed = {**slot, 'parent_review_sha256': schema.sha(old_path)}
            with self.subTest(change=change), self.assertRaises(ValueError): schema.validate_parent_reuse(changed, public, created, 'round.json')
        save(old_path, old)
        save(prov_path, {**prov, 'parent_candidate_id': 'other'})
        with self.assertRaisesRegex(ValueError, 'wrong_parent'):
            schema.validate_parent_reuse({**slot, 'parent_provenance_sha256': schema.sha(prov_path)}, public, created, 'round.json')


if __name__ == '__main__':
    unittest.main()
