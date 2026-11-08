"""Synthetic pre-request Gold and independent semantic-anchor safety tests."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HOME = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HOME))
import ordering_generation as generation
import ordering_plan
import ordering_review as review
from common import bind, load_ordering_gate, read, write


def fixture(run):
    """Complete private plan fixture, no replacement of production validation."""
    run.mkdir(parents=True, exist_ok=True)
    labels = ['I3', 'I1', 'I6', 'I2', 'I5', 'I4']
    root, qid = 'synthetic-source', 'synthetic-question'
    source = {'root_id': root, 'turns': [
        {'turn_ref': f'CANDIDATE:A:S01:T{i:03d}', 'speaker': 'Actor',
         'text': f'Actor recorded event {i}. Actor discussed topic {i}.'}
        for i in range(1, 7)]}
    plan = {'candidate_id': qid, 'root_id': root, 'scene_id': 'CANDIDATE:A:S01',
            'question': ordering_plan.STEM.format(scene_id='CANDIDATE:A:S01'),
            'anchors': {label: {'turn_ref': source['turns'][i]['turn_ref'], 'speaker': 'Actor',
                               'quote': f'Actor recorded event {i + 1}.'}
                        for i, label in enumerate(labels)}}
    anchor_plan = {'root_id':root,'source_sha256':'synthetic-source-binding','questions': [plan]}
    write(run / 'SOURCE.json', source)
    anchor_plan['source_sha256']=bind(run / 'SOURCE.json')['sha256']
    write(run / 'ANCHOR_PLAN.json', anchor_plan)
    write(run / 'WRITER_PLAN.json', ordering_plan.writer_plan(source, anchor_plan))
    record = {'version': 'v88_program_gold_plan_v1', 'created_at_utc': '2026-10-14T00:00:00+00:00',
              'source': bind(run / 'SOURCE.json'), 'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
              'gold_origin': ordering_plan.GOLD_ORIGIN, 'semantic_certification': False,
              'gold_supplied_to_writer': False, 'questions': ordering_plan.gold_questions(source, anchor_plan)}
    write(run / 'GOLD_PLAN.json', record)
    manifest = {'root_id': root, 'candidate_ids': [qid], 'source': bind(run / 'SOURCE.json'),
                'anchor_plan': bind(run / 'ANCHOR_PLAN.json'), 'gold_plan': bind(run / 'GOLD_PLAN.json'),
                'writer_plan': bind(run / 'WRITER_PLAN.json'),
                'gold_origin': ordering_plan.GOLD_ORIGIN, 'prepared_at_utc': '2026-10-14T00:01:00+00:00'}
    write(run / 'MANIFEST.json', manifest)
    raw = {'candidate_id': qid, 'root_id': root,
           'records': [{'anchor_id':turn['turn_ref'],'predicate':f'recorded event {i+1}.'} for i,turn in enumerate(source['turns'])]}
    return source, plan, record, manifest, raw


class ProgramGoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v88_program_gold_synthetic_')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.run = self.home / 'ordering/fixture'
        self.source, self.plan, self.record, self.manifest, self.raw = fixture(self.run)

    def validate(self, raw):
        with patch.object(generation, 'engine', return_value=SimpleNamespace(
                strict_json=json.loads, legacy=SimpleNamespace(MODEL='synthetic-no-model-call'))):
            return generation.validate_raw(json.dumps({'questions': [raw]}), self.manifest,
                self.source, self.run / 'SOURCE.json', [])

    def test_verbatim_predicates_receive_frozen_subject_labels_and_gold(self):
        self.assertEqual(self.record['questions'][0]['gold'], ['I3', 'I1', 'I6', 'I2', 'I5', 'I4'])
        before = deepcopy(self.raw)
        accepted, rejected = self.validate(self.raw)
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['gold'], self.record['questions'][0]['gold'])
        self.assertEqual(accepted[0]['items'], {label:anchor['quote'] for label,anchor in self.plan['anchors'].items()})
        self.assertEqual(accepted[0]['program_gold_plan'], bind(self.run / 'GOLD_PLAN.json'))
        self.assertEqual(self.raw, before)

    def test_legacy_gold_reason_and_any_extra_field_are_rejected_never_repaired(self):
        for key, value in [('gold', ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']),
                           ('order_reason', 'Old writer contract'), ('question', 'New scope'),
                           ('item_evidence', {}), ('items', {}), ('labels', ['I1']), ('unknown_extra', True)]:
            with self.subTest(key=key):
                raw = {**deepcopy(self.raw), key: value}
                before = deepcopy(raw)
                accepted, rejected = self.validate(raw)
                self.assertEqual(accepted, [])
                self.assertEqual(rejected[0]['reasons'], ['ordering_bound_writer_schema'])
                self.assertEqual(raw, before)

    def test_records_array_order_does_not_change_items_or_answer(self):
        expected, errors = self.validate(self.raw)
        self.assertEqual(errors, [])
        reordered = deepcopy(self.raw)
        reordered['records'] = reordered['records'][2:] + reordered['records'][:2]
        actual, errors = self.validate(reordered)
        self.assertEqual(errors, [])
        self.assertEqual(actual, expected)

    def test_missing_duplicate_unknown_ids_and_record_fields_rejected(self):
        changes = {
            'missing': lambda r: r['records'].pop(),
            'duplicate': lambda r: r['records'][1].update(anchor_id=r['records'][0]['anchor_id']),
            'unknown': lambda r: r['records'][0].update(anchor_id='invented'),
            'subject': lambda r: r['records'][0].update(subject='OtherActor'),
            'label': lambda r: r['records'][0].update(label='I1'),
            'gold': lambda r: r['records'][0].update(gold=True),
            'blank': lambda r: r['records'][0].update(predicate=''),
            'whitespace': lambda r: r['records'][0].update(predicate=' recorded event 1.'),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                altered = deepcopy(self.raw);change(altered);before=deepcopy(altered)
                accepted, rejected = self.validate(altered)
                self.assertEqual(accepted, []);self.assertTrue(rejected[0]['reasons'])
                self.assertEqual(altered, before)

    def test_gendered_pronoun_inside_predicate_is_rejected_without_replacement(self):
        altered=deepcopy(self.raw);altered['records'][0]['predicate']='reported that he recorded event 1.'
        before=deepcopy(altered);accepted,rejected=self.validate(altered)
        self.assertEqual(accepted,[])
        self.assertIn('ordering_actor_reference_policy_forbidden_pronoun:I3:he',rejected[0]['reasons'])
        self.assertEqual(altered,before)

    def test_writer_plan_is_source_order_without_label_or_answer_mapping(self):
        public=read(self.run/'WRITER_PLAN.json')
        records=public['questions'][0]['anchor_records']
        self.assertEqual([r['anchor_id'] for r in records],[t['turn_ref'] for t in self.source['turns']])
        self.assertTrue(all(set(r)=={'anchor_id','speaker','quote','statement_prefix'} for r in records))
        self.assertNotIn('"I1"',json.dumps(public));self.assertNotIn('"gold"',json.dumps(public))
        self.assertFalse(public['display_labels_supplied']);self.assertFalse(public['gold_supplied_to_writer'])

    def test_tampered_writer_plan_rejected_even_with_new_binding(self):
        path=self.run/'WRITER_PLAN.json';value=read(path)
        value['questions'][0]['anchor_records'][0]['statement_prefix']='OtherActor '
        path.write_text(json.dumps(value));self.manifest['writer_plan']=bind(path)
        with self.assertRaisesRegex(ValueError,'writer_plan_reconstruction_mismatch'):
            self.validate(self.raw)

    def test_missing_or_changed_plan_stops_even_malformed_raw(self):
        path = self.run / 'GOLD_PLAN.json'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate(None)
        path.write_bytes(original)
        value = read(path); value['questions'][0]['gold'].reverse()
        path.write_text(json.dumps(value))
        self.manifest['gold_plan'] = bind(path)
        with self.assertRaisesRegex(ValueError, 'program_gold_reconstruction_mismatch'):
            self.validate(self.raw)

    def test_answer_plan_cannot_be_created_after_generation_started(self):
        write(self.run / 'GENERATION.STARTED.json', {'started_at_utc': '2026-10-13T23:59:00+00:00'})
        with self.assertRaisesRegex(ValueError, 'program_gold_after_generation_started'):
            self.validate(self.raw)

    def test_existing_failure_artifact_and_plan_are_append_only(self):
        failure = self.run / 'REJECTIONS.json'
        write(failure, [{'candidate_id': self.raw['candidate_id'], 'reasons': ['historical_failure']}])
        before = failure.read_bytes()
        with self.assertRaises(FileExistsError):
            write(failure, [])
        with self.assertRaises(FileExistsError):
            write(self.run / 'GOLD_PLAN.json', self.record)
        self.assertEqual(failure.read_bytes(), before)

    def semantic_review(self, wrong_topic=False):
        raw = deepcopy(self.raw)
        if wrong_topic:
            raw['records'][0]['predicate'] = 'discussed topic 1.'
        candidates, rejected = self.validate(raw)
        self.assertEqual(rejected, [])
        q = candidates[0]; public = generation.public(q)
        judgment = deepcopy(public)
        judgment.update(complete_source_turns_read=6, coherent=True, unique_order=True,
                        neutral_stem_ok=True, alternative_orders=[])
        judgment['items'] = {}
        for label, anchor in self.plan['anchors'].items():
            quote = q['items'][label]
            judgment['items'][label] = {'proposition': quote,
                **{key: True for key in review.ITEM_BOOLEANS},
                'order_anchor': {'turn_ref': anchor['turn_ref'], 'quote': quote},
                'support_refs': [anchor['turn_ref']], 'alternative_matching_anchors': []}
        checked = review._review_row(judgment, public, self.source, load_ordering_gate())
        self.assertEqual(checked['reasons'], [])
        self.assertEqual(checked['computed_order'], q['gold'])
        source_review_path = self.home / 'ordering/SOURCE_JUDGMENTS.json'
        write(source_review_path, {'fixture_only': True, 'questions': [judgment]})
        write(self.home / 'ordering/POOL.json', {'questions': candidates})
        source_binding = bind(source_review_path)
        pool_binding = bind(self.home / 'ordering/POOL.json')
        context = {'created_at_utc': '2026-10-14T00:02:00+00:00',
                   'generation_files': {pool_binding['path']: pool_binding},
                   'questions': {q['candidate_id']: public}, 'sources': {q['root_id']: self.source},
                   'checks': [checked], 'bindings': [source_binding], 'judgment_binding': source_binding,
                   'judgment': {'reviewer': 'synthetic-fixture', 'public_questions': {'sha256': '0' * 64}}}
        return review._private_review(self.home, context)

    def test_matching_semantic_anchors_pass_private_intersection(self):
        result = self.semantic_review()
        self.assertEqual(result['checks'][0]['status'], 'USABLE')
        self.assertEqual(len(result['accepted']), 1)

    def test_correct_program_gold_cannot_override_wrong_fixed_anchor_meaning(self):
        result = self.semantic_review(wrong_topic=True)
        self.assertEqual(result['checks'][0]['status'], 'EXCLUDED')
        self.assertIn('I3:fixed_anchor_disagrees_with_independent_semantic_anchor', result['checks'][0]['reasons'])
        self.assertNotIn('private_gold_disagrees_with_independent_source_order', result['checks'][0]['reasons'])
        self.assertEqual(result['accepted'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
