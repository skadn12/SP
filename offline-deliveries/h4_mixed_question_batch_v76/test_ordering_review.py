"""Synthetic saved-review lifecycle tests; no generation/model/solver calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ordering_review as review
from common import bind, read, sha, write


def timestamp(offset=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


def replace(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fixture(home, repeated=False):
    """All behavior declarations below describe this synthetic fixture only."""
    folder = home / 'ordering'; slot = folder / 'root_ordering'
    source_path = slot / 'SOURCE.json'; original_path = home / 'frozen_source.json'
    source = {'root_id': 'root', 'turns': [
        {'turn_ref': 'S1:T' + str(n + 1), 'speaker': 'A' if n % 2 else 'B', 'text': text}
        for n, text in enumerate([
            'Context.', 'First act. First act.' if repeated else 'First act.',
            'Second alternate act.', 'First alternate act.', 'Second act.',
            'Other context.', 'More context.', 'Third act.', 'Fourth act.',
            'Fifth act.', 'Sixth act.', 'Closing context.',
        ])]}
    write(original_path, source); write(source_path, source)
    write(home / 'CONFIG.json', {'ordering_slots': [{
        'slot_id': 'root_ordering', 'root_id': 'root', 'source_path': str(original_path),
        'source_sha256': sha(original_path), 'history_path': str(original_path),
        'history_sha256': sha(original_path), 'candidate_count': 1,
    }]})
    public = {'candidate_id': 'question-1', 'root_id': 'root', 'question_type': 'ordering',
              'chronology_kind': 'conversation_order',
              'question': 'Order these statements as they APPEAR IN THE SUPPLIED DIALOGUE.',
              'items': {label: text for label, text in zip(review.LABELS,
                  ['First act.', 'Second act.', 'Third act.', 'Fourth act.', 'Fifth act.', 'Sixth act.'])},
              'source_path': str(source_path), 'source_sha256': sha(source_path)}
    private = {**deepcopy(public), 'gold': list(review.LABELS),
               'item_evidence': {'private': 'Never exposed to the source phase'}}
    public_path = folder / 'PUBLIC_QUESTIONS.json'; pool_path = folder / 'POOL.json'
    write(public_path, {'questions': [public]}); write(pool_path, {'questions': [private]})
    write(folder / 'GENERATION.COMPLETE.json', {
        'version': 'v76_ordering_generation_complete_v1', 'status': 'GENERATED',
        'completed_at_utc': timestamp(-2), 'candidate_count': 1,
        'files': [bind(public_path), bind(pool_path), bind(source_path)],
    })
    judgment = {
        'review_schema_version': 'source_judgment_v1', 'reviewer': 'synthetic-test-reviewer',
        'created_at_utc': timestamp(-1), **review.DECLARATIONS,
        'public_questions': bind(public_path),
        'sources': [{'root_id': 'root', **bind(source_path), 'complete_turn_count': len(source['turns'])}],
        'questions': [{**deepcopy(public), 'complete_source_turns_read': len(source['turns']),
            'coherent': True, 'unique_order': True, 'neutral_stem_ok': True, 'alternative_orders': [],
            'items': {label: {
                'proposition': public['items'][label], **{key: True for key in review.ITEM_BOOLEANS},
                'order_anchor': {'turn_ref': 'S1:T' + str(n), 'quote': public['items'][label]},
                'support_refs': ['S1:T' + str(n)], 'alternative_matching_anchors': [],
            } for label, n in zip(review.LABELS, [2, 5, 8, 9, 10, 11])},
        }],
    }
    write(folder / 'SOURCE_JUDGMENTS.json', judgment)
    return judgment


class OrderingReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name).resolve()
        self.judgment = fixture(self.home)
        self.folder = self.home / 'ordering'
        self.judgment_path = self.folder / 'SOURCE_JUDGMENTS.json'

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        replace(self.judgment_path, self.judgment)

    def rebind_output(self, name):
        path = self.folder / 'CONTENT_REVIEW.COMPLETE.json'
        complete = read(path)
        complete['files'] = [bind(row['path']) if Path(row['path']).name == name else row
                             for row in complete['files']]
        replace(path, complete)

    def assert_no_private_read(self, action):
        original_read = review.read; original_sha = review.sha
        seen = []
        def guarded_read(path):
            if Path(path).name == 'POOL.json':
                seen.append('read')
                self.fail('Invalid source review opened the private pool')
            return original_read(path)
        def guarded_sha(path):
            if Path(path).name == 'POOL.json':
                seen.append('sha')
                self.fail('Invalid source review hashed the private pool')
            return original_sha(path)
        with patch.object(review, 'read', guarded_read), patch.object(review, 'sha', guarded_sha):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(seen, [])

    def test_usable_pool_is_frozen_and_replayed(self):
        before = bind(self.folder / 'POOL.json')
        report = review.finish_review(self.home)
        self.assertEqual(report['accepted_count'], 1)
        self.assertEqual(report['rejected_count'], 0)
        row = read(self.folder / 'EVALUATION_POOL.json')['questions'][0]
        self.assertEqual(row['gold'], list(review.LABELS))
        self.assertEqual(row['order_anchor_spans']['I1']['char_start'], 0)
        self.assertEqual(set(row['allowed_matching_anchor_spans']), set(review.LABELS))
        self.assertEqual(row['independent_source_review']['source_judgments'], bind(self.judgment_path))
        self.assertNotIn('qwen_review', row)
        self.assertEqual(review.verify_review(self.home), report)
        self.assertEqual(bind(self.folder / 'POOL.json'), before)
        with self.assertRaisesRegex(ValueError, 'already_exist'):
            review.finish_review(self.home)

    def test_missing_and_wrong_declarations_block_private_access(self):
        original = deepcopy(self.judgment)
        for key, expected in review.DECLARATIONS.items():
            for value in ('missing', 0, int(expected), not expected):
                with self.subTest(key=key, value=value):
                    self.judgment = deepcopy(original)
                    if value == 'missing':
                        self.judgment.pop(key)
                    else:
                        self.judgment[key] = value
                    self.save()
                    self.assert_no_private_read(lambda: review.finish_review(self.home))
        self.assertFalse((self.folder / 'CONTENT_REVIEW.json').exists())

    def test_public_and_whole_question_coverage_block_private_access(self):
        original = deepcopy(self.judgment)
        modifications = [
            lambda j: j.update(questions=[]),
            lambda j: j['questions'].append(deepcopy(j['questions'][0])),
            lambda j: j['questions'][0]['items'].pop('I6'),
            lambda j: j['questions'][0]['items']['I1'].update(proposition='Changed claim.'),
            lambda j: j['questions'][0].update(question='Changed stem.'),
            lambda j: j['questions'][0].update(root_id='other'),
            lambda j: j['questions'][0].update(complete_source_turns_read=11),
            lambda j: j['sources'][0].update(complete_turn_count=11),
            lambda j: j['questions'][0]['items']['I1'].update(support_refs=['S1:T999']),
            lambda j: j['questions'][0]['items']['I1'].update(supported=1),
            lambda j: j['questions'][0].update(gold=list(review.LABELS)),
        ]
        for modification in modifications:
            with self.subTest(modification=modification):
                self.judgment = deepcopy(original); modification(self.judgment); self.save()
                self.assert_no_private_read(lambda: review.finish_review(self.home))

    def test_naive_future_and_pregeneration_timestamps_block_private_access(self):
        for value in (datetime.now().isoformat(), timestamp(60), timestamp(-60), None):
            with self.subTest(value=value):
                self.judgment['created_at_utc'] = value; self.save()
                self.assert_no_private_read(lambda: review.finish_review(self.home))

    def test_wrong_public_source_and_generation_binding_block_private_access(self):
        original = deepcopy(self.judgment)
        for key in ('public_questions', 'sources'):
            with self.subTest(key=key):
                self.judgment = deepcopy(original)
                row = self.judgment[key][0] if key == 'sources' else self.judgment[key]
                row['sha256'] = '0' * 64; self.save()
                self.assert_no_private_read(lambda: review.finish_review(self.home))
        self.judgment = original; self.save()
        generation_path = self.folder / 'GENERATION.COMPLETE.json'
        generation = read(generation_path)
        generation['files'][0]['sha256'] = '0' * 64; replace(generation_path, generation)
        self.assert_no_private_read(lambda: review.finish_review(self.home))

    def test_private_gold_disagreement_excludes_without_repair(self):
        path = self.folder / 'POOL.json'; pool = read(path)
        pool['questions'][0]['gold'] = list(reversed(review.LABELS)); replace(path, pool)
        generation_path = self.folder / 'GENERATION.COMPLETE.json'; generation = read(generation_path)
        generation['files'][1] = bind(path); replace(generation_path, generation)
        before = bind(path)
        report = review.finish_review(self.home)
        self.assertEqual(report['accepted_count'], 0)
        self.assertIn('private_gold_disagrees_with_independent_source_order', report['questions'][0]['reasons'])
        self.assertEqual(bind(path), before)
        self.assertEqual(read(self.folder / 'EVALUATION_POOL.json')['questions'], [])
        review.verify_review(self.home)

    def test_cartesian_alternatives_cannot_jointly_cross(self):
        items = self.judgment['questions'][0]['items']
        items['I1']['alternative_matching_anchors'] = [{'turn_ref': 'S1:T4', 'quote': 'First alternate act.'}]
        items['I2']['alternative_matching_anchors'] = [{'turn_ref': 'S1:T3', 'quote': 'Second alternate act.'}]
        self.save()
        report = review.finish_review(self.home)
        self.assertEqual(report['accepted_count'], 0)
        self.assertIn('alternative_anchor_changes_or_ties_order:I1:I2', report['questions'][0]['reasons'])

    def test_alternative_order_cannot_change_gold(self):
        self.judgment['questions'][0]['alternative_orders'] = [list(reversed(review.LABELS))]
        self.save()
        report = review.finish_review(self.home)
        self.assertIn('alternative_order_changes_gold', report['questions'][0]['reasons'])
        self.assertEqual(report['accepted_count'], 0)

    def test_repeated_quote_is_substantive_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve(); fixture(home, repeated=True)
            report = review.finish_review(home)
            self.assertEqual(report['accepted_count'], 0)
            self.assertIn('I1:quote_not_unique_in_turn', report['questions'][0]['reasons'])

    def test_review_false_boolean_is_exclusion_not_overridden(self):
        self.judgment['questions'][0]['unique_order'] = False
        self.save(); before = bind(self.judgment_path)
        report = review.finish_review(self.home)
        self.assertIn('unique_order_failed', report['questions'][0]['reasons'])
        self.assertEqual(bind(self.judgment_path), before)

    def test_frozen_source_judgment_change_blocks_private_access_on_replay(self):
        review.finish_review(self.home)
        self.judgment['full_source_read'] = False; self.save()
        self.assert_no_private_read(lambda: review.verify_review(self.home))

    def test_frozen_inputs_changes_reject_even_with_valid_shape(self):
        review.finish_review(self.home)
        self.judgment['reviewer'] = 'different-reviewer'; self.save()
        self.assert_no_private_read(lambda: review.verify_review(self.home))

    def test_altered_alternative_span_fails_replay(self):
        review.finish_review(self.home)
        path = self.folder / 'EVALUATION_POOL.json'; pool = read(path)
        pool['questions'][0]['allowed_matching_anchor_spans']['I1'][0]['char_end'] += 1
        replace(path, pool); self.rebind_output('EVALUATION_POOL.json')
        with self.assertRaisesRegex(ValueError, 'evaluation_pool_replay_mismatch'):
            review.verify_review(self.home)

    def test_review_time_is_rechecked_even_after_rebinding_outputs(self):
        review.finish_review(self.home)
        path = self.folder / 'CONTENT_REVIEW.json'; report = read(path)
        report['first_private_pool_read_at_utc'] = timestamp(-60)
        replace(path, report); self.rebind_output('CONTENT_REVIEW.json')
        self.assert_no_private_read(lambda: review.verify_review(self.home))

    def test_public_must_contain_no_private_answer_fields(self):
        path = self.folder / 'PUBLIC_QUESTIONS.json'; public = read(path)
        public['questions'][0]['gold'] = list(review.LABELS); replace(path, public)
        self.assert_no_private_read(lambda: review.finish_review(self.home))


if __name__ == '__main__':
    unittest.main()
