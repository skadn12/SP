from pathlib import Path
import tempfile
import json
import unittest
import dedup


def question(qid='new', root='r'):
    return {'candidate_id': qid, 'root_id': root, 'question_type': 'multi_select',
            'question': 'Which?', 'options': {x: 'statement ' + x for x in 'ABCDEF'}}


class DedupTest(unittest.TestCase):
    def test_unselected_warning_does_not_ban_but_batch_duplicate_does(self):
        q = question()
        prior = [{'candidate_id': 'old-perfect', 'facts': list(q['options'].values()),
                  'hard_dedup': False, 'history_status': 'HISTORICAL_CANDIDATE_WARNING'}]
        accepted, rejected = dedup.deduplicate([q, question('copy')], prior)
        self.assertEqual([x['candidate_id'] for x in accepted], ['new'])
        self.assertEqual(rejected[0]['errors'], ['duplicate_same_batch_statement_set'])

    def test_selected_export_is_bound_and_blocks_even_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / 'past/SELECTED_QUESTIONS.json'
            path.parent.mkdir();path.write_text(json.dumps({'questions': [question('old')]}))
            ignored = folder / 'current/SELECTED_QUESTIONS.json'
            ignored.parent.mkdir();ignored.write_text(json.dumps({'questions': [question('self')]}))
            rows, files = dedup.selected_history(folder, 'r', folder / 'current')
            self.assertEqual(files, [path])
            self.assertEqual(rows[0]['solver_model'], 'historical_not_inferred')
            q = question();q['options'] = dict(zip('ABCDEF', reversed(list(q['options'].values()))))
            accepted, rejected = dedup.deduplicate([q], rows)
            self.assertFalse(accepted)
            self.assertEqual(rejected[0]['matches'][0]['source_path'], str(path))
            self.assertEqual(rejected[0]['errors'], ['duplicate_historical_selected_statement_set'])

    def test_solver_result_alone_does_not_promote(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'RESULTS.json').write_text(json.dumps({'questions': [{**question(), 'selected_by_policy': True}]}))
            self.assertEqual(dedup.selected_history(folder, 'r', folder / 'current'), ([], []))


if __name__ == '__main__':
    unittest.main()
