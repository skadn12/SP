"""Focused checks for the explicitly requested citation-free diagnostic."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'evaluation'))
import protocol
from fractions import Fraction
from scoring import score_response
from selection import select


class CitationAblationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        source = self.folder / 'source.json'
        source.write_text(json.dumps({'root_id': 'r', 'turns': [
            {'turn_ref': 'R02:A:S01:T001', 'speaker': 'X', 'text': 'Original dialogue.'}]}))
        self.question = {'candidate_id': 'r-q', 'root_id': 'r', 'question_type': 'multi_select',
            'question': 'Select all that apply.', 'options': {c: 'Statement ' + c for c in 'ABCDEF'},
            'gold': ['A', 'C'], 'source_path': str(source), 'source_sha256': protocol.sha(source),
            'option_evidence': {c: {'turn_ref': 'WRONG_REF', 'speaker': 'WRONG_SPEAKER',
                'quote': 'Not an exact quote...', 'relation': 'supports' if c in 'AC' else 'contradicts',
                'reason': ''} for c in 'ABCDEF'}}
        self.pool = {'version': 'h4_r02_multiselect_format_ablation_v13',
            'citation_gate': 'DIAGNOSTIC_ONLY_DISABLED_BY_USER', 'diagnostic_only': True,
            'provenance': {'requested_model': 'Qwen/Qwen3.6-35B-A3B'}, 'questions': [self.question]}

    def validate(self):
        path = self.folder / 'pool.json'
        path.write_text(json.dumps(self.pool))
        return protocol.validate_pool(path)

    def test_bad_citations_do_not_block_or_get_repaired(self):
        original = deepcopy(self.question)
        _, qs, _ = self.validate()
        self.assertEqual(qs[0], original)

    def test_scope_is_explicit_and_cannot_bypass_other_pools(self):
        self.pool['diagnostic_only'] = False
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_CITATION_ABLATION_POOL_REQUIRED'):
            self.validate()

    def test_empty_gold_remains_unscorable(self):
        self.question['gold'] = []
        with self.assertRaisesRegex(ValueError, 'GOLD_CARDINALITY_INVALID'):
            self.validate()

    def test_relation_and_gold_cannot_be_relabelled(self):
        self.question['gold'] = ['A', 'B']
        with self.assertRaisesRegex(ValueError, 'RAW_RELATION_GOLD_MISMATCH'):
            self.validate()

    def test_changed_dialogue_remains_rejected(self):
        self.question['source_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SOURCE_PATH_OR_SHA_INVALID'):
            self.validate()

    def test_permutations_and_public_prompts_ignore_gold_and_bad_evidence(self):
        variants = protocol.make_variants(self.question, seed=123, count=6)
        changed = deepcopy(self.question)
        changed['gold'] = ['B', 'D', 'F']
        changed['option_evidence'] = {}
        self.assertEqual(variants, protocol.make_variants(changed, seed=123, count=6))
        source = {'turns': [{'speaker': 'X', 'text': 'Original dialogue.'}]}
        self.assertEqual(protocol.prompt(self.question, variants[0], source),
                         protocol.prompt(changed, variants[0], source))
        self.assertNotIn('Not an exact quote', protocol.prompt(self.question, variants[0], source))
        content = {text: c for c, text in variants[0]['content_to_text'].items()}
        q = {**self.question, 'gold_content_ids': [content[self.question['options'][c]] for c in 'AC']}
        labels = [label for label, cid in variants[0]['display_to_content'].items() if cid in q['gold_content_ids']]
        score = score_response(q, {'variant': variants[0]}, '<box>' + ','.join(labels) + '</box>')
        self.assertEqual(score['partial_score_fraction'], '1')
        scores = {f'perm_{i:02}': [Fraction(1, 3)] * 3 for i in range(6)}
        self.assertTrue(select(scores, list(scores))['selected_by_policy'])


if __name__ == '__main__':
    unittest.main()
