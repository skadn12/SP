"""Negative controls for the derivation boundary; no disk writes or model calls."""
import sys
from pathlib import Path
from copy import deepcopy
from unittest import TestCase, main
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch

class DerivationBoundaries(TestCase):
    def mutated(self, target, mutation):
        original_read = batch.read
        def changed(path):
            value = original_read(path)
            if str(path).endswith(target):
                value = deepcopy(value); mutation(value)
            return value
        # Config binding verification is separate; exercise the transformation
        # contract itself against adversarial input while skipping that cache.
        frozen_config = batch.config()
        with patch.object(batch, 'read', changed), patch.object(batch, 'config', return_value=frozen_config):
            return batch.projected_rows()

    def test_exact_preservation(self):
        rows, edits = batch.projected_rows()
        old = batch.read(batch.PREVIOUS / 'ordering/POOL.json')['questions']
        self.assertEqual((len(rows), len(edits)), (3, 12))
        for before, after in zip(old, rows):
            for key in set(before) - {'candidate_id', 'items', 'source_path'}:
                self.assertEqual(before[key], after[key], key)

    def test_disallow_other_proposition_changes(self):
        def mutate(d):
            item = next(iter(d['questions'][0]['edits'].values()))
            item['substitutions'].append({'old':'amazing', 'new':'terrible'})
            item['after'] = item['after'].replace('amazing','terrible')
        with self.assertRaisesRegex(ValueError, 'only_known_actor'):
            self.mutated('REPLACEMENT_PLAN.json', mutate)

    def test_disallow_wrong_identity(self):
        with self.assertRaisesRegex(ValueError, 'new_identity_required'):
            self.mutated('REPLACEMENT_PLAN.json', lambda d:d['questions'][0].update(candidate_id='wrong'))

    def test_disallow_missing_failed_item(self):
        with self.assertRaisesRegex(ValueError, 'exact_failed_item_coverage'):
            self.mutated('REPLACEMENT_PLAN.json', lambda d:d['questions'][0]['edits'].pop('I2'))

    def test_disallow_wrong_gold_parent(self):
        with self.assertRaisesRegex(ValueError, 'original_gold_must_be_retained'):
            self.mutated('v81/ordering/POOL.json', lambda d:d['questions'][0]['gold'].reverse())

    def test_disallow_other_failure_type(self):
        with self.assertRaisesRegex(ValueError, 'only_actor_failures'):
            self.mutated('v81/ordering/CONTENT_REVIEW.json', lambda d:d['questions'][0]['reasons'].append('I1:state_ok_failed'))

if __name__ == '__main__':
    main()
