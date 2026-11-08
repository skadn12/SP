"""Counterfactual replay control flow over the bounded 22 frozen responses."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from . import replay as replay_module

SOURCE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_writer_model_comparison_v1_live_1')


class ReplayCounterfactualTests(unittest.TestCase):
    def test_unneeded_historical_repairs_are_not_read_or_admitted(self):
        with tempfile.TemporaryDirectory() as temp, forbid_network(), patch.object(
                replay_module, 'repair_required', return_value=False):
            target = Path(temp) / 'replay'
            result = replay_module.replay(SOURCE, target)
            slots = [r for r in replay_module.read(target / 'slot_replay.json') if r['admission_policy'] == 'partial_v1']
            self.assertTrue(all(r['stages_used'] == ['primary'] for r in slots))
            self.assertEqual(result['unused_historical_repairs_partial'], 10)
            self.assertEqual(result['counterfactual_unobserved'], [])
            self.assertEqual(len(list((target / 'partial_v1').rglob('*-repair-*.json.txt'))), 0)

    def test_needed_unobserved_repairs_are_not_borrowed(self):
        calls = []
        original = replay_module.normalize_stage
        def observed(raw, **kwargs):
            calls.append((kwargs['ordering_context']['arm_id'], kwargs['root_id'],
                          kwargs['intent_id'], kwargs['stage'], kwargs['request_id']))
            return original(raw, **kwargs)
        with tempfile.TemporaryDirectory() as temp, forbid_network(), patch.object(
                replay_module, 'repair_required', return_value=True), patch.object(
                replay_module, 'normalize_stage', side_effect=observed):
            target = Path(temp) / 'replay'
            result = replay_module.replay(SOURCE, target)
            missing = result['counterfactual_unobserved']
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(r['admission_policy'] == 'partial_v1' for r in missing))
            self.assertTrue(all(r['counterfactual_status'] == 'COUNTERFACTUAL_OUTPUT_UNOBSERVED' for r in missing))
            self.assertTrue(all(r['stages_used'] == ['primary'] and not r['historical_repair_observed'] for r in missing))
            self.assertEqual(len(calls), 22)
            self.assertEqual(sum(r[3] == 'repair' for r in calls), 10)
            for arm, root, intent, stage, request_id in calls:
                original_root = replay_module.directory(SOURCE, {'arm_id': arm, 'root_id': root})
                self.assertTrue((original_root / 'raw/responses' / (request_id + '.json.txt')).is_file())
            self.assertTrue(result['legacy_final_identity_match'])
            self.assertEqual(result['new_external_calls'], 0)


if __name__ == '__main__':
    unittest.main()
