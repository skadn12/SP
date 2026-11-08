"""Regression checks for historical SOURCE_HOLD staging and exact partial resume."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch_prepare_v121 as stage


class PrefixStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / 'current'
        self.old = Path(self.tmp.name) / 'history' / 'units/root74644'
        self.unit = self.batch / 'units/root74644'
        self.root = 'cc-train-episode-74644'
        self.patch = patch.object(stage, 'B', self.batch)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.write(self.old / 'source_repair/AUTHOR_PAYLOAD.json',
                   {'continuation_constraints': ['Keep the inherited synthetic fact boundary.']})
        self.write(self.unit / 'inputs/BASELINE.json', {'root_id': self.root, 'turns': []})
        self.write(self.unit / 'inputs/CONTINUATION_CONSTRAINTS.json',
                   {'root_id': self.root, 'continuation_constraints': ['Fresh dual-review constraint.']})
        self.write(self.batch / 'ROLE_ASSIGNMENTS.json', {'roots': []})
        self.write(self.batch / 'tools/prefix_review_v2.py', {'fixture': 'prefix tool'})
        self.write(self.batch / 'tools/batch_prepare_v121.py', {'fixture': 'staging tool'})
        reviews = []
        for n in (1, 2):
            p = self.unit / f'baseline_reviews/reader{n}/REVIEW.json'
            self.write(p, {'reviewer': f'/root/fixture{n}'})
            reviews.append(stage.bind(p))
        output = self.unit / 'source_repair/SOURCE_OUTPUT.json'
        self.write(output, {'root_id': self.root, 'baseline_binding': stage.bind(self.unit / 'inputs/BASELINE.json'),
                            'review_bindings': reviews})
        self.write(self.unit / 'source_repair/FINALIZE_RECEIPT.json',
                   {'status': 'SOURCE_READY', 'source_output_binding': stage.bind(output), 'review_bindings': reviews})
        self.row = {'root_id': self.root, 'unit_path': str(self.unit), 'old_unit_path': str(self.old),
                    'proposed_retained_scene_refs': [], 'a_plan_mode': 'NEW_SELECTION',
                    'proposed_retry_start': 'R02:A:S01'}
        self.write(self.batch / 'REPAIR_SCOPE.json', {'roots': [self.row]})
        self.manifest = {'roots': [{'root_id': self.root, 'inputs': [
            {'label': 'inherited_source_constraints',
             **stage.bind(self.old / 'source_repair/AUTHOR_PAYLOAD.json')}]}]}
        self.freeze_manifest()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def freeze_manifest(self):
        self.write(self.batch / 'SOURCE_INPUTS.json', self.manifest)
        self.write(self.batch / 'SOURCE_PROTOCOL_FREEZE.json',
                   {'bindings': [stage.bind(self.batch / 'SOURCE_INPUTS.json')]})

    def run_prepare(self):
        with redirect_stdout(io.StringIO()):
            stage.prepare_prefix()

    def test_source_hold_history_without_old_gate_uses_declared_author_constraints(self):
        self.assertFalse((self.old / 'inputs/CONTINUATION_CONSTRAINTS.json').exists())
        self.run_prepare()
        adaptation = stage.read(self.unit / 'CONSTRAINTS_PATH_ADAPTATION.json')
        self.assertEqual(adaptation['original_binding'],
                         stage.bind(self.old / 'source_repair/AUTHOR_PAYLOAD.json'))
        self.assertEqual(adaptation['historical_unit_path'], str(self.old.resolve()))
        self.assertEqual(adaptation['source_inputs_binding'], stage.bind(self.batch / 'SOURCE_INPUTS.json'))
        self.assertEqual(len(stage.read(self.batch / 'PREFIX_INPUTS.json')['roots']), 1)
        for record in stage.read(self.batch / 'PREFIX_PROTOCOL_FREEZE.json')['bindings']:
            stage.check(record)

    def test_exact_preexisting_aplan_is_resumed_without_rewrite(self):
        p = self.unit / 'A_PLAN_POLICY.json'
        self.write(p, {'root_id': self.root, 'plan_mode': 'NEW_SELECTION', 'a_continuations': [],
                       'selection_rule': 'Choose three source-evidence-bound continuation plans during actual frozen matching; align numbered R01 scenes where all three exist.'})
        before = p.read_bytes(), p.stat().st_mtime_ns
        self.run_prepare()
        self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before)

    def test_partial_staging_resumes_exact_aplan_and_adaptation(self):
        actual_save = stage.save
        def interrupted(path, value):
            if Path(path).name == 'PREFIX_INPUTS.json':
                raise RuntimeError('fixture interrupted before prefix manifest')
            actual_save(path, value)
        with patch.object(stage, 'save', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
                self.run_prepare()
        paths = [self.unit / n for n in ('A_PLAN_POLICY.json', 'CONSTRAINTS_PATH_ADAPTATION.json')]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        self.run_prepare()
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}, before)

    def test_different_existing_aplan_is_rejected_without_overwrite(self):
        p = self.unit / 'A_PLAN_POLICY.json'
        self.write(p, {'root_id': self.root, 'plan_mode': 'INHERIT_A', 'a_continuations': []})
        before = p.read_bytes()
        with self.assertRaisesRegex(ValueError, 'EXISTING_STAGING_OBJECT_DIFFERS'):
            self.run_prepare()
        self.assertEqual(p.read_bytes(), before)
        self.assertFalse((self.batch / 'PREFIX_INPUTS.json').exists())

    def test_changed_historical_constraints_hash_is_rejected(self):
        self.write(self.old / 'source_repair/AUTHOR_PAYLOAD.json', {'continuation_constraints': ['Changed.']})
        with self.assertRaisesRegex(ValueError, 'BOUND_FILE_CHANGED'):
            self.run_prepare()
        self.assertFalse((self.unit / 'A_PLAN_POLICY.json').exists())

    def test_changed_frozen_source_manifest_is_rejected(self):
        p = self.batch / 'SOURCE_INPUTS.json'
        p.write_text(p.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'BOUND_FILE_CHANGED'):
            self.run_prepare()

    def test_missing_declared_historical_constraints_is_rejected(self):
        self.manifest['roots'][0]['inputs'] = []
        self.freeze_manifest()
        with self.assertRaisesRegex(ValueError, 'DECLARED_HISTORICAL_CONSTRAINTS_REQUIRED'):
            self.run_prepare()

    def test_frozen_constraints_can_come_from_a_different_prior_gate_batch(self):
        foreign = self.batch / 'foreign.json'
        self.write(foreign, {'root_id': self.root, 'continuation_constraints': ['Newer inherited gate.']})
        self.manifest['roots'][0]['inputs'][0] = {'label': 'inherited_source_constraints', **stage.bind(foreign)}
        self.freeze_manifest()
        self.run_prepare()
        adaptation = stage.read(self.unit / 'CONSTRAINTS_PATH_ADAPTATION.json')
        self.assertEqual(adaptation['original_binding'], stage.bind(foreign))
        self.assertEqual(adaptation['historical_unit_path'], str(self.old.resolve()))

    def test_declared_constraints_for_a_different_root_are_rejected(self):
        foreign = self.batch / 'foreign.json'
        self.write(foreign, {'root_id': 'cc-train-episode-999', 'continuation_constraints': ['Wrong root.']})
        self.manifest['roots'][0]['inputs'][0] = {'label': 'inherited_source_constraints', **stage.bind(foreign)}
        self.freeze_manifest()
        with self.assertRaisesRegex(ValueError, 'HISTORICAL_CONSTRAINTS_ROOT_MISMATCH'):
            self.run_prepare()


if __name__ == '__main__':
    unittest.main()
