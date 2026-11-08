"""Isolated synthetic staging/denominator tests, never production judgments."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import json
import tempfile
import unittest

import batch_prepare_v121 as p
import prefix_review_v2 as review


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.previous = p.B
        p.B = Path(self.tmp.name) / 'SYNTHETIC_ONLY'
        self.root = 'cc-train-episode-1'; self.unit = p.B / 'units/root1'; self.old = p.B.parent / 'HISTORY_ONLY/root1'
        base = {'root_id': self.root, 'turns': [{'turn_ref': 'R01:S01:T001', 'speaker': 'A', 'text': 'Synthetic baseline.'}]}
        save(self.old / 'inputs/BASELINE.json', base); save(self.unit / 'inputs/BASELINE.json', base)
        save(self.old / 'inputs/CONTINUATION_CONSTRAINTS.json', {'continuation_constraints': ['Synthetic only.']})
        save(self.unit / 'inputs/CONTINUATION_CONSTRAINTS.json', {'continuation_constraints': ['Synthetic only.']})
        save(p.B / 'SOURCE_INPUTS.json', {'roots': [{'root_id': self.root, 'inputs': [
            {'label': 'inherited_source_constraints', **p.bind(self.old / 'inputs/CONTINUATION_CONSTRAINTS.json')}]}]})
        save(p.B / 'SOURCE_PROTOCOL_FREEZE.json', {'status': 'SYNTHETIC_FIXTURE_ONLY',
            'bindings': [p.bind(p.B / 'SOURCE_INPUTS.json')]})
        save(p.B / 'tools/prefix_review_v2.py', {'not_executed': 'Synthetic binding placeholder.'})
        save(p.B / 'tools/batch_prepare_v121.py', {'not_executed': 'Synthetic binding placeholder.'})
        self.rows = []
        for i in range(1, 25):
            unit = p.B / 'units' / f'root{i}'
            candidate = self.unit / 'inputs/BASELINE.json' if i == 1 else unit / 'source_repair/CANDIDATE_BASELINE.json'
            if i != 1: save(candidate, {'root_id': f'cc-train-episode-{i}', 'turns': base['turns']})
            reviews = []
            for reviewer in (1, 2):
                rp = unit / f'baseline_reviews/reader{reviewer}/REVIEW.json'; save(rp, {'synthetic': True}); reviews.append(p.bind(rp))
            output = unit / 'source_repair/SYNTHETIC_OUTPUT.json'
            save(output, {'root_id': f'cc-train-episode-{i}', 'baseline_binding': p.bind(candidate), 'review_bindings': reviews})
            save(unit / 'source_repair/FINALIZE_RECEIPT.json', {'status': 'SOURCE_READY' if i == 1 else 'SOURCE_HOLD',
                'source_output_binding': p.bind(output), 'source_status': 'UNCHANGED', 'review_bindings': reviews})
            self.rows.append({'root_id': f'cc-train-episode-{i}', 'unit_path': str(unit), 'old_unit_path': str(self.old),
                'a_plan_mode': 'NEW_SELECTION', 'proposed_retained_scene_refs': [],
                'proposed_new_scene_refs': p.SCENES, 'proposed_retry_start': p.SCENES[0]})
        save(p.B / 'REPAIR_SCOPE.json', {'fixed_candidate_count': 24, 'roots': self.rows})
        save(p.B / 'ROLE_ASSIGNMENTS.json', {'roots': [{'root_id': r['root_id'], 'author': '/root/fixture_author',
            'reviewer': '/root/fixture_reviewer', 'prefix_reviewer': '/root/fixture_reviewer'} for r in self.rows]})

    def tearDown(self):
        p.B = self.previous; self.tmp.cleanup()

    def quiet(self, function):
        with redirect_stdout(io.StringIO()): function()

    def bind_actual_helper(self):
        # Isolated fixture binds the actual helper executed, never a substitute.
        fp = p.B / 'PREFIX_PROTOCOL_FREEZE.json'; frozen = p.read(fp)
        frozen['bindings'] = [r for r in frozen['bindings'] if not r['path'].endswith('/tools/prefix_review_v2.py')]
        frozen['bindings'].append(p.bind(review.__file__)); save(fp, frozen)

    def prefix_review(self):
        self.bind_actual_helper()
        args = SimpleNamespace(command='read', batch=str(p.B), root=self.root, executor='/root/fixture_reviewer')
        self.quiet(lambda: review.run(args))
        folder = p.B / 'prefix_audits/root1'; context = p.read(folder / 'READ_CONTEXT.json')
        raw = {'root_id': self.root, 'executor': args.executor, 'decision': 'PASS', 'findings': [],
            'checks': {k: {'status': 'PASS', 'reason': 'Synthetic temporary fixture only.'} for k in review.CHECKS},
            'read_attestation': {**{k: True for k in ['full_baseline_read', 'full_candidate_prefix_read', 'all_constraints_read',
                'selection_read', 'retained_native_events_read', 'no_private_qa_or_prior_source_judgments_read']},
                'turn_refs_read': context['turn_refs_read'], 'continuation_constraints_read': context['continuation_constraints']}}
        rawp = folder / 'SYNTHETIC_JUDGMENT.json'; save(rawp, raw)
        args.command = 'save'; args.judgment = str(rawp); self.quiet(lambda: review.run(args))

    def retained_six(self):
        row = self.rows[0]; row.update({'a_plan_mode': 'INHERIT_A', 'proposed_retained_scene_refs': p.SCENES,
                                      'proposed_new_scene_refs': [], 'proposed_retry_start': 'QA'})
        save(p.B / 'REPAIR_SCOPE.json', {'fixed_candidate_count': 24, 'roots': self.rows})
        save(self.old / 'selection.json', {'a_continuations': [{'scene_ref': s, 'thread': 'Synthetic only.'} for s in p.SCENES[:3]]})
        previous = p.read(self.old / 'inputs/BASELINE.json')['turns']; reservations = []
        for scene in p.SCENES:
            folder = p.scene_dir(self.old, self.root, scene)
            turns = [{'turn_ref': scene + ':T001', 'scene_ref': scene, 'speaker': 'A', 'text': 'Synthetic only ' + scene}]
            save(folder / 'EXPANSION_DRAFT.json', {'scene_ref': scene, 'author': '/root/historical_fixture_author', 'turns': turns})
            previous = previous + turns; source = {'root_id': self.root, 'turns': previous}
            save(folder / 'CUMULATIVE_SOURCE.json', source)
            save(folder / 'SOURCE_REVIEW.json', {'decision': 'READY', 'reviewer': '/root/historical_fixture_reviewer',
                                               'source_sha256': p.bind(folder / 'CUMULATIVE_SOURCE.json')['sha256']})
            if ':B:' in scene:
                eventp = self.old / 'inputs' / (scene.replace(':', '_') + '_EVENT.json')
                save(eventp, {'turns': [{'turn_id': '0', 'utterance': 'Synthetic event only.'}]})
                reservations.append({'root_id': self.root, 'scene_ref': scene, 'source_item_path': str(eventp),
                                     'source_item_sha256': p.bind(eventp)['sha256']})
        save(self.old / 'SOURCE_RESERVATIONS.json', {'allocation': reservations})

    def test_requires_every_source_terminal_before_prefix_freeze(self):
        (p.B / 'units/root24/source_repair/FINALIZE_RECEIPT.json').unlink()
        with self.assertRaisesRegex(ValueError, 'SOURCE_NOT_FINALIZED'):
            self.quiet(p.prepare_prefix)
        self.assertFalse((p.B / 'PREFIX_INPUTS.json').exists())

    def test_source_hold_keeps_fixed_denominator_and_no_prefix_attempt(self):
        self.quiet(p.prepare_prefix); self.prefix_review(); self.quiet(p.import_prefix)
        manifest = p.read(p.B / 'MANIFEST.json')
        self.assertEqual((manifest['fixed_candidates'], manifest['source_ready'], manifest['source_hold']), (24, 1, 23))
        self.assertEqual((manifest['prefix_pass'], manifest['prefix_hold'], len(manifest['roots'])), (1, 0, 24))
        self.assertEqual((manifest['new_scene_budget'], manifest['new_b_event_budget']), (6, 3))

    def test_six_scene_import_preserves_exact_history_and_zero_new_budget(self):
        self.retained_six(); self.quiet(p.prepare_prefix); self.prefix_review(); self.quiet(p.import_prefix)
        manifest = p.read(p.B / 'MANIFEST.json')
        self.assertEqual((manifest['new_scene_budget'], manifest['new_b_event_budget']), (0, 0))
        imported = p.read(self.unit / 'PREFIX_IMPORT.json')
        self.assertEqual(len(imported['scenes']), 6)
        for row in imported['scenes']:
            for entry in row['verbatim_artifacts'].values():
                self.assertEqual(Path(entry['historical_binding']['path']).read_bytes(), Path(entry['imported_binding']['path']).read_bytes())

    def test_edited_baseline_cannot_import_historical_prefix(self):
        self.retained_six(); document = p.read(self.unit / 'inputs/BASELINE.json')
        document['turns'][0]['text'] = 'Synthetic changed source.'; save(self.unit / 'inputs/BASELINE.json', document)
        # Model a legitimately finalized new edited source, incompatible with
        # the otherwise untouched historical prefix.
        outp = self.unit / 'source_repair/SYNTHETIC_OUTPUT.json'; output = p.read(outp)
        output['baseline_binding'] = p.bind(self.unit / 'inputs/BASELINE.json'); save(outp, output)
        fp = self.unit / 'source_repair/FINALIZE_RECEIPT.json'; final = p.read(fp)
        final['source_output_binding'] = p.bind(outp); save(fp, final)
        with self.assertRaisesRegex(ValueError, 'EDITED_SOURCE_CANNOT_IMPORT_OLD_PREFIX'):
            self.quiet(p.prepare_prefix)

    def test_prefix_frozen_input_change_blocks_import(self):
        self.quiet(p.prepare_prefix); self.prefix_review()
        (self.unit / 'inputs/BASELINE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'BOUND_FILE_CHANGED'):
            self.quiet(p.import_prefix)
        self.assertFalse((self.unit / 'PREFIX_IMPORT.json').exists())

    def test_source_ready_cannot_accept_baseline_changed_after_dual_review(self):
        (self.unit / 'inputs/BASELINE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'BOUND_FILE_CHANGED'):
            self.quiet(p.prepare_prefix)
        self.assertFalse((p.B / 'PREFIX_INPUTS.json').exists())

    def test_prefix_reader_requires_protocol_freeze(self):
        self.quiet(p.prepare_prefix)
        (p.B / 'PREFIX_PROTOCOL_FREEZE.json').unlink()
        args = SimpleNamespace(command='read', batch=str(p.B), root=self.root, executor='/root/fixture_reviewer')
        with self.assertRaisesRegex(ValueError, 'PREFIX_PROTOCOL_NOT_FROZEN'):
            self.quiet(lambda: review.run(args))
        self.assertFalse((p.B / 'prefix_audits/root1/READ_CONTEXT.json').exists())

    def test_prefix_reader_rejects_unfrozen_actual_runtime(self):
        self.quiet(p.prepare_prefix)
        args = SimpleNamespace(command='read', batch=str(p.B), root=self.root, executor='/root/fixture_reviewer')
        with self.assertRaisesRegex(ValueError, 'PREFIX_RUNTIME_NOT_FROZEN'):
            self.quiet(lambda: review.run(args))

    def test_prefix_reader_rejects_omitted_input_binding(self):
        self.quiet(p.prepare_prefix); self.bind_actual_helper()
        fp = p.B / 'PREFIX_PROTOCOL_FREEZE.json'; frozen = p.read(fp)
        frozen['bindings'] = [r for r in frozen['bindings'] if not r['path'].endswith('/inputs/BASELINE.json')]
        save(fp, frozen)
        args = SimpleNamespace(command='read', batch=str(p.B), root=self.root, executor='/root/fixture_reviewer')
        with self.assertRaisesRegex(ValueError, 'PREFIX_DECLARED_INPUT_NOT_FROZEN'):
            self.quiet(lambda: review.run(args))

    def test_prefix_save_rejects_context_predating_freeze(self):
        self.quiet(p.prepare_prefix); self.bind_actual_helper()
        args = SimpleNamespace(command='read', batch=str(p.B), root=self.root, executor='/root/fixture_reviewer')
        self.quiet(lambda: review.run(args))
        cp = p.B / 'prefix_audits/root1/READ_CONTEXT.json'; context = p.read(cp)
        context['started_at_utc'] = '2000-01-01T00:00:00+00:00'; save(cp, context)
        args.command = 'save'; args.judgment = str(p.B / 'UNREAD_SYNTHETIC_INPUT.json')
        with self.assertRaisesRegex(ValueError, 'PREFIX_READ_PRECEDES_FREEZE_OR_FUTURE'):
            self.quiet(lambda: review.run(args))


if __name__ == '__main__': unittest.main(verbosity=2)
