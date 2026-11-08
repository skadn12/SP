"""Offline Gate regression and frozen-response diagnostic; no credentials/API."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import difflib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BASE = Path('/root/autodl-tmp/gxy-selfplay')
DEST = Path(__file__).resolve().parent
OLD_WORKTREE = BASE / 'worktrees/h4-planner-output-contract-v1'
WORKTREE = BASE / 'worktrees/h4-gate-mixed-targets-v1'
SOURCE = BASE / 'offline-deliveries/h4_planner_output_contract_v1/prepared_run'
PACKAGE = 'scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1'
REL = Path(*PACKAGE.split('.'))
ROOTS = [f'cc-train-episode-{n}' for n in (259192, 96462, 112812, 218519)]
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import scoped_gate
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze
from scripts.current_method.restart_n5_spell.core import sha256_file


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate(raw, plan, context, **kwargs):
    return scoped_gate.validate_gate(raw, plan=plan, catalog=context['catalog'], minimum=1,
        expected_root_id=context['root_id'], expected_catalog_sha256=context['catalog']['sha256'], **kwargs)


def main():
    original = read(SOURCE / 'freeze_manifest.json')
    check_freeze(original)
    source_files = {p: sha256_file(p) for p in SOURCE.rglob('*') if p.is_file()}
    modules = read(SOURCE.parent / 'TEST_RESULTS.json')['modules'] + [
        'test_gate_mixed_targets', 'test_gate_mixed_targets_replay']
    stream = io.StringIO()
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('socket.socket.connect_ex', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         redirect_stdout(stream), redirect_stderr(stream):
        suite = unittest.defaultTestLoader.loadTestsFromNames([PACKAGE + '.' + n for n in modules])
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (DEST / 'test_output.txt').write_text(stream.getvalue())
    tests = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                 skipped=len(result.skipped), modules=modules, external_calls=0, network_forbidden=True)
    write('TEST_RESULTS.json', tests)
    if not result.wasSuccessful():
        raise AssertionError('offline_tests_failed_see_test_output')
    replay, probes = [], []
    for root in ROOTS:
        exp = SOURCE / 'expansion/roots' / root
        for stage, suffix, validation_name in [('gate', '', 'validation.json'),
                                              ('gate_supplement', '_supplement', 'supplement_validation.json')]:
            directory = exp / 'semantic_gate'
            context = read(directory / ('context' + suffix + '.json'))
            raw = read(directory / ('output' + suffix + '.json'))
            recorded = read(directory / ('validation' + suffix + '.json'))
            plan = read(exp / 'planner' / validation_name)['compiled_plan']
            legacy = validate(raw, plan, context, target_state_policy=None)
            new = validate(raw, plan, context)
            for key in ('passed_candidate_refs', 'rejected_candidate_refs', 'quarantined_candidate_refs', 'row_errors'):
                assert legacy[key] == recorded[key], (root, stage, key)
            replay.append({'root_id': root, 'stage': stage, 'source_output': str(directory / ('output' + suffix + '.json')),
                           'source_output_sha256': sha256_file(directory / ('output' + suffix + '.json')),
                           'legacy': legacy, 'new_policy_unchanged_raw': new,
                           'interpretation': 'Unchanged old labels are not repaired by code. This is not a fresh model output.'})
            for row in raw['candidate_reviews']:
                ref = row['candidate_ref']
                errors = legacy['row_errors'].get(ref, [])
                if 'existing_state_requires_exact_identity_and_source_support' not in errors:
                    continue
                proposed = deepcopy(raw)
                target = next(r for r in proposed['candidate_reviews'] if r['candidate_ref'] == ref)
                target['relation'] = 'UNKNOWN'
                old_probe = validate(proposed, plan, context, target_state_policy=None)
                new_probe = validate(proposed, plan, context)
                assert ref in old_probe['quarantined_candidate_refs']
                assert ref in new_probe['deferred_candidate_refs']
                normalized = next(r for r in new_probe['normalized_candidate_reviews'] if r['candidate_ref'] == ref)
                assert all({k: normalized['target_checks'][f][k] for k in ('state', 'support', 'evidence_refs')}
                           == target['target_checks'][f] for f in target['target_checks'])
                probes.append({'root_id': root, 'stage': stage, 'candidate_ref': ref,
                               'kind': 'OFFLINE_COUNTERFACTUAL_RELATION_ONLY_NOT_MODEL_RESULT',
                               'original_relation': row['relation'], 'counterfactual_relation': 'UNKNOWN',
                               'original_target_checks': row['target_checks'],
                               'legacy_errors': old_probe['row_errors'][ref],
                               'new_normalized_review': normalized,
                               'source_raw_edited': False, 'semantic_support_certified': False})
    assert len(replay) == 8 and len(probes) == 4
    write('reports/OLD_RESPONSE_REPLAY.json', replay)
    write('reports/MIXED_STATE_EXPRESSIBILITY_PROBES.json', probes)
    production = ['gate_source_policy.py', 'scoped_gate.py']
    added = ['gate_mixed_targets_replay.py', 'test_gate_mixed_targets.py', 'test_gate_mixed_targets_replay.py']
    changes, patches = [], []
    for name in production + added:
        before, after = OLD_WORKTREE / REL / name, WORKTREE / REL / name
        a = before.read_text().splitlines(keepends=True) if before.is_file() else []
        b = after.read_text().splitlines(keepends=True)
        patches.extend(difflib.unified_diff(a, b, fromfile='a/' + str(REL / name), tofile='b/' + str(REL / name)))
        changes.append({'path': str(after), 'sha256': sha256_file(after),
                        'previous_sha256': sha256_file(before) if before.is_file() else None})
    (DEST / 'changes.patch').write_text(''.join(patches))
    write('CHANGES.json', changes)
    assert all(sha256_file(p) == digest for p, digest in source_files.items())
    check_freeze(original)
    write('VERIFICATION.json', {'status': 'PASS_OFFLINE_READY_FOR_GATE_ONLY', 'external_calls': 0,
          'tests': tests, 'legacy_gate_replays_identical': 8, 'counterfactual_mixed_rows_expressible': 4,
          'counterfactual_is_not_fresh_quality_evidence': True,
          'source_run_files_unchanged': len(source_files), 'source_freeze_entries_unchanged': len(original['files']),
          'old_worktree_preserved': str(OLD_WORKTREE), 'new_worktree': str(WORKTREE),
          'planner_outline_realizer_qa_calls_in_this_verification': 0})
    print(json.dumps({'tests': tests['tests_run'], 'status': 'PASS', 'source_frozen_unchanged': len(original['files'])}))


if __name__ == '__main__':
    main()
