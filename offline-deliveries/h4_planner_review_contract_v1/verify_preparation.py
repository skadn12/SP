"""Offline verification of this delivery and its explicitly referenced freezes."""
from pathlib import Path
import hashlib
import json

BASE = Path(__file__).resolve().parent
CURRENT = BASE / 'prepared_run_final'
PREVIOUS = BASE.parent / 'h4_planner_method_v2/prepared_run'
ROOTS = ['cc-train-episode-259192', 'cc-train-episode-96462',
         'cc-train-episode-112812', 'cc-train-episode-218519']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    current, previous = [read(p / 'freeze_manifest.json') for p in (CURRENT, PREVIOUS)]
    roots = []
    for root in ROOTS:
        identities = []
        for name in ('inputs.json', 'planner_prompt.json', 'planner_request.json',
                     'planner_capacity.json'):
            new, old = [p / 'frozen' / root / name for p in (CURRENT, PREVIOUS)]
            assert new.read_bytes() == old.read_bytes(), (root, name)
            identities.append({'file': str(new), 'sha256': sha(new),
                               'phase1_byte_identity': True})
        inputs = read(CURRENT / 'frozen' / root / 'inputs.json')
        roots.append({'root_id': root, 'files': identities,
                      'full_history_turns': len(inputs['prior_expanded']['turns']),
                      'conversation_date_segments': len(inputs['conversation_date_segments'])})
    assert current['public_profile'] == previous['public_profile']
    assert current['input_bundle_manifest_path'] == previous['input_bundle_manifest_path']
    profile_diffs = {k: {'old': previous['execution_profile'].get(k), 'new': v}
                     for k, v in current['execution_profile'].items()
                     if previous['execution_profile'].get(k) != v}
    assert set(profile_diffs) == {'name', 'review_contract_policy'}, profile_diffs
    schedule_diffs = {k: {'old': previous['schedule'].get(k), 'new': v}
                      for k, v in current['schedule'].items()
                      if previous['schedule'].get(k) != v}
    assert set(schedule_diffs) == {'version'}, schedule_diffs
    assert current['execution_profile']['minimum_qualified_scenes'] == 3
    assert current['execution_profile']['planner_supplement_limit'] == 1
    assert current['schedule']['maximum_http_attempts'] == 68
    assert current['schedule']['automatic_retries'] == 0
    freezes = []
    for path in (CURRENT / 'freeze_manifest.json', PREVIOUS / 'freeze_manifest.json',
                 BASE.parent / 'h4_planner_codex_feedback_v1/prepared_run/freeze_manifest.json',
                 BASE.parent / 'h4_planner_codex_feedback_v1/inputs/manifest.json'):
        records = read(path)['files']
        changed = [r['path'] for r in records if not Path(r['path']).is_file()
                   or sha(Path(r['path'])) != r['sha256']]
        assert not changed, changed
        freezes.append({'manifest': str(path), 'file_count': len(records), 'changed': changed})
    prohibited = [str(p) for p in CURRENT.rglob('*') if p.is_file() and
                  (p.name == 'EXECUTION_STARTED.json' or p.name.startswith('response')
                   or 'ledger' in p.name or 'generation_summary' in p.name)]
    assert not prohibited, prohibited
    status = read(CURRENT / 'run_status.json')
    assert status['external_calls'] == 0
    assert status['status'] == 'WAITING_FOR_CALL_BUDGET_CONFIRMATION'
    tests = read(BASE / 'reports/TEST_RESULTS.json')
    assert tests['failures'] == tests['errors'] == 0
    replay = read(BASE / 'reports/INTERFACE_REPLAY.json')
    assert replay['status'] == 'INTERFACE_REPLAY_PASS'
    report = {'status': 'OFFLINE_VERIFICATION_PASS', 'experimental_api_calls': 0,
              'live_quality': 'NOT_EVALUATED', 'prepared_run': str(CURRENT),
              'initial_planner_identity': roots,
              'provider_profile_unchanged': True, 'profile_differences': profile_diffs,
              'schedule_differences': schedule_diffs, 'freeze_checks': freezes,
              'no_generation_artifacts': True, 'run_status': status,
              'tests': tests, 'interface_replay_report': str(BASE / 'reports/INTERFACE_REPLAY.json'),
              'superseded_preparation': str(BASE / 'prepared_run'),
              'superseded_reason': 'Exact duplicate check was corrected after the first preparation; its original freeze was not rewritten.',
              'limits': ['Reference identity does not certify source entailment.',
                         'Model prerequisite scope and novelty judgments remain unverified.',
                         'No live scene count, QA accuracy or cost improvement measured.']}
    (BASE / 'reports/VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'tests_passed': tests['passed'],
                      'frozen_file_counts': [f['file_count'] for f in freezes],
                      'experimental_api_calls': 0}))


if __name__ == '__main__':
    verify()
