"""One offline preparation of the repaired formal route, with network blocked."""
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import difflib
import hashlib
import io
import json
import sys

BASE = Path(__file__).resolve().parent
WORK = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-realizer-state-contract-v1')
PREVIOUS_WORK = WORK.parent/'h4-planner-review-contract-v1'
PREVIOUS = BASE.parent/'h4_planner_review_contract_v1/prepared_run_final'
PACKAGE = Path('scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1')
sys.path.insert(0, str(WORK))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.full_pipeline_planner_recovery_more import main


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    auth = read(BASE/'AUTHORIZATION.json')
    run = Path(auth['run_root'])
    with patch('socket.socket.connect', side_effect=AssertionError('offline only')), \
            patch('socket.socket.connect_ex', side_effect=AssertionError('offline only')), \
            redirect_stdout(io.StringIO()):
        main(['prepare', '--planner-method', 'v2', '--review-contract', 'scoped-v1',
              '--realizer-contract', 'keyed-state-v1', '--run-root', str(run),
              '--input-bundle', auth['input_bundle']])
    manifest = read(run/'freeze_manifest.json')
    prior = read(PREVIOUS/'freeze_manifest.json')
    assert manifest['public_profile'] == prior['public_profile']
    schedule = {k: v for k, v in manifest['schedule'].items() if k != 'version'}
    assert schedule == {k: v for k, v in prior['schedule'].items() if k != 'version'}
    assert schedule['maximum_http_attempts'] == auth['maximum_http_attempts'] == 68
    identities = []
    for root in auth['root_order']:
        files = []
        for name in ('inputs.json', 'planner_prompt.json', 'planner_request.json', 'planner_capacity.json'):
            new, old = run/'frozen'/root/name, PREVIOUS/'frozen'/root/name
            assert new.read_bytes() == old.read_bytes(), (root, name)
            files.append({'path': str(new), 'sha256': sha(new), 'phase2_byte_identity': True})
        identities.append({'root_id': root, 'files': files})
    freezes = []
    for p in (run/'freeze_manifest.json', PREVIOUS/'freeze_manifest.json',
              BASE.parent/'h4_planner_method_v2/prepared_run/freeze_manifest.json',
              Path(auth['input_bundle'])/'manifest.json'):
        rows = read(p)['files']
        changed = [r['path'] for r in rows if not Path(r['path']).is_file() or sha(Path(r['path'])) != r['sha256']]
        assert not changed, changed
        freezes.append({'manifest': str(p), 'file_count': len(rows), 'changed': changed})
    changes, patches = [], []
    for new in sorted((WORK/PACKAGE).glob('*.py')):
        old = PREVIOUS_WORK/PACKAGE/new.name
        a = old.read_text() if old.is_file() else ''
        b = new.read_text()
        if a == b:
            continue
        changes.append({'file': str(new), 'baseline': str(old) if old.is_file() else None,
                        'old_sha256': sha(old) if old.is_file() else None, 'new_sha256': sha(new)})
        patches.extend(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                       fromfile=str(old) if old.is_file() else '/dev/null', tofile=str(new)))
    (BASE/'reports/CHANGES.json').write_text(json.dumps({'baseline': str(PREVIOUS_WORK),
        'worktree': str(WORK), 'files': changes}, indent=2)+'\n')
    (BASE/'reports/changes.patch').write_text(''.join(patches))
    tests = read(BASE/'reports/TEST_RESULTS.json')
    assert tests['failures'] == tests['errors'] == 0
    report = {'status': 'OFFLINE_PREPARATION_PASS', 'new_api_calls_during_preparation': 0,
        'live_execution_authorized': True, 'run_root': str(run), 'tests': tests,
        'initial_planner_identity': identities, 'frozen_records': freezes,
        'same_service_and_parameters': True, 'same_call_budget': True,
        'changed_python_files': len(changes), 'profile': manifest['execution_profile'],
        'capacity': manifest['planner_capacities'],
        'limitations': ['This preflight does not measure live generation quality.',
                       'Exact selected source IDs do not certify semantic completeness or consistency.',
                       'Service capacity, reasoning and verified monetary cost remain UNKNOWN.']}
    (BASE/'reports/PREPARATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'changed_python_files': len(changes),
                      'tests_passed': tests['passed'], 'frozen_records': len(manifest['files']),
                      'actual_api_calls': 0, 'authorized_http_maximum': 68}))


if __name__ == '__main__':
    prepare()
