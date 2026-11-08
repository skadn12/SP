"""Exercise reporting/export helpers on mixed historical fixtures in a temp home."""
from pathlib import Path
import tempfile, json, shutil, subprocess, hashlib, datetime
HOME = Path(__file__).resolve().parents[1]
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
OLD81 = HOME.parent / 'h4_bound_scene_ordering_mixed_batch_v81'
OLD82 = HOME.parent / 'h4_actor_reference_minfix_ordering_v82'
def read(path): return json.loads(path.read_text())
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
def bind(path): return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
checks = []
with tempfile.TemporaryDirectory(prefix='v83_evidence_helpers_') as temporary:
    task = Path(temporary) / 'h4_name_policy_mixed_batch_v83'
    (task / 'evidence').mkdir(parents=True)
    for source in [OLD81, OLD82]:
        (task.parent / source.name).symlink_to(source)
    shutil.copyfile(HOME / 'common.py', task / 'common.py')
    for name in ['write_result_details.py', 'write_runtime_receipt.py', 'export_audited.py', 'complete_batch.py']:
        shutil.copyfile(HOME / 'evidence' / name, task / 'evidence' / name)
    pools = [read(old / 'EVALUATION_POOL.json')['questions'] for old in [OLD81, OLD82]]
    pairs = [read(old / 'PAIRED_CALLS.json')['calls'] for old in [OLD81, OLD82]]
    questions = pools[0] + pools[1]
    write(task / 'EVALUATION_POOL.json', {'questions': questions})
    write(task / 'PAIRED_CALLS.json', {'calls': pairs[0] + pairs[1]})
    selected = read(OLD81 / 'SELECTED_CANDIDATES.json')['questions'] + read(OLD82 / 'SELECTED_CANDIDATES.json')['questions']
    write(task / 'SELECTED_CANDIDATES.json', {'questions': selected})
    write(task / 'BATCH_REVIEW.COMPLETE.json', {'status': 'offline fixture'})
    mixed_report = read(OLD81 / 'MIXED_EVALUATION_REPORT.json')
    mixed_report.update(selected_candidate_ids=[q['candidate_id'] for q in selected], calls_per_model=180)
    mixed_report['by_type']['ordering'] = read(OLD82 / 'MIXED_EVALUATION_REPORT.json')['by_type']['ordering']
    write(task / 'MIXED_EVALUATION_REPORT.json', mixed_report)
    for side in ['base', 'trained']:
        native = [read(old / 'solver' / side / 'RESULTS.json')['questions'] for old in [OLD81, OLD82]]
        write(task / 'solver' / side / 'RESULTS.json', {'questions': native[0] + native[1]})
    for relative in ['multiselect/ALLOCATION.json', 'ordering/root194259_ordering/REJECTIONS.json']:
        write(task / relative, read(OLD81 / relative))
    for relative in ['ordering/CONTENT_REVIEW.json', 'ordering/SOURCE_JUDGMENTS.json']:
        write(task / relative, read(OLD82 / relative))
    for path in (OLD81 / 'multiselect/assets').glob('*/rounds/*/ASSET_REVIEW.json'):
        write(task / path.relative_to(OLD81), read(path))
    outcome = subprocess.run([PYTHON, '-B', 'evidence/write_result_details.py'], cwd=task, capture_output=True, text=True)
    assert outcome.returncode == 0, outcome.stderr
    details = read(task / 'evidence/RESULT_DETAILS.json')
    for side in ['base', 'trained']:
        metrics = details['metrics'][side]
        assert metrics['all_questions']['calls'] == 180
        assert metrics['new_multiselect']['calls'] == 108
        assert metrics['new_ordering']['calls'] == 36
        assert metrics['historical_control']['calls'] == 36
        assert metrics['new_child'] == metrics['new_candidates']
        assert metrics['new_candidates']['calls'] == 144
    assert details['ordering_content_failures'] == []
    assert details['aggregate']['all']['base']['all_six_selected_calls'] == sum(r['base']['selected_count'] == 6 for r in pairs[0])
    assert details['content_usable_new_by_type'] == {'multi_select': 6, 'ordering': 3}
    checks.append('Mixed108newmulti/36ordering/36controls: allroles and all-six counts correctly separate ordering')
    for name in ['CHARACTER_REGISTRY.json', 'ACTOR_REFERENCE_POLICY.json', 'SCENE_PLANS.json']:
        shutil.copyfile(HOME / name, task / name)
    for auditname in ['GENERATION_AUDIT', 'INPUT_AUDIT', 'RESULT_AUDIT', 'UNION_AUDIT']:
        write(task / 'evidence' / (auditname + '.json'), read(OLD81 / 'evidence' / (auditname + '.json')))
    outcome = subprocess.run([PYTHON, '-B', 'evidence/export_audited.py'], cwd=task, capture_output=True, text=True)
    assert outcome.returncode == 0, outcome.stderr
    exported = read(task / 'AUDITED_NEW_CANDIDATES.json')
    union = read(task / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
    assert exported['new_question_count'] == 7
    assert exported['type_distribution'] == {'multi_select': 4, 'ordering': 3}
    assert sum(exported['gold_distribution'].values()) == 4
    assert union['candidate_count'] == 20 and len(union['duplicates_removed']) == 7
    assert union['type_distribution'] == {'multi_select': 17, 'ordering': 3}
    checks.append('Export7noncontrols, ordering excluded from multiGoldcounts, v82union duplicates retained as receipts')
    protected = read(HOME / 'PRESERVATION_BEFORE.json')
    write(task / 'PRESERVATION_BEFORE.json', protected)
    config = read(HOME / 'CONFIG.json')
    for key, name in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'), ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
        config[key] = bind(task / name)
    write(task / 'CONFIG.json', config)
    write(task / 'BATCH_PREPARATION.COMPLETE.json', {'files': [bind(task / 'common.py')]})
    outcome = subprocess.run([PYTHON, '-B', 'evidence/complete_batch.py'], cwd=task, capture_output=True, text=True)
    assert outcome.returncode == 0, outcome.stderr
    complete = read(task / 'evidence/RUN_COMPLETE.json')
    assert complete['new_selected_count'] == 7 and complete['ordering_new_selected_count'] == 3
    assert complete['preserved_previous_file_count'] == 4939
    reporttext = (task / 'evidence/RESULTS.md').read_text()
    assert '本轮新多选' in reporttext and '本轮新排序' in reporttext
    checks.append('Completion gates audits/protection/freeze/profile, reports actualmixedtypes dynamically')
    for action, log in [('freeze', 'FREEZE'), ('prepare-evaluation', 'PREPARE_EVALUATION'), ('live-trained', 'LIVE_TRAINED'), ('live-base', 'LIVE_BASE'), ('report', 'REPORT')]:
        (task / 'evidence' / (log + '.log')).write_text('synthetic offline fixture; no actual command execution\n')
    for name in ['EVALUATION_PREPARATION.COMPLETE.json', 'TRAINED_EVALUATION.COMPLETE.json']:
        write(task / name, {'status': 'fixture'})
    write(task / 'BASE_EVALUATION.STARTED.json', {'time': '2026-10-14T00:00:00+00:00'})
    write(task / 'BASE_EVALUATION.COMPLETE.json', {'time': '2026-10-14T00:01:00+00:00'})
    write(task / 'solver/base/STARTED.json', {'started_at_utc': '2026-10-14T00:00:10+00:00'})
    for observed, expectedstage in [('2026-10-14T00:00:05+00:00', 'startup'), ('2026-10-14T00:00:20+00:00', 'live after')]:
        write(task / 'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json', {'observed_at_utc': observed,
              'shell_pythonpath_present': False, 'live_processes': [
                  {'command': [PYTHON, '-B', 'batch.py', 'live-base'], 'pythonpath_present': False},
                  {'command': [PYTHON, '-B', '/fixture/run_solver.py', 'live'], 'pythonpath_present': False}],
              'limitation': 'Synthetic fixture only; not actual runtime evidence.'})
        receiptpath = task / 'evidence/EVALUATION_RUNTIME_RECEIPT.json'
        if receiptpath.exists(): receiptpath.unlink()
        outcome = subprocess.run([PYTHON, '-B', 'evidence/write_runtime_receipt.py'], cwd=task, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
        assert expectedstage in read(receiptpath)['environment_observation_stage']['stage']
    checks.append('Runtime receipt bothstartup andafterSTARTED classification fromtrueparentinterval; fixtureonly')
record = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS',
          'scope': 'Offline mixed historical fixtures in a temporary directory; no production writes, model calls or audit execution.',
          'checks': checks, 'helper_files': [bind(HOME / 'evidence' / name) for name in ['write_result_details.py', 'write_runtime_receipt.py', 'observe_live_environment.py', 'export_audited.py', 'complete_batch.py', 'ARTIFACT_INDEX.md']]}
with (HOME / 'evidence/HELPERS_OFFLINE_VALIDATION.json').open('x') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2); stream.write('\n')
print(record['status'], len(checks), checks)
