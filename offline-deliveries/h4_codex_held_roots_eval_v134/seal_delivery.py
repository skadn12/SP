"""Seal completed QA, all three live runs, and independently audited results."""
from pathlib import Path
import json
import hashlib
import datetime
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')

def verify(rows):
    for r in rows:
        p = Path(r['path'])
        if not p.is_absolute():
            p = B / p
        assert p.is_file() and sha(p) == r['sha256'], str(p)

def main():
    qa = read(B / 'QA_FREEZE.json')
    assert qa['status'] == 'FROZEN' and qa['question_count'] == 49
    for name in ['MASTER_FREEZE.json', 'DEV_QA_CLOSED.json', 'QA_FREEZE.json',
                 'evaluation/TOOLS_FREEZE.json', 'evaluation/PREFLIGHT_FREEZE.json']:
        verify(read(B / name)['files'])
    for name in ['evidence/INDEPENDENT_EVALUATION_PREFLIGHT_AUDIT.json',
                 'evidence/INDEPENDENT_RESULTS_AUDIT.json']:
        audit = read(B / name)
        assert audit['status'] == 'PASS', name
        verify(audit['audited_files'])
    assert read(B / 'evidence/SEALED_PREDECESSORS_AFTER_V134.json')['status'] == 'PASS'
    assert read(B / 'evaluation/RUN_COMPLETE.json')['status'] == 'COMPLETE'
    summary = read(B / 'evaluation/reports/SUMMARY.json')
    assert summary['status'] == 'COMPLETE' and not summary['errors']
    for side in ['base', 'clean', 'new']:
        run = B / 'evaluation/run' / side
        assert read(run / 'COMPLETE.json')['actual_call_count'] == 798
        assert len(list((run / 'responses').glob('*.json'))) == 798
        assert not (run / 'FAILED.json').exists()
    assert (B / 'README_ZH.md').is_file()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(B / 'FINAL_STATUS.json', {
        'status': 'COMPLETE_INDEPENDENTLY_AUDITED_NEW_ONLY_EVALUATION',
        'created_at_utc': now, 'fixed_retry_roots': 7, 'content_pass_questions': 49,
        'new_dev_questions': 35, 'new_test_questions': 14,
        'extended_dev_questions': 83, 'extended_test_questions': 101,
        'extended_library_evaluated_here': False, 'calls_per_model': 798,
        'total_live_calls': 2394, 'model_sides': ['base', 'clean', 'new'],
        'score_based_selection': False, 'baseline_conflicts_resolved': False,
        'training_run': False, 'existing_pool_or_model_modified': False,
        'same_model_shared_filesystem_review': True,
        'test_has_historical_exposure': True,
        'results_report': 'evaluation/reports/REPORT_ZH.md',
        'independent_results_audit': 'evidence/INDEPENDENT_RESULTS_AUDIT.json'})
    paths = sorted(p for p in B.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc'
                   and p.name != 'COMPLETE_MANIFEST.json')
    rows = [{'path': str(p.relative_to(B)), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in paths]
    save(B / 'COMPLETE_MANIFEST.json', {
        'schema_version': 'v134_complete_manifest_v1',
        'status': 'SEALED_QA_AND_THREE_MODEL_EVALUATION_COMPLETE',
        'created_at_utc': now,
        'scope': 'All delivery files except this self-excluded manifest and Python caches.',
        'file_count': len(rows), 'payload_bytes': sum(r['bytes'] for r in rows),
        'questions_evaluated': 49, 'live_calls': 2394, 'files': rows})
    print(json.dumps({'status': 'SEALED', 'files': len(rows),
                      'payload_bytes': sum(r['bytes'] for r in rows),
                      'manifest_sha256': sha(B / 'COMPLETE_MANIFEST.json')}))

if __name__ == '__main__':
    main()
