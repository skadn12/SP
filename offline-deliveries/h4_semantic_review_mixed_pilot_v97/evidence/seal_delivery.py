"""Seal completed v97 after independent reviews; no generation or old-file edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = BASE / 'COMPLETE_MANIFEST.json'
    assert not destination.exists(), 'already_sealed'
    for name in ('COMPLETE_RESULTS_ZH.md', 'RESULTS.json', 'NEXT_MINFIX_PLAN.md',
                 'evidence/ACTUAL_FUNCTION_GATE_OBSERVATIONS.json',
                 'evidence/INDEPENDENT_FINAL_RESULT_AUDIT.json',
                 'evidence/FINAL_REPORT_SECOND_READER.json'):
        assert (BASE / name).is_file(), 'incomplete:' + name
    for name in ('evidence/INDEPENDENT_FINAL_RESULT_AUDIT.json',
                 'evidence/FINAL_REPORT_SECOND_READER.json'):
        result = read(BASE / name)
        assert str(result['status']).startswith('PASS'), 'review_not_passed:' + name
        assert not result.get('failures'), 'review_failures:' + name
    protected = read(BASE / 'PRESERVATION_BEFORE.json')['files']
    assert len(protected) == 1863
    for row in protected:
        assert sha(Path(row['path'])) == row['sha256'], 'old_file_changed:' + row['path']
    for name in ('evidence/EXPANSION_MINFIX_RECEIPT.json',
                 'evidence/ORDERING_MINFIX_RECEIPT.json'):
        for row in read(BASE / name)['files']:
            assert sha(Path(row['path'])) == row['sha256'], 'frozen_file_changed:' + row['path']
    results = read(BASE / 'RESULTS.json')
    assert results['totals']['independent_ready_roots'] == 0
    assert results['totals']['qa_http_calls'] == results['totals']['new_selected_candidates'] == 0
    assert read(BASE / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json')['status'] == 'SKIPPED'
    assert not (BASE / 'shards').exists()
    files = []
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        assert path != destination
        files.append({'path': str(path.resolve()), 'sha256': sha(path),
                      'bytes': path.stat().st_size})
    record = {'version': 'v97_complete_manifest_v1', 'status': 'SEALED',
              'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
              'listed_file_count_excluding_this_manifest': len(files),
              'protected_file_count_including_this_manifest': len(files) + 1,
              'self_excluded': True, 'excluded': ['__pycache__/**', '*.pyc'],
              'files': files, 'previous_delivery_protected_files': len(protected),
              'model_accepted_scenes': 1, 'model_accepted_new_turns': 14,
              'independent_ready_roots': 0, 'independent_ready_new_turns': 0,
              'qa_http_calls': 0, 'solver_calls': 0, 'training_jobs': 0,
              'semantic_generation_reliability_certified': False,
              'reason_for_qa_skip': 'No independently READY newly expanded source.'}
    with destination.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for row in files:
        assert sha(Path(row['path'])) == row['sha256'], 'postseal_file_changed:' + row['path']
    print(json.dumps({'status': 'SEALED_AND_HASH_VERIFIED',
                      'listed_files': len(files), 'protected_files_including_manifest': len(files) + 1,
                      'manifest_sha256': sha(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
