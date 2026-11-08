#!/usr/bin/env python3
"""Seal a terminated local attempt. No model loading, generation, or scoring."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'solver'
V = S / 'evaluation'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bind(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-exit-code', required=True, type=int)
    args = parser.parse_args()
    output = V / 'EXECUTION_RESULT_FREEZE.json'
    assert not output.exists(), 'EXCLUSIVE_RESULT_FREEZE_ALREADY_EXISTS'
    assert (S / 'RUN_ATTEMPT.json').exists(), 'ACTUAL_ATTEMPT_REQUIRED'
    requests = read(V / 'REQUESTS.json')
    assert len(requests) == 1008
    expected = [r['request_id'] for r in requests]
    assert len(set(expected)) == len(expected)
    attempt = read(S / 'RUN_ATTEMPT.json')
    assert attempt['plan_freeze'] == bind(V / 'PLAN_FREEZE.json')
    raw_files = sorted((V / 'raw').glob('*.json'))
    raw_ids = []
    statuses = Counter()
    for f in raw_files:
        row = read(f)
        assert f.name == row['request_id'] + '.json'
        assert row['request_id'] in set(expected)
        raw_ids.append(row['request_id'])
        statuses[row['status']] += 1
    assert len(set(raw_ids)) == len(raw_ids)
    collected_path = V / 'RAW_ANSWERS.json'
    scores_path = V / 'SCORES.json'
    assert collected_path.exists(), 'RAW_COLLECTION_MISSING_DO_NOT_RESUME_INFERENCE'
    collected = read(collected_path)
    assert [r['request_id'] for r in collected] == expected
    collected_status = Counter(r['status'] for r in collected)
    actual = {read(f)['request_id']: read(f) for f in raw_files}
    for r in collected:
        if r['request_id'] in actual:
            assert r == actual[r['request_id']], 'COLLECTED_RAW_CONTENT_MISMATCH'
        else:
            assert r['status'] == 'MISSING', 'UNPRESERVED_RAW_NOT_MISSING'
    score_rows = None
    if scores_path.exists():
        scores = read(scores_path)
        score_rows = scores['requests']
        assert [r['request_id'] for r in score_rows] == expected
        for req, raw, score in zip(requests, collected, score_rows):
            for field in ('request_id', 'candidate_id', 'root_id', 'group', 'question_type',
                          'context_mode', 'variant_id', 'repeat_index', 'sampling_seed',
                          'protocol_markers', 'clean_contrast_eligible', 'structure_result', 'overlap_result'):
                assert req[field] == score[field], ('SCORE_METADATA_MISMATCH', field)
            assert raw['status'] == score['execution_status']
            assert raw.get('raw_prediction') == score.get('raw_prediction')
    completed = V / 'COMPLETED.json'
    interrupted = V / 'INTERRUPTED.json'
    if args.process_exit_code == 0 and not interrupted.exists():
        assert completed.exists() and scores_path.exists(), 'SUCCESS_WITHOUT_COMPLETE_RESULTS'
        assert len(raw_files) == len(expected)
        assert collected_status == {'RETURNED': len(expected)}
        assert read(completed)['returned_requests'] == len(expected)
    names = ['PLAN_FREEZE.json', 'REQUESTS.json', 'PREFLIGHT.json', 'INPUT_BINDINGS.json',
             'RAW_ANSWERS.json', 'SCORES.json', 'STARTED.json', 'COMPLETED.json', 'INTERRUPTED.json']
    files = [V / name for name in names if (V / name).exists()]
    files += [S / 'RUN_ATTEMPT.json', S / 'ATTEMPT_LEDGER.json', S / 'RUN_STDOUT.log', Path(__file__)]
    for name in ('PROTOCOL_AMENDMENT.json', 'PROTOCOL_AMENDMENT.md'):
        if (ROOT / name).exists(): files.append(ROOT / name)
    returned = collected_status.get('RETURNED', 0)
    result = {
        'schema_version': 'amended_solver_execution_result_freeze_v1',
        'frozen': True, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'process_exit_code': args.process_exit_code,
        'namespace': read(V / 'PLAN_FREEZE.json')['namespace'],
        'status': 'COMPLETED' if args.process_exit_code == 0 and completed.exists() and not interrupted.exists() else 'INTERRUPTED_OR_FAILED',
        'counts': {'planned': len(expected), 'returned': returned,
                   'missing': len(expected) - returned, 'preserved_raw_files': len(raw_files)},
        'raw_status_counts': dict(collected_status), 'score_file_present': scores_path.exists(),
        'automatic_retry': False, 'training': False, 'training_export': False,
        'all_request_ids_and_score_metadata_match': score_rows is not None,
        'files': [bind(f) for f in files], 'raw_files': [bind(f) for f in raw_files],
        'limits': 'Byte/metadata result seal, not semantic correctness verification. Infrastructure missing/backend records are not model errors. Prior failed execution remains immutable.'
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'result_freeze': bind(output), 'status': result['status'],
                      'counts': result['counts'], 'score_file_present': result['score_file_present']}))


if __name__ == '__main__':
    main()
