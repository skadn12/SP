"""Run hermetic v76 tests; bind their exact code and preserve older deliveries."""
from pathlib import Path
import re
import subprocess

from common import HERE, PYTHON, bind, need, now, read, sha, write


def preservation(version):
    baseline_path = HERE / (version + '_PRESERVATION_BEFORE.json')
    before = read(baseline_path); root = Path(before['source'])
    actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
    saved = before['files']
    changed = sorted(k for k in set(actual) & set(saved) if actual[k] != saved[k])
    missing = sorted(set(saved) - set(actual)); added = sorted(set(actual) - set(saved))
    result = {'status': 'FAIL' if changed or missing or added else 'PASS', 'checked_at_utc': now(),
              'source': str(root), 'before_file_count': len(saved), 'after_file_count': len(actual),
              'changed': changed, 'missing': missing, 'added': added, 'baseline': bind(baseline_path)}
    write(HERE / 'evidence' / (version + '_PRESERVATION_RECEIPT.json'), result)
    need(result['status'] == 'PASS', version + '_preservation_failed')
    return result


def main():
    evidence = HERE / 'evidence'; evidence.mkdir(exist_ok=True)
    need(not (evidence / 'FINAL_OFFLINE_TEST_RECEIPT.json').exists(), 'offline_receipt_already_exists')
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.md')) + [HERE / 'CONFIG.json']
    bindings = [bind(p) for p in sorted(paths)]
    command = [str(PYTHON), '-B', '-m', 'unittest', 'discover', '-v', '-p', 'test_*.py']
    started = now()
    done = subprocess.run(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = evidence / 'FINAL_OFFLINE_TESTS.log'
    with log.open('x') as stream:
        stream.write(done.stdout)
    unchanged = all(sha(row['path']) == row['sha256'] for row in bindings)
    count = re.search(r'Ran (\d+) tests? in', done.stdout)
    need(count is not None, 'test_count_missing')
    older = {version: preservation(version) for version in ('V74', 'V75')}
    result = {'status': 'PASS' if done.returncode == 0 and unchanged else 'FAIL',
              'started_at_utc': started, 'completed_at_utc': now(), 'command': command,
              'test_count': int(count.group(1)), 'exit_code': done.returncode,
              'code_unchanged_during_test': unchanged, 'files': bindings + [bind(log)],
              'older_deliveries': {version: {'status': row['status'], 'file_count': row['before_file_count']}
                                   for version, row in older.items()},
              'actual_generation_http_calls': 0, 'actual_solver_calls': 0, 'training_calls': 0,
              'scope': 'synthetic offline tests; network and model execution replaced by test fakes',
              'real_candidate_content_certified': False, 'real_batch_result_audited': False}
    write(evidence / 'FINAL_OFFLINE_TEST_RECEIPT.json', result)
    need(result['status'] == 'PASS', 'offline_tests_failed')
    print({k:v for k,v in result.items() if k not in ('files', 'command')})


if __name__ == '__main__':
    main()
