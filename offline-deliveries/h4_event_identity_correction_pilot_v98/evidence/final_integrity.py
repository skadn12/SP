"""Verify preserved archives and pre-call code/input receipts after execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p): return {'path': str(p.resolve()), 'sha256': sha(p)}
def main():
    checks = []
    for name in ('PRESERVATION_BEFORE.json', 'evidence/EXPANSION_MINFIX_RECEIPT.json',
                 'evidence/ORDERING_MINFIX_RECEIPT.json'):
        rows = read(HOME / name)['files']
        failures = [r['path'] for r in rows if sha(Path(r['path'])) != r['sha256']]
        checks.append({'receipt': bind(HOME / name), 'file_count': len(rows), 'failures': failures})
    plan = read(HOME / 'PRECALL_PLAN.json')
    failures = [r['path'] for r in plan['frozen_inputs'] if sha(Path(r['path'])) != r['sha256']]
    checks.append({'receipt': bind(HOME / 'PRECALL_PLAN.json'),
                   'file_count': len(plan['frozen_inputs']), 'failures': failures})
    result = {'version': 'v98_final_preservation_and_precall_integrity_v1',
              'status': 'FAIL' if any(c['failures'] for c in checks) else 'PASS',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'semantic_certification': False}
    with (HOME / 'evidence/FINAL_CODE_INPUT_INTEGRITY.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    assert result['status'] == 'PASS', result
    print(json.dumps({'status': result['status'], 'checked_files': sum(c['file_count'] for c in checks)}))
if __name__ == '__main__': main()
