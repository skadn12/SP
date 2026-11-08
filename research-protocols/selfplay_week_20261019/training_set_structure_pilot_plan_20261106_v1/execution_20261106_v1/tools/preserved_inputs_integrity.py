#!/usr/bin/env python3
"""Bind preservation checks only; never modify source or historical files."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json
import re
import pilot_artifacts as pa

E = Path(__file__).resolve().parents[1]


def main():
    coverage_path = E / 'manifests/HISTORICAL_COVERAGE.json'
    coverage = pa.read(coverage_path)
    expected = defaultdict(set)
    kinds = defaultdict(set)

    def gather(value, kind):
        if isinstance(value, dict):
            if (isinstance(value.get('path'), str)
                    and isinstance(value.get('sha256'), str)
                    and re.fullmatch('[0-9a-f]{64}', value['sha256'])):
                expected[value['path']].add(value['sha256'])
                kinds[value['path']].add(kind)
            for child in value.values():
                gather(child, kind)
        elif isinstance(value, list):
            for child in value:
                gather(child, kind)

    gather(coverage, 'historical_inventory')
    construction = pa.read(E / 'manifests/CONSTRUCTION_COMPLETION_FREEZE.json')
    gather(construction['drafts'], 'current_construction_freeze')
    execution = pa.read(E / 'manifests/EXECUTION_FREEZE.json')
    for source in execution['source_manifest']:
        for prefix in ('source', 'neutral'):
            gather({'path': source[prefix + '_path'],
                    'sha256': source[prefix + '_sha256']}, 'current_' + prefix)
    records = []
    for path, hashes in sorted(expected.items()):
        assert len(hashes) == 1, ('CONFLICTING_EXPECTED_BYTES', path)
        actual = pa.digest(path)
        records.append({'path': path, 'expected_sha256': next(iter(hashes)),
                        'actual_sha256': actual, 'roles': sorted(kinds[path]),
                        'unchanged': actual in hashes})
    assert all(x['unchanged'] for x in records), 'PRESERVED_INPUT_CHANGED'
    result = {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'stage': 'PRE_SOLVER',
              'coverage_sha256': pa.digest(coverage_path),
              'counts_by_role': dict(Counter(k for x in records for k in x['roles'])),
              'all_bound_files_unchanged': True, 'files': records,
              'limits': 'Byte integrity against the stated freezes only; not semantic validation or a claim that OPP01 was never rewritten before construction freeze.'}
    out = E / 'review/PRESERVED_INPUT_INTEGRITY.json'
    pa.write_new(out, result)
    print(json.dumps({'output': str(out), 'status': result['status'],
                      'counts_by_role': result['counts_by_role']}))


if __name__ == '__main__':
    main()
