#!/usr/bin/env python3
"""Audit the additive observation v2 repair without importing its implementation."""
from pathlib import Path
import collections
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math

B = Path(__file__).resolve().parent.parent


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'bytes': p.stat().st_size}


def main():
    olddir = B / 'analysis/results_01'
    newdir = B / 'analysis/results_02'
    old = json.loads((olddir / 'OBSERVATIONS.json').read_text())
    new = json.loads((newdir / 'OBSERVATIONS.json').read_text())
    changes = []
    float_changes = []
    set_order_changes = []
    oldcopy = copy.deepcopy(old)
    newcopy = copy.deepcopy(new)
    for rname in old['runs']:
        ar = oldcopy['runs'][rname]['requests']
        br = newcopy['runs'][rname]['requests']
        assert len(ar) == len(br)
        for i, (a, b) in enumerate(zip(ar, br)):
            assert a['request_id'] == b['request_id']
            if a['raw_binding'] != b['raw_binding']:
                expected = B / ('solver/direct' if rname == 'direct' else 'solver/evidence_v2') / 'responses' / (b['request_id'] + '.json')
                assert b['raw_binding'] == bind(expected)
                assert '/sources/' in a['raw_binding']['path']
                changes.append({'run': rname, 'request_id': a['request_id'],
                                'previous_binding': a['raw_binding'], 'correct_binding': b['raw_binding']})
            a.pop('raw_binding')
            b.pop('raw_binding')
    assert len(changes) == 40
    assert collections.Counter(x['run'] for x in changes) == {'direct': 20, 'evidence': 20}
    for obj in [oldcopy, newcopy]:
        obj.pop('created_at_utc')
        obj.pop('tool_binding')

    def compare(a, b, path=''):
        assert type(a) is type(b), path
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                compare(a[k], b[k], path + '/' + k)
        elif isinstance(a, list):
            assert len(a) == len(b), path
            if path.endswith(('/false_positive_labels', '/false_negative_labels')):
                assert sorted(a) == sorted(b), path
                if a != b:
                    set_order_changes.append({'path': path, 'old': a, 'new': b})
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, path + '/' + str(i))
        elif isinstance(a, float):
            assert math.isclose(a, b, rel_tol=0, abs_tol=1e-15), (path, a, b)
            if a != b:
                float_changes.append({'path': path, 'old': a, 'new': b, 'absolute_difference': abs(a - b)})
        else:
            assert a == b, (path, a, b)

    compare(oldcopy, newcopy)
    same_bytes = {}
    for f in ['CITATION_OBSERVATIONS.csv', 'REPORT_ZH.md']:
        same_bytes[f] = (olddir / f).read_bytes() == (newdir / f).read_bytes()
        assert same_bytes[f], f
    a = list(csv.DictReader(io.StringIO((olddir / 'QUESTION_OBSERVATIONS.csv').read_text())))
    b = list(csv.DictReader(io.StringIO((newdir / 'QUESTION_OBSERVATIONS.csv').read_text())))
    assert len(a) == len(b)
    csv_float_changes = []
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for k in x:
            if x[k] != y[k]:
                assert math.isclose(float(x[k]), float(y[k]), rel_tol=0, abs_tol=1e-15), (i, k, x[k], y[k])
                csv_float_changes.append({'row': i, 'field': k, 'old': x[k], 'new': y[k]})
    source = (B / 'analysis/observe.py').read_text()
    replacement = "'raw_binding':bind(run/'responses'/f'{rid}.json')"
    assert (B / 'analysis/observe_v2.py').read_text() == source.replace("'raw_binding':bindings[-1]", replacement, 1)
    audit = json.loads((B / 'audit/RESULTS_AUDIT_OBSERVATIONS_02.json').read_text())
    assert audit['status'] == 'PASS'
    out = {
        'status': 'PASS_WITH_EXPLICIT_NON_SEMANTIC_SERIALIZATION_DIFFERENCES',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'bindings': [bind(p) for p in [B / 'analysis/observe.py', B / 'analysis/observe_v2.py',
                                      olddir / 'OBSERVATIONS.json', newdir / 'OBSERVATIONS.json',
                                      B / 'audit/RESULTS_AUDIT_OBSERVATIONS_01.json',
                                      B / 'audit/RESULTS_AUDIT_OBSERVATIONS_02.json', Path(__file__)]],
        'repaired_raw_bindings': changes,
        'repaired_raw_binding_count': len(changes),
        'new_observation_independent_check_count': audit['check_count'],
        'byte_identical_files': same_bytes,
        'float_serialization_differences': float_changes,
        'label_set_serialization_order_differences': set_order_changes,
        'question_csv_float_serialization_differences': csv_float_changes,
        'interpretation': [
            'The only implementation change explicitly binds every request to its response JSON.',
            'The initial 40 provenance failures remain preserved in results_01 and its independent audit.',
            'Predictions, validity, exact scores, citation contents and label sets did not change.',
            'New-process set iteration alters FP/FN label list ordering and floating accumulation tails; these are reported rather than claimed byte-identical.',
            'A mechanically valid source quote is not a new semantic certification or a diagnosed failure mechanism.'
        ],
        'new_solver_calls': 0
    }
    with (B / 'audit/RESULTS_AUDIT_OBSERVATIONS_BINDING_REPAIR.json').open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ['status', 'repaired_raw_binding_count', 'new_observation_independent_check_count', 'byte_identical_files']}))
    print(json.dumps({'json_float_tail_differences': len(float_changes),
                      'label_set_order_differences': len(set_order_changes),
                      'question_csv_float_tail_differences': len(csv_float_changes)}))


if __name__ == '__main__':
    main()
