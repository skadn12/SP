"""Preserve v1; bind exact S0 submitted choice arrays in a new CPU addendum."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prepare as b

HERE = b.HERE


def main():
    original = b.read(HERE / 'PREPARATION_FREEZE.json')
    for f in original['files'] + original['upstream_bindings']:
        assert b.sha(f['path']) == f['sha256'], f['path']
    ids = {r['request_id'] for r in b.read(HERE / 'public/DIRECT_REQUESTS.json')['requests']}
    byid, choices, bindings = {}, {}, []
    for p in sorted((b.OLD / 'batches').glob('*_SUBMITTED.json')):
        rows = b.read(p)
        selected = [(rid, x) for rid, x in zip(rows['request_ids'], rows['sampling_params']) if rid in ids]
        if not selected:
            continue
        bindings.append(b.binding(p))
        for rid, x in selected:
            if isinstance(x, str):
                matched = re.search(r'choice=(\[[^\]]*\])', x)
                assert matched, 'S0_CHOICE_ARRAY_NOT_FOUND:' + rid
                v = ast.literal_eval(matched.group(1))
            else:
                v = x['structured_outputs']['choice']
            assert isinstance(v, list) and v and all(isinstance(i, str) for i in v) and len(v) == len(set(v))
            key = hashlib.sha256(json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            if key in choices:
                assert choices[key] == v
            choices[key] = v
            assert rid not in byid
            byid[rid] = key
    assert set(byid) == ids and len(ids) == 360
    b.save(HERE / 'public/FINITE_CHOICES.json', {'source': 'Exact ordered choice arrays extracted with ast.literal_eval from actual S0 submitted batch SamplingParams; no response or Gold reads',
        'choices': choices, 'request_keys': byid, 'S0_batch_bindings': bindings})
    files = original['files'] + [b.binding(HERE / name) for name in ('run_v2.py', 'prepare_addendum_v2.py', 'public/FINITE_CHOICES.json')]
    new = {**original, 'schema_version': 'day5_train_diagnosis_execution_addendum_v2',
        'status': 'CPU_PREPARED_V2_ENDPOINT_BINDINGS_AND_INDEPENDENT_AUDIT_REQUIRED',
        'files': files, 'original_freeze': b.binding(HERE / 'PREPARATION_FREEZE.json'),
        'mechanical_changes': ['Bind actual S0 finite-choice sequence from submitted parameters, compare newly generated choice sets and lengths only, execute original ordered sequence.',
            'Make old scorer common module import explicit before reuse; no scoring rule changes.', 'Record actual submitted SamplingParams.'],
        'unchanged': ['Selected24 questions and6 roots', 'Source and Gold', 'Prompt bytes and original seeds', 'All request counts and diagnostic gates'],
        'original_v1_preserved': True, 'executor': b.binding(HERE / 'run_v2.py'), 'new_solver_calls': 0}
    b.save(HERE / 'EXECUTION_ADDENDUM_V2.json', new)
    print(json.dumps({'status': 'CPU_V2_ADDED_ORIGINAL_V1_PRESERVED', 'selected_requests': len(byid), 'distinct_ordered_choice_arrays': len(choices), 'new_solver_calls': 0}))


if __name__ == '__main__':
    main()
