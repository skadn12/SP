#!/usr/bin/env python3
"""Freeze a separate Solver input after auditing the directed-pair contract."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / 'solver_gate_20261108_v1'
MANIFEST = BASE / 'candidate_manifest_20261108_v1/CANDIDATE_MANIFEST.json'
OUT = BASE / 'solver_gate_20261108_schema_hold_v1'
OLD_ADAPTER = (BASE.parent / 'training_set_structure_pilot_plan_20261106_v1/'
               'execution_20261106_v1/solver/adapter.py')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_new(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)


def main():
    full_path = ORIGINAL / 'SOLVER_INPUT_FULL_CONTEXT.jsonl'
    no_path = ORIGINAL / 'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl'
    full, no = jsonl(full_path), jsonl(no_path)
    assert len(full) == len(no) == 138
    assert len({r['candidate_id'] for r in full}) == 138
    assert [r['candidate_id'] for r in full] == [r['candidate_id'] for r in no]
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    records = {r['candidate_id']: r for r in manifest['records']}
    holds = []
    for row in full:
        if row['question_type'] != 'partial_order':
            continue
        assert row['gold_option_ids'] == [] and row['gold_pairs'] and row['query_pairs']
        gold = set(map(tuple, row['gold_pairs']))
        domain = set(map(tuple, row['query_pairs']))
        outside = sorted(gold - domain)
        if not outside:
            continue
        record = records[row['candidate_id']]
        draft = Path(record['author_draft_path'])
        assert digest(draft) == record['author_draft_sha256']
        candidate = record['candidate']
        assert candidate['gold_pairs'] == row['gold_pairs']
        assert candidate['query_pairs'] == row['query_pairs']
        holds.append({
            'candidate_id': row['candidate_id'], 'packet_id': row['packet_id'],
            'root_id': row['root_id'], 'status': 'EVAL_HOLD',
            'reason': 'SOURCE_UNORDERED_QUERY_PAIRS_INCOMPATIBLE_WITH_FROZEN_DIRECTED_QUERY_DOMAIN',
            'gold_pairs_outside_directed_query_domain': outside,
            'source_query_scope': candidate.get('query_scope'),
            'source_pair_audit': candidate.get('pair_audit'),
            'original_gold_pairs': row['gold_pairs'], 'original_query_pairs': row['query_pairs'],
            'author_draft_path': str(draft), 'author_draft_sha256': digest(draft),
            'source_path': row['source_path'], 'source_sha256': row['source_sha256'],
            'explanation': ('The author draft describes unordered pairs and explicitly records reverse '
                            'entailed relations. The frozen adapter requires directed gold pairs to be '
                            'members of query_pairs. This is an interface HOLD, not a finding that Gold is wrong.'),
            'gold_or_query_modified': False,
        })
    assert [r['candidate_id'] for r in holds] == ['FGX1_F_77882_01'], holds
    excluded = {r['candidate_id'] for r in holds}
    kept_full = [r for r in full if r['candidate_id'] not in excluded]
    kept_no = [r for r in no if r['candidate_id'] not in excluded]
    assert len(kept_full) == len(kept_no) == 137
    OUT.mkdir(exist_ok=False)
    write_new(OUT / 'SOLVER_INPUT_FULL_CONTEXT.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept_full))
    write_new(OUT / 'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept_no))
    audit = {
        'schema': 'solver_schema_hold_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_count': len(full), 'eligible_count': len(kept_full), 'hold_count': len(holds),
        'eligible_by_type': dict(Counter(r['question_type'] for r in kept_full)),
        'eligible_by_arm': dict(Counter(r['arm'] for r in kept_full)),
        'planned_requests_if_six_displays_three_repeats_two_contexts': len(kept_full) * 6 * 3 * 2,
        'holds': holds, 'original_inputs_unchanged': True, 'automatic_retry': False,
        'input_bindings': [{'path': str(p), 'sha256': digest(p)} for p in (full_path, no_path, MANIFEST, OLD_ADAPTER, Path(__file__))],
        'output_bindings': [{'path': str(OUT / name), 'sha256': digest(OUT / name)} for name in
                            ('SOLVER_INPUT_FULL_CONTEXT.jsonl', 'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl')],
        'limitations': 'Provisional Solver eligibility only; prior global lineage and training-admission limitations remain in force.',
    }
    write_new(OUT / 'SCHEMA_HOLD_AUDIT.json', json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('input_count', 'eligible_count', 'hold_count', 'eligible_by_type', 'eligible_by_arm', 'planned_requests_if_six_displays_three_repeats_two_contexts')}, indent=2))


if __name__ == '__main__':
    main()
