#!/usr/bin/env python3
"""Freeze a provisional gate after the historical adapter's input checks."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / 'solver_gate_20261108_v1'
MANIFEST = BASE / 'candidate_manifest_20261108_v1/CANDIDATE_MANIFEST.json'
OUT = BASE / 'solver_gate_20261108_schema_hold_v3'
ADAPTER = (BASE.parent / 'training_set_structure_pilot_plan_20261106_v1/'
           'execution_20261106_v1/solver/adapter.py')
EXPECTED_ID_HOLDS = {
    'FGX1_F_146700_01', 'FGX1_F_70315_01', 'FGX1_G_146700_01',
    'FGX1_G_162896_01', 'FGX1_G_57679_01', 'FGX1_G_761_01', 'FGX1_G_78203_01',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def put(path, text):
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def main():
    full_path = ORIGINAL / 'SOLVER_INPUT_FULL_CONTEXT.jsonl'
    no_path = ORIGINAL / 'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl'
    full, no = load_jsonl(full_path), load_jsonl(no_path)
    assert len(full) == len(no) == 138
    assert [r['candidate_id'] for r in full] == [r['candidate_id'] for r in no]
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    records = {r['candidate_id']: r for r in data['records']}
    assert data['denominator'] == 140 and len(records) == 140
    spec = importlib.util.spec_from_file_location('frozen_structure_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    holds = []
    for record in data['records']:
        if record['canonical_status'] == 'PASS':
            continue
        holds.append({k: record.get(k) for k in ('candidate_id', 'packet_id', 'root_id', 'arm',
                      'question_type', 'canonical_status', 'canonical_resolution_reason', 'review_source',
                      'author_draft_path', 'author_draft_sha256')})
        holds[-1].update(status='QUALITY_HOLD', reason='canonical_review_hold')
    for row in full:
        if row['question_type'] != 'partial_order':
            continue
        record = records[row['candidate_id']]
        c = record['candidate']
        draft = Path(record['author_draft_path'])
        assert sha(draft) == record['author_draft_sha256']
        assert row['gold_pairs'] == c['gold_pairs'] and row['query_pairs'] == c['query_pairs']
        ids = [o['option_id'] for o in c['options']]
        pairs = [tuple(p) for p in c['query_pairs']]
        assert len({frozenset(p) for p in pairs}) == len(pairs)
        assert {frozenset(p) for p in pairs} == {frozenset(p) for p in combinations(ids, 2)}
        missing_reverse = sorted([list(reversed(p)) for p in pairs if tuple(reversed(p)) not in pairs])
        assert len(missing_reverse) == len(pairs)
        outside = sorted(set(map(tuple, c['gold_pairs'])) - set(pairs))
        pair_evidence = []
        for entry in c.get('pair_audit', []):
            pair_evidence.append({k: entry[k] for k in ('pair', 'query_pair', 'entailed_pair', 'entailed_pairs', 'relation') if k in entry})
        holds.append({
            'candidate_id': row['candidate_id'], 'packet_id': row['packet_id'], 'root_id': row['root_id'],
            'arm': row['arm'], 'question_type': row['question_type'], 'status': 'EVAL_HOLD',
            'reason': 'UNORDERED_SOURCE_QUERY_PAIRS_INCOMPATIBLE_WITH_FROZEN_DIRECTED_DOMAIN',
            'question': c['question'], 'source_query_scope': c.get('query_scope'),
            'source_query_pairs': c['query_pairs'], 'source_gold_pairs': c['gold_pairs'],
            'source_pair_audit_relations': pair_evidence, 'gold_pairs_outside_directed_domain': outside,
            'unordered_coverage_complete': True, 'reverse_directions_absent': missing_reverse,
            'author_draft_path': str(draft), 'author_draft_sha256': sha(draft),
            'source_path': row['source_path'], 'source_sha256': row['source_sha256'],
            'explanation': ('The draft asks for all unordered item pairs, but the frozen adapter treats '
                            'query_pairs as a directed domain. Restricting predictions to that list would '
                            'reject opposite-direction answers even where the pair is in scope. Expanding '
                            'each unordered pair to both directed orientations requires a recorded protocol '
                            'revision and is not applied here.'),
            'gold_or_query_modified': False,
        })
    pair_holds = {r['candidate_id'] for r in holds}
    id_holds = set()
    for row in full:
        if row['candidate_id'] in pair_holds:
            continue
        record = records[row['candidate_id']]
        c = record['candidate']
        ids = [o['option_id'] for o in c['options']]
        public = {key: c[key] for key in adapter.PUBLIC_FIELDS if key in c}
        try:
            adapter.check_visible_id_references(c['question'], [o['text'] for o in c['options']], public, ids)
        except AssertionError as exc:
            assert str(exc) == 'EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE', (row['candidate_id'], str(exc))
            id_holds.add(row['candidate_id'])
            texts = [('question', c['question'])] + [(f'options[{i}].text', o['text']) for i, o in enumerate(c['options'])]
            texts += [(key, c[key]) for key in adapter.PUBLIC_FIELDS if isinstance(c.get(key), str)]
            matches = [{'stable_id': oid, 'field': field, 'text': value} for oid in ids if len(oid) > 1
                       for field, value in texts if re.search(r'(?<![A-Za-z0-9_])' + re.escape(oid) + r'(?![A-Za-z0-9_])', value)]
            assert matches, row['candidate_id']
            draft = Path(record['author_draft_path'])
            assert sha(draft) == record['author_draft_sha256']
            holds.append({'candidate_id': row['candidate_id'], 'packet_id': row['packet_id'],
                          'root_id': row['root_id'], 'arm': row['arm'], 'question_type': row['question_type'],
                          'status': 'EVAL_HOLD', 'reason': str(exc), 'visible_word_matches': matches,
                          'explanation': 'Historical conservative stable-ID reference check triggered. A match can be an ordinary word and does not establish actual leakage.',
                          'author_draft_path': str(draft), 'author_draft_sha256': sha(draft)})
    assert id_holds == EXPECTED_ID_HOLDS, (sorted(id_holds), sorted(EXPECTED_ID_HOLDS))
    assert len(holds) == 12 and Counter(r['status'] for r in holds) == {'QUALITY_HOLD': 2, 'EVAL_HOLD': 10}
    excluded = {r['candidate_id'] for r in holds}
    kept_full = [r for r in full if r['candidate_id'] not in excluded]
    kept_no = [r for r in no if r['candidate_id'] not in excluded]
    assert len(kept_full) == len(kept_no) == 128
    assert set(records) == excluded | {r['candidate_id'] for r in kept_full}
    OUT.mkdir(exist_ok=False)
    for name, rows in [('SOLVER_INPUT_FULL_CONTEXT.jsonl', kept_full), ('SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl', kept_no)]:
        put(OUT / name, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    audit = {
        'schema': 'solver_schema_hold_v3', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_denominator': 140, 'previous_quality_holds': 2, 'previous_provisional_count': 138,
        'additional_solver_schema_holds': 3, 'additional_stable_id_check_holds': 7,
        'eligible_count': 128, 'total_holds': 12,
        'eligible_by_type': dict(Counter(r['question_type'] for r in kept_full)),
        'eligible_by_arm': dict(Counter(r['arm'] for r in kept_full)),
        'planned_requests_if_six_displays_three_repeats_two_contexts': 128 * 6 * 3 * 2,
        'holds': holds, 'gold_or_query_modified': False, 'automatic_retry': False,
        'input_bindings': [{'path': str(p), 'sha256': sha(p)} for p in (full_path, no_path, MANIFEST, ADAPTER, Path(__file__))],
        'output_bindings': [{'path': str(OUT / n), 'sha256': sha(OUT / n)} for n in
                            ('SOLVER_INPUT_FULL_CONTEXT.jsonl', 'SOLVER_INPUT_NO_CONTEXT_DIAGNOSTIC.jsonl')],
        'limitations': ('Provisional Solver eligibility only; quality and global lineage limitations remain. '
                        'The earlier 137-row schema_hold_v1 is retained as an intermediate audit, not the final gate.'),
    }
    put(OUT / 'SCHEMA_HOLD_AUDIT.json', json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('candidate_denominator', 'previous_quality_holds', 'additional_solver_schema_holds', 'additional_stable_id_check_holds', 'eligible_count', 'eligible_by_type', 'eligible_by_arm', 'planned_requests_if_six_displays_three_repeats_two_contexts')}, indent=2))


if __name__ == '__main__':
    main()
