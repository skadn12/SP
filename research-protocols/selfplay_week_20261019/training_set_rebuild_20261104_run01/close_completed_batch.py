#!/usr/bin/env python3
"""Join disjoint completed execution components without scoring or inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def quality_path(d):
    path = d / 'QUALITY_FINAL.json'
    for view in ('review_enum_interface_view', 'target_hash_interface_view', 'review_citation_interface_view', 'warning_policy_view'):
        candidate = d / view / 'QUALITY_FINAL.json'
        if candidate.exists():
            path = candidate
    return path


def close(index):
    batch = R / f'batch_{index:02d}'
    roots = read(R / ('SOURCE_FREEZE.json' if index == 1 else f'batch_{index:02d}/SOURCE_FREEZE.json'))['sources']
    items, qualities, qbindings = {}, {}, []
    for root in roots:
        for arm in ('F', 'G'):
            base = batch / root['root_id'] / arm
            draft = read(base / 'author/CANDIDATES_FIRST_DRAFT.json')
            source_items = read(base / 'batch/ITEMS_PRIVATE.json')
            assert len(draft['candidates']) == len(source_items) <= 2
            path = quality_path(base / 'batch')
            qs = read(path)['items']
            assert {i['candidate_id'] for i in source_items} == {q['candidate_id'] for q in qs}
            qbindings.extend([bind(path), bind(base / 'batch/ITEMS_PRIVATE.json'), bind(base / 'author/CANDIDATES_FIRST_DRAFT.json')])
            for i in source_items:
                assert i['candidate_id'] not in items
                items[i['candidate_id']] = i
            qualities.update({q['candidate_id']: q for q in qs})
    # The first batch also has a pre-run independence HOLD which was frozen
    # at aggregate intake, while the raw per-root blind verdict is retained.
    content_addendum = batch / 'PUBLIC_QUERY_SCOPE_HOLD_APPLICATION.json'
    if content_addendum.exists():
        addendum = read(content_addendum)
        original_gate = Path(addendum['original_quality_gate']['path'])
        assert bind(original_gate) == addendum['original_quality_gate']
        original_rows = {q['candidate_id']: q for q in read(original_gate)['items']}
        for q in addendum['items']:
            assert q == original_rows[q['candidate_id']] and not q['content_pass']
            qualities[q['candidate_id']] = q
        qbindings.append(bind(content_addendum))
    components = []
    requests, raw_ids, score_ids, scores, score_paths = {}, set(), set(), {}, {}
    for comp in sorted((R / 'new_evaluation').glob(f'batch_{index:02d}*')):
        solver = comp / 'solver'
        if not (solver / 'COMPLETED.json').exists():
            continue
        req = read(solver / 'REQUESTS.json')
        raw = read(solver / 'RAW_ANSWERS.json')
        ss = read(solver / 'SCORES.json')
        rid = {x['request_id'] for x in req}
        assert len(rid) == len(req) == read(solver / 'COMPLETED.json')['returned_requests']
        assert len(raw) == len({x['request_id'] for x in raw}) == len(req)
        assert len(ss['requests']) == len({x['request_id'] for x in ss['requests']}) == len(req)
        assert rid == {x['request_id'] for x in raw} == {x['request_id'] for x in ss['requests']}
        assert not rid & set(requests), 'REPEATED_REQUEST_ACROSS_COMPONENTS'
        assert not set(ss['items']) & set(scores), 'REPEATED_CANDIDATE_EVALUATION'
        assert all(items[i['candidate_id']] == i for i in read(comp / 'ITEMS_PRIVATE.json'))
        for x in req:
            assert x['candidate_id'] in items and qualities[x['candidate_id']]['content_pass']
            requests[x['request_id']] = x
        scores.update(ss['items'])
        score_paths.update({qid: bind(solver / 'SCORES.json') for qid in ss['items']})
        raw_ids |= {x['request_id'] for x in raw}
        score_ids |= {x['request_id'] for x in ss['requests']}
        components.append({'path': str(comp), 'requests': len(req), 'bindings': [bind(solver / name) for name in ('REQUESTS.json', 'RAW_ANSWERS.json', 'SCORES.json', 'COMPLETED.json', 'MODEL_CONFIG.json')]})
    assert requests and set(requests) == raw_ids == score_ids
    source_holds = set()
    hold_file = R / f'source_preparation/BATCH{index:02d}_SOURCE_HOLD_APPLICATION.json'
    if hold_file.exists():
        source_holds.update(read(hold_file)['source_HOLD_candidate_ids'])
        qbindings.append(bind(hold_file))
    rows = []
    for qid, i in items.items():
        q, ss = qualities[qid], scores.get(qid, {})
        actual = [x for x in requests.values() if x['candidate_id'] == qid]
        planned_per_mode = min(6, math.factorial(len(i['options']))) * 3
        if not q['content_pass']:
            assert not actual and not ss
            state = 'QUALITY_HOLD_NOT_EVALUATED'
        else:
            assert len(actual) == planned_per_mode * 2, ('MISSING_PLANNED_CALLS', qid)
            for mode in ('full_context', 'no_context'):
                assert ss[mode]['planned_calls'] == sum(x['context_mode'] == mode for x in actual) == planned_per_mode
            full = ss['full_context']
            mean = full.get('mean_directed_pair_f1', full.get('mean_J'))
            if full['valid_calls'] != planned_per_mode:
                state = 'OUTPUT_PROTOCOL_PENDING'
            elif mean == 1:
                state = 'MASTERED_RESERVE'
            elif mean <= 0:
                state = 'HARD_EXAMPLE_DIAGNOSTIC'
            else:
                assert 0 < mean < 1
                state = 'MAIN_POOL_CANDIDATE'
        full = ss.get('full_context', {})
        rows.append({'candidate_id': qid, 'root_id': i['root_id'], 'arm': i['arm'], 'question_type': i['question_type'],
            'quality_status': q['status'], 'quality_projection': q,
            'source_qualification_status': 'HOLD' if qid in source_holds else 'PASS_EXISTING_CITED_USE',
            'training_value_status': state, 'metric': 'directed_pair_F1' if i['question_type'] == 'partial_order' else 'signed_J',
            'mean_full_score': full.get('mean_directed_pair_f1', full.get('mean_J')),
            'full_statistics': full, 'all_mode_statistics': ss, 'score_binding': score_paths.get(qid),
            'selection_status': 'EXCLUDE_SOURCE_HOLD' if qid in source_holds else 'PENDING_UNION_ALL68',
            'feedback_evidence_eligible': q['content_pass'] and qid not in source_holds,
            'final_answer_only_or_no_context_confers_eligibility': False})
    now = datetime.now(timezone.utc).isoformat()
    write(batch / 'CANDIDATE_LAYER_DECISIONS.json', {'created_at_utc': now, 'batch_index': index, 'items': rows,
        'quality_training_value_source_qualification_collection_separated': True,
        'all_original_items_answers_and_scores_preserved': True, 'no_rescoring_or_new_calls': True})
    value = {'created_at_utc': now, 'batch_index': index, 'status': 'COMPLETE', 'root_count': len(roots),
        'both_arms_frozen_and_first_review_complete': True, 'first_drafts': len(items),
        'quality_counts': dict(Counter(q['status'] for q in qualities.values())),
        'source_hold_ids': sorted(source_holds), 'training_value_counts': dict(Counter(x['training_value_status'] for x in rows)),
        'source_qualified_main_pool_candidates': sum(x['training_value_status'] == 'MAIN_POOL_CANDIDATE' and x['source_qualification_status'] != 'HOLD' for x in rows),
        'unique_completed_requests': len(requests), 'duplicate_requests': 0, 'components': components,
        'quality_and_item_bindings': qbindings, 'no_question_or_first_answer_rewrites': True, 'training_started': False}
    write(batch / 'COMPLETED_BATCH.json', value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('components', 'quality_and_item_bindings')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, required=True, choices=range(1, 10))
    close(p.parse_args().index)
