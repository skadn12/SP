#!/usr/bin/env python3
"""Freeze an abstract card only after mainline output inspection and batch closure."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main(index, inspection):
    closed_path = R / f'batch_{index:02d}/COMPLETED_BATCH.json'
    closed = read(closed_path)
    assert closed['status'] == 'COMPLETE' and closed['both_arms_frozen_and_first_review_complete']
    review = read(inspection)
    assert review['batch_index'] == index and review['mainline_read_frozen_outputs'] is True
    assert review['mechanistic_internal_causality_claim'] is False
    assert review['abstract_card_assessment'] and review['alternative_explanations'] and review['counterevidence']
    next_index = index + 1
    assert next_index <= 9
    assert not any((d / 'F/author/CANDIDATES_FIRST_DRAFT.json').exists()
                   for d in (R / f'batch_{next_index:02d}').glob('cc-*'))
    source_holds = set(closed['source_hold_ids'])
    inspected_requests, raw_requests, planned_requests, evidence_bindings = {}, {}, {}, []
    for comp in closed['components']:
        for recorded in comp['bindings']:
            assert bind(Path(recorded['path'])) == recorded
        solver = Path(comp['path']) / 'solver'
        scores = read(solver / 'SCORES.json')
        raw = read(solver / 'RAW_ANSWERS.json')
        planned = read(solver / 'REQUESTS.json')
        score_map = {x['request_id']: x for x in scores['requests']}
        raw_map = {x['request_id']: x for x in raw}
        planned_map = {x['request_id']: x for x in planned}
        assert len(score_map) == len(scores['requests']) == len(raw_map) == len(raw) == len(planned_map) == len(planned)
        assert score_map.keys() == raw_map.keys() == planned_map.keys()
        assert not inspected_requests.keys() & score_map.keys()
        inspected_requests.update(score_map)
        raw_requests.update(raw_map)
        planned_requests.update(planned_map)
        evidence_bindings.extend(bind(solver / name) for name in ('REQUESTS.json', 'RAW_ANSWERS.json', 'SCORES.json'))
    for obs in review['observations']:
        request = inspected_requests[obs['request_id']]
        assert request['context_mode'] == 'full_context'
        assert request['valid'] is True and request['protocol_valid'] is True
        assert request['candidate_id'] not in source_holds
        assert obs['observation'].strip()
        raw = raw_requests[obs['request_id']]
        planned = planned_requests[obs['request_id']]
        assert raw['status'] == 'RETURNED' and raw['prompt_sha256'] == planned['prompt_sha256']
        obs['raw_prediction_sha256'] = hashlib.sha256(raw['raw_prediction'].encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    review.update(created_at_utc=now, batch_completion=bind(closed_path),
        inspection_input=bind(inspection), all_unique_requests_complete=closed['unique_completed_requests'],
        excluded_source_holds=sorted(source_holds), first_answers_preserved=True,
        feedback_export_is_abstract_only=True, frozen_output_bindings=evidence_bindings)
    private = R / f'feedback_cards/BATCH_{index:02d}_REVIEW_PRIVATE.json'
    write(private, review)
    previous = R / f'feedback_cards/BATCH_{index:02d}_AUTHOR_CARD.json'
    card = copy.deepcopy(read(previous))
    card.update(card_id=f'FC{next_index:02d}_MCLEAN_LIMITED_20261104_B{next_index:02d}',
        confidence=f'Limited calibration and {index} completed batches. Repeated samples, shared sources and semantic families are correlated; no general or internal causal weakness is established.',
        batch_review_assessment=review['abstract_card_assessment'],
        alternative_explanations=review['alternative_explanations'])
    new_card = R / f'feedback_cards/BATCH_{next_index:02d}_AUTHOR_CARD.json'
    write(new_card, card)
    write(R / f'feedback_cards/BATCH_{next_index:02d}_FREEZE.json', {
        'created_at_utc': now, 'batch_index': next_index, 'card': bind(new_card),
        'previous_completed_batch': bind(closed_path), 'private_review': bind(private),
        'frozen_before_next_F_authoring': True, 'G_receives_card': False,
        'root_list': [x['root_id'] for x in read(R / f'batch_{next_index:02d}/SOURCE_FREEZE.json')['sources']],
        'max_candidates_per_root_per_arm': 2, 'no_within_batch_score_guided_rewrite': True})
    print(json.dumps({'status': 'FROZEN', 'card': str(new_card), 'previous_batch': index,
                      'previous_unique_calls': closed['unique_completed_requests']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--completed-index', type=int, required=True)
    p.add_argument('--inspection', type=Path, required=True)
    a = p.parse_args()
    main(a.completed_index, a.inspection)
