#!/usr/bin/env python3
"""Offline integration check. Never loads the model or calls the GPU."""
import copy
import json
from pathlib import Path
import tempfile

import adapter


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def main():
    old = adapter.WEEK / 'training_set_rebuild_20261104_run01/new_evaluation/batch_02_F/ITEMS_PRIVATE.json'
    item = copy.deepcopy(adapter.read(old)[0])
    source = next(row for row in adapter.read(adapter.EXECUTION_FREEZE)['roots'] if row['root_id'] == item['root_id'])
    snapshot = (adapter.EXPERIMENT / source['source_snapshot']).resolve()
    item.update(candidate_id='supplement_offline_fixture', source_path=str(snapshot),
                source_sha256=source['source_snapshot_sha256'])
    with tempfile.TemporaryDirectory(prefix='offline_integration_', dir=adapter.HERE) as tmp:
        batch = Path(tmp)
        write(batch / 'ITEMS_PRIVATE.json', [item])
        write(batch / 'QUALITY_FINAL.json', {'items': [{'candidate_id': item['candidate_id'], 'content_pass': True}]})
        write(batch / 'SOURCE_QUALIFICATION.json', {'items': [{'candidate_id': item['candidate_id'], 'source_qualified': True}]})
        write(batch / 'SOURCE_FREEZE.json', {'sources': [{'root_id': item['root_id'], 'path': str(snapshot),
              'sha256': item['source_sha256'], 'turn_count': source['turn_count']}]})
        original_read = adapter.read
        def fixture_read(path):
            value = original_read(path)
            if Path(path) == adapter.IDENTITY_AUDIT:
                value = {**value, 'files': []}
            return value
        adapter.read = fixture_read
        adapter.configure(batch, batch / 'solver', 1, 'supplement')
        adapter.h.HERE.mkdir()
        adapter.h.prepare()
        requests = adapter.h.read(adapter.h.HERE / 'REQUESTS.json')
        assert len(requests) == 36
        assert min(r['sampling_seed'] for r in requests) >= 2026110502 + 2000 * 100000
        answers = []
        for idx, request in enumerate(requests):
            labels = request['gold_labels']
            final = '<box>' + ' '.join(labels) + '</box>'
            if idx == 0:
                final = '<box>' + labels[0] + '</box>'
            # Deliberately invalid auxiliary evidence; valid answers must survive.
            raw = json.dumps({'item_evidence': {}, 'answer_reason': '', 'final_answer': final})
            answers.append({'request_id': request['request_id'], 'status': 'RETURNED',
                            'finish_reason': 'stop', 'raw_prediction': raw})
        write(adapter.h.HERE / 'RAW_ANSWERS.json', answers)
        adapter.h.score()
        score = adapter.h.read(adapter.h.HERE / 'SCORES.json')
        full = score['items'][item['candidate_id']]['full_context']
        assert full['planned_calls'] == 18 and full['valid_calls'] == 17, full
        assert 0 < full['mean_J'] < 1, full
        layers = adapter.h.read(adapter.h.HERE / 'CANDIDATE_LAYER_DECISIONS.json')
        assert layers['items'][0]['status'] == 'MAIN_POOL_CANDIDATE'
        assert all(not row['auxiliary_protocol_valid'] for row in score['requests'])
        request = requests[0]
        evidence = {label: {'judgment': 'not_supported', 'turn_refs': [],
                            'reason': 'Checked the dialogue and found no support.'}
                    for label in request['display_to_option_id']}
        payload = {'item_evidence': evidence, 'answer_reason': 'Example reason.',
                   'final_answer': '<box>' + ' '.join(request['gold_labels']) + '</box>'}
        raw = {'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': json.dumps(payload)}
        checked = adapter.validate(raw, request)
        assert checked['final_answer_valid'] and checked['auxiliary_protocol_valid'], checked
        assert not checked['citation_entailment_verified']
        evidence[next(iter(evidence))]['judgment'] = 'supported'
        raw['raw_prediction'] = json.dumps(payload)
        checked = adapter.validate(raw, request)
        assert checked['final_answer_valid'] and not checked['auxiliary_protocol_valid'], checked
        adapter.HERE = batch
        reserved = adapter.reserve_budget(batch)
        assert reserved['reserved_requests'] == 36
        try:
            adapter.reserve_budget(batch)
            raise AssertionError('DUPLICATE_RESERVATION_ALLOWED')
        except AssertionError as exc:
            assert str(exc) == 'BATCH_ALREADY_RESERVED_NO_RETRY'
        print('PASS: offline prepare, paired 36 requests, invalid answer fixed denominator, auxiliary independence; no GPU calls')


if __name__ == '__main__':
    main()
