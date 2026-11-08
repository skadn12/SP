"""Mechanical packet/receipt mapping; never generates questions or decisions."""
import pathlib
from workflow_io import BASE, bind, load, save, now, candidate_errors


def make_packet(candidates, folder):
    folder = pathlib.Path(folder)
    assert candidates
    assert len({c['root_id'] for c in candidates}) == 1
    assert len({str(c['source_binding']) for c in candidates}) == 1
    source = load(candidates[0]['source_binding']['path'])
    assert bind(candidates[0]['source_binding']['path']) == candidates[0]['source_binding']
    entries, questions = [], []
    for index, candidate in enumerate(candidates, 1):
        review_id = f'Q{index:03d}'
        fields = ['question_type', 'task_operation', 'question',
                  'items' if candidate['question_type'] == 'ordering' else 'options']
        questions.append({'review_id': review_id, **{key: candidate[key] for key in fields}})
        entries.append({'review_id': review_id, 'candidate': candidate})
    assert not (folder / 'PUBLIC_PACKET.json').exists(), 'No packet overwrite'
    save(folder / 'PUBLIC_PACKET.json', {'root_id': source['root_id'], 'full_source': source, 'questions': questions})
    save(folder / 'PRIVATE_MAPPING.json', {'entries': entries})
    save(folder / 'PACKET_RECEIPT.json', {'created_at': now(), 'public': bind(folder / 'PUBLIC_PACKET.json'),
                                       'private': bind(folder / 'PRIVATE_MAPPING.json')})


def map_reviews(folder):
    folder = pathlib.Path(folder)
    packet = load(folder / 'PUBLIC_PACKET.json')
    entries = load(folder / 'PRIVATE_MAPPING.json')['entries']
    ids = {e['review_id'] for e in entries}
    responses = {}
    receipts = {}
    mapped = {}
    for name, filename in [('gold', 'INDEPENDENT_GOLD.json'), ('review', 'CONTENT_REVIEW.json')]:
        path = folder / filename
        if not path.exists():
            return {'folder': str(folder), 'status': 'PENDING', 'missing': filename}, []
        response = load(path)
        assert response['root_id'] == packet['root_id']
        assert set(r['review_id'] for r in response['slots']) == ids
        assert len(response['slots']) == len(ids)
        responses[name] = response
        for arm in sorted({e['candidate']['arm'] for e in entries}):
            slots = []
            for entry in entries:
                if entry['candidate']['arm'] != arm:
                    continue
                result = next(r for r in response['slots'] if r['review_id'] == entry['review_id'])
                row = {**result, 'slot_id': entry['candidate']['slot_id']}
                slots.append(row)
                mapped[(name, entry['review_id'])] = row
            receipt = {'transport': 'direct_collaboration', 'reviewer_id': response['reviewer_id'],
                       'public_packet_binding': bind(folder / 'PUBLIC_PACKET.json'),
                       'private_mapping_binding': bind(folder / 'PRIVATE_MAPPING.json'),
                       'response_binding': bind(path), 'root_id': packet['root_id'], 'arm': arm,
                       'completed_at': response.get('completed_at'), 'slots': slots}
            rp = folder / f'{arm}_{name.upper()}_ADMISSION_RECEIPT.json'
            save(rp, receipt)
            receipts[(name, arm)] = str(rp.resolve())
    outcomes = []
    turns = {t['turn_ref']: t['text'] for t in packet['full_source']['turns']}
    for entry in entries:
        candidate, rid = entry['candidate'], entry['review_id']
        gold, content = mapped[('gold', rid)], mapped[('review', rid)]
        errors = candidate_errors(candidate)
        if gold['decision'] != 'PASS': errors.append('independent_gold_not_pass')
        if content['decision'] != 'PASS': errors.append('content_review_not_pass')
        if gold.get('gold') != candidate['gold']: errors.append('independent_gold_disagreement')
        for name, row in [('gold', gold), ('review', content)]:
            if not row.get('source_quotes'): errors.append(name + ':missing_quotes')
            for q in row.get('source_quotes', []):
                if q.get('turn_ref') not in turns or not q.get('quote') or q['quote'] not in turns[q['turn_ref']]:
                    errors.append(name + ':quote_mismatch')
        reviewer_ids = [r['reviewer_id'] for r in responses.values()]
        if len(set(reviewer_ids + [candidate['author_id']])) != 3:
            errors.append('author_gold_content_not_distinct')
        row = {k: candidate[k] for k in ['root_id', 'arm', 'slot_id', 'candidate_id']}
        row.update(status='FAIL' if errors else 'PASS', candidate=candidate,
                   review={'status': 'FAIL' if errors else 'PASS', 'errors': errors, 'gold': gold, 'review': content,
                           'gold_receipt': receipts[('gold', candidate['arm'])],
                           'review_receipt': receipts[('review', candidate['arm'])]})
        outcomes.append(row)
    result = {'folder': str(folder), 'status': 'COMPLETE', 'pass_count': sum(r['status'] == 'PASS' for r in outcomes),
              'count': len(outcomes), 'at': now(), 'slots': outcomes}
    save(folder / 'ADMISSION.json', result)
    return {k: v for k, v in result.items() if k != 'slots'}, outcomes


if __name__ == '__main__':
    import sys, json
    for folder in sys.argv[1:]:
        print(json.dumps(map_reviews(folder)[0], ensure_ascii=False))
