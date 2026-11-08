"""Write a reviewer's explicit decisions and verify their mechanical bindings.

No question authoring, answer derivation, or review decisions occur here.
"""
from workflow_io import BASE, load, save, bind, now, candidate_errors


def emit(folder, decisions, reviewer_id='/root/content_review_new_20261025'):
    folder = BASE / folder
    packet = load(folder / 'PUBLIC_PACKET.json')
    source = packet['full_source']
    turns = {t['turn_ref']: t for t in source['turns']}
    entries = {e['review_id']: e for e in load(folder / 'PRIVATE_MAPPING.json')['entries']}
    assert set(entries) == {d['review_id'] for d in decisions}
    slots = []
    for decision in decisions:
        decision = dict(decision)
        refs = decision.pop('refs')
        decision['source_quotes'] = [{'turn_ref': r, 'quote': turns[r]['text']} for r in refs]
        decision.setdefault('decision', 'PASS')
        decision.setdefault('failure_reasons', [])
        candidate = entries[decision['review_id']]['candidate']
        errors = candidate_errors(candidate)
        decision['mechanical_errors'] = errors
        decision['author_mapping_agrees'] = decision['independently_derived_gold'] == candidate['gold']
        assert reviewer_id != candidate['author_id']
        if decision['decision'] == 'PASS':
            assert decision['author_mapping_agrees'] and not errors
        slots.append(decision)
    path = folder / 'CONTENT_REVIEW.json'
    assert not path.exists(), 'Preserve completed review; use a new revision packet'
    save(path, {'root_id': packet['root_id'], 'reviewer_id': reviewer_id, 'completed_at': now(),
                'review_method': 'Reviewer read complete public source and public questions and independently derived answers before inspecting author mapping. Explicit decisions supplied by reviewer; this helper only validates/writes receipts. No Solver output used.',
                'public_packet_binding': bind(folder / 'PUBLIC_PACKET.json'),
                'full_source_read': {'turn_count': len(turns), 'turn_refs': list(turns)}, 'slots': slots})
    return {'path': str(path), 'decisions': {s['review_id']: s['decision'] for s in slots}}
