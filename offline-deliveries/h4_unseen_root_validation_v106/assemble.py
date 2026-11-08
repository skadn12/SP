"""Append-only content assembly, with source-first review before private Gold."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def rng(root, purpose):
    return random.Random(f'v106|{root}|{purpose}')


def project(root):
    directory = HERE / 'roots' / root
    source = read(directory / 'SOURCE.json')
    review = read(directory / 'SOURCE_REVIEW.json')
    assert review['source_sha256'] == sha(directory / 'SOURCE.json') and review['full_source_read']
    if review['decision'] != 'READY':
        return
    pairs = read(directory / 'FACT_ASSETS.json')
    assert [f['fact_id'] for f in pairs['facts']] == [f'F{i}' for i in range(1, 7)]
    candidates = []
    for fact in pairs['facts']:
        for side in ('true', 'false'):
            if fact.get(side + '_statement'):
                candidates.append({'fact_id': fact['fact_id'], 'side': side,
                                   'proposition': fact[side + '_statement']})
    rng(root, 'asset-review-display').shuffle(candidates)
    public, private = [], []
    for i, row in enumerate(candidates, 1):
        asset_id = f'P{i:02d}'
        public.append({'asset_id': asset_id, 'proposition': row['proposition']})
        private.append({**row, 'asset_id': asset_id})
    draft = read(directory / 'ORDERING_DRAFT.json')
    orders = []
    for slot in draft['slots']:
        orders.append({'slot': slot['slot'], 'scene_ref': slot['scene_ref'],
                       'question': 'Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the ' + slot['scene_ref'] + ' scene.',
                       'items': {k: v['statement'] for k, v in slot.get('items', {}).items()},
                       'writer_failure': slot.get('failure')})
    write(directory / 'ASSET_REVIEW_QUERY.json', {
        'root_id': root, 'source_sha256': sha(directory / 'SOURCE.json'),
        'scene_ref': pairs['scene_ref'],
        'question_scope': 'Which statements are supported by the supplied dialogue within the ' + pairs['scene_ref'] + ' scene?',
        'propositions': public, 'ordering': orders,
        'private_gold_included': False, 'intended_relations_or_writer_citations_included': False})
    write(directory / 'ASSET_BINDINGS_PRIVATE.json', {'rows': private})


def citation_ok(c, turns):
    return (c.get('turn_ref') in turns and c.get('speaker', turns[c['turn_ref']]['speaker']) == turns[c['turn_ref']]['speaker']
            and isinstance(c.get('quote'), str) and bool(c['quote']) and c['quote'] in turns[c['turn_ref']]['text'])


def content_gate(root):
    directory = HERE / 'roots' / root
    source = read(directory / 'SOURCE.json')
    turns = {t['turn_ref']: t for t in source['turns']}
    assert len(turns) == len(source['turns'])
    query = read(directory / 'ASSET_REVIEW_QUERY.json')
    review = read(directory / 'CONTENT_REVIEW.json')
    assert review['full_source_read'] and review['private_gold_read'] is False
    assert review['writer_assets_read'] is False and review['solver_scores_read'] is False
    assert review['source_sha256'] == sha(directory / 'SOURCE.json')
    judged = {r['asset_id']: r for r in review['propositions']}
    assert len(judged) == len(review['propositions']) == len(query['propositions'])
    assert set(judged) == {r['asset_id'] for r in query['propositions']}
    for p in query['propositions']:
        r = judged[p['asset_id']]
        assert r['proposition'] == p['proposition']
        assert r['relation'] in ('supports', 'contradicts', 'unknown')
        assert r['citations'] and all(citation_ok(c, turns) for c in r['citations'])
    orders = {r['slot']: r for r in review['ordering']}
    assert len(orders) == len(review['ordering']) == len(query['ordering'])
    for q in query['ordering']:
        r = orders[q['slot']]
        assert set(r['items']) == set(q['items'])
        for label, text in q['items'].items():
            item = r['items'][label]
            assert item['proposition'] == text
            for c in [item.get('order_anchor'), *item.get('alternative_matching_anchors', [])]:
                if c is not None:
                    assert citation_ok(c, turns), (root, q['slot'], label, c)
    return directory, source, query, review, judged, orders


def checked_bindings(directory, query):
    rows = read(directory / 'ASSET_BINDINGS_PRIVATE.json')['rows']
    public = {p['asset_id']: p['proposition'] for p in query['propositions']}
    assert len(rows) == len(public) == len({r['asset_id'] for r in rows})
    assert {r['asset_id']: r['proposition'] for r in rows} == public
    assert len({(r['fact_id'], r['side']) for r in rows}) == len(rows)
    assert {r['fact_id'] for r in rows} == {f'F{i}' for i in range(1, 7)}
    assert all(r['side'] in ('true', 'false') for r in rows)
    assert {r['fact_id'] for r in rows if r['side'] == 'true'} == {f'F{i}' for i in range(1, 7)}
    writer = read(directory / 'FACT_ASSETS.json')['facts']
    originals = {(r['fact_id'], side): r[side + '_statement']
                 for r in writer for side in ('true', 'false') if r.get(side + '_statement')}
    assert {(r['fact_id'], r['side']): r['proposition'] for r in rows} == originals
    return rows


def compose(root):
    directory, source, query, review, judged, orders = content_gate(root)
    turns = {t['turn_ref']: t for t in source['turns']}
    # All source-derived judgments above were saved before reading intended relations.
    bindings = checked_bindings(directory, query)
    by_fact = {}
    for row in bindings:
        by_fact.setdefault(row['fact_id'], {})[row['side']] = row
    author = read(directory / 'SOURCE_REVIEW.json')
    ready = author['decision'] == review['source_decision'] == 'READY'
    public, private, outcomes = [], [], []
    for g in range(2, 7):
        order = list(range(6))
        rng(root, f'gold|{g}').shuffle(order)
        selected_true = set(order[:g])
        choices = [by_fact[f'F{i + 1}'].get('true' if i in selected_true else 'false') for i in range(6)]
        rng(root, f'display|{g}').shuffle(choices)
        cid = f'v106-root{root.rsplit("-", 1)[1]}-ms-g{g}'
        role = 'control6' if g == 6 else 'new_child'
        issues = []
        if not ready:
            issues.append('SOURCE_NOT_READY')
        if any(c is None for c in choices):
            outcomes.append({'candidate_id': cid, 'root_id': root, 'question_type': 'multi_select',
                             'role': role, 'intended_gold_count': g, 'status': 'CONTENT_FAIL',
                             'issues': issues + ['MISSING_FIXED_FACT_VARIANT'], 'source_derived_answer': None})
            continue
        options = {label: c['proposition'] for label, c in zip('ABCDEF', choices)}
        q = {'candidate_id': cid, 'root_id': root, 'question_type': 'multi_select',
             'question': query['question_scope'], 'options': options,
             'source_path': str(directory / 'SOURCE.json'), 'source_sha256': sha(directory / 'SOURCE.json')}
        gold = [label for label, c in zip('ABCDEF', choices) if c['side'] == 'true']
        derived = []
        evidence = {}
        for label, c in zip('ABCDEF', choices):
            r = judged[c['asset_id']]
            if r['relation'] == 'supports':
                derived.append(label)
            expected = 'supports' if c['side'] == 'true' else 'contradicts'
            if r['relation'] != expected or not r.get('scope_valid') or not r.get('fact_distinct'):
                issues.append('ASSET_' + c['asset_id'] + '_' + r['relation'].upper())
            if re.search(r'\b(?:he|she|him|her|his|hers|himself|herself)\b', c['proposition'], re.I):
                issues.append('FORBIDDEN_GENDER_PRONOUN')
            evidence[label] = r['citations']
        if len(set(options.values())) != 6:
            issues.append('DUPLICATE_OPTIONS')
        if gold != derived:
            issues.append('SOURCE_GOLD_MISMATCH')
        public.append(q)
        private.append({'candidate_id': cid, 'role': role, 'gold': gold, 'evidence': evidence})
        outcomes.append({'candidate_id': cid, 'root_id': root, 'question_type': 'multi_select', 'role': role,
                         'intended_gold_count': g, 'status': 'CONTENT_FAIL' if issues else 'CONTENT_PASS',
                         'issues': sorted(set(issues)), 'source_derived_answer': derived,
                         'asset_ids_by_label': {l: c['asset_id'] for l, c in zip('ABCDEF', choices)},
                         'review_saved_before_private_gold': True})
    plan = read(directory / 'ORDERING_PLAN.json')
    index = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
    for q in query['ordering']:
        slot = q['slot']
        cid = f'v106-root{root.rsplit("-", 1)[1]}-ordering{slot:02d}'
        r = orders[slot]
        planned = next(s for s in plan['slots'] if s['slot'] == slot)
        issues = [] if ready else ['SOURCE_NOT_READY']
        labels = [f'I{i}' for i in range(1, 7)]
        if set(q['items']) != set(labels):
            issues.append('INCOMPLETE_ORDERING_DRAFT')
        gold = sorted(labels, key=lambda l: index[planned['anchors'][l]['turn_ref']])
        evidence, possible = {}, []
        for label in labels:
            if label not in r['items']:
                continue
            item = r['items'][label]
            if not all(item.get(k) is True for k in ('supported', 'actor_ok', 'modality_ok', 'state_ok', 'neutral_stem_ok', 'single_event')):
                issues.append('ORDERING_ITEM_REJECTED_' + label)
            anchor = item.get('order_anchor')
            alternatives = item.get('alternative_matching_anchors', [])
            anchors = [anchor, *alternatives] if anchor else []
            spans = []
            for a in anchors:
                text = turns[a['turn_ref']]['text']
                if text.count(a['quote']) != 1:
                    issues.append('ORDERING_ANCHOR_NOT_UNIQUE_' + label)
                start = text.find(a['quote'])
                spans.append((index[a['turn_ref']], start, start + len(a['quote'])))
                if turns[a['turn_ref']]['scene_ref'] != q['scene_ref']:
                    issues.append('ORDERING_ANCHOR_OUTSIDE_PUBLIC_SCOPE_' + label)
            if anchor:
                evidence[label] = {**anchor, 'speaker': turns[anchor['turn_ref']]['speaker']}
                if anchor['turn_ref'] != planned['anchors'][label]['turn_ref']:
                    issues.append('ORDERING_ANCHOR_MOVED_' + label)
            possible.append((label, spans))
        possible_orders = set()
        combinations = 0
        if len(possible) == 6 and all(p for _, p in possible):
            for combo in itertools.product(*(p for _, p in possible)):
                combinations += 1
                positions = dict(zip(labels, combo))
                perm = tuple(sorted(labels, key=lambda l: positions[l]))
                if any(positions[a][0] == positions[b][0] and positions[a][2] > positions[b][1]
                       for a, b in zip(perm, perm[1:])):
                    issues.append('ORDERING_OVERLAP_OR_TIE')
                possible_orders.add(perm)
        if possible_orders != {tuple(gold)} or not all(r.get(k) is True for k in ('coherent', 'unique_order', 'neutral_stem_ok', 'scope_valid')):
            issues.append('ORDERING_NOT_UNIQUE_OR_GOLD_MISMATCH')
        question = {'candidate_id': cid, 'root_id': root, 'question_type': 'ordering',
                    'chronology_kind': 'conversation_order', 'question': q['question'],
                    'items': q['items'], 'source_path': str(directory / 'SOURCE.json'),
                    'source_sha256': sha(directory / 'SOURCE.json')}
        public.append(question)
        private.append({'candidate_id': cid, 'role': 'ordering', 'gold': gold, 'evidence': evidence})
        outcomes.append({'candidate_id': cid, 'root_id': root, 'question_type': 'ordering', 'role': 'ordering',
                         'status': 'CONTENT_FAIL' if issues else 'CONTENT_PASS', 'issues': sorted(set(issues)),
                         'source_derived_answer': list(next(iter(possible_orders))) if len(possible_orders) == 1 else None,
                         'enumerated_anchor_combinations': combinations,
                         'alternative_orders': [list(o) for o in sorted(possible_orders)],
                         'review_saved_before_private_gold': True})
    write(directory / 'PUBLIC_QUESTIONS.json', {'questions': public})
    write(directory / 'PRIVATE_GOLD.json', {'questions': private})
    write(directory / 'FINAL_CROSSCHECK.json', {'created_at_utc': now(), 'results': outcomes,
        'source_review': {'path': str(directory / 'SOURCE_REVIEW.json'), 'sha256': sha(directory / 'SOURCE_REVIEW.json')},
        'content_review': {'path': str(directory / 'CONTENT_REVIEW.json'), 'sha256': sha(directory / 'CONTENT_REVIEW.json')},
        'source_sha256': sha(directory / 'SOURCE.json'), 'source_unchanged': True})
    print(json.dumps({'root': root, 'pass': sum(r['status'] == 'CONTENT_PASS' for r in outcomes), 'total': len(outcomes)}))


def finalize():
    config = read(HERE / 'RUN_CONFIG.json')
    all_q, all_p, all_outcomes = [], [], []
    for root in config['roots']:
        directory = HERE / 'roots' / root['root_id']
        assert sha(directory / 'SOURCE.json') == root['source_sha256']
        assert sha(directory / 'ORDERING_PLAN.json') == root['ordering_plan_sha256']
        if not (directory / 'FINAL_CROSSCHECK.json').exists():
            review = read(directory / 'SOURCE_REVIEW.json')
            assert review['decision'] == 'HOLD'
            for g in range(2, 7):
                all_outcomes.append({'root_id': root['root_id'], 'intended_gold_count': g, 'status': 'SOURCE_HOLD', 'issues': review['blocking_issues']})
            for _ in range(2):
                all_outcomes.append({'root_id': root['root_id'], 'question_type': 'ordering', 'status': 'SOURCE_HOLD', 'issues': review['blocking_issues']})
            continue
        _, source, query, review, judged, order_reviews = content_gate(root['root_id'])
        checked_bindings(directory, query)
        crosscheck = read(directory / 'FINAL_CROSSCHECK.json')
        assert sha(directory / 'CONTENT_REVIEW.json') == crosscheck['content_review']['sha256']
        assert sha(directory / 'SOURCE_REVIEW.json') == crosscheck['source_review']['sha256']
        q_by = {q['candidate_id']: q for q in read(directory / 'PUBLIC_QUESTIONS.json')['questions']}
        p_by = {q['candidate_id']: q for q in read(directory / 'PRIVATE_GOLD.json')['questions']}
        assert set(q_by) == set(p_by) == {r['candidate_id'] for r in crosscheck['results']}
        plan = read(directory / 'ORDERING_PLAN.json')
        positions_by_ref = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        turns = {t['turn_ref']: t for t in source['turns']}
        for row in crosscheck['results']:
            q = q_by[row['candidate_id']]
            if q['question_type'] == 'multi_select':
                assert all(q['options'][label] == judged[aid]['proposition']
                           for label, aid in row['asset_ids_by_label'].items())
            else:
                slot = int(q['candidate_id'].rsplit('ordering', 1)[1])
                original = next(s for s in query['ordering'] if s['slot'] == slot)
                fixed = next(s for s in plan['slots'] if s['slot'] == slot)
                independent = order_reviews[slot]
                assert q['items'] == original['items'] and q['question'] == original['question']
                expected_gold = sorted(fixed['anchors'], key=lambda l: positions_by_ref[fixed['anchors'][l]['turn_ref']])
                assert p_by[q['candidate_id']]['gold'] == expected_gold
                spans = []
                fixed_labels = [f'I{i}' for i in range(1, 7)]
                for label in fixed_labels:
                    item = independent['items'][label]
                    main = item.get('order_anchor')
                    if main:
                        assert p_by[q['candidate_id']]['evidence'][label] == {**main, 'speaker': turns[main['turn_ref']]['speaker']}
                    possible = [main, *item.get('alternative_matching_anchors', [])] if main else []
                    spans.append([(positions_by_ref[a['turn_ref']], turns[a['turn_ref']]['text'].find(a['quote']),
                                   turns[a['turn_ref']]['text'].find(a['quote']) + len(a['quote'])) for a in possible])
                combinations = list(itertools.product(*spans)) if all(spans) else []
                orders = {tuple(l for l, span in sorted(zip(fixed_labels, combo), key=lambda pair: pair[1])) for combo in combinations}
                assert row['enumerated_anchor_combinations'] == len(combinations)
                assert {tuple(o) for o in row['alternative_orders']} == orders
            if row['status'] == 'CONTENT_PASS':
                assert row['source_derived_answer'] == p_by[row['candidate_id']]['gold'] and not row['issues']
        all_q += read(directory / 'PUBLIC_QUESTIONS.json')['questions']
        all_p += read(directory / 'PRIVATE_GOLD.json')['questions']
        all_outcomes += read(directory / 'FINAL_CROSSCHECK.json')['results']
    assert len(all_outcomes) == 28
    assert len({q['candidate_id'] for q in all_q}) == len(all_q) == len(all_p)
    passed = {r['candidate_id'] for r in all_outcomes if r['status'] == 'CONTENT_PASS'}
    selected_q = [q for q in all_q if q['candidate_id'] in passed]
    selected_p = [q for q in all_p if q['candidate_id'] in passed]
    controls = {q['candidate_id'] for q in selected_p if q['role'] == 'control6'}
    for name, rows in (('NEW_PUBLIC_QUESTIONS.json', [q for q in selected_q if q['candidate_id'] not in controls]),
                       ('NEW_PRIVATE_GOLD.json', [q for q in selected_p if q['candidate_id'] not in controls]),
                       ('CONTROL_PUBLIC_QUESTIONS.json', [q for q in selected_q if q['candidate_id'] in controls]),
                       ('CONTROL_PRIVATE_GOLD.json', [q for q in selected_p if q['candidate_id'] in controls])):
        write(HERE / name, {'questions': rows})
    write(HERE / 'ALL_OUTCOMES_INDEX.json', {'results': all_outcomes})
    rejected = {q['candidate_id'] for q in all_q} - passed
    write(HERE / 'REJECTED_CANDIDATES.json', {'public_questions': [q for q in all_q if q['candidate_id'] in rejected],
        'private_gold': [q for q in all_p if q['candidate_id'] in rejected],
        'outcomes': [r for r in all_outcomes if r['status'] != 'CONTENT_PASS']})
    counts = dict(Counter(q['question_type'] for q in selected_q))
    gold_counts = dict(Counter(str(len(q['gold'])) for q in selected_p if q['role'] != 'ordering'))
    calls = sum(18 if q['question_type'] == 'multi_select' else 12 for q in selected_q)
    write(HERE / 'FINAL_BATCH_ACCOUNTING.json', {'status': 'CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_VALIDATED',
        'attempted_roots': 4, 'planned_slots': 28, 'public_candidates': len(selected_q),
        'content_failed_slots': len(all_outcomes) - len(passed),
        'new_source_turns': 0, 'source_substitutions': 0, 'semantic_repairs': 0,
        'evaluation_counts': {'candidate_count': len(selected_q), 'type_counts': counts,
                              'multi_gold_counts': gold_counts, 'calls_per_model': calls},
        'training_started': False, 'test_to_training_export': False,
        'sampling': config['source_selection'], 'historical_test_evaluated': True,
        'same_inherited_model_generation_review': True, 'exact_generator_model_id': None})
    files = [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    write(HERE / 'COMPLETE_MANIFEST.json', {'status': 'SEALED_CONTENT_ONLY', 'sealed_at_utc': now(),
        'files': files, 'file_count': len(files), 'self_hash_excluded': True,
        'training_started': False, 'test_to_training_export': False,
        'content_freeze_not_external_semantic_certification': True})
    print(json.dumps({'status': 'CONTENT_SEALED', 'counts': counts, 'gold': gold_counts, 'calls_per_model': calls}))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('project', 'compose', 'finalize'))
    p.add_argument('--root')
    a = p.parse_args()
    if a.phase == 'finalize':
        finalize()
    else:
        assert a.root
        {'project': project, 'compose': compose}[a.phase](a.root)
