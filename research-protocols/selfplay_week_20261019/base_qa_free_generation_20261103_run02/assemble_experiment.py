"""Assemble frozen research drafts for display; does not generate or revise QA."""
import hashlib
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEED = 2026110302


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def gold_of(item):
    for key in ('gold_option_ids', 'gold_order', 'gold_semantic_ids'):
        if key in item:
            return item[key]
    raise ValueError(f'No explicit semantic Gold: {list(item)}')


def assemble(author):
    folder = BASE / author
    raw = read(folder / 'CANDIDATES_FIRST_DRAFT.json')
    candidates = raw['candidates'] if isinstance(raw, dict) else raw
    targets_raw = read(folder / 'TARGETS_FROZEN.json')
    targets = targets_raw['targets'] if isinstance(targets_raw, dict) else targets_raw
    target_index = {t.get('candidate_id', t.get('target_id')): t for t in targets}
    source_context = read(BASE / 'SOURCE_CONTEXT.json')['roots']
    checks = []
    for candidate in candidates:
        cid = candidate['candidate_id']
        root = candidate['root_id']
        problems = []
        source_path = Path(candidate['source_path'])
        source = read(source_path)
        source_hash = digest(source_path)
        if source_hash != candidate['source_sha256']:
            problems.append('SOURCE_HASH_MISMATCH')
        if source_hash != source_context[root]['sha256']:
            problems.append('INPUT_BINDING_MISMATCH')
        qtype = candidate['question_type']
        if qtype in ('multi_select', 'multiple_select', 'multiple_choice_multiple_select'):
            normalized_type = 'multi_select'
        elif qtype in ('ordering', 'full_order', 'full_ordering', 'complete_order'):
            normalized_type = 'full_order'
        else:
            raise ValueError(f'Unrecognized type {qtype}; do not silently convert')
        options = candidate.get('options', candidate.get('items'))
        if isinstance(options, dict):
            options = list(options.values())
        ids = [option['option_id'] for option in options]
        if len(set(ids)) != len(ids):
            problems.append('DUPLICATE_OPTION_IDS')
        gold = gold_of(candidate)
        if not isinstance(gold, list) or not set(gold) <= set(ids):
            raise ValueError(f'Invalid explicit semantic Gold: {cid}')
        if normalized_type == 'multi_select':
            if not (2 <= len(gold) <= 6 and len(gold) < len(ids)):
                problems.append('MULTI_SELECT_GOLD_RANGE_OR_NO_DISTRACTOR')
        elif len(gold) != len(ids) or set(gold) != set(ids):
            problems.append('ORDER_NOT_A_FULL_PERMUTATION')
        target_id = candidate.get('target_id', cid)
        target = target_index.get(target_id)
        if target is None:
            problems.append('NO_EXPLICIT_FROZEN_TARGET_BINDING')
        else:
            frozen_gold = gold_of(target)
            match = (set(gold) == set(frozen_gold) if normalized_type == 'multi_select'
                     else gold == frozen_gold)
            if not match:
                problems.append('GOLD_DIFFERS_FROM_FROZEN_TARGET')
        turns = {t['turn_ref']: t for t in source['turns']}
        evidence_checks = []
        for evidence in candidate['evidence']:
            ref = evidence['turn_ref']
            turn = turns.get(ref)
            quote = evidence.get('text', evidence.get('quote'))
            exact = bool(turn and isinstance(quote, str) and quote and quote in turn['text'])
            evidence_checks.append({'turn_ref': ref, 'ref_resolves': turn is not None,
                                    'exact_quote': exact,
                                    'actual_origin': turn.get('origin') if turn else None})
            if not exact:
                problems.append('UNRESOLVED_OR_NONEXACT_EVIDENCE:' + ref)
        order = list(range(len(options)))
        item_seed = int(hashlib.sha256(f'{SEED}:{cid}'.encode()).hexdigest(), 16)
        random.Random(item_seed).shuffle(order)
        labels = ([chr(65 + i) for i in range(len(options))]
                  if normalized_type == 'multi_select'
                  else [f'I{i + 1}' for i in range(len(options))])
        display = [{'label': labels[i], 'text': options[index]['text']}
                   for i, index in enumerate(order)]
        label_to_id = {labels[i]: ids[index] for i, index in enumerate(order)}
        id_to_label = {oid: label for label, oid in label_to_id.items()}
        if normalized_type == 'multi_select':
            display_gold = [label for label in labels if label_to_id[label] in gold]
        else:
            display_gold = [id_to_label[oid] for oid in gold]
        recovered = [label_to_id[label] for label in display_gold]
        round_trip = (set(recovered) == set(gold) if normalized_type == 'multi_select'
                      else recovered == gold)
        if not round_trip:
            problems.append('DISPLAY_GOLD_ROUND_TRIP_FAILURE')
        packet = {'candidate_id': cid, 'question_type': normalized_type,
                  'question': candidate['question'], 'options': display,
                  'source': {'root_id': root, 'path': str(source_path),
                             'sha256': source_hash, 'turn_count': len(source['turns']),
                             'source_status': source.get('source_status'),
                             'warnings': source.get('warnings'),
                             'existing_review_context': source_context[root]},
                  'review_scope': 'This single question and this complete frozen source only; '
                                  'no author Gold, certificates or other questions. '
                                  'Research draft, no formal source approval implied.'}
        private = {'candidate_id': cid, 'author': author, 'root_id': root,
                   'question_type': normalized_type, 'question': candidate['question'],
                   'options': display, 'label_to_option_id': label_to_id,
                   'semantic_gold': gold, 'display_gold': display_gold,
                   'shuffle_seed': SEED, 'permutation_original_indices': order,
                   'shuffle_count': 1, 'gold_round_trip': round_trip,
                   'author_candidate_path': str(folder / 'CANDIDATES_FIRST_DRAFT.json'),
                   'author_candidate_sha256': digest(folder / 'CANDIDATES_FIRST_DRAFT.json'),
                   'target_path': str(folder / 'TARGETS_FROZEN.json'),
                   'target_sha256': digest(folder / 'TARGETS_FROZEN.json')}
        destination = BASE / 'items' / cid
        save(destination / 'PUBLIC_PACKET.json', packet)
        save(destination / 'DISPLAY_PRIVATE.json', private)
        check = {'candidate_id': cid, 'status': 'PASS' if not problems else 'HOLD',
                 'problems': problems, 'evidence_checks': evidence_checks,
                 'public_packet_sha256': digest(destination / 'PUBLIC_PACKET.json'),
                 'display_private_sha256': digest(destination / 'DISPLAY_PRIVATE.json'),
                 'gold_round_trip': round_trip, 'target_found': target is not None,
                 'note': 'Data integrity and mapping only, not a proof of semantic correctness.'}
        save(destination / 'PROGRAM_CHECK.json', check)
        checks.append(check)
    save(folder / 'ASSEMBLY_CHECKS.json', checks)
    print(json.dumps({'author': author, 'candidate_count': len(candidates),
                      'checks': [{k: c[k] for k in ('candidate_id', 'status', 'problems')}
                                 for c in checks]}, ensure_ascii=False))


if __name__ == '__main__':
    assemble(sys.argv[1])
