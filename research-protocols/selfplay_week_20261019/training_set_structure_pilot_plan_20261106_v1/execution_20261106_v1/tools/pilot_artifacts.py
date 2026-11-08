#!/usr/bin/env python3
"""Read-only candidate checks and answer-free role packets for this pilot."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def candidate(path):
    raw = read(path)
    items = raw.get('candidates') if isinstance(raw, dict) else raw
    if items is None:
        items = [raw] if raw.get('question_type') else []
    assert len(items) <= 1, ('FIRST_DRAFT_CEILING', str(path))
    if not items:
        return None
    item = dict(items[0])
    for key in ('root_id', 'source_path', 'source_sha256', 'source_turn_count'):
        if key not in item and key in raw:
            item[key] = raw[key]
    for key, alias in (('source_sha256', 'source_hash'), ('source_turn_count', 'turn_count')):
        if key not in item and alias in item:
            item[key] = item[alias]
    if 'source_path' not in item:
        manifest = read(HERE / 'manifests/EXECUTION_FREEZE.json')
        item['source_path'] = next(r['source_path'] for r in manifest['source_manifest']
                                   if r['root_id'] == item['root_id'])
    return item


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('turn_ref'), str) and any(k in value for k in ('quote', 'text')):
            yield value
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def audit(path):
    c = candidate(path)
    if c is None:
        return {'draft_path': str(path), 'draft_sha256': digest(path), 'status': 'NO_CANDIDATE'}
    bad = []
    source = read(c['source_path'])
    turns = {t['turn_ref']: t for t in source['turns']}
    if digest(c['source_path']) != c.get('source_sha256'):
        bad.append('SOURCE_HASH_MISMATCH')
    if len(turns) != c.get('source_turn_count'):
        bad.append('SOURCE_TURN_COUNT_MISMATCH')
    opts = c.get('options', [])
    ids = [o.get('option_id') for o in opts]
    if not ids or len(ids) != len(set(ids)) or None in ids:
        bad.append('OPTION_IDS_INVALID')
    gold = c.get('gold_option_ids', c.get('gold'))
    kind = c.get('question_type')
    if kind not in ('multi_select', 'ordering', 'partial_order'):
        bad.append('QUESTION_TYPE_INVALID')
    if kind == 'multi_select':
        if not isinstance(gold, list) or len(gold) != len(set(gold)) or not set(gold) <= set(ids):
            bad.append('MULTISELECT_GOLD_INVALID')
    if kind == 'ordering' and (not isinstance(gold, list) or len(gold) != len(ids) or set(gold) != set(ids)):
        bad.append('ORDERING_GOLD_INVALID')
    if kind == 'partial_order':
        pairs = c.get('gold_pairs', [])
        if gold != [] or not c.get('query_scope'):
            bad.append('PARTIAL_QUERY_INVALID')
        if any(len(p) != 2 or p[0] == p[1] or not set(p) <= set(ids) for p in pairs):
            bad.append('PARTIAL_PAIRS_INVALID')
    refs = list(references(c))
    for r in refs:
        t = turns.get(r['turn_ref'])
        quote = r.get('quote', r.get('text'))
        if t is None:
            bad.append('UNKNOWN_REF:' + r['turn_ref'])
        elif not isinstance(quote, str) or quote not in t['text']:
            bad.append('NONEXACT_QUOTE:' + r['turn_ref'])
        elif r.get('origin') is not None and r['origin'] != t.get('origin'):
            bad.append('ORIGIN_MISMATCH:' + r['turn_ref'])
    if not refs:
        bad.append('MISSING_EVIDENCE')
    return {'draft_path': str(path), 'draft_sha256': digest(path),
            'candidate_id': c.get('candidate_id'), 'root_id': c['root_id'],
            'status': 'PASS' if not bad else 'HOLD_MECHANICAL', 'question_type': kind,
            'evidence_occurrences_checked': len(refs), 'issues': sorted(set(bad))}


def blind(path):
    c = candidate(path)
    assert c is not None
    code = 'BQ_' + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    # Only explicitly allowlisted fields; no inherited IDs, provenance, Gold,
    # author truth-status, number of correct options, group, target or card.
    q = {k: c[k] for k in ('question_type', 'question', 'options', 'query_scope', 'query_pairs',
                             'answer_semantics') if k in c}
    q['options'] = [{k: o[k] for k in ('option_id', 'text')} for o in c['options']]
    packet = {'blind_id': code, 'source_path': c['source_path'],
              'source_sha256': c['source_sha256'], 'source_turn_count': c['source_turn_count'],
              'question_display': q,
              'read_policy': 'Read complete compact source turns in chunks <=30 turns; inspect each tool result for truncation. Read this packet only; no parent or sibling files.',
              'review_policy': 'First exclusive-create FIRST_ANSWER.json with independently recovered answer and exact evidence, without seeing author Gold. Then write REVIEW.json with quality scope/uniqueness/naturalness/evidence/option findings and actual sufficient solution operation. D1=independent status checks; D2=final version then item matching; D3=direct agreed procedure; D4=source narration order directly yields full event chain. Record shortcuts and non-D positive evidence; structure does not determine quality. Full source text must be visible before first answer is frozen.',
              'limitations': 'Same-model independent context review, not external human review.'}
    dest = HERE / 'review/fresh_blind_packets' / (code + '.json')
    write_new(dest, packet)
    return {'blind_id': code, 'packet_path': str(dest), 'packet_sha256': digest(dest),
            'draft_path': str(path), 'draft_sha256': digest(path),
            'candidate_id': c.get('candidate_id'), 'root_id': c['root_id']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'blind'))
    parser.add_argument('drafts', nargs='+')
    args = parser.parse_args()
    rows = [audit(Path(p)) if args.mode == 'audit' else blind(Path(p)) for p in args.drafts]
    print(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'items': rows}, ensure_ascii=False, indent=2))
