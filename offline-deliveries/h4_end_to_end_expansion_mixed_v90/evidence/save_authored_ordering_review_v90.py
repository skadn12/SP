"""Serialize source-only authored ordering judgments; infer no semantics or Gold."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DECLARATIONS = {
    'parent_all_true_provenance_known': False,
    'parent_gold_known_before_review': False,
    'fresh_blind_parent_review': False,
    'current_solver_scores_read': False,
    'solver_scores_used': False,
    'full_source_read': True,
    'new_variant_private_artifacts_read_before_save': False,
    'child_gold_read_before_save': False,
    'current_pool_read_before_save': False,
}
IDENTITY = ('candidate_id', 'root_id', 'question_type', 'chronology_kind',
            'question', 'source_path', 'source_sha256')
ITEM_FLAGS = ('supported', 'actor_ok', 'modality_ok', 'state_ok', 'neutral_stem_ok')
QUESTION_FLAGS = ('coherent', 'unique_order', 'neutral_stem_ok')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(home, authored_path):
    public_path = home / 'ordering/PUBLIC_QUESTIONS.json'
    authored = read(authored_path)
    assert authored['public_questions'] == bind(public_path)
    assert all(type(authored[k]) is bool and authored[k] is v for k, v in DECLARATIONS.items())
    public = {q['candidate_id']: q for q in read(public_path)['questions']}
    assert len(authored['questions']) == len(public)
    assert {q['candidate_id'] for q in authored['questions']} == set(public)
    sources = {}
    rows = []
    for row in authored['questions']:
        q = public[row['candidate_id']]
        source = read(q['source_path'])
        assert bind(q['source_path'])['sha256'] == q['source_sha256']
        assert source['root_id'] == q['root_id']
        assert row['complete_source_turns_read'] == len(source['turns'])
        assert all(type(row[k]) is bool for k in QUESTION_FLAGS)
        turns = {t['turn_ref']: t for t in source['turns']}
        sources[q['root_id']] = {**bind(q['source_path']), 'root_id': q['root_id'],
                                 'complete_turn_count': len(source['turns'])}
        items = {}
        assert set(row['items']) == set(q['items'])
        for label, item in row['items'].items():
            assert item['proposition'] == q['items'][label]
            assert all(type(item[k]) is bool for k in ITEM_FLAGS)
            assert item['reason'].strip() and item['support_refs']
            assert set(item['support_refs']) <= set(turns)
            def anchor(ref):
                assert ref in turns
                return {'turn_ref': ref, 'quote': turns[ref]['text']}
            items[label] = {k: item[k] for k in ('proposition', 'reason', 'support_refs', *ITEM_FLAGS)}
            items[label]['order_anchor'] = anchor(item['primary_turn_ref'])
            items[label]['alternative_matching_anchors'] = [anchor(ref) for ref in item['alternative_turn_refs']]
        rows.append({**{k: q[k] for k in IDENTITY},
                     **{k: row[k] for k in (*QUESTION_FLAGS, 'alternative_orders', 'reason',
                                          'complete_source_turns_read')}, 'items': items})
    record = {'review_schema_version': 'source_judgment_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer': authored['reviewer'], **DECLARATIONS,
              'public_questions': bind(public_path), 'sources': list(sources.values()),
              'authored_judgments': bind(authored_path), 'questions': rows}
    path = home / 'ordering/SOURCE_JUDGMENTS.json'
    with path.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'saved': str(path), 'questions': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    parser.add_argument('authored', type=Path)
    args = parser.parse_args()
    save(args.home, args.authored)
