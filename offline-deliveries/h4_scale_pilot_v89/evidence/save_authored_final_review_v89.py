"""Serialize explicit whole-question source judgments, without deriving semantics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(home, group, authored_path):
    public_path = home / 'multiselect/PUBLIC_QUESTIONS.json'
    authored = read(authored_path)
    assert authored['group'] == group
    assert authored['public_questions_sha256'] == sha(public_path)
    public = {q['candidate_id']: q for q in read(public_path)['questions']}
    sources = {}
    rows = []
    for row in authored['questions']:
        q = public[row['candidate_id']]
        assert q['root_id'] == authored['root_id']
        path = Path(q['source_path'])
        source = read(path)
        assert sha(path) == q['source_sha256'] == authored['source_sha256']
        assert authored['complete_source_turns_read'] == len(source['turns'])
        sources[q['root_id']] = {'root_id': q['root_id'], 'path': str(path.resolve()),
                                 'sha256': sha(path), 'complete_turn_count': len(source['turns'])}
        turns = {t['turn_ref']: t for t in source['turns']}
        assert set(row['options']) == set(q['options'])
        options = {}
        for label, judgment in row['options'].items():
            assert judgment['proposition'] == q['options'][label]
            assert judgment['relation'] in ('supports', 'contradicts', 'unknown')
            assert type(judgment['scope_valid']) is bool and type(judgment['fact_distinct']) is bool
            assert judgment['reason'].strip() and judgment['citation_turn_refs']
            options[label] = {k: judgment[k] for k in
                              ('proposition', 'relation', 'scope_valid', 'fact_distinct', 'reason')}
            options[label]['citations'] = [
                {'turn_ref': ref, 'speaker': turns[ref]['speaker'], 'quote': turns[ref]['text']}
                for ref in judgment['citation_turn_refs']]
        assert type(row['stem_valid']) is bool
        rows.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                     'source_sha256': q['source_sha256'], 'question': q['question'],
                     'complete_source_turns_read': len(source['turns']),
                     'stem_valid': row['stem_valid'], 'options': options})
    assert len({r['candidate_id'] for r in rows}) == len(rows)
    expected = {q['candidate_id'] for q in public.values() if q['root_id'] == authored['root_id']}
    assert {r['candidate_id'] for r in rows} == expected
    record = {'review_schema_version': 'source_judgment_v1', 'group': group,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer': 'Codex root; independent complete-source and whole-question review; prior parent and round crosschecks known',
              'parent_all_true_provenance_known': True, 'parent_gold_known_before_review': True,
              'fresh_blind_parent_review': False, 'current_solver_scores_read': False,
              'solver_scores_used': False, 'child_gold_read_before_save': False,
              'current_pool_read_before_save': False, 'new_child_private_artifacts_read': False,
              'writer_raw_responses_read': False,
              'previous_round_private_crosschecks_known': True,
              'semantic_judgments_derived_from_round_records': False,
              'public_questions': {'path': str(public_path.resolve()), 'sha256': sha(public_path)},
              'source': list(sources.values()),
              'current_complete_source_read': {
                  root: {'complete': True, 'turn_positions_inclusive': [1, item['complete_turn_count']]}
                  for root, item in sources.items()},
              'authored_judgments': {'path': str(authored_path.resolve()), 'sha256': sha(authored_path)},
              'questions': rows}
    output = home / 'multiselect/final_review' / f'group_{group}_SOURCE_JUDGMENTS.json'
    with output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'saved': str(output), 'whole_questions': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    parser.add_argument('group')
    parser.add_argument('authored', type=Path)
    args = parser.parse_args()
    save(args.home, args.group, args.authored)
