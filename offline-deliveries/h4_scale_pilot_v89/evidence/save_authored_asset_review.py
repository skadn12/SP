"""Serialize explicitly authored judgments; this helper does not judge facts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(run, round_id, authored_path):
    folder = run / 'rounds' / round_id
    source = read(run / 'SOURCE.json')
    public = read(folder / 'PUBLIC_ASSETS.json')
    authored = read(authored_path)
    assert authored['root_id'] == source['root_id'] == public['root_id']
    assert authored['public_assets_sha256'] == sha(folder / 'PUBLIC_ASSETS.json')
    assert authored['source_sha256'] == sha(run / 'SOURCE.json')
    assert authored['complete_source_turns_read'] == len(source['turns'])
    turns = {t['turn_ref']: t for t in source['turns']}
    out = {'review_schema_version': 'source_judgment_v1', 'root_id': source['root_id'],
           'family_id': public['family_id'], 'round_id': round_id,
           'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'reviewer': 'Codex root source-first reviewer; parent Gold6 known, current variant private artifacts unread',
           'source_sha256': authored['source_sha256'],
           'public_assets_sha256': authored['public_assets_sha256'],
           'complete_source_turns_read': authored['complete_source_turns_read'],
           'parent_all_true_provenance_known': True, 'parent_gold_known_before_review': True,
           'fresh_blind_parent_review': False, 'current_solver_scores_read': False,
           'solver_scores_used': False, 'new_variant_private_artifacts_read_before_save': False,
           'full_source_read': True, 'derived_from_rounds': False, 'stem_valid': authored['stem_valid'],
           'authored_judgments': {'path': str(authored_path.resolve()), 'sha256': sha(authored_path)}}
    for key, public_key, id_key in [('parents', 'parent_facts', 'fact_id'), ('versions', 'versions', 'version_id')]:
        expected = {r[id_key]: r for r in public[public_key]}
        given = {r[id_key]: r for r in authored[key]}
        assert len(given) == len(authored[key]) and set(given) == set(expected)
        out[key] = []
        for identity, row in given.items():
            assert row['proposition'] == expected[identity]['proposition']
            assert row['relation'] in ('supports', 'contradicts', 'unknown')
            assert type(row['scope_valid']) is bool and type(row['fact_distinct']) is bool
            assert row['reason'].strip() and row['citation_turn_refs']
            citations = [{'turn_ref': ref, 'speaker': turns[ref]['speaker'], 'quote': turns[ref]['text']}
                         for ref in row['citation_turn_refs']]
            out[key].append({id_key: identity, **{k: row[k] for k in
                ('proposition', 'relation', 'scope_valid', 'fact_distinct', 'reason')}, 'citations': citations})
    with (folder / 'SOURCE_JUDGMENTS.json').open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write(chr(10))
    print({'saved': str(folder / 'SOURCE_JUDGMENTS.json'), 'parents': len(out['parents']), 'versions': len(out['versions'])})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('round', choices=['initial', 'supplement'])
    p.add_argument('authored', type=Path)
    a = p.parse_args()
    save(a.run, a.round, a.authored)
