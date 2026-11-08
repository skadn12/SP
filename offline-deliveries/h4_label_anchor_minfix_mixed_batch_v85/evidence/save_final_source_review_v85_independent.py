"""Source-only final review serializer; decisions must be supplied by reviewer.

Reads only public questions, current config and complete frozen sources.
Never reads private pool, Gold, assets, rounds, RAW, solver, or other reviews.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def save(decisions):
    home = BASE / 'multiselect'
    config = read(home / 'CONFIG.json')
    public_path = home / 'PUBLIC_QUESTIONS.json'
    public = read(public_path)
    slots = config['slots']
    roots = {s['root_id']: s for s in slots}
    groups = config['final_review_groups']
    assert len(groups) == len(slots) == 4
    assigned = dict(zip((s['root_id'] for s in slots), groups))
    questions = public['questions']
    assert set(decisions) == {q['candidate_id'] for q in questions}
    sources = {r: read(s['source_path']) for r,s in roots.items()}
    out = home / 'final_review'
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for g in groups:
        selected = [q for q in questions if assigned[q['root_id']] == g]
        assert selected
        group_roots = sorted({q['root_id'] for q in selected})
        record = {
            'review_schema_version': 'source_judgment_v1',
            'reviewer': 'v85_final_source_review_independent_agent',
            'group': g,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'review_mode': 'independent_final_complete_source_public_question_review',
            'parent_all_true_provenance_known': True,
            'parent_gold_known_before_review': True,
            'fresh_blind_parent_review': False,
            'current_solver_scores_read': False,
            'solver_scores_used': False,
            'child_gold_read_before_save': False,
            'current_pool_read_before_save': False,
            'new_child_private_artifacts_read': False,
            'asset_reviews_read': False,
            'writer_raw_responses_read': False,
            'round_judgments_read': False,
            'derived_from_rounds': False,
            'other_reviewers_judgments_read': False,
            'target_answer_quotas_used_to_infer_relations': False,
            'parent_review_disclosure': 'Historical parents were known to have all six Gold options; the parent review is not fresh blind. Current child Gold and private materials were not read. Each current public option is judged from complete source, without target quotas.',
            'public_questions': bind(public_path),
            'source': [{'root_id':r, **bind(roots[r]['source_path']), 'complete_turn_count':len(sources[r]['turns'])} for r in group_roots],
            'current_complete_source_read': {r:{'complete':True,'turn_positions_inclusive':[1,len(sources[r]['turns'])]} for r in group_roots},
            'questions': [],
        }
        for q in selected:
            r = q['root_id']
            d = decisions[q['candidate_id']]
            assert set(d['options']) == set('ABCDEF')
            row = {
                'candidate_id':q['candidate_id'], 'root_id':r,
                'source_sha256':q['source_sha256'], 'question':q['question'],
                'complete_source_turns_read':len(sources[r]['turns']),
                'stem_valid':d['stem_valid'],
                'stem_reason':d['stem_reason'],
                'options':{},
            }
            turns = {t['turn_ref']:t for t in sources[r]['turns']}
            for label in 'ABCDEF':
                option = dict(d['options'][label])
                assert option['relation'] in ('supports','contradicts','unknown')
                assert option.get('reason')
                option['proposition'] = q['options'][label]
                citations = []
                for ref,quote in option.pop('citation_specs'):
                    t = turns[ref]
                    assert quote and quote in t['text'], (q['candidate_id'], label, ref, quote)
                    citations.append({'turn_ref':ref,'speaker':t['speaker'],'quote':quote})
                assert citations
                option['citations'] = citations
                row['options'][label] = option
            record['questions'].append(row)
        path = out / f'group_{g}_SOURCE_JUDGMENTS.json'
        assert not path.exists(), f'Existing final review must not be silently overwritten: {path}'
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        paths.append(bind(path))
    return paths
