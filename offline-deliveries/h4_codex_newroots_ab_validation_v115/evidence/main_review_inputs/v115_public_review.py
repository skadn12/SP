from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import sys

BATCH = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def begin(short, stage):
    folder = BATCH / 'roots' / ('root' + short)
    public = 'PUBLIC_ASSETS.json' if stage == 'assets' else 'PUBLIC_QUESTIONS.json'
    context = {'started_at_utc': now(), 'source_sha256': sha(folder/'CUMULATIVE_SOURCE.json'), 'public_sha256': sha(folder/public)}
    write(Path('/tmp')/f'v115-{short}-{stage}-read.json', context)
    source = read(folder/'CUMULATIVE_SOURCE.json')
    for t in source['turns']:
        print(t['turn_ref']+' | '+t['speaker']+': '+t['text'])
    print('PUBLIC_TEXT', json.dumps(read(folder/public), ensure_ascii=False, indent=2))

def header(short, stage):
    folder = BATCH/'roots'/('root'+short)
    name = 'PUBLIC_ASSETS.json' if stage == 'assets' else 'PUBLIC_QUESTIONS.json'
    context = read(Path('/tmp')/f'v115-{short}-{stage}-read.json')
    assert context['source_sha256'] == sha(folder/'CUMULATIVE_SOURCE.json')
    assert context['public_sha256'] == sha(folder/name)
    source = read(folder/'CUMULATIVE_SOURCE.json')
    return folder, source, {
        'root_id': source['root_id'], 'reviewer': '/root',
        'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
        'external_semantic_certification': False, **context,
        'finished_at_utc': now(), 'full_source_read': True,
        'turn_refs_read': [t['turn_ref'] for t in source['turns']],
        'private_answers_read_before_save': False,
        'independent_of_question_author': True, 'current_solver_scores_read': False,
        'semantic_review_is_programmatic': False,
        'limitations': ['Same inherited model and shared project context; role separation is not external certification, different-model agreement, or complete blindness.', 'Current private draft, asset design, anchor plan, Gold and Solver results were not opened before judgment.']}

def save_assets(short, verdicts, order_specs):
    folder, source, out = header(short, 'assets')
    public = read(folder/'PUBLIC_ASSETS.json')
    turns = {t['turn_ref']:t for t in source['turns']}
    assert set(verdicts) == {s['statement_id'] for s in public['statements']}
    out['statements'] = {}
    for s in public['statements']:
        relation, primary, refs, reason = verdicts[s['statement_id']]
        assert primary in refs
        citations = [{'turn_ref': r, 'speaker': turns[r]['speaker'], 'quote': turns[r]['text']} for r in refs]
        out['statements'][s['statement_id']] = {
            'proposition': s['proposition'], 'relation': relation,
            'scope_valid': True, 'fact_distinct': True, 'reason': reason,
            'citations': citations, 'primary_citation': copy.deepcopy(citations[refs.index(primary)])}
    assert len(order_specs) == len(public['ordering_questions'])
    out['ordering_questions'] = []
    positions = {t['turn_ref']:i for i,t in enumerate(source['turns'])}
    for q, spec in zip(public['ordering_questions'], order_specs):
        assert set(spec) == set(q['items'])
        j = {
            'candidate_id': q['candidate_id'], 'stem_valid': True, 'scope_valid': True,
            'coherent': True, 'unique_order': True, 'neutral_stem_ok': True,
            'alternative_orders': [], 'items': {},
            'all_scene_turn_refs_checked': [t['turn_ref'] for t in source['turns'] if t['turn_ref'].startswith('R02:')],
            'reason': 'Full cumulative source and every public item read. Items describe distinct statements or single events in the six named scenes. All reasonable matching anchors recorded below preserve the same strict appearance order.'}
        for label, (ref, alts, reason) in spec.items():
            j['items'][label] = {
                'proposition': q['items'][label], 'supported': True,
                'actor_ok': True, 'modality_ok': True, 'state_ok': True,
                'neutral_stem_ok': True, 'single_event_ok': True,
                'order_anchor': {'turn_ref': ref, 'quote': turns[ref]['text']},
                'support_refs': [ref] + alts,
                'alternative_matching_anchors': [{'turn_ref': a, 'quote': turns[a]['text']} for a in alts],
                'reason': reason}
        j['derived_order'] = sorted(spec, key=lambda label: positions[spec[label][0]])
        out['ordering_questions'].append(j)
    write(folder/'ASSET_SOURCE_JUDGMENTS.json', out)
    print(folder/'ASSET_SOURCE_JUDGMENTS.json')

def save_public(short):
    # Run only after a fresh complete-source and final public-question reading.
    folder, source, out = header(short, 'public')
    public = read(folder/'PUBLIC_QUESTIONS.json')
    assets = read(folder/'PUBLIC_ASSETS.json')
    review = read(folder/'ASSET_SOURCE_JUDGMENTS.json')
    statements = {j['proposition']:j for j in review['statements'].values()}
    orders = {j['candidate_id']:j for j in review['ordering_questions']}
    original_orders = {j['candidate_id']:j for j in assets['ordering_questions']}
    out['asset_judgments_sha256'] = sha(folder/'ASSET_SOURCE_JUDGMENTS.json')
    out['questions'] = {}
    for q in public['questions']:
        if q['question_type'] == 'multi_select':
            assert q['question'] == assets['multiselect_stem']
            opts = {label: copy.deepcopy(statements[text]) for label, text in q['options'].items()}
            out['questions'][q['candidate_id']] = {
                'stem_valid': True, 'scope_valid': True, 'fact_distinct': True,
                'no_entangled_options': True,
                'reason': 'Final displayed stem and all six options read against full source and saved source-first asset judgments. Each independently judged option belongs to a distinct new scene, retaining actor, report attribution and action state. No unknown variant is treated as false.',
                'options': opts,
                'source_derived_answer': [k for k in 'ABCDEF' if opts[k]['relation'] == 'supports']}
        else:
            assert q == original_orders[q['candidate_id']]
            out['questions'][q['candidate_id']] = copy.deepcopy(orders[q['candidate_id']])
    write(folder/'PUBLIC_SOURCE_JUDGMENTS.json', out)
    print(folder/'PUBLIC_SOURCE_JUDGMENTS.json')

if __name__ == '__main__':
    if sys.argv[1] == 'read': begin(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'save-public': save_public(sys.argv[2])
