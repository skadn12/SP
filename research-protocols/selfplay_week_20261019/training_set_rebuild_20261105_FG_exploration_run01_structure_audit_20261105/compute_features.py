"""Read sealed drafts; write separate descriptive structural-audit artifacts."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / 'training_set_rebuild_20261105_FG_exploration_run01'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def counts(values):
    return dict(Counter(values).most_common())


seal = read(RUN / 'EXPLORATION_SEAL.json')
bad = [x['path'] for x in seal['files']
       if digest(RUN / x['path']) != x['sha256']]
assert not bad, bad
cfg = read(RUN / 'RUN_CONFIG.json')
old = cfg['sealed_135_training_export']
assert digest(old['path']) == old['sha256']
source_cache = {}
items = []
bindings = []
for p in sorted(RUN.glob('authoring/*/*/CANDIDATES_FIRST_DRAFT.json')):
    doc = read(p)
    bindings.append({'path': str(p), 'sha256': digest(p)})
    for c in doc['candidates']:
        arm = c.get('arm', doc['arm'])
        source_path = c['source_path']
        if source_path not in source_cache:
            src = read(source_path)
            source_cache[source_path] = {
                t['turn_ref']: (j, t['text']) for j, t in enumerate(src['turns'])
            }
        turns = source_cache[source_path]
        opts = c['options']
        gold = set(c['gold_option_ids'])
        evidence = c['evidence']
        refs = {e['turn_ref'] for e in evidence}
        scenes = sorted({re.sub(r':T\d+$', '', ref) for ref in refs})
        pos = [j + 1 for j, o in enumerate(opts) if o['option_id'] in gold]
        row = {
            'candidate_id': c['candidate_id'], 'root_id': c['root_id'], 'arm': arm,
            'question_type': c['question_type'], 'option_count': len(opts),
            'gold_cardinality': len(c.get('gold_pairs', [])) if c['question_type'] == 'partial_order' else len(gold),
            'first_draft_correct_positions': pos if c['question_type'] == 'multi_select' else None,
            'declared_evidence_turn_count': len(refs),
            'declared_evidence_scene_count': len(scenes),
            'declared_evidence_scenes': scenes,
            'select_all_instruction': bool(re.search(r'select all|all that apply|请选择所有|多选', c['question'], re.I)),
            'authored_macrostructure': c.get('macrostructure', c.get('relation_plan', {}).get('macrostructure')),
        }
        if c['question_type'] == 'multi_select':
            audits = {a['option_id']: a for a in c['option_audit']}
            support_counts = [len({e['turn_ref'] for e in audits[o['option_id']]['evidence']})
                              for o in opts if o['option_id'] in gold]
            row['positive_option_declared_evidence_turn_counts'] = support_counts
            row['all_positive_options_cite_at_most_one_turn_each'] = all(n <= 1 for n in support_counts)
        if c['question_type'] == 'ordering':
            anchors = {}
            for a in c['option_audit']:
                citations = a['evidence']
                positions = [(turns[e['turn_ref']][0], turns[e['turn_ref']][1].find(e['quote'])) for e in citations]
                anchors[a['option_id']] = min(positions) if positions else None
            goldanchors = [anchors[o] for o in c['gold_option_ids']]
            row['gold_monotone_by_earliest_declared_option_evidence'] = (
                all(a is not None for a in goldanchors) and goldanchors == sorted(goldanchors)
            )
            row['earliest_evidence_anchor_ties'] = len(goldanchors) != len(set(goldanchors))
            row['option_text_order_cue_ids'] = [
                o['option_id'] for o in opts if re.search(
                    r'\b(first|then|next|finally|before|after|again|initial|later|earlier|revised|corrected|final)\b|先|随后|之后|再次|最终|更正',
                    o['text'], re.I)
            ]
        items.append(row)
assert len(items) == 140 and len({x['candidate_id'] for x in items}) == 140

summary = {}
for arm in ['ALL', 'F', 'G']:
    group = [x for x in items if arm == 'ALL' or x['arm'] == arm]
    mult = [x for x in group if x['question_type'] == 'multi_select']
    ordering = [x for x in group if x['question_type'] == 'ordering']
    summary[arm] = {
        'items': len(group),
        'types': counts(x['question_type'] for x in group),
        'multi_select_option_count_gold_count': counts(f"{x['option_count']} options/{x['gold_cardinality']} correct" for x in mult),
        'multi_select_first_draft_position_patterns': counts(f"n={x['option_count']};correct={','.join(map(str,x['first_draft_correct_positions']))}" for x in mult),
        'ordering_option_counts': counts(str(x['option_count']) for x in ordering),
        'declared_evidence_scene_counts': counts(str(x['declared_evidence_scene_count']) for x in group),
        'multi_select_all_positives_cite_one_turn_each': sum(x['all_positive_options_cite_at_most_one_turn_each'] for x in mult),
        'multi_select_positive_options_total': sum(len(x['positive_option_declared_evidence_turn_counts']) for x in mult),
        'multi_select_positive_options_citing_one_turn': sum(n <= 1 for x in mult for n in x['positive_option_declared_evidence_turn_counts']),
        'ordering_gold_monotone_by_earliest_declared_evidence': sum(x['gold_monotone_by_earliest_declared_option_evidence'] for x in ordering),
        'ordering_with_lexical_order_cues_in_options': sum(bool(x['option_text_order_cue_ids']) for x in ordering),
    }

byroot = defaultdict(dict)
for c in items:
    byroot[c['root_id']][c['arm']] = c
summary['paired_types'] = counts(f"F:{p['F']['question_type']} G:{p['G']['question_type']}" for p in byroot.values())
summary['interpretation_limits'] = [
    'No Solver calls: lexical clues and positional patterns are descriptive, not measured shortcut success.',
    'First-draft position patterns do not describe future shuffled evaluation displays.',
    'Citation counts describe author evidence footprints, not proven minimal sufficient evidence or cognitive difficulty.',
    'Independent human coding of operations is in separate review records; authored macrostructure is not evidence of diversity.',
]
write('AUDIT_INPUT_FREEZE.json', {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'run_path': str(RUN), 'run_seal_sha256': digest(RUN / 'EXPLORATION_SEAL.json'),
    'run_sealed_files_verified': len(seal['files']), 'run_seal_mismatches': 0,
    'candidate_count': len(items), 'draft_files': bindings,
    'sealed_135_path': old['path'], 'sealed_135_sha256': old['sha256'],
    'codebook_sha256': digest(OUT / 'CODEBOOK.md'),
    'structural_audit_is_not_blind_quality_review': True,
})
write('ITEM_FEATURES.json', {'items': items})
write('QUANTITATIVE_STATS.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
