"""Read-only derived diagnostics; repeated permutations never increase question n."""
from pathlib import Path
import collections
import hashlib
import json
import statistics
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution_v2'

def read(path):
    return json.loads(path.read_text())

def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    cs = {c['candidate_id']: c for c in read(HERE / 'pilot/CANDIDATES_FINAL.json')}
    private = {r['request_id']: r for r in read(OUT / 'PRIVATE_SCORING.json')['requests']}
    answer_only = '--answer-only' in sys.argv
    score_path = OUT / ('ANSWER_ONLY_SCORES.json' if answer_only else 'SCORES.json')
    scores = read(score_path)['records']
    rows = []
    for r in scores:
        if r['cohort'] != 'new':
            continue
        c = cs[r['candidate_id']]
        target = c['dependency_certificate']['target_option']
        mapping = private[r['request_id']]['presentation']['display_to_canonical']
        label = next(k for k, v in mapping.items() if v == target)
        parsed = r.get('parsed')
        judgment = parsed['item_evidence'][label]['judgment'] if r['valid'] else None
        rows.append({k: r[k] for k in ('request_id', 'candidate_id', 'root_id', 'condition', 'variant_id', 'valid', 'exact', 'J')} |
                    {'target_option': target, 'target_selected': target in r['prediction'] if r['valid'] else None,
                     'target_judgment': judgment})
    summaries = []
    for cond in sorted({r['condition'] for r in rows}):
        xs = [r for r in rows if r['condition'] == cond]
        valid = [r for r in xs if r['valid']]
        summaries.append({'condition': cond, 'questions': len({r['candidate_id'] for r in xs}),
                          'requests': len(xs), 'valid': len(valid),
                          'target_selected': sum(r['target_selected'] for r in valid),
                          'target_judgments': dict(collections.Counter(r['target_judgment'] for r in valid))})
    by_key = {(r['candidate_id'], r['variant_id'], r['condition']): r for r in rows}
    paired = []
    for c in cs.values():
        for index in range(len(read(HERE / 'PILOT_PLAN.json')['display_seeds'])):
            variant = f'perm_{index:02}'
            full = by_key[(c['candidate_id'], variant, 'FULL')]
            for cond in sorted({r['condition'] for r in rows if r['candidate_id'] == c['candidate_id']} - {'FULL'}):
                r = by_key[(c['candidate_id'], variant, cond)]
                paired.append({'candidate_id': c['candidate_id'], 'variant_id': variant, 'condition': cond,
                               'full_exact': full['exact'], 'altered_exact': r['exact'],
                               'full_target_selected': full['target_selected'], 'altered_target_selected': r['target_selected'],
                               'full_target_judgment': full['target_judgment'], 'altered_target_judgment': r['target_judgment']})
    result = {'score_basis': 'answer_only_post_hoc' if answer_only else 'strict',
              'inputs': [binding(score_path), binding(OUT / 'PRIVATE_SCORING.json'),
                         binding(HERE / 'pilot/CANDIDATES_FINAL.json')],
              'records': rows, 'summaries': summaries, 'paired': paired,
              'interpretation': 'All six certified targets are true in FULL. Selection on reduced contexts measures retention of FULL targets, not correctness under a new reduced-context Gold. Underdetermination is independently reviewed. Target guesses, low exact scores and ablation drops alone do not prove logical necessity.'}
    behavior = []
    for cohort in ('baseline', 'new'):
        for condition in sorted({r['condition'] for r in scores if r['cohort'] == cohort}):
            xs = [r for r in scores if r['cohort'] == cohort and r['condition'] == condition and r['valid']]
            behavior.append({'cohort': cohort, 'condition': condition, 'valid': len(xs),
                             'empty_predictions': sum(not r['prediction'] for r in xs),
                             'mean_selected': statistics.mean(len(r['prediction']) for r in xs) if xs else None})
    result['answer_behavior'] = behavior
    filename = 'DEPENDENCY_DIAGNOSTICS_ANSWER_ONLY.json' if answer_only else 'DEPENDENCY_DIAGNOSTICS.json'
    (OUT / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summaries': summaries}, ensure_ascii=False))

if __name__ == '__main__':
    main()
