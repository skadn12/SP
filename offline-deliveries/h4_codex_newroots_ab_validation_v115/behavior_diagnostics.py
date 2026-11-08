"""Describe option-selection errors from completed, independently audited runs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def bind(p): return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    audit = read(HERE/'INDEPENDENT_SOLVER_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['errors'] == []
    pool = {q['candidate_id']:q for q in read(HERE/'evaluation/EVALUATION_POOL.json')['questions']}
    sides = {}
    for side in ('base','trained'):
        run = HERE/'evaluation/run'/side
        complete = read(run/'COMPLETE.json')
        assert complete['status'] == 'COMPLETE'
        manifest = read(run/'manifest.json')
        requests = {r['request_id']:r for r in manifest['requests']}
        canonical_gold = read(run/'gold.json')
        assert canonical_gold == read(HERE/'evaluation/run/base/gold.json')
        rows = []
        for p in sorted((run/'responses').glob('*.json')):
            response = read(p)
            if response['question_type'] != 'multi_select': continue
            assert response['evaluation_valid']
            q = pool[response['candidate_id']]
            match = re.fullmatch(r'\s*<box>\s*([^<>]*)\s*</box>\s*',response['raw_output'])
            assert match
            labels = [x.strip().upper() for x in match.group(1).split(',') if x.strip()]
            mapping = requests[response['request_id']]['variant']['display_to_content']
            prediction = {mapping[label] for label in labels}
            gold = set(canonical_gold[q['candidate_id']])
            assert len(gold) == len(q['gold'])
            extras, missing = len(prediction-gold), len(gold-prediction)
            kind = 'extra_and_missing' if extras and missing else 'extra_only' if extras else 'missing_only' if missing else 'exact'
            assert (kind == 'exact') == response['score']['exact_match']
            rows.append({'request_id':response['request_id'],'candidate_id':q['candidate_id'],'gold_count':len(gold),'selected_count':len(prediction),'false_positive_count':extras,'false_negative_count':missing,'error_kind':kind})
        groups = {}
        for name, selected_rows in [
            ('gold2_5',[r for r in rows if r['gold_count'] < 6]),
            ('gold6_control',[r for r in rows if r['gold_count'] == 6]),
            *[(f'gold{k}',[r for r in rows if r['gold_count'] == k]) for k in (2,3,4,5)]]:
            groups[name] = {'calls':len(selected_rows),'error_kinds':dict(Counter(r['error_kind'] for r in selected_rows)),
                'all_six_selected':sum(r['selected_count']==6 for r in selected_rows),
                'more_than_gold_selected':sum(r['selected_count']>r['gold_count'] for r in selected_rows),
                'any_false_positive':sum(r['false_positive_count']>0 for r in selected_rows),
                'any_false_negative':sum(r['false_negative_count']>0 for r in selected_rows),
                'total_false_positives':sum(r['false_positive_count'] for r in selected_rows),
                'total_false_negatives':sum(r['false_negative_count'] for r in selected_rows)}
        sides[side] = {'groups':groups,'responses':rows}
    out = {'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Descriptive option-selection counts on the frozen current batch; repeated permutations/answers and same-root question families are correlated, not independent samples. No new training or selection-rule change.',
        'definitions':{'extra_only':'Selects at least one false option and misses no true option.','missing_only':'Misses at least one true option and selects no false option.','extra_and_missing':'Both errors in the same answer.','all_six_selected':'Selects every option, irrespective of Gold count.'},
        'sides':sides,'input_bindings':[bind(HERE/'INDEPENDENT_SOLVER_AUDIT.json'),bind(HERE/'evaluation/EVALUATION_POOL.json'),bind(Path(__file__))],
        'training_started':False,'pool_merge_performed':False}
    with (HERE/'BEHAVIOR_DIAGNOSTICS.json').open('x') as f:
        json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({side:value['groups'] for side,value in sides.items()},ensure_ascii=False))

if __name__ == '__main__': main()
