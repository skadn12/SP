#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
from collections import Counter
import json,itertools
P=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def main():
 items=read(P/'ITEMS_PRIVATE.json');audit=read(P/'SCORE_RECOMPUTATION_AUDIT.json');requests={x['request_id']:x for x in read(P/'solver/REQUESTS.json')};out=[]
 for c in items:
  modes={}
  for mode in ['full_context','no_context']:
   rows=[x for x in audit['rows'] if x['candidate_id']==c['candidate_id'] and x['context_mode']==mode]
   patterns=[tuple(sorted(x['prediction_option_ids'])) if c['question_type']=='multi_select' else tuple(x['prediction_option_ids']) for x in rows]
   counts=Counter(patterns);fp=Counter();fn=Counter();supported_final_conflicts=0
   for row in rows:
    if not row['valid']:continue
    fp.update(set(row['prediction_option_ids'])-set(c['gold_option_ids']));fn.update(set(c['gold_option_ids'])-set(row['prediction_option_ids']))
   pairs=list(itertools.combinations(patterns,2));consistency=sum(a==b for a,b in pairs)/len(pairs) if pairs else None
   modes[mode]={'calls':len(rows),'valid':sum(x['valid'] for x in rows),'exact':sum(x['exact_correct'] for x in rows),'mean_J_fraction':str(sum((Fraction(x['J_fraction']) for x in rows),Fraction(0))/len(rows)) if rows else None,'unique_semantic_answers':len(counts),'modal_answer_share':max(counts.values())/len(rows) if rows else None,'pairwise_answer_agreement':consistency,'answer_distribution':[{'option_ids':list(k),'count':v} for k,v in counts.most_common()],'FP_by_option_id':dict(fp),'FN_by_option_id':dict(fn),'permutation_accuracy':{v:sum(x['exact_correct'] for x in rows if requests[x['request_id']]['variant_id']==v)/sum(requests[x['request_id']]['variant_id']==v for x in rows) for v in sorted({requests[x['request_id']]['variant_id'] for x in rows})}}
  out.append({'candidate_id':c['candidate_id'],'review_id':c['review_id'],'arm':c['arm'],'root_id':c['root_id'],'modes':modes})
 with (P/'CONSISTENCY_AND_SELECTION_ERRORS.json').open('x') as f:json.dump({'unit':'semantic answers; repeated calls are not independent source questions','items':out},f,ensure_ascii=False,indent=2)
 print('wrote per-item consistency and FP/FN counts')
if __name__=='__main__':main()
