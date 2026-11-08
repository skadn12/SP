from pathlib import Path
import sys
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import HERE,read,write,bind,now

pool=read(HERE/'EVALUATION_POOL.json')['questions']; by={q['candidate_id']:q for q in pool}
metrics={};questions=[]
for q in pool:
    questions.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count':len(q['gold']),'role':'new_child' if len(q['gold'])<6 else 'historical_control','models':{}})
for side in ['trained','base']:
    run=HERE/'solver'/side
    rows=[read(p) for p in sorted((run/'responses').glob('*.json'))]
    results={q['candidate_id']:q for q in read(run/'RESULTS.json')['questions']}
    metrics[side]={}
    for role in ['new_child','historical_control','all_questions']:
        selected=[r for r in rows if role=='all_questions' or (len(by[r['candidate_id']]['gold'])<6)==(role=='new_child')]
        n=len(selected);correct=sum(r['score']['exact_match'] for r in selected)
        metrics[side][role]={'calls':n,'exact_correct':correct,'exact_fraction':str(Fraction(correct,n)) if n else None,'exact_percent':100*correct/n if n else None,'partial_mean':str(sum((Fraction(r['score']['partial_score_fraction']) for r in selected),Fraction())/n) if n else None}
    for q in questions:
        result=results[q['candidate_id']]
        q['models'][side]={k:result[k] for k in ['full_exact_rate','full_partial_mean','selected_by_policy','selection_reason']}
write(HERE/'evidence/RESULT_DETAILS.json',{'created_at_utc':now(),'metrics':metrics,'questions':questions,'selection_rule':'0 < trained mean partial < 1; exact rate is descriptive and does not select questions.','new_content_scope':'Four new six-option combinations from the same six facts and counterversions on root194259; no added roots or independent facts.','report':bind(HERE/'MIXED_EVALUATION_REPORT.json'),'limitations':['Current paired comparison covers these questions only.','Same-family questions are correlated.','Historical controls are not newly generated questions.','No usable ordering questions, hence no ordering Solver evidence.']})
print(metrics)
