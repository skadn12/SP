from pathlib import Path
from fractions import Fraction
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now
pool=read(HOME/'EVALUATION_POOL.json')['questions']
report=read(HOME/'MIXED_EVALUATION_REPORT.json')
pairs=read(HOME/'PAIRED_CALLS.json')['calls']
metrics={};questions=[]
for q in pool:
    questions.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],
      'question_type':'ordering','gold_count':6,'role':'new_ordering','models':{}})
for side in ['trained','base']:
    metrics[side]={}
    for role in ['new_child','historical_control','all_questions']:
        rows=[] if role=='historical_control' else [r[side] for r in pairs]
        n=len(rows);correct=sum(r['exact'] for r in rows)
        metrics[side][role]={'calls':n,'exact_correct':correct,
          'exact_fraction':str(Fraction(correct,n)) if n else None,
          'exact_percent':100*correct/n if n else None,
          'partial_mean':str(sum((Fraction(r['partial']) for r in rows),Fraction())/n) if n else None}
    results={q['candidate_id']:q for q in read(HOME/'solver'/side/'RESULTS.json')['questions']}
    for q in questions:
        q['models'][side]={k:results[q['candidate_id']][k] for k in ['full_exact_rate','full_partial_mean','selected_by_policy','selection_reason']}
write(HOME/'evidence/RESULT_DETAILS.json',{'created_at_utc':now(),'report':bind(HOME/'MIXED_EVALUATION_REPORT.json'),
 'metrics':metrics,'questions':questions,'new_selected_count':report['selected_new_candidate_count'],
 'selection_rule':'all answers valid and 0 < trained mean partial < 1',
 'limitations':['These are three deliberately repaired prior failures, not a random generation sample.',
  'No v81 ordering Solver results exist; no before/after ordering accuracy comparison is possible.',
  'The legacy new_child aggregate key means new candidates here; all current questions are ordering.',
  'Repeated fact families and two sources do not establish independent new facts or broad reliability.'],
 'training_started':False,'formal_training_admission':False})
print({'new_selected':report['selected_new_candidate_count'],'metrics':metrics})
