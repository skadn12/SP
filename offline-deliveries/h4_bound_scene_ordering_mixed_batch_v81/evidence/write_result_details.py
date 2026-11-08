"""Summarize actual new/control outcomes and preserve every failure category."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now
pool=read(HOME/'EVALUATION_POOL.json')['questions']
byid={q['candidate_id']:q for q in pool}
report=read(HOME/'MIXED_EVALUATION_REPORT.json')
pairs=read(HOME/'PAIRED_CALLS.json')['calls']
newids={q['candidate_id'] for q in pool if q['question_type']=='ordering' or len(q['gold'])<6}
selected=set(report['selected_candidate_ids'])
def accuracy(rows):
 n=len(rows);correct=sum(r['exact'] for r in rows)
 return {'correct':correct,'calls':n,'accuracy_fraction':str(Fraction(correct,n)) if n else None,
  'accuracy_percent':round(100*correct/n,4) if n else None,
  'all_six_selected_calls':sum(r['selected_count']==6 for r in rows),
  'all_six_selected_correct':sum(r['selected_count']==6 and r['exact'] for r in rows)}
perquestion=[]
for q in pool:
 rows=[r for r in pairs if r['candidate_id']==q['candidate_id']]
 perquestion.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],
  'gold_count':len(q['gold']),'role':('new_ordering' if q['question_type']=='ordering' else 'new_child') if q['candidate_id'] in newids else 'historical_control',
  'selected':q['candidate_id'] in selected,
  'models':{side:accuracy([r[side] for r in rows]) for side in ['base','trained']}})
groups={}
for role in ['new_children','historical_controls','all']:
 chosen=[r for r in pairs if role=='all' or ((r['candidate_id'] in newids)==(role=='new_children'))]
 groups[role]={side:accuracy([r[side] for r in chosen]) for side in ['base','trained']}
ordering_failures=[]
for p in sorted((HOME/'ordering').glob('*/REJECTIONS.json')):
 for r in read(p):ordering_failures.append({**r,'record':bind(p)})
rounds=[]
for p in sorted((HOME/'multiselect/assets').glob('*/rounds/*/ASSET_REVIEW.json')):
 a=read(p);rounds.append({'family_id':a['family_id'],'round_id':a['round_id'],
  'usable_parents':sum(v['usable'] for v in a['parents']),
  'usable_contradictions':sum(v['usable'] for v in a['versions']),
  'rejected_versions':[v for v in a['versions'] if not v['usable']],
  'record':bind(p)})
prior=HOME.parent/'h4_scoped_ordering_mixed_batch_v80/MIXED_EVALUATION_REPORT.json'
details={'created_at_utc':now(),'status':'ACTUAL_BATCH_RESULTS','content_usable_new_count':len(newids),
 'new_selected_count':len(newids&selected),
 'content_usable_new_by_multi_gold':{str(g):sum(q['candidate_id'] in newids and q['question_type']=='multi_select' and len(q['gold'])==g for q in pool) for g in range(2,6)},
 'selected_new_by_multi_gold':{str(g):sum(q['candidate_id'] in newids&selected and q['question_type']=='multi_select' and len(q['gold'])==g for q in pool) for g in range(2,6)},
 'aggregate':groups,'per_question':perquestion,'ordering_generation_failures':ordering_failures,
 'multiselect_asset_round_reviews':rounds,'allocation':bind(HOME/'multiselect/ALLOCATION.json'),
 'prior_v80_report':bind(prior),'prior_v80_by_type':read(prior)['by_type'],
 'limitations':['Current and prior batches contain different question mixtures, so between-batch accuracy is not a controlled difficulty change estimate.',
  'Within-batch base/trained use identical prompts, permutations, seeds, decoder and scoring.',
  'Historically known all-true parent provenance is disclosed; actual current source-review disclosures are bound in the freeze.',
  'Ordering conclusions are limited to the actual reviewed and evaluated candidate count.',
  'These questions share source roots and existing parent facts; split by root/family if later admitted.'],
 'training_allowed':False,'formal_training_admission':False}
metrics={};questions=[]
for q in pool:
 questions.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],
  'question_type':q['question_type'],'gold_count':len(q['gold']),
  'role':('historical_control' if q['candidate_id'] not in newids else
          ('new_ordering' if q['question_type']=='ordering' else 'new_child')),'models':{}})
for side in ['trained','base']:
 metrics[side]={}
 for role in ['new_child','historical_control','all_questions']:
  rows=[r[side] for r in pairs if role=='all_questions' or ((r['candidate_id'] in newids)==(role=='new_child'))]
  n=len(rows);correct=sum(r['exact'] for r in rows)
  metrics[side][role]={'calls':n,'exact_correct':correct,
   'exact_fraction':str(Fraction(correct,n)) if n else None,
   'exact_percent':100*correct/n if n else None,
   'partial_mean':str(sum((Fraction(r['partial']) for r in rows),Fraction())/n) if n else None}
 results={q['candidate_id']:q for q in read(HOME/'solver'/side/'RESULTS.json')['questions']}
 for q in questions:
  q['models'][side]={k:results[q['candidate_id']][k] for k in ['full_exact_rate','full_partial_mean','selected_by_policy','selection_reason']}
details.update(metrics=metrics,questions=questions,report=bind(HOME/'MIXED_EVALUATION_REPORT.json'),
 selection_rule='0 < trained mean partial < 1; exact accuracy is descriptive and does not select questions.')
write(HOME/'evidence/RESULT_DETAILS.json',details)
print({'new_content':len(newids),'new_selected':len(newids&selected),'aggregate':groups})
