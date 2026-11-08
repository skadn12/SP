"""Export the actual audited selection and cumulative candidate inventory."""
from collections import Counter
from pathlib import Path
import sys
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now

for name in ['RESULT_AUDIT.json','INPUT_AUDIT.json']:
 assert read(HOME/'evidence'/name)['status']=='PASS',name
generation=read(HOME/'evidence/GENERATION_AUDIT.json')
assert generation['status'] in ('PASS','PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS')
selected=read(HOME/'SELECTED_CANDIDATES.json')['questions']
new=[q for q in selected if q['question_type']=='ordering' or len(q['gold'])<6]
controls=[q for q in selected if q not in new]
def counts(rows):
 return {'by_type':dict(Counter(q['question_type'] for q in rows)),
  'type_distribution':dict(Counter(q['question_type'] for q in rows)),
  'multiselect_gold_distribution':{str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in rows) for g in range(2,6)}}
write(HOME/'AUDITED_NEW_CANDIDATES.json',{
 'created_at_utc':now(),'status':'CONTENT_REVIEW_SOLVER_SELECTION_AND_INDEPENDENT_AUDIT_PASS',
 'questions':new,'new_question_count':len(new),**counts(new),
 'gold_distribution':{str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in new) for g in range(2,6)},
 'role':'new_whole_question_combinations_from_existing_facts',
 'content_review':bind(HOME/'BATCH_REVIEW.COMPLETE.json'),
 'paired_report':bind(HOME/'MIXED_EVALUATION_REPORT.json'),
 'independent_result_audit':bind(HOME/'evidence/RESULT_AUDIT.json'),
 'independent_generation_audit':bind(HOME/'evidence/GENERATION_AUDIT.json'),
 'historical_controls_excluded':[q['candidate_id'] for q in controls],
 'training_allowed':False,'training_started':False,'formal_training_admission':False,'merged_into_baseline':False,
 'review_limitation':'Reviewers save actual source judgments before current private Gold; historically known parent provenance and any prior exposure remain disclosed in bound review records.'})
previous=HOME.parent/'h4_scoped_ordering_mixed_batch_v80'
assert read(previous/'evidence/RESULT_AUDIT.json')['status']=='PASS'
assert read(previous/'evidence/UNION_AUDIT.json')['status']=='PASS'
pools=[previous/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',HOME/'AUDITED_NEW_CANDIDATES.json']
all_rows=[];members=[];seen={};duplicates=[]
for source in pools:
 data=read(source);assert data['training_allowed'] is False
 for q in data['questions']:
  assert q['question_type']=='ordering' or len(q['gold'])<6
  field='options' if q['question_type']=='multi_select' else 'items'
  key=(q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values())))
  if key in seen:
   duplicates.append({'candidate_id':q['candidate_id'],'matches':seen[key]});continue
  seen[key]=q['candidate_id'];all_rows.append(q)
  members.append({'candidate_id':q['candidate_id'],'source':str(source),'root_id':q['root_id'],'gold_count':len(q['gold'])})
write(HOME/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',{
 'created_at_utc':now(),'status':'DEDUPLICATED_CANDIDATE_INVENTORY','questions':all_rows,
 'candidate_count':len(all_rows),**counts(all_rows),
 'gold_distribution':{str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in all_rows) for g in range(2,6)},
 'root_distribution':dict(Counter(q['root_id'] for q in all_rows)),
 'source_pools':[bind(p) for p in pools],
 'independent_batch_audits':[bind(previous/'evidence/RESULT_AUDIT.json'),bind(HOME/'evidence/RESULT_AUDIT.json')],
 'members':members,'deduplication':'root/type/sorted casefold-whitespace-normalized complete six-statement set; controls excluded before union',
 'duplicates_removed':duplicates,'training_allowed':False,'training_started':False,
 'formal_training_admission':False,'merged_into_baseline':False,
 'limitations':['Questions remain dependent on two source roots and parent fact families.',
  'Whole-question novelty does not imply independent new facts or semantic novelty.',
  'This inventory does not recertify or modify the older v73 baseline.',
  'Current review has explicitly disclosed prior source and parent exposure; no fresh blind parent claim.']})
print({'new':len(new),'cumulative':len(all_rows),**counts(all_rows)})
