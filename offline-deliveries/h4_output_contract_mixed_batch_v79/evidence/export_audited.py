from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import HERE,read,write,bind,sha,now

for name in ['RESULT_AUDIT_FINAL.json','INPUT_AUDIT.json']:
    assert read(HERE/'evidence'/name)['status']=='PASS', name
selected=read(HERE/'SELECTED_CANDIDATES.json')['questions']
new=[q for q in selected if q['question_type']=='ordering' or len(q['gold'])<6]
controls=[q for q in selected if q not in new]
assert len(new)==3 and len(controls)==2
write(HERE/'AUDITED_NEW_CANDIDATES.json',{'created_at_utc':now(),'status':'CONTENT_REVIEW_SOLVER_SELECTION_AND_INDEPENDENT_AUDIT_PASS','questions':new,'new_question_count':len(new),'gold_distribution':dict(Counter(str(len(q['gold'])) for q in new)),'role':'new_whole_question_combinations_from_existing_facts','content_review':bind(HERE/'BATCH_REVIEW.COMPLETE.json'),'paired_report':bind(HERE/'MIXED_EVALUATION_REPORT.json'),'independent_result_audit':bind(HERE/'evidence/RESULT_AUDIT_FINAL.json'),'historical_controls_excluded':[q['candidate_id'] for q in controls],'training_allowed':False,'training_started':False,'formal_training_admission':False,'merged_into_baseline':False})
previous=HERE.parent/'h4_ready_roots_mixed_batch_v78'
pools=[previous/'AUDITED_NEW_CANDIDATES.json',HERE/'AUDITED_NEW_CANDIDATES.json']
assert read(previous/'evidence/RESULT_AUDIT.json')['status']=='PASS'
all_rows=[];members=[];seen={};duplicates=[]
for source in pools:
    data=read(source)
    assert data['training_allowed'] is False
    for q in data['questions']:
        field='options' if q['question_type']=='multi_select' else 'items'
        key=(q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values())))
        if key in seen:
            duplicates.append({'candidate_id':q['candidate_id'],'matches':seen[key]});continue
        seen[key]=q['candidate_id'];all_rows.append(q);members.append({'candidate_id':q['candidate_id'],'source':str(source),'root_id':q['root_id'],'gold_count':len(q['gold'])})
assert len(all_rows)==7 and not duplicates
write(HERE/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',{'created_at_utc':now(),'status':'DEDUPLICATED_CANDIDATE_INVENTORY','questions':all_rows,'candidate_count':len(all_rows),'gold_distribution':{str(k):sum(len(q['gold'])==k for q in all_rows) for k in range(2,6)},'root_distribution':dict(Counter(q['root_id'] for q in all_rows)),'source_pools':[bind(p) for p in pools],'independent_batch_audits':[bind(previous/'evidence/RESULT_AUDIT.json'),bind(HERE/'evidence/RESULT_AUDIT_FINAL.json')],'members':members,'deduplication':'root/type/sorted casefold-whitespace-normalized complete six-statement set; controls excluded before union','duplicates_removed':duplicates,'training_allowed':False,'training_started':False,'formal_training_admission':False,'merged_into_baseline':False,'limitations':['Seven questions come from two source roots and two parent families.','Whole-question novelty does not imply independent new facts.','Zero ordering questions in this two-batch addition.','This inventory does not recertify or modify the older v73 baseline.']})
print({'new':len(new),'cumulative':len(all_rows),'gold_distribution':read(HERE/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')['gold_distribution']})
