"""Append independently verified current ordering selections to v81 inventory."""
from pathlib import Path
from collections import Counter
import sys
HOME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HOME))
from common import read,write,bind,now
previous=HOME.parent/'h4_bound_scene_ordering_mixed_batch_v81'
for name in ['INPUT_AUDIT','RESULT_AUDIT']:
    assert read(HOME/'evidence'/(name+'.json'))['status']=='PASS'
for name in ['RESULT_AUDIT','UNION_AUDIT']:
    assert read(previous/'evidence'/(name+'.json'))['status']=='PASS'
selected=read(HOME/'SELECTED_CANDIDATES.json')['questions']
assert all(q['question_type']=='ordering' for q in selected)
def counts(rows):
    return {'type_distribution':dict(Counter(q['question_type'] for q in rows)),
       'gold_distribution':{str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in rows) for g in range(2,6)}}
write(HOME/'AUDITED_NEW_CANDIDATES.json',{'created_at_utc':now(),
 'status':'CONTENT_REVIEW_SOLVER_SELECTION_AND_INDEPENDENT_AUDIT_PASS',
 'questions':selected,'new_question_count':len(selected),**counts(selected),
 'role':'derived_actor_reference_repairs_of_existing_ordering_drafts',
 'historical_controls_excluded':[],
 'content_review':bind(HOME/'BATCH_REVIEW.COMPLETE.json'),
 'paired_report':bind(HOME/'MIXED_EVALUATION_REPORT.json'),
 'independent_result_audit':bind(HOME/'evidence/RESULT_AUDIT.json'),
 'derivation':bind(HOME/'DERIVATION.json'),
 'training_allowed':False,'training_started':False,'formal_training_admission':False,'merged_into_baseline':False})
previous_path=previous/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json'
current_path=HOME/'AUDITED_NEW_CANDIDATES.json'
all_rows=[];seen={};members=[];duplicates=[]
for path in [previous_path,current_path]:
    for q in read(path)['questions']:
        field='items' if q['question_type']=='ordering' else 'options'
        key=(q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values())))
        if key in seen:
            duplicates.append({'candidate_id':q['candidate_id'],'matches':seen[key]});continue
        seen[key]=q['candidate_id'];all_rows.append(q)
        members.append({'candidate_id':q['candidate_id'],'source':str(path),'root_id':q['root_id'],'gold_count':len(q['gold'])})
write(HOME/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',{'created_at_utc':now(),'status':'DEDUPLICATED_CANDIDATE_INVENTORY',
 'questions':all_rows,'candidate_count':len(all_rows),**counts(all_rows),
 'root_distribution':dict(Counter(q['root_id'] for q in all_rows)),
 'source_pools':[bind(previous_path),bind(current_path)],
 'independent_batch_audits':[bind(previous/'evidence/RESULT_AUDIT.json'),bind(HOME/'evidence/RESULT_AUDIT.json')],
 'members':members,'duplicates_removed':duplicates,
 'deduplication':'root/type/sorted casefold-whitespace-normalized complete six-statement set; no controls',
 'training_allowed':False,'training_started':False,'formal_training_admission':False,'merged_into_baseline':False,
 'limitations':['Prior17 multiselect unchanged. New ordering are explicit repairs, not fresh fact generation.',
  'Exact whole-set deduplication does not imply semantic novelty or independent source families.',
  'No historical baseline is recertified or changed.']})
print({'new_ordering':len(selected),'cumulative':len(all_rows),**counts(all_rows)})
