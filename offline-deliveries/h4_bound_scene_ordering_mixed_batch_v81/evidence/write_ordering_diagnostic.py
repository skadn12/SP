"""Post-review diagnosis only; preserve every original generated item and Gold."""
from pathlib import Path
import sys,re
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now
review=read(HOME/'ordering/CONTENT_REVIEW.json')
reviewed={q['candidate_id']:q for q in review['questions']}
rows=[]
for run in sorted((HOME/'ordering').glob('root*_ordering')):
 source=read(run/'SOURCE.json');turns={t['turn_ref']:t for t in source['turns']}
 positions={t['turn_ref']:i for i,t in enumerate(source['turns'])}
 plans={p['candidate_id']:p for p in read(run/'ANCHOR_PLAN.json')['questions']}
 rejected={r['candidate_id']:r['reasons'] for r in read(run/'REJECTIONS.json')}
 response=next(run.glob('requests/*/RESPONSE.json'))
 import json
 raw=json.loads(read(response)['text'])
 prior=read(run/'PRIOR.json')
 def whole(q):return tuple(sorted(' '.join(t.split()).casefold() for t in q['items'].values()))
 for q in raw['questions']:
  if q is None:continue
  p=plans[q['candidate_id']];a=p['anchors']
  order=sorted(a,key=lambda k:(positions[a[k]['turn_ref']],turns[a[k]['turn_ref']]['text'].index(a[k]['quote'])))
  rows.append({'candidate_id':q['candidate_id'],'scene_id':p['scene_id'],
   'writer_items':q['items'],'writer_gold':q['gold'],'anchor_order_diagnostic_only':order,
   'original_gold_agrees':q['gold']==order,
   'all_six_fixed_evidence_exact':all(turns[v['turn_ref']]['speaker']==v['speaker'] and turns[v['turn_ref']]['text'].count(v['quote'])==1 for v in a.values()),
   'all_anchors_inside_fixed_scene':all(v['turn_ref'].rsplit(':',1)[0]==p['scene_id'] for v in a.values()),
   'matching_prior_whole_sets':[x['candidate_id'] for x in prior if whole(x)==whole(q)],
   'public_gendered_pronoun_tokens':{k:re.findall(r'\b(?:he|she|him|her|his|hers|himself|herself)\b',t,re.I) for k,t in q['items'].items() if re.search(r'\b(?:he|she|him|her|his|hers|himself|herself)\b',t,re.I)},
   'mechanical_rejection':rejected.get(q['candidate_id'],[]),
   'source_review':reviewed.get(q['candidate_id']),
   'raw_response':bind(response),'plan':bind(run/'ANCHOR_PLAN.json')})
write(HOME/'evidence/ORDERING_FAILURE_DIAGNOSIS.json',{
 'created_at_utc':now(),'status':'DIAGNOSIS_ONLY','current_private_writer_gold_read':True,
 'blind_review_claimed':False,'current_solver_scores_read':False,'model_calls':0,
 'source_review':bind(HOME/'ordering/CONTENT_REVIEW.json'),'candidates':rows,
 'summary':{'raw_count':len(rows),'correct_raw_gold_count':sum(r['original_gold_agrees'] for r in rows),
 'exact_fixed_citation_count':sum(r['all_six_fixed_evidence_exact'] for r in rows)*6,
 'inside_scene_count':sum(r['all_anchors_inside_fixed_scene'] for r in rows),
 'prior_whole_set_duplicate_count':sum(bool(r['matching_prior_whole_sets']) for r in rows),
 'content_usable':review['accepted_count']},
 'question_or_gold_or_evidence_repairs':[],
 'limits':['Gendered token detection identifies wording to inspect, not a generic proof of unsupported actor identity. Saved source judgments determine semantic failure.',
 'Fixed anchors solve citation binding and constrain scope; they cannot certify generated paraphrases or independent Gold.',
 'No failed question is repaired or admitted by this diagnostic.']})
print('ordering diagnostic saved')
