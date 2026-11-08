"""Lossless export and content freezing, never run a solver or alter old pools."""
from pathlib import Path
import sys,json,copy
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import qa_workflow as w
sys.path.insert(0,str(B/'scripts'))
from evidence_export import export_root_questions
OLD=B.parent/'h4_codex_dev_test_ab_v124'
def export_split(split):
 w.verify(w.read(B/'MASTER_FREEZE.json')['files'])
 rows=[r for r in w.read(B/'ROOT_MANIFEST.json')['roots'] if r['split']==split]
 assert len(rows)==(5 if split=='dev' else 2)
 if split=='test':w.verify(w.read(B/'DEV_QA_CLOSED.json')['files'])
 questions=[];statuses=[]
 for r in rows:
  rid=r['root_id'];st=w.status(rid);assert st['status'] in ('QA_TERMINAL','QA_AUTHOR_FAILED','QA_CONTENT_HOLD'),(rid,st)
  src=w.ready(w.folder(rid));qq=export_root_questions(w.folder(rid)) if st['status']=='QA_TERMINAL' else []
  for q in qq:
   q.update(split=split,evaluation_only=True,solver_evaluated=False,baseline_conflicts_resolved=False,question_evidence_scope=list(w.SCENES))
   w.citations(q,src)
  questions.extend(qq);statuses.append({'root_id':rid,'split':split,**st,'question_count':len(qq)})
  text=[f'# {rid}',f'\n{split} | {len(qq)} content-pass questions | original conflicts retained', '\n## Complete dialogue']
  for t in src['turns']:text.append(f"\n[{t['turn_ref']}] **{t['speaker']}**: {t['text']}")
  text.append('\n## Questions (scope: six new R02 scenes)')
  for q in qq:
   text.extend(['\n### '+q['candidate_id'],'\n'+q['question']])
   for k,v in q.get('options',q.get('items')).items():text.append(f'\n{k}. {v}')
   text.append('\nGold: '+', '.join(q['gold']))
  p=B/'delivery'/split/'readable'/(rid+'.md');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('x') as f:f.write('\n'.join(text)+'\n')
 out=B/'delivery'/split
 w.save(out/'ROOT_STATUS.json',{'fixed_retry_roots':len(rows),'rows':statuses})
 w.save(out/'QUESTIONS_WITH_ANSWERS.json',{'created_at_utc':w.now(),'evaluation_only':True,'solver_run':False,'questions':questions})
 w.save(out/'QUESTIONS_PUBLIC.json',{'questions':[{k:copy.deepcopy(v) for k,v in q.items() if k in ('candidate_id','root_id','question_type','question','options','items','source_path','source_sha256')} for q in questions]})
 old=w.read(OLD/'delivery'/split/'QUESTIONS_WITH_ANSWERS.json')['questions']
 assert not {q['candidate_id'] for q in old}&{q['candidate_id'] for q in questions}
 w.save(out/'EXTENDED_QUESTIONS_WITH_ANSWERS.json',{'status':'ADDITIVE_VERSION_NOT_PRIOR_FROZEN_RESULT','old_pool_path':str(OLD/'delivery'/split/'QUESTIONS_WITH_ANSWERS.json'),'old_pool_sha256':w.sha(OLD/'delivery'/split/'QUESTIONS_WITH_ANSWERS.json'),'fixed_parent_roots':15 if split=='dev' else 16,'old_question_count':len(old),'new_question_count':len(questions),'questions':old+questions})
 if split=='dev':
  files=[]
  for r in rows:
   files.extend(p for p in w.unit(r['root_id']).rglob('*') if p.is_file())
  files.extend(p for p in out.rglob('*') if p.is_file())
  w.save(B/'DEV_QA_CLOSED.json',{'status':'FROZEN','created_at_utc':w.now(),'fixed_roots':5,'question_count':len(questions),'solver_run':False,'files':[{'path':str(p),'sha256':w.sha(p)} for p in sorted(files)]})
 print(split,len(questions),'exported')
def freeze():
 w.verify(w.read(B/'MASTER_FREEZE.json')['files']);w.verify(w.read(B/'DEV_QA_CLOSED.json')['files'])
 qs=[]
 for split in ('dev','test'):qs.extend(w.read(B/'delivery'/split/'QUESTIONS_WITH_ANSWERS.json')['questions'])
 assert len({q['candidate_id'] for q in qs})==len(qs) and qs
 w.save(B/'EVALUATION_POOL.json',{'status':'FROZEN_NEW_RETRY_QUESTIONS_NO_SCORE_SELECTION','provenance':{'backend':'CURRENT_CODEX_SESSION','source_batch_id':B.name,'exact_model_id':None,'same_inherited_model':True},'questions':qs})
 paths=[]
 for d in (B/'units',B/'delivery'):
  paths.extend(p for p in d.rglob('*') if p.is_file())
 paths.extend([B/'MASTER_FREEZE.json',B/'DEV_QA_CLOSED.json',B/'EVALUATION_POOL.json'])
 w.save(B/'QA_FREEZE.json',{'status':'FROZEN','created_at_utc':w.now(),'fixed_roots':7,'question_count':len(qs),'solver_run_before_freeze':False,'score_based_selection':False,'files':[{'path':str(p),'sha256':w.sha(p)} for p in sorted(paths)]})
 print('QA_FROZEN',len(qs))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev','test','freeze']);a=p.parse_args()
 freeze() if a.phase=='freeze' else export_split(a.phase)
