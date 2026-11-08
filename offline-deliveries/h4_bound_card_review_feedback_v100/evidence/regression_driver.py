"""Bounded card-source review and semantic-feedback regression; never production yield."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, sys

BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'h4_planner_schema_regression_v99'
RUN=BASE/'regression'
sys.path.insert(0,str(BASE/'expansion_engine'))
import runner as r

def read(p):return json.loads(Path(p).read_text())
def write(p,value):r.base.write(Path(p),value)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def now():return datetime.now(timezone.utc).isoformat()
def context(payload):return json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nINPUT_JSON\n',1)[1])[0]

def prepare():
 assert not RUN.exists();RUN.mkdir()
 shutil.copytree(OLD/'regression/feedback',RUN/'feedback')
 shutil.copytree(OLD/'regression/sources',RUN/'sources')
 profile,transport,tokenizer=r.transport_inputs();write(RUN/'provider_profile.json',profile)
 cells=[];old_inputs=[]
 for short,seq in [('19053','003'),('25652','006')]:
  original=OLD/'regression/prepared'/f'{short}_treatment.json'
  original_plan=next((OLD/'regression/requests').glob(seq+'-*'))/'CANDIDATE_PLAN.json'
  c=context(read(original)['payload']);candidate=read(original_plan)
  # Validate input without changing the old plan, history, card or initial state.
  assert r.validate_plan(candidate,c['root_id'],c['route'],c['cumulative_history'],c['dataset_cards'],'S01',latest_state=c['latest_state'])==candidate
  data={'context':c,'initial_plan':candidate,'exposed_regression_only':True,
        'provenance':[bind(original),bind(original_plan)]}
  ip=RUN/'inputs'/f'{short}.json';write(ip,data)
  payload=r.event_contract.build_request(c['root_id'],c['route'],candidate['scenes'][0],c['cumulative_history'],c['dataset_cards'],c['latest_state'])
  pp=RUN/'prepared'/f'{short}_initial_event_review.json';write(pp,{'payload':payload})
  cells.append({'root_id':c['root_id'],'input':bind(ip),'prepared_request':bind(pp)})
  old_inputs.extend([original,original_plan])
 _prof,_factory,annotate,capacity,dev=r.base.transport()
 tok=dev.tokenizer_from(str(r.base.TOKENIZER));caps=[]
 for cell in cells:
  payload=read(cell['prepared_request']['path'])['payload']
  cap=annotate(capacity(profile,tok,payload['messages'],4096));assert cap['status']=='PASS'
  caps.append({'root_id':cell['root_id'],'prepared_request_sha256':cell['prepared_request']['sha256'],'capacity':cap})
 write(RUN/'CAPACITY_PREFLIGHT.json',{'status':'PASS','network_calls':0,'cells':caps})
 paths=set((BASE/'expansion_engine').glob('*.py'))|{Path(__file__),BASE/'PRECALL_PLAN_ZH.md',BASE/'PRESERVATION_BEFORE.json'}
 paths.update(BASE/'expansion_engine'/n for n in ('SOURCE_REGISTRY.json','SOURCE_SELECTION.json'))
 paths.update(p for p in RUN.rglob('*') if p.is_file())
 paths.update(old_inputs);paths.update(transport);paths.update(tokenizer)
 paths.update(r.LEGACY.glob('*.py'))
 for module in tuple(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   p=Path(name).resolve()
   if p.suffix=='.py' and p.is_file() and p.is_relative_to(r.base.PROJECT) and '.venv' not in str(p):paths.add(p)
 manifest={'version':'v100_bound_card_review_feedback_regression_v1','created_at_utc':now(),
  'scope':'Two exposed v99 candidate plans re-reviewed with source binding; diagnostic endpoints only.',
  'cells':cells,'max_initial_review_attempts_per_root':2,'max_semantic_replan_rounds_per_root':1,
  'max_structural_planner_attempts_per_semantic_round':2,'max_post_replan_review_attempts_per_root':2,
  'max_logical_requests':12,'http_max':24,'max_tokens':4096,
  'budget_note':'Separate exposed-regression budget; production PLAN_ATTEMPTS remains 2 total, not 2 per semantic round.',
  'endpoint':'Stop after a no-replan event review, exhausted correction/one semantic round, or transport/capacity failure. No realizer or QA.',
  'fresh_source_count':0,'new_source_reservations':0,'training_allowed':False,
  'files':[bind(p) for p in sorted(paths)]}
 write(RUN/'REGRESSION_MANIFEST.json',manifest)
 print(json.dumps({'status':'PREPARED','cells':len(cells),'frozen_files':len(paths)}),flush=True)

class RegressionCaller(r.Caller):
 def __init__(self):
  self.run=RUN;self.limit=read(RUN/'REGRESSION_MANIFEST.json')['http_max']
  prof,self.factory,self.annotate,self.capacity,dev=r.base.transport()
  self.profile=prof();assert self.profile==read(RUN/'provider_profile.json')
  self.tokenizer=dev.tokenizer_from(str(r.base.TOKENIZER));self.http_counter=0;self.request_counter=0
  self.current_root=None;self.stage=None;self.initial_prepared=None;self.initial_seen=False
 def call(self,label,payload,proof):
  assert self.request_counter<12,'logical_request_budget_exhausted'
  if self.stage=='initial_event_review' and not self.initial_seen:
   assert payload==self.initial_prepared,'initial_review_changed_after_freeze'
   self.initial_seen=True
  return super().call(label,payload,{**proof,'regression_root':self.current_root,'regression_stage':self.stage,
   'regression_only':True,'fresh_source':False})

def review(caller,c,plan,stage,number):
 caller.stage=stage;before=caller.request_counter
 result=r.extract_event_state(caller,RUN,c['root_id'],c['route'],c['cumulative_history'],c['dataset_cards'],plan['scenes'][0],c['latest_state'],number)
 row={'logical_requests':caller.request_counter-before,'structurally_valid':result is not None,
      'requires_replan':r.event_contract.requires_replan(result) if result is not None else None}
 path=RUN/'roots'/c['root_id']/(stage+'.json')
 write(path,{'summary':row,'review':result});row['artifact']=bind(path)
 return row,result

def run_cell(caller,cell):
 data=read(cell['input']['path']);c=data['context'];candidate=data['initial_plan']
 root=c['root_id'];caller.current_root=root
 caller.initial_prepared=read(cell['prepared_request']['path'])['payload'];caller.initial_seen=False
 out={'root_id':root,'initial_review':None,'semantic_replan_rounds':0,'planner_attempts':[],
      'post_replan_review':None,'endpoint_review_without_replan':False,'terminal_failure':None}
 try:
  out['initial_review'],snapshot=review(caller,c,candidate,'initial_event_review',0)
  if snapshot is None:out['terminal_failure']='initial_review_rejected_after_correction_limit';return out
  if not r.event_contract.requires_replan(snapshot):out['endpoint_review_without_replan']=True;return out
  repair=r.semantic_replan_feedback(candidate,snapshot)
  fp=RUN/'roots'/root/'SEMANTIC_REPLAN_FEEDBACK.json';write(fp,repair)
  out['semantic_feedback']=bind(fp);out['semantic_replan_rounds']=1
  caller.stage='semantic_replan'
  for attempt in (1,2):
   payload,proof=r.make_request(RUN,root,c['route'],'planner',c['cumulative_history'],c['dataset_cards'],'S01',
    repair=repair,latest_state=c['latest_state'],event_state=snapshot)
   raw,directory=caller.call(f'{root}-semantic-replan-structure-{attempt}',payload,proof)
   try:
    revised=r.validate_plan(raw,root,c['route'],c['cumulative_history'],c['dataset_cards'],'S01',latest_state=c['latest_state'])
   except (ValueError,TypeError) as error:
    repair=r.planner_contract.build_repair_feedback(error,raw,c['dataset_cards'][0],route=c['route'])
    validation={'status':'REJECT','reason':str(error),'errors':repair['errors']}
    write(directory/'VALIDATION.json',validation)
    out['planner_attempts'].append({'attempt':attempt,'directory':str(directory),'validation':validation})
    out['terminal_failure']='semantic_replan_structure_rejected_after_limit';continue
   validation={'status':'PASS','semantic_certified':False};write(directory/'VALIDATION.json',validation)
   write(directory/'CANDIDATE_PLAN.json',revised)
   out['planner_attempts'].append({'attempt':attempt,'directory':str(directory),'validation':validation})
   if not revised['scenes']:out['terminal_failure']='semantic_replan_empty';return out
   out['post_replan_review'],post=review(caller,c,revised,'post_replan_event_review',1)
   if post is None:out['terminal_failure']='post_replan_review_rejected_after_correction_limit'
   elif r.event_contract.requires_replan(post):out['terminal_failure']='semantic_replan_still_requires_revision_after_limit'
   else:out['endpoint_review_without_replan']=True;out['terminal_failure']=None
   return out
 except (RuntimeError,ValueError) as error:
  out['terminal_failure']=f'{type(error).__name__}:{error}'
 return out

def live():
 manifest=read(RUN/'REGRESSION_MANIFEST.json')
 for row in manifest['files']:assert sha(row['path'])==row['sha256'],row['path']
 gate=read(BASE/'evidence/REGRESSION_PREFLIGHT_AUDIT.json')
 assert gate['status']=='PASS' and gate['manifest_sha256']==sha(RUN/'REGRESSION_MANIFEST.json')
 assert not (RUN/'LIVE_STARTED.json').exists();write(RUN/'LIVE_STARTED.json',{'at_utc':now(),'http_max':24})
 caller=RegressionCaller();cells=[]
 for cell in manifest['cells']:
  row=run_cell(caller,cell);cells.append(row)
  print(json.dumps({'root_id':row['root_id'],'result':row},ensure_ascii=False),flush=True)
  assert caller.request_counter<=12 and caller.http_counter<=24
 for row in manifest['files']:assert sha(row['path'])==row['sha256'],row['path']
 write(RUN/'REPORT.json',{'version':'v100_bound_source_feedback_results_v1','status':'COMPLETE','cells':cells,
  'logical_requests':caller.request_counter,'http_attempts':caller.http_counter,
  'new_sources':0,'new_dialogue_turns':0,'ready_sources':0,'new_questions':0,'solver_calls':0,'training_jobs':0,
  'fresh_yield_measured':False,'semantic_reliability_certified':False})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','live']);args=parser.parse_args()
 prepare() if args.action=='prepare' else live()
