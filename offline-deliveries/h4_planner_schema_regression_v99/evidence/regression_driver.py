"""Bounded exposed Planner regression, separate from fresh expansion production."""
from pathlib import Path
import argparse, hashlib, importlib.util, inspect, json, subprocess, sys
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'h4_event_identity_correction_pilot_v98'
OLD_RUN=OLD/'expansion_engine/runs/four_roots_B_scene01'
RUN=BASE/'regression'
PY=Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
sys.path.insert(0,str(BASE/'expansion_engine'))
import runner as r

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def context(payload): return json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nINPUT_JSON\n',1)[1])[0]
def old_folder(root):return next((OLD_RUN/'requests').glob('*'+root+'-B-S01-planner-1'))
def load_case(root):
 payload=read(old_folder(root)/'REQUEST.json');return context(payload)
def control_request(root,repair):
 # Separate interpreter keeps the v98 module graph and its prompt unchanged.
 code="""from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1]);import runner as r
x=json.load(sys.stdin);c=x['context'];p,q=r.make_request(Path(x['run']),c['root_id'],c['route'],'planner',c['cumulative_history'],c['dataset_cards'],c['target_scene_id'],repair=x['repair'],latest_state=c['latest_state']);print(json.dumps({'payload':p,'proof':q}))
"""
 out=subprocess.run([str(PY),'-B','-c',code,str(OLD/'expansion_engine')],input=json.dumps({'context':load_case(root),'repair':repair,'run':str(OLD_RUN)}),text=True,capture_output=True,check=True)
 return json.loads(out.stdout)
def treatment_request(root,repair):
 c=load_case(root)
 p,q=r.make_request(RUN,c['root_id'],c['route'],'planner',c['cumulative_history'],c['dataset_cards'],c['target_scene_id'],repair=repair,latest_state=c['latest_state'])
 return {'payload':p,'proof':q}

def control_repair(root,raw):
 # Retain the control's original validator and feedback generator on retries.
 code="""import json,sys
sys.path.insert(0,sys.argv[1]);import runner as r
x=json.load(sys.stdin);c=x['context']
try:r.validate_plan(x['raw'],c['root_id'],c['route'],c['cumulative_history'],c['dataset_cards'],'S01',latest_state=c['latest_state'])
except (ValueError,TypeError) as error:print(json.dumps(r.planner_contract.build_repair_feedback(error,x['raw'],c['dataset_cards'][0])))
else:raise RuntimeError('control_validator_diverged_from_treatment')
"""
 out=subprocess.run([str(PY),'-B','-c',code,str(OLD/'expansion_engine')],input=json.dumps({'context':load_case(root),'raw':raw}),text=True,capture_output=True,check=True)
 return json.loads(out.stdout)

class RegressionCaller(r.Caller):
 """Inherit the exact production transport, capacity and immutable raw writer."""
 def __init__(self):
  self.run=RUN;self.limit=read(RUN/'REGRESSION_MANIFEST.json')['http_max']
  profile,self.factory,self.annotate,self.capacity,dev=r.base.transport()
  self.profile=profile();assert self.profile==read(RUN/'provider_profile.json')
  self.tokenizer=dev.tokenizer_from(str(r.base.TOKENIZER));self.http_counter=0;self.request_counter=0
  self.current_arm=None;self.current_root=None

 def call(self,label,payload,proof):
  return super().call(label,payload,{**proof,'experiment_arm':self.current_arm,
   'regression_cell_root':self.current_root,'regression_only':True,'fresh_source':False})

def prepare():
 assert not RUN.exists();RUN.mkdir()
 roots=['cc-train-episode-19053','cc-train-episode-25652']
 import shutil
 shutil.copytree(OLD_RUN/'feedback',RUN/'feedback')
 for root in roots:
  src=OLD_RUN/'sources'/(root+'.json');dst=RUN/'sources'/(root+'.json');dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
 profile,transport,tokenizer=r.transport_inputs();write(RUN/'provider_profile.json',profile)
 cells=[]
 for short,arms in [('19053',['control','treatment']),('25652',['treatment','control'])]:
  root='cc-train-episode-'+short
  pair={}
  for arm in arms:
   req=control_request(root,None) if arm=='control' else treatment_request(root,None)
   if arm=='control': assert req['payload']==read(old_folder(root)/'REQUEST.json')
   path=RUN/'prepared'/f'{short}_{arm}.json';write(path,req);pair[arm]=req
   cells.append({'root_id':root,'arm':arm,'prepared_request':bind(path)})
  assert context(pair['control']['payload'])==context(pair['treatment']['payload'])
  assert {k:v for k,v in pair['control']['payload'].items() if k!='messages'}=={k:v for k,v in pair['treatment']['payload'].items() if k!='messages'}
 _profile,_factory,annotate,capacity,dev=r.base.transport()
 tok=dev.tokenizer_from(str(r.base.TOKENIZER));capacity_rows=[]
 for cell in cells:
  payload=read(cell['prepared_request']['path'])['payload']
  result=annotate(capacity(profile,tok,payload['messages'],4096))
  capacity_rows.append({'root_id':cell['root_id'],'arm':cell['arm'],'prepared_request_sha256':cell['prepared_request']['sha256'],'capacity':result})
  assert result['status']=='PASS','capacity_blocked_no_truncation'
 write(RUN/'CAPACITY_PREFLIGHT.json',{'status':'PASS','network_calls':0,'cells':capacity_rows})
 paths=set(p for p in BASE.glob('expansion_engine/*.py'))|{Path(__file__),BASE/'expansion_engine/SOURCE_REGISTRY.json',BASE/'expansion_engine/SOURCE_SELECTION.json'}
 paths.update(p for p in (RUN/'feedback').rglob('*') if p.is_file())
 paths.update(p for p in (RUN/'sources').rglob('*') if p.is_file())
 paths.update(p for p in (RUN/'prepared').rglob('*') if p.is_file())
 paths.add(RUN/'provider_profile.json')
 paths.add(RUN/'CAPACITY_PREFLIGHT.json')
 paths.update(transport);paths.update(tokenizer)
 paths.update(p for p in r.LEGACY.glob('*.py'))
 for module in tuple(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   p=Path(name).resolve()
   if p.suffix=='.py' and p.is_file() and p.is_relative_to(r.base.PROJECT) and '.venv' not in str(p):paths.add(p)
 paths.update(p for p in (OLD/'expansion_engine').glob('*.py'))
 paths.update(OLD/'expansion_engine'/n for n in ('SOURCE_REGISTRY.json','SOURCE_SELECTION.json','POLICY.md') if (OLD/'expansion_engine'/n).exists())
 paths.update(p for p in (OLD_RUN/'feedback').rglob('*') if p.is_file())
 for root in roots:
  paths.update(old_folder(root)/n for n in ('REQUEST.json','PROOF.json','RESPONSE.json','VALIDATION.json'))
  paths.add(OLD_RUN/'sources'/(root+'.json'))
  paths.add(OLD_RUN/'roots'/root/'B/S01.LATEST_STATE.json')
 write(RUN/'REGRESSION_MANIFEST.json',{'version':'v99_exposed_paired_planner_regression_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Exposed regression only; no fresh source reservation, no source READY, no generation yield claim.','roots':roots,'cells':cells,'max_planner_attempts_per_cell':2,'max_event_review_attempts_per_cell':2,'max_logical_requests':16,'http_max':32,'max_tokens':4096,'endpoint':'First valid nonempty plan is independently event-reviewed then cell stops; no replan, realizer, scene review, QA, Solver or training.','first_attempt_contexts_paired':True,'retry_contexts_paired':False,'retries':'Only structural Planner repair; event-review format/citation correction inherits production bounded route.','files':[bind(p) for p in sorted(paths)],'fresh_source_count':0,'new_source_reservations':0,'training_allowed':False})
 print(json.dumps({'status':'PREPARED_EXPOSED_REGRESSION','cells':len(cells),'frozen_files':len(paths)}))

def live():
 manifest=read(RUN/'REGRESSION_MANIFEST.json')
 for row in manifest['files']:assert sha(row['path'])==row['sha256'],row['path']
 gate=read(BASE/'evidence/REGRESSION_PAIRED_PREFLIGHT_AUDIT.json')
 assert gate['status']=='PASS' and gate['manifest_sha256']==sha(RUN/'REGRESSION_MANIFEST.json'),'independent_preflight_required'
 assert not (RUN/'LIVE_STARTED.json').exists();write(RUN/'LIVE_STARTED.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'http_max':32})
 caller=RegressionCaller();results=[]
 for cell in manifest['cells']:
  root,arm=cell['root_id'],cell['arm'];caller.current_root=root;caller.current_arm=arm
  c=load_case(root);repair=None;out={'root_id':root,'arm':arm,'attempts':[],'structurally_valid_nonempty_plan':False,'event_review_reached':False,'event_review_logical_requests':0,'event_review_structurally_valid':False,'event_review_requires_replan':None}
  for attempt in (1,2):
   req=read(cell['prepared_request']['path']) if attempt==1 else (control_request(root,repair) if arm=='control' else treatment_request(root,repair))
   proof={**req['proof'],'experiment_arm':arm,'regression_only':True,'fresh_source':False}
   try:raw,directory=caller.call(f"{root}-{arm}-planner-{attempt}",req['payload'],proof)
   except RuntimeError as error:
    out['terminal_failure']=str(error);break
   try:
    candidate=r.validate_plan(raw,root,c['route'],c['cumulative_history'],c['dataset_cards'],'S01',latest_state=c['latest_state'])
   except (ValueError,TypeError) as error:
    repair=(control_repair(root,raw) if arm=='control' else
     r.planner_contract.build_repair_feedback(error,raw,c['dataset_cards'][0],route=c['route']))
    validation={'status':'REJECT','reason':str(error),'errors':repair['errors']};write(directory/'VALIDATION.json',validation)
    out['attempts'].append({'attempt':attempt,'request_directory':str(directory),'validation':validation});out['terminal_failure']='plan_rejected_after_retry';continue
   validation={'status':'PASS','semantic_certified':False};write(directory/'VALIDATION.json',validation);write(directory/'CANDIDATE_PLAN.json',candidate)
   out['attempts'].append({'attempt':attempt,'request_directory':str(directory),'validation':validation})
   if not candidate['scenes']:out['terminal_failure']='empty_scene';break
   out['structurally_valid_nonempty_plan']=True;before_event=caller.request_counter
   try:
    review=r.extract_event_state(caller,RUN,root,c['route'],c['cumulative_history'],c['dataset_cards'],candidate['scenes'][0],c['latest_state'],attempt)
   except RuntimeError as error:out['terminal_failure']=str(error);break
   finally:
    out['event_review_logical_requests']=caller.request_counter-before_event
    out['event_review_reached']=out['event_review_logical_requests']>0
   if review is None:out['terminal_failure']='event_review_rejected_after_retry'
   else:
    out['event_review_structurally_valid']=True;out['event_review_requires_replan']=r.event_contract.requires_replan(review)
    out['plan_diagnostics']=r.event_contract.plan_diagnostics(review);write(directory/'EVENT_REVIEW_RESULT.json',review)
    out['terminal_failure']=None if not out['event_review_requires_replan'] else 'event_review_requires_replan_diagnostic_endpoint'
   break
  results.append(out);print(json.dumps({'cell':root+'-'+arm,'result':out},ensure_ascii=False),flush=True)
  assert caller.request_counter<=16 and caller.http_counter<=32
 for row in manifest['files']:assert sha(row['path'])==row['sha256'],row['path']
 write(RUN/'REPORT.json',{'version':'v99_exposed_regression_results_v1','status':'COMPLETE','cells':results,'logical_requests':caller.request_counter,'http_attempts':caller.http_counter,'new_sources':0,'new_dialogue_turns':0,'new_questions':0,'solver_calls':0,'training_jobs':0,'fresh_yield_measured':False,'semantic_reliability_certified':False})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','live']);args=parser.parse_args()
 prepare() if args.action=='prepare' else live()
