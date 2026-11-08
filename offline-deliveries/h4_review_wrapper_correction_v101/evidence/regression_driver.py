"""Replay exact failed wrapper repair, then bounded live semantic replan."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,sys
BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'h4_bound_card_review_feedback_v100'
RUN=BASE/'regression'
sys.path.insert(0,str(BASE/'expansion_engine'))
import runner as r
def read(p):return json.loads(Path(p).read_text())
def write(p,x):r.base.write(Path(p),x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def now():return datetime.now(timezone.utc).isoformat()

def prepare():
 assert not RUN.exists();RUN.mkdir()
 shutil.copytree(OLD/'regression/feedback',RUN/'feedback');shutil.copytree(OLD/'regression/sources',RUN/'sources')
 old_input=OLD/'regression/inputs/19053.json';data=read(old_input);c=data['context'];candidate=data['initial_plan']
 first=next((OLD/'regression/requests').glob('001-*'));second=next((OLD/'regression/requests').glob('002-*'))
 raw=read(second/'RESPONSE.json')['text']
 old_request=read(second/'REQUEST.json')
 old_context=json.JSONDecoder().raw_decode(old_request['messages'][0]['content'].split('EVENT_STATE_INPUT_JSON\n',1)[1])[0]
 repair=old_context['targeted_repair']
 assert repair['previous_raw']==read(first/'RESPONSE.json')['text']
 correction=r.event_contract.correction_progress(raw,repair)
 review=r.event_contract.validate(raw,c['root_id'],c['route'],candidate['scenes'][0],c['cumulative_history'],cards=c['dataset_cards'])
 r.event_contract.require_correction_progress(correction)
 assert r.event_contract.requires_replan(review),'historical_review_has_no_binding_negative'
 write(RUN/'REPLAY_VALIDATION.json',{'status':'PASS','classification':'Offline replay of exact historical responses; zero new model calls.',
  'correction_progress':correction,'requires_replan':True,'full_validation_pass':True,'semantic_certified':False,
  'first_raw':bind(first/'RESPONSE.json'),'second_raw':bind(second/'RESPONSE.json'),'original_reject':bind(second/'VALIDATION.json')})
 write(RUN/'REPLAY_REVIEW.json',review);write(RUN/'INPUT.json',data)
 feedback=r.semantic_replan_feedback(candidate,review);write(RUN/'SEMANTIC_REPLAN_FEEDBACK.json',feedback)
 payload,proof=r.make_request(RUN,c['root_id'],c['route'],'planner',c['cumulative_history'],c['dataset_cards'],'S01',
  repair=feedback,latest_state=c['latest_state'],event_state=review)
 pp=RUN/'prepared/19053_semantic_replan_1.json';write(pp,{'payload':payload,'proof':proof})
 profile,transport,tokenizer=r.transport_inputs();write(RUN/'provider_profile.json',profile)
 _p,_factory,annotate,capacity,dev=r.base.transport()
 cap=annotate(capacity(profile,dev.tokenizer_from(str(r.base.TOKENIZER)),payload['messages'],4096));assert cap['status']=='PASS'
 write(RUN/'CAPACITY_PREFLIGHT.json',{'status':'PASS','network_calls':0,'prepared_request_sha256':sha(pp),'capacity':cap})
 paths=set((BASE/'expansion_engine').glob('*.py'))|{Path(__file__),BASE/'PRECALL_PLAN_ZH.md',BASE/'PRESERVATION_BEFORE.json',old_input,OLD/'regression/REPORT.json',OLD/'COMPLETE_MANIFEST.json'}
 paths.update(BASE/'expansion_engine'/n for n in ['SOURCE_REGISTRY.json','SOURCE_SELECTION.json'])
 paths.update(p for p in RUN.rglob('*') if p.is_file());paths.update(transport);paths.update(tokenizer);paths.update(r.LEGACY.glob('*.py'))
 for folder in [first,second]:paths.update(folder/n for n in ['REQUEST.json','PROOF.json','RESPONSE.json','VALIDATION.json'])
 for module in tuple(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.suffix=='.py' and p.is_file() and p.is_relative_to(r.base.PROJECT) and '.venv' not in str(p):paths.add(p)
 prior=read(OLD/'regression/REPORT.json')
 write(RUN/'REGRESSION_MANIFEST.json',{'version':'v101_wrapper_feedback_continuation_v1','created_at_utc':now(),
  'root_id':c['root_id'],'scope':'One exposed root; exact historical replay then live Planner plus event review, no fresh yield.',
  'prepared_request':bind(pp),'max_logical_requests':4,'http_max':8,'max_planner_attempts':2,'max_event_review_attempts':2,
  'prior_phase_logical_requests':prior['logical_requests'],'prior_phase_http_attempts':prior['http_attempts'],
  'joint_max_logical_requests':12,'joint_max_http_attempts':24,'max_tokens':4096,'fresh_source_count':0,'training_allowed':False,
  'files':[bind(p) for p in sorted(paths)]})
 print(json.dumps({'status':'PREPARED','frozen_files':len(paths),'replay_status':correction['status']}),flush=True)

class Caller(r.Caller):
 def __init__(self):
  self.run=RUN;self.limit=8
  p,self.factory,self.annotate,self.capacity,dev=r.base.transport();self.profile=p()
  assert self.profile==read(RUN/'provider_profile.json')
  self.tokenizer=dev.tokenizer_from(str(r.base.TOKENIZER));self.http_counter=0;self.request_counter=0;self.stage='semantic_replan'
 def call(self,label,payload,proof):
  assert self.request_counter<4
  return super().call(label,payload,{**proof,'regression_root':'cc-train-episode-19053','regression_stage':self.stage,
   'regression_only':True,'fresh_source':False,'historical_review_replay_sha256':sha(RUN/'REPLAY_REVIEW.json')})

def live():
 manifest=read(RUN/'REGRESSION_MANIFEST.json')
 for x in manifest['files']:assert sha(x['path'])==x['sha256'],x['path']
 gate=read(BASE/'evidence/REGRESSION_PREFLIGHT_AUDIT.json')
 assert gate['status']=='PASS' and gate['manifest_sha256']==sha(RUN/'REGRESSION_MANIFEST.json')
 assert not (RUN/'LIVE_STARTED.json').exists();write(RUN/'LIVE_STARTED.json',{'at_utc':now()})
 caller=Caller();data=read(RUN/'INPUT.json');c=data['context'];snapshot=read(RUN/'REPLAY_REVIEW.json')
 feedback=read(RUN/'SEMANTIC_REPLAN_FEEDBACK.json')
 out={'root_id':c['root_id'],'historical_replay':'PASS, not a new model review','planner_attempts':[],
      'post_replan_review':None,'endpoint_review_without_replan':False,'terminal_failure':None}
 try:
  for attempt in [1,2]:
   if attempt==1:req=read(manifest['prepared_request']['path']);payload,proof=req['payload'],req['proof']
   else:payload,proof=r.make_request(RUN,c['root_id'],c['route'],'planner',c['cumulative_history'],c['dataset_cards'],'S01',repair=feedback,latest_state=c['latest_state'],event_state=snapshot)
   raw,directory=caller.call(f"{c['root_id']}-semantic-replan-structure-{attempt}",payload,proof)
   try:plan=r.validate_plan(raw,c['root_id'],c['route'],c['cumulative_history'],c['dataset_cards'],'S01',latest_state=c['latest_state'])
   except (ValueError,TypeError) as error:
    feedback=r.planner_contract.build_repair_feedback(error,raw,c['dataset_cards'][0],route=c['route'])
    v={'status':'REJECT','reason':str(error),'errors':feedback['errors']};write(directory/'VALIDATION.json',v)
    out['planner_attempts'].append({'attempt':attempt,'directory':str(directory),'validation':v});out['terminal_failure']='planner_rejected_after_limit';continue
   v={'status':'PASS','semantic_certified':False};write(directory/'VALIDATION.json',v);write(directory/'CANDIDATE_PLAN.json',plan)
   out['planner_attempts'].append({'attempt':attempt,'directory':str(directory),'validation':v})
   if not plan['scenes']:out['terminal_failure']='empty_scene';break
   caller.stage='post_replan_event_review';before=caller.request_counter
   review=r.extract_event_state(caller,RUN,c['root_id'],c['route'],c['cumulative_history'],c['dataset_cards'],plan['scenes'][0],c['latest_state'],1)
   out['post_replan_review']={'logical_requests':caller.request_counter-before,'structurally_valid':review is not None,
    'requires_replan':r.event_contract.requires_replan(review) if review is not None else None}
   write(RUN/'POST_REPLAN_REVIEW.json',review)
   if review is None:out['terminal_failure']='post_replan_review_rejected_after_limit'
   elif r.event_contract.requires_replan(review):out['terminal_failure']='post_replan_still_requires_revision'
   else:out['endpoint_review_without_replan']=True;out['terminal_failure']=None
   break
 except (RuntimeError,ValueError) as error:out['terminal_failure']=f'{type(error).__name__}:{error}'
 for x in manifest['files']:assert sha(x['path'])==x['sha256'],x['path']
 joint_l=manifest['prior_phase_logical_requests']+caller.request_counter;joint_h=manifest['prior_phase_http_attempts']+caller.http_counter
 assert joint_l<=12 and joint_h<=24
 write(RUN/'REPORT.json',{'version':'v101_results_v1','status':'COMPLETE','result':out,'logical_requests':caller.request_counter,
  'http_attempts':caller.http_counter,'joint_logical_requests':joint_l,'joint_http_attempts':joint_h,
  'new_sources':0,'new_dialogue_turns':0,'ready_sources':0,'new_questions':0,'solver_calls':0,'training_jobs':0,'semantic_certified':False})
 print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','live']);a=p.parse_args()
 prepare() if a.action=='prepare' else live()
