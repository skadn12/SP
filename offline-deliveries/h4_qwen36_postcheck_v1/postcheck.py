"""Frozen eight-item post-generation review using existing bounded transport.

No Planner/Writer regeneration, no retries, no automatic resume. Reviewer sees
original candidate and full source, never Codex diagnoses or corrected output.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from jsonschema import Draft202012Validator

DEST=Path(__file__).resolve().parent
RUN=DEST/'prepared_run'
BASE=Path('/root/autodl-tmp/gxy-selfplay')
WT=BASE/'worktrees/h4-gate-downstream-v1'
SOURCE=BASE/'offline-deliveries/h4_gate_downstream_v1/prepared_run'
sys.path.insert(0,str(WT))
from scripts.current_method.restart_n5_spell.core import atomic_json, atomic_text
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_runtime import DevelopmentRuntimeError, DevelopmentContentError, capacity_check
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_school_service import SchoolDevelopmentCaller, annotate_school_capacity
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import tokenizer_from, now, cost_summary
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload
from scripts.current_method.restart_n5_spell.h4_ordering_grounded_options_v2.contract import derive_ordering

VERSION='qwen36_postcheck_ids_v1'
MAX_HTTP=8
MAX_OUTPUT=8192
POLICY={'version':'ordering_grounded_options_v2','node_count':4}
ROOTS=('cc-train-episode-259192','cc-train-episode-96462','cc-train-episode-112812')
SCHEDULE=(('259192','I001',3,'primary'),('259192','I003',4,'primary'),
          ('96462','I001',4,'primary'),('96462','I002',3,'primary'),('96462','I003',4,'primary'),
          ('112812','I001',3,'primary'),('112812','I002',2,'repair'),('112812','I003',4,'repair'))
SLOTS=tuple({'slot_id':f'Q{i:02d}','root_id':'cc-train-episode-'+s,'intent_id':intent,'raw_row_index':row,'phase':phase,
             'candidate_id':f'cc-train-episode-{s}:{intent}-C-Q{row:02d}:{phase}'} for i,(s,intent,row,phase) in enumerate(SCHEDULE,1))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def strict_object(properties,required=None):return {'type':'object','additionalProperties':False,'properties':properties,'required':list(properties) if required is None else required}
TEXT={'type':'string','minLength':1}
IDS={'type':'array','uniqueItems':True,'items':TEXT}
EVENT_ID={'enum':['I1','I2','I3','I4']}
PAIR={'type':'array','minItems':2,'maxItems':2,'uniqueItems':True,'items':EVENT_ID}
SUPPORT={'enum':['SUPPORTED','UNSUPPORTED','UNCERTAIN']}
NODE=strict_object({'item_id':EVENT_ID,'support':SUPPORT,'evidence_refs':IDS,'reason':TEXT})
EDGE_REVIEW=strict_object({'ordered_pair':PAIR,'support':SUPPORT,'evidence_refs':IDS,'reason':TEXT})
EDGE=strict_object({'ordered_pair':PAIR,'evidence_refs':{**IDS,'minItems':1},'reason':TEXT})
SCHEMA=strict_object({
 'version':{'const':VERSION},'candidate_id':TEXT,
 'item_checks':{'type':'array','minItems':4,'maxItems':4,'items':NODE},
 'original_relation_checks':{'type':'array','items':EDGE_REVIEW},
 'question_check':strict_object({'status':{'enum':['CONSISTENT','MINOR_WORDING_FIX','MATERIAL_MISMATCH']},'reason':TEXT,'suggested_question':{'type':['string','null']}}),
 'supported_relations':{'type':'array','maxItems':6,'items':EDGE},
 'decision':{'enum':['READY','REJECT','UNDETERMINED']},'decision_reason':TEXT})

INSTRUCTIONS="""ROLE: Post-generation factual and temporal reviewer.
Review ONE existing ordering question against the COMPLETE TRANSCRIPT below.
The transcript is the only factual source. Existing event metadata, evidence
bindings and temporal edges are proposals to inspect, not trusted truth.
You are not generating new questions, choosing a new topic or certifying a model.

Tasks:
1. Check each displayed item_text, its actor, action, event instance and state.
   Does the source support the WHOLE claim? Mere topical relevance is insufficient.
   Metadata does not silently change the meaning of the displayed option.
   Use SUPPORTED / UNSUPPORTED / UNCERTAIN and a short factual reason.
2. Select the exact canonical turn IDs needed to substantiate each item. You may
   replace or add IDs from anywhere in the complete transcript. Include the
   context needed for pronouns, answers, negation, actor, state and time. One
   event may need several turns. Return IDs only; the program retrieves their
   complete source text. Do not copy or rewrite quotations in output.
3. Assess each original ordered_pair, then independently return only relations
   justified by source evidence. [A,B] means event A is strictly earlier than B.
   The order of mentioned memories, conversation position, a usual workflow or
   an expected causal story alone does not prove event chronology. Direct
   references to an earlier event, clearly linked instances, responsive acts,
   or explicit relative/absolute times can establish it. Allow ordinary
   contextual reference when supported; do not demand an absolute date for
   every event. Do not turn an unknown relation into its reverse.
4. Distinguish capability, proposal, decision, start and completion. A decision
   may be a valid event, but does not prove its intended action happened.
   Repeated mentions of the same occurrence are not separate occurrences.
   Existing synthetic date labels date conversations, not all recalled events.
   UNKNOWN is not a date; do not invent dates or infer them from section order.
5. Preserve all four item IDs and displayed item texts. Do not quietly rewrite
   an unsupported item to make it true. Report it instead. You may suggest a
   minor question-stem wording fix only if it does not change the four items,
   requested temporal criterion, or task. Explain any mismatch explicitly.
6. Return READY only when all four items are supported and your supported
   relations establish exactly one full strict ordering. If an item is false
   or unsupported, REJECT. If evidence cannot settle the unique chronology or
   an essential claim, UNDETERMINED. Do not manufacture a chain to be helpful.
   Report the actual outcome, not an expected balance of decisions.

Each item and each original relation must be checked once. evidence_refs must
be exact IDs in this candidate's transcript; do not add/remove root prefixes.
For unsupported/uncertain claims cite the relevant diagnostic context, or []
if no source location exists. supported_relations must cite sufficient source
IDs for the relationship, not just two isolated topic-matching sentences.
Do not return Gold or a final answer permutation. The program derives it from
supported_relations. Do not output additional fields or code fences.
Keep reasons concise (one or two sentences each). Return JSON matching SCHEMA.
SCHEMA:
"""

def build_messages(slot,raw,transcript):
 prompt=INSTRUCTIONS+json.dumps(SCHEMA,ensure_ascii=False,separators=(',',':'))
 prompt+='\nCANDIDATE_ID: '+slot['candidate_id']
 prompt+='\nBEGIN ORIGINAL CANDIDATE DATA\n'+json.dumps(raw,ensure_ascii=False,indent=2)+'\nEND ORIGINAL CANDIDATE DATA'
 prompt+='\nBEGIN COMPLETE TRANSCRIPT\n'+transcript+'\nEND COMPLETE TRANSCRIPT'
 return [{'role':'user','content':prompt}]

def validate_review(review,slot,raw,source):
 errors=['SCHEMA:'+('/'.join(map(str,e.absolute_path)))+':'+e.validator for e in Draft202012Validator(SCHEMA).iter_errors(review)]
 if errors:return {'status':'FAIL','errors':errors,'ready':False}
 if review['candidate_id']!=slot['candidate_id']:errors.append('CANDIDATE_ID_MISMATCH')
 nodes={n['item_id']:n for n in raw['event_nodes']}
 checks=review['item_checks']
 if len({n['item_id'] for n in checks})!=4 or {n['item_id'] for n in checks}!=set(nodes):errors.append('ITEM_COVERAGE_INVALID')
 original_pairs=[tuple(e['ordered_pair']) for e in raw['temporal_relations']]
 checked_pairs=[tuple(e['ordered_pair']) for e in review['original_relation_checks']]
 if len(checked_pairs)!=len(set(checked_pairs)) or set(checked_pairs)!=set(original_pairs):errors.append('ORIGINAL_RELATION_COVERAGE_INVALID')
 catalog={t['turn_ref']:t for t in source['turns']}
 entities=checks+review['original_relation_checks']+review['supported_relations']
 for e in entities:
  if any(ref not in catalog for ref in e['evidence_refs']):errors.append('EVIDENCE_ID_UNKNOWN')
  if e.get('support')=='SUPPORTED' and not e['evidence_refs']:errors.append('SUPPORTED_CLAIM_WITHOUT_IDS')
 pairs=[tuple(e['ordered_pair']) for e in review['supported_relations']]
 if len(pairs)!=len(set(pairs)):errors.append('DUPLICATE_SUPPORTED_RELATION')
 q=review['question_check']
 if q['status']=='MINOR_WORDING_FIX' and not (isinstance(q['suggested_question'],str) and q['suggested_question'].strip()):errors.append('QUESTION_FIX_MISSING')
 if q['status']!='MINOR_WORDING_FIX' and q['suggested_question'] is not None:errors.append('UNREQUESTED_QUESTION_REWRITE')
 if review['decision']=='READY' and (any(n['support']!='SUPPORTED' for n in checks) or q['status']=='MATERIAL_MISMATCH'):errors.append('READY_WITH_UNSUPPORTED_CONTENT')
 errors=sorted(set(errors))
 return {'status':'FAIL' if errors else 'PASS','errors':errors,'ready':review['decision']=='READY' and not errors,
         'evidence_reference_scope':'EXISTENCE_ONLY_NOT_SEMANTIC_CERTIFICATION'}

def materialize(review,slot,raw,source):
 check=validate_review(review,slot,raw,source)
 result={'review_validation':check,'model_decision':review.get('decision'),'program_acceptance':False}
 if not check['ready']:return result
 candidate=copy.deepcopy(raw)
 catalog={t['turn_ref']:t for t in source['turns']}
 byid={n['item_id']:n for n in review['item_checks']}
 def binding(refs):return {'evidence_refs':refs,'evidence_spans':[{'turn_ref':r,'quote':catalog[r]['text']} for r in refs]}
 for n in candidate['event_nodes']:
  n.pop('date_anchors',None)
  n.update(binding(byid[n['item_id']]['evidence_refs']))
 candidate['temporal_relations']=[{'ordered_pair':e['ordered_pair'],**binding(e['evidence_refs'])} for e in review['supported_relations']]
 # Old evidence_groups are raw Writer metadata. Re-project them rather than
 # retaining a stale ref outside the rebound source evidence.
 candidate['evidence_groups']={'events':list(dict.fromkeys(r for n in candidate['event_nodes'] for r in n['evidence_refs'])),
                               'temporal_support':list(dict.fromkeys(r for e in candidate['temporal_relations'] for r in e['evidence_refs']))}
 candidate['gold']=None
 if review['question_check']['status']=='MINOR_WORDING_FIX':candidate['question']=review['question_check']['suggested_question']
 identity={'run_id':VERSION,'arm_id':'QWEN36_POSTCHECK','root_id':slot['root_id'],'intent_id':slot['intent_id'],
           'source':'post_generation_review','stage':'postcheck','request_id':slot['slot_id'],'raw_row_index':1,
           'candidate_id':slot['candidate_id']+':qwen36_postcheck','raw_sha256':digest(review)}
 derived=derive_ordering(candidate,expanded=source,policy=POLICY,identity=identity)
 result.update(program_acceptance=derived['status']=='PASS',rebound_contract=candidate,
               program_derivation=derived['record'],derived_gold=derived['record']['derived_gold'],
               graph_errors=derived['errors'],semantic_support_status='NOT_ASSESSED_BY_PROGRAM')
 return result

class PostcheckCaller(SchoolDevelopmentCaller):
 MAX_HTTP_ATTEMPTS=MAX_HTTP
 MAX_CALLS_PER_ROOT=3
 ALLOWED_STAGES=frozenset(s['slot_id'] for s in SLOTS)
 def __init__(self,*args,http_max=MAX_HTTP,per_root_max=3,**kwargs):
  if http_max!=MAX_HTTP or per_root_max!=3:raise DevelopmentRuntimeError('fixed_postcheck_budget_required')
  super().__init__(*args,http_max=http_max,per_root_max=per_root_max,**kwargs)
 def _validate_slot(self,root_id,stage):
  if (root_id,stage) not in {(s['root_id'],s['slot_id']) for s in SLOTS}:raise DevelopmentRuntimeError('postcheck_slot_not_allowed')

def verify_frozen(manifest):
 for row in manifest['files']:
  if sha(row['path'])!=row['sha256']:raise RuntimeError('FROZEN_FILE_CHANGED:'+Path(row['path']).name)

def prepare():
 if RUN.exists():raise FileExistsError('prepared_run_exists_no_overwrite')
 old=read(SOURCE/'freeze_manifest.json')
 profile=copy.deepcopy(read(SOURCE/'frozen/provider_profile.json'))
 assert profile['endpoint']=='https://ai.tju.edu.cn/api/v3/chat/completions' and profile['model']=='tju-llm'
 assert profile['http_retry_limit']==0 and all(profile[k] is False for k in ('redirects','ssl_fallback','parameter_fallback'))
 profile['request_parameters']={'temperature':0.1,'max_tokens':MAX_OUTPUT,'response_format':{'type':'json_object'}}
 profile['output_limit']=MAX_OUTPUT
 profile['profile_id']=VERSION
 profile['comparison']='SINGLE_MODEL_POSTCHECK_ON_FIXED_ORIGINAL_ORDERING_CANDIDATES'
 tokenizer=tokenizer_from(old['tokenizer_path'])
 RUN.mkdir(parents=True)
 atomic_json(RUN/'frozen/provider_profile.json',profile)
 atomic_json(RUN/'frozen/output_schema.json',SCHEMA)
 all_sources={SOURCE/'freeze_manifest.json', SOURCE/'frozen/provider_profile.json',Path(__file__).resolve()}
 capacities=[]
 slots=[]
 for slot in SLOTS:
  q=SOURCE/'qa/roots'/slot['root_id']
  pool=read(q/'candidate_pool.json')
  c=next(c for c in pool if c['candidate_id']==slot['candidate_id'])
  raw_path=q/'raw/responses'/f"{c['admission_request_id']}.json.txt"
  raw=read(raw_path)['candidates'][slot['raw_row_index']-1]
  assert raw['question_type']=='ordering' and raw.get('gold') is None
  source=read(q/'qa_source.json')
  transcript=(q/'qa_transcript.txt').read_text()
  # Full source and source text must agree; no token-driven truncation.
  assert all(t['text'] in transcript and t['turn_ref'] in transcript for t in source['turns'])
  messages=build_messages(slot,raw,transcript)
  payload=build_payload(profile,messages)
  cap=annotate_school_capacity(capacity_check(profile,tokenizer,messages,MAX_OUTPUT),profile)
  assert cap['status']=='PASS','LOCAL_CAPACITY_EXCEEDED_NO_TRUNCATION'
  folder=RUN/'frozen'/slot['slot_id']
  for name,value in [('slot.json',slot),('original_raw.json',raw),('source.json',source),('messages.json',messages),('payload.json',payload),('capacity.json',cap)]:atomic_json(folder/name,value)
  atomic_text(folder/'prompt.txt',messages[0]['content'])
  atomic_text(folder/'transcript.txt',transcript)
  all_sources.update((q/'candidate_pool.json',raw_path,q/'qa_source.json',q/'qa_transcript.txt'))
  capacities.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],**cap})
  slots.append({**slot,'original_request_id':c['admission_request_id'],'original_raw_path':str(raw_path),'original_pool_path':str(q/'candidate_pool.json'),'original_gold_not_sent':c['gold']})
 # Snapshot pre-existing diagnosis solely for reporting. This file never goes
 # into build_messages, and is not consulted during dispatch or materialization.
 reference=SOURCE.parent/'reports/CONTENT_ALL_ROWS.json'
 rows=read(reference)
 evaluation=[r for r in rows if r['candidate_id'] in {s['candidate_id'] for s in SLOTS}]
 assert len(evaluation)==8
 atomic_json(RUN/'evaluation_only/frozen_prior_codex_diagnosis.json',evaluation)
 all_sources.add(reference)
 atomic_json(RUN/'frozen/slots.json',slots)
 plan={'version':VERSION,'user_authorization':'那调用qwen3.6试试事后检查和事实绑定？错误率如何',
  'scope':'All 8 original program-accepted ordering candidates from last run; 1 request each; fixed order; no selective success sampling.',
  'maximum_http_attempts':MAX_HTTP,'maximum_logical_calls':MAX_HTTP,'automatic_retries':0,'repair_calls':0,'automatic_resume':False,
  'failure_policy':'Any transport failure/non-stop/unknown completion halts all unsent requests. Invalid JSON halts. A completed schema-invalid review is recorded and not retried; continue next fixed slot.',
  'parameters':profile['request_parameters'],'reasoning':'SERVICE_DEFAULT_UNKNOWN','actual_parameter_adoption':'UNKNOWN','cost':'UNKNOWN',
  'max_output_tokens_total':MAX_OUTPUT*MAX_HTTP,'estimated_input_tokens_total':sum(c['input_tokens'] for c in capacities),
  'input_identity':'Original raw candidate and complete unmodified source; no Codex diagnosis/correction, no existing Gold, no new Memory.',
  'immutable_fields':'Four item_ids/item_texts/actors/states preserved; only evidence and temporal relations change; minor stem clarification may be proposed.',
  'evaluation':'Report source-supported surviving questions, missed bad/uncertain items, lost recoverable items, ID validity, binding support, graph consistency, and total tokens. Frozen Codex judgments are a fallible reference, not human Gold.',
  'no_other_stages':['Planner','Gate','Outline','Realizer','Miner','Writer_regeneration','Solver','Training','Memory_API','extra_audit_model'],
  'capacity':capacities,'service_capacity':'UNKNOWN_LOCAL_PROXY_GUARD_ONLY','no_truncation':True}
 atomic_json(RUN/'frozen/plan.json',plan)
 atomic_text(RUN/'frozen/AUTHORIZATION_AND_BUDGET.md','用户本轮明确要求调用Qwen3.6试事后检查。范围由当前讨论的8道排序确定；每题1次，共最多8 HTTP。本轮新预算不结转旧实验余额。无测试请求、重试、修复调用或技术回退；停止条件见plan.json。8是本次执行范围推导的硬上限，不冒称用户逐字给出该数字。\n')
 for module in list(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   p=Path(name).resolve()
   if p.is_relative_to(WT/'scripts') and p.suffix=='.py':all_sources.add(p)
 all_sources.update(p for p in Path(old['tokenizer_path']).iterdir() if p.is_file())
 all_sources.update(p for p in (RUN/'frozen').rglob('*') if p.is_file())
 all_sources.add(RUN/'evaluation_only/frozen_prior_codex_diagnosis.json')
 manifest={'version':VERSION,'status':'PREPARED','prepared_at':now(),'tokenizer_path':old['tokenizer_path'],'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(all_sources)]}
 atomic_json(RUN/'freeze_manifest.json',manifest)
 atomic_json(RUN/'run_status.json',{'status':'PREPARED','http_attempts':0,'maximum_http_attempts':MAX_HTTP})
 return {'status':'PREPARED','slots':len(slots),'estimated_input_tokens':plan['estimated_input_tokens_total'],'maximum_output_tokens':MAX_OUTPUT*MAX_HTTP,'frozen_files':len(all_sources)}

def execute(approved_http_budget):
 if approved_http_budget!=MAX_HTTP:raise ValueError('EXACT_EIGHT_HTTP_BOUND_REQUIRED')
 manifest=read(RUN/'freeze_manifest.json');verify_frozen(manifest)
 with (RUN/'EXECUTION_STARTED.json').open('x') as h:json.dump({'started_at':now(),'http_hard_cap':MAX_HTTP,'automatic_resume':False},h)
 profile=read(RUN/'frozen/provider_profile.json')
 tokenizer=tokenizer_from(manifest['tokenizer_path'])
 slots=read(RUN/'frozen/slots.json')
 assert len(slots)==MAX_HTTP
 results=[];stop_reason=None
 def guard(root,stage,prompt,maximum):
  verify_frozen(manifest)
  if maximum!=MAX_OUTPUT or prompt.get('messages')!=read(RUN/'frozen'/stage/'messages.json'):raise DevelopmentRuntimeError('FROZEN_DISPATCH_CHANGED')
 with PostcheckCaller(RUN/'execution',profile,tokenizer,before_stage=guard) as caller:
  for slot in slots:
   frozen=RUN/'frozen'/slot['slot_id']
   print(json.dumps({'event':'START','slot':slot['slot_id'],'candidate':slot['candidate_id']},ensure_ascii=False),flush=True)
   atomic_json(RUN/'run_status.json',{'status':'RUNNING','current_slot':slot['slot_id'],'completed_reviews':len(results),'http_attempts':caller.state['http_attempt_count'],'maximum_http_attempts':MAX_HTTP})
   try:
    parsed=caller.call(slot['root_id'],slot['slot_id'],{'messages':read(frozen/'messages.json'),'prompt':(frozen/'prompt.txt').read_text(),'prompt_version':VERSION},MAX_OUTPUT)
   except (DevelopmentRuntimeError,DevelopmentContentError) as e:
    stop_reason=e.status;break
   try:
    outcome=materialize(parsed,slot,read(frozen/'original_raw.json'),read(frozen/'source.json'))
   except Exception as e:
    # No exception text or provider inputs in public errors. No second call.
    outcome={'review_validation':{'status':'FAIL','errors':['LOCAL_MATERIALIZATION_FAILURE:'+type(e).__name__]},'program_acceptance':False}
    stop_reason='LOCAL_MATERIALIZATION_FAILURE'
   atomic_json(RUN/'results'/slot['slot_id']/'outcome.json',outcome)
   results.append({'slot_id':slot['slot_id'],'candidate_id':slot['candidate_id'],'model_decision':parsed.get('decision'),'review_validation':outcome['review_validation'],'program_acceptance':outcome['program_acceptance'],'derived_gold':outcome.get('derived_gold')})
   print(json.dumps({'event':'DONE',**results[-1]},ensure_ascii=False),flush=True)
   if stop_reason:break
  records=caller.records
  if not stop_reason:
   caller.state['status']='FINISHED';caller.state['finished_at']=now();caller._save()
 summary={'version':VERSION,'status':'STOPPED_TECHNICAL_OR_CONTENT_FAILURE' if stop_reason else 'GENERATION_FINISHED','stop_reason':stop_reason,
  'completed_review_count':len(results),'maximum_http_attempts':MAX_HTTP,'results':results,'calls':records,'cost':cost_summary(records),
  'semantic_evaluation':'PENDING_CODEX_OFFLINE_REVIEW','automatic_next_experiment':False}
 atomic_json(RUN/'generation_summary.json',summary)
 atomic_json(RUN/'run_status.json',{'status':summary['status'],'stop_reason':stop_reason,'http_attempts':summary['cost']['http_attempts'],'completed_reviews':len(results)})
 return {k:summary[k] for k in ['status','completed_review_count','stop_reason','cost']}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','execute']);parser.add_argument('--approved-http-budget',type=int)
 args=parser.parse_args()
 print(json.dumps(prepare() if args.action=='prepare' else execute(args.approved_http_budget),ensure_ascii=False,indent=2),flush=True)
