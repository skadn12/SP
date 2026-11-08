"""Independent v134 fixture-only preflight; no model, solver or real QA reads."""
import copy, hashlib, importlib.util, json, tempfile, traceback
from pathlib import Path
REAL=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134')
spec=importlib.util.spec_from_file_location('v134_qa_preflight',REAL/'qa_workflow.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SCENES=m.SCENES
RESULTS=[]

def w(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');return p

def fixture(split='dev'):
 b=Path(tempfile.mkdtemp(prefix='v134-frontier-fixture-'));rid='900001';root='cc-train-episode-'+rid
 m.B=b
 u=b/'prior/unit';f=b/'units'/('root'+rid)/'roots'/('root'+rid)
 base={'root_id':root,'turns':[{'turn_ref':'OLD1','scene_ref':'R01:A:S01','speaker':'Old','text':'Old history remains conflicted.'}]}
 w(u/'inputs/BASELINE.json',base);w(u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json',{'decision':'HOLD'});w(u/'SAFE_PLAN_REVIEW.json',{'decision':'READY'});w(u/'SAFE_PLAN.json',{'policy':'conflict quarantine'})
 previous=copy.deepcopy(base['turns'])
 scene_turns=[]
 for i,s in enumerate(SCENES):
  turns=[{'turn_ref':f'N{i}-{j}','scene_ref':s,'speaker':'Ada' if j==0 else 'Ben','text':f'Unique scene {i} event {j}: key {i}-{j} is blue.'} for j in range(2)]
  d=u/'roots'/('root'+rid)/'scenes'/s.removeprefix('R02:').replace(':','_')
  w(d/'EXPANSION_DRAFT.json',{'author':'fixture-expander','turns':turns});previous+=turns
  p=w(d/'CUMULATIVE_SOURCE.json',{'root_id':root,'turns':copy.deepcopy(previous)})
  w(d/'SOURCE_REVIEW.json',{'decision':'READY','source_sha256':m.sha(p),'baseline_conflicts_resolved':False,'reviewer':'fixture-source-reviewer'})
  w(d/'receipts/a.json',{'status':'REVIEW_READY'});w(d/'receipts/b.json',{'status':'STRUCTURAL_PASS'})
  scene_turns.append(turns)
 p=w(f/'CUMULATIVE_SOURCE.json',{'root_id':root,'turns':previous})
 cfg={'controller_version':'codex_sequential_ab_v1','batch_id':'fixture_v134','roots':[{'root_id':root}], 'scene_sequence':list(SCENES),'scope_scene_refs':list(SCENES),'new_turn_count':{'min':12,'max':16},'multiselect_target_cardinalities':[2,3,4,5],'seed_namespace':'fixture'}
 w(b/'units'/('root'+rid)/'RUN_CONFIG.json',cfg)
 w(b/'PROTOCOL_ZH.md',{'fixture_protocol':True})
 w(b/'ROOT_MANIFEST.json',{'roots':[{'root_id':root,'split':split,'author':'fixture-author','reviewer':'fixture-reviewer','source_sha256':m.sha(p),'v133_unit':str(u)}]})
 w(b/'MASTER_FREEZE.json',{'status':'FROZEN','files':[]})
 return b,rid,u,f,scene_turns

def authored(x):
 b,rid,u,f,st=x
 m.context(rid,'fixture-author','author')
 slots=[]
 for j in range(2):slots.append({'slot_id':str(j+1),'anchors':[{'anchor_id':f'A{j}-{i}','turn_ref':ts[j]['turn_ref'],'speaker':ts[j]['speaker'],'full_turn_quote':ts[j]['text']} for i,ts in enumerate(st)]})
 m.save_anchors(rid,'fixture-author',{'source_sha256':m.sha(f/'CUMULATIVE_SOURCE.json'),'slots':slots})
 facts=[{'fact_id':f'F{i}','source_turn_ref':ts[0]['turn_ref'],'true_text':ts[0]['text'],'false_text':ts[0]['text'].replace('is blue','is red')} for i,ts in enumerate(st)]
 ordering=[{'slot_id':str(j+1),'items':[{'item_id':f'I{i+1}','anchor_id':f'A{j}-{i}','text':ts[j]['text'],'support_quote':ts[j]['text']} for i,ts in enumerate(st)]} for j in range(2)]
 m.save_draft(rid,'fixture-author',{'stem':'Select supported claims in '+', '.join(SCENES),'facts':facts,'ordering':ordering})
 return facts

def cit(t):return {k:t[k] for k in ['turn_ref','speaker']}|{'quote':t['text']}
def stmt(text,st):
 t=next(t for ts in st for t in ts if text==t['text'] or text==t['text'].replace('is blue','is red'))
 return {'proposition':text,'relation':'supports' if text==t['text'] else 'contradicts','scope_valid':True,'reason':'Explicit fixture assertion or opposite color.','citations':[cit(t)]}
def order(q,st):
 turns=[t for ts in st for t in ts];items={}
 for label,text in q['items'].items():
  t=next(t for t in turns if t['text']==text)
  items[label]={'proposition':text,**{k:True for k in ['supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok']},'order_anchor':cit(t),'support_refs':[t['turn_ref']],'alternative_matching_anchors':[],'reason':'Exact unique fixture event.'}
 positions={t['turn_ref']:i for i,t in enumerate(turns)}
 return {'candidate_id':q['candidate_id'],**{k:True for k in ['stem_valid','scope_valid','coherent','unique_order','neutral_stem_ok']},'items':items,'derived_order':sorted(items,key=lambda l:positions[items[l]['order_anchor']['turn_ref']]),'alternative_orders':[]}
def apayload(x):
 b,rid,u,f,st=x;m.context(rid,'fixture-reviewer','assets');a=m.read(f/'PUBLIC_ASSETS.json')
 return {'quarantine_compliance':{'status':'PASS','reason':'Fixture uses new-scope evidence only.'},'findings':[],'statements':{s['statement_id']:stmt(s['proposition'],st) for s in a['statements']},'ordering_questions':[order(q,st) for q in a['ordering_questions']]}
def ppayload(x):
 b,rid,u,f,st=x;m.context(rid,'fixture-reviewer','public');qs=m.read(f/'PUBLIC_QUESTIONS.json')['questions'];jj={}
 for q in qs:
  if q['question_type']=='ordering':j=order(q,st)
  else:j={**{k:True for k in ['stem_valid','scope_valid','fact_distinct','no_entangled_options']},'options':{l:stmt(text,st)|{'fact_distinct':True} for l,text in q['options'].items()}}
  jj[q['candidate_id']]=j
 return {'quarantine_compliance':{'status':'PASS','reason':'Fixture uses new-scope evidence only.'},'findings':[],'questions':jj}

def record(name,fn):
 try:
  detail=fn();RESULTS.append({'test':name,'status':'PASS','detail':detail})
 except Exception as e:RESULTS.append({'test':name,'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()})

def full_happy():
 x=fixture();b,rid,u,f,st=x;assert m.native.source_ready is m.ready;authored(x);m.review_assets(rid,'fixture-reviewer',apayload(x));assert not (f/'PRIVATE_GOLD.json').exists();m.review_public(rid,'fixture-reviewer',ppayload(x));assert m.status(rid)['content_pass']==7
 import evidence_export
 out=evidence_export.export_root_questions(f);assert len(out)==7
 return {'content_pass':7,'export_count':len(out),'historical_hold':m.read(u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')['decision'],'native_bridge':True}
record('happy_path_seven_questions_lossless_export',full_happy)

def false_unknown():
 x=fixture();b,rid,u,f,st=x;authored(x);p=apayload(x)
 for j in p['statements'].values():
  if j['relation']=='contradicts':j['relation']='unknown'
 p['ordering_questions'][0]['stem_valid']=False
 m.review_assets(rid,'fixture-reviewer',p)
 receipt=m.read(f/'COMPOSITION_RECEIPT.json');assert receipt['public_count']==2 and len(receipt['unfilled_slots'])==5
 return {'public_count':receipt['public_count'],'unfilled':len(receipt['unfilled_slots']),'judgments_saved':(f/'ASSET_SOURCE_JUDGMENTS.json').exists()}
record('asset_unknown_and_false_flag_saved_then_native_excluded',false_unknown)

def final_false_unknown():
 x=fixture();b,rid,u,f,st=x;authored(x);m.review_assets(rid,'fixture-reviewer',apayload(x));p=ppayload(x)
 qs=m.read(f/'PUBLIC_QUESTIONS.json')['questions'];ms=next(q for q in qs if q['question_type']=='multi_select');oq=next(q for q in qs if q['question_type']=='ordering')
 p['questions'][ms['candidate_id']]['options']['A']['relation']='unknown';p['questions'][oq['candidate_id']]['coherent']=False
 m.review_public(rid,'fixture-reviewer',p);status=m.status(rid);assert status['content_pass']==5 and status['excluded']==2
 return {'content_pass':5,'excluded':2,'original_gold_unchanged':all(r['original_gold_unchanged'] for r in status['results'])}
record('final_unknown_and_false_flag_recorded_then_excluded',final_false_unknown)

def blocking(stage):
 x=fixture();b,rid,u,f,st=x;authored(x)
 if stage=='assets':p=apayload(x)
 else:m.review_assets(rid,'fixture-reviewer',apayload(x));p=ppayload(x)
 p['findings']=[{'severity':'BLOCKING','issue':'Synthetic semantic blocker.'}]
 getattr(m,'review_'+stage)(rid,'fixture-reviewer',p)
 assert m.status(rid)['status']=='QA_CONTENT_HOLD'
 assert not (f/('PUBLIC_QUESTIONS.json' if stage=='assets' else 'PRIVATE_GOLD.json')).exists()
 return {'stage':stage,'content_pass':0,'hold_saved':True}
record('blocking_asset_review_holds_before_compose',lambda:blocking('assets'))
record('blocking_public_review_holds_before_gold',lambda:blocking('public'))

def history_hold():
 x=fixture();b,rid,u,f,st=x;m.ready(f);w(u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json',{'decision':'READY'})
 try:m.ready(f)
 except AssertionError:return {'changed_hold_rejected':True}
 raise AssertionError('changed historical HOLD accepted')
record('historical_hold_must_remain_hold',history_hold)

def testgate():
 x=fixture('test');b,rid,u,f,st=x
 try:m.context(rid,'fixture-author','author')
 except FileNotFoundError:pass
 else:raise AssertionError('TEST proceeded before DEV closure')
 bound=w(b/'dev-frozen.json',{'frozen':True});w(b/'DEV_QA_CLOSED.json',{'status':'FROZEN','files':[{'path':str(bound),'sha256':m.sha(bound)}]});m.context(rid,'fixture-author','author');w(bound,{'frozen':False})
 try:m.author_read(rid,'fixture-author')
 except AssertionError:return {'missing_closure_rejected':True,'mutated_dev_binding_rejected':True}
 raise AssertionError('TEST proceeded after DEV binding mutation')
record('test_waits_for_verified_dev_closure',testgate)

def anchor_failure():
 x=fixture();b,rid,u,f,st=x;m.context(rid,'fixture-author','author');bad={'source_sha256':'bad','slots':[]}
 try:m.save_anchors(rid,'fixture-author',bad)
 except AssertionError:pass
 else:raise AssertionError('bad anchors accepted')
 assert m.read(f/'AUTHOR_ANCHOR_PAYLOAD.json')==bad and m.status(rid)['status']=='QA_AUTHOR_FAILED'
 try:m.save_anchors(rid,'fixture-author',bad)
 except AssertionError:return {'raw_first_payload_preserved':True,'failure_terminal':True,'second_attempt_rejected':True}
 raise AssertionError('retry allowed')
record('anchor_failure_preserves_raw_payload_and_terminal_state',anchor_failure)

def malformed_review():
 x=fixture();b,rid,u,f,st=x;authored(x);p=apayload(x);p['findings']=['not a finding object']
 try:m.review_assets(rid,'fixture-reviewer',p)
 except AssertionError:assert not (f/'ASSET_SOURCE_JUDGMENTS.json').exists();return {'pre_save_rejection':True}
 raise AssertionError('malformed review written')
record('malformed_schema_rejected_before_review_write',malformed_review)

def duplicate_order():
 x=fixture();b,rid,u,f,st=x;authored(x);p=apayload(x);duplicate=copy.deepcopy(p['ordering_questions'][0]);duplicate['stem_valid']=False;p['ordering_questions'].append(duplicate)
 try:m.review_assets(rid,'fixture-reviewer',p)
 except AssertionError:assert not (f/'ASSET_SOURCE_JUDGMENTS.json').exists();return {'duplicates_rejected':True}
 raise AssertionError('Duplicate candidate judgments accepted; wrapper prechecks first but native uses last, so audit payload has conflicting judgments.')
record('duplicate_ordering_judgments_rejected_pre_save',duplicate_order)


def multicitation_primary():
 x=fixture();b,rid,u,f,st=x;authored(x);m.review_assets(rid,'fixture-reviewer',apayload(x));p=ppayload(x)
 q=next(q for q in m.read(f/'PUBLIC_QUESTIONS.json')['questions'] if q['question_type']=='multi_select')
 p['questions'][q['candidate_id']]['options']['A']['citations'].append(cit(st[0][1]))
 try:m.review_public(rid,'fixture-reviewer',p)
 except AssertionError:
  assert not (f/'PUBLIC_SOURCE_JUDGMENTS.json').exists();return {'missing_primary_rejected_pre_save':True}
 import evidence_export
 try:evidence_export.export_root_questions(f)
 except ValueError as e:raise AssertionError('Review saved and candidate CONTENT_PASS, but canonical exporter later fails: '+str(e))
 raise AssertionError('Missing explicit multi-citation primary not rejected')
record('multi_citation_requires_recorded_primary_before_save',multicitation_primary)

def anchor_in_support_refs():
 x=fixture();b,rid,u,f,st=x;authored(x);p=apayload(x);p['ordering_questions'][0]['items']['I1']['support_refs']=[st[1][0]['turn_ref']]
 try:m.review_assets(rid,'fixture-reviewer',p)
 except AssertionError:
  assert not (f/'ASSET_SOURCE_JUDGMENTS.json').exists();return {'anchor_not_in_support_refs_rejected':True}
 p=ppayload(x);q=next(q for q in m.read(f/'PUBLIC_QUESTIONS.json')['questions'] if q['question_type']=='ordering')
 p['questions'][q['candidate_id']]['items']['I1']['support_refs']=[st[1][0]['turn_ref']]
 try:m.review_public(rid,'fixture-reviewer',p)
 except AssertionError:
  assert not (f/'PUBLIC_SOURCE_JUDGMENTS.json').exists();return {'anchor_not_in_support_refs_rejected':True}
 import evidence_export
 try:evidence_export.export_root_questions(f)
 except ValueError as e:raise AssertionError('Invalid support/anchor schema passed native CONTENT_PASS, canonical exporter fails: '+str(e))
 raise AssertionError('Missing anchor support link not rejected')
record('ordering_anchor_required_in_support_refs',anchor_in_support_refs)


def explicit_primary_positive():
 x=fixture();b,rid,u,f,st=x;authored(x);m.review_assets(rid,'fixture-reviewer',apayload(x));p=ppayload(x)
 q=next(q for q in m.read(f/'PUBLIC_QUESTIONS.json')['questions'] if q['question_type']=='multi_select');cid=q['candidate_id']
 j=p['questions'][cid]['options']['A'];j['citations'].append(cit(st[0][1]));j['primary_citation']=copy.deepcopy(j['citations'][0]);expected=copy.deepcopy(j['citations'])
 m.review_public(rid,'fixture-reviewer',p)
 import evidence_export
 out=next(q for q in evidence_export.export_root_questions(f) if q['candidate_id']==cid)
 assert out['option_evidence']['A']['citations']==expected
 assert len(out['option_evidence']['A']['citations'])==2 and out['option_evidence']['A']['primary_citation']==expected[0]
 return {'all_citations_preserved':2,'explicit_primary_preserved':True}
record('valid_explicit_multicitation_primary_exports_losslessly',explicit_primary_positive)

def source_negative(mode):
 x=fixture();b,rid,u,f,st=x
 scene=u/'roots'/('root'+rid)/'scenes/A_S01'
 if mode=='local_hash':
  v=m.read(f/'CUMULATIVE_SOURCE.json');v['turns'][0]['text']='changed';w(f/'CUMULATIVE_SOURCE.json',v)
 elif mode=='duplicate_ready':w(scene/'receipts/duplicate.json',{'status':'REVIEW_READY'})
 elif mode=='failed_receipt':w(scene/'receipts/failed.json',{'status':'REVIEW_HOLD'})
 elif mode=='prefix':
  v=m.read(scene/'CUMULATIVE_SOURCE.json');v['turns'][0]['text']='changed';w(scene/'CUMULATIVE_SOURCE.json',v)
 try:m.ready(f)
 except AssertionError:return {'rejected':True,'mutation':mode}
 raise AssertionError('source mutation accepted: '+mode)
for mode in ['local_hash','duplicate_ready','failed_receipt','prefix']:record('source_ready_rejects_'+mode,lambda mode=mode:source_negative(mode))

print(json.dumps({'controller_sha256':m.sha(REAL/'qa_workflow.py'),'fixture_only':True,'results':RESULTS},indent=2,ensure_ascii=False))
Path('/tmp/v134_preflight_frontier_result.json').write_text(json.dumps({'controller_sha256':m.sha(REAL/'qa_workflow.py'),'fixture_only':True,'results':RESULTS},indent=2,ensure_ascii=False)+'\n')
