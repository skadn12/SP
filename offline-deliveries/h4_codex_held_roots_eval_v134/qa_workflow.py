"""Append-only v133 scope bridge to unchanged v124 question composition."""
from pathlib import Path
import json,hashlib,sys,datetime
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
PRIOR=B.parent/'h4_codex_held_roots_retry_v133'
sys.path.insert(0,str(B/'scripts'))
import codex_sequential_questions as native
SCENES=tuple(f'R02:{r}:S{n:02}' for r in 'AB' for n in range(1,4))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def row(rid):return next(r for r in read(B/'ROOT_MANIFEST.json')['roots'] if r['root_id'].rsplit('-',1)[-1]==str(rid).rsplit('-',1)[-1])
def unit(rid):return B/'units'/('root'+str(rid).rsplit('-',1)[-1])
def folder(rid):return unit(rid)/'roots'/('root'+str(rid).rsplit('-',1)[-1])
def source(rid):return read(folder(rid)/'CUMULATIVE_SOURCE.json')
def verify(rows):
 for r in rows:assert sha(r['path'])==r['sha256'],('BOUND_FILE_CHANGED',r['path'])
def guard(rid,who,role):
 r=row(rid);assert who==r[role],('ASSIGNED_ROLE_REQUIRED',who,role)
 verify(read(B/'MASTER_FREEZE.json')['files'])
 if r['split']=='test':
  closed=read(B/'DEV_QA_CLOSED.json');assert closed['status']=='FROZEN'
  verify(closed['files'])
 return r
def ready(f):
 rid=f.name.removeprefix('root');r=row(rid);u=Path(r['v133_unit'])
 assert sha(f/'CUMULATIVE_SOURCE.json')==r['source_sha256']
 assert read(u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')['decision']=='HOLD'
 assert read(u/'SAFE_PLAN_REVIEW.json')['decision']=='READY'
 previous=read(u/'inputs/BASELINE.json')['turns']
 for s in SCENES:
  d=u/'roots'/('root'+rid)/'scenes'/s.removeprefix('R02:').replace(':','_')
  draft=read(d/'EXPANSION_DRAFT.json');src=read(d/'CUMULATIVE_SOURCE.json');rv=read(d/'SOURCE_REVIEW.json')
  assert src['turns']==previous+draft['turns'];previous=src['turns']
  assert rv['decision']=='READY' and rv['source_sha256']==sha(d/'CUMULATIVE_SOURCE.json') and rv['baseline_conflicts_resolved'] is False
  assert rv['reviewer']!=draft['author']
  receipts=[read(p) for p in (d/'receipts').glob('*.json')]
  assert sum(x['status']=='REVIEW_READY' for x in receipts)==1
  assert sum(x['status']=='STRUCTURAL_PASS' for x in receipts)==1
  assert not any(x['status'] in {'STRUCTURAL_FAILURE','INTEGRITY_FAILURE','REVIEW_INVALID','REVIEW_HOLD'} for x in receipts)
 assert previous==read(f/'CUMULATIVE_SOURCE.json')['turns']
 return read(f/'CUMULATIVE_SOURCE.json')
native.source_ready=ready
def citations(x,src):
 lookup={t['turn_ref']:t for t in src['turns']}
 def walk(v):
  if isinstance(v,dict):
   if {'turn_ref','speaker','quote'}<=set(v):
    t=lookup[v['turn_ref']];assert t.get('scene_ref') in SCENES,'EVIDENCE_OUTSIDE_NEW_SCOPE'
    assert v['speaker']==t['speaker'] and v['quote'] and v['quote'] in t['text'],'CITATION_MISMATCH'
   for y in v.values():walk(y)
  elif isinstance(v,list):
   for y in v:walk(y)
 walk(x)
def context(rid,who,stage):
 role='author' if stage=='author' else 'reviewer';r=guard(rid,who,role);src=ready(folder(rid));u=Path(r['v133_unit'])
 files=[folder(rid)/'CUMULATIVE_SOURCE.json',u/'SAFE_PLAN.json',u/'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json',B/'PROTOCOL_ZH.md']
 if role=='reviewer':
  assert stage in ('assets','public');files.append(folder(rid)/('PUBLIC_ASSETS.json' if stage=='assets' else 'PUBLIC_QUESTIONS.json'))
 text='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in src['turns'])
 text+='\n\n'+'\n\n'.join(p.name+'\n'+p.read_text() for p in files[1:])
 d=unit(rid)/'reads'/stage
 if not (d/'READ_CONTEXT.json').exists():
  d.mkdir(parents=True,exist_ok=True)
  with (d/'FULL_INPUT.txt').open('x') as f:f.write(text+'\n')
  save(d/'READ_CONTEXT.json',{'actual_reader':who,'stage':stage,'created_at_utc':now(),'files':[{'path':str(p),'sha256':sha(p)} for p in files],'turn_refs_read':[t['turn_ref'] for t in src['turns']],'private_answers_exposed':False if role=='reviewer' else None})
 else:
  c=read(d/'READ_CONTEXT.json');assert c['actual_reader']==who;verify(c['files'])
 return text
def author_read(rid,who):
 guard(rid,who,'author');c=read(unit(rid)/'reads/author/READ_CONTEXT.json');assert c['actual_reader']==who;verify(c['files'])
def _save_anchors(rid,who,p):
 author_read(rid,who);src=ready(folder(rid));lookup={t['turn_ref']:t for t in src['turns']}
 assert p['source_sha256']==sha(folder(rid)/'CUMULATIVE_SOURCE.json') and len(p['slots'])==2
 assert len({str(s['slot_id']) for s in p['slots']})==2
 for s in p['slots']:
  aa=s['anchors'];assert len(aa)==6 and len({a['anchor_id'] for a in aa})==6 and len({a['turn_ref'] for a in aa})==6
  assert {lookup[a['turn_ref']]['scene_ref'] for a in aa}==set(SCENES)
  for a in aa:
   t=lookup[a['turn_ref']];assert a['speaker']==t['speaker'] and a['full_turn_quote']==t['text']
 assert {a['turn_ref'] for a in p['slots'][0]['anchors']}!={a['turn_ref'] for a in p['slots'][1]['anchors']},'TWO_DISTINCT_ANCHOR_SETS_REQUIRED'
 save(folder(rid)/'ORDERING_ANCHOR_PLAN.json',{**p,'actual_author':who,'created_at_utc':now()})
def save_anchors(rid,who,p):
 author_read(rid,who);f=folder(rid);assert not (f/'QA_AUTHOR_FAILURE.json').exists()
 save(f/'AUTHOR_ANCHOR_PAYLOAD.json',p)
 try:_save_anchors(rid,who,p)
 except Exception as e:
  save(f/'QA_AUTHOR_FAILURE.json',{'stage':'anchors','error_type':type(e).__name__,'error':str(e),'created_at_utc':now(),'payload_sha256':sha(f/'AUTHOR_ANCHOR_PAYLOAD.json')});raise
def save_draft(rid,who,p):
 author_read(rid,who);f=folder(rid);assert not (f/'QA_AUTHOR_FAILURE.json').exists();assert (f/'ORDERING_ANCHOR_PLAN.json').exists(),'ANCHORS_FIRST'
 save(f/'AUTHOR_QA_PAYLOAD.json',p)
 try:
  assert set(p)=={'stem','facts','ordering'}
  assert len(p['facts'])==6 and len({x['fact_id'] for x in p['facts']})==6
  lookup={t['turn_ref']:t for t in source(rid)['turns']}
  assert {lookup[x['source_turn_ref']]['scene_ref'] for x in p['facts']}==set(SCENES)
  assert all(x.get('true_text') and (x.get('false_text') is None or x['false_text']) for x in p['facts'])
  assert len(p['ordering'])==2 and all(s in p['stem'] for s in SCENES)
  save(f/'QA_DRAFT.json',p)
  native.publish_assets(unit(rid))
  public=read(f/'PUBLIC_ASSETS.json')
  assert set(public)=={'root_id','source_sha256','scope_scene_refs','multiselect_stem','statements','ordering_questions','ordering_failures'}
  save(f/'AUTHOR_QA_RECEIPT.json',{'actual_author':who,'created_at_utc':now(),'attempt':1,'semantic_repairs':0,'full_source_read':True,'current_solver_results_read':False,'baseline_conflicts_resolved':False,'payload_sha256':sha(f/'AUTHOR_QA_PAYLOAD.json'),'anchor_plan_sha256':sha(f/'ORDERING_ANCHOR_PLAN.json')})
 except Exception as e:
  save(f/'QA_AUTHOR_FAILURE.json',{'error_type':type(e).__name__,'error':str(e),'created_at_utc':now(),'payload_sha256':sha(f/'AUTHOR_QA_PAYLOAD.json')});raise
def review_header(rid,who,stage,p):
 r=guard(rid,who,'reviewer');f=folder(rid);c=read(unit(rid)/'reads'/stage/'READ_CONTEXT.json');assert c['actual_reader']==who;verify(c['files'])
 assert p['quarantine_compliance']['status'] in ('PASS','FAIL') and p['quarantine_compliance']['reason']
 findings=p.get('findings',[]);assert isinstance(findings,list) and all(isinstance(x,dict) and x.get('severity') in ('NONBLOCKING','BLOCKING') and (x.get('issue') or x.get('description')) for x in findings)
 citations(p,source(rid))
 name='PUBLIC_ASSETS.json' if stage=='assets' else 'PUBLIC_QUESTIONS.json'
 return {**p,'actual_reviewer':who,'reviewer':who,'created_at_utc':now(),'source_sha256':sha(f/'CUMULATIVE_SOURCE.json'),'public_sha256':sha(f/name),'full_source_read':True,'turn_refs_read':[t['turn_ref'] for t in source(rid)['turns']],'private_answers_read_before_save':False,'independent_of_question_author':True,'baseline_conflicts_resolved':False,**({'asset_judgments_sha256':sha(f/'ASSET_SOURCE_JUDGMENTS.json')} if stage=='public' else {})}
def precheck_order(q,j,src):
 assert all(isinstance(j.get(k),bool) for k in ('stem_valid','scope_valid','coherent','unique_order','neutral_stem_ok'))
 assert set(j.get('items',{}))==set(q['items'])
 for label,x in j['items'].items():
  assert {'proposition','supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok','order_anchor','support_refs','alternative_matching_anchors','reason'}<=set(x)
  refs=x['support_refs'];lookup={t['turn_ref']:t for t in src['turns']};assert refs and all(lookup[ref].get('scene_ref') in SCENES for ref in refs)
  assert all(isinstance(x[k],bool) for k in ('supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok'))
  assert isinstance(x['alternative_matching_anchors'],list) and x['reason'] and x['proposition']==q['items'][label]
  assert {'turn_ref','speaker','quote'}<=set(x['order_anchor'])
  assert x['order_anchor']['turn_ref'] in refs,'ORDER_ANCHOR_NOT_IN_SUPPORT_REFS'
  for anchor in [x['order_anchor'],*x['alternative_matching_anchors']]:
   assert isinstance(anchor,dict) and {'turn_ref','speaker','quote'}<=set(anchor),'ANCHOR_CITATION_SCHEMA'
 assert isinstance(j.get('derived_order'),list)
def precheck_statement(j,text,lookup):
 assert j.get('proposition')==text and j.get('relation') in ('supports','contradicts','unknown')
 assert isinstance(j.get('scope_valid'),bool) and j.get('reason') and j.get('citations')
 assert isinstance(j['citations'],list)
 if len(j['citations'])>1:assert j.get('primary_citation') in j['citations'],'EXPLICIT_PRIMARY_CITATION_REQUIRED'
 elif 'primary_citation' in j:assert j['primary_citation'] in j['citations'],'PRIMARY_CITATION_NOT_REVIEWED'
 for c in j['citations']:
  t=lookup[c['turn_ref']];assert c['speaker']==t['speaker'] and c['quote'] and c['quote'] in t['text']
def content_hold(rid,p,stage):
 if p['quarantine_compliance']['status']=='FAIL' or any(x['severity']=='BLOCKING' for x in p.get('findings',[])):
  save(folder(rid)/'QA_CONTENT_HOLD.json',{'stage':stage,'created_at_utc':now(),'findings':p.get('findings',[]),'quarantine_compliance':p['quarantine_compliance'],'content_pass':0});return True
 return False
def review_assets(rid,who,p):
 f=folder(rid);value=review_header(rid,who,'assets',p);a=read(f/'PUBLIC_ASSETS.json');src=source(rid);lookup={t['turn_ref']:t for t in src['turns']}
 assert set(p['statements'])=={x['statement_id'] for x in a['statements']}
 for x in a['statements']:
  precheck_statement(p['statements'][x['statement_id']],x['proposition'],lookup)
 ordering_ids=[x['candidate_id'] for x in p['ordering_questions']]
 assert len(ordering_ids)==len(set(ordering_ids))==len(a['ordering_questions']),'DUPLICATE_OR_MISSING_ORDERING_REVIEW'
 assert set(ordering_ids)=={x['candidate_id'] for x in a['ordering_questions']}
 for q in a['ordering_questions']:precheck_order(q,next(x for x in p['ordering_questions'] if x['candidate_id']==q['candidate_id']),src)
 save(f/'ASSET_SOURCE_JUDGMENTS.json',value)
 if content_hold(rid,p,'assets'):return
 native.compose(unit(rid))
def review_public(rid,who,p):
 f=folder(rid);value=review_header(rid,who,'public',p);qs=read(f/'PUBLIC_QUESTIONS.json')['questions'];src=source(rid);lookup={t['turn_ref']:t for t in src['turns']}
 assert set(p['questions'])=={q['candidate_id'] for q in qs}
 for q in qs:
  j=p['questions'][q['candidate_id']]
  if q['question_type']=='ordering':precheck_order(q,j,src)
  else:
   assert all(isinstance(j.get(k),bool) for k in ('stem_valid','scope_valid','fact_distinct','no_entangled_options'))
   assert set(j['options'])==set('ABCDEF')
   for label,text in q['options'].items():
    x=j['options'][label];assert isinstance(x.get('fact_distinct'),bool)
    precheck_statement(x,text,lookup)
 save(f/'PUBLIC_SOURCE_JUDGMENTS.json',value)
 if content_hold(rid,p,'public'):return
 native.finalize(unit(rid))
def status(rid):
 f=folder(rid)
 if (f/'QA_CONTENT_HOLD.json').exists():return {'status':'QA_CONTENT_HOLD','content_pass':0}
 if (f/'FINAL_CROSSCHECK.json').exists():
  x=read(f/'FINAL_CROSSCHECK.json')['results'];return {'status':'QA_TERMINAL','content_pass':sum(r['status']=='CONTENT_PASS' for r in x),'excluded':sum(r['status']!='CONTENT_PASS' for r in x),'results':x}
 if (f/'QA_AUTHOR_FAILURE.json').exists():return {'status':'QA_AUTHOR_FAILED','content_pass':0}
 return {'status':'IN_PROGRESS','public_assets':(f/'PUBLIC_ASSETS.json').exists(),'public_questions':(f/'PUBLIC_QUESTIONS.json').exists()}
