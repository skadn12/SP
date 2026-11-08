"""Read v124 artifacts and export fixed-denominator ledgers; never certify semantics."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,sys
B=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'tools'));sys.path.insert(0,str(B/'scripts'))
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def classify(row):
 u=Path(row['unit_path']);f=u/'roots'/u.name;c=u/'inputs/CONTINUATION_CONSTRAINTS.json'
 out={'root_id':row['root_id'],'split':row['split'],'cohort':row['cohort'],'terminal_state':None,'status':'BASELINE_PENDING','terminal_evidence_bindings':[],'ready_scene_count':0,'content_pass_questions':0,'generated_scene_count':len(list(f.glob('scenes/*/EXPANSION_DRAFT.json')))}
 if c.exists():
  if read(c)['decision']=='HOLD':out.update(status='BASELINE_HOLD',terminal_state='BASELINE_HOLD',terminal_evidence_bindings=[bind(c)]);return out
 elif not (u/'RUN_CONFIG.json').exists():return out
 p=u/'evidence/MATCHING/MATCHING_FAILURE.json'
 if p.exists():out.update(status='MATCHING_HOLD',terminal_state='MATCHING_HOLD',terminal_evidence_bindings=[bind(p)]);return out
 if not (u/'RUN_CONFIG.json').exists():out['status']='MATCHING_PENDING';return out
 import scene_controller_v120 as scene
 s=scene.inspect_root(u,scene.load_config(u),row['root_id'],record_reviews=False);out.update(scene_state=s,ready_scene_count=s['ready_scene_count'],status='EXPANSION_PENDING')
 if s['root_stopped']:
  evidence=[bind(p) for pattern in ['scenes/*/SOURCE_REVIEW.json','scenes/*/AUTHOR_FAILURE.json','scenes/*/receipts/*.json'] for p in sorted(f.glob(pattern))]
  out.update(status='EXPANSION_STOPPED',terminal_state='EXPANSION_STOPPED',terminal_evidence_bindings=evidence);return out
 if not s['full_expansion_complete']:return out
 out['status']='QA_PENDING'
 failures=[p for n in ['ORDERING_ANCHOR_FAILURE.json','QA_AUTHOR_FAILURE.json','QA_REVIEW_FAILURE.json','ROOT_TERMINAL.json'] if (p:=f/n).exists()]
 if failures:out.update(status='QA_FAILED',terminal_state='QA_FAILED',terminal_evidence_bindings=[bind(p) for p in failures]);return out
 p=f/'FINAL_CROSSCHECK.json'
 if p.exists():
  d=read(p);out.update(status='QA_COMPLETE',terminal_state='QA_COMPLETE',terminal_evidence_bindings=[bind(p)],content_pass_questions=sum(q['status']=='CONTENT_PASS' for q in d['results']),question_results=d['results'])
 return out
def states(split=None):return [classify(r) for r in read(B/'MANIFEST.json')['roots'] if split is None or r['split']==split]
def accounting(split):
 rows=states(split);assert all(r['terminal_state'] for r in rows),'NONTERMINAL_ROOTS:'+str([(r['root_id'],r['status']) for r in rows if not r['terminal_state']])
 p=B/'evidence'/f'{split.upper()}_TERMINAL_ACCOUNTING.json'
 save(p,{'schema_version':f'v124_{split}_terminal_accounting_v1','status':f'ALL_{split.upper()}_TERMINAL','created_at_utc':now(),'solver_run':False,'score_based_selection':False,'roots':rows});print(str(p))
def export(split):
 from codex_evidence_export import export_root_questions
 rows=states(split);assert all(r['terminal_state'] for r in rows),'split not terminal'
 D=B/'delivery'/split;D.mkdir(parents=True,exist_ok=True);meta={r['root_id']:r for r in read(B/'MANIFEST.json')['roots']};exports=[];lengths=[]
 for row in rows:
  m=meta[row['root_id']];u=Path(m['unit_path']);f=u/'roots'/u.name;latest=Path(m['baseline_path'])
  for key in ['A_S01','A_S02','A_S03','B_S01','B_S02','B_S03']:
   p=f/'scenes'/key/'CUMULATIVE_SOURCE.json'
   if p.exists():latest=p
  if (f/'CUMULATIVE_SOURCE.json').exists():latest=f/'CUMULATIVE_SOURCE.json'
  source=read(latest);assert source['turns'][:m['baseline_turn_count']]==read(m['baseline_path'])['turns'],'baseline mutated'
  qs=export_root_questions(f) if row['terminal_state']=='QA_COMPLETE' else []
  for q in qs:q.update(split=split,dialogue_path=str(latest),dialogue_sha256=bind(latest)['sha256'],evaluation_only=True,solver_evaluated=False)
  exports.extend(qs)
  pub=[{k:v for k,v in q.items() if k not in ['gold','option_evidence','item_evidence','order_anchor_spans','role','evidence_export']} for q in qs]
  save(D/f'{u.name}_PUBLIC.json',{'root_id':row['root_id'],'split':split,'status':row['terminal_state'],'source_binding':bind(latest),'turns':source['turns'],'questions':pub})
  save(D/f'{u.name}_ANSWERS_AND_EVIDENCE.json',{'root_id':row['root_id'],'split':split,'questions':qs})
  lines=[f'# {u.name}：{split} 对话及题目','',f'终态：{row["terminal_state"]}；已审场：{row["ready_scene_count"]}/6；内容通过题：{len(qs)}。','','## 对话',''];prior=None
  for t in source['turns']:
   sc=t.get('scene_ref')or t.get('session_ref')or'原始对话'
   if sc!=prior:lines+=['',f'### {sc}',''];prior=sc
   lines+=[f'**{t["turn_ref"]} — {t["speaker"]}**：{t["text"]}','']
  lines+=['## 题目、答案与完整证据','']
  for i,q in enumerate(qs,1):
   lines+=[f'### 题{i}：{q["question_type"]}'+('（Gold6对照）' if q.get('role')=='control6' else ''),'',q['question'],'']
   for k,v in q.get('options',q.get('items',{})).items():lines+=[f'- **{k}**：{v}']
   lines+=['','答案：'+(' → ' if q['question_type']=='ordering' else '、').join(q['gold']),'','~~~json',json.dumps(q.get('option_evidence',q.get('item_evidence',{})),ensure_ascii=False,indent=2),'~~~','']
  if not qs:lines+=['本root无可交付题目，终态与失败证据：','','~~~json',json.dumps(row,ensure_ascii=False,indent=2),'~~~']
  with (D/f'{u.name}_DIALOGUE_AND_QUESTIONS.md').open('x') as h:h.write('\n'.join(lines)+'\n')
  lengths.append({'root_id':row['root_id'],'baseline_turns':m['baseline_turn_count'],'latest_materialized_turns':len(source['turns']),'new_turns':len(source['turns'])-m['baseline_turn_count'],'ready_scene_count':row['ready_scene_count'],'source_binding':bind(latest)})
 save(D/'QUESTION_DISPOSITION.json',{'created_at_utc':now(),'roots':rows,'questions':[q for r in rows for q in r.get('question_results',[])]})
 save(D/'QUESTIONS_WITH_ANSWERS.json',{'created_at_utc':now(),'evaluation_only':True,'solver_run':False,'questions':exports})
 totals={'created_at_utc':now(),'split':split,'fixed_roots':len(rows),'terminal_states':dict(collections.Counter(r['terminal_state'] for r in rows)),'roots_six_scenes_ready':sum(r['ready_scene_count']==6 for r in rows),'generated_scenes':sum(r['generated_scene_count'] for r in rows),'ready_scenes':sum(r['ready_scene_count'] for r in rows),'content_pass_questions':len(exports),'question_types':dict(collections.Counter(q['question_type'] for q in exports)),'multi_select_gold_counts':dict(collections.Counter(str(len(q['gold'])) for q in exports if q['question_type']=='multi_select')),'gold6_control_questions':sum(q.get('role')=='control6' for q in exports),'solver_run':False,'training_or_pool_merge':False,'dialogue_lengths':lengths}
 save(D/'TOTALS.json',totals);print(json.dumps(totals,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['status','accounting','export']);p.add_argument('--split',choices=['dev','test']);a=p.parse_args()
 if a.action=='status':print(json.dumps(states(a.split),ensure_ascii=False,indent=2))
 else:assert a.split;globals()[a.action](a.split)
