from pathlib import Path
from datetime import datetime, timezone
import collections, hashlib, json, subprocess, sys
B=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_dev_test_ab_v124')
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'delivery_tools'))
import state_and_export as d
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def now():return datetime.now(timezone.utc).isoformat()
split=sys.argv[1] if len(sys.argv)>1 else 'dev'
assert split in ('dev','test')
roles={x['root_id']:x for x in read(B/'ROLE_ASSIGNMENTS.json')['roots']}
for row in read(B/'MANIFEST.json')['roots']:
 if row['split']!=split:continue
 u=Path(row['unit_path']);f=u/'roots'/u.name
 for action,need,out in [('publish','QA_AUTHOR_RECEIPT.json','PUBLIC_ASSETS.json'),('compose','ASSET_SOURCE_JUDGMENTS.json','PUBLIC_QUESTIONS.json'),('finalize','PUBLIC_SOURCE_JUDGMENTS.json','FINAL_CROSSCHECK.json')]:
  if not (f/need).is_file() or (f/out).exists():continue
  if action!='publish':
   phase='asset' if action=='compose' else 'public'
   receipts=[read(p) for p in (u/'evidence/reads').glob('*/SAVE_RECEIPT.json')]
   receipts=[x for x in receipts if x.get('phase')==phase and x.get('executor_id')==roles[row['root_id']]['reviewer'] and x.get('output_binding')==bind(f/need)]
   if len(receipts)!=1:continue
  e=u/'evidence/PARENT_STAGES';e.mkdir(parents=True,exist_ok=True)
  receipt=e/(action.upper()+'_RUN.json')
  if receipt.exists():continue
  lock=e/(action.upper()+'_RUN.lock')
  try:lock.mkdir()
  except FileExistsError:continue
  if action=='publish':cmd=[sys.executable,'-B',str(B/'tools/question_controller_v120.py'),action,'--batch-dir',str(u)]
  else:cmd=[sys.executable,'-B',str(B/'tools/run_guarded_questions_v2.py'),action,'--batch',str(B),'--root',row['root_id']]
  start=now();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
  value={'schema_version':'v124_parent_mechanical_stage_receipt_v1','executor':'/root','root_id':row['root_id'],'action':action,'started_at_utc':start,'finished_at_utc':now(),'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'input_binding':bind(f/need),'output_binding':bind(f/out) if (f/out).is_file() else None,'semantic_judgment_created':False,'execution_script_binding':bind(Path(__file__))}
  with receipt.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
  lock.rmdir();print(json.dumps({'root':row['root_id'],'action':action,'returncode':p.returncode,'output_exists':(f/out).is_file(),'stderr':p.stderr[:700]},ensure_ascii=False))
rows=d.states(split)
print(json.dumps({'created_at_utc':now(),'split':split,'status_counts':dict(collections.Counter(x['status'] for x in rows)),'ready_scenes':sum(x['ready_scene_count'] for x in rows),'generated_scenes':sum(x['generated_scene_count'] for x in rows),'content_pass_questions':sum(x['content_pass_questions'] for x in rows),'roots':[{'root':x['root_id'].rsplit('-',1)[1],'state':x['status'],'ready':x['ready_scene_count'],'drafts':x['generated_scene_count']} for x in rows]},ensure_ascii=False))
