"""Freeze a four-root serialization validation before source matching/authoring."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, re
HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
D=P/'SP-820-publish/first_training_pipeline/data'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def copy(p,q):
 q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('xb') as f:f.write(p.read_bytes())
def walk(v):
 if isinstance(v,dict):
  for k,x in v.items():
   if k=='source_id' and isinstance(x,str):yield x
   yield from walk(x)
 elif isinstance(v,list):
  for x in v:yield from walk(x)
def main():
 created=datetime.now(timezone.utc).isoformat()
 train=D/'matched_training/m_clean_policy_training_manifest.jsonl'
 train_roots={json.loads(l)['root_id'] for l in train.read_text().splitlines()}
 inv={x['root_id']:x for x in read(D/'qa_generation_only/manifest.json')['roots']}
 originals={x['root_id']:x for x in read(D/'selection_manifest_h4_unexpanded122.json')['records']}
 exposed={};protected_dirs=[]
 for path in sorted(HERE.parent.iterdir()):
  m=re.search(r'_v(\d+)$',path.name)
  if not m or not 104<=int(m.group(1))<=114 or not path.is_dir():continue
  if path.name.startswith(('h4_batch_codex_','h4_codex_','h4_unseen_root_')):
   protected_dirs.append(path)
   cp=path/'RUN_CONFIG.json'; ap=path/'FINAL_BATCH_ACCOUNTING.json'
   if cp.exists(): roots=read(cp).get('roots',[])
   elif ap.exists(): roots=read(ap)['roots']
   else:continue
   for r in roots:
    rid=r['root_id'] if isinstance(r,dict) else r
    exposed.setdefault(rid,[]).append({'path':str(cp if cp.exists() else ap),'sha256':sha(cp if cp.exists() else ap)})
 selected=[]; eligibility=[]
 for rid in sorted(train_roots,key=lambda r:int(r.rsplit('-',1)[1])):
  if rid in {'cc-train-episode-222937','cc-train-episode-225575','cc-train-episode-761','cc-train-episode-9557','cc-train-episode-14304','cc-train-episode-20432'}:continue
  s=read(inv[rid]['source_path']);scenes=sorted({t.get('scene_ref') for t in s['turns'] if t.get('scene_ref')})
  speakers=sorted({t['speaker'] for t in s['turns']})
  ok=scenes==['R01:S01','R01:S02','R01:S03'] and len(speakers)==2
  eligibility.append({'root_id':rid,'eligible':ok,'baseline_scene_refs':scenes,'baseline_speakers':speakers,'reason':'structural_eligibility' if ok else 'requires_three_R01_scenes_and_two_speakers'})
  if ok:selected.append(rid)
  if len(selected)==4:break
 assert selected==['cc-train-episode-29753','cc-train-episode-31457','cc-train-episode-45849','cc-train-episode-55788']
 roots=[]
 for rid in selected:
  item=inv[rid];p=Path(item['source_path']);s=read(p);assert sha(p)==item['source_sha256']
  target=HERE/'inputs'/f"root{rid.rsplit('-',1)[1]}_BASELINE.json";copy(p,target)
  original=read(originals[rid]['input_path'])['source']
  roots.append({'root_id':rid,'baseline_path':str(target),'baseline_sha256':sha(target),'upstream_path':str(p),'upstream_sha256':sha(p),'baseline_turn_count':len(s['turns']),'baseline_speakers':sorted({t['speaker'] for t in s['turns']}),'original_source_id':original['source_id'],'original_source_record_sha256':original['source_record_sha256']})
 holdouts=[D/'matched_training'/x for x in ['dev_full_eval_manifest.jsonl','test_full_eval_manifest.jsonl']]
 holdout_roots={json.loads(l)['root_id'] for path in holdouts for l in path.read_text().splitlines()}
 assert not set(selected)&holdout_roots
 hold_sources=[read(originals[rid]['input_path'])['source'] for rid in holdout_roots]
 for key,sourcekey in [('original_source_id','source_id'),('original_source_record_sha256','source_record_sha256')]:assert not {r[key] for r in roots}&{r[sourcekey] for r in hold_sources}
 consumed=set();scan=[]
 for path in sorted(P.rglob('SOURCE_RESERVATIONS.json')):
  if HERE in path.parents:continue
  ids=sorted(set(walk(read(path))));consumed.update(ids);scan.append({'path':str(path),'sha256':sha(path),'source_ids':ids})
 old=read(HERE.parent/'h4_codex_controlled_batch_v105/evidence/SOURCE_SELECTION.json');consumed.update(old['consumed_source_ids'])
 pool=HERE.parent/'h4_external_event_source_refresh_v1/multiwoz_train_sample.json'
 assert sha(pool)=='2f3ea771d4e01cb2780357738cff7f7496b87d34c221cc240df74501312438d3'
 available=sorted([e for e in read(pool) if 'multiwoz22:train:'+e['dialogue_id'] not in consumed],key=lambda e:e['dialogue_id'])
 groups=[]
 for i,r in enumerate(roots):
  target=HERE/'pre_inputs'/f"root{r['root_id'].rsplit('-',1)[1]}_EVENT_POOL.json"
  es=available[i::4];write(target,{'root_id':r['root_id'],'partition_index_zero_based':i,'partition_count':4,'events':es})
  groups.append({'root_id':r['root_id'],'event_pool_path':str(target),'event_pool_sha256':sha(target),'event_count':len(es),'partition_index_zero_based':i})
 write(HERE/'EXPOSURE_AUDIT.json',{'status':'PASS','purpose':'TRAINING_SIDE_SERIALIZED_A3B3_WORKFLOW_VALIDATION','selected_root_ids':selected,'selected_roots':roots,'already_exposed_training_roots':True,'prior_codex_authoring':{rid:exposed.get(rid,[]) for rid in selected},'unexposed_clean_training_roots_remaining':len(train_roots-set(exposed)),'new_unexposed_generalization_claimed':False,'structural_selection_trace':eligibility,'baseline_has_C1':False,'prior_Codex_scene_text_reused':False,'clean_training_manifest':{'path':str(train),'sha256':sha(train)},'holdout_manifests':[{'path':str(p),'sha256':sha(p)} for p in holdouts],'holdout_root_overlap':[],'holdout_source_id_overlap':[],'holdout_source_record_overlap':[],'new_training_started':False})
 protection={}
 for directory in protected_dirs:
  for path in directory.rglob('*'):
   if path.is_file():protection[str(path)]=sha(path)
 # v112 protected paths include previous production/evaluation/export/delivery artifacts.
 for item in read(HERE.parent/'h4_codex_sequential_ab_pilot_v112/evidence/HISTORICAL_PROTECTION.json')['files']:
  assert sha(item['path'])==item['sha256'];protection[item['path']]=item['sha256']
 write(HERE/'evidence/HISTORICAL_PROTECTION.json',{'created_at_utc':created,'scope':'All files in sealed v104-v114 Codex production/validation directories plus transitive v112 historical protected files','directories':[str(x) for x in protected_dirs],'files':[{'path':p,'sha256':h} for p,h in sorted(protection.items())]})
 plan={'batch_id':HERE.name,'created_at_utc':created,'status':'FROZEN_BEFORE_EVENT_MATCHING_AND_AUTHORING','roots':roots,'groups':groups,'root_selection':'Numeric clean-training order, excluding v112 and v113 roots; original P+R01 must have exactly S01/S02/S03 and two established speakers. 19053 and19708 lack S03 and are structurally ineligible. All selected roots already exposed; no semantic/output/performance filtering or replacements.','scene_sequence':[f'R02:{r}:S{n:02}' for r in 'AB' for n in range(1,4)],'source_selection':'Compatibility matching of one unused restaurant, attraction and train event per root in four disjoint preassigned modulo partitions of sorted native MultiWOZ train events. Full selected native events and baseline read before final allocation; reservations permanent.','source_pool_path':str(pool),'source_pool_sha256':sha(pool),'available_event_count':len(available),'consumed_source_ids':sorted(consumed),'consumption_scan':scan,'new_turn_count':{'min':12,'max':16},'author_attempts_per_scene':1,'semantic_repairs':0,'source_replacements':0,'scene_payload_writer':'scripts/codex_scene_payload.py; authors supply only speaker/text. Metadata serialization is not semantic repair.','qa_author_attempts_per_complete_root':1,'slots_per_ready_root':{'gold2':1,'gold3':1,'gold4':1,'gold5':1,'gold6_control':1,'ordering':2},'planned_new_questions_max':24,'planned_controls_max':4,'solver_protocol':'v15_full_only_random_mcq6x3_order12x1_v1','solver_models':{'trained':'M_clean_policy','base':'M0'},'solver_calls_per_model_max':456,'solver_calls_both_models_max':912,'solver_automatic_retry':False,'selection':'Complete-valid 0 < equal-display partial mean < 1; Gold6 excluded from training candidates.','training_started':False,'training_or_pool_merge_authorized':False,'backend':'CURRENT_CODEX_SESSION','exact_model_id':None,'sampling_parameters':None,'usage_tokens':None,'billed_cost':None,'external_semantic_certification':False,'statistical_unit':'root/family','formal_backend_comparison':False,'primary_metrics':['6READY root / all4 frozen roots','READY scenes /24 and attempted scene count','actor_identity failures among reviewed scenes','content pass and Solver-selected counts by root/type/Gold','wall time and failure categories']}
 write(HERE/'PRECALL_PLAN.json',plan)
 contract=(HERE.parent/'h4_codex_sequential_ab_pilot_v112/evidence/CONTENT_CONTRACT.md').read_text().replace('恢复验证内容合同 v112','新四root流程验证内容合同 v115').replace('两条已暴露训练root的探索性流程恢复','四条已暴露训练root的固定人物流程复测').replace('另一root继续','其他root继续').replace('最大每模型228调用','最大每模型456调用、两模型912调用')
 contract+='\n## v115 固定人物与编号\n- 固定4root尝试分母，不因结果换root；恰两名原speaker，不新增发言人物。服务人员只能作为明确归属的转述，不把朋友、亲属改为服务人员。\n- A1/A2/A3分别延续原R01:S01/S02/S03，每场有新进展，允许原三场关联同一主事件；不把原3场伪称新3场，不复用既有Codex稿。\n- 作者只提交speaker/text对，codex_scene_payload统一构造scene/session/turn_ref和origin，保存一次；结构失败、语义HOLD均停止依赖后续，不重写。\n- 基线原文中已有角色/性别/年龄歧义保持原样列入继承warning；新增内容不能放大矛盾，也不能把猜测当作已知人物事实。\n'
 contract+='\n## v115 无损导出与措辞检查\n- 按v114 WORDING_RULE_ADDENDUM检查总量/单次量/次数及完整范围；仅来源明确支持或反驳的命题进入组合。\n- 私有证据导出保留所有citations、真实relation/reason，逐来源hash/turn_ref/speaker/quote检查，不截断到首引；无损覆盖检查通过后才能冻结评估。\n'
 (HERE/'evidence/CONTENT_CONTRACT.md').write_text(contract)
 print(json.dumps({'roots':selected,'available_events':len(available),'protected_files':len(protection),'precall_sha256':sha(HERE/'PRECALL_PLAN.json')}))
if __name__=='__main__':main()
