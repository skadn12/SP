"""Read-only final consistency checks; write only final check reports."""
from pathlib import Path
from collections import Counter
import json,hashlib,re,sys
from urllib.parse import unquote
sys.path.insert(0,'/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live import _load_feedback_context
from scripts.current_method.restart_n5_spell.h4_round1_r03_feedback_view_v2.compiler import expansion_view_v3
D=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_v1');R=D/'prepared_run'
roots=tuple('cc-train-episode-'+x for x in ('259192','96462','112812','218519'))
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,status='PASS' if ok else 'FAIL',detail=detail))

final=read(D/'reports/FINAL_REPORT.json');allrows=read(D/'reports/CONTENT_ALL_ROWS.json')
ck('42_raw_rows_unique',len(allrows)==42 and len({(r['request_id'],r['raw_row_index']) for r in allrows})==42)
ck('29_final_review_rows',sum(r['in_final_pool'] for r in allrows)==29)
raw_identity=[];total_pool=0;strict_total=0;dedup_total=0;ord_counts=Counter();whole=Counter();link_paths=[];feedback=[]
for root in roots:
 fp=D/'filtered'/root;mp=read(fp/'filter_manifest.json');inputdata=read(R/'frozen'/root/'inputs.json');pp=R/'qa/roots'/root/'candidate_pool.json'
 pool=read(pp) if pp.exists() else [];by={r['candidate_id']:r for r in pool};total_pool+=len(pool)
 decisions=read(fp/'content_review_decisions.json');dec={r['candidate_id']:r for r in decisions}
 if root==roots[-1]:
  ck('218519_not_reached',not pp.exists() and mp['status']=='NOT_REACHED_NOT_EVALUATED' and mp['source_pool'] is None and not decisions)
 else:
  ck('source_pool_sha_'+root,mp['source_pool']==str(pp) and mp['source_pool_sha256']==sha(pp))
  ck('one_decision_per_final_'+root,set(dec)==set(by) and len(dec)==len(decisions))
  report=read(D/'reports'/('CONTENT_'+root.rsplit('-',1)[-1]+'.json'));review=report.get('raw_candidate_reviews',report.get('qa_candidates'))
  for row in review:
   raw=read(row['raw_response_path'])['candidates'][row['raw_row_index']-1]
   ok=raw==row.get('raw_candidate',row.get('original_raw_candidate'))
   if row['in_final_pool'] and 'final_candidate' in row:ok=ok and row['final_candidate']==by[row['candidate_id']]
   raw_identity.append(dict(candidate_id=row['candidate_id'],identity_intact=ok))
 sets={}
 for fn in ('candidate_pool.json','ordering_keep.json','ordering_hold.json','ordering_excluded.json','strict_content_pass_candidates.json','strict_content_pass_deduplicated.json'):
  data=read(fp/fn);sets[fn]=data
  ck('unmodified_'+root+'_'+fn,all(r['candidate_id'] in by and r==by[r['candidate_id']] for r in data) and len(data)==len({r['candidate_id'] for r in data}))
  ck('output_sha_'+root+'_'+fn,mp['output_sha256'][fn]==sha(fp/fn))
 for fn,v in mp['output_sha256'].items():ck('manifest_hash_'+root+'_'+fn,sha(fp/fn)==v)
 original_ordering=[r for r in pool if r['question_type']=='ordering'];parts=[]
 for fn,status in [('ordering_keep.json','PASS'),('ordering_hold.json','HOLD'),('ordering_excluded.json','FAIL')]:
  part=sets[fn];parts+=part;ord_counts[status]+=len(part)
  ck('partition_decision_'+root+'_'+status,all(r['question_type']=='ordering' and dec[r['candidate_id']]['verdict']==status for r in part))
 ck('partition_complete_'+root,len(parts)==len({r['candidate_id'] for r in parts})==len(original_ordering) and {r['candidate_id'] for r in parts}=={r['candidate_id'] for r in original_ordering})
 expected=[r for r in pool if r['question_type']!='ordering' or dec[r['candidate_id']]['verdict']=='PASS']
 ck('ordering_filter_exact_pool_'+root,sets['candidate_pool.json']==expected)
 ck('SC_MS_unchanged_'+root,[r for r in sets['candidate_pool.json'] if r['question_type']!='ordering']==[r for r in pool if r['question_type']!='ordering'])
 expected_strict=[r for r in pool if dec[r['candidate_id']]['verdict']=='PASS']
 ck('strict_pass_exact_'+root,sets['strict_content_pass_candidates.json']==expected_strict)
 excluded={x['candidate_id'] for x in mp['semantic_duplicate_exclusions']}
 ck('semantic_dedup_exact_'+root,sets['strict_content_pass_deduplicated.json']==[r for r in expected_strict if r['candidate_id'] not in excluded])
 strict_total+=len(expected_strict);dedup_total+=len(sets['strict_content_pass_deduplicated.json'])
 for r in original_ordering:whole[dec[r['candidate_id']]['whole_source_support']]+=1
 fbpath=D/'feedback'/root/'feedback_input.json';fb=read(fbpath);rec,dialogue=_load_feedback_context(inputdata['parent'],feedback_input=fbpath);view=expansion_view_v3(rec);perf=rec['performance'];post=perf['post_generation_review'];oldperf=inputdata['feedback_record']['performance']
 ck('fact_checks_unchanged_'+root,perf['fact_checks']==oldperf['fact_checks'])
 ck('loader_renderer_post_review_'+root,view['solver_observations']['post_generation_review']==post and len(dialogue['turns'])==len(inputdata['parent']['dialogue']['turns']))
 ck('feedback_source_record_sha_'+root,fb['source_feedback']['record_sha256']==hashlib.sha256(json.dumps(inputdata['feedback_record'],ensure_ascii=False,sort_keys=True).encode()).hexdigest())
 ck('no_solver_or_activation_'+root,perf['solver_behavior_observations']=='NOT_PROVIDED_SOLVER_NOT_RUN' and post['solver_observations']=='NOT_PROVIDED_SOLVER_NOT_RUN' and fb['activation'].startswith('NOT_AUTOMATICALLY_ACTIVE') and post['filter_activation']=='LOCAL_REVIEW_ARTIFACT_ONLY_NOT_AUTOMATICALLY_ACTIVE' and fb['model_calls']==0 and not mp['automatic_activation'])
 ck('feedback_history_boundary_'+root,post['historical_fact_checks_scope']=='FROZEN_P_PLUS_R01_ONLY_UNCHANGED' and bool(post['branch_boundary']) and view['historical_qa'] is None and view['historical_qa_status']=='SOURCE_DATA_INCOMPLETE_NOT_MODEL_VISIBLE')
 if root==roots[-1]:ck('218519_feedback_no_invented_evaluation',post['status']=='PRESTOP_GATE_POOL_1_LT_3' and post['program_counts']['raw']==post['program_counts']['accepted']==0 and not post['qa_observations'] and post['source_transcript_path'] is None)
 feedback.append(dict(root=root,loader_renderer='PASS',fact_checks_preserved=perf['fact_checks']==oldperf['fact_checks'],review_in_performance=True,automatic_activation=False))
ck('raw_original_objects',len(raw_identity)==42 and all(x['identity_intact'] for x in raw_identity))
ck('total29_pool',total_pool==29==final['program_accepted'])
ck('strict_ordering_0_4_4',dict(ord_counts)=={'PASS':0,'HOLD':4,'FAIL':4},dict(ord_counts))
ck('whole_ordering_3_3_2',dict(whole)=={'PASS':3,'FAIL':3,'UNDETERMINED':2},dict(whole))
ck('strict14_dedup13',strict_total==14==final['strict_PASS_SC_MS'] and dedup_total==13==final['strict_PASS_SC_MS_semantic_dedup'])
ck('report_numerics',final['raw_candidates']==42 and final['strict_valid_ordering']==0 and final['final_ordering_review']=={'HOLD':4,'FAIL':4} and final['final_ordering_whole_source_support']==dict(whole) and final['cost']['total_tokens']==445962 and final['cost']['http_attempts']==31 and final['tokens_per_valid_ordering']=='UNDEFINED_ZERO_VALID_ORDERING')
ck('no_automatic_followup',not final['automatic_next_experiment'] and not final['feedback_automatically_activated'] and all(final[k]==0 for k in ('solver_calls','training_calls','free_calls','remote_memory_calls','external_audit_calls')))
manifest=read(R/'freeze_manifest.json');freezes={}
for label,p,n in [('current',R,358),('gate',Path(manifest['gate_run']),229),('original_planner',Path(manifest['source_run']),293)]:
 fs=read(p/'freeze_manifest.json')['files'];changed=[x['path'] for x in fs if not Path(x['path']).exists() or sha(x['path'])!=x['sha256']];freezes[label]=dict(count=len(fs),expected=n,changed=changed);ck('freeze_'+label,not changed and len(fs)==n,freezes[label])
for file in [D/'reports/FINAL_REPORT.md',D/'README.md']:
 text=file.read_text()
 for target in re.findall(r'\[[^\]]+\]\(([^\)]+)\)',text):
  if target.startswith(('http:','https:','#')):continue
  target=unquote(target.split('#',1)[0]);target=re.sub(r':\d+$','',target);p=Path(target) if target.startswith('/') else (file.parent/target).resolve();link_paths.append(dict(file=str(file),target=str(p),exists=p.exists()))
ck('final_report_readme_paths',all(x['exists'] for x in link_paths),dict(count=len(link_paths),missing=[x for x in link_paths if not x['exists']]))
result=dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',audit_kind='CODEX_OFFLINE_FINAL_ARTIFACT_CONSISTENCY_CHECK',new_api_calls=0,raw_candidates=42,program_final_candidates=29,ordering_partition=dict(ord_counts),ordering_whole_source_support=dict(whole),strict_pass_before_dedup=strict_total,strict_pass_dedup=dedup_total,checks=checks,freeze=freezes,feedback=feedback,report_links=link_paths,raw_identity=raw_identity,limitations=['Consistency and provenance checks do not independently certify semantic judgments.','HOLD is not proof of wrong Gold; three ordering sequences have whole-source support.','Original SC/MS retained in ordering-filtered pool are not all certified; use separately labeled strict files.'])
(D/'reports/FINAL_DELIVERY_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# 最终交付一致性检查','','**'+result['status']+'**。'+str(len(checks))+'项只读核对；新增API调用0。','','- 42条raw诊断身份与实际响应位置逐对象匹配，覆盖29条程序最终候选。','- 四root过滤文件中的候选均逐对象来自本轮原pool，没有改题干、选项、引用或Gold。','- 排序分区完整且互斥：严格保留0、HOLD4、FAIL4；全文支持分别PASS3、FAIL3、UNDETERMINED2。不能说8个Gold全错。','- 排序过滤后的candidate_pool保留全部原SC/MS原样，不等同训练合格池。严格PASS14，另按声明的1组安全攀岩近重复去重后13。','- 218519无原candidate_pool，正确记录NOT_REACHED/NOT_EVALUATED，0生成和0内容评估。','- 四份反馈保留冻结performance.fact_checks，新复盘在performance.post_generation_review，真实loader和renderer可见。Solver仍NOT_RUN；顶层激活说明未自动执行。','- 当前358、Gate229、旧Planner293条冻结记录无变化。','- 最终报告及README中'+str(len(link_paths))+'个本地链接均存在。','','本检查审计对象身份和交付一致性，不是对Codex语义诊断的独立人工认证。','','失败检查：']
failed=[c for c in checks if c['status']=='FAIL'];lines+=['- '+json.dumps(c,ensure_ascii=False) for c in failed] if failed else ['无。']
(D/'reports/FINAL_DELIVERY_CHECK.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),failed=failed),ensure_ascii=False,indent=2))
