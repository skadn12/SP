"""Write a reviewable batch report, manifest and current pointers after validation."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import Counter
import hashlib,json
D=Path(__file__).resolve().parent;P=D.parent/'day3_production_v1';W=D.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
s=read(D/'SUMMARY.json');old=read(P/'SUMMARY.json');v=read(D/'qa/FINAL_VALIDATION.json');pv=read(D/'qa/PRESENTATION_VALIDATION.json')
assert v['status']=='PASS' and v['records']==175 and pv['status']=='PASS'
assert s['accepted_questions']+s['held_questions']==175 and s['resolved_slots']==216
assert len(read(D/'dedup/DECISIONS.json')['rows'])==175
now=datetime.now(timezone.utc).isoformat()
combined=dict(schema_version='DAY3_TRAIN_PRODUCTION_TWO_BATCHES_V1',created_at_utc=now,
 roots=20,planned_slots=240,slot_decisions=240,actual_questions=old['authored_questions']+s['actual_initial_questions'],
 accepted_questions=old['accepted_questions']+s['accepted_questions'],held_questions=old['held_questions']+s['held_questions'],
 shortfall_slots=old['shortfall_slots']+s['shortfall_slots'],generation_holds=s['generation_holds'],
 matched_slot_pairs=old['matched_slot_pairs']+s['matched_slot_pairs'],
 first_batch_summary=binding(P/'SUMMARY.json'),batch2_summary=binding(D/'SUMMARY.json'),
 source_dialogue_extensions_created=0,new_solver_calls=0,new_training_runs=0,
 common_evaluation_actual_questions_frozen=0,training_pool_frozen=False,full_day3_complete=False,
 facts_independent_new_knowledge=False,
 interpretation='Content-reviewed candidates from20 historically exposed Train roots; shared facts across arms/slots and cross-root contact facts explicitly linked in dedup. Independent training pool and common evaluation not frozen.')
assert combined['accepted_questions']+combined['held_questions']+combined['shortfall_slots']+combined['generation_holds']==240
combined['arms']={}
for arm in ['A','B']:
 a=old['arms'][arm];b=s['arms'][arm]
 combined['arms'][arm]=dict(accepted=a['accepted']+b['accepted'],content_hold=a['held']+b['content_hold'],source_shortfall=a['shortfall']+b['source_shortfalls'],generation_holds=b['generation_holds'],accepted_by_type=dict(Counter(a['accepted_by_type'])+Counter(b['accepted_by_type'])),accepted_MC_gold_counts=dict(Counter(a['accepted_MC_gold_counts'])+Counter(b['accepted_MC_gold_counts'])))
dump(D/'TRAIN_PRODUCTION_SUMMARY.json',combined)
qa=read(D/'qa/EFFECTIVE_REVIEW_INDEX.json');held=[r for r in qa['rows'].values() if r['content_status']!='PASS']
lines=[f'本批18个Train root的216个固定题位已全部给出处理结果。175道实际初稿中，{s["accepted_questions"]}道通过来源、内容、目标关系和语义去重审核，{s["held_questions"]}道HOLD；另38槽素材短缺、3槽在生成阶段拦截。这里的通过题是内容候选池，训练池尚未冻结。',
'| 组别 | 题位 | 实际初稿 | 入选候选 | 内容HOLD | 素材短缺 | 生成拦截 |','|---|---:|---:|---:|---:|---:|---:|']
for arm,b in s['arms'].items():lines.append(f'| {arm} | {b["planned_slots"]} | {b["actual_initial_questions"]} | {b["accepted"]} | {b["content_hold"]} | {b["source_shortfalls"]} | {b["generation_holds"]} |')
lines+=['',f'与前批2个root合并：20个root、240个题位，实际{combined["actual_questions"]}道题，入选候选{combined["accepted_questions"]}道、内容HOLD {combined["held_questions"]}道、素材短缺{combined["shortfall_slots"]}槽、生成拦截{combined["generation_holds"]}槽。共{combined["matched_slot_pairs"]}个两臂同时入选的匹配题位，尚未用于训练池选择。前批262个封存文件SHA256保持一致。',
'',f'本批完整阅读并准入18份parent+R01+R02来源，共{s["source_turns_semantically_reviewed"]}个turn。逐题先读公开题和完整来源、独立重建封存，再读取作者答案比对；全部175题都有签署记录。该流程在共享文件系统上执行，不能称技术盲法。',
'','| 本批入选 | A | B |','|---|---:|---:|']
for typ in ['multi_select','ordering']:lines.append(f'| {typ} | {s["arms"]["A"]["accepted_by_type"].get(typ,0)} | {s["arms"]["B"]["accepted_by_type"].get(typ,0)} |')
lines+=['','| 多选Gold数量 | A | B |','|---|---:|---:|']
for n in ['2','3','4','5']:lines.append(f'| {n} | {s["arms"]["A"]["accepted_MC_gold_counts"].get(n,0)} | {s["arms"]["B"]["accepted_MC_gold_counts"].get(n,0)} |')
lines+=['','38个素材短缺中，36个是两臂各18个联合否定/条件关系槽：原文不足以证明两个非冗余前提相互约束，未用普通并列事实凑题；另外2个是14304两臂的同事项明确修订素材不足。因此本批覆盖binding与显式revision，并未完成联合条件关系的目标覆盖。',
'','3个生成拦截：B122711排序03的冻结锚点晚于已标同义提及，在写公开题前拦截；B179186多选01/02把继承Gold4/5误按3/4处理，在PLAN和公开题写出前拦截。原失败尝试与拦截原因保留，未改Gold补槽。',
'','7道内容HOLD的实质原因：',
'- A156136排序01/02：Green Haven会合地点在R01已有可成立的接受表述，R02的再选择不能无条件当首次。',
'- A148314排序01/02/03：便签修订在R02:A:S02:T007已表达，作者选了后一次T009。',
'- B135466排序03：模拟面试答案在R02:A:S03:T005已语义出现，不能排除它而选T007。',
'- B136202多选01：James也明确断言了博物馆免费入场，独立答案ACEF与原Gold ACE不一致。',
'','上述排序题部分字母顺序碰巧未变，但首次锚点审核仍不通过。原始公开题、作者Gold、锚点和封存重建全部保留；追加审核结论单独绑定。全部作者occurrence的纳入/排除已审核，只有元数据更正在有效索引内合并，真实题面/Gold/锚点问题保留HOLD。',
'','已实施的流程修复：',
'- 统一全部175题的有效审核入口，后续HOLD覆盖初始PASS；逐题绑定审核、追加注释与语义去重文件的SHA256。',
'- 归一化正式输出拒绝覆盖旧record；冻结检查要求175题完整ID覆盖，并对照当前有效QA/去重，防止沿用过期入选记录。',
'- A排序初稿存在固定BDFAEC答案位置模式。已在Solver/训练前登记按共同slot确定的随机重标算法，两臂同slot用同一映射；原稿保留。导出通过双射、文本回映和Gold往返检查。排列版本不算新增题或独立知识，有限样本分布不保证完美均匀。',
'','语义去重以全部175题与首批20题为范围，按人物、动作、对象和模态核对，披露同root和跨root共享事实；具体限制与判断见dedup目录。此审核不证明与所有历史训练/诊断题独立。共同事实复用和模板化措辞仍需在后续训练池与评估中控制。',
'',f'机械检查通过：{v["checks"]}项；展示转换通过：{pv["transformed_candidates"]}题。检查覆盖schema、原文跨度、配额、15pairs、首次纳入位置、审核绑定与入选门。对HOLD原稿保留的已知错误只核对其被隔离，不把它们宣称为内容正确。',
'','目录入口：',
'- exports/content_accepted.public.jsonl：入选候选原稿；exports/content_accepted.private.jsonl：对应Gold与来源/record索引。',
'- presentation/public/：统一重标后的候选；presentation/PRIVATE_LABEL_MAP.json：新旧标签与Gold映射。',
'- candidates/{A,B}/{root}/public、private、shortfall：全部实际初稿及缺额；GENERATION_HOLD文件保留生成拦截。',
'- qa/EFFECTIVE_REVIEW_INDEX.json、qa/independent/：有效审核入口、原封存与追加复核。',
'- final_records/、record_provenance/、exports/SLOT_LEDGER.json：统一记录、来源链和216槽全账。',
'- TRAIN_PRODUCTION_SUMMARY.json：两批合并计数；FINAL_MANIFEST.json：本批文件散列。',
'','第三天尚未整体完成。共同评估仍只有15 DEV+16 TEST来源和186个抽象题位；实际题面、Gold、事实族、请求冻结均为0。31来源散列核对通过，13槽已知场景数不足，其余仅场景数可行，不代表语义素材已准入。见evaluation_preflight/REPORT_ZH.md。下一步先完成不读取模型成绩的历史事实排除索引、共同评估来源/题面审核和冻结，再进行Day4对称训练池与资源冻结。',
'','本批未新增对话扩充、未调用Solver、未训练；没有证明反馈提高难度或训练收益。B只使用已准入的窄排序反馈，多选仍是共同设计假设；实际provider cap、usage、request ID未知，不作等实际算力结论。']
report=D/'REPORT_ZH.md'
with report.open('x') as fp:fp.write('\n'.join(lines)+'\n')
# Everything in this append-only batch except self and transient bytecode is sealed.
files=[]
for p in sorted(D.rglob('*')):
 if p.is_file() and p.name!='FINAL_MANIFEST.json' and '__pycache__' not in p.parts and p.suffix!='.pyc':files.append(dict(path=str(p.relative_to(D)),sha256=sha(p),bytes=p.stat().st_size))
manifest=D/'FINAL_MANIFEST.json'
dump(manifest,dict(schema_version='DAY3_BATCH2_MANIFEST_V1',created_at_utc=now,files=files,first_batch_manifest=binding(P/'FINAL_MANIFEST.json'),file_count=len(files),scope='Original drafts, failed attempts, source reviews, independent seals/addenda, semantic dedup, normalized records, presentation exports and validation. No mutable top-level status pointers in this manifest.'))
# Only pointers are mutable; their previous versions were snapshotted first.
for name in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
 snapshot=D/'history'/('BEFORE_BATCH2_'+name);assert snapshot.exists()
 assert sha(W/name)==sha(snapshot),'Pointer changed after snapshot; merge explicitly.'
status=read(W/'CURRENT_STATUS.json')
previous=status['day3']
status.update(status='DAY3_TRAIN_PRODUCTION_COMPLETE_COMMON_EVALUATION_PENDING',updated_at_utc=now,updated_at_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),next_action='Freeze no-score historical fact exclusions, full source admission and actual common evaluation content/Gold/request; then Day4 symmetric matched training pools and resources.')
status['daily_planning']['day3']='TRAIN_PRODUCTION_COMPLETE_COMMON_EVALUATION_PENDING'
status['before_execution_freeze_remaining']=[x for x in status['before_execution_freeze_remaining'] if not x.startswith('Remaining18 Train roots')]
status['day3']=dict(summary=binding(D/'TRAIN_PRODUCTION_SUMMARY.json'),report=binding(report),first_batch=previous,batch2_summary=binding(D/'SUMMARY.json'),manifest=binding(manifest),roots=20,planned_slots_executed=240,actual_questions=combined['actual_questions'],accepted=combined['accepted_questions'],hold=combined['held_questions'],shortfall=combined['shortfall_slots'],generation_hold=combined['generation_holds'],matched_slot_pairs=combined['matched_slot_pairs'],remaining_roots=0,remaining_slots=0,full_day_complete=False,new_solver_calls=0,new_training_runs=0,source_extensions_created=0,common_evaluation_actual_questions_frozen=0,common_evaluation_abstract_slots=186,data_fact_independence='HISTORICALLY_EXPOSED_TRAIN_SOURCE_REUSE',first_batch_read_only_after_manifest=True)
status['execution_progress_v6']=dict(scope='Day3 remaining18 Train roots216 slot decisions and independent content/dedup review complete',new_questions=175,accepted_questions=s['accepted_questions'],held_questions=s['held_questions'],shortfall_slots=38,generation_holds=3,new_solver_calls=0,new_training_runs=0,sealed_first_batch_files_unchanged=262)
latest=dict(schema_version='selfplay_week_day3_latest_v2',updated_at_utc=now,status=status['status'],summary=binding(D/'TRAIN_PRODUCTION_SUMMARY.json'),batch2_summary=binding(D/'SUMMARY.json'),report=binding(report),manifest=binding(manifest),accepted_questions=combined['accepted_questions'],held_questions=combined['held_questions'],shortfall_slots=combined['shortfall_slots'],generation_holds=combined['generation_holds'],remaining_root_count=0,remaining_slot_count=0,full_day_complete=False,common_evaluation_actual_questions_frozen=0,new_solver_calls=0,new_training_runs=0,training_pool_frozen=False)
for name,obj in [('CURRENT_STATUS.json',status),('DAY3_LATEST.json',latest)]:
 tmp=W/('.'+name+'.batch2.tmp')
 with tmp.open('x') as fp:json.dump(obj,fp,ensure_ascii=False,indent=2);fp.write('\n')
 tmp.replace(W/name)
for e in files:assert sha(D/e['path'])==e['sha256']
print(json.dumps(dict(combined={k:combined[k] for k in ['roots','planned_slots','actual_questions','accepted_questions','held_questions','shortfall_slots','generation_holds','matched_slot_pairs']},sealed_batch2_files=len(files),report=str(report))))
