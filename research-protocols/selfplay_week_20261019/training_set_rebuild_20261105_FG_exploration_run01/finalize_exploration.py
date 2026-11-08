"""Seal completed exploration and review preparation, never training/evaluation."""
import json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
R=Path(__file__).resolve().parent

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
a=json.loads((R/'ARTIFACT_AUDIT.json').read_text());assert a['frozen_opportunities']==140 and a['pending_opportunities']==0
cfg=json.loads((R/'RUN_CONFIG.json').read_text());sealed=cfg['sealed_135_training_export'];assert h(sealed['path'])==sealed['sha256']
# Immutable author bytes are reverified after packet generation.
for x in a['rows']:
 for b in x['artifacts']:assert h(R/b['path'])==b['sha256'],b['path']
pf=json.loads((R/'PROTOCOL_FREEZE.json').read_text())
for x in pf['files']:assert h(R/x['path'])==x['sha256'],x['path']
for x in json.loads((R/'SOURCE_ADMISSION_ADDENDUM.json').read_text())['rows']:
 assert h(x['source_snapshot'])==x['source_sha256']
 assert h(x['source_review_path'])==x['source_review_sha256']
 assert h(x['neutral_source_info_snapshot'])==x['neutral_sha256']
for x in json.loads((R/'private_semantic_reference/REFERENCE_FREEZE.json').read_text())['input_bindings']:
 assert h(x['path'])==x['sha256'],x['path']
n=json.loads((R/'NEXT_STAGE.json').read_text());assert n['review_packet_count']==a['draft_count']
normalize={'state_updates':'state updates','constraint_integration':'constraint integration','entity_set_comparison':'entity/set comparison','event_chronology':'event/revision chronology','event_revision_chronology':'event/revision chronology','plan_dependencies':'plan dependencies','planned_dependencies':'plan dependencies','partial_precedence':'partial precedence'}
mac=Counter();byarm={};option_counts={};gold_counts={}
for x in a['rows']:
 if x.get('draft_count'):mac[normalize.get(x['macrostructure'],x['macrostructure'])]+=x['draft_count']
 for c in json.loads((R/'authoring'/x['arm']/x['root_id']/'CANDIDATES_FIRST_DRAFT.json').read_text())['candidates']:
  qt=c['question_type'];option_counts.setdefault(qt,Counter())[str(len(c['options']))]+=1
  gc=len(c.get('gold_pairs',[])) if qt=='partial_order' else len(c['gold_option_ids'])
  gold_counts.setdefault(qt,Counter())[str(gc)]+=1
for arm in ['F','G']:
 rr=[x for x in a['rows'] if x['arm']==arm]
 byarm[arm]={'roots_explored':len(rr),'first_drafts':sum(x['draft_count'] for x in rr),'no_safe_qa_roots':sum(x['draft_count']==0 for x in rr),'question_types':dict(Counter(x['question_type'] for x in rr if x.get('draft_count')))}
summary={'run_id':R.name,'status':'EXPLORATION_COMPLETE_REVIEW_PENDING','completed_at_utc':datetime.now(timezone.utc).isoformat(),'roots_required':70,'root_arm_opportunities_completed':140,'draft_cap':140,'first_drafts':a['draft_count'],'by_arm':byarm,'question_types':a['candidate_types'],'macrostructures_as_authored_normalized':dict(mac),'macrostructure_classification_is_author_report_not_independent_diversity_proof':True,'mechanical_pass_opportunities':a['mechanical_pass_opportunities'],'mechanical_hold_opportunities':a['mechanical_hold_opportunities'],'quality_approved_items':None,'semantically_novel_items':None,'Mclean_evaluated_items':0,'new_training_eligible_items':None,'new_solver_calls':0,'new_training_exports':0,'training_runs_started':0,'sealed_135_unchanged':True,'sealed_135_sha256':sealed['sha256'],'review_packets':n['review_packet_count'],'all_first_drafts_preserved':True,'extra_roots_source_admission':['194259','87273']}
summary['option_count_distribution_by_type']={k:dict(v) for k,v in option_counts.items()}
summary['gold_cardinality_distribution_by_type']={k:dict(v) for k,v in gold_counts.items()}
write(R/'SUMMARY.json',summary)
grouped={}
for x in a['rows']:
 grouped.setdefault(x['root_id'],{'root_id':x['root_id'],'arms':{},'both_author_opportunities_completed':True})['arms'][x['arm']]=x
assert len(grouped)==70 and all(set(x['arms'])=={'F','G'} for x in grouped.values())
write(R/'ROOT_COVERAGE.json',{'required_roots':70,'root_count':70,'opportunities_completed':140,'author_coverage_is_not_training_admission':True,'roots':list(grouped.values())})
(R/'PROGRESS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
handoff=R/'RUN_HANDOFF.md'
handoff.write_text('Exploration completed and sealed. All 140 author opportunities are complete; no remaining author batches. Use SUMMARY.json, REPORT_ZH.md, ROOT_COVERAGE.json and NEXT_STAGE.json for current state. Independent candidate quality/source reconstruction, semantic dedup and M-clean evaluation remain pending. No solver calls/export/training occurred.\n\nThe initial operational handoff below is retained as history; its active-worker queue is obsolete.\n\n'+handoff.read_text())
report=f'''# 新 F/G 独立探索轮报告\n\n本轮已完成原始 70 个 Train roots 的 F、G 全部 140 个探索机会。实际冻结 {a['draft_count']} 道首稿：F {byarm['F']['first_drafts']}、G {byarm['G']['first_drafts']}。每 root 每臂最多一稿；零候选保留原因。两臂独立，F 使用固定抽象反馈，G 不看反馈；作者均不看历史题面、Gold 或 Solver 输出。\n\n题型：{json.dumps(a['candidate_types'],ensure_ascii=False)}。作者记录的宏观结构（描述性标签）：{json.dumps(dict(mac),ensure_ascii=False)}。这些首稿数量和结构标签尚不能证明新增独立结构、去重后增量或去模板化成功。具体新颖性需要比较任务意图、关系、实体、事件证据和答案操作。\n\n来源：68 个既有精确来源重新绑定；194259 的 92 回合及 87273 的 91 回合另做全文来源矛盾/归属审查，限制记录在 SOURCE_ADMISSION_ADDENDUM.json 和各 root 的来源审查中。合成续写仅支持对话层面的陈述与计划，不是现实事实认证。审查是同模型内部工作；来源审查者接触过历史候选，但未接触本轮作者首稿。\n\n机械检查：{a['mechanical_pass_opportunities']} 个机会 PASS、{a['mechanical_hold_opportunities']} 个 HOLD；核验来源哈希、全文阅读证书、冻结目标哈希/落盘次序、目标与首稿 Gold 一致、选项/证书覆盖、逐字引用与 origin，及适用的排序/部分顺序结构。机械 PASS 不等于质量合格或语义新增。所有首稿和 HOLD 原因保留，未根据检查结果修改题面/选项/Gold。\n\n已生成 {n['review_packet_count']} 个不含作者 Gold 的独立审核包；质量与来源的逐题独立重建、语义去重和 M-clean 评估均待下一阶段。本阶段 Solver 调用 0，训练导出 0，训练 0。已封存 135 题原文件哈希核验未变：`{sealed['sha256']}`。\n\n去重参考冻结覆盖 70 roots，含旧 F/G、legacy、F 补充全量首稿、三-root F/G retry 和两个补入 root 的历史候选池；当前 135 题另有完整无答案题面投影。额外历史 parent-family 资产可在语义审查时补检；本轮尚未给出“新增合格题数”。\n\n下一阶段入口：NEXT_STAGE.json、blind_review_packets/PACKET_MANIFEST.json、private_review_bindings.json、private_semantic_reference/REFERENCE_FREEZE.json 及 SEALED135_QUESTION_PROJECTION_ADDENDUM.json。新评估须先核验执行环境、绑定模型/适配器/新 seeds/请求表和无重试账本；预留调用上限 5040，不沿用任何旧轮余量或诊断分数。\n'''
report+='\n选项数分布（各题型：选项数→首稿数）：'+json.dumps(summary['option_count_distribution_by_type'],ensure_ascii=False)+'。\n\nGold 基数分布：'+json.dumps(summary['gold_cardinality_distribution_by_type'],ensure_ascii=False)+'。部分顺序的 Gold 基数为完整闭包中的关系对数。\n'
with (R/'REPORT_ZH.md').open('x') as f:f.write(report)
files=[{'path':str(p.relative_to(R)),'sha256':h(p),'bytes':p.stat().st_size} for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('EXPLORATION_SEAL.json',)]
write(R/'EXPLORATION_SEAL.json',{'status':summary['status'],'sealed_at_utc':datetime.now(timezone.utc).isoformat(),'file_count':len(files),'files':files,'sealed_135_unchanged':True,'review_and_evaluation_not_claimed':True})
print(json.dumps(summary,ensure_ascii=False,indent=2))
