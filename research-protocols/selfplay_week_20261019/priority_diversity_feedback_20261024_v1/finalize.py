"""Publish this bounded batch, keeping production and follow-up denominators separate."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

P=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def binding(path):
    p=Path(path).resolve(); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(name,value):
    with (P/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def pct(value):return f'{value:.1%}' if value is not None else '—'
def main():
    sem=P/'semantic_audit/assembled'
    for name in ['MANIFEST.json']:
        for b in read(sem/name)['bindings']:assert binding(b['path'])==b
    summary=read(sem/'SUMMARY.json')
    analysis=read(P/'execution/ANALYSIS_SUMMARY.json')
    content=read(P/'CONTENT_FREEZE.json')
    trace=read(P/'FEEDBACK_CONSUMPTION_TRACE.json')
    ledger=read(P/'SLOT_LEDGER.json')['slots']
    candidates=read(P/'CANDIDATES_PRIVATE.json')
    content_reviews={r['candidate_id']:r for r in read(P/'CONTENT_REVIEW_AI.json')['reviews']}
    concerns=read(sem/'POSTFREEZE_CONTENT_CONCERNS.json')['concerns']
    blocked={r['candidate_id'] for r in concerns}
    available=[c for c in candidates if content_reviews[c['candidate_id']]['decision']=='PASS' and c['candidate_id'] not in blocked]
    # Resolve and validate dependent artifacts before exclusive publication.
    next_path=P/'next_generation/SUMMARY.json'
    assert next_path.exists(), 'Complete bounded feedback handoff before final report'
    nxt=read(next_path)
    raw=read(P/'execution/RAW_RESPONSE_FREEZE.json')
    model=read(P/'execution/MODEL_LOAD_RECEIPT.json')
    assert model['effective_rope_theta']==1000000
    assert nxt['additional_solver_requests']==nxt['training_runs']==nxt['new_dialogue_turns']==0
    consumed=bool(nxt['feedback_consumed'] and nxt['generated']>0)
    assert consumed, 'No actual follow-up feedback consumption; report as incomplete instead'
    packets=read(P/'semantic_audit/REVIEW_PACKET_A.json')['questions']
    audits={r['request_id']:r for r in read(sem/'FULL_RESPONSE_REVIEWS.json')['records']}
    for cid in ['PDF1-218519-MC2','PDF1-218519-ORDER1']:
        q=next(q for q in packets if q['candidate']['candidate_id']==cid)
        r=next(r for r in q['responses'] if r['variant_id']=='perm_00')
        assert r['request_id'] in audits
    assert raw['returned_requests']==analysis['requests']==120
    for row in summary['primary_control_scores']:
        assert {'condition','group','exact','requests','exact_accuracy'}<=row.keys()
    outputs=['CANDIDATE_POOL_INDEX.json','MATERIAL_SHORTFALL_FEEDBACK.json','SUMMARY.json',
             'REPORT_ZH.md','EXAMPLES_ZH.md','FINAL_MANIFEST.json']
    assert not any((P/name).exists() for name in outputs), 'Do not overwrite an earlier closeout'
    save('CANDIDATE_POOL_INDEX.json',{'schema_version':'priority_content_candidate_pool_v1',
       'questions':[{'candidate_id':c['candidate_id'],'root_id':c['root_id'],
           'question_type':c['question_type'],'task_operation':c['task_operation'],
           'gold_count':len(c['gold']) if c['question_type']=='multi_select' else None,
           'feedback_card_id':c['feedback_target']['card_id'],'source_binding':c['source_binding']} for c in available],
       'content_admitted':len(available),'solver_score_filtering':False,
       'training_mixture_frozen':False,'training_admission':False,
       'historical_pools_modified':False,'postfreeze_concerns_excluded_from_new_pool':sorted(blocked),
       'bindings':[binding(P/'CANDIDATES_PRIVATE.json'),binding(P/'CONTENT_REVIEW_AI.json'),binding(sem/'POSTFREEZE_CONTENT_CONCERNS.json')]})
    material=[r for r in ledger if r['status']=='MATERIAL_SHORTFALL']
    save('MATERIAL_SHORTFALL_FEEDBACK.json',{'type':'SOURCE_MATERIAL_NOT_SOLVER_WEAKNESS',
       'slots':material,'source_population_stability_established':False,
       'future_expansion_request':'If a later, separately authorized dialogue expansion is run, obtain source-consistent explicit time/constraint facts with different values inside a coherent event. Do not insert numbers into existing source or invent a relation merely to fill an ordering slot.',
       'current_batch_new_dialogue_turns':0})
    final={
      'schema_version':'priority_diversity_feedback_closeout_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'primary_batch':trace['counts'],'solver_requests':analysis['requests'],
      'raw_returned':raw['returned_requests'],'valid_requests':analysis['valid'],
      'wall_seconds':raw['wall_seconds'],'source_turns':sum(len(read(r['path'])['turns']) for r in read(P/'SELECTION_FREEZE.json')['roots']),
      'primary_control_scores':summary['primary_control_scores'],
      'all_full_responses_reviewed':summary['full_responses'],
      'all_semantic_dimensions_pass':summary['all_semantic_dimensions_pass'],
      'card_outcomes':summary['card_outcomes'],'new_candidate_pool':len(available),
      'next_generation':nxt,'postfreeze_content_concerns':concerns,
      'actual_feedback_consumption_into_new_tasks':consumed,
      'population_stable_production_proven':False,'formal_new_solver_weakness_admission':False,
      'causal_feedback_training_gain_proven':False,'new_training_runs':0,'new_dialogue_turns':0,
      'dev_test_calls':0,'faithful_internal_cot_claim':False,
      'claim_boundary':'Fixed historical Train roots; dependent displays; Gold-visible nonauthor AI review; no randomized feedback control or new training. Primary and follow-up question counts remain separate.'}
    save('SUMMARY.json',final)
    initial=trace['counts']['initial_review']
    lines=['# 优先项一、二：题型质量与反馈驱动出题', '',
      '本轮只做既有Train对话上的出题和B1评估，无新增对话、训练或Dev/Test调用。', '',
      '## 出题质量与覆盖', '',
      f"固定4个历史Train roots、32题位，初审 {initial.get('PASS',0)} PASS、{initial.get('HOLD',0)} HOLD、2处素材不足；一轮修订后30题通过（20多选、10排序），最终30/32=93.8%。初审按计划槽通过率25/32=78.1%，按已提交稿为25/30=83.3%，不能混同终审产率。", '',
      '多选gold 2/3/4/5/6各4道；其中6-gold仅为全选控制。覆盖数量关系、实体合取、证据选择、因果支持；排序覆盖钟点比较、具体流程、话语重建。显示程序固定打乱，保存双向映射；题干不暴露正确数量，模型提示中无包含gold数量的候选ID。', '',
      '初审5处HOLD：3道证据选择题直接展示完整支持/反证文本，已改为来源位置引用；另修复人物动机干扰项和包含多重错误的理由。缺额来自222937、225575各只有3个独立钟点，未改成别的操作或补编事实。4个root的最终产出为8/7/7/8题。', '',
      f"以上30题指Solver前冻结的内容准入；回答后的再次核查发现{len(concerns)}道内容疑点，新候选索引保留{len(available)}道，原题与原分数不回改。", '',
      f"实际B1请求{analysis['requests']}次，返回{raw['returned_requests']}，结构有效{analysis['valid']}；运行耗时{raw['wall_seconds']/60:.1f}分钟（含加载/收尾）。实际引擎RoPE theta={model['effective_rope_theta']}，配置回执单独保存；没有重试或按分数删题。", '',
      '| 分析组 | 完整对话正确/请求 | 正确率 | 仅题面正确/请求 | 正确率 |',
      '|---|---:|---:|---:|---:|']
    for group,label in [('MC2_5_PRIMARY','多选2–5 gold'),('ORDERING','排序'),('MC6_CONTROL','多选6 gold控制')]:
        a=next(r for r in summary['primary_control_scores'] if r['condition']=='FULL' and r['group']==group)
        b=next(r for r in summary['primary_control_scores'] if r['condition']=='QUESTION_ONLY' and r['group']==group)
        lines.append(f"| {label} | {a['exact']}/{a['requests']} | {pct(a['exact_accuracy'])} | {b['exact']}/{b['requests']} | {pct(b['exact_accuracy'])} |")
    lines += ['', '每题两个显示相关，不能把响应数充作独立题数。完整对话与仅题面的差异用于诊断来源依赖，不单独证明因果反馈收益或总体难度。', '',
      '## 反馈实际使用', '',
      '此前4张已审核探索卡 → 两位作者的逐槽计划/读取回执 → 新题具体目标与独立审查 → B1真实响应 → 全部FULL响应逐项语义审核 → 新版探索卡 → 下一代有限出题。每个环节保存版本及hash；原历史卡和原始记录保持可追溯。', '',
      f"逐项审核{summary['full_responses']}条FULL响应，判断、引用、理由及适用一致性全部通过{summary['all_semantic_dimensions_pass']}条；答案正确不自动视为理由或引用正确。这个严格指标要求一题的所有项与适用响应维度均通过，不等于其余每个单项都错。逐项引用计数为{summary['item_status_counts']['citation_support']}，逐项理由为{summary['item_status_counts']['reason_support']}。", '',
      '| 输入卡维度 | 观察到目标错误的响应 | 独立题 | root | 双显示均观察到的题 |',
      '|---|---:|---:|---:|---:|']
    for r in summary['card_outcomes']:
        lines.append(f"| {r['card_id'].removeprefix('PROD1-HYP-')} | {r['responses_observed']} | {r['unique_questions_observed']} | {r['roots_observed_count']} | {r['questions_observed_in_both_displays_count']} |")
    lines += ['', f"下一代单列：计划{nxt['planned_roots']}个root，生成{nxt['generated']}题，独立终审准入{nxt['content_admitted']}题；初审{nxt['initial_decisions']}，终审{nxt['final_decisions']}。目标为{nxt['target_dimensions']}。新增Solver请求、训练和对话均为0，不能把这些4题计入主批产率或120请求的准确率。", '',
      '下一代225575的正确比较来自判断与理由正确、实际引用失败的单项；原citation FAIL保留，作者另读完整来源并重构确切证据，独立审核确认。它不被宣称为全维正确响应。其他root优先使用实际逐项判断、引用和理由均通过的对照；229686排序还核对了完整正确顺序。', '',
      '## 可以下的结论', '',
      '已实现：7类操作和2–6 gold覆盖；可发现并修复题面捷径及局部干扰项问题；旧卡被实际用于新题，本批真实错误被逐项审核、形成新版卡并进入下一代。素材短缺与Solver错误分开回传。', '',
      '尚不能宣称：全量自动稳定生产、正式新弱点受控复现、反馈导致训练收益或对未见root泛化。首审仍需修复的比例不可忽略；这批没有新训练和随机无反馈对照。可见理由不是内部思维链。', '',
      '## 文件导航', '',
      '- `CANDIDATES_PRIVATE.json`：主批30题终稿及私有答案、逐字证据。',
      '- `CANDIDATE_POOL_INDEX.json`：仅按内容审核形成的新候选索引，不是已冻结训练混合池。',
      '- `author_a/`、`author_b/`、`review_a/`、`review_b/`：计划、初稿、一轮修订、初审与终审。',
      '- `execution/raw/`：120条原始输出；`execution/REASON_REVIEW_PACKET.json`含可见理由和最终答案。',
      '- `semantic_audit/assembled/FULL_RESPONSE_REVIEWS.json`：全部FULL逐项审核；同目录新版卡与汇总。',
      '- `next_generation/`：本批反馈的下一代真实消费、计划、题目和独立内容审核。',
      '- `next_generation/PRESENTATION_PUBLIC_PREVIEW.json`、`PRESENTATION_MAPPING_PRIVATE.json`：程序打乱后的公开预览与私有双向映射；未发送模型、未赋历史分数。',
      '- `EXAMPLES_ZH.md`：一个多选、一个排序的题目、实际理由和审核对照。', '']
    with (P/'REPORT_ZH.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    examples=['# 真实题目、回答与反馈示例', '', '固定展示218519的数量多选与时间排序的perm_00，未按回答好坏挑选。完整来源在对应source_binding.path；原始输出路径和hash保留。', '']
    for cid in ['PDF1-218519-MC2','PDF1-218519-ORDER1']:
        q=next(q for q in packets if q['candidate']['candidate_id']==cid)
        r=next(r for r in q['responses'] if r['variant_id']=='perm_00')
        a=audits[r['request_id']]
        examples += [f'## {cid}', '', q['candidate']['question'], '']
        examples += [f'{k}. {v}' for k,v in r['display'].items()]
        examples += ['',f"显示Gold：{' '.join(r['score']['display_gold'])}；显示预测：{' '.join(r['score']['display_prediction'])}；完全正确：{r['score']['exact']}", '',
            '实际可见输出（仅输出，不是内部思维链）：', '```json', json.dumps(r['parsed'],ensure_ascii=False,indent=2) if r.get('parsed') is not None else r['raw_output'], '```', '',
            '审核：'+json.dumps({'predicted_error_assessment':a['predicted_error_assessment'],
               'response_notes':a['notes'],'item_reviews':{k:{f:i[f] for f in ['judgment_correct','citation_support','reason_support','note']} for k,i in a['items'].items()}},ensure_ascii=False,indent=2), '',
            '原始响应：'+json.dumps(r['raw_binding'],ensure_ascii=False), '']
    with (P/'EXAMPLES_ZH.md').open('x',encoding='utf-8') as f:f.write('\n'.join(examples))
    paths=[x for x in P.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.name!='FINAL_MANIFEST.json' and not any(part.startswith('runner_cpu_fixture') for part in x.parts)]
    save('FINAL_MANIFEST.json',{'schema_version':'priority_diversity_feedback_manifest_v1',
       'artifact_bindings':[binding(x) for x in sorted(paths)],
       'source_bindings':[{k:r[k] for k in ['path','sha256','bytes']} for r in read(P/'SELECTION_FREEZE.json')['roots']],
       'primary_question_count':len(candidates),'raw_response_count':analysis['requests'],
       'training_launched':False,'new_dialogue_turns':0})
    print(json.dumps({'primary_questions':len(candidates),'candidate_pool':len(available),
         'full_responses_reviewed':summary['full_responses'],'next_generation':nxt},ensure_ascii=False))
if __name__=='__main__':main()
