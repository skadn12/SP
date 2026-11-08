"""Post-score D2 disposition, descriptive tables, and active status pointers."""
import collections,csv,hashlib,json,statistics
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent;R=P/'runtime/direct'
def read(p):return json.loads(p.read_text())
def bind(p):
    p=p.resolve();b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def pct(n,d):return f'{100*n/d:.2f}%' if d else 'NA'
assert not (P/'SUMMARY.json').exists()
scores=read(R/'SCORES.json');freeze=read(P/'CONTENT_FREEZE.json')
assert read(R/'RAW_RESPONSE_FREEZE.json')['status']=='COMMITTED'
assert scores['denominator_calls']<=360
private={q['candidate_id']:q for q in read(P/'private/PRIVATE_QUESTIONS.json')['questions']}
old=read(W/'day2_feedback_readiness_v1/SUMMARY.json')
reviewB=read(P/'reviews/B/INDEPENDENT_REVIEW.json')
reviewmap={q['candidate_id']:q for q in reviewB['questions']}
rows=[];lineage=[]
structures=collections.Counter()
for candidate in private.values():
    if candidate['question_type']=='ordering':
        scene_counts=collections.Counter(ev['primary_turn_ref'].rsplit(':',1)[0] for ev in candidate['evidence'].values())
        structures['+'.join(map(str,sorted(scene_counts.values(),reverse=True)))]+=1
for q in scores['questions']:
    src=private[q['candidate_id']];arm=src['arm'];typ=src['question_type']
    base={k:q[k] for k in ['candidate_id','root_id','question_type','denominator_calls','valid_calls','exact_calls','exact_rate','stratum']}
    base.update(arm=arm,gold_count=len(src['gold']),target_realized=reviewmap.get(q['candidate_id'],{}).get('target_realized'),
        lineage_valid=reviewmap.get(q['candidate_id'],{}).get('lineage_valid'))
    if typ=='multi_select':
        target=set(src['design']['target_labels'])
        base.update(fp=q['fp_calls'],fn=q['fn_calls'],
          target_fp=sum(x['fp_calls'] for x in q['items'] if x['canonical_label'] in target),
          target_negative_exposures=sum(x['valid_calls'] for x in q['items'] if x['canonical_label'] in target and not x['gold_selected']),
          negative_exposures=sum(x['valid_calls'] for x in q['items'] if not x['gold_selected']),
          positive_exposures=sum(x['valid_calls'] for x in q['items'] if x['gold_selected']),
          targets=sorted(target))
    else:base.update(correct_pairs=q['correct_pairs'],total_pairs=q['total_pairs'],pair_accuracy=q['pair_accuracy'])
    rows.append(base)
    if arm=='B':
        lineage.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':typ,
          'card_ids':src['design']['retrieved_card_ids'],'feedback_case_ids':src['design'].get('feedback_case_ids',[]),
          'target_labels':src['design']['target_labels'],'source_material':src['design']['source_material'],
          'content_pass':True,'target_realized':base['target_realized'],'lineage_valid':base['lineage_valid'],
          'content_artifact':reviewmap[q['candidate_id']]['selected_artifact'],'result':base})
save(P/'FEEDBACK_TO_RESULT_LINEAGE.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'questions':lineage})
groups=[]
for arm in ['A','B']:
    for typ in ['multi_select','ordering']:
        subset=[r for r in rows if r['arm']==arm and r['question_type']==typ]
        g={'arm':arm,'question_type':typ,'questions':len(subset),'roots':len({r['root_id'] for r in subset}),
           'calls':sum(r['denominator_calls'] for r in subset),'valid_calls':sum(r['valid_calls'] for r in subset),
           'exact_calls':sum(r['exact_calls'] for r in subset),
           'question_equal_exact':statistics.mean(r['exact_rate'] for r in subset) if subset else None}
        for key in (['fp','fn','target_fp','target_negative_exposures','negative_exposures','positive_exposures'] if typ=='multi_select' else ['correct_pairs','total_pairs']):
            g[key]=sum(r[key] for r in subset)
        groups.append(g)
target=[r for r in rows if r['arm']=='B' and r['question_type']=='ordering' and r['target_realized'] and r['lineage_valid']]
roots=sorted({q['root_id'] for q in target})
mc_targets=[r for r in rows if r['arm']=='B' and r['question_type']=='multi_select']
errors_roots=sorted({r['root_id'] for r in target if r['exact_calls']<r['denominator_calls']})
gate={
 'D2_01_to_03':old['source_gold_agree']==39 and old['operational_reproduced_count']>=1,
 'independent_execution_preflight':read(P/'audit/EXECUTION_PREFLIGHT.json').get('errors',[]) in [[],0],
 'actual_B_lineage_min_3_roots':len(roots)>=3,
 'content_qualified_target_min_3_families':len(roots)>=3,
 'new_direct_evaluation_terminal_frozen':scores['status']=='COMPLETE' and scores['valid_calls']==scores['denominator_calls'],
 'other_type_gaps_and_unknown_costs_recorded':True,
 'formal_MC_from_hypothesis_not_admitted':True
}
go=all(gate.values())
summary={'schema_version':'day2_pilot_completion_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'GO_NARROW_ORDERING_FEEDBACK_PROCESS' if go else 'NO_GO_FOR_FEEDBACK_B',
 'day2_all_goals_complete':go,'experiment_records_closed':True,
 'gates':gate,'planned_slots':24,'generated_questions':len(rows),'admitted_questions':freeze['admitted_count'],
 'shortfalls':freeze['shortfalls'],'roots':6,'new_solver_calls':scores['denominator_calls'],
 'new_training_runs':0,'source_dialogue_expansion':False,'historically_exposed_train':True,
 'groups':groups,'per_question':rows,'cost':scores['cost'],
 'author_cost':{'actual_tokens':None,'currency_cost':None,'requested_cap_per_question':3072,'backend_enforced':None,'strict_equal_compute_claim':False},
 'qualified_new_ordering_lineage_roots':roots,'qualified_new_ordering_families':len(roots),
 'qualified_ordering_roots_with_any_direct_error':errors_roots,
 'MC_attribute_hypothesis_target_error_roots':sorted({r['root_id'] for r in mc_targets if r['target_fp']>0}),
 'MC_status':'HYPOTHESIS_NO_RELATION_CONTROLLED_PROBE_NO_FORMAL_TARGETED_QUOTA',
 'fact_memory':{'adopted':False,'context':'full_history','cost_conclusion':'NO_COST_CONCLUSION'},
 'weakness_memory':'Actual assigned card retrieval/read->plan->candidate->review->direct-result lineage now recorded; one first-root full-card-read limitation retained.',
 'cot':'Prior 39 short evidence-and-answer diagnostics reviewed; no new long CoT or CoT training labels.',
 'feedback_utility_causal_claim':False,'training_utility_proven':False,
 'structural_diversity_complete':False,'fixed_70_percent_formal_B_ready':False,
 'actual_ordering_scene_compositions':dict(structures),
 'content_clarification_slots':{'A':2,'B':2},
 'scope':'Process readiness only for narrow source-qualified cross-scene occurrence ordering; MC relation remains hypothesis.',
 'next_action':'Day 3 common structural diversity design and source/slot freeze; revise formal B coverage prospectively before production, because current narrow target does not justify original 70% whole-type quota.',
 'inputs':{k:bind(v) for k,v in {'content_freeze':P/'CONTENT_FREEZE.json','scores':R/'SCORES.json',
 'source_reviews_B':P/'reviews/B/INDEPENDENT_REVIEW.json','source_reviews_A':P/'reviews/A/INDEPENDENT_REVIEW.json',
 'raw_response_freeze':R/'RAW_RESPONSE_FREEZE.json','historical_readiness_summary':W/'day2_feedback_readiness_v1/SUMMARY.json',
 'runtime_code_review':P/'audit/RUNTIME_CODE_REVIEW.json','public_preflight':P/'audit/PUBLIC_PREFLIGHT_PASS.json'}.items()}}
save(P/'SUMMARY.json',summary)
with (P/'RESULTS_PER_QUESTION.csv').open('x') as f:
    fields=['root_id','arm','question_type','gold_count','exact_calls','denominator_calls','exact_rate','valid_calls','fp','fn','target_fp','target_negative_exposures','correct_pairs','total_pairs','lineage_valid','target_realized','candidate_id']
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
lines=['# 第二天整体执行报告','',
 f"结论：**{summary['status']}**。D2-01—D2-06已有实际产物；这里的GO仅是窄范围反馈流程准入，不是反馈训练有效的结论。",'',
 '6个Train root、固定24题位，A/B各6道多选＋6道排序。原全文复用，未扩充对话。两臂同来源、粗结构、Gold数量计划；B额外读指定反馈。生成/审核全部完成并冻结后才开始S0=M-clean评估。根和数据族均已历史暴露，不是独立新测试。','',
 '| 臂 | 题型 | 题数 | 整题全对/调用 | 整题正确率 | 细项 |','|---|---|---:|---:|---:|---|']
for g in groups:
    detail=f"FP {g['fp']}/{g['negative_exposures']}；FN {g['fn']}/{g['positive_exposures']}" if g['question_type']=='multi_select' else f"顺序对 {g['correct_pairs']}/{g['total_pairs']}（{pct(g['correct_pairs'],g['total_pairs'])}）"
    lines.append(f"| {g['arm']} | {g['question_type']} | {g['questions']} | {g['exact_calls']}/{g['calls']} | {pct(g['exact_calls'],g['calls'])} | {detail} |")
lines+=['','每道多选6种显示排列×3次；每道排序12种显示排列×1次。360次是重复作答次数，独立新题24道、root仅6个。各组同样本数，调用加权与题等权一致。A/B题目不同，分数差不能证明反馈因果。','',
 '## 六项交付','',
 '| 任务 | 实际结果 |','|---|---|',
 '| D2-01 旧诊断语义审核 | 39题/20root、234条item；39/39来源重建Gold一致；1处旧题代词需中性改写，不改Gold。 |',
 '| D2-02 结构化真实反馈 | 引文位置、支持/反证、判断与答案一致性、FP/FN及错序、重复稳定性、来源/响应hash、竞争解释均已落盘。 |',
 '| D2-03 操作性裁决 | 1条窄排序方向获历史操作性支持；4条MC仍是假说。旧固定引文探针是V2回溯裁决，不冒充事前正阈值。 |',
 f"| D2-04 Challenger读取 | 已形成B实际读取→计划→题位→候选链；{len(roots)}个root的排序链完整；首root的完整卡读取证明不足，内容保留但不计正式链路门槛。 |",
 f"| D2-05 前瞻新题小试 | 24固定题位，内容准入{len(rows)}/24；所有原稿、有限澄清及审稿保留。先内容冻结，后{scores['denominator_calls']}次direct。 |",
 f"| D2-06 裁决 | {summary['status']}；目标实现及新题错误复现分开统计。 |",'',
 '## 可以推进的范围','',
 f"合格新排序目标实现覆盖{len(roots)}个root/数据族，达到事前至少3个root＋3个family的流程门；其中{len(errors_roots)}个root存在直接作答错序。family按root|type计，不代表结构体裁多样。",
 '多选只验证能按属性绑定假说形成合法错项，并报告实际误选；本次未实施该关系的单变量受控探针，不把误选升级为已证实机制，也不准入正式70%定向比例。',
'新旧题不同，因此本轮准确率与旧39题不能直接作为难度上升/下降的因果比较；本轮也没有训练，尚不能证明反馈提高训练收益。','',
 '## memory与推理输出','',
 '事实memory仍暂不采用，主输入保留完整历史；旧事实memory试验保持NO_COST_CONCLUSION。本轮实际采用的是带模型版本、资格状态及证据绑定的弱点卡检索，二者分开。旧39条简短依据＋答案已核查；不把它当忠实内部思维过程，不生成长CoT训练标签。','',
 '## 成本与限制','',
 f"本轮direct：{scores['cost']['planned_calls']}次，输入token {scores['cost']['known_prompt_tokens']}，输出token {scores['cost']['known_completion_tokens']}；推理总耗时{scores['cost']['live_wall_seconds']:.1f}秒（含模型载入）。本地GPU货币费用未知。作者及审核实际API token不可取得，记null；3072仅请求上限，未证明后端强制执行或两臂等算力。",'',
 'A完成2个题位的措辞澄清，Gold不变；B的全部澄清详见独立审核记录。生成者与审核者共享文件系统，隔离由任务输入边界和读取日志记录，不宣称外部盲法。B作者此前写过runtime但未读A新题，A作者无B反馈。',
 f"排序场景组成实际为{dict(structures)}（数值为各场景选取的事件数，字典值为题数）。仍以文本首次出现为核心；变换动词和报告焦点不能解决结构过于一致的问题。原来的全题型70%定向B配额仍未就绪。",'',
 '## 下一步','',
 '进入第3天：冻结共同的实质结构库、题位和共同评估边界；排序落实3＋2＋1、2＋2＋2、4＋2的同场景组合及重复报告区分。正式B仅可使用已经通过的窄方向，若原配额与覆盖不匹配，先前瞻修订范围，不用MC假说或自由题暗中补齐。训练池、选项置换和重训仍属第4—5天，当前未启动。','',
 '## 文件入口','',
 '- SUMMARY.json：逐门状态、分组及逐题统计。','- public/PUBLIC_QUESTIONS.json：准入题面与来源。','- private/PRIVATE_QUESTIONS.json：Gold、证据与设计。','- drafts/A、drafts/B：全部计划、原稿、澄清和读取日志。','- reviews/A、reviews/B：独立来源语义审核。','- CONTENT_FREEZE.json：先评分前的准入冻结。','- runtime/direct：真实提示词、映射、种子、原始回答、全终态冻结及SCORES。','- FEEDBACK_TO_RESULT_LINEAGE.json：反馈→候选→结果。','- RESULTS_PER_QUESTION.csv：逐root/题结果。','']
(P/'REPORT_ZH.md').write_text('\n'.join(lines))
# Active pointers are updated after the independent completion audit, not here.
print(json.dumps({'status':summary['status'],'gates':gate,'groups':groups,'report':str(P/'REPORT_ZH.md')},ensure_ascii=False))
