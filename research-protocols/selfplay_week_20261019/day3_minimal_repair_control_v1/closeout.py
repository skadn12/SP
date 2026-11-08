#!/usr/bin/env python3
"""Close out frozen inference; never invokes a model or overwrites raw/scoring files."""
import collections, copy, datetime, hashlib, json, pathlib, sys

P = pathlib.Path(__file__).resolve().parent
NL = chr(10)

def read(f):
    return json.loads((P / f).read_text())

def binding(f):
    p = (P / f).resolve()
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def save(f, value):
    p = P / f
    assert not p.exists(), f'NO_OVERWRITE:{p}'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + NL)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def prepare():
    mc_path = 'review/INTERFACE_V3_MC_SOURCE_FIRST_REVIEW.json'
    order_path = 'review/INTERFACE_V3_ORDER_SOURCE_FIRST_REVIEW_V2.json'
    mc, order = read(mc_path), read(order_path)
    assert mc['status'] == 'COMPLETE' and order['status'].startswith('COMPLETE')
    score, manifest = read('runtime/interface_v3/SCORES.json'), read('runtime/interface_v3/manifest.json')
    requests = {r['request_id']: r for r in manifest['requests']}
    scored = {r['request_id']: r for r in score['responses']}
    contract = read('review/SEMANTIC_COVERAGE_CONTRACT.json')
    audit = read('review/INDEPENDENT_MC_AGGREGATION_ADJUDICATION.json')
    assert audit['MC_checks']['all_checks_passed']
    rows, seen = [], set()
    for rr in mc['responses'] + order['responses']:
        r = copy.deepcopy(rr)
        q = requests[r['request_id']]
        assert r['request_id'] not in seen
        seen.add(r['request_id'])
        assert r['format_valid'] == scored[r['request_id']]['format_valid']
        displayed = q['display'].get('options', q['display'].get('items'))
        assert {i['display_label'] for i in r['items']} == set(displayed)
        assert len(r['items']) == len(displayed)
        r['question_type'] = q['question_type']
        r['root_id'] = q['root_id']
        for item in r['items']:
            assert set(contract['per_item_fields']) <= set(item)
            assert not any('PENDING' in str(item.get(k, '')) for k in ['status','rationale'])
            item['as_reviewed_source_faithful_item'] = item['source_faithful_item']
            eligible = r['format_valid'] and all(item[k] is True for k in ['judgment_correct','reason_correct','final_alignment','citations_exact'])
            relevant = item['citation_semantic_support'] is True
            complete = item.get('citation_covers_all_required_premises', relevant) is True
            if r['candidate_id'] == 'FBR-R4-ORD1' and r['arm'] == 'simplified' and item['display_label'] == 'I2':
                relevant, complete = True, False
                item['adjudication_note'] = 'Independent audit: exact open-email excerpt is relevant to the compound item; read-request premise is omitted. Uniform relevance-only credit; no complete-premise credit.'
            item['uniform_relevance_semantic_support'] = relevant if r['format_valid'] else None
            item['uniform_complete_premise_support'] = complete if r['format_valid'] else None
            item['source_faithful_item'] = bool(eligible and relevant)
            item['complete_premise_source_faithful_item'] = bool(eligible and relevant and complete)
            if not r['format_valid']:
                assert item['source_faithful_item'] is False
        r['uniform_source_faithful_items'] = sum(i['source_faithful_item'] for i in r['items'])
        r['uniform_source_faithful_response'] = r['format_valid'] and all(i['source_faithful_item'] for i in r['items'])
        rows.append(r)
    assert len(seen) == 22 and sum(len(r['items']) for r in rows) == 120
    arms = {}
    for arm in ['original','simplified']:
        rr = [r for r in rows if r['arm'] == arm]
        ii = [i for r in rr for i in r['items']]
        a = copy.deepcopy(score['arms'][arm])
        a['as_reviewed_mixed_rubric_items'] = sum(i['as_reviewed_source_faithful_item'] for i in ii)
        a['source_faithful_items'] = sum(i['source_faithful_item'] for i in ii)
        a['source_faithful_expected_item_coverage'] = a['source_faithful_items'] / 60
        a['source_faithful_valid_only_item_coverage'] = a['source_faithful_items'] / a['strict_reviewable_items']
        a['complete_premise_source_faithful_items'] = sum(i['complete_premise_source_faithful_item'] for i in ii)
        a['complete_premise_expected_item_coverage'] = a['complete_premise_source_faithful_items'] / 60
        a['source_faithful_responses'] = sum(r['uniform_source_faithful_response'] for r in rr)
        a['semantic_source_support'] = 'COMPLETE_UNIFORM_RELEVANCE_WITH_COMPLETE_PREMISE_SENSITIVITY'
        a['source_faithful_reviewable_coverage'] = a['source_faithful_expected_item_coverage']
        a['raw_citations_all_responses'] = {'original':76,'simplified':48}[arm]
        a['raw_exact_citations_all_responses'] = {'original':34,'simplified':16}[arm]
        a['target_error_item_counts_overlapping'] = dict(collections.Counter(c for r in rr if r['format_valid'] for i in r['items'] for c in i['target_error_class'] if c != 'none'))
        a['by_type'] = {}
        for typ in ['multi_select','ordering']:
            subset = [r for r in rr if r['question_type'] == typ]
            items = [i for r in subset for i in r['items']]
            a['by_type'][typ] = {'responses':len(subset),'expected_items':len(items),'strict_valid':sum(r['format_valid'] for r in subset),'source_faithful_items':sum(i['source_faithful_item'] for i in items),'complete_premise_source_faithful_items':sum(i['complete_premise_source_faithful_item'] for i in items)}
        arms[arm] = a
    assert [arms[a]['source_faithful_items'] for a in arms] == [5,8]
    assert [arms[a]['complete_premise_source_faithful_items'] for a in arms] == [5,5]
    decision = {'status':'ADOPT_SIMPLIFIED_FOR_EXPLORATORY_DIAGNOSTICS_ONLY','direct_primary_unchanged':True,'frozen_necessary_gate_passes':True,'strict_validity_non_decreasing':True,'uniform_relevance_expected_coverage_improves':True,'new_schema_ambiguity_demonstrated':False,'complete_premise_gain':False,'robust_improvement_established':False,'universal_reliability_established':False,'rationale_zh':'按冻结字面口径，格式8/11持平，可来源核查项5/60至8/60，未发现新增schema歧义。仅将简化接口作为后续探索性诊断默认；引文完整前提敏感性5/60对5/60，无稳健改善证据。无效checks结构是遵循失败，不冒称schema歧义。原direct主评估不改。','no_more_interface_probe_in_this_run':True}
    agg = {'schema_version':'mrc_semantic_aggregation_v1','created_at_utc':now(),'status':'COMPLETE','expected_responses':22,'expected_items':120,'review_bindings':[binding(mc_path),binding(order_path)],'adjudication_binding':binding('review/INDEPENDENT_MC_AGGREGATION_ADJUDICATION.json'),'scores_binding':binding('runtime/interface_v3/SCORES.json'),'scores_file_remains_historical_pending_artifact':True,'uniform_primary_rubric':contract['source_faithful_item_definition'],'rubric_disclosure':'Original MC and ORDER reviewers differed on partial premise quotations. Independent postrun adjudication follows frozen literal relevance contract uniformly; complete-premise support reported separately. Mixed 5/60 vs7/60 is historical reviewer sum, not the final comparable metric.','arms':arms,'adoption_decision':decision,'responses':rows,'technical_blinding':False,'human_review':False,'review_provenance':{'MC':mc['sequencing_disclosure'],'ORDER':order['review_provenance'],'audit':audit['provenance']},'invalid_policy':contract['strict_invalid']}
    save('SEMANTIC_AGGREGATION.json',agg)
    prior = read('../day3_feedback_realization_v1/FEEDBACK_SNAPSHOT.json')
    entries=[]
    for c in prior['entries']:
        entries.append({'feedback_id':c['feedback_id'],'parent_card_status':c['status'],'prior_card_preserved':True,'status':'HISTORICAL_LOCAL_EVIDENCE_RETAINED','next_action':c['next_action'],'formal_reproduced':False,'training_use':False})
    source = next(c for c in entries if c['feedback_id']=='T4E-SOURCE_FACT_REJECTION')
    source.update(status='NEW_SCOPE_REPAIR_CONTENT_ACCEPTED_DIRECT_FAILURE_REASON_UNESTABLISHED',candidate_id='MRC-R4-MC1',gold_count=5,content_accepted=True,direct_observation={'valid':6,'exact':0,'jaccard':7/9,'missed_original_D_calls':6,'false_positive_original_F_calls':1},diagnostic_status='NOT_REVIEWABLE',old_candidate_disposition='FBR-R4-MC2 remains HOLD',next_action='不再次追问此单题；保留同family修复变体，后续按family去重。不得由无效诊断推断漏选D原因，不把Gold5强行配额化。')
    iface=next(c for c in entries if c['feedback_id']=='FBR-EVIDENCE_FIDELITY_INTERFACE')
    iface.update(status='PAIRED22_COMPLETE_EXPLORATORY_SIMPLIFIED_DEFAULT',new_observation={'strict_valid':[8,8],'expected_item_denominator_each':60,'uniform_relevance_faithful':[5,8],'complete_premise_faithful':[5,5],'fully_faithful_responses':[0,0]},next_action='结束本轮接口试验；后续探索性诊断使用冻结简化schema，仍须严格格式与引用/语义审核。短依据不得直接作为训练理由标签。')
    feedback={'schema_version':'mrc_reviewed_feedback_snapshot_v1','version':'train.feedback.minimal_repair_control.v1.reviewed','created_at_utc':now(),'parent_version':prior['version'],'parent_snapshot':binding('../day3_feedback_realization_v1/FEEDBACK_SNAPSHOT.json'),'generation_freeze':binding('GENERATION_FREEZE.json'),'semantic_aggregation':binding('SEMANTIC_AGGREGATION.json'),'candidate_review':binding('review/CANDIDATE_DIAGNOSTIC_SEMANTIC_REVIEW.json'),'entries':entries,'adoption_decision':decision,'formal_reproduced':False,'matched_feedback_AB':False,'feedback_causal_effect_established':False,'training_benefit_established':False,'faithful_hidden_reasoning_claim':False,'training_start_go':False,'all_targets_loop_closed':False}
    save('FEEDBACK_SNAPSHOT.json',feedback)
    text=['# 最小修复与接口对照反馈 v1','', '本版本接续上一轮六张卡，原卡及原始评分保留。29次固定调用已用完，本轮不新增推理。','', '简化接口仅采用为探索性诊断默认：格式8/11对8/11；统一相关性口径5/60对8/60；引文完整前提口径5/60对5/60；两臂均无整份全部可靠的响应。不能宣称可靠性、难度、memory或训练收益已获证明。','']
    for c in entries:text += [f"## {c['feedback_id']}",'',f"状态：{c['status']}",'',f"下一消费者动作：{c['next_action']}",'']
    text += ['## 交接边界','', '下一步回到共同评估准入、family去重和对称A/B训练池准备；不得将本轮同源变体计为独立新题族。素材缺额才定向补料，仍按A旧事件延续3场＋B新事件3场。下一消费者本轮只写可执行计划，不开始训练、扩充或调用Solver。']
    (P/'FEEDBACK_ZH.md').write_text(NL.join(text)+NL)
    packet={'schema_version':'mrc_next_consumer_packet_v1','created_at_utc':now(),'scope':'PLAN_ONLY_EXISTING_TRAIN_DIAGNOSTIC_AND_MAINLINE_READINESS','feedback_snapshot':binding('FEEDBACK_SNAPSHOT.json'),'feedback_content':binding('FEEDBACK_ZH.md'),'semantic_aggregation':binding('SEMANTIC_AGGREGATION.json'),'generation_freeze':binding('GENERATION_FREEZE.json'),'candidate_content':binding('review/CANDIDATE_CONTENT_REVIEW.json'),'candidate_diagnostic':binding('review/CANDIDATE_DIAGNOSTIC_SEMANTIC_REVIEW.json'),'public_questions':binding('runtime/interface_v3/PUBLIC_QUESTIONS.json'),'source_bindings':[binding(x) for x in sorted((P/'runtime/interface_v3/sources').glob('*.json'))],'common_evaluation_status_bindings':[binding('../day3_eval_recovery_v2/SUMMARY.json'),binding('../day3_eval_clearance_v1/SUMMARY.json')],'protocol_bindings':[binding('../DAILY_EXECUTION_V2_ZH.md'),binding('../PLAN_CONTINUITY_REPAIR_ZH.md')],'required_actions':['Read actual feedback semantics and source observations; verify every consumed file binding.','Record actual tool-visible read manifest yourself; do not claim API prompt trace or technical blinding.','Plan exact next deliverable for common evaluation clearance, family deduplication, symmetric A/B training pool/resource gates; no further open-ended interface experiments.','Retain literal relevance and complete-premise sensitivity; do not call observed malformed output schema ambiguity.','If source shortage truly requires expansion, preserve A3+B3; no roots or Solver calls now.','Keep training_start_go=false, full_day_complete=false; no training or model calls in this handoff.'],'next_candidates_generated':False,'program_packet_written':True,'actual_consumption_record_path':str(P/'next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json'),'training_start_go':False}
    save('next_generation/INPUT_PACKET.json',packet)
    print(json.dumps({'status':'PREPARED','arms':{a:{k:v[k] for k in ['strict_valid','source_faithful_items','complete_premise_source_faithful_items']} for a,v in arms.items()},'next_packet':binding('next_generation/INPUT_PACKET.json')},ensure_ascii=False))

def finish():
    agg=read('SEMANTIC_AGGREGATION.json')
    receipt=read('next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json')
    plan=read('next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json')
    assert receipt and plan
    tech=read('review/POSTRUN_TECHNICAL_RECOMPUTATION_V3.json')
    assert tech['status']=='PASS' and tech['actual_submitted_calls']==29
    s={'schema_version':'mrc_closeout_summary_v1','created_at_utc':now(),'status':'BOUNDED_MINIMAL_REPAIR_AND_INTERFACE_STUDY_COMPLETE','semantic_aggregation':binding('SEMANTIC_AGGREGATION.json'),'feedback_snapshot':binding('FEEDBACK_SNAPSHOT.json'),'candidate':{'candidate_id':'MRC-R4-MC1','content_accepted':True,'gold_count':5,'same_fact_family_variant':True,'old_candidate_remains_hold':True,'direct_calls':6,'direct_valid':6,'direct_exact':0,'direct_jaccard':7/9,'diagnostic_calls':1,'diagnostic_valid':0,'diagnostic_disposition':'NOT_REVIEWABLE','source_verified_diagnostic_target_failure':False,'training_admitted':0},'interface':{'questions':11,'multi_select':7,'ordering':4,'calls':22,'expected_items_each_arm':60,'arms':agg['arms'],'adoption_decision':agg['adoption_decision']},'new_solver_calls':29,'candidate_calls':7,'interface_calls':22,'new_roots':0,'new_source_turns':0,'new_training_runs':0,'training_admitted':0,'dev_test_calls':0,'training_start_go':False,'full_day_complete':False,'formal_reproduced':False,'matched_feedback_AB':False,'feedback_causal_effect_established':False,'controlled_difficulty_increase_established':False,'training_benefit_established':False,'faithful_hidden_reasoning_claim':False,'all_targets_loop_closed':False,'actual_next_consumer_read':True,'actual_next_plan':binding('next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json'),'actual_read_manifest':binding('next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json'),'cost':{k:tech[k] for k in ['total_input_tokens','total_output_tokens','total_tokens','wall_seconds','model_load_seconds','interface_shared_wall_counted_once']},'technical_recomputation':binding('review/POSTRUN_TECHNICAL_RECOMPUTATION_V3.json'),'technical_checks':5018,'technical_failures':0,'prior_sealed_files_verified':1354,'day3_cumulative_solver_calls':3417,'next_action':'Execute the bound next plan for common-evaluation clearance and family-deduplicated symmetric A/B training-pool preparation. No training until gates pass; no open-ended interface retesting.','technical_blinding':False,'human_review':False}
    save('SUMMARY.json',s)
    report='''# 最小修复与诊断接口对照结果

## 完成范围

已完成冻结的1题范围修复＋11题两臂诊断对照：6次直接答案、1次修复题依据诊断、22次接口调用，共29次。原始返回冻结后评分，来源审核覆盖14份多选响应84项＋8份排序响应36项，合计120项。新版反馈已由下一消费者实际读取并写具体后续计划。本轮没有新root、对话、训练或Dev/Test调用；不等于第三天全部完成。

## 单题范围修复

MRC-R4-MC1通过内容审核，Gold为A/B/C/D/E；旧FBR-R4-MC2仍HOLD。每个证据包独立判断充分性，全文不能补包内缺失cutoff。6次direct全部格式有效，exact为0/6，平均Jaccard77.78%；6次均漏D、1次误选F。唯一依据诊断超过每项最多2条引文，整份NOT_REVIEWABLE，不能提取final救分或据此解释漏选原因。

这是同来源、同事实family修复变体，不能算新增独立题族或难度提升。A–E共享核心事实，可能存在按cutoff是否出现的浅层解法。内容接受不等于训练准入，本轮训练准入为0。

## 配对接口结果

11题（7多选、4排序），同题同seed、实际展示映射和完整source相同；original逐字节复用旧prompt，simplified只替换任务尾部。3072输出预算、temperature0.8、top_p0.9、thinking=false相同，无grammar、重试或截断。简化同时改变键控结构、引文上限2→1及reason320→160，是接口包对照，不能归因某一组件。

| 指标 | original | simplified |
|---|---:|---:|
| strict格式有效 | 8/11 | 8/11 |
| final exact（无效仍留11分母） | 1/11 | 2/11 |
| strict可审核项 | 44/60 | 42/60 |
| 有精确引文的项 | 23/60 | 15/60 |
| 有效响应精确引文/引文数 | 27/65 | 15/42 |
| 全部响应原始精确引文/引文数（只作描述） | 34/76 | 16/48 |
| 统一字面相关性：来源可靠项 | 5/60 | 8/60 |
| 统一字面相关性：有效项分母 | 5/44 | 8/42 |
| 引文完整前提敏感性：来源可靠项 | 5/60 | 5/60 |
| 全部项来源可靠的响应 | 0/11 | 0/11 |
| input tokens | 110126 | 110346 |
| output tokens | 7145 | 4436 |

审核者最初对部分前提引文处理不同，直接相加的5/60对7/60是混合口径，不作正式比较。独立裁决按冻结契约字面统一：非空模型引文全部精确且与所述证据链相关、判断/理由及最终选择或全序对齐正确；未额外强制引文囊括全部前提。因此MC R3-MC3 simplified C/D以及ORDER R4 simplified I2可按相关性计入，但完整前提指标均排除。此解释与敏感性为事后显式裁决，原审核文件保留；不得把审核者补的参考证据算为模型引用。

冻结必要门：格式不下降、统一来源可靠覆盖改善、无新增schema歧义，在字面口径通过。决定仅将simplified作为后续探索性诊断默认，direct主评估不改；不再为本轮增加接口实验。完整前提口径无改善、每臂仅11次且无整份可靠响应，不能宣称稳健可靠性提升。字符串checks、缺外层对象等是schema遵循失败，并不能直接证明schema本身歧义。1/11到2/11不能单独决定采纳。

## 实际可用的反馈

保留来源抽取、对象/状态绑定、时间计算、选项或全序对齐、引用位置/语义与结构遵循等分开的维度。精确引文不保证推理正确：R4-MC1 original六项引文均精确，仍将12:00−10:08写成1h43；正确应为1h52。排序R4两臂仍把lighting报告置于较晚位置。无效输出只标不可审核，不从中推断模型弱点。

本轮内容修复接上了来源事实卡，但没有有效诊断原因，不授予formal reproduced。其他历史卡保留版本及证据边界；本轮不声称feedback/no-feedback、memory、训练或难度的因果收益。短理由是可见可审输出，不是隐藏CoT真值，不直接成为监督理由标签。

## 复核、成本与局限

技术重算V3通过5018项、0失败；此前工具Gold绑定及branch路径错误版本保留。旧971＋122＋261＝1354个封存文件逐项hash不变。接口V1/V2在live前失败、调用均0；V3重复JSON键防护等预检通过后才执行22次，未重跑或覆盖原SCORES。

29调用共输入274435、输出12805、合计287240 tokens。推理墙时109.69秒，包含模型加载52.77秒；接口共享56.82秒只计一次。这些时间不包括作者、审核、编排和报告耗时，本轮端到端时间明显更长，不能用推理墙时代表完整生产速度。本地GPU，无可核实货币账单。第三天累计3417次，第二天另计。

审核为LLM角色共享文件系统，不是人工独立标注或技术盲法。多选先见raw后查source，恢复工作时曾看旧candidate结构审核；排序主审核在source-only helper完成前已见raw。独立裁决也见过原始回答及审核，均披露。采用manifest实际display映射，不把PUBLIC_QUESTIONS原始标签混入评分。排序逐项final_alignment检查该项与所有其他项的相对序，不能只核rank或相邻项。

## 下一步与入口

下一消费者已经读取反馈语义、绑定和协议，产出next_generation/agent_plan/中的具体计划；文件读取回执不等于完整API prompt trace。回到共同评估准入、family去重及对称A/B训练池和资源冻结，不继续围绕个别题无止境重试。只有真实素材缺额才按A旧事件延续3场＋B新事件3场补料。训练启动与第三天全部完成保持false。

- SUMMARY.json：本轮机器摘要；SEMANTIC_AGGREGATION.json：统一口径全部22响应/120项。
- exports/QUESTIONS_WITH_GOLD_ZH.md：修复题与答案；runtime/interface_v3/PUBLIC_QUESTIONS.json：对照11题。
- exports/CANDIDATE_RESPONSES_ZH.md、exports/INTERFACE_RESPONSES_ZH.md：全部可见返回；runtime中保留完整raw。
- FEEDBACK_ZH.md、FEEDBACK_SNAPSHOT.json：版本化反馈；next_generation/INPUT_PACKET.json：实际交接输入。
- next_generation/agent_plan/PLAN_ZH.md：实际读取后的后续计划；review/：来源审核、独立裁决及版本修复。
- FINAL_MANIFEST.json：最终封存清单；最终独立审计另见review/FINAL_CLOSEOUT_AUDIT.json。
'''
    (P/'REPORT_ZH.md').write_text(report)
    (P/'README_ZH.md').write_text('''# 本轮已完成

本轮只执行冻结的最小修复与诊断接口对照：29次Solver调用，1题内容修复、11题两臂诊断。来源审核22响应/120项已完整，反馈已由下一消费者实际读取并写计划。

- [结果报告](REPORT_ZH.md)与[机器摘要](SUMMARY.json)。
- [修复题及Gold](exports/QUESTIONS_WITH_GOLD_ZH.md)、[修复题回答](exports/CANDIDATE_RESPONSES_ZH.md)、[接口全部回答](exports/INTERFACE_RESPONSES_ZH.md)。
- [统一语义审核](SEMANTIC_AGGREGATION.json)、[反馈](FEEDBACK_ZH.md)、[后续计划](next_generation/agent_plan/PLAN_ZH.md)。
- [最终封存清单](FINAL_MANIFEST.json)、[最终审核](review/FINAL_CLOSEOUT_AUDIT.json)。

修复题direct0/6；接口格式8/11对8/11，来源可靠项字面口径5/60对8/60，完整前提敏感性5/60对5/60。简化接口仅作探索性诊断默认，direct主评估不改，不宣称稳健提升。原SCORES保留其历史pending状态，最终结果见统一语义审核和本摘要。没有扩充、训练或Dev/Test调用，训练准入0，第三天尚未全部完成。
''')
    print('SUMMARY/report/README saved; pending independent final audit and manifest seal')

def seal():
    audit=read('review/FINAL_CLOSEOUT_AUDIT.json')
    assert audit['status'] == 'PASS', 'Final audit must pass before sealing'
    summary=read('SUMMARY.json')
    assert summary['actual_next_consumer_read'] and not summary['training_start_go']
    old_verified=[]
    for name in ['day3_targeted_4root_v1','day3_diagnostic_handoff_v1','day3_feedback_realization_v1']:
        old=read(f'../{name}/FINAL_MANIFEST.json')
        for b in old['files']:
            actual=binding(b['path'])
            assert actual['sha256']==b['sha256'] and actual['bytes']==b['bytes'], b['path']
        old_verified.append({'manifest':binding(f'../{name}/FINAL_MANIFEST.json'),'files':len(old['files'])})
    assert sum(x['files'] for x in old_verified)==1354
    files=[binding(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='FINAL_MANIFEST.json' and '__pycache__' not in f.parts]
    final={'schema_version':'minimal_repair_control_final_manifest_v1','created_at_utc':now(),'status':summary['status'],'files':files,'new_solver_calls':29,'direct_calls':6,'candidate_diagnostic_calls':1,'interface_calls':22,'old_sealed_verified':old_verified,'new_roots':0,'new_source_turns':0,'training_admitted':0,'new_training_runs':0,'dev_test_calls':0,'actual_next_consumer_read':True,'training_start_go':False,'full_day_complete':False,'formal_reproduced':False,'matched_feedback_AB':False,'source_faithful_items_each_arm':[5,8],'complete_premise_source_faithful_items_each_arm':[5,5],'adoption_status':summary['interface']['adoption_decision']['status'],'no_further_writes_to_sealed_directory':True}
    save('FINAL_MANIFEST.json',final)
    for b in final['files']:
        actual=binding(b['path']);assert actual==b
    print(json.dumps({'status':'SEALED','file_count':len(files),'manifest':binding('FINAL_MANIFEST.json'),'old_sealed_files_verified':1354},ensure_ascii=False))

def navigation():
    final=read('FINAL_MANIFEST.json')
    for b in final['files']:
        assert binding(b['path'])==b
    # Navigation is mutable and outside the sealed run. Keep the pre-update
    # copies already saved in this run; do not mutate the sealed directory.
    progress={'status':final['status'],'summary':binding('SUMMARY.json'),'report':binding('REPORT_ZH.md'),'manifest':binding('FINAL_MANIFEST.json'),'feedback_snapshot':binding('FEEDBACK_SNAPSHOT.json'),'semantic_aggregation':binding('SEMANTIC_AGGREGATION.json'),'actual_next_plan':binding('next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json'),'actual_read_manifest':binding('next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json'),'new_solver_calls':29,'candidate_calls':7,'interface_calls':22,'new_candidate_content_pass':1,'old_candidate_hold_preserved':True,'diagnostic_strict_valid_each_arm':[8,8],'diagnostic_expected_responses_each_arm':11,'source_faithful_expected_items_each_arm':[5,8],'source_faithful_expected_denominator_each_arm':60,'complete_premise_sensitivity_each_arm':[5,5],'adoption_status':final['adoption_status'],'actual_next_consumer_read':True,'new_roots':0,'new_source_turns':0,'training_admitted':0,'new_training_runs':0,'dev_test_calls':0,'training_start_go':False,'full_day_complete':False}
    breakdown={'day3_old187_diagnostic':2688,'day3_diversity_crossroot':258,'day3_targeted_four_root':342,'day3_new_batch_evidence_diagnostic':23,'day3_feedback_realization':77,'day3_minimal_repair_candidate':7,'day3_minimal_repair_interface':22,'day3_train_diagnostic_total':3417,'day2_excluded_from_this_total':True}
    assert sum(v for k,v in breakdown.items() if k not in ['day3_train_diagnostic_total','day2_excluded_from_this_total'])==3417
    next_action='Execute day3_minimal_repair_control_v1/next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json: common-evaluation clearance, explicit family deduplication and matched A/B training-pool/resource readiness. Preserve exploratory-only simplified diagnostic use and complete-premise sensitivity; no new open-ended interface tuning. Training waits for original protocol gates.'
    for filename in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
        p=P.parent/filename; d=json.loads(p.read_text())
        d['updated_at_utc']=now()
        d['updated_at_beijing']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        d['status']=final['status']
        d['active_workstream']='MINIMAL_REPAIR_AND_INTERFACE_STUDY_CLOSED_NEXT_MAINLINE_READINESS'
        d['next_action']=next_action
        d['new_solver_calls']=3417
        d['latest_iteration_solver_calls']=29
        d['solver_call_scope_breakdown']=breakdown
        d['minimal_repair_control_progress']=progress
        d['current_execution_summary']=binding('SUMMARY.json')
        d['current_execution_report']=binding('REPORT_ZH.md')
        d['training_start_go']=False
        d['full_day_complete']=False
        d['new_training_runs']=0
        p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+NL)
    p=P.parent/'README_ZH.md'; old=p.read_text()
    lead='''# 当前方案与进度

最新完成[最小修复与接口对照](day3_minimal_repair_control_v1/REPORT_ZH.md)：29次固定Solver调用已用完，1道范围修复内容通过、direct0/6，旧HOLD保留；11题两臂诊断与120项来源审核完成。格式8/11对8/11，统一字面相关性来源可靠项5/60对8/60，完整前提敏感性5/60对5/60，均无整份全部可靠响应。简化接口仅作后续探索性诊断默认，不宣称稳健提升；direct主评估不改。

[新版反馈](day3_minimal_repair_control_v1/FEEDBACK_ZH.md)已由下一消费者真实读取并形成[后续具体计划](day3_minimal_repair_control_v1/next_generation/agent_plan/PLAN_ZH.md)。回到共同评估准入、family去重及对称A/B训练池/资源准备，不继续无限接口微调。本轮未新增root、对话、训练或Dev/Test调用，训练准入0，training_start_go=false，第三天未全部完成。累计3417次＝2688＋258＋342＋23＋77＋7＋22，第二天另计。见[机器摘要](day3_minimal_repair_control_v1/SUMMARY.json)、[全部可见返回](day3_minimal_repair_control_v1/exports/INTERFACE_RESPONSES_ZH.md)、[封存清单](day3_minimal_repair_control_v1/FINAL_MANIFEST.json)。以下内容为历史阶段快照，当前状态以本段及CURRENT_STATUS.json为准。

'''
    if old.startswith('# 当前方案与进度'):
        old=old[len('# 当前方案与进度'):].lstrip()
    p.write_text(lead+old)
    # Read back both mutable navigation anchors.
    for filename in ['CURRENT_STATUS.json','DAY3_LATEST.json']:
        d=json.loads((P.parent/filename).read_text());assert d['new_solver_calls']==3417 and not d['training_start_go'] and not d['full_day_complete']
    print('Weekly navigation updated; sealed run left unchanged.')

if __name__ == '__main__':
    {'prepare':prepare,'finish':finish,'seal':seal,'navigation':navigation}[sys.argv[1]]()
