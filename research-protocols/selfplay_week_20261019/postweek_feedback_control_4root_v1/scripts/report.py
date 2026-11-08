"""Assemble final paired-root descriptive report from immutable real artifacts."""
import csv, io, json
from collections import Counter
from control import BASE, read, write, bind, verify, now, SCENES


def status(value):
    if isinstance(value, str): return value
    assert isinstance(value, dict)
    return value.get('status', value.get('verdict', value.get('decision')))


def metric(rows):
    n = len(rows)
    pairwise_correct = sum(r.get('pairwise_correct') or 0 for r in rows)
    pairwise_total = sum(r.get('pairwise_total') or 0 for r in rows)
    return {'n':n, 'valid':sum(r['valid'] for r in rows), 'correct':sum(r['exact'] for r in rows),
            'accuracy':sum(r['exact'] for r in rows)/n if n else None,
            'false_positive':sum(r['false_positive'] or 0 for r in rows),
            'false_negative':sum(r['false_negative'] or 0 for r in rows),
            'ordering_pairwise_correct':pairwise_correct,'ordering_pairwise_total':pairwise_total,
            'ordering_pairwise_accuracy':pairwise_correct/pairwise_total if pairwise_total else None}


def attribution_metric(counter):
    opportunities = sum(counter[s] for s in ('PASS','HOLD','FAIL'))
    return {'opportunities':opportunities, 'pass':counter['PASS'], 'hold':counter['HOLD'],
            'fail':counter['FAIL'], 'not_applicable':counter['NOT_APPLICABLE'],
            'pass_rate_on_opportunities':counter['PASS']/opportunities if opportunities else None}


def report():
    protocol = read(BASE/'PROTOCOL_FREEZE.json')
    roots, arms = protocol['roots'], protocol['conditions']
    raw = read(BASE/'ORIGINAL_QUESTION_QUALITY_FREEZE.json')
    frozen = read(BASE/'QUESTION_FREEZE.json')
    validation = read(BASE/'REASON_REVIEW_VALIDATION.json')
    assert validation['status']=='PASS_STRUCTURAL_BINDINGS_AND_ACCOUNTING_ONLY'
    assert validation['records']==frozen['admitted_questions']
    for binding in validation['bindings']:
        assert bind(binding['path'])==binding
    out = BASE/'solver_runtime/execution'
    scores = read(out/'SCORES.json')['records']
    raw_freeze = read(out/'RAW_RESPONSE_FREEZE.json')
    assert raw_freeze['status'] == 'ALL_REQUESTS_TERMINAL_RAW_SEALED'
    units, reviews, reason_records = [], [], []
    for root in roots:
        for arm in arms:
            unit = BASE/'sources'/root/arm
            initial = read(BASE/'inputs/initial_sources'/(root+'.json'))
            final = unit/'FINAL_SOURCE.json'
            scenes = [unit/s.replace(':','_') for s in SCENES]
            accepted = [p for p in scenes if (p/'ACCEPTED.json').exists()]
            warnings = [p for p in accepted if any(c['status'] != 'PASS' for c in read(p/'REVIEW.json')['checks'].values())]
            first = [r for r in raw['decisions'] if r['root_id']==root and r['condition']==arm]
            post = [r for r in frozen['decisions'] if r['root_id']==root and r['condition']==arm]
            assert len(first) == len(post) == 7
            u = {'root_id':root,'condition':arm,'initial_turns':len(initial['turns']),
                 'accepted_scenes':len(accepted),'warning_scenes':len(warnings),
                 'accepted_added_turns':14*len(accepted),'final_turns':len(read(final)['turns']) if final.exists() else None,
                 'source_complete':final.exists(),'first_pass_admitted':sum(r['status']=='ADMIT' for r in first),
                 'post_transform_admitted':sum(r['status']=='ADMIT' for r in post),
                 'source_binding':bind(final) if final.exists() else None}
            u['question_quality_dimensions'] = {}
            for phase, directory in (('original_first_pass','question_reviews'),('presentation_supplement','normalized_question_reviews')):
                qrows = read(BASE/directory/root/arm/'REVIEW.json')['questions']
                u['question_quality_dimensions'][phase] = {
                    'submitted_gold_matches':sum(r.get('submitted_gold_matches') is True for r in qrows),
                    'operation_alignment_failures':sum(r.get('operation_alignment')=='FAIL' for r in qrows),
                    'check_statuses':{k:dict(Counter(status(r['checks'][k]) for r in qrows))
                        for k in ('grounding','unique_key','actual_operation','presentation','foils')},
                }
            admitted = {r['candidate_id'] for r in post if r['status']=='ADMIT'}
            for branch in ('direct','reason'):
                cell = [s for s in scores if s['root_id']==root and s['condition']==arm and s['branch']==branch]
                assert {s['candidate_id'] for s in cell} == admitted
                u[branch] = metric(cell)
                u[branch+'_main_mc2_5'] = metric([s for s in cell if s['question_type']=='multi_select' and not s['mc6_control']])
                u[branch+'_ordering'] = metric([s for s in cell if s['question_type']=='ordering'])
                u[branch+'_mc6_control'] = metric([s for s in cell if s['mc6_control']])
            if admitted:
                rp = BASE/'reason_semantic_reviews'/root/arm/'REVIEW.json'
                review = read(rp)
                assert review['root_id']==root and review['condition']==arm
                assert review['private_scores_read'] is False and review['weakness_cards_read'] is False
                assert review['raw_freeze_binding']==bind(out/'RAW_RESPONSE_FREEZE.json')
                assert review['source_binding']==bind(final)
                assert {r['candidate_id'] for r in review['records']} == admitted
                verify(review['key_freeze_binding'])
                for r in review['records']:
                    verify(r['raw_binding'])
                    assert status(r['decision']) in ('PASS','HOLD','FAIL')
                    reason_records.append({**r,'root_id':root,'condition':arm})
                reviews.append(bind(rp))
                u['strict_reason_review'] = dict(Counter(status(r['decision']) for r in review['records']))
            else:
                u['strict_reason_review'] = {}
            units.append(u)
    assert len(scores) == 2*frozen['admitted_questions'] <= 112
    assert len(reason_records) == frozen['admitted_questions']
    condition_cells = []
    for arm in arms:
        for branch in ('direct','reason'):
            rows = [r for r in scores if r['condition']==arm and r['branch']==branch]
            condition_cells.append({'condition':arm,'branch':branch,'all':metric(rows),
                'main_mc2_5':metric([r for r in rows if r['question_type']=='multi_select' and not r['mc6_control']]),
                'ordering':metric([r for r in rows if r['question_type']=='ordering']),
                'mc6_control':metric([r for r in rows if r['mc6_control']])})
    paired = []
    for root in roots:
        g,t = [next(u for u in units if u['root_id']==root and u['condition']==a) for a in arms]
        common_slots = set(r['slot_id'] for r in frozen['decisions'] if r['root_id']==root and r['condition']=='GENERIC' and r['status']=='ADMIT') & set(r['slot_id'] for r in frozen['decisions'] if r['root_id']==root and r['condition']=='TARGETED' and r['status']=='ADMIT')
        p = {'root_id':root,'original_admission_targeted_minus_generic':t['first_pass_admitted']-g['first_pass_admitted'],
             'supplement_admission_targeted_minus_generic':t['post_transform_admitted']-g['post_transform_admitted'],
             'common_admitted_slots':sorted(common_slots),'common_slots_n':len(common_slots)}
        common_groups = {
            'main_mc2_5': common_slots & {'MC2','MC3','MC4','MC5'},
            'ordering': common_slots & {'ORDER1','ORDER2'},
            'mc6_control': common_slots & {'MC6'},
        }
        p['common_slots_by_question_group'] = {k:sorted(v) for k,v in common_groups.items()}
        p['group_contrasts'] = {}
        for branch in ('direct','reason'):
            for arm in arms:
                rows = [r for r in scores if r['root_id']==root and r['condition']==arm and r['branch']==branch
                        and r['candidate_id'].rsplit('__',1)[-1] in common_slots]
                assert len(rows)==len(common_slots)
                p[arm+'_'+branch+'_common_slots']=metric(rows)
            gm,tm=p['GENERIC_'+branch+'_common_slots'],p['TARGETED_'+branch+'_common_slots']
            p[branch+'_common_accuracy_targeted_minus_generic']=tm['accuracy']-gm['accuracy'] if common_slots else None
            for group, group_slots in common_groups.items():
                cells = {}
                for arm in arms:
                    cells[arm] = metric([r for r in scores if r['root_id']==root and r['condition']==arm
                        and r['branch']==branch and r['candidate_id'].rsplit('__',1)[-1] in group_slots])
                    assert cells[arm]['n'] == len(group_slots)
                p['group_contrasts'][branch+'_'+group] = {
                    'common_slots':sorted(group_slots), 'conditions':cells,
                    'accuracy_targeted_minus_generic':cells['TARGETED']['accuracy']-cells['GENERIC']['accuracy'] if group_slots else None,
                }
        paired.append(p)
    semantic = []
    for arm in arms:
        rows=[r for r in reason_records if r['condition']==arm]
        item_dimensions={k:Counter() for k in ('judgment_verdict','citation_sufficiency','citation_precision','reason_source_support','actor_attribution')}
        answer_dimensions={k:Counter() for k in ('answer_reason_relation','answer_reason_final_consistency','answer_reason_source_support','answer_reason_actor_attribution')}
        citation_classes=Counter()
        for r in rows:
            items=r['item_findings']
            if isinstance(items,dict):items=list(items.values())
            for item in items:
                for k,counter in item_dimensions.items():
                    v=status(item[k]); assert v in ('PASS','HOLD','FAIL','NOT_APPLICABLE');counter[v]+=1
                for citation in item['citations']:citation_classes[citation['classification']]+=1
            for k,counter in answer_dimensions.items():counter[status(r[k])]+=1
        semantic.append({'condition':arm,'records':len(rows),'strict_verdicts':dict(Counter(status(r['decision']) for r in rows)),
                         'item_dimensions':{k:dict(v) for k,v in item_dimensions.items()},
                         'answer_dimensions':{k:dict(v) for k,v in answer_dimensions.items()},
                         'citation_classes':dict(citation_classes),
                         'actor_attribution':{'item_reasons':attribution_metric(item_dimensions['actor_attribution']),
                            'answer_reasons':attribution_metric(answer_dimensions['answer_reason_actor_attribution'])},
                         'not_applicable_attribution_excluded_from_opportunity_denominator':True})
    result={'schema_version':'fbc1_final_descriptive_paired_report_v1','created_at_utc':now(),'status':'COMPLETED',
            'independent_units':4,'unit':'paired root','roots':roots,'units':units,'paired_root_contrasts':paired,
            'accepted_scenes':sum(u['accepted_scenes'] for u in units),'accepted_added_turns':sum(u['accepted_added_turns'] for u in units),
            'source_complete_units':sum(u['source_complete'] for u in units),'planned_question_slots':56,
            'paired_roots_with_common_questions_by_group':{group:sum(bool(p['common_slots_by_question_group'][group]) for p in paired)
                for group in ('main_mc2_5','ordering','mc6_control')},
            'original_first_pass_admitted':sum(u['first_pass_admitted'] for u in units),
            'question_quality_dimensions':{phase:{
                'submitted_gold_matches':sum(u['question_quality_dimensions'][phase]['submitted_gold_matches'] for u in units),
                'operation_alignment_failures':sum(u['question_quality_dimensions'][phase]['operation_alignment_failures'] for u in units),
                'check_statuses':{k:dict(sum((Counter(u['question_quality_dimensions'][phase]['check_statuses'][k]) for u in units),Counter()))
                    for k in ('grounding','unique_key','actual_operation','presentation','foils')},
                } for phase in ('original_first_pass','presentation_supplement')},
            'admission_by_question_group':{
                phase:{group:sum(r['status']=='ADMIT' and r['slot_id'] in slots for r in ledger['decisions'])
                       for group,slots in {'main_mc2_5':{'MC2','MC3','MC4','MC5'},'ordering':{'ORDER1','ORDER2'},'mc6_control':{'MC6'}}.items()}
                for phase,ledger in (('original_first_pass',raw),('exploratory_presentation_supplement',frozen))},
            'supplement_admitted':frozen['admitted_questions'],'solver_requests':len(scores),
            'condition_solver_cells':condition_cells,'reason_semantic_cells':semantic,
            'mc_gold_coverage':{a:dict(Counter(r['gold_count'] for r in scores if r['condition']==a and r['branch']=='direct' and r['question_type']=='multi_select')) for a in arms},
            'raw_returned_requests':raw_freeze['returned_requests'],'solver_wall_seconds':raw_freeze['wall_seconds'],
            'limitations':['Exploratory four paired Train roots; initial sources historically seen by M-clean, not unseen-root generalization.',
                'Shared presentation supplement chosen after observing the first root. Original yields are preserved; post-transform results are not the unchanged preregistered primary analysis.',
                'Generation and independent reviews use AI agents; conditions visible. No claim of fully blinded human annotation.',
                'Conditional difficulty on admitted questions and visible-reason errors do not establish feedback training benefit or stable formal model weakness.',
                'Different conditional admission sets can confound raw difficulty comparisons. Common admitted operation slots are reported per paired root.',
                'MC6 is an all-correct audit control. Always-select-all succeeds there and is not evidence of distractor quality.',
                'Visible explanations are observable outputs, not validated faithful hidden chain of thought. Citation quality and final correctness are separate.',
                'Each branch has one sampled response per admitted question with its own fixed seed at temperature 0.8; observed direct-versus-reason differences are descriptive, not replicated causal effects.',
                'No training, final training-pool admission, question-only controls, retries or extra permutations. Memory held fixed as full source.'],
            'formal_weakness_claim':False,'causal_feedback_benefit_claim':False,'training_performed':False,
            'bindings':[bind(BASE/p) for p in ('PROTOCOL_FREEZE.json','PRESENTATION_AMENDMENT.json','PRESENTATION_AMENDMENT_REVIEW.json','ORIGINAL_QUESTION_QUALITY_FREEZE.json','QUESTION_FREEZE.json','inputs/EXPORT_FREEZE.json','REASON_SEMANTIC_REVIEW_RUBRIC.json','REASON_REVIEW_IMPLEMENTATION_CLARIFICATION.json','REASON_REVIEW_VALIDATION.json')]+[bind(out/p) for p in ('RAW_RESPONSE_FREEZE.json','SCORES.json','ANALYSIS_SUMMARY.json')]+reviews}
    write(BASE/'FINAL_REPORT.json',result)
    text=['# FBC1：4-root 配对反馈小试结果','',f'完成 {result["source_complete_units"]}/8 组完整来源，审核通过 {result["accepted_scenes"]}/48 场，新增 {result["accepted_added_turns"]} 条发言（turn，每条为一名参与者的一次发言）。独立分析单位是四对 root。','',
          f'原始首次题目准入 {result["original_first_pass_admitted"]}/56；统一呈现处理后准入 {result["supplement_admitted"]}/56。后者是观察到共有提示问题后的探索性补充，不替代原始结果。',
          f'原始 MC2–5 准入 {result["admission_by_question_group"]["original_first_pass"]["main_mc2_5"]}/32；呈现处理后 {result["admission_by_question_group"]["exploratory_presentation_supplement"]["main_mc2_5"]}/32。原始总准入数包含排序和 MC6 控制题，不能等同于多选主任务已稳定。',
          f'运行 {result["solver_requests"]} 次 B1 请求，返回 {result["raw_returned_requests"]} 次；每题一次完整来源直接作答和一次完整来源可见理由。GPU运行耗时 {result["solver_wall_seconds"]:.1f} 秒。','',
          '| Root | 条件 | 完成场数 | 新增发言条数 | 原始准入/7 | 处理后准入/7 | 直接答案正确 | 可见理由最终答案正确 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for u in units:text.append(f'| {u["root_id"].removeprefix("cc-train-episode-")} | {u["condition"]} | {u["accepted_scenes"]} | {u["accepted_added_turns"]} | {u["first_pass_admitted"]} | {u["post_transform_admitted"]} | {u["direct"]["correct"]}/{u["direct"]["n"]} | {u["reason"]["correct"]}/{u["reason"]["n"]} |')
    text += ['', '## 题型分开统计','', '| 条件 | 输出方式 | MC2–5正确 | 排序整题正确 | 排序成对关系正确 | MC6全对控制正确 |','|---|---|---:|---:|---:|---:|']
    for c in condition_cells:
        values = [str(c[k]['correct'])+'/'+str(c[k]['n']) for k in ('main_mc2_5','ordering')]
        values += [str(c['ordering']['ordering_pairwise_correct'])+'/'+str(c['ordering']['ordering_pairwise_total']),
                   str(c['mc6_control']['correct'])+'/'+str(c['mc6_control']['n'])]
        text.append('| '+c['condition']+' | '+c['branch']+' | '+' | '.join(values)+' |')
    coverage=result['paired_roots_with_common_questions_by_group']
    text += ['', '配对比较可用root数：MC2–5 '+str(coverage['main_mc2_5'])+'/4，排序 '+str(coverage['ordering'])+'/4，MC6控制 '+str(coverage['mc6_control'])+'/4。分题型共同准入槽位及两条件结果见 JSON 的 paired_root_contrasts；不能把仅有一侧准入的题直接当成配对收益。']
    text += ['', '## 可见理由的独立语义审核','', '| 条件 | 整条PASS | HOLD | FAIL |','|---|---:|---:|---:|']
    for c in semantic:text.append('| '+c['condition']+' | '+' | '.join(str(c['strict_verdicts'].get(k,0)) for k in ('PASS','HOLD','FAIL'))+' |')
    text += ['', '最终答案正确不等于理由、说话人归属或引用可靠。逐项判断、引用充分性、精确度、来源支持和归属机会分母见 FINAL_REPORT.json；全部原始响应保留在 solver_runtime/execution/raw。','', '## 解释边界','']
    text += ['- '+s for s in result['limitations']]
    text += ['', '## 文件索引','', '- FINAL_REPORT.json：完整机器可读指标、四对root对照、绑定和限制。','- ORIGINAL_QUESTION_QUALITY_FREEZE.json：原始首轮准入/失败。','- QUESTION_FREEZE.json：固定呈现处理后的准入/失败。','- sources/：原始来源、逐场不可变草稿、消费回执和独立审核。','- questions/ 与 normalized_questions/：原稿和统一处理版本，失败稿均保留。','- question_reviews/ 与 normalized_question_reviews/：原始及处理后独立内容审核。','- solver_runtime/execution/：完整请求、prompt、schema、原始响应、封存及评分。','- reason_semantic_reviews/：先冻结独立key、后读raw的理由/引用/归属审核。','']
    path=BASE/'FINAL_REPORT_ZH.md'
    with path.open('x') as f:f.write('\n'.join(text))
    csv_path=BASE/'PAIRED_ROOT_RESULTS.csv'
    with csv_path.open('x',newline='') as f:
        fields=['root_id','condition','initial_turns','accepted_scenes','warning_scenes','accepted_added_turns','final_turns','source_complete','first_pass_admitted','post_transform_admitted','direct_correct','direct_n','reason_correct','reason_n']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for u in units:writer.writerow({**{k:u[k] for k in fields if k in u},'direct_correct':u['direct']['correct'],'direct_n':u['direct']['n'],'reason_correct':u['reason']['correct'],'reason_n':u['reason']['n']})
    write(BASE/'FINAL_ARTIFACT_MANIFEST.json',{'status':'COMPLETED','created_at_utc':now(),'report':bind(BASE/'FINAL_REPORT.json'),'readme':bind(path),'paired_table':bind(csv_path),'script_binding':bind(BASE/'scripts/report.py')})
    return {k:result[k] for k in ('status','source_complete_units','accepted_scenes','accepted_added_turns','original_first_pass_admitted','supplement_admitted','solver_requests','condition_solver_cells','reason_semantic_cells')}


if __name__=='__main__':
    print(json.dumps(report(),ensure_ascii=False))
