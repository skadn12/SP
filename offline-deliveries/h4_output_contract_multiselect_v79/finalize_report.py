"""Report v79 dependent parent families without promoting controls or deleting baseline rows."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import evaluate as ev
import compose
HERE=Path(__file__).resolve().parent
COUNTS=(2,3,4,5)

def display_module():
    key = 'v79_unchanged_display_export'
    if key not in sys.modules:
        path = HERE.parent / 'h4_option_composition_v63_qa/display_export.py'
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def distribution(questions):
    multi = [q for q in questions if q['question_type'] == 'multi_select']
    counts = Counter(len(q['gold']) for q in multi)
    roots = sorted({q['root_id'] for q in questions})
    return {'questions':len(questions),'by_type':dict(Counter(q['question_type'] for q in questions)),
        'multiselect_gold_count_histogram':{str(n):counts[n] for n in range(2,7)},
        'missing_multiselect_gold_counts':[n for n in range(2,7) if not counts[n]],
        'five_or_six_fraction':str(Fraction(counts[5]+counts[6],len(multi))) if multi else None,
        'by_root':{root:{'questions':sum(q['root_id']==root for q in questions),
            'by_type':dict(Counter(q['question_type'] for q in questions if q['root_id']==root)),
            'multiselect_gold_count_histogram':{str(n):sum(q['root_id']==root and q['question_type']=='multi_select' and len(q['gold'])==n for q in questions) for n in range(2,7)}} for root in roots}}


def statement_key(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    return q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values()))


def canonical_gold(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    values=[' '.join(q[field][label].casefold().split()) for label in q['gold']]
    return tuple(sorted(values) if field=='options' else values)


def aggregate(rows):
    calls=len(rows)
    exact=sum(r['exact'] for r in rows)
    partial=sum((Fraction(r['partial']) for r in rows),Fraction())
    return {'questions':len({r['candidate_id'] for r in rows}),'calls':calls,'exact_correct':exact,
        'exact_rate':str(Fraction(exact,calls)) if calls else None,
        'exact_rate_percent':round(100*exact/calls,2) if calls else None,
        'partial_mean':str(partial/calls) if calls else None,
        'partial_mean_percent':round(100*float(partial/calls),2) if calls else None}


def compare_pairs(pairs):
    base=aggregate([{'candidate_id':p['candidate_id'],**p['base']} for p in pairs])
    trained=aggregate([{'candidate_id':p['candidate_id'],**p['trained']} for p in pairs])
    return {'base':base,'trained':trained,
        'trained_minus_base_exact_percentage_points':round(100*(float(Fraction(trained['exact_rate']))-float(Fraction(base['exact_rate']))),2) if pairs else None,
        'trained_minus_base_partial_points':str(Fraction(trained['partial_mean'])-Fraction(base['partial_mean'])) if pairs else None,
        'paired_exact':{'both_correct':sum(p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'trained_only_correct':sum(not p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'base_only_correct':sum(p['base']['exact'] and not p['trained']['exact'] for p in pairs),
            'both_wrong':sum(not p['base']['exact'] and not p['trained']['exact'] for p in pairs)}}


def blind_all_six(questions):
    n=ev.native()
    rows=[]
    for q in questions:
        if q['question_type']!='multi_select':
            continue
        variants,_=n._variants_for(q)
        variant=variants[0]
        mapping={text:cid for cid,text in variant['content_to_text'].items()}
        gold=[mapping[q['options'][label]] for label in q['gold']]
        question={**q,'gold_content_ids':gold}
        request={'question_type':'multi_select','variant':variant}
        score=n.score_response(question,request,'<box>A,B,C,D,E,F</box>')
        independent=ev.independent_score('<box>A,B,C,D,E,F</box>',request,gold)
        ev.require(score['partial_score_fraction']==independent['partial'] and score['exact_match']==independent['exact'],'blind_score_disagreement')
        rows.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],**independent})
    return {'scope':'multiselect_only_native_scorer_equal_question_weights','aggregate':aggregate(rows),
        'by_correct_count':{str(k):aggregate([r for r in rows if r['gold_count']==k]) for k in range(2,7)},
        'questions':rows,'formula':'TP/k - FP/(6-k); k=6 has no false-positive term',
        'all_six_partial_is_zero_for_gold_two_to_five':True}


def cost_summary(requests, usable_count=None):
    actual=[r for r in requests if r['remote_http_attempt']]
    known=[r for r in actual if r['token_usage_status']=='known']
    complete=len(known)==len(actual)
    inputs=sum(r['reported_input_tokens'] for r in known)
    outputs=sum(r['reported_output_tokens'] for r in known)
    return {'http_attempts':len(actual),'prepared_requests':len(requests),'known_usage_http_attempts':len(known),
        'unknown_usage_http_attempts':len(actual)-len(known),'usage_complete':complete,
        'input_tokens':inputs if complete else 'unknown','output_tokens':outputs if complete else 'unknown',
        'known_input_token_subtotal':inputs,'known_output_token_subtotal':outputs,
        'money_cost':'unknown_no_frozen_provider_pricing','standalone_planner_calls':0,
        'content_usable_questions':usable_count,
        'http_per_content_usable_question':str(Fraction(len(actual),usable_count)) if usable_count else None}


def export_questions(home,stem,questions,title,provenance):
    display=display_module()
    views=[display.display_view(q) for q in questions]
    for q,view in zip(questions,views): display.replay_view(q,view)
    ev.write(Path(home)/(stem+'_CANDIDATES.json'),{'questions':questions,'candidate_count':len(questions),
        'original_question_and_gold_unchanged':True,'training_allowed':False,'formal_training_admission':False,'provenance':provenance})
    ev.write(Path(home)/(stem+'_DISPLAY.json'),{'questions':views,'gold_blind_permutation_replayed':True})
    with (Path(home)/(stem+'_QUESTIONS.md')).open('x') as stream:
        stream.write(display.markdown(views,title=title)+'\n')


def sidecar_index(sidecar,questions):
    members=sidecar['members'];indexed={m['candidate_id']:m for m in members}
    ev.require(len(indexed)==len(members) and set(indexed)=={q['candidate_id'] for q in questions},'family_member_coverage')
    for q in questions:
        m=indexed[q['candidate_id']]
        ev.require(m['root_id']==q['root_id']==m['train_root_id']==m['split_group'] and m['gold_count']==len(q['gold']),'family_root_or_count_changed')
        ev.require(m['role'] in ('control6','child') and m['new_candidate']==(m['role']=='child'),'family_role_changed')
        ev.require((len(q['gold'])==6)==(m['role']=='control6'),'control_count_changed')
    for family in {m['family_id'] for m in members}:
        rows=[m for m in members if m['family_id']==family]
        controls=[m for m in rows if m['role']=='control6']
        ev.require(len(controls)==1 and len({m['root_id'] for m in rows})==1,'family_control_or_root_coverage')
        ev.require(all(m['parent_candidate_id']==controls[0]['candidate_id'] and m['fixed_fact_ids']==controls[0]['fixed_fact_ids'] for m in rows),'family_parent_or_fixed_facts_changed')
    return indexed


def merge_selected_children(baseline,selected,members):
    """Preserve the entire baseline verbatim; only novel selected child rows append."""
    merged=list(baseline);seen={statement_key(q):q for q in baseline};ids={q['candidate_id'] for q in baseline}
    added=[];duplicates=[];controls=[]
    for q in selected:
        m=members[q['candidate_id']]
        if m['role']=='control6':
            controls.append({'candidate_id':q['candidate_id'],'already_in_baseline':q['candidate_id'] in ids,'reason':'CONTROL_ONLY_NEVER_NEW_ADMISSION'})
            continue
        ev.require(m['role']=='child','unknown_family_role')
        key=statement_key(q)
        if key in seen:
            original=seen[key];disagreement=canonical_gold(q)!=canonical_gold(original)
            duplicates.append({'candidate_id':q['candidate_id'],'retained_candidate_id':original['candidate_id'],'root_id':q['root_id'],'reason':'same_root_type_normalized_full_six_statement_set','gold_content_disagreement':disagreement})
            ev.require(not disagreement,'duplicate_gold_content_disagreement')
            continue
        ev.require(q['candidate_id'] not in ids,'child_candidate_id_collision')
        seen[key]=q;ids.add(q['candidate_id']);merged.append(q);added.append(q)
    ev.require(merged[:len(baseline)]==baseline,'baseline_changed_during_merge')
    return merged,added,duplicates,controls


def required_count_coverage(questions,config):
    hist=distribution(questions)['multiselect_gold_count_histogram']
    targets={str(k):config['target_distribution'][str(k)] for k in COUNTS}
    actual={str(k):hist[str(k)] for k in COUNTS}
    shortfalls={str(k):max(0,targets[str(k)]-actual[str(k)]) for k in COUNTS}
    return {'required_counts':list(COUNTS),'targets':targets,'actual_counts':actual,'shortfalls':shortfalls,
        'missing_counts':[k for k in COUNTS if actual[str(k)]==0],
        'all_required_counts_present':all(actual[str(k)]>0 for k in COUNTS),'all_target_counts_met':not any(shortfalls.values()),
        'scope':'current_batch_children_only','controls_counted_as_new':False,'target_is_not_actual_coverage':True}


def family_stage_report(assembled,usable,evaluated,selected,added,baseline,members,config):
    stages={'assembled':assembled,'content_usable':usable,'evaluated':evaluated,'selected':selected,'merged':added}
    rows=[];baseline_ids={q['candidate_id'] for q in baseline}
    for family in sorted({m['family_id'] for m in members.values()}):
        family_ids={cid for cid,m in members.items() if m['family_id']==family}
        control_id=next(cid for cid in family_ids if members[cid]['role']=='control6')
        children={cid for cid in family_ids if members[cid]['role']=='child'}
        by_count={}
        for k in COUNTS:
            ids=sorted(cid for cid in children if members[cid]['gold_count']==k)
            by_count[str(k)]={'target':1,'candidate_ids':ids,**{name:sum(q['candidate_id'] in ids for q in qs) for name,qs in stages.items()}}
        rows.append({'family_id':family,'root_id':members[control_id]['root_id'],'parent_candidate_id':control_id,'parent_origin':members[control_id].get('parent_origin','existing_frozen_six_gold_question'),'parent_pool_path':members[control_id].get('parent_pool_path'),'parent_control':{
            **{name:any(q['candidate_id']==control_id for q in qs) for name,qs in stages.items() if name!='merged'},
            'newly_merged':False,'already_in_baseline':control_id in baseline_ids,'retained_in_baseline':control_id in baseline_ids,
            'reevaluation_changes_old_admission':False},'children_by_count':by_count,'independent_sampling_unit':'root/family','siblings_are_independent':False})
    child_stages={name:[q for q in qs if members[q['candidate_id']]['role']=='child'] for name,qs in stages.items()}
    control_stages={name:[q for q in qs if members[q['candidate_id']]['role']=='control6'] for name,qs in stages.items() if name!='merged'}
    return {'families':rows,'family_count':len(rows),'children_stages':{name:distribution(qs) for name,qs in child_stages.items()},
        'control_stages':{name:distribution(qs) for name,qs in control_stages.items()},
        'required_count_coverage':{name:required_count_coverage(qs,config) for name,qs in child_stages.items()},
        'independent_family_count':len(rows),'siblings_are_independent_samples':False,'max_family_weight_not_question_count':True}


def comparison(verification,usable,members):
    responses={side:{r['request_id']:r for r in verification['models'][side]['responses']} for side in ('base','trained')}
    ev.require(set(responses['base'])==set(responses['trained']),'unpaired_responses')
    qs={q['candidate_id']:q for q in usable}
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    pairs=[]
    for rid,left in responses['base'].items():
        right=responses['trained'][rid];cid=left['candidate_id'];q=qs[cid];m=members[cid]
        ev.require(cid==right['candidate_id'],'paired_question_changed')
        pairs.append({'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'family_id':m['family_id'],'role':m['role'],'gold_count':len(q['gold']),'base':left,'trained':right,'trained_selected':cid in selected_ids})
    children=[p for p in pairs if p['role']=='child'];controls=[p for p in pairs if p['role']=='control6']
    families={family:{'children':compare_pairs([p for p in children if p['family_id']==family]),'control6':compare_pairs([p for p in controls if p['family_id']==family]),
        'children_by_correct_count':{str(k):compare_pairs([p for p in children if p['family_id']==family and p['gold_count']==k]) for k in COUNTS}}
        for family in sorted({m['family_id'] for m in members.values()})}
    macro={}
    for side in ('base','trained'):
        values=[f['children'][side] for f in families.values() if f['children'][side]['calls']]
        macro[side]={'contributing_families':len(values),'equal_family_weight_partial_mean':str(sum((Fraction(v['partial_mean']) for v in values),Fraction())/len(values)) if values else None,
            'equal_family_weight_exact_rate':str(sum((Fraction(v['exact_rate']) for v in values),Fraction())/len(values)) if values else None}
    return {'primary_scope':'frozen_content_usable_children_descriptive_with_family_dependence','primary':compare_pairs(children),'control6':compare_pairs(controls),
        'all_evaluated_descriptive':compare_pairs(pairs),'by_family':families,'equal_family_weight_children':macro,
        'by_correct_count':{str(k):compare_pairs([p for p in children if p['gold_count']==k]) for k in COUNTS},
        'selected_children_secondary':compare_pairs([p for p in children if p['trained_selected']]),
        'questions':[{'candidate_id':cid,'family_id':members[cid]['family_id'],'role':members[cid]['role'],'gold_count':len(q['gold']),'trained_selected':cid in selected_ids,**compare_pairs([p for p in pairs if p['candidate_id']==cid])} for cid,q in qs.items()],
        'paired_inputs':verification['paired_inputs'],'training_started':False,'independent_family_count':len(families),'siblings_are_independent_samples':False,
        'limitation':f"{len(families)} source-root families with shared fixed facts within each family. Every family uses its frozen complete parent source without expansion. Parent review modes and previously known Gold are disclosed per route; reused parent judgments are not fresh blind reviews. Each current child requires a saved public-text/full-source judgment before private child Gold and current Solver results are read. Writer fields and root coverage change together, so outcomes cannot be attributed to format alone. Sibling questions and repeated calls are dependent. Descriptive paired checkpoint comparison only; no independent-sample significance claim and no new training."},pairs


def planned_scope(config):
    slots=config['slots']
    families=[slot.get('family_id',slot['slot_id']) for slot in slots]
    roots=[slot['root_id'] for slot in slots]
    ev.require(len(set(families))==len(families) and len(set(roots))==len(roots),'planned_family_or_root_collision')
    return {'family_count':len(families),'planned_child_count':sum(len(slot.get('target_counts',COUNTS)) for slot in slots),
        'planned_control_count':len(slots),'independent_family_count':len(set(roots))}


def validate_baseline(baseline,config):
    ev.require(len({q['candidate_id'] for q in baseline})==len(baseline),'baseline_duplicate_candidate_id')
    if 'baseline_expected_question_count' in config:
        ev.require(len(baseline)==config['baseline_expected_question_count'],'baseline_count_changed')
    if 'baseline_expected_type_counts' in config:
        ev.require(dict(Counter(q['question_type'] for q in baseline))==config['baseline_expected_type_counts'],'baseline_type_counts_changed')


def parent_routes(config):
    fields=('family_id','root_id','parent_candidate_id','previous_parent_candidate_id','parent_origin','parent_pool_path',
        'parent_review_path','parent_review_sha256','parent_provenance_path','parent_provenance_sha256',
        'parent_content_reused_from','parent_review_mode','parent_gold_known_before_current_review')
    return [{k:slot.get(k) for k in fields} for slot in config['slots']]


def generation_records(home,config):
    import asset_generation
    requests=[];slots=[]
    for slot in config['slots']:
        fid=slot.get('family_id',slot['slot_id']);run=Path(home)/'assets'/fid
        asset_generation.verify(run,complete=True)
        complete=ev.read(run/'GENERATION.COMPLETE.json');provenance=ev.read(run/'ROUND_PROVENANCE.json')
        round_ids=[row['round_id'] for row in provenance['rounds']]
        ev.require(round_ids in (['initial'],['initial','supplement']),'generation_round_order_or_budget')
        ev.require({p.name for p in (run/'rounds').iterdir()}==set(round_ids),'generation_round_coverage')
        local=[];rounds=[]
        for row in provenance['rounds']:
            rid=row['round_id'];folder=run/'rounds'/rid;rc=ev.read(folder/'GENERATION.COMPLETE.json')
            paths=sorted((folder/'requests').glob('*/REQUEST.json'));rr=[]
            for path in paths:
                request_folder=path.parent;remote=(request_folder/'HTTP_ATTEMPT.json').exists()
                response=ev.read(request_folder/'RESPONSE.json') if (request_folder/'RESPONSE.json').exists() else {}
                metadata=response.get('metadata',{});usage=metadata.get('usage') or {}
                ins=metadata.get('input_tokens',usage.get('prompt_tokens'));outs=metadata.get('output_tokens',usage.get('completion_tokens'))
                known=type(ins) is int and type(outs) is int and ins>=0 and outs>=0
                rr.append({'family_id':fid,'root_id':slot['root_id'],'round_id':rid,'target_fact_ids':rc['target_fact_ids'],
                    'request':ev.bind(path),'remote_http_attempt':remote,'reported_input_tokens':ins,'reported_output_tokens':outs,
                    'token_usage_status':('known' if known else 'unknown') if remote else 'not_called',
                    'token_usage_source':'provider_metadata_no_estimation'})
            ev.require(len(rr)==1 and sum(x['remote_http_attempt'] for x in rr)==rc['http_attempts']<=1,'generation_round_call_budget_or_count')
            ev.require(rc['round_id']==rid and rc['target_fact_ids']==row['target_fact_ids'] and rc['status']==row['status']
                and rc['http_attempts']==row['http_attempts'],'generation_round_provenance_disagreement')
            review=ev.read(folder/'ASSET_REVIEW.json');assets=ev.read(folder/'ASSETS.json')
            accepted=sum(v['usable'] and v['relation']=='contradicts' for v in review['versions'])
            ev.require(accepted==row['approved_distinct_false_facts'],'generation_round_approval_count')
            errors=[]
            for path in paths:
                validation=path.parent/'VALIDATION.json'
                if validation.exists():errors.extend(ev.read(validation)['mechanical_errors'])
            rounds.append({'round_id':rid,'target_fact_ids':row['target_fact_ids'],'status':rc['status'],'http_attempts':rc['http_attempts'],
                'generation_complete':ev.bind(folder/'GENERATION.COMPLETE.json'),'provenance':row,
                'mechanically_accepted_false_versions':sum(isinstance(f.get('contradicted'),dict) for f in assets['facts']),
                'source_approved_false_versions':accepted,'mechanical_rejections':errors,
                'usable_parent_facts':sum(v['usable'] and v['relation']=='supports' for v in review['parents']),
                'cost':cost_summary(rr),'source_first_reviewed':True})
            local.extend(rr)
        ev.require(len(local)<=2 and sum(x['remote_http_attempt'] for x in local)==complete['http_attempts']
            and complete['bounded_supplements']==len(rounds)-1,'generation_family_call_budget_or_count')
        review=ev.read(run/'ASSET_REVIEW.json');assets=ev.read(run/'ASSETS.json')
        final_count=sum(isinstance(f.get('contradicted'),dict) for f in assets['facts'])
        ev.require(final_count==sum(row['source_approved_false_versions'] for row in rounds),'generation_final_approval_count')
        slots.append({'family_id':fid,'root_id':slot['root_id'],'http_attempts':complete['http_attempts'],
            'generation_complete':ev.bind(run/'GENERATION.COMPLETE.json'),'round_provenance':ev.bind(run/'ROUND_PROVENANCE.json'),
            'rounds':rounds,'bounded_supplements':complete['bounded_supplements'],'fixed_parent_facts':len(assets['facts']),
            'accepted_false_versions':final_count,'usable_parent_facts':sum(v['usable'] and v['relation']=='supports' for v in review['parents']),
            'usable_false_versions':sum(v['usable'] and v['relation']=='contradicts' for v in review['versions']),
            'parent_origin':slot.get('parent_origin'),'parent_review_mode':slot.get('parent_review_mode'),
            'parent_gold_known_before_current_review':slot.get('parent_gold_known_before_current_review'),
            'new_variants_source_first_reviewed':True,'aggregate_is_new_independent_blind_review':False})
        requests.extend(local)
    ev.require(sum(x['remote_http_attempt'] for x in requests)<=config.get('max_http',8)<=8,'generation_total_budget')
    return requests,slots


def generation_round_statistics(requests,families):
    by_round={}
    for rid in ('initial','supplement'):
        rows=[row for family in families for row in family['rounds'] if row['round_id']==rid]
        by_round[rid]={'families_attempted':len(rows),'targeted_facts':sum(len(row['target_fact_ids']) for row in rows),
            'mechanically_accepted_false_versions':sum(row['mechanically_accepted_false_versions'] for row in rows),
            'source_approved_false_versions':sum(row['source_approved_false_versions'] for row in rows),
            'cost':cost_summary([request for request in requests if request['round_id']==rid])}
    return {'scope':'descriptive_generation_round_accounting','by_round':by_round,'families':families,
        'accepted_initial_versions_preserved':True,'supplement_targets_only_initial_unresolved_facts':True,
        'aggregate_is_new_independent_blind_review':False,'solver_scores_used_for_generation':False,
        'limitation':'Targeted supplements may add source-approved contradictions; they do not guarantee child count coverage, difficulty admission, or independent samples.'}


def report_markdown(s):
    b=s['base_trained_comparison']['primary']['base'];t=s['base_trained_comparison']['primary']['trained']
    lines=['# v79 精简替换句输出与扩大 root 覆盖的 Gold 数量修复试验','',f"{s['family_count']} 组母题共组装 {s['assembled_count']} 题，其中对照 {s['control_count']} 题、子题 {s['assembled_child_count']} 题。子题内容可用 {s['content_usable_child_count']} 题，原难度规则入选 {s['selected_count']} 题，实际新增 {s['added_count']} 题；原 {s['baseline_count']} 题完整保留，合并后 {s['merged_count']} 题。",'',
        '本轮按冻结 CONFIG 的逐 root 来源和母题审核模式执行：既有母题内容复用及先前已知 Gold 明确披露，复用判断不称为新的盲审；新增来源母题先依据公开文本和完整 SOURCE 保存判断，再核对私有 Gold。每 family 的完整来源字节保持冻结，不作新对话扩展。每道子题固定使用所属母题的六事实，各取一个核实版本；对照只用于评估，不作为新增候选入池。历史 prior 分数仅作来源记录，不作为本轮结果或新子题生成反馈。','',
        'writer 每行只返回 fact_id 和 replacement_proposition（字符串或 null），取消 unavailable_reason，也不输出 relation、rationale 或引文。程序绑定冻结母题引文，内部关系/理由仅作未审核占位，新替换句经独立完整来源判定通过后才可用。固定引文、复制拦截、来源语义审核、有界补充、评分和入选阈值保持原规则。本轮同时精简 writer 字段并扩大来源 root 覆盖，结果不能单独归因于格式改动。','',
        '母题来源、逐项复用或修复记录及审核绑定见 [PARENT_SELECTION](execution/PARENT_SELECTION.json)、[CONFIG](CONFIG.json) 及逐 family 的 parent_review 记录；下表按配置披露本轮母题身份与审核模式。','',
        '| Family / root | 母题 | 来源模式 | 当前来源审查前已知母题 Gold |','|---|---|---|---|']
    for route in s['parent_routes']:
        known=route.get('parent_gold_known_before_current_review')
        lines.append('| '+str(route.get('family_id'))+' / '+str(route.get('root_id'))+' | '+str(route.get('parent_candidate_id'))+' | '+str(route.get('parent_review_mode') or '未声明，见绑定来源')+' | '+('是' if known is True else '否' if known is False else '未声明，见绑定来源')+' |')
    lines.extend(['',
        '| 子题阶段 | 2项 | 3项 | 4项 | 5项 |','|---|---:|---:|---:|---:|']
    )
    labels={'assembled':'组装','content_usable':'内容可用','evaluated':'已评估','selected':'难度入选','merged':'实际新增'}
    for stage,label in labels.items():
        h=s['family_stages']['children_stages'][stage]['multiselect_gold_count_histogram'];lines.append('| '+label+' | '+' | '.join(str(h[str(k)]) for k in COUNTS)+' |')
    target_text='、'.join(str(k)+'项 '+str(s['target_distribution'][str(k)])+' 题' for k in COUNTS)
    lines.extend(['','各档目标：'+target_text+'；目标不代表审核通过或已入池。','', '| Family / 母题 | 2项 组/审/评/选/增 | 3项 组/审/评/选/增 | 4项 组/审/评/选/增 | 5项 组/审/评/选/增 |','|---|---:|---:|---:|---:|'])
    for family in s['family_stages']['families']:
        cells=['/'.join(str(family['children_by_count'][str(k)][stage]) for stage in labels) for k in COUNTS]
        lines.append('| '+family['family_id']+' / '+family['parent_candidate_id']+' / '+family['parent_origin']+' | '+' | '.join(cells)+' |')
    lines.extend(['','| 六项全真对照 | 已组装 | 内容可用 | 已评估 | 本次难度入选 | 原已在池 | 新增入池 |','|---|---:|---:|---:|---:|---:|---:|'])
    for family in s['family_stages']['families']:
        x=family['parent_control'];lines.append('| '+family['family_id']+' | '+' | '.join(str(int(x[k])) for k in ('assembled','content_usable','evaluated','selected','already_in_baseline','newly_merged'))+' |')
    lines.extend(['',f"子题主比较：Base 完全正确 {b['exact_correct']}/{b['calls']}，部分分均值 {b['partial_mean']}；已有训练检查点完全正确 {t['exact_correct']}/{t['calls']}，部分分均值 {t['partial_mean']}。",'',
        f"本轮有 {s['independent_family_count']} 个来源 root/family；同组子题共享事实，重复调用和兄弟题不视为独立样本。逐组结果、母题对照与等 family 权重均值另列，不作显著性或因果结论。",
        f"既有 {s['baseline_count']} 题逐对象保留，包括此前已记录的内容问题；本轮未重新认证整个旧池的答案可靠性，也不提高旧题可靠性声明。",
        f"生成 HTTP {s['generation_cost']['http_attempts']} 次；每模型评估 {s['calls_per_model']} 次。生成输入/输出 tokens：{s['generation_cost']['input_tokens']}/{s['generation_cost']['output_tokens']}；费用未获提供方定价，记为未知。",
        f"首轮生成 HTTP {s['generation_round_statistics']['by_round']['initial']['cost']['http_attempts']} 次，补充生成 HTTP {s['generation_round_statistics']['by_round']['supplement']['cost']['http_attempts']} 次；经来源审核可用的反例版本分别为 {s['generation_round_statistics']['by_round']['initial']['source_approved_false_versions']} / {s['generation_round_statistics']['by_round']['supplement']['source_approved_false_versions']}。补充只处理首轮未通过的事实，已通过版本保持原样；补充不保证覆盖全部档位或达到难度要求。",
        '引用由程序绑定到冻结母题，模型原始输出原样留存。母题审核按逐 root 的复用或来源先审记录披露；每轮新替换句和最终新子题先依据公开命题和完整 SOURCE 保存判断，再检查私有关系或 Gold；最终汇总明确记录其派生性质。',
        '继续使用原生难度规则与 TP/k − FP/(6−k) 部分评分；k=6 不计假阳性项。盲选全六对2–5项子题得0分，对六项全真对照得1分。未启动训练。','',
        '[分轮生成统计](GENERATION_ROUND_STATISTICS.json) · [Family阶段和缺口](FAMILY_STAGES.json) · [新增入选子题](SELECTED_QUESTIONS.md) · [六正确项对照](CONTROLS_QUESTIONS.md) · [累计候选池](MERGED_USABLE_QUESTIONS.md) · [逐组模型比较](BASE_TRAINED_COMPARISON.json)'])
    return '\n'.join(lines)+'\n'


def finalize(home=HERE):
    home=Path(home).resolve();ev.require(not (home/'RESULT_SUMMARY.json').exists(),'final_report_already_exists')
    verification=ev.verify(home);pool,usable,statuses,_=ev.reviewed_content(home);config=ev.read(home/'CONFIG.json')
    assembled=pool['questions'];members=sidecar_index(ev.read(home/'FAMILY_SIDECAR.json'),assembled)
    baseline=ev.read(config['baseline_pool'])['questions'];validate_baseline(baseline,config)
    scope=planned_scope(config)
    previous_bindings=ev.read(config['baseline_result_bindings'])['files']
    for binding in previous_bindings:ev.verify_binding(binding)
    ev.require(ev.bind(config['baseline_pool']) in previous_bindings,'baseline_not_previously_bound')
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    selected_all=[q for q in usable if q['candidate_id'] in selected_ids]
    merged,added,duplicates,excluded_controls=merge_selected_children(baseline,selected_all,members)
    selected=[q for q in selected_all if members[q['candidate_id']]['role']=='child']
    controls=[q for q in assembled if members[q['candidate_id']]['role']=='control6']
    evaluated_ids={q['candidate_id'] for q in verification['models']['trained']['questions']}
    ev.require(evaluated_ids=={q['candidate_id'] for q in usable},'evaluation_content_coverage')
    evaluated=[q for q in usable if q['candidate_id'] in evaluated_ids]
    stages=family_stage_report(assembled,usable,evaluated,selected_all,added,baseline,members,config)
    compare,pairs=comparison(verification,usable,members);requests,slots=generation_records(home,config)
    round_statistics=generation_round_statistics(requests,slots)
    generation_complete=ev.read(home/'GENERATION.COMPLETE.json')
    ev.require(generation_complete.get('training_allowed') is False and generation_complete.get('automatic_followup') is False,'generation_scope_changed')
    ev.require(generation_complete['http_attempts']==sum(r['remote_http_attempt'] for r in requests),'generation_total_count_changed')
    if 'plan' in generation_complete:
        ev.verify_binding(generation_complete['plan'])
        for b in ev.read(generation_complete['plan']['path'])['files']:ev.verify_binding(b)
    for row in generation_complete.get('slots',[]):
        if 'completion' in row:ev.verify_binding(row['completion'])
    children=lambda qs:[q for q in qs if members[q['candidate_id']]['role']=='child']
    ev.require(stages['family_count']==scope['family_count'],'assembled_family_count_changed')
    summary={'version':'v79_immutable_parent_family_results_v1',**scope,
        'baseline_count':len(baseline),'assembled_count':len(assembled),'assembled_child_count':len(children(assembled)),'control_count':len(controls),
        'content_usable_count':len(usable),'content_usable_child_count':len(children(usable)),'known_issue_count':len(assembled)-len(usable),
        'evaluated_count':len(evaluated),'selected_count':len(selected),'selected_control_count':len(selected_all)-len(selected),
        'added_count':len(added),'merged_count':len(merged),'exact_merge_duplicates':duplicates,'selected_controls_excluded_from_new_admission':excluded_controls,
        'baseline_verbatim_preserved':merged[:len(baseline)]==baseline,'baseline_content_recertified':False,'known_baseline_content_issues_preserved_without_reliability_upgrade':True,'control_reevaluation_changes_old_admission':False,
        'target_distribution':config['target_distribution'],'family_stages':stages,'required_count_coverage':stages['required_count_coverage'],
        'evaluation_status':verification['evaluation_status'],'calls_per_model':verification['calls_per_model'],'content_statuses':statuses,
        'allocation':ev.read(home/'ALLOCATION.json'),'base_trained_comparison':compare,'generation_cost':cost_summary(requests,len(children(usable))),
        'generation_families':slots,'generation_round_statistics':round_statistics,'siblings_are_independent_samples':False,
        'training_allowed':False,'training_started':False,'automatic_followup':False,'new_expansion_tokens':0,
        'parent_routes':parent_routes(config),'parent_semantic_review_mode':config.get('parent_semantic_review_mode'),
        'parent_gold_known_before_current_review_is_per_route':True,'parent_content_reused_from':config.get('parent_content_reused_from'),
        'writer_wire_contract':config.get('writer_output_contract'),'writer_unavailable_reason_field':False,'writer_relation_and_rationale_fields':False,
        'current_route_changes':config.get('current_route_changes',['writer_fields_simplified','source_root_coverage_expanded']),
        'format_and_root_coverage_changed_together':True,'format_only_causal_attribution_supported':False,
        'full_family_sources_frozen_without_expansion':True,'new_variants_source_first_review_required':True,'historical_difficulty_use':'provenance_or_preselection_only_no_current_solver_feedback',
        'previous_bound_files_unchanged':len(previous_bindings),'raw_native_evaluation_replay_status':verification['status'],
        'distributions':{name:distribution(qs) for name,qs in [('assembled',assembled),('content_usable',usable),('evaluated',evaluated),('selected',selected),('added',added),('controls',controls),('baseline',baseline),('merged',merged)]}}
    provenance={'requested_model':pool['provenance']['requested_model'],'baseline':ev.bind(config['baseline_pool']),'assembly':ev.bind(home/'ASSEMBLY_RECEIPTS.json'),'family_sidecar':ev.bind(home/'FAMILY_SIDECAR.json'),'content_review':ev.bind(home/'FINAL_CONTENT_REVIEW.COMPLETE.json')}
    for stem,qs,title in [('ASSEMBLED',assembled,'v79 全部母题和子题'),('CONTROLS',controls,'v79 六项全真母题对照'),('CONTENT_USABLE',usable,'v79 内容可用题'),('SELECTED',selected,'v79 原难度规则入选子题'),('ADDED',added,'v79 实际新增子题'),('MERGED_USABLE',merged,'累计可用候选题')]:export_questions(home,stem,qs,title,provenance)
    ev.write(home/'FAMILY_STAGES.json',stages)
    ev.write(home/'COVERAGE.json',{'family_stages':stages,'required_count_coverage':stages['required_count_coverage'],'allocation':summary['allocation'],'target_is_not_actual_coverage':True})
    ev.write(home/'GENERATION_COST.json',{'summary':summary['generation_cost'],'by_round':round_statistics['by_round'],'requests':requests})
    ev.write(home/'GENERATION_ROUND_STATISTICS.json',round_statistics)
    ev.write(home/'BASE_TRAINED_COMPARISON.json',compare);ev.write(home/'PAIRED_RAW_SCORES.json',{'pairs':pairs})
    ev.write(home/'BLIND_ALL_SIX_BASELINE.json',{name:blind_all_six(qs) for name,qs in [('controls',controls),('content_usable_children',children(usable)),('selected_children',selected),('merged',merged)]})
    ev.write(home/'RAW_RECOMPUTATION_AUDIT.json',{'status':'PASS','audit_kind':'raw_recomputation_by_implementation_author','models_called':0,'frozen_files_modified':0,'evaluation':verification,'previous_bound_files_unchanged':len(previous_bindings),'baseline_verbatim_preserved':True,'control_not_new_admission':True})
    ev.write(home/'RESULT_SUMMARY.json',summary)
    with (home/'REPORT.md').open('x') as f:f.write(report_markdown(summary))
    files={p for p in home.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='RESULT_BINDINGS.json' and not p.name.endswith('.log')}
    files|={Path(config['baseline_pool']),Path(config['baseline_result_bindings'])}
    files|={Path(b['path']) for b in ev.read(home/'ASSEMBLY_RECEIPTS.json')['inputs']}
    files|={Path(b['path']) for b in ev.read(home/'EVALUATION_PREPARATION.COMPLETE.json')['files']}
    ev.write(home/'RESULT_BINDINGS.json',{'scope':'all_v79_final_artifacts_and_direct_frozen_inputs_except_mutable_logs','files':[ev.bind(p) for p in sorted(files)]})
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=HERE);args=parser.parse_args();s=finalize(args.report)
    print(json.dumps({k:s[k] for k in ('assembled_count','content_usable_count','selected_count','added_count','merged_count','calls_per_model')},indent=2))
