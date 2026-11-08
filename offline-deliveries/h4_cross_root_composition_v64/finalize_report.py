"""Recompute bounded v64 outcomes from frozen assets, reviews and native responses."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import evaluate as ev

HERE = Path(__file__).resolve().parent


def display_module():
    key = 'v64_unchanged_display_export'
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


def merge_exact(baseline, new):
    seen, kept, duplicates = {}, [], []
    for q in baseline + new:
        key=statement_key(q)
        if key in seen:
            original=seen[key]
            duplicates.append({'candidate_id':q['candidate_id'],'retained_candidate_id':original['candidate_id'],
                'root_id':q['root_id'],'reason':'same_root_type_normalized_full_six_statement_set',
                'gold_content_disagreement':canonical_gold(q)!=canonical_gold(original)})
        else:
            seen[key]=q
            kept.append(q)
    return kept,duplicates


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


def comparison(verification, usable):
    by_side={side:{r['request_id']:r for r in verification['models'][side]['responses']} for side in ('base','trained')}
    ev.require(set(by_side['base'])==set(by_side['trained']),'unpaired_responses')
    questions={q['candidate_id']:q for q in usable}
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    pairs=[]
    for rid,left in by_side['base'].items():
        right=by_side['trained'][rid]
        q=questions[left['candidate_id']]
        ev.require(left['candidate_id']==right['candidate_id'],'paired_question_mismatch')
        pairs.append({'request_id':rid,'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count':len(q['gold']),
            'base':left,'trained':right,'trained_selected':q['candidate_id'] in selected_ids})
    result={'primary_scope':'all_content_usable_questions_frozen_before_solver','primary':compare_pairs(pairs),
        'by_root':{root:compare_pairs([p for p in pairs if p['root_id']==root]) for root in sorted({q['root_id'] for q in usable})},
        'by_correct_count':{str(n):compare_pairs([p for p in pairs if p['gold_count']==n]) for n in range(2,7)},
        'selected_secondary':compare_pairs([p for p in pairs if p['trained_selected']]),
        'questions':[{'candidate_id':qid,'root_id':q['root_id'],'gold_count':len(q['gold']),
            'trained_selected':qid in selected_ids,**compare_pairs([p for p in pairs if p['candidate_id']==qid])} for qid,q in questions.items()],
        'paired_inputs':verification['paired_inputs'],'training_started':False,
        'limitation':'historical_checkpoint_vs_base_on_this_bounded_five_root_question_set; not_second_round_training_effect_or_randomized_generalization_estimate'}
    return result,pairs


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


def generation_records(home, config):
    import asset_generation
    requests,slots=[],[]
    for slot in config['slots']:
        directory=Path(home)/'assets'/slot['slot_id']
        asset_generation.verify(directory,complete=True)
        complete=ev.read(directory/'GENERATION.COMPLETE.json')
        rows=[]
        for request_path in sorted((directory/'requests').glob('*/REQUEST.json')):
            folder=request_path.parent
            remote=(folder/'HTTP_ATTEMPT.json').exists()
            response=ev.read(folder/'RESPONSE.json') if (folder/'RESPONSE.json').exists() else {}
            metadata=response.get('metadata',{})
            usage=metadata.get('usage') or {}
            inputs=metadata.get('input_tokens',usage.get('prompt_tokens'))
            outputs=metadata.get('output_tokens',usage.get('completion_tokens'))
            error=ev.read(folder/'ERROR.json') if (folder/'ERROR.json').exists() else None
            row={'slot_id':slot['slot_id'],'root_id':slot['root_id'],'request':ev.bind(request_path),
                'remote_http_attempt':remote,'reported_input_tokens':inputs,'reported_output_tokens':outputs,
                'token_usage_status':('known' if type(inputs) is int and type(outputs) is int else 'unknown') if remote else 'not_called',
                'token_usage_source':'provider_response_metadata_no_estimation','error':error}
            rows.append(row)
        ev.require(len(rows)<=2 and len(rows)==complete['requests_prepared']
                   and sum(r['remote_http_attempt'] for r in rows)==complete['http_attempts'],
                   'asset_request_budget_or_count_mismatch')
        assets=ev.read(directory/'ASSETS.json') if (directory/'ASSETS.json').exists() else None
        facts=assets.get('facts',[]) if assets else []
        slots.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],'initial_target_count':slot['initial_target_count'],
            'generation_complete':complete,'requests':len(rows),'http_attempts':sum(r['remote_http_attempt'] for r in rows),
            'asset_fact_groups':len(facts),'asset_supported_versions':sum(bool(f.get('supported')) for f in facts),
            'asset_contradicted_versions':sum(bool(f.get('contradicted')) for f in facts),
            'asset_review':ev.read(directory/'ASSET_REVIEW.json') if (directory/'ASSET_REVIEW.json').exists() else None})
        requests.extend(rows)
    ev.require(sum(r['remote_http_attempt'] for r in requests)<=config['max_http'],'total_asset_http_budget')
    return requests,slots


def source_summary(home,config):
    inventory=ev.read(Path(home)/'ROOT_INVENTORY.json')
    roots={r['root_id']:r for r in inventory['roots']}
    out=[]
    for root in sorted({s['root_id'] for s in config['slots']}):
        row=roots[root]
        ev.verify_binding(row['history'])
        history=ev.read(row['history']['path'])
        ev.require(len(history['turns'])==row['full_history']['turns'],'source_turn_count_changed')
        out.append({'root_id':root,'history':row['history'],'full_history':row['full_history'],
            'formal_original_plus_R01':row['formal_original_plus_R01'],'retained_by_route':row['retained_by_route'],
            'body_token_definition':row['body_token_definition'],'new_expansion_tokens':0})
    return out


def export_questions(home,stem,questions,title,provenance):
    display=display_module()
    views=[display.display_view(q) for q in questions]
    for q,view in zip(questions,views): display.replay_view(q,view)
    ev.write(Path(home)/(stem+'_CANDIDATES.json'),{'questions':questions,'candidate_count':len(questions),
        'original_question_and_gold_unchanged':True,'training_allowed':False,'formal_training_admission':False,'provenance':provenance})
    ev.write(Path(home)/(stem+'_DISPLAY.json'),{'questions':views,'gold_blind_permutation_replayed':True})
    with (Path(home)/(stem+'_QUESTIONS.md')).open('x') as stream:
        stream.write(display.markdown(views,title=title)+'\n')


def report_markdown(summary):
    primary=summary['base_trained_comparison']['primary']
    b,t=primary['base'],primary['trained']
    lines=['# v64 跨 root 命题组装结果','',
        f"{summary['planned_slots']} 个名额，组装 {summary['assembled_count']} 题，内容可用 {summary['content_usable_count']} 题，原难度规则入选 {summary['selected_count']} 题。与原57题合并后 {summary['merged_count']} 题。",
        '每题从六个不同事实中各取一个已核实命题版本，Gold由实际支持关系导出。源对话未扩充；当前只评估内容可用题。','',
        '| 分布口径 | 2项 | 3项 | 4项 | 5项 | 6项 |','|---|---:|---:|---:|---:|---:|']
    for label,key in [('起始构造目标','planned'),('已组装','assembled'),('内容可用新题','content_usable'),('难度筛选入选','selected'),('合并可用候选','merged')]:
        hist=summary['distributions'][key]['multiselect_gold_count_histogram']
        lines.append('| '+label+' | '+' | '.join(str(hist[str(n)]) for n in range(2,7))+' |')
    lines.extend(['',f"最终新增入池尚缺数量：{summary['distributions']['selected']['missing_multiselect_gold_counts']}；合并池尚缺：{summary['distributions']['merged']['missing_multiselect_gold_counts']}。",
        '', '| 内容可用集主比较 | Base | 已有训练检查点 |','|---|---:|---:|',
        f"| 完全正确 | {b['exact_correct']}/{b['calls']} | {t['exact_correct']}/{t['calls']} |",
        f"| 部分得分均值 | {b['partial_mean']} | {t['partial_mean']} |",'',
        '主比较使用评分前冻结的全部内容可用题。两模型输入、随机呈现、seed与评分一致；难度筛选结果单列，第二轮训练未启动。',
        f"生成 HTTP {summary['generation_cost']['http_attempts']} 次；每模型实际 {summary['calls_per_model']} 次。提供方输入/输出 tokens：{summary['generation_cost']['input_tokens']}/{summary['generation_cost']['output_tokens']}。",
        '原生部分得分为 TP/k − FP/(6−k)，k=6不计后一项；盲选六项对2–5真题得0分、全6真题得1分。',
        '这是有限五root题集上的已有训练检查点与Base比较，不表示第二轮训练收益。当前批次结束，不自动追加补题。','',
        '[新增入选题](SELECTED_QUESTIONS.md) · [合并候选题](MERGED_USABLE_QUESTIONS.md) · [逐题模型比较](BASE_TRAINED_COMPARISON.json) · [实际覆盖](COVERAGE.json)'])
    return '\n'.join(lines)+'\n'


def finalize(home=HERE):
    home=Path(home)
    ev.require(not (home/'RESULT_SUMMARY.json').exists(),'final_report_already_exists')
    verification=ev.verify(home)
    pool,usable,statuses,_=ev.reviewed_content(home)
    config=ev.read(home/'CONFIG.json')
    generation_complete=ev.read(home/'GENERATION.COMPLETE.json')
    ev.require(generation_complete['training_allowed'] is False and generation_complete['automatic_followup'] is False,
               'generation_completion_scope_changed')
    ev.verify_binding(generation_complete['plan'])
    for item in ev.read(generation_complete['plan']['path'])['files']: ev.verify_binding(item)
    assembled=pool['questions']
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    selected=[q for q in usable if q['candidate_id'] in selected_ids]
    old=ev.read(config['baseline_pool'])
    baseline=old['questions']
    ev.require(len(baseline)==57 and len({q['candidate_id'] for q in baseline})==57,'baseline57_changed')
    previous_bindings=ev.read(config['baseline_result_bindings'])['files']
    for binding in previous_bindings: ev.verify_binding(binding)
    ev.require(ev.bind(config['baseline_pool']) in previous_bindings,'baseline_pool_not_previously_bound')
    merged,duplicates=merge_exact(baseline,selected)
    ev.require(not any(r['gold_content_disagreement'] for r in duplicates),'duplicate_gold_content_disagreement')
    compare,pairs=comparison(verification,usable)
    requests,slots=generation_records(home,config)
    ev.require(generation_complete['http_attempts']==sum(r['remote_http_attempt'] for r in requests),
               'aggregate_generation_http_count_mismatch')
    ev.require({s['slot_id'] for s in generation_complete['slots']}=={s['slot_id'] for s in config['slots']},
               'generation_completion_slot_coverage')
    for slot in generation_complete['slots']: ev.verify_binding(slot['completion'])
    # The asset/assembly implementation owns exact raw-response and review reconstruction.
    generation_audit={'status':'PASS','slots_verified':len(config['slots']),
        'exact_request_feedback_source_and_asset_projection_replayed':True,
        'models_called_by_audit':0}
    allocation=ev.read(home/'ALLOCATION.json')
    planned=[{'candidate_id':s['slot_id'],'root_id':s['root_id'],'question_type':'multi_select','gold':list(range(s['initial_target_count']))} for s in config['slots']]
    distributions={name:distribution(rows) for name,rows in [('planned',planned),('assembled',assembled),
        ('content_usable',usable),('selected',selected),('baseline',baseline),('merged',merged)]}
    summary={'version':'v64_verified_composition_results_v1','planned_slots':len(config['slots']),
        'assembled_count':len(assembled),'content_usable_count':len(usable),'known_issue_count':len(assembled)-len(usable),
        'selected_count':len(selected),'merged_count':len(merged),'exact_merge_duplicates':duplicates,
        'calls_per_model':verification['calls_per_model'],'distributions':distributions,'allocation':allocation,
        'content_statuses':statuses,'generation_cost':cost_summary(requests,len(usable)),
        'generation_slots':slots,'base_trained_comparison':compare,'source_summaries':source_summary(home,config),
        'new_expansion_tokens':0,'training_allowed':False,'training_started':False,'automatic_followup':False,
        'previous_bound_files_unchanged':len(previous_bindings),
        'generation_audit':generation_audit,'raw_native_evaluation_replay_status':verification['status']}
    provenance={'requested_model':pool['provenance']['requested_model'],'baseline':ev.bind(config['baseline_pool']),
        'assembly':ev.bind(home/'ASSEMBLY_RECEIPTS.json'),'content_review':ev.bind(home/'FINAL_CONTENT_REVIEW.COMPLETE.json')}
    for stem,rows,title in [('ASSEMBLED',assembled,'v64 全部组装题'),('CONTENT_USABLE',usable,'v64 内容可用题'),
                           ('SELECTED',selected,'v64 原难度规则入选题'),('MERGED_USABLE',merged,'累计可用候选题')]:
        export_questions(home,stem,rows,title,provenance)
    ev.write(home/'COVERAGE.json',{'distributions':distributions,'allocation':allocation,'target_is_not_actual_coverage':True})
    ev.write(home/'GENERATION_COST.json',{'summary':summary['generation_cost'],'requests':requests})
    ev.write(home/'BASE_TRAINED_COMPARISON.json',compare)
    ev.write(home/'PAIRED_RAW_SCORES.json',{'pairs':pairs})
    ev.write(home/'BLIND_ALL_SIX_BASELINE.json',{name:blind_all_six(rows) for name,rows in
        [('baseline',baseline),('content_usable',usable),('selected',selected),('merged',merged)]})
    ev.write(home/'RAW_RECOMPUTATION_AUDIT.json',{'status':'PASS','audit_kind':'raw_recomputation_by_implementation_author',
        'models_called':0,'frozen_files_modified':0,'generation':generation_audit,
        'evaluation':verification,'previous_bound_files_unchanged':len(previous_bindings)})
    ev.write(home/'RESULT_SUMMARY.json',summary)
    with (home/'REPORT.md').open('x') as stream: stream.write(report_markdown(summary))
    files=[p for p in home.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name!='RESULT_BINDINGS.json' and not p.name.endswith('.log')]
    files += [Path(config['baseline_pool']),Path(config['baseline_result_bindings'])]
    files += [Path(s['history_path']) for s in config['slots']]
    files += [Path(p) for s in config['slots'] for p in s['prior_pool_paths']]
    files += [Path(row['path']) for row in ev.read(home/'EVALUATION_PREPARATION.COMPLETE.json')['files']]
    ev.write(home/'RESULT_BINDINGS.json',{'scope':'all_v64_final_artifacts_and_direct_frozen_inputs_except_mutable_process_logs',
        'files':[ev.bind(p) for p in sorted(set(files))]})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=HERE)
    args=parser.parse_args()
    result=finalize(args.report)
    print(json.dumps({k:result[k] for k in ('assembled_count','content_usable_count','selected_count','merged_count','calls_per_model')},indent=2))
