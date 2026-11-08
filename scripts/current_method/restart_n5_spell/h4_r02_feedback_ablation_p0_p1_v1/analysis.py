"""P1 denominator audits, descriptive paired metrics and blinded review packets."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import io
import itertools
import json
from pathlib import Path
from .freeze import ARMS, ROOTS, DEFAULT, RUNS, METHOD, load, rows, source, write, write_rows, frozen_bytes, sha_obj
from ..core import canonical_json
from ..h4_round0_b0_k3_difficulty_v1 import candidate_request_count, ROLLOUTS
from ..h4_round0_b0_k3_difficulty_v1.contract import position_variants, score_output
from ..h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round1_question_bank_lifecycle_weight_freeze_v1.run import stem_signature, gate_fingerprint_from_candidate_result

MODELS=('M0','M_clean_policy','M_inclusive_policy')
TYPES=('single_choice','multi_select','ordering')
MODES=('contract_guided','free_exploration')
CONTEXTS=('full_context','no_context')

def mean(values):
    return sum(values)/len(values) if values and all(x is not None for x in values) else None

def selected_payload(c):
    return {k:c.get(k) for k in ('root_id','candidate_id','question_type','question','options','items','gold','evidence_refs')}

def blinded_case(candidate, historical_qa, transcript, checks):
    """Whitelist factual content, retaining every turn while removing provenance."""
    question_fields=('question_type','question','options','items','gold','evidence_refs')
    return {'blind_id':'B'+sha_obj(selected_payload(candidate)), 'root_id':candidate['root_id'],
        'candidate':{k:candidate.get(k) for k in question_fields},
        'historical_relation':{k:historical_qa.get(k) for k in question_fields},
        'transcript':{'root_id':transcript['root_id'],'turns':[
            {k:turn[k] for k in ('turn_ref','session_ref','scene_ref','speaker','text','origin') if k in turn}
            for turn in transcript['turns']]},
        'review_instructions':checks,
        'prohibited':'Do not inspect arm mapping or Solver scores; do not alter labels/admission.'}

def audit_candidate_requests(candidate,result):
    """Verify original position/rollout/seed schedule and rescore raw output."""
    cid,root=candidate['candidate_id'],candidate['root_id']
    assert result['candidate_id']==cid and result['root_id']==root and result['question_type']==candidate['question_type']
    variants={v['variant_id']:v for v in position_variants(candidate)}
    expected={(context,variant,slot) for context in CONTEXTS for variant in variants for slot in range(ROLLOUTS)}
    assert result['request_count']==len(result['requests'])==candidate_request_count(candidate)
    actual=set(); ids=set()
    for request in result['requests']:
        key=(request['context_mode'],request['variant_id'],request['slot_id'])
        assert key in expected and key not in actual, f'duplicate_or_unknown_request_slot:{cid}:{key}'
        actual.add(key)
        variant=variants[request['variant_id']]
        expected_id=f"{cid}__{request['variant_id']}__{request['context_mode']}__s{request['slot_id']:02d}"
        assert request['request_id']==expected_id and expected_id not in ids
        ids.add(expected_id)
        assert request['candidate_id']==cid and request['root_id']==root and request['question_type']==candidate['question_type']
        assert request['seed']==stable_seed(root,cid,request['slot_id'])
        for field in ('display_to_canonical','displayed_gold','canonical_gold'):
            assert request[field]==variant[field], f'request_gold_mapping_drift:{cid}:{field}'
        assert type(request['scored']) is bool
        if request['scored']:
            assert type(request['correct']) is bool
        else:
            assert request['correct'] is None
        if request.get('raw_output') is None or (request.get('backend_metadata') or {}).get('finish_reason')=='length':
            assert request['scored'] is False
        else:
            rescored=score_output(request['raw_output'],candidate,variant)
            assert all(request[k]==rescored[k] for k in ('scored','correct','canonical_prediction')), f'raw_scoring_drift:{expected_id}'
    assert actual==expected
    return len(actual)

def formatted_rate(value):
    return 'NA' if value is None else f'{value:.6f}'

def qa_roots(run,arm):
    qa=run/arm/'qa'; manifest=load(qa/'manifest.json')
    for entry in manifest['roots']:
        yield entry['root_id'],qa/'shards'/entry['shard']/'roots'/entry['root_id']

def review(run):
    policy=load(run/'audit/statistics_and_review_policy.json')['review']
    mapping=[]; diagnostics=[]; corpus={}; selection_groups=[]
    for arm in ARMS:
        for root,rd in qa_roots(run,arm):
            candidates=load(rd/'candidates_valid.json'); transcript=load(rd/'expanded_dialogue_clean.json')
            frozen=load(run/'inputs/shared'/root/'expanded_dialogue_clean.json'); assert transcript==frozen
            packet=load(run/'inputs/feedback_on'/root/'feedback_packet.json')
            for qtype in TYPES:
                group=[c for c in candidates if c['question_type']==qtype]
                chosen=sorted(group,key=lambda c:(sha_obj(selected_payload(c)),canonical_json(selected_payload(c))))[:2]
                selection_groups.append({'arm':arm,'root_id':root,'question_type':qtype,
                                         'available_count':len(group),'selected_count':len(chosen),
                                         'all_available_selected':len(group)<=2})
                for c in chosen:
                    digest=sha_obj(selected_payload(c)); bid='B'+digest
                    mapping.append({'blind_id':bid,'arm':arm,'root_id':root,'candidate_id':c['candidate_id'],'question_type':qtype,'selection_hash':digest,'group_candidate_count':len(group)})
                    case=blinded_case(c,packet['historical_qa'],transcript,policy['checks'])
                    write(run/'diagnostics/blind_review/cases'/(bid+'.json'),case)
            stems=Counter(stem_signature(c) for c in candidates)
            r02=sum(any(ref.startswith('R02:') for ref in c.get('evidence_refs',[])) for c in candidates)
            contracts=load(rd/'reasoning_contracts.json')
            intent_r02=sum('R02:' in json.dumps(c,ensure_ascii=False) for c in contracts)
            refs=Counter('R02' if r.startswith('R02:') else 'R01' if r.startswith('R01:') else 'Parent' for c in candidates for r in c.get('evidence_refs',[]))
            diagnostics.append({'arm':arm,'root_id':root,'candidate_count':len(candidates),'type_counts':dict(Counter(c['question_type'] for c in candidates)),'mode_counts':dict(Counter(c['generation_source'] for c in candidates)),'r02_evidence_candidate_count':r02,'valid_contract_count':len(contracts),'r02_referencing_contract_count':intent_r02,'evidence_ref_distribution':dict(refs),'duplicate_stem_excess':sum(n-1 for n in stems.values()),'duplicate_stem_groups':[{'signature':s,'count':n} for s,n in stems.items() if n>1]})
            corpus[(arm,root)]=candidates
    assert len(mapping)<=72
    write_rows(run/'diagnostics/blind_review/arm_mapping_do_not_show_reviewers.jsonl',mapping)
    write_rows(run/'diagnostics/blind_review/selection_groups_do_not_show_reviewers.jsonl',selection_groups)
    write(run/'diagnostics/generation_content_diagnostics.json',{'rows':diagnostics,'selection_count':len(mapping),'unique_blind_cases':len({r['blind_id'] for r in mapping}),'admission_changed':False,'rule_source':source(run/'audit/statistics_and_review_policy.json'),'stem_code':source(METHOD/'h4_round1_question_bank_lifecycle_weight_freeze_v1/run.py')})
    print(json.dumps({'review_selections':len(mapping),'unique_blind_cases':len({r['blind_id'] for r in mapping})}))

def group_metrics(values, expected_roots=ROOTS, expected_types=('single_choice','multi_select')):
    """Null for a missing candidate/root/type; request and candidate means differ."""
    correct=sum(x['correct'] for x in values); scored=sum(x['scored'] for x in values)
    root_means={r:mean([x['accuracy'] for x in values if x['root_id']==r]) for r in expected_roots}
    type_means={t:mean([x['accuracy'] for x in values if x['question_type']==t]) for t in expected_types}
    return {'candidate_count':len(values),'correct':correct,'scored_requests':scored,'total_requests':sum(x['requests'] for x in values),
        'unscored_requests':sum(x['requests']-x['scored'] for x in values),'request_weighted_accuracy':correct/scored if scored else None,
        'candidate_mean_accuracy':mean([x['accuracy'] for x in values]),'root_macro_candidate_accuracy':mean(list(root_means.values())),
        'question_type_equal_macro_candidate_accuracy':mean(list(type_means.values())),'root_candidate_means':root_means,'type_candidate_means':type_means,
        'missing_roots':[r for r,v in root_means.items() if v is None]}

def compare(run):
    observed=[]; candidate_deltas=[]; request_audits=[]; costs=[]; fingerprints=[]
    for arm in ARMS:
        sr=run/arm/'solver'; pre=load(sr/'preflight_report.json'); assert pre['status']=='PASS'
        candidates=rows(sr/'candidate_manifest.jsonl'); by_id={x['candidate_id']:x for x in candidates}; assert len(by_id)==len(candidates)
        expected_requests=sum(candidate_request_count(c) for c in candidates); assert expected_requests==pre['request_count_per_model']
        assert all(c['root_id'] in ROOTS for c in candidates)
        candidate_pool_sha=source(sr/'candidate_manifest.jsonl')['sha256']
        model_request_sets={}
        for model in MODELS:
            md=sr/model; results=rows(md/'candidate_results.jsonl'); report=load(md/'report.json')
            assert report['status']=='PASS' and {x['candidate_id'] for x in results}==set(by_id) and len(results)==len(candidates)
            seen=set(); prompt_tokens=completion_tokens=0; unscored=0
            for result in results:
                cid=result['candidate_id']; c=by_id[cid]; req=result['requests']; expected=candidate_request_count(c)
                assert audit_candidate_requests(c,result)==expected
                assert result['candidate_pool_sha256']==candidate_pool_sha
                assert result['dialogue_sha256']==source(sr/'dialogues'/(c['root_id']+'.json'))['sha256']
                assert result['solver_contract_sha256']==source(md/'solver_contract.json')['sha256']
                summary=summarize_candidate_result(result); contexts={}
                for r in req:
                    assert r['candidate_id']==cid and r['root_id']==c['root_id'] and r['canonical_gold']==c['gold']
                    key=(cid,r['request_id'],r['context_mode'],r['variant_id'],r['slot_id'],r['seed']); assert key not in seen; seen.add(key)
                    meta=r.get('backend_metadata') or {}; prompt_tokens+=int(meta.get('prompt_token_count') or 0); completion_tokens+=int(meta.get('completion_token_count') or 0)
                for context in CONTEXTS:
                    rs=[r for r in req if r['context_mode']==context]; scored=[r for r in rs if r.get('scored') is True]
                    assert len(rs)==expected//2
                    correct=sum(r.get('correct') is True for r in scored); n=len(scored); acc=correct/n if n else None
                    if summary['technical_exclusion_reason'] is None:
                        assert correct==summary[context+'_correct_count'] and n==summary[context+'_total_count'] and acc==summary[context+'_accuracy']
                    record={'arm':arm,'model':model,'candidate_id':cid,'root_id':c['root_id'],'question_type':c['question_type'],'generation_source':c['generation_source'],'context_mode':context,'correct':correct,'scored':n,'requests':len(rs),'accuracy':acc,'technical_failure_count':sum(bool(r.get('failure_type')) and r.get('failure_type')!='reasoning_error' for r in rs),'context_helpful':summary['context_helpful'],'technical_complete':summary['technical_exclusion_reason'] is None}
                    observed.append(record); contexts[context]=record; unscored+=len(rs)-n
                full,no=contexts['full_context']['accuracy'],contexts['no_context']['accuracy']
                candidate_deltas.append({'arm':arm,'model':model,'root_id':c['root_id'],'candidate_id':cid,'question_type':c['question_type'],'generation_source':c['generation_source'],'full_minus_no':full-no if full is not None and no is not None else None,'context_helpful':summary['context_helpful'],'technical_complete':summary['technical_exclusion_reason'] is None})
                if model=='M0': fingerprints.append({'arm':arm,'root_id':c['root_id'],'candidate_id':cid,'stem':stem_signature(c),'solver_prompt_fingerprint':gate_fingerprint_from_candidate_result(result)})
            assert len(seen)==expected_requests
            assert report['ledger']['completed_request_count']==expected_requests and report['ledger']['ambiguous_request_count']==0
            model_request_sets[model]=seen
            request_audits.append({'arm':arm,'model':model,'candidate_count':len(results),'expected_requests':expected_requests,'actual_unique_requests':len(seen),'unscored_requests':unscored,'ledger':report['ledger'],'results_source':source(md/'candidate_results.jsonl')})
            costs.append({'arm':arm,'model':model,'local_solver_prompt_tokens':prompt_tokens,'local_solver_completion_tokens':completion_tokens,'request_count':len(seen),'monetary_cost':None,'monetary_cost_reason':'local inference; no provider billing price'})
        assert all(model_request_sets[m]==model_request_sets['M0'] for m in MODELS)
    write_rows(run/'reports/candidate_context_metrics.jsonl',observed)
    write_rows(run/'reports/candidate_full_minus_no.jsonl',candidate_deltas)
    write(run/'audit/solver_denominator_audit.json',{'status':'PASS','rows':request_audits,'total_requests':sum(r['actual_unique_requests'] for r in request_audits),'all_models_candidate_request_schedule_equal':True})
    scopes={'main':('single_choice','multi_select'),'all_diagnostic':TYPES,**{t:(t,) for t in TYPES}}
    groups=[]; gaps=[]
    for arm,model,root,mode,scope in itertools.product(ARMS,MODELS,('all',)+ROOTS,('all',)+MODES,scopes):
        common=lambda r:r['arm']==arm and r['model']==model and (root=='all' or r['root_id']==root) and (mode=='all' or r['generation_source']==mode) and r['question_type'] in scopes[scope]
        expected_roots=ROOTS if root=='all' else (root,)
        for context in CONTEXTS:
            group=[r for r in observed if common(r) and r['context_mode']==context]
            groups.append({'arm':arm,'model':model,'root_id':root,'generation_source':mode,'scope':scope,'context_mode':context,**group_metrics(group,expected_roots,scopes[scope])})
        ds=[r for r in candidate_deltas if common(r)]
        root_gaps={rid:mean([r['full_minus_no'] for r in ds if r['root_id']==rid]) for rid in expected_roots}
        root_help={rid:mean([float(r['context_helpful']) for r in ds if r['root_id']==rid]) for rid in expected_roots}
        gaps.append({'arm':arm,'model':model,'root_id':root,'generation_source':mode,'scope':scope,'candidate_count':len(ds),'candidate_mean_full_minus_no':mean([r['full_minus_no'] for r in ds]),'root_macro_full_minus_no':mean(list(root_gaps.values())),'context_helpful_count':sum(r['context_helpful'] for r in ds),'context_helpful_denominator':len(ds),'context_helpful_rate':mean([float(r['context_helpful']) for r in ds]),'root_macro_context_helpful_rate':mean(list(root_help.values())),'technical_incomplete_candidate_count':sum(not r['technical_complete'] for r in ds)})
    write_rows(run/'reports/group_metrics.jsonl',groups); write_rows(run/'reports/context_dependency_metrics.jsonl',gaps)
    fields=[k for k in groups[0] if k not in ('root_candidate_means','type_candidate_means','missing_roots')]
    buf=io.StringIO(); writer=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(groups);frozen_bytes(run/'reports/group_metrics.csv',buf.getvalue().encode())
    def get(arm,model,root='all',scope='main',context='full_context',mode='all'):
        return next(g for g in groups if g['arm']==arm and g['model']==model and g['root_id']==root and g['scope']==scope and g['context_mode']==context and g['generation_source']==mode)
    paired=[]
    for model,root,scope,context,mode in itertools.product(MODELS,ROOTS,scopes,CONTEXTS,('all',)+MODES):
        off,on=get('feedback_off',model,root,scope,context,mode),get('feedback_on',model,root,scope,context,mode)
        a,b=off['candidate_mean_accuracy'],on['candidate_mean_accuracy']
        paired.append({'model':model,'root_id':root,'scope':scope,'context_mode':context,'generation_source':mode,'off':a,'on':b,'on_minus_off':b-a if a is not None and b is not None else None})
    write_rows(run/'reports/root_paired_differences.jsonl',paired)
    off=get('feedback_off','M0')['root_macro_candidate_accuracy']; on=get('feedback_on','M0')['root_macro_candidate_accuracy']
    duplicate_groups=[]
    for key in ('stem','solver_prompt_fingerprint'):
        grouped=defaultdict(list)
        for r in fingerprints:grouped[(r['arm'],r['root_id'],r[key])].append(r['candidate_id'])
        duplicate_groups.extend({'kind':key,'arm':a,'root_id':r,'signature':sig,'candidate_ids':ids} for (a,r,sig),ids in grouped.items() if len(ids)>1)
    write(run/'diagnostics/duplicate_diagnostics.json',{'groups':duplicate_groups,'code_source':source(METHOD/'h4_round1_question_bank_lifecycle_weight_freeze_v1/run.py'),'admission_changed':False,'fingerprint_includes_original_solver_contract_identity':True})
    write(run/'reports/solver_token_costs.json',costs)
    summary={'status':'COMPLETE','primary':{'definition':'Base full-context, SC+MS, candidate accuracy averaged within root then equal-weight six roots','off':off,'on':on,'on_minus_off':on-off if on is not None and off is not None else None,'root_count':6},'primary_root_pairs':[r for r in paired if r['model']=='M0' and r['scope']=='main' and r['context_mode']=='full_context'],'all_models_main':[{k:g[k] for k in g if k!='type_candidate_means'} for g in groups if g['root_id']=='all' and g['generation_source']=='all' and g['scope']=='main'],'request_audit':source(run/'audit/solver_denominator_audit.json'),'group_metrics':source(run/'reports/group_metrics.csv'),'limitations':['six exploratory paired roots and one generation per arm','same fixed R02 transcripts previously exposed to old feedback','treatment includes current appendix instructions, not isolated Solver numbers','accuracy against unmodified generated Gold; lower accuracy alone does not prove useful difficulty','ordering only raw diagnosis; corrected R02 refers to expansion route, not corrected QA Gold'],'confidence_interval':None,'confidence_interval_reason':'not planned; request positions/rollouts are not independent questions'}
    summary['primary_root_pairs']=[r for r in summary['primary_root_pairs'] if r['generation_source']=='all']
    summary['primary']['missing_roots']={arm:get(arm,'M0')['missing_roots'] for arm in ARMS}
    summary['primary']['available_root_count']={arm:6-len(get(arm,'M0')['missing_roots']) for arm in ARMS}
    summary['primary']['metric_complete']=off is not None and on is not None
    from . import finalize
    summary=finalize.enrich_report(run,summary)
    write(run/'reports/p1_comparison.json',summary)
    lines=['# P1 feedback on/off 比较','',f"主指标（Base full、SC+MS、六 root 等权）：off={off}, on={on}, on−off={summary['primary']['on_minus_off']}。",'', '该差值只描述当前原标签下的答对难度；效果解释须结合独立内容核查。','', '| 模型 | 臂 | context | correct/scored | request 加权 | candidate 均值 | root-macro |', '|---|---|---|---:|---:|---:|---:|']
    for g in summary['all_models_main']:
        lines.append(f"| {g['model']} | {g['arm']} | {g['context_mode']} | {g['correct']}/{g['scored_requests']} | {formatted_rate(g['request_weighted_accuracy'])} | {formatted_rate(g['candidate_mean_accuracy'])} | {formatted_rate(g['root_macro_candidate_accuracy'])} |")
    lines+=['','完整 root、题型和 Guided/Free 表：`group_metrics.csv`；候选分母：`candidate_context_metrics.jsonl`；配对差：`root_paired_differences.jsonl`。','', '排序分保存在 all_diagnostic/ordering 分组，不进入主结论。六 root、一次生成仅提供探索性证据，不证明训练收益或所有反馈方案的效果。']
    lines.append(finalize.render_report(summary))
    frozen_bytes(run/'reports/p1_comparison.md',('\n'.join(lines)+'\n').encode())
    print(json.dumps(summary['primary']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);p.add_argument('--phase',choices=['review','compare'],required=True);a=p.parse_args();{'review':review,'compare':compare}[a.phase](a.run)
