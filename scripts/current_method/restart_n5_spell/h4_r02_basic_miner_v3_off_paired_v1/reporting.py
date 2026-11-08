"""Six-dimensional attributed review and fixed paired denominators, offline only."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from ..core import atomic_json, atomic_text, sha256_file
from ..h4_r02_miner_three_arm_diagnostic_v1.reporting import collect_group as collect_original_group
from .constants import VERSION, ROOTS, ARMS, SCHEDULE, WORK_ORDER, root_path
from .adapter import audit_response_schema, output_schema
from jsonschema import Draft202012Validator
from .runtime import read, now, audit_reservations, verify_checkpoint

REVIEWER='Codex offline diagnostic / direct transcript review'
DIMENSIONS=('DISTINCTION_SPECIFICITY','EVIDENCE_SUPPORT','STATE_CORRECTNESS','TEMPORAL_CORRECTNESS','CAUSAL_CORRECTNESS','EVIDENCE_GROUPING')
REQUIRED={'DISTINCTION_SPECIFICITY','EVIDENCE_SUPPORT','EVIDENCE_GROUPING'}
NOT_EVALUATED='NOT_EVALUATED_STRUCTURAL_OR_MISSING'
PASS='INTENT_EVIDENCE_PASS'
FAIL='INTENT_EVIDENCE_FAIL'
UNDETERMINED='INTENT_EVIDENCE_UNDETERMINED'
TAGS={'VAGUE_DISTINCTION','IRRELEVANT_EVIDENCE','CROSS_EVENT_MIXING','STATE_CONFUSION','TEMPORAL_CONFUSION','CAUSAL_OVERREACH','UNSUPPORTED_RELATION','OTHER'}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).is_file() else []


def write_jsonl(path,rows):
    atomic_text(Path(path),''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows))


def review_policy():
    order=WORK_ORDER.read_text()
    return {'version':VERSION+'_six_dimension_review_v1','reviewer':REVIEWER,
        'planned_total':12,'planned_per_arm':6,'planned_per_group':3,
        'dimensions':list(DIMENSIONS),'required_dimensions':sorted(REQUIRED),
        'judgments':['PASS','FAIL','UNDETERMINED','NOT_APPLICABLE'],
        'NA_only_for':['STATE_CORRECTNESS','TEMPORAL_CORRECTNESS','CAUSAL_CORRECTNESS'],
        'aggregation':'Any applicable FAIL -> INTENT_EVIDENCE_FAIL; otherwise any UNDETERMINED -> INTENT_EVIDENCE_UNDETERMINED; all required and applicable PASS -> INTENT_EVIDENCE_PASS. Structural/missing is separate, never NA-derived PASS.',
        'slots':'First three raw array positions; extras never replace slots; canonicalizer defaults unchanged; full response JSON Schema separately audited.',
        'metric_policy':{'scene':'frozen scene_ref, else session_ref, else explicit turn_ref prefix before :T; unmappable null',
            'event':'UNKNOWN_NO_FROZEN_EVENT_ANNOTATION; always null','span':'1-based min/max/span of exact valid-chain refs; unknown null','no_threshold':True},
        'statistical_unit':'root; two roots equal weight; no same-ID pairing or significance test',
        'failure_tag_count_unit':'distinct intent raw position per tag; dimensions counted separately; extras excluded from planned denominator',
        'acceptance':'Engineering all requirements PASS; V3 instance PASS only if both full responses match frozen schema and six evaluable slots all content PASS; any content FAIL -> FAIL; remaining missing/structural/uncertain -> INCONCLUSIVE. READY only engineering+instance PASS. Never execute more roles.',
        'exact_work_order_review_and_statistics':order.split('## 8. 内容诊断：',1)[1].split('## 10.',1)[0],
        'no_production_semantic_gate':True,'extra_review_api_calls':0}


def key(row):
    return row['root_id'],row['arm_id'],row.get('raw_index',row.get('intent_index'))


def call_accounting(run):
    run=Path(run);events=jsonl(run/'audit/transport_events.jsonl');logical=jsonl(run/'audit/call_events.jsonl')
    known=set();groups=[]
    for row in SCHEDULE:
        root=root_path(run,row['arm_id'],row['root_id']);paths=list((root/'ledger/requests').glob('*.json'));known.update(paths)
        states=[read(p) for p in paths];complete=[s for s in states if s.get('status')=='completed']
        transport=root/'ledger/transport_outcomes'/f"{row['request_id']}.json"
        outcome=read(transport) if transport.is_file() else {}
        own=[e for e in events if e.get('arm_id',e.get('arm'))==row['arm_id'] and e.get('root_id')==row['root_id']]
        own_log=[e for e in logical if e.get('global_request_id')==row['global_request_id']]
        usage=Counter()
        for s in complete:
            usage.update({k:v for k,v in (s.get('response_metadata',{}).get('usage') or {}).items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
        attempts=sum(e.get('event')=='http_attempt_started' for e in own)
        known_fail=int(bool(states) and outcome.get('service_completion_status')=='KNOWN_HTTP_REJECTION')
        groups.append({**row,'logical_calls':len(states),'completed_calls':len(complete),'known_failed_calls':known_fail,
            'unknown_calls':len(states)-len(complete)-known_fail,'http_attempts':attempts,'http_retries':max(0,attempts-1),
            'service_completion_status':outcome.get('service_completion_status','NOT_RUN'),
            'finish_reasons':dict(Counter(str(s.get('response_metadata',{}).get('finish_reason')) for s in complete)),
            'usage':dict(usage),'usage_missing':sum(s.get('response_metadata',{}).get('usage') is None for s in complete),
            'cache_hits':sum(e.get('event')=='completed_cache_reused_no_send' for e in own_log)})
    all_paths=set(run.glob('**/ledger/requests/*.json'))
    usage=Counter()
    for g in groups:usage.update(g['usage'])
    return {'planned_logical_calls':4,'maximum_logical_calls':4,'groups':groups,
        **{k:sum(g[k] for g in groups) for k in ('logical_calls','completed_calls','known_failed_calls','unknown_calls','http_attempts','http_retries','usage_missing','cache_hits')},
        'usage':dict(usage),'unexpected_ledger_paths':sorted(map(str,all_paths-known)),
        'monetary_cost':None,'monetary_cost_status':'UNKNOWN_NO_BILLING_OR_PRICING_EVIDENCE',
        'non_miner_calls':len(all_paths-known),'zero_roles':['Writer','repair','Solver','Planner','Realizer','Memory','training','independent_review_API']}


def mechanical_report(run):
    run=Path(run);errors=[]
    try:
        from .run import verify_frozen
        verify_frozen(run);audit_reservations(run)
    except BaseException as error:
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
        errors.append('frozen_or_runtime_guard:'+str(error))
    groups=[];items=[];pairing={(r['arm_id'],r['root_id']):r for r in jsonl(run/'pairing_manifest.jsonl')}
    for row in SCHEDULE:
        group,original=collect_original_group(run,row['arm_id'],row['root_id'])
        root=root_path(run,row['arm_id'],row['root_id'])
        schema_path=root/'response_schema_audit.json'
        schema=read(schema_path) if schema_path.is_file() else {'schema_valid':False,'errors':['MISSING_AUDIT']}
        group.update(row);group['full_response_schema_valid']=schema['schema_valid'];group['full_schema_audit']=schema
        prompt=root/'raw/prompts'/f"{row['request_id']}.txt"
        group['checks']['prompt_frozen_identity']=prompt.is_file() and sha256_file(prompt)==pairing.get((row['arm_id'],row['root_id']),{}).get('prompt_sha256')
        try:group['checks']['checkpoint']=verify_checkpoint(root)
        except BaseException as error:group['checks']['checkpoint']=False;errors.append(str(error))
        if schema_path.is_file():
            from ..simple_append_v1.live_runtime import load_json_object_strict
            try:raw=load_json_object_strict((root/'raw/responses'/f"{row['request_id']}.json.txt").read_text())
            except ValueError:raw=None
            expected=audit_response_schema(raw)
            group['checks']['schema_audit_reproduction']=all(schema.get(k)==v for k,v in expected.items())
        item_schema=Draft202012Validator(output_schema()['properties']['intents']['items'])
        for item in original:
            metrics=item['metrics'];span=metrics['evidence_span']
            item.update(raw_index=item['intent_index'],full_item_schema_valid=item_schema.is_valid(item['raw_intent']) if item['returned'] else False,declared_intent_id=item['raw_intent'].get('intent_id') if isinstance(item['raw_intent'],dict) else None,
                turn_position_min=span['min_position'],turn_position_max=span['max_position'],turn_position_span=span['width'],
                event_annotation_status='UNKNOWN_NO_FROZEN_EVENT_ANNOTATION')
            items.append(item)
        errors.extend(row['global_request_id']+':'+k for k,v in group['checks'].items() if v is False)
        group['full_item_schema_valid_count']=sum(i['full_item_schema_valid'] for i in original if i['planned_slot'])
        groups.append(group)
    accounting=call_accounting(run)
    if accounting['logical_calls']!=4 or any(g['logical_calls']!=1 for g in accounting['groups']):errors.append('four_planned_calls_not_complete')
    if accounting['unknown_calls'] or accounting['unexpected_ledger_paths']:errors.append('unknown_or_out_of_scope_requests')
    forbidden=[str(p) for p in (run/'arms').rglob('*') if p.is_file() and
        (p.name in ('candidate_pool.json','candidate_generation.json','feedback_bindings.json','binding_validation.json','guided_brief_preview.json') or '-generator-' in p.name or '-repair-' in p.name)]
    if forbidden:errors.append('forbidden_non_miner_artifacts')
    report={'status':'PASS' if not errors else 'FAIL','groups':groups,'items':items,'accounting':accounting,'errors':errors,'forbidden_files':forbidden}
    atomic_json(run/'audit/mechanical_report.json',report);atomic_json(run/'audit/call_accounting.json',accounting)
    atomic_json(run/'reports/structural_and_extra_outputs.json',{'groups':groups,'extras':[i for i in items if not i['planned_slot']]})
    write_jsonl(run/'reports/evidence_width_metrics.jsonl',[{**{k:i[k] for k in ('root_id','arm_id','raw_index','intent_id','planned_slot','turn_position_min','turn_position_max','turn_position_span','event_annotation_status')},**i['metrics']} for i in items])
    return report


def prepare_review_views(run):
    report=mechanical_report(run);views=[];mapping=[]
    for n,item in enumerate(report['items'],1):
        vid=f'V{n:03d}'
        views.append({'view_id':vid,'raw_intent':item['raw_intent'],'original_intent':item['original_intent'],
            'derived_evidence':item['derived_evidence'],'structurally_valid':item['structurally_valid'],
            'complete_transcript':read(Path(item['frozen_transcript_path']))})
        mapping.append({'view_id':vid,**{k:item[k] for k in ('root_id','arm_id','raw_index','intent_id','planned_slot')}})
    write_jsonl(Path(run)/'reports/intent_review_views.jsonl',views)
    write_jsonl(Path(run)/'audit/intent_review_view_mapping.jsonl',mapping)
    return report


def aggregate_dimensions(dimensions):
    if set(dimensions)!=set(DIMENSIONS):raise ValueError('six_dimensions_required')
    judgments=[]
    for name,dim in dimensions.items():
        judgment=dim.get('judgment')
        if judgment not in ('PASS','FAIL','UNDETERMINED','NOT_APPLICABLE'):raise ValueError('invalid_dimension_judgment')
        if name in REQUIRED and judgment=='NOT_APPLICABLE':raise ValueError('required_dimension_cannot_be_NA')
        judgments.append(judgment)
    if 'FAIL' in judgments:return FAIL
    if 'UNDETERMINED' in judgments:return UNDETERMINED
    return PASS


def assemble_reviews(items,authored):
    expected={key(i):i for i in items};provided={};errors=[];result=[]
    for r in authored:
        k=key(r)
        if k in provided:errors.append('duplicate_review:'+str(k))
        if k not in expected:errors.append('unknown_raw_slot:'+str(k))
        provided[k]=r
    for k,item in expected.items():
        original=item['original_intent'] or {}
        common={**{f:item[f] for f in ('root_id','arm_id','raw_index','intent_id','declared_intent_id','planned_slot','structurally_valid')},
            'discriminative_variable':original.get('discriminative_variable'), 'intent_type':original.get('intent_type'),
            'reasoning_chain':original.get('reasoning_chain'), 'original_intent_errors':item['intent_errors'],
            'reviewer':REVIEWER}
        row=provided.get(k)
        if not item['structurally_valid']:
            if row and row.get('judgment')!=NOT_EVALUATED:errors.append('semantic_review_on_unusable_slot:'+str(k))
            result.append({**common,'judgment':NOT_EVALUATED,'dimensions':{},'tags':[],
                'reason':'Original canonicalizer/contract/evidence unavailable or invalid; not semantically assessed.'});continue
        if row is None:errors.append('missing_review:'+str(k));continue
        try:judgment=aggregate_dimensions(row.get('dimensions',{}))
        except ValueError as error:errors.append(str(error)+':'+str(k));continue
        if row.get('reviewer')!=REVIEWER or not row.get('reason'):errors.append('review_attribution_or_reason:'+str(k))
        if set(row.get('tags',[]))-TAGS:errors.append('unknown_failure_tag:'+str(k))
        if row.get('judgment',judgment)!=judgment:errors.append('aggregate_judgment_conflict:'+str(k))
        turns={t['turn_ref']:t for t in read(Path(item['frozen_transcript_path']))['turns']}
        selected=set(item['current_refs']);dimensions=deepcopy(row['dimensions'])
        for name,dim in dimensions.items():
            if not dim.get('reason'):errors.append('dimension_reason_missing:'+str(k)+name)
            quotes=dim.get('quotes')
            if not isinstance(quotes,list) or (dim['judgment']!='NOT_APPLICABLE' and not quotes):
                errors.append('dimension_quotes_missing:'+str(k)+name);continue
            for quote in quotes:
                ref=quote.get('turn_ref');text=quote.get('quote')
                if ref not in turns or not isinstance(text,str) or not text or text not in turns[ref]['text']:
                    errors.append('nonexact_transcript_quote:'+str(k)+name)
                quote['in_selected_chain']=ref in selected
                if ref not in selected and quote.get('usage')!='OUTSIDE_CHAIN_CONTEXT_ONLY':
                    errors.append('outside_chain_quote_not_labeled:'+str(k)+name)
        result.append({**row,**common,'dimensions':dimensions,'judgment':judgment,'tags':sorted(set(row.get('tags',[])))})
    return result,errors


def summarize(items,reviews,groups):
    bykey={key(r):r for r in reviews}
    def counts(selected,denominator):
        planned=[i for i in selected if i['planned_slot']];review=[bykey[key(i)] for i in planned if key(i) in bykey]
        c=Counter(r['judgment'] for r in review)
        return {'planned_slots':denominator,'returned':sum(i['returned'] for i in planned),
            'returned_total':sum(i['returned'] for i in selected),'original_valid_intents':sum(i['structurally_valid'] for i in planned),
            'content_pass':c[PASS],'content_fail':c[FAIL],'content_undetermined':c[UNDETERMINED],
            'not_evaluated':c[NOT_EVALUATED],'missing':sum(not i['returned'] for i in planned),'review_missing':denominator-len(review),
            'extras':sum(not i['planned_slot'] for i in selected),'pass_rate':c[PASS]/denominator,
            'reviewable_pass_fraction':c[PASS]/sum(i['structurally_valid'] for i in planned) if any(i['structurally_valid'] for i in planned) else None,
            'dimension_counts':{d:dict(Counter(r.get('dimensions',{}).get(d,{}).get('judgment','STRUCTURAL_NOT_EVALUATED') for r in review)) for d in DIMENSIONS},
            'failure_tag_counts':dict(Counter(t for r in review for t in set(r.get('tags',[])))),
            'failure_tag_count_unit':'distinct planned raw slot per tag',
            'extra_reviews':[bykey.get(key(i),{'review_missing':True}) for i in selected if not i['planned_slot']]}
    out=[]
    for row in SCHEDULE:
        selected=[i for i in items if i['root_id']==row['root_id'] and i['arm_id']==row['arm_id']]
        group=next(g for g in groups if g['root_id']==row['root_id'] and g['arm_id']==row['arm_id'])
        out.append({**row,**counts(selected,3),'full_response_schema_valid':group['full_response_schema_valid'],
                    'full_item_schema_valid_count':group['full_item_schema_valid_count']})
    arms=[{'arm_id':a,**counts([i for i in items if i['arm_id']==a],6)} for a in ARMS]
    paired=[]
    for rid in ROOTS:
        base=next(g for g in out if g['root_id']==rid and g['arm_id']=='BASE_OFF')
        v3=next(g for g in out if g['root_id']==rid and g['arm_id']=='V3_OFF')
        paired.append({'root_id':rid,'BASE_OFF_pass':base['content_pass'],'V3_OFF_pass':v3['content_pass'],
            'BASE_OFF_pass_rate':base['pass_rate'],'V3_OFF_pass_rate':v3['pass_rate'],'delta':v3['pass_rate']-base['pass_rate'],
            'dimension_counts':{'BASE_OFF':base['dimension_counts'],'V3_OFF':v3['dimension_counts']}})
    v3=next(a for a in arms if a['arm_id']=='V3_OFF')
    acceptance=('FAIL' if v3['content_fail'] else 'PASS' if v3['content_pass']==6 and
                all(g['full_response_schema_valid'] for g in out if g['arm_id']=='V3_OFF') else 'INCONCLUSIVE')
    return {'groups':out,'arms':arms,'root_paired':paired,'equal_weight_mean_delta':sum(p['delta'] for p in paired)/2,
            'V3_INSTANCE_ACCEPTANCE':acceptance,'paired_unit':'root; not equal intent ID', 'production_admission_changed':False}


def preservation(run):
    histories=[]
    for label in ('P1_RUN','V1_RUN','V2_RUN','THREE_RUN'):
        baseline=read(run/f'audit/{label}_inventory_before.json');folder=Path(baseline['root'])
        current={str(p.relative_to(folder)):{'sha256':sha256_file(p),'size':p.stat().st_size} for p in sorted(folder.rglob('*')) if p.is_file()}
        old=baseline['files'];histories.append({'history':label,'root':str(folder),'before_files':len(old),'after_files':len(current),
            'status':'PASS' if current==old else 'FAIL','changed_or_missing':[k for k in old if current.get(k)!=old[k]],'added':sorted(set(current)-set(old))})
    result={'status':'PASS' if all(h['status']=='PASS' for h in histories) else 'FAIL','histories':histories,'at_utc':now()}
    atomic_json(run/'audit/historical_preservation_report.json',result);return result


def final_report(run):
    run=Path(run);mechanical=mechanical_report(run);history=preservation(run)
    reviews,errors=assemble_reviews(mechanical['items'],jsonl(run/'reports/intent_evidence_review.jsonl'))
    if not errors:write_jsonl(run/'reports/intent_evidence_review.jsonl',reviews)
    comparison=summarize(mechanical['items'],reviews,mechanical['groups'])
    atomic_json(run/'reports/paired_comparison.json',comparison)
    checks={name:read(run/f'audit/{name}.json')['status']=='PASS' for name in ('test_report','preflight_report','prompt_pairing_report','dependency_coverage_report')}
    checks.update(frozen_and_actual_runtime=mechanical['status']=='PASS',historical_preservation=history['status']=='PASS')
    engineering='PASS' if all(checks.values()) else 'FAIL'
    result={'status':'COMPLETED' if not errors else 'BLOCKED','version':VERSION,'ENGINEERING_COMPARABILITY':engineering,
        'V3_INSTANCE_ACCEPTANCE':comparison['V3_INSTANCE_ACCEPTANCE'],
        'READY_FOR_THREE_ARM_REVIEW':'YES' if engineering=='PASS' and comparison['V3_INSTANCE_ACCEPTANCE']=='PASS' and not errors else 'NOT_READY',
        'WRITER_STATUS':'NOT_RUN_OUT_OF_SCOPE','checks':checks,'review_errors':errors,'mechanical_errors':mechanical['errors'],
        'comparison':comparison,'accounting':mechanical['accounting'],'history_preservation':history,'at_utc':now()}
    atomic_json(run/'audit/final_contract_audit.json',result)
    lines=['# 基础 Intent Miner v3：四次 OFF 配对诊断','',
        f"ENGINEERING_COMPARABILITY={engineering}; V3_INSTANCE_ACCEPTANCE={result['V3_INSTANCE_ACCEPTANCE']}; READY_FOR_THREE_ARM_REVIEW={result['READY_FOR_THREE_ARM_REVIEW']}; WRITER_STATUS=NOT_RUN_OUT_OF_SCOPE。",'',
        '| root | arm | 原有效/3 | 完整 schema | PASS/FAIL/UNDETERMINED/未评 | extras |','|---|---|---:|---|---|---:|']
    for g in comparison['groups']:
        lines.append(f"| {g['root_id']} | {g['arm_id']} | {g['original_valid_intents']}/3 | {g['full_response_schema_valid']} | {g['content_pass']}/{g['content_fail']}/{g['content_undetermined']}/{g['not_evaluated']} | {g['extras']} |")
    lines+=['','每组计划分母 3、每臂 6、总计 12。原 canonicalizer 准入与完整冻结 JSON Schema 分别审计；内容维度仅离线诊断，不修改原 Guided 合同或生产 QA 准入。','']
    narrative=run/'reports/diagnostic_narrative.md'
    if narrative.is_file():lines+=[narrative.read_text(),'']
    else:errors.append('authored_narrative_missing')
    lines+=['配对计数：','```json',json.dumps(comparison['root_paired'],ensure_ascii=False,indent=2),'```',
        '两 root 等权平均 delta='+str(comparison['equal_weight_mean_delta']),
        '逐槽六维结论、原始变量/组 refs、精确短引用见 intent_evidence_review.jsonl；宽度见 evidence_width_metrics.jsonl。事件数均 null，UNKNOWN_NO_FROZEN_EVENT_ANNOTATION；不设宽度阈值。',
        '账务：','```json',json.dumps(mechanical['accounting'],ensure_ascii=False,indent=2),'```',
        f"历史保全：{history['status']}。工程错误：{mechanical['errors']}；审阅错误：{errors}。",
        '仅两个已知 root、每格一次生成，不作显著性、泛化、稳定因果、Solver accuracy 或训练收益结论。prompt 要求构造顺序不等于可由 JSON 证明内部推理顺序改变。',
        'Writer、repair、Solver、Planner、Realizer、Memory、训练、独立审核 API 均为零。四格诊断到此停止；不自动恢复三臂或新增样本。']
    if errors:result['status']='BLOCKED';result['review_errors']=errors;result['READY_FOR_THREE_ARM_REVIEW']='NOT_READY';atomic_json(run/'audit/final_contract_audit.json',result)
    atomic_text(run/'reports/miner_v3_off_paired_diagnostic.md','\n'.join(lines)+'\n')
    atomic_text(run/'RUN_STATUS.md',f"Report: {result['status']}\nENGINEERING_COMPARABILITY: {engineering}\nV3_INSTANCE_ACCEPTANCE: {result['V3_INSTANCE_ACCEPTANCE']}\nREADY_FOR_THREE_ARM_REVIEW: {result['READY_FOR_THREE_ARM_REVIEW']}\nWRITER_STATUS: NOT_RUN_OUT_OF_SCOPE\nSTOPPED. All non-Miner model roles: 0.\n")
    artifacts=['execution_contract.json','implementation_manifest.json','review_policy.json','schedule.json','pairing_manifest.jsonl','input_path_audit.json','audit/test_report.json','audit/preflight_report.json','audit/prompt_pairing_report.json','audit/dependency_coverage_report.json','audit/frozen_manifest.json','audit/call_accounting.json','audit/token_report.json','audit/transport_events.jsonl','audit/historical_preservation_report.json','audit/final_contract_audit.json','reports/intent_evidence_review.jsonl','reports/structural_and_extra_outputs.json','reports/evidence_width_metrics.jsonl','reports/paired_comparison.json','reports/miner_v3_off_paired_diagnostic.md','RUN_STATUS.md']
    atomic_text(run/'ARTIFACT_INDEX.md','# Artifacts\n\n'+''.join(f'- [{p}]({p})\n' for p in artifacts)+'\nFrozen inputs: inputs/shared/<root>/. Actual raw/ledger/checkpoints: arms/<arm>/qa/shards/shard-00/roots/<root>/.\n')
    return result
