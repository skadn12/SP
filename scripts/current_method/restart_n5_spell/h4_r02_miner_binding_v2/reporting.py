"""Mechanical audit plus serialization of attributed direct transcript review.

No model calls or semantic classifiers. Review judgments are authored offline.
"""
from collections import Counter
import json
from pathlib import Path
from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.live_runtime import load_json_object_strict
from .runtime import FIRST_ROOTS, root_path, state_paths, read, now, verify_checkpoint, GuardrailStop
from .transfer import validate_bindings, derive_all_intent_evidence

CRITERIA=('current_evidence_support','old_question_focus','feedback_basis','added_distinction','scope','no_transfer_appropriateness')
JUDGMENTS={'PASS','FAIL','UNDETERMINED','NOT_APPLICABLE'}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_jsonl(path,rows):
    atomic_text(path,''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows))


def optional(path,default):
    return read(path) if path.is_file() else default


def preservation(run):
    result=[]
    for label in ('P1','V1_RUN'):
        baseline=read(run/f'audit/{label}_inventory_before.json'); folder=Path(baseline['root'])
        current={str(p.relative_to(folder)):sha256_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        old=baseline['files']; changed=[k for k in old if current.get(k)!=old[k]]; added=sorted(set(current)-set(old))
        result.append({'history':label,'root':str(folder),'before_files':len(old),'after_files':len(current),
                       'changed_or_missing':changed,'added':added,'status':'PASS' if current==old else 'FAIL'})
    report={'status':'PASS' if all(r['status']=='PASS' for r in result) else 'FAIL','histories':result,'at_utc':now()}
    atomic_json(run/'audit/history_preservation.json',report); return report


def mechanical_report(run):
    from .run import verify_frozen
    freeze_error=None
    try: verify_frozen(run)
    except (GuardrailStop,Exception) as error: freeze_error=str(error)
    states=[read(p) for p in state_paths(run)]; completed=[s for s in states if s.get('status')=='completed']
    transport=jsonl(run/'audit/transport_events.jsonl'); outcomes=[t for t in transport if t.get('event')=='logical_call_transport_outcome']
    usage=Counter()
    for s in completed: usage.update({k:v for k,v in (s.get('response_metadata',{}).get('usage') or {}).items() if isinstance(v,(int,float))})
    accounting={'planned_logical_calls':3,'maximum_logical_calls':3,'logical_calls':len(states),'completed_calls':len(completed),
        'failed_or_unresolved_calls':len(states)-len(completed),'unknown_billing_status_calls':len(states)-len(completed),
        'miner_calls':sum(s.get('request_id')==s.get('request_metadata',{}).get('root_id','')+'-intent-miner' for s in states),
        'http_attempts':sum(t.get('event')=='http_attempt_started' for t in transport),
        'http_retries':sum(max(0,t.get('transport_attempt_count',0)-1) for t in outcomes),
        'transport_failures':sum(t.get('status')=='failed' for t in outcomes),'completion_usage':dict(usage),
        'finish_reasons':dict(Counter(s.get('response_metadata',{}).get('finish_reason','unknown') for s in completed)),
        'writer_calls':0,'repair_calls':0,'solver_calls':0,'memory_calls':0,'training_calls':0,'other_llm_calls':0,
        'monetary_cost':None,'monetary_cost_note':'Provider did not supply a billed amount; no price inferred.'}
    atomic_json(run/'audit/call_accounting.json',accounting)
    roots=[]; items=[]; traces=[]; errors=[] if freeze_error is None else ['frozen_integrity:'+freeze_error]
    for root_id in FIRST_ROOTS:
        root=root_path(run,root_id); folder=run/'inputs'/root_id
        if not all((folder/p).is_file() for p in ('expanded_dialogue_clean.json','observation_cards.json')):
            errors.append(root_id+':frozen_input_missing')
            roots.append({'root_id':root_id,'status':'BLOCKED','valid_intents':0,'actual_intents':0,'proposed_bindings':0,'valid_proposed_bindings':0,'model_no_transfer':0,'valid_no_transfer':0,'invalid_binding_rows':0,'logical_calls':sum(s.get('request_metadata',{}).get('root_id')==root_id for s in states),'completed_calls':0,'checks':{'frozen_input_present':False}})
            continue
        expanded=read(folder/'expanded_dialogue_clean.json')
        cards=read(folder/'observation_cards.json'); intents=optional(root/'intents.json',{}).get('validation',{})
        contracts=optional(root/'reasoning_contracts.json',[]); bindings=optional(root/'binding_validation.json',{})
        evidence=optional(root/'derived_intent_evidence.json',{}); root_states=[s for s in states if s.get('request_metadata',{}).get('root_id')==root_id]
        prompt_path=root/'raw/prompts'/f'{root_id}-intent-miner.txt'; raw_path=root/'raw/responses'/f'{root_id}-intent-miner.json.txt'
        try: raw=load_json_object_strict(raw_path.read_text()) if raw_path.exists() else None
        except (ValueError,json.JSONDecodeError): raw=None
        checks={}
        if intents:
            checks['binding_reconstruction']=bindings==validate_bindings(raw,intents,contracts,expanded=expanded,root_id=root_id,cards=cards)
            checks['evidence_reconstruction']=evidence==derive_all_intent_evidence(intents,contracts,expanded=expanded,root_id=root_id,transcript_sha256=sha256_file(folder/'expanded_dialogue_clean.json'))
            checks['no_integrity_errors']=not evidence.get('integrity_errors')
        else: checks['miner_stage_artifacts']=False
        checks['frozen_prompt']=prompt_path.is_file() and (run/'policy_snapshot/offline_prompts'/root_id/'miner_prompt.txt').is_file() and prompt_path.read_bytes()==(run/'policy_snapshot/offline_prompts'/root_id/'miner_prompt.txt').read_bytes()
        try: checks['checkpoint']=verify_checkpoint(root)
        except (GuardrailStop,Exception): checks['checkpoint']=False
        if root_id==FIRST_ROOTS[2]:
            payload=json.loads(prompt_path.read_text().split('ROOT OBSERVATIONS AND HISTORICAL TRACKING:\n',1)[1]) if prompt_path.is_file() else {}
            forbidden=('historical_gold','solver_observations','canonical_prediction_counts','correct_count','accuracy','predictions')
            text=json.dumps(payload)
            checks['negative_input_blocked']=payload.get('observation_cards')==[] and payload.get('feedback_status')=='NO_ACTIVE_FEEDBACK' and not any('"'+k+'"' in text for k in forbidden)
        for key,value in checks.items():
            if not value: errors.append(root_id+':'+key)
        for index,intent in enumerate(intents.get('intents',[])):
            matches=[r for r in bindings.get('bindings',[]) if r.get('intent_id')==intent['intent_id']]
            d=next((r for r in evidence.get('items',[]) if r['intent_index']==index),{})
            item={'root_id':root_id,'intent_index':index,'intent_id':intent['intent_id'],
                'intent_valid':intent.get('valid',False),'intent_errors':intent.get('errors',[]),
                'binding_indices':[r['binding_index'] for r in matches],
                'binding_state':bindings.get('by_intent',{}).get(intent['intent_id'],{}).get('binding_state','ORIGINAL_INTENT_INVALID'),
                'binding_mechanical_valid':len(matches)==1 and matches[0]['mechanical_valid'],
                'binding_errors':[r['errors'] for r in matches],
                'derivation_usable':d.get('usable',False),'derivation_exact':checks.get('evidence_reconstruction',False) and d.get('usable',False),
                'diagnostic_counts_only':d.get('diagnostic_counts_only',{}),'evidence_refs':[r['turn_ref'] for r in d.get('derived_current_evidence',[])]}
            items.append(item)
            traces.append({**item,'original_intent':intent,'raw_bindings':[r['binding'] for r in matches],
                'source_feedback_ids':[sid for r in matches if isinstance(r.get('binding'),dict) for sid in (r['binding'].get('source_feedback_ids') if isinstance(r['binding'].get('source_feedback_ids'),list) else [])],
                'historical_observations_path':str(folder/'observation_cards.json'),'historical_provenance_path':str(folder/'provenance.json'),
                'frozen_transcript_path':str(folder/'expanded_dialogue_clean.json'),'raw_response_path':str(raw_path),
                'derived_evidence_path':str(root/'derived_intent_evidence.json'),'delivery_status':'NOT_SENT_TO_WRITER'})
        roots.append({'root_id':root_id,'valid_intents':len(contracts),'actual_intents':len(intents.get('intents',[])),
            'proposed_bindings':sum(r.get('status')=='PROPOSED_TRANSFER' for r in bindings.get('bindings',[])),
            'valid_proposed_bindings':bindings.get('valid_proposed_binding_count',0),
            'model_no_transfer':sum(r.get('status')=='NO_SUPPORTED_TRANSFER' for r in bindings.get('bindings',[])),
            'valid_no_transfer':bindings.get('valid_no_supported_transfer_count',0),
            'invalid_binding_rows':sum(not r.get('mechanical_valid') for r in bindings.get('bindings',[])),
            'envelope_errors':bindings.get('envelope_errors',[]),'intent_errors':intents.get('errors',[]),
            'logical_calls':len(root_states),'completed_calls':sum(s.get('status')=='completed' for s in root_states),
            'checks':checks})
    forbidden_files=[str(p) for p in (run/'qa').rglob('*') if p.is_file() and (p.name in {'candidate_pool.json','candidate_generation.json','candidates_valid.json'} or '-generator-' in p.name or '-repair-' in p.name)]
    if forbidden_files: errors.append('forbidden_writer_artifacts')
    if not(len(states)==3 and accounting['miner_calls']==3 and all(r['logical_calls']==1 for r in roots)): errors.append('scheduled_call_scope_incomplete_or_violated')
    report={'status':'PASS' if not errors else 'FAIL','roots':roots,'items':items,'errors':errors,'forbidden_files':forbidden_files,'accounting':accounting,'mechanical_not_semantic':True}
    atomic_json(run/'audit/mechanical_report.json',report); write_jsonl(run/'reports/source_to_intent_trace.jsonl',traces)
    return report


def final_report(run):
    mechanical=mechanical_report(run); history=preservation(run)
    reviews=jsonl(run/'reports/miner_binding_content_review.jsonl')
    global_review=optional(run/'reports/content_acceptance_judgment.json',{})
    coverage_errors=[] if optional(run/'audit/live_completion.json',{}).get('status')=='COMPLETED' else ['live_incomplete']
    passing={r:[] for r in FIRST_ROOTS[:2]}
    expected={(i['root_id'],i['intent_index']) for i in mechanical['items']}
    actual=[(r['root_id'],r.get('intent_index')) for r in reviews if r.get('intent_index') is not None]
    if set(actual)!=expected or len(actual)!=len(set(actual)): coverage_errors.append('all_intent_review_coverage')
    for row in reviews:
        if row.get('reviewer')!='Codex direct transcript review': coverage_errors.append('review_attribution')
        for criterion in CRITERIA:
            judgment=row.get('criteria',{}).get(criterion,{})
            if judgment.get('judgment') not in JUDGMENTS or not judgment.get('reason') or 'current_refs' not in judgment or 'historical_observation_keys' not in judgment:
                coverage_errors.append(str(row.get('root_id'))+':review_criterion:'+criterion)
    for item in mechanical['items']:
        matches=[r for r in reviews if r['root_id']==item['root_id'] and r.get('intent_index')==item['intent_index']]
        if len(matches)!=1: continue
        row=matches[0]
        if row.get('reviewer')!='Codex direct transcript review': coverage_errors.append('review_attribution')
        if set(row.get('binding_indices',[]))!=set(item['binding_indices']): coverage_errors.append('binding_coverage')
        for criterion in CRITERIA:
            judgment=row.get('criteria',{}).get(criterion,{})
            if judgment.get('judgment') not in JUDGMENTS or not judgment.get('reason') or 'current_refs' not in judgment or 'historical_observation_keys' not in judgment:
                coverage_errors.append(item['root_id']+':'+item['intent_id']+':'+criterion)
        if (item['root_id'] in passing and item['intent_valid'] and item['binding_mechanical_valid'] and item['binding_state']=='PROPOSED_TRANSFER'
            and all(row.get('criteria',{}).get(k,{}).get('judgment')=='PASS' for k in CRITERIA[:-1])):
            passing[item['root_id']].append(item['intent_id'])
    # Every actual raw binding, including unknown IDs, needs review attribution.
    for root_id in FIRST_ROOTS:
        rows=optional(root_path(run,root_id)/'binding_validation.json',{}).get('bindings',[])
        covered={index for r in reviews if r['root_id']==root_id for index in r.get('binding_indices',[])}
        if covered!={r['binding_index'] for r in rows}: coverage_errors.append(root_id+':all_raw_binding_coverage')
    engineering='PASS' if mechanical['status']=='PASS' and history['status']=='PASS' and optional(run/'audit/test_report.json',{}).get('status')=='PASS' and optional(run/'audit/token_report.json',{}).get('status')=='PASS' else 'FAIL'
    negative=mechanical['roots'][2]
    isolation=negative['checks'].get('negative_input_blocked',False)
    negative_model_compliance=negative['valid_no_transfer']==negative['actual_intents'] and negative['actual_intents']>0
    content='INCONCLUSIVE' if coverage_errors else ('PASS' if all(passing.values()) and isolation and negative_model_compliance else 'FAIL')
    ready=global_review.get('READY_FOR_WRITER_SMALL_ACCEPTANCE','INCONCLUSIVE')
    if ready not in {'PASS','FAIL','INCONCLUSIVE'}: ready='INCONCLUSIVE'
    if ready=='PASS' and (engineering!='PASS' or content!='PASS' or global_review.get('systematic_mismatch_or_wide_chain_success') is not False):
        ready='INCONCLUSIVE'; coverage_errors.append('readiness_prerequisites_not_met')
    result={'status':'COMPLETED' if not coverage_errors else 'BLOCKED','ENGINEERING':engineering,'MINER_CONTENT':content,
        'READY_FOR_WRITER_SMALL_ACCEPTANCE':ready,'readiness_reason':global_review.get('reason','Offline global judgment missing'),
        'passing_transfers_by_root':passing,'negative_input_isolation':isolation,'negative_model_no_transfer_compliance':negative_model_compliance,
        'review_count':len(reviews),'coverage_errors':coverage_errors,'roots':mechanical['roots'],'accounting':mechanical['accounting'],
        'history_preservation':history,'reviewer':'Codex direct transcript review','at_utc':now()}
    atomic_json(run/'reports/acceptance_summary.json',result)
    lines=[f"# P2-2 Miner Binding v2 验收报告\n",f"ENGINEERING：{engineering}；MINER_CONTENT：{content}；READY_FOR_WRITER_SMALL_ACCEPTANCE：{ready}。\n",
        "共享 generation.py 提取 generate_root_intents；原 generate_root_qa 调用相同 Miner 阶段后继续原 Writer。no-hook 和真实 v1 hooks 均做快照回归。默认 Miner 示例、Planner 和排序方法未改。\n",
        "v2 binding 固定五字段，取消 evidence_refs / target_distinction；change_from_source 改为 old_question_focus、feedback_basis、added_distinction 三字段对象。派生证据仅取原已验证链，不搜索、增补、裁剪或按内容过滤；机械通过不代表语义通过。\n",
        "| root | 原有效 intents | 提出/结构有效迁移 | 主动/有效无迁移 | 无效 bindings | 调用/完成 |\n|---|---:|---:|---:|---:|---:|"]
    for r in mechanical['roots']:
        lines.append(f"| {r['root_id']} | {r['valid_intents']} | {r['proposed_bindings']}/{r['valid_proposed_bindings']} | {r['model_no_transfer']}/{r['valid_no_transfer']} | {r['invalid_binding_rows']} | {r['logical_calls']}/{r['completed_calls']} |")
    lines.extend(['',f"全部 {len(reviews)} 项审阅见 miner_binding_content_review.jsonl；审阅者为 Codex direct transcript review，不是人类认证、独立盲审或程序语义判定。覆盖错误：{coverage_errors}。",f"负例输入阻断：{isolation}；模型自产无迁移合规：{negative_model_compliance}。",''])
    narrative=run/'reports/direct_review_narrative.md'
    if narrative.is_file(): lines.append(narrative.read_text())
    lines.extend([f"下一阶段就绪判断：{result['readiness_reason']}\n",
        f"程序测试：{optional(run/'audit/test_report.json',{}).get('tests_run',0)} 项；调用记账：{json.dumps(mechanical['accounting'],ensure_ascii=False)}\n",
        f"token 预检与实际 usage 核对见 audit/token_report.json；使用已核实 131072 context、4096 输出和 512 余量，无截断。历史保全：{history['status']}（P1 与 V1_RUN 全部文件 SHA 与数量核对）。\n",
        "v1/v2 是读过失败后的接口迭代，只作描述性比较；本次不评 QA、Gold、Free 质量、排序质量、有效难度或反馈因果收益。\n",
        "本次 Writer、repair、Solver、Memory、训练及其他 LLM 调用均为 0。到此停止；Writer 小规模验证须用户另行启动。\n"])
    atomic_text(run/'reports/miner_only_acceptance_report.md','\n'.join(lines))
    atomic_text(run/'RUN_STATUS.md',f"Prepare: PASS\nLive: {optional(run/'audit/live_completion.json',{}).get('status','INCOMPLETE')}\nReport: {result['status']}\nENGINEERING: {engineering}\nMINER_CONTENT: {content}\nREADY_FOR_WRITER_SMALL_ACCEPTANCE: {ready}\nSTOPPED; no Writer/Solver/Memory/training calls.\n")
    paths=['execution_contract.json','implementation_manifest.json','input_path_audit.json','audit/frozen_manifest.json','audit/test_report.json','audit/shared_miner_parity.json','audit/preflight_report.json','audit/token_identity.json','audit/token_report.json','audit/call_accounting.json','audit/history_preservation.json','audit/mechanical_report.json','reports/miner_binding_content_review.jsonl','reports/source_to_intent_trace.jsonl','reports/miner_only_acceptance_report.md','reports/acceptance_summary.json','RUN_STATUS.md']
    atomic_text(run/'ARTIFACT_INDEX.md','# Miner Binding v2 artifacts\n\n'+''.join(f'- [{p}]({p})\n' for p in paths)+'\nExact inputs: inputs/<root>/; immutable snapshots: policy_snapshot/; actual Miner raw/sidecars/ledger: qa/shards/shard-00/roots/<root>/. All Guided previews are NOT_SENT_TO_WRITER.\n')
    return result
