"""Read-only post-run verification. Never calls the backend or edits frozen files."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;W=WORK.parent
sys.path.insert(0,str(WORK/'execution'))
import run_pilot as runtime
import score_pilot as scoring
import provider_client as client
CHECKS=[];ISSUES=[];API_OBSERVATIONS=[]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def check(ok,label,detail=None):
    CHECKS.append(label)
    if not ok:ISSUES.append({'check':label,'detail':detail})
    return ok
def verified(b,label):
    try:return check(sha(b['path'])==b['sha256'],label)
    except (KeyError,TypeError,OSError) as e:return check(False,label,str(e))
def old_seals():
    provenance=load(WORK/'diagnosis/PROVENANCE.json');rows=[]
    for old in provenance['sealed_snapshots']:
        p=Path(old['path']);verified(old,'unchanged old manifest '+p.parent.name);m=load(p)
        for row in m['files']:
            fp=Path(row['path']);fp=fp if fp.is_absolute() else p.parent/fp
            verified({'path':str(fp),'sha256':row['sha256']},'old sealed file '+str(fp.relative_to(W)))
        rows.append({'manifest_binding':bind(p),'verified_file_count':len(m['files'])})
    # The complete failed v1 remains part of the preserved experimental record.
    previous=WORK.with_name('day2_completion_v1')/'COMPLETE_MANIFEST.json'
    if previous.exists():
        m=load(previous)
        for row in m['files']:
            fp=previous.parent/row['path'];verified({'path':str(fp),'sha256':row['sha256']},'preserved v1 file '+row['path'])
        check(m.get('day2_all_goals_complete') is False,'v1 remains infrastructure-aborted, not whole Day2 complete')
        rows.append({'manifest_binding':bind(previous),'verified_file_count':len(m['files'])})
    return rows
def checks_before():
    rows=old_seals();freeze=load(WORK/'EXECUTION_FREEZE.json')
    for b in freeze['files']:verified(b,'execution freeze '+b['path'])
    preflight=load(HERE/'RUNTIME_PREFLIGHT.json')
    for b in preflight['reviewed_files']:verified(b,'auditor preflight '+b['path'])
    for root,h in freeze['source_hashes'].items():verified({'path':str(runtime.OLD/'sources'/(root+'.json')),'sha256':h},'frozen source '+root)
    started=load(WORK/'pilot/STARTED.json')
    verified(started['freeze_binding'],'live start exact freeze');verified(started['audit_binding'],'live start exact preflight')
    check(any(b['path']==str(HERE/'RUNTIME_PREFLIGHT.json') for b in freeze['files']),'freeze contains audited code-bound preflight')
    return rows
def check_role(d,expected_payload,cap,role_name,root,arm,scene):
    task=load(d/'TASK.json');prefix=f'{root} {arm} S{scene} {role_name}'
    check(task['role']==role_name,prefix+' role identity')
    check(task['payload']==expected_payload,prefix+' exact isolated payload')
    check(task['requested_max_output_tokens']==cap,prefix+' requested cap')
    rendered=task['instructions']+'\n'+json.dumps({'input':task['payload'],'response_structure':task['response_structure']},ensure_ascii=False,separators=(',',':'))
    request=(d/'REQUEST.txt').read_text();check(request==rendered,prefix+' exact task rendering')
    sub=load(d/'api/SUBMITTED.json');payload=sub['payload']
    check(payload['input']==[{'role':'user','content':request}],prefix+' exact fresh API input')
    check(payload.get('tools')==[] and payload.get('tool_choice')=='none' and payload.get('store') is False and 'previous_response_id' not in payload,prefix+' fresh tool-free request')
    check(payload['max_output_tokens']==cap and sub['automatic_retry'] is False,prefix+' call-count and requested budget')
    amendment=load(WORK/'EXECUTION_AMENDMENT.json')['backend']
    check(sub['endpoint']==amendment['endpoint'] and payload['model']==amendment['model_requested'],prefix+' frozen provider/model')
    c=d/'api/COMPLETE.json';f=d/'api/FAILED.json'
    check(c.exists()!=f.exists(),prefix+' exactly one terminal API record')
    observation={'root_id':root,'arm':arm,'scene_index':scene,'role':role_name,'terminal':'FAILED' if f.exists() else 'COMPLETE'}
    raw_path=d/'api/RAW_RESPONSE.json'
    if raw_path.exists():
        try:raw=client.strict_json(raw_path.read_bytes())
        except (ValueError,TypeError) as exc:
            observation['raw_response_parse_error']=str(exc);API_OBSERVATIONS.append(observation)
            check(f.exists(),prefix+' unparsable raw response was stopped')
            return
        observed_output_types=[x.get('type') for x in raw.get('output',[])]
        observed_activity=client.tool_activity(raw.get('tool_usage',{})) or any(x not in ('message','reasoning') for x in observed_output_types)
        observation.update(tool_choice_returned=raw.get('tool_choice'),tools_returned=raw.get('tools'),tool_usage_nonzero_or_tool_output=observed_activity,output_types=observed_output_types,control_echo_match=not raw.get('tools') and raw.get('tool_choice')=='none',response_chain_observed=raw.get('previous_response_id') is not None)
        instructions=raw.get('instructions')
        observation['provider_instructions_sha256']=hashlib.sha256(instructions.encode()).hexdigest() if isinstance(instructions,str) else None
        if isinstance(instructions,str):check(not any(r in instructions for r in runtime.ROOTS) and 'selfplay_week_20261019' not in instructions,prefix+' no root or experiment material in provider instructions')
        if role_name=='realizer':
            texts=[v['text'] for item in raw.get('output',[]) if item.get('type')=='message' for v in item.get('content',[]) if v.get('type')=='output_text']
            try:observation['raw_candidate_json_present']=len(texts)==1 and isinstance(client.strict_json(texts[0]).get('turns'),list)
            except (ValueError,TypeError,AttributeError):observation['raw_candidate_json_present']=False
    if f.exists():
        failed=load(f);observation['error']=failed.get('error');observation['error_type']=failed.get('error_type');observation['failed_attempt_usage_available']=isinstance(failed.get('usage'),dict)
        API_OBSERVATIONS.append(observation);return
    meta=load(c);raw=client.strict_json((d/'api/RAW_RESPONSE.json').read_bytes())
    check(raw.get('model')==payload['model'] and raw.get('status')=='completed',prefix+' completed requested model')
    check(raw.get('previous_response_id') is None and not raw.get('tools') and not client.tool_activity(raw.get('tool_usage',{})),prefix+' no observed tools or response chain')
    check(all(x.get('type') in ('message','reasoning') for x in raw.get('output',[])),prefix+' output item types')
    chunks=[v['text'] for item in raw.get('output',[]) if item.get('type')=='message' for v in item.get('content',[]) if v.get('type')=='output_text']
    check(len(chunks)==1,prefix+' one final JSON output')
    value=client.strict_json(chunks[0])
    check(value==load(d/'RESPONSE.json'),prefix+' response exact provider output')
    rr=load(d/'ROLE_RECEIPT.json')
    for k in ('task_binding','request_binding','response_binding','api_binding'):verified(rr[k],prefix+' role receipt '+k)
    check(meta.get('usage')==raw.get('usage') and meta.get('input_tokens')==raw.get('usage',{}).get('input_tokens') and meta.get('output_tokens')==raw.get('usage',{}).get('output_tokens'),prefix+' raw actual usage preserved')
    generic=load(WORK.with_name('day2_completion_v1')/'execution/PROVIDER_CONTROLS_PROBE.json')['response']['instructions']
    observation['instructions_equal_controls_probe']=raw.get('instructions')==generic
    API_OBSERVATIONS.append(observation)
def check_branch(root,arm):
    branch=WORK/'pilot/branches'/(root+'__'+arm);prefix_path=runtime.OLD/'sources'/(root+'.json')
    memory_path=WORK/'initial_memory'/root/'FACT_MEMORY_SNAPSHOT.json';review_path=memory_path.parent/'SEMANTIC_REVIEW.json'
    stopped=False;calls=0
    for scene in range(1,4):
        sd=branch/('scene_'+str(scene).zfill(2));r=load(sd/'SCENE_RECEIPT.json');label=root+' '+arm+' S'+str(scene)
        check(r['root_id']==root and r['arm']==arm and r['scene_index']==scene,label+' fixed slot identity')
        verified(r['accepted_prefix_binding'],label+' frozen accepted input prefix')
        check(r['accepted_prefix_binding']['sha256']==sha(prefix_path),label+' own accepted-prefix chronology')
        submitted=list(sd.glob('*/api/SUBMITTED.json'));calls+=len(submitted)
        check(not stopped or (r['status']=='MISSING' and not submitted),label+' stopped descendants remain missing without calls')
        if not submitted:
            if r['status']!='PASS':stopped=True
            continue
        source=load(prefix_path);memory=None
        if arm==runtime.ARMS[1] and submitted:memory=runtime.memory_gate(memory_path,review_path,prefix_path)
        context=runtime.context(source,arm,memory)
        planner=sd/'planner';realizer=sd/'realizer';reviewer=sd/'reviewer'
        common=None
        if (planner/'TASK.json').exists():
            actual=load(planner/'TASK.json')['payload'];common={**context,'high_level_objective':runtime.PROTOCOL['context']['shared_objectives'][scene-1],'requirements':actual.get('requirements')}
            check(set(actual)=={'context','prefix_fact_memory','high_level_objective','requirements'},label+' planner payload allowlist')
            check_role(planner,common,1024,'planner',root,arm,scene)
        if (realizer/'TASK.json').exists():
            plan=load(planner/'RESPONSE.json');check_role(realizer,{**common,'approved_plan_for_this_request':plan},4096,'realizer',root,arm,scene)
        if (reviewer/'TASK.json').exists():
            actual=load(reviewer/'TASK.json')['payload'];candidate=load(realizer/'RESPONSE.json')
            sessions=list(dict.fromkeys(t['session_ref'] for t in load(runtime.OLD/'sources'/(root+'.json'))['turns']))
            source0=load(runtime.OLD/'sources'/(root+'.json'));boundary=next(i for i,t in enumerate(source0['turns']) if t['session_ref'] in sessions[-2:])
            opaque=hashlib.sha256((root+'|'+arm).encode()).hexdigest()[:16]
            expected={'root_id':root,'branch_id':opaque,'scene_index':scene,'candidate_id':opaque+':'+str(scene),'candidate_sha256':sha(realizer/'RESPONSE.json'),'accepted_prefix_sha256':sha(prefix_path),'full_accepted_prefix':source,'candidate':candidate,'initial_older_history_before_turn_index':boundary,'mechanical_duplicate_turn_ratio':runtime.duplicate_ratio(source['turns'],candidate['turns']),'instructions':actual.get('instructions')}
            check_role(reviewer,expected,3072,'source_reviewer',root,arm,scene)
        if r['status']=='PASS':
            row=scoring.read_receipt(WORK,root,arm,scene);check(row['status']=='PASS' and row['receipt_valid'],label+' independent accepted artifact validation',row.get('reason'))
            previous_path=prefix_path;prefix_path=Path(r['accepted_output_prefix_binding']['path'])
            if arm==runtime.ARMS[1] and scene<3:
                updater=sd/'memory_update';mreviewer=sd/'memory_review'
                if (updater/'TASK.json').exists():
                    actual=load(updater/'TASK.json')['payload'];base=load(memory_path);new=load(prefix_path)
                    expected={'previous_approved_memory':runtime.memory_projection(base),'new_accepted_turns':[{'turn_index':i,**t} for i,t in enumerate(new['turns']) if i>=len(load(previous_path)['turns'])],'instructions':actual.get('instructions')}
                    check_role(updater,expected,2048,'memory_update',root,arm,scene)
                snapshot=updater/'FACT_MEMORY_SNAPSHOT.json'
                if (mreviewer/'TASK.json').exists():
                    actual=load(mreviewer/'TASK.json')['payload'];expected={'root_id':root,'snapshot_sha256':sha(snapshot),'full_source_sha256':sha(prefix_path),'full_accepted_prefix':load(prefix_path),'candidate_memory':load(snapshot),'instructions':actual.get('instructions')}
                    check_role(mreviewer,expected,3072,'memory_review',root,arm,scene)
                gate=sd/'MEMORY_GATE.json'
                if gate.exists():
                    g=load(gate)
                    for k in ('snapshot_binding','review_binding','accepted_prefix_binding'):verified(g[k],label+' maintenance '+k)
                    runtime.memory_gate(g['snapshot_binding']['path'],g['review_binding']['path'],prefix_path)
                    check(g['decision']=='PASS',label+' maintenance accepted exact hash')
                    memory_path=Path(g['snapshot_binding']['path']);review_path=Path(g['review_binding']['path'])
                else:stopped=True
        else:stopped=True
    return calls
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before-only',action='store_true');args=parser.parse_args()
    old=checks_before()
    if args.before_only:
        result={'status':'PASS' if not ISSUES else 'FAIL','check_count':len(CHECKS),'issues':ISSUES,'old_seals':old,'model_calls':0,'network_calls':0}
        p=HERE/'LIVE_BINDINGS_AND_OLD_SEALS.json'
    else:
        check(load(WORK/'pilot/COMPLETE.json')['status']=='ALL_54_SLOTS_TERMINAL','controller fixed54 complete')
        calls=sum(check_branch(r,a) for r in runtime.ROOTS for a in runtime.ARMS)
        allsubmitted=list((WORK/'pilot/branches').glob('*/scene_*/*/api/SUBMITTED.json'))
        check(calls==len(allsubmitted) and calls<=186,'all API attempts accounted within hard186 call bound')
        recorded=load(WORK/'pilot/SCORES.json');recalc=scoring.score(WORK,runtime.PROTOCOL)
        rc=copy.deepcopy(recorded);rc.pop('original_protocol_binding',None);rc.pop('created_at_utc',None);recalc.pop('created_at_utc',None)
        check(rc==recalc,'independent re-execution of committed scorer exact outcome equality')
        check(len(recorded['receipts'])==54 and all(recorded['arms'][a]['planned_scenes']==18 for a in runtime.ARMS),'score fixed54 /18 denominators')
        check(recorded['cost_status']=='NO_COST_CONCLUSION' and not recorded['memory_adopted'] and recorded['selected_context_policy']=='full_history','predeclared unknown-cost fallback preserved')
        aggregate={}
        for arm in runtime.ARMS:
            rows=[load(WORK/'pilot/branches'/(r+'__'+arm)/('scene_'+str(i).zfill(2))/'SCENE_RECEIPT.json') for r in runtime.ROOTS for i in range(1,4)]
            accepted=sum(r['status']=='PASS' for r in rows);aggregate[arm]={'accepted':accepted,'denominator':18,'root_equal_quality':sum(sum(x['status']=='PASS' for x in rows if x['root_id']==r)/3 for r in runtime.ROOTS)/6}
            check(accepted==recorded['arms'][arm]['accepted_initial_scenes'] and aggregate[arm]['root_equal_quality']==recorded['arms'][arm]['root_equal_quality'],'independent raw counts '+arm)
        blocked=[x for x in API_OBSERVATIONS if x.get('error') in ('TOOLS_NOT_DISABLED','RETURNED_TOOLS_NOT_EMPTY')]
        ledger=recorded['receipts'];reviews=list((WORK/'pilot/branches').glob('*/scene_*/reviewer/RESPONSE.json'))
        review_decisions={k:sum(load(p).get('decision')==k for p in reviews) for k in ('PASS','FAIL','HOLD')}
        format_holds=[]
        for row in ledger:
            rp=WORK/'pilot/branches'/(row['root_id']+'__'+row['arm'])/('scene_'+str(row['scene_index']).zfill(2))/'reviewer/RESPONSE.json'
            if row['recorded_status']=='HOLD' and rp.exists() and row['review_binding'] is None:
                review=load(rp);badpairs=[i for i,e in enumerate(review.get('evidence_refs',[])) if (e.get('prefix_turn_index') is None)!=(e.get('prefix_quote') is None)]
                format_holds.append({'root_id':row['root_id'],'arm':row['arm'],'scene_index':row['scene_index'],'model_decision':review.get('decision'),'review_binding':bind(rp),'unpaired_optional_prefix_evidence_indices':badpairs,'classification':'REVIEW_RECORD_EVIDENCE_FORMAT_HOLD'})
        limitations={'interface_control_mismatch_failed_calls':len(blocked),'interface_control_mismatch_root_ids':sorted({x['root_id'] for x in blocked}),'interface_control_mismatch_root_count':len({x['root_id'] for x in blocked}),'interface_control_mismatch_branch_count':len({(x['root_id'],x['arm']) for x in blocked}),'interface_blocked_realizer_calls':sum(x['role']=='realizer' for x in blocked),'interface_scene_HOLD_receipts':sum(x['recorded_status']=='HOLD' and 'TOOLS_NOT_DISABLED' in (x.get('reason') or '') for x in ledger),'interface_maintenance_failures':sum(x['role'] in ('memory_update','memory_review') for x in blocked),'review_record_format_HOLD_count':len(format_holds),'review_record_format_HOLD_details':format_holds,'actual_tool_activity_observed_calls':sum(x.get('tool_usage_nonzero_or_tool_output') is True for x in API_OBSERVATIONS),'official_candidate_response_files':len(list((WORK/'pilot/branches').glob('*/scene_*/realizer/RESPONSE.json'))),'raw_candidate_json_returned_calls':sum(x.get('raw_candidate_json_present') is True for x in API_OBSERVATIONS),'complete_source_review_response_files':len(reviews),'source_review_decisions':review_decisions,'non_admitted_HOLD_receipts':sum(x['recorded_status']=='HOLD' for x in ledger),'reviewed_content_FAIL_receipts':sum(x['recorded_status']=='FAIL' and x['review_binding'] is not None for x in ledger),'reviewed_content_HOLD_receipts':sum(x['recorded_status']=='HOLD' and x['review_binding'] is not None for x in ledger),'memory_quality_comparison':'Descriptive six-root initial acceptance comparison only; process/format HOLD must be separated from source-reviewed content failures. No causal or population quality claim; inspect all process-failure counts before interpreting differences.','cost_comparison':'NO_COST_CONCLUSION','diagnosed_model_weakness_from_interface_failures':False}
        failed_calls=[x for x in API_OBSERVATIONS if x['terminal']=='FAILED']
        errors=sorted({x.get('error_type','UNKNOWN') for x in failed_calls})
        limitations['api_failure_counts_by_error_type']={e:sum(x.get('error_type','UNKNOWN')==e for x in failed_calls) for e in errors}
        limitations['api_failed_attempts_without_known_usage']=sum(not x.get('failed_attempt_usage_available') for x in failed_calls)
        limitations['network_or_backend_failure_details']=[x for x in failed_calls if x.get('error_type') not in ('AssertionError',)]
        limitations['source_reviewed_HOLD_details']=[{'root_id':x['root_id'],'arm':x['arm'],'scene_index':x['scene_index'],'review_binding':x['review_binding']} for x in ledger if x['recorded_status']=='HOLD' and x['review_binding'] is not None]
        maintenance_failures=[]
        for p in sorted((WORK/'pilot/branches').glob('*/scene_*/MEMORY_GATE_FAILURE.json')):
            failure=load(p);rp=p.parent/'memory_review/RESPONSE.json'
            maintenance_failures.append({'failure_binding':bind(p),'branch':p.parent.parent.name,'after_scene_directory':p.parent.name,'reason':failure.get('reason'),'maintenance_review_decision':load(rp).get('decision') if rp.exists() else None,'next_scene_blocked':failure.get('next_scene_blocked')})
        limitations['maintenance_failure_count']=len(maintenance_failures);limitations['maintenance_failure_details']=maintenance_failures
        def known_cost(s):
            return {'api_attempts':sum(a['api_cost']['api_attempt_count'] for a in s['arms'].values()),'input_tokens':sum(a['api_cost']['known_input_tokens'] for a in s['arms'].values()),'output_tokens':sum(a['api_cost']['known_output_tokens'] for a in s['arms'].values()),'total_tokens':sum(a['api_cost']['known_total_tokens'] for a in s['arms'].values()),'sum_call_wall_seconds':sum(a['api_cost']['sum_call_wall_seconds'] for a in s['arms'].values()),'attempts_without_complete_usage_and_timing':sum(sum(not c['actual_usage_and_timing_complete'] for c in a['api_cost']['calls']) for a in s['arms'].values())}
        previous_scores=WORK.with_name('day2_completion_v1')/'pilot/SCORES.json';previous=known_cost(load(previous_scores));current=known_cost(recorded)
        accounting={'v1_preserved_score_binding':bind(previous_scores),'v1_known_pilot_cost':previous,'v2_known_pilot_cost':current,'v1_plus_v2_known_pilot_cost':{k:previous[k]+current[k] for k in current},'scope':'Both live memory-pilot attempts, including their failures; initial internal review actual usage/timing remain unknown. Setup compatibility probes are separately retained and not part of these pilot-only subtotals. Call-wall sums overlap under concurrency and are not experiment makespan.','cost_adoption_conclusion':'NO_COST_CONCLUSION'}
        result={'status':'PASS' if not ISSUES else 'FAIL','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/frozen_audit','check_count':len(CHECKS),'issues':ISSUES,'old_seals':old,'old_files_verified':sum(x['verified_file_count'] for x in old),'api_attempts_verified':calls,'api_observations':API_OBSERVATIONS,'failure_classification_rule':'Returned tool_choice auto/null alone is metadata mismatch, not actual tool activity when tool_usage is zero and output types are message/reasoning. V2 continues rejecting returned nonempty tool availability, unexpected tool output, nonzero usage, response chain and contamination; separate these observed controls from source-reviewed content decisions.','operational_failure_analysis':limitations,'cost_accounting':accounting,'aggregate':aggregate,'scores_binding':bind(WORK/'pilot/SCORES.json'),'freeze_binding':bind(WORK/'EXECUTION_FREEZE.json'),'runtime_preflight_binding':bind(HERE/'RUNTIME_PREFLIGHT.json'),'audit_script_binding':bind(Path(__file__)),'model_calls_by_auditor':0,'network_calls_by_auditor':0,'semantic_limit':'Stored one-pass reviewer decisions verified mechanically; no new human blind adjudication, causal or significance claim.','sealed_input_mutations':0}
        p=HERE/'RESULTS_AUDIT.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'check_count':len(CHECKS),'issues':ISSUES,'result':str(p)},ensure_ascii=False))
if __name__=='__main__':main()
