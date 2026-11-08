#!/usr/bin/env python3
"""Question-free fixed-denominator score of the sealed Day2 memory pilot."""
import argparse, datetime as dt, difflib, hashlib, json, math, re
from pathlib import Path
ARMS=('recent2','recent2_plus_prefix_memory','full_history')
MEM=ARMS[1]
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
def binding_ok(b):
    try: return isinstance(b,dict) and Path(b['path']).is_file() and sha(b['path'])==b['sha256']
    except (KeyError,OSError,TypeError): return False
def ratio(a,b): return a/b if finite(a) and finite(b) and b else None

def duplicate_ratio(prefix,new):
    """Recompute the frozen character-similarity metric from stored dialogue."""
    normal=lambda s:' '.join(re.findall(r'\w+',s.lower()))
    prior=[normal(t['text']) for t in prefix];num=den=0
    greetings={'hi','hello','hey','good morning','good afternoon','good evening','goodbye','bye','thanks','thank you','see you','see you later'}
    for turn in new:
        n=normal(turn['text'])
        if not (len(n.split())<=5 and n in greetings):
            den+=1;num+=int(any(difflib.SequenceMatcher(None,n,p).ratio()>=.90 for p in prior if p))
        prior.append(n)
    if not den:raise ValueError('ZERO_DUPLICATE_DENOMINATOR')
    return num/den

def original_source(work,root):return work.parent/'day2/fact_memory/sources'/f'{root}.json'

def validate_pass_artifacts(work,root,arm,i,r):
    """Mechanical evidence validation only; semantic judgement remains the reviewer’s."""
    candidate=load(r['candidate_binding']['path']);prefix=load(r['accepted_prefix_binding']['path'])
    review=load(r['review_binding']['path']);output=load(r['accepted_output_prefix_binding']['path'])
    source=load(original_source(work,root));turns=candidate['turns'];prior=prefix['turns'];m=r['metrics']
    if i==1 and r['accepted_prefix_binding']['sha256']!=sha(original_source(work,root)):raise ValueError('SCENE1_PREFIX_NOT_FROZEN_SOURCE')
    if prior[:len(source['turns'])]!=source['turns'] or {k:v for k,v in prefix.items() if k!='turns'}!={k:v for k,v in source.items() if k!='turns'}:raise ValueError('PREFIX_CHANGED_ORIGINAL_SOURCE')
    if not isinstance(candidate,dict) or set(candidate)!={'turns'} or not isinstance(turns,list) or not 12<=len(turns)<=20:raise ValueError('CANDIDATE_SCHEMA_OR_LENGTH')
    sessions=list(dict.fromkeys(t['session_ref'] for t in prior))
    visible={t['speaker'] for t in prior if arm=='full_history' or t['session_ref'] in sessions[-2:]}
    for t in turns:
        if not isinstance(t,dict) or set(t)!={'speaker','text'} or t['speaker'] not in visible or not isinstance(t['text'],str) or not t['text'].strip():raise ValueError('CANDIDATE_TURN_OR_SPEAKER')
    computed=duplicate_ratio(prior,turns)
    if computed>.25 or not math.isclose(m['duplicate_turn_ratio'],computed,rel_tol=0,abs_tol=1e-12):raise ValueError('RECOMPUTED_DUPLICATION_GATE')
    opaque=hashlib.sha256((root+'|'+arm).encode()).hexdigest()[:16]
    ids={'root_id':root,'branch_id':opaque,'scene_index':i,'candidate_id':opaque+':'+str(i),'candidate_sha256':r['candidate_binding']['sha256'],'accepted_prefix_sha256':r['accepted_prefix_binding']['sha256']}
    if any(review.get(k)!=v for k,v in ids.items()) or review.get('decision')!='PASS' or review.get('all_plausible_prior_states_checked') is not True:raise ValueError('REVIEW_IDENTITY_HASH_OR_DECISION')
    for key in ('critical_major_error_count','new_progress_count','old_history_linkage_count'):
        if type(review.get(key)) is not int or review[key]!=m[key]:raise ValueError('REVIEW_RECEIPT_METRIC_MISMATCH')
    if 'duplicate_turn_ratio' not in review or not finite(review['duplicate_turn_ratio']):raise ValueError('REVIEW_DUPLICATION_FIELD')
    initial_sessions=list(dict.fromkeys(t['session_ref'] for t in source['turns']))
    old_boundary=next(j for j,t in enumerate(source['turns']) if t['session_ref'] in initial_sessions[-2:])
    evidence=review.get('evidence_refs');supported={'new_progress':set(),'old_history_linkage':set(),'conflict':set()}
    if not isinstance(evidence,list):raise ValueError('REVIEW_EVIDENCE_NOT_LIST')
    for e in evidence:
        if not isinstance(e,dict) or e.get('kind') not in supported:raise ValueError('REVIEW_EVIDENCE_KIND')
        kind=e['kind'];j=e.get('candidate_turn_index');q=e.get('quote');p=e.get('prefix_turn_index');pq=e.get('prefix_quote')
        if type(j) is not int or not 0<=j<len(turns) or not isinstance(q,str) or not q or q not in turns[j]['text']:raise ValueError('REVIEW_CANDIDATE_QUOTE')
        if kind in ('old_history_linkage','conflict') or p is not None or pq is not None:
            if type(p) is not int or not 0<=p<len(prior) or not isinstance(pq,str) or not pq or pq not in prior[p]['text']:raise ValueError('REVIEW_PREFIX_QUOTE')
        if kind=='old_history_linkage' and not p<old_boundary:raise ValueError('LINKAGE_NOT_INITIAL_OLDER_HISTORY')
        supported[kind].add((j,q,p,pq))
    if review['new_progress_count']>len(supported['new_progress']) or review['old_history_linkage_count']>len(supported['old_history_linkage']):raise ValueError('REVIEW_COUNTS_WITHOUT_UNIQUE_SUPPORT')
    if review['critical_major_error_count']>0 and not supported['conflict']:raise ValueError('CONFLICT_WITHOUT_SUPPORT')
    if {k:v for k,v in output.items() if k!='turns'}!={k:v for k,v in prefix.items() if k!='turns'}:raise ValueError('OUTPUT_PREFIX_METADATA_CHANGED')
    appended=output.get('turns',[])
    if len(appended)!=len(prior)+len(turns) or appended[:len(prior)]!=prior:raise ValueError('OUTPUT_PREFIX_NOT_EXACT_APPEND')
    session='D2P:'+opaque+':S'+str(i)
    for j,(record,turn) in enumerate(zip(appended[len(prior):],turns)):
        expected={'turn_ref':session+':T'+str(j).zfill(3),'session_ref':session,'scene_ref':session,**turn}
        if record!=expected:raise ValueError('OUTPUT_APPEND_DIFFERS_FROM_CANDIDATE')
    return {'status':'PASS','recomputed_duplicate_turn_ratio':computed,'verified_new_progress_reference_count':len(supported['new_progress']),'verified_old_history_reference_count':len(supported['old_history_linkage']),'semantic_limit':'Exact quotes, indices, identities and counts checked mechanically; the stored source reviewer supplies the semantic judgement.'}

def read_receipt(work,root,arm,i):
    p=work/'pilot/branches'/f'{root}__{arm}'/f'scene_{i:02d}/SCENE_RECEIPT.json'
    row={'root_id':root,'arm':arm,'scene_index':i,'planned':True,'receipt_binding':None,'recorded_status':'MISSING','status':'MISSING','reason':'RECEIPT_FILE_MISSING','metrics':None,'candidate_binding':None,'accepted_prefix_binding':None,'review_binding':None,'accepted_output_prefix_binding':None,'receipt_valid':False}
    if not p.exists(): return row
    row['receipt_binding']=bind(p)
    try: r=load(p)
    except (OSError,ValueError): row.update(status='HOLD',reason='MALFORMED_RECEIPT');return row
    row['recorded_status']=r.get('status');row['reason']=r.get('reason');row['metrics']=r.get('metrics')
    for k in ('candidate_binding','accepted_prefix_binding','review_binding','accepted_output_prefix_binding'): row[k]=r.get(k)
    identity=(r.get('root_id')==root and r.get('arm')==arm and r.get('scene_index')==i)
    status=r.get('status');row['status']=status if identity and status in ('PASS','FAIL','HOLD','MISSING') else 'HOLD'
    if not identity:row['reason']='RECEIPT_IDENTITY_MISMATCH';return row
    if status not in ('PASS','FAIL','HOLD','MISSING'):row['reason']='INVALID_STATUS';return row
    if status!='PASS':row['receipt_valid']=True;return row
    m=r.get('metrics') or {};progress=m.get('new_progress_count');dup=m.get('duplicate_turn_ratio');link=m.get('old_history_linkage_count');crit=m.get('critical_major_error_count')
    metrics=(finite(progress) and progress>=1 and isinstance(progress,int) and finite(dup) and dup<=.25 and finite(link) and isinstance(link,int) and crit==0 and not isinstance(crit,bool))
    bindings=all(binding_ok(r.get(k)) for k in ('candidate_binding','accepted_prefix_binding','review_binding','accepted_output_prefix_binding'))
    if not metrics or not bindings:
        row.update(status='HOLD',reason='PASS_RECEIPT_METRICS_OR_BYTE_BINDINGS_INVALID',receipt_valid=False)
    else:
        try:row['artifact_validation']=validate_pass_artifacts(work,root,arm,i,r);row['receipt_valid']=True
        except (OSError,ValueError,KeyError,TypeError,IndexError,StopIteration) as exc:row.update(status='HOLD',reason='PASS_ARTIFACT_VALIDATION_FAILED:'+str(exc),receipt_valid=False)
    return row

def cost_for_arm(work,arm):
    base=work/'pilot/branches';calls=[]
    dirs=sorted(p for p in base.glob(f'*__{arm}/scene_*/*/api/SUBMITTED.json'))
    for submitted in dirs:
        d=submitted.parent;s=load(submitted);complete=d/'COMPLETE.json';failed=d/'FAILED.json';terminal=complete if complete.exists() else failed if failed.exists() else None
        try:t=load(terminal) if terminal else {}
        except (OSError,ValueError):t={}
        usage=t.get('usage') or {}
        if not usage and (d/'RAW_RESPONSE.json').exists():
            try: usage=load(d/'RAW_RESPONSE.json').get('usage') or {}
            except (OSError,ValueError): pass
        inp=t.get('input_tokens',usage.get('input_tokens'));out=t.get('output_tokens',usage.get('output_tokens'));wall=t.get('wall_seconds')
        if inp is None:inp=usage.get('input_tokens')
        if out is None:out=usage.get('output_tokens')
        valid=finite(inp) and finite(out) and finite(wall) and terminal is not None
        calls.append({'directory':str(d),'role':d.parent.name,'scene_directory':str(d.parent.parent),'submitted_binding':bind(submitted),'terminal_binding':bind(terminal) if terminal else None,'terminal_status':'COMPLETE' if complete.exists() else 'FAILED' if failed.exists() else 'MISSING','model':t.get('returned_model',t.get('model',(s.get('payload') or {}).get('model'))),'input_tokens':inp if finite(inp) else None,'output_tokens':out if finite(out) else None,'wall_seconds':wall if finite(wall) else None,'actual_usage_and_timing_complete':valid})
    return {'api_attempt_count':len(calls),'completed_attempts':sum(c['terminal_status']=='COMPLETE' for c in calls),'failed_attempts':sum(c['terminal_status']=='FAILED' for c in calls),'missing_terminal_attempts':sum(c['terminal_status']=='MISSING' for c in calls),'known_input_tokens':sum(c['input_tokens'] or 0 for c in calls),'known_output_tokens':sum(c['output_tokens'] or 0 for c in calls),'known_total_tokens':sum((c['input_tokens'] or 0)+(c['output_tokens'] or 0) for c in calls),'sum_call_wall_seconds':sum(c['wall_seconds'] or 0 for c in calls),'all_recorded_api_calls_have_usage_and_timing':bool(calls) and all(c['actual_usage_and_timing_complete'] for c in calls),'wall_semantics':'Sum of observed per-call wall; concurrent calls may overlap. This is not full experiment makespan.','calls':calls}

def initial_memory(work,roots):
    records=[]
    for root in roots:
        d=work/'initial_memory'/root;mfile=d/'FACT_MEMORY_SNAPSHOT.json';rfile=d/'SEMANTIC_REVIEW.json';cfile=d/'ROLE_CALL.json'
        try:
            m=load(mfile);r=load(rfile);c=load(cfile);passed=(r.get('root_id')==root and r.get('decision')=='PASS' and r.get('snapshot_sha256')==sha(mfile) and r.get('full_source_sha256')==m.get('source_sha256') and r.get('all_facts_reviewed') is True and set(r.get('accepted_fact_ids',[]))=={f['fact_id'] for f in m.get('facts',[])} and r.get('coverage_status') in ('BOUNDED','REVIEWED_COMPLETE_WITHIN_DECLARED_SCOPE'))
            actual=c.get('actual_usage');inp=actual.get('input_tokens') if isinstance(actual,dict) else None;out=actual.get('output_tokens') if isinstance(actual,dict) else None;wall=c.get('actual_model_call_wall_seconds');costok=finite(inp) and finite(out) and finite(wall)
            records.append({'root_id':root,'semantic_gate_pass':passed,'snapshot_binding':bind(mfile),'review_binding':bind(rfile),'role_call_binding':bind(cfile),'actual_usage':actual,'actual_model_call_wall_seconds':wall,'input_tokens':inp,'output_tokens':out,'actual_cost_complete':costok,'tokenizer_estimates':c.get('tokenizer_estimates'),'timing_limit':c.get('timing_note')})
        except (OSError,ValueError,KeyError,TypeError):records.append({'root_id':root,'semantic_gate_pass':False,'actual_cost_complete':False,'reason':'INITIAL_MEMORY_RECORD_MISSING_OR_INVALID'})
    return {'root_count':len(roots),'all_six_exact_hash_semantic_pass':all(x['semantic_gate_pass'] for x in records),'all_six_actual_cost_complete':all(x['actual_cost_complete'] for x in records),'known_input_tokens':sum(x.get('input_tokens') or 0 for x in records),'known_output_tokens':sum(x.get('output_tokens') or 0 for x in records),'known_model_wall_seconds':sum(x.get('actual_model_call_wall_seconds') or 0 for x in records),'records':records,'actual_initial_cost_unknown':not all(x['actual_cost_complete'] for x in records)}

def score(work,protocol):
    work=Path(work);roots=protocol['selection']['root_ids'];assert len(roots)==6 and len(set(roots))==6
    rows=[read_receipt(work,root,arm,i) for arm in ARMS for root in roots for i in range(1,4)]
    # Enforce accepted-prefix lineage without deleting rows or replacing failures.
    for arm in ARMS:
        for root in roots:
            branch=[r for r in rows if r['arm']==arm and r['root_id']==root]
            for before,after in zip(branch,branch[1:]):
                if after['status']=='PASS' and (before['status']!='PASS' or not before['accepted_output_prefix_binding'] or before['accepted_output_prefix_binding']['sha256']!=after['accepted_prefix_binding']['sha256']):after.update(status='HOLD',receipt_valid=False,reason='BROKEN_ACCEPTED_PREFIX_LINEAGE')
    arms={}
    for arm in ARMS:
        ar=[r for r in rows if r['arm']==arm];byroot=[]
        for root in roots:
            rr=[r for r in ar if r['root_id']==root];passed=[r for r in rr if r['status']=='PASS'];linked=[r for r in passed if r['metrics']['old_history_linkage_count']>0]
            byroot.append({'root_id':root,'accepted_initial_scenes':len(passed),'denominator':3,'quality':len(passed)/3,'consistent_old_history_linkage_scenes':len(linked),'full_branch_complete':len(passed)==3,'scene3_reached':rr[2]['recorded_status'] in ('PASS','FAIL','HOLD') and binding_ok(rr[2]['candidate_binding']) and binding_ok(rr[2]['review_binding'])})
        counts={s:sum(r['status']==s for r in ar) for s in ('PASS','FAIL','HOLD','MISSING')}
        arms[arm]={'planned_scenes':18,'status_counts':counts,'accepted_initial_scenes':counts['PASS'],'accepted_scene_fraction':counts['PASS']/18,'root_equal_quality':sum(r['quality'] for r in byroot)/6,'independent_root_count':6,'root_details':byroot,'completed_accepted_branches':sum(r['full_branch_complete'] for r in byroot),'all_branches_reached_scene3':all(r['scene3_reached'] for r in byroot),'all18_scenes_attempted_and_reviewed':all(r['recorded_status'] in ('PASS','FAIL','HOLD') and binding_ok(r['candidate_binding']) and binding_ok(r['review_binding']) for r in ar),'consistent_old_history_linkage_scene_count':sum(r['consistent_old_history_linkage_scenes'] for r in byroot),'linkage_denominator':18,'api_cost':cost_for_arm(work,arm)}
    initial=initial_memory(work,roots);memory=arms[MEM];full=arms['full_history'];recent=arms['recent2'];mr={r['root_id']:r['accepted_initial_scenes'] for r in memory['root_details']};fr={r['root_id']:r['accepted_initial_scenes'] for r in full['root_details']};rr={r['root_id']:r['accepted_initial_scenes'] for r in recent['root_details']}
    diffs_full={root:mr[root]-fr[root] for root in roots};diffs_recent={root:mr[root]-rr[root] for root in roots}
    maintenance=[]
    for r in rows:
        if r['arm']!=MEM or r['scene_index']>=3 or r['status']!='PASS':continue
        nextrow=next(x for x in rows if x['root_id']==r['root_id'] and x['arm']==MEM and x['scene_index']==r['scene_index']+1)
        if nextrow['recorded_status']=='MISSING':continue
        gate_path=work/'pilot/branches'/f"{r['root_id']}__{MEM}"/f"scene_{r['scene_index']:02d}/MEMORY_GATE.json";ok=False;reason='MEMORY_GATE_MISSING';gate_binding=None
        try:
            gate=load(gate_path);gate_binding=bind(gate_path);mb=gate.get('snapshot_binding');rb=gate.get('review_binding');pb=gate.get('accepted_prefix_binding')
            if not all(binding_ok(b) for b in (mb,rb,pb)):raise ValueError('MEMORY_GATE_BINDING_INVALID')
            rev=load(rb['path']);snap=load(mb['path']);prefix_hash=r['accepted_output_prefix_binding']['sha256']
            ok=(gate.get('decision')=='PASS' and rev.get('decision')=='PASS' and rev.get('root_id')==r['root_id'] and snap.get('root_id')==r['root_id'] and rev.get('snapshot_sha256')==mb['sha256'] and rev.get('full_source_sha256')==prefix_hash and snap.get('source_sha256')==prefix_hash and pb['sha256']==prefix_hash and nextrow['accepted_prefix_binding']['sha256']==prefix_hash and rev.get('all_facts_reviewed') is True and rev.get('coverage_status') in ('BOUNDED','REVIEWED_COMPLETE_WITHIN_DECLARED_SCOPE') and set(rev.get('accepted_fact_ids',[]))=={f['fact_id'] for f in snap.get('facts',[])})
            reason='EXACT_HASH_MEMORY_GATE_PASS' if ok else 'MEMORY_GATE_IDENTITY_OR_SEMANTIC_REVIEW_MISMATCH'
        except (OSError,ValueError,KeyError,TypeError):pass
        maintenance.append({'root_id':r['root_id'],'after_scene_index':r['scene_index'],'required_before_attempted_next_scene':True,'exact_hash_semantic_gate_pass':ok,'gate_binding':gate_binding,'reason':reason})
    maintenance_ok=all(x['exact_hash_semantic_gate_pass'] for x in maintenance)
    recent_ceiling=recent['accepted_initial_scenes']>=17
    quality_gates={'all_six_initial_memory_semantic_pass':initial['all_six_exact_hash_semantic_pass'],'all_used_maintenance_memory_semantic_pass':maintenance_ok,'memory_at_least17_initial_accepted':memory['accepted_initial_scenes']>=17,'memory_no_more_than1_scene_below_full':memory['accepted_initial_scenes']>=full['accepted_initial_scenes']-1,'memory_at_least_full_on5roots_and_remaining_loss_at_most1':sum(d>=0 for d in diffs_full.values())>=5 and min(diffs_full.values())>=-1,'memory_vs_recent_operational_threshold':memory['accepted_initial_scenes']>=recent['accepted_initial_scenes'] if recent_ceiling else memory['accepted_initial_scenes']>=recent['accepted_initial_scenes']+2 and sum(d>0 for d in diffs_recent.values())>=2 and sum(d<0 for d in diffs_recent.values())<=1,'memory_linkage_scene_count_at_least_full_minus1':memory['consistent_old_history_linkage_scene_count']>=full['consistent_old_history_linkage_scene_count']-1}
    cost_eligible=(memory['all18_scenes_attempted_and_reviewed'] and full['all18_scenes_attempted_and_reviewed'] and memory['all_branches_reached_scene3'] and full['all_branches_reached_scene3'] and memory['api_cost']['all_recorded_api_calls_have_usage_and_timing'] and full['api_cost']['all_recorded_api_calls_have_usage_and_timing'] and initial['all_six_actual_cost_complete'])
    mt=memory['api_cost']['known_total_tokens']+initial['known_input_tokens']+initial['known_output_tokens'];ft=full['api_cost']['known_total_tokens'];mw=memory['api_cost']['sum_call_wall_seconds']+initial['known_model_wall_seconds'];fw=full['api_cost']['sum_call_wall_seconds'];tr=ratio(mt,ft) if cost_eligible else None;wr=ratio(mw,fw) if cost_eligible else None
    cost_gates={'both_arms_all18_attempted_and_reviewed':memory['all18_scenes_attempted_and_reviewed'] and full['all18_scenes_attempted_and_reviewed'],'both_arms_every_branch_reached_scene3':memory['all_branches_reached_scene3'] and full['all_branches_reached_scene3'],'all_comparison_api_cost_recorded':memory['api_cost']['all_recorded_api_calls_have_usage_and_timing'] and full['api_cost']['all_recorded_api_calls_have_usage_and_timing'],'initial_memory_actual_cost_recorded':initial['all_six_actual_cost_complete'],'complete_cost_comparison_available':cost_eligible,'token_ratio_at_most_0_8':tr is not None and tr<=.8,'wall_ratio_at_most_1_1':wr is not None and wr<=1.1}
    adopt=all(quality_gates.values()) and all(cost_gates.values())
    return {'schema_version':'day2_memory_pilot_scores_v1','created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'COMPLETE_FIXED_DENOMINATOR' if all(r['receipt_binding'] is not None for r in rows) else 'INCOMPLETE_FIXED_DENOMINATOR','planned_root_count':6,'planned_scene_count':54,'independent_unit':'root, n=6; repeated scenes are not independent roots','arms':arms,'initial_memory':initial,'maintenance_memory_gates':maintenance,'root_deltas':{'memory_minus_full':diffs_full,'memory_minus_recent':diffs_recent},'quality_gates':quality_gates,'quality_gates_all_pass':all(quality_gates.values()),'recent_arm_ceiling_case':recent_ceiling,'cost_gates':cost_gates,'cost_status':'COST_COMPARISON_AVAILABLE' if cost_eligible else 'NO_COST_CONCLUSION','memory_over_full_token_ratio':tr,'memory_over_full_call_wall_ratio':wr,'memory_adopted':adopt,'selected_context_policy':MEM if adopt else 'full_history','fallback_reason':'Every semantic, quality, linkage and complete-cost gate must pass; otherwise frozen default full_history.','no_causal_claim':'Operational six-root pilot only. No significance, general population, selfplay effectiveness, feedback-training effect or model-weakness causal claim.','receipts':rows}

def report(s):
    lines=['# Day2 三种上下文方式的小样本扩充结果','', '固定6个Train root，每种上下文18场景，共54个计划场景。未完成、停止分支和失败均留在分母；主指标为每root接受场景数/3后等权平均。','', '| 上下文 | 初审通过/18 | root等权质量 | 完整通过分支/6 | 有一致旧历史联系场景/18 |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        a=s['arms'][arm];lines.append(f"| {arm} | {a['accepted_initial_scenes']}/18 | {a['root_equal_quality']:.2%} | {a['completed_accepted_branches']}/6 | {a['consistent_old_history_linkage_scene_count']}/18 |")
    lines+=['','## 冻结门槛','']
    for k,v in s['quality_gates'].items():lines.append(f'- {k}: '+('PASS' if v else 'FAIL/UNKNOWN'))
    lines+=['',f"费用结论：{s['cost_status']}。初始记忆实际usage/独立模型时延不可用时，完整成本比较必为NO_COST_CONCLUSION，不把提前停止产生的少用量说成效率。",'','| 上下文 | API尝试 | 已知输入tokens | 已知输出tokens | 已知调用wall总秒 |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        c=s['arms'][arm]['api_cost'];lines.append(f"| {arm} | {c['api_attempt_count']} | {c['known_input_tokens']} | {c['known_output_tokens']} | {c['sum_call_wall_seconds']:.3f} |")
    lines+=['','这些费用是明确已知的API记录部分，含失败尝试、复写和维护调用；调用wall可能并发重叠，不等于全实验耗时。初始记忆的tokenizer估算在SCORES.json独立保留，不混作真实provider usage。','',f"是否采用memory：{'是' if s['memory_adopted'] else '否'}；当前默认上下文：{s['selected_context_policy']}。",'','## root明细','', '| root | recent2通过 | memory通过 | full通过 | memory−full | memory−recent |','|---|---:|---:|---:|---:|---:|']
    for root in s['root_deltas']['memory_minus_full']:
        counts=[next(x['accepted_initial_scenes'] for x in s['arms'][arm]['root_details'] if x['root_id']==root) for arm in ARMS];lines.append('| '+root+' | '+' | '.join(map(str,counts))+f" | {s['root_deltas']['memory_minus_full'][root]} | {s['root_deltas']['memory_minus_recent'][root]} |")
    lines+=['','recent2若已通过至少17场景，按冻结方案报告天花板，不声称已证明memory提升质量。此为6root运行选择，不作统计显著性、总体泛化、selfplay/反馈训练收益或排序因果机制结论。完整54个receipt、费用记录、缺失与逐门槛结果见SCORES.json。','']
    return chr(10).join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);a=p.parse_args();w=Path(a.work).resolve();protocol_path=w.parent/'day2/fact_memory/PILOT_PROTOCOL.json';protocol=load(protocol_path);s=score(w,protocol);s['original_protocol_binding']=bind(protocol_path)
    for target in (w/'pilot/SCORES.json',w/'pilot/REPORT_ZH.md'):assert not target.exists(),f'OUTPUT_ALREADY_EXISTS: {target}'
    (w/'pilot').mkdir(exist_ok=True)
    with (w/'pilot/SCORES.json').open('x') as f:json.dump(s,f,ensure_ascii=False,indent=2);f.write(chr(10))
    with (w/'pilot/REPORT_ZH.md').open('x') as f:f.write(report(s))
    print(json.dumps({'status':s['status'],'accepted':{a:s['arms'][a]['accepted_initial_scenes'] for a in ARMS},'cost_status':s['cost_status'],'memory_adopted':s['memory_adopted']},ensure_ascii=False))
if __name__=='__main__':main()
