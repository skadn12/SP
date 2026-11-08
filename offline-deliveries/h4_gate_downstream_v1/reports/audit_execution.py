"""Offline execution audit. Writes reports only, never run/source/code."""
from pathlib import Path
from collections import Counter
import hashlib,json
D=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_v1'); R=D/'prepared_run'
ROOTS=tuple('cc-train-episode-'+x for x in ('259192','96462','112812','218519'))
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def prompt(base):
 return {**read(base.with_name(base.name+'_manifest.json')),'prompt':base.with_suffix('.txt').read_text()}

def cost(rows):
 x=dict(logical_calls=len(rows),http_attempts=sum(r['http_attempt_count'] for r in rows),retries=sum(r['retry_count'] for r in rows),verified_cost='UNKNOWN')
 for key in ('input_tokens','output_tokens','total_tokens','reasoning_tokens'):
  vs=[r.get(key) for r in rows];x[key]=sum(vs) if all(type(v)is int for v in vs) else 'UNKNOWN'
 return x

def main():
 if not (R/'generation_summary.json').exists(): print('WAITING_FOR_GENERATION_SUMMARY');return
 summary=read(R/'generation_summary.json');m=read(R/'freeze_manifest.json');old=Path(m['source_run']);g=Path(m['gate_run'])
 checks=[];freezes={};sources=[];locks=[];details=[]
 def ck(name,ok,detail=None):checks.append(dict(name=name,status='PASS' if ok else 'FAIL',detail=detail))
 for label,p,expected in [('current',R,358),('previous_gate',g,229),('previous_planner',old,293)]:
  fs=read(p/'freeze_manifest.json')['files'];changed=[x['path'] for x in fs if not Path(x['path']).exists() or sha(x['path'])!=x['sha256']]
  freezes[label]=dict(count=len(fs),expected=expected,changed=changed);ck(label+'_freeze',not changed and len(fs)==expected,freezes[label])
 exp=[];allpaths=[]
 for root in ROOTS[:3]:
  p=R/'expansion/execution'/root;x=read(p/'development_call_ledger.json');exp+=x['records'];allpaths += [(r,p,'expansion') for r in x['records']]
  ck('exp_ledger_'+root,x['logical_call_count']==len(x['records'])<=6 and x['http_attempt_count']==sum(z['http_attempt_count'] for z in x['records'])<=6 and x['automatic_retry_count']==0)
 p=R/'qa/execution';x=read(p/'qa_call_ledger.json');qa=x['records'];allpaths += [(r,p,'qa') for r in qa]
 ck('qa_ledger',x['logical_call_count']==len(qa)<=21 and x['http_attempt_count']==sum(z['http_attempt_count'] for z in qa)<=21 and x['automatic_retry_count']==0)
 calls=exp+qa;ck('complete_record_identity',read(R/'complete_call_records.json')==dict(expansion=exp,qa=qa))
 for k,rows in [('cost',calls),('expansion_cost',exp),('qa_cost',qa)]:
  c=cost(rows);ck(k+'_summary_identity',all(summary[k].get(a)==v for a,v in c.items()),c)
 est={'outline','outline_supplement','realizer','adherence','continuation','reaudit'};qst={'miner','writer_primary','writer_repair'}
 ck('caps_allowlist',len(calls)<=39 and len(exp)<=18 and len(qa)<=21 and all(z['root_id'] in ROOTS[:3] for z in calls) and all(z['stage'] in est for z in exp) and all(z['stage'] in qst for z in qa))
 ck('no_new_planner_gate',not any(z['stage'] in ('planner','gate','planner_supplement','gate_supplement') for z in calls))
 ck('no_retries_or_timeouts',all(z['http_attempt_count']==1 and z['retry_count']==0 and z.get('timeout') is False and z['status']=='COMPLETED' for z in calls))
 budgets=dict(outline=8192,outline_supplement=8192,realizer=8192,adherence=4096,continuation=4096,reaudit=4096,miner=4096,writer_primary=8192,writer_repair=8192)
 for row,p,kind in allpaths:
  t=p/row['artifact_directory'];req=read(t/'request.json');meta=read(t/'response_metadata.json');pay=req['payload'];raw=meta['raw_response']
  params=dict(temperature=.25 if kind=='expansion' else .1,max_tokens=budgets[row['stage']],response_format={'type':'json_object'})
  ck('request_'+row['root_id']+'_'+row['stage'],pay==dict(model='tju-llm',messages=pay['messages'],**params) and req['requested_parameters']==params and req['confirmed_parameter_adoption']=='UNKNOWN' and req['endpoint']=='https://ai.tju.edu.cn/api/v3/chat/completions' and digest(pay)==row['payload_sha256'] and digest(pay['messages'])==row['messages_sha256'])
  ck('response_usage_'+row['root_id']+'_'+row['stage'],raw['usage']==meta['usage']==row['usage'] and raw['model']==meta['actual_model']==row['actual_model']=='qwen3.6-35b-a3b' and raw['choices'][0]['finish_reason']==meta['finish_reason']==row['finish_reason']=='stop' and row['http_status']==meta['http_status']==200 and raw['choices'][0]['message']['content']==(t/'response.txt').read_text() and meta['transport_attempt_count']==meta['http_attempt_count']==1 and meta['transport_retry_count']==meta['rate_limit_retry_count']==0 and meta['usage'].get('prompt_tokens')==row['input_tokens'] and meta['usage'].get('completion_tokens')==row['output_tokens'] and meta['usage'].get('total_tokens')==row['total_tokens'])
  details.append({k:row.get(k) for k in ('root_id','stage','request_id','intent_id','status','actual_model','finish_reason','usage','input_tokens','output_tokens','reasoning_tokens','total_tokens','http_attempt_count','retry_count','timeout')})
 for root in ROOTS:
  sv=read(R/'frozen'/root/'saved_planning.json');t=R/'expansion/roots'/root
  ck('input_'+root,read(R/'frozen'/root/'inputs.json')==read(old/'frozen'/root/'inputs.json'))
  for slot,gs,ps,suf in [('primary','gate','planner',''),('supplement','gate_supplement','planner_supplement','_supplement')]:
   f=g/'frozen'/root/gs;o=g/'results'/root/gs
   paths=dict(planner_prompt=old/'expansion/execution'/root/'calls'/root/ps/'prompt.json',planner_raw=f/'planner_output.json',planner_validation=f/'planner_validation.json',gate_context=f/'context.json',gate_prompt=f/'prompt.json',gate_output=o/'output.json',gate_validation=o/'validation.json')
   mismatch=[k for k,p in paths.items() if sv[slot][k]!=read(p)];sources.append(dict(root=root,slot=slot,mismatch=mismatch));ck('saved_source_'+root+'_'+slot,not mismatch)
   pout=t/'planner'/('output.json' if slot=='primary' else 'supplement_output.json')
   if pout.exists():
    vals=[('planner_raw',pout),('planner_validation',t/'planner'/('validation.json' if slot=='primary' else 'supplement_validation.json')),('gate_context',t/'semantic_gate'/('context'+suf+'.json')),('gate_output',t/'semantic_gate'/('output'+suf+'.json')),('gate_validation',t/'semantic_gate'/('validation'+suf+'.json'))]
    ck('imports_'+root+'_'+slot,all(sv[slot][k]==read(p) for k,p in vals))
    ck('gate_prompt_'+root+'_'+slot,prompt(t/'semantic_gate'/('prompt'+suf))==sv[slot]['gate_prompt'])
    ck('planner_prompt_'+root+'_'+slot,prompt(t/'planner'/('prompt' if slot=='primary' else 'supplement_prompt'))==sv[slot]['planner_prompt'])
  sp=t/'planner/supplement_import_audit.json'
  if sp.exists():
   selected=read(t/'planner/outline_semantic_selection.json')['selected_candidate_refs'];sa=read(sp);ga=read(t/'semantic_gate/import_audit_supplement.json');op=t/'planner/outline_semantic_prompt_supplement.txt';ol=None
   if op.exists():
    tail=op.read_text().split('\n\nCANDIDATE RECOVERY AND ACTOR CONTRACT\n',1)[1];ol=json.loads(tail[tail.index('{'):])['locked_candidate_refs']
   item=dict(root=root,current_locked=selected,historical_locked=sv['supplement']['planner_prompt']['qualified_candidate_refs'],outline_locked=ol);locks.append(item)
   ck('current_locks_'+root,sa['current_locked_candidate_refs']==selected and ga['current_locked_candidate_refs']==selected and (ol is None or ol==selected),item)
  else:locks.append(dict(root=root,supplement_imported=False))
  lp=t/'local_memory_accounting.json'
  if lp.exists():
   a=read(lp);ck('local_memory_'+root,a['remote_api_calls']==0 and a['writes']==0,a)
 ck('zero_call_shortage_root',not any(z['root_id']==ROOTS[3] for z in calls),read(R/'expansion/roots'/ROOTS[3]/'outcome.json'))
 per={r:dict(expansion=cost([z for z in exp if z['root_id']==r]),qa=cost([z for z in qa if z['root_id']==r]),all=cost([z for z in calls if z['root_id']==r])) for r in ROOTS}
 result=dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',audit_kind='CODEX_OFFLINE_EXECUTION_COST_NOT_SEMANTIC_CERTIFICATION',new_api_calls=0,checks=checks,freeze=freezes,per_root=per,stage_counts=dict(Counter(z['stage'] for z in calls)),cost=cost(calls),previous_gate_cost_separate=summary['previous_gate_cost_separate'],actual_models=sorted(set(z['actual_model'] for z in calls)),finish_reasons=sorted(set(z['finish_reason'] for z in calls)),source_identity=sources,selection_locks=locks,calls=details,service_capacity='UNKNOWN',parameter_adoption='UNKNOWN',verified_money='UNKNOWN')
 (D/'reports/EXECUTION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 c=result['cost'];lines=['# 执行身份与成本审计','','**'+result['status']+'**。只读离线核对，新增API调用0。','',f"本轮{c['logical_calls']}次逻辑调用＝{c['http_attempts']}次HTTP，上限39；重试{c['retries']}。",f"输入{c['input_tokens']}＋输出{c['output_tokens']}＝总{c['total_tokens']} tokens；推理细分和可核实金额UNKNOWN。",'','| Root | 扩充调用 | QA调用 | 总tokens |','|---|---:|---:|---:|']
 for r,x in per.items():lines.append(f"| {r} | {x['expansion']['http_attempts']} | {x['qa']['http_attempts']} | {x['all']['total_tokens']} |")
 lines+=['','冻结核对：'+'；'.join(f"{k} {v['count']}/{v['expected']}，变化{len(v['changed'])}" for k,v in freezes.items())+'。','','原Planner输出、validation、实际prompt和新Gate输入输出均与保存来源一致。导入不是新增模型调用。补充仅在本次初次Outline不足时使用；锁定来自本次初次选择，原历史上下文保留。','','完整usage、实际模型和finish_reason与原始服务响应逐条交叉核对。请求参数核对不等于服务采用认证。','','本审计不认证场景或题目语义质量。前一轮Gate成本单列，未当成本轮新调用重复相加。','','失败检查：']
 failures=[x for x in checks if x['status']!='PASS'];lines+=['- '+json.dumps(x,ensure_ascii=False) for x in failures] if failures else ['无。']
 (D/'reports/EXECUTION_AUDIT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:result[k] for k in ('status','cost','stage_counts','freeze')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
