import json,hashlib
from pathlib import Path
A=Path(__file__).parent;W=A.parent;F=W/'f_reason_selection_analysis_20261025_v1';G=W/'g_solver_error_analysis_20261025_v1';E=W/'feedback_closed_loop_20261024_v1'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def ck(name,x):
 checks.append(dict(name=name,passed=bool(x)))
 assert x,name
for b in rd(A/'INPUT_BINDINGS.json'):ck('input '+b['path'],sha(Path(b['path']))==b['sha256'])
for base in [E,F,G]:
 m=rd(base/'FINAL_MANIFEST.json')
 for b in m['files']:ck('sealed '+str(base/b['path']),sha(base/b['path'])==b['sha256'])
records=[]
for arm in ['F','G']:
 for root in ['218519','222937','225575','229686']:
  P=rd((F if arm=='F' else A)/'reviews'/root/'PACKET.json');R=rd((F if arm=='F' else A)/'reviews'/root/'REVIEW.json')
  T={t['turn_ref']:t['text'] for t in rd(Path(P['source_binding']['path']))['turns']};by={r['candidate_id']:r for r in R['records']}
  rows=[x for x in P['records'] if x['candidate']['question_type']=='multi_select'];ck(arm+root+' five MC',len(rows)==5)
  for row in rows:
   c=row['candidate'];rev={x['request_id']:x for x in by[c['candidate_id']]['responses']};ck(c['candidate_id']+' two responses',len(rev)==2)
   for x in row['responses']:
    r=rev[x['request_id']];st=x['score']['strict'];p=x['presentation'];labels=set();ck(x['request_id']+' raw',sha(Path(x['raw_binding']['path']))==x['raw_binding']['sha256'])
    for i,ev in enumerate(r['evidence']):
     ck(x['request_id']+f' quote{i}',bool(ev['raw_quote']) and ev['raw_quote'] in x['raw_output'])
     ck(x['request_id']+f' labels{i}',[p['canonical_to_display'][k] for k in ev['canonical_labels']]==ev['display_labels'])
     for sc in ev['source_checks']:ck(x['request_id']+f' source{i} '+sc['turn_ref'],bool(sc['source_quote']) and sc['source_quote'] in T[sc['turn_ref']])
     if ev['type']=='L1':labels.update(ev['canonical_labels'])
    ck(x['request_id']+' L1 supported',bool(labels)==r['l1_local_correct_reason_wrong_decision'])
    gold=set(c['gold']);pred=set(st['prediction'] or []);affected=sorted(k for k in labels if (k in gold)!=(k in pred))
    judgment={p['display_to_canonical'][k]:v['judgment']=='supported' for k,v in x['score']['parsed']['item_evidence'].items()}
    mechanical_l2=any(judgment[k]==(k in gold) and (k in pred)!=(k in gold) for k in judgment)
    records.append(dict(arm=arm,root_id=root,candidate_id=c['candidate_id'],request_id=x['request_id'],slot_id=c['slot_id'],exact=st['exact'],valid=st['valid'],l1_labels=sorted(labels),l1_final_error_labels=affected,l1_affects_final=bool(affected),l1=r['l1_local_correct_reason_wrong_decision'],l2=r['l2_correct_judgment_wrong_box'],l3=r['l3_whole_visible_solution_correct_wrong_box'],mechanical_l2=mechanical_l2,uncertain=r['uncertain'],rationale=r['rationale']))
def stats(rows):
 affected=[x for x in rows if x['l1_affects_final']];l1=[x for x in rows if x['l1']];n=len(rows);wrong=sum(not x['exact'] for x in rows);qs={x['candidate_id'] for x in rows};aq={x['candidate_id'] for x in affected}
 return dict(responses=n,semantic_questions=len(qs),correct=sum(x['exact'] for x in rows),wrong=wrong,L1_responses=len(l1),L1_final_wrong_responses=len(affected),L1_final_wrong_semantic_questions=len(aq),L1_final_wrong_both_displays=sum(sum(x['candidate_id']==q and x['l1_affects_final'] for x in rows)==2 for q in qs),rate_all_responses=len(affected)/n if n else None,rate_wrong_responses=len(affected)/wrong if wrong else None,rate_questions_any_display=len(aq)/len(qs) if qs else None,L2_responses=sum(x['l2'] for x in rows),L3_responses=sum(x['l3'] for x in rows),mechanical_gold_correct_judgment_wrong_box=sum(x['mechanical_l2'] for x in rows),affected_request_ids=[x['request_id'] for x in affected])
summary=dict(schema_version='fg_reason_selection_same_criterion_summary_v1',analysis_date='2026-10-25',new_solver_calls=0,training=False,main_definition='决定性局部理由正确，但同一选项在最终box仍选错；不等于整题或内部思考正确',by_arm={},sensitivity=dict(case='CL1-G-218519-MC5__perm_00__FULL__THINK_OFF__CL1',primary_counted=False,reason='正确替代目的avoid guess未明确否定already knows；保守排除，仅宽语义口径敏感性纳入。',G_final_wrong_responses_if_included=6,G_rate_all_noncontrol_MC=6/32,G_rate_wrong_noncontrol_MC=6/25,G_semantic_questions_if_included=5,G_rate_questions_any_display=5/16),scope_limits=['事后非盲、同模型家族审核。','F/G每组4个Train roots；两排列是同题重复测量，不是独立样本。','主比较只含Gold2—5的多选；本轮G不重审排序，不能外推为所有题型的完整推理正确率。','仅描述本样本，不能据比例差证明反馈因果收益或总体显著差异。'])
for arm in ['F','G']:
 rows=[x for x in records if x['arm']==arm]
 summary['by_arm'][arm]=dict(noncontrol_MC=stats([x for x in rows if x['slot_id']!='MC6']),all_MC_including_control=stats(rows),control_MC6=stats([x for x in rows if x['slot_id']=='MC6']),by_root={root:stats([x for x in rows if x['root_id']==root and x['slot_id']!='MC6']) for root in ['218519','222937','225575','229686']})
 ck(arm+' 40 MC responses',len(rows)==40)
 ck(arm+' 32 noncontrol',summary['by_arm'][arm]['noncontrol_MC']['responses']==32)
 ck(arm+' 25 MC wrong',summary['by_arm'][arm]['noncontrol_MC']['wrong']==25)
 ck(arm+' L2 zero',summary['by_arm'][arm]['all_MC_including_control']['L2_responses']==summary['by_arm'][arm]['all_MC_including_control']['mechanical_gold_correct_judgment_wrong_box']==0)
ck('F primary unchanged',summary['by_arm']['F']['noncontrol_MC']['L1_final_wrong_responses']==rd(F/'SUMMARY.json')['L1_same_item_wrong_in_final']['responses'])
ck('G primary five',summary['by_arm']['G']['noncontrol_MC']['L1_final_wrong_responses']==5)
summary['observed_difference_percentage_points']=100*(summary['by_arm']['F']['noncontrol_MC']['rate_all_responses']-summary['by_arm']['G']['noncontrol_MC']['rate_all_responses'])
(A/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(A/'RESPONSE_INDEX.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
(A/'VALIDATION.json').write_text(json.dumps(dict(status='PASS',check_count=len(checks),checks=checks),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='by_arm'},ensure_ascii=False,indent=2))
for arm in ['F','G']:print(arm,json.dumps(summary['by_arm'][arm]['noncontrol_MC'],ensure_ascii=False))
