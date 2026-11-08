import json,hashlib,re,stat,datetime
from pathlib import Path
D=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/question_form_only_fg_pilot_20261026_v1/question_authoring/F')
read=lambda p:json.loads(Path(p).read_text())
I=read(D/'AUTHOR_INPUT.json');P={p['candidate_id']:p for p in read(D/'TASK_PLANS_FIXED.json')};C=read(D/'CANDIDATES_DRAFT.json');CT={p['candidate_id']:p for p in read(D/'PLAN_CONTINUATION_RECEIPT.json')['continuations']}
S={};T={};W={}
for path in I['allowed_read_files']:
 if '/sources/' in path:
  x=read(path);n=x['root_id'];S[n]=x;T[n]={t['turn_ref']:t for t in x['turns']};W[n]=read(D/f"SOURCE_WARNINGS_{n.split('-')[-1]}.json")['warnings']
checks=[]
def check(name,ok,detail=None):
 checks.append({'check':name,'pass':bool(ok),'detail':detail})
ids=[c['candidate_id'] for c in C]
expected={f'CL1-F-{n}-MC{k}' for n in ['218519','222937','225575','229686'] for k in range(2,7)}
check('exactly_20_unique_required_candidates',len(C)==20 and len(set(ids))==20 and set(ids)==expected)
check('all_fixed_plans_present',expected<=P.keys())
feedback_keys=['feedback_id','trigger_realization','predicted_observable_error','counterexample','target_acceptance_test']
word_stats=[]
count_hints=[]
for c in C:
 cid=c['candidate_id'];p=P[cid];turns=T[c['root_id']];opts=c['options'];a=c['item_analysis'];refs={e['turn_ref'] for z in a.values() for e in z['evidence']}
 check(cid+':schema_and_quota',c['question_type']=='multi_select' and c['arm']=='F' and set(opts)==set('ABCDEF') and set(a)==set('ABCDEF') and len(c['gold'])==int(c['slot_id'][2:]) and c['gold']==[k for k in 'ABCDEF' if a[k]['should_select']])
 check(cid+':fixed_plan_bindings',c['task_operation']==p['task_operation'] and c['source_binding']==p['source_binding'] and c['preserved_source_relations']==p['source_relations'] and c['preserved_plan_candidate_id']==cid and c['feedback_target']=={k:p[k] for k in feedback_keys} and c['hypothesis_status']==p['hypothesis_status'])
 check(cid+':continuation_fields',c['form_version']=='question_form_only_v1' and c['form_task_description']==CT[cid]['form_task_description'] and c['form_change_rationale']==CT[cid]['form_change_reason'])
 check(cid+':status_and_warnings_unchanged',c['source_status']==S[c['root_id']]['source_status'] and c['warnings']==W[c['root_id']])
 check(cid+':planned_source_relations_cited',set(p['source_relations'])<=refs,sorted(set(p['source_relations'])-refs))
 check(cid+':r03_present_and_removal_explained',any(r.startswith('R03:') for r in refs) and bool(c['r03_necessity']))
 check(cid+':english_public_text',not re.search('[\u3400-\u9fff]',c['question']+' '.join(opts.values())))
 b=Path(c['source_binding']['path']).read_bytes()
 check(cid+':source_hash_and_bytes',hashlib.sha256(b).hexdigest()==c['source_binding']['sha256'] and len(b)==c['source_binding']['bytes'])
 for lab in 'ABCDEF':
  z=a[lab];valid=bool(z['evidence']) and all(e['turn_ref'] in turns and e['speaker']==turns[e['turn_ref']]['speaker'] and e['quote']==turns[e['turn_ref']]['text'] for e in z['evidence'])
  check(cid+':'+lab+':full_verbatim_source_evidence',valid)
  check(cid+':'+lab+':analysis_complete',isinstance(z['should_select'],bool) and bool(z['rationale']) and (z['should_select'] or bool(z['changed_relation'])))
  if c['slot_id']=='MC4':
   g=z['given_fragment'];check(cid+':'+lab+':single_literal_fragment',len(z['evidence'])==1 and g['turn_ref']==z['evidence'][0]['turn_ref'] and g['speaker']==turns[g['turn_ref']]['speaker'] and g['quote'] in turns[g['turn_ref']]['text'] and g['quote'] in opts[lab] and bool(z['attached_claim']) and 'single' in z['judgment_scope'].lower())
 if c['slot_id']=='MC4':check(cid+':explicit_no_pooling_scope','single' in c['evidence_selection_scope'] and any(v in c['question'].lower() for v in ['do not combine','do not pool','without combining','cannot fill']))
 if c['slot_id']=='MC6':check(cid+':private_all_true_control',c['gold']==list('ABCDEF') and c['private_control_type']=='all_true_control; does not establish distractor quality')
 wc=lambda t:len(t.split())
 word_stats.append({'candidate_id':cid,'stem_words':wc(c['question']),'option_words':{lab:wc(opts[lab]) for lab in 'ABCDEF'}})
 if re.search(r'\b(all six|exactly\s+(two|three|four|five|six)|select\s+(two|three|four|five|six))\b',c['question'],re.I):count_hints.append(cid)
check('no_explicit_answer_count_hints_in_stems',not count_hints,count_hints)
M=read(D/'RECEIPTS_FREEZE_MANIFEST.json')
for binding in M['bindings']:
 path=Path(binding['path']);b=path.read_bytes()
 check(path.name+':frozen_hash_and_size',hashlib.sha256(b).hexdigest()==binding['sha256'] and len(b)==binding['bytes'])
 check(path.name+':read_only_mode',not bool(path.stat().st_mode & 0o222))
receipt=read(D/'AUTHOR_READ_RECEIPT.json')
report={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local mechanical checks only; not an independent semantic review and not production_question_gate.','candidate_count':len(C),'by_root':{n:sum(c['root_id']==n for c in C) for n in S},'source_turn_counts':{n:len(x['turns']) for n,x in S.items()},'source_turn_total':sum(len(x['turns']) for x in S.values()),'check_count':len(checks),'pass_count':sum(z['pass'] for z in checks),'failure_count':sum(not z['pass'] for z in checks),'checks':checks,'word_counts':word_stats,'stem_target_30_90_violations':[x for x in word_stats if not 30<=x['stem_words']<=90],'option_target_10_35_violations':[{'candidate_id':x['candidate_id'],'option':lab,'words':v} for x in word_stats for lab,v in x['option_words'].items() if not 10<=v<=35],'length_note':'MC4 option lengths include source coordinates, verbatim evidence and attached claim; approximate contract guidance is reported without substantive redrafting.','production_gate_run':False,'independent_semantic_review_run':False}
(D/'MECHANICAL_CHECKS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'candidates':len(C),'checks':report['check_count'],'passed':report['pass_count'],'failed':report['failure_count'],'failures':[z for z in checks if not z['pass']],'stem_length_violations':report['stem_target_30_90_violations'],'option_length_violations':report['option_target_10_35_violations'],'receipt_top_keys':list(receipt)},ensure_ascii=False,indent=2))
# Additional receipt coverage checks; no candidate changes.
for z in receipt['sources']:
 x=S[z['root_id']];bb=Path(z['source_binding']['path']).read_bytes();wb=Path(z['warnings_binding']['path']).read_bytes()
 check(z['root_id']+':receipt_complete_refs',z['complete_turn_count']==len(x['turns']) and z['complete_turn_refs']==[t['turn_ref'] for t in x['turns']] and z['source_status']==x['source_status'])
 check(z['root_id']+':receipt_source_binding',len(bb)==z['source_binding']['bytes'] and hashlib.sha256(bb).hexdigest()==z['source_binding']['sha256'])
 check(z['root_id']+':receipt_warning_binding',len(wb)==z['warnings_binding']['bytes'] and hashlib.sha256(wb).hexdigest()==z['warnings_binding']['sha256'])
for z in receipt['allowed_input_bindings']:
 bb=Path(z['path']).read_bytes();check(Path(z['path']).name+':receipt_allowed_input_binding',len(bb)==z['bytes'] and hashlib.sha256(bb).hexdigest()==z['sha256'])
report.update({'checks':checks,'check_count':len(checks),'pass_count':sum(z['pass'] for z in checks),'failure_count':sum(not z['pass'] for z in checks)})
(D/'MECHANICAL_CHECKS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'final_checks':len(checks),'passed':report['pass_count'],'failed':report['failure_count'],'failures':[z for z in checks if not z['pass']]},ensure_ascii=False))
