#!/usr/bin/env python3
"""Final artifact and arithmetic audit; no runtime mutation or new model calls."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,statistics
from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parents[1];R=H/'runtime/eval_v1'
checks=[]
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def ck(ok,name,detail=''):checks.append({'name':name,'pass':bool(ok),'detail':str(detail)})
def near(a,b):return isinstance(a,(int,float)) and abs(a-b)<1e-12
def vb(b,base=H):
 p=Path(b['path']);p=p if p.is_absolute() else base/p
 ck(p.is_file(),'binding_exists',p)
 if p.is_file():ck(sha(p)==b['sha256'],'binding_sha256',p);ck(p.stat().st_size==b['bytes'],'binding_bytes',p)
def recursive_bindings(obj):
 if isinstance(obj,dict):
  if {'path','sha256','bytes'}<=set(obj):vb(obj)
  else:
   for x in obj.values():recursive_bindings(x)
 elif isinstance(obj,list):
  for x in obj:recursive_bindings(x)

summary=read(H/'SUMMARY.json');feedback=read(H/'FEEDBACK_SNAPSHOT.json');ledger=read(H/'exports/CANDIDATE_LEDGER.json');scores=read(R/'SCORES.json');manifest=read(R/'manifest.json');gen=read(H/'GENERATION_FREEZE.json')
report=(H/'REPORT_ZH.md').read_text();readme=(H/'README_ZH.md').read_text();semantic=rows(H/'SEMANTIC_REVIEWS.jsonl');sem_by={x['candidate_id']:x for x in semantic}
old_count=0;old_reports=[]
for name,count,expected_manifest_sha in [('day3_targeted_4root_v1',971,'44f5d62179affbe5125477eb83754c6eb3f0a4700619722e1f07d45fc54992e2'),('day3_diagnostic_handoff_v1',122,'a83a7d378a21e6b21c4ad3ca3f40e3f950840417c0d6f0972ee232d5d19732b9')]:
 p=H.parent/name/'FINAL_MANIFEST.json';ck(sha(p)==expected_manifest_sha,'prior_manifest_expected_hash',name);m=read(p);ck(len(m['files'])==count,'prior_manifest_file_count',name)
 for b in m['files']:vb(b)
 old_count+=count;old_reports.append({'directory':name,'files':count,'manifest':bind(p)})
ck(old_count==summary['old_sealed_files_verified']==1093,'1093_old_files_preserved')
ck(sha(R/'SCORES.json')=='dee012c06e525fcbefdf240b9c1e22ccf6faa6af4bca9ce43cf203cf6460edec','original_scores_unchanged')
post=read(H/'review/POSTRUN_RECOMPUTATION_VERIFIED.json');ck(post['status']=='PASS' and post['checks']==1734 and not post['errors'],'root_independent_arithmetic_recheck_pass')
for b in post['audited_files']:vb(b)
public={q['candidate_id']:q for q in read(R/'PUBLIC_QUESTIONS.json')['questions']};private={q['candidate_id']:q for q in read(H/'exports/PRIVATE_QUESTIONS.json')['questions']}
lc={q['candidate_id']:q for q in ledger['questions']};accepted={x['candidate_id'] for x in ledger['questions'] if x['content_disposition']=='ACCEPT'};held={x['candidate_id'] for x in ledger['questions'] if x['content_disposition']=='HOLD'}
ck(len(lc)==len(ledger['questions'])==12 and len(accepted)==11 and held=={'FBR-R4-MC2'},'12_slots_11_accept_1_hold')
ck(accepted==set(public)==set(private),'public_private_accepted_scope')
ck(all(not x['training_admitted'] for x in lc.values()),'no_question_training_admitted')
gc=Counter(len(q['gold']) for q in private.values() if q['question_type']=='multi_select')
ck(gc=={2:2,3:3,4:2} and summary['accepted_MC_gold_counts']=={'2':2,'3':3,'4':2,'5':0},'Gold5_not_admitted')
ck(summary['planned_slots']==summary['generated_candidates']==12 and summary['content_accepted']==11 and summary['content_hold']==1,'summary_candidate_counts')
ck(summary['new_roots']==summary['new_source_turns']==summary['source_shortfall']==0,'no_expansion_or_root_in_this_iteration')
maps=read(R/'DISPLAY_MAPPINGS.json');req_by={r['request_id']:r for r in manifest['requests']};direct_by={r['request_id']:r for r in scores['direct_requests']};diag_by={r['request_id']:r for r in scores['diagnostics']}
all_raw={};token_inputs=token_outputs=0;live_wall=load_wall=0
for branch,count in [('direct',66),('evidence',11)]:
 f=read(R/branch/'RAW_RESPONSE_FREEZE.json');complete=read(R/branch/'COMPLETE.json')
 ck(f['status']=='COMMITTED' and f['planned_request_count']==count,'branch_raw_count',branch)
 ck(complete['actual_submitted_requests']==complete['returned_requests']==count,'branch_submitted_returned',branch)
 for b in f['files']+f['execution_ledger_bindings']+[f['manifest_binding']]:vb(b)
 submitted=[rid for p in sorted((R/branch/'batches').glob('*_SUBMITTED.json')) for rid in read(p)['request_ids']]
 ck(len(submitted)==len(set(submitted))==count,'each_request_once',branch)
 live_wall+=complete['wall_seconds'];load_wall+=complete['model_load_wall_seconds'];branch_input=branch_output=0
 for b in f['files']:
  raw=read(Path(b['path']));rid=raw['request_id'];all_raw[rid]=raw;row=req_by[rid]
  ck(raw['status']=='RETURNED' and raw['automatic_retry'] is False and raw['source_truncation'] is False,'returned_no_retry_truncation',rid)
  ck(raw['metadata']['finish_reason']!='length','raw_not_output_truncated',rid)
  ck(raw['seed']==row['seed'] and raw['prompt_sha256']==row['prompt_sha256'],'raw_identity_matches_frozen_request',rid)
  branch_input+=raw['metadata']['prompt_token_count'];branch_output+=raw['metadata']['completion_token_count']
 cost=summary['cost_by_branch'][branch]
 ck(cost==scores['cost_by_branch'][branch],'cost_summary_matches_score',branch)
 ck(cost['known_prompt_tokens']==branch_input and cost['known_completion_tokens']==branch_output and cost['known_total_tokens']==branch_input+branch_output,'cost_independent_raw_token_sum',branch)
 token_inputs+=branch_input;token_outputs+=branch_output
ck(len(all_raw)==summary['total_solver_calls']==77,'77_total_calls')
ck(token_inputs==760424 and token_outputs==7850 and token_inputs+token_outputs==summary['total_known_tokens']==768274,'768274_tokens')
ck(near(live_wall,summary['total_live_wall_seconds']) and round(live_wall,2)==87.22,'87_22s_live_wall_includes_load')
ck(round(load_wall,2)==33.63,'33_63s_loading_separately_explained')
recomputed={};per_question=defaultdict(list)
for rid,s in direct_by.items():
 row=req_by[rid];q=public[row['candidate_id']];raw=all_raw[rid];answer=re.fullmatch(r'<box>([^<]+)</box>',raw['raw_output']);ck(answer is not None,'direct_box_parse',rid)
 labels=answer.group(1).split(',');var=maps['requests'][rid];reverse={v:k for k,v in maps['public_to_canonical'][q['candidate_id']].items()};pred=[reverse[var['display_to_content'][x]] for x in labels];gold=private[q['candidate_id']]['gold']
 is_mc=q['question_type']=='multi_select';exact=set(pred)==set(gold) if is_mc else pred==gold
 ck(s['exact']==exact and (set(s['canonical_prediction'])==set(pred) if is_mc else s['canonical_prediction']==pred),'direct_label_domain_exact_recomputed',rid)
 if is_mc:
  fp=len(set(pred)-set(gold));fn=len(set(gold)-set(pred));j=len(set(pred)&set(gold))/len(set(pred)|set(gold))
  ck(s['metrics']['fp']==fp and s['metrics']['fn']==fn and near(s['metrics']['jaccard'],j),'MC_FP_FN_Jaccard_recomputed',rid)
  item={'exact':exact,'jaccard':j,'fp':fp,'fn':fn,'selected':len(pred),'gold':len(gold),'all_selected':len(pred)==6}
 else:
  rank={x:i for i,x in enumerate(pred)};correct=sum(rank[a]<rank[b] for i,a in enumerate(gold) for b in gold[i+1:]);total=len(gold)*(len(gold)-1)//2
  ck(s['metrics']['correct_pairs']==correct and s['metrics']['total_pairs']==total,'ordering_pairs_recomputed',rid);item={'exact':exact,'pair_correct':correct,'pair_total':total,'pair_acc':correct/total}
 per_question[q['candidate_id']].append(item)
for kind in ['multi_select','ordering']:
 subset=[q for q in public.values() if q['question_type']==kind];roots=defaultdict(list);q_rates=[];all_items=[]
 for q in subset:
  vals=per_question[q['candidate_id']];ck(len(vals)==6,'each_accepted_six_direct',q['candidate_id']);rate=statistics.mean(v['exact'] for v in vals);q_rates.append(rate);roots[q['root_id']].append(rate);all_items+=vals
 stats={'questions':len(subset),'calls':len(all_items),'exact':sum(v['exact'] for v in all_items),'question_equal_exact':statistics.mean(q_rates),'root_equal_exact':statistics.mean(statistics.mean(v) for v in roots.values())}
 s=summary['by_type'][kind];ck(s==scores['by_type'][kind],'type_summary_equals_immutable_score',kind)
 ck(s['questions']==stats['questions'] and s['direct_calls']==stats['calls'] and near(s['direct_question_equal_exact'],stats['question_equal_exact']) and near(s['direct_root_equal_exact'],stats['root_equal_exact']),'question_root_means_recomputed',kind)
 if kind=='multi_select':
  stats.update(jaccard=statistics.mean(x['jaccard'] for x in all_items),fp=sum(x['fp'] for x in all_items),fn=sum(x['fn'] for x in all_items),mean_selected=statistics.mean(x['selected'] for x in all_items),mean_gold=statistics.mean(x['gold'] for x in all_items),all_selected_calls=sum(x['all_selected'] for x in all_items))
  ck(stats['fp']==51 and stats['fn']==20 and stats['all_selected_calls']==1,'MC51FP20FN_one_full_selection')
  ck(near(stats['jaccard'],s['direct_question_equal_jaccard']),'MC_partial_mean')
 else:
  q_pairs=[sum(v['pair_correct'] for v in per_question[q['candidate_id']])/sum(v['pair_total'] for v in per_question[q['candidate_id']]) for q in subset]
  stats.update(question_equal_pair_accuracy=statistics.mean(q_pairs),pair_correct=sum(x['pair_correct'] for x in all_items),pair_total=sum(x['pair_total'] for x in all_items))
  stats['pooled_pair_accuracy']=stats['pair_correct']/stats['pair_total'];ck(stats['pair_correct']==147 and stats['pair_total']==198 and near(stats['question_equal_pair_accuracy'],s['direct_question_equal_pair_accuracy']),'pair_147of198_vs_question_mean')
 recomputed[kind]=stats
ck(recomputed['multi_select']['exact']==13 and recomputed['ordering']['exact']==2,'direct_exact13of42_2of24')
ck(len(diag_by)==len(semantic)==len(sem_by)==11 and set(sem_by)==accepted,'all_diagnostics_semantically_disposed')
valid=[s for s in diag_by.values() if s['format_valid']];invalid=[s for s in diag_by.values() if not s['format_valid']]
ck(len(valid)==summary['diagnostic_valid']==8 and len(invalid)==summary['diagnostic_invalid']==3,'8valid3invalid_retained')
ck({x['candidate_id'] for x in invalid}=={'FBR-R2-ORD1','FBR-R3-MC1','FBR-R3-MC3'},'invalid_exact_ids')
ck(Counter(x['review_status'] for x in semantic)=={'FAIL':8,'NOT_REVIEWABLE':3},'semantic8errors3not_reviewable')
for x in invalid:
 ck(x['display_prediction'] is None and x['canonical_prediction'] is None and not x['exact'] and bool(x['invalid_reasons']),'invalid_not_salvaged',x['candidate_id'])
 ck(sem_by[x['candidate_id']]['review_status']=='NOT_REVIEWABLE' and not sem_by[x['candidate_id']]['source_verified_target_failure'],'invalid_no_target_claim',x['candidate_id'])
quote_count=quote_exact=0
for x in valid:
 row=req_by[x['request_id']];raw=all_raw[x['request_id']];parsed=json.loads(raw['raw_output'].strip());ck(parsed==x['parsed_response'],'diagnostic_parsed_actual_model_text',x['request_id'])
 source=read(R/'sources'/(row['root_id']+'.json'));turns={t['turn_ref']:t['text'] for t in source['turns']};cits=[e for item in parsed['item_evidence'] for e in item['citations']];qe=sum(e['turn_ref'] in turns and e['quote'] in turns[e['turn_ref']] for e in cits)
 ck(x['citation_count']==len(cits) and x['exact_quote_count']==qe,'citation_fidelity_independent_recount',x['request_id']);quote_count+=len(cits);quote_exact+=qe
ck(quote_count==64 and quote_exact==24 and summary['total_quotes_at_valid_diagnostics']==64 and summary['exact_quotes_at_valid_diagnostics']==24,'24of64_exact_quotes')
target_failure={x['candidate_id'] for x in semantic if x['source_verified_target_failure']}
ck(len(target_failure)==summary['source_verified_target_failure_records']==7,'seven_source_verified_target_errors')
ck(all(x['training_use'] is False and not x['formal_reproduced'] for x in semantic),'semantic_not_training_or_formal')
cards=[e for e in feedback['entries'] if e['feedback_id'].startswith('T4E-')];cb={e['feedback_id']:e for e in cards}
ck(len(cards)==summary['planned_feedback_cards']==5 and sum(e['candidate_realization'] for e in cards)==summary['content_realized_feedback_cards']==4,'five_cards_four_content_realizations')
ck(sum(bool(e['source_verified_target_failure_ids']) for e in cards)==summary['cards_with_source_verified_target_failure']==3,'three_cards_supported_target_failures')
for fid,card in cb.items():
 planned={x['candidate_id'] for x in ledger['questions'] if x['feedback_id']==fid};ac=planned&accepted;vi=ac&{x['candidate_id'] for x in valid};iv=ac&{x['candidate_id'] for x in invalid}
 ck(set(card['planned_candidates'])==planned and set(card['accepted_candidates'])==ac and card['candidate_realization']==bool(ac),'card_candidates_realization',fid)
 ck(set(card['valid_diagnostic_ids'])==vi and set(card['invalid_diagnostic_ids'])==iv and set(card['source_verified_target_failure_ids'])==ac&target_failure,'card_diagnostic_target_scope',fid)
 ck(card['training_use'] is False and card['formal_reproduced'] is False,'card_not_training_or_formal',fid)
 recursive_bindings(card)
source_card=cb['T4E-SOURCE_FACT_REJECTION'];ck(source_card['status']=='CANDIDATE_HOLD_NO_SOLVER_TEST' and not source_card['candidate_realization'] and not source_card['source_verified_operational_feedback'] and not source_card['response_bindings'],'source_fact_rejection_not_closed_or_tested')
ck(cb['T4E-CONDITION_AND_ANSWER_ALIGNMENT']['status']=='NO_NEW_VALID_TARGET_FAILURE_THIS_BATCH' and not cb['T4E-CONDITION_AND_ANSWER_ALIGNMENT']['source_verified_target_failure_ids'],'condition_no_new_target_failure_not_mastered')
for key in ['all_five_targets_realized','formal_reproduced','matched_AB','feedback_causal_effect_established','training_started','feedback_loop_closed','all_targets_loop_closed','next_candidates_generated']:
 ck(summary.get(key) is False,'summary_negative_gate',key)
for key in ['all_five_targets_realized','formal_reproduced','matched_AB','feedback_causal_effect_established','all_targets_loop_closed']:
 ck(feedback.get(key) is False,'feedback_negative_gate',key)
ck(summary['training_admitted']==summary['dev_test_calls']==0 and summary['primary_evaluation']=='direct' and summary['day3_overall_complete'] is False,'no_training_devtest_full_day_claim')
plan=read(H/'next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json');receipt=read(H/'next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json');packet=read(H/'next_generation/INPUT_PACKET.json')
ck(summary['actual_next_consumer_read'] is True and summary['feedback_to_candidate_to_review_to_next_plan_executed'] is True and plan['actual_plan_consumption'] is True and receipt['actual_plan_consumption'] is True,'actual_next_consumer_plan_recorded')
for key in ['next_candidates_generated','all_targets_loop_closed','formal_reproduced','matched_feedback_AB','feedback_causal_effect_established','training_start_go']:
 ck(plan[key] is False and receipt[key] is False,'next_plan_negative_gate',key)
ck(receipt['actual_agent_consumption_not_program_packet_only'] is True and receipt['raw_api_trace'] is False and receipt['api_message_capture'] is False,'tool_visible_read_not_raw_api_claim')
ck(plan['priority_1_new_evidence_sufficiency_version']['original_disposition']=='HOLD_UNCHANGED' and plan['priority_1_new_evidence_sufficiency_version']['original_solver_calls']==0,'next_plan_preserves_hold')
ck(plan['priority_2_diagnostic_interface_small_control']['status']=='DESIGN_ONLY_NO_PROMPT_EDIT_NO_MODEL_CALL' and plan['priority_2_diagnostic_interface_small_control']['proposed_total_calls']==22,'next_interface22_only_proposed')
ck(plan['material_policy']['current_action']=='REUSE_SOURCE_NO_A3_B3_EXPANSION','plan_no_unneeded_expansion')
for obj in [packet,plan,receipt,summary.get('actual_next_plan_binding'),summary.get('actual_next_read_receipt_binding')]:recursive_bindings(obj)
# Old packet flags remain a historical preparation record; later read manifest
# records subsequent actual consumption without retroactively editing that packet.
ck(packet['next_llm_consumption'] is False and receipt['actual_plan_consumption'] is True,'prepared_packet_and_later_consumption_distinct')
for required in ['11 ACCEPT/1 HOLD','66次直接作答＋11次结构化依据诊断','Gold5为0','13/42','30.95%','18.06%','2/24','8.33%','69.58%','147/198=74.24%','51次错选','20次漏选','42次中只有1次全选','8格式有效、3无效','0截断、0重试','64引用，仅24','768274','87.22秒','33.63秒','1093','1734','1882','全目标闭环仍false','matched A/B=false']:
 ck(required in report,'report_key_numeric_or_negative_statement',required)
ck('Gold5候选被HOLD' in readme and '不能声称全目标闭环或反馈收益' in readme,'readme_limits_clear')
struct=read(H/'review/AUTHOR_STRUCTURAL_VALIDATION.json');ck(struct['status']=='PASS','structural_validation_pass');
recursive_bindings(feedback.get('semantic_reviews'));recursive_bindings(feedback.get('reviewer_bindings'))
for filename in ['exports/QUESTIONS_WITH_GOLD_ZH.md','exports/DIAGNOSTIC_OUTPUTS_ZH.md','exports/per_question.csv','FEEDBACK_ZH.md','next_generation/agent_plan/PLAN_ZH.md']:
 ck((H/filename).is_file() and (H/filename).stat().st_size>0,'user_visible_artifact_exists',filename)
errors=[x for x in checks if not x['pass']]
audit_paths=[H/'REPORT_ZH.md',H/'README_ZH.md',H/'SUMMARY.json',H/'FEEDBACK_SNAPSHOT.json',H/'SEMANTIC_REVIEWS.jsonl',H/'exports/CANDIDATE_LEDGER.json',H/'next_generation/INPUT_PACKET.json',H/'next_generation/agent_plan/EXPANSION_QUESTION_PLAN.json',H/'next_generation/agent_plan/ACTUAL_AGENT_READ_MANIFEST.json',R/'SCORES.json',R/'manifest.json',R/'direct/RAW_RESPONSE_FREEZE.json',R/'evidence/RAW_RESPONSE_FREEZE.json',H/'review/POSTRUN_RECOMPUTATION_VERIFIED.json',Path(__file__)]
result={'schema_version':'feedback_realization_final_completeness_audit_v1','status':'PASS' if not errors else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'checks_count':len(checks),'failure_count':len(errors),'failures':errors,'checks':checks,'audited_files':[bind(p) for p in audit_paths],
 'recomputed_metrics':recomputed,'old_sealed_files_verified':old_count,'old_sealed':old_reports,'fixed_calls':77,'tokens':{'input':token_inputs,'output':token_outputs,'total':token_inputs+token_outputs},'live_wall_seconds':live_wall,'model_load_seconds':load_wall,
 'planned_slots':12,'accepted':11,'held':1,'Gold5_admitted':0,'task_feedback_cards':5,'content_realized_cards':4,'source_verified_target_failure_cards':3,'source_verified_target_failure_records':7,
 'diagnostic_valid':8,'diagnostic_invalid':3,'diagnostic_truncated':0,'source_fact_rejection_loop_closed':False,'actual_next_agent_plan_consumption':True,'all_targets_loop_closed':False,'formal_reproduced':False,'matched_AB':False,'training_started':False,
 'reviewer_disclosure':'Auditor also wrote the runtime adapter and preflight; separate role, not independent implementation or technical blinding. Postrun private Gold and diagnostics read only for this final audit after immutable raw/score.',
 'semantic_review_scope':'Reviewed semantic audit completeness, binding and stated scope; did not replace the two substantive full-source reviewers or claim fresh full item-by-item semantic annotation.',
 'new_solver_calls':0,'mutation_scope':'This audit script and final review artifacts only; no runtime, raw, score, feedback or navigation mutation.'}
out=H/'review/FINAL_COMPLETENESS_AUDIT.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(H/'review/AUDIT_FINAL_COMPLETENESS_TOOL_ARCHIVE.json').write_text(json.dumps({'status':'ARCHIVED','script':bind(Path(__file__)),'audit':bind(out),'new_solver_calls':0},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'checks','audited_files','old_sealed','reviewer_disclosure','semantic_review_scope'}},ensure_ascii=False))
if errors:raise SystemExit(2)
