#!/usr/bin/env python3
"""Finite independent closeout; preserves passed prior TP88 recomputation."""
import json,hashlib,collections,itertools
from pathlib import Path
from datetime import datetime,timezone
RUN=Path(__file__).resolve().parents[1];W=RUN.parent;OUT=RUN/'review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
prior=read(OUT/'INDEPENDENT_CLOSEOUT_AUDIT.json')
archive=OUT/'INDEPENDENT_TP88_PRIOR_RECOMPUTATION.json'
if not archive.exists():archive.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+chr(10))
prior=read(archive);checks=prior['checks'].copy();bindings=[];resolutions=[]
def check(n,v,d=None):checks.append({'name':n,'passed':bool(v),'details':d})
def binding(b):
 p=Path(b['path'])
 if p==W/'CURRENT_STATUS.json':
  p=RUN/'navigation_history/CURRENT_STATUS.json';resolutions.append({'declared_path':b['path'],'resolved_path':str(p),'sha256':b['sha256'],'reason':'Execution-before-navigation immutable snapshot; navigation publication is not original-source mutation.'})
 return p.exists()and sha(p)==b['sha256']and('bytes'not in b or p.stat().st_size==b['bytes'])
def rd(rel):
 p=RUN/rel;bindings.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size});return read(p)
FZ=rd('EXECUTION_FREEZE.json');D=rd('common_eval/HISTORICAL184_ADMISSION.json');F=rd('common_eval/FAMILY_DEDUP_LEDGER.json');A=rd('common_eval/evidence/SOURCE_QA_ATOM_RECONSTRUCTION.json');E=rd('common_eval/evidence/NAMED_PROPERTY_SEMANTIC_ADJUDICATION.json');O=rd('common_eval/evidence/ORDERING_ALTERNATIVE_POSITION_CHECKS.json');B=rd('training/TRAINING_BUDGET_FREEZE.json');L=rd('training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json');R=rd('common_eval/COMMON_REGRESSION_REQUEST_FREEZE.json');V=rd('common_eval/COVERAGE_DECISION.json');P=rd('READINESS_DECISION.json')
fixed=next(s for s in FZ['streams']if s['id']=='CE-184')['fixed_candidate_ids'];rows=D['records'];by={r['candidate_id']:r for r in rows};atoms={r['candidate_id']:r for r in A['records']};counts=dict(collections.Counter(r['status']for r in rows));accepted={r['candidate_id']for r in rows if r['status']=='PASS'}
check('prior203_TP88_independent_recomputations_passed',prior['checks_total']==203 and prior['checks_passed']==203 and not prior['failures'],{'provenance':str(archive),'scope':'Includes actual B93 lineage,352 maps/161600 exhaustive roundtrips,27 model file hashes,187 candidates and88 pairs. Model hashing not repeated in this finalizer.'})
check('fixed184_exact_original_order', [r['candidate_id']for r in rows]==fixed and len(set(fixed))==184)
check('final115_11_58_dispositions',counts==D['counts']=={'PASS':115,'HOLD':58,'EXCLUDE':11})
check('all27controls_separate16admitted',sum(r['reporting_role']=='GOLD6_CONTROL_ONLY'for r in rows)==27 and sum(by[c]['reporting_role']=='GOLD6_CONTROL_ONLY'for c in accepted)==16)
check('all184_content_QA_checks_pass',all(all(r['checks'].values())for r in rows)and all(not r['issues']for r in A['records']))
check('all184_evidence_and_inputs_actual_hashes',bool(D['evidence_bindings'])and all(binding(b)for b in D['evidence_bindings']+D['input_bindings']))
check('frozen_original_inputs_with_navigation_snapshot_resolution',all(binding(b)for b in FZ['input_bindings']))
fm=[c for f in F['families']for c in f['candidate_ids']];fb={c:f['family_id']for f in F['families']for c in f['candidate_ids']}
check('family_partition34_exact184',len(fm)==len(set(fm))==184 and set(fm)==set(fixed)and len(F['families'])==34 and all(r['family_id']==fb[r['candidate_id']]for r in rows)and F['root_id_is_not_family_identity']and F['all_gold_subsets_and_permutations_one_family'])
proofs={p['rule_id']:p for p in E['overlaps']};proven={k for k,p in proofs.items()if p['adjudication_status']=='CONFIRMED_NECESSARY_FACT_OVERLAP'}
check('fullsource_no_automatic_exclusion_3confirmed20specificHOLD',len(proofs)==23 and proven=={'CE184-PROP-02','CE184-PROP-11','CE184-PROP-22'}and all(p['context_exposure']['is_automatic_fact_family_exclusion']is False for p in proofs.values())and 'cannot replace exact necessary' in E['full_source_input_exposure_rule'])
proof_results=[]
for k in sorted(proven):
 for b in proofs[k]['necessary_train_branch_proofs']:
  q=next(q for q in read(b['record_binding']['path'])['questions']if q['candidate_id']==b['candidate_id']);text=(q.get('options')or q.get('items'))[b['label']];s=read(b['source_binding']['path']);t=next(t for t in s['turns']if t['turn_ref']==b['necessary_citation']['turn_ref']);semantic=('hospital address'in text and 'Hills Rd'in text)if k.endswith('02')else('Broughton House Gallery'in text and 'free entry'in text);ok=binding(b['record_binding'])and binding(b['source_binding'])and text==b['public_proposition']and b['necessary_citation']['quote']in t['text']and t==b['source_turn']and bool(b['exposure_occurrences'])and all('train'in o['role']for o in b['exposure_occurrences'])and semantic;proof_results.append({'rule_id':k,'candidate_id':b['candidate_id'],'label':b['label'],'necessary_proposition':text,'passed':bool(ok)})
check('actual3Train_candidate_branch_necessary_attribute_proofs',len(proof_results)==3 and all(x['passed']for x in proof_results))
check('EXCLUDE11_require_confirmed_actual_Train_necessity',all(r['current_exposure']['overlap_rule_ids']and set(r['current_exposure']['overlap_rule_ids'])<=proven for r in rows if r['status']=='EXCLUDE'))
check('HOLD58_specific_not_content_failure',all(r['current_exposure']['unresolved_scopes']and all(r['checks'].values())and not r['current_exposure']['overlap_rule_ids']for r in rows if r['status']=='HOLD'))
check('PASS115_no_open_mapping_or_overlap',all(not r['current_exposure']['unresolved_scopes']and not r['current_exposure']['overlap_rule_ids']and all(b['semantic_status']=='NO_SAME_NECESSARY_FACT_FOUND_IN_BOUND_CURRENT_EXPOSURE'for b in r['current_exposure']['all_option_or_item_checks'])for r in rows if r['status']=='PASS'))
check('independently_discovered_hotel_mutation_cross_type_fixed',by['h4_codex_dev_test_ab_v124-ms-fe6ec339cc8c8baf']['status']=='HOLD'and by['h4_codex_dev_test_ab_v124-root215999-ordering1']['status']=='HOLD')
check('failed_fullsource_first_version_preserved',(RUN/'common_eval/evidence/superseded_full_source_only_v1/HISTORICAL184_ADMISSION.json').is_file())
source_ok=[];sources={}
for r in A['records']:
 p=r['source_binding']['path']
 if p not in sources:sources[p]=read(p)
 ts={t['turn_ref']:t for t in sources[p]['turns']};aa={a['atom_id']:a for a in r['necessary_fact_union']};source_ok.append(binding(r['source_binding'])and all(a['source_quote'].casefold()in ts[a['turn_ref']]['text'].casefold()and a['speaker']==ts[a['turn_ref']]['speaker']for a in aa.values())and all(b['atom_ids']and set(b['atom_ids'])<=set(aa)for b in r['all_option_or_item_branches'].values()))
check('all184_actual_source_and_all_branch_atoms',len(source_ok)==184 and all(source_ok))
alt_ok=[]
for r in O['records']:
 choices=list(itertools.product(*r['all_anchors']));orders={tuple(x['label']for x in sorted(c,key=lambda z:(z['turn_index'],z['offset'])))for c in choices};alt_ok.append(orders=={tuple(r['gold'])}and len(choices)==r['cartesian_combinations_checked']and r['status']=='PASS')
check('all50_ordering_alternative_anchor_invariance',len(alt_ok)==50 and all(alt_ok))
M=B['current_protocol_mix_policy'];check('current_v21_mix_supersedes_historical602020',M['protocol_detail_version']=='v2.1'and binding(M['protocol_binding'])and M['target_share_formula']=={'new_structure':'0.75 * (1 - q)','audited_legacy':'q','existing_round2_foundation':'0.25 * (1 - q)'}and M['candidate_mix_target_if_q0']=={'new_structure':.75,'audited_legacy':0.,'existing_round2_foundation':.25}and M['superseded_for_current_mix']and M['actual_supervised_token_shares']is None and M['final_schedule_q']is None and not M['final_budget_frozen'])
check('requests_evidence_hashes_actual',all(binding(b)for b in R['evidence_bindings']+[R['admission_binding'],R['inherited_request_policy'],R['adapter_binding'],R['builder_binding'],R['native_contract_binding'],V['request_bytes_freeze']]))
pubs=lines(RUN/'common_eval/evidence/ADMITTED_PUBLIC.jsonl');privs=lines(RUN/'common_eval/evidence/ADMITTED_PRIVATE.jsonl');reqs=lines(RUN/'common_eval/evidence/ADMITTED_REQUESTS.jsonl');prompts=lines(RUN/'common_eval/evidence/ADMITTED_PROMPTS.jsonl');pr={p['prompt_id']:p for p in prompts};RT=rd('common_eval/evidence/REQUEST_TOKENIZATION_AND_ROUNDTRIPS.json')
check('public_private115_exact_PASS_no_HOLD_EXCLUDE',len(pubs)==len(privs)==115 and {r['candidate_id']for r in pubs}=={r['candidate_id']for r in privs}=={r['candidate_id']for r in reqs}==accepted)
check('public_whitelist_exact_no_private_payload',all(set(r)<=set(R['public_input']['schema_keys'])and bool(r.get('options'))!=bool(r.get('items'))for r in pubs))
check('all906_prompt_bytes_hashes',len(pr)==len(prompts)==906 and all(hashlib.sha256(p['prompt'].encode()).hexdigest()==p['prompt_sha256']for p in prompts))
check('1854requests_fixed184_index_sampling_seed',len(reqs)==R['request_count_per_model']==1854 and all(r['fixed184_index']==fixed.index(r['candidate_id'])and r['sampling']['seed']==20261020+r['fixed184_index']*18+int(r['variant_id'].split('_')[-1])*(3 if r['question_type']=='multi_select'else 1)+r['sample_index']and r['prompt_sha256']==pr[r['prompt_id']]['prompt_sha256']for r in reqs))
check('request_bytes_frozen_execution_identity_still_pending',all(r['execution_status']=='NOT_EXECUTED'and r['runtime_binding']is None for r in reqs)and not R['execution_lock']['executable_request_lock']and not R['execution_lock']['future_checkpoint_identities_bound']and not R['execution_lock']['full_common_AB_evaluation_ready'])
check('requests_zero_truncation_actual906variants',RT['zero_truncation']and RT['public_questions']==115 and RT['variants']==906 and RT['requests_per_model']==1854 and not RT['model_weights_loaded']and all(not r['truncated']and r['total_budget']<=r['limit']for r in RT['records']))
check('coverage_counts115_primary63MC36order16control',V['dispositions']==counts and V['admitted_count']==115 and V['primary']=={'question_count':99,'multi_select_gold2_to5':63,'ordering':36}and V['gold6_controls']['question_count']==16 and V['root_count_admitted']==25 and V['families_with_admitted_rows']==28 and not V['novel_family_or_unseen_root_claim']and not V['full_common_AB_evaluation_ready'])
check('parent_output_hashes_actual',all(binding(b)for b in P['input_bindings']))
H=P['historical_regression'];check('parent_historical_and_request_counts_reconcile',H['dispositions']==counts and (H['admitted_total'],H['admitted_primary_MC'],H['admitted_ordering'],H['admitted_control6'],H['admitted_roots'],H['admitted_family_groups'])==(115,63,36,16,25,28)and P['stream_request_status']==R['status']and P['stream_coverage_decision']==V)
check('parent_TP88_counts_resources_reconcile',P['candidate_training_preflight']['counts']==L['counts']and P['candidate_training_preflight']['resource_summary']==B['candidate_only_resources']and P['current_treatment']==L['treatment_scope']and P['final_training_pool_counts']=={'A':0,'B':0})
check('Day3_bounded_complete_Day4_final_pending_Day5_no_training',P['day3_production_scope_review_complete']is True and P['day3_full_goal_ready']is False and P['training_start_go']is False and P['full_day_complete']is False and 'Day4'in P['day3_day4_boundary']and 'Day5'in P['day3_day4_boundary'])
check('zero_new_execution_material',all(P[k]==0 for k in ['new_solver_calls','new_model_inference','scorer_runs','new_questions','new_dialogue_turns','new_training_runs'])and not P['old_sealed_modified'])
passed=all(c['passed']for c in checks)
out={'schema_version':'day3_mainline_independent_closeout_audit_v2','checked_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/day3_progress_boundary','status':'PASS_BOUNDED_READINESS_FINAL_TRAINING_NO_GO'if passed else'FAIL_REQUIRES_OWNER_REPAIR','all_pass':passed,'checks_total':len(checks),'checks_passed':sum(c['passed']for c in checks),'failures':[c for c in checks if not c['passed']],'checks':checks,'scope':'Independent CE184,TP88,request-content and parent closeout. This reviewer authored CE-ORDER-4; that branch is excluded from independent semantic certification.','CE184_semantic_review':{'dispositions':counts,'HOLD_meaning':'58 specific necessary-property/identity mapping uncertainties; all184 content/Gold/QA checks pass.','confirmed_actual_Train_necessary_branch_proofs':proof_results,'corrections_required_and_completed':['Full-source-only exclusions changed to specific HOLD.','Hamilton Lodge hotel mutation and explicit guesthouse ordering item held in same uncertain attribute family.'],'PASS_opinion':'Bounded necessary-fact comparisons found no same necessary fact in115 admitted historical regression rows; historical source-first non-author QA reused. No fresh blinded/human certification or unseen-root claim.','six_narrow_uncertainties_reviewed':['Acorn north','University Arms Regent Street','Cambridge Belfry stars','Primavera telephone','Parkside identity','Primavera free admission']},'B_actual_lineage_recomputed':prior['B_actual_lineage_recomputed'],'prior_TP88_actual_recomputation_binding':{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size},'navigation_snapshot_resolutions':list({r['declared_path']:r for r in resolutions}.values()),'input_bindings':bindings,'day_boundary':'Day3 bounded content/feedback/diversity/source-family and CPU request readiness; Day4 final common pool,mixed schedule,checkpoints/runtime identity pending; Day5 training not run.','new_solver_calls':0,'new_training_runs':0,'technical_blinding':False,'human_review':False}
(OUT/'INDEPENDENT_CLOSEOUT_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10));print(json.dumps({k:out[k]for k in ['status','all_pass','checks_total','checks_passed','failures']},ensure_ascii=False))
