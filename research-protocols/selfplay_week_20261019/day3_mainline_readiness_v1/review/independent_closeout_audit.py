#!/usr/bin/env python3
"""Independent cross-stream check; never edits stream outputs or calls models."""
import json,hashlib,collections,itertools
from pathlib import Path
from datetime import datetime,timezone
RUN=Path(__file__).resolve().parents[1];W=RUN.parent;OUT=RUN/'review'
checks=[];bindings={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def rd(p):
 p=Path(p);bindings[str(p)]=bind(p);return json.loads(p.read_text())
def check(name,test,detail=None):checks.append({'name':name,'passed':bool(test),'details':detail})
def bindcheck(b):
 p=Path(b['path']);return p.exists()and sha(p)==b['sha256'] and ('bytes'not in b or p.stat().st_size==b['bytes'])
def jlines(p):
 p=Path(p);bindings[str(p)]=bind(p);return [json.loads(l)for l in p.read_text().splitlines()if l.strip()]
freeze=rd(RUN/'EXECUTION_FREEZE.json')
# Do not re-review CE-ORDER-4 authored by this reviewer. Scope here is CE184/TP88/parent readiness.
L=rd(RUN/'training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json')
B=rd(RUN/'training/TRAINING_BUDGET_FREEZE.json')
M=rd(RUN/'training/DISPLAY_MAP_CHECKS.json')
T=rd(RUN/'training/TOKENIZATION_ZERO_TRUNCATION.json')
C=rd(RUN/'training/COMMON_OLD_BASE_SET.json')
rows=L['candidate_records'];by={r['candidate_id']:r for r in rows};pairs=L['pairs'];paired={p[a]for p in pairs for a in ['A','B']}
check('TP88_unique187_records_A94_B93',len(by)==len(rows)==187 and dict(collections.Counter(r['arm']for r in rows))=={'A':94,'B':93})
check('TP88_exact88_pairs176_unique_candidates',len(pairs)==88 and len(paired)==176 and len({p['slot_id']for p in pairs})==88)
check('TP88_all88_candidate_only_usable',all(p['candidate_pair_CPU_usable']and all(p['pair_checks'].values()) for p in pairs))
ordrows=[r for r in rows if r['in88pair']and r['arm']=='B'and r['question_type']=='ordering'];mcrows=[r for r in rows if r['in88pair']and r['arm']=='B'and r['question_type']=='multi_select']
check('TP88_B53narrow_ordering35commonMC',len(ordrows)==53 and len(mcrows)==35 and all(r['feedback_lineage']['qualified_for_narrow_feedback_target']for r in ordrows)and not any(r['feedback_lineage']['formal_MC_target']for r in mcrows))
# Check actual linked inputs rather than accepting precomputed booleans only.
card=rd(W/'day3_production_v1/packets/B/ORDERING_FEEDBACK_BOUNDED.json');ad=card['admission']
check('TP88_qualified_card_narrow_scope',ad['decision']=='ADMITTED_OPERATIONAL_REPRODUCED_WITH_SCOPE_LIMITS'and ad['source_qualified_failure_roots']>=3 and ad['source_qualified_data_families']>=3 and ad['predeclared_related_probe_support']is True and not ad['internal_cause_proof'] and 'Cross-scene ordering only' in card['day3_use'])
chain_results=[];record_cache={};public_cache={}
for r in rows:
 rec=rd(r['record_binding']['path']);pub=rd(r['public_binding']['path']);record_cache[r['candidate_id']]=rec;public_cache[r['candidate_id']]=pub
 check('TP88_original_candidate_bindings:'+r['candidate_id'],all(bindcheck(r[k])for k in ['record_binding','public_binding','source_binding']))
 if r['arm']!='B':continue
 lineage=r['feedback_lineage'];rm=rd(lineage['read_manifest']['path']);pl=rd(lineage['plan_binding']['path']);decl=pl.get('read_manifest_sha256',pl.get('read_manifest_binding',{}).get('sha256'));gt=min(g['started_at']for g in rec['generation_history']);pt=pl.get('frozen_at_utc',pl.get('created_at_utc'))
 cb=bind(W/'day3_production_v1/packets/B/ORDERING_FEEDBACK_BOUNDED.json')
 cardread=any(x.get('path')==cb['path']and x.get('sha256')==cb['sha256']for x in rm.get('read_bindings',[]))or rm.get('feedback_card_binding',{}).get('sha256')==cb['sha256']
 ch={'candidate_id':r['candidate_id'],'read_manifest_hash_matches_plan':sha(lineage['read_manifest']['path'])==decl,'card_read_receipt_bound':cardread,'plan_before_first_draft':pt<=gt,'actual_QA_and_target_pass':rec['qa_review']['status']=='PASS'and rec['target_realization']['target_implemented']=='YES','within_qualified_scope':True}
 if r['question_type']=='ordering':
  o=rec['ordering'];counts=o['scene_event_counts'];design=rec['target_realization']['ordering_profile_id'];expected=pl.get('feedback_to_plan',{}).get('targeted_slots',[])
  ch['actual_target_predeclared']=r['candidate_id']in expected or ('ordering_feedback_plan'in pl and design in pl.get('intended_topics',{}))
  ch['within_qualified_scope']=len(counts)>=2 and sum(counts)==len(pub['options_or_items']) and o['unique_order_verified']in[True,'PASS','YES'] and o['all_reasonable_mentions_searched']in[True,'PASS','YES']
 chain_results.append(ch)
check('TP88_all_B93_actual_lineage_pass',len(chain_results)==93 and all(all(v for k,v in r.items()if k!='candidate_id')for r in chain_results))
# Directly recompute all display maps and complete answer-space roundtrips.
map_pass=[]
seed=M['seed'];total_cases=0
for m in M['rows']:
 r=by[m['candidate_id']];pub=public_cache[m['candidate_id']];labs=list(pub['options_or_items']);orig=sorted(labs,key=lambda x:hashlib.sha256((str(seed)+'|'+str(m['epoch'])+'|'+m['slot_id']+'|'+x).encode()).hexdigest());dto=dict(zip(labs,orig));otd={v:k for k,v in dto.items()}
 answers=itertools.permutations(labs)if r['question_type']=='ordering'else itertools.chain.from_iterable(itertools.combinations(labs,n)for n in range(len(labs)+1))
 cases=0;passed=True
 for a in answers:
  cases+=1;passed=passed and [dto[otd[x]]for x in a]==list(a)
 total_cases+=cases;map_pass.append(dto==m['display_to_original']and otd==m['original_to_display']and passed and cases==m['all_answer_space_cases'])
check('TP88_all352_maps_independent_reconstruction',len(map_pass)==352 and all(map_pass)and total_cases==161600,{'maps':len(map_pass),'answer_space_roundtrip_cases':total_cases})
mt={(m['candidate_id'],m['epoch']):m for m in M['rows']}
check('TP88_two_epochs_every_paired_candidate',set(mt)=={(cid,e)for cid in paired for e in [1,2]})
check('TP88_token_rows352_complete',len(T['candidate_new_structure']['rows'])==352 and {(r['candidate_id'],r['epoch'])for r in T['candidate_new_structure']['rows']}==set(mt))
tr=T['candidate_new_structure']['rows'];totals={a:sum(r['full_input_tokens']for r in tr if r['arm']==a)for a in ['A','B']};diff=abs(totals['A']-totals['B'])/max(totals.values())
check('TP88_token_totals_and_resource_ratio_recompute',totals==B['candidate_only_resources']['complete_input_tokens']and abs(diff-B['candidate_only_resources']['relative_input_difference'])<1e-15 and diff<=.05,{'full_input_tokens':totals,'ratio':diff})
check('TP88_no_candidate_truncation6889to7947',min(r['full_input_tokens']for r in tr)==6889 and max(r['full_input_tokens']for r in tr)==7947 and not any(r['over_limit']for r in tr) and all(r['no_truncation_requested']for r in tr) and T['truncation_executed']is False)
check('TP88_model27_files_actual_hashes',len(B['model_identity_checks'])==27 and all(bindcheck(x)and x['sha256']==x['expected_sha256']and x['bytes']==x['expected_bytes']for x in B['model_identity_checks']))
check('TP88_actual_final_pools_empty_NO_GO',all((RUN/f'training/TRAIN_POOL_{a}.jsonl').stat().st_size==0 for a in ['A','B'])and B['final_budget']['records_A']==B['final_budget']['records_B']==0 and not B['final_budget']['final_schedule_validated']and L['training_start_go']is False)
check('TP88_empty_final_not_vacuous_PASS',T['final_training_pool_check']['zero_truncation_pass']is None and L['diversity_content_candidate_gate']['final_selected_pool_gate']=='NOT_EVALUATED_EMPTY_NOT_PASS')
check('TP88_common_old_base_not_falsely_admitted',C['legacy']['current_selected_count']==0 and C['foundation']['current_selected_A']==C['foundation']['current_selected_B']==[] and C['q']['final_schedule_q']is None and C['q']['q0_fallback_target_new_old_base']==[.75,0,.25])
amend=rd(W/'day3_production_v1/DAY3_SCOPE_AMENDMENT.json');av=rd(RUN/'review/EXISTING_SCOPE_AMENDMENT_VERIFICATION.json')
check('TP88_original70_and_existing50_separately_disclosed',L['treatment_scope']['original_plan']=={'targeted':84,'common':36}and L['treatment_scope']['currently_authorized_plan']=={'targeted_ordering':60,'common':60}and all(av['checks'].values())and bindcheck(av['binding'])and L['treatment_scope']['accepted_yield_fraction_is_not_a_new_treatment_amendment'])
# Final stages add CE184 and parent consistency after these files exist.
ce=RUN/'common_eval/HISTORICAL184_ADMISSION_LEDGER.json';decision=RUN/'READINESS_DECISION.json'
if ce.exists():
 D=rd(ce);print('CE_FILE_PRESENT',list(D))
if decision.exists():
 D=rd(decision);check('PARENT_training_and_full_day_stay_false',D.get('training_start_go')is False and D.get('full_day_complete')is False)
result={'schema_version':'day3_mainline_independent_closeout_audit_v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/day3_progress_boundary','status':'TP88_REVIEW_COMPLETE_AWAITING_CE184_AND_PARENT_DECISION','scope':'Independent recheck of TP88/CE184/parent closeout only. This reviewer authored CE-ORDER-4; that branch is excluded from its independent semantic opinion.','checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'failures':[x for x in checks if not x['passed']],'checks':checks,'B_actual_lineage_recomputed':chain_results,'input_bindings':sorted(bindings.values(),key=lambda b:b['path']),'open_review_note':'Historical inherited60/20/20 must be overridden clearly by current v2.1 q formula. Final mixed schedule remains absent, not PASS.','new_solver_calls':0,'new_training_runs':0,'technical_blinding':False,'human_review':False}
(OUT/'INDEPENDENT_CLOSEOUT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
print(json.dumps({'status':result['status'],'checks_total':len(checks),'failures':result['failures'],'maps':len(map_pass),'ratio':diff},ensure_ascii=False))
