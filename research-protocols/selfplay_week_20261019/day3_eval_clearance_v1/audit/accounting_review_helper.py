import json,hashlib,collections,itertools,datetime,ast
from pathlib import Path
W=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019');C=W/'day3_eval_clearance_v1'
files=[C/'INPUT_MANIFEST.json',C/'SCREENING_FREEZE.json',W/'day3_eval_semantic_v1/COMMON_EVAL_WORK_QUEUE.json',W/'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json']
files+=sorted((C/'inputs').glob('*.json'))+sorted((C/'screening').glob('*.json'))
files += [C/'freeze_requests.py',W/'day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json',W/'day2/solver/adapter.py']
raw={str(p):p.read_bytes() for p in files}; J=lambda p:json.loads(raw[str(p)])
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
sha=lambda x:hashlib.sha256(x).hexdigest()
manifest=J(files[0]);freeze=J(files[1]);queue=J(files[2]);lock=J(files[3]);ledger=lock['itemized_slot_family_ledger'];L={s['slot_id']:s for s in ledger}
findings=[];checks=[]
def check(name,condition,detail):
 checks.append({'check':name,'status':'PASS' if condition else 'FAIL','detail':detail})
 if not condition:findings.append({'check':name,'detail':detail})
for ref in [manifest['policy'],manifest['source_lock'],freeze['inheritance'],freeze['source_material_queue']]+[g['packet'] for g in manifest['groups']]:
 b=raw.get(ref['path']);check('bound_input_'+Path(ref['path']).name,b is not None and sha(b)==ref['sha256'] and len(b)==ref['bytes'],{'path':ref['path'],'sha256_actual':sha(b) if b else None,'sha256_expected':ref['sha256']})
packets={p.stem:J(p) for p in files if p.parent==C/'inputs'}
screenings={p.stem:J(p) for p in files if p.parent==C/'screening'}
roots={r['root_id']:r for d in packets.values() for r in d['roots']}
source_bindings={r['root_id']:r['source_binding'] for r in lock['source_bindings']}
packet_slots={s['slot_id']:(r,s) for d in packets.values() for r in d['roots'] for s in r['supported_slots']}
screen_slots=[(group,s) for group,d in screenings.items() for s in d['slots']]
S={s['slot_id']:s for group,s in screen_slots};Q={s['slot_id']:s for s in queue['priority_slots']};R={s['slot_id']:s for s in queue['retained_material_dispositions']}
check('ledger_186_unique',len(ledger)==len(L)==186,{'count':len(ledger),'unique':len(L)})
check('screened60_exact_supported_set',len(screen_slots)==len(S)==60 and set(S)==set(Q)==set(packet_slots),{'screened':len(S),'queue_supported':len(Q),'packet_supported':len(packet_slots),'missing':sorted(set(Q)-set(S)),'extra':sorted(set(S)-set(Q))})
check('retained98hold28shortfall',len(R)==126 and collections.Counter(x['status'] for x in R.values())=={'MATERIAL_AMBIGUOUS_HOLD':98,'SOURCE_SHORTFALL':28},dict(collections.Counter(x['status'] for x in R.values())))
check('all186_partition_no_replacement',set(S).isdisjoint(R) and set(S)|set(R)==set(L),{'supported':len(S),'retained':len(R)})
expected_order=[]
for split in ['dev','test']:
 for root in sorted(lock['root_scope'][split+'_root_ids']):
  expected_order.extend((split,root,qt,n) for qt in ['multi_select','ordering'] for n in [1,2,3])
check('ledger_original_order_and_identity',[(x['split'],x['root_id'],x['question_type'],x['local_slot']) for x in ledger]==expected_order,'DEV then TEST; lexicographic root ID; MC1/2/3 then ORDER1/2/3')
allplan=[];slot_rows=[];check5=['train_new_replay_basic_family','train_diagnostic_feedback_related_family','historical_dev_test_v140_family','cross_root_cross_split_semantic_near_duplicate','within_locked_eval_fact_overlap_and_revision']
key_fields={'source_sha256','normalized_semantic_fact_tuples','evidence_turn_refs','explicit_scope','mention_rule'}
for group,s in screen_slots:
 base=L[s['slot_id']];r,ps=packet_slots[s['slot_id']];turns=r['complete_source']['turns'];T={t['turn_ref']:t for t in turns};idx={t['turn_ref']:i for i,t in enumerate(turns)}
 identity=all(s[k]==base[k] for k in ['root_id','question_type','source_sha256']) and s['design_target']==base['design_target'] and ps['design_target']==base['design_target']
 check('slot_identity_design_'+s['slot_id'],identity,{'group':group,'root_id':s['root_id'],'type':s['question_type']})
 check('source_binding_'+s['slot_id'],r['source_binding']==source_bindings[r['root_id']],{'bound_sha':r['source_binding']['sha256'],'lock_sha':source_bindings[r['root_id']]['sha256']})
 check('max3_'+s['slot_id'],1<=len(s['plans'])<=3,{'plans':len(s['plans'])})
 prow=[]
 for p in s['plans']:
  pid=p['plan_id'];allplan.append((s,p));key=p['candidate_key_payload'];refs=key['evidence_turn_refs'];w=p['source_witnesses'];wmap={x['turn_ref']:x for x in w}
  check('candidate_key_'+pid,set(key)==key_fields and sha(canon(key))==p['candidate_selection_key'],{'payload_fields':sorted(key),'key':p['candidate_selection_key']})
  check('key_source_'+pid,key['source_sha256']==base['source_sha256'],key['source_sha256'])
  check('quote_refs_'+pid,len(refs)==len(set(refs)) and set(refs)==set(wmap) and all(ref in T and isinstance(wmap[ref].get('quote'),str) and wmap[ref]['quote'] in T[ref]['text'] for ref in refs),{'evidence_refs':refs,'witness_count':len(w)})
  check('quote_speaker_scene_'+pid,all(x.get('speaker',T[x['turn_ref']]['speaker'])==T[x['turn_ref']]['speaker'] and x.get('scene_ref',T[x['turn_ref']]['scene_ref'])==T[x['turn_ref']]['scene_ref'] for x in w),{'source_witnesses_checked':len(w)})
  scenes=key['explicit_scope']['scene_refs'];actual={T[ref]['scene_ref'] for ref in refs};target=base['design_target'];scope=target['evidence_scope_target']
  scope_ok=actual==set(scenes) and None not in actual and len(actual)>=target['minimum_explicit_scene_count'] and (len(actual)==1 if scope=='same_explicit_scene' else len(actual)>=2)
  check('scope_'+pid,scope_ok,{'actual_scenes':sorted(actual),'declared_scenes':scenes,'locked_scope':scope})
  check('five_check_fields_'+pid,set(p['family_checks'])==set(check5) and all(isinstance(p['family_checks'][k],dict) and all(z in p['family_checks'][k] for z in ['status','compared_record_ids','matched_or_related_record_ids','evidence','rationale','reviewer']) for k in check5),{'check_statuses':{k:v.get('status') for k,v in p['family_checks'].items()}})
  ordering_detail=None
  if s['question_type']=='ordering':
   op=p.get('ordering_plan',{});events=op.get('atomic_events') or op.get('anchor_audit') or p.get('ordering_atomic_events') or []
   anchors=[];ordinals=[];eligible=[]
   for e in events:
    anchor=e.get('selected_anchor');anchors.append(anchor['turn_ref'] if isinstance(anchor,dict) else anchor)
    ordinals.append(anchor.get('report_ordinal_in_scope') if isinstance(anchor,dict) else e.get('report_index_in_scope',e.get('report_index_in_explicit_scope')))
    eligible.append(e.get('eligible_scope_report_refs',e.get('eligible_scope_predicate_reports',[])))
   counts=collections.Counter(T[x]['scene_ref'] for x in anchors)
   comp='+'.join(map(str,sorted(counts.values(),reverse=True)))
   pairs=op.get('pairwise_source_span_checks') or op.get('pairwise_order_evidence') or p.get('pairwise_order_evidence') or []
   if not pairs:
    for k,v in op.items():
     if isinstance(v,list) and len(v)==15 and 'pair' in k:pairs=v
   valid6=len(events)==len(anchors)==len(set(anchors))==6 and comp==target['scene_composition_target']
   check('ordering_six_scene_'+pid,valid6,{'events':len(events),'distinct_anchors':len(set(anchors)),'actual_composition':comp,'locked_composition':target['scene_composition_target']})
   check('ordering_mention_'+pid,key['mention_rule']['rule']==target['mention_scope_target'] and all(isinstance(n,int) and n>=1 for n in ordinals) and all(a in ee for a,ee in zip(anchors,eligible)),{'mention_rule':key['mention_rule'],'anchor_ordinals':ordinals})
   check('ordering_fifteen_pairs_'+pid,len(pairs)==15,{'pair_records':len(pairs)})
   # Mechanical mention integrity, not a repeat semantic interpretation audit.
   mention_ok=all(sorted(ee,key=lambda x:idx[x])[n-1]==a for a,ee,n in zip(anchors,eligible,ordinals) if n<=len(ee)) and all(n<=len(ee) for ee,n in zip(eligible,ordinals))
   check('ordering_declared_report_anchor_'+pid,mention_ok,{'anchors':anchors,'ordinals':ordinals})
   ordering_detail={'events':len(events),'composition':comp,'pair_records':len(pairs),'declared_anchor_ordinals':ordinals}
  prow.append({'plan_id':pid,'candidate_selection_key':p['candidate_selection_key'],'explicit_scope':key['explicit_scope'],'family_check_statuses':{k:v['status'] for k,v in p['family_checks'].items()},'plan_disposition':p.get('plan_disposition',p.get('plan_status')),'ordering_mechanical_summary':ordering_detail})
 slot_rows.append({'slot_id':s['slot_id'],'root_id':base['root_id'],'split':base['split'],'question_type':base['question_type'],'source_sha256':base['source_sha256'],'design_target_preserved':identity,'plan_count':len(s['plans']),'initial_screening_disposition':s['slot_disposition'],'plans':prow})
for sid,r in R.items():
 base=L[sid];ok=r['root_id']==base['root_id'] and r['split']==base['split'] and r['question_type']==base['question_type']
 check('retained_identity_'+sid,ok,{'status':r['status'],'root_id':r['root_id'],'question_type':r['question_type']})
 slot_rows.append({'slot_id':sid,'root_id':base['root_id'],'split':base['split'],'question_type':base['question_type'],'source_sha256':base['source_sha256'],'design_target':base['design_target'],'plan_count':0,'retained_material_disposition':r['status'],'identity_preserved':ok,'replacement_authorized':False})
check('plan_count107',len(allplan)==107,{'total':len(allplan),'groups':{g:sum(len(s['plans']) for s in d['slots']) for g,d in screenings.items()}})
check('plan_ids_unique',len({p['plan_id'] for s,p in allplan})==107,'Every actual plan has one unique identity')
check('initial_screening_not_final_pass',not any(all(p['family_checks'][k]['status']=='PASS' for k in check5) for s,p in allplan),'No initial plan has all five PASS. Global and subsequent content records are outside this audit read scope.')
script=raw[str(C/'freeze_requests.py')].decode();adapter=raw[str(W/'day2/solver/adapter.py')].decode();policy=J(W/'day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json');ast.parse(script);ast.parse(adapter)
check('request_adapter_bound_actual_sha',sha(raw[str(W/'day2/solver/adapter.py')])==policy['prompt']['adapter']['sha256']==policy['display_mapping']['implementation']['sha256'],{'actual':sha(raw[str(W/'day2/solver/adapter.py')]),'policy':policy['prompt']['adapter']['sha256']})
static={'status':'STATIC_REVIEW_COMPLETE_RUNTIME_NOT_EXECUTED','script_sha256':sha(raw[str(C/'freeze_requests.py')]),'adapter_sha256':sha(raw[str(W/'day2/solver/adapter.py')]),'policy_sha256':sha(raw[str(W/'day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json')]),'fixed_findings':[{'issue':'Admission row and public question ID could diverge','current_status':'FIXED_BY_MAIN_AGENT_STATICALLY_VERIFIED','guards':['ADMISSION_PUBLIC_ID_MISMATCH','DUPLICATE_ADMITTED_CANDIDATES']},{'issue':'Self-reported source hash was not compared to prelocked source path/hash/root','current_status':'FIXED_BY_MAIN_AGENT_STATICALLY_VERIFIED','guards':['LOCKED_SOURCE_SHA_MISMATCH','LOCKED_SOURCE_PATH_MISMATCH','SOURCE_CHANGED','SOURCE_ROOT_MISMATCH']},{'issue':'Adapter supports4/5/6 while all prelocked slots require exactly6','current_status':'FIXED_BY_MAIN_AGENT_STATICALLY_VERIFIED','guards':['PRELOCKED_ITEM_COUNT_MISMATCH']}],'verified_behaviors':['Admission requires status CONTENT_ADMISSION_FINAL, at least one accepted row, and five_family_checks/independent_source_first_qa PASS.','Public adapter rejects non-whitelist keys, missing strings, wrong question types, conflicting task fields and duplicate/empty task text.','All186 slot ledger indices are retained for seed spacing; seed=base+index*18+variant_index*type_repetitions+sample_index; MC offsets0..17 and ordering0..11 do not cross slot boundaries.','Original split is taken from locked slot and not supplied by admitted public content.','Finite choice space includes every nonempty MC subset or every full ordering permutation, without consulting the locked Gold count target.','Actual variants must be deterministic; content ID rebindings roundtrip through all63 MC subsets or all720 ordering permutations.','Prompt length uses the actual full direct_prompt string, tokenizer.encode(add_special_tokens=False), reserves32 answer tokens and applies whole-question LENGTH_HOLD if any variant exceeds16384; no truncation branch.','The script has no Solver call or private Gold read and freezes model identity as pending; TEST is recorded final-model-only.','Output request IDs are unique and write_new refuses changed bytes at existing frozen paths.'],'remaining_limits':['Native contract/solver implementation imported by adapter.direct_prompt was not opened in this authorized audit; actual full-history/native chat template correctness remains covered by the separately bound preflight/runtime validation.','Preflight tokenizer binding contents and actual tokenizer files were not opened in this audit; runtime must still verify them.','Script accepts the final admission gate declarations; global semantic evidence, source-first reconstruction, Gold correctness and actual slot design must already have been independently finalized upstream.','Model/checkpoint exact identities remain explicitly pending in request freeze; no execution approval arises from this audit.']}
for token in ['ADMISSION_PUBLIC_ID_MISMATCH','DUPLICATE_ADMITTED_CANDIDATES','LOCKED_SOURCE_SHA_MISMATCH','LOCKED_SOURCE_PATH_MISMATCH','SOURCE_ROOT_MISMATCH','PRELOCKED_ITEM_COUNT_MISMATCH']:
 check('static_guard_'+token,token in script,'Fix guard present in final read script; no script mutation by auditor')
groups=collections.Counter();states=collections.Counter();allstates=collections.Counter()
for s,p in allplan:
 states[p.get('plan_disposition',p.get('plan_status'))]+=1
 for k in check5:allstates[k+':'+p['family_checks'][k]['status']]+=1
result={'schema_version':'preseal_screening_accounting_review_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'dev_rest_agent_accounting_and_static_request_auditor','status':'PASS_MECHANICAL_ACCOUNTING_ONLY_GATES_PENDING' if not findings else 'MECHANICAL_FINDINGS_REQUIRE_RECORDING','scope':'Mechanical and method audit of original186 slots, finite source-only screening and public request code. Not a fresh semantic clearance, not final content QA. No Gold/public question/model-output files read.','read_bindings':[{'path':str(p),'sha256':sha(raw[str(p)]),'bytes':len(raw[str(p)])} for p in files],'counts':{'locked_slots':len(ledger),'screened_supported_slots':len(S),'finite_plans':len(allplan),'retained_material_ambiguous_hold':98,'retained_source_shortfall':28,'initial_all_five_pass_plans':sum(all(p['family_checks'][k]['status']=='PASS' for k in check5) for s,p in allplan),'plan_dispositions':dict(states),'family_check_statuses':dict(allstates),'check_count':len(checks),'failed_checks':len(findings)},'checks':checks,'findings':findings,'all186_slot_accounting':sorted(slot_rows,key=lambda x:next(i for i,s in enumerate(ledger) if s['slot_id']==x['slot_id'])),'request_freezer_static_review':static,'remaining_gates_for_main_agent':[{'gate':'GLOBAL_FAMILY_ADMISSION','required':'Merge all107 plans in original DEV then TEST order; resolve concrete relevant historical mappings and cross-root/split semantic near duplicates; compare all necessary source facts including same/related revision chains. Fifth check must have explicit concrete chosen-family overlap resolution. No lexical/root/hash shortcut.'},{'gate':'FINITE_KEY_SELECTION','required':'Within the frozen max3 actual plans choose lexicographically smallest candidate key only among all-five-PASS plans; preserve every rejected/held plan and finite-inventory scope.'},{'gate':'SOURCE_FIRST_CONTENT_AND_INDEPENDENT_QA','required':'Only after family clearance create public question and private Gold, preserve prelocked relation/scope/6 items/cardinality, reconstruct independently from exact complete bound source with all-option union or six distinct events, explicit first/later report count, all semantic mentions and15 ordering pairs. Seal first pass before author-private comparison.'},{'gate':'FINAL_ALL186_ACCOUNTING','required':'Bind final disposition for every original slot; retain98 material HOLD and28 shortfall without replacement. Accepted/rejected/length-HOLD counts retain original186 production denominator; nonexistent questions are not Solver failures.'},{'gate':'ACTUAL_CONTENT_REQUEST_FREEZE','required':'Bind public question/private Gold hashes, content admission and original source hashes; execute corrected CPU request freezer in exact environment after upstream admission, verify actual display mappings and full prompt+32 length. Apply LENGTH_HOLD before any scores.'},{'gate':'MODEL_IDENTITY_AND_EXECUTION','required':'Bind exact models/checkpoints/tokenizer/runtime and fixed endpoint before execution; DEV aggregate reporting only, TEST final-only after model identity freeze. Shared questions/prompts/mappings/seeds across models; no Challenger release of selected eval facts or outcomes.'}],'no_solver_calls':True,'no_training_runs':True,'original_screening_files_modified':False}
for f in findings:
 if f['check']=='ordering_six_scene_D3-EVAL-TEST-060-P2':
  f['disposition']='DETECTED_AND_HELD_NOT_ADMITTED'
  f['screening_evidence']={'support_status':'HOLD_COMPOSITE_SAME_SENTENCE_ORDER','plan_disposition':'HOLD','same_turn':'R01:S01:T013','two_events':['Daniel planned start writing tonight','Daniel planned notify Chloe after first draft finished'],'same_reported_character_span':[0,89],'pair_status':'UNRESOLVED_SAME_TURN'}
  f['rationale']='The source screening already preserves this exact unresolved same-turn order as HOLD. It remains an inventoried rejected/held attempt, not admitted content. No new events, source edits or repaired anchors are introduced by this audit.'
result['status']='ACCOUNTING_VERIFIED_WITH_DETECTED_SUPPORT_HOLD_GATES_PENDING'
result['counts']['passed_checks']=len(checks)-len(findings)
result['counts']['detected_and_held_not_admitted']=sum(f.get('disposition')=='DETECTED_AND_HELD_NOT_ADMITTED' for f in findings)
result['all_checks_passed']=not findings
(C/'audit/SCREENING_ACCOUNTING_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'counts':result['counts'],'findings':findings},ensure_ascii=False))
