#!/usr/bin/env python3
"""Independent CPU-only check of selected paired evidence-hint requests."""
import collections,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent.parent
H=B/'probes/evidence_hint';D=B/'solver/direct';checks=[]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def check(n,c):
 checks.append(n)
 if not c:raise AssertionError(n)
def vr(r):
 p=Path(r['path']);check('sha:'+str(p),p.is_file()and sha(p)==r['sha256'])
 if'bytes'in r:check('bytes:'+str(p),p.stat().st_size==r['bytes'])
def main():
 freeze=read(H/'PREFLIGHT_FREEZE.json');m=read(H/'manifest.json');dm=read(D/'manifest.json');ds=read(D/'SCORES.json');policy=read(B/'PROBE_SELECTION_POLICY.json');pf=read(B/'PROBE_POLICY_FREEZE.json')
 check('not_started_before_independent_audit',not(H/'STARTED.json').exists())
 check('policy_before_gold_scoring',pf['created_at_utc']<read(D/'SCORING_STARTED.json')['gold_first_read_at_utc']and pf['direct_scoring_started']is False)
 for r in pf['files']:vr(r)
 for r in freeze['files']:vr(r)
 for r in m['code_bindings']:vr(r)
 for r in m['input_bindings']:vr(r)
 vr(m['baseline_manifest']);vr(m['probe_policy']);vr(m['private_answer_binding'])
 check('direct_complete',ds['status']=='COMPLETE'and ds['denominator_calls']==582 and ds['denominator_questions']==39)
 public=read(D/'PUBLIC_QUESTIONS.json')['questions'];pub={q['candidate_id']:q for q in public};priv={q['candidate_id']:q for q in read(m['private_answer_binding']['path'])['questions']}
 eligible={r['candidate_id']for r in ds['requests']if r['valid']and not r['exact']};used=set();selected=[]
 for typ in policy['selection']['type_order']:
  candidates=sorted([q for q in public if q['question_type']==typ and q['candidate_id']in eligible],key=lambda q:hashlib.sha256(('2026101900|evidence-hint|'+q['candidate_id']).encode()).hexdigest());n=0
  for q in candidates:
   if q['root_id']in used:continue
   selected.append(q);used.add(q['root_id']);n+=1
   if n==policy['selection']['per_type_max']:break
 hp=read(H/'PUBLIC_QUESTIONS.json')['questions'];check('independent_hash_greedy_selection',hp==selected)
 check('distinct_train_roots_max6',len(hp)<=6 and len(used)==len(hp)and used<=set(read(B.parent/'day1/evaluation/ROOT_SELECTION.json')['wave1']['root_ids']))
 for key in ['model','tokenizer','solver_model','namespace','decoder_contract','random_display','batch_size','max_model_len','prefix_caching','automatic_retry','source_truncation','model_bindings','private_answer_binding','content_admission_binding']:
  check('held_fixed:'+key,m[key]==dm[key])
 check('diagnostic_purpose_not_benchmark',m['purpose']=='controlled_evidence_hint_probe'and m['use_as_final_eval']is False)
 oldrows={r['request_id']:r for r in dm['requests']};expectedold=[r for r in dm['requests']if r['candidate_id']in {q['candidate_id']for q in selected}]
 rows=m['requests'];check('all_permutation_sample_slots_retained',len(rows)==len(expectedold)<=90 and {r['baseline_request_id']for r in rows}=={r['request_id']for r in expectedold})
 check('exact_count6question90request',len(hp)==6 and len(rows)==90 and freeze['requests']==90 and freeze['questions']==6)
 oldmaps=read(D/'private/DISPLAY_MAPPINGS.json');maps=read(H/'private/DISPLAY_MAPPINGS.json');hints=read(H/'EVIDENCE_HINTS.json');pairs=read(H/'MATCHED_PAIRS.json')['pairs'];tokens=read(H/'PROMPT_TOKEN_IDS.json')
 check('pair_index_complete',len(pairs)==len(rows)and len({p['probe_request_id']for p in pairs})==len(rows))
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(m['tokenizer'],local_files_only=True)
 reminders={}
 for q in hp:
  cid=q['candidate_id'];field='options'if q['question_type']=='multi_select'else'items';ev=priv[cid]['option_evidence'if field=='options'else'item_evidence'];check('all_labels_evidence_covered:'+cid,set(ev)==set(q[field]))
  source=read(q['source_path']);check('original_source_hash:'+cid,sha(q['source_path'])==q['source_sha256'])
  check('source_copy_identical:'+cid,Path(q['source_path']).read_bytes()==(H/'sources'/f'{q["root_id"]}.json').read_bytes())
  byref={t['turn_ref']:t for t in source['turns']};refs=set()
  for label,e in ev.items():
   cites=e.get('citations')or[e.get('primary_citation')or e]
   for c in cites:
    ref=c['turn_ref'];check('evidence_turn_exists:'+cid+':'+label,ref in byref)
    if c.get('quote'):check('quote_verbatim:'+cid+':'+label,c['quote']in byref[ref]['text'])
    if c.get('speaker'):check('speaker_attribution:'+cid+':'+label,c['speaker']==byref[ref]['speaker'])
    refs.add(ref)
  relevant=[t for t in source['turns']if t['turn_ref']in refs]
  reminder='\n\nRelevant-source reminder (use the complete dialogue and evaluate every item independently; this reminder does not identify correct labels):\n'+'\n'.join(f'[{t["turn_ref"]}] {t["speaker"]}: {t["text"]}'for t in relevant)
  check('source_order_all_item_turn_union:'+cid,hints[cid]['turn_refs']==[t['turn_ref']for t in relevant]and hints[cid]['text']==reminder)
  check('no_author_relation_or_mapping_added:'+cid,hints[cid]['gold_relations_or_label_mapping_included']is False and hints[cid]['all_item_evidence_included']is True)
  check('canonical_map_unchanged:'+cid,maps['public_to_canonical'][cid]==oldmaps['public_to_canonical'][cid]);reminders[cid]=reminder
 for r in rows:
  rid=r['request_id'];old=oldrows[r['baseline_request_id']];cid=r['candidate_id']
  check('request_id_suffix:'+rid,rid==old['request_id']+'__evidence_hint')
  changed={'request_id','prompt_sha256','prompt_tokens'}
  check('all_original_conditions_fixed:'+rid,all(r[k]==v for k,v in old.items()if k not in changed))
  check('only_permitted_request_fields_added:'+rid,set(r)-set(old)=={'baseline_request_id','intervention'})
  check('private_variant_unchanged:'+rid,maps['requests'][rid]==oldmaps['requests'][old['request_id']])
  original=(D/'prompts'/f'{old["request_id"]}.txt').read_text();marker='<|im_end|>\n<|im_start|>assistant';check('unique_public_user_insertion_boundary:'+rid,original.count(marker)==1)
  pos=original.index(marker);expected=original[:pos]+reminders[cid]+original[pos:];actual=(H/'prompts'/f'{rid}.txt').read_text()
  check('exact_whitelisted_reminder_insertion:'+rid,actual==expected)
  check('prompt_hash:'+rid,hashlib.sha256(actual.encode()).hexdigest()==r['prompt_sha256'])
  actualids=tok.encode(actual,add_special_tokens=False);check('token_ids_count_exact:'+rid,actualids==tokens[rid]and len(actualids)==r['prompt_tokens'])
  check('no_truncation_overflow:'+rid,r['prompt_tokens']+r['max_output_tokens']<=m['max_model_len']and r['max_output_tokens']==32)
  pair=next(p for p in pairs if p['probe_request_id']==rid)
  check('matched_pair_identity:'+rid,pair=={'baseline_request_id':old['request_id'],'probe_request_id':rid,'candidate_id':cid,'root_id':r['root_id'],'same_seed':old['seed'],'same_variant_id':old['variant_id'],'slot':old['slot']})
 check('preflight_still_not_started',not(H/'STARTED.json').exists())
 report={'status':'PASS','auditor':'/root/day2_fact_memory_prep','created_at_utc':datetime.now(timezone.utc).isoformat(),'review_scope':'Independent paired request reconstruction; source-author evidence intervention explicitly disclosed. Shared model/filesystem internal audit.',
  'audited_files':[bind(H/'PREFLIGHT_FREEZE.json'),bind(H/'manifest.json'),bind(B/'PROBE_POLICY_FREEZE.json'),bind(Path(__file__))],
  'checks':len(checks),'check_names':checks,'questions':len(hp),'requests':len(rows),'new_solver_calls':0,
  'selected_roots':[q['root_id']for q in hp],
  'limits':['Author evidence selects relevant turns; not a Gold-blind benchmark.','Paired same seed is a controlled request condition, not a guarantee of identical GPU random streams.','Positive results support evidence-salience/instruction hypotheses only, not learned improvement or established internal causality.']}
 with(HERE_PATH:=B/'audit/INDEPENDENT_HINT_PREFLIGHT.json').open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in report.items()if k not in ['check_names','audited_files','limits']}))
if __name__=='__main__':main()
