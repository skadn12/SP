#!/usr/bin/env python3
"""Independent postraw recomputation, not a semantic audit. Writes only this review."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import jsonschema
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parents[1]
R=H/'runtime/evidence_v1'
P=H.parent/'day3_targeted_4root_v1'
checks=[]
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def binding(p):return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def ck(ok,name,detail=''):checks.append({'name':name,'pass':bool(ok),'detail':str(detail)})
def vb(b):
 p=Path(b['path']);ck(p.is_file(),'bound_file_exists',p)
 if p.is_file():ck(p.stat().st_size==b['bytes'],'bound_bytes',p);ck(sha(p)==b['sha256'],'bound_sha256',p)

baseline=read(H/'DIRECT_BASELINE_BINDING.json')
ck(sha(P/'FINAL_MANIFEST.json')=='44f5d62179affbe5125477eb83754c6eb3f0a4700619722e1f07d45fc54992e2','sealed_manifest_expected_sha')
old_manifest=read(P/'FINAL_MANIFEST.json')
ck(len(old_manifest['files'])==971,'sealed_manifest_971_files')
for b in old_manifest['files']:vb(b)
for b in baseline['bindings']:vb(b)
ck(baseline['calls']==342 and baseline['unchanged'] is True,'old_342_primary_preserved')
direct_rows=rows(P/'exports/SOLVER_RESPONSES.jsonl')
ck(len(direct_rows)==342 and len({r['request_id'] for r in direct_rows})==342,'old_342_unique_raw_records')
direct={r['request_id']:r for r in direct_rows}
raw_freeze=read(R/'RAW_RESPONSE_FREEZE.json')
preflight=read(R/'PREFLIGHT_FREEZE.json')
manifest=read(R/'manifest.json')
for b in preflight['files']+raw_freeze['files']+raw_freeze['execution_ledger_bindings']+[raw_freeze['manifest_binding'],raw_freeze['preflight_binding']]:vb(b)
ck(raw_freeze['status']=='COMMITTED' and raw_freeze['planned_request_count']==23,'raw_frozen_23')
ck(raw_freeze['automatic_retry'] is False and raw_freeze['source_truncation'] is False and raw_freeze['gold_opened_by_live'] is False,'raw_contract_boundaries')
reqs=manifest['requests'];req_by={r['request_id']:r for r in reqs}
ck(len(reqs)==len(req_by)==23,'fixed_23_unique_requests')
ck(all(r['variant_id']=='perm_00' and r['slot']==0 for r in reqs),'fixed_perm00_sample00')
complete=read(R/'COMPLETE.json')
ck(complete['status']=='COMPLETE' and complete['actual_submitted_requests']==complete['returned_requests']==23,'complete_23_returned')
ck(not (R/'FAILED.json').exists(),'no_failed_live_marker')
actual_files={p.name for p in (R/'responses').glob('*.json')}
ck(actual_files=={r+'.json' for r in req_by},'raw_terminal_exact_set')
submitted=[]
for b in sorted((R/'batches').glob('*_SUBMITTED.json')):
 x=read(b);submitted+=x['request_ids'];ck(x['automatic_retry'] is False,'batch_no_retry',b.name)
 ck(len(x['request_ids'])==len(x['sampling_params']),'batch_params_count',b.name)
 for rid,prm in zip(x['request_ids'],x['sampling_params']):
  ck(isinstance(prm,str) and 'structured_outputs=None' in prm and 'stop=[]' in prm and 'max_tokens=2048' in prm,'no_constraint_actual_sampling',rid)
  ck(f"seed={req_by[rid]['seed']}" in prm,'actual_sampling_seed',rid)
 done=read(b.with_name(b.name.replace('_SUBMITTED','_COMPLETE')))
 ck(done['request_ids']==x['request_ids'],'batch_complete_matches_submission',b.name)
ck(len(submitted)==len(set(submitted))==23 and set(submitted)==set(req_by),'23_once_only_submitted')
ck(not list((R/'batches').glob('*EXTRA_OUTPUTS*')),'no_extra_outputs')
public={q['candidate_id']:q for q in read(R/'PUBLIC_QUESTIONS.json')['questions']}
original_public={q['candidate_id']:q for q in read(P/'exports/PUBLIC_QUESTIONS.json')['questions']}
gold={q['candidate_id']:q['gold'] for q in read(P/'exports/PRIVATE_QUESTIONS.json')['questions']}
maps=read(R/'DISPLAY_MAPPINGS.json');token_ids=read(R/'PROMPT_TOKEN_IDS.json')
parsed_saved={r['request_id']:r for r in read(R/'PARSED_RESPONSES.json')['questions']}
scores=read(H/'DIAGNOSTIC_SCORES.json');score_by={r['request_id']:r for r in scores['questions']}
normalized={r['request_id']:r for r in rows(H/'DIAGNOSTIC_RESPONSES.jsonl')}
coverage={r['request_id']:r for r in read(H/'COVERAGE_PLAN.json')['items']}
hist=H/'review/history/label_domain_correction'
historical_scores=read(hist/'DIAGNOSTIC_SCORES.json')
historical_norm=rows(hist/'DIAGNOSTIC_RESPONSES.jsonl')
ck(sha(hist/'DIAGNOSTIC_RESPONSES.jsonl')==sha(H/'DIAGNOSTIC_RESPONSES.jsonl'),'label_fix_did_not_modify_normalized_model_fields')
ck(all(not s['diagnostic_exact'] for s in historical_scores['questions']),'initial_zero_score_failure_preserved')
ck((hist/'assemble_diagnostics.py').is_file() and (hist/'COVERAGE_PLAN.json').is_file(),'initial_bug_files_preserved')
correction=read(H/'review/LABEL_DOMAIN_CORRECTION.json')
ck(correction['raw_or_prompt_or_model_modified'] is False and correction['new_solver_calls']==0 and correction['old342_direct_scores_modified'] is False,'correction_declarations')
totals=defaultdict(Counter);citation_total=quote_total=0;invalid_details=[];raw_hash_records=[]
for rid,r in req_by.items():
 rawpath=R/'responses'/(rid+'.json');raw=read(rawpath);raw_hash_records.append(binding(rawpath))
 ck(raw['request_id']==rid and raw['candidate_id']==r['candidate_id'],'raw_identity',rid)
 ck(raw['status']=='RETURNED' and isinstance(raw['raw_output'],str),'returned_text',rid)
 ck(raw['automatic_retry'] is False and raw['source_truncation'] is False and raw['gold_read_by_solver'] is False,'raw_no_retry_truncation_gold',rid)
 ck(raw['seed']==r['seed'] and raw['prompt_sha256']==r['prompt_sha256'],'raw_frozen_request_identity',rid)
 ck(raw['prompt_token_ids']==token_ids[rid] and len(raw['prompt_token_ids'])==r['prompt_tokens'],'full_frozen_prompt_tokens',rid)
 ck(not raw['output_validation_errors'],'no_prompt_or_output_validation_errors',rid)
 ck(len(raw['all_completions'])==1 and raw['raw_output']==raw['all_completions'][0]['text'],'one_verbatim_completion',rid)
 ck(raw['completion_token_ids']==raw['all_completions'][0]['token_ids'],'completion_token_ids',rid)
 meta=raw['metadata'];ck(meta['finish_reason']!='length' and meta['completion_token_count']<2048,'no_output_length_truncation',rid)
 ck(meta['constrained_decoding'] is False and meta['structured_output_api'] is None,'unconstrained_actual_output',rid)
 q=public[r['candidate_id']];oq=original_public[r['candidate_id']]
 ck(q==oq,'original_public_question_unchanged',rid)
 field='options' if q['question_type']=='multi_select' else 'items';labels=set(q[field])
 dm=maps['requests'][rid]['display_to_content'];cm=maps['public_to_canonical'][q['candidate_id']]
 inv={v:k for k,v in cm.items()};restore={k:inv[v] for k,v in dm.items()}
 ck(len(inv)==len(cm) and set(restore)==set(restore.values())==labels,'content_id_to_original_label_bijection',rid)
 ck(coverage[rid]['display_to_canonical']==restore,'fixed_coverage_original_label_domain',rid)
 for label,target in restore.items():ck(r['display'][field][label]==q[field][target],'display_original_text_identity',rid+':'+label)
 ck(set(gold[q['candidate_id']])<=labels,'gold_original_label_domain',rid)
 try:
  parsed=json.loads(raw['raw_output'].strip());jsonschema.validate(parsed,r['json_schema'])
  ck(set(i['label'] for i in parsed['item_evidence'])==labels and len(parsed['item_evidence'])==len(labels),'valid_item_coverage',rid)
  if len(parsed['item_evidence'])!=len(labels) or {i['label'] for i in parsed['item_evidence']}!=labels:raise ValueError('missing/duplicate item labels')
  valid=True
 except Exception as error:valid=False;invalid_details.append({'request_id':rid,'candidate_id':q['candidate_id'],'error_type':type(error).__name__,'reason':str(error)[:400]})
 saved=parsed_saved[rid];n=normalized[rid];s=score_by[rid]
 ck(valid==saved['format_valid']==n['diagnostic_valid']==s['format_valid'],'independent_strict_validity',rid)
 ck(n['raw_output']==raw['raw_output'],'normalized_verbatim_raw',rid);vb(n['raw_response_binding'])
 ck(n['training_use'] is False,'diagnostics_not_training_labels',rid)
 prediction=None
 if valid:
  prediction=[restore[z] for z in parsed['final_answer']]
  expected_normalized={'item_judgments':[{'label':i['label'],'judgment':i['judgment'],'evidence':[{'turn':e['turn_ref'],'quote':e['quote']} for e in i['citations']],'rationale':i['reason'],**({'evidence_status':'no_support_found'} if not i['citations'] else {})} for i in parsed['item_evidence']],'final_answer':parsed['final_answer']}
  ck(n['parsed']==expected_normalized,'all_normalized_fields_model_emitted',rid)
  ck(saved['canonical_prediction']==[dm[z] for z in parsed['final_answer']],'parse_content_id_domain_preserved',rid)
  ck(s['canonical_prediction']==prediction,'score_restores_original_public_labels',rid)
  source=read(R/'sources'/(r['root_id']+'.json'));texts={t['turn_ref']:t['text'] for t in source['turns']}
  cits=[(i['label'],e) for i in parsed['item_evidence'] for e in i['citations']]
  cq=sum(e['turn_ref'] in texts and e['quote'] in texts[e['turn_ref']] for _,e in cits)
  citation_total+=len(cits);quote_total+=cq
  ck(s['citation_count']==len(cits) and s['exact_quote_count']==cq,'citation_fidelity_recomputed',rid)
 else:
  ck(n['parsed'] is None and s['display_prediction'] is None and s['canonical_prediction'] is None and bool(n['invalid_reason']),'invalid_not_repaired_or_answer_extracted',rid)
 exact=bool(valid and (set(prediction)==set(gold[q['candidate_id']]) if field=='options' else prediction==gold[q['candidate_id']]))
 d=direct[r['direct_request_id']]
 direct_exact=(set(d['canonical_prediction'])==set(gold[q['candidate_id']]) if field=='options' else d['canonical_prediction']==gold[q['candidate_id']])
 ck(direct_exact==d['exact']==s['paired_direct_exact'],'paired_direct_exact_recomputed',rid)
 ck(d['display_to_canonical']==restore and d['request_id']==r['direct_request_id'],'paired_direct_identical_display_mapping',rid)
 ck(s['diagnostic_exact']==exact and s['canonical_gold']==gold[q['candidate_id']],'diagnostic_exact_gold_recomputed',rid)
 ck(s['same_answer']==(prediction==d['canonical_prediction']),'same_answer_field_recomputed',rid)
 ck(s['final_answer_extracted_from_invalid_output'] is False,'no_invalid_answer_salvage',rid)
 totals[q['question_type']].update(questions=1,valid=valid,invalid=not valid,diagnostic_exact=exact,paired_direct_exact=direct_exact,direct_wrong_diagnostic_correct=not direct_exact and exact,direct_correct_diagnostic_wrong=direct_exact and not exact)
ck(sum(t['valid'] for t in totals.values())==15 and sum(t['invalid'] for t in totals.values())==8,'strict_valid15_invalid8')
ck(totals['multi_select']['diagnostic_exact']==2 and totals['ordering']['diagnostic_exact']==7,'new_exact_MC2of12_ORD7of11')
ck(totals['multi_select']['paired_direct_exact']==5 and totals['ordering']['paired_direct_exact']==4,'paired_direct_MC5of12_ORD4of11')
for kind,t in totals.items():
 for k,value in scores['by_type'][kind].items():ck(t[k]==value,'aggregate_independent_recomputation',kind+':'+k)
ck(citation_total==80 and quote_total==48,'citation80_quote48_recomputed')
ck(scores['not_comparable_to_342_direct_as_equal_budget'] is True,'different_protocol_not_cot_causal_claim')
# Recheck raw after all score/correction reads to bind the unchanged inference evidence.
for b in raw_freeze['files']:vb(b)
audit_files=[H/'DIRECT_BASELINE_BINDING.json',H/'COVERAGE_PLAN.json',H/'DIAGNOSTIC_RESPONSES.jsonl',H/'DIAGNOSTIC_SCORES.json',H/'assemble_diagnostics.py',H/'review/LABEL_DOMAIN_CORRECTION.json',R/'RAW_RESPONSE_FREEZE.json',R/'PARSED_RESPONSES.json',R/'manifest.json',R/'COMPLETE.json',P/'FINAL_MANIFEST.json',Path(__file__)]
result={'schema_version':'independent_diagnostic_postrun_v1','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'checks_count':len(checks),'failure_count':sum(not x['pass'] for x in checks),'checks':checks,'audited_files':[binding(p) for p in audit_files],
 'sealed_pilot_files_verified':len(old_manifest['files']),'old_direct_calls_unchanged':342,'planned_calls':23,'actual_submitted_once':len(submitted),'returned':23,
 'strict_valid':15,'strict_invalid':8,'truncated':0,'by_type':{k:dict(v) for k,v in totals.items()},'citation_count':citation_total,'exact_quote_count':quote_total,
 'invalid_details':invalid_details,'raw_files_after_correction':raw_hash_records,'label_domain_correction_verified':True,'raw_modified_by_correction':False,
 'initial_derived_zero_score_retained_and_withdrawn':True,'normalization_model_field_fidelity_verified':True,'new_solver_calls_by_auditor':0,
 'reviewer_prior_exposure':'This agent inspected the first old private-question example while designing an interface before preflight. Postrun scoring explicitly reads private Gold only after RAW_RESPONSE_FREEZE. Reviewer was never content blind.',
 'semantic_review_status':'separate_in_progress_not_certified_here','limits':['This audit verifies raw integrity, mapping, strict format, and arithmetic; semantic validity of explanations is a separate review.','One mapping/sample and changed output budget/protocol support descriptive comparison only, not a causal CoT benefit estimate.','Shared filesystem gives procedural independence, not technical blinding.']}
out=H/'review/INDEPENDENT_DIAGNOSTIC_POSTRUN.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
archive={'status':'ARCHIVED','created_at_utc':datetime.now(timezone.utc).isoformat(),'script':binding(Path(__file__)),'audit':binding(out),'script_execution':'Run with .venv-local4b/bin/python -B','new_solver_calls':0}
(H/'review/AUDIT_DIAGNOSTIC_POSTRUN_TOOL_ARCHIVE.json').write_text(json.dumps(archive,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'checks','audited_files','invalid_details','raw_files_after_correction'}},ensure_ascii=False))
if result['status']!='PASS':raise SystemExit(2)
