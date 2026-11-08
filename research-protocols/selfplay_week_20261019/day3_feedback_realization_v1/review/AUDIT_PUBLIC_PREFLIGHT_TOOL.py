#!/usr/bin/env python3
"""Public execution-input audit only; no Gold/author files, outputs, scores or GPU."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parents[1]
R=H/'runtime/eval_v1'
checks=[]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def ck(ok,name,detail=''):checks.append({'name':name,'pass':bool(ok),'detail':str(detail)})
def vb(b):
 p=Path(b['path']);ck(p.is_file(),'binding_exists',p)
 if p.is_file():ck(p.stat().st_size==b['bytes'],'binding_bytes',p);ck(sha(p)==b['sha256'],'binding_sha256',p)
frozen=read(R/'PREFLIGHT_FREEZE.json');m=read(R/'manifest.json');policy=read(R/'PROTOCOL.json');parent_policy=read(H/'EXECUTION_POLICY.json')
# Only public source bindings are consumed from the authoring freeze; no private
# answer labels or author evidence are opened by this audit.
gen_source_bindings=read(H/'GENERATION_FREEZE.json')['sources']
ck(policy==parent_policy,'frozen_policy_matches_parent')
ck(m['status']==frozen['status']=='PREPARED_AWAITING_INDEPENDENT_PREFLIGHT' and frozen['inference_started'] is False,'frozen_before_inference')
ck(not (R/'direct').exists() and not (R/'evidence').exists(),'branches_not_started')
for b in frozen['files']:
 ck(Path(b['path']).resolve().is_relative_to(R),'frozen_input_local_public',b['path'])
 ck('PRIVATE' not in Path(b['path']).name.upper(),'no_private_input_file',b['path']);vb(b)
for group in (m['public_input_bindings'],m['model_bindings'],m['code_bindings'],gen_source_bindings):
 for b in group:vb(b)
sys.path.insert(0,str(H/'runtime'))
spec=importlib.util.spec_from_file_location('realization_public_audit_adapter',H/'runtime/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
ev.c.environment();ev.policy_check(policy)
authoritative_models=ev.c.rt.models()
ck(m['model_bindings']==authoritative_models,'model_bindings_match_prior_authoritative_S0')
ck(m['model']==str(ev.c.MODEL) and m['tokenizer']==str(ev.c.TOKENIZER) and m['solver_model']=='S0_M_clean','same_model_identity')
ck(m['versions']==ev.c.rt.versions() and m['python_prefix']==sys.prefix,'library_versions_python_identity')
_,native=ev.c.a.native();decoder=native.generation_config('multi_select')
ck(m['decoder_contract']==decoder,'native_direct_decoder_contract')
public=read(R/'PUBLIC_QUESTIONS.json')['questions'];qby={q['candidate_id']:q for q in public};admission=read(R/'CONTENT_ADMISSION.json')
ck(len(public)==len(qby)==11 and admission['admitted_count']==11,'11_admitted_questions')
ck(admission['planned_slots']==m['planned_slots']==len(policy['planned_slots'])==12,'planned12_denominator_preserved')
ck(set(qby)==set(admission['candidate_ids']),'admitted_public_ids')
ck('FBR-R4-MC2' not in qby and set(s['candidate_id'] for s in policy['planned_slots'])-set(qby)=={'FBR-R4-MC2'},'known_hold_excluded_no_replacement')
ck(Counter(q['question_type'] for q in public)=={'multi_select':7,'ordering':4},'7MC_4ordering')
ck(len({q['root_id'] for q in public})==4 and {q['root_id'] for q in public}==set(policy['allowed_roots']),'four_fixed_roots')
maps=read(R/'DISPLAY_MAPPINGS.json');ids=read(R/'PROMPT_TOKEN_IDS.json');rows=m['requests']
ck(len(rows)==len({r['request_id'] for r in rows})==77,'77_unique_requests')
ck(Counter(r['branch'] for r in rows)==m['request_counts']=={'direct':66,'evidence':11},'66direct11diagnostic')
ck(policy['budget']=={'max_direct_calls':72,'max_diagnostic_calls':12},'max72plus12_frozen_budget')
ck(set(ids)==set(maps['requests'])=={r['request_id'] for r in rows},'complete_token_map_request_sets')
ck(policy['source_truncation'] is False and policy['automatic_retry'] is False and not m['training_started'] and not m['use_as_final_eval'],'no_retry_truncation_training_finaleval')
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(str(ev.c.TOKENIZER),local_files_only=True)
source_by_hash={b['sha256']:b for b in gen_source_bindings};direct_seed_index=0
for q in public:
 cid=q['candidate_id'];field='options' if q['question_type']=='multi_select' else 'items'
 ck(set(q)<={'candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'},'public_whitelist_no_gold_family_author_evidence',cid)
 ev.c.a.public_question(q)
 sourcepath=R/'sources'/(q['root_id']+'.json');source=read(sourcepath)
 ck(sha(sourcepath)==q['source_sha256'] and q['source_sha256'] in source_by_hash,'source_matches_pre_generation_public_binding',cid)
 ck(source['root_id']==q['root_id'] and len(source['turns'])==len({t['turn_ref'] for t in source['turns']}),'source_unique_turns',cid)
 canonical=ev.c.a.content_ids(q);ck(maps['public_to_canonical'][cid]==canonical,'canonical_content_ids_recomputed',cid)
 variants=ev.c.a.variants(q,policy['direct']['display_seed'],3)
 subset=[r for r in rows if r['candidate_id']==cid];ds=[r for r in subset if r['branch']=='direct'];es=[r for r in subset if r['branch']=='evidence']
 ck(len(ds)==6 and len(es)==1,'all_admitted_6direct1diag',cid)
 ck(len({tuple(v['display_order_content_ids']) for v in variants})==3,'three_distinct_gold_blind_permutations',cid)
 for variant in variants:
  chosen=[r for r in ds if r['variant_id']==variant['variant_id']]
  ck(len(chosen)==2 and {r['slot'] for r in chosen}=={0,1},'two_samples_each_permutation',cid+variant['variant_id'])
  direct_prompt=ev.c.a.direct_prompt(tok,q,variant,source)
  for row in chosen:
   ck(row['seed']==policy['direct']['sampling_seed']+direct_seed_index,'sequential_direct_seed',row['request_id']);direct_seed_index+=1
   ck(maps['requests'][row['request_id']]==variant and row['display']=={field:variant[field]},'direct_frozen_display_reconstructed',row['request_id'])
   ck((R/'prompts'/(row['request_id']+'.txt')).read_text()==direct_prompt,'direct_public_prompt_reconstructed',row['request_id'])
   prm=ev.params(row,decoder['structured_output_api']);constraint=getattr(prm,'structured_outputs',None) or getattr(prm,'guided_decoding',None)
   ck(constraint is not None and getattr(constraint,'choice',None)==ev.c.a.choices(row['question_type'],row['display'][field]),'actual_native_finite_answer_choices',row['request_id'])
   ck(prm.max_tokens==32 and prm.stop==['</box>'] and prm.include_stop_str_in_output,'direct32_stopbox',row['request_id'])
 e=es[0];base=next(r for r in ds if r['variant_id']=='perm_00' and r['slot']==0)
 ck(e['variant_id']=='perm_00' and e['slot']==0 and e['direct_request_id']==base['request_id'] and e['seed']==base['seed'] and e['display']==base['display'],'diagnostic_fixed_paired_mapping_seed',cid)
 ck(e['max_output_tokens']==3072 and e['json_schema']==ev.evidence.schema(q),'diagnostic3072_posthoc_schema',cid)
 ck((R/'prompts'/(e['request_id']+'.txt')).read_text()==ev.evidence.prompt(tok,q,variants[0],source),'diagnostic_public_prompt_reconstructed',cid)
 prm=ev.params(e,decoder['structured_output_api']);ck(prm.max_tokens==3072 and getattr(prm,'structured_outputs',None) is None and getattr(prm,'guided_decoding',None) is None and not prm.stop,'diagnostic_actual_api_unconstrained_no_stopbox',cid)
 for row in subset:
  rid=row['request_id'];variant=maps['requests'][rid];mapping=variant['display_to_content']
  ck(set(mapping)==set(q[field]) and set(mapping.values())==set(canonical.values()) and len(set(mapping.values()))==len(mapping),'display_bijection',rid)
  reverse={v:k for k,v in canonical.items()}
  ck(all(row['display'][field][label]==q[field][reverse[content]] for label,content in mapping.items()),'display_to_content_to_original_text_identity',rid)
  text=(R/'prompts'/(rid+'.txt')).read_text()
  ck(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in text for t in source['turns']),'every_full_source_turn_present',rid)
  tk=tok.encode(text,add_special_tokens=False)
  ck(tk==ids[rid] and len(tk)==row['prompt_tokens'],'independent_actual_tokenization',rid)
  ck(hashlib.sha256(text.encode()).hexdigest()==row['prompt_sha256'],'actual_prompt_hash',rid)
  ck(row['prompt_tokens']+row['max_output_tokens']<=16384,'zero_context_truncation_with_frozen_output_budget',rid)
  ck(row['source_turns']==len(source['turns']) and row['source_sha256']==q['source_sha256'],'row_source_identity',rid)
for branch in ('direct','evidence'):
 subset=[r for r in rows if r['branch']==branch]
 computed={'min_prompt_tokens':min(r['prompt_tokens'] for r in subset),'max_prompt_tokens':max(r['prompt_tokens'] for r in subset),'min_context_headroom':min(16384-r['prompt_tokens']-r['max_output_tokens'] for r in subset)}
 ck(computed==m['branch_limits'][branch],'manifest_token_headroom_recomputed',branch)
 ck(all(r['max_output_tokens']==(32 if branch=='direct' else 3072) for r in subset),'uniform_branch_output_budget',branch)
# This static review certifies separation of prepare/live from scorer entry points,
# rather than claiming process isolation from authors sharing this filesystem.
tree=ast.parse((H/'runtime/evaluate.py').read_text())
live=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='live')
ck(not any(isinstance(n,ast.Name) and n.id in {'golds','private','bindings'} for n in ast.walk(live)),'live_function_no_private_gold_read_path')
score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
score_text=ast.get_source_segment((H/'runtime/evaluate.py').read_text(),score)
ck(score_text.index("for branch in ('direct','evidence')")<score_text.index("golds={"),'both_raw_freezes_before_private_gold_parse')
errors=[x for x in checks if not x['pass']]
audit_paths=[R/'PREFLIGHT_FREEZE.json',R/'manifest.json',R/'PROTOCOL.json',R/'PUBLIC_QUESTIONS.json',R/'DISPLAY_MAPPINGS.json',R/'PROMPT_TOKEN_IDS.json',R/'CONTENT_ADMISSION.json',H/'EXECUTION_POLICY.json',H/'GENERATION_FREEZE.json',H/'runtime/evaluate.py',Path(__file__)]
result={'schema_version':'feedback_realization_public_preflight_v1','status':'PASS' if not errors else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'checks_count':len(checks),'errors':errors,'error_count':len(errors),'audited_files':[bind(p) for p in audit_paths],
 'review_role_independent_from_root_execution':True,'reviewer_also_adapter_author':True,'independent_code_author_claim':False,'technical_blinding_claim':False,
 'private_gold_labels_author_evidence_raw_or_scores_opened':False,'exposure_disclosure':'Reviewer wrote this adapter and knows prior pilot cases. Before audit, inspecting GENERATION_FREEZE metadata displayed planned target Gold cardinalities; no private Gold labels, private evidence, raw answers or scores for this batch were opened. Audit consumes its public source bindings only.',
 'planned_slots':12,'admitted_questions':11,'held_slot':'FBR-R4-MC2','direct_calls':66,'diagnostic_calls':11,'max_frozen_calls':84,'solver_calls_by_auditor':0,'gpu_started':False,
 'branch_limits':m['branch_limits'],'model_files_verified':len(m['model_bindings']),'model_bytes_verified':sum(b['bytes'] for b in m['model_bindings']),
 'interpretation':'GO for this immutable input set: 66 direct calls and 11 structured-evidence diagnostic calls. Semantic quality, target realization and feedback utility require separate postrun review.',
 'limits':['Role separation is procedural; reviewer also authored runtime adapter, so this is not an independent implementation audit.','Finite-choice direct decoding and unconstrained diagnostic decoding have distinct validity and output budgets.','All failed/invalid terminal outcomes must remain in fixed request denominators.']}
out=H/'review/INDEPENDENT_PUBLIC_PREFLIGHT.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'checks','errors','audited_files'}},ensure_ascii=False))
if errors:raise SystemExit(2)
