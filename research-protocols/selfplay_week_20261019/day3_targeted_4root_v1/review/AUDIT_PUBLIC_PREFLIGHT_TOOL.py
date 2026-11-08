"""Independent PUBLIC-ONLY preflight. Never opens author/private/answer data."""
from pathlib import Path
import argparse,collections,hashlib,importlib.metadata,itertools,json,math,random,sys
from datetime import datetime,timezone
P=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_targeted_4root_v1')
R=P/'runtime/direct_v1'
checks=[]
def ck(name,ok):
 checks.append({'name':name,'pass':bool(ok)})
 if not ok: raise RuntimeError(name)
def read(p):
 p=Path(p).resolve()
 ck('public_read_path:'+str(p),p.is_relative_to(R) and not any(x.lower() in ('private','authors','answers') or x.lower().startswith('private_') for x in p.parts))
 return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def verify(b,public=False):
 p=Path(b['path']).resolve()
 if public: ck('frozen_public_path:'+str(p),p.is_relative_to(R) and not any(x.lower().startswith(('private','answers','authors')) for x in p.parts))
 ck('binding:'+str(p),p.is_file() and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'])
def seed(qid,v):return int.from_bytes(hashlib.sha256(f'2026102004|{qid}|{v}'.encode()).digest()[:8],'big')
def variants(q):
 typ=q['question_type'];field='options' if typ=='multi_select' else 'items';keys=list(q[field])
 canonical={k:'content_'+hashlib.sha256(t.strip().encode()).hexdigest()[:16] for k,t in q[field].items()}
 content=list(canonical.values());texts={canonical[k]:q[field][k] for k in keys}
 random.Random(seed(q['candidate_id'],-1)).shuffle(content)
 used=set();result=[]
 for i in range(6 if typ=='multi_select' else min(12,math.factorial(len(keys)))):
  rng=random.Random(seed(q['candidate_id'],i))
  while True:
   order=content[:];rng.shuffle(order)
   if tuple(order) not in used:break
  used.add(tuple(order))
  result.append({'variant_id':f'perm_{i:02d}',field:{keys[p]:texts[c] for p,c in enumerate(order)},'display_to_content':dict(zip(keys,order)),'canonical_order_content_ids':content[:],'display_order_content_ids':order,'content_to_text':texts})
 return canonical,result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 ck('expected_venv',Path(sys.prefix).resolve()==Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b'))
 ck('python_B',sys.dont_write_bytecode)
 ck('run_exists',R.is_dir());ck('no_started_or_results',not any((R/n).exists() for n in ('STARTED.json','SCORES.json','RAW_RESPONSE_FREEZE.json','SCORING_STARTED.json','COMPLETE.json','FAILED.json','responses','batches')))
 freeze=read(R/'PREFLIGHT_FREEZE.json');ck('awaiting_preflight',freeze['status']=='PREPARED_AWAITING_INDEPENDENT_PREFLIGHT' and freeze['inference_started'] is False)
 for b in freeze['files']:verify(b,True)
 public=read(R/'PUBLIC_QUESTIONS.json');manifest=read(R/'manifest.json');policy=read(R/'PROTOCOL.json');admission=read(R/'CONTENT_ADMISSION.json');maps=read(R/'DISPLAY_MAPPINGS.json');tokids=read(R/'PROMPT_TOKEN_IDS.json')
 ck('wrapper_public_only',set(public)=={'questions'})
 qs=public['questions'];solver=policy['solver'];budget=policy['budget'];rows=manifest['requests'];slots=policy['slots'];slotmap={s['candidate_id']:s for s in slots}
 ck('namespace',policy['namespace']==manifest['namespace']==admission['namespace']=='train.feedback.targeted4root_v1')
 fixed={'name':'S0_M_clean','model':'/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model','tokenizer':'/root/autodl-tmp/Qwen3-4B','MC_mappings':6,'MC_samples_each_mapping':3,'ordering_mappings':12,'ordering_samples_each_mapping':1,'temperature':.8,'top_p':.9,'max_tokens':32,'enable_thinking':False,'max_model_len':16384,'source_truncation':False,'automatic_retry':False,'display_seed':2026102004,'sampling_seed':2026102040}
 ck('fixed_solver_contract',solver==fixed)
 expectedids={f'T4R-R{r}-{task}{n}' for r in range(1,5) for task in ('MC','ORD') for n in range(1,4)}
 ck('24unique_slots',len(slots)==len(slotmap)==budget['production_planned_slots']==24 and {s['candidate_id'] for s in slots}==expectedids and len({s['slot_id'] for s in slots})==24)
 ck('four_planned_roots',len({s['root_id'] for s in slots})==4 and manifest['planned_root_count']==4)
 ck('admitted_count',len(qs)==len({q['candidate_id'] for q in qs})==budget['initial_question_slots']==manifest['question_count']==manifest['planned_question_slots']==admission['admitted_count'])
 ck('admission_set',admission['status']=='PASS' and set(admission['candidate_ids'])=={q['candidate_id'] for q in qs}<=set(slotmap) and admission['planned_slots']==24)
 missing=sorted(set(slotmap)-{q['candidate_id'] for q in qs})
 ck('production_denominator',manifest['production_denominator']==24 and manifest['production_admitted_numerator']==len(qs) and manifest['production_unadmitted_slots']==len(missing) and manifest['missing_candidate_ids']==admission['missing_candidate_ids']==missing)
 ck('backend_settings',manifest['branch']=='direct' and manifest['solver_model']=='S0_M_clean' and manifest['model']==fixed['model'] and manifest['tokenizer']==fixed['tokenizer'] and manifest['batch_size']==8 and manifest['max_model_len']==16384 and manifest['gpu_memory_utilization']==.8 and manifest['prefix_caching'] is False and manifest['automatic_retry'] is False and manifest['source_truncation'] is False)
 ck('no_training_or_finaleval',manifest['training_started'] is False and manifest['use_as_final_eval'] is False)
 ck('gold_boundary_declarations',manifest['gold_opened_by_prepare'] is False and manifest['gold_parsed_by_prepare'] is False and manifest['gold_bytes_copied_by_prepare'] is True)
 ck('request_count_and_ids',len(rows)==len({r['request_id'] for r in rows})==budget['max_direct_requests']==manifest['request_count'] and len({r['seed'] for r in rows})==len(rows))
 ck('public_artifact_sets',set(maps['requests'])==set(tokids)=={r['request_id'] for r in rows} and set(maps['public_to_canonical'])=={q['candidate_id'] for q in qs})
 sys.path.insert(0,str(P/'runtime'));import adapter
 contract,native_solver=adapter.native()
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(fixed['tokenizer'],local_files_only=True)
 ck('decoder_contract',manifest['decoder_contract']==native_solver.generation_config('multi_select'))
 reconstructed=[];roots={};arity=collections.Counter();prompt_lengths=[]
 for q in qs:
  qid=q['candidate_id'];typ=q['question_type'];field='options' if typ=='multi_select' else 'items'
  ck(qid+':public_whitelist',set(q)=={'candidate_id','root_id','question_type','question',field,'source_path','source_sha256'})
  ck(qid+':public_strings',all(isinstance(q.get(k),str) and q[k].strip() for k in ('candidate_id','root_id','question_type','question','source_path','source_sha256')))
  labels=tuple('ABCDEF') if typ=='multi_select' else tuple(f'I{i+1}' for i in range(len(q[field])))
  ck(qid+':labels',typ in ('multi_select','ordering') and (len(q[field])==6 if typ=='multi_select' else len(q[field]) in (3,4,5,6)) and tuple(q[field])==labels)
  ck(qid+':nonempty_unique_text',all(isinstance(t,str) and t.strip() for t in q[field].values()) and len({' '.join(t.casefold().split()) for t in q[field].values()})==len(labels))
  slot=slotmap[qid];ck(qid+':slot',slot['root_id']==q['root_id'] and slot['question_type']==typ and slot['item_count']==len(labels))
  sp=R/'sources'/(q['root_id']+'.json');source=read(sp)
  ck(qid+':source_hash',sha(sp)==q['source_sha256'])
  ck(qid+':source_schema',source['root_id']==q['root_id'] and isinstance(source['turns'],list) and source['turns'] and all(all(isinstance(t.get(k),str) and t[k].strip() for k in ('turn_ref','speaker','text')) for t in source['turns']) and len({t['turn_ref'] for t in source['turns']})==len(source['turns']))
  ck(qid+':one_source_per_root',q['root_id'] not in roots or roots[q['root_id']]['sha256']==q['source_sha256']);roots[q['root_id']]={'turns':len(source['turns']),'sha256':q['source_sha256']};arity[typ+':'+str(len(labels))]+=1
  canonical,vs=variants(q);ck(qid+':canonical_map',maps['public_to_canonical'][qid]==canonical)
  for v in vs:
   prompt=contract.build_prompt(tokenizer,{'question_type':typ,'question':q['question']},{field:v[field]},contract.dialogue_rows(source),context_mode='full_context')
   ids=tokenizer.encode(prompt,add_special_tokens=False)
   ck(qid+v['variant_id']+':full_source',all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns']))
   ck(qid+v['variant_id']+':full_context_fit',len(ids)+32<=16384)
   for sample in range(3 if typ=='multi_select' else 1):
    rid=f"{qid}__{v['variant_id']}__direct__s{sample:02d}"
    ck(rid+':mapping',maps['requests'][rid]==v)
    ck(rid+':prompt_and_tokens',(R/'prompts'/(rid+'.txt')).read_text()==prompt and tokids[rid]==ids)
    reconstructed.append({'request_id':rid,'candidate_id':qid,'root_id':q['root_id'],'question_type':typ,'branch':'direct','variant_id':v['variant_id'],'display':{field:v[field]},'slot':sample,'seed':2026102040+len(reconstructed),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_tokens':len(ids),'source_turns':len(source['turns']),'source_sha256':q['source_sha256'],'max_output_tokens':32})
    prompt_lengths.append(len(ids))
 ck('complete_independent_schedule',reconstructed==rows)
 ck('source_and_arity_counts',manifest['root_count']==len(roots) and manifest['question_arity_counts']==dict(arity) and manifest['question_type_counts']==dict(collections.Counter(q['question_type'] for q in qs)))
 ck('manifest_prompt_headroom',manifest['max_prompt_tokens']==max(prompt_lengths,default=0) and manifest['minimum_context_headroom']==min((16384-n-32 for n in prompt_lengths),default=None))
 for b in manifest['public_input_bindings']:verify(b,True)
 for b in manifest['code_bindings']:verify(b)
 for b in manifest['model_bindings']:verify(b)
 versions={k:importlib.metadata.version(k) for k in ('transformers','torch','vllm','tokenizers')};ck('library_versions',versions==manifest['versions'])
 result={'schema_version':'targeted4root_independent_public_preflight_v1','status':'PASS','created_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/frozen_scoring','review_type':'NON_BLIND_SHARED_WORKSPACE_PUBLIC_ONLY','prior_runtime_implementation_involvement_disclosed':True,'private_gold_author_answers_opened':False,'inference_calls':0,'prepared_run_modified':False,'scope':'Public frozen artifacts and independent seed/shuffle/request/prompt/token reconstruction; does not assess Gold or semantic correctness. Public-only review follows earlier runtime implementation involvement.','checks_total':len(checks),'checks':checks,'questions':len(qs),'planned_slots':24,'planned_roots':4,'admitted_roots':len(roots),'requests':len(rows),'missing_candidate_ids':missing,'question_arity_counts':dict(arity),'sources':roots,'max_prompt_tokens':max(prompt_lengths,default=0),'minimum_context_headroom':min((16384-n-32 for n in prompt_lengths),default=None),'model_files_rehashed':len(manifest['model_bindings']),'versions':versions,'audited_files':[bind(R/n) for n in ('PREFLIGHT_FREEZE.json','manifest.json','PROTOCOL.json','CONTENT_ADMISSION.json','PUBLIC_QUESTIONS.json','DISPLAY_MAPPINGS.json','PROMPT_TOKEN_IDS.json')],'auditor_code':bind(Path(__file__))}
 with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'PASS','checks':len(checks),'requests':len(rows),'questions':len(qs),'output':bind(args.output)},ensure_ascii=False))
if __name__=='__main__':main()
