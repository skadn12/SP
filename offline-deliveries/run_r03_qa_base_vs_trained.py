import json, hashlib, pathlib, sys, time
from collections import defaultdict
from transformers import AutoTokenizer
sys.path.insert(0,'/root/autodl-tmp/gxy-selfplay/current_method_github')
from scripts.current_method.restart_n5_spell.h4_qa_mining_probe_v1.solver_adapter import evaluate_candidates
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
from scripts.current_method.restart_n5_spell.ledger import RequestLedger

BASE=pathlib.Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1_live_2')
OUT=pathlib.Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r03_qa_base_vs_trained_eval_live_2')
TOKENIZER=pathlib.Path('/root/autodl-tmp/Qwen3-4B')
MODELS={'base':TOKENIZER,'trained':pathlib.Path('/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged')}

def load_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 # Use all structurally valid composition outputs and content-valid fresh outputs from frozen pilot.
 attempts=[]
 for fn in ['composition_attempt_results.jsonl','fresh_attempt_results.jsonl']:
  for row in load_jsonl(BASE/fn):
   c=row.get('closure',{}); o=row.get('parsed_output') or {}
   if not c.get('schema_valid'): continue
   if fn.startswith('fresh') and not c.get('content_valid'): continue
   if o.get('outcome')!='QA': continue
   qtype=o.get('question_type'); opts=o.get('options_or_items'); gold=o.get('canonical_gold')
   if qtype not in ('single_choice','multi_select','ordering') or not isinstance(opts,dict) or not gold: continue
   attempts.append((row,fn))
 # de-dupe by root/pass
 seen=set(); candidates_by_root=defaultdict(list)
 for row,fn in attempts:
  key=(str(row['root_id']),str(row['pass']))
  if key in seen: continue
  seen.add(key); o=row['parsed_output']; qtype=o['question_type']; opts={str(k):str(v) for k,v in o['options_or_items'].items()}; gold=[str(x) for x in o['canonical_gold']]
  cid=f"{row['root_id']}:{str(row['pass']).lower()}:attempt-001"
  cand={'candidate_id':cid,'root_id':str(row['root_id']),'question_type':qtype,'question':str(o['new_question']),'options':opts if qtype!='ordering' else {},'gold':gold,'evidence_refs':[str(x) for x in o.get('evidence_refs') or []],'generation_source':str(row['pass'])}
  if qtype=='single_choice': cand['correct_option']=gold[0]
  if qtype=='ordering': cand['items']=opts; cand.pop('options',None)
  candidates_by_root[cand['root_id']].append(cand)
 # source dialogues
 dm={str(x['root_id']):x for x in load_jsonl(BASE/'authoritative_dialogue_manifest.jsonl')}
 manifest={'version':'h4_r03_qa_base_vs_trained_eval_v1','source_pilot':str(BASE),'source_pilot_sha256':sha(BASE/'pilot_contract.json'),'models':{k:str(v) for k,v in MODELS.items()},'candidate_count':sum(map(len,candidates_by_root.values())),'candidate_ids':[c['candidate_id'] for cs in candidates_by_root.values() for c in cs],'semantic_admission':'NOT_PERFORMED; composition candidates structurally valid only; fresh candidates content-valid under pilot structural gate','model_calls':{'qa_writer':0,'memorax':0,'training':0}}
 (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,trust_remote_code=True,local_files_only=True)
 for name,model in MODELS.items():
  run=OUT/name; run.mkdir()
  ledger=RequestLedger(run, hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(), 'live')
  backend=VLLMSolverBackend(model_path=model,tokenizer_path=TOKENIZER,max_model_len=16384,temperature=0.7,top_p=0.9,max_tokens=64,gpu_memory_utilization=0.70)
  allres=[]; started=time.time()
  for root,cs in sorted(candidates_by_root.items()):
   expanded=json.loads(pathlib.Path(dm[root]['path']).read_text())
   res=evaluate_candidates(candidates=cs,expanded=expanded,run_root=run,ledger=ledger,tokenizer=tokenizer,backend=backend)
   allres.extend(res)
  # aggregate exact-match rates from variants
  rows=[]
  for r in allres:
   variants=r.get('variants') or []; total=sum(int(v.get('total',1)) for v in variants); correct=sum(int(v.get('correct',0)) for v in variants)
   bymode={}
   for mode in ('no_context','full_context'):
    vv=[v for v in variants if v.get('mode')==mode]; t=sum(int(v.get('total',1)) for v in vv); c=sum(int(v.get('correct',0)) for v in vv); bymode[mode]={'correct':c,'total':t,'accuracy':(c/t if t else None)}
   rows.append({'candidate_id':r['candidate_id'],'question_type':r['question_type'],'overall':{'correct':correct,'total':total,'accuracy':correct/total if total else None},'by_context':bymode})
  report={'status':'PASS','model_name':name,'model_path':str(model),'candidate_count':len(allres),'results':rows,'elapsed_seconds':time.time()-started,'request_count':len(ledger.completed()),'semantic_support':'UNKNOWN_NO_SEMANTIC_AUDITOR'}
  (run/'evaluation_results.json').write_text(json.dumps({'raw':allres,'summary':report},ensure_ascii=False,indent=2)+'\n')
  (run/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  backend.close()
 print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
