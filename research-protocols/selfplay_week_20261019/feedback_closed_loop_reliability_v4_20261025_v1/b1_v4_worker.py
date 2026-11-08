#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time,traceback
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rows=[json.loads(x) for x in Path(a.requests).read_text().splitlines() if x.strip()]; records=[]; started=time.time(); meta={}
 try:
  import torch,transformers,vllm
  from transformers import AutoTokenizer
  from vllm import LLM,SamplingParams
  model='/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model'; tokenizer='/root/autodl-tmp/Qwen3-4B'; tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
  # Qwen's no-thinking template already opens the final-answer channel.  The
  # former manually-appended think block made the model stop immediately with
  # an empty visible completion, so never append another assistant prefix.
  prompts=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
  llm=LLM(model=model,tokenizer=tokenizer,dtype='bfloat16',max_model_len=16384,gpu_memory_utilization=.85,max_num_seqs=8,seed=0,enable_prefix_caching=False,enforce_eager=True,disable_log_stats=True)
  for i in range(0,len(rows),8):
   batch=rows[i:i+8]; params=[SamplingParams(temperature=.8,top_p=.9,top_k=-1,repetition_penalty=1.0,max_tokens=6144,min_tokens=1,seed=r['sampling_seed']) for r in batch]
   outs=llm.generate(prompts[i:i+8],params,use_tqdm=False)
   for r,o in zip(batch,outs): records.append({'request_id':r['request_id'],'text':o.outputs[0].text if o.outputs else '', 'returned':bool(o.outputs and o.outputs[0].text),'finish_reason':o.outputs[0].finish_reason if o.outputs else None,'gold_read':False})
  meta={'backend':'vllm','model':model,'tokenizer':tokenizer,'mode':'FULL THINK_OFF','torch':torch.__version__,'vllm':vllm.__version__,'transformers':transformers.__version__}
 except Exception as e: meta={'backend':'error','error':repr(e),'traceback':traceback.format_exc(),'mode':'FULL THINK_OFF'}
 out={'prepared_count':len(rows),'returned_count':len(records),'records':records,'backend':meta,'started_at':started,'finished_at':time.time(),'gold_read':False,'training':False}
 q=Path(a.out);q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,q)
 return 0 if len(records)==len(rows) and all(x['returned'] for x in records) else 1
if __name__=='__main__':raise SystemExit(main())
