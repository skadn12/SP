"""CPU-only exact prompt length check, before expensive execution preparation."""
import argparse,hashlib,importlib.util,json,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('closed_loop_preflight_engine',P/'solver_runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
 a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('output');o=a.parse_args()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(m.orig.TOKENIZER),local_files_only=True)
 cs=json.loads(Path(o.input).read_text());rows=[]
 for c in cs:
  source=json.loads(Path(c['source_binding']['path']).read_text())
  for index,public,presentation in m.display_variants(c):
   field='options' if c['question_type']=='multi_select' else 'items'
   row={k:v for k,v in public.items() if k!=field};row.update(condition='FULL',cohort='new',display=public[field],labels=list(public[field]))
   old=m.orig.diagnostic.make_prompt(tok,row,row['display'],source)
   match=re.fullmatch(r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n',old,re.S);assert match
   system,user=match.groups()
   oldtail='Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.'
   newtail='Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.'
   assert user.count(oldtail)==1
   messages=[dict(role='system',content=system.replace('Give concise, verifiable explanations','In the final response, give concise, verifiable explanations')),dict(role='user',content=user.replace(oldtail,newtail))]
   prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   tokens=len(tok.encode(prompt,add_special_tokens=False));rows.append(dict(candidate_id=c['candidate_id'],variant=index,prompt_tokens=tokens,pass_budget=tokens+6144<=16384))
 result={'status':'PASS' if all(r['pass_budget'] for r in rows) else 'FAIL','input_binding':m.bind(Path(o.input)),'runner_binding':m.bind(P/'solver_runner.py'),'max_prompt_tokens':max(r['prompt_tokens'] for r in rows),'prompt_budget':10240,'max_output_tokens':6144,'context':16384,'source_truncation':False,'gpu_started':False,'records':rows}
 m.save(Path(o.output),result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','input_binding','runner_binding']}))
if __name__=='__main__':main()
