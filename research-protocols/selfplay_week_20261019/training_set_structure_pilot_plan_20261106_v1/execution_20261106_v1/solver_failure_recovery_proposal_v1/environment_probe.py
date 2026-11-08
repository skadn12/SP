#!/usr/bin/env python3
import sys,inspect,json,importlib.metadata as m
r={'executable':sys.executable,'python_version':sys.version,'prefix':sys.prefix,'vllm_version':m.version('vllm')}
try:
 from vllm import SamplingParams
 import vllm.sampling_params as p
 r['sampling_params_file']=p.__file__
 r['has_StructuredOutputsParams']=hasattr(p,'StructuredOutputsParams')
 r['SamplingParams_has_structured_outputs']='structured_outputs' in inspect.signature(SamplingParams).parameters
 if r['has_StructuredOutputsParams']:
  s=p.StructuredOutputsParams(json={'type':'object','properties':{'final_answer':{'type':'string'}},'required':['final_answer'],'additionalProperties':False})
  x=SamplingParams(n=1,temperature=.8,top_p=.9,top_k=-1,repetition_penalty=1.,max_tokens=6144,seed=2026110604,stop=None,structured_outputs=s)
  r['historical_sampling_constructor_compatible']=True
except Exception as e:r['error']={'type':type(e).__name__,'message':str(e)}
r.update(model_loaded=False,tokenizer_loaded=False,model_calls=0,LLM_instantiated=False)
print(json.dumps(r))

