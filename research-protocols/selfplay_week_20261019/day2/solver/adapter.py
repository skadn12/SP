"""Dynamic 4/5/6 public input adapter; v15 permutation algorithm, native prompts/decoder.

No private answer/evidence file is opened by this module. The original v15/v140
files stay unchanged. Only the cardinality restriction is generalized.
"""
from __future__ import annotations
import hashlib, importlib.util, math, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
WORKTREE = REPO / 'worktrees/h4-gate-downstream-v1'
PUBLIC_KEYS = {'candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'}
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKTREE))

def native():
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, solver
    return contract, solver

def native_scoring():
    key = 'day2_native_v15_scoring'
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,NATIVE/'scoring.py')
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules[key]

def labels(qtype, n):
    if qtype not in ('multi_select','ordering') or n not in (4,5,6):
        raise ValueError('ONLY_MULTI_ORDER_FOUR_TO_SIX')
    return tuple('ABCDEF'[:n]) if qtype=='multi_select' else tuple(f'I{i+1}' for i in range(n))

def public_question(q):
    if not isinstance(q,dict) or set(q)-PUBLIC_KEYS:
        raise ValueError('PUBLIC_QUESTION_WHITELIST_VIOLATION')
    for k in ('candidate_id','root_id','question_type','question','source_path','source_sha256'):
        if not isinstance(q.get(k),str) or not q[k].strip(): raise ValueError('INVALID_PUBLIC_FIELD:'+k)
    field='options' if q['question_type']=='multi_select' else 'items'
    if not isinstance(q.get(field),dict): raise ValueError('PUBLIC_ITEMS_REQUIRED')
    keys=labels(q['question_type'],len(q[field]))
    if tuple(q[field])!=keys: raise ValueError('CANONICAL_INPUT_LABELS_INVALID')
    values=list(q[field].values())
    if any(not isinstance(v,str) or not v.strip() for v in values) or len(set(' '.join(v.casefold().split()) for v in values))!=len(keys):
        raise ValueError('ITEMS_NOT_NONEMPTY_UNIQUE')
    if ('items' if field=='options' else 'options') in q: raise ValueError('CONFLICTING_TASK_FIELDS')
    return q

def content_ids(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    mapping={label:'content_'+hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label,text in q[field].items()}
    if len(set(mapping.values()))!=len(mapping): raise ValueError('CONTENT_ID_COLLISION')
    return mapping

def _seed(seed,qid,variant):
    return int.from_bytes(hashlib.sha256(f'{seed}|{qid}|{variant}'.encode()).digest()[:8],'big')

def variants(q,seed,count):
    # Same random rebind algorithm as native v15 make_variants; dynamic labels only.
    field='options' if q['question_type']=='multi_select' else 'items'
    keys=list(labels(q['question_type'],len(q[field])))
    mapping=content_ids(q); content=list(mapping.values());texts={mapping[k]:q[field][k] for k in keys}
    if not 1<=count<=math.factorial(len(content)): raise ValueError('INVALID_VARIANT_COUNT')
    random.Random(_seed(seed,q['candidate_id'],-1)).shuffle(content)
    used=set(); result=[]
    for i in range(count):
        rng=random.Random(_seed(seed,q['candidate_id'],i))
        while True:
            order=content[:];rng.shuffle(order)
            if tuple(order) not in used: break
        used.add(tuple(order)); display={keys[p]:texts[c] for p,c in enumerate(order)}
        result.append({'variant_id':f'perm_{i:02d}',field:display,'display_to_content':dict(zip(keys,order)),
                       'canonical_order_content_ids':content[:],'display_order_content_ids':order,'content_to_text':texts})
    return result

def choices(qtype,display):
    _,solver=native(); out=solver.structured_box_choices(qtype,allowed_item_ids=display.keys())
    expected=2**len(display)-1 if qtype=='multi_select' else math.factorial(len(display))
    if len(out)!=expected: raise RuntimeError('NATIVE_FINITE_CHOICES_CARDINALITY_MISMATCH')
    return out

def direct_prompt(tokenizer,q,variant,source):
    contract,_=native()
    public={'question_type':q['question_type'],'question':q['question']}
    field='options' if q['question_type']=='multi_select' else 'items'
    return contract.build_prompt(tokenizer,public,{field:variant[field]},contract.dialogue_rows(source),context_mode='full_context')

def evidence_schema(q):
    field='options' if q['question_type']=='multi_select' else 'items'; keys=list(q[field]); n=len(keys)
    return {'type':'object','additionalProperties':False,'required':['evidence','answer'], 'properties':{
      'evidence':{'type':'array','minItems':n,'maxItems':n,'items':{'type':'object','additionalProperties':False,
         'required':['label','status','turn_ref','quote'],'properties':{
           'label':{'type':'string','enum':keys},'status':{'type':'string','enum':['supported','contradicted','not_supported','ambiguous']},
           'turn_ref':{'type':'string'},'quote':{'type':'string'}}}},
      'answer':{'type':'array','minItems':n if field=='items' else 1,'maxItems':n,'uniqueItems':True,'items':{'type':'string','enum':keys}}}}

def evidence_prompt(tokenizer,q,variant,source):
    field='options' if q['question_type']=='multi_select' else 'items'
    rows='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
    task='For every displayed label provide a brief source citation: its status, turn_ref and short exact quote. Use empty turn_ref and quote when no citation exists. For ordering, cite the occurrence used to order each item. Return the final answer using displayed labels only. Return one JSON object with evidence and answer. Do not give extended reasoning.'
    user=f"Dialogue:\n{rows}\n\nQuestion ({q['question_type']}):\n{q['question']}\n\n"+field.title()+':\n'+'\n'.join(f'{k}. {v}' for k,v in variant[field].items())+'\n\n'+task
    return tokenizer.apply_chat_template([{'role':'system','content':'Use only the supplied dialogue. Return brief auditable source citations and a final answer in the required JSON schema.'},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

def sampling_params(row,expected_api):
    from vllm import SamplingParams,sampling_params as sp
    _,solver=native(); api,_=solver.detect_structured_output_api()
    if api!=expected_api: raise RuntimeError('STRUCTURED_API_CHANGED')
    field='options' if row['question_type']=='multi_select' else 'items'
    direct=row['branch']=='direct'
    constraint={'choice':choices(row['question_type'],row['display'][field])} if direct else {'json':row['json_schema']}
    kwargs={'temperature':0.8,'top_p':0.9,'max_tokens':32 if direct else 1024,'seed':row['seed']}
    if direct: kwargs.update(stop=['</box>'],include_stop_str_in_output=True)
    if api.startswith('structured_outputs='):
        kwargs['structured_outputs']=sp.StructuredOutputsParams(**constraint)
    else: kwargs['guided_decoding']=sp.GuidedDecodingParams(**constraint)
    return SamplingParams(**kwargs)
