#!/usr/bin/env python3
"""CPU behavioral checks; no frozen training export, W&B run, or GPU launch."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import types

from audited_training_runner import Audit
from build_final_training import normalize_candidate
from training_contract import CANDIDATES, FORMATTER, TRAINER, TRAINER_SHA, bind, load, now, read, sha, write_once


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    import torch
    checks=[]
    source_rows=read(CANDIDATES)['records']
    formatter=load('controller_interface_frozen_formatter',FORMATTER)
    deps=formatter.dependencies()
    for qtype in ('multi_select','ordering'):
        row=next(r for r in source_rows if r['question_type']==qtype and r['canonical_status']=='PASS')
        item=normalize_candidate(row,{'semantic_family_id':'INTERFACE_ONLY'})
        source=read(row['source_path'])
        messages,variant,*_=formatter.public_and_gold(item,source,deps)
        counts,fits=formatter.checked_features(messages,deps)
        changed=deepcopy(item)
        ids=list(item['label_to_option_id'].values())
        if qtype=='multi_select':
            changed['gold_option_ids']=next(g for g in (ids[:2],ids[-2:]) if set(g)!=set(item['gold_option_ids']))
        else:
            changed['gold_option_ids']=list(reversed(item['gold_option_ids']))
        other,other_variant,*_=formatter.public_and_gold(changed,source,deps)
        assert messages[:-1]==other[:-1] and variant['display_to_option_id']==other_variant['display_to_option_id']
        assert fits and counts['supervised_token_count']>0
        assert messages[-1]['content'].startswith('<box>') and '{' not in messages[-1]['content']
        checks.append(dict(check='actual_frozen_formatter_gold_blind',question_type=qtype,candidate_id=row['candidate_id'],status='PASS',**counts))

    # Exercise real Torch backward/clip/AdamW behavior on a tiny CPU tensor.
    # The audit is advanced past step 1 to avoid an unrelated model snapshot.
    synthetic=args.out_dir/'synthetic_gradient'
    synthetic.mkdir()
    audit=Audit(synthetic,'smoke',1,2,None,None,None)
    native=types.SimpleNamespace(load_initial_adapter=lambda *a: {},tokenize_record=lambda *a: {},
        trainable_parameters=lambda model: (2,2),save_adapter=lambda *a: None)
    audit.install(native)
    model=torch.nn.Linear(2,1,bias=False)
    audit.model=model
    audit.native_args=types.SimpleNamespace(grad_accum=1)
    audit.optimizer_steps=1
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    initial=model.weight.detach().clone()
    loss=(model(torch.tensor([[1.,2.]]))-3.).pow(2).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    assert audit.backward_calls==1 and audit.optimizer_steps==2
    assert torch.count_nonzero(initial-model.weight.detach()).item()>0
    assert audit.gradient_records[0]['gradient_nonzero_elements']>0
    checks.append(dict(check='actual_torch_backward_clip_and_optimizer_instrumentation',status='PASS',device='cpu'))
    model.weight.grad=torch.zeros_like(model.weight)
    rejected_zero=False
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    except AssertionError as exc:
        rejected_zero=str(exc)=='GRADIENT_NOT_FINITE_NONZERO'
    assert rejected_zero
    checks.append(dict(check='zero_gradient_rejected_before_optimizer',status='PASS'))
    rejected_nan=False
    try:
        (model.weight*float('nan')).sum().backward()
    except AssertionError as exc:
        rejected_nan=str(exc)=='NONFINITE_LOSS'
    assert rejected_nan and audit.backward_calls==1
    checks.append(dict(check='nonfinite_loss_rejected_before_backward',status='PASS'))

    evaluation=args.out_dir/'evaluation_state'
    evaluation.mkdir()
    manifest=evaluation/'FIXTURE_MANIFEST.json'
    write_once(manifest,{'fixture':True})
    callback=evaluation/'fixture_callback.py'
    callback.write_text('''import json,hashlib,random,torch\ndef evaluate(*,model,tokenizer,step,output_dir,validation_manifest):\n    random.random(); torch.rand(3); model.train()\n    raw=output_dir/'RAW_COT.jsonl'\n    raw.write_text(json.dumps({'raw_prediction':'{"item_evidence":{},"answer_reason":"fixture","final_answer":"<box>A,B</box>"}'})+'\\n')\n    return {'metrics':{'valid_loss':1.,'exact':.5,'mean_J':.75},'evaluated_records':1,'raw_cot_saved':True,'artifacts':[{'path':str(raw.resolve()),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}\n''',encoding='utf-8')
    class Artifact:
        def __init__(self,*a,**k): pass
        def add_file(self,*a,**k): pass
    eval_audit=Audit(evaluation,'formal',0,0,manifest,callback,None)
    eval_audit.model=model
    eval_audit.tokenizer=object()
    eval_audit.wandb=types.SimpleNamespace(log=lambda *a,**k:None,log_artifact=lambda *a,**k:None)
    eval_audit.wandb_sdk=types.SimpleNamespace(Artifact=Artifact)
    model.eval()
    py_before=random.getstate()
    torch_before=torch.random.get_rng_state().clone()
    eval_audit.evaluate(0,'interface_fixture')
    assert not model.training and random.getstate()==py_before and torch.equal(torch.random.get_rng_state(),torch_before)
    eval_audit.evaluate(0,'same_endpoint_reuse_fixture')
    assert len(eval_audit.evaluations)==1
    checks.append(dict(check='evaluation_preserves_rng_model_mode_and_reuses_same_endpoint',status='PASS',actual_model_weights_loaded=False))
    assert sha(TRAINER)==TRAINER_SHA
    result=dict(status='PASS',checked_at_utc=now(),scope='CPU interface checks only; synthetic records are not training data',
                model_weights_loaded=False,gpu_calls=0,wandb_calls=0,formal_data_frozen=False,formal_training_started=False,
                native_trainer_unchanged=True,checks=checks,script=bind(Path(__file__)))
    write_once(args.out_dir/'INTERFACE_CHECK.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
