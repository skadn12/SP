#!/usr/bin/env python3
"""Materialize an isolated HF model with the frozen custom LoRA applied.

The training code stores LoRALinear weights in a PyTorch state dict.  vLLM
cannot load that format as an adapter, so this script applies the exact
LoRA delta (B @ A * alpha/rank) to a byte-isolated copy of the base model.
"""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

BASE = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter')
OUT = Path(__file__).resolve().parent / 'mclean_merged_model'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def resolve(root, dotted):
    obj=root
    for part in dotted.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj

def main():
    if (OUT/'config.json').exists():
        raise SystemExit('refusing to overwrite existing merged model')
    cfg=json.loads((ADAPTER/'adapter_config.json').read_text())
    rank=int(cfg['lora_rank']); alpha=int(cfg['lora_alpha'])
    state=torch.load(ADAPTER/'adapter_model.pt', map_location='cpu')
    model=AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True)
    applied=0
    with torch.no_grad():
        for key,A in state.items():
            if not key.endswith('.lora_a.weight'): continue
            prefix=key[:-len('.lora_a.weight')]
            B=state[prefix+'.lora_b.weight']
            linear=resolve(model, prefix.rsplit('.',1)[0])
            linear.weight.add_(((B.float() @ A.float())*(alpha/rank)).to(linear.weight.dtype))
            applied += 1
    if applied != 252:
        raise RuntimeError(f'expected 252 LoRA matrices, applied {applied}')
    OUT.mkdir(parents=True)
    model.save_pretrained(str(OUT), safe_serialization=True, max_shard_size='2GB')
    tok_files=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','added_tokens.json','chat_template.jinja']
    for name in tok_files:
        src=BASE/name
        if src.exists(): shutil.copy2(src,OUT/name)
    manifest={'base_model':str(BASE),'base_config_sha256':sha(BASE/'config.json'),
        'adapter_dir':str(ADAPTER),'adapter_config_sha256':sha(ADAPTER/'adapter_config.json'),
        'adapter_model_sha256':sha(ADAPTER/'adapter_model.pt'),'rank':rank,'alpha':alpha,
        'applied_lora_matrices':applied,'output':str(OUT)}
    (OUT/'MERGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
