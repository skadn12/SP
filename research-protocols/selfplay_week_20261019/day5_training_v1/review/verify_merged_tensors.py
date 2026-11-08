#!/usr/bin/env python3
"""Check six fixed merged-weight matrices against original base plus final LoRA.

CPU tensors only; no model construction, inference or GPU use. This is a fixed
arithmetic spot check in addition to full-file hashes and merge metadata.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DAY5 = HERE.parent


def binding(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}


def read(p):
    return json.loads(Path(p).read_text())


def tensor(directory, index, name):
    with safe_open(directory / index[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['A1', 'B1'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = read(DAY5 / 'MERGE_PLAN.json')
    stage = plan['stages'][args.role]
    base = Path('/root/autodl-tmp/Qwen3-4B')
    merged = Path(stage['output_dir'])
    adapter = Path(stage['final_adapter'])
    state = torch.load(adapter / 'adapter_model.pt', map_location='cpu', weights_only=True)
    config = read(adapter / 'adapter_config.json')
    bi = read(base / 'model.safetensors.index.json')['weight_map']
    mi = read(merged / 'model.safetensors.index.json')['weight_map']
    targets = [(0, 'self_attn.q_proj'), (0, 'mlp.gate_proj'),
               (17, 'self_attn.q_proj'), (17, 'mlp.down_proj'),
               (35, 'self_attn.q_proj'), (35, 'mlp.down_proj')]
    checks = []
    for layer, module in targets:
        stem = f'model.layers.{layer}.{module}'
        name = stem + '.weight'
        initial = tensor(base, bi, name)
        actual = tensor(merged, mi, name)
        a, b = state[stem + '.lora_a.weight'], state[stem + '.lora_b.weight']
        delta = (b.float() @ a.float()) * (float(config['lora_alpha']) / float(config['lora_rank']))
        expected = initial.clone()
        expected.add_(delta.to(expected.dtype))
        checks.append({'weight': name, 'shape': list(actual.shape),
                       'exact_arithmetic_match': torch.equal(actual, expected),
                       'finite': bool(torch.isfinite(actual).all()),
                       'different_from_original_base': not torch.equal(actual, initial),
                       'max_absolute_error': float((actual.float() - expected.float()).abs().max())})
    status = 'PASS' if all(c['exact_arithmetic_match'] and c['finite']
                           and c['different_from_original_base'] for c in checks) else 'FAIL'
    report = {'schema_version': 'day5_cpu_merged_tensor_spotcheck_v1', 'status': status,
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model_role': args.role, 'audit_script': binding(__file__),
              'model_directory': str(merged), 'adapter_binding': binding(adapter / 'adapter_model.pt'),
              'checks': checks, 'tensor_count': len(checks),
              'scope': 'Six fixed matrices: original base plus full final updated LoRA once, CPU only.',
              'limit': 'Arithmetic spot check; complete model identity uses independent full-file hashes.'}
    path = HERE / (args.role + '_MERGED_TENSOR_AUDIT.json')
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': status, 'model_role': args.role, 'checks': len(checks), 'output': str(path)}))
    return status != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
