"""Shared immutable bindings for the isolated final-training controller."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

FINAL = Path(__file__).resolve().parents[1]
RUN = FINAL.parent
WEEK = RUN.parent
OLD_TRAIN = WEEK / 'training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl'
OLD_SHA = '46acd3d1725c8af4fea264a5cad2dbdc5b36e604a75ff04f5ea2db1b4b2b50e1'
TRAINER = Path('/root/autodl-tmp/gxy-selfplay/current_method_github/train_qwen3_lora_sft.py')
TRAINER_SHA = '918a45232e4287a26f46a5b4eb48175476a82136f5c3e202c94bf0d8cecb87dc'
MODEL = Path('/root/autodl-tmp/Qwen3-4B')
INIT = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter')
INIT_SHA = '92b5706e2e2e4201e1aeff91bad8137e9662fc95ce96e9b71a2475e9097a31bf'
OUTPUT = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_final_merged_training_20261108_run01')
FORMATTER = WEEK / 'training_set_rebuild_20261104_run01/packing/export_training.py'
CANDIDATES = RUN / 'candidate_manifest_20261108_v1/CANDIDATE_MANIFEST.json'
SCORES = RUN / 'solver_eval_20261108_strict128_formal/SCORES.json'
RECIPE = dict(max_length=12288, epochs=2.0, batch_size=1, grad_accum=4, lr=1e-4,
              min_lr_ratio=.1, warmup_ratio=.05, weight_decay=0., max_grad_norm=1.,
              lora_rank=16, lora_alpha=32, lora_dropout=.05,
              target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
              seed=20261018, eval_steps=20, save_steps=50)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def verify_binding(binding):
    actual = bind(binding['path'])
    assert actual['sha256'] == binding['sha256'], ('BINDING_CHANGED', binding['path'])
    if 'size_bytes' in binding:
        assert actual['size_bytes'] == binding['size_bytes'], ('BINDING_SIZE_CHANGED', binding['path'])
    return actual


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_fixed_inputs():
    assert sha(OLD_TRAIN) == OLD_SHA, 'SEALED_135_CHANGED'
    assert sha(TRAINER) == TRAINER_SHA, 'NATIVE_TRAINER_CHANGED'
    assert sha(INIT / 'adapter_model.pt') == INIT_SHA, 'MCLEAN_START_ADAPTER_CHANGED'


def final_manifest():
    manifest = read(FINAL / 'FINAL_MANIFEST.json')
    assert manifest['status'] == 'FINAL_DATA_FROZEN_NO_TRAINING'
    assert manifest['recipe'] == RECIPE
    verify_fixed_inputs()
    for item in manifest['bindings']:
        verify_binding(item)
    raw = (FINAL / 'FINAL_TRAIN.jsonl').read_bytes()
    old = OLD_TRAIN.read_bytes()
    assert raw[:len(old)] == old, 'OLD_135_BYTES_NOT_PRESERVED'
    assert len([line for line in raw.splitlines() if line.strip()]) == manifest['record_count']
    return manifest
