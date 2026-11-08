"""CPU-only one-time native custom LoRA merge, tensor audit and identity freeze."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parent.parent
BASE = Path('/root/autodl-tmp/Qwen3-4B')
TRAIN = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'
DELIVERY = REPO / 'offline-deliveries/h4_codex_training_v126'
POOL = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
ADAPTER = TRAIN / 'adapter'
OUTPUT = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'
PENDING = OUTPUT.with_name(OUTPUT.name + '.pending')
LEGACY = REPO / 'current_method_github/scripts/merge_custom_lora_adapter.py'
FINAL_STATE_SHA = '0e89757df099234cc6a4f81dd1bfbbf869dd4a4425b6a8098f8c00a65a170ecc'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': p.stat().st_size}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def verify_seal(folder):
    manifest_path = folder / 'COMPLETE_MANIFEST.json'
    manifest = read(manifest_path)
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'COMPLETE_MANIFEST.json'}
    assert actual == {row['path'] for row in manifest['files']}, 'UNLISTED_OR_MISSING_SEAL_FILE'
    assert len(actual) == manifest['file_count']
    for row in manifest['files']:
        p = folder / row['path']
        assert p.stat().st_size == row['size_bytes'] and sha(p) == row['sha256'], str(p)
    return bind(manifest_path)


def main():
    started = time.monotonic()
    assert not OUTPUT.exists() and not PENDING.exists(), 'REFUSE_EXISTING_OUTPUT'
    assert not (HERE / 'MODEL_IDENTITY.json').exists(), 'REFUSE_EXISTING_IDENTITY'
    input_bindings = [verify_seal(DELIVERY), verify_seal(TRAIN)]
    assert read(DELIVERY / 'FINAL_STATUS.json')['training_complete'] is True
    result = read(TRAIN / 'TRAINING_RESULT.json')
    assert result['status'] == 'PASS' and result['backward_calls'] == 726 and result['optimizer_steps'] == 182
    output_binding = read(DELIVERY / 'MODEL_OUTPUT_BINDING.json')
    assert sha(TRAIN / 'COMPLETE_MANIFEST.json') == output_binding['complete_manifest_sha256']
    assert sha(ADAPTER / 'adapter_model.pt') == result['adapter_state_sha256'] == output_binding['adapter_state_sha256'] == FINAL_STATE_SHA
    config = read(ADAPTER / 'adapter_config.json')
    assert sha(ADAPTER / 'adapter_config.json') == result['adapter_config_sha256'] == output_binding['adapter_config_sha256']
    assert config['model'] == config['tokenizer'] == str(BASE)
    assert config['step'] == 182 and config['max_length'] == 12288
    assert config['lora_rank'] == 16 and config['lora_alpha'] == 32
    assert config['init_adapter_dir'] == str(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter')
    expected_targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    assert set(config['target_modules']) == expected_targets
    freeze = {r['path']: r for r in read(POOL / 'INPUT_FREEZE.json')['files']}
    for path in sorted(BASE.iterdir()):
        if path.is_file():
            row = freeze[str(path)]
            assert path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], str(path)
            input_bindings.append(bind(path))
    for path in [LEGACY, Path(__file__), ADAPTER / 'adapter_config.json', ADAPTER / 'adapter_model.pt',
                 TRAIN / 'TRAINING_RESULT.json', DELIVERY / 'MODEL_OUTPUT_BINDING.json',
                 DELIVERY / 'evidence/INDEPENDENT_FINAL_TRAINING_AUDIT.json']:
        input_bindings.append(bind(path))
    print('PASS: sealed formal outputs and all original base files verified', flush=True)

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), 'CPU_MERGE_GPU_MUST_BE_HIDDEN'
    state = torch.load(ADAPTER / 'adapter_model.pt', map_location='cpu', weights_only=True)
    expected_modules = {f'model.layers.{layer}.{part}.{target}'
                        for layer in range(36) for part, targets in [('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                                                                    ('mlp', ['gate_proj', 'up_proj', 'down_proj'])]
                        for target in targets}
    expected_keys = {module + suffix for module in expected_modules for suffix in ['.lora_a.weight', '.lora_b.weight']}
    assert set(state) == expected_keys and len(state) == 504
    assert all(t.dtype == torch.float32 and t.device.type == 'cpu' and torch.isfinite(t).all().item() for t in state.values())
    assert sum(t.numel() for t in state.values()) == 33030144
    for module in expected_modules:
        a, b = state[module + '.lora_a.weight'], state[module + '.lora_b.weight']
        assert a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == 16

    command = [sys.executable, str(LEGACY), '--model', str(BASE), '--tokenizer', str(BASE),
               '--adapter-dir', str(ADAPTER), '--output-dir', str(PENDING), '--dtype', 'bfloat16', '--max-shard-size', '4GB']
    merge_start = time.monotonic()
    with (HERE / 'merge.log').open('x') as log:
        process = subprocess.run(command, cwd=REPO / 'current_method_github', env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    assert process.returncode == 0, 'NATIVE_MERGE_FAILED'
    merge_seconds = time.monotonic() - merge_start
    metadata = read(PENDING / 'merged_adapter_info.json')
    assert metadata['merge_info'] == {'merged_linear_modules': 252, 'skipped_missing_pairs': []}
    assert metadata['base_model'] == str(BASE) and metadata['adapter_dir'] == str(ADAPTER)
    assert metadata['adapter_config'] == config
    print('PASS: native custom merge completed, validating every saved tensor', flush=True)

    base_index = read(BASE / 'model.safetensors.index.json')['weight_map']
    new_index = read(PENDING / 'model.safetensors.index.json')['weight_map']
    assert set(base_index) == set(new_index), 'MERGED_STANDARD_WEIGHT_KEYS_CHANGED'
    base_handles = {name: safe_open(str(BASE / name), framework='pt', device='cpu') for name in set(base_index.values())}
    new_handles = {name: safe_open(str(PENDING / name), framework='pt', device='cpu') for name in set(new_index.values())}
    checked, merged_count, unchanged_count = 0, 0, 0
    for key in sorted(base_index):
        original = base_handles[base_index[key]].get_tensor(key)
        actual = new_handles[new_index[key]].get_tensor(key)
        assert original.device.type == actual.device.type == 'cpu'
        assert original.dtype == actual.dtype == torch.bfloat16 and original.shape == actual.shape
        module = key[:-len('.weight')] if key.endswith('.weight') else None
        if module in expected_modules:
            a, b = state[module + '.lora_a.weight'], state[module + '.lora_b.weight']
            delta = (b @ a) * 2.0
            assert delta.shape == original.shape and torch.isfinite(delta).all().item()
            expected = original.clone().add_(delta.to(dtype=original.dtype))
            assert torch.equal(actual, expected), 'MERGED_WEIGHT_MATH_MISMATCH:' + key
            assert not torch.equal(actual, original), 'EXPECTED_TARGET_DID_NOT_CHANGE:' + key
            merged_count += 1
        else:
            assert torch.equal(actual, original), 'NON_TARGET_WEIGHT_CHANGED:' + key
            unchanged_count += 1
        assert torch.isfinite(actual).all().item(), 'NONFINITE_MERGED_WEIGHT:' + key
        checked += 1
    assert merged_count == 252 and checked == len(new_index)
    del base_handles, new_handles
    print(f'PASS: {checked} tensors exact, {merged_count} merged targets, {unchanged_count} unchanged others', flush=True)

    base_config, new_config = read(BASE / 'config.json'), read(PENDING / 'config.json')
    for key in ['architectures', 'model_type', 'num_hidden_layers', 'hidden_size', 'vocab_size', 'num_attention_heads',
                'num_key_value_heads', 'head_dim', 'max_position_embeddings', 'rope_theta', 'rope_scaling', 'tie_word_embeddings']:
        assert new_config[key] == base_config[key], key
    assert new_config['max_position_embeddings'] == 40960 and new_config['use_cache'] is True
    base_tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    merged_tok = AutoTokenizer.from_pretrained(str(PENDING), local_files_only=True)
    assert base_tok.get_vocab() == merged_tok.get_vocab()
    assert base_tok.chat_template == merged_tok.chat_template
    assert base_tok.special_tokens_map == merged_tok.special_tokens_map
    assert sha(PENDING / 'chat_template.jinja') == hashlib.sha256(base_tok.chat_template.encode()).hexdigest() == 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'
    train_rows = [json.loads(line) for line in (POOL / 'data/TRAIN_NEW_ONLY.jsonl').read_text().splitlines()]
    prompt_cases = []
    for row in train_rows:
        messages = row['messages'][:-1]
        base_prompt = base_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        merged_prompt = merged_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert base_prompt == merged_prompt
        assert base_tok.encode(base_prompt, add_special_tokens=False) == merged_tok.encode(merged_prompt, add_special_tokens=False)
        prompt_cases.append(row['candidate_id'])
    assert len(prompt_cases) == 363
    assert not (PENDING / 'adapter_config.json').exists() and not (PENDING / 'adapter_model.pt').exists()
    assert not any('lora_' in key for key in new_index)

    for binding in input_bindings:
        assert sha(binding['path']) == binding['sha256'], 'INPUT_CHANGED_DURING_MERGE:' + binding['path']
    PENDING.rename(OUTPUT)
    files = [{'name': str(p.relative_to(OUTPUT)), 'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(OUTPUT.rglob('*')) if p.is_file()]
    fingerprint = digest([{'name': r['name'], 'size_bytes': r['size_bytes'], 'sha256': r['sha256']} for r in files])
    checkpoint_files = [{'path': r['path'], 'name': r['name'], 'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in files]
    # Native paired evaluation deliberately retains the original base tokenizer.
    for name in ['tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json']:
        p = BASE / name
        checkpoint_files.append({'path': str(p), 'name': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    identity = {
        'schema_version': 'v127_M_clean_new_merged_identity_v1', 'status': 'PASS', 'model_identifier': 'M_clean_new',
        'model_name': 'M_clean_new', 'model_path': str(OUTPUT), 'inference_model_path': str(OUTPUT),
        'tokenizer_path': str(BASE), 'base_model_identifier': 'Qwen/Qwen3-4B', 'base_model_path': str(BASE),
        'adapter_path': str(ADAPTER), 'adapter_enabled': True, 'adapter_already_merged': True,
        'adapter_config_sha256': result['adapter_config_sha256'], 'adapter_state_sha256': FINAL_STATE_SHA,
        'checkpoint_files': checkpoint_files, 'merged_files': files, 'merged_file_count': len(files),
        'merged_size_bytes': sum(r['size_bytes'] for r in files), 'model_fingerprint_sha256': fingerprint,
        'fingerprint_method': 'SHA256 of canonical UTF8 JSON list sorted by name, with name,size_bytes,sha256 for every merged output file.',
        'merged_linear_modules': 252, 'merge_algorithm': 'Native custom LoRALinear W_bf16.add_((B_fp32@A_fp32*2).to(bf16)), original base once',
        'chat_template_sha256': hashlib.sha256(base_tok.chat_template.encode()).hexdigest(),
        'architecture_max_position_embeddings': 40960, 'training_max_length': 12288, 'evaluation_max_model_len': 16384,
        'source_training_delivery_manifest_binding': input_bindings[0],
        'source_training_outputs_manifest_binding': input_bindings[1], 'input_bindings': input_bindings,
        'identity_creation_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'CPU_only': True, 'inference_started': False,
    }
    save(HERE / 'MODEL_IDENTITY.json', identity)
    audit = {
        'schema_version': 'v127_model_merge_audit_v1', 'status': 'PASS', 'actual_executor': '/root/repair_v122_engineering',
        'identity_binding': bind(HERE / 'MODEL_IDENTITY.json'), 'model_identifier': 'M_clean_new', 'model_path': str(OUTPUT),
        'model_fingerprint_sha256': fingerprint, 'merged_file_count': len(files), 'merged_size_bytes': identity['merged_size_bytes'],
        'final_adapter_state_sha256': FINAL_STATE_SHA, 'native_merge_script_binding': bind(LEGACY),
        'reproduction_script_binding': bind(Path(__file__)), 'merge_log_binding': bind(HERE / 'merge.log'),
        'merge_command': command, 'environment': {key: os.environ[key] for key in ['CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'merge_elapsed_seconds': merge_seconds, 'total_elapsed_seconds': time.monotonic() - started,
        'checks': {'v126_delivery_and_training_output_seals_verified': True, 'formal726_backward182_updates_PASS': True,
                   'all_base_files_hash_match_frozen_training_inputs': True, 'final_adapter504_float32_keys_finite': True,
                   'rank16_alpha32_target_shapes_valid': True, 'exactly252_modules_merged_no_missing_pairs': True,
                   'all_saved_target_weights_equal_native_bf16_formula': True, 'all_non_target_weights_byte_equal_base': True,
                   'all_saved_tensors_finite': True, 'base_only_once_not_historical_merged': True,
                   'config_context_architecture_preserved': True, 'tokenizer_vocab_and_special_tokens_identical': True,
                   'all363_public_chat_renderings_and_token_ids_equal': True, 'standard_HF_weights_no_adapter_autoload': True,
                   'all_merged_files_hashed': True, 'input_hashes_unchanged_before_after': True, 'GPU_hidden_CPU_only': True},
        'tensor_verification': {'total_saved_tensors': checked, 'merged_target_tensors': merged_count,
                                'unchanged_non_target_tensors': unchanged_count, 'adapter_tensors': 504,
                                'adapter_float32_parameters': 33030144, 'prompt_cases': len(prompt_cases)},
        'outputs': files, 'input_bindings': input_bindings,
        'limitations': ['No inference or CUDA kernels were run; model-load/evaluation validity remains for the paired evaluation stage.',
                        'BF16 merged logits can differ from unmerged FP32-LoRA forward due to normal finite-precision arithmetic; identical historical merge policy was retained.'],
        'historical_models_or_sealed_training_packages_modified': False,
    }
    save(PACKAGE / 'evidence/MODEL_MERGE_AUDIT.json', audit)
    print(json.dumps({'status': 'PASS', 'identity': bind(HERE / 'MODEL_IDENTITY.json'), 'audit': bind(PACKAGE / 'evidence/MODEL_MERGE_AUDIT.json'),
                      'model_path': str(OUTPUT), 'file_count': len(files), 'size_bytes': identity['merged_size_bytes'], 'fingerprint': fingerprint}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
