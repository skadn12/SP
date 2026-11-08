#!/usr/bin/env python3
"""Freeze public-only requests after real source-first content admission.

CPU only. No model backend, private Gold read, score selection or truncation.
"""
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def check(value, message):
    if not value:
        raise RuntimeError(message)


def verify(ref):
    check(binding(ref['path'])['sha256'] == ref['sha256'], 'HASH_CHANGED:' + ref['path'])


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        check(path.read_bytes() == data, 'REFUSING_TO_REWRITE_FROZEN_REQUESTS')
    else:
        path.write_bytes(data)


def main():
    check(Path(sys.prefix).resolve() == REPO / '.venv-local4b', 'EXACT_LOCAL4B_ENV_REQUIRED')
    admission_path = HERE / 'content/CONTENT_ADMISSION.json'
    admission = json.loads(admission_path.read_text())
    check(admission['status'] == 'CONTENT_ADMISSION_FINAL', 'CONTENT_ADMISSION_NOT_FINAL')
    admitted = [r for r in admission['items'] if r['status'] == 'CONTENT_ACCEPTED']
    check(bool(admitted), 'NO_ACCEPTED_CONTENT_NO_REQUEST_FREEZE')
    check(len({r['candidate_id'] for r in admitted}) == len(admitted), 'DUPLICATE_ADMITTED_CANDIDATES')
    policy_path = WEEK / 'day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json'
    policy = json.loads(policy_path.read_text())
    verify(policy['source_lock'])
    verify(policy['prompt']['adapter'])
    verify(policy['display_mapping']['implementation'])
    lock = json.loads(Path(policy['source_lock']['path']).read_text())
    slots = {r['slot_id']: (i, r) for i, r in enumerate(lock['itemized_slot_family_ledger'])}
    sources = {r['root_id']: r['source_binding'] for r in lock['source_bindings']}
    preflight_path = WEEK / 'day3_common_evaluation_v1/audit/SOURCE_AND_MAPPING_PREFLIGHT.json'
    preflight = json.loads(preflight_path.read_text())
    check(preflight['status'] == 'STRUCTURAL_AND_SYNTHETIC_MAPPING_PASS_ONLY', 'SYNTHETIC_PREFLIGHT_MISSING')
    for ref in preflight['actual_tokenizer_bindings']:
        verify(ref)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy['length']['tokenizer_directory'], local_files_only=True)
    spec = importlib.util.spec_from_file_location('common_eval_request_adapter', policy['prompt']['adapter']['path'])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    requests, mappings, lengths, held = [], [], [], []
    sample = policy['sampling']
    for row in sorted(admitted, key=lambda r: slots[r['candidate_id']][0]):
        check(row['five_family_checks'] == 'PASS' and row['independent_source_first_qa'] == 'PASS',
              'CONTENT_GATE_NOT_PASS:' + row['candidate_id'])
        verify(row['public_question'])
        q = json.loads(Path(row['public_question']['path']).read_text())
        adapter.public_question(q)
        check(q['candidate_id'] == row['candidate_id'], 'ADMISSION_PUBLIC_ID_MISMATCH')
        index, slot = slots[q['candidate_id']]
        check(q['root_id'] == slot['root_id'] and q['question_type'] == slot['question_type'], 'SLOT_IDENTITY_CHANGED')
        check(q['source_sha256'] == slot['source_sha256'] == sources[q['root_id']]['sha256'], 'LOCKED_SOURCE_SHA_MISMATCH')
        check(Path(q['source_path']).resolve() == Path(sources[q['root_id']]['path']).resolve(), 'LOCKED_SOURCE_PATH_MISMATCH')
        check(binding(q['source_path'])['sha256'] == q['source_sha256'], 'SOURCE_CHANGED')
        source = json.loads(Path(q['source_path']).read_text())
        check(source['root_id'] == q['root_id'], 'SOURCE_ROOT_MISMATCH')
        count = sample['multi_select_permutations'] if q['question_type'] == 'multi_select' else sample['ordering_permutations']
        repetitions = sample['multi_select_samples_per_permutation'] if q['question_type'] == 'multi_select' else sample['ordering_samples_per_permutation']
        variants = adapter.variants(q, sample['display_seed'], count)
        check(variants == adapter.variants(q, sample['display_seed'], count), 'NONDETERMINISTIC_DISPLAY')
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        target = slot['design_target']['options_target' if field == 'options' else 'items_target']
        check(len(q[field]) == target == 6, 'PRELOCKED_ITEM_COUNT_MISMATCH')
        canonical = adapter.content_ids(q)
        labels = list(q[field])
        answer_space = ([list(c) for n in range(1, len(labels) + 1) for c in itertools.combinations(labels, n)]
                        if field == 'options' else list(itertools.permutations(labels)))
        pending_requests, pending_mappings, pending_lengths = [], [], []
        roundtrips = 0
        for vi, variant in enumerate(variants):
            inverse = {v: k for k, v in variant['display_to_content'].items()}
            check(set(inverse) == set(canonical.values()), 'NON_BIJECTIVE_DISPLAY')
            for answer in answer_space:
                content = [canonical[k] for k in answer]
                display = [inverse[k] for k in content]
                check([variant['display_to_content'][k] for k in display] == content, 'ACTUAL_CONTENT_ROUNDTRIP_FAILED')
                roundtrips += 1
            prompt = adapter.direct_prompt(tokenizer, q, variant, source)
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            pending_lengths.append({'candidate_id': q['candidate_id'], 'variant_id': variant['variant_id'],
                                    'prompt_tokens': tokens, 'reserved_answer_tokens': sample['max_output_tokens']})
            pending_mappings.append({'candidate_id': q['candidate_id'], **variant})
            for si in range(repetitions):
                pending_requests.append({
                    'request_id': f"{q['candidate_id']}:{variant['variant_id']}:sample_{si:02d}",
                    'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                    'original_split': slot['split'], 'question_type': q['question_type'],
                    'global_slot_index_zero_based': index, 'variant_id': variant['variant_id'],
                    'sample_index': si, 'branch': 'direct',
                    'seed': sample['seed_base'] + index * 18 + vi * repetitions + si,
                    'display': variant, 'prompt': prompt, 'prompt_tokens': tokens,
                    'source_sha256': q['source_sha256'],
                    'public_question_sha256': row['public_question']['sha256'],
                    'temperature': sample['temperature'], 'top_p': sample['top_p'],
                    'max_output_tokens': sample['max_output_tokens'], 'stop': sample['stop'],
                    'include_stop_str_in_output': sample['include_stop_str_in_output'],
                    'choice_space': 'all_nonempty_subsets' if field == 'options' else 'all_full_permutations',
                })
        overflow = any(x['prompt_tokens'] + sample['max_output_tokens'] > policy['length']['max_model_len'] for x in pending_lengths)
        if overflow:
            held.append({'candidate_id': q['candidate_id'], 'status': 'LENGTH_HOLD', 'variants': pending_lengths})
            continue
        requests.extend(pending_requests)
        mappings.extend(pending_mappings)
        lengths.append({'candidate_id': q['candidate_id'], 'status': 'PASS', 'variants': pending_lengths,
                        'all_answer_space_roundtrips': roundtrips})
    check(len({r['request_id'] for r in requests}) == len(requests), 'DUPLICATE_REQUESTS')
    output = HERE / 'requests'
    # Refuse an already completed freeze before writing any output files.
    check(not (output / 'REQUEST_FREEZE.json').exists(), 'REQUEST_FREEZE_ALREADY_EXISTS')
    request_data = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in requests).encode()
    write_new(output / 'REQUESTS.jsonl', request_data)
    write_new(output / 'DISPLAY_MAPS.json', (json.dumps(mappings, ensure_ascii=False, indent=2) + '\n').encode())
    report = {'schema_version': 'actual_common_eval_request_freeze_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'CONTENT_REQUESTS_FROZEN_MODEL_BINDINGS_PENDING',
              'admission': binding(admission_path), 'policy': binding(policy_path),
              'script': binding(__file__), 'synthetic_mapping_preflight': binding(preflight_path),
              'actual_tokenizer_bindings': preflight['actual_tokenizer_bindings'],
              'requests': binding(output / 'REQUESTS.jsonl'), 'display_maps': binding(output / 'DISPLAY_MAPS.json'),
              'question_count': len(lengths), 'request_count': len(requests), 'length_and_mapping_checks': lengths,
              'length_holds': held, 'private_gold_opened': False, 'zero_truncation': True,
              'historical_role': 'REGRESSION_ONLY', 'new_solver_calls': 0, 'new_training_runs': 0,
              'actual_model_identity_frozen': False, 'full_evaluation_ready': False,
              'test_execution': 'FINAL_MODEL_ONLY_AFTER_MODEL_IDENTITY_FREEZE'}
    # The report is written once: reruns must use a separate version.
    (output / 'REQUEST_FREEZE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'question_count', 'request_count', 'new_solver_calls', 'full_evaluation_ready']}))


if __name__ == '__main__':
    main()
