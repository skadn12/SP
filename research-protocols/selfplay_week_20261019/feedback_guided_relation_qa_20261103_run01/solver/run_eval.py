#!/usr/bin/env python3
"""Finite local Qwen evaluation of frozen, independently reviewed F/G drafts.

Only transport/preparation differs from the frozen visible-reason evaluator.
Challenger and Reviewer are Codex; no school model or API is used here.
prepare/run/score never replace an existing artifact or retry a model call.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WEEK = EXPERIMENT.parent
REFERENCE = WEEK / 'base_qa_free_generation_reason_restore_eval_20261103_run02/run_eval.py'
PROJECT = WEEK.parent.parent
PERMUTATION_CODE = PROJECT / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/protocol.py'
MODEL = Path('/root/autodl-tmp/Qwen3-4B')
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 12
DISPLAY_SEED, SAMPLE_SEED, ENGINE_SEED = 2026110351, 2026110352, 2026110303
MAX_ITEMS, MAX_REQUESTS = 8, 288


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load('feedback_frozen_visible_reason', REFERENCE)
# The reused validator resolves the new request's frozen schema under HERE.
reference.HERE = HERE
baseline = reference.baseline
protocol = load('feedback_official_gold_blind_permutations', PERMUTATION_CODE)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path.read_bytes())}


def input_paths():
    """Snapshot all existing author/review inputs, never new Solver outputs."""
    return sorted(path for path in EXPERIMENT.rglob('*')
                  if path.is_file() and HERE not in path.parents)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_input_item(original, sources):
    item = copy.deepcopy(original)
    required = ('candidate_id', 'root_id', 'question_type', 'question', 'options',
                'label_to_option_id', 'gold_labels', 'gold_option_ids',
                'source_path', 'source_sha256', 'number', 'arm')
    assert all(key in item for key in required), ('ITEM_FIELD_MISSING', item.get('candidate_id'))
    qid, qtype = item['candidate_id'], item['question_type']
    assert isinstance(qid, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', qid), 'UNSAFE_CANDIDATE_ID'
    assert qtype in ('multi_select', 'ordering'), ('UNSUPPORTED_ANSWER_FORM', qid)
    assert item['arm'] in ('F', 'G'), ('INVALID_ARM', qid)
    assert isinstance(item['question'], str) and item['question'].strip(), ('EMPTY_QUESTION', qid)
    options, mapping, gold = item['options'], item['label_to_option_id'], item['gold_option_ids']
    assert isinstance(options, list) and len(options) >= 3, ('OPTIONS_REQUIRED', qid)
    labels = [option['id'] for option in options]
    expected = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(labels)]) if qtype == 'multi_select' else [f'I{i+1}' for i in range(len(labels))]
    assert labels == expected, ('DISPLAY_LABEL_CONTRACT', qid, labels)
    assert set(mapping) == set(labels) and len(set(mapping.values())) == len(labels), ('OPTION_MAPPING_INVALID', qid)
    assert all(isinstance(oid, str) and oid for oid in mapping.values()), ('OPTION_ID_INVALID', qid)
    assert all(isinstance(option['text'], str) and option['text'].strip() for option in options), ('OPTION_TEXT_INVALID', qid)
    assert len({' '.join(option['text'].casefold().split()) for option in options}) == len(labels), ('DUPLICATE_OPTION', qid)
    assert isinstance(gold, list) and len(gold) == len(set(gold)) and set(gold) <= set(mapping.values()), ('GOLD_ID_INVALID', qid)
    assert isinstance(item['gold_labels'], list) and len(item['gold_labels']) == len(set(item['gold_labels'])), ('GOLD_LABEL_INVALID', qid)
    mapped = [mapping[label] for label in item['gold_labels']]
    if qtype == 'multi_select':
        assert 2 <= len(gold) <= 6 and len(gold) < len(labels), ('UNSCORABLE_OR_UNSUPPORTED_GOLD_CARDINALITY', qid)
        assert set(mapped) == set(gold), ('GOLD_MAPPING_MISMATCH', qid)
    else:
        assert mapped == gold and set(gold) == set(mapping.values()), ('COMPLETE_ORDER_REQUIRED', qid)
    source_path = Path(item['source_path']).resolve()
    assert source_path.is_file() and sha(source_path.read_bytes()) == item['source_sha256'], ('SOURCE_HASH_MISMATCH', qid)
    source = read(source_path)
    assert source.get('root_id') == item['root_id'], ('SOURCE_ROOT_MISMATCH', qid)
    turns = source.get('turns')
    assert isinstance(turns, list) and turns, ('SOURCE_TURNS_REQUIRED', qid)
    assert all(isinstance(turn, dict) and all(isinstance(turn.get(key), str) and turn[key].strip()
               for key in ('turn_ref', 'speaker', 'text')) for turn in turns), ('SOURCE_TURN_SCHEMA', qid)
    refs = [turn['turn_ref'] for turn in turns]
    assert len(refs) == len(set(refs)), ('DUPLICATE_SOURCE_TURN_ID', qid)
    frozen = sources[item['root_id']]
    assert str(source_path) == str(Path(frozen['path']).resolve()) and item['source_sha256'] == frozen['sha256'], ('OUTSIDE_FROZEN_SOURCE', qid)
    assert len(turns) == frozen['turn_count'], ('SOURCE_TURN_COUNT_MISMATCH', qid)
    item.update(source_path=str(source_path), turn_count=len(turns),
                permutations=6 if qtype == 'multi_select' else min(12, math.factorial(len(labels))),
                repeats_per_permutation=3 if qtype == 'multi_select' else 1)
    return item, turns


def make_variants(item):
    """Call only the official Gold-blind pure permutation function."""
    qtype = item['question_type']
    labels = [option['id'] for option in item['options']]
    field = 'options' if qtype == 'multi_select' else 'items'
    protocol.LABELS[qtype] = tuple(labels)
    public = {'candidate_id': item['candidate_id'], 'question_type': qtype,
              field: {option['id']: option['text'] for option in item['options']}}
    # Gold, sources, feedback and arm are absent from the permutation input.
    variants = protocol.make_variants(public, seed=DISPLAY_SEED, count=item['permutations'])
    text_to_stable = {option['text']: item['label_to_option_id'][option['id']] for option in item['options']}
    rebound = []
    for variant in variants:
        display = {label: text_to_stable[variant['content_to_text'][content_id]]
                   for label, content_id in variant['display_to_content'].items()}
        reverse = {oid: label for label, oid in display.items()}
        gold_labels = [reverse[oid] for oid in item['gold_option_ids']]
        rebound.append({'variant_id': variant['variant_id'], 'display_to_option_id': display,
            'options': [{'id': label, 'option_id': display[label], 'text': text}
                        for label, text in variant[field].items()],
            'gold_labels': sorted(gold_labels) if qtype == 'multi_select' else gold_labels,
            'official_content_to_text': variant['content_to_text'],
            'official_display_to_content': variant['display_to_content']})
    assert len({tuple(row['display_to_option_id'].values()) for row in rebound}) == item['permutations']
    return rebound


def seed_for(root_index, variant_index, repeat):
    # Same root/variant/repeat has the same seed in both F/G and context modes.
    return SAMPLE_SEED + root_index * 1000 + variant_index * 3 + repeat


def self_checks():
    baseline.self_checks()
    fixture = {'candidate_id': 'offline_fixture', 'question_type': 'multi_select',
        'options': [{'id': label, 'text': 'distinct statement ' + label} for label in 'ABCDE'],
        'label_to_option_id': {label: 'stable_' + label for label in 'ABCDE'},
        'gold_option_ids': ['stable_A', 'stable_C'], 'permutations': 6}
    variants = make_variants(fixture)
    changed = {**fixture, 'gold_option_ids': ['stable_B', 'stable_D', 'stable_E']}
    other = make_variants(changed)
    assert [variant['display_to_option_id'] for variant in variants] == [variant['display_to_option_id'] for variant in other]
    for variant in variants:
        assert {variant['display_to_option_id'][label] for label in variant['gold_labels']} == set(fixture['gold_option_ids'])
    assert seed_for(0, 1, 2) == seed_for(0, 1, 2)
    assert seed_for(0, 1, 2) != seed_for(1, 1, 2)
    return 'historical scoring/parser checks and Gold-independent permutation/stable-ID/seed checks passed; no model inference'


def prepare():
    reference.offline()
    assert not (HERE / 'REQUESTS.json').exists() and not (HERE / 'PREPARE_STARTED.json').exists(), 'NO_PREPARE_RETRY_OR_OVERWRITE'
    checks = self_checks()
    quality_path, pool_path = EXPERIMENT / 'QUALITY_FINAL.json', EXPERIMENT / 'ITEMS_PRIVATE.json'
    quality, pool = read(quality_path), read(pool_path)
    assert isinstance(quality, dict) and isinstance(quality.get('items'), list), 'QUALITY_SCHEMA_INVALID'
    assert isinstance(pool, list) and len(pool) <= MAX_ITEMS, 'CANDIDATE_POOL_INVALID'
    assert all(isinstance(row, dict) and type(row.get('content_pass')) is bool for row in quality['items']), 'BOOLEAN_CONTENT_PASS_REQUIRED'
    quality_by_id = {row['candidate_id']: row for row in quality['items']}
    assert len(quality_by_id) == len(quality['items']), 'DUPLICATE_QUALITY_ID'
    assert len({item['candidate_id'] for item in pool}) == len(pool), 'DUPLICATE_POOL_ID'
    assert {item['candidate_id'] for item in pool} <= set(quality_by_id), 'UNREVIEWED_CANDIDATE'
    assert set(quality_by_id) <= {item['candidate_id'] for item in pool}, 'QUALITY_WITHOUT_CANDIDATE'
    selected = [item for item in pool if quality_by_id[item['candidate_id']]['content_pass'] is True]
    source_freeze = read(EXPERIMENT / 'SOURCE_FREEZE.json')
    source_by_root = {row['root_id']: row for row in source_freeze['sources']}
    roots = sorted(source_by_root)
    assert len(roots) == 4 and len(selected) <= MAX_ITEMS, 'EXPERIMENT_SIZE_EXCEEDED'
    root_indices = {root: index for index, root in enumerate(roots)}
    seen_pairs, items, turns_by_id = set(), [], {}
    for original in selected:
        item, turns = check_input_item(original, source_by_root)
        pair = (item['root_id'], item['arm'])
        assert pair not in seen_pairs, ('DUPLICATE_ROOT_ARM', pair)
        seen_pairs.add(pair)
        items.append(item)
        turns_by_id[item['candidate_id']] = turns
    bindings = {str(path.resolve()): bind(path) for path in input_paths()}
    dependencies = [Path(__file__), REFERENCE, reference.REASON, reference.REPAIR,
        reference.BASE / 'run_eval.py', baseline.ENGINE_PATH, baseline.engine.LEGACY_FILE,
        baseline.ORDER_PROMPT_PATH, baseline.SCORER_PATH, PERMUTATION_CODE,
        MODEL / 'config.json', MODEL / 'tokenizer_config.json', MODEL / 'tokenizer.json']
    for item in items:
        dependencies.append(Path(item['source_path']))
    for path in dependencies:
        bindings[str(path.resolve())] = bind(path)
    requests, permutations, assets = [], {}, []
    if items:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    for item in items:
        variants = make_variants(item)
        permutations[item['candidate_id']] = variants
        labels = [option['id'] for option in item['options']]
        qtype = item['question_type']
        for mode in ('full_context', 'no_context'):
            turns = turns_by_id[item['candidate_id']] if mode == 'full_context' else []
            refs = [turn['turn_ref'] for turn in turns]
            for variant_index, variant in enumerate(variants):
                request_public = {**variant, 'question_type': qtype}
                prompt = reference.prompt_for(tokenizer, item, request_public, turns)
                tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                assert tokens + MAX_TOKENS <= MODEL_LEN, ('NO_TRUNCATION', item['candidate_id'], mode, tokens)
                schema = reference.schema_for(labels, refs)
                for repeat in range(item['repeats_per_permutation']):
                    rid = f"{item['candidate_id']}__{mode}__{variant['variant_id']}__s{repeat:02d}"
                    prompt_path, schema_path = f'prompts/{rid}.txt', f'schemas/{rid}.json'
                    schema_text = json.dumps(schema, ensure_ascii=False, indent=2) + '\n'
                    assets.extend([(prompt_path, prompt), (schema_path, schema_text)])
                    n = len(labels) if qtype == 'multi_select' else len(labels) * (len(labels)-1)
                    g = len(item['gold_option_ids']) if qtype == 'multi_select' else n // 2
                    requests.append({'request_id': rid, 'candidate_id': item['candidate_id'],
                        'root_id': item['root_id'], 'arm': item['arm'], 'number': item['number'],
                        'question_type': qtype, 'context_mode': mode, 'variant_id': variant['variant_id'],
                        'repeat_index': repeat, 'sampling_seed': seed_for(root_indices[item['root_id']], variant_index, repeat),
                        'display_to_option_id': variant['display_to_option_id'], 'options': variant['options'],
                        'gold_labels': variant['gold_labels'], 'gold_option_ids': item['gold_option_ids'],
                        'n': n, 'g': g, 'all_true_control': False,
                        'prompt_path': prompt_path, 'prompt_sha256': sha(prompt.encode()), 'prompt_token_count': tokens,
                        'schema_path': schema_path, 'schema_sha256': sha(schema_text.encode()),
                        'turn_count': len(turns), 'source_sha256_bytes': item['source_sha256'],
                        'content_pass': True, 'formal_training_export': False})
    assert len(requests) <= MAX_REQUESTS, 'REQUEST_LIMIT_EXCEEDED'
    assert len({row['request_id'] for row in requests}) == len(requests), 'DUPLICATE_REQUEST'
    write('PREPARE_STARTED.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False})
    for relative, text in assets:
        path = HERE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    write('ITEMS_PRIVATE.json', items)
    write('PERMUTATIONS_PRIVATE.json', permutations)
    write('REQUESTS.json', requests)
    write('INPUT_BINDINGS.json', {'files': list(bindings.values()),
        'policy': 'all existing author/review/protocol/source inputs plus exact reused implementation; immutable byte bindings'})
    write('MODEL_CONFIG.json', {'model': str(MODEL), 'tokenizer': str(MODEL), 'base_model_only': True,
        'challenger': 'Codex', 'reviewer': 'Codex same-model blind review', 'solver': 'local Qwen3-4B',
        'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1.,
        'max_new_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'dtype': 'bfloat16',
        'gpu_memory_utilization': .8, 'engine_seed': ENGINE_SEED, 'batch_size': BATCH,
        'native_thinking': False, 'visible_reasons': True, 'structured_outputs': True,
        'enable_prefix_caching': False, 'stop_sequences': None,
        'offline': True, 'school_api': False, 'network_api': False, 'automatic_retry': False,
        'training': False, 'planned_requests': len(requests),
        'installed_vllm_version_at_preparation': package_version('vllm'),
        'frozen_prompt_and_schema_reference': str(REFERENCE),
        'permutation_generator': str(PERMUTATION_CODE), 'display_seed': DISPLAY_SEED,
        'sampling_by_type': {'multi_select': {'permutations': 6, 'repeats': 3},
            'ordering': {'permutations': 'min(12,k!)', 'repeats': 1}},
        'sampling_seed_rule': '2026110352 + sorted_full_root_index*1000 + variant_index*3 + repeat_index; same across F/G arms and Full/No-context',
        'sampling_root_indices': root_indices, 'context_modes': ['full_context', 'no_context'],
        'primary_scoring': 'strict evidence/reason JSON protocol; invalid=0 in fixed denominators',
        'secondary_scoring': 'final-answer-only; isolates evidence/reason protocol errors',
        'multi_select_J': 'TP/g-FP/(n-g)', 'ordering_J': '(correct_pairs-wrong_pairs)/C(k,2)',
        'source_scope': 'complete frozen dialogue turn_ref/speaker/text; no metadata warnings or Gold included in prompt',
        'question_isolation': 'one question per request; no other candidate, feedback card or expected error supplied',
        'formal_training_export': False})
    write('PREFLIGHT.json', {'status': 'PASS' if items else 'NO_CONTENT_PASS', 'candidate_count': len(items),
        'all_candidate_count': len(pool), 'excluded_candidate_ids': [item['candidate_id'] for item in pool if quality_by_id[item['candidate_id']]['content_pass'] is False],
        'request_count': len(requests), 'full_context_requests': sum(row['context_mode']=='full_context' for row in requests),
        'no_context_requests': sum(row['context_mode']=='no_context' for row in requests),
        'min_prompt_tokens': min((row['prompt_token_count'] for row in requests), default=None),
        'max_prompt_tokens': max((row['prompt_token_count'] for row in requests), default=None),
        'max_output_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'truncation': False,
        'requests_sha256': sha((HERE / 'REQUESTS.json').read_bytes()), 'self_checks': checks,
        'quality_gate': 'only content_pass is exactly true', 'gold_independent_permutation': True,
        'stable_option_ids_rebound': True, 'paired_context_and_root_arm_sampling_seeds': True,
        'author_review_source_inputs_bound': True, 'new_model_calls': 0})
    print(json.dumps(read(HERE / 'PREFLIGHT.json'), ensure_ascii=False), flush=True)


def summarize(rows):
    if not rows:
        return {'planned_calls': 0, 'valid_calls': 0, 'invalid_calls': 0,
                'exact_correct_calls': 0, 'exact_accuracy': None, 'mean_J': None,
                'mean_J_fraction': None}
    return baseline.summarize(rows)


def score():
    requests, raws = read(HERE / 'REQUESTS.json'), read(HERE / 'RAW_ANSWERS.json')
    assert len(requests) == len(raws) <= MAX_REQUESTS
    scored, answer_only_rows = [], []
    keys = ('request_id', 'candidate_id', 'root_id', 'arm', 'number', 'question_type', 'context_mode',
            'variant_id', 'repeat_index', 'sampling_seed', 'n', 'g')
    for request, raw in zip(requests, raws):
        assert request['request_id'] == raw['request_id']
        checked = reference.validate(raw, request)
        invalid = baseline.parse_and_score({'status': 'INVALID'}, request)
        answer = checked.get('answer_only_score', invalid)
        strict = answer if checked['protocol_valid'] else invalid
        metadata = {key: request[key] for key in keys}
        scored.append({**metadata, **strict, **checked, 'raw_prediction': raw.get('raw_prediction'),
                       'finish_reason': raw.get('finish_reason')})
        answer_only_rows.append({**metadata, **answer, 'raw_prediction': raw.get('raw_prediction'),
                                 'finish_reason': raw.get('finish_reason'), 'protocol_valid': checked['protocol_valid']})
    items = read(HERE / 'ITEMS_PRIVATE.json')
    def group(rows):
        by_item = {}
        for item in items:
            modes = {}
            for mode in ('full_context', 'no_context'):
                selected = [row for row in rows if row['candidate_id']==item['candidate_id'] and row['context_mode']==mode]
                modes[mode] = summarize(selected)
                modes[mode]['by_permutation'] = {f'perm_{index:02d}': summarize([row for row in selected if row['variant_id']==f'perm_{index:02d}'])
                    for index in range(item['permutations'])}
            by_item[item['candidate_id']] = modes
        return by_item
    aggregate = {mode: summarize([row for row in scored if row['context_mode']==mode])
                 for mode in ('full_context', 'no_context')}
    by_arm = {arm: {mode: summarize([row for row in scored if row['arm']==arm and row['context_mode']==mode])
                    for mode in ('full_context', 'no_context')} for arm in ('F', 'G')}
    write('SCORES.json', {'items': group(scored), 'requests': scored,
        'aggregate': aggregate, 'by_arm': by_arm, 'items_final_answer_only': group(answer_only_rows),
        'final_answer_only_aggregate': {mode: summarize([row for row in answer_only_rows if row['context_mode']==mode])
                                      for mode in ('full_context', 'no_context')},
        'comparison_limit': 'descriptive finite F/G comparison; quality, target realization and semantic errors require independent review'})
    write('SCORES_FINAL_ANSWER_ONLY.json', {'items': group(answer_only_rows), 'requests': answer_only_rows})
    print(json.dumps({'status': 'SCORED', 'requests': len(scored), 'aggregate': aggregate}), flush=True)


def run():
    assert not (HERE / 'STARTED.json').exists(), 'NO_RETRY_OR_OVERWRITE'
    reference.offline()
    requests, preflight = read(HERE / 'REQUESTS.json'), read(HERE / 'PREFLIGHT.json')
    assert sha((HERE / 'REQUESTS.json').read_bytes()) == preflight['requests_sha256']
    assert preflight['status'] in ('PASS', 'NO_CONTENT_PASS') and len(requests) <= MAX_REQUESTS
    for binding in read(HERE / 'INPUT_BINDINGS.json')['files']:
        assert bind(binding['path']) == binding, ('FROZEN_INPUT_CHANGED', binding['path'])
    for row in requests:
        assert sha((HERE / row['prompt_path']).read_bytes()) == row['prompt_sha256']
        assert sha((HERE / row['schema_path']).read_bytes()) == row['schema_sha256']
    if not requests:
        write('STARTED.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'planned_requests': 0, 'gpu_model_loaded': False})
        write('RAW_ANSWERS.json', [])
        write('COMPLETED.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'returned_requests': 0, 'gpu_model_loaded': False})
        score()
        return
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    write('STARTED.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'planned_requests': len(requests), 'vllm_version': vllm.__version__,
        'model': str(MODEL), 'automatic_retry': False, 'school_api': False})
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=True,
        dtype='bfloat16', max_model_len=MODEL_LEN, gpu_memory_utilization=.8,
        seed=ENGINE_SEED, enable_prefix_caching=False)
    raws = []
    for start in range(0, len(requests), BATCH):
        batch = requests[start:start+BATCH]
        prompts = [(HERE / row['prompt_path']).read_text(encoding='utf-8') for row in batch]
        params = [SamplingParams(n=1, temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1.,
            max_tokens=MAX_TOKENS, seed=row['sampling_seed'], stop=None,
            structured_outputs=StructuredOutputsParams(json=read(HERE / row['schema_path']))) for row in batch]
        # A failed run is preserved at STARTED/raw files; never silently retried.
        outputs = llm.generate(prompts, params, use_tqdm=False)
        assert len(outputs) == len(batch)
        for row, output in zip(batch, outputs):
            assert len(output.outputs) == 1
            completion = output.outputs[0]
            raw = {'request_id': row['request_id'], 'status': 'RETURNED',
                'raw_prediction': completion.text, 'finish_reason': completion.finish_reason,
                'stop_reason': completion.stop_reason, 'completion_token_ids': list(completion.token_ids),
                'prompt_sha256': row['prompt_sha256'], 'automatic_retry': False}
            write('raw/' + row['request_id'] + '.json', raw)
            raws.append(raw)
        print(json.dumps({'status': 'INFERENCE_PROGRESS', 'completed': len(raws), 'planned': len(requests)}), flush=True)
    write('RAW_ANSWERS.json', raws)
    write('COMPLETED.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'returned_requests': len(raws), 'vllm_version': vllm.__version__, 'automatic_retry': False})
    score()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('self-check', 'prepare', 'run', 'score'))
    args = parser.parse_args()
    if args.mode == 'self-check':
        print(self_checks())
    else:
        {'prepare': prepare, 'run': run, 'score': score}[args.mode]()
