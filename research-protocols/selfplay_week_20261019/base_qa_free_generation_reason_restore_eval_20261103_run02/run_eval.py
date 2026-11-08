#!/usr/bin/env python3
"""Paired rerun restoring historical visible reasons; no historical writes."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
BASE = WEEK / 'base_qa_free_generation_solver_eval_20261103_run02'
MODEL = Path('/root/autodl-tmp/Qwen3-4B')
HISTORY = WEEK / 'natural_question_forms_fg_15root_extension_20261026_v1'
REASON = WEEK / 'postweek_diversity_reason_v1/run.py'
REPAIR = WEEK / 'postweek_reason_feedback_repair_v1/repair.py'
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 12


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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


baseline = load('frozen_free02_score', BASE / 'run_eval.py')
historical = load('historical_visible_reason', REASON)
repair = load('historical_reason_schema', REPAIR)


def offline():
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'VLLM_USE_FLASHINFER_SAMPLER': '0', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}.items():
        os.environ[key] = value


def schema_for(labels, refs):
    # Keep historical fields/limits, generalize only label alphabet and count.
    result = copy.deepcopy(repair.schema(labels, refs))
    alternative = '(?:' + '|'.join(re.escape(label) for label in labels) + ')'
    result['properties']['final_answer']['pattern'] = (
        r'^<box>' + alternative + r'( ' + alternative + r'){0,' + str(len(labels)-1) + r'}</box>$')
    if not refs:
        for item in result['properties']['item_evidence']['properties'].values():
            item['properties']['turn_refs'] = {'type': 'array', 'maxItems': 0,
                                             'items': {'type': 'string'}}
    return result


def prompt_for(tokenizer, item, request, turns):
    labels = list(request['display_to_option_id'])
    example = '<box>' + ' '.join(labels[:min(3, len(labels))]) + '</box>'
    if turns:
        task = historical.COMMON.replace('<box>A B C</box>', example)
        task = task.replace(
            'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.',
            'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.')
        dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        source_text = f'Dialogue:\n{dialogue}\n\n'
        system = "Use only the supplied dialogue. Follow the question's requested operation. In the final response, give concise, verifiable explanations before the final boxed answer."
    else:
        task = f'''No source dialogue or source references are supplied in this condition.
Give your best prediction using the question and displayed items only. For each
displayed label, give a judgment (supported, contradicted, not_supported or
ambiguous) and a concise reason (at most 320 characters). Here supported means
your best assessment from the visible wording, not a verified source claim.
Every turn_refs array must be empty []; do not invent or cite hidden dialogue.
Give answer_reason (at most 800 characters), then final_answer as
{example}: selected labels for multi_select, every label exactly once for
ordering. The example only illustrates format. Return only JSON, with
item_evidence first, answer_reason second and final_answer last. Give brief
visible explanations, not extended private deliberation.'''
        source_text = ''
        system = 'Make a best prediction from the question and displayed items. No dialogue is supplied. Never fabricate source references. Give a brief explanation and final boxed answer.'
    # No operation class inferred from Gold or supplied as a new solution hint.
    user = (source_text + f"Question ({item['question_type']}):\n{item['question']}\n\nDisplayed items:\n"
            + '\n'.join(f"{o['id']}. {o['text']}" for o in request['options'])
            + '\n\n' + task)
    return tokenizer.apply_chat_template([{'role': 'system', 'content': system},
        {'role': 'user', 'content': user}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def validate(raw, request):
    """No repair/fallback: final answer must occur in its designated JSON field."""
    result = {'protocol_valid': False, 'protocol_errors': [], 'parsed_output': None,
              'final_answer': None, 'final_answer_valid': False}
    try:
        assert raw['status'] == 'RETURNED', 'BACKEND_NOT_RETURNED'
        assert raw.get('finish_reason') in ('stop', 'eos'), 'TRUNCATED_OR_INCOMPLETE'
        def unique_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('DUPLICATE_JSON_KEY:' + key)
                value[key] = item
            return value
        obj = json.loads(raw['raw_prediction'], object_pairs_hook=unique_pairs)
        result['parsed_output'] = obj
        assert isinstance(obj, dict), 'JSON_OBJECT_REQUIRED'
        final = obj.get('final_answer')
        result['final_answer'] = final
        assert isinstance(final, str), 'FINAL_ANSWER_STRING_REQUIRED'
        alternative = '(?:' + '|'.join(re.escape(label) for label in request['display_to_option_id']) + ')'
        match = re.fullmatch(r'<box>(' + alternative + r'(?: ' + alternative + r')*)</box>', final)
        assert match, 'FINAL_BOX_SYNTAX'
        labels = match.group(1).split(' ')
        adapter = {'status': 'RETURNED', 'finish_reason': raw['finish_reason'],
                   'raw_prediction': (',' if request['question_type'] == 'multi_select' else '>').join(labels)}
        scored = baseline.parse_and_score(adapter, request)
        result['final_answer_valid'] = scored['valid']
        result['answer_only_score'] = scored
        assert scored['valid'], scored['parse_reason']
        from jsonschema import validate as json_validate
        json_validate(obj, read(HERE / request['schema_path']))
        assert list(obj) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
        assert obj['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
        for item in obj['item_evidence'].values():
            assert item['reason'].strip(), 'EMPTY_ITEM_REASON'
            assert len(item['turn_refs']) == len(set(item['turn_refs'])), 'DUPLICATE_REFERENCE'
            if request['context_mode'] == 'full_context':
                assert item['judgment'] != 'supported' or item['turn_refs'], 'SUPPORTED_WITHOUT_CITATION'
                if request['question_type'] == 'ordering':
                    assert item['turn_refs'], 'ORDER_ITEM_WITHOUT_CITATION'
            else:
                assert item['turn_refs'] == [], 'NO_CONTEXT_REFERENCE_FORBIDDEN'
        result['protocol_valid'] = True
    except Exception as exc:
        result['protocol_errors'].append(str(exc))
    return result


def prepare():
    offline()
    assert not (HERE / 'REQUESTS.json').exists(), 'NO_OVERWRITE'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    frozen_manifest = read(BASE / 'MANIFEST_FINAL.json')
    for relative, digest in frozen_manifest['files'].items():
        assert sha((BASE / relative).read_bytes()) == digest, relative
    items = read(BASE / 'ITEMS_PRIVATE.json')
    by_id = {i['candidate_id']: i for i in items}
    sources = {i['candidate_id']: read(i['source_path']) for i in items}
    requests, bindings = [], [bind(BASE / 'MANIFEST_FINAL.json'), bind(REASON), bind(REPAIR),
        bind(HISTORY / 'runner.py'), bind(HISTORY / 'PROTOCOL_FREEZE.json'), bind(Path(__file__)),
        bind(BASE / 'run_eval.py'), bind(MODEL / 'config.json'), bind(MODEL / 'tokenizer_config.json')]
    for relative in frozen_manifest['files']:
        bindings.append(bind(BASE / relative))
    for item in items:
        bindings.append(bind(item['source_path']))
        assert sha(Path(item['source_path']).read_bytes()) == item['source_sha256']
    for old in read(BASE / 'REQUESTS.json'):
        item = by_id[old['candidate_id']]
        turns = sources[old['candidate_id']]['turns'] if old['context_mode'] == 'full_context' else []
        prompt = prompt_for(tokenizer, item, old, turns)
        tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        assert tokens + MAX_TOKENS <= MODEL_LEN, ('NO_TRUNCATION', old['request_id'], tokens)
        refs = [t['turn_ref'] for t in turns]
        schema = schema_for(list(old['display_to_option_id']), refs)
        prompt_path = 'prompts/' + old['request_id'] + '.txt'
        schema_path = 'schemas/' + old['request_id'] + '.json'
        (HERE / prompt_path).parent.mkdir(exist_ok=True)
        with (HERE / prompt_path).open('x', encoding='utf-8') as stream:
            stream.write(prompt)
        write(schema_path, schema)
        row = copy.deepcopy(old)
        row.update(prompt_path=prompt_path, schema_path=schema_path, prompt_sha256=sha(prompt.encode()),
                   prompt_token_count=tokens, schema_sha256=sha((HERE / schema_path).read_bytes()),
                   baseline_prompt_sha256=old['prompt_sha256'])
        requests.append(row)
    write('ITEMS_PRIVATE.json', items)
    write('PERMUTATIONS_PRIVATE.json', read(BASE / 'PERMUTATIONS_PRIVATE.json'))
    write('REQUESTS.json', requests)
    write('INPUT_BINDINGS.json', {'files': bindings})
    write('MODEL_CONFIG.json', {'model': str(MODEL), 'tokenizer': str(MODEL),
        'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1.0,
        'max_new_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'dtype': 'bfloat16',
        'gpu_memory_utilization': .8, 'engine_seed': 2026110303, 'batch_size': BATCH,
        'native_thinking': False, 'visible_reasons': True, 'structured_outputs': True,
        'stop_sequences': None, 'offline': True, 'automatic_retry': False, 'training': False,
        'checkpoint_unchanged_from_recent_baseline': True,
        'historical_format_reference': str(HISTORY),
        'comparison': 'restore visible-reason prompt+JSON schema+6144-token output protocol; not an isolated native-thinking or CoT causal test',
        'historical_differences': ['keep recent base checkpoint, not historical B1 checkpoint',
            'keep recent questions, source versions, source turn IDs, permutations, repeat counts and seeds',
            'generalize historical A-F schema to actual A-H or I1-In labels',
            'use neutral historical common instructions without adding operation-specific hints',
            'current vLLM runtime differs from historical v0.21.0, recorded at execution'],
        'primary_scoring': 'strict output protocol; invalid=0 in fixed denominators',
        'secondary_scoring': 'final-answer-only validity isolated from evidence/reason protocol errors',
        'multi_select_J': 'TP/g-FP/(n-g)', 'ordering_J': '(correct_pairs-wrong_pairs)/C(k,2)'})
    write('PREFLIGHT.json', {'status': 'PASS', 'request_count': len(requests), 'candidate_count': len(items),
        'full_context_requests': sum(r['context_mode'] == 'full_context' for r in requests),
        'min_prompt_tokens': min(r['prompt_token_count'] for r in requests),
        'max_prompt_tokens': max(r['prompt_token_count'] for r in requests),
        'max_output_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'truncation': False,
        'requests_sha256': sha((HERE / 'REQUESTS.json').read_bytes()),
        'source_gold_mapping_sampling_unchanged': True})
    print(json.dumps(read(HERE / 'PREFLIGHT.json')), flush=True)


def score():
    scored = []
    requests, raws = read(HERE / 'REQUESTS.json'), read(HERE / 'RAW_ANSWERS.json')
    assert len(requests) == len(raws) == 396
    for request, raw in zip(requests, raws):
        assert request['request_id'] == raw['request_id']
        checked = validate(raw, request)
        invalid = baseline.parse_and_score({'status': 'INVALID'}, request)
        answer = checked.get('answer_only_score', invalid)
        strict = answer if checked['protocol_valid'] else invalid
        scored.append({**{k: request[k] for k in ('request_id', 'candidate_id', 'question_type', 'context_mode',
            'variant_id', 'repeat_index', 'sampling_seed', 'n', 'g')}, **strict, **checked,
            'raw_prediction': raw.get('raw_prediction'), 'finish_reason': raw.get('finish_reason')})
    summaries = {}
    for item in read(HERE / 'ITEMS_PRIVATE.json'):
        modes = {}
        for mode in ('full_context', 'no_context'):
            rows = [s for s in scored if s['candidate_id'] == item['candidate_id'] and s['context_mode'] == mode]
            modes[mode] = baseline.summarize(rows)
        summaries[item['candidate_id']] = modes
    write('SCORES.json', {'items': summaries, 'requests': scored})
    print(json.dumps({'status': 'SCORED', 'requests': len(scored)}), flush=True)


def run():
    assert not (HERE / 'STARTED.json').exists(), 'NO_RETRY_OR_OVERWRITE'
    offline()
    requests = read(HERE / 'REQUESTS.json')
    assert sha((HERE / 'REQUESTS.json').read_bytes()) == read(HERE / 'PREFLIGHT.json')['requests_sha256']
    for binding in read(HERE / 'INPUT_BINDINGS.json')['files']:
        assert bind(binding['path']) == binding, binding['path']
    for row in requests:
        assert sha((HERE / row['prompt_path']).read_bytes()) == row['prompt_sha256']
        assert sha((HERE / row['schema_path']).read_bytes()) == row['schema_sha256']
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    write('STARTED.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'planned_requests': len(requests), 'vllm_version': vllm.__version__, 'automatic_retry': False})
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=True,
        dtype='bfloat16', max_model_len=MODEL_LEN, gpu_memory_utilization=.8,
        seed=2026110303, enable_prefix_caching=False)
    raws = []
    for start in range(0, len(requests), BATCH):
        batch = requests[start:start+BATCH]
        prompts = [(HERE / r['prompt_path']).read_text(encoding='utf-8') for r in batch]
        params = [SamplingParams(n=1, temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1.,
            max_tokens=MAX_TOKENS, seed=r['sampling_seed'],
            structured_outputs=StructuredOutputsParams(json=read(HERE / r['schema_path']))) for r in batch]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        assert len(outputs) == len(batch)
        for row, output in zip(batch, outputs):
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
        'returned_requests': len(raws), 'automatic_retry': False})
    score()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.mode]()
