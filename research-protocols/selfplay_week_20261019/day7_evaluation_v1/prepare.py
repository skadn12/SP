"""Freeze the existing final TEST requests without regenerating prompts or tasks.

CPU only. Private answers are copied into an isolated scoring packet; no model
is imported or called. All upstream prompt, mapping, choice, and sampling bytes
are retained. Existing sealed packages are never changed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = WEEK / 'day2/solver/adapter.py'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.dont_write_bytecode = True


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def textsha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canon(value):
    return textsha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def bindfile(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(bindings):
    for ref in bindings:
        require(bindfile(ref['path']) == ref, 'BOUND_FILE_CHANGED:' + ref['path'])


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def save_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    require(not (HERE / 'TEST_REQUEST_FREEZE.json').exists(), 'ALREADY_FROZEN')
    spec = importlib.util.spec_from_file_location('day7_public_adapter', ADAPTER)
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    lockpath = WEEK / 'day4_schedule_v1/EVALUATION_LOCK.json'
    lock = read(lockpath)
    cohorts = {r['cohort_id']: r for r in lock['cohorts']}
    inputs = [bindfile(lockpath), lock['unchanged_sampling_policy'], bindfile(ADAPTER)]
    for c in cohorts.values():
        inputs += [c['request_freeze']] + c['request_evidence_bindings']
    verify(inputs)
    histroot = Path(cohorts['historical115']['request_freeze']['path']).parent / 'evidence'
    hp = {r['candidate_id']: r for r in lines(histroot / 'ADMITTED_PRIVATE.jsonl') if r['split'] == 'test'}
    hq = {r['candidate_id']: r for r in lines(histroot / 'ADMITTED_PUBLIC.jsonl') if r['candidate_id'] in hp}
    hr = [r for r in lines(histroot / 'ADMITTED_REQUESTS.jsonl') if r['split'] == 'test']
    hm = {r['prompt_id']: r for r in read(histroot / 'ADMITTED_DISPLAY_MAPS.json')['variants'] if r['split'] == 'test'}
    prompts = {r['prompt_id']: r for r in lines(histroot / 'ADMITTED_PROMPTS.jsonl') if r['candidate_id'] in hp}
    choices = read(histroot / 'FINITE_ANSWER_CHOICES.json')['choices']
    require((len(hp), len(hr), len(hm)) == (75, 1230, 570), 'HISTORICAL_TEST_DENOMINATOR_CHANGED')
    rows, questions, private, maps = [], [], [], {}
    for cid, q in hq.items():
        a.public_question(q)
        questions.append({'cohort_id': 'historical115_TEST', 'split': 'test', **q})
        private.append({'cohort_id': 'historical115_TEST', 'root_id': q['root_id'], 'question_type': q['question_type'], **hp[cid]})
    native_keys = ('variant_id', 'options', 'items', 'display_to_content', 'canonical_order_content_ids', 'display_order_content_ids', 'content_to_text')
    for r in hr:
        pid = r['prompt_id']
        m = hm[pid]
        require(canon({k: m[k] for k in native_keys if k in m}) == r['display_mapping_sha256'], 'HISTORICAL_MAPPING_DRIFT')
        maps[pid] = m
        rows.append({'cohort_id': 'historical115_TEST', 'root_id': hq[r['candidate_id']]['root_id'], **r, 'prompt_tokens': prompts[pid]['prompt_tokens']})

    prevroot = Path(cohorts['previous4MC']['request_freeze']['path']).parent
    pr = [r for r in lines(prevroot / 'REQUESTS.jsonl') if r['original_split'] == 'test']
    pm = {(r['candidate_id'], r['variant_id']): r for r in read(prevroot / 'DISPLAY_MAPS.json')}
    require(len(pr) == 36 and len({r['candidate_id'] for r in pr}) == 2, 'PREVIOUS_TEST_DENOMINATOR_CHANGED')
    for cid in sorted({r['candidate_id'] for r in pr}):
        pubpath, privpath = (prevroot.parent / 'content' / sub / (cid + '.json') for sub in ('public', 'private'))
        inputs += [bindfile(pubpath), bindfile(privpath)]
        q = {k: v for k, v in read(pubpath).items() if k in a.PUBLIC_KEYS}
        a.public_question(q)
        p = read(privpath)
        require(p['original_split'] == 'test', 'NON_TEST_PREVIOUS_PRIVATE')
        ids = a.content_ids(q)
        questions.append({'cohort_id': 'previous2MC_TEST', 'split': 'test', **q})
        private.append({'cohort_id': 'previous2MC_TEST', 'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'], 'split': 'test', 'reporting_role': 'SEPARATE_PREVIOUS_MC_SUPPLEMENT', 'gold': p['gold'], 'gold_content_ids': [ids[label] for label in p['gold']]})
    for r in pr:
        pid = r['candidate_id'] + '::' + r['variant_id']
        require(pm[r['candidate_id'], r['variant_id']] == {'candidate_id': r['candidate_id'], **r['display']}, 'PREVIOUS_MAPPING_DRIFT')
        phash = textsha(r['prompt'])
        pp = {'prompt_id': pid, 'candidate_id': r['candidate_id'], 'variant_id': r['variant_id'], 'prompt': r['prompt'], 'prompt_sha256': phash, 'prompt_tokens': r['prompt_tokens']}
        require(pid not in prompts or prompts[pid] == pp, 'CONFLICTING_PROMPT_BYTES')
        prompts[pid] = pp
        maps[pid] = pm[r['candidate_id'], r['variant_id']]
        rows.append({'cohort_id': 'previous2MC_TEST', 'request_id': r['request_id'], 'candidate_id': r['candidate_id'], 'root_id': r['root_id'], 'split': 'test', 'reporting_role': 'SEPARATE_PREVIOUS_MC_SUPPLEMENT', 'question_type': r['question_type'], 'branch': 'direct', 'variant_id': r['variant_id'], 'sample_index': r['sample_index'], 'prompt_id': pid, 'prompt_sha256': phash, 'prompt_tokens': r['prompt_tokens'], 'sampling': {'temperature': r['temperature'], 'top_p': r['top_p'], 'max_tokens': r['max_output_tokens'], 'seed': r['seed'], 'stop': r['stop'], 'include_stop_str_in_output': r['include_stop_str_in_output']}, 'constraint_key': r['question_type'], 'context_mode': 'full_context'})

    supproot = Path(cohorts['supplement12']['request_freeze']['path']).parent
    sf = read(cohorts['supplement12']['request_freeze']['path'])
    verify([sf['content_admission'], sf['execution_freeze'], sf['native_contract'], sf['tokenization_proof']])
    inputs += [sf[k] for k in ('content_admission', 'execution_freeze', 'native_contract', 'tokenization_proof')]
    admission = read(sf['content_admission']['path'])
    verify(admission['input_bindings'])
    inputs += admission['input_bindings']
    sr = lines(supproot / 'REQUESTS.jsonl')
    sp = {r['prompt_id']: r for r in lines(supproot / 'PROMPTS.jsonl')}
    sm = {r['prompt_id']: r for r in read(supproot / 'DISPLAY_MAPS.json')['variants']}
    choices.update(read(supproot / 'FINITE_CHOICES.json')['choices'])
    require((len({r['candidate_id'] for r in sr}), len(sr), len(sp), len(sm)) == (12, 180, 108, 108), 'SUPPLEMENT_TEST_DENOMINATOR_CHANGED')
    for cid in sorted({r['candidate_id'] for r in sr}):
        admit = next(r for r in admission['records'] if r['candidate_id'] == cid)
        require(admit['status'] == 'CONTENT_ACCEPTED' and admit['original_split'] == 'test', 'SUPPLEMENT_NOT_ADMITTED_TEST')
        verify([admit['public_question']])
        pubpath = Path(admit['public_question']['path'])
        privpath = supproot.parent / 'content/private' / (cid + '.json')
        inputs += [admit['public_question'], bindfile(privpath)]
        q = a.public_question(read(pubpath))
        p = read(privpath)
        ids = a.content_ids(q)
        require(p['candidate_id'] == cid, 'SUPPLEMENT_PRIVATE_ID_DRIFT')
        questions.append({'cohort_id': 'supplement12_TEST', 'split': 'test', **q})
        private.append({'cohort_id': 'supplement12_TEST', 'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'], 'split': 'test', 'reporting_role': 'SEPARATE_MULTITYPE_SUPPLEMENT', 'gold': p['gold'], 'gold_content_ids': [ids[label] for label in p['gold']]})
    require(not (set(sp) & set(prompts)), 'SUPPLEMENT_PROMPT_ID_COLLISION')
    prompts.update(sp)
    for r in sr:
        m = sm[r['prompt_id']]
        require(canon({k: m[k] for k in native_keys if k in m}) == r['display_mapping_sha256'], 'SUPPLEMENT_MAPPING_DRIFT')
        maps[r['prompt_id']] = m
        rows.append({'cohort_id': 'supplement12_TEST', 'reporting_role': 'SEPARATE_MULTITYPE_SUPPLEMENT', **r, 'prompt_tokens': sp[r['prompt_id']]['prompt_tokens'], 'context_mode': 'full_context'})
    require(len(rows) == len({r['request_id'] for r in rows}) == 1446 and len(questions) == 89, 'TOTAL_TEST_DENOMINATOR_CHANGED')
    require(len({q['candidate_id'] for q in questions}) == len(private) == 89, 'DUPLICATE_QUESTION_ID')
    qmap = {q['candidate_id']: q for q in questions}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    checks = []
    for pid, p in prompts.items():
        require(textsha(p['prompt']) == p['prompt_sha256'], 'PROMPT_BYTES_CHANGED')
        ids = tok.encode(p['prompt'], add_special_tokens=False)
        require(len(ids) == p['prompt_tokens'] and len(ids) + 32 <= 16384, 'TOKEN_COUNT_OR_LENGTH_DRIFT')
        require('token_ids_sha256' not in p or canon(ids) == p['token_ids_sha256'], 'TOKEN_IDS_DRIFT')
        m = maps[pid]
        q = qmap[p['candidate_id']]
        canonical = a.content_ids({k: v for k, v in q.items() if k in a.PUBLIC_KEYS})
        require(set(m['display_to_content'].values()) == set(canonical.values()), 'MAPPING_CONTENT_DRIFT')
        inverse = {v: k for k, v in m['display_to_content'].items()}
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        require(all(m['content_to_text'][c] == m[field][label] for label, c in m['display_to_content'].items()), 'DISPLAY_TEXT_DRIFT')
        labels = tuple(canonical)
        answers = (itertools.chain.from_iterable(itertools.combinations(labels, n) for n in range(1, len(labels) + 1)) if field == 'options' else itertools.permutations(labels))
        rounds = 0
        for answer in answers:
            content = [canonical[x] for x in answer]
            require([m['display_to_content'][inverse[x]] for x in content] == content, 'ANSWER_ROUNDTRIP_FAILED')
            rounds += 1
        target = HERE / 'prompts' / (textsha(pid) + '.txt')
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:
            f.write(p['prompt'].encode())
        p['prompt_file'] = str(target)
        checks.append({'prompt_id': pid, 'prompt_sha256': p['prompt_sha256'], 'prompt_tokens': len(ids), 'token_ids_sha256': canon(ids), 'answer_space_roundtrips': rounds, 'truncated': False})
    for q in questions:
        require(sha(q['source_path']) == q['source_sha256'], 'SOURCE_CHANGED')
        inputs.append(bindfile(q['source_path']))
    for r in rows:
        require(r['split'] == 'test' and r['branch'] == 'direct', 'NON_TEST_OR_NON_DIRECT_REQUEST')
        p = prompts[r['prompt_id']]
        require(r['prompt_sha256'] == p['prompt_sha256'], 'REQUEST_PROMPT_DRIFT')
        require(r['constraint_key'] in choices, 'MISSING_FINITE_SPACE')
        require(r['sampling']['max_tokens'] == 32 and r['sampling']['stop'] == ['</box>'] and r['sampling']['include_stop_str_in_output'] is True, 'SAMPLING_DRIFT')
        r['prompt_file'] = p['prompt_file']
    for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
        inputs.append(bindfile(TOKENIZER / name))
    save_lines(HERE / 'TEST_REQUESTS.jsonl', rows)
    save_lines(HERE / 'TEST_PUBLIC_QUESTIONS.jsonl', questions)
    save_lines(HERE / 'private/TEST_GOLD.jsonl', private)
    save(HERE / 'TEST_DISPLAY_MAPS.json', maps)
    save(HERE / 'FINITE_CHOICES.json', choices)
    save(HERE / 'PROMPT_INDEX.json', [{k: v for k, v in p.items() if k != 'prompt'} for p in prompts.values()])
    save(HERE / 'CPU_PREFLIGHT.json', {'status': 'PASS', 'question_count': 89, 'request_count_per_model': 1446, 'prompt_count': len(prompts), 'prompt_checks': checks, 'prompt_tokens_max': max(r['prompt_tokens'] for r in rows), 'source_truncation': False, 'preserved_upstream_sampling': True, 'private_access': 'Preparation copied answers into isolated scoring packet; no model loaded or called', 'model_weights_loaded': False, 'new_solver_requests': 0})
    artifact_paths = [HERE / name for name in ('TEST_REQUESTS.jsonl', 'TEST_PUBLIC_QUESTIONS.jsonl', 'private/TEST_GOLD.jsonl', 'TEST_DISPLAY_MAPS.json', 'FINITE_CHOICES.json', 'PROMPT_INDEX.json', 'CPU_PREFLIGHT.json')] + sorted((HERE / 'prompts').glob('*.txt'))
    freeze = {'schema_version': 'day7_same_request_TEST_only_v1', 'status': 'CPU_PASS_AWAITING_FINAL_TEST_MODEL_BINDINGS_AND_INDEPENDENT_REVIEW', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model_roles': ['S0', 'A1', 'B1'], 'missing_model_roles': {'A2': 'NOT_PRODUCED', 'B2': 'NOT_PRODUCED'}, 'question_count': 89, 'requests_per_model': 1446, 'total_planned_solver_requests': 4338, 'cohorts': [{'cohort_id': 'historical115_TEST', 'questions': 75, 'requests_per_model': 1230, 'primary_questions': 64, 'primary_requests_per_model': 1032, 'gold6_control_questions': 11, 'gold6_control_requests_per_model': 198}, {'cohort_id': 'previous2MC_TEST', 'questions': 2, 'requests_per_model': 36, 'separate_denominator': True}, {'cohort_id': 'supplement12_TEST', 'questions': 12, 'requests_per_model': 180, 'separate_denominator': True}], 'prompt_count': len(prompts), 'planned_prompt_tokens_per_model': sum(r['prompt_tokens'] for r in rows), 'maximum_output_tokens_per_model': 1446 * 32, 'maximum_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'batch_size': 8, 'max_model_len': 16384, 'gpu_memory_utilization': .8, 'prefix_caching': False, 'source_truncation': False, 'automatic_retry': False, 'sampling_inherited_byte_equivalent': True, 'all_prompt_and_token_bytes_verified': True, 'test_requests_per_model': 1446, 'dev_requests': 0, 'checkpoint_selection': 'FIXED_FINAL_OPTIMIZER_STEP_56_NO_DEV_OR_TEST_SELECTION', 'challenger_release': 'NONE: TEST outputs, facts, examples, gold or diagnoses cannot be fed to Challenger', 'historically_exposed': True, 'overlapping_roots_across_cohorts': True, 'no_unseen_root_claim': True, 'raw_generation_gold_access': False, 'preparation_private_access': 'Read TEST answers to create isolated scoring packet only; no model called', 'execution_permitted': False, 'new_solver_calls': 0, 'model_weights_loaded': False, 'input_bindings': list({x['path']: x for x in inputs}.values()), 'artifact_bindings': [bindfile(p) for p in artifact_paths], 'wrapper': bindfile(__file__), 'runtime_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'vllm')}}
    save(HERE / 'TEST_REQUEST_FREEZE.json', freeze)
    print(json.dumps({k: freeze[k] for k in ('status', 'question_count', 'requests_per_model', 'prompt_count', 'total_planned_solver_requests', 'maximum_prompt_tokens', 'new_solver_calls')}), flush=True)


if __name__ == '__main__':
    main()
