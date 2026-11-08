"""Independent frozen-citation preflight and post-freeze result audit.

Does not import preparation, inference, or scoring implementations. Preflight
uses public questions, complete sources and old raw responses, never old scores
or private Gold. The post-freeze audit opens Gold only after all raw bytes pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'day2_order_probe_v1'
DAY1, DAY2 = HERE.parent / 'day1', HERE.parent / 'day2'
ARMS = ('freeze_quotes', 'explicit_positions')
LABELS = ['I' + str(i) for i in range(1, 7)]
PINNED_SEALS = {
    DAY1: '8918702665d1965e3dc54015239a871213df24c7d9d1db983d66f0bb013c30e8',
    DAY2: '4411bdaedfbedc4c0b478f59ceda9e9a70b65f15c13e2a915b76c555a5dfafa7',
    OLD: '2b8a0d0565e094860f91d94f67e99a0bf8a3d566f47ecf630df659d41e24528b',
}
CHECKS, ISSUES, HASH_CACHE = [], [], {}


def now():
    return datetime.now(timezone.utc).isoformat()


def moment(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def read(path):
    return json.loads(Path(path).read_text())


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('DUPLICATE_JSON_KEY')
        value[key] = item
    return value


def strict_json(value):
    return json.loads(value, object_pairs_hook=strict_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE')))


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check(name, value, detail=None):
    ok = bool(value)
    CHECKS.append({'name': name, 'pass': ok})
    if not ok:
        ISSUES.append({'name': name, 'detail': detail})
    return ok


def need(name, value):
    if not check(name, value):
        raise AssertionError(name)


def equal(name, actual, expected):
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        return check(name, math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), {'actual': actual, 'expected': expected})
    return check(name, actual == expected, {'actual': actual, 'expected': expected})


def compare(name, actual, expected):
    for key, value in expected.items():
        if isinstance(value, dict):
            need(name + ':' + key + ':dict', isinstance(actual.get(key), dict))
            compare(name + ':' + key, actual[key], value)
        else:
            equal(name + ':' + key, actual.get(key), value)


def bindings(rows, name):
    for i, row in enumerate(rows):
        p = Path(row['path'])
        need(f'{name}:file:{i}', p.is_file())
        need(f'{name}:hash:{i}', p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])


def sealed_snapshots():
    for directory, pinned in PINNED_SEALS.items():
        path = directory / 'COMPLETE_MANIFEST.json'
        need(directory.name + ':pinned_seal', sha(path) == pinned)
        value = read(path)
        need(directory.name + ':sealed', value.get('status') == 'SEALED_DAY1_COMPLETE' if directory == DAY1 else value.get('sealed') is True)
        need(directory.name + ':counts', len(value['files']) == value['file_count'] and sum(r['bytes'] for r in value['files']) == value['total_bytes'])
        expected = set()
        for row in value['files']:
            rel = Path(row['path'])
            need(directory.name + ':relative_path', not rel.is_absolute() and '..' not in rel.parts)
            expected.add((directory / rel).resolve())
        bindings([{**r, 'path': str(directory / r['path'])} for r in value['files']], directory.name + ':snapshot')
        actual = {p.resolve() for p in directory.rglob('*') if p.is_file() and p.name != 'COMPLETE_MANIFEST.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        need(directory.name + ':file_set', actual == expected)


def independent_positions(evidence, source):
    locations = {t['turn_ref']: (i, t['text']) for i, t in enumerate(source['turns'])}
    positions = []
    for item in evidence:
        located = locations.get(item['turn_ref'])
        quote = item['quote']
        offsets = [] if not located or not quote else [i for i in range(len(located[1])) if located[1].startswith(quote, i)]
        if len(offsets) != 1:
            return None
        positions.append({'label': item['label'], 'source_turn_index': located[0], 'quote_start_offset': offsets[0]})
    if len(positions) != 6 or len({(p['source_turn_index'], p['quote_start_offset']) for p in positions}) != 6:
        return None
    return positions


def reconstruct_selection():
    import jsonschema
    old = read(OLD / 'run/manifest.json')
    oldfreeze = read(OLD / 'run/RAW_RESPONSE_FREEZE.json')
    need('prior_raw_committed', oldfreeze['status'] == 'COMMITTED')
    bindings(oldfreeze['files'] + [oldfreeze['manifest_binding']], 'prior_raw')
    controls = sorted([r for r in old['requests'] if r['probe_arm'] == 'control'], key=lambda x: x['request_id'])
    need('population54_controls', len(controls) == 54)
    selected, excluded, inventory = [], [], []
    for row in controls:
        rid = row['request_id']
        rawpath = OLD / 'run/responses' / (rid + '.json')
        sourcepath = OLD / 'run/sources' / (row['root_id'] + '.json')
        raw, source = read(rawpath), read(sourcepath)
        need('old_source_unique_refs:' + rid, len({t['turn_ref'] for t in source['turns']}) == len(source['turns']))
        reason = None
        try:
            if raw['status'] != 'RETURNED' or raw['metadata']['finish_reason'] not in ('stop', 'eos'):
                raise AssertionError('not returned')
            value = strict_json(raw['raw_output'])
            jsonschema.validate(value, row['json_schema'])
            if len(value['evidence']) != 6 or {x['label'] for x in value['evidence']} != set(LABELS):
                raise AssertionError('invalid evidence labels')
            # Structural validity is checked, but answer ordering/correctness is
            # deliberately not used in eligibility or reconstruction.
            if len(value['answer']) != 6 or set(value['answer']) != set(LABELS):
                raise AssertionError('invalid answer labels')
            evidence = sorted(value['evidence'], key=lambda x: LABELS.index(x['label']))
            positions = independent_positions(evidence, source)
            if positions is None:
                reason = 'NONEXACT_NONUNIQUE_OR_TIED_POSITIONS'
        except Exception as error:
            reason = 'STRICT_INVALID:' + type(error).__name__
        base = {key: row[key] for key in ('candidate_id', 'root_id', 'variant_id')}
        base.update(frozen_record_id=rid, raw_binding=bind(rawpath), source_binding=bind(sourcepath))
        inventory.append({**base, 'selected': reason is None, 'reason': reason})
        if reason is not None:
            excluded.append({**base, 'reason': reason})
        else:
            ordered = sorted(positions, key=lambda p: (p['source_turn_index'], p['quote_start_offset']))
            selected.append({**base, 'display': row['display'], 'evidence': evidence, 'positions': positions,
                             'citation_sorted_display': [p['label'] for p in ordered],
                             'semantic_support': 'REVIEWED_SEPARATELY_DOES_NOT_SELECT_SAMPLE'})
    need('fixed11_records5_roots43excluded', len(selected) == 11 and len({x['root_id'] for x in selected}) == 5 and len(excluded) == 43)
    return old, selected, excluded, inventory


def independent_prompt(tok, question, record, source, policy, arm):
    parts = ['Dialogue:\n' + '\n'.join('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in source['turns']),
             'Question (ordering):\n' + question['question'],
             'Items:\n' + '\n'.join(k + '. ' + v for k, v in record['display']['items'].items()),
             'Frozen citations:\n' + '\n'.join(e['label'] + ': turn_ref=' + json.dumps(e['turn_ref'], ensure_ascii=False) + '; quote=' + json.dumps(e['quote'], ensure_ascii=False) for e in record['evidence']),
             policy['prompt']['task']]
    if arm == 'explicit_positions':
        parts.append(policy['prompt']['positions_heading'] + '\n' + '\n'.join(str(p['label']) + ': turn_index=' + str(p['source_turn_index']) + ', quote_char_offset=' + str(p['quote_start_offset']) for p in record['positions']))
    return tok.apply_chat_template([{'role': 'system', 'content': policy['prompt']['system']}, {'role': 'user', 'content': '\n\n'.join(parts)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def common_inputs(run):
    m = read(run / 'manifest.json')
    freeze = read(run / 'PREFLIGHT_FREEZE.json')
    need('prefreeze_no_inference', freeze['inference_started'] is False)
    bindings(freeze['files'], 'preflightfreeze')
    for key in ('code_bindings', 'input_bindings', 'model_bindings'):
        need('nonempty:' + key, bool(m[key]))
        bindings(m[key], key)
    for key in ('private_answer_binding', 'content_admission_binding', 'protocol_binding', 'frozen_evidence_binding', 'baseline_inventory_binding'):
        bindings([m[key]], key)
    need('new_code_bound', {p.resolve() for p in HERE.glob('*.py')} <= {Path(x['path']).resolve() for x in m['code_bindings']})
    need('no_prepare_gold', m['gold_opened_by_prepare'] is False)
    need('fixed_train_S0_direct', m['namespace'] == 'train.feedback' and m['solver_model'] == 'S0_M_clean' and m['branch'] == 'direct' and m['use_as_final_eval'] is False)
    need('fixed_runtime', m['batch_size'] == 8 and m['max_model_len'] == 16384 and not m['prefix_caching'] and not m['automatic_retry'] and not m['source_truncation'])
    need('fixed_denominators', (m['request_count'], m['question_count'], m['root_count'], m['frozen_record_count']) == (88, 5, 5, 11))
    policy = read(HERE / 'PROTOCOL.json')
    need('fixed_repeats', policy['sampling']['repeat_ids'] == ['r00', 'r01', 'r02', 'r03'])
    need('fixed_sampling', all(policy['sampling'][k] == v for k, v in {'temperature': .8, 'top_p': .9, 'max_output_tokens': 32, 'max_model_len': 16384, 'batch_size': 8, 'prefix_caching': False, 'automatic_retry': False, 'source_truncation': False, 'enable_thinking': False}.items()))
    need('fixed88_planned', policy['planned'] == {'frozen_records': 11, 'independent_questions': 5, 'independent_roots': 5, 'matched_pairs': 44, 'requests': 88, 'requests_per_arm': 44})
    policy_freeze = read(HERE / 'POLICY_FREEZE.json')
    need('policy_before_new_inference', policy_freeze['status'] == 'FROZEN_BEFORE_NEW_INFERENCE' and policy_freeze['new_solver_calls_at_freeze'] == 0)
    bindings(policy_freeze['files'] + [policy_freeze['prior_sealed_snapshot']], 'policyfreeze')
    equal('prior_seal_binding', policy_freeze['prior_sealed_snapshot'], bind(OLD / 'COMPLETE_MANIFEST.json'))
    return m, policy


def preflight(run):
    need('not_live', not any((run / p).exists() for p in ('STARTED.json', 'RAW_RESPONSE_FREEZE.json', 'SCORES.json', 'SCORING_STARTED.json')))
    sealed_snapshots()
    m, policy = common_inputs(run)
    frozen = read(run / 'PREFLIGHT_FREEZE.json')
    actual = {p.resolve() for p in run.rglob('*') if p.is_file() and p.name != 'PREFLIGHT_FREEZE.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    need('all_prepared_files_frozen', actual == {Path(x['path']).resolve() for x in frozen['files']})
    old, expected, excluded, inventory = reconstruct_selection()
    selected = read(run / 'FROZEN_EVIDENCE.json')
    equal('selected_records_independent', selected['records'], expected)
    equal('excluded43_independent', selected['excluded_records'], excluded)
    equal('inventory54_independent', read(HERE / 'BASELINE_INVENTORY.json')['records'], inventory)
    need('semantic_does_not_select', selected['semantics_does_not_select_samples'] is True)
    public = read(run / 'PUBLIC_QUESTIONS.json')['questions']
    old_public = {q['candidate_id']: q for q in read(OLD / 'run/PUBLIC_QUESTIONS.json')['questions']}
    equal('five_original_public_questions', public, [old_public[qid] for qid in sorted({r['candidate_id'] for r in expected})])
    questions = {q['candidate_id']: q for q in public}
    ids = read(run / 'PROMPT_TOKEN_IDS.json')
    maps = read(run / 'private/DISPLAY_MAPPINGS.json')
    old_maps = read(OLD / 'run/private/DISPLAY_MAPPINGS.json')
    rows = m['requests']
    need('88_unique_requests', len(rows) == len({r['request_id'] for r in rows}) == 88)
    need('mapping_token_denominator', set(ids) == set(maps['requests']) == {r['request_id'] for r in rows})
    equal('44_each_arm', dict(Counter(r['probe_arm'] for r in rows)), dict.fromkeys(ARMS, 44))
    equal('same_original_model', m['model_bindings'], old['model_bindings'])
    need('same_model_tokenizer_paths', m['model'] == old['model'] and m['tokenizer'] == old['tokenizer'])
    equal('same_direct_decoder_contract', m['decoder_contract'], old['decoder_contract'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    sys.path.insert(0, str(DAY2 / 'solver'))
    import adapter
    from vllm import SamplingParams, sampling_params as sp
    from vllm.config import ModelConfig, StructuredOutputsConfig
    import xgrammar as xgr
    choices = ['<box>' + ','.join(p) + '</box>' for p in itertools.permutations(LABELS)]
    need('720_choices_original_decoder', len(choices) == 720 and choices == adapter.choices('ordering', dict.fromkeys(LABELS)))
    expected_order, expected_pairs, cursor = [], [], 0
    last_params = None
    for record in expected:
        question = questions[record['candidate_id']]
        sourcepath = run / 'sources' / (record['root_id'] + '.json')
        need('source_exact:' + record['root_id'], sha(sourcepath) == question['source_sha256'] == sha(question['source_path']))
        source = read(sourcepath)
        canonical = {k: 'content_' + hashlib.sha256(v.strip().encode()).hexdigest()[:16] for k, v in question['items'].items()}
        equal('canonical_map:' + record['candidate_id'], maps['public_to_canonical'][record['candidate_id']], canonical)
        for repeat in ['r00', 'r01', 'r02', 'r03']:
            pid = record['frozen_record_id'] + '__' + repeat
            seed = int.from_bytes(hashlib.sha256(('2026101907|' + pid).encode()).digest()[:8], 'big') % (2**31 - 1)
            order = list(ARMS)
            if hashlib.sha256(('frozen-arm-order|' + pid).encode()).digest()[0] % 2:
                order.reverse()
            pair = {k: record[k] for k in ('frozen_record_id', 'candidate_id', 'root_id', 'variant_id')}
            pair.update(pair_id=pid, repeat_id=repeat, seed=seed)
            row_pair = pair.copy()
            for arm in order:
                row = rows[cursor]; cursor += 1
                rid = pid + '__' + arm
                expected_order.append(rid)
                compare('metadata:' + rid, row, {**row_pair, 'request_id': rid, 'question_type': 'ordering', 'branch': 'direct', 'probe_arm': arm, 'slot': int(repeat[1:]), 'display': record['display'], 'source_turns': len(source['turns']), 'source_sha256': question['source_sha256'], 'max_output_tokens': 32})
                need('no_cross_arm_id_metadata:' + rid, not any(k.endswith('_request_id') for k in row))
                equal('unchanged_display_variant:' + rid, maps['requests'][rid], old_maps['requests'][record['frozen_record_id']])
                prompt = (run / 'prompts' / (rid + '.txt')).read_text()
                expected_prompt = independent_prompt(tokenizer, question, record, source, policy, arm)
                equal('prompt_reconstruction:' + rid, prompt, expected_prompt)
                need('promptsha:' + rid, hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'])
                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                equal('tokenids:' + rid, ids[rid], encoded)
                need('full_context_budget:' + rid, len(encoded) == row['prompt_tokens'] and len(encoded) + 32 <= 16384)
                need('all_full_turns:' + rid, all('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] in prompt for t in source['turns']))
                need('no_sorted_labels_or_gold_fields:' + rid, not any(k in row for k in ('gold', 'citation_sorted_display', 'positions')))
                params = adapter.sampling_params(row, m['decoder_contract']['structured_output_api'])
                need('sampling_values:' + rid, params.temperature == .8 and params.top_p == .9 and params.max_tokens == 32 and params.seed == seed and params.stop == ['</box>'] and params.include_stop_str_in_output)
                constraint = getattr(params, 'structured_outputs', None) or getattr(params, 'guided_decoding', None)
                need('same720_finite_choices:' + rid, constraint.choice == choices and constraint.json is None)
                last_params = params
                pair[arm + '_request_id'] = rid
            expected_pairs.append(pair)
    equal('request_order', [r['request_id'] for r in rows], expected_order)
    equal('matched44_pairs', read(run / 'MATCHED_PAIRS.json')['pairs'], expected_pairs)
    need('headroom_metadata', m['max_prompt_tokens'] == max(r['prompt_tokens'] for r in rows) and m['minimum_context_headroom'] == min(16384 - r['prompt_tokens'] - 32 for r in rows))
    model_config = ModelConfig(model=m['model'], tokenizer=m['tokenizer'], max_model_len=16384, trust_remote_code=True)
    last_params.verify(model_config=model_config, speculative_config=None, structured_outputs_config=StructuredOutputsConfig(backend='auto'), tokenizer=tokenizer)
    need('real_samplingparams_CPU_verify', True)
    expression = '(?:' + '|'.join(re.escape(c) for c in choices) + ')'
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model_config.get_vocab_size()), max_threads=4)
    grammar = compiler.compile_regex(expression)
    need('compiled_choice_grammar_accepts_all720', all(xgr.GrammarMatcher(grammar).accept_string(c) for c in choices))
    need('compiled_choice_grammar_rejects_duplicate', not xgr.GrammarMatcher(grammar).accept_string('<box>I1,I2,I3,I4,I5,I1</box>'))
    need('compiled_choice_grammar_rejects_explanation', not xgr.GrammarMatcher(grammar).accept_string('Answer: ' + choices[0]))
    need('still_not_started', not (run / 'STARTED.json').exists())
    return {'new_solver_calls': 0, 'private_gold_json_read_by_this_audit': False, 'old_scores_json_read_by_this_audit': False,
            'old_answers_correctness_used_for_selection': False, 'model_weights_loaded': False,
            'actual_vllm_verify': True, 'actual_grammar_compile': True,
            'questions': 5, 'roots': 5, 'frozen_records': 11, 'excluded_records': 43, 'pairs': 44, 'requests': 88}


def parse_answer(raw):
    if raw['status'] != 'RETURNED':
        return None, False
    text = raw['raw_output']
    match = re.fullmatch(r'<box>(I[1-6](?:,I[1-6]){5})</box>', text)
    prediction = match.group(1).split(',') if match else None
    valid = prediction is not None and len(set(prediction)) == 6 and raw.get('metadata', {}).get('finish_reason') in ('stop', 'eos')
    return prediction, bool(valid)


def pair_correct(prediction, target, valid):
    return sum(prediction.index(target[i]) < prediction.index(target[j]) for i in range(6) for j in range(i + 1, 6)) if valid else 0


def results(run):
    m, policy = common_inputs(run)
    frozen_raw = read(run / 'RAW_RESPONSE_FREEZE.json')
    need('raw_committed_before_gold', frozen_raw['status'] == 'COMMITTED')
    bindings(frozen_raw['files'] + [frozen_raw['manifest_binding']], 'rawfreeze')
    rows = m['requests']
    ids = {r['request_id'] for r in rows}
    need('88_unique_before_gold', len(ids) == len(rows) == 88)
    expected_raw = {run / 'responses' / (rid + '.json') for rid in ids}
    need('exact88_frozen_terminal_paths', len(frozen_raw['files']) == 88 and {Path(x['path']).resolve() for x in frozen_raw['files']} == expected_raw)
    need('exact88_response_files', set((run / 'responses').glob('*.json')) == expected_raw)
    raws = {}
    for row in rows:
        rid = row['request_id']
        raw = read(run / 'responses' / (rid + '.json'))
        need('terminal_before_gold:' + rid, raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED', 'MISSING') and moment(raw['response_committed_at_utc']) <= moment(frozen_raw['created_at_utc']))
        need('rawids:' + rid, all(raw[k] == row[k] for k in ('request_id', 'candidate_id', 'root_id')))
        raws[rid] = raw
    first_gold_read = now()
    need('audit_gold_after_rawfreeze', moment(first_gold_read) >= moment(frozen_raw['created_at_utc']))
    private = {q['candidate_id']: q for q in read(m['private_answer_binding']['path'])['questions']}
    scores = read(run / 'SCORES.json')
    started = read(run / 'SCORING_STARTED.json')
    need('scorer_gold_after_rawfreeze', moment(started['gold_first_read_at_utc']) >= moment(frozen_raw['created_at_utc']))
    equal('scorer_goldtimestamp', scores['gold_first_read_at_utc'], started['gold_first_read_at_utc'])
    need('scorer_timestamps', moment(scores['scored_at_utc']) >= moment(scores['gold_read_completed_at_utc']) >= moment(scores['gold_first_read_at_utc']))
    bindings([started['raw_response_freeze_binding'], started['private_answer_binding']], 'scorerstart')
    for key in ('manifest_binding', 'raw_response_freeze_binding', 'preflight_freeze_binding', 'frozen_evidence_binding', 'public_questions_binding', 'private_answer_binding', 'display_mappings_binding', 'scoring_code_binding', 'protocol_binding', 'baseline_inventory_binding'):
        bindings([scores[key]], 'score:' + key)
    public = {q['candidate_id']: q for q in read(run / 'PUBLIC_QUESTIONS.json')['questions']}
    maps = read(run / 'private/DISPLAY_MAPPINGS.json')
    tokenids = read(run / 'PROMPT_TOKEN_IDS.json')
    frozen = {x['frozen_record_id']: x for x in read(run / 'FROZEN_EVIDENCE.json')['records']}
    sources, references = {}, {}
    for qid, question in public.items():
        equal('private_public:' + qid, {k: private[qid][k] for k in question}, question)
        path = run / 'sources' / (question['root_id'] + '.json')
        need('sourcebound:' + qid, sha(path) == question['source_sha256'])
        sources[question['root_id']] = read(path)
        expected_canonical = {k: 'content_' + hashlib.sha256(v.strip().encode()).hexdigest()[:16] for k, v in question['items'].items()}
        equal('contentids:' + qid, maps['public_to_canonical'][qid], expected_canonical)
    for fid, item in frozen.items():
        positions = independent_positions(item['evidence'], sources[item['root_id']])
        need('frozen_positions_known:' + fid, positions is not None)
        equal('frozen_positions:' + fid, positions, item['positions'])
        references[fid] = [p['label'] for p in sorted(positions, key=lambda p: (p['source_turn_index'], p['quote_start_offset']))]
        equal('frozen_target:' + fid, item['citation_sorted_display'], references[fid])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    parsed_scores = {r['request_id']: r for r in scores['requests']}
    need('scorer88', len(scores['requests']) == len(parsed_scores) == 88 and set(parsed_scores) == ids)
    records = []
    usage = {arm: Counter() for arm in ARMS}
    for row in rows:
        rid, arm, qid, fid = (row[k] for k in ('request_id', 'probe_arm', 'candidate_id', 'frozen_record_id'))
        raw = raws[rid]
        prompt = (run / 'prompts' / (rid + '.txt')).read_text()
        need('promptsha:' + rid, hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'])
        equal('prompt_encoding:' + rid, tokenizer.encode(prompt, add_special_tokens=False), tokenids[rid])
        if raw['status'] == 'RETURNED':
            need('response_model_seed_prompt:' + rid, raw['model_path'] == m['model'] and all(raw[k] == row[k] for k in ('seed', 'variant_id', 'prompt_sha256')))
            equal('raw_promptIDs:' + rid, raw['prompt_token_ids'], tokenids[rid])
            equal('raw_decode:' + rid, tokenizer.decode(raw['completion_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False), raw['raw_output'])
            md = raw['metadata']
            need('raw_sampling:' + rid, md['temperature'] == .8 and md['top_p'] == .9 and md['max_tokens'] == 32 and md['enable_thinking'] is False and raw['gold_read_by_solver'] is False)
            need('rawbudget:' + rid, len(raw['completion_token_ids']) <= 32)
            need('rawtimeline:' + rid, moment(raw['request_submitted_at_utc']) <= moment(raw['response_committed_at_utc']) <= moment(frozen_raw['created_at_utc']))
            for key, id_key in (('prompt_token_count', 'prompt_token_ids'), ('completion_token_count', 'completion_token_ids')):
                equal('tokens:' + rid + ':' + key, md[key], len(raw[id_key]))
                usage[arm][key] += len(raw[id_key])
        prediction, valid = parse_answer(raw)
        display_map = maps['requests'][rid]['display_to_content']
        equal('display_map_text:' + rid, display_map, {k: 'content_' + hashlib.sha256(v.strip().encode()).hexdigest()[:16] for k, v in row['display']['items'].items()})
        canonical = maps['public_to_canonical'][qid]
        gold_content = [canonical[label] for label in private[qid]['gold']]
        reverse = {value: key for key, value in display_map.items()}
        gold = [reverse[value] for value in gold_content]
        citation = references[fid]
        record = {k: row[k] for k in ('request_id', 'candidate_id', 'root_id', 'probe_arm', 'pair_id', 'frozen_record_id', 'variant_id', 'repeat_id', 'seed')}
        record.update(status=raw['status'], valid=valid, prediction_display=prediction,
                      prediction_content=[display_map[x] for x in prediction] if prediction else None,
                      citation_reference_display=citation, citation_reference_content=[display_map[x] for x in citation],
                      gold_reference_display=gold, gold_reference_content=gold_content,
                      citation_order_matches_gold=citation == gold,
                      citation_exact=bool(valid and prediction == citation), citation_correct_pairs=pair_correct(prediction, citation, valid),
                      gold_exact=bool(valid and prediction == gold), gold_correct_pairs=pair_correct(prediction, gold, valid),
                      total_pairs=15, citation_positions=frozen[fid]['positions'],
                      semantic_support='NOT_REVIEWED_BY_SCORER', finish_reason=raw.get('metadata', {}).get('finish_reason'),
                      response_committed_at_utc=raw['response_committed_at_utc'])
        record['citation_pair_accuracy'] = record['citation_correct_pairs'] / 15
        record['gold_pair_accuracy'] = record['gold_correct_pairs'] / 15
        compare('score_request:' + rid, parsed_scores[rid], record)
        equal('strict_parse_error:' + rid, parsed_scores[rid]['parse_error'] is None, valid)
        for key, path in (('response_binding', run / 'responses' / (rid + '.json')), ('source_binding', run / 'sources' / (row['root_id'] + '.json')), ('frozen_evidence_binding', run / 'FROZEN_EVIDENCE.json'), ('display_mappings_binding', run / 'private/DISPLAY_MAPPINGS.json')):
            equal('score_record_binding:' + rid + ':' + key, parsed_scores[rid][key], bind(path))
        records.append(record)

    def aggregate(subset):
        n = len(subset)
        value = {'calls': n, 'questions': len({r['candidate_id'] for r in subset}), 'roots': len({r['root_id'] for r in subset}),
                 'frozen_records': len({r['frozen_record_id'] for r in subset}), 'valid_calls': sum(r['valid'] for r in subset), 'total_pairs': 15 * n}
        value['invalid_calls'] = n - value['valid_calls']
        for target in ('citation', 'gold'):
            exact = sum(r[target + '_exact'] for r in subset)
            pairs = sum(r[target + '_correct_pairs'] for r in subset)
            value.update({target + '_exact_calls': exact, target + '_exact_rate': exact / n,
                          target + '_correct_pairs': pairs, target + '_pair_accuracy': pairs / (15 * n)})
        return value

    by_arm = {arm: aggregate([r for r in records if r['probe_arm'] == arm]) for arm in ARMS}
    compare('by_arm', scores['by_arm'], by_arm)
    by_root, by_frozen = [], []
    for key, output in (('root_id', by_root), ('frozen_record_id', by_frozen)):
        for identifier in sorted({r[key] for r in records}):
            subset = [r for r in records if r[key] == identifier]
            item = {key: identifier, 'root_id': subset[0]['root_id'], **{arm: aggregate([r for r in subset if r['probe_arm'] == arm]) for arm in ARMS}}
            for target in ('citation', 'gold'):
                item[target + '_exact_rate_difference'] = item[ARMS[1]][target + '_exact_rate'] - item[ARMS[0]][target + '_exact_rate']
                item[target + '_exact_net_change'] = item[ARMS[1]][target + '_exact_calls'] - item[ARMS[0]][target + '_exact_calls']
                item[target + '_correct_pair_net_change'] = item[ARMS[1]][target + '_correct_pairs'] - item[ARMS[0]][target + '_correct_pairs']
            if key == 'frozen_record_id':
                item.update({k: subset[0][k] for k in ('candidate_id', 'variant_id', 'citation_order_matches_gold')})
            output.append(item)
    equal('by_root', scores['by_root'], by_root)
    equal('by_frozen_record', scores['by_frozen_record'], by_frozen)
    root_equal = {}
    for target in ('citation', 'gold'):
        root_equal[target] = {arm: sum(r[arm][target + '_exact_rate'] for r in by_root) / 5 for arm in ARMS}
        root_equal[target]['difference'] = sum(r[target + '_exact_rate_difference'] for r in by_root) / 5
    compare('root_equal_exact', scores['root_equal_exact'], root_equal)
    equal('primary_root_equal_citation_difference', scores['primary_root_equal_citation_exact_difference'], root_equal['citation']['difference'])
    equal('secondary_root_equal_gold_difference', scores['secondary_root_equal_gold_exact_difference'], root_equal['gold']['difference'])
    paired = defaultdict(dict)
    for record in records:
        need('no_duplicate_pair_arm:' + record['request_id'], record['probe_arm'] not in paired[record['pair_id']])
        paired[record['pair_id']][record['probe_arm']] = record
    need('44_complete_pairs', len(paired) == 44 and all(set(v) == set(ARMS) for v in paired.values()))
    pair_rows, transitions = [], {target: Counter() for target in ('citation', 'gold')}
    for pid, arms in sorted(paired.items()):
        a, b = [arms[arm] for arm in ARMS]
        for key in ('candidate_id', 'root_id', 'frozen_record_id', 'variant_id', 'repeat_id', 'seed', 'citation_reference_display', 'gold_reference_display'):
            equal('paired_metadata:' + pid + ':' + key, a[key], b[key])
        item = {k: a[k] for k in ('pair_id', 'candidate_id', 'root_id', 'frozen_record_id', 'variant_id', 'repeat_id', 'seed')}
        item.update({arm + '_request_id': arms[arm]['request_id'] for arm in ARMS})
        item.update({arm + '_valid': arms[arm]['valid'] for arm in ARMS})
        for target in ('citation', 'gold'):
            transition = ('correct' if a[target + '_exact'] else 'wrong') + '_to_' + ('correct' if b[target + '_exact'] else 'wrong')
            transitions[target][transition] += 1
            item.update({target + '_transition': transition, target + '_exact_net_change': int(b[target + '_exact']) - int(a[target + '_exact']),
                         target + '_correct_pair_net_change': b[target + '_correct_pairs'] - a[target + '_correct_pairs']})
        pair_rows.append(item)
    compare('matched_pairs', scores['matched_pairs'], {'pairs': 44, 'records': pair_rows, **{target + '_transitions': {k: transitions[target][k] for k in ('correct_to_correct', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong')} for target in ('citation', 'gold')}})
    statuses = Counter(r['status'] for r in raws.values())
    submitted, batch_wall = [], 0.
    for path in sorted((run / 'batches').glob('*_SUBMITTED.json')):
        batch = read(path); submitted.extend(batch['request_ids'])
        need('no_automatic_retry:' + path.name, batch['automatic_retry'] is False)
        donepath = Path(str(path).replace('_SUBMITTED.json', '_COMPLETE.json'))
        if donepath.exists():
            done = read(donepath)
            equal('batchids:' + path.name, done['request_ids'], batch['request_ids'])
            need('batch_time:' + path.name, moment(done['completed_at_utc']) >= moment(batch['submitted_at_utc']))
            batch_wall += done['wall_seconds']
    need('no_repeated_submission', len(submitted) == len(set(submitted)))
    execution_path = run / ('COMPLETE.json' if (run / 'COMPLETE.json').exists() else 'FAILED.json')
    execution = read(execution_path)
    prompt_tokens = sum(u['prompt_token_count'] for u in usage.values())
    completion_tokens = sum(u['completion_token_count'] for u in usage.values())
    cost = {'planned_calls': 88, 'returned_calls': statuses['RETURNED'], 'submitted_attempts': len(submitted),
            'not_submitted_calls': statuses['NOT_SUBMITTED'], 'missing_terminal_calls': statuses['MISSING'],
            'requests_with_unknown_usage': statuses['BACKEND_ERROR'], 'known_prompt_tokens': prompt_tokens, 'known_completion_tokens': completion_tokens,
            'known_total_tokens': prompt_tokens + completion_tokens, 'live_wall_seconds': execution['wall_seconds'],
            'model_load_wall_seconds': execution['model_load_wall_seconds'], 'execution_binding': bind(execution_path)}
    compare('cost', scores['cost'], cost)
    for arm in ARMS:
        compare('armcost:' + arm, scores['cost']['by_arm'][arm], {'planned_calls': 44, 'returned_calls': sum(raws[r['request_id']]['status'] == 'RETURNED' for r in rows if r['probe_arm'] == arm), 'known_prompt_tokens': usage[arm]['prompt_token_count'], 'known_completion_tokens': usage[arm]['completion_token_count'], 'requests_with_unknown_usage': 0})
    need('wall_covers_batches_and_model_load', execution['wall_seconds'] + 1e-6 >= batch_wall + (execution['model_load_wall_seconds'] or 0))
    equal('total_valid', scores['valid_calls'], sum(r['valid'] for r in records))
    compare('denominators', scores, {'denominator_calls': 88, 'denominator_questions': 5, 'denominator_roots': 5, 'denominator_frozen_records': 11, 'denominator_matched_pairs': 44})
    need('no_cause_or_training_claims', scores['cause_auto_published'] is False and scores['weakness_memory_upgraded'] is False and scores['training_started'] is False and scores['new_training_questions_generated'] == 0 and scores['use_as_final_eval'] is False and scores['semantic_support_reviewed_by_scorer'] is False)
    sealed_snapshots()
    return {'gold_first_read_by_audit_at_utc': first_gold_read, 'raw_status_counts': dict(statuses), 'by_arm': by_arm,
            'by_root': by_root, 'by_frozen_record': by_frozen, 'root_equal_exact': root_equal, 'cost': cost,
            'all_scored_results_independently_recomputed': True, 'semantic_support_certified_by_this_audit': False,
            'results_bindings': [bind(run / 'RAW_RESPONSE_FREEZE.json'), bind(run / 'SCORES.json'), bind(execution_path)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'results'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.out.resolve()
    need('new_directory_only', run.is_relative_to(HERE) and output.is_relative_to(HERE))
    need('exclusive_output', not output.exists())
    result = preflight(run) if args.mode == 'preflight' else results(run)
    payload = {'status': 'PASS' if not ISSUES else 'FAIL', 'mode': args.mode,
               'auditor': '/root/frozen_audit', 'created_at_utc': now(),
               'review_scope': 'Independent input reconstruction and raw rescoring; shared filesystem/model. Literal positions do not certify semantic support or an internal causal mechanism.',
               'checks_count': len(CHECKS), 'issues_count': len(ISSUES), 'issues': ISSUES,
               'checks': CHECKS, **result,
               'audited_files': [bind(run / 'PREFLIGHT_FREEZE.json'), bind(HERE / 'PROTOCOL.json'), bind(__file__)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: payload[key] for key in ('status', 'mode', 'checks_count', 'issues_count')}, ensure_ascii=False))
    if ISSUES:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
