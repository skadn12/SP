#!/usr/bin/env python3
"""One-time execution subset with the full original candidate pool retained.

No GPU/model import occurs in self-check or score. Partial orders retain the
existing directed-pair F1 metric, separately from multi/full-order J.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import inspect
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WEEK = EXPERIMENT.parent
HISTORICAL = WEEK / 'feedback_guided_relation_qa_20261103_run01/solver/run_eval.py'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
MCLEAN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model')
INVENTORY = WEEK / 'source_round_inventory_20261104_run01/INVENTORY.json'
PLAN_CONFIG = WEEK / 'training_set_rebuild_plan_20261104_v1/PLAN_CONFIG.json'
ORIGINAL_ADAPTER = WEEK / 'root_capacity_feedback_20261104_run01/solver_adapter/adapter.py'
IDENTITY_AUDIT = HERE / 'MODEL_IDENTITY_AUDIT.json'
EXECUTION_FREEZE = EXPERIMENT / 'EXECUTION_FREEZE.json'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


h = load('capacity_historical_solver', HISTORICAL)
old_check, old_variants = h.check_input_item, h.make_variants
old_validate, old_parse, old_summary = h.reference.validate, h.baseline.parse_and_score, h.summarize
old_schema, old_prompt = h.reference.schema_for, h.reference.prompt_for
partial_legacy = h.baseline.order_prompt
read, sha = h.read, h.sha


def bind(path):
    # The new exact model bindings include multi-GB shards: stream hashes.
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


h.bind = bind
h.MODEL, h.TOKENIZER = MCLEAN, TOKENIZER
h.DISPLAY_SEED, h.SAMPLE_SEED = 2026110401, 2026110402
check_source = inspect.getsource(old_check)
check_assertion = 'assert isinstance(options, list) and len(options) >= 3'
assert check_source.count(check_assertion) == 1, 'HISTORICAL_ITEM_CHECK_CHANGED'
check_source = check_source.replace(check_assertion,
    "assert isinstance(options, list) and len(options) >= (2 if original.get('_capacity_partial_order') else 3)")
check_source = check_source.replace('def check_input_item(', 'def capacity_check_input_item(', 1)
exec(compile(check_source, str(HISTORICAL) + ':capacity_check', 'exec'), h.__dict__)
capacity_check = h.capacity_check_input_item


def check_item(original, sources):
    item = copy.deepcopy(original)
    item.setdefault('arm', 'C')
    assert item['arm'] in ('C', 'F', 'G'), 'INVALID_ARM'
    qtype, arm = item['question_type'], item['arm']
    assert qtype in ('multi_select', 'ordering', 'partial_order'), 'UNSUPPORTED_ANSWER_FORM'
    proxy = copy.deepcopy(item)
    proxy['arm'] = 'F'  # Only the old experiment's F/G gate is generalized.
    if qtype == 'partial_order':
        pairs, mapping = item['gold_option_ids'], item['label_to_option_id']
        assert isinstance(pairs, list) and pairs, 'NONEMPTY_PARTIAL_GOLD_REQUIRED'
        assert all(isinstance(p, list) and len(p) == 2 and p[0] != p[1]
                   and set(p) <= set(mapping.values()) for p in pairs), 'INVALID_PARTIAL_GOLD_PAIR'
        assert len({tuple(p) for p in pairs}) == len(pairs), 'DUPLICATE_PARTIAL_GOLD_PAIR'
        assert [[mapping[a], mapping[b]] for a, b in item['gold_labels']] == pairs, 'PARTIAL_GOLD_MAPPING_MISMATCH'
        # Existing typed-answer normalization rejects cyclic partial Gold and
        # treats its transitive closure as the canonical relation set.
        vertices = set(mapping.values())
        closure = {tuple(p) for p in pairs}
        while True:
            expanded = closure | {(a, d) for a, b in closure for c, d in closure if b == c}
            if expanded == closure:
                break
            closure = expanded
        assert all(a != b for a, b in closure), 'CYCLIC_PARTIAL_GOLD'
        assert closure == {tuple(p) for p in pairs}, 'PARTIAL_GOLD_MUST_BE_TRANSITIVE_CLOSURE'
        assert vertices, 'EMPTY_PARTIAL_ITEMS'
        proxy.update(question_type='ordering', gold_labels=list(mapping), gold_option_ids=list(mapping.values()))
        if len(item['options']) == 2:
            proxy['_capacity_partial_order'] = True
    checked, turns = capacity_check(proxy, sources)
    checked.pop('_capacity_partial_order', None)
    checked.update(question_type=qtype, arm=arm, gold_labels=item['gold_labels'], gold_option_ids=item['gold_option_ids'],
                   permutations=min(6, math.factorial(len(item['options']))), repeats_per_permutation=3)
    return checked, turns


def variants(item):
    if item['question_type'] != 'partial_order':
        return old_variants(item)
    proxy = {**item, 'question_type': 'ordering', 'gold_option_ids': list(item['label_to_option_id'].values())}
    result = old_variants(proxy)
    for variant in result:
        reverse = {oid: label for label, oid in variant['display_to_option_id'].items()}
        variant['gold_labels'] = [[reverse[a], reverse[b]] for a, b in item['gold_option_ids']]
    return result


def schema_for(labels, refs, qtype):
    schema = old_schema(labels, refs)
    if qtype == 'partial_order':
        atom = '(?:' + '|'.join(re.escape(x) for x in labels) + ')'
        pair = atom + '<' + atom
        schema['properties']['final_answer']['pattern'] = '^<box>(?:' + pair + '(?:;' + pair + ')*)?</box>$'
    return schema


def prompt_for(tokenizer, item, request, turns):
    if item['question_type'] != 'partial_order':
        return old_prompt(tokenizer, item, request, turns)
    labels = list(request['display_to_option_id'])
    example = '<box>' + ' '.join(labels[:min(3, len(labels))]) + '</box>'
    instruction = (' For partial_order, A<B means A occurs before B. Return the complete supported directed-pair set, '
        'including transitive pairs; use only displayed I-labels. Separate pairs with semicolons, '
        'and use <box></box> for no supported pair. Cite evidence for each displayed item. '
        'Do not invent a total order.')
    # Rebuild from the public chat messages rather than append after the
    # assistant-generation prefix returned by the tokenizer.
    class Capture:
        def apply_chat_template(self, messages, **kwargs):
            return messages, kwargs
    messages, kwargs = old_prompt(Capture(), item, request, turns)
    query_pairs = item.get('query_pairs', [])
    if query_pairs:
        reverse = {oid: label for label, oid in request['display_to_option_id'].items()}
        public_pairs = [[reverse[a], reverse[b]] for a, b in query_pairs]
        messages[-1]['content'] += '\nquery_pairs (unordered action pairs defining the requested scope): ' + json.dumps(public_pairs)
    messages[-1]['content'] = messages[-1]['content'].replace(example, '<box>' + labels[0] + '<' + labels[1] + '</box>')
    messages[-1]['content'] = messages[-1]['content'].replace('every label exactly\nonce for ordering',
        'supported before/after pairs for partial_order, separated by semicolons').replace(
        'every label exactly once for\nordering', 'supported before/after pairs for partial_order, separated by semicolons') + instruction
    return tokenizer.apply_chat_template(messages, **kwargs)


def partial_invalid(reason):
    return {'valid': False, 'parse_reason': reason, 'parsed': None, 'exact_correct': False,
            'partial_score': 0.0, 'score_type': 'directed_pair_f1', 'precision': 0.0, 'recall': 0.0,
            'predicted_option_ids': None, 'invalid_policy': 'zero in fixed denominators; no repair or retry'}


def parse_and_score(raw, request):
    if request['question_type'] != 'partial_order':
        return old_parse(raw, request)
    try:
        assert raw.get('status') == 'RETURNED' and raw.get('finish_reason') in ('stop', 'eos'), 'BACKEND_OR_TRUNCATION'
        labels = list(request['display_to_option_id'])
        atom = '(?:' + '|'.join(re.escape(x) for x in labels) + ')'
        pair = atom + '<' + atom
        text = raw['raw_prediction']
        assert re.fullmatch('(?:' + pair + '(?:;' + pair + ')*)?', text), 'EXACT_PAIR_SYNTAX_REQUIRED'
        pairs = [p.split('<') for p in text.split(';')] if text else []
        assert all(a != b for a, b in pairs), 'SELF_PAIR'
        assert len({tuple(p) for p in pairs}) == len(pairs), 'DUPLICATE_PAIR'
        result = partial_legacy.score({'question_type': 'partial_order', 'options': request['options'],
                                       'gold_labels': request['gold_labels']}, text)
        assert result['parsed'] == pairs, 'LEGACY_PAIR_PARSER_DISAGREEMENT'
        return {**result, 'valid': True, 'parse_reason': None,
                'predicted_option_ids': [[request['display_to_option_id'][a], request['display_to_option_id'][b]] for a, b in pairs],
                'scorer_path': str(h.baseline.ORDER_PROMPT_PATH)}
    except Exception as exc:
        return partial_invalid(str(exc))


def validate(raw, request):
    if request['question_type'] != 'partial_order':
        return old_validate(raw, request)
    result = {'protocol_valid': False, 'protocol_errors': [], 'parsed_output': None,
              'final_answer': None, 'final_answer_valid': False}
    try:
        assert raw.get('status') == 'RETURNED' and raw.get('finish_reason') in ('stop', 'eos'), 'BACKEND_OR_TRUNCATION'
        def unique(pairs):
            value = {}
            for key, entry in pairs:
                assert key not in value, 'DUPLICATE_JSON_KEY'
                value[key] = entry
            return value
        obj = json.loads(raw['raw_prediction'], object_pairs_hook=unique)
        result['parsed_output'] = obj
        final = obj['final_answer']
        result['final_answer'] = final
        match = re.fullmatch(r'<box>(.*)</box>', final)
        assert match, 'FINAL_BOX_SYNTAX'
        answer = parse_and_score({**raw, 'raw_prediction': match.group(1)}, request)
        result.update(final_answer_valid=answer['valid'], answer_only_score=answer)
        assert answer['valid'], answer['parse_reason']
        from jsonschema import validate as json_validate
        json_validate(obj, read(h.HERE / request['schema_path']))
        assert list(obj) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
        assert obj['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
        for item in obj['item_evidence'].values():
            assert item['reason'].strip(), 'EMPTY_ITEM_REASON'
            assert len(item['turn_refs']) == len(set(item['turn_refs'])), 'DUPLICATE_REFERENCE'
            if request['context_mode'] == 'full_context':
                assert item['turn_refs'], 'ORDER_ITEM_WITHOUT_CITATION'
            else:
                assert item['turn_refs'] == [], 'NO_CONTEXT_REFERENCE_FORBIDDEN'
        result['protocol_valid'] = True
    except Exception as exc:
        result['protocol_errors'].append(str(exc))
    return result


def summarize(rows):
    if not rows or all(r['question_type'] != 'partial_order' for r in rows):
        return old_summary(rows)
    if any(r['question_type'] != 'partial_order' for r in rows):
        return {'planned_calls': len(rows), 'exact_correct_calls': sum(r['exact_correct'] for r in rows),
                'exact_accuracy': sum(r['exact_correct'] for r in rows) / len(rows),
                'metrics_by_type': {q: summarize([r for r in rows if r['question_type'] == q])
                                    for q in sorted({r['question_type'] for r in rows})}}
    return {'planned_calls': len(rows), 'valid_calls': sum(r['valid'] for r in rows),
            'invalid_calls': sum(not r['valid'] for r in rows),
            'exact_correct_calls': sum(r['exact_correct'] for r in rows),
            'exact_accuracy': sum(r['exact_correct'] for r in rows) / len(rows),
            'mean_directed_pair_f1': sum(r['partial_score'] for r in rows) / len(rows),
            'score_type': 'directed_pair_f1', 'scorer_path': str(h.baseline.ORDER_PROMPT_PATH)}


def configure(batch_dir, outdir, max_items, seed_namespace):
    assert 1 <= max_items <= 32, 'BATCH_MAXIMUM_IS_32'
    h.EXPERIMENT, h.HERE = Path(batch_dir).resolve(), Path(outdir).resolve()
    assert EXPERIMENT in h.EXPERIMENT.parents, 'BATCH_OUTSIDE_NEW_EXPERIMENT'
    assert h.EXPERIMENT in h.HERE.parents, 'OUTPUT_MUST_BE_WITHIN_BATCH'
    h.reference.HERE = h.HERE
    assert seed_namespace in ('new', 'calibration', 'legacy'), 'INVALID_SEED_NAMESPACE'
    identity = read(IDENTITY_AUDIT)
    assert identity['status'] == 'PASS' and identity['model'] == str(MCLEAN)
    assert identity['tokenizer'] == str(TOKENIZER)
    planned = read(PLAN_CONFIG)['proposed_defaults']
    roots = sorted((row['root_id'] for row in read(INVENTORY)['rows']),
                   key=lambda root: int(root.rsplit('-', 1)[1]))
    assert len(roots) == len(set(roots)) == 68, 'GLOBAL_ROOT_INVENTORY_REQUIRED'
    root_indices = {root: index for index, root in enumerate(roots)}
    pool = read(h.EXPERIMENT / 'ITEMS_PRIVATE.json')
    assert len({item['candidate_id'] for item in pool}) == len(pool)
    slots, seed_metadata = {}, {}
    calibration_ids = planned['feedback_calibration']['item_ids']
    legacy_indices = {}
    if seed_namespace == 'legacy':
        # Global representative indices must come from a pre-call frozen list.
        legacy_ids = read(EXPERIMENT / 'LEGACY_REPRESENTATIVES_FREEZE.json')['candidate_ids_numeric_sorted']
        assert len(legacy_ids) == len(set(legacy_ids))
        legacy_indices = {qid: index for index, qid in enumerate(legacy_ids)}
    for item in pool:
        qid, root, arm = item['candidate_id'], item['root_id'], item['arm']
        assert root in root_indices, ('ROOT_OUTSIDE_TRAIN_INVENTORY', root)
        key = (root, arm)
        slot = slots.get(key, 0)
        slots[key] = slot + 1
        if seed_namespace == 'calibration':
            index = calibration_ids.index(qid)
            offset = 100 * 100000 + index * 1000
        elif seed_namespace == 'legacy':
            index = legacy_indices[qid]
            offset = 1000 * 100000 + index * 1000
        else:
            assert arm in ('F', 'G') and slot < 2, 'NEW_ROOT_ARM_CANDIDATE_BUDGET'
            index = root_indices[root]
            offset = index * 100000 + slot * 1000
        seed_metadata[qid] = {'namespace': seed_namespace, 'root_index': root_indices[root],
            'item_slot': slot if seed_namespace == 'new' else index, 'base_seed': 2026110402 + offset}
    def sampling_seed_for(item, variant_index, repeat):
        return seed_metadata[item['candidate_id']]['base_seed'] + variant_index * 3 + repeat
    h.capacity_seed_for = sampling_seed_for
    selector = read(h.EXPERIMENT / 'EXECUTION_SELECTOR.json')
    execution_ids = set(selector['previously_uncalled_candidate_ids'])
    assert execution_ids and execution_ids <= {i['candidate_id'] for i in pool}
    assert not execution_ids & set(selector['already_completed_candidate_ids'])
    assert execution_ids | set(selector['already_completed_candidate_ids']) == {i['candidate_id'] for i in pool}
    assert len(pool) == selector['original_pool_count']
    h.capacity_execution_ids = execution_ids
    h.MAX_ITEMS, h.MAX_REQUESTS = max_items, max_items * 36
    h.check_input_item, h.make_variants = check_item, variants
    h.reference.prompt_for, h.reference.validate = prompt_for, validate
    h.baseline.parse_and_score, h.summarize = parse_and_score, summarize
    # Reuse historical preparation. These four exact, checked substitutions
    # remove prior experiment quotas and select the partial-order schema/g.
    source = inspect.getsource(h.prepare)
    substitutions = [
        ("selected = [item for item in pool if quality_by_id[item['candidate_id']]['content_pass'] is True]",
         "selected = [item for item in pool if quality_by_id[item['candidate_id']]['content_pass'] is True and item['candidate_id'] in capacity_execution_ids]"),
        ("assert len(roots) == 4 and len(selected) <= MAX_ITEMS, 'EXPERIMENT_SIZE_EXCEEDED'",
         "assert roots and len(selected) <= MAX_ITEMS, 'EXPERIMENT_SIZE_EXCEEDED'"),
        ("assert pair not in seen_pairs, ('DUPLICATE_ROOT_ARM', pair)",
         "assert True  # Capacity batches may contain several questions for one root/arm."),
        ('schema = reference.schema_for(labels, refs)', 'schema = capacity_schema_for(labels, refs, qtype)'),
        ("g = len(item['gold_option_ids']) if qtype == 'multi_select' else n // 2",
         "g = len(item['gold_option_ids']) if qtype in ('multi_select', 'partial_order') else n // 2"),
        ('AutoTokenizer.from_pretrained(str(MODEL),', 'AutoTokenizer.from_pretrained(str(TOKENIZER),'),
        ("seed_for(root_indices[item['root_id']], variant_index, repeat)",
         'capacity_seed_for(item, variant_index, repeat)'),
        ("MODEL / 'config.json', MODEL / 'tokenizer_config.json', MODEL / 'tokenizer.json']",
         "MODEL / 'config.json', TOKENIZER / 'tokenizer_config.json', TOKENIZER / 'tokenizer.json'] + capacity_model_paths")]
    for before, after in substitutions:
        assert source.count(before) == 1, 'HISTORICAL_PREPARATION_CHANGED'
        source = source.replace(before, after)
    h.capacity_schema_for = schema_for
    h.capacity_model_paths = [Path(row['path']) for row in identity['files']]
    exec(compile(source, str(HERE / 'adapter.py') + ':historical_prepare', 'exec'), h.__dict__)
    # Reuse the historical inference loop with a distinct tokenizer path.
    run_source = inspect.getsource(h.run)
    assert run_source.count('tokenizer=str(MODEL)') == 1
    run_source = run_source.replace('tokenizer=str(MODEL)', 'tokenizer=str(TOKENIZER)')
    exec(compile(run_source, str(HERE / 'adapter.py') + ':historical_run', 'exec'), h.__dict__)
    # Add the adapter and global execution freeze to historical byte bindings.
    h.input_paths = lambda: sorted(set([p for p in h.EXPERIMENT.rglob('*') if p.is_file() and h.HERE not in p.parents]
        + [Path(__file__), ORIGINAL_ADAPTER, EXECUTION_FREEZE, PLAN_CONFIG, INVENTORY, IDENTITY_AUDIT]))
    historical_write = h.write
    def write(name, value):
        if name == 'MODEL_CONFIG.json':
            value.update(model=str(MCLEAN), tokenizer=str(TOKENIZER), base_model_only=False,
                solver='local Qwen3-4B + M-clean (exact verified merged weights)',
                init_adapter=identity['init_adapter'], adapter_applied_at_runtime=False,
                model_load_mode=identity['load_mode'], model_identity_audit=str(IDENTITY_AUDIT),
                global_root_indices=root_indices, seed_namespace=seed_namespace, seed_metadata=seed_metadata,
                sampling_root_indices=root_indices,
                local_only=True, batch_maximum_items=max_items, maximum_requests=max_items * 36,
                sampling_by_type={q: {'permutations': 'min(6,k!) unique', 'repeats': 3}
                                  for q in ('multi_select', 'ordering', 'partial_order')},
                partial_order_scoring='existing directed_pair_f1, separate from J',
                partial_order_scorer=str(h.baseline.ORDER_PROMPT_PATH), adapter=str(Path(__file__)),
                inherited_local_executor=str(HISTORICAL),
                sampling_seed_rule=planned['solver_protocol']['sampling_seed_rule'],
                calibration_seed_namespace=planned['solver_protocol']['calibration_seed_namespace'],
                legacy_seed_namespace=planned['solver_protocol']['legacy_seed_namespace'])
        historical_write(name, value)
    h.write = write


def self_check():
    before = {str(p): bind(p) for p in (Path(__file__), HISTORICAL, h.REFERENCE, h.baseline.SCORER_PATH,
                                       h.baseline.ORDER_PROMPT_PATH, EXECUTION_FREEZE)}
    old_h_checks = h.self_checks()
    reports = []
    with tempfile.TemporaryDirectory(prefix='schema_fixture_', dir=HERE) as temp:
        previous_here, previous_ref_here = h.HERE, h.reference.HERE
        h.HERE = h.reference.HERE = Path(temp)
        for qtype, labels, gold in [('multi_select', list('ABCD'), ['stable_A', 'stable_C']),
                                ('ordering', ['I1', 'I2', 'I3', 'I4'], ['stable_I1', 'stable_I2', 'stable_I3', 'stable_I4']),
                                ('partial_order', ['I1', 'I2', 'I3', 'I4'], [['stable_I1', 'stable_I2'], ['stable_I1', 'stable_I3']])]:
            fixture = {'candidate_id': 'synthetic_' + qtype, 'question_type': qtype,
                'options': [{'id': x, 'text': 'Synthetic statement ' + x} for x in labels],
                'label_to_option_id': {x: 'stable_' + x for x in labels},
                'gold_option_ids': gold, 'permutations': 6}
            displayed = variants(fixture)
            changed = copy.deepcopy(fixture)
            changed['gold_option_ids'] = gold[::-1] if qtype != 'multi_select' else ['stable_B', 'stable_D']
            assert [v['display_to_option_id'] for v in displayed] == [v['display_to_option_id'] for v in variants(changed)], 'GOLD_LEAKAGE'
            assert len(displayed) == len({tuple(v['display_to_option_id'].values()) for v in displayed}) == 6
            small = {**fixture, 'candidate_id': fixture['candidate_id'] + '_two_items',
                     'options': fixture['options'][:2],
                     'label_to_option_id': {key: fixture['label_to_option_id'][key] for key in labels[:2]},
                     'permutations': 2,
                     'gold_option_ids': (['stable_' + labels[0]] if qtype == 'multi_select' else
                         [['stable_' + labels[0], 'stable_' + labels[1]]] if qtype == 'partial_order' else
                         ['stable_' + labels[0], 'stable_' + labels[1]])}
            small_variants = variants(small)
            assert len(small_variants) == 2 and len({tuple(v['display_to_option_id'].values()) for v in small_variants}) == 2
            for variant in displayed:
                answer = ';'.join('<'.join(p) for p in variant['gold_labels']) if qtype == 'partial_order' else (',' if qtype == 'multi_select' else '>').join(variant['gold_labels'])
                req = {**variant, 'question_type': qtype, 'gold_option_ids': gold,
                       'n': 4 if qtype == 'multi_select' else 12,
                       'g': 2 if qtype == 'multi_select' else (6 if qtype == 'ordering' else len(gold)),
                       'schema_path': f'{qtype}.json', 'context_mode': 'full_context'}
                schema_path = h.HERE / req['schema_path']
                if not schema_path.exists():
                    schema_path.write_text(json.dumps(schema_for(labels, ['t1'], qtype)))
                evidence = {label: {'judgment': 'supported', 'turn_refs': ['t1'], 'reason': 'Synthetic cited reason.'}
                            for label in labels}
                displayed_answer = ' '.join(variant['gold_labels']) if qtype != 'partial_order' else answer
                boxed = '<box>' + displayed_answer + '</box>'
                raw_json = json.dumps({'item_evidence': evidence, 'answer_reason': 'Synthetic answer reason.',
                                       'final_answer': boxed})
                raw = {'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': raw_json}
                checked = validate(raw, req)
                assert checked['protocol_valid'], (qtype, boxed, checked['protocol_errors'])
                if qtype == 'partial_order':
                    class FakeTokenizer:
                        def apply_chat_template(self, messages, **kwargs):
                            return json.dumps({'messages': messages, 'kwargs': kwargs})
                    item = {'question_type': qtype, 'question': 'Which displayed item precedes which?', 'options': fixture['options']}
                    prompt = prompt_for(FakeTokenizer(), item, variant,
                        [{'turn_ref': 't1', 'speaker': 'A', 'text': 'First happens, then second.'}])
                    assert '<box>I1<I2</box>' in prompt and 'complete supported directed-pair set' in prompt
                    assert 'Do not invent a total order.' in prompt
                    assert 'every label exactly once for\nordering' not in prompt
                    assert 'every label exactly\nonce for ordering' not in prompt
                malformed = json.loads(raw_json)
                malformed['item_evidence'][labels[0]]['reason'] = ''
                assert not validate({**raw, 'raw_prediction': json.dumps(malformed)}, req)['protocol_valid']
                scored = parse_and_score({**raw, 'raw_prediction': answer}, req)
                assert scored['valid'] and scored['exact_correct']
                if qtype != 'partial_order':
                    wrong = [x for x in labels if x not in variant['gold_labels']] if qtype == 'multi_select' else variant['gold_labels'][::-1]
                    negative = parse_and_score({'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': (',' if qtype == 'multi_select' else '>').join(wrong)}, req)
                    assert negative['J_fraction'] == '-1', negative
                else:
                    a, b = variant['gold_labels'][0]
                    incomplete = parse_and_score({'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': a + '<' + b}, req)
                    assert incomplete['partial_score'] == 0.666667 and not incomplete['exact_correct']
                    assert 'J' not in incomplete and 'J_fraction' not in incomplete
                    bad = parse_and_score({'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': 'Answer: ' + answer}, req)
                    assert not bad['valid']
            reports.append({'type': qtype, 'unique_permutations': 6, 'stable_id_roundtrips': 6,
                            'two_item_permutations': len(small_variants),
                            'reason_schema_and_box_validation': 'PASS',
                            'partial_prompt_format': 'PASS' if qtype == 'partial_order' else None,
                            'correct_and_wrong_or_incomplete_scores': 'PASS'})
        h.HERE, h.reference.HERE = previous_here, previous_ref_here
    with tempfile.TemporaryDirectory(prefix='fixture_', dir=HERE) as temp:
        fixture_path = Path(temp) / 'frozen_fixture.json'
        fixture_path.write_text('{"synthetic":true}\n')
        frozen = bind(fixture_path)
        assert bind(fixture_path) == frozen
        fixture_path.write_text('{"synthetic":false}\n')
        assert bind(fixture_path) != frozen
    assert {str(p): bind(p) for p in map(Path, before)} == before, 'HISTORICAL_FREEZE_CHANGED'
    return {'status': 'PASS', 'historical_self_checks': old_h_checks, 'synthetic_fixtures': reports,
            'freeze_unchanged': True, 'synthetic_freeze_mutation_detected': True,
            'gpu_model_loaded': False, 'new_model_calls': 0, 'training': False,
            'bindings': list(before.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('self-check', 'prepare', 'run', 'score'))
    parser.add_argument('--batch-dir', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--max-items', type=int, default=5)
    parser.add_argument('--seed-namespace', choices=('new', 'calibration', 'legacy'), default='new')
    args = parser.parse_args()
    if args.mode == 'self-check':
        result = self_check()
        path = HERE / 'OFFLINE_FIXTURE_CHECK.json'
        with path.open('w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(json.dumps(result, ensure_ascii=False))
        return
    assert args.batch_dir and args.outdir, 'BATCH_DIR_AND_OUTDIR_REQUIRED'
    configure(args.batch_dir, args.outdir, args.max_items, args.seed_namespace)
    h.HERE.mkdir(parents=True, exist_ok=True)
    {'prepare': h.prepare, 'run': h.run, 'score': h.score}[args.mode]()


if __name__ == '__main__':
    main()
