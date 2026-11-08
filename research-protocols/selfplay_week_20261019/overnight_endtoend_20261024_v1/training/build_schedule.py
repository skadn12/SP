"""Compile an independently admitted matched pool into a fixed zero-truncation schedule."""
from __future__ import annotations
import argparse
import random
from common import *


def extract_records(frozen):
    records = frozen.get('records')
    if isinstance(records, dict):
        return [dict(r, condition=arm) for arm, values in records.items() for r in values]
    if isinstance(records, list): return records
    if isinstance(frozen.get('arms'), dict):
        out = []
        for arm, entry in frozen['arms'].items():
            values = entry if isinstance(entry, list) else entry.get('records')
            if values is None:
                b = entry.get('records_binding') or entry.get('binding')
                verify([b]); values = lines(b['path'])
            out.extend(dict(r, condition=arm) for r in values)
        return out
    raise RuntimeError('POOL_RECORDS_SCHEMA_REQUIRED')


def messages_for(q, variant, source):
    system = ('You are a strict structured dialogue QA solver. Use only the supplied '
              'dialogue when it is present. Return the final answer in exactly one '
              '<box>...</box> pair. Do not provide explanations, reasoning, or any '
              'text outside the box.')
    turns = source.get('turns')
    require(isinstance(turns, list) and turns, 'FULL_SOURCE_TURNS_REQUIRED')
    dialogue = '\n'.join(f"[{row['turn_ref']}] {row['speaker']}: {row['text']}" for row in turns)
    typ = q['question_type']; field = 'options' if typ == 'multi_select' else 'items'
    chunks = ['Dialogue:\n' + dialogue, f"Question ({typ}):\n{q['question']}",
              field.title() + ':\n' + '\n'.join(f'{k}. {v}' for k, v in variant[field].items())]
    chunks.append('Output only selected letters separated by commas, e.g. <box>A,C</box>.' if typ == 'multi_select' else 'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': '\n\n'.join(chunks)}]


def variants_for(q, gold, slot_key, seed, forbidden_orders=None):
    """Same label rebindings on both arms; no correctness-based selection."""
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    keys = list(q[field]); original = list(q[field].values())
    require(len(keys) in (4, 5, 6) and len(set(original)) == len(keys), 'FOUR_TO_SIX_UNIQUE_ITEMS_REQUIRED')
    ids = {k: 'content_' + hashlib.sha256(q[field][k].strip().encode()).hexdigest()[:16] for k in keys}
    require(len(set(ids.values())) == len(keys), 'CONTENT_HASH_COLLISION')
    output = []; used = set()
    forbidden = {tuple(x) for x in (forbidden_orders or [])}
    if q['question_type'] == 'ordering':
        forbidden.update((tuple(gold), tuple(reversed(gold))))
    for v in range(2):
        rng = random.Random(canonical_sha([seed, slot_key, v]))
        # Non-identity rebinding has deterministic paired permutation budget.
        # Draws depend only on seed/slot/variant; not solver answers or scores.
        order = keys[:]
        while tuple(order) in used or order == keys or tuple(order) in forbidden:
            rng.shuffle(order)
        used.add(tuple(order))
        mapping = {label: ids[k] for label, k in zip(keys, order)}
        display = {label: q[field][k] for label, k in zip(keys, order)}
        inv = {value: key for key, value in mapping.items()}
        gold_content = [ids[k] for k in gold]
        display_gold = [inv[x] for x in gold_content]
        if q['question_type'] == 'multi_select': display_gold.sort()
        # Public order must actually be shuffled, including relative to semantic target.
        if q['question_type'] == 'ordering':
            require([mapping[k] for k in keys] != gold_content, 'ORDER_DISPLAY_EQUALS_GOLD')
        require([mapping[k] for k in display_gold] == gold_content if q['question_type'] == 'ordering' else set(mapping[k] for k in display_gold) == set(gold_content), 'GOLD_ROUNDTRIP_FAILED')
        output.append({'variant_id': f'perm_{v:02d}', field: display, 'display_to_content': mapping,
                       'display_to_canonical': dict(zip(keys, order)),
                       'canonical_to_display': {canonical: display for display, canonical in zip(keys, order)},
                       'gold_content_ids': gold_content, 'gold_display_labels': display_gold})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, default=HERE / 'schedule')
    p.add_argument('--max-length', type=int, default=16384)
    args = p.parse_args()
    from transformers import AutoTokenizer
    require(not (args.output / 'SCHEDULE_FREEZE.json').exists(), 'SCHEDULE_ALREADY_FROZEN')
    policy = read(PROTOCOL); train = policy['training']; seed = train['seed']
    require(train['starting_checkpoint'] == str(MODEL), 'B1_STARTING_CHECKPOINT_MISMATCH')
    pool = read(args.pool_freeze)
    require(pool.get('status') == 'FROZEN_BEFORE_TRAINING', 'POOL_NOT_ADMITTED')
    require(pool.get('schema_version') == 'oet1_matched_training_pool_v1', 'POOL_SCHEMA_NOT_INDEPENDENTLY_ADMITTED')
    records = extract_records(pool)
    # Bind actual training text to the independently reviewed immutable question.
    prepared = {}; forbidden_by_slot = collections.defaultdict(list)
    for item in records:
        verify([item['source_binding'], item['question_binding'], item['content_review_binding']])
        bound = read(item['question_binding']['path'])
        require('question' not in item or item['question'] == bound, 'EMBEDDED_QUESTION_DIFFERS_FROM_REVIEWED_BINDING')
        require(bound.get('source_binding') == item['source_binding'], 'QUESTION_SOURCE_BINDING_MISMATCH')
        q = dict(bound)
        typ = q['question_type']; field = 'options' if typ == 'multi_select' else 'items'
        keys = list('ABCDEF') if typ == 'multi_select' else [f'I{i}' for i in range(1, len(q[field]) + 1)]
        require(set(q[field]) == set(keys), 'CANONICAL_LABEL_SET_INVALID')
        q[field] = {key: q[field][key] for key in keys}
        arm = item.get('condition', item.get('arm'))
        key = (arm, item['paired_slot_key'])
        require(key not in prepared, 'DUPLICATE_PAIRED_SLOT')
        prepared[key] = q
        if typ == 'ordering':
            forbidden_by_slot[item['paired_slot_key']].extend([q['gold'], list(reversed(q['gold']))])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    eval_lock = read(HERE / 'evaluation/EVALUATION_FREEZE.json')
    eval_records = lines(eval_lock['final_requests']['path'])
    eval_roots = {r['root_id'] for r in eval_records}; eval_ids = {r['candidate_id'] for r in eval_records}
    result = {}; checks = []; content_keys = {}; arms = {}
    for arm in ('GENERIC', 'TARGETED'):
        selected = [r for r in records if r.get('condition', r.get('arm')) == arm]
        require(selected, 'EMPTY_MATCHED_ARM')
        require(len({r['paired_slot_key'] for r in selected}) == len(selected), 'DUPLICATE_PAIRED_SLOT')
        rows = []; arm_content = {}
        for item in selected:
            require(item.get('split', 'train') == 'train' and item['root_id'] not in eval_roots and item['candidate_id'] not in eval_ids, 'TRAIN_DEV_BOUNDARY_VIOLATION')
            require(item['slot'] in ('MC2', 'MC3', 'MC4', 'MC5', 'ORDER1', 'ORDER2'), 'CONTROL_OR_UNKNOWN_SLOT_NOT_TRAINABLE')
            bindings = [item[k] for k in ('source_binding', 'question_binding', 'content_review_binding') if item.get(k)]
            require(len(bindings) == 3, 'AUDITED_SOURCE_QUESTION_REVIEW_BINDINGS_REQUIRED')
            verify(bindings)
            q = prepared[arm, item['paired_slot_key']]
            typ = q.get('question_type', item.get('question_type'))
            q['question_type'] = typ
            require(typ in ('multi_select', 'ordering'), 'UNKNOWN_TRAIN_TASK')
            gold = q.get('gold') or item.get('gold')
            require(isinstance(gold, list) and len(gold) == len(set(gold)), 'CANONICAL_GOLD_REQUIRED')
            field = 'options' if typ == 'multi_select' else 'items'
            expected_keys = list('ABCDEF') if typ == 'multi_select' else [f'I{i}' for i in range(1, len(q[field]) + 1)]
            require(list(q[field]) == expected_keys and set(gold) <= set(expected_keys), 'CANONICAL_LABEL_SCHEMA_INVALID')
            require(len(gold) == (int(item['slot'][2:]) if typ == 'multi_select' else len(q[field])), 'GOLD_QUOTA_MISMATCH')
            source = read(item['source_binding']['path'])
            variants = variants_for(q, gold, item['paired_slot_key'], seed, forbidden_by_slot[item['paired_slot_key']])
            for v in variants:
                messages = messages_for(q, v, source)
                # Check our message assembly is byte-equivalent to the preserved prompt.
                public_q = {'question_type': typ, 'question': q['question']}
                contract, _ = adapter().native()
                prompt = contract.build_prompt(tokenizer, public_q, {field: v[field]}, contract.dialogue_rows(source), context_mode='full_context')
                require(prompt == native().render_chat(tokenizer, messages, True), 'PROMPT_CONTRACT_DRIFT')
                messages.append({'role': 'assistant', 'content': '<box>' + ','.join(v['gold_display_labels']) + '</box>'})
                _, tokens = tokenize(tokenizer, messages, args.max_length)
                for epoch in (1, 2):
                    schedule_key = item['paired_slot_key'] + '::' + v['variant_id']
                    row = {k: item[k] for k in ('candidate_id', 'root_id', 'slot', 'paired_slot_key', 'source_binding', 'question_binding', 'content_review_binding', 'historical_exposure')}
                    row.update(arm=arm, split='train', question_type=typ, gold_cardinality=len(gold), category=item['slot'], epoch=epoch,
                               variant_id=v['variant_id'], schedule_key=schedule_key, messages=messages,
                               display_to_content=v['display_to_content'], gold_content_ids=v['gold_content_ids'],
                               display_to_canonical=v['display_to_canonical'], canonical_to_display=v['canonical_to_display'],
                               gold_display_labels=v['gold_display_labels'], **tokens)
                    rows.append(row)
            arm_content[item['paired_slot_key']] = {'slot': item['slot'], 'type': typ, 'gold': len(gold), 'root_id': item['root_id']}
        rows.sort(key=lambda r: (r['epoch'], canonical_sha([seed, r['epoch'], r['schedule_key']]), r['schedule_key']))
        for i, row in enumerate(rows): row['ordinal'] = i + 1
        require(len(rows) % 4 == 0 and len(rows) == len(selected) * 4, 'OPTIMIZER_BUDGET_MISMATCH')
        output = args.output / ('TRAIN_' + arm + '.jsonl'); save_lines(output, rows)
        arms[arm] = rows; content_keys[arm] = arm_content
        result[arm] = {'schedule': binding(output), 'questions': len(selected), 'microsteps': len(rows), 'optimizer_steps': len(rows) // 4,
                       'full_input_tokens': sum(r['expected_full_input_tokens'] for r in rows),
                       'supervised_tokens': sum(r['expected_supervised_tokens'] for r in rows),
                       'max_full_input_tokens': max(r['expected_full_input_tokens'] for r in rows),
                       'historical_layers': dict(collections.Counter(r['historical_exposure'] for r in selected))}
    require(content_keys['GENERIC'] == content_keys['TARGETED'], 'MATCHED_ROOT_SLOT_TYPE_GOLD_BUDGET_REQUIRED')
    require([(r['epoch'], r['schedule_key']) for r in arms['GENERIC']] == [(r['epoch'], r['schedule_key']) for r in arms['TARGETED']], 'PAIRED_SCHEDULE_ORDER_DRIFT')
    require([r['display_to_canonical'] for r in arms['GENERIC']] == [r['display_to_canonical'] for r in arms['TARGETED']], 'PAIRED_DISPLAY_TRANSFORM_MISMATCH')
    totals = [result[a]['full_input_tokens'] for a in ('GENERIC', 'TARGETED')]
    relative = abs(totals[0] - totals[1]) / min(totals)
    supervised = [result[a]['supervised_tokens'] for a in ('GENERIC', 'TARGETED')]
    require(supervised[0] == supervised[1], 'SUPERVISED_TOKEN_BUDGET_MISMATCH')
    code = [HERE / name for name in ('common.py', 'build_schedule.py', 'runner.py')]
    identity = [binding(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    tokenizer_bindings = [binding(TOKENIZER / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt') if (TOKENIZER / n).exists()]
    save(args.output / 'SCHEDULE_FREEZE.json', {'schema_version': 'oet1_matched_preexpanded_schedule_v1', 'status': 'FROZEN_CPU_PREFLIGHT_PASS',
            'created_at_utc': now(), 'pool_freeze': binding(args.pool_freeze), 'protocol': binding(PROTOCOL),
            'evaluation_freeze': binding(HERE / 'evaluation/EVALUATION_FREEZE.json'),
            'arms': result, 'training': train, 'max_length': args.max_length, 'runtime_epochs': 1, 'runtime_shuffle': False,
            'source_truncation': False, 'input_token_relative_difference': relative, 'formal_token_fairness_within_5pct': relative <= .05,
            'interpretation': 'Exploratory single paired run; historical QFP2 exposure separately declared',
            'model_directory': str(MODEL), 'model_bindings': identity, 'tokenizer_directory': str(TOKENIZER), 'tokenizer_bindings': tokenizer_bindings,
            'code_bindings': [binding(p) for p in code] + [binding(NATIVE)], 'train_eval_root_overlap': [],
            'initialization': 'B1_MERGED_PLUS_NEW_SEEDED_LORA; NO_OLD_ADAPTER_REINJECTION',
            'fixed_final_step': result['GENERIC']['optimizer_steps']})
    print(json.dumps({'status': 'FROZEN', 'arms': result, 'input_token_relative_difference': relative, 'schedule_freeze': binding(args.output / 'SCHEDULE_FREEZE.json')}))

if __name__ == '__main__': main()
