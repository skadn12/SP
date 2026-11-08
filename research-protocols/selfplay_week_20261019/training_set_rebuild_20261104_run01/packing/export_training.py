#!/usr/bin/env python3
"""One-shot packaging of the final selected list; never generate, score, or train."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
WEEK = RUN.parent
REPO = WEEK.parent.parent
SELECTED = RUN / 'union_selection/FINAL_SELECTED_ITEMS.json'
OUTPUT = RUN / 'training_export_v1'
INVENTORY = WEEK / 'source_round_inventory_20261104_run01/INVENTORY.json'
V125 = REPO / 'offline-deliveries/h4_codex_train_pool_v125/build_package.py'
ADAPTER = RUN / 'model_baseline/adapter_batch_querypairs.py'
ANSWERS = REPO / 'scripts/current_method/base_qa_answers_v2_2_1.py'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
MAX_LENGTH = 12288


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dependencies():
    package = load('rebuild_export_v125', V125)
    adapter = load('rebuild_export_display_adapter', ADAPTER)
    display = load('rebuild_export_public_prompt', package.DISPLAY)
    trainer = load('rebuild_export_trainer', package.TRAINER)
    answers = load('rebuild_export_typed_answers', ANSWERS)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    paths = [Path(__file__), V125, ADAPTER, package.DISPLAY, package.TRAINER, ANSWERS,
             adapter.HISTORICAL, adapter.h.PERMUTATION_CODE, INVENTORY]
    paths += [TOKENIZER / name for name in ('tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt')
              if (TOKENIZER / name).is_file()]
    return package, adapter, display, trainer, answers, tokenizer, [file_binding(p) for p in dict.fromkeys(paths)]


def public_and_gold(item, source, deps):
    package, adapter, display, _, answers, _, _ = deps
    qtype = item['question_type']
    assert qtype in ('multi_select', 'ordering', 'partial_order'), 'UNSUPPORTED_QUESTION_TYPE'
    labels = [option['id'] for option in item['options']]
    assert labels == (list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(labels)] if qtype == 'multi_select'
                      else ['I' + str(i + 1) for i in range(len(labels))]), 'DISPLAY_LABEL_CONTRACT'
    mapping = item['label_to_option_id']
    assert set(mapping) == set(labels) and len(set(mapping.values())) == len(labels), 'INPUT_MAPPING_NOT_BIJECTIVE'
    kind = {'multi_select': 'option_set', 'ordering': 'option_sequence', 'partial_order': 'partial_order'}[qtype]
    field = 'pairs' if qtype == 'partial_order' else 'option_ids'
    gold = answers.canonical_answer({'kind': kind, field: item['gold_option_ids']}, qtype, list(mapping.values()))
    if qtype == 'multi_select':
        assert 2 <= len(gold[field]) <= 6 and len(labels) > len(gold[field]), 'MULTISELECT_GOLD_CARDINALITY'
    if 'gold_labels' in item:
        frozen = answers.map_answer({'kind': kind, field: item['gold_labels']}, qtype, mapping)
        assert frozen == gold, 'INPUT_GOLD_MAPPING_DRIFT'
    copied = copy.deepcopy(item)
    copied['permutations'] = 1
    # Normalize only the answer representation; the official display input is Gold-blind.
    copied['gold_option_ids'] = gold[field]
    variant = adapter.variants(copied)[0]
    display_to_id = variant['display_to_option_id']
    assert set(display_to_id) == set(labels) and set(display_to_id.values()) == set(mapping.values()), 'DISPLAY_MAPPING_DRIFT'
    reverse = {oid: label for label, oid in display_to_id.items()}
    displayed = answers.map_answer(gold, qtype, reverse)
    rank = {label: i for i, label in enumerate(labels)}
    if qtype == 'multi_select':
        answer_labels = sorted(displayed[field], key=rank.__getitem__)
        answer = ','.join(answer_labels)
    elif qtype == 'ordering':
        answer_labels = displayed[field]
        answer = ','.join(answer_labels)
    else:
        answer_labels = sorted(displayed[field], key=lambda pair: (rank[pair[0]], rank[pair[1]]))
        answer = ';'.join(a + '<' + b for a, b in answer_labels)
    public_field = 'options' if qtype == 'multi_select' else 'items'
    question = {'question_type': qtype, 'question': item['question']}
    public_variant = {public_field: {o['id']: o['text'] for o in variant['options']}}
    user = display.prompt(question, public_variant, source)
    public_pairs = []
    if qtype == 'partial_order':
        query_pairs = item.get('query_pairs', [])
        assert isinstance(query_pairs, list), 'QUERY_PAIRS_MUST_BE_LIST'
        assert all(isinstance(p, list) and len(p) == 2 and p[0] != p[1]
                   and set(p) <= set(mapping.values()) for p in query_pairs), 'INVALID_PUBLIC_QUERY_PAIRS'
        assert len({frozenset(p) for p in query_pairs}) == len(query_pairs), 'DUPLICATE_UNORDERED_QUERY_PAIR'
        public_pairs = [[reverse[a], reverse[b]] for a, b in query_pairs]
        if query_pairs:
            domain = {frozenset(p) for p in query_pairs}
            assert all(frozenset(p) in domain for p in gold['pairs']), 'GOLD_OUTSIDE_QUERY_PAIR_DOMAIN'
        old_instruction = 'Output only one <box>... </box> answer. For multi-select use comma-separated letters; for ordering use comma-separated item IDs.'
        assert user.endswith(old_instruction), 'HISTORICAL_PROMPT_CONTRACT_CHANGED'
        instruction = ('Output only one <box>...</box> answer. For partial_order, I1<I2 means '
                       'the first displayed item precedes the second in the relation requested by the question. '
                       'Return all supported directed pairs, including transitive pairs, separated by semicolons. '
                       'Use only displayed I-labels; use <box></box> if no pair is supported. Do not invent a total order.')
        if public_pairs:
            instruction += '\nquery_pairs (unordered action pairs defining the requested scope): ' + json.dumps(public_pairs)
        user = user[:-len(old_instruction)] + instruction
    target = '<box>' + answer + '</box>'
    messages = [{'role': 'system', 'content': package.SYSTEM}, {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': target}]
    return messages, variant, gold, answer_labels, public_pairs


def checked_features(messages, deps):
    _, _, _, trainer, _, tokenizer, _ = deps
    full = tokenizer(trainer.render_chat(tokenizer, messages, False), add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not full or full[-1] != tokenizer.eos_token_id):
        full.append(tokenizer.eos_token_id)
    prompt = tokenizer(trainer.render_chat(tokenizer, messages[:-1], True), add_special_tokens=False).input_ids
    assert full[:len(prompt)] == prompt, 'CHAT_PROMPT_IS_NOT_FULL_PREFIX'
    counts = {'input_token_count': len(full), 'prompt_token_count': len(prompt),
              'supervised_token_count': len(full) - len(prompt)}
    if len(full) > MAX_LENGTH:
        return counts, False
    features = trainer.tokenize_record(tokenizer, {'messages': messages}, MAX_LENGTH)
    assert features['input_ids'] == full, 'TRAINING_WOULD_TRUNCATE'
    assert features['labels'][:len(prompt)] == [-100] * len(prompt), 'PROMPT_LABELS_NOT_MASKED'
    assert features['labels'][len(prompt):] == full[len(prompt):] and len(full) > len(prompt), 'ASSISTANT_LABEL_DRIFT'
    assert features['attention_mask'] == [1] * len(full), 'ATTENTION_MASK_DRIFT'
    return counts, True


def write_json_x(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl_x(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(canonical(row) + '\n')


def self_check():
    report_path = HERE / 'EXPORT_INTERFACE_CHECK.json'
    assert not report_path.exists(), 'INTERFACE_CHECK_ALREADY_EXISTS'
    deps = dependencies()
    source = {'root_id': 'INTERFACE_FIXTURE_ONLY', 'turns': [
        {'turn_ref': 'FIXTURE:T001', 'speaker': 'Fixture', 'text': 'Tokenization interface fixture only.'}]}
    results = []
    for qtype, n, gold, other in [
        ('multi_select', 7, ['stable_0', 'stable_2', 'stable_3', 'stable_4', 'stable_5', 'stable_6'], ['stable_0', 'stable_1']),
        ('ordering', 4, ['stable_2', 'stable_0', 'stable_1', 'stable_3'], ['stable_3', 'stable_1', 'stable_0', 'stable_2']),
        ('partial_order', 5, [['stable_0', 'stable_1'], ['stable_1', 'stable_2']], [['stable_0', 'stable_3']]),
    ]:
        labels = list('ABCDEFG')[:n] if qtype == 'multi_select' else ['I' + str(i + 1) for i in range(n)]
        item = {'candidate_id': 'INTERFACE_FIXTURE_' + qtype, 'question_type': qtype,
                'question': 'Interface fixture; serialize the requested answer.',
                'options': [{'id': label, 'text': 'Fixture item ' + str(i)} for i, label in enumerate(labels)],
                'label_to_option_id': dict(zip(labels, ['stable_' + str(i) for i in range(n)])),
                'gold_option_ids': gold, 'query_scope': 'PRIVATE_QUERY_SCOPE_SENTINEL',
                'author_reason': 'PRIVATE_AUTHOR_SENTINEL', 'reviewer': 'PRIVATE_REVIEWER_SENTINEL'}
        if qtype == 'partial_order':
            item['query_pairs'] = [['stable_' + str(a), 'stable_' + str(b)] for a in range(n) for b in range(a + 1, n)]
        messages, variant, canonical_gold, displayed, public_pairs = public_and_gold(item, source, deps)
        changed = {**item, 'gold_option_ids': other}
        changed_messages, changed_variant, _, _, changed_pairs = public_and_gold(changed, source, deps)
        assert variant['display_to_option_id'] == changed_variant['display_to_option_id'], 'GOLD_DEPENDENT_DISPLAY'
        assert messages[:-1] == changed_messages[:-1] and public_pairs == changed_pairs, 'GOLD_DEPENDENT_PUBLIC_PROMPT'
        assert messages[-1] != changed_messages[-1], 'GOLD_NOT_REBOUND'
        assert not any(s in canonical(messages) for s in ('PRIVATE_QUERY_SCOPE_SENTINEL', 'PRIVATE_AUTHOR_SENTINEL', 'PRIVATE_REVIEWER_SENTINEL', 'stable_')), 'PRIVATE_FIELD_LEAK'
        counts, accepted = checked_features(messages, deps)
        assert accepted
        if qtype == 'partial_order':
            assert len(canonical_gold['pairs']) == 3 and len(displayed) == 3, 'TRANSITIVE_CLOSURE_NOT_SERIALIZED'
            assert public_pairs and all(a.startswith('I') and b.startswith('I') for a, b in public_pairs)
        results.append({'question_type': qtype, 'option_count': n, 'gold_blind_mapping': True,
                        'gold_blind_public_prompt': True, 'assistant_only_mask': True,
                        'no_truncation': True, 'private_fields_excluded': True, **counts})
    overflow = copy.deepcopy(messages)
    overflow[1]['content'] += '\n' + 'overflow_token ' * (MAX_LENGTH + 100)
    trainer = deps[3]
    original = trainer.tokenize_record
    def forbidden(*args, **kwargs):
        raise AssertionError('TOKENIZE_RECORD_CALLED_FOR_OVERFLOW')
    trainer.tokenize_record = forbidden
    try:
        over_counts, accepted = checked_features(overflow, deps)
        assert not accepted and over_counts['input_token_count'] > MAX_LENGTH
    finally:
        trainer.tokenize_record = original
    report = {'status': 'PASS', 'created_at': datetime.now(timezone.utc).isoformat(),
              'fixture_scope': 'In-memory interface fixtures only; no candidate training rows written.',
              'checks': results, 'overflow_rejected_before_native_tokenize_record': True,
              'overflow_input_token_count': over_counts['input_token_count'], 'max_length': MAX_LENGTH,
              'final_selection_path': str(SELECTED), 'final_selection_exists': SELECTED.exists(),
              'training_export_directory_created': False, 'model_calls': 0, 'model_weights_loaded': False,
              'training_started': False, 'replay_count': 0, 'dependencies': deps[-1]}
    write_json_x(report_path, report)
    return report


def export():
    # Nothing is written, including the output directory, unless the final list exists.
    if not SELECTED.is_file():
        raise FileNotFoundError('FINAL_SELECTION_REQUIRED: ' + str(SELECTED))
    if OUTPUT.exists():
        raise FileExistsError('EXPORT_ALREADY_EXISTS: ' + str(OUTPUT))
    raw = SELECTED.read_bytes()
    envelope = json.loads(raw)
    items = envelope['items']
    assert isinstance(items, list), 'SELECTED_ITEMS_MUST_BE_LIST'
    assert len({q['candidate_id'] for q in items}) == len(items), 'DUPLICATE_SELECTED_CANDIDATE'
    inventory = json.loads(INVENTORY.read_text())['rows']
    train_roots = {r['root_id']: r for r in inventory if r['split'] == 'train'}
    assert len(inventory) == len(train_roots) == 68, 'EXISTING_68_TRAIN_ROOT_CONTRACT'
    deps = dependencies()
    rows, lineage, pending, seen_prompts = [], [], [], set()
    for item in items:
        for key in ('origin', 'family_id', 'selection_reason'):
            assert item.get(key), 'MISSING_SELECTION_LINEAGE:' + key
        root = item['root_id']
        assert root in train_roots and item.get('split', 'train') == 'train', 'ROOT_NOT_IN_EXISTING_TRAIN_SPLIT'
        path = Path(item['source_path'])
        assert path.is_absolute() and path.is_file(), 'SOURCE_PATH_INVALID'
        source_bytes = path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        assert source_sha == item['source_sha256'], 'SOURCE_HASH_MISMATCH:' + item['candidate_id']
        source = json.loads(source_bytes)
        assert source['root_id'] == root and source['turns'], 'SOURCE_ROOT_OR_TURNS_INVALID'
        refs = [t['turn_ref'] for t in source['turns']]
        assert len(set(refs)) == len(refs), 'DUPLICATE_SOURCE_TURN_REF'
        assert all(all(isinstance(t.get(k), str) and t[k].strip() for k in ('turn_ref', 'speaker', 'text'))
                   for t in source['turns']), 'INVALID_SOURCE_TURN'
        assert item.get('source_turn_count', len(refs)) == len(refs), 'SOURCE_TURN_COUNT_DRIFT'
        messages, variant, gold, displayed, query_pairs = public_and_gold(item, source, deps)
        prompt_fp = digest(messages[:-1])
        assert prompt_fp not in seen_prompts, 'DUPLICATE_SELECTED_PUBLIC_PROMPT'
        seen_prompts.add(prompt_fp)
        counts, fits = checked_features(messages, deps)
        rounds = Counter((re.match(r'R\d{2}(?=:)', ref).group(0) if re.match(r'R\d{2}(?=:)', ref) else 'parent') for ref in refs)
        actual_rounds = sorted(k for k in rounds if k != 'parent')
        lin = {'candidate_id': item['candidate_id'], 'root_id': root, 'question_type': item['question_type'],
               'origin': item['origin'], 'family_id': item['family_id'], 'selection_reason': item['selection_reason'],
               'selected_item_sha256': digest(item), 'split': 'train', 'source_binding': {
                   'path': str(path), 'sha256': source_sha, 'size_bytes': len(source_bytes)},
               'source_turn_count': len(refs), 'source_turn_counts_by_round': dict(rounds),
               'actual_source_latest_round': actual_rounds[-1] if actual_rounds else 'parent',
               'matches_inventory_latest_source': source_sha == train_roots[root]['source_sha256'],
               'display_variant': variant, 'display_to_option_id': variant['display_to_option_id'],
               'canonical_gold': gold, 'displayed_gold': displayed, 'public_query_pairs': query_pairs,
               'prompt_sha256': prompt_fp, 'messages_sha256': digest(messages), **counts,
               'export_status': 'EXPORTED' if fits else 'EXPORT_PENDING_OVER_MAX_LENGTH'}
        lineage.append(lin)
        if not fits:
            pending.append({'candidate_id': item['candidate_id'], 'reason': 'OVER_MAX_LENGTH_NO_TRUNCATION',
                            'max_length': MAX_LENGTH, 'selected_item_sha256': digest(item),
                            'source_binding': lin['source_binding'], **counts})
            continue
        rows.append({'messages': messages, 'candidate_id': item['candidate_id'], 'root_id': root,
                     'question_type': item['question_type'], 'split': 'train', 'selection_role': item['origin'],
                     'dependence_group_id': item['family_id'],
                     'gold_cardinality': len(gold['pairs' if item['question_type'] == 'partial_order' else 'option_ids'])})
    assert hashlib.sha256(SELECTED.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest(), 'FINAL_SELECTION_CHANGED_DURING_EXPORT'
    summary = {'status': 'SEALED_WITH_EXPORT_PENDING' if pending else 'SEALED',
               'created_at': datetime.now(timezone.utc).isoformat(), 'selected_count': len(items),
               'exported_count': len(rows), 'export_pending_count': len(pending), 'max_length': MAX_LENGTH,
               'assistant_only_loss': True, 'full_source': True, 'truncate': False,
               'training_display_count_per_question': 1, 'display_seed': deps[1].h.DISPLAY_SEED,
               'question_types': dict(Counter(r['question_type'] for r in rows)),
               'origins': dict(Counter(str(q['origin']) for q in items)),
               'source_versions': dict(Counter(l['actual_source_latest_round'] for l in lineage)),
               'source_binding_policy': 'Each selected item retains its exact source path/hash; same-root versions may differ.',
               'root_split_policy': 'Existing 68 Train roots; parent sources were used by M-clean; no unseen-root claim.',
               'selection_binding': {'path': str(SELECTED), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)},
               'dependencies': deps[-1], 'model_calls': 0, 'model_weights_loaded': False,
               'training_started': False, 'replay_count': 0}
    OUTPUT.mkdir(exist_ok=False)
    write_jsonl_x(OUTPUT / 'TRAIN.jsonl', rows)
    write_jsonl_x(OUTPUT / 'QUESTION_LINEAGE.jsonl', lineage)
    write_json_x(OUTPUT / 'EXPORT_PENDING.json', {'items': pending})
    summary['outputs'] = [file_binding(OUTPUT / name) for name in ('TRAIN.jsonl', 'QUESTION_LINEAGE.jsonl', 'EXPORT_PENDING.json')]
    write_json_x(OUTPUT / 'SUMMARY.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', help='CPU tokenizer checks only; exclusive-create interface report, no training export.')
    args = parser.parse_args()
    result = self_check() if args.self_check else export()
    print(json.dumps({k: v for k, v in result.items() if k not in ('dependencies', 'checks', 'outputs')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
