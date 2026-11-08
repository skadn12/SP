#!/usr/bin/env python3
"""Verify the completed finite run without inference, rescoring or training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

R = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())


def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def check_seed_metadata(directory, config):
    """Derive seed slots independently, including unevaluated intake slots."""
    inventory_path = R.parent / 'source_round_inventory_20261104_run01/INVENTORY.json'
    rows = read(inventory_path)['rows']
    roots = sorted((row['root_id'] for row in rows if row['split'] == 'train'),
                   key=lambda root: int(root.rsplit('-', 1)[-1]))
    assert len(roots) == len(set(roots)) == 68
    root_indices = {root: index for index, root in enumerate(roots)}
    assert config['global_root_indices'] == config['sampling_root_indices'] == root_indices
    pool_path = directory.parent / 'ITEMS_PRIVATE.json'
    pool = read(pool_path)
    assert len(pool) == len({item['candidate_id'] for item in pool})
    paths = [inventory_path, pool_path]
    if 'feedback_calibration' in directory.parts:
        namespace = 'calibration'
        plan_path = R.parent / 'training_set_rebuild_plan_20261104_v1/PLAN_CONFIG.json'
        order = read(plan_path)['proposed_defaults']['feedback_calibration']['item_ids']
        paths.append(plan_path)
        slots = {cid: index for index, cid in enumerate(order)}
        offset = 100 * 100000
    elif 'legacy_evaluation' in directory.parts:
        namespace = 'legacy'
        freeze_path = R / 'LEGACY_REPRESENTATIVES_FREEZE.json'
        order = read(freeze_path)['candidate_ids_numeric_sorted']
        paths.append(freeze_path)
        slots = {cid: index for index, cid in enumerate(order)}
        offset = 1000 * 100000
    else:
        namespace, offset, slots, per_arm = 'new', 0, {}, Counter()
        for item in pool:
            key = (item['root_id'], item['arm'])
            slots[item['candidate_id']] = per_arm[key]
            per_arm[key] += 1
        assert all(count <= 2 for count in per_arm.values())
    expected = {}
    for item in pool:
        cid, root = item['candidate_id'], item['root_id']
        slot, root_index = slots[cid], root_indices[root]
        base = 2026110402 + offset + slot * 1000
        if namespace == 'new':
            base += root_index * 100000
        expected[cid] = {'namespace': namespace, 'root_index': root_index,
                         'item_slot': slot, 'base_seed': base}
    assert config['seed_namespace'] == namespace
    assert config['seed_metadata'] == expected, 'SEED_METADATA_NOT_DERIVED_FROM_FROZEN_POOL'
    return [bind(path) for path in paths]


def check_sampling(directory, requests, items, config):
    """Check frozen schedules/files without regenerating prompts or scores."""
    per_item, request_files = {}, {}
    seed_bindings = check_seed_metadata(directory, config)
    for request in requests:
        cid, mode = request['candidate_id'], request['context_mode']
        assert mode in ('full_context', 'no_context')
        variant, repeat = request['variant_id'], request['repeat_index']
        assert repeat in (0, 1, 2)
        assert request['request_id'] == f'{cid}__{mode}__{variant}__s{repeat:02d}'
        assert request['content_pass'] is True and request['formal_training_export'] is False
        item = items[cid]
        for field in ('root_id', 'arm', 'number', 'question_type', 'gold_option_ids'):
            assert request[field] == item[field], ('REQUEST_ITEM_DRIFT', cid, field)
        assert request['source_sha256_bytes'] == item['source_sha256']
        labels = [option['id'] for option in item['options']]
        mapping = request['display_to_option_id']
        option_ids = set(item['label_to_option_id'].values())
        assert list(mapping) == labels and set(mapping.values()) == option_ids and len(mapping) == len(option_ids), 'DISPLAY_MAPPING_DOMAIN_DRIFT'
        text_by_id = {item['label_to_option_id'][option['id']]: option['text'] for option in item['options']}
        expected_options = [{'id': label, 'option_id': oid, 'text': text_by_id[oid]} for label, oid in mapping.items()]
        assert request['options'] == expected_options, 'DISPLAYED_OPTION_TEXT_DRIFT'
        if item['question_type'] == 'partial_order':
            mapped_gold = [[mapping[a], mapping[b]] for a, b in request['gold_labels']]
        else:
            mapped_gold = [mapping[label] for label in request['gold_labels']]
        if item['question_type'] == 'multi_select':
            assert len(mapped_gold) == len(set(mapped_gold)) and set(mapped_gold) == set(item['gold_option_ids'])
        else:
            assert mapped_gold == item['gold_option_ids'], 'DISPLAY_GOLD_MAPPING_DRIFT'
        for kind in ('prompt', 'schema'):
            path = (directory / request[kind + '_path']).resolve()
            assert directory.resolve() in path.parents, 'REQUEST_FILE_OUTSIDE_SOLVER_DIRECTORY'
            binding = bind(path)
            assert binding['sha256'] == request[kind + '_sha256'], ('FROZEN_REQUEST_FILE_CHANGED', str(path))
            previous = request_files.get(binding['path'])
            assert previous is None or previous == binding, 'REQUEST_FILE_CHANGED_DURING_AUDIT'
            request_files[binding['path']] = binding
        assert request['prompt_token_count'] + config['max_new_tokens'] <= config['max_model_len']
        assert re.fullmatch(r'perm_\d{2}', variant), 'INVALID_VARIANT_ID'
        vi = int(variant.removeprefix('perm_'))
        assert request['sampling_seed'] == config['seed_metadata'][cid]['base_seed'] + vi * 3 + repeat
        group = per_item.setdefault(cid, {})
        key = (mode, variant, repeat)
        assert key not in group
        group[key] = tuple(request['display_to_option_id'].items())
    assert set(per_item) == set(items), 'EVALUATED_ITEM_MISSING_ALL_REQUESTS'
    for cid, group in per_item.items():
        count = min(6, math.factorial(len(items[cid]['options'])))
        variants = [f'perm_{index:02d}' for index in range(count)]
        assert set(group) == {(m, v, s) for m in ('full_context', 'no_context') for v in variants for s in range(3)}
        displays = []
        for variant in variants:
            display = group[('full_context', variant, 0)]
            assert all(group[(mode, variant, repeat)] == display for mode in ('full_context', 'no_context') for repeat in range(3))
            assert len(set(value for _, value in display)) == len(items[cid]['options'])
            displays.append(display)
        assert len(set(displays)) == count, 'REPEATED_DISPLAY_PERMUTATION'
    return [*seed_bindings, *[request_files[key] for key in sorted(request_files)]]


def main():
    output = R / 'FINAL_EXECUTION_AUDIT.json'
    assert not output.exists(), 'FINAL_EXECUTION_AUDIT_ALREADY_EXISTS'
    observed = {}

    def observe(record):
        path = Path(record['path']).resolve()
        actual = bind(path)
        assert actual['sha256'] == record['sha256'], ('FROZEN_INPUT_CHANGED', str(path))
        previous = observed.get(str(path))
        assert previous is None or previous == actual, ('INPUT_CHANGED_DURING_AUDIT', str(path))
        observed[str(path)] = actual

    def check_embedded(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value:
                assert Path(value['path']).is_absolute(), 'FROZEN_BINDING_MUST_BE_ABSOLUTE'
                observe(value)
            for child in value.values():
                check_embedded(child)
        elif isinstance(value, list):
            for child in value:
                check_embedded(child)

    def frozen_read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        observe({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        value = json.loads(raw)
        check_embedded(value)
        return value

    coverage = frozen_read(R / 'ROOT_COVERAGE.json')
    assert coverage['required_roots'] == 68 and coverage['required_author_opportunities'] == 136
    assert coverage['author_opportunities_completed'] == 136
    assert coverage['completed_both_arms'] == 68
    assert len(coverage['roots']) == len({row['root_id'] for row in coverage['roots']}) == 68
    for row in coverage['roots']:
        assert row['both_author_opportunities_completed'] is True and set(row['arms']) == {'F', 'G'}
        for arm, opportunity in row['arms'].items():
            assert opportunity['state'] in ('DRAFTS_FROZEN', 'NO_SAFE_QA_FROZEN')
            expected = R / f'batch_{row["batch_index"]:02d}' / row['root_id'] / arm / 'author/CANDIDATES_FIRST_DRAFT.json'
            assert Path(opportunity['path']).resolve() == expected
            draft = frozen_read(expected)
            author_input = frozen_read(expected.parent / 'AUTHOR_INPUT.json')
            assert draft['root_id'] == row['root_id']
            assert author_input['root_id'] == row['root_id'] and author_input['arm'] == arm
            assert draft.get('arm', author_input['arm']) == arm
            assert Path(author_input['output_directory']).resolve() == expected.parent
            assert 0 <= opportunity['first_drafts'] == len(draft['candidates']) <= 2
            assert (opportunity['state'] == 'NO_SAFE_QA_FROZEN') == (not draft['candidates'])
    first_drafts = sum(a['first_drafts'] for row in coverage['roots'] for a in row['arms'].values())
    assert first_drafts <= 272
    closed = [frozen_read(R / f'batch_{i:02d}/COMPLETED_BATCH.json') for i in range(1, 10)]
    for index, batch in enumerate(closed, 1):
        assert batch['status'] == 'COMPLETE' and batch['batch_index'] == index
        assert batch['both_arms_frozen_and_first_review_complete'] is True
        assert batch['duplicate_requests'] == 0 and batch['training_started'] is False
        assert batch['root_count'] == sum(row['batch_index'] == index for row in coverage['roots'])
        assert batch['first_drafts'] == sum(arm['first_drafts'] for row in coverage['roots'] if row['batch_index'] == index for arm in row['arms'].values())
    completed_paths = [R / 'feedback_calibration/solver/COMPLETED.json',
        *sorted(R.glob('legacy_evaluation/part_*/solver/COMPLETED.json')),
        *sorted(R.glob('new_evaluation/*/solver/COMPLETED.json'))]
    seen, candidate_modes, expected_modes, components, totals = {}, Counter(), {}, [], Counter()
    for completion in completed_paths:
        d = completion.parent
        group = ('calibration' if 'feedback_calibration' in d.parts else
                 'legacy' if 'legacy_evaluation' in d.parts else 'new')
        req = frozen_read(d / 'REQUESTS.json')
        raw = frozen_read(d / 'RAW_ANSWERS.json')
        score_document = frozen_read(d / 'SCORES.json')
        scores = score_document['requests']
        maps = [{x['request_id']: x for x in rows} for rows in (req, raw, scores)]
        item_rows = frozen_read(d / 'ITEMS_PRIVATE.json')
        item_map = {x['candidate_id']: x for x in item_rows}
        assert len(item_rows) == len(item_map) and set(item_map) == set(score_document['items'])
        assert all(len(m) == len(rows) for m, rows in zip(maps, (req, raw, scores)))
        assert maps[0].keys() == maps[1].keys() == maps[2].keys()
        summary_counts = defaultdict(Counter)
        completed = frozen_read(completion)
        assert len(req) == completed['returned_requests'] and completed['automatic_retry'] is False
        assert not seen.keys() & maps[0].keys(), 'DUPLICATE_REQUEST_ACROSS_RUN'
        for request_id, request in maps[0].items():
            answer, score = maps[1][request_id], maps[2][request_id]
            assert answer['status'] == 'RETURNED'
            assert answer['automatic_retry'] is False
            assert answer['prompt_sha256'] == request['prompt_sha256']
            for field in ('candidate_id', 'root_id', 'arm', 'number', 'question_type',
                          'context_mode', 'variant_id', 'repeat_index', 'sampling_seed'):
                assert score[field] == request[field], ('SCORE_REQUEST_IDENTITY_DRIFT', request_id, field)
            assert score['raw_prediction'] == answer['raw_prediction'] and score['finish_reason'] == answer['finish_reason'], 'SCORE_RAW_RESPONSE_DRIFT'
            assert isinstance(score['valid'], bool)
            assert not score['valid'] or score['protocol_valid'] is True
            counts = summary_counts[(score['candidate_id'], score['context_mode'])]
            counts['planned_calls'] += 1
            counts['valid_calls' if score['valid'] else 'invalid_calls'] += 1
            seen[request_id] = str(d)
            key = (group, request['candidate_id'], request['context_mode'])
            candidate_modes[key] += 1
            expected_modes[key] = min(6, math.factorial(len(item_map[request['candidate_id']]['options']))) * 3
        config = frozen_read(d / 'MODEL_CONFIG.json')
        assert config['planned_requests'] == len(req), 'PLANNED_REQUEST_COUNT_DRIFT'
        for cid, modes in score_document['items'].items():
            assert set(modes) == {'full_context', 'no_context'}
            for mode, summary in modes.items():
                for field in ('planned_calls', 'valid_calls', 'invalid_calls'):
                    assert summary[field] == summary_counts[(cid, mode)][field], ('FROZEN_SCORE_COUNT_DRIFT', cid, mode, field)
        request_bindings = check_sampling(d, req, item_map, config)
        for binding in request_bindings:
            observe(binding)
        assert config['model'] == '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
        assert config['tokenizer'] == '/root/autodl-tmp/Qwen3-4B'
        for key, value in {'temperature': 0.8, 'top_p': 0.9, 'top_k': -1,
            'max_new_tokens': 6144, 'max_model_len': 16384,
            'engine_seed': 2026110303, 'display_seed': 2026110401,
            'native_thinking': False, 'automatic_retry': False,
            'training': False, 'adapter_applied_at_runtime': False}.items():
            assert config[key] == value, ('MODEL_CONTRACT_DRIFT', str(d), key)
        # Inventory/plan seed proofs are shared across components and belong in
        # audit_input_bindings; component bindings remain disjoint execution files.
        component_bindings = {binding['path']: binding for binding in request_bindings
                              if Path(binding['path']).is_relative_to(d.parent)}
        for name in ('REQUESTS.json', 'RAW_ANSWERS.json', 'SCORES.json', 'MODEL_CONFIG.json', 'COMPLETED.json', 'ITEMS_PRIVATE.json'):
            binding = observed[str((d / name).resolve())]
            component_bindings[binding['path']] = binding
        components.append({'group': group, 'requests': len(req), 'bindings': [component_bindings[key] for key in sorted(component_bindings)]})
        totals[group] += len(req)
    assert totals['calibration'] == 144 and totals['legacy'] == 2448
    assert totals['new'] <= 9792 and sum(totals.values()) <= 23004
    assert totals['new'] == sum(x['unique_completed_requests'] for x in closed)
    assert candidate_modes == expected_modes
    audited_components = {binding['path']: binding['sha256'] for component in components for binding in component['bindings']}
    for batch in closed:
        for component in batch['components']:
            for binding in component['bindings']:
                assert audited_components.get(binding['path']) == binding['sha256'], 'CLOSED_COMPONENT_NOT_IN_FINAL_AUDIT'
    observe(bind(Path(__file__).resolve()))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'required_roots': 68, 'completed_author_opportunities': 136,
        'actual_new_first_drafts': first_drafts, 'unique_requests': len(seen),
        'requests_by_group': dict(totals), 'duplicate_requests': 0,
        'each_evaluated_candidate_each_mode_calls': 'min(6,n!) unique permutations x 3 repeats', 'rescoring': False,
        'prompt_and_schema_bytes_verified': True, 'paired_context_displays_and_sampling_schedule_verified': True,
        'training_started': False, 'components': components,
        'coverage_binding': observed[str(R / 'ROOT_COVERAGE.json')],
        'batch_closure_bindings': [observed[str(R / f'batch_{i:02d}/COMPLETED_BATCH.json')] for i in range(1, 10)],
        'audit_input_bindings': [observed[key] for key in sorted(observed)]}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    for binding in list(observed.values()):
        observe(binding)
    with output.open('x', encoding='utf-8') as f:
        f.write(payload)
    print(json.dumps({k: v for k, v in report.items() if k not in ('components', 'batch_closure_bindings', 'audit_input_bindings')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
