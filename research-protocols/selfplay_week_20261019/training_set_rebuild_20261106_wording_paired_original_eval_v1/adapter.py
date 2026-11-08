#!/usr/bin/env python3
"""Frozen v2 paired original-wording evaluation; no retries or training."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'training_set_rebuild_20261106_wording_solver_eval_v2'
spec = importlib.util.spec_from_file_location('wording_v2', V2 / 'adapter.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v1, h = v2.v1, v2.h
DATA, OUTPUT = HERE / 'inputs', HERE / 'evaluation'
read, bind, write_new = v1.read, v1.bind, v1.write_new


def configure():
    v1.HERE, v1.DATA, v1.OUTPUT = HERE, DATA, OUTPUT
    v1.old.HERE = HERE


def verify_pairs():
    old_dir = V2 / 'evaluation'
    old_items = {x['candidate_id']: x for x in read(V2 / 'inputs/ITEMS_PRIVATE.json')}
    items = {x['candidate_id']: x for x in read(DATA / 'ITEMS_PRIVATE.json')}
    old_requests, requests = read(old_dir / 'REQUESTS.json'), read(OUTPUT / 'REQUESTS.json')
    assert len(old_requests) == len(requests) == 144
    assert set(items) == set(old_items) and len(items) == 4
    assert read(old_dir / 'PERMUTATIONS_PRIVATE.json') == read(OUTPUT / 'PERMUTATIONS_PRIVATE.json')
    assert read(old_dir / 'MODEL_CONFIG.json') == read(OUTPUT / 'MODEL_CONFIG.json'), 'MODEL_CONFIG_DIFFERS'
    for qid, item in items.items():
        old = copy.deepcopy(old_items[qid])
        assert old['question'] != item['question'], ('IDENTICAL_WORDING', qid)
        old['question'] = item['question']
        assert old == item, ('ITEM_DIFFERENCE_OUTSIDE_QUESTION', qid)
    for old, new in zip(old_requests, requests):
        qid = old['candidate_id']
        for key in old:
            if key not in ('prompt_sha256', 'prompt_token_count'):
                assert new[key] == old[key], ('REQUEST_DIFFERENCE', old['request_id'], key)
        assert set(old) == set(new)
        original, rewritten = items[qid]['question'], old_items[qid]['question']
        old_prompt = (old_dir / old['prompt_path']).read_text(encoding='utf-8')
        new_prompt = (OUTPUT / new['prompt_path']).read_text(encoding='utf-8')
        assert old_prompt.count(rewritten) == 1, ('QUESTION_NOT_UNIQUE', old['request_id'])
        assert old_prompt.replace(rewritten, original, 1) == new_prompt, ('PROMPT_DIFFERENCE', old['request_id'])
        assert (old_dir / old['schema_path']).read_bytes() == (OUTPUT / new['schema_path']).read_bytes(), ('SCHEMA_DIFFERENCE', old['request_id'])
    return {'status': 'PASS', 'paired_requests': len(requests), 'candidate_ids': sorted(items),
            'identical_request_order_seed_display_source_schema_and_config': True,
            'only_prompt_difference_is_question_text': True,
            'model_calls': 0}


def verify():
    v2.verify_bindings()
    result = verify_pairs()
    recorded = read(HERE / 'PAIR_PREFLIGHT.json')
    assert recorded == result, 'PAIR_PREFLIGHT_CHANGED'
    return result


def prepare():
    assert not DATA.exists() and not OUTPUT.exists() and not (HERE / 'ATTEMPT_LEDGER.json').exists(), 'NO_PREPARE_RETRY_OR_OVERWRITE'
    selected, sources, held, _, dependencies = v2.checked_inputs()
    v2.verify_bindings()
    assert read(V2 / 'evaluation/COMPLETED.json')['returned_requests'] == 144
    original_rows = {r['candidate_id']: r for r in read(v1.PILOT / 'DERIVED_CANDIDATES.json')['items']}
    frozen_items = read(V2 / 'inputs/ITEMS_PRIVATE.json')
    assert selected == frozen_items, 'V2_SELECTION_CHANGED'
    for item in selected:
        row = original_rows[item['candidate_id']]
        assert row['derived_question'] == item['question']
        item['question'] = row['original_question']
    prior_files = [p for p in (V2 / 'evaluation').rglob('*') if p.is_file()]
    dependencies.update(prior_files)
    dependencies.update((Path(__file__), HERE / 'README.md'))
    configure()
    write_new(HERE / 'ATTEMPT_LEDGER.json', {'state': 'RESERVED', 'automatic_retry': False,
        'candidate_ids': [i['candidate_id'] for i in selected], 'held': held,
        'reserved_calls': 144, 'comparison_run': str(V2),
        'seed_namespace': 'wording_20261106_v2',
        'prior_results_bindings': [bind(V2 / 'evaluation' / x) for x in
            ('REQUESTS.json', 'RAW_ANSWERS.json', 'COMPLETED.json', 'SCORES.json', 'SCORES_FINAL_ANSWER_ONLY.json')]})
    write_new(DATA / 'ITEMS_PRIVATE.json', selected)
    write_new(DATA / 'QUALITY_FINAL.json', {'items': [{'candidate_id': i['candidate_id'], 'content_pass': True} for i in selected]})
    write_new(DATA / 'SOURCE_FREEZE.json', {'sources': sources})
    dependencies.add(HERE / 'ATTEMPT_LEDGER.json')
    v2.setup(selected, dependencies)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    h.prepare()
    result = verify_pairs()
    write_new(HERE / 'PAIR_PREFLIGHT.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'run'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
        return
    configure()
    v2.setup(read(DATA / 'ITEMS_PRIVATE.json'), set())
    print(json.dumps(verify(), ensure_ascii=False), flush=True)
    if args.mode == 'run':
        h.run()


if __name__ == '__main__':
    main()
