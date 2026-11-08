#!/usr/bin/env python3
"""Isolated wording-pilot M-clean evaluation; no automatic retry or training export."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PILOT = WEEK / 'training_set_rebuild_20261105_FG_wording_pilot_20261105'
EXPLORATION = WEEK / 'training_set_rebuild_20261105_FG_exploration_run01'
PREVIOUS = WEEK / 'training_set_rebuild_20261105_three_root_G_retry_eval_v1/model_baseline/adapter.py'
REVIEWS = HERE / 'EXTERNAL_REVIEWS.json'
DATA = HERE / 'inputs'
OUTPUT = HERE / 'evaluation'
IDS = {
    'FGX1_F_162896_01': ('cc-train-episode-162896', 'F'),
    'FGX1_F_191749_01': ('cc-train-episode-191749', 'F'),
    'FGX1_F_225575_01': ('cc-train-episode-225575', 'F'),
    'FGX1_F_45849_01': ('cc-train-episode-45849', 'F'),
    'FGX1_G_75241_01': ('cc-train-episode-75241', 'G'),
}

import importlib.util
spec = importlib.util.spec_from_file_location('mclean_g_retry_adapter', PREVIOUS)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
h = old.h
read, bind = h.read, old.bind


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def checked_inputs():
    final_path = PILOT / 'FINAL_RESEARCH_CANDIDATES_REFINED.json'
    derived_path = PILOT / 'DERIVED_CANDIDATES.json'
    freeze_path = EXPLORATION / 'INPUT_FREEZE.json'
    final = read(final_path)['items']
    chosen = [row for row in final if row['selected_version'] == 'V1_REWRITE']
    assert len(chosen) == 5 and {r['candidate_id'] for r in chosen} == set(IDS), 'UNEXPECTED_PILOT_SET'
    derived = {row['candidate_id']: row for row in read(derived_path)['items']}
    freeze = read(freeze_path)
    assert freeze['root_count'] == len(freeze['roots']) == 70, 'EXPLORATION_ROSTER_CHANGED'
    frozen = {row['root_id']: row for row in freeze['roots']}
    assert len(frozen) == 70
    reviews = read(REVIEWS)['items']
    by_id = {row['candidate_id']: row for row in reviews}
    assert len(reviews) == len(by_id) == 5 and set(by_id) == set(IDS), 'REVIEW_COVERAGE_REQUIRED'
    identity = read(old.IDENTITY_AUDIT)
    assert identity['status'] == 'PASS' and identity['model'] == str(old.MCLEAN) and identity['tokenizer'] == str(old.TOKENIZER)
    dependencies = {Path(__file__), PREVIOUS, old.HISTORICAL, final_path, derived_path, freeze_path, REVIEWS,
                    old.IDENTITY_AUDIT, PILOT / 'FINAL_SEAL.json', EXPLORATION / 'RUN_PROTOCOL.md',
                    EXPLORATION / 'private_semantic_reference/REFERENCE_FREEZE.json',
                    EXPLORATION / 'private_semantic_reference/SEALED135_QUESTION_PROJECTION_ADDENDUM.json'}
    selected, sources, held, review_details = [], [], [], []
    for row in chosen:
        qid = row['candidate_id']
        expected_root, arm = IDS[qid]
        candidate, prior = row['candidate'], derived[qid]
        assert candidate == prior['derived_candidate'], ('PILOT_CANDIDATE_CHANGED', qid)
        original = copy.deepcopy(prior['original_candidate'])
        assert original['question'] == prior['original_question'] and candidate['question'] == prior['derived_question']
        original['question'] = candidate['question']
        assert candidate == original and prior['changed_fields'] == ['question'], ('NON_WORDING_CHANGE', qid)
        assert candidate['root_id'] == expected_root and row['arm'] == arm and candidate.get('arm', arm) == arm
        assert candidate['question_type'] in ('multi_select', 'ordering')
        draft = Path(prior['original_draft_path']).resolve()
        assert bind(draft)['sha256'] == prior['original_draft_sha256'], ('DRAFT_HASH_CHANGED', qid)
        matching = [entry for entry in read(draft)['candidates'] if entry['candidate_id'] == qid]
        assert len(matching) == 1 and matching[0] == prior['original_candidate'], ('DRAFT_CONTENT_MISMATCH', qid)
        src = frozen[expected_root]
        source = Path(src['source_snapshot']).resolve()
        assert source == Path(candidate['source_path']).resolve() and source == Path(prior['source_path']).resolve()
        digest = bind(source)['sha256']
        assert digest == candidate['source_sha256'] == prior['source_sha256'] == src['source_sha256']
        assert len(read(source)['turns']) == candidate['source_turn_count'] == src['source_turn_count']
        review = by_id[qid]
        assert review['candidate_sha256'] == canonical_hash(candidate) and review['source_sha256'] == digest, ('REVIEW_BINDING_MISMATCH', qid)
        statuses = []
        for category in ('quality', 'source', 'dedup'):
            entry = review[category]
            assert entry['status'] in ('PASS', 'HOLD'), ('REVIEW_STATUS_REQUIRED', qid, category)
            assert isinstance(entry['reviewer'], str) and entry['reviewer'].strip()
            if category in ('quality', 'source'):
                assert entry['blind'] is True, ('BLIND_REVIEW_REQUIRED', qid, category)
            artifact = Path(entry['artifact_path']).resolve()
            assert artifact.is_file() and bind(artifact)['sha256'] == entry['artifact_sha256'], ('REVIEW_ARTIFACT_MISMATCH', qid, category)
            dependencies.add(artifact)
            statuses.append(entry['status'])
        dependencies.update((source, draft, EXPLORATION / 'private_semantic_reference' / f'REFERENCE_{expected_root}.json'))
        review_details.append({'candidate_id': qid, 'statuses': statuses})
        if any(status != 'PASS' for status in statuses):
            held.append(qid)
            continue
        labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(candidate['options'])]) if candidate['question_type'] == 'multi_select' else [f'I{i+1}' for i in range(len(candidate['options']))]
        mapping = {label: opt['option_id'] for label, opt in zip(labels, candidate['options'])}
        inverse = {oid: label for label, oid in mapping.items()}
        item = {'candidate_id': qid, 'root_id': expected_root, 'arm': arm,
                'question_type': candidate['question_type'], 'question': candidate['question'],
                'options': [{'id': label, 'text': option['text']} for label, option in zip(labels, candidate['options'])],
                'label_to_option_id': mapping, 'gold_option_ids': candidate['gold_option_ids'],
                'gold_labels': [inverse[oid] for oid in candidate['gold_option_ids']],
                'source_path': str(source), 'source_sha256': digest, 'number': 1}
        selected.append(item)
        sources.append({'root_id': expected_root, 'path': str(source), 'sha256': digest, 'turn_count': src['source_turn_count']})
    for record in identity['files']:
        path = Path(record['path'])
        assert bind(path)['sha256'] == record['expected_sha256'] == record['sha256'], ('MODEL_IDENTITY_CHANGED', str(path))
        dependencies.add(path)
    return selected, sources, held, review_details, dependencies


def setup(selected, dependencies):
    h.EXPERIMENT, h.HERE = DATA, OUTPUT
    h.reference.HERE = OUTPUT
    h.MODEL, h.TOKENIZER = old.MCLEAN, old.TOKENIZER
    h.DISPLAY_SEED, h.SAMPLE_SEED = 2026110601, 2026110602
    h.MAX_ITEMS, h.MAX_REQUESTS = 5, 180
    h.check_input_item, h.make_variants = old.check_item, old.variants
    h.reference.prompt_for, h.reference.validate = old.prompt_for, old.validate
    h.baseline.parse_and_score, h.summarize = old.parse_and_score, old.summarize
    index = {qid: i for i, qid in enumerate(sorted(IDS))}
    h.capacity_seed_for = lambda item, variant, repeat: 2026110602 + index[item['candidate_id']] * 1000 + variant * 3 + repeat
    source = inspect.getsource(h.prepare)
    changes = [
        ("assert len(roots) == 4 and len(selected) <= MAX_ITEMS, 'EXPERIMENT_SIZE_EXCEEDED'", "assert len(selected) <= MAX_ITEMS, 'EXPERIMENT_SIZE_EXCEEDED'"),
        ("assert pair not in seen_pairs, ('DUPLICATE_ROOT_ARM', pair)", "assert pair not in seen_pairs, ('DUPLICATE_ROOT_ARM', pair)"),
        ('AutoTokenizer.from_pretrained(str(MODEL),', 'AutoTokenizer.from_pretrained(str(TOKENIZER),'),
        ("seed_for(root_indices[item['root_id']], variant_index, repeat)", 'capacity_seed_for(item, variant_index, repeat)'),
        ("MODEL / 'config.json', MODEL / 'tokenizer_config.json', MODEL / 'tokenizer.json']", "MODEL / 'config.json', TOKENIZER / 'tokenizer_config.json', TOKENIZER / 'tokenizer.json'] + capacity_dependencies"),
    ]
    for before, after in changes:
        assert source.count(before) == 1, ('HISTORICAL_PREPARE_CHANGED', before)
        source = source.replace(before, after)
    h.capacity_dependencies = sorted(dependencies)
    exec(compile(source, str(Path(__file__)) + ':prepare', 'exec'), h.__dict__)
    run = inspect.getsource(h.run)
    assert run.count('tokenizer=str(MODEL)') == 1
    exec(compile(run.replace('tokenizer=str(MODEL)', 'tokenizer=str(TOKENIZER)'), str(Path(__file__)) + ':run', 'exec'), h.__dict__)
    score = inspect.getsource(h.score)
    needle = "strict = answer if checked['protocol_valid'] else invalid"
    assert score.count(needle) == 1
    exec(compile(score.replace(needle, 'strict = answer'), str(Path(__file__)) + ':score', 'exec'), h.__dict__)
    h.input_paths = lambda: sorted(p for p in DATA.rglob('*') if p.is_file())
    old_write = h.write
    def write(name, value):
        if name == 'MODEL_CONFIG.json':
            value.update(model=str(old.MCLEAN), tokenizer=str(old.TOKENIZER), base_model_only=False,
                solver='local Qwen3-4B + M-clean merged weights', adapter_applied_at_runtime=False,
                max_new_tokens=6144, display_seed=h.DISPLAY_SEED,
                sampling_seed_rule='2026110602 + sorted five-candidate index*1000 + variant_index*3 + repeat_index',
                sampling_candidate_indices=index, seed_namespace='wording_20261106_v1',
                sampling_by_type={kind: {'permutations': 'min(6,k!)', 'repeats': 3} for kind in ('multi_select', 'ordering')},
                primary_scoring='final answer only, invalid zero in fixed denominator',
                secondary_scoring='auxiliary reason/citation format diagnostic, no entailment verification',
                formal_training_export=False)
        elif name == 'PREFLIGHT.json':
            value['quality_gate'] = 'external blind quality/source and semantic dedup review completed; hold excluded'
        old_write(name, value)
    h.write = write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check-inputs', 'self-check', 'prepare', 'run', 'score'))
    args = parser.parse_args()
    if args.mode == 'self-check':
        old.HERE = HERE
        print(json.dumps(old.self_check(), ensure_ascii=False))
        return
    if args.mode in ('check-inputs', 'prepare'):
        selected, sources, held, details, dependencies = checked_inputs()
        print(json.dumps({'selected': [i['candidate_id'] for i in selected], 'held': held,
                          'reviews': details, 'new_model_calls': 0}, ensure_ascii=False), flush=True)
        if args.mode == 'check-inputs':
            return
        assert selected, 'NO_REVIEWED_CANDIDATES'
        assert not DATA.exists() and not OUTPUT.exists() and not (HERE / 'ATTEMPT_LEDGER.json').exists(), 'NO_PREPARE_RETRY_OR_OVERWRITE'
        write_new(HERE / 'ATTEMPT_LEDGER.json', {'state': 'RESERVED', 'automatic_retry': False,
                  'candidate_ids': [i['candidate_id'] for i in selected], 'held': held,
                  'reserved_calls': sum(2 * min(6, __import__('math').factorial(len(i['options']))) * 3 for i in selected),
                  'budget_limit': 180})
        write_new(DATA / 'ITEMS_PRIVATE.json', selected)
        write_new(DATA / 'QUALITY_FINAL.json', {'items': [{'candidate_id': i['candidate_id'], 'content_pass': True} for i in selected]})
        write_new(DATA / 'SOURCE_FREEZE.json', {'sources': sources})
        setup(selected, dependencies)
        OUTPUT.mkdir(parents=True, exist_ok=False)
        h.prepare()
        requests = read(OUTPUT / 'REQUESTS.json')
        ledger = read(HERE / 'ATTEMPT_LEDGER.json')
        assert len(requests) == ledger['reserved_calls'] <= 180
        return
    assert (HERE / 'ATTEMPT_LEDGER.json').is_file() and (OUTPUT / 'REQUESTS.json').is_file(), 'PREPARE_REQUIRED'
    # Existing prepared bindings are rechecked by historical run before model load.
    setup(read(DATA / 'ITEMS_PRIVATE.json'), set())
    if args.mode == 'run':
        h.run()
    else:
        assert (OUTPUT / 'COMPLETED.json').is_file(), 'COMPLETED_RUN_REQUIRED'
        assert not (OUTPUT / 'SCORES.json').exists(), 'NO_SCORE_OVERWRITE'
        h.score()


if __name__ == '__main__':
    main()
