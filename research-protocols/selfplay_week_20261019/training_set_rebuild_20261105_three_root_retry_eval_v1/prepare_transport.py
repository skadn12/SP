#!/usr/bin/env python3
"""Bind the existing three reviewed items to byte-identical local snapshots."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
WEEK = RUN.parent
OLD = WEEK / 'training_set_rebuild_20261105_F_supplement_run01'
RETRY = WEEK / 'training_set_rebuild_20261105_three_root_retry_v1'
BATCH = RUN / 'review_batches/batch_01'
EXPECTED = {'cc-train-episode-189272', 'cc-train-episode-249232', 'cc-train-episode-59126'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not (RUN / 'TRANSPORT_BINDINGS.json').exists(), 'ALREADY_TRANSPORTED'
assert sha(RUN / 'INPUT_FREEZE.json') == sha(OLD / 'INPUT_FREEZE.json')
freeze = read(RUN / 'INPUT_FREEZE.json')
assert len(freeze['roots']) == len({r['root_id'] for r in freeze['roots']}) == 68
for row in freeze['roots']:
    for path_key, hash_key in [('source_snapshot', 'source_snapshot_sha256'),
                               ('neutral_source_info_snapshot', 'neutral_source_info_sha256')]:
        path = RUN / row[path_key]
        assert path.is_relative_to(RUN / 'inputs') and sha(path) == row[hash_key]

original_batch = RETRY / 'review_batches/batch_01'
items_path, source_path = BATCH / 'ITEMS_PRIVATE.json', BATCH / 'SOURCE_FREEZE.json'
assert sha(items_path) == sha(original_batch / items_path.name)
assert sha(source_path) == sha(original_batch / source_path.name)
items, sources = read(items_path), read(source_path)
assert {i['root_id'] for i in items} == EXPECTED and len(items) == 3
assert {s['root_id'] for s in sources['sources']} == EXPECTED and len(sources['sources']) == 3
assert all(i['arm'] == 'F' and i['question_type'] == 'multi_select' and len(i['options']) >= 3 for i in items)
bindings = []
for item in items:
    old = Path(item['source_path'])
    new = RUN / 'inputs' / item['root_id'] / 'SOURCE.json'
    assert old.is_relative_to(OLD / 'inputs') and sha(old) == sha(new) == item['source_sha256']
    assert len(read(new)['turns']) == item['source_turn_count']
    bindings.append({'candidate_id': item['candidate_id'], 'old_source': str(old),
                     'new_source': str(new), 'source_sha256': sha(new),
                     'turn_count': item['source_turn_count']})
    item['source_path'] = str(new)
for source in sources['sources']:
    for key, hash_key, filename in [('path', 'sha256', 'SOURCE.json'),
                                    ('neutral_source_info_path', 'neutral_source_info_sha256', 'NEUTRAL_SOURCE_INFO.json')]:
        old = Path(source[key])
        new = RUN / 'inputs' / source['root_id'] / filename
        assert old.is_relative_to(OLD / 'inputs') and sha(old) == sha(new) == source[hash_key]
        source[key] = str(new)
write(items_path, items)
write(source_path, sources)
write(RUN / 'TRANSPORT_BINDINGS.json', {
    'input_freeze_original_run_id': freeze['run_id'],
    'input_freeze_note': 'Unchanged old frozen inventory and snapshots, reused for global root indices and source bindings, not a claim this run covers all roots.',
    'input_freeze_sha256': sha(RUN / 'INPUT_FREEZE.json'),
    'review_original_items_sha256': sha(original_batch / items_path.name),
    'review_transported_items_sha256': sha(items_path),
    'review_original_source_freeze_sha256': sha(original_batch / source_path.name),
    'review_transported_source_freeze_sha256': sha(source_path),
    'changes': 'Only source_path and source freeze path/neutral_source_info_path are redirected to byte-identical local snapshots; item content, answers and source bytes unchanged.',
    'all_68_source_and_neutral_snapshots_verified': True,
    'sources': bindings,
})
