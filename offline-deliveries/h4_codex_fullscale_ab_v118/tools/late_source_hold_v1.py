"""Record a later assigned-reader HOLD without replacing any accepted review.

This additive path can only stop a root. It cannot rescue an earlier failed
draft or revise source, reviews, questions or Gold. Semantic findings and full
reading remain explicit declarations by the actual assigned non-author reader.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_v3 as review


def inputs(batch, root, executor):
    batch, unit, folder, role, roles = review.layout(batch, root, executor)
    state = review.controller_state(unit, root)
    review.require(not state['root_stopped'], 'ROOT_ALREADY_STOPPED')
    materialized = [s for s in state['scenes']
                    if (Path(s['folder']) / 'CUMULATIVE_SOURCE.json').exists()]
    review.require(bool(materialized), 'NO_MATERIALIZED_SOURCE')
    current = materialized[-1]
    source_path = Path(current['folder']) / 'CUMULATIVE_SOURCE.json'
    source = review.read(source_path)
    turns = review.validate_turns(source, root)
    cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
    constraints = review.read(cp)
    review.require(constraints['root_id'] == root and constraints['decision'] in {'PASS', 'WARNING'},
                   'BASELINE_NOT_ELIGIBLE')
    paths = [roles, unit / 'RUN_CONFIG.json', unit / 'SOURCE_RESERVATIONS.json',
             unit / 'selection.json', cp, source_path, Path(__file__).resolve(),
             Path(review.__file__).resolve()]
    paths.append(review.verify_binding(constraints['baseline_binding'], unit, 'BASELINE'))
    paths.extend(review.verify_binding(b, unit, 'BASELINE_REVIEW') for b in constraints['review_bindings'])
    for scene in materialized:
        sf = Path(scene['folder'])
        paths.extend([sf / 'EXPANSION_DRAFT.json', sf / 'CUMULATIVE_SOURCE.json'])
        if (sf / 'SOURCE_REVIEW.json').exists():
            paths.append(sf / 'SOURCE_REVIEW.json')
    native = []
    for item in review.read(unit / 'SOURCE_RESERVATIONS.json')['allocation']:
        p = review.verify_binding({'path': item['source_item_path'],
                                   'sha256': item['source_item_sha256']}, unit, 'NATIVE_EVENT')
        paths.append(p)
        native.append(review.read(p))
    rendered = '\n'.join([
        'LATE CUMULATIVE SOURCE RECHECK: assigned reader only; can only HOLD',
        'ROOT ' + root + ' EXECUTOR ' + executor,
        'ALL_CONTINUATION_CONSTRAINTS', __import__('json').dumps(constraints, ensure_ascii=False),
        'FROZEN_SELECTION', (unit / 'selection.json').read_text(),
        'FULL_CUMULATIVE_SOURCE',
        *[t['turn_ref'] + ' | ' + t['speaker'] + ': ' + t['text'] for t in source['turns']],
        'FULL_RESERVED_NATIVE_EVENTS', __import__('json').dumps(native, ensure_ascii=False),
        'PRIOR_REVIEW_BINDINGS', __import__('json').dumps(
            [review.binding(p) for p in paths if p.name == 'SOURCE_REVIEW.json'], ensure_ascii=False),
        review.READ_LIMITATION,
    ]) + '\n'
    directory = unit / 'evidence' / ('late_source_recheck_' + hashlib.sha256(executor.encode()).hexdigest()[:16])
    return {'batch': batch, 'unit': unit, 'folder': folder, 'root': root,
            'executor': executor, 'source': source, 'turns': turns, 'constraints': constraints,
            'constraints_binding': review.binding(cp), 'source_binding': review.binding(source_path),
            'input_bindings': [review.binding(p) for p in dict.fromkeys(paths)],
            'directory': directory, 'rendered': rendered, 'last_scene_ref': current['scene_ref']}


def start(batch, root, executor):
    data = inputs(batch, root, executor)
    directory = data['directory']
    directory.mkdir(parents=True, exist_ok=False)
    render = directory / 'INPUT_RENDER.txt'
    with render.open('x') as stream:
        stream.write(data['rendered'])
    review.save_new(directory / 'READ_CONTEXT.json', {
        'root_id': root, 'executor_id': executor, 'started_at_utc': review.now(),
        'source_binding': data['source_binding'], 'input_bindings': data['input_bindings'],
        'input_render_binding': review.binding(render), 'last_scene_ref': data['last_scene_ref']})
    print(data['rendered'], end='')


def validate(raw, data):
    att = raw.get('read_attestation', {})
    review.require(att.get('full_source_read') is True and
                   att.get('all_continuation_constraints_read') is True and
                   att.get('all_reserved_native_events_read') is True,
                   'EXPLICIT_FULL_RECHECK_ATTESTATION_REQUIRED')
    review.require(raw.get('decision') == 'HOLD', 'LATE_RECHECK_CAN_ONLY_HOLD')
    semantic = review.validate_scene(raw, data)
    review.require(any(c['status'] == 'FAIL' for c in semantic['checks'].values()),
                   'SPECIFIC_FAILED_CHECK_REQUIRED')
    blocking = [f for f in semantic['findings'] if isinstance(f, dict) and f.get('severity') == 'BLOCKING']
    review.require(bool(blocking), 'BLOCKING_FINDING_REQUIRED')
    for finding in blocking:
        review.text(finding.get('reason'), 'LATE_FINDING')
        review.require(isinstance(finding.get('citations'), list) and bool(finding['citations']),
                       'BLOCKING_EXACT_CITATIONS_REQUIRED')
        for citation in finding['citations']:
            review.citation(citation, data['turns'], 'LATE_FINDING')
    return semantic


def save(batch, root, executor, judgment):
    data = inputs(batch, root, executor)
    directory = data['directory']
    context_path = directory / 'READ_CONTEXT.json'
    context = review.read(context_path)
    review.require((context['root_id'], context['executor_id']) == (root, executor), 'READ_IDENTITY_CHANGED')
    review.require(context['input_bindings'] == data['input_bindings'], 'RECHECK_INPUTS_CHANGED')
    for item in [*context['input_bindings'], context['input_render_binding']]:
        review.verify_binding(item, data['unit'], 'LATE_RECHECK')
    review.require((directory / 'INPUT_RENDER.txt').read_text() == data['rendered'], 'RECHECK_RENDER_CHANGED')
    raw_path = directory / 'RAW_JUDGMENT.json'
    with raw_path.open('xb') as stream:
        stream.write(Path(judgment).read_bytes())
    try:
        raw = review.read(raw_path)
        semantic = validate(raw, data)
        output = directory / 'LATE_SOURCE_HOLD.json'
        review.save_new(output, {
            'schema_version': 'v118_late_cumulative_source_hold_v1',
            'root_id': root, 'reviewer': executor, 'created_at_utc': review.now(),
            'started_at_utc': context['started_at_utc'], 'last_scene_ref': data['last_scene_ref'],
            'source_binding': data['source_binding'], 'read_context_binding': review.binding(context_path),
            'raw_judgment_binding': review.binding(raw_path), 'full_source_read': True,
            'turn_refs_read': list(data['turns']), 'external_semantic_certification': False,
            'prior_source_and_reviews_unchanged': True, **semantic})
        from codex_sequential_ab import append_receipt
        receipt = append_receipt(data['folder'], 'REVIEW_HOLD', root_id=root,
                                 reason='Later cumulative source recheck found a blocking defect; prior reviews retained.',
                                 late_recheck_binding=review.binding(output), reviewer=executor)
        review.save_new(directory / 'SAVE_RECEIPT.json', {
            'status': 'LATE_SOURCE_HOLD_RECORDED', 'root_id': root,
            'review_binding': review.binding(output), 'controller_receipt_binding': review.binding(receipt),
            'reviewed_source_binding': data['source_binding']})
    except Exception as exc:
        review.save_new(directory / 'SAVE_FAILURE.json', {
            'status': 'FIRST_LATE_RECHECK_REJECTED_PRESERVED', 'error': str(exc),
            'raw_judgment_binding': review.binding(raw_path)})
        raise
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['read', 'save'])
    p.add_argument('--batch', type=Path, default=review.DEFAULT_BATCH)
    p.add_argument('--root', required=True)
    p.add_argument('--executor', required=True)
    p.add_argument('--judgment', type=Path)
    a = p.parse_args()
    if a.action == 'read':
        start(a.batch, a.root, a.executor)
    else:
        review.require(a.judgment is not None, 'JUDGMENT_REQUIRED')
        save(a.batch, a.root, a.executor, a.judgment)


if __name__ == '__main__':
    main()
