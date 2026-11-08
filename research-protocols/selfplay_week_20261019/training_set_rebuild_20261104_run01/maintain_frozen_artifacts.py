#!/usr/bin/env python3
"""Run existing intake/finalization commands and refresh factual progress indexes.

No authoring, model inference, retry, content repair, or collection selection.
"""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parent
FINALIZER = R.parent / 'root_capacity_feedback_20261104_run01/finalize_batch.py'


def read(path):
    return json.loads(path.read_text())


def copy_once(source, target):
    data = source.read_bytes()
    if target.exists():
        assert target.read_bytes() == data, ('FROZEN_BYTES_CHANGED', str(target))
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    return True


def run(args):
    result = subprocess.run(['python', *map(str, args)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def main():
    new_packets, finalized = [], []
    maps = [R / 'BLIND_CONTEXT_MAP_PRIVATE.json', *sorted(R.glob('BLIND_CONTEXT_MAP_BATCH_*_PRIVATE.json'))]
    for mapping in maps:
        for context in read(mapping)['contexts']:
            batch = Path(context['private_batch'])
            author = batch.parent / 'author'
            public = Path(context['public_context'])
            if not (author / 'CANDIDATES_FIRST_DRAFT.json').exists():
                continue
            batch_name = batch.parents[2].name
            if not (batch / 'ITEMS_PRIVATE.json').exists():
                prepare = R / ('prepare_batch.py' if batch_name == 'batch_01' else batch_name + '/prepare_batch.py')
                run([prepare, '--author', author, '--batch', batch, '--arm', context['arm']])
            for packet in sorted((batch / 'blind_packets').glob('Q*.json')):
                target = public / 'blind_packets' / packet.name
                if copy_once(packet, target):
                    new_packets.append(str(target.relative_to(R / 'blind_review')))
            for review in sorted((public / 'blind_results').glob('Q*.json')):
                copy_once(review, batch / 'blind_results' / review.name)
            if not (batch / 'QUALITY_FINAL.json').exists():
                items = read(batch / 'ITEMS_PRIVATE.json')
                if all((batch / 'blind_results' / (i['review_id'] + '.json')).exists() for i in items):
                    finalized.append(json.loads(run([FINALIZER, '--batch', batch])))
            target_view = batch / 'target_hash_interface_view'
            if target_view.exists():
                for review in sorted((batch / 'blind_results').glob('Q*.json')):
                    copy_once(review, target_view / 'blind_results' / review.name)
                if not (target_view / 'QUALITY_FINAL.json').exists():
                    items = read(target_view / 'ITEMS_PRIVATE.json')
                    if all((target_view / 'blind_results' / (i['review_id'] + '.json')).exists() for i in items):
                        finalized.append(json.loads(run([FINALIZER, '--batch', target_view])))

    coverage = []
    for index in range(1, 10):
        source_freeze = R / ('SOURCE_FREEZE.json' if index == 1 else f'batch_{index:02d}/SOURCE_FREEZE.json')
        for source in read(source_freeze)['sources']:
            row = {'root_id': source['root_id'], 'batch_index': index, 'arms': {}}
            for arm in ('F', 'G'):
                author = R / f'batch_{index:02d}' / row['root_id'] / arm / 'author'
                draft = author / 'CANDIDATES_FIRST_DRAFT.json'
                if draft.exists():
                    count = len(read(draft)['candidates'])
                    state = 'DRAFTS_FROZEN' if count else 'NO_SAFE_QA_FROZEN'
                    value = {'state': state, 'first_drafts': count, 'path': str(draft),
                             'sha256': hashlib.sha256(draft.read_bytes()).hexdigest()}
                else:
                    started = any((author / name).exists() for name in ('EXPLORATION.json', 'TARGETS_FROZEN.json'))
                    value = {'state': 'EXPLORING' if started else 'NOT_STARTED', 'first_drafts': 0}
                row['arms'][arm] = value
            row['both_author_opportunities_completed'] = all(
                a['state'] in ('DRAFTS_FROZEN', 'NO_SAFE_QA_FROZEN') for a in row['arms'].values())
            coverage.append(row)
    assert len(coverage) == len({r['root_id'] for r in coverage}) == 68
    now = datetime.now(timezone.utc).isoformat()
    covered = {'updated_at_utc': now, 'required_roots': 68, 'required_author_opportunities': 136,
               'completed_both_arms': sum(r['both_author_opportunities_completed'] for r in coverage),
               'author_opportunities_completed': sum(a['state'] in ('DRAFTS_FROZEN', 'NO_SAFE_QA_FROZEN')
                   for r in coverage for a in r['arms'].values()),
               'source_preparation_alone_counts_as_author_coverage': False,
               'batch_yield_can_stop_remaining_coverage': False, 'roots': coverage}
    (R / 'ROOT_COVERAGE.json').write_text(json.dumps(covered, ensure_ascii=False, indent=2) + '\n')
    quality = {}
    for path in sorted(R.glob('batch_*/cc-*/[FG]/batch/QUALITY_FINAL.json')):
        interface = path.parent / 'review_enum_interface_view/QUALITY_FINAL.json'
        target_interface = path.parent / 'target_hash_interface_view/QUALITY_FINAL.json'
        if target_interface.exists():
            interface = target_interface
        citation_interface = path.parent / 'review_citation_interface_view/QUALITY_FINAL.json'
        if citation_interface.exists():
            interface = citation_interface
        for item in read(interface if interface.exists() else path)['items']:
            quality[item['candidate_id']] = item
    for path in sorted(R.glob('new_evaluation/*/QUALITY_FINAL.json')):
        for item in read(path)['items']:
            quality[item['candidate_id']] = item
    # A separately frozen policy application supersedes the classification
    # in historical aggregates; the first answer and original HOLD remain.
    for path in sorted(R.glob('batch_*/cc-*/[FG]/batch/warning_policy_view/QUALITY_FINAL.json')):
        for item in read(path)['items']:
            quality[item['candidate_id']] = item
    completed = []
    for path in sorted([*R.glob('legacy_evaluation/part_*/solver/COMPLETED.json'),
                        *R.glob('new_evaluation/*/solver/COMPLETED.json')]):
        completed.append({'path': str(path), 'requests': read(path)['returned_requests']})
    previous = read(R / 'PROGRESS.json')
    previous.update(updated_at_utc=now, phase='ALL68_AUTHORING_BLIND_REVIEW_AND_MCLEAN_EVALUATION',
        status='IN_PROGRESS', required_roots=68,
        author_opportunities_completed=covered['author_opportunities_completed'],
        new_roots_completed_both_arms=covered['completed_both_arms'],
        actual_new_first_drafts=sum(a['first_drafts'] for r in coverage for a in r['arms'].values()),
        quality_finalized_candidates=len(quality), quality_pass=sum(q['status'] == 'PASS' for q in quality.values()),
        quality_hold=sum(q['status'] == 'HOLD' for q in quality.values()),
        completed_solver_calls=144 + sum(c['requests'] for c in completed), completed_evaluations=completed,
        all68_coverage_required_regardless_of_batch_yield=True, training_started=False)
    (R / 'PROGRESS.json').write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'new_public_packets': new_packets, 'new_quality_finalizations': finalized,
        'progress': {k: previous[k] for k in ('author_opportunities_completed', 'new_roots_completed_both_arms',
            'actual_new_first_drafts', 'quality_pass', 'quality_hold', 'completed_solver_calls')}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
