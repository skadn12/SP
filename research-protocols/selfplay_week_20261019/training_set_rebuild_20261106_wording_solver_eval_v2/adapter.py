#!/usr/bin/env python3
"""Explicit second CPU preparation of the wording research evaluation."""
import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'training_set_rebuild_20261106_wording_solver_eval_v1'
spec = importlib.util.spec_from_file_location('wording_v1', V1 / 'adapter.py')
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
v1.HERE, v1.DATA, v1.OUTPUT, v1.REVIEWS = HERE, HERE / 'inputs', HERE / 'evaluation', HERE / 'EXTERNAL_REVIEWS.json'
v1.old.HERE = HERE
h = v1.h


def setup(selected, dependencies):
    source = inspect.getsource(v1.setup)
    for before, after in [('2026110601', '2026110611'), ('2026110602', '2026110612'),
                          ('wording_20261106_v1', 'wording_20261106_v2')]:
        assert before in source, ('V1_SETUP_CHANGED', before)
        source = source.replace(before, after)
    exec(compile(source, str(Path(__file__)) + ':v2_setup', 'exec'), v1.__dict__)
    v1.setup(selected, dependencies)


def checked_inputs():
    selected, sources, held, details, dependencies = v1.checked_inputs()
    crossarm = V1 / 'CROSSARM_REVIEW_ZH.md'
    reviews = v1.read(v1.REVIEWS)
    for row in reviews['items']:
        item = row['dedup']['additional_artifact']
        assert Path(item['path']).resolve() == crossarm.resolve()
        assert v1.bind(crossarm)['sha256'] == item['sha256'], 'CROSSARM_REVIEW_CHANGED'
    dependencies.update((Path(__file__), V1 / 'adapter.py', V1 / 'ATTEMPT_LEDGER.json', crossarm,
                         HERE / 'README.md', HERE / 'EXTERNAL_REVIEWS.json'))
    drafts = list((v1.EXPLORATION / 'authoring').glob('*/*/CANDIDATES_FIRST_DRAFT.json'))
    assert len(drafts) == 140, 'CROSSARM_DRAFT_CORPUS_CHANGED'
    dependencies.update(drafts)
    shard_root = v1.WEEK.parent.parent / 'SP-820-publish/first_training_pipeline/data/qa_generation_only/shards'
    shard_pools = list(shard_root.rglob('candidate_pool.json'))
    assert shard_pools, 'HISTORICAL_SHARD_POOLS_MISSING'
    dependencies.update(shard_pools)
    assert len(selected) == 4 and held == ['FGX1_F_45849_01'], 'REVIEW_SET_CHANGED'
    return selected, sources, held, details, dependencies


def verify_bindings():
    preflight = v1.read(v1.OUTPUT / 'PREFLIGHT.json')
    assert h.sha((v1.OUTPUT / 'REQUESTS.json').read_bytes()) == preflight['requests_sha256']
    for binding in v1.read(v1.OUTPUT / 'INPUT_BINDINGS.json')['files']:
        assert v1.bind(binding['path']) == binding, ('FROZEN_INPUT_CHANGED', binding['path'])
    for request in v1.read(v1.OUTPUT / 'REQUESTS.json'):
        assert h.sha((v1.OUTPUT / request['prompt_path']).read_bytes()) == request['prompt_sha256']
        assert h.sha((v1.OUTPUT / request['schema_path']).read_bytes()) == request['schema_sha256']


def manifest():
    blind, dedup, crossarm = (V1 / name for name in ('BLIND_REVIEW_ZH.md', 'DEDUP_REVIEW_ZH.md', 'CROSSARM_REVIEW_ZH.md'))
    assert all(path.is_file() for path in (blind, dedup, crossarm)), 'COMPLETED_REVIEW_RECORDS_REQUIRED'
    chosen = [r for r in v1.read(v1.PILOT / 'FINAL_RESEARCH_CANDIDATES_REFINED.json')['items'] if r['selected_version'] == 'V1_REWRITE']
    assert len(chosen) == 5 and {r['candidate_id'] for r in chosen} == set(v1.IDS)
    def record(status, path, reviewer, blind_review=False):
        result = {'status': status, 'reviewer': reviewer, 'artifact_path': str(path),
                  'artifact_sha256': v1.bind(path)['sha256']}
        if blind_review:
            result['blind'] = True
        return result
    rows = []
    for row in chosen:
        qid, candidate = row['candidate_id'], row['candidate']
        comparison = record('PASS', dedup, 'pre-solver semantic comparison; PASS means comparison performed')
        comparison['additional_artifact'] = v1.bind(crossarm)
        rows.append({'candidate_id': qid, 'candidate_sha256': v1.canonical_hash(candidate),
            'source_sha256': candidate['source_sha256'],
            'quality': record('HOLD' if qid == 'FGX1_F_45849_01' else 'PASS', blind,
                'two independent same-model blind contexts and timing adjudication', True),
            'source': record('PASS', blind, 'two independent same-model blind contexts', True),
            'dedup': comparison})
    v1.write_new(v1.REVIEWS, {'items': rows, 'scope': 'Research only; PASS denotes review performed, not novelty or training eligibility.',
        'review_limits': 'Same-model review; semantic comparison historical coverage incomplete. See both dedup records.'})
    print(json.dumps({'created': str(v1.REVIEWS), 'items': len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('manifest', 'self-check', 'check-inputs', 'prepare', 'run', 'score'))
    args = parser.parse_args()
    if args.mode == 'manifest':
        manifest()
        return
    if args.mode == 'self-check':
        print(json.dumps(v1.old.self_check(), ensure_ascii=False))
        return
    if args.mode in ('prepare', 'check-inputs'):
        selected, sources, held, details, dependencies = checked_inputs()
        print(json.dumps({'selected': [i['candidate_id'] for i in selected], 'held': held,
                          'reviews': details, 'new_model_calls': 0}, ensure_ascii=False), flush=True)
        if args.mode == 'check-inputs':
            return
        assert not v1.DATA.exists() and not v1.OUTPUT.exists() and not (HERE / 'ATTEMPT_LEDGER.json').exists(), 'NO_PREPARE_RETRY_OR_OVERWRITE'
        assert not (V1 / 'evaluation/REQUESTS.json').exists() and not (V1 / 'evaluation/PREPARE_STARTED.json').exists()
        assert not (V1 / 'evaluation/STARTED.json').exists() and not (V1 / 'evaluation/RAW_ANSWERS.json').exists(), 'V1_EXECUTION_NOT_EMPTY'
        calls = sum(2 * min(6, math.factorial(len(i['options']))) * 3 for i in selected)
        assert calls == 144
        v1.write_new(HERE / 'ATTEMPT_LEDGER.json', {'state': 'RESERVED', 'automatic_retry': False,
            'candidate_ids': [i['candidate_id'] for i in selected], 'held': held, 'reserved_calls': calls,
            'v1': 'CPU preparation interrupted before any requests or model calls; no replay',
            'v1_ledger': v1.bind(V1 / 'ATTEMPT_LEDGER.json'), 'seed_namespace': 'wording_20261106_v2'})
        v1.write_new(v1.DATA / 'ITEMS_PRIVATE.json', selected)
        v1.write_new(v1.DATA / 'QUALITY_FINAL.json', {'items': [{'candidate_id': i['candidate_id'], 'content_pass': True} for i in selected]})
        v1.write_new(v1.DATA / 'SOURCE_FREEZE.json', {'sources': sources})
        dependencies.add(HERE / 'ATTEMPT_LEDGER.json')
        setup(selected, dependencies)
        v1.OUTPUT.mkdir(parents=True, exist_ok=False)
        h.prepare()
        assert len(v1.read(v1.OUTPUT / 'REQUESTS.json')) == calls
        return
    assert (HERE / 'ATTEMPT_LEDGER.json').is_file() and (v1.OUTPUT / 'REQUESTS.json').is_file(), 'PREPARE_REQUIRED'
    setup(v1.read(v1.DATA / 'ITEMS_PRIVATE.json'), set())
    if args.mode == 'run':
        h.run()
    else:
        assert (v1.OUTPUT / 'COMPLETED.json').is_file(), 'COMPLETED_RUN_REQUIRED'
        assert not (v1.OUTPUT / 'SCORES.json').exists(), 'NO_SCORE_OVERWRITE'
        verify_bindings()
        h.score()


if __name__ == '__main__':
    main()
