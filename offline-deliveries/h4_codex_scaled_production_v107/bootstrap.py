"""Freeze a twelve-root training-side scale pilot before source matching."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = PROJECT / 'SP-820-publish/first_training_pipeline/data'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    assert not (HERE / 'PRECALL_PLAN.json').exists()
    train = DATA / 'matched_training/m_clean_policy_training_manifest.jsonl'
    rows = [json.loads(l) for l in train.read_text().splitlines()]
    excluded = set()
    protected = [HERE.parent / name for name in ('h4_batch_codex_benchmark_v104',
        'h4_codex_controlled_batch_v105', 'h4_unseen_root_validation_v106')]
    for directory in protected:
        cfg = read(directory / 'RUN_CONFIG.json') if (directory / 'RUN_CONFIG.json').exists() else None
        if cfg:
            excluded.update(r if isinstance(r, str) else r['root_id'] for r in cfg['roots'])
        else:
            excluded.update(r['root_id'] for r in read(directory / 'FINAL_BATCH_ACCOUNTING.json')['roots'])
    selected = sorted({r['root_id'] for r in rows} - excluded,
                      key=lambda r: int(r.rsplit('-', 1)[1]))[:12]
    assert len(selected) == 12
    inventory = {r['root_id']: r for r in read(DATA / 'qa_generation_only/manifest.json')['roots']}
    original_index = {r['root_id']: r for r in read(DATA / 'selection_manifest_h4_unexpanded122.json')['records']}
    source_bindings = {}
    for rid in selected:
        item = inventory[rid]
        assert sha(item['source_path']) == item['source_sha256']
        source = read(item['source_path'])
        assert source['root_id'] == rid and len(source['turns']) == source['turn_count']
        target = HERE / 'pre_inputs' / f'{rid}_BASELINE.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(Path(item['source_path']).read_bytes())
        original = read(original_index[rid]['input_path'])['source']
        source_bindings[rid] = {'root_id': rid, 'baseline_path': str(target),
            'baseline_sha256': sha(target), 'baseline_turn_count': len(source['turns']),
            'baseline_speakers': sorted({t['speaker'] for t in source['turns']}),
            'original_source_id': original['source_id'],
            'original_source_record_sha256': original['source_record_sha256']}
    # Read identities only from common dev/test; never read their model responses.
    holdout_files = []
    for filename in ('dev_full_eval_manifest.jsonl', 'test_full_eval_manifest.jsonl'):
        matches = list(DATA.rglob(filename))
        assert len(matches) == 1, (filename, matches)
        holdout_files.append(matches[0])
    holdout_roots = {json.loads(l)['root_id'] for p in holdout_files for l in p.read_text().splitlines()}
    holdout_sources = {rid: read(original_index[rid]['input_path'])['source'] for rid in holdout_roots}
    assert not set(selected) & holdout_roots
    for field, target_field in (('source_id', 'original_source_id'),
                                ('source_record_sha256', 'original_source_record_sha256')):
        assert not {v[target_field] for v in source_bindings.values()} & {v[field] for v in holdout_sources.values()}
    old_selection = read(HERE.parent / 'h4_codex_controlled_batch_v105/evidence/SOURCE_SELECTION.json')
    consumed = set(old_selection['consumed_source_ids'])
    scan = []
    for p in sorted(PROJECT.rglob('SOURCE_RESERVATIONS.json')):
        if HERE in p.parents:
            continue
        doc = read(p)
        ids = set()
        def walk(v):
            if isinstance(v, dict):
                for k, val in v.items():
                    if k == 'source_id' and isinstance(val, str):
                        ids.add(val)
                    walk(val)
            elif isinstance(v, list):
                for val in v:
                    walk(val)
        walk(doc)
        consumed.update(ids)
        scan.append({'path': str(p), 'sha256': sha(p), 'source_ids': sorted(ids)})
    pool_path = Path(old_selection['source_pool_path'])
    assert sha(pool_path) == old_selection['source_pool_sha256']
    available = sorted([r for r in read(pool_path)['rows']
        if f"soda:train:{r['row']['original_index']}" not in consumed],
        key=lambda r: int(r['row']['original_index']))
    assert len(available) >= 12
    groups = []
    for group in range(3):
        roots = selected[group * 4:(group + 1) * 4]
        # Fixed disjoint event partitions; purposive compatibility matching within them.
        events = available[group::3]
        path = HERE / 'pre_inputs' / f'group{group + 1}_EVENT_POOL.json'
        write(path, {'rows': events, 'consumed_source_ids': sorted(consumed)})
        groups.append({'group_id': group + 1, 'roots': [source_bindings[r] for r in roots],
                       'event_pool_path': str(path), 'event_pool_sha256': sha(path),
                       'selection_output': str(HERE / 'selection' / f'group{group + 1}.json')})
    write(HERE / 'EXPOSURE_AUDIT.json', {'status': 'PASS', 'purpose': 'TRAINING_SIDE_DATA_PRODUCTION',
        'selected_root_ids': selected, 'selected_roots': list(source_bindings.values()),
        'clean_training_manifest': {'path': str(train), 'sha256': sha(train)},
        'holdout_manifests': [{'path': str(p), 'sha256': sha(p)} for p in holdout_files],
        'holdout_root_overlap': [], 'holdout_source_id_overlap': [], 'holdout_source_record_overlap': [],
        'first_round_training_exposure': True, 'untouched_test_claimed': False,
        'v106_validation_reused_for_training': False})
    write(HERE / 'PRECALL_PLAN.json', {'batch_id': HERE.name,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FROZEN_BEFORE_EVENT_MATCHING_AND_AUTHORING', 'groups': groups,
        'root_selection': 'First twelve numeric clean-training root IDs excluding v104/v105/v106 roots; no performance filtering.',
        'event_selection': 'Full-history compatibility matching within three preassigned disjoint unconsumed SODA event partitions; assignments frozen before authoring.',
        'source_pool_path': str(pool_path), 'source_pool_sha256': sha(pool_path),
        'consumed_source_ids': sorted(consumed), 'consumption_scan': scan,
        'qa_rule_version': 'v105_content_contract_inheriting_v98_v103',
        'scene_ref': 'R02:C:S01', 'expansion_turns': {'min': 12, 'max': 16},
        'expansion_attempts_per_root': 1, 'qa_attempts_per_root': 1,
        'semantic_repairs': 0, 'source_replacements': 0,
        'slots_per_ready_root': {'gold2': 1, 'gold3': 1, 'gold4': 1, 'gold5': 1,
                                'gold6_control': 1, 'ordering': 2},
        'planned_new_questions_max': 72, 'planned_controls_max': 12,
        'solver_protocol': 'v15_full_only_random_mcq6x3_order12x1_v1',
        'solver_models': {'trained': 'M_clean_policy', 'base': 'M0'},
        'solver_calls_per_model_max': 1368, 'solver_automatic_retry': False,
        'selection': 'Native complete-valid 0 < equal-display partial mean < 1; Gold6 controls excluded from training package.',
        'training_package': 'Provisional eligible pool and quarantines, with root/family provenance; no training execution.',
        'external_review': 'Not performed; same inherited model and shared project background, separate author/reviewer roles.',
        'feedback': 'Historical training-side feedback where available; new validation/test answers and scores are excluded.',
        'primary_metrics': ['source READY roots / all 12 attempted roots',
            'content-pass and Solver-selected counts by root/type/Gold', 'wall time and failure categories'],
        'statistical_unit': 'root/family; no independent-call significance claims',
        'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
        'sampling_parameters': None, 'usage_tokens': None, 'billed_cost': None,
        'metadata_limitation': 'Generation model/usage not exposed; artifact token counts are not billing usage.',
        'training_started': False, 'formal_backend_comparison': False,
        'protected_batches': [str(p) for p in protected]})
    contract = HERE.parent / 'h4_codex_controlled_batch_v105/evidence/CONTENT_CONTRACT.md'
    target = HERE / 'evidence/CONTENT_CONTRACT.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(contract.read_bytes())
    print(json.dumps({'status': 'PRECALL_FIXED', 'roots': selected,
                      'available_unconsumed_events': len(available)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
