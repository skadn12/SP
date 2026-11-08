"""Frozen 70-train-root P+R01 sources and previous-round feedback; no model calls.

The source JSON is copied without changes. Provenance is recorded separately;
this never synthesizes CANDIDATE_HISTORY or promotes an expansion/training scene.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERIES = PROJECT / 'offline-deliveries'
ROOT_MANIFEST = PROJECT / 'restart-n5-runs/h4_round1_cumulative_continuation_fresh_mining_v1_live_47/round1_root_manifest.jsonl'
ROOT_MANIFEST_SHA = '8ed47f06cf3f92ed91d9dd23ab1f7469aef25aac5196e3124583e27ead5c6db5'
OLD_FEEDBACK = DELIVERIES / 'h4_groundless_random_feedback_v1/feedback/compiled_v4'
OLD_LOCK_SHA = '6618db74e2f81f97953c19ef8b10ffc64b1021b5cd7d8ff99e81d8b4413236c9'
OLD_COMPILED_SHA = '54f72b0984147da2d985fdfc211e305e6506b165053534a6be487b1b5535a55e'
COMPILER = OLD_FEEDBACK.parent / 'compiler_v4.py'
COMPILER_SHA = 'ed1e8bd005379dd4b43bd8944bde228968bb23b7f9c0d9819cd5203093fe11ee'
TARGET = 'M_clean_policy'
SOURCE_MODE = 'frozen_formal_original_plus_R01'
SOURCE_STATUS = 'FROZEN_FORMAL_ORIGINAL_PLUS_R01'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify(path, expected):
    require(Path(path).is_file() and sha(path) == expected, 'frozen_source_hash_mismatch:' + str(path))
    return Path(path).resolve()


def root_rows():
    verify(ROOT_MANIFEST, ROOT_MANIFEST_SHA)
    rows = [json.loads(x) for x in ROOT_MANIFEST.read_text().splitlines() if x.strip()]
    require(len(rows) == 70 and len({r['root_id'] for r in rows}) == 70
            and all(r['split'] == 'train' for r in rows), 'fixed_70_train_manifest_invalid')
    return {r['root_id']: r for r in rows}


def validate_original_r01(source, row):
    root = row['root_id']
    require(source.get('root_id') == root, 'source_root_mismatch')
    require(source.get('expanded_dialogue_version') == 'simple_append_expanded_dialogue_v1'
            and source.get('append_mode') == 'parent_tail_only', 'not_original_plus_R01_append')
    require(source.get('complete_dialogue_sha256') == row['complete_dialogue_sha256']
            and digest({k: v for k, v in source.items() if k != 'complete_dialogue_sha256'}) == row['complete_dialogue_sha256'],
            'complete_dialogue_hash_mismatch')
    turns = source.get('turns')
    require(isinstance(turns, list) and len(turns) == row['turn_count'] == source.get('turn_count'), 'full_source_turn_count_mismatch')
    require(all(isinstance(t, dict) and all(isinstance(t.get(k), str) and t[k].strip()
                for k in ['turn_ref', 'speaker', 'text']) for t in turns), 'source_turn_schema_invalid')
    require(len({t['turn_ref'] for t in turns}) == len(turns), 'source_duplicate_turn_refs')
    parent = source.get('parent_turn_count'); child = source.get('child_turn_count')
    require(type(parent) is int and type(child) is int and parent > 0 and child > 0
            and parent + child == len(turns) and child == row['r01_turn_count'], 'P_R01_boundary_count_mismatch')
    require(all(re.fullmatch(re.escape(root) + r':P\d+:T\d+', t['turn_ref']) and t.get('origin') == 'source_grounded'
                for t in turns[:parent]), 'non_parent_turn_in_P_prefix')
    require(all(re.fullmatch(r'R01:S\d+:T\d+', t['turn_ref']) and t.get('origin') == 'generated_extension'
                for t in turns[parent:]), 'non_R01_turn_in_suffix')
    require(turns[parent]['turn_ref'] == row['r01_first_turn_ref'] and turns[-1]['turn_ref'] == row['r01_last_turn_ref'], 'R01_endpoints_changed')
    require(source['child_boundary'] == {'first_child_turn_index_1_based': parent + 1,
            'last_parent_turn_ref': turns[parent - 1]['turn_ref'], 'first_child_turn_ref': turns[parent]['turn_ref']}, 'source_boundary_changed')
    require(list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in turns[parent:])) == source['child_scene_refs'], 'R01_scene_refs_changed')
    require(source.get('qa_fact_source') == 'complete_parent_plus_child' and source.get('memory_is_qa_fact_source') is False,
            'complete_dialogue_not_QA_fact_source')


def checked_source(history_path, root_id):
    rows = root_rows()
    require(root_id in rows, 'root_outside_fixed_70_train')
    row = rows[root_id]
    path = Path(history_path).resolve()
    require(path == Path(row['canonical_dialogue_path']).resolve(), 'source_not_manifest_canonical_path')
    verify(path, row['canonical_dialogue_sha256'])
    source = read(path)
    validate_original_r01(source, row)
    audit = {'status': 'FROZEN_ORIGINAL_R01_VERIFIED', 'root_id': root_id,
        'source_mode': SOURCE_MODE, 'source_admission_status': SOURCE_STATUS,
        'fixed_training_root_count': 70, 'root_manifest': bind(ROOT_MANIFEST), 'source': bind(path),
        'complete_dialogue_sha256': row['complete_dialogue_sha256'], 'parent_turns': source['parent_turn_count'],
        'r01_turns': source['child_turn_count'], 'full_turns': len(source['turns']),
        'all_turns_preserved': True, 'new_expansion_tokens': 0, 'new_admitted_scenes': 0,
        'semantic_certified': False, 'training_pool_eligible': False, 'training_started': False,
        'note': 'Existing formal P+R01 evidence is reused unchanged; this adapter grants no new semantic or training admission.'}
    return {'root_id': root_id, 'path': path, 'source': deepcopy(source), 'sha256': sha(path),
            'files': [ROOT_MANIFEST, path, Path(__file__).resolve()], 'audit': audit}


def _compiler():
    verify(COMPILER, COMPILER_SHA)
    spec = importlib.util.spec_from_file_location('v66_readonly_old_feedback_compiler', COMPILER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feedback_data(root_id):
    """Derive exact historical feedback for any fixed train root with locked records."""
    rows = root_rows()
    require(root_id in rows, 'feedback_root_outside_fixed_70_train')
    row = rows[root_id]
    lock_path = verify(OLD_FEEDBACK / 'source_lock.json', OLD_LOCK_SHA)
    compiled_path = verify(OLD_FEEDBACK / 'COMPILED_FEEDBACK.json', OLD_COMPILED_SHA)
    lock = read(lock_path)
    require(lock['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
            and lock['automatic_latest_run_selection'] is False and lock['merge_current_generation_feedback'] is False,
            'feedback_not_previous_frozen_round')
    files = {ROOT_MANIFEST, lock_path, compiled_path, COMPILER, Path(__file__).resolve()}
    for key in ['model_identity_source', 'solver_checkpoint_source', 'formal_training_checkpoint_source']:
        a = lock[key]; files.add(verify(a['path'], a['sha256']))
    for a in lock['evaluation_sources'].values(): files.add(verify(a['path'], a['sha256']))
    require(TARGET in read(lock['model_identity_source']['path'])['models'], 'previous_trained_model_missing')
    training = read(lock['formal_training_checkpoint_source']['path'])
    require(training['status'] == 'FORMAL_TRAINING_COMPLETE' and TARGET in training['policy_reports'], 'previous_training_not_complete')
    metadata = verify(row['source_metadata_path'], row['source_metadata_sha256']); files.add(metadata)
    pool_path = metadata.parent / 'candidate_pool.json'
    verify(pool_path, row['round0_candidate_pool_sha256']); files.add(pool_path)
    pool_values = read(pool_path)
    require(isinstance(pool_values, list) and len(pool_values) == row['round0_candidate_count'], 'previous_pool_count_changed')
    pool = {q['candidate_id']: q for q in pool_values}
    require(len(pool) == len(pool_values) and all(q['root_id'] == root_id for q in pool_values), 'previous_pool_identity_invalid')
    matrix_path = Path(lock['evaluation_sources']['candidate_transition_matrix']['path'])
    exposure_path = Path(lock['evaluation_sources']['candidate_training_exposure']['path'])
    matrix = [json.loads(x) for x in matrix_path.read_text().splitlines() if x.strip()]
    exposures = [json.loads(x) for x in exposure_path.read_text().splitlines() if x.strip()]
    exposure = {x['candidate_id']: x for x in exposures}
    require(len(exposure) == len(exposures), 'duplicate_historical_exposure_id')
    selected = sorted([x for x in matrix if x['root_id'] == root_id], key=lambda x: x['candidate_id'])
    require(selected and len({x['candidate_id'] for x in selected}) == len(selected)
            and {x['candidate_id'] for x in selected} == set(pool), 'frozen_feedback_root_records_incomplete')
    require(all(x['candidate_id'] in exposure and TARGET in x.get('models', {}) for x in selected), 'historical_target_or_exposure_missing')
    compiler = _compiler()
    history = [compiler._candidate_history(x, pool[x['candidate_id']], exposure[x['candidate_id']]) for x in selected]
    aggregate = compiler._aggregate(history)
    suffix = compiler._suffix(root_id, aggregate, history, 'writer', TARGET)
    existing = root_id in lock['root_order']
    if existing:
        record = next(x for x in read(compiled_path)['records'] if x['root_id'] == root_id)
        paths = {name: OLD_FEEDBACK / root_id / name for name in ['candidate_history.json', 'aggregate.json', 'writer__M_clean_policy_feedback.txt']}
        verify(paths['candidate_history.json'], record['candidate_history_sha256'])
        verify(paths['aggregate.json'], record['aggregate_sha256'])
        verify(paths['writer__M_clean_policy_feedback.txt'], record['suffix_sha256']['writer'][TARGET])
        require(read(paths['candidate_history.json']) == history and read(paths['aggregate.json']) == aggregate
                and paths['writer__M_clean_policy_feedback.txt'].read_text() == suffix, 'existing_feedback_recompile_not_identical')
        files.update(paths.values())
    derived = deepcopy(lock)
    derived['version'] = 'v66_previous_training_round_root_projection_v1'
    derived['root_order'] = [root_id]
    derived['candidate_pools'] = {root_id: bind(pool_path)}
    derived['parent_source_lock'] = bind(lock_path)
    derived['fixed_training_root_manifest'] = bind(ROOT_MANIFEST)
    derived['compilation'] = {'compiler': bind(COMPILER), 'same_historical_functions': ['_candidate_history', '_aggregate', '_suffix'],
                            'current_solver_data_used': False, 'new_evaluation': False}
    receipt = {'version': 'v66_locked_previous_round_feedback_projection_v1', 'root_id': root_id,
        'target_solver_model': TARGET, 'source_lock_parent': bind(lock_path), 'root_manifest': bind(ROOT_MANIFEST),
        'historical_pool': bind(pool_path), 'compiler': bind(COMPILER), 'candidate_count': len(history),
        'existing_compiled_v4_byte_identical': True if existing else None,
        'existing_compiled_v4_missing_root': not existing, 'current_solver_data_used': False,
        'model_calls': 0, 'source_files': [bind(p) for p in sorted(files)]}
    return {'source_lock': derived, 'candidate_history': history, 'aggregate': aggregate,
            'writer_suffix': suffix, 'files': sorted(files), 'receipt': receipt}


def freeze_feedback(root_id, output_dir, *, data=None):
    data = feedback_data(root_id) if data is None else data
    require(data['receipt']['root_id'] == root_id, 'feedback_root_mismatch')
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), 'feedback_output_already_exists')
    objects = {'source_lock.json': data['source_lock'], root_id + '/candidate_history.json': data['candidate_history'],
               root_id + '/aggregate.json': data['aggregate'], 'ADAPTER_RECEIPT.json': data['receipt']}
    outputs = []
    for name, value in objects.items():
        p = output_dir / name; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
        outputs.append(p)
    p = output_dir / root_id / ('writer__' + TARGET + '_feedback.txt')
    with p.open('x') as f: f.write(data['writer_suffix'])
    outputs.append(p)
    return data, sorted(set(data['files'] + outputs))


def prior_candidates(root_id, data, local_history_path, v15):
    """Preserve v15's complete public-prior projection for roots beyond compiled8."""
    rows = []
    local_history_path = Path(local_history_path)
    for h in data['candidate_history']:
        q = h.get('historical_question') or {}; texts = q.get('options') or q.get('items') or {}
        rows.append({'candidate_id': h['candidate_id'], 'question_type': q.get('question_type'),
            'question': q.get('question'), 'facts': list(texts.values()), 'history_status': 'PREVIOUS_ROUND_CANDIDATE_HISTORY',
            'hard_dedup': any(h.get('flags', {}).get(x) is True for x in ['directly_trained_by_clean', 'directly_trained_by_inclusive']),
            'training_exposure': h.get('training_exposure'), 'source_path': str(local_history_path),
            'source_sha256': sha(local_history_path)})
    files = [local_history_path]
    for path in [DELIVERIES / 'h4_groundless_random_feedback_v1/evaluation/SOLVER_POOL_V3.json',
                 DELIVERIES / 'h4_r02_qa_batch_v1/runs/pilot_02/SOLVER_POOL.json',
                 DELIVERIES / 'h4_r02_qa_batch_v1/runs/pilot_02_transport_retry_01/SOLVER_POOL.json']:
        if path.exists():
            qs = [q for q in read(path)['questions'] if q['root_id'] == root_id]
            rows += [{**x, 'history_status': 'HISTORICAL_CANDIDATE_WARNING', 'hard_dedup': False,
                      'source_path': str(path), 'source_sha256': sha(path)} for x in v15.legacy.compact_questions(qs)]
            files.append(path)
    selected, selected_files = v15.dedup.selected_history(v15.DELIVERIES, root_id, v15.HERE, v15.HISTORICAL_SELECTED_EXPORTS)
    return rows + selected, files + selected_files
