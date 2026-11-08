"""Bounded 8- or 20-slot collection: generation, then evaluation and merge.

The v53 question contracts and established full-only selection policy are used.
Each stage executes once; failed or interrupted stages remain recorded and are
never resumed automatically. Only public mechanical questions enter later chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
import evaluate_and_pool as pipeline
import slot_focus

gen, base = pipeline.generation, pipeline.base
HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
VERSION = 'h4_sequential_ab_v55_bounded_qa_v1'
ROOTS = ('cc-train-episode-138343', 'cc-train-episode-135466', 'cc-train-episode-210046')
DEFAULT_PRIORS = ()
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
CHUNK_COUNT, SLOTS_PER_TYPE = 5, 2
BUDGET = {'candidate_slots': 20, 'qwen_http_attempts_max': 30, 'full_solver_calls_max': 300}
STAGES = ('GENERATION', 'EVALUATION', 'FINALIZATION')
read, require = base.read, base.require


def now():
    return gen.v15.now()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def chunk_paths(run, index):
    require(type(index) is int and 1 <= index <= CHUNK_COUNT, 'batch_chunk_index_invalid')
    run = Path(run)
    prefix = run.name + f'_c{index:02d}'
    return run.parent / (prefix + '_generation'), run.parent / (prefix + '_evaluation')


def public_question(question):
    """Project unchanged statement text, with no Gold, evidence or Solver data."""
    kind = question['question_type']
    require(kind in gen.TYPES, 'batch_public_question_type_invalid')
    field = 'options' if kind == 'multi_select' else 'items'
    values = question.get(field)
    require(isinstance(values, dict) and len(values) == 6
            and all(isinstance(value, str) for value in values.values()), 'batch_public_six_statements_required')
    return {key: deepcopy(question[key]) for key in ('candidate_id', 'root_id', 'question_type', 'question', field)}


def _files(directory):
    return sorted(path for path in Path(directory).rglob('*') if path.is_file()
                  and '__pycache__' not in path.parts)


def budget_for(slots):
    require(type(slots) is int and slots in (8, 20), 'batch_slots_must_be_8_or_20')
    return {'candidate_slots': slots, 'qwen_http_attempts_max': slots // 4 * 6,
            'full_solver_calls_max': slots * 15}


def _chunk_config(run, chunk_count=CHUNK_COUNT):
    return [{'index': index, 'generation_run': str(chunk_paths(run, index)[0]),
             'evaluation_run': str(chunk_paths(run, index)[1]),
             'comparison_id': run.name + f'_c{index:02d}'} for index in range(1, chunk_count + 1)]


def prepare(run, candidate_history, prior_pools=DEFAULT_PRIORS, root=None, focus_plan=None, candidate_slots=20):
    run, candidate_history = Path(run).resolve(), Path(candidate_history).resolve()
    root = root or read(candidate_history.parent / 'MANIFEST.json')['root_id']
    budget = budget_for(candidate_slots)
    chunk_count = candidate_slots // 4
    require(not run.exists(), 'batch_run_already_exists')
    require(all(not path.exists() for index in range(1, chunk_count + 1)
                for path in chunk_paths(run, index)), 'batch_chunk_run_already_exists')
    bindings, sources, excluded = gen.checked_sources(candidate_history, [root])
    require(len(sources) == 1 and sources[0]['root_id'] == root and not excluded,
            'batch_requires_one_complete_root')
    focus_plan_path = Path(focus_plan).resolve() if isinstance(focus_plan, (str, Path)) else None
    plan = read(focus_plan_path) if focus_plan_path else deepcopy(focus_plan) if focus_plan is not None else slot_focus.build_plan(
        root, run.name, sources[0]['source']['turns'], slots=candidate_slots)
    slot_focus.validate_plan(plan, root, run.name, sources[0]['source']['turns'])
    require(plan['slots'] == candidate_slots, 'batch_focus_slot_count_changed')
    _, dependencies = gen._dependency_paths()
    priors = list(dict.fromkeys([gen.DEFAULT_PRIOR.resolve(), *(Path(path).resolve() for path in prior_pools)]))
    require(all(path.is_file() for path in priors), 'batch_historical_prior_missing')
    paths = set(dependencies) | {Path(__file__).resolve(), Path(pipeline.__file__), Path(pipeline.display_export.__file__), candidate_history, *priors}
    if focus_plan_path:
        paths.add(focus_plan_path)
    paths.update(gen.collection_source.audit_dependencies(candidate_history.parent))
    historical_prior, historical_paths = gen.v15.prior_candidates(root)
    paths.update(Path(path).resolve() for path in historical_paths)
    for path in priors:
        historical_prior += gen._prior_pool_rows(path, root)
    historical_prior_path = run / 'HISTORICAL_PRIOR.json'
    write_once(historical_prior_path, historical_prior)
    paths.add(historical_prior_path)
    slot_focus_path = run / 'SLOT_FOCUS_PLAN.json'
    write_once(slot_focus_path, plan)
    paths.add(slot_focus_path)
    paths.update(path for path in gen.legacy.FEEDBACK.rglob('*') if path.is_file())
    paths.update(Path(row['path']).resolve() for row in bindings.get('files', []))
    paths.update(gen.checked_expansion(candidate_history))
    config = {'version': VERSION, 'root_id': root, 'candidate_history': str(candidate_history),
        'candidate_history_sha256': base.digest(candidate_history.read_bytes()),
        'source_turn_count': len(sources[0]['source']['turns']),
        'prior_pools': [str(path) for path in priors], 'prompt_arm': 'current',
        'historical_prior_path': str(historical_prior_path),
        'historical_prior_count': len(historical_prior),
        'slot_focus_plan_path': str(slot_focus_path),
        'slot_focus_contract': slot_focus.CONTRACT,
        'requested_scene_refs': plan['requested_scene_refs'], 'available_scene_refs': plan['available_scene_refs'],
        'missing_scene_refs': plan['missing_scene_refs'], 'expansion_complete': plan['expansion_complete'],
        'chunk_count': chunk_count, 'candidate_slots_per_type_per_chunk': SLOTS_PER_TYPE,
        'chunks': _chunk_config(run, chunk_count), 'budget': budget,
        'generation_before_all_evaluation': True, 'automatic_transport_retries': 0,
        'automatic_resume': False, 'automatic_replacement_slots': False,
        'within_batch_prior': 'cumulative_previous_MECHANICAL_POOL_public_text_only',
        'feedback': 'frozen_previous_training_round_only', 'current_solver_feedback_in_generation': False,
        'solver_python': str(PYTHON), 'training_started': False, 'training_calls': 0,
        'source_training_admission': False, 'training_pool_kind': 'provisional_qa_selected_candidates'}
    write_once(run / 'CONFIG.json', config)
    paths.add(run / 'CONFIG.json')
    manifest = {'version': VERSION, 'prepared_at_utc': now(), 'config': config,
                'config_sha256': base.digest((run / 'CONFIG.json').read_bytes()), 'files': base.bindings(paths)}
    write_once(run / 'MANIFEST.json', manifest)
    verify(run)
    return manifest


def _stage_record(run, stage, suffix):
    return Path(run) / (stage + '_' + suffix + '.json')


def _begin(run, stage):
    require(stage in STAGES, 'batch_stage_invalid')
    require(not _stage_record(run, stage, 'STARTED').exists(), 'batch_stage_already_started_no_replay:' + stage)
    require(not _stage_record(run, stage, 'COMPLETE').exists(), 'batch_stage_already_complete:' + stage)
    write_once(_stage_record(run, stage, 'STARTED'), {'version': VERSION, 'stage': stage,
        'started_at_utc': now(), 'automatic_resume': False})


def _error(run, stage, exc, chunk=None):
    path = _stage_record(run, stage, 'ERROR')
    if not path.exists():
        write_once(path, {'version': VERSION, 'stage': stage, 'chunk': chunk,
            'failed_at_utc': now(), 'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'automatic_resume': False,
            'completed_chunks_preserved': True})


def _complete(run, stage, value, paths):
    result = {'version': VERSION, 'stage': stage, 'status': 'COMPLETE',
        'completed_at_utc': now(), **value, 'files': base.bindings(paths)}
    write_once(_stage_record(run, stage, 'COMPLETE'), result)
    return result


def verify(run):
    """Verify frozen configuration and every completed-stage artifact binding."""
    run = Path(run).resolve()
    manifest = read(run / 'MANIFEST.json')
    config = read(run / 'CONFIG.json')
    require(manifest['version'] == config['version'] == VERSION, 'batch_version_changed')
    require(manifest['config'] == config and manifest['config_sha256'] == base.digest((run / 'CONFIG.json').read_bytes()),
            'batch_config_changed')
    require(config['budget'] == budget_for(config['budget']['candidate_slots'])
            and config['chunk_count'] == config['budget']['candidate_slots'] // 4
            and config['candidate_slots_per_type_per_chunk'] == SLOTS_PER_TYPE
            and config['chunks'] == _chunk_config(run, config['chunk_count']), 'batch_budget_or_chunks_changed')
    require(config['slot_focus_contract'] == slot_focus.CONTRACT
            and Path(config['slot_focus_plan_path']).is_file()
            and read(config['slot_focus_plan_path'])['contract'] == slot_focus.CONTRACT,
            'batch_slot_focus_config_changed')
    require(config['prompt_arm'] == 'current' and config['generation_before_all_evaluation'] is True
            and config['automatic_resume'] is False and config['automatic_transport_retries'] == 0
            and config['automatic_replacement_slots'] is False
            and config['current_solver_feedback_in_generation'] is False
            and config['training_started'] is False and config['training_calls'] == 0,
            'batch_scope_changed')
    base.verify_bindings(manifest['files'])
    for stage in STAGES:
        path = _stage_record(run, stage, 'COMPLETE')
        if path.exists():
            value = read(path)
            require(value['version'] == VERSION and value['stage'] == stage and value['status'] == 'COMPLETE',
                    'batch_stage_completion_changed')
            require(_stage_record(run, stage, 'STARTED').is_file()
                    and not _stage_record(run, stage, 'ERROR').exists(), 'batch_stage_state_conflict')
            base.verify_bindings(value['files'])
    if _stage_record(run, 'EVALUATION', 'STARTED').exists():
        require(_stage_record(run, 'GENERATION', 'COMPLETE').is_file(), 'evaluation_before_generation_complete')
    if _stage_record(run, 'EVALUATION', 'COMPLETE').exists():
        evaluation = read(_stage_record(run, 'EVALUATION', 'COMPLETE'))
        require(evaluation['generation_completion_sha256']
                == base.digest(_stage_record(run, 'GENERATION', 'COMPLETE').read_bytes()),
                'batch_generation_evaluation_completion_binding_changed')
    if _stage_record(run, 'FINALIZATION', 'STARTED').exists():
        require(_stage_record(run, 'EVALUATION', 'COMPLETE').is_file(), 'finalization_before_evaluation_complete')
    if _stage_record(run, 'FINALIZATION', 'COMPLETE').exists():
        pool = run / 'training_pool'
        combined = read(pool / 'MANIFEST.json')
        require(combined['version'] == VERSION, 'batch_combined_version_changed')
        pipeline._verify_candidate_scope(combined)
        require(combined['generation_completion_sha256']
                == base.digest(_stage_record(run, 'GENERATION', 'COMPLETE').read_bytes())
                and combined['evaluation_completion_sha256']
                == base.digest(_stage_record(run, 'EVALUATION', 'COMPLETE').read_bytes()),
                'batch_combined_completion_binding_changed')
        base.verify_bindings(combined['files'])
        base.verify_bindings(combined['original_pool_bindings'])
        final = read(_stage_record(run, 'FINALIZATION', 'COMPLETE'))
        report = read(pool / 'REPORT.json')
        require(final['report'] == report and final['qa_selected_unique_candidates'] == report['new_unique_added'],
                'batch_final_completion_report_changed')
        for line in (pool / 'TRAINING_POOL.jsonl').read_text().splitlines():
            if line.strip():
                _check_row_references(json.loads(line), pool)
        rows = [json.loads(line) for line in (pool / 'TRAINING_POOL.jsonl').read_text().splitlines() if line.strip()]
        views = [json.loads(line) for line in (pool / 'DISPLAY_CANDIDATES.jsonl').read_text().splitlines() if line.strip()]
        require(len(rows) == len(views), 'batch_display_count_mismatch')
        for row, view in zip(rows, views):
            pipeline.display_export.replay_view(row['original_selected_question'], view)
    return manifest


def _generation_complete(run):
    config = read(Path(run) / 'CONFIG.json')
    require(_stage_record(run, 'GENERATION', 'COMPLETE').is_file(), 'all_generation_not_complete')
    result = read(_stage_record(run, 'GENERATION', 'COMPLETE'))
    require([row['index'] for row in result['chunks']] == list(range(1, config['chunk_count'] + 1)),
            'generation_completion_missing_chunks')
    require(result['candidate_slots'] == config['budget']['candidate_slots']
            and result['qwen_http_attempts'] <= config['budget']['qwen_http_attempts_max'], 'generation_total_budget_invalid')
    base.verify_bindings(result['files'])
    return result


def verify_chunk_generation(run, index):
    generation_run, _ = chunk_paths(run, index)
    config = read(Path(run) / 'CONFIG.json')
    manifest, questions, _, _ = pipeline.verify_generation(generation_run)
    require(manifest['roots'] == [config['root_id']] and manifest['prompt_arm'] == 'current'
            and manifest['candidate_slots_per_type_per_root'] == SLOTS_PER_TYPE
            and manifest['comparison_id'] == Path(run).name + f'_c{index:02d}'
            and manifest['candidate_history_path'] == config['candidate_history'], 'batch_chunk_identity_changed')
    plan = read(config['slot_focus_plan_path'])
    chunk_focus = next(row for row in plan['chunks'] if row['index'] == index)
    require(manifest.get('slot_focus_by_candidate_id') == chunk_focus['slot_focus_by_candidate_id'],
            'batch_chunk_slot_focus_manifest_changed')
    expected_prior = read(Path(config['historical_prior_path']))
    require(len(expected_prior) == config['historical_prior_count'], 'batch_historical_prior_count_changed')
    if index > 1:
        inventory = Path(run) / 'priors' / f'after_c{index - 1:02d}.json'
        previous_public = []
        for previous_index in range(1, index):
            previous_run, _ = chunk_paths(run, previous_index)
            previous_public += [public_question(question)
                for question in read(previous_run / 'MECHANICAL_POOL.json')['questions']]
        require(read(inventory) == {'questions': previous_public}, 'batch_public_inventory_not_exact_mechanical_projection')
        expected_prior += gen._prior_pool_rows(inventory, config['root_id'])
    require(read(generation_run / 'prior' / (config['root_id'] + '.json')) == expected_prior,
            'batch_chunk_prior_not_frozen_history_plus_previous_public_inventory')
    for kind in gen.TYPES:
        marker = 'multi' if kind == 'multi_select' else 'order'
        paths = sorted((generation_run / 'requests').glob('*-merged-' + marker + '-generate-1/REQUEST.json'))
        require(len(paths) == 1, 'batch_initial_request_missing_or_ambiguous')
        snapshot = read(Path(run) / 'initial' / f'c{index:02d}' / (kind + '.json'))
        request, proof = read(paths[0]), read(paths[0].with_name('PROOF.json'))
        require(request == snapshot['payload'] and proof['feedback_receipt'] == snapshot['receipt'],
                'batch_actual_initial_request_changed')
        rebuilt, receipt = gen.make_payload(generation_run, config['root_id'], kind,
                                            gen.initial_context(generation_run, config['root_id'], kind))
        require(request == rebuilt and receipt == snapshot['receipt'], 'batch_initial_snapshot_changed')
    return manifest, questions


def generate(run):
    run = Path(run).resolve()
    config = verify(run)['config']
    require(not _stage_record(run, 'EVALUATION', 'STARTED').exists()
            and all(not chunk_paths(run, index)[1].exists() for index in range(1, config['chunk_count'] + 1)),
            'batch_evaluation_exists_before_generation')
    _begin(run, 'GENERATION')
    chunks, cumulative, paths, index = [], [], [], None
    try:
        for index in range(1, config['chunk_count'] + 1):
            verify(run)
            generation_run, evaluation_run = chunk_paths(run, index)
            require(not generation_run.exists() and not evaluation_run.exists(), 'batch_chunk_already_exists_no_replay')
            priors = [Path(path) for path in config['prior_pools']]
            if index > 1:
                inventory = run / 'priors' / f'after_c{index - 1:02d}.json'
                require(read(inventory) == {'questions': cumulative}, 'batch_previous_public_inventory_changed')
                priors.append(inventory)
            gen.prepare(generation_run, Path(config['candidate_history']), roots=[config['root_id']],
                        slots=SLOTS_PER_TYPE, prior_pools=priors, prompt_arm='current',
                        comparison_id=run.name + f'_c{index:02d}',
                        slot_focus_map=next(row for row in read(config['slot_focus_plan_path'])['chunks']
                                            if row['index'] == index)['slot_focus_by_candidate_id'])
            for kind in gen.TYPES:
                payload, receipt = gen.make_payload(generation_run, config['root_id'], kind,
                                                    gen.initial_context(generation_run, config['root_id'], kind))
                snapshot = run / 'initial' / f'c{index:02d}' / (kind + '.json')
                write_once(snapshot, {'payload': payload, 'receipt': receipt})
                paths.append(snapshot)
            gen.generate(generation_run)
            _, ready = verify_chunk_generation(run, index)
            mechanical = read(generation_run / 'MECHANICAL_POOL.json')['questions']
            report = read(generation_run / 'REPORT.json')
            require(report['candidate_slots'] == 4 and report['http_attempts'] <= 6,
                    'batch_chunk_generation_budget_invalid')
            cumulative += [public_question(question) for question in mechanical]
            inventory = run / 'priors' / f'after_c{index:02d}.json'
            write_once(inventory, {'questions': cumulative})
            paths += [inventory, *_files(generation_run)]
            chunks.append({'index': index, 'generation_run': str(generation_run),
                'candidate_slots': report['candidate_slots'], 'http_attempts': report['http_attempts'],
                'mechanical_pass': len(mechanical), 'ready_for_solver': len(ready),
                'public_inventory_count': len(cumulative), 'public_inventory_path': str(inventory)})
            print(json.dumps({'event': 'batch_generation_chunk_complete', **chunks[-1]}), flush=True)
        result = _complete(run, 'GENERATION', {'chunks': chunks, 'candidate_slots': sum(row['candidate_slots'] for row in chunks),
            'qwen_http_attempts': sum(row['http_attempts'] for row in chunks),
            'mechanical_pass': sum(row['mechanical_pass'] for row in chunks),
            'ready_for_solver': sum(row['ready_for_solver'] for row in chunks),
            'training_started': False, 'current_solver_feedback_in_generation': False}, paths)
        _generation_complete(run)
        verify(run)
        return result
    except BaseException as exc:
        _error(run, 'GENERATION', exc, index)
        raise


def run_evaluation_action(action, evaluation_run, generation_run=None, log_path=None):
    """One explicit subprocess per stage; no retry or fallback interpreter."""
    require(action in ('preflight', 'live', 'finalize'), 'batch_evaluation_action_invalid')
    command = [str(PYTHON), '-u', str(HERE / 'evaluate_and_pool.py'), action, '--run', str(evaluation_run)]
    if generation_run is not None:
        command += ['--generation', str(generation_run)]
    require(log_path is not None, 'batch_evaluation_log_required')
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open('x', encoding='utf-8') as stream:
        subprocess.run(command, cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluate(run):
    run = Path(run).resolve()
    config = verify(run)['config']
    generation_record = _generation_complete(run)
    _begin(run, 'EVALUATION')
    chunks, paths, total_calls, index = [], [], 0, None
    try:
        for index in range(1, config['chunk_count'] + 1):
            verify(run)
            _generation_complete(run)
            generation_run, evaluation_run = chunk_paths(run, index)
            _, questions = verify_chunk_generation(run, index)
            require(not evaluation_run.exists(), 'batch_evaluation_chunk_already_exists_no_replay')
            planned = sum(18 if question['question_type'] == 'multi_select' else 12 for question in questions)
            require(planned <= 60 and total_calls + planned <= config['budget']['full_solver_calls_max'],
                    'batch_solver_budget_exceeded')
            for action in ('preflight', 'live', 'finalize'):
                log = run / 'logs' / f'c{index:02d}_{action}.log'
                run_evaluation_action(action, evaluation_run,
                    generation_run=generation_run if action == 'preflight' else None, log_path=log)
                paths.append(log)
            pipeline.verify(evaluation_run)
            report = read(evaluation_run / 'training_pool/REPORT.json')
            require(report['full_solver_calls'] == planned, 'batch_solver_call_count_changed')
            total_calls += report['full_solver_calls']
            paths += _files(evaluation_run)
            chunks.append({'index': index, 'evaluation_run': str(evaluation_run), 'evaluated': len(questions),
                'full_solver_calls': report['full_solver_calls'], 'qa_selected': report['new_unique_added']})
            print(json.dumps({'event': 'batch_evaluation_chunk_complete', **chunks[-1]}), flush=True)
        result = _complete(run, 'EVALUATION', {'chunks': chunks, 'full_solver_calls': total_calls,
            'generation_completion_sha256': base.digest(_stage_record(run, 'GENERATION', 'COMPLETE').read_bytes()),
            'qa_selected_before_combined_dedup': sum(row['qa_selected'] for row in chunks),
            'generation_finished_at_utc': generation_record['completed_at_utc'], 'training_started': False}, paths)
        verify(run)
        return result
    except BaseException as exc:
        _error(run, 'EVALUATION', exc, index)
        raise


def _copy_exact(source, destination):
    import shutil
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(source.read_bytes() == destination.read_bytes(), 'batch_snapshot_name_collision')
    else:
        shutil.copyfile(source, destination)
    require(base.digest(source.read_bytes()) == base.digest(destination.read_bytes()), 'batch_snapshot_copy_changed')


def _check_row_references(row, pool):
    source = Path(row['source_path'])
    require(source.is_file() and base.digest(source.read_bytes()) == row['source_sha256'], 'batch_row_source_reference_invalid')
    require(read(source) == row['full_context'], 'batch_row_full_context_changed')
    local = pool / 'contexts' / (row['source_sha256'] + '.json')
    require(local.is_file() and local.read_bytes() == source.read_bytes(), 'batch_local_context_missing_or_changed')
    def inspect(value):
        if isinstance(value, dict):
            if 'snapshot_path' in value:
                path = (pool / value['snapshot_path']).resolve()
                require(path.is_relative_to(pool.resolve()) and path.is_file()
                        and base.digest(path.read_bytes()) == value['sha256'], 'batch_relative_provenance_reference_invalid')
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(row['provenance_chain'])


def finalize(run):
    run = Path(run).resolve()
    config = verify(run)['config']
    _generation_complete(run)
    require(_stage_record(run, 'EVALUATION', 'COMPLETE').is_file(), 'all_evaluation_not_complete')
    evaluation_record = read(_stage_record(run, 'EVALUATION', 'COMPLETE'))
    require([row['index'] for row in evaluation_record['chunks']] == list(range(1, config['chunk_count'] + 1)),
            'evaluation_completion_missing_chunks')
    require(evaluation_record['full_solver_calls'] <= config['budget']['full_solver_calls_max'], 'batch_solver_total_budget_invalid')
    target = run / 'training_pool'
    require(not target.exists(), 'batch_combined_pool_already_exists')
    _begin(run, 'FINALIZATION')
    index = None
    try:
        rows, duplicates, by_key, originals = [], [], {}, []
        for index in range(1, config['chunk_count'] + 1):
            _, evaluation_run = chunk_paths(run, index)
            pipeline.verify(evaluation_run)
            pool = evaluation_run / 'training_pool'
            original_rows = [json.loads(line) for line in (pool / 'TRAINING_POOL.jsonl').read_text().splitlines() if line.strip()]
            originals += _files(pool)
            for directory in ('contexts', 'provenance'):
                for path in _files(pool / directory):
                    _copy_exact(path, target / path.relative_to(pool))
            for row in original_rows:
                _check_row_references(row, pool)
                require(row['training_allowed'] is False and row['training_started'] is False
                        and row['training_admitted_questions'] == 0, 'batch_row_promoted_to_training')
                key = pipeline.assembler.statement_key(row['original_selected_question'])
                if key in by_key:
                    retained = by_key[key]
                    duplicates.append({'candidate_id': row['candidate_id'], 'retained_candidate_id': retained['candidate_id'],
                        'chunk': index, 'reason': 'SAME_ROOT_TYPE_NORMALIZED_FULL_STATEMENT_SET',
                        'gold_content_disagreement': pipeline.assembler.canonical_gold_statements(row['original_selected_question'])
                            != pipeline.assembler.canonical_gold_statements(retained['original_selected_question'])})
                else:
                    by_key[key] = row
                    rows.append(row)
                _check_row_references(row, target)
        target.mkdir(parents=True, exist_ok=True)
        with (target / 'TRAINING_POOL.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        display_views = pipeline.display_export.write_views(target, [row['original_selected_question'] for row in rows])
        write_once(target / 'SELECTED_QUESTIONS.json', {'questions': [row['original_selected_question'] for row in rows],
                   **pipeline._candidate_scope()})
        report = {'version': VERSION, 'status': 'IMMUTABLE_PROVISIONAL_QA_BATCH_POOL_PREPARED',
            **pipeline._candidate_scope(), 'candidate_slots': config['budget']['candidate_slots'],
            'chunks': config['chunk_count'], 'selected_before_exact_dedup': sum(row['qa_selected'] for row in evaluation_record['chunks']),
            'new_unique_added': len(rows), 'duplicates': duplicates,
            'by_type': dict(Counter(row['question_type'] for row in rows)),
            'by_root': dict(Counter(row['root_id'] for row in rows)),
            'full_solver_calls': evaluation_record['full_solver_calls'],
            'qwen_http_attempts': read(_stage_record(run, 'GENERATION', 'COMPLETE'))['qwen_http_attempts'],
            'display_export': {'version': pipeline.display_export.VERSION,
                'future_training_render_input': 'DISPLAY_CANDIDATES.jsonl',
                'canonical_records_are_not_rendered_training_examples': True,
                'stats': pipeline.display_export.display_stats(display_views)},
            'original_pool_rows_unchanged': True, 'source_paths': 'original_bound_absolute_chunk_context_paths',
            'relative_provenance_snapshots_copied': True, 'semantic_gold_certified': False,
            'ground_calls': 0, 'training_calls': 0}
        require(report['selected_before_exact_dedup'] == len(rows) + len(duplicates), 'batch_selected_total_changed')
        write_once(target / 'REPORT.json', report)
        write_once(target / 'MANIFEST.json', {'version': VERSION, 'batch_manifest': str(run / 'MANIFEST.json'),
            **pipeline._candidate_scope(), 'files': base.bindings(_files(target)),
            'original_pool_bindings': base.bindings(originals),
            'generation_completion_sha256': base.digest(_stage_record(run, 'GENERATION', 'COMPLETE').read_bytes()),
            'evaluation_completion_sha256': base.digest(_stage_record(run, 'EVALUATION', 'COMPLETE').read_bytes())})
        result = _complete(run, 'FINALIZATION', {'qa_selected_unique_candidates': len(rows),
            'combined_pool': str(target), 'report': report, 'training_started': False}, _files(target))
        verify(run)
        return result
    except BaseException as exc:
        _error(run, 'FINALIZATION', exc, index)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'generate', 'evaluate', 'finalize', 'verify'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate-history', type=Path)
    parser.add_argument('--root')
    parser.add_argument('--candidate-slots', type=int, choices=(8, 20), default=20)
    parser.add_argument('--prior-pool', action='append', type=Path, default=[])
    parser.add_argument('--focus-plan', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.parent == HERE / 'runs', 'batch_run_must_be_direct_child_of_v52_runs')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('interrupted_by_signal_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    if args.action == 'prepare':
        require(args.candidate_history is not None and args.root is not None, 'batch_prepare_requires_root_and_history')
        value = prepare(run, args.candidate_history, root=args.root, prior_pools=args.prior_pool,
                        candidate_slots=args.candidate_slots, focus_plan=args.focus_plan)
    else:
        value = globals()[args.action](run)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
