"""Schedule both types in one bounded batch; freeze both reviews before Solver."""
import argparse
from collections import Counter
from pathlib import Path

from common import HERE, V75, aware, bind, load_multi, need, now, read, require_bindings, require_shard, verify_bindings, write
import ordering_generation

VERSION = 'v97_full_turn_mixed_shard_v1'
import source_subset


def config(home=HERE):
    home = require_shard(home); value = source_subset.validate(read(home / 'CONFIG.json')); multi = value['multiselect_config']; slots = value['ordering_slots']
    need(value['version'] == VERSION and value['question_types'] == ['multi_select', 'ordering']
         and value['training_allowed'] is False and value['automatic_followup'] is False
         and all(value[k] == 0 for k in ('transport_retries', 'semantic_retries', 'mechanical_repairs')),
         'mixed_batch_policy_changed')
    need(all(type(value[k]) is int for k in ('transport_retries', 'semantic_retries', 'mechanical_repairs',
              'max_generation_http', 'max_candidates', 'max_solver_calls_per_model', 'ordering_output_max_tokens'))
         and value['ordering_output_max_tokens'] == 6144 and value['source_mode'] == 'complete_frozen_sources'
         and value['baseline_pool'] == multi['baseline_pool'] and isinstance(value['batch_id'], str)
         and value['batch_id'].replace('_', '').isalnum(), 'mixed_batch_field_types_or_source_mode')
    need(Path(value['multiselect_engine']).resolve() == V75, 'multiselect_strategy_changed')
    multi_by_root = {s['root_id']: s for s in multi['slots']}
    need(len(slots) == len(multi_by_root) and len({s['slot_id'] for s in slots}) == len(slots)
         and {s['root_id'] for s in slots} == set(multi_by_root)
         and all(type(s['candidate_count']) is int and s['candidate_count'] == 2
                 and s['slot_id'] == multi_by_root[s['root_id']]['slot_id'].replace('_family', '_ordering')
                 for s in slots), 'both_types_require_same_distinct_roots')
    for slot in slots:
        m = multi_by_root[slot['root_id']]
        need(all(slot[k] == m[k] for k in ('source_path', 'source_sha256', 'history_path', 'history_sha256'))
             and slot['prior_pool_path'] == multi['baseline_pool'], 'ordering_full_source_contract_changed')
    ordering_count = sum(s['candidate_count'] for s in slots)
    n = len(slots)
    need(1 <= n <= 4 and value['max_generation_http'] == 3*n
         and value['max_candidates'] == 7*n and value['max_solver_calls_per_model'] == 114*n
         and multi['max_http'] == 2*n and multi['max_questions'] == 5*n
         and multi['max_calls_per_model'] == 90*n,
         'shard_one_to_four_ready_roots_dynamic_budget')
    need(value['max_generation_http'] == multi['max_http'] + len(slots)
         and value['max_candidates'] == multi['max_questions'] + ordering_count
         and value['max_solver_calls_per_model'] == multi['max_calls_per_model'] + 12 * ordering_count
         and value['ordering_review_required'] is True, 'typed_batch_budget_changed')
    return value


def prepare(home=HERE):
    home = Path(home).resolve(); value = config(home)
    need(not (home / 'BATCH_PREPARATION.STARTED.json').exists(), 'batch_preparation_no_replay')
    write(home / 'BATCH_PREPARATION.STARTED.json', {'started_at_utc': now(), 'real_http_calls': 0})
    multi = home / 'multiselect'; write(multi / 'CONFIG.json', value['multiselect_config'])
    errors = []; completed = []
    for lane, action in [('multi_select', lambda: load_multi('generation_driver').prepare(multi)),
                         ('ordering', lambda: ordering_generation.prepare(home))]:
        try:
            action(); completed.append(lane)
        except Exception as error:
            errors.append({'question_type': lane, 'error': str(error)})
    result = {'version': VERSION, 'completed_at_utc': now(), 'completed_types': completed, 'errors': errors,
              'status': 'ERROR' if errors else 'PREPARED', 'training_allowed': False, 'automatic_retry': False}
    if errors:
        write(home / 'BATCH_PREPARATION.ERROR.json', result); raise ValueError('batch_preparation_incomplete')
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.md')) + [home / 'CONFIG.json',
            home / 'SHARD_ENGINE_INSTALLATION.json', home / 'SHARD_INPUTS.COMPLETE.json',
            multi / 'CONFIG.json', multi / 'EXECUTION_PLAN.json', home / 'ordering/PREPARATION.COMPLETE.json']
    result['files'] = [bind(p) for p in sorted(set(paths))]
    write(home / 'BATCH_PREPARATION.COMPLETE.json', result); return result


def verify_plan(home=HERE):
    home = Path(home).resolve(); config(home); plan = read(home / 'BATCH_PREPARATION.COMPLETE.json')
    need(plan['status'] == 'PREPARED' and plan['completed_types'] == ['multi_select', 'ordering'] and not plan['errors'], 'both_type_preparation_required')
    required = [home / 'CONFIG.json', home / 'SHARD_ENGINE_INSTALLATION.json', home / 'SHARD_INPUTS.COMPLETE.json',
                home / 'multiselect/CONFIG.json', home / 'multiselect/EXECUTION_PLAN.json',
                home / 'ordering/PREPARATION.COMPLETE.json'] + list(HERE.glob('*.py')) + list(HERE.glob('*.md'))
    require_bindings(plan['files'], required)
    need(plan['version'] == VERSION and plan['training_allowed'] is False
         and aware(plan['completed_at_utc']) <= aware(now()), 'batch_preparation_contract')
    load_multi('generation_driver').verify_plan(home / 'multiselect')
    ordering_generation.verify(home); return plan


def generate(home=HERE):
    home = Path(home).resolve(); verify_plan(home)
    need(not (home / 'BATCH_GENERATION.STARTED.json').exists(), 'batch_generation_no_replay')
    write(home / 'BATCH_GENERATION.STARTED.json', {'started_at_utc': now(), 'types': ['multi_select', 'ordering']})
    completed = []; errors = []
    # Try both scheduled types even if the first lane fails. Preserve each
    # outcome; a failed lane never becomes a successful batch completion.
    for lane, action in [('multi_select', lambda: load_multi('generation_driver').generate(home / 'multiselect')),
                         ('ordering', lambda: ordering_generation.generate(home))]:
        try:
            action(); completed.append(lane)
        except Exception as error:
            errors.append({'question_type': lane, 'error': str(error)})
    attempts = len(list((home / 'multiselect/assets').glob('*/rounds/*/requests/*/HTTP_ATTEMPT.json')))
    attempts += len(list((home / 'ordering').glob('*/requests/*/HTTP_ATTEMPT.json')))
    need(attempts <= config(home)['max_generation_http'], 'mixed_http_budget_exceeded')
    result = {'version': VERSION, 'completed_at_utc': now(), 'completed_types': completed, 'errors': errors,
              'status': 'ERROR' if errors else 'GENERATED', 'http_attempts': attempts,
              'training_allowed': False, 'automatic_retry': False}
    if errors:
        write(home / 'BATCH_GENERATION.ERROR.json', result); raise ValueError('batch_generation_incomplete')
    result['files'] = [bind(home / 'multiselect/INITIAL_GENERATION.COMPLETE.json'), bind(home / 'ordering/GENERATION.COMPLETE.json')]
    write(home / 'BATCH_GENERATION.COMPLETE.json', result); return result


def finalize_multiselect(home=HERE):
    home = Path(home).resolve(); verify_plan(home)
    load_multi('generation_driver').finalize(home / 'multiselect')
    return load_multi('compose').compose(home / 'multiselect')


def finish_asset_review(home, slot, round_id):
    home = Path(home).resolve(); value = config(home)
    need(slot in {s['slot_id'] for s in value['multiselect_config']['slots']}
         and round_id in ('initial', 'supplement'), 'unknown_asset_review_slot_or_round')
    need(not (home / 'BATCH_REVIEW.COMPLETE.json').exists()
         and not (home / 'EVALUATION_PREPARATION.STARTED.json').exists(), 'review_cannot_change_after_batch_freeze')
    return load_multi('asset_generation').finish_review(home / 'multiselect/assets' / slot, round_id)


def reviewed_pools(home):
    import ordering_review
    home = Path(home).resolve()
    multi_engine = load_multi('evaluate'); multi_engine.verify_review(home / 'multiselect')
    _, multi, _, _ = multi_engine.reviewed_content(home / 'multiselect')
    ordering_review.verify_review(home)
    ordered = read(home / 'ordering/EVALUATION_POOL.json')['questions']
    need(all(q['question_type'] == 'multi_select' for q in multi) and all(q['question_type'] == 'ordering' for q in ordered), 'lane_question_type_changed')
    return multi + ordered


def validate_review_sources(home):
    # Check both saved public judgments before any assembly/generation replay
    # can open private drafts or Gold during the review/evaluation stages.
    import ordering_review
    load_multi('review_schema').validate_final_sources(Path(home) / 'multiselect')
    ordering_review.validate_sources(home)


def verify_generation(home):
    home = Path(home).resolve(); generation = read(home / 'BATCH_GENERATION.COMPLETE.json')
    need(generation['version'] == VERSION and generation['status'] == 'GENERATED'
         and generation['completed_types'] == ['multi_select', 'ordering'] and not generation['errors']
         and generation['training_allowed'] is False and generation['automatic_retry'] is False
         and not (home / 'BATCH_GENERATION.ERROR.json').exists(), 'both_generated_types_required')
    require_bindings(generation['files'], [home / 'multiselect/INITIAL_GENERATION.COMPLETE.json', home / 'ordering/GENERATION.COMPLETE.json'])
    attempts = len(list((home / 'multiselect/assets').glob('*/rounds/*/requests/*/HTTP_ATTEMPT.json')))
    attempts += len(list((home / 'ordering').glob('*/requests/*/HTTP_ATTEMPT.json')))
    need(type(generation['http_attempts']) is int and 0 <= generation['http_attempts'] <= attempts <= config(home)['max_generation_http'],
         'mixed_http_budget_or_receipt_changed')
    need(aware(read(home / 'BATCH_PREPARATION.COMPLETE.json')['completed_at_utc']) <= aware(generation['completed_at_utc']) <= aware(now()),
         'mixed_generation_time')
    return generation


def finish_multiselect_review(home=HERE):
    home = Path(home).resolve(); config(home)
    need(not (home / 'EVALUATION_PREPARATION.STARTED.json').exists()
         and not (home / 'BATCH_REVIEW.COMPLETE.json').exists(), 'review_cannot_change_after_batch_freeze')
    import importlib.util
    import sys
    import io
    from contextlib import redirect_stdout
    path = V75 / 'execution/merge_final_review.py'
    spec = importlib.util.spec_from_file_location('_v76_frozen_multi_review_merge', path)
    module = importlib.util.module_from_spec(spec); original = sys.path[:]
    try:
        spec.loader.exec_module(module)
        with redirect_stdout(io.StringIO()):
            module.main(home / 'multiselect')
    finally:
        sys.path[:] = original
    return load_multi('evaluate').freeze_review(home / 'multiselect')


def freeze(home=HERE):
    home = Path(home).resolve(); config(home); validate_review_sources(home); verify_plan(home); verify_generation(home)
    need(not any((home / 'solver').glob('*/STARTED.json')), 'cannot_freeze_after_solver')
    questions = reviewed_pools(home)
    need(len({q['candidate_id'] for q in questions}) == len(questions) and len(questions) <= config(home)['max_candidates'], 'mixed_candidate_ids_or_budget')
    child_freezes = [home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', home / 'ordering/CONTENT_REVIEW.COMPLETE.json']
    completed = now()
    need(all(aware(read(p)['completed_at_utc']) <= aware(completed) for p in child_freezes), 'mixed_freeze_before_child_review')
    pool = {'questions': questions, 'provenance': {'requested_model': 'Qwen/Qwen3.6-35B-A3B', 'mixed_batch_version': VERSION},
            'training_allowed': False, 'formal_training_admission': False}
    write(home / 'EVALUATION_POOL.json', pool)
    paths = child_freezes + [home / 'EVALUATION_POOL.json', home / 'CONFIG.json', home / 'BATCH_PREPARATION.COMPLETE.json', home / 'BATCH_GENERATION.COMPLETE.json']
    paths += list(HERE.glob('*.py'))
    result = {'version': VERSION, 'status': 'FROZEN', 'completed_at_utc': completed,
              'by_type': dict(Counter(q['question_type'] for q in questions)), 'before_any_solver': True,
              'solver_scores_used': False, 'training_allowed': False, 'files': [bind(p) for p in sorted(set(paths))]}
    write(home / 'BATCH_REVIEW.COMPLETE.json', result); return result


def verify_frozen(home=HERE):
    home = Path(home).resolve(); config(home); validate_review_sources(home); verify_plan(home); verify_generation(home)
    frozen = read(home / 'BATCH_REVIEW.COMPLETE.json')
    need(frozen['version'] == VERSION and frozen['status'] == 'FROZEN' and frozen['before_any_solver'] is True
         and frozen['solver_scores_used'] is False and frozen['training_allowed'] is False
         and aware(frozen['completed_at_utc']) <= aware(now()), 'mixed_freeze_contract')
    paths = [home / name for name in ('EVALUATION_POOL.json', 'CONFIG.json', 'BATCH_PREPARATION.COMPLETE.json', 'BATCH_GENERATION.COMPLETE.json')]
    paths += [home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', home / 'ordering/CONTENT_REVIEW.COMPLETE.json']
    require_bindings(frozen['files'], paths + list(HERE.glob('*.py'))); questions = reviewed_pools(home)
    need(read(home / 'EVALUATION_POOL.json')['questions'] == questions
         and frozen['by_type'] == dict(Counter(q['question_type'] for q in questions))
         and len(questions) <= config(home)['max_candidates']
         and len({q['candidate_id'] for q in questions}) == len(questions), 'mixed_frozen_pool_changed')
    pool = read(home / 'EVALUATION_POOL.json')
    need(pool['training_allowed'] is False and pool['formal_training_admission'] is False
         and pool['provenance'] == {'requested_model': 'Qwen/Qwen3.6-35B-A3B', 'mixed_batch_version': VERSION}, 'mixed_pool_contract')
    for path in [home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', home / 'ordering/CONTENT_REVIEW.COMPLETE.json']:
        need(aware(read(path)['completed_at_utc']) <= aware(frozen['completed_at_utc']), 'mixed_freeze_before_child_review')
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'generate', 'verify', 'finish-asset-review', 'finalize-multiselect', 'finish-multiselect-review', 'finish-ordering-review',
                                         'freeze', 'prepare-evaluation', 'live-trained', 'live-base', 'report'])
    parser.add_argument('--home', type=Path, default=HERE)
    parser.add_argument('--slot'); parser.add_argument('--round', choices=('initial', 'supplement'), default='initial')
    args = parser.parse_args()
    config(args.home)
    if args.action == 'prepare': result = prepare(args.home)
    elif args.action == 'generate': result = generate(args.home)
    elif args.action == 'verify': result = verify_plan(args.home)
    elif args.action == 'finish-asset-review': result = finish_asset_review(args.home, args.slot, args.round)
    elif args.action == 'finalize-multiselect': result = finalize_multiselect(args.home)
    elif args.action == 'finish-multiselect-review': result = finish_multiselect_review(args.home)
    elif args.action == 'finish-ordering-review':
        import ordering_review
        result = ordering_review.finish_review(args.home)
    elif args.action == 'freeze': result = freeze(args.home)
    else:
        import mixed_evaluation
        if args.action == 'prepare-evaluation': result = mixed_evaluation.prepare(args.home)
        elif args.action.startswith('live-'): result = mixed_evaluation.live(args.home, args.action[5:])
        else: result = mixed_evaluation.report(args.home)
    import json
    print(json.dumps(result, ensure_ascii=False))
