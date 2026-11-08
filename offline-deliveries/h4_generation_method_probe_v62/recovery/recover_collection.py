"""Additive v62 recovery. close is local; generation runs five untouched chunks."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
sys.path.insert(0, str(REPORT))
import run_collection as c
VERSION = 'v62_interrupted_first_chunk_recovery_v1'
LOST = 'root194259_v62_direct_c01_generation'
MISSING = ('RESPONSE.json', 'HTTP_RESULT.json', 'ERROR.json', 'VALIDATION.json')

def require(value, message):
    if not value: raise ValueError(message)

def write(name, value):
    c.save(HERE / name, value)

def live_processes():
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(path.parent.name)
            if pid == os.getpid(): continue
            args = path.read_bytes().decode(errors='replace').split(chr(0))
            if not args or not any('python' in Path(x).name for x in args[:1]): continue
            gen = any(x == str(c.CODE / 'generate.py') for x in args)
            orch = any(Path(x).name == 'run_collection.py' for x in args) and 'generation' in args
            if gen or orch: matches.append({'pid': pid, 'args': args})
        except (OSError, ValueError): pass
    return matches

def interrupted_identity(plan):
    first = plan['chunks'][0]
    require(first['chunk_name'] == 'direct_c01' and first['arm'] == 'direct', 'first_chunk_changed')
    run = Path(first['generation_run'])
    require(run.name == LOST, 'unexpected_interrupted_run')
    requests = sorted((run / 'requests').glob('*/REQUEST.json'))
    require(len(requests) == 1, 'interrupted_request_count_changed')
    folder = requests[0].parent
    require((folder / 'HTTP_ATTEMPT.json').is_file(), 'no_interrupted_HTTP_attempt')
    for name in MISSING: require(not (folder / name).exists(), 'interrupted_request_has_observed_result:' + name)
    require(len(list((run / 'requests').glob('*/HTTP_ATTEMPT.json'))) == 1, 'interrupted_attempt_replayed')
    gen, _ = c.modules()
    gen.verify(run)
    gen.verify_prompt_request(run, c.read(folder / 'REQUEST.json'), c.read(folder / 'PROOF.json'))
    return first, run, folder

def close():
    require(not (HERE / 'RECOVERY_PLAN.json').exists(), 'recovery_already_prepared_no_replay')
    plan = c.verify_plan()
    require((REPORT / 'generation.STARTED.json').is_file(), 'original_generation_not_started')
    require(not (REPORT / 'generation.COMPLETE.json').exists(), 'original_generation_already_complete')
    require(not (REPORT / 'generation.ERROR.json').exists(), 'unexpected_original_generation_error')
    first, run, folder = interrupted_identity(plan)
    require(not live_processes(), 'live_generation_process_present')
    require(all(not Path(ch['generation_run']).exists() for ch in plan['chunks'][1:]), 'remaining_chunk_not_untouched')
    require(all(not Path(ch['evaluation_run']).exists() for ch in plan['chunks']), 'evaluation_already_started')
    require(all(not (run / n).exists() for n in ('REPORT.json', 'SOLVER_POOL.json', 'MECHANICAL_POOL.json')), 'closure_outputs_already_exist')
    decision = Path('/tmp/v62_recovery_parent_decision.txt')
    require(decision.is_file() and decision.read_text().startswith('APPROVED'), 'parent_recovery_design_approval_missing')
    snapshot = [c.binding(p) for home in (REPORT, c.CODE) for p in sorted(home.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(HERE)]
    write('ORIGINAL_BINDINGS.json', {'scope': 'all_existing_v62_artifacts_before_additive_recovery', 'files': snapshot})
    with (HERE / 'PARENT_DECISION.txt').open('x') as stream: stream.write(decision.read_text())
    observation = {'version': VERSION, 'observed_at_utc': c.now(), 'observed_at_epoch': time.time(),
        'observed_at_monotonic': time.monotonic(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'observation_type': 'LOCAL_RECOVERY_OBSERVATION', 'reason': 'parent_reports_deliberate_user_interruption',
        'local_matching_processes': [], 'request': c.binding(folder / 'REQUEST.json'),
        'http_attempt': c.binding(folder / 'HTTP_ATTEMPT.json'), 'throttle': c.binding(folder / 'THROTTLE.json'),
        'http_attempted': True, 'response_observed': False, 'remote_outcome': 'unknown',
        'remote_completion_at_utc': None, 'remote_error_observed': False, 'token_usage': 'unknown',
        'replayed_http_calls': 0, 'replacement_slots': 0, 'lost_candidate_ids': first['candidate_ids'],
        'no_synthetic_response_result_error_or_validation': True}
    write('INTERRUPTED_ATTEMPT.json', observation)
    receipt = c.binding(HERE / 'INTERRUPTED_ATTEMPT.json')
    gen, pipeline = c.modules()
    manifest = gen.verify(run)
    common = {'execution_recovery': receipt, 'execution_status': 'INTERRUPTED_NO_SAVED_RESPONSE_CLOSED_WITH_EMPTY_POOL',
              'remote_outcome': 'unknown', 'local_observed_candidates': 0,
              'returned_model_observed': False, 'unresolved_interrupted_slots': 2}
    outcome = {'root_id': c.ROOT, 'question_type': 'multi_select', 'candidate_slots': 2,
        'mechanical_pass': 0, 'ready_for_solver': 0, 'mechanical_rejected': [], 'abstentions': [], 'review_rejected': [],
        'interrupted_slots': [{'candidate_id': qid, 'reason': 'InterruptedNoSavedResponse',
             'failure_class': 'interrupted_unobserved', 'mechanical_validation_performed': False,
             'observed_draft': False} for qid in first['candidate_ids']],
        'actual_generation_requests': 1, 'generation_method': 'direct', 'evidence_plan_requests': 0, **common}
    c.save(run / 'outcomes' / (c.ROOT + '.multi_select.json'), outcome)
    c.save(run / 'MECHANICAL_POOL.json', {'questions': [], **common})
    c.save(run / 'SOLVER_POOL.json', {'version': gen.VERSION, 'candidate_count': 0, 'questions': [],
        'provenance': {'requested_model': gen.MODEL, 'returned_model': None, 'returned_model_observed': False,
            'codex_question_edits': False, 'multi_select_semantic_review': False, 'ordering_review_owner': None,
            'ordering_review_active': False, 'source_mode': 'one_full_provisional_candidate_context_per_root',
            'source_context_version': gen.VERSION, 'feedback_target_solver': gen.TARGET,
            'prompt_arm': manifest['prompt_arm'], 'comparison_id': manifest['comparison_id'],
            'generation_method': 'direct', 'method_contract': gen.METHOD_CONTRACT,
            'comparison_contract': gen.COMPARISON_CONTRACT},
        'ground_calls': 0, 'training_started': False, 'source_training_pool_eligible': False,
        'upstream_scene_promotion': False, **common})
    distribution = gen.display_export.multi_answer_distribution([])
    report = {'version': gen.VERSION, 'http_attempts': 1, 'requests_prepared': 1, 'candidate_slots': 2,
        'mechanical_pass': 0, 'ready_for_solver': 0, 'roots': [outcome],
        'mechanical_answer_distribution': distribution, 'ready_answer_distribution': distribution,
        'feedback_in_every_request': True, 'semantic_novelty_gate': False, 'paraphrases_allowed': True,
        'prompt_arm': manifest['prompt_arm'], 'comparison_id': manifest['comparison_id'],
        'generation_method': 'direct', 'method_contract': gen.METHOD_CONTRACT,
        'actual_generation_requests': 1, 'evidence_plan_requests': 0, 'evidence_plan_status': 'NOT_APPLICABLE',
        'request_costs': gen.request_costs(run), 'comparison_contract': gen.COMPARISON_CONTRACT,
        'no_current_solver_feedback_in_generation': True, 'ground_calls': 0, 'training_calls': 0,
        'saved_generation_responses': 0, 'unobserved_response_requests': 1,
        'remote_error_requests': 0, **common}
    c.save(run / 'REPORT.json', report)
    pipeline.verify_generation(run)
    frozen = {'version': VERSION, 'prepared_at_utc': c.now(), 'experiment_plan': c.binding(REPORT / 'EXPERIMENT_PLAN.json'),
        'original_bindings': c.binding(HERE / 'ORIGINAL_BINDINGS.json'), 'interrupted_attempt': receipt,
        'parent_decision': c.binding(HERE / 'PARENT_DECISION.txt'), 'runner': c.binding(__file__),
        'tests': c.binding(HERE / 'test_recovery.py'),
        'closure_outputs': [c.binding(run / n) for n in ('REPORT.json', 'SOLVER_POOL.json', 'MECHANICAL_POOL.json', 'outcomes/' + c.ROOT + '.multi_select.json')],
        'remaining_chunks': deepcopy(plan['chunks'][1:]), 'total_HTTP_budget_original': 15,
        'interrupted_HTTP_attempts': 1, 'remaining_HTTP_max': sum(x['http_max'] for x in plan['chunks'][1:]),
        'maximum_total_HTTP_attempts': 1 + sum(x['http_max'] for x in plan['chunks'][1:]),
        'replayed_requests': 0, 'replacement_slots': 0, 'training_allowed': False,
        'remote_interrupted_completion_gap_verifiable': False, 'local_resume_cooldown_seconds': 35,
        'ordinary_transport_completion_marker_unchanged_by_recovery': True}
    write('RECOVERY_PLAN.json', frozen)
    verify()
    print(json.dumps({'status': 'CLOSED_LOCALLY', 'lost_slots': 2, 'HTTP_attempts': 1, 'remote_outcome': 'unknown', 'new_remote_calls': 0}))

def verify():
    plan = c.verify_plan()
    recovery = c.read(HERE / 'RECOVERY_PLAN.json')
    require(recovery['version'] == VERSION, 'recovery_version_changed')
    for key in ('experiment_plan', 'original_bindings', 'interrupted_attempt', 'parent_decision', 'runner', 'tests'): c.verify_binding(recovery[key])
    for row in c.read(HERE / 'ORIGINAL_BINDINGS.json')['files'] + recovery['closure_outputs']: c.verify_binding(row)
    first, run, folder = interrupted_identity(plan)
    require(recovery['remaining_chunks'] == plan['chunks'][1:], 'remaining_order_changed')
    require(recovery['maximum_total_HTTP_attempts'] == 14 <= plan['remote_http_max'], 'recovery_budget_changed')
    _, pipeline = c.modules()
    require(pipeline.verify_generation(run)[1] == [], 'interrupted_chunk_has_candidates')
    return plan, recovery

def chunk_summary(chunk, capacity, prior, log):
    run = Path(chunk['generation_run'])
    result = c.read(run / 'REPORT.json')
    cap = c.read(capacity)['capacity']
    value = {'index': chunk['index'], 'arm': chunk['arm'], 'chunk_name': chunk['chunk_name'],
        'generation_run': str(run), 'public_prior_rows': len(c.read(prior)['questions']),
        'initial_required_context_tokens': cap['required_context_tokens'],
        **{k: result[k] for k in ('candidate_slots', 'http_attempts', 'mechanical_pass', 'ready_for_solver')},
        'bindings': [c.binding(run / 'MANIFEST.json'), c.binding(run / 'REPORT.json'),
             c.binding(run / 'MECHANICAL_POOL.json'), c.binding(run / 'SOLVER_POOL.json'),
             c.binding(prior), c.binding(capacity), c.binding(log)]}
    if chunk['chunk_name'] == 'direct_c01':
        value.update(interrupted_HTTP_attempts=1, unresolved_interrupted_slots=2,
                     remote_outcome='unknown', closure=c.binding(HERE / 'INTERRUPTED_ATTEMPT.json'))
    return value

def generation():
    plan, recovery = verify()
    require(not live_processes(), 'live_generation_process_present')
    require(not (HERE / 'generation.STARTED.json').exists(), 'recovery_generation_already_started_no_replay')
    require(not (REPORT / 'generation.COMPLETE.json').exists(), 'generation_already_complete')
    require(all(not Path(ch['generation_run']).exists() for ch in recovery['remaining_chunks']), 'remaining_chunk_not_untouched')
    write('generation.STARTED.json', {'version': VERSION, 'started_at_utc': c.now(),
          'recovery_plan': c.binding(HERE / 'RECOVERY_PLAN.json'), 'automatic_replay': False})
    observed = c.read(HERE / 'INTERRUPTED_ATTEMPT.json')
    current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    elapsed = time.monotonic() - observed['observed_at_monotonic'] if observed['boot_id'] == current_boot else 0
    waited = max(0, 35 - elapsed)
    if waited: time.sleep(waited)
    write('LOCAL_RESUME_COOLDOWN.json', {'completed_at_utc': c.now(), 'waited_seconds': waited,
        'basis': 'local_recovery_observation_only', 'observed_interval_seconds': time.monotonic() - observed['observed_at_monotonic'] if observed['boot_id'] == current_boot else None,
        'remote_interrupted_completion_time': None, 'remote_completion_gap_verified': False,
        'ordinary_transport_marker_modified': False})
    gen, _ = c.modules()
    prior = Path(plan['historical_prior']['path'])
    first = plan['chunks'][0]
    first_log = REPORT / 'logs' / (Path(first['generation_run']).name + '_generate.log')
    summaries = [chunk_summary(first, REPORT / 'capacity/c01_initial.json', prior, first_log)]
    total_http = 1
    current = None
    try:
        for chunk in recovery['remaining_chunks']:
            verify()
            current = {'chunk': chunk['chunk_name'], 'action': 'prepare_untouched_frozen_chunk'}
            run = Path(chunk['generation_run'])
            gen.prepare(run, c.HISTORY, roots=[c.ROOT], slots=c.SLOTS, prior_pools=[prior],
                prompt_arm='current', comparison_id=chunk['comparison_id'],
                slot_focus_map=chunk['slot_focus_by_candidate_id'], generation_method=chunk['arm'])
            capacity = REPORT / 'capacity' / f"c{chunk['index']:02d}_initial.json"
            c.capacity_check(gen, run, capacity)
            current['action'] = 'generate'
            log = c.command('generate.py', 'generate', run)
            result = c.read(run / 'REPORT.json')
            total_http += result['http_attempts']
            require(result['candidate_slots'] == 2 and result['http_attempts'] <= chunk['http_max'] and total_http <= 14, 'generation_budget_exceeded')
            summary = chunk_summary(chunk, capacity, prior, Path(log['path']))
            summaries.append(summary)
            write('receipts/' + chunk['chunk_name'] + '.json', summary)
            print(json.dumps({'event': 'recovery_chunk_complete', 'chunk': chunk['chunk_name'], 'http_attempts': result['http_attempts'], 'ready': result['ready_for_solver']}, ensure_ascii=False), flush=True)
        complete = {'version': c.VERSION, 'stage': 'generation', 'completed_at_utc': c.now(),
            'experiment_plan': c.binding(REPORT / 'EXPERIMENT_PLAN.json'), 'chunks': summaries,
            'http_attempts': total_http, 'solver_calls': 0, 'candidate_slots': 12,
            'mechanical_pass': sum(x['mechanical_pass'] for x in summaries),
            'ready_for_solver': sum(x['ready_for_solver'] for x in summaries),
            'execution_recovery': c.binding(HERE / 'RECOVERY_PLAN.json'),
            'interrupted_HTTP_attempts': 1, 'unresolved_interrupted_slots': 2,
            'completed_at_meaning': 'local_collection_accounting_complete_not_remote_interrupted_request_completion',
            'remote_interrupted_outcome': 'unknown', 'replayed_requests': 0, 'replacement_slots': 0}
        c.save(REPORT / 'generation.COMPLETE.json', complete)
        write('generation.COMPLETE.json', {'version': VERSION, 'completed_at_utc': c.now(),
            'collection_completion': c.binding(REPORT / 'generation.COMPLETE.json'), 'remote_HTTP_attempts_total': total_http,
            'old_request_replayed': False, 'training_allowed': False})
        c.completion('generation')
        print(json.dumps({'event': 'recovered_generation_complete', 'HTTP_attempts': total_http, 'lost_slots': 2}), flush=True)
    except BaseException as error:
        write('generation.ERROR.json', {'version': VERSION, 'recorded_at_utc': c.now(), 'error': repr(error),
            'traceback': traceback.format_exc(), 'current': current, 'completed_chunks': summaries,
            'automatic_replay': False, 'original_artifacts_preserved': True})
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('close', 'verify', 'generation'))
    action = parser.parse_args().action
    if action == 'verify':
        plan, recovery = verify()
        print(json.dumps({'status': 'PASS', 'remaining_chunks': len(recovery['remaining_chunks']), 'no_remote_calls': True}))
    else: {'close': close, 'generation': generation}[action]()
