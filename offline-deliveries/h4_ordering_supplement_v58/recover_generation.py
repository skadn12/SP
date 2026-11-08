"""Explicit v58 c01 empty-output sealing and continuation at c02; no c01 replay."""
from pathlib import Path
import argparse
import json
import traceback
import run_collection as c

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / 'GENERATION_RECOVERY.PREFLIGHT.json'
ARCHIVE = HERE / 'generation.ERROR.recovered.json'


def require(value, why):
    if not value:
        raise ValueError(why)


def inspect():
    plan = c.verify_plan()
    gen, pipeline = c.modules()
    chunk = plan['chunks'][0]
    run = Path(chunk['generation_run'])
    manifest = gen.verify(run)
    error_path = HERE / 'generation.ERROR.json'
    require(error_path.is_file() != ARCHIVE.is_file(), 'need_exactly_one_original_error_location')
    error_path = error_path if error_path.exists() else ARCHIVE
    error = c.read(error_path)
    require(error['stage'] == 'generation' and error['completed_chunks'] == []
            and error['current_operation'] == {'chunk': 1, 'action': 'generate'}, 'unexpected_original_error')
    log = HERE / 'logs' / (run.name + '_generate.log')
    require('frozen_file_changed:' + str(c.CODE / 'test_v58_ordering.py') in log.read_text(), 'unexpected_c01_process_error')
    folders = list((run / 'requests').iterdir())
    require(len(folders) == 1, 'c01_must_have_exactly_one_request')
    request = folders[0]
    payload, proof = c.read(request / 'REQUEST.json'), c.read(request / 'PROOF.json')
    attempt, result = c.read(request / 'HTTP_ATTEMPT.json'), c.read(request / 'HTTP_RESULT.json')
    response, remote_error = c.read(request / 'RESPONSE.json'), c.read(request / 'ERROR.json')
    require(proof['role'] == 'question_generator' and proof['question_type'] == 'ordering', 'c01_not_order_generator')
    require(attempt['payload_sha256'] == proof['payload_sha256'] == gen.base.digest(payload)
            and attempt['actual_payload_equals_frozen'] and attempt['transport_retries'] == 0, 'c01_actual_request_binding_failed')
    gen.verify_prompt_request(run, payload, proof)
    context = gen.initial_context(run, c.ROOT, 'ordering')
    expected, _ = gen.make_payload(run, c.ROOT, 'ordering', context)
    require(payload == expected, 'c01_actual_payload_differs_frozen_plan')
    require(response['metadata']['finish_reason'] == 'length' and response['metadata']['output_tokens'] == 4096
            and remote_error['error_type'] == 'ValueError' and remote_error['http_attempted'] is True
            and result['outcome'] == 'error', 'c01_must_be_recorded_incomplete_response')
    try:
        json.loads(response['text'])
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError('c01_response_unexpectedly_complete_json')
    outcome_path = run / 'outcomes' / (c.ROOT + '.ordering.json')
    outcome = c.read(outcome_path)
    require(outcome == {'root_id': c.ROOT, 'question_type': 'ordering', 'candidate_slots': 2,
        'mechanical_pass': 0, 'mechanical_rejected': [{'candidate_id': qid, 'errors': ['ValueError']} for qid in chunk['candidate_ids']],
        'abstentions': [], 'review_rejected': [], 'actual_generation_requests': 1, 'ready_for_solver': 0}, 'c01_outcome_mismatch')
    require(not (request / 'VALIDATION.json').exists(), 'truncated_response_must_not_have_accepted_validation')
    for later in plan['chunks'][1:]:
        require(not Path(later['generation_run']).exists(), 'later_chunk_already_exists_before_explicit_resume')
    require(not c.marker('evaluation', 'STARTED').exists(), 'solver_started_before_generation_recovery')
    prior, capacity = c.cumulative_prior_path(1), HERE / 'capacity/c01_initial.json'
    require(c.read(capacity)['capacity']['status'] == 'PASS', 'c01_capacity_not_passed')
    existing = [p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files = sorted(set(existing + [log, prior, capacity, c.marker('generation', 'STARTED'), error_path]))
    return plan, gen, run, manifest, outcome, files


def preflight():
    plan, gen, run, manifest, outcome, files = inspect()
    require(not RECEIPT.exists(), 'recovery_preflight_already_exists')
    value = {'status': 'PASS', 'at_utc': c.now(), 'script': c.binding(__file__),
        'experiment_plan': c.binding(HERE / 'EXPERIMENT_PLAN.json'), 'frozen_inputs': [c.binding(p) for p in files],
        'original_error': c.binding(HERE / 'generation.ERROR.json'), 'c01_http_calls': 1,
        'c01_mechanical_pass': 0, 'c01_ready': 0, 'c01_replay_calls': 0,
        'reason': 'Concurrent test-only append changed a frozen binding after the HTTP error. Test restored byte-identically. Seal native empty postamble then continue frozen c02-c06.',
        'remote_failure_preserved': 'c01 finish_reason=length, 4096 tokens, incomplete JSON; no extraction or salvage',
        'remaining_chunks': [r['index'] for r in plan['chunks'][1:]],
        'remaining_remote_http_max': 20, 'total_remote_http_max': 24, 'solver_calls': 0,
        'actions': ['write missing native empty MECHANICAL_POOL, SOLVER_POOL and REPORT only',
            'rename complete original generation.ERROR.json to generation.ERROR.recovered.json without byte changes',
            'continue original frozen c02-c06, retaining c01 HTTP in totals']}
    c.save(RECEIPT, value)
    print(json.dumps({'status': 'PASS', 'receipt': str(RECEIPT), 'live_calls': 0}))


def verify_receipt():
    receipt = c.read(RECEIPT)
    c.verify_binding(receipt['script']); c.verify_binding(receipt['experiment_plan'])
    for row in receipt['frozen_inputs']:
        if row['path'] == str(HERE / 'generation.ERROR.json') and ARCHIVE.exists():
            require(c.binding(ARCHIVE)['sha256'] == row['sha256'], 'archived_error_changed')
        else:
            c.verify_binding(row)
    return receipt


def seal():
    receipt = verify_receipt()
    plan, gen, run, manifest, outcome, files = inspect()
    require(not (HERE / 'GENERATION_RECOVERY.SEALED.json').exists(), 'c01_already_sealed')
    for name in ('MECHANICAL_POOL.json', 'SOLVER_POOL.json', 'REPORT.json'):
        require(not (run / name).exists(), 'refuse_overwrite_c01_postamble:' + name)
    c.save(HERE / 'GENERATION_RECOVERY.STARTED.json', {'started_at_utc': c.now(), 'preflight': c.binding(RECEIPT),
        'original_error': receipt['original_error'], 'no_c01_replay': True, 'script': c.binding(__file__)})
    mechanical, ready = [], []
    c.save(run / 'MECHANICAL_POOL.json', {'questions': mechanical})
    c.save(run / 'SOLVER_POOL.json', {'version': gen.VERSION, 'candidate_count': 0, 'questions': ready,
        'provenance': {'requested_model': gen.MODEL, 'returned_model': gen.MODEL, 'codex_question_edits': False,
            'multi_select_semantic_review': False, 'ordering_review_owner': gen.MODEL,
            'source_mode': 'one_full_provisional_candidate_context_per_root',
            'source_context_version': gen.VERSION, 'feedback_target_solver': gen.TARGET,
            'prompt_arm': manifest['prompt_arm'], 'comparison_id': manifest['comparison_id'],
            'comparison_contract': gen.COMPARISON_CONTRACT},
        'ground_calls': 0, 'training_started': False, 'source_training_pool_eligible': False,
        'upstream_scene_promotion': False})
    report = {'version': gen.VERSION, 'http_attempts': 1, 'requests_prepared': 1,
        'candidate_slots': manifest['budget']['candidate_slots'], 'mechanical_pass': 0, 'ready_for_solver': 0, 'roots': [outcome],
        'mechanical_answer_distribution': gen.display_export.multi_answer_distribution(mechanical),
        'ready_answer_distribution': gen.display_export.multi_answer_distribution(ready),
        'feedback_in_every_request': True, 'semantic_novelty_gate': False, 'paraphrases_allowed': True,
        'prompt_arm': manifest['prompt_arm'], 'comparison_id': manifest['comparison_id'],
        'comparison_contract': gen.COMPARISON_CONTRACT,
        'no_current_solver_feedback_in_generation': True, 'ground_calls': 0, 'training_calls': 0}
    c.save(run / 'REPORT.json', report)
    gen.verify(run)
    _, pipeline = c.modules(); pipeline.verify_generation(run)
    (HERE / 'generation.ERROR.json').rename(ARCHIVE)
    require(c.binding(ARCHIVE)['sha256'] == receipt['original_error']['sha256'], 'archived_error_not_byte_identical')
    verify_receipt()
    c.save(HERE / 'GENERATION_RECOVERY.SEALED.json', {'sealed_at_utc': c.now(), 'remote_calls': 0,
        'original_error_archive': c.binding(ARCHIVE), 'original_error_original_path': receipt['original_error']['path'],
        'empty_outputs': [c.binding(run / name) for name in ('MECHANICAL_POOL.json', 'SOLVER_POOL.json', 'REPORT.json')],
        'question_salvage': False, 'c01_repeated_inference': False})
    print(json.dumps({'status': 'PASS', 'c01_ready': 0, 'remote_calls': 0}))


def summary(chunk, *, log, prior):
    run = Path(chunk['generation_run']); result = c.read(run / 'REPORT.json')
    capacity = HERE / 'capacity' / f"c{chunk['index']:02d}_initial.json"
    return {'index': chunk['index'], 'generation_run': str(run), 'public_prior_rows': len(c.read(prior)['questions']),
        'initial_required_context_tokens': c.read(capacity)['capacity']['required_context_tokens'],
        **{key: result[key] for key in ('candidate_slots', 'http_attempts', 'mechanical_pass', 'ready_for_solver')},
        'bindings': [c.binding(run / 'MANIFEST.json'), c.binding(run / 'REPORT.json'),
            c.binding(run / 'MECHANICAL_POOL.json'), c.binding(run / 'SOLVER_POOL.json'),
            c.binding(prior), c.binding(capacity), log]}


def live():
    receipt = verify_receipt(); plan = c.verify_plan()
    require((HERE / 'GENERATION_RECOVERY.SEALED.json').exists(), 'seal_required')
    require(not (HERE / 'GENERATION_RECOVERY.CONTINUATION_STARTED.json').exists(), 'no_automatic_replay')
    for row in c.read(HERE / 'GENERATION_RECOVERY.SEALED.json')['empty_outputs']:
        c.verify_binding(row)
    require(not c.marker('generation', 'ERROR').exists() and not c.marker('generation', 'COMPLETE').exists(), 'original_generation_state_invalid')
    c.save(HERE / 'GENERATION_RECOVERY.CONTINUATION_STARTED.json', {'started_at_utc': c.now(), 'script': c.binding(__file__),
        'preflight': c.binding(RECEIPT), 'sealed': c.binding(HERE / 'GENERATION_RECOVERY.SEALED.json')})
    gen, _ = c.modules()
    first = plan['chunks'][0]; first_run = Path(first['generation_run'])
    rows = [summary(first, log=c.binding(HERE / 'logs' / (first_run.name + '_generate.log')), prior=c.cumulative_prior_path(1))]
    cumulative = [Path(plan['historical_prior']['path']), first_run / 'MECHANICAL_POOL.json']
    current = None
    try:
        for chunk in plan['chunks'][1:]:
            c.verify_plan(); verify_receipt()
            run = Path(chunk['generation_run']); current = {'chunk': chunk['index'], 'action': 'prepare'}
            prior = c.cumulative_prior_path(chunk['index'])
            c.project_prior(cumulative, prior)
            gen.prepare(run, c.HISTORY, roots=[c.ROOT], slots=c.SLOTS, prior_pools=[prior], prompt_arm='current',
                comparison_id=chunk['comparison_id'], slot_focus_map=chunk['slot_focus_by_candidate_id'])
            c.capacity_check(gen, run, HERE / 'capacity' / f"c{chunk['index']:02d}_initial.json")
            current['action'] = 'generate'
            log = c.command('generate.py', 'generate', run)
            row = summary(chunk, log=log, prior=prior); rows.append(row)
            require(row['http_attempts'] <= 4 and sum(r['http_attempts'] for r in rows) <= c.MAX_HTTP, 'recovery_budget_exceeded')
            cumulative.append(run / 'MECHANICAL_POOL.json')
            print(json.dumps({'event': 'recovered_generation_chunk_complete', **row}), flush=True)
        recovery = {'completed_at_utc': c.now(), 'preflight': c.binding(RECEIPT), 'script': c.binding(__file__),
            'original_error_archive': c.binding(ARCHIVE), 'sealed': c.binding(HERE / 'GENERATION_RECOVERY.SEALED.json'),
            'repeated_http_calls': 0, 'c01_http_calls_reused': 1, 'continued_chunks': [2,3,4,5,6],
            'total_http_calls': sum(r['http_attempts'] for r in rows), 'solver_calls': 0}
        c.save(HERE / 'GENERATION_RECOVERY.COMPLETE.json', recovery)
        c.save(c.marker('generation', 'COMPLETE'), {'version': c.VERSION, 'stage': 'generation',
            'completed_at_utc': c.now(), 'experiment_plan': c.binding(HERE / 'EXPERIMENT_PLAN.json'),
            'chunks': rows, 'http_attempts': sum(r['http_attempts'] for r in rows), 'solver_calls': 0,
            'candidate_slots': 12, 'mechanical_pass': sum(r['mechanical_pass'] for r in rows),
            'ready_for_solver': sum(r['ready_for_solver'] for r in rows),
            'explicit_recovery': c.binding(HERE / 'GENERATION_RECOVERY.COMPLETE.json'),
            'ordinary_uninterrupted_execution': False})
        c.completion('generation')
    except BaseException as error:
        c.save(HERE / 'GENERATION_RECOVERY.ERROR.json', {'failed_at_utc': c.now(), 'error': repr(error),
            'traceback': traceback.format_exc(), 'completed_chunks': rows, 'current_operation': current,
            'no_automatic_replay': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'seal', 'live'))
    globals()[parser.parse_args().action]()
