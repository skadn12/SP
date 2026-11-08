"""Read-only audit of v55's explicit capacity supplement; no model calls."""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_supplement as supplement

spec = importlib.util.spec_from_file_location('supplement_collection_provenance', HERE.parent / 'audit_provenance.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.gen = supplement.gen
audit.pipeline = supplement.pipeline


def read(path):
    return json.loads(Path(path).read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def latest_generation():
    paths = [HERE.parent / 'generation.COMPLETE.json', HERE / 'generation.COMPLETE.json']
    rows = [{'path': str(path), 'completed_at_utc': read(path)['completed_at_utc']} for path in paths]
    latest = max(rows, key=lambda row: audit.stamp(row['completed_at_utc']))
    return rows, audit.stamp(latest['completed_at_utc']), latest['completed_at_utc']


def evaluation_audit_alternate(recovery):
    """Replay completed inference and inspect the separate empty output."""
    supplement.pipeline.verify(supplement.EVAL)
    _, questions, _, inspector = supplement.pipeline.verify_generation(supplement.RUN)
    solver = supplement.EVAL / 'solver'
    manifest, records, _ = inspector.solver(solver)
    results = read(solver / 'RESULTS.json')
    selected = supplement.pipeline.verify_scores(questions, manifest, records, results)
    require(selected == [] and len(records) == 18 and len(questions) == 1,
            'alternate_finalization_score_replay_changed')
    by_id = {q['candidate_id']: q for q in questions}
    gold = read(solver / 'gold.json')
    canonical = {q['candidate_id']: q for q in read(solver / 'canonical_bindings.json')['candidates']}
    for request in manifest['requests']:
        question = by_id[request['candidate_id']]
        require(gold[question['candidate_id']] ==
                [canonical[question['candidate_id']]['content_id_by_label'][label] for label in question['gold']],
                'alternate_solver_gold_content_changed')
        prompt = solver / 'prompts' / (request['request_id'] + '.txt')
        require(audit.sha(prompt.read_bytes()) == request['prompt_sha256'], 'alternate_solver_prompt_hash_changed')
        source = read(question['source_path'])
        dialogue = '\n'.join('[' + turn['turn_ref'] + '] ' + turn['speaker'] + ': ' + turn['text']
                             for turn in source['turns'])
        require('Dialogue:\n' + dialogue in prompt.read_text(), 'alternate_solver_full_dialogue_changed')
        variant = request['variant']
        require(request['mode'] == 'full_context' and set(variant['content_to_text'].values()) == set(question['options'].values()),
                'alternate_solver_mode_or_statements_changed')
        require(all(variant['options'][label] == variant['content_to_text'][cid]
                    for label, cid in variant['display_to_content'].items()), 'alternate_solver_display_mapping_changed')
    pool = Path(recovery['alternate_pool_directory'])
    require(pool == supplement.EVAL / 'alternate_finalization_empty_pool', 'unexpected_alternate_pool_directory')
    report = read(pool / 'REPORT.json')
    require(report['new_unique_added'] == 0 and report['full_solver_calls'] == 18
            and read(pool / 'SELECTED_QUESTIONS.json')['questions'] == [], 'alternate_empty_report_changed')
    require((pool / 'TRAINING_POOL.jsonl').read_text() == ''
            and (pool / 'DISPLAY_CANDIDATES.jsonl').read_text() == '', 'alternate_pool_not_empty')
    require((HERE / 'evaluation.ERROR.json').is_file() and not (HERE / 'evaluation.COMPLETE.json').exists(),
            'stock_evaluation_failure_not_preserved')
    return {'status': 'PASS', 'full_solver_calls': len(records), 'selected': 0,
            'solver_score_replay': True, 'full_dialogue_and_gold_content_mappings_preserved': True,
            'selected_display_permutations': 'vacuously_true_empty', 'stock_evaluation_error_preserved': True,
            'alternate_finalization_verified': True}


def audit_run(phase):
    manifest = supplement.verify()
    complete = read(HERE / 'generation.COMPLETE.json')
    supplement.verify_bindings(complete['files'])
    row = {'root_id': supplement.ROOT, 'chunk': 'c05_capacity_supplement',
           'generation_run': str(supplement.RUN), 'evaluation_run': str(supplement.EVAL),
           'history': supplement.binding(supplement.BASE / 'h4_sequential_ab_v51/runs/root135466_a3b3_01/CANDIDATE_HISTORY.json'),
           'prior': supplement.binding(HERE.parent / 'priors/cc-train-episode-135466_v51_v52b_v53_v54_public.json')}
    result = {'status': 'PASS', 'phase': phase, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'audit_script': supplement.binding(__file__), 'manifest': supplement.binding(HERE / 'MANIFEST.json'),
              'same_unattempted_candidate_ids': manifest['same_unattempted_candidate_ids'],
              'supplement_slot_count': 2, 'additional_distinct_slots': 0,
              'collection_distinct_planned_slots': 40, 'actual_output_limit': 4096,
              'question_rules_changed': False, 'original_runs_unchanged': True,
              'frozen_old_file_count': len(read(HERE / 'ORIGINAL_FILES_BEFORE.json')['files']),
              'models_called_by_audit': 0, 'semantic_correctness_certified': False,
              'generation': audit.generation_audit({'roots': [row]})}
    require(result['generation']['counts']['candidate_slots'] == 2
            and result['generation']['counts']['http_requests'] <= 2, 'supplement_generation_budget_changed')
    initial = read(HERE / 'INITIAL_REQUEST.json')
    actual_first = read(supplement.RUN / 'requests/001-135466-merged-multi-generate-1/REQUEST.json')
    require(actual_first == initial['payload'], 'supplement_actual_initial_request_not_frozen_preflight')
    require(not (supplement.BLOCKED_REQUEST / 'HTTP_ATTEMPT.json').exists(), 'originalblockedrequestlaterattempted')
    question_ids = {q['candidate_id'] for q in read(supplement.RUN / 'MECHANICAL_POOL.json')['questions']}
    require(question_ids <= set(manifest['same_unattempted_candidate_ids']), 'supplementaddednewslots')
    if phase == 'evaluation':
        recovery = read(HERE / 'FINALIZATION_RECOVERY.COMPLETE.json')
        require(recovery['status'] == 'COMPLETE'
                and recovery['recovery_kind'] == 'offline_export_only'
                and recovery['new_solver_calls'] == recovery['new_http_calls'] == 0
                and recovery['original_failure_preserved'] is True,
                'alternate_finalization_marker_invalid')
        supplement.verify_bindings(recovery['inputs'])
        supplement.verify_bindings(recovery['outputs'])
        result['evaluation'] = evaluation_audit_alternate(recovery)
        generations, latest_epoch, latest_utc = latest_generation()
        checked = []
        starts = list(supplement.ORIGINAL.glob('runs/root*_collect20_01_c*_evaluation/solver/STARTED.json'))
        starts += list(supplement.EVAL.glob('solver/STARTED.json'))
        starts += list((HERE.parent / 'recovery').glob('solver/STARTED.json'))
        for path in starts:
            started = read(path)['started_at_utc']
            require(audit.stamp(started) > latest_epoch, 'solver_started_before_original_and_supplement_generation_done')
            checked.append({'path': str(path), 'started_at_utc': started})
        require(result['evaluation']['full_solver_calls'] <= 36, 'supplement_solver_budget_exceeded')
        result['alternate_finalization'] = {'marker': supplement.binding(HERE / 'FINALIZATION_RECOVERY.COMPLETE.json'),
            'status': 'COMPLETE', 'selected': 0, 'evaluated': 1,
            'full_solver_calls': result['evaluation']['full_solver_calls'],
            'stock_evaluation_error_preserved': True,
            'alternate_pool': recovery['alternate_pool_directory']}
        result['all_solver_timing'] = {'generation_completions': generations,
            'latest_generation_complete_at_utc': latest_utc,
            'solver_start_files_verified': checked,
            'all_observed_collection_supplement_and_recovery_solver_starts_after_both_generation_runs': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['generation', 'evaluation'], required=True)
    args = parser.parse_args()
    result = audit_run(args.phase)
    name = 'PROVENANCE_AUDIT' if args.phase == 'evaluation' else 'PROVENANCE_GENERATION_AUDIT'
    (HERE / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (HERE / (name + '.md')).write_text('\n'.join([
        '# v55 capacity supplement provenance audit', '',
        'Status: PASS. No model calls or original-input mutations by this audit.', '',
        '- Two original c05 MC slots were blocked before any HTTP request and retain their exact candidate IDs and soft focuses.',
        '- Only actual output cap and execution scope changed:4096 output tokens; two MC slots, at most2 HTTP and36 Solver calls.',
        '- Full source, frozen previous-round feedback, all old public priors, and final c05 public inventory are preserved.',
        '- All original file bindings, payloads, source/feedback receipts, public priors, and35-second call gaps replayed.',
        '- This restores2 blocked slots among40 distinct planned collection slots; it does not add2 new distinct slots.',
        '', 'Generation counts: ' + json.dumps(result['generation']['counts']),
        'Evaluation: ' + json.dumps(result.get('evaluation', 'not yet audited')),
        '', 'These provenance checks do not certify semantic correctness.'
    ]) + '\n')
    print(json.dumps({'status': 'PASS', 'phase': args.phase, 'counts': result['generation']['counts'],
                      'evaluation': result.get('evaluation'), 'report': str(HERE / (name + '.json'))}, ensure_ascii=False))
