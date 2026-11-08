"""Bounded live validation of v53: two frozen roots, four slots per root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
BASE = REPORT.parent
CODE = BASE / 'h4_sequential_ab_v53_qa'
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
sys.path.insert(0, str(CODE))
import generate as gen

ROOTS = ('135466', '210046')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def generation_run(root):
    return CODE / 'runs' / f'root{root}_rules4_01_generation'

def evaluation_run(root):
    return CODE / 'runs' / f'root{root}_rules4_01_evaluation'

def prepare():
    assert not (REPORT / 'EXPERIMENT_PLAN.json').exists()
    profile, factory, annotate, capacity, dev = gen.base.transport()
    tokenizer = dev.tokenizer_from(str(gen.base.TOKENIZER))
    rows, checks = [], []
    for short in ROOTS:
        root = 'cc-train-episode-' + short
        history = BASE / f'h4_sequential_ab_v51/runs/root{short}_a3b3_01/CANDIDATE_HISTORY.json'
        prior_sources = sorted((BASE / 'h4_sequential_ab_v51_qa/runs').glob(f'root{short}_*_generation/MECHANICAL_POOL.json'))
        prior_sources += sorted((BASE / 'h4_sequential_ab_v52b_qa/runs').glob(f'root{short}_*_generation/MECHANICAL_POOL.json'))
        assert len(prior_sources) == 8
        questions, seen, original_count = [], set(), 0
        for path in prior_sources:
            for q in read(path)['questions']:
                assert q['root_id'] == root
                original_count += 1
                field = 'options' if q['question_type'] == 'multi_select' else 'items'
                key = (q['question_type'], tuple(sorted(' '.join(x.split()).casefold() for x in q[field].values())))
                if key in seen:
                    continue
                seen.add(key)
                questions.append({k: q[k] for k in ('candidate_id', 'root_id', 'question_type', 'question', field)})
        prior = REPORT / 'priors' / (root + '_v51_v52b_public.json')
        save(prior, {'questions': questions, 'source_files': [binding(p) for p in prior_sources],
                     'original_rows': original_count, 'unique_whole_sets': len(questions),
                     'projection': 'public_only_no_gold_no_evidence_no_solver; exact duplicate whole sets collapsed'})
        comparison = f'v53_root{short}_rules4_01'
        scenes = (['CANDIDATE:A:S02', 'CANDIDATE:B:S03', 'CANDIDATE:B:S02', 'CANDIDATE:A:S03']
                  if short == '135466' else
                  ['CANDIDATE:A:S01', 'CANDIDATE:B:S02', 'CANDIDATE:B:S03', 'CANDIDATE:A:S01'])
        hints = [gen.slot_focus.NEUTRAL_FOCI[i] for i in (2, 3, 10, 11)]
        ids = sum([gen.slot_focus.candidate_ids(root, comparison, kind, 2) for kind in gen.TYPES], [])
        focuses = {qid: {'preferred_scene_refs': [scene], 'focus': hint,
                         'requirement': gen.slot_focus.REQUIREMENT}
                   for qid, scene, hint in zip(ids, scenes, hints)}
        run = generation_run(short)
        gen.prepare(run, history, roots=[root], slots=2, prior_pools=[prior],
                    prompt_arm='current', comparison_id=comparison, slot_focus_map=focuses)
        for kind in gen.TYPES:
            payload, receipt = gen.make_payload(run, root, kind, gen.initial_context(run, root, kind))
            result = annotate(capacity(read(run / 'provider_profile.json'), tokenizer,
                                       payload['messages'], payload['max_tokens']))
            snapshot = REPORT / 'initial' / short / (kind + '.json')
            save(snapshot, {'payload': payload, 'receipt': receipt, 'capacity': result})
            checks.append({'root_id': root, 'question_type': kind, **result, 'snapshot': binding(snapshot)})
        rows.append({'root_id': root, 'history': binding(history), 'prior': binding(prior),
                     'public_prior_rows': len(questions), 'prior_source_files': [binding(p) for p in prior_sources],
                     'generation_run': str(run), 'evaluation_run': str(evaluation_run(short)),
                     'slot_focus': focuses})
    save(REPORT / 'CAPACITY_CHECK.json', {'checks': checks, 'all_pass': all(c['status'] == 'PASS' for c in checks)})
    save(REPORT / 'EXPERIMENT_PLAN.json', {
        'created_at_utc': now(), 'scope': 'v53_live_question_rules_small_validation',
        'roots': rows, 'candidate_slots': 8, 'slots_per_root': 4,
        'slots_per_type_per_root': 2, 'remote_http_max': 12, 'solver_full_calls_max': 120,
        'all_generation_before_any_solver': True, 'current_solver_feedback_to_writer': False,
        'full_v51_source_and_frozen_previous_round_feedback': True,
        'prior_scope': 'all_v51_and_v52b_mechanical_public_sets_plus_existing_default_history',
        'maximum_mechanical_repairs_per_type': 1, 'replacement_slots': 0,
        'minimum_remote_completion_gap_seconds': 35, 'training_calls': 0,
        'generation_code': binding(CODE / 'generate.py'), 'orchestrator': binding(Path(__file__)),
        'four_slots_are_validation_size_not_final_per_root_quota': True})
    assert all(c['status'] == 'PASS' for c in checks), 'initial_capacity_blocked_before_any_remote_call'
    print(json.dumps({'event': 'prepared', 'candidate_slots': 8,
                      'max_context': max(c['required_context_tokens'] for c in checks),
                      'prior_rows': [r['public_prior_rows'] for r in rows]}), flush=True)

def command(script, action, run, extra=()):
    log = REPORT / 'logs' / (run.name + '_' + action + '.log')
    log.parent.mkdir(exist_ok=True)
    cmd = [str(PYTHON), '-B', '-u', str(CODE / script), action, '--run', str(run), *extra]
    with log.open('x') as stream:
        subprocess.run(cmd, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=True)

def execute(stage):
    plan = read(REPORT / 'EXPERIMENT_PLAN.json')
    assert read(REPORT / 'CAPACITY_CHECK.json')['all_pass']
    for row in plan['roots']:
        for frozen in (row['history'], row['prior']):
            assert binding(Path(frozen['path'])) == frozen
    if stage == 'evaluation':
        assert (REPORT / 'generation.COMPLETE.json').exists()
    save(REPORT / f'{stage}.STARTED.json', {'started_at_utc': now(), 'automatic_replay': False})
    summaries = []
    try:
        for short in ROOTS:
            run = generation_run(short)
            if stage == 'generation':
                command('generate.py', 'generate', run)
                summary = read(run / 'REPORT.json')
                summaries.append({'root': short, 'report': binding(run / 'REPORT.json'),
                                  **{k: summary[k] for k in ('candidate_slots', 'http_attempts', 'mechanical_pass', 'ready_for_solver')}})
            else:
                target = evaluation_run(short)
                command('evaluate_and_pool.py', 'preflight', target, ('--generation', str(run)))
                command('evaluate_and_pool.py', 'live', target)
                command('evaluate_and_pool.py', 'finalize', target)
                command('evaluate_and_pool.py', 'verify', target)
                summary = read(target / 'training_pool/REPORT.json')
                summaries.append({'root': short, 'report': binding(target / 'training_pool/REPORT.json'),
                                  'full_solver_calls': summary['full_solver_calls'],
                                  'selected': summary['new_unique_added']})
            print(json.dumps({'event': stage + '_root_complete', **summaries[-1]}), flush=True)
        save(REPORT / f'{stage}.COMPLETE.json', {'completed_at_utc': now(), 'roots': summaries})
    except BaseException as error:
        save(REPORT / f'{stage}.ERROR.json', {'failed_at_utc': now(), 'error_type': type(error).__name__,
                                             'completed_roots': summaries})
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'generation', 'evaluation'))
    action = parser.parse_args().action
    prepare() if action == 'prepare' else execute(action)
