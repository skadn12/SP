"""Replay all 123 historical zero-to-partial Train candidates, unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CODE = REPO / 'current_method_github'
HIST = REPO / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1'
BASE_HIST = REPO / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1'
V127 = REPO / 'offline-deliveries/h4_codex_dev_eval_v127'
EXPECTED_PREFIX = REPO / '.venv-local4b'
FIELDS = ['request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id',
          'shift', 'slot_id', 'seed', 'context_mode', 'prompt_sha256',
          'prompt_token_count', 'display_to_canonical', 'canonical_gold', 'displayed_gold']


def read(path):
    return json.loads(Path(path).read_text())


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def htext(text):
    return hashlib.sha256(text.encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], 'FILE_CHANGED:' + row['path']


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def update(path, data):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def native():
    sys.path.insert(0, str(CODE))
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, solver
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
    return contract, solver, SolverRequest, VLLMSolverBackend


def sidecar(path):
    path = Path(path)
    assert sha(path) == path.with_name(path.name + '.sha256').read_text().split()[0], 'OLD_SIDECAR_MISMATCH:' + str(path)


def prepare():
    assert Path(sys.prefix) == EXPECTED_PREFIX, 'USE_LOCAL4B_VENV'
    assert not (HERE / 'manifest.json').exists(), 'ALREADY_PREPARED'
    contract, solver, _, _ = native()
    matrix_path = HIST / 'candidate_transition_matrix.jsonl'
    report_path = HIST / 'm0_full_zero_transition_report.json'
    for path in [matrix_path, report_path, HIST / 'preflight_report.json', HIST / 'solver_contract.json']:
        sidecar(path)
    matrix = readl(matrix_path)
    selected = [row for row in matrix if row['models']['M0']['full_context_correct_count'] == 0
                and 0 < row['models']['M_clean_policy']['full_context_accuracy'] < 1]
    assert len(selected) == 123 and len({r['root_id'] for r in selected}) == 58
    assert Counter(r['question_type'] for r in selected) == {'ordering': 62, 'multi_select': 51, 'single_choice': 10}
    assert all(r['split'] == 'train' and r['training_exposure'] == 'UNEXPOSED_TO_BOTH' for r in selected)
    assert read(report_path)['models']['M_clean_policy']['full_status_counts']['partial'] == 123
    assert sha(V127 / 'COMPLETE_MANIFEST.json') == '4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61'
    frozen_v127 = {row['path']: row for row in read(V127 / 'COMPLETE_MANIFEST.json')['files']}
    identities = {}
    paths = {matrix_path, report_path, HIST / 'preflight_report.json', HIST / 'solver_contract.json', V127 / 'COMPLETE_MANIFEST.json'}
    for side, old_side in [('live_clean', 'baseline'), ('live_new', 'new')]:
        path = V127 / 'run' / old_side / 'solver_identity.json'
        assert sha(path) == frozen_v127[str(path.relative_to(V127))]['sha256']
        identities[side] = read(path)
        paths.add(path)
        for row in identities[side]['checkpoint_files']:
            assert sha(row['path']) == row['sha256'], 'MODEL_CHECKPOINT_DRIFT'
            paths.add(Path(row['path']))
        audit = Path(identities[side]['audit_path'])
        assert sha(audit) == identities[side]['audit_sha256']
        paths.add(audit)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    inventory = read(HIST / 'preflight_report.json')['source_root_inventory']
    questions, candidates, requests = [], [], []
    source_cache = {}
    historical_replay_checks = 0
    for transition in selected:
        cid, root = transition['candidate_id'], transition['root_id']
        if root not in source_cache:
            source = inventory[root]
            pool_path, dialogue_path = Path(source['candidate_pool_path']), Path(source['dialogue_path'])
            assert sha(pool_path) == source['candidate_pool_sha256']
            assert sha(dialogue_path) == source['dialogue_sha256']
            paths.update([pool_path, dialogue_path])
            pool = {q['candidate_id']: q for q in read(pool_path)}
            dialogue = read(dialogue_path)
            source_cache[root] = pool, dialogue
            destination = HERE / 'sources' / (root + '.json')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dialogue_path, destination)
        pool, dialogue = source_cache[root]
        question = pool[cid]
        assert question['root_id'] == root and question['question_type'] == transition['question_type']
        questions.append(question)
        variant_map = {v['variant_id']: v for v in contract.position_variants(question)}
        m0_path = BASE_HIST / 'candidate_results' / (htext(cid) + '.json')
        clean_path = HIST / 'clean_train/candidate_results' / (htext(cid) + '.json')
        for path in [m0_path, clean_path]:
            sidecar(path)
            paths.add(path)
        m0, clean = read(m0_path), read(clean_path)
        candidates.append({'candidate_id': cid, 'root_id': root,
            'question_type': question['question_type'], 'training_exposure': transition['training_exposure'],
            'historical_m0_path': str(m0_path), 'historical_clean_path': str(clean_path),
            'transition': transition})
        m0_full = {r['request_id']: r for r in m0['requests'] if r['context_mode'] == 'full_context'}
        clean_full = [r for r in clean['requests'] if r['context_mode'] == 'full_context']
        assert set(m0_full) == {r['request_id'] for r in clean_full}
        for old in clean_full:
            other = m0_full[old['request_id']]
            assert {k: old[k] for k in FIELDS} == {k: other[k] for k in FIELDS}
            variant = variant_map[old['variant_id']]
            for key in ['display_to_canonical', 'canonical_gold', 'displayed_gold']:
                assert old[key] == variant[key], 'VARIANT_REPLAY_DRIFT'
            prompt = contract.build_prompt(tokenizer, question, variant, contract.dialogue_rows(dialogue), context_mode='full_context')
            assert htext(prompt) == old['prompt_sha256'], 'PROMPT_REPLAY_DRIFT'
            assert len(tokenizer.encode(prompt, add_special_tokens=False)) == old['prompt_token_count']
            assert old['prompt_token_count'] + 32 <= 16384
            for old_scope, old_row in [(BASE_HIST, other), (HIST / 'clean_train', old)]:
                old_prompt = old_scope / 'raw_prompts' / (old_row['prompt_sha256'] + '.txt')
                assert old_prompt.read_bytes() == prompt.encode(), 'HISTORICAL_PROMPT_BYTES_CHANGED'
                paths.add(old_prompt)
                calculated = contract.score_output(old_row['raw_output'], question, variant)
                for key in ['parse_status', 'parser_version', 'scored', 'correct', 'display_prediction', 'canonical_prediction', 'failure_type', 'metrics']:
                    assert calculated[key] == old_row[key], 'HISTORICAL_SCORE_REPLAY_DRIFT:' + key
                md = old_row['backend_metadata']
                assert md['sampling_seed'] == old['seed'] and md['prompt_token_count'] == old['prompt_token_count']
                assert md['temperature'] == (.7 if question['question_type'] == 'single_choice' else .8)
                assert md['top_p'] == .9 and md['max_tokens'] == 32
                historical_replay_checks += 1
            prompt_path = HERE / 'prompts' / (old['prompt_sha256'] + '.txt')
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            if not prompt_path.exists():
                prompt_path.write_text(prompt)
            assert prompt_path.read_bytes() == prompt.encode()
            requests.append({**{k: old[k] for k in FIELDS}, 'variant': variant,
                             'prompt_path': str(prompt_path), 'original_response': old})
    assert len(requests) == len({r['request_id'] for r in requests}) == 1629
    assert historical_replay_checks == 3258
    save(HERE / 'QUESTIONS.json', {'candidates': questions})
    runtime = {name: importlib.metadata.version(name) for name in ['vllm', 'torch', 'transformers']}
    configuration = {t: solver.generation_config(t) for t in ['single_choice', 'multi_select', 'ordering']}
    scope = {'schema_version': 'v131_historical_frontier_v1', 'created_at_utc': now(),
        'selection_rule': 'M0.full_context_correct_count == 0 and 0 < M_clean_policy.full_context_accuracy < 1, from the frozen historical transition matrix only.',
        'questions': 123, 'roots': 58, 'question_types': {'ordering': 62, 'multi_select': 51, 'single_choice': 10},
        'historical_selection_partial_means': 'Some repeated exact answers succeeded; this is not within-answer partial credit.',
        'split': 'train', 'historical_question_exposure': 'UNEXPOSED_TO_BOTH',
        'context_modes': ['full_context'], 'requests_per_side': 1629,
        'new_inference_calls': 3258, 'reused_historical_calls': 3258,
        'new_sides': ['live_clean', 'live_new'], 'historical_sides': ['historical_m0', 'historical_clean'],
        'native_generation_algorithm_changed': False,
        'historical_runtime': read(HIST / 'solver_contract.json')['generation_config'],
        'current_runtime': runtime,
        'runtime_bridge': 'Re-evaluate M_clean under the same current runtime as M_v125_newonly; retain historical M0 and M_clean scores separately.',
        'prompt_and_gold_replay_checks': historical_replay_checks,
        'historical_code_sha_equal_to_current_claimed': False,
        'native_contract_semantics_verified': 'All 3258 reused historical raw scores reproduce exactly; every original full-context prompt reproduces byte-for-byte.',
        'ordering_display': 'Four original IDs retain identity; historical shift cycles displayed item lines without rebinding IDs.',
        'test_evaluation': False, 'new_training': False, 'new_question_generation': False,
        'automatic_retry': False, 'score_based_filtering_of_this_selected_set': False,
        'gold_modification': False,
        'limits': ['Outcome-selected historical Train-root diagnostic, not an unbiased DEV/TEST estimate.',
                   'Unexposed refers to these questions before first training, not unseen roots or proof of no semantic overlap with later data.',
                   'Historical Gold is replayed unchanged; selection does not certify semantic correctness.',
                   'Repeated positions and slots are correlated; no statistical significance claim.']}
    save(HERE / 'SCOPE.json', scope)
    save(HERE / 'manifest.json', {'schema_version': 'v131_full_context_exact_projection_v1',
        'created_at_utc': now(), 'questions_path': str(HERE / 'QUESTIONS.json'),
        'native_import_root': str(CODE), 'native_contract_path': str(Path(contract.__file__).resolve()),
        'candidates': candidates, 'requests': requests, 'identities': identities,
        'decoder_contract': configuration, 'runtime': runtime,
        'question_count': 123, 'request_count_per_side': 1629,
        'response_filename': 'sha256(request_id UTF-8).json'})
    imported = {Path(obj.__file__).resolve() for obj in sys.modules.values()
                if getattr(obj, '__file__', None) and str(Path(obj.__file__).resolve()).startswith(str(CODE))}
    tool_paths = imported | {Path(__file__).resolve(), HERE / 'report_frontier.py'}
    save(HERE / 'TOOLS_FREEZE.json', {'status': 'FROZEN_BEFORE_INFERENCE', 'created_at_utc': now(),
        'files': [binding(path) for path in sorted(tool_paths)]})
    paths.update(p for p in HERE.rglob('*') if p.is_file() and p.name not in ['prepare.log', 'evaluate.py', 'report_frontier.py'])
    paths.update(path.with_name(path.name + '.sha256') for path in list(paths)
                 if path.with_name(path.name + '.sha256').is_file())
    save(HERE / 'PREFLIGHT_FREEZE.json', {'status': 'READY_FOR_INDEPENDENT_AUDIT',
        'created_at_utc': now(), 'files': [binding(path) for path in sorted(paths)]})
    print(json.dumps({'status': 'PREPARED', 'questions': 123, 'new_calls': 3258,
                      'historical_score_replay_checks': historical_replay_checks}))


def frozen():
    verify(read(HERE / 'TOOLS_FREEZE.json')['files'])
    verify(read(HERE / 'PREFLIGHT_FREEZE.json')['files'])
    audit = read(HERE / 'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json')
    assert audit['status'] == 'PASS' and not audit.get('errors')
    verify(audit['audited_files'])


def live_side(side):
    assert side in ['live_clean', 'live_new'] and Path(sys.prefix) == EXPECTED_PREFIX
    assert not (HERE / 'responses' / side).exists(), 'SIDE_ALREADY_STARTED'
    frozen()
    contract, solver, SolverRequest, Backend = native()
    plan = read(HERE / 'manifest.json')
    identity = plan['identities'][side]
    output = HERE / 'responses' / side
    output.mkdir(parents=True)
    started = time.monotonic()
    record = {'status': 'LOADING', 'side': side, 'planned_calls': 1629,
              'started_at_utc': now(), 'actual_calls': 0, 'pid': os.getpid()}
    update(HERE / (side + '_STATUS.json'), record)
    backend = None
    try:
        solver.ensure_vllm_environment()
        for qtype, expected in plan['decoder_contract'].items():
            assert solver.generation_config(qtype) == expected, 'RUNTIME_DECODER_DRIFT'
        backend = Backend(model_path=Path(identity['model_path']),
                          tokenizer_path=Path(identity['tokenizer_path']),
                          max_model_len=16384, temperature=.7, top_p=.9,
                          max_tokens=32, gpu_memory_utilization=.8)
        backend.model_identifier = identity['model_identifier']
        questions = {q['candidate_id']: q for q in read(plan['questions_path'])['candidates']}
        record.update(status='RUNNING', model_load_seconds=backend.load_latency_seconds)
        update(HERE / (side + '_STATUS.json'), record)
        for index, row in enumerate(plan['requests'], 1):
            question, variant = questions[row['candidate_id']], row['variant']
            prompt = Path(row['prompt_path']).read_text()
            assert htext(prompt) == row['prompt_sha256']
            request = SolverRequest(request_id=row['request_id'], qa_id=row['candidate_id'],
                evaluation_mode='full_context', order_id=row['variant_id'],
                target_slot=row['slot_id'], retry_index=0, sampling_seed=row['seed'],
                options=variant.get('options') or variant.get('items'),
                canonical_answer_text=','.join(row['displayed_gold']),
                prompt_sha256=row['prompt_sha256'], prompt=prompt)
            answer = (backend.generate(request) if row['question_type'] == 'single_choice'
                      else solver._structured_generate(backend, request, question_type=row['question_type']))
            metadata = answer.metadata
            assert metadata['sampling_seed'] == row['seed']
            assert metadata['prompt_token_count'] == row['prompt_token_count']
            score = contract.score_output(answer.raw_text, question, variant)
            valid = bool(score['scored']) and metadata['finish_reason'] in ['stop', 'eos']
            save(output / (htext(row['request_id']) + '.json'), {
                **{k: row[k] for k in FIELDS}, 'side': side,
                'solver_model': identity['model_identifier'], 'model_path': identity['model_path'],
                'model_fingerprint_sha256': identity['model_fingerprint_sha256'],
                'raw_output': answer.raw_text, 'score': score, 'metadata': metadata,
                'runtime': plan['runtime'], 'evaluation_valid': valid,
                'created_at_utc': now(), 'automatic_retry': False})
            record['actual_calls'] = index
            if index % 25 == 0 or index == 1629:
                record.update(elapsed_seconds=time.monotonic() - started, last_request_id=row['request_id'])
                update(HERE / (side + '_STATUS.json'), record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        record.update(status='COMPLETE', finished_at_utc=now(),
                      elapsed_seconds=time.monotonic() - started, pid=None)
        update(HERE / (side + '_STATUS.json'), record)
    except BaseException as error:
        record.update(status='STOPPED_WITH_ERROR', error_type=type(error).__name__, error=str(error), stopped_at_utc=now())
        update(HERE / (side + '_STATUS.json'), record)
        raise
    finally:
        if backend is not None:
            backend.close()


def live():
    assert not (HERE / 'RUN_STATUS.json').exists(), 'ALREADY_STARTED_NO_AUTOMATIC_RERUN'
    started = time.monotonic()
    record = {'status': 'VERIFYING', 'started_at_utc': now(), 'pid': os.getpid(),
              'planned_new_calls': 3258, 'automatic_retry': False, 'test_evaluation': False}
    update(HERE / 'RUN_STATUS.json', record)
    try:
        frozen()
        for side in ['live_clean', 'live_new']:
            record['status'] = side.upper() + '_RUNNING'
            update(HERE / 'RUN_STATUS.json', record)
            with (HERE / (side + '.log')).open('x') as output:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), side],
                               stdout=output, stderr=subprocess.STDOUT, check=True)
            assert read(HERE / (side + '_STATUS.json'))['status'] == 'COMPLETE'
        record['status'] = 'REPORTING'
        update(HERE / 'RUN_STATUS.json', record)
        subprocess.run([sys.executable, str(HERE / 'report_frontier.py')], check=True)
        assert read(HERE / 'SUMMARY.json')['status'] == 'COMPLETE'
        record.update(status='REPORTED_AWAITING_INDEPENDENT_AUDIT', actual_new_calls=3258,
                      elapsed_seconds=time.monotonic() - started, finished_at_utc=now(), pid=None)
        update(HERE / 'RUN_STATUS.json', record)
        print(json.dumps(record, ensure_ascii=False))
    except BaseException as error:
        record.update(status='STOPPED_WITH_ERROR', error_type=type(error).__name__, error=str(error), stopped_at_utc=now())
        update(HERE / 'RUN_STATUS.json', record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'verify', 'live', 'live_clean', 'live_new'])
    phase = parser.parse_args().phase
    {'prepare': prepare, 'verify': frozen, 'live': live,
     'live_clean': lambda: live_side('live_clean'),
     'live_new': lambda: live_side('live_new')}[phase]()
