"""Paired evaluation of the two existing round-one models on frozen R03 QA."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
WORK = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1')
sys.path.insert(0, str(WORK))
for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'VLLM_USE_FLASHINFER_SAMPLER': '0', 'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ[key] = value

from scripts.current_method.restart_n5_spell.h4_memory_light_gate_solver_v1 import run as baseline
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2 import evaluate as runtime
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.contract import LEGACY_POLICY
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, gate, solver
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import RunLock, write_once_json
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
from scripts.current_method.restart_n5_spell.core import atomic_json, atomic_text, sha256_file
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1 import sources as renderer

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
PRIOR = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_writer_coverage_solver_eval_v1/run/model_identity.json')
TRAINED = ('M_clean_policy', 'M_inclusive_policy')
MODELS = ('M0',) + TRAINED
TYPES = ('single_choice', 'multi_select', 'ordering')
PAIR_FIELDS = ('request_id', 'candidate_id', 'question_type', 'variant_id', 'context_mode',
               'slot_id', 'seed', 'prompt_sha256', 'prompt_token_count', 'display_to_canonical',
               'displayed_gold', 'canonical_gold')


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def inputs():
    return [read(baseline.RUN / name) for name in ('questions.json', 'qa_source.json', 'program_checks.json')]


def model_results(name):
    directory = baseline.RUN / 'M0' if name == 'M0' else RUN / name
    return [read(directory / 'candidate_results' / f'{i:02d}.json') for i in range(1, 10)]


def check_pair(result, old, question):
    for field in ('candidate_id', 'question_type', 'candidate_pool_sha256', 'dialogue_sha256', 'request_count'):
        require(result[field] == old[field], 'candidate_pair_drift:' + field)
    variants = {v['variant_id']: v for v in contract.position_variants(question)}
    require(len(result['requests']) == len(old['requests']), 'request_count_drift')
    for row, previous in zip(result['requests'], old['requests']):
        for field in PAIR_FIELDS:
            require(row[field] == previous[field], 'paired_request_drift:' + field)
        score = contract.score_output(row['raw_output'], question, variants[row['variant_id']])
        for field in ('scored', 'correct', 'canonical_prediction', 'display_prediction', 'parse_status'):
            require(score[field] == row[field], 'response_rescore_drift:' + field)
        require(row['scored'] and row['backend_metadata']['finish_reason'] in ('stop', 'eos'), 'failed_response')
    return len(result['requests'])


def prepare():
    from transformers import AutoTokenizer
    require(not RUN.exists(), 'new_run_required')
    oldpre = baseline.verify()
    questions, source, checks = inputs()
    require(len(questions) == 9 and len(source['turns']) == 188, 'input_scope_drift')
    original = model_results('M0')
    require(sum(check_pair(r, r, q) for q, r in zip(questions, original)) == 234, 'M0_rescore_count')
    baseline.install(source)
    tokenizer = AutoTokenizer.from_pretrained(baseline.BASE_SOLVER_MODEL, local_files_only=True)
    for q, old in zip(questions, checks):
        actual = gate.gate_candidate(q, source, tokenizer, root_id=baseline.ROOT)
        require(actual['gate_status'] == gate.PROGRAM_PASS, 'program_gate_failure')
        require(actual['prompt_audit'] == old['prompt_audit'], 'M0_prompt_preflight_drift')
    runtime.verify_sha_sidecar(PRIOR)
    prior = read(PRIOR)
    models = {m: prior['models'][m] for m in MODELS}
    require(models['M0']['base_model_identity'] == read(baseline.RUN / 'solver_contract.json')['checkpoint_identity'],
            'base_model_identity_mismatch')
    identities = {m: models[m]['base_model_identity' if m == 'M0' else 'merged_model_identity'] for m in MODELS}
    stats = {}
    for name, identity in identities.items():
        before = runtime.checkpoint_stats(identity)
        runtime.verify_records([{'path': str(Path(identity['path']) / f['name']), 'sha256': f['sha256']}
                                for f in identity['files']])
        stats[name] = runtime.checkpoint_stats(identity)
        require(before == stats[name], 'checkpoint_changed_during_hash')
    records = list(oldpre['sources'])
    paths = [Path(__file__), PRIOR, baseline.RUN / 'preflight.json', baseline.RUN / 'COMPLETED.json',
             baseline.DELIVERY / 'RESULTS.json', Path(renderer.__file__), Path(runtime.__file__)]
    paths += list((baseline.RUN / 'M0/candidate_results').glob('*.json'))
    records += [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in paths]
    records = list({r['path']: r for r in records}.values())
    protocol = {'version': 'light_gate_trained_solver_paired_v1', 'new_models': list(TRAINED),
        'reuse_M0': str(baseline.RUN), 'new_requests_per_model': 234, 'maximum_new_generations': 468,
        'generation_by_type': {t: solver.generation_config(t) for t in TYPES},
        'ordering_display_policy': LEGACY_POLICY, 'prompt_adapter': 'original_light_gate_M0_dated_prompt',
        'questions_sha256': sha256_file(baseline.RUN / 'questions.json'),
        'dialogue_sha256': sha256_file(baseline.RUN / 'qa_source.json'),
        'pair_fields': list(PAIR_FIELDS), 'sources': records,
        'max_model_len': baseline.MAX_MODEL_LEN, 'max_output_tokens': baseline.MAX_TOKENS,
        'tokenizer': str(baseline.BASE_SOLVER_MODEL), 'gpu_memory_utilization': 0.80,
        'gold_policy': 'UNCHANGED_ORIGINAL_GOLD; content notes do not filter questions',
        'authorization': 'User requested Clean and Inclusive on the same nine new questions',
        'training_calls': 0, 'http_calls': 0, 'memory_calls': 0, 'automatic_retries': 0}
    RUN.mkdir()
    write_once_json(RUN / 'protocol.json', protocol)
    write_once_json(RUN / 'model_identity.json', {'models': models, 'identities': identities, 'checkpoint_stats': stats})
    for name in TRAINED:
        write_once_json(RUN / name / 'solver_contract.json', {**protocol, 'model': name,
            'model_path': models[name]['merged_model_path'], 'protocol_sha256': sha256_file(RUN / 'protocol.json')})
    out = {'status': 'PASS', 'new_models': list(TRAINED), 'reused_M0_requests': 234,
        'new_requests_per_model': 234, 'maximum_new_generations': 468, 'matched_unique_prompts': 78,
        'source_turns': 188, 'question_count': 9, 'real_model_calls': 0,
        'created_at': datetime.now(timezone.utc).isoformat()}
    write_once_json(RUN / 'preflight.json', out)
    return out


def verify_sources():
    baseline.verify()
    runtime.verify_sha_sidecar(RUN / 'protocol.json')
    runtime.verify_sha_sidecar(RUN / 'model_identity.json')
    protocol, identity = read(RUN / 'protocol.json'), read(RUN / 'model_identity.json')
    runtime.verify_records(protocol['sources'])
    for name in MODELS:
        require(runtime.checkpoint_stats(identity['identities'][name]) == identity['checkpoint_stats'][name],
                'checkpoint_changed:' + name)
    for kind, config in protocol['generation_by_type'].items():
        require(config == solver.generation_config(kind), 'generation_config_drift:' + kind)
    return protocol


def live(name):
    from transformers import AutoTokenizer
    verify_sources()
    md = RUN / name
    frozen = read(md / 'solver_contract.json')
    contract_sha = runtime.verify_sha_sidecar(md / 'solver_contract.json')
    require(frozen['protocol_sha256'] == sha256_file(RUN / 'protocol.json'), 'model_protocol_drift')
    questions, source, checks = inputs()
    original = model_results('M0')
    baseline.install(source)
    tokenizer = AutoTokenizer.from_pretrained(baseline.BASE_SOLVER_MODEL, local_files_only=True)
    with RunLock(RUN):
        require(not (md / 'STARTED.json').exists(), 'already_started_no_automatic_resume')
        baseline.assert_gpu_idle()
        ledger = baseline.NoRetryLedger(md, solver_contract_sha256=contract_sha, limit=234)
        require(not ledger.states(), 'prior_requests_exist')
        write_once_json(md / 'STARTED.json', {'model': name, 'limit': 234, 'at': datetime.now(timezone.utc).isoformat()})
        backend, results = None, []
        started = time.monotonic()
        try:
            solver.ensure_vllm_environment()
            backend = VLLMSolverBackend(model_path=Path(frozen['model_path']), tokenizer_path=baseline.BASE_SOLVER_MODEL,
                max_model_len=baseline.MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
                max_tokens=baseline.MAX_TOKENS, gpu_memory_utilization=0.80)
            for number, (q, check, old) in enumerate(zip(questions, checks, original), 1):
                result = solver.evaluate_candidate(candidate=q, expanded=source, gate_row=check,
                    tokenizer=tokenizer, backend=backend, ledger=ledger, solver_contract_sha256=contract_sha)
                write_once_json(md / 'candidate_results' / f'{number:02d}.json', result)
                check_pair(result, old, q)
                results.append(result)
                progress = {'model': name, 'completed_questions': number, 'new_generations': ledger.used}
                atomic_json(md / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            verify_sources()
            counts = ledger.summary()
            require(counts['completed_request_count'] == counts['actual_solver_generation_count'] == 234,
                    'generation_count_mismatch')
            require(counts['retried_generation_count'] == counts['failed_request_count'] == counts['ambiguous_request_count'] == 0,
                    'nonclean_ledger')
            rr = [a for r in results for a in r['requests']]
            report = {'status': 'COMPLETE', 'model': name, 'ledger': counts, 'new_generations': 234,
                'input_tokens': sum(a['backend_metadata']['prompt_token_count'] for a in rr),
                'output_tokens': sum(a['backend_metadata']['completion_token_count'] for a in rr),
                'elapsed_seconds': round(time.monotonic() - started, 2), 'paired_requests_verified': 234}
            write_once_json(md / 'COMPLETED.json', report)
            return report
        except BaseException as error:
            atomic_json(md / 'STOPPED.json', {'error_type': type(error).__name__, 'error': str(error),
                'completed_questions': len(results), 'ledger': ledger.summary(), 'automatic_retry': False})
            raise
        finally:
            if backend is not None:
                backend.close()


def metric(results):
    requests = [a for r in results for a in r['requests']]
    out = {'question_count': len(results)}
    for mode in ('full_context', 'no_context'):
        rr = [a for a in requests if a['context_mode'] == mode]
        count = sum(a['correct'] for a in rr)
        out[mode] = {'matches': count, 'total': len(rr), 'rate': count / len(rr)}
    return out


def report():
    verify_sources()
    questions, _, _ = inputs()
    results = {m: model_results(m) for m in MODELS}
    for name in TRAINED:
        require(read(RUN / name / 'COMPLETED.json')['status'] == 'COMPLETE', 'model_incomplete')
        require(sum(check_pair(r, o, q) for r, o, q in zip(results[name], results['M0'], questions)) == 234, 'pair_count')
    data = {'status': 'COMPLETE', 'question_count': 9, 'source_turns': 188,
            'new_generations': 468, 'reused_M0_generations': 234, 'total_compared_responses': 702,
            'score_semantics': 'AGREEMENT_WITH_UNCHANGED_ORIGINAL_GOLD', 'models': {}, 'paired_vs_M0': {}}
    for name, rows in results.items():
        data['models'][name] = {'all': metric(rows),
            'by_type': {t: metric([r for r in rows if r['question_type'] == t]) for t in TYPES},
            'excluding_flagged_ordering_3_9': metric([r for i, r in enumerate(rows, 1) if i not in (3, 9)])}
    for name in TRAINED:
        data['paired_vs_M0'][name] = {}
        for mode in ('full_context', 'no_context'):
            pairs = [(a['correct'], b['correct']) for old, new in zip(results['M0'], results[name])
                     for a, b in zip(old['requests'], new['requests']) if a['context_mode'] == mode]
            data['paired_vs_M0'][name][mode] = {
                'both_match': sum(a and b for a, b in pairs), 'M0_only': sum(a and not b for a, b in pairs),
                'trained_only': sum(b and not a for a, b in pairs), 'neither_matches': sum(not a and not b for a, b in pairs),
                'delta_percentage_points': 100 * sum(int(b) - int(a) for a, b in pairs) / len(pairs)}
    data['execution'] = {m: read(RUN / m / 'COMPLETED.json') for m in TRAINED}
    per_question = []
    for i, q in enumerate(questions):
        row = {'number': i + 1, 'candidate_id': q['candidate_id'], 'question_type': q['question_type'], 'gold': q['gold'], 'models': {}}
        for name in MODELS:
            row['models'][name] = metric([results[name][i]])
            for mode in ('full_context', 'no_context'):
                dist = Counter(tuple(a['canonical_prediction']) for a in results[name][i]['requests'] if a['context_mode'] == mode)
                row['models'][name][mode]['answer_distribution'] = [{'answer': list(a), 'count': n} for a, n in dist.most_common()]
        per_question.append(row)
    atomic_json(HERE / 'RESULTS.json', data)
    atomic_json(HERE / 'QUESTION_RESULTS.json', per_question)
    lines = ['# New nine questions: M0, Clean, Inclusive paired comparison', '',
        'Same 188-turn dialogue, unchanged questions/options/Gold, paired prompt hashes, display mappings and seeds.',
        'M0 reuses 234 frozen responses. Clean and Inclusive each add 234 responses; no retraining or HTTP calls.',
        'Rates are exact agreement with original Gold. Each type has only 3 distinct questions; repeated answers are not independent questions.', '']
    for mode in ('full_context', 'no_context'):
        lines += ['## ' + mode, '', '| Type | M0 | Clean | Inclusive |', '|---|---:|---:|---:|']
        for t in (*TYPES, 'all'):
            cells = []
            for name in MODELS:
                d = data['models'][name]
                v = (d['all'] if t == 'all' else d['by_type'][t])[mode]
                cells.append(f"{v['matches']}/{v['total']} ({v['rate']:.2%})")
            lines.append('| ' + t + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Per question', '', '| Question | Type | M0 full/no | Clean full/no | Inclusive full/no |', '|---|---|---|---|---|']
    for row in per_question:
        cells = []
        for name in MODELS:
            d = row['models'][name]
            cells.append(' / '.join(str(d[m]['matches']) + ':' + str(d[m]['total']) for m in ('full_context', 'no_context')))
        lines.append('| ' + str(row['number']) + ' | ' + row['question_type'] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Limitations', '',
        'Questions 3 and 9 have known chronology ambiguities. Question 8 option B has a broad-paraphrase concern. These notes do not change selection or Gold.',
        'This is one known training-domain root, not an independent generalization benchmark. The historical 2,294-question results are not pooled here.', '',
        '| Excluding Q3/Q9 | Full context | No context |', '|---|---:|---:|']
    for name in MODELS:
        d = data['models'][name]['excluding_flagged_ordering_3_9']
        cells = [f"{d[m]['matches']}/{d[m]['total']} ({d[m]['rate']:.2%})" for m in ('full_context', 'no_context')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    atomic_text(HERE / 'RESULTS.md', chr(10).join(lines) + chr(10))
    atomic_json(HERE / 'VERIFICATION.json', {'status': 'PASS', 'new_requests_rescored_and_paired': 468,
        'frozen_sources_unchanged': True, 'total_compared_responses': 702, 'new_inference_during_verification': 0})
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('prepare', 'live', 'report'))
    p.add_argument('--model', choices=TRAINED)
    args = p.parse_args()
    if args.phase == 'live' and args.model is None:
        p.error('--model required')
    output = prepare() if args.phase == 'prepare' else live(args.model) if args.phase == 'live' else report()
    print(json.dumps(output, ensure_ascii=False), flush=True)
