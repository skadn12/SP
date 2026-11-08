"""Frozen Writer coverage input adapter; generation stays in the shared Solver.

The display-v2 live runner supplies model loading, no-retry ledgers, generation,
parsing, canonical scoring and shutdown. This adapter selects all frozen types,
adds the Inclusive checkpoint, and binds its limit to the audited prompt count.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1')
sys.path.insert(0, str(WORKTREE))
for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'VLLM_USE_FLASHINFER_SAMPLER': '0', 'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ[key] = value

from scripts.current_method.restart_n5_spell.h4_ordering_display_v2 import evaluate as runtime
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.contract import RELABEL_POLICY, permutation_version
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import candidate_request_count
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, gate, solver
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1 import sources as writer_sources

HERE = Path(__file__).resolve().parent
PREPARED, RUN = HERE / 'prepared', HERE / 'run'
MODELS = ('M0', 'M_clean_policy', 'M_inclusive_policy')
TOKENIZER = runtime.TOKENIZER
MAX_MODEL_LEN, MAX_TOKENS, ROLLOUTS = runtime.MAX_MODEL_LEN, runtime.MAX_TOKENS, runtime.ROLLOUTS
require, load_json = runtime.require, runtime.load_json
sha256_file, verify_records = runtime.sha256_file, runtime.verify_records
write_once_json, verify_sha_sidecar = runtime.write_once_json, runtime.verify_sha_sidecar
dated_context = runtime.dated_context
NoRetryLedger = runtime.NoRetryLedger


def input_rows(prepared):
    rows = load_json(prepared / 'candidate_manifest.json')
    require(isinstance(rows, list) and bool(rows), 'nonempty_candidate_list_required')
    require(len({r['candidate_id'] for r in rows}) == len(rows), 'duplicate_candidate_id')
    require(set(r['question_type'] for r in rows) <= {'single_choice', 'multi_select', 'ordering'},
            'unsupported_question_type')
    review = load_json(prepared / 'SELECTION_REVIEW.json')
    require(review.get('status') == 'PASS', 'passed_selection_review_required')
    require(review.get('candidate_count', len(rows)) == len(rows), 'selection_count_drift')
    dialogues = {root: load_json(prepared / 'dialogues' / (root + '.json'))
                 for root in sorted({r['root_id'] for r in rows})}
    return rows, dialogues


def load_inputs(prepared):
    rows, dialogues = input_rows(prepared)
    gates = load_json(prepared / 'program_gate_results.json')
    require([g['candidate_id'] for g in gates] == [r['candidate_id'] for r in rows],
            'gate_candidate_order_drift')
    return rows, gates, dialogues


def install_runtime(limit):
    """Only scope and budget are adapted; the existing live function is unchanged."""
    runtime.MODELS = MODELS
    runtime.LIMIT = limit
    runtime.load_inputs = load_inputs

    class ScopedNoRetryLedger(NoRetryLedger):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('limit', limit)
            super().__init__(*args, **kwargs)

    runtime.NoRetryLedger = ScopedNoRetryLedger


def prepare(prepared, run):
    from transformers import AutoTokenizer
    require(not run.exists(), 'run_exists_do_not_overwrite')
    rows, dialogues = input_rows(prepared)
    source_paths = sorted(p for p in prepared.rglob('*') if p.is_file())
    source_paths += [Path(__file__).resolve(), HERE / 'check_offline.py',
                     runtime.PRIOR_IDENTITY, Path(writer_sources.__file__).resolve()]
    sources = [{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in source_paths]
    sources += gate._contract_source_identity()
    sources = list({r['path']: r for r in sources}.values())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    with dated_context(dialogues):
        gates = [gate.gate_candidate(row, dialogues[row['root_id']], tokenizer,
                                    root_id=row['root_id'], ordering_display_policy=RELABEL_POLICY)
                 for row in rows]
    require(all(g['gate_status'] == gate.PROGRAM_PASS for g in gates),
            'program_gate_failed:' + json.dumps([g for g in gates if g['gate_status'] != gate.PROGRAM_PASS]))
    limit = sum(candidate_request_count(row) for row in rows)
    prompts = [{**p, 'candidate_id': g['candidate_id'], 'root_id': g['root_id']}
               for g in gates for p in g['prompt_audit']]
    require(len(prompts) * ROLLOUTS == limit, 'prompt_request_budget_mismatch')
    verify_sha_sidecar(runtime.PRIOR_IDENTITY)
    prior = load_json(runtime.PRIOR_IDENTITY)
    models = {name: prior['models'][name] for name in MODELS}
    identities = {name: models[name]['base_model_identity' if name == 'M0' else 'merged_model_identity']
                  for name in MODELS}
    stats = {}
    for name, identity in identities.items():
        before = runtime.checkpoint_stats(identity)
        verify_records([{'path': str(Path(identity['path']) / r['name']), 'sha256': r['sha256']}
                        for r in identity['files']])
        stats[name] = runtime.checkpoint_stats(identity)
        require(before == stats[name], 'checkpoint_changed_during_hash:' + name)
    verify_records(sources)
    write_once_json(prepared / 'program_gate_results.json', gates)
    write_once_json(prepared / 'prompt_manifest.json', prompts)
    for path in (prepared / 'program_gate_results.json', prepared / 'prompt_manifest.json'):
        sources.append({'path': str(path.resolve()), 'sha256': sha256_file(path)})
    generation = {kind: solver.generation_config(kind) for kind in sorted({r['question_type'] for r in rows})}
    shared = {
        'version': 'writer_coverage_three_model_solver_eval_v1',
        'ordering_display_policy': RELABEL_POLICY,
        'permutation_contract_version': permutation_version(RELABEL_POLICY),
        'generation_config': solver.generation_config('ordering'),
        'generation_config_by_type': generation,
        'tokenizer': str(TOKENIZER), 'max_model_len': MAX_MODEL_LEN,
        'gpu_memory_utilization': 0.80, 'tensor_parallel_size': 1,
        'enable_prefix_caching': False, 'dtype': 'auto',
        'rollouts': ROLLOUTS, 'request_limit_per_model': limit, 'models': list(MODELS),
        'model_order': list(MODELS), 'candidate_order': [r['candidate_id'] for r in rows],
        'request_order': 'candidate_manifest order, cyclic/relabel variant, no_context/full_context, slot_0..2',
        'seed_policy': 'shared stable_seed(root_id,candidate_id,slot_id); paired across models/context/display',
        'output_mode': 'B0_label; single <box> letter, multi-select letter set, ordering item sequence; no reasoning',
        'scoring': 'Map displayed labels to canonical IDs; single exact, multi-select exact set, ordering exact sequence.',
        'context_policy': 'Complete frozen dated dialogue, including UNKNOWN; unchanged no-context template.',
        'automatic_retries': 0, 'failure_policy': 'stop_on_first_technical_parse_truncation_failure_no_resume',
        'sources': sources, 'prepared': str(prepared.resolve()),
        'selection_review_sha256': sha256_file(prepared / 'SELECTION_REVIEW.json'),
        'runtime_path': str(Path(runtime.__file__).resolve()),
        'gold_policy': 'Frozen reviewed evaluation copies; any correction is documented before generation.',
        'authorization': 'User authorized local Solver evaluation of latest Writer coverage candidates with M0, Clean and Inclusive; no remote generation.'}
    solver.validate_gate_display_policy(shared, gates)
    run.mkdir(parents=True)
    write_once_json(run / 'protocol.json', shared)
    write_once_json(run / 'model_identity.json', {
        'models': models, 'verified_checkpoints': identities, 'checkpoint_stats': stats,
        'identity_scope': 'ACTUAL_LOADED_CHECKPOINTS_AND_BASE_TOKENIZER',
        'adapter_identity': 'Historical provenance only; merged checkpoints load directly with no additional adapter.'})
    for name in MODELS:
        write_once_json(run / name / 'solver_contract.json', {
            **shared, 'model': name, 'model_path': models[name]['merged_model_path'],
            'protocol_sha256': sha256_file(run / 'protocol.json')})
    report = {
        'status': 'PASS', 'candidate_count': len(rows), 'by_type': dict(Counter(r['question_type'] for r in rows)),
        'root_count': len(dialogues), 'unique_prompts': len(prompts), 'models': list(MODELS),
        'per_model_requests': limit, 'maximum_local_generations': limit * len(MODELS),
        'input_tokens_per_model': sum(r['prompt_token_count'] for r in prompts) * ROLLOUTS,
        'maximum_output_tokens_total': limit * len(MODELS) * MAX_TOKENS,
        'max_prompt_tokens': max(r['prompt_token_count'] for r in prompts),
        'model_calls': 0, 'http_calls': 0, 'created_at': runtime.now(),
        'protocol_sha256': sha256_file(run / 'protocol.json')}
    write_once_json(run / 'preflight.json', report)
    return report


def live(model, run):
    pre = load_json(run / 'preflight.json')
    protocol = load_json(run / 'protocol.json')
    check_path = run.parent / 'OFFLINE_CHECKS.json'
    verify_sha_sidecar(check_path)
    checks = load_json(check_path)
    require(checks['status'] == 'PASS' and checks['real_model_calls'] == 0
            and checks['mock_responses_scored'] == pre['per_model_requests']
            and checks['protocol_sha256'] == sha256_file(run / 'protocol.json'),
            'complete_matching_offline_checks_required')
    require(protocol['models'] == list(MODELS), 'model_scope_drift')
    rows, _, _ = load_inputs(Path(protocol['prepared']))
    limit = sum(candidate_request_count(r) for r in rows)
    require(limit == pre['per_model_requests'] == protocol['request_limit_per_model'], 'request_limit_drift')
    for kind, config in protocol['generation_config_by_type'].items():
        require(config == solver.generation_config(kind), 'runtime_generation_drift:' + kind)
    for prior in MODELS[:MODELS.index(model)]:
        require(load_json(run / prior / 'report.json')['status'] == 'COMPLETE', 'prior_model_must_complete:' + prior)
    install_runtime(limit)
    return runtime.live(model, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'live'))
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.phase == 'live' and args.model is None:
        parser.error('--model is required for live')
    result = prepare(args.prepared.resolve(), args.run.resolve()) if args.phase == 'prepare' else live(args.model, args.run.resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
