"""Evaluate the 29 frozen downstream questions with the established local Solvers.

Only this delivery's input adapter adds the already-frozen conversation dates.
Production generation, Gold, semantic labels, and training data are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file, sha256_bytes
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import (
    MAX_MODEL_LEN, MAX_TOKENS, ROLLOUTS, SINGLE_TEMPERATURE, TOP_P,
    candidate_request_count,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, gate, solver
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import (
    RunLock, load_json, load_jsonl, write_once_json, write_once_jsonl, write_once_text,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.request_ledger import (
    RecordedSolverFailure, Round0RequestLedger, request_cache_key,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle
from scripts.current_method.restart_n5_spell.h4_round1_paired_blind_solver_eval_v1.run import _identity_preflight
from scripts.current_method.restart_n5_spell.h4_round1_paired_blind_solver_eval_v1 import BASE_MODEL
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.sources import render_dialogue
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
SOURCE = HERE.parent / 'h4_gate_downstream_v1'
ROOTS = ('cc-train-episode-259192', 'cc-train-episode-96462', 'cc-train-episode-112812')
MODELS = ('M0', 'M_clean_policy', 'M_inclusive_policy')
ORIGINAL_BUILD_PROMPT = contract.build_prompt
DIALOGUES = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def dated_build_prompt(tokenizer, candidate, variant, dialogue, *, context_mode):
    """Reuse the original question/system templates, adding frozen source dates."""
    if context_mode == 'no_context':
        return ORIGINAL_BUILD_PROMPT(tokenizer, candidate, variant, dialogue, context_mode=context_mode)
    expanded = DIALOGUES[candidate['root_id']]
    if list(dialogue) != contract.dialogue_rows(expanded):
        raise RuntimeError('date_adapter_dialogue_identity_drift')
    label = 'Context' if candidate['question_type'] == 'single_choice' else 'Dialogue'
    old_context = label + ':\n' + '\n'.join(
        f"[{r['turn_id']}] {r['speaker']}: {r['text']}" for r in dialogue)
    new_context = label + ':\n' + render_dialogue(expanded, numbered=False)

    class ContextTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            copied = [dict(m) for m in messages]
            if copied[1]['role'] != 'user' or not copied[1]['content'].startswith(old_context + '\n\n'):
                raise RuntimeError('date_adapter_context_boundary_mismatch')
            copied[1]['content'] = new_context + copied[1]['content'][len(old_context):]
            return tokenizer.apply_chat_template(copied, **kwargs)

    return ORIGINAL_BUILD_PROMPT(ContextTokenizer(), candidate, variant, dialogue, context_mode=context_mode)


def install_date_adapter(dialogues):
    DIALOGUES.clear()
    DIALOGUES.update(dialogues)
    gate.build_prompt = dated_build_prompt
    solver.build_prompt = dated_build_prompt


class NoRetryLedger(Round0RequestLedger):
    """Preserve the established ledger, prohibit retry and stop on technical failure."""
    def __init__(self, *args, limit, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.used = 0

    def invoke(self, *, contract, prompt, call):
        existing = self.read(str(contract['root_id']), request_cache_key(contract))
        if existing is not None and existing.get('status') != 'completed':
            raise RuntimeError('noncompleted_request_cannot_be_resent')
        if existing is None:
            if self.used >= self.limit:
                raise RuntimeError('local_generation_budget_exhausted')
            self.used += 1
        try:
            response, reused = super().invoke(contract=contract, prompt=prompt, call=call)
        except RecordedSolverFailure as error:
            raise RuntimeError('technical_failure_stop_without_retry') from error
        if response.metadata.get('finish_reason') not in ('stop', 'eos'):
            raise RuntimeError('nonexplicit_or_truncated_completion_stop')
        return response, reused

    def record_decision(self, root_id, cache_key, decision):
        super().record_decision(root_id, cache_key, decision)
        if not decision.get('scored'):
            raise RuntimeError('unscorable_output_stop_without_retry')


def verify_records(records):
    for record in records:
        if sha256_file(Path(record['path'])) != record['sha256']:
            raise RuntimeError('frozen_source_changed:' + record['path'])


def prepare():
    from transformers import AutoTokenizer
    if RUN.exists():
        raise RuntimeError('run_exists_do_not_overwrite')
    sources = []

    def read(path):
        sources.append({'path': str(path), 'sha256': sha256_file(path)})
        return load_json(path)

    final = read(SOURCE / 'reports/FINAL_REPORT.json')
    if final['roots_reached_qa'] != 3 or final['program_accepted'] != 29:
        raise RuntimeError('source_scope_changed')
    reviews = read(SOURCE / 'reports/CONTENT_ALL_ROWS.json')
    reviews = {r['candidate_id']: r for r in reviews if r['in_final_pool']}
    candidates, dialogues, strict_ids, details = [], {}, set(), []
    for root in ROOTS:
        rd = SOURCE / 'prepared_run/qa/roots' / root
        pool = read(rd / 'candidate_pool.json')
        dialogue = read(rd / 'qa_source.json')
        strict = read(SOURCE / 'filtered' / root / 'strict_content_pass_deduplicated.json')
        strict_ids.update(r['candidate_id'] for r in strict)
        rendered = render_dialogue(dialogue, numbered=False)
        text_path = rd / 'qa_transcript.txt'
        sources.append({'path': str(text_path), 'sha256': sha256_file(text_path)})
        if rendered.strip() != text_path.read_text().strip():
            raise RuntimeError('writer_source_rendering_mismatch:' + root)
        if any(r['root_id'] != root or r['candidate_id'] not in reviews for r in pool):
            raise RuntimeError('candidate_review_identity_mismatch')
        candidates.extend(pool)
        dialogues[root] = dialogue
        details.append({'root_id': root, 'candidate_count': len(pool),
                        'by_type': dict(Counter(r['question_type'] for r in pool)),
                        'turn_count': len(dialogue['turns'])})
    if len(candidates) != 29 or len({r['candidate_id'] for r in candidates}) != 29 or len(strict_ids) != 13:
        raise RuntimeError('candidate_count_mismatch')
    identity, models = _identity_preflight()
    models['M0'] = {'model_name': 'M0', 'adapter_enabled': False, 'adapter_path': None,
                    'merged_model_path': str(BASE_MODEL), 'base_model_identity': identity['base_model_identity']}
    identity['models'] = models
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    install_date_adapter(dialogues)
    protocol = gate.build_solver_contract(tokenizer, model_path=BASE_MODEL, tokenizer_path=BASE_MODEL)
    protocol['evaluation_adapter'] = {
        'version': 'gate_downstream_dated_solver_v1', 'path': str(Path(__file__).resolve()),
        'sha256': sha256_file(Path(__file__)),
        'date_renderer_path': str(Path(sys.modules[render_dialogue.__module__].__file__)),
        'date_renderer_sha256': sha256_file(Path(sys.modules[render_dialogue.__module__].__file__)),
        'full_context': 'Complete original Writer transcript with frozen date headers; R02 UNKNOWN preserved.',
        'no_context': 'Byte-identical original no-context template; no dates or dialogue supplied.',
        'turn_format': 'turn_ref | speaker | text, exactly the frozen Writer qa_transcript.txt',
        'semantic_filter': False, 'gold_changed': False, 'retries': 0,
        'failure_policy': 'Stop on first technical/parse/truncation failure. Never resend an uncertain request.',
    }
    gates = [gate.gate_candidate(r, dialogues[r['root_id']], tokenizer, root_id=r['root_id']) for r in candidates]
    if any(r['gate_status'] != gate.PROGRAM_PASS for r in gates):
        raise RuntimeError('program_gate_failed:' + json.dumps([r for r in gates if r['gate_status'] != gate.PROGRAM_PASS]))
    RUN.mkdir(parents=True)
    write_once_jsonl(RUN / 'candidate_manifest.jsonl', candidates)
    write_once_json(RUN / 'model_identity_report.json', identity)
    write_once_json(RUN / 'solver_protocol.json', protocol)
    for root, d in dialogues.items():
        write_once_json(RUN / 'dialogues' / (root + '.json'), d)
        write_once_text(RUN / 'dialogues' / (root + '.txt'), render_dialogue(d) + '\n')
    for g in gates:
        dpath = RUN / 'dialogues' / (g['root_id'] + '.json')
        g.update(candidate_pool_path=str(RUN / 'candidate_manifest.jsonl'),
                 candidate_pool_sha256=sha256_file(RUN / 'candidate_manifest.jsonl'),
                 dialogue_path=str(dpath), dialogue_sha256=sha256_file(dpath))
    write_once_jsonl(RUN / 'program_gate_results.jsonl', gates)
    labels = [{**reviews[r['candidate_id']], 'strict_pass_deduplicated': r['candidate_id'] in strict_ids}
              for r in candidates]
    write_once_jsonl(RUN / 'review_labels.jsonl', labels)
    for model in MODELS:
        write_once_json(RUN / model / 'solver_contract.json', {
            'version': 'gate_downstream_dated_solver_v1', 'model_identity': models[model],
            'protocol_sha256': sha256_file(RUN / 'solver_protocol.json')})
    count = sum(candidate_request_count(r) for r in candidates)
    input_tokens = sum(p['prompt_token_count'] * ROLLOUTS for g in gates for p in g['prompt_audit'])
    token_max = max(g['max_prompt_token_count'] for g in gates)
    if count != 744:
        raise RuntimeError('unexpected_call_count')
    verify_records(sources)
    pre = {'status': 'PASS', 'created_at': now(), 'root_count': 3, 'root_details': details,
           'candidate_count': 29, 'by_type': dict(Counter(r['question_type'] for r in candidates)),
           'models': list(MODELS), 'request_count_per_model': count, 'maximum_local_generations': count * 3,
           'input_token_budget_per_model': input_tokens, 'input_token_budget_total': input_tokens * 3,
           'maximum_output_tokens': count * 3 * MAX_TOKENS, 'maximum_prompt_tokens': token_max,
           'output_reserve': MAX_TOKENS, 'max_model_len': MAX_MODEL_LEN,
           'sources': sources, 'review_labels_used_for_input_selection': False,
           'training_calls': 0, 'external_api_calls': 0, 'automatic_retries': 0,
           'source_and_gold_unchanged': True, 'automatic_next_experiment': False,
           'files': {str(p.relative_to(RUN)): sha256_file(p) for p in RUN.rglob('*') if p.is_file()}}
    write_once_json(RUN / 'preflight_report.json', pre)
    print(json.dumps({k:v for k,v in pre.items() if k not in ('sources','files')}, ensure_ascii=False), flush=True)


def load_preflight():
    pre = load_json(RUN / 'preflight_report.json')
    verify_records(pre['sources'])
    verify_records([{'path': str(RUN / p), 'sha256': s} for p,s in pre['files'].items()])
    protocol = load_json(RUN / 'solver_protocol.json')
    if sha256_file(Path(__file__)) != protocol['evaluation_adapter']['sha256']:
        raise RuntimeError('runner_drift')
    verify_records(protocol['contract_source_files'])
    adapter = protocol['evaluation_adapter']
    if sha256_file(Path(adapter['date_renderer_path'])) != adapter['date_renderer_sha256']:
        raise RuntimeError('date_renderer_drift')
    return pre


def metrics(rows):
    out = {'candidate_count': len(rows), 'by_type': dict(Counter(r['question_type'] for r in rows)),
           'difficulty_buckets_against_original_gold': dict(Counter(r['difficulty_bucket'] for r in rows))}
    for mode in ('no_context', 'full_context'):
        correct = sum(r[mode + '_correct_count'] for r in rows)
        total = sum(r[mode + '_total_count'] for r in rows)
        out[mode] = {'gold_matches': correct, 'responses': total,
                     'gold_match_rate': correct / total if total else None}
    return out


def live(model):
    from transformers import AutoTokenizer
    pre = load_preflight()
    md = RUN / model
    with RunLock(RUN):
        if (md / 'EXECUTION_STARTED.json').exists():
            raise RuntimeError('model_already_started_no_automatic_resume')
        identity = load_json(RUN / 'model_identity_report.json')
        info = identity['models'][model]
        expected = info['base_model_identity'] if model == 'M0' else info['merged_model_identity']
        if gate.directory_identity(Path(info['merged_model_path'])) != expected:
            raise RuntimeError('model_identity_drift')
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
        rows = load_jsonl(RUN / 'candidate_manifest.jsonl')
        gates = {r['candidate_id']: r for r in load_jsonl(RUN / 'program_gate_results.jsonl')}
        labels = {r['candidate_id']: r for r in load_jsonl(RUN / 'review_labels.jsonl')}
        dialogues = {root: load_json(RUN / 'dialogues' / (root + '.json')) for root in ROOTS}
        install_date_adapter(dialogues)
        assert_gpu_idle()
        solver.ensure_vllm_environment()
        contract_sha = sha256_file(md / 'solver_contract.json')
        ledger = NoRetryLedger(md, solver_contract_sha256=contract_sha, limit=pre['request_count_per_model'])
        if ledger.states():
            raise RuntimeError('unexpected_prior_requests')
        start = time.monotonic()
        write_once_json(md / 'EXECUTION_STARTED.json', {'started_at': now(), 'model': model,
                        'authorized_by': 'User requested Solver evaluation of the three completed roots.',
                        'maximum_local_generations': pre['request_count_per_model'], 'http_calls': 0})
        backend = None
        results, summaries = [], []
        try:
            backend = VLLMSolverBackend(model_path=Path(info['merged_model_path']), tokenizer_path=BASE_MODEL,
                max_model_len=MAX_MODEL_LEN, temperature=SINGLE_TEMPERATURE, top_p=TOP_P,
                max_tokens=MAX_TOKENS, gpu_memory_utilization=.80)
            for row in sorted(rows, key=lambda r: (ROOTS.index(r['root_id']), r['candidate_id'])):
                cid = row['candidate_id']
                result = solver.evaluate_candidate(candidate=row, expanded=dialogues[row['root_id']],
                    gate_row=gates[cid], tokenizer=tokenizer, backend=backend, ledger=ledger,
                    solver_contract_sha256=contract_sha)
                write_once_json(md / 'candidate_results' / (sha256_bytes(cid.encode()) + '.json'), result)
                label = labels[cid]
                summary = {**summarize_candidate_result(result), 'candidate_id': cid, 'root_id': row['root_id'],
                           'question_type': row['question_type'], 'model': model,
                           'codex_offline_verdict': label['verdict'],
                           'whole_source_support': label.get('whole_source_support'),
                           'strict_pass_deduplicated': label['strict_pass_deduplicated']}
                results.append(result)
                summaries.append(summary)
                progress = {'model': model, 'completed_candidates': len(results), 'total_candidates': len(rows),
                            'local_generations': ledger.used, 'elapsed_seconds': round(time.monotonic()-start, 2),
                            'candidate_id': cid}
                atomic_json(RUN / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            counts = ledger.summary()
            if counts['completed_request_count'] != pre['request_count_per_model'] or counts['retried_generation_count'] or counts['ambiguous_request_count']:
                raise RuntimeError('ledger_completion_mismatch')
            write_once_jsonl(md / 'candidate_results.jsonl', results)
            write_once_jsonl(md / 'summary.jsonl', summaries)
            reqs = [q for r in results for q in r['requests']]
            cost = {'local_generations': len(reqs), 'http_requests': 0, 'retries': 0,
                    'input_tokens': sum(q['backend_metadata']['prompt_token_count'] for q in reqs),
                    'output_tokens': sum(q['backend_metadata']['completion_token_count'] for q in reqs),
                    'elapsed_seconds': round(time.monotonic()-start, 2),
                    'infrastructure_cost': 'UNKNOWN', 'paid_api_cost': 0}
            report = {'status': 'COMPLETE', 'model': model, 'cost': cost, 'ledger': counts,
                      'all_original_questions': metrics(summaries),
                      'strict_pass_deduplicated': metrics([r for r in summaries if r['strict_pass_deduplicated']]),
                      'by_review': {s: metrics([r for r in summaries if r['codex_offline_verdict'] == s]) for s in ('PASS','HOLD','FAIL')},
                      'by_type': {s: metrics([r for r in summaries if r['question_type'] == s]) for s in ('single_choice','multi_select','ordering')},
                      'by_root': {s: metrics([r for r in summaries if r['root_id'] == s]) for s in ROOTS},
                      'completed_at': now(), 'scoring': 'Agreement with unchanged original Gold; diagnostic labels are not independent truth.'}
            verify_records(pre['sources'])
            write_once_json(md / 'report.json', report)
        except BaseException as error:
            atomic_json(md / 'STOPPED.json', {'status': 'STOPPED', 'error_type': type(error).__name__,
                        'error': str(error), 'ledger': ledger.summary(), 'completed_candidates': len(results),
                        'timestamp': now(), 'automatic_retry': False})
            raise
        finally:
            if backend is not None:
                backend.close()
        print(json.dumps({'status': 'COMPLETE', 'model': model, 'cost': cost}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('prepare','live'))
    ap.add_argument('--model', choices=MODELS)
    args = ap.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        if not args.model:
            ap.error('--model required for live')
        live(args.model)
