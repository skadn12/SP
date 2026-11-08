"""Frozen nine-question input adapter for the existing local Solver runner.

The existing runtime owns generation, date rendering, permutations, seed policy,
parsing, scoring, and no-retry ledgers. No second generation implementation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'h4_qa_acceptance_policy_v1'
TRANSCRIPT_SOURCE = HERE.parent / 'h4_qa_feedback_constraints_v2'
RUNTIME_PATH = HERE.parent / 'h4_gate_downstream_solver_eval_v1/evaluate.py'
spec = importlib.util.spec_from_file_location('existing_local_solver_runtime', RUNTIME_PATH)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
legacy.HERE, legacy.RUN, legacy.SOURCE = HERE, HERE / 'run', SOURCE
# Keep the established helper API available to the offline verifier and report.
for name in dir(legacy):
    if not name.startswith('__') and name not in ('prepare', 'live'):
        globals()[name] = getattr(legacy, name)


def prepare():
    from transformers import AutoTokenizer
    if RUN.exists():
        raise RuntimeError('run_exists_do_not_overwrite')
    sources = []

    def read(path):
        sources.append({'path': str(path), 'sha256': sha256_file(path)})
        return load_json(path)

    final = read(SOURCE / 'REPORT.json')
    if final['new_total'] != 9 or [r['new_total'] for r in final['roots']] != [3, 4, 2]:
        raise RuntimeError('selected_nine_question_scope_changed')
    candidates, dialogues, labels, details = [], {}, [], []
    for root in ROOTS:
        pool = read(SOURCE / 'final' / root / 'new_vs_previous_formal.json')
        read(SOURCE / 'final' / root / 'PROVENANCE.json')
        rd = TRANSCRIPT_SOURCE / 'prepared_run/live/roots' / root
        dialogue = read(rd / 'qa_source.json')
        original = {q['candidate_id']: q for q in read(rd / 'candidate_pool.json')}
        text_path = rd / 'qa_transcript.txt'
        sources.append({'path': str(text_path), 'sha256': sha256_file(text_path)})
        if render_dialogue(dialogue, numbered=False).strip() != text_path.read_text().strip():
            raise RuntimeError('writer_source_rendering_mismatch:' + root)
        corrected = read(SOURCE / 'ordering' / root / 'deduplicated_candidates.json')
        read(SOURCE / 'ordering' / root / 'CORRECTION_AUDIT.json')
        corrected_map = {q['candidate_id']: q for q in corrected}
        for q in pool:
            if q['root_id'] != root or q['candidate_id'] not in original:
                raise RuntimeError('candidate_source_identity_mismatch')
            old = original[q['candidate_id']]
            if q['question_type'] == 'ordering':
                if q != corrected_map[q['candidate_id']]:
                    raise RuntimeError('ordering_correction_copy_drift')
                if any(q[k] != old[k] for k in ('question', 'items', 'gold')):
                    raise RuntimeError('ordering_text_or_gold_changed')
            elif q != old:
                raise RuntimeError('choice_candidate_changed')
            labels.append({'candidate_id': q['candidate_id'], 'root_id': root,
                'verdict': 'PASS' if q['question_type'] == 'ordering' else 'NOT_APPLIED',
                'whole_source_support': 'PASS' if q['question_type'] == 'ordering' else 'NOT_ASSESSED_FOR_ADMISSION',
                'strict_pass_deduplicated': q['question_type'] == 'ordering',
                'selection_policy': 'CODEX_CORRECTED_ORDERING' if q['question_type'] == 'ordering' else 'PROGRAM_ACCEPTED_CHOICE_NO_SEMANTIC_GATE',
                'reason': 'Frozen selected question; choice semantic/evidence-completeness diagnoses are not acceptance gates.'})
        candidates.extend(pool)
        dialogues[root] = dialogue
        details.append({'root_id': root, 'candidate_count': len(pool),
            'by_type': dict(Counter(q['question_type'] for q in pool)), 'turn_count': len(dialogue['turns'])})
    if (len(candidates) != 9 or len({q['candidate_id'] for q in candidates}) != 9
            or Counter(q['question_type'] for q in candidates) != Counter(single_choice=4, multi_select=3, ordering=2)):
        raise RuntimeError('candidate_count_or_type_mismatch')
    identity, models = legacy._identity_preflight()
    models['M0'] = {'model_name': 'M0', 'adapter_enabled': False, 'adapter_path': None,
        'merged_model_path': str(BASE_MODEL), 'base_model_identity': identity['base_model_identity']}
    identity['models'] = models
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    install_date_adapter(dialogues)
    protocol = gate.build_solver_contract(tokenizer, model_path=BASE_MODEL, tokenizer_path=BASE_MODEL)
    protocol['evaluation_adapter'] = {
        'version': 'acceptance_selected_nine_dated_solver_v1',
        'path': str(RUNTIME_PATH), 'sha256': sha256_file(RUNTIME_PATH),
        'selection_adapter_path': str(Path(__file__).resolve()),
        'selection_adapter_sha256': sha256_file(Path(__file__)),
        'date_renderer_path': str(Path(sys.modules[render_dialogue.__module__].__file__)),
        'date_renderer_sha256': sha256_file(Path(sys.modules[render_dialogue.__module__].__file__)),
        'full_context': 'Complete frozen Writer transcript and dates, including R02 UNKNOWN.',
        'no_context': 'Unchanged original no-context template.',
        'choice_semantic_gate': False, 'choice_evidence_completeness_gate': False,
        'ordering_input': 'Two Codex-corrected evidence-binding copies; original text/items/Gold unchanged.',
        'selection_scope': 'Nine final questions after within-batch and previous-formal-round deduplication.',
        'gold_changed_for_solver': False, 'retries': 0,
        'failure_policy': 'Stop on technical/parse/truncation failure; no uncertain request resend.',
    }
    gates = [gate.gate_candidate(q, dialogues[q['root_id']], tokenizer, root_id=q['root_id']) for q in candidates]
    if any(g['gate_status'] != gate.PROGRAM_PASS for g in gates):
        raise RuntimeError('program_gate_failed:' + json.dumps([g for g in gates if g['gate_status'] != gate.PROGRAM_PASS]))
    for path in (Path(__file__), RUNTIME_PATH, HERE / 'check_offline.py'):
        sources.append({'path': str(path), 'sha256': sha256_file(path)})
    count = sum(candidate_request_count(q) for q in candidates)
    if count != 234:
        raise RuntimeError('unexpected_local_generation_count')
    RUN.mkdir(parents=True)
    write_once_jsonl(RUN / 'candidate_manifest.jsonl', candidates)
    write_once_json(RUN / 'model_identity_report.json', identity)
    write_once_json(RUN / 'solver_protocol.json', protocol)
    for root, dialogue in dialogues.items():
        write_once_json(RUN / 'dialogues' / (root + '.json'), dialogue)
        write_once_text(RUN / 'dialogues' / (root + '.txt'), render_dialogue(dialogue) + '\n')
    for g in gates:
        path = RUN / 'dialogues' / (g['root_id'] + '.json')
        g.update(candidate_pool_path=str(RUN / 'candidate_manifest.jsonl'),
            candidate_pool_sha256=sha256_file(RUN / 'candidate_manifest.jsonl'),
            dialogue_path=str(path), dialogue_sha256=sha256_file(path))
    write_once_jsonl(RUN / 'program_gate_results.jsonl', gates)
    write_once_jsonl(RUN / 'review_labels.jsonl', labels)
    for model in MODELS:
        write_once_json(RUN / model / 'solver_contract.json', {
            'version': 'acceptance_selected_nine_dated_solver_v1', 'model_identity': models[model],
            'protocol_sha256': sha256_file(RUN / 'solver_protocol.json')})
    input_tokens = sum(p['prompt_token_count'] * ROLLOUTS for g in gates for p in g['prompt_audit'])
    verify_records(sources)
    pre = {'status': 'PASS', 'created_at': now(), 'root_count': 3, 'root_details': details,
        'candidate_count': len(candidates), 'by_type': dict(Counter(q['question_type'] for q in candidates)),
        'models': list(MODELS), 'request_count_per_model': count, 'maximum_local_generations': count * len(MODELS),
        'input_token_budget_per_model': input_tokens, 'input_token_budget_total': input_tokens * len(MODELS),
        'maximum_output_tokens': count * len(MODELS) * MAX_TOKENS,
        'maximum_prompt_tokens': max(g['max_prompt_token_count'] for g in gates),
        'output_reserve': MAX_TOKENS, 'max_model_len': MAX_MODEL_LEN,
        'sources': sources, 'training_calls': 0, 'external_api_calls': 0, 'automatic_retries': 0,
        'source_and_gold_unchanged': True, 'automatic_next_experiment': False,
        'choice_semantic_gate': False, 'ordering_already_codex_corrected': 2,
        'authorization': 'User requested one Solver evaluation of the latest nine selected questions; established local three-model protocol.',
        'files': {str(p.relative_to(RUN)): sha256_file(p) for p in RUN.rglob('*') if p.is_file()}}
    write_once_json(RUN / 'preflight_report.json', pre)
    print(json.dumps({k: v for k, v in pre.items() if k not in ('sources', 'files')}, ensure_ascii=False), flush=True)


def live(model):
    checks = load_json(HERE / 'OFFLINE_CHECKS.json')
    if (checks['status'] != 'PASS' or checks['candidates_unchanged'] != 9
            or checks['real_model_calls'] != 0 or checks['mock_responses_scored'] != 234):
        raise RuntimeError('complete_offline_checks_required')
    return legacy.live(model)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('prepare', 'live'))
    ap.add_argument('--model', choices=MODELS)
    args = ap.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        if not args.model:
            ap.error('--model required for live')
        live(args.model)
