"""Build a reviewable selected-question delivery from completed frozen evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def deliver(generation, solver, output):
    assert not output.exists(), 'delivery_already_exists'
    gen = read(generation / 'REPORT.json'); pool = read(generation / 'SOLVER_POOL.json')
    results = read(solver / 'RESULTS.json'); manifest = read(solver / 'manifest.json')
    complete = read(solver / 'COMPLETE.json')
    assert manifest['pool_sha256'] == sha(generation / 'SOLVER_POOL.json')
    assert not results['call_set_errors']
    assert complete['actual_call_count'] == manifest['request_count']
    rows = {q['candidate_id']: q for q in pool['questions']}
    assert len(rows) == len(results['questions']) == gen['ready_for_solver']
    selected, per_root, rejected = [], {}, []
    for result in results['questions']:
        assert result['actual_full_calls'] == result['expected_full_calls']
        candidate = rows[result['candidate_id']]
        if result['selected_by_policy'] is True:
            assert 0 < Fraction(result['full_partial_mean']) < 1
            selected.append({**candidate, 'training_selection': result,
                'training_target': 'complete_canonical_gold', 'loss_weight_from_partial_score': False})
        else:
            rejected.append(result)
    for row in gen['roots']:
        root = row['root_id']; kind = row['question_type']
        counts = per_root.setdefault(root, {})
        counts[kind] = {'candidate_slots': row['candidate_slots'], 'mechanical_pass': row['mechanical_pass'],
            'ready_for_solver': row['ready_for_solver'],
            'selected': sum(q['root_id'] == root and q['question_type'] == kind for q in selected),
            'generator_abstentions': len(row['abstentions']),
            'mechanical_rejection_reasons': dict(Counter(e for r in row['mechanical_rejected'] for e in r['errors'])),
            'qwen_review_rejected': len(row['review_rejected'])}
    for root, counts in per_root.items():
        total = sum(row['selected'] for row in counts.values())
        counts['selected_total'] = total
        counts['trial_target_4_to_8_met'] = 4 <= total <= 8
    source_bindings = {q['root_id']: {'path': q['source_path'], 'sha256': q['source_sha256']} for q in pool['questions']}
    report = {'version': 'h4_r02_question_contract_v14_selected_delivery', 'status': 'EVALUATION_COMPLETE',
        'generator_model': pool['provenance']['requested_model'], 'ordering_reviewer_model': pool['provenance']['ordering_review_owner'],
        'solver_model': results['solver_model'], 'solver_model_path': results['model_path'],
        'policy_version': results['policy_version'], 'selection_rule': '0 < full_partial_mean < 1 after all planned valid full calls',
        'per_root': per_root, 'candidate_slots': gen['candidate_slots'], 'ready_for_solver': len(rows),
        'selected_total': len(selected), 'selected_multi_select': sum(q['question_type'] == 'multi_select' for q in selected),
        'selected_ordering': sum(q['question_type'] == 'ordering' for q in selected),
        'full_solver_calls': complete['actual_call_count'], 'ground_calls': 0, 'training_calls': 0,
        'feedback_in_every_generation_request': gen['feedback_in_every_request'],
        'multi_select_semantic_review': False, 'codex_question_edits': False, 'semantic_dedup_certified': False,
        'source_bindings': source_bindings,
        'generation_report': {'path': str(generation / 'REPORT.json'), 'sha256': sha(generation / 'REPORT.json')},
        'solver_results': {'path': str(solver / 'RESULTS.json'), 'sha256': sha(solver / 'RESULTS.json')},
        'solver_identity': {'path': str(solver / 'solver_identity.json'), 'sha256': sha(solver / 'solver_identity.json')},
        'selected_wording': 'Passed this training-selection policy; permutation behavior recorded separately.'}
    retry_report = generation / 'RETRY_REPORT.json'
    report['generation_http_attempts_including_parent'] = read(retry_report)['combined_http_attempts'] if retry_report.exists() else gen['http_attempts']
    output.mkdir(parents=True)
    write(output / 'REPORT.json', report)
    write(output / 'SELECTED_QUESTIONS.json', {'questions': selected, 'training_started': False,
        'selection_policy': report['selection_rule'], 'solver_model': results['solver_model']})
    write(output / 'NOT_SELECTED.json', {'questions': rejected})
    lines = ['# R02 question generation and full-only Solver selection', '',
        f"Generator/reviewer: `{report['generator_model']}`. Solver: `{report['solver_model']}`.", '',
        '| Root | Multi ready / selected | Ordering ready / selected | Selected total |', '|---|---:|---:|---:|']
    for root, counts in per_root.items():
        a, b = counts['multi_select'], counts['ordering']
        lines.append(f"| {root.split('-')[-1]} | {a['ready_for_solver']} / {a['selected']} | {b['ready_for_solver']} / {b['selected']} | {counts['selected_total']} |")
    lines += ['', f"{report['full_solver_calls']} valid full-context answers; 0 ground calls; 0 training calls.",
        'Every selected question retains its complete Gold. No Solver answer replaces the training target.',
        'Multi-select uses mechanical checks only; ordering uses independent Qwen review. Semantic novelty is not certified.',
        'The 4–8 selected-question range is an experimental target, not a quota that overrides filtering.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.generation.resolve(), args.solver.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
