#!/usr/bin/env python3
"""Build Day 1 boundary records from frozen historical manifests; no inference."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERIES = BASE / 'offline-deliveries'
SOURCES = {
    'week_plan': OUT.parents[1] / 'PLAN_ZH.md',
    'authoritative_split': BASE / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl',
    'v125_training_roots': DELIVERIES / 'h4_codex_train_pool_v125/data/ROOT_MANIFEST.json',
    'v125_summary': DELIVERIES / 'h4_codex_train_pool_v125/SUMMARY.json',
    'v124_all_heldout_sources': DELIVERIES / 'h4_codex_dev_test_ab_v124/MANIFEST.json',
    'v127_dev_scope': DELIVERIES / 'h4_codex_dev_eval_v127/SCOPE.json',
    'v131_old_questions': DELIVERIES / 'h4_codex_historical_frontier_eval_v131/QUESTIONS.json',
    'v134_heldout_scope': DELIVERIES / 'h4_codex_held_roots_eval_v134/evaluation/SCOPE.json',
    'v134_final_status': DELIVERIES / 'h4_codex_held_roots_eval_v134/FINAL_STATUS.json',
    'v140_design_exposure': DELIVERIES / 'h4_codex_structure_pilot_v140/PLAN.json',
    'historical_training_diagnostics': BASE / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/candidate_transition_matrix.jsonl',
}


def read(name):
    return json.loads(SOURCES[name].read_text())


def binding(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    now = datetime.now(timezone.utc)
    stamp = {'created_at_utc': now.isoformat(), 'created_at_beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}
    split = [json.loads(line) for line in SOURCES['authoritative_split'].read_text().splitlines() if line.strip()]
    split_by_root = {row['root_id']: row for row in split}
    roots = {s: sorted(row['root_id'] for row in split if row['split'] == s) for s in ('train', 'dev', 'test')}
    trained_rows = read('v125_training_roots')
    trained = {row['root_id'] for row in trained_rows}
    unused = sorted(set(roots['train']) - trained)
    v140 = {row['root_id'] for row in read('v140_design_exposure')['roots']}
    old = read('v131_old_questions')['candidates']
    old_mc_order = [row for row in old if row['question_type'] in ('multi_select', 'ordering')]
    old_roots = {row['root_id'] for row in old_mc_order}
    history_counts = Counter()
    with SOURCES['historical_training_diagnostics'].open() as source:
        for line in source:
            row = json.loads(line)
            history_counts[row['root_id']] += 1
    v124 = {row['root_id']: row for row in read('v124_all_heldout_sources')['roots']}
    v134 = read('v134_heldout_scope')
    wave1, wave2, memory = roots['train'][:20], roots['train'][20:30], roots['train'][:6]
    train_source_rows = {row['root_id']: row for row in trained_rows}
    records = []
    for rid in sorted(split_by_root):
        original = split_by_root[rid]
        s = original['split']
        historical = []
        if s == 'train':
            if history_counts[rid]:
                historical.append('historical_train_solver_diagnostics')
            if rid in trained:
                historical.append('v125_training_pool')
        else:
            historical.append('v124_source_generation_review_accounting')
            historical.append('historically_exposed_heldout_regression')
        if rid in v140:
            historical.append('v140_design_and_solver_diagnostic')
        records.append({
            'root_id': rid, 'original_split': s,
            'original_manifest_analysis_exposed': original.get('analysis_exposed'),
            'original_flag_interpretation': 'Historical snapshot only; false does not undo later exposure.',
            'current_role': 'TRAIN_ONLY' if s == 'train' else 'REGRESSION_ONLY',
            'historical_exposure_records': historical,
            'historical_train_diagnostic_row_count': history_counts[rid],
            'v125_training_used': rid in trained,
            'v140_exposed': rid in v140,
            'old_mc_order_review_source_overlap': rid in old_roots,
            'week_wave': 1 if rid in wave1 else 2 if rid in wave2 else None,
            'memory_pilot': rid in memory,
            'source_lineage_bindings': ([train_source_rows[rid]['source_binding']] if rid in trained else
                [v124[rid]['baseline_binding']] if rid in v124 else []),
            'not_a_pristine_unseen_root': True,
        })
    bindings = {key: binding(path) for key, path in SOURCES.items()}
    boundary = {
        'schema_version': 'selfplay_week_day1_root_boundary_v1', **stamp,
        'status': 'BOUNDARY_ACCOUNTED_WITH_HISTORICAL_EXPOSURE',
        'source_bindings': bindings,
        'counts': {**{s: len(roots[s]) for s in roots}, 'total': len(split), 'v125_actual_train_roots': len(trained), 'v125_unused_train_roots': len(unused), 'v140_exposed_dev_roots': len(v140), 'pristine_existing_roots': 0},
        'split_root_ids': roots,
        'v125_unused_train_root_ids': unused,
        'v125_unused_does_not_mean_unseen': 'Both omitted training roots have historical solver diagnostic records; no reclassification as TEST.',
        'v140_exposed_dev_root_ids': sorted(v140),
        'exposure_scope': 'All 101 roots have prior development exposure. This is a conservative source/experiment boundary; it does not claim every generated question or every turn was individually inspected by a human.',
        'heldout_evaluation_coverage_limit': 'DEV/TEST source generation/review accounting is broader than delivered question or completed solver coverage. Do not interpret all 31 roots as all having scored questions.',
        'roots': records,
    }
    selection = {
        'schema_version': 'selfplay_week_day1_root_selection_v1', **stamp,
        'status': 'ROOT_IDS_FROZEN_SOURCE_ADMISSION_STILL_REQUIRED',
        'selection_rule': 'Sort complete root_id strings lexicographically in Python; use the authoritative 70 TRAIN roots. No model score or question correctness is used.',
        'numeric_suffix_sort': False,
        'source_binding': bindings['authoritative_split'],
        'wave1': {'root_ids': wave1, 'count': len(wave1), 'per_arm_question_upper_bound': 120, 'same_roots_for_A_and_B': True},
        'wave2': {'root_ids': wave2, 'count': len(wave2), 'per_arm_question_upper_bound': 60, 'same_roots_for_A_and_B': True},
        'memory_pilot': {'root_ids': memory, 'count': len(memory), 'rule': 'First six wave1 roots; all three context arms use these same roots.', 'arms': ['recent_two_scenes', 'memory_plus_recent_two_scenes', 'full_context']},
        'selected_roots_without_v125_training_source': sorted((set(wave1) | set(wave2)) - trained),
        'source_admission_gate': 'Root ID selection does not approve a source. Before generation, bind and review the exact dialogue prefix; unused v125 roots require an explicitly admitted historical source or reviewed expansion. Preserve failures; no score-based replacement.',
        'old_question_review': {
            'all_v131_questions': len(old), 'mc_order_questions': len(old_mc_order), 'mc_order_roots': len(old_roots),
            'wave1_overlap_root_ids': sorted(old_roots & set(wave1)),
            'wave2_overlap_root_ids': sorted(old_roots & set(wave2)),
            'memory_overlap_root_ids': sorted(old_roots & set(memory)),
            'interpretation': 'Allowed within TRAIN for reuse/diagnosis; track facts and question families. Reuse is not independent transfer evidence. Review PASS and source admission remain required.',
        },
    }
    namespace = 'gxy-selfplay:selfplay-week-20261019'
    policy = {
        'schema_version': 'selfplay_week_day1_evaluation_policy_v1', **stamp,
        'status': 'DAY1_POLICY_FROZEN_FUTURE_DATA_LOCKS_PENDING',
        'namespace': namespace,
        'historical_boundary': {
            'train': '70 original TRAIN roots; source-grounded diagnosis, memory pilot, A/B generation, and training permitted after content admission.',
            'dev': '15 historical DEV roots; REGRESSION_ONLY. Predeclared aggregate metric may select checkpoints. No item-level errors, answers, source examples, or generated diagnosis may enter Challenger memory.',
            'test': '16 historical TEST roots; REGRESSION_ONLY. Freeze question/request manifests before runs; evaluate frozen final models for reporting. No checkpoint selection, feedback, repair, or generation targeting based on results.',
            'v140': 'Four historically exposed DEV roots remain DEV regression; their design diagnostic is prior methodological exposure, not new unseen-root evidence.',
        },
        'roles': {
            'A_generator': {'reads': ['shared_design_library', 'admitted_train_sources', 'shared_coarse_quotas'], 'forbidden': ['solver_failure_memory', 'DEV_or_TEST_item_feedback', 'locked_eval_questions']},
            'B_generator': {'reads': ['shared_design_library', 'admitted_train_sources', 'own_branch_train_weakness_memory', 'shared_coarse_quotas'], 'forbidden': ['DEV_or_TEST_item_feedback', 'other_branch_future_results', 'locked_eval_questions']},
            'reviewer': {'reads': ['public_candidate', 'complete_admitted_source'], 'rule': 'Reconstruct answer from source before private comparison; no solver-score quality selection.'},
            'evaluator': {'rule': 'Read only frozen question/request/model bindings for the assigned namespace; report root/family dependence and historical exposure.'},
        },
        'namespaces': {
            'training_feedback_A': namespace + ':train-feedback:A',
            'training_feedback_B': namespace + ':train-feedback:B',
            'memory_pilot': namespace + ':train-memory-pilot',
            'train_root_new_family_locked': namespace + ':locked-new-family-on-train-root',
            'historical_dev_regression': namespace + ':regression:dev',
            'historical_test_regression': namespace + ':regression:test',
            'future_unseen_root': namespace + ':prospective-unseen-root',
        },
        'guards': [
            {'id': 'split_immutable', 'rule': 'No automatic split reassignment. Neither of the two v125-unused Train roots may be called unseen test.'},
            {'id': 'all_permutations_one_family', 'rule': 'One semantic question, its revisions, near-duplicate facts and all display permutations remain one family; never cross train/feedback/locked-eval boundaries.'},
            {'id': 'feedback_taints_family', 'rule': 'Any family or related fact family whose item feedback enters diagnosis is training-feedback thereafter; it cannot remain unexposed transfer evidence.'},
            {'id': 'root_overlap_report', 'rule': 'New locked question families on training roots are labeled seen-root/new-family. Do not call them new-root generalization.'},
            {'id': 'lock_before_solver', 'rule': 'Lock evaluator question/source/Gold/request hashes, selection rule, family exclusions, model IDs and release policy before viewing the relevant scores.'},
            {'id': 'dev_aggregate_only', 'rule': 'Expose only frozen aggregate checkpoint criteria to selection. DEV item analysis is kept outside Challenger memory.'},
            {'id': 'test_final_only', 'rule': 'No TEST feedback, source examples, Gold-specific targeting or score-driven question fixes; test is final regression reporting only.'},
            {'id': 'shared_AB_conditions', 'rule': 'Same selected source roots, exact admitted source suffix, memory production policy, review budget, old-data budget and training budget across A/B. Branch-specific TRAIN diagnosis is the studied difference.'},
            {'id': 'unseen_import_required', 'rule': 'For a genuinely new-root benchmark, import genuinely unused compatible source roots under a new provenance manifest; check IDs, hashes, dialogue/semantic near-duplicates, historical use and family overlap before lock. Do not silently use existing Train roots.'},
            {'id': 'protocol_not_access_control', 'rule': 'These are frozen workflow requirements and validation records, not an implemented filesystem access-control sandbox. Shared agent filesystem is disclosed.'},
        ],
        'future_unseen_root_gate': {
            'status': 'UNRESOLVED_NO_IMPORTED_UNSEEN_ROOTS', 'day1_blocker': False,
            'before_unseen_generalization_claim': ['new_source_provenance', 'historical_exposure_and_duplicate_audit', 'root_and_family_disjointness', 'source_first_Gold_review', 'pre_score_evaluation_lock'],
            'fallback': 'Continue TRAIN diagnosis and controlled A/B preparation; report historical regression and prospectively locked new families honestly. Omit a pristine unseen-root claim if the source gate is not met.',
        },
        'execution': {'inference_launched_here': False, 'training_launched_here': False, 'new_root_download_or_import_here': False, 'historical_artifacts_modified': False},
        'source_bindings': bindings,
    }
    checks = {
        'total_101_unique_roots': len(split) == len(split_by_root) == 101,
        'split_counts_70_15_16': {s: len(roots[s]) for s in roots} == {'train': 70, 'dev': 15, 'test': 16},
        'v125_68_roots_all_train': len(trained) == 68 and trained <= set(roots['train']),
        'v125_two_unused_exact': unused == ['cc-train-episode-194259', 'cc-train-episode-87273'],
        'unused_both_historically_evaluated': all(history_counts[rid] > 0 for rid in unused),
        'v140_four_all_dev': len(v140) == 4 and v140 <= set(roots['dev']),
        'v124_all_31_heldout_accounted': set(v124) == set(roots['dev']) | set(roots['test']),
        'v134_dev_boundary_matches': set(v134['fixed_full_heldout_roots']['dev']) == set(roots['dev']),
        'v134_test_boundary_matches': set(v134['fixed_full_heldout_roots']['test']) == set(roots['test']),
        'v134_historical_test_exposure_true': read('v134_final_status')['test_has_historical_exposure'] is True,
        'all_70_train_have_historical_diagnostics': set(roots['train']) <= set(history_counts),
        'wave_counts_and_disjointness': len(wave1) == 20 and len(wave2) == 10 and not set(wave1) & set(wave2),
        'memory_first_six_wave1': len(memory) == 6 and memory == wave1[:6],
        'no_selected_heldout_roots': (set(wave1) | set(wave2)) <= set(roots['train']),
        'old_review_113_on_56_train_roots': len(old_mc_order) == 113 and len(old_roots) == 56 and old_roots <= set(roots['train']),
        'all_source_files_present': all(path.is_file() for path in SOURCES.values()),
        'source_hashes_rechecked': all(binding(SOURCES[key]) == value for key, value in bindings.items()),
    }
    write('ROOT_BOUNDARY.json', boundary)
    write('ROOT_SELECTION.json', selection)
    write('EVALUATION_POLICY.json', policy)
    outputs = {name: binding(OUT / name) for name in ('ROOT_BOUNDARY.json', 'ROOT_SELECTION.json', 'EVALUATION_POLICY.json')}
    write('VALIDATION.json', {
        'schema_version': 'selfplay_week_day1_boundary_validation_v1', **stamp,
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'check_count': len(checks), 'checks': checks,
        'errors': [key for key, value in checks.items() if not value],
        'artifact_bindings': outputs,
        'day1_gate': 'PASS_BOUNDARY_AND_SELECTION_READY' if all(checks.values()) else 'FAIL',
        'future_gates': ['Exact source-prefix admission before generation', 'Evaluation question/family/request lock before scores', 'New source import before any pristine unseen-root claim'],
        'future_gates_do_not_block_day1_completion': True,
    })
    assert all(checks.values()), checks
    print(json.dumps({'status': 'PASS', 'check_count': len(checks), 'counts': boundary['counts'], 'output_directory': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
