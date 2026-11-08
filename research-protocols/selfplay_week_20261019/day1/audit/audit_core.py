"""Independent read-only Day1 checks; all outputs stay in this audit directory.

Deliberately does not read legacy private Gold, INVENTORY, or semantic reviews.
No model loading, generation, inference, training, or historical file mutation.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import copy
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CHECKS = []
READ_BINDINGS = {}


def sha(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    p = Path(path)
    row = {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
    READ_BINDINGS[str(p)] = row
    return row


def read(rel):
    p = BASE / rel
    bind(p)
    return json.loads(p.read_text())


def check(name, passed, detail=None):
    CHECKS.append({'name': name, 'passed': bool(passed), 'detail': detail})


def check_ref(ref, name):
    p = Path(ref['path'])
    row = bind(p)
    check(name, row['sha256'] == ref['sha256'] and
          ('bytes' not in ref or row['bytes'] == ref['bytes']) and
          ('size_bytes' not in ref or row['bytes'] == ref['size_bytes']))


def main():
    protocol = read('PROTOCOL_V1.json')
    roots = read('evaluation/ROOT_SELECTION.json')
    boundary = read('evaluation/ROOT_BOUNDARY.json')
    policy = read('evaluation/EVALUATION_POLICY.json')
    whitelist = read('TRAIN_ROOTS.json')
    allowed = set(whitelist['root_ids'])
    check_ref(protocol['prior_plan'], 'protocol_plan_hash')
    check_ref(whitelist['source_manifest'], 'train_manifest_hash')
    check_ref(roots['source_binding'], 'selection_manifest_hash')
    for name, rec in boundary['source_bindings'].items():
        check_ref(rec, 'boundary_source_hash:' + name)
    for name, rec in policy['source_bindings'].items():
        check_ref(rec, 'policy_source_hash:' + name)
    split = [json.loads(line) for line in Path(roots['source_binding']['path']).read_text().splitlines() if line.strip()]
    actual_roots = {k: sorted(x['root_id'] for x in split if x['split'] == k) for k in ['train', 'dev', 'test']}
    check('authoritative_70_15_16', [len(actual_roots[k]) for k in ['train', 'dev', 'test']] == [70, 15, 16])
    check('boundary_matches_authoritative_roots', boundary['split_root_ids'] == actual_roots)
    check('train_whitelist_exact', len(allowed) == 70 and allowed == set(actual_roots['train']))
    check('split_disjointness', len(set().union(*[set(v) for v in actual_roots.values()])) == 101)
    for key, expected in [('wave1', actual_roots['train'][:20]), ('wave2', actual_roots['train'][20:30]), ('memory_pilot', actual_roots['train'][:6])]:
        check('lexicographic_score_free_selection:' + key, roots[key]['root_ids'] == expected)
    check('all_101_roots_honestly_exposed', boundary['counts']['pristine_existing_roots'] == 0 and all(x['not_a_pristine_unseen_root'] for x in boundary['roots']))
    check('unused_train_not_relabelled_unseen', len(boundary['v125_unused_train_root_ids']) == 2 and all(x['historical_train_diagnostic_row_count'] > 0 and x['current_role'] == 'TRAIN_ONLY' for x in boundary['roots'] if x['root_id'] in boundary['v125_unused_train_root_ids']))
    check('A_B_diagnosis_difference_declared', protocol['arms']['A']['reads_solver_failure_memory'] is False and protocol['arms']['B']['reads_solver_failure_memory'] is True)
    for rel in [protocol['root_schedule']['file'], protocol['legacy_reuse']['queue'], protocol['model_identity']['binding_file'], protocol['diagnosis']['taxonomy'], protocol['diagnosis']['schema'], protocol['evaluation']['policy']]:
        check('protocol_filename_exists:' + rel, (BASE / rel).is_file())
    check('two_stage_lineage_declared', protocol['update_lineage'] == {'stage1': 'A1 and B1 independently start from S0=M_clean', 'stage2': 'A2 starts from A1; B2 starts from B1; independent optimizer/scheduler state resets per stage', 'frozen_solver_within_stage': True})
    check('new_execution_not_claimed', all(protocol['execution_this_package'][k] is False for k in ['new_solver_inference', 'new_training', 'new_dialogue_expansion']))
    check('CoT_separate_limited_nontraining', protocol['diagnosis']['structured_evidence_branch_is_separate'] and protocol['diagnosis']['max_evidence_output_tokens'] == 1024 and protocol['diagnosis']['diagnosis_repeats_per_question'] == 1 and not protocol['diagnosis']['solver_sees_gold_or_author_evidence_before_response'] and not protocol['diagnosis']['self_generated_reasoning_is_training_target'])
    check('direct_answer_limits_32_tokens', protocol['evaluation']['primary_direct_protocol']['max_output_tokens'] == 32 and protocol['evaluation']['primary_direct_protocol']['finite_choice_constrained_answers'])
    check('generation_and_review_limits', protocol['production']['max_output_tokens_per_authored_question'] == protocol['production']['max_output_tokens_per_reconstruction'] == 3072 and protocol['production']['initial_generation_attempts_per_slot'] == 1 and protocol['production']['maximum_content_revisions_per_slot'] == 1 and protocol['production']['mechanical_error_retries'] == protocol['production']['automatic_solver_retries'] == 0)
    check('paired_training_budget_design', protocol['training']['budget_matching']['full_input_training_token_difference_max_ratio'] == .05 and protocol['training']['budget_matching']['optimizer_steps'] == 'equal per corresponding A/B stage' and protocol['training']['allow_truncation'] is False and protocol['training']['reset_optimizer_and_scheduler'] is True)
    check('same_AB_context_and_roots', protocol['fact_memory_pilot']['main_AB_context_identical'] and roots['wave1']['same_roots_for_A_and_B'] and roots['wave2']['same_roots_for_A_and_B'])
    slots = read('SLOT_PLAN.json')
    check('slots_bind_protocol_and_root_selection', slots['protocol_sha256'] == sha(BASE / 'PROTOCOL_V1.json') and slots['root_selection_sha256'] == sha(BASE / 'evaluation/ROOT_SELECTION.json'))
    rows = slots['slots']
    check('360_unique_unexecuted_slots', len(rows) == len({s['candidate_slot_id'] for s in rows}) == 360 and all(s['question_generated'] is False and s['source_binding'] is None for s in rows))
    expected_slots = []
    idx = 0
    for stage, key in [(1, 'wave1'), (2, 'wave2')]:
        stage_rows = []
        for root in roots[key]['root_ids']:
            for kind in ['multi_select', 'ordering']:
                for local in range(1, 4):
                    sid = f'week1-stage{stage}-{root}-{kind}-{local:02d}'
                    gold = [2, 3, 4, 5][idx % 4] if kind == 'multi_select' else None
                    idx += kind == 'multi_select'
                    stage_rows.append({'slot_id': sid, 'stage': stage, 'root_id': root, 'question_type': kind, 'local_slot': local, 'target_gold_count': gold, 'options_or_items': 6})
        ranked = sorted(stage_rows, key=lambda x: hashlib.sha256(f"2026101900|{stage}|{x['slot_id']}".encode()).hexdigest())
        targeted = {x['slot_id'] for x in ranked[:len(ranked) * 7 // 10]}
        for arm in ['A', 'B']:
            for s in stage_rows:
                target = arm == 'B' and s['slot_id'] in targeted
                expected_slots.append({**s, 'arm': arm, 'candidate_slot_id': arm + '-' + s['slot_id'], 'generation_mode': 'diagnosis_targeted' if target else 'free_diverse', 'target_cause_id': None, 'cause_selection_state': 'WAITING_FOR_REPRODUCED_TRAIN_CAUSE' if target else 'NOT_APPLICABLE', 'source_binding': None, 'source_admission_state': 'PENDING_EXACT_PREFIX_ADMISSION', 'question_generated': False})
    check('all_slot_values_independently_reproduced', rows == expected_slots)
    for arm in ['A', 'B']:
        arm_rows = [s for s in rows if s['arm'] == arm]
        check('arm_slot_arithmetic:' + arm, Counter(s['stage'] for s in arm_rows) == {1: 120, 2: 60} and Counter(s['question_type'] for s in arm_rows) == {'multi_select': 90, 'ordering': 90} and Counter(s['target_gold_count'] for s in arm_rows if s['question_type'] == 'multi_select') == {2: 23, 3: 23, 4: 22, 5: 22})
    check('B_84_42_targeted_waiting', Counter(s['stage'] for s in rows if s['generation_mode'] == 'diagnosis_targeted') == {1: 84, 2: 42} and all(s['cause_selection_state'] == 'WAITING_FOR_REPRODUCED_TRAIN_CAUSE' for s in rows if s['generation_mode'] == 'diagnosis_targeted'))
    legacy = read('reuse/public/QUESTIONS.json')['questions']
    public_keys = {'candidate_id', 'root_id', 'question_type', 'question', 'options', 'items', 'source_binding'}
    check('113_legacy_51_multi_62_order_56_train_roots', len(legacy) == 113 and Counter(q['question_type'] for q in legacy) == {'multi_select': 51, 'ordering': 62} and len({q['root_id'] for q in legacy}) == 56 and all(q['root_id'] in allowed for q in legacy))
    check('legacy_public_field_whitelist', all(set(q) == public_keys for q in legacy))
    for q in legacy:
        check_ref(q['source_binding'], 'legacy_source:' + q['candidate_id'])
    public_freeze = read('reuse/PUBLIC_FREEZE.json')
    for rec in public_freeze['files']:
        check_ref({**rec, 'path': str(BASE / 'reuse' / rec['path'])}, 'public_freeze:' + rec['path'])
    next_plan = read('diagnosis_inputs/INPUT_PLAN.json')
    next_public = read('diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    next_private = read('diagnosis_inputs/PRIVATE_QUESTIONS.json')['questions']
    check('39_next_19_multi_20_order_20_roots', len(next_public) == len(next_private) == next_plan['question_count'] == 39 and Counter(q['question_type'] for q in next_public) == {'multi_select': 19, 'ordering': 20} and len({q['root_id'] for q in next_public}) == next_plan['root_count'] == 20)
    direct = protocol['evaluation']['primary_direct_protocol']
    count = sum(direct[q['question_type'] + '_permutations'] * direct[q['question_type'] + '_samples'] for q in next_public)
    check('582_planned_0_executed_not_final_eval', count == next_plan['planned_direct_calls'] == 582 and next_plan['new_solver_calls'] == 0 and next_plan['use_as_final_eval'] is False and next_plan['max_structured_diagnostic_calls'] == 39)
    projection = {'candidate_id', 'root_id', 'question_type', 'question', 'source_path', 'source_sha256', 'options', 'items'}
    check('next_public_exact_private_projection', next_public == [{k: v for k, v in q.items() if k in projection} for q in next_private])
    for q in next_public:
        check('next_root_train:' + q['candidate_id'], q['root_id'] in allowed)
        check_ref({'path': q['source_path'], 'sha256': q['source_sha256']}, 'next_source:' + q['candidate_id'])
    check_ref(next_plan['source_pool'], 'diagnosis_source_pool_hash')
    pool = json.loads(Path(next_plan['source_pool']['path']).read_text())['questions']
    selected = []
    shortfalls = []
    for root in roots['wave1']['root_ids']:
        for kind in ['multi_select', 'ordering']:
            candidates = [q for q in pool if q['root_id'] == root and q['question_type'] == kind]
            if candidates:
                selected.append(min(candidates, key=lambda q: hashlib.sha256(('2026101900|diagnosis-input|' + q['candidate_id']).encode()).hexdigest())['candidate_id'])
            else:
                shortfalls.append({'root_id': root, 'type': kind})
    check('diagnosis_selection_independently_reproduced', selected == [q['candidate_id'] for q in next_public] and shortfalls == next_plan['shortfalls'])
    spec = importlib.util.spec_from_file_location('independent_day1_diagnosis', BASE / 'diagnosis/validate_diagnosis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    memory = read('WEAKNESS_MEMORY_B_v000.json')
    check('empty_memory_schema_and_train_boundary_valid', not module.validate(memory, allowed))
    check('empty_memory_no_fabricated_causes', all(memory[k] == [] for k in ['failure_cases', 'probes', 'causes']))
    for k in ['taxonomy', 'allowed_train_roots_manifest']:
        check_ref(memory[k], 'empty_memory_ref:' + k)
    val = read('diagnosis/VALIDATION.json')
    check('diagnosis_27_27', val['status'] == 'PASS' and val['test_count'] == val['passed_count'] == 27)
    for name, rec in val['files'].items():
        check_ref({**rec, 'path': str(BASE / 'diagnosis' / name)}, 'diagnosis_validation_hash:' + name)
    for rel in ['evaluation/VALIDATION.json', 'reuse/VALIDATION_RECHECK.json']:
        check('child_validation_pass:' + rel, read(rel)['status'] == 'PASS')
    models = read('MODEL_BINDINGS.json')
    hash_checked, large_stat_only = [], []
    for name, model in models['models'].items():
        for rec in model['files']:
            p = Path(rec['path'])
            check('model_file_present_size:' + name + ':' + p.name, p.is_file() and p.stat().st_size == rec['bytes'])
            if rec['bytes'] <= 32 * 1024 * 1024:
                check_ref(rec, 'model_small_file_hash:' + name + ':' + p.name)
                hash_checked.append(rec)
            else:
                large_stat_only.append(rec)
    for merged, adapter in [('S0_M_clean_merged', 'S0_M_clean_adapter'), ('old_Round2_merged', 'old_Round2_adapter')]:
        d = Path(models['models'][merged]['directory'])
        info = json.loads((d / 'merged_adapter_info.json').read_text())
        check('merge_lineage:' + merged, info['base_model'] == models['models']['base']['directory'] and info['adapter_dir'] == models['models'][adapter]['directory'] and info['merge_info']['skipped_missing_pairs'] == [])
    launch = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_codex_combined_sft_v125/LAUNCH_BINDING.json')
    bind(launch)
    argv = json.loads(launch.read_text())['formal_argv']
    fields = {'max-length': 'max_length', 'epochs': 'epochs', 'batch-size': 'microbatch', 'grad-accum': 'gradient_accumulation', 'lr': 'lr', 'min-lr-ratio': 'min_lr_ratio', 'warmup-ratio': 'warmup_ratio', 'weight-decay': 'weight_decay', 'max-grad-norm': 'max_grad_norm', 'lora-rank': 'lora_rank', 'lora-alpha': 'lora_alpha', 'lora-dropout': 'lora_dropout'}
    check('frozen_recipe_matches_actual_prior_launcher_values', all(float(argv[argv.index('--' + flag) + 1]) == protocol['training'][key] for flag, key in fields.items()) and argv[argv.index('--target-modules') + 1].split(',') == protocol['training']['target_modules'])
    for rel in ['build_protocol.py', 'build_slots.py', 'validate_day1.py', 'assemble_reviews.py', 'evaluation/build_boundary.py', 'diagnosis/build_schemas.py', 'diagnosis/validate_diagnosis.py']:
        p = BASE / rel
        bind(p)
        ast.parse(p.read_text())
        check('python_syntax:' + rel, True)
    trainer = Path('/root/autodl-tmp/gxy-selfplay/current_method_github/train_qwen3_lora_sft.py')
    adapter = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_structure_pilot_v140/evaluation/adapter_protocol.py')
    bind(trainer)
    bind(adapter)
    trainer_text, adapter_text = trainer.read_text(), adapter.read_text()
    known_runtime = {'existing_trainer_left_truncates_overlength': 'full_ids = full_ids[overflow:]' in trainer_text, 'existing_trainer_tokenizes_once_then_shuffles': 'train_features = [tokenize_record' in trainer_text and 'random.shuffle(indices)' in trainer_text, 'existing_adapter_six_item_only': 'SIX_UNIQUE_OPTIONS_OR_ITEMS_REQUIRED' in adapter_text and 'ORDERING_REQUIRES_SIX_GOLD_ITEMS' in adapter_text}
    synthetic = []
    synthetic_roots = {'r0', 'r1', 'r2'}
    fixture, _ = module.fixture()
    improving = copy.deepcopy(fixture)
    c = improving['causes'][0]
    c['confidence_status'] = 'improving'
    c['state_history'][-1]['to'] = 'improving'
    c['new_question_results'] = [{'result_id': 'decline', 'provenance': {**improving['failure_cases'][0]['provenance'], 'family_id': 'new-family'}, 'new_family_for_memory': True, 'before_model_id': 'unrelated-before', 'after_model_id': 'unrelated-after', 'same_protocol': True, 'before_exact_rate': .9, 'after_exact_rate': .1, 'record': improving['taxonomy'], 'interpretation': 'Synthetic worsening, not experimental evidence.'}]
    synthetic.append({'name': 'improving_decline_unrelated_model_record_currently_accepted', 'current_validator_errors': module.validate(improving, synthetic_roots), 'interpretation': 'Before real improving status, require positive change plus explicit current-model lineage and reviewed before/after records.'})
    recurred = copy.deepcopy(fixture)
    c = recurred['causes'][0]
    c['confidence_status'] = 'recurred'
    c['state_history'] = [{**c['state_history'][0], 'from': 'none', 'to': 'recurred', 'model_id': 'unrelated-model'}]
    synthetic.append({'name': 'recurred_without_prior_improvement_or_matching_history_model_currently_accepted', 'current_validator_errors': module.validate(recurred, synthetic_roots), 'interpretation': 'Before recurrence status, verify chronologically linked improvement and recurrence records.'})
    ref = fixture['taxonomy']
    trace = {'document_type': 'solver_diagnostic_trace', 'schema_version': '1.0', 'trace_id': 'synthetic-trace', 'request_id': 'synthetic-request', 'created_at': '2026-10-19T00:00:00Z', 'model_id': 'S0', 'provenance': fixture['failure_cases'][0]['provenance'], 'question_id': 'q0', 'task_type': 'multi_select', 'public_question': ref, 'source_input': ref, 'display_mapping': ref, 'protocol': ref, 'gold_visible': False, 'author_evidence_visible': False, 'output_budget_tokens': 2048, 'actual_output_tokens': 2048, 'per_item': [{'content_id': 'i0', 'judgment': 'supported', 'evidence_refs': [], 'brief_basis': 'Synthetic only.'}], 'order_pairs': [], 'final_prediction_content_ids': ['i0'], 'format_valid': True, 'raw_response': ref}
    synthetic.append({'name': 'self_declared_2048_token_diagnosis_above_frozen_1024_limit_currently_accepted', 'current_validator_errors': module.validate(trace, synthetic_roots), 'interpretation': 'Collection controller must compare declared and actual budgets against PROTOCOL_V1; schema currently only enforces actual <= locally declared budget.'})
    report = {'audit_type': 'independent_day1_mechanical_and_design_audit', 'status': 'CORE_PASS_LEDGER_PENDING', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'check_count': len(CHECKS), 'passed_count': sum(c['passed'] for c in CHECKS), 'failed_checks': [c['name'] for c in CHECKS if not c['passed']], 'checks': CHECKS, 'truthful_counts': {'legacy_public_candidates': len(legacy), 'legacy_multi_select': 51, 'legacy_ordering': 62, 'planned_new_slots_both_arms': len(rows), 'generated_new_slot_questions': 0, 'prepared_diagnosis_questions': len(next_public), 'planned_direct_calls': count, 'executed_direct_calls_declared': 0, 'new_training_runs_declared': 0, 'empty_memory_causes': len(memory['causes'])}, 'model_binding_validation': {'small_file_hash_checked_count': len(hash_checked), 'large_file_size_only_count': len(large_stat_only), 'large_file_size_only_total_bytes': sum(x['bytes'] for x in large_stat_only), 'large_file_size_only_records': large_stat_only, 'limit': 'Independent audit did not rehash model tensors or adapters, load weights, or claim independently verified 24GB tensor hashes; frozen supplied hashes are retained.'}, 'known_runtime_behavior': known_runtime, 'synthetic_memory_gate_evidence': synthetic, 'ledger_review': {'status': 'WAITING_FOR_PARENT_RECONSTRUCTION_FREEZE_AND_LEDGER', 'restricted_material_read': False}, 'evidence_bindings': list(READ_BINDINGS.values()), 'no_new_model_calls_or_training': True}
    if report['failed_checks']:
        report['status'] = 'CORE_FAIL_LEDGER_PENDING'
    (OUT / 'CORE_EVIDENCE.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'check_count', 'passed_count', 'failed_checks', 'truthful_counts', 'known_runtime_behavior', 'synthetic_memory_gate_evidence']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
