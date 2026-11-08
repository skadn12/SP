"""Prepare the Day 1 protocol and bind existing models without loading them."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = Path('/root/autodl-tmp/gxy-selfplay')
RUNS = REPO / 'restart-n5-runs'
DELIVERY = REPO / 'offline-deliveries'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    p = Path(path)
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def save(name, data):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    assert not (HERE / 'PROTOCOL_V1.json').exists()
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    base = Path('/root/autodl-tmp/Qwen3-4B')
    adapter = RUNS / 'h4_round0_b0_formal_training_v1_live_2/clean_adapter'
    clean = RUNS / 'h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
    old = RUNS / 'h4_codex_combined_sft_v125_merged'
    old_adapter = RUNS / 'h4_codex_combined_sft_v125/adapter'
    for merged, expected in [(clean, adapter), (old, old_adapter)]:
        info = json.loads((merged / 'merged_adapter_info.json').read_text())
        assert Path(info['adapter_dir']).resolve() == expected.resolve()
        assert Path(info['base_model']).resolve() == base.resolve()
    models = {}
    for key, directory in [('base', base), ('S0_M_clean_merged', clean), ('old_Round2_merged', old)]:
        files = []
        index = json.loads((directory / 'model.safetensors.index.json').read_text())
        names = sorted(set(index['weight_map'].values()))
        names += [p.name for p in directory.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')]
        for name in sorted(set(names)):
            files.append(bind(directory / name))
        models[key] = {'directory': str(directory), 'files': files}
    for key, directory in [('S0_M_clean_adapter', adapter), ('old_Round2_adapter', old_adapter)]:
        models[key] = {'directory': str(directory), 'files': [bind(directory / x) for x in ['adapter_config.json', 'adapter_model.pt']]}
    save('MODEL_BINDINGS.json', {
        'status': 'BOUND_EXISTING_MODELS_NO_LOADING', 'created_at_beijing': now,
        'S0': 'S0_M_clean_merged', 'train_initialization': 'base + S0_M_clean_adapter',
        'old_round2_role': 'regression_comparator_only',
        'merge_rule': 'Apply the custom adapter to original base once; never add the adapter to an already merged model.',
        'models': models,
    })
    protocol = {
        'schema_version': 1,
        'experiment_id': 'selfplay_week_20261019_v1',
        'status': 'DESIGN_FROZEN_BEFORE_NEW_INFERENCE_GENERATION_TRAINING',
        'frozen_at_beijing': now,
        'prior_plan': bind(HERE.parent / 'PLAN_ZH.md'),
        'primary_question': 'Does train-side diagnosis feedback improve new-question transfer beyond diversity-only generation at comparable cost?',
        'arms': {
            'A': {'generation': 'diversity_only', 'reads_solver_failure_memory': False},
            'B': {'generation': 'diversity_plus_diagnosis', 'reads_solver_failure_memory': True,
                  'targeted_slots_ratio': 0.7, 'free_exploration_slots_ratio': 0.3,
                  'memory_ready_rule': 'Only reproduced train-side causes with audited failures on at least 3 distinct train roots; otherwise record unmet targeted quota and wait rather than invent causes.'},
        },
        'update_lineage': {'stage1': 'A1 and B1 independently start from S0=M_clean',
                           'stage2': 'A2 starts from A1; B2 starts from B1; independent optimizer/scheduler state resets per stage',
                           'frozen_solver_within_stage': True},
        'seeds': {'selection': 2026101900, 'training': 2026101901,
                  'training_display': 2026101902, 'evaluation_display': 2026101903,
                  'direct_sampling': 2026101904,
                  'interactive_generator_seed': None,
                  'interactive_generator_seed_reason': 'Backend seed control not exposed; do not claim deterministic generation.'},
        'root_schedule': {'file': 'evaluation/ROOT_SELECTION.json',
                          'selection_rule': 'Sort complete root_id lexicographically among the authoritative 70 Train roots; first20 stage1, next10 stage2; first6 memory pilot.',
                          'selection_uses_solver_scores': False},
        'production': {'stages': [{'stage':1,'roots':20,'slots_per_root':6,'planned_slots_per_arm':120,'B_targeted':84,'B_free':36},
                                 {'stage':2,'roots':10,'slots_per_root':6,'planned_slots_per_arm':60,'B_targeted':42,'B_free':18}],
                       'per_root': {'multi_select':3,'ordering':3},
                       'new_options_or_items':6,
                       'gold_count_schedule': 'Across sorted root/slot IDs, multi-select Gold targets cycle [2,3,4,5]; same schedule in A/B, no score-based reassignment.',
                       'B_targeted_slot_assignment': 'Within each stage, sort slot_id by SHA256(selection_seed|stage|slot_id); lowest70percent targeted, remainder free.',
                       'initial_generation_attempts_per_slot':1,'maximum_content_revisions_per_slot':1,
                       'mechanical_error_retries':0,'automatic_solver_retries':0,
                       'repair_rule': 'Clarify wording only; changed facts/Gold/anchors require a new version not silently admitted in the original slot.',
                       'retain_all_attempts_and_shortfalls':True,
                       'content_admission_uses_solver_scores':False,
                       'ordering_definition':'appearance in the complete publicly specified source range; not actual event time',
                       'max_output_tokens_per_authored_question':3072,
                       'max_output_tokens_per_reconstruction':3072,
                       'interactive_generator_note':'Token ceiling is requested where supported; unavailable settings logged null. Current internal tasks share model/filesystem; not external blind review.',
                       'final_controller_preflight_required':True},
        'model_identity': {'binding_file':'MODEL_BINDINGS.json',
                           'challenger_backend':'current Codex session and explicitly logged internal tasks',
                           'exact_challenger_model_identifier':None,
                           'exact_challenger_model_identifier_status':'not exposed; must not infer from branding',
                           'challenger_weight_updates':False},
        'fact_memory_pilot': {'roots':6,'arms':['recent2','recent2_plus_prefix_memory','full_history'],
                              'scenes_per_branch':3,'recent_context_unit':'two complete sessions',
                              'common_input':'Same accepted prefix and high-level objective; no hidden-history facts injected through shared plan.',
                              'memory_backend':'local source-bound JSON snapshot',
                              'generation_calls_per_scene':{'planner':1,'realizer':1},
                              'max_output_tokens':{'planner':1024,'realizer':4096,'memory_update':2048},
                              'review_calls_per_scene':1,'max_revisions_per_scene':1,
                              'equal_budget_meaning':'same call counts and output limits; actual input and memory maintenance tokens are measured, not forced equal by truncation',
                              'default_main_pipeline':'full_history until pilot meets source-consistency and cost criteria',
                              'main_AB_context_identical':True},
        'diagnosis': {'taxonomy':'diagnosis/WEAKNESS_TAXONOMY.json','schema':'diagnosis/MEMORY_SCHEMA.json',
                      'source':'audited train-side questions only','minimum_reproduced_roots':3,
                      'frequency_unit':'distinct root and question family, not repeated sampling calls',
                      'structured_evidence_branch_is_separate':True,
                      'max_evidence_output_tokens':1024,'diagnosis_repeats_per_question':1,
                      'solver_sees_gold_or_author_evidence_before_response':False,
                      'self_generated_reasoning_is_training_target':False,
                      'diagnostic_sample_rule':'Deterministic hash sampling within incorrect/occasionally-correct/stably-correct strata, retaining IDs and denominators; feedback set cannot be final eval.'},
        'legacy_reuse': {'queue':'reuse/public/QUESTIONS.json','expected_multi':51,'expected_ordering':62,
                         'single_choice_excluded':10,'source_first_reconstruction_required':True,
                         'missing_old_review_is_not_failure_or_pass':True,
                         'preserve_original_option_counts':True,
                         'previously_untrained_generated_questions_are_not_seen_training_replay':True},
        'training': {'initialization':'original Qwen3-4B + actual M_clean custom adapter',
                     'reset_optimizer_and_scheduler':True,'max_length':12288,'allow_truncation':False,
                     'epochs':2.0,'microbatch':1,'gradient_accumulation':4,'lr':0.0001,
                     'min_lr_ratio':0.1,'warmup_ratio':0.05,'weight_decay':0.0,'max_grad_norm':1.0,
                     'lora_rank':16,'lora_alpha':32,'lora_dropout':0.05,
                     'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                     'loss':'assistant answer tokens only; no unverified explanation supervision',
                     'mix_target_supervised_tokens':{'new_structure':0.6,'audited_legacy':0.2,'existing_round2_foundation':0.2},
                     'mix_shortfall_rule':'Record legacy shortfall; use a symmetric score-independent reduction/reallocation fixed before training. Never loosen semantic review.',
                     'permutation':'fresh deterministic gold-blind bijection each epoch; preserve original 4/5/6 item formats',
                     'independent_question_count_inflated_by_permutation':False,
                     'budget_matching':{'optimizer_steps':'equal per corresponding A/B stage',
                                        'full_input_training_token_difference_max_ratio':0.05,
                                        'procedure':'Hash-order eligible pools and construct equal-microstep paired schedules before training; reject budget freeze if full-token difference exceeds5percent, without reading outcomes.',
                                        'record':['unique_questions','families','roots','supervised_tokens','full_input_tokens','microsteps','optimizer_steps','generation_calls','review_calls']},
                     'exact_schedules':'Frozen separately after semantic admission and tokenization; this design file does not claim ready training inputs.',
                     'optional_C1':'Same B1 question pool and resource budget, fixed single permutation, start from S0; cannot compare one-step C1 with two-step B2 as a permutation effect.'},
        'evaluation': {'policy':'evaluation/EVALUATION_POLICY.json',
                       'primary_direct_protocol':{'context':'complete public source','multi_select_permutations':6,'multi_select_samples':3,
                                                  'ordering_permutations':12,'ordering_samples':1,'temperature':0.8,'top_p':0.9,
                                                  'max_output_tokens':32,'batch_size':8,'finite_choice_constrained_answers':True,
                                                  'adaptation_needed':'v140 adapter is six-item only; dynamic4/5/6 compatibility must pass preflight, with immutable legacy files preserved.'},
                       'primary_metric':'root-equal and question-equal exact, separately reported',
                       'secondary':['multi_select_FP_FN_Jaccard_overselection','ordering_pair_accuracy_within_across','display_permutation_consistency','source_coverage','cost'],
                       'fixed_denominator':True,'invalid_or_missing_exact':0,
                       'no_test_feedback_to_generator':True,'no_same_family_cross_split':True,
                       'legacy_dev_test_role':'historically exposed regression; never described as pristine unseen test',
                       'new_prospective_eval':'Not yet collected/frozen; required before a claim about genuinely unseen-root generalization, not a reason to relabel existing Train roots.'},
        'implementation_gates_before_expensive_execution':[
            'semantic admission manifest for exact data version',
            'source/model/tokenizer hash verification',
            'gold-blind mapping inverse checks for 4/5/6 item formats',
            'source-only public prompt whitelist and split namespace guards',
            'complete-context tokenization without truncation',
            'paired training token/microstep budget freeze',
            'actual backend params/calls recorded, unknown parameters explicitly null',
        ],
        'execution_this_package':{'new_solver_inference':False,'new_training':False,'new_dialogue_expansion':False,'internal_semantic_review_tasks':3},
    }
    save('PROTOCOL_V1.json', protocol)
    print(json.dumps({'status':'PROTOCOL_AND_MODELS_BOUND','models':len(models),'model_bound_files':sum(len(v['files']) for v in models.values()),'created_at_beijing':now}))


if __name__ == '__main__':
    main()
