"""Freeze the authorized prospective experiment before any author requests."""
from pathlib import Path
import json, hashlib, datetime, os
BASE=Path(__file__).resolve().parents[1]
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bind(p):
 p=Path(p).resolve(); b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,x):
 with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2); f.write('\n')
if __name__=='__main__':
 plan=json.loads((BASE/'PLAN.json').read_text()); parent=BASE.parent
 protocol=dict(schema_version='oet1_formal_balanced_protocol_v1',version=1,status='FROZEN_BEFORE_AUTHORING',created_at_utc=now(),
   authorized_objective=plan['authorized_objective'],roots=plan['new_train_roots'],conditions=['GENERIC','TARGETED'],
   historical_parent_seen_by_M_clean=True,unseen_root_claim=False,exploratory_single_paired_experiment=True,
   scene_sequence=[f'OET1:{r}:S{i:02}' for r in 'AB' for i in range(1,4)],turns_per_scene=14,
   cumulative_routes=True,paired_initial_source_and_native_events=True,only_arm_difference='Frozen abstract material goals from verified Train cards; exploratory cross-root transfer.',
   planned=dict(roots=8,sources=16,scenes=96,question_slots=112),first_wave_roots=plan['new_train_roots'][:4],
   expansion_gate=dict(overall_admission_min=.90,every_root_admission_min=.85,every_slot_admission_min=.75,source_or_gold_ambiguity_failures_max=0),
   failure_policy='One immutable semantic attempt per scene/slot. Retain every failure and planned denominator. Source failure stops its descendants, not independent units. Gate failure stops roots 5-8.',
   transport_retry=dict(max_attempts=3,only_if_no_successful_semantic_response=True,retain_original=True),
   backend=dict(executable='/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex',model_override=None,max_concurrent_data_review_processes=3,read_only=True,separate_new_session_per_call=True,return_json_only=True),
   audit=dict(type='Independent AI review',condition_visible=True,human_blind_review=False,mechanical_is_not_semantic=True,question_gold_hidden_until_independent_derivation=True,exact_quote_substrings=True),
   slots={'MC2':'MC_ENTITY_BINDING','MC3':'MC_STATE_UPDATE_CONJUNCTION','MC4':'MC_CONCLUSION_EVIDENCE','MC5':'MC_JOINT_CONSTRAINT','MC6':'MC_RECORD_AUDIT_CONTROL','ORDER1':'ORDER_EXPLICIT_CLOCK_VALUE','ORDER2':'ORDER_ENQUIRY_PROGRESSION'},
   private_mc_cardinality=[2,3,4,5,6],mc6=dict(control_only=True,exclude_primary_training=True,exclude_primary_benefit=True,public_count_and_control_hidden=True),
   training=dict(starting_checkpoint=str((BASE.parents[2]/'restart-n5-runs/day5_B1_schedule_v1/merged_model').resolve()),new_independent_lora=True,seed=20261024,rank=16,alpha=32,dropout=.05,lr=1e-4,min_lr_ratio=.1,warmup_ratio=.05,weight_decay=0,max_grad_norm=1,microbatch=1,gradient_accumulation=4,epochs=2,permutations_per_question=2,schedule='Frozen paired-slot hash order; single runtime traversal of pre-expanded rows',fixed_final_checkpoint=True,dev_every_optimizer_steps=20,no_truncation=True,context_cap_initial=16384,longest_sample_smoke=True,offline_wandb=True,full_resume=True,gpu_flock=True),
   training_pool=dict(primary='Independently admitted common root/slot MC2-5 and ORDER1-2; no score filtering',historical_layer='Predeclared QFP2 24 primary questions per arm, separately reported',foundation=False,old_133_candidate_pool=False,gate_failure_fallback='QFP2-only explicitly labelled pilot, if independently bound audits and fairness gates pass; new-root main objective remains incomplete',matched_question_count_and_optimizer_steps=True,input_token_difference_policy='Report actual; >5% prevents formal fairness claim; exploratory only if explicitly disclosed'),
   evaluation=dict(boundary='Existing DEV only; no Test content read for production',controller_reads_item_level_eval=False,freeze_before_training=True,models=['B1','B1_GENERIC','B1_TARGETED'],same_prompts_permutations_and_decoding=True,greedy=True,max_new_tokens=32,strict_box=True,periodic_panel='Frozen 42 DEV questions x one display',final_scope='Frozen 42 DEV questions x two distinct displays',baseline_cache='Only exact checkpoint/prompt/seed/hash match; otherwise rerun',training_fit_separate=True,metrics=['MC exact/partial/FP/FN/overselection','ordering exact/pairwise','paired root','display consistency','MC6 separate','new vs historical layer']),
   interpretation='Single seed exploratory paired experiment. Difficulty or training fit is not evidence of generalization or feedback benefit. Visible reasons are not faithful internal CoT. No positive-result chasing.',
   deadline=json.loads((BASE/'SUPERVISOR_STATUS.json').read_text())['deadline_at_utc'],closeout_reserve_minutes=45,
   bindings=[bind(BASE/n) for n in ['PLAN.json','WORKER_INSTRUCTIONS_ZH.md','PRODUCTION_HANDOFF_ZH.md','TRAINING_HANDOFF_ZH.md']]+[bind(parent/'postweek_question_presentation_repair_v1/QUESTION_AUTHOR_RULES_V2.txt'),bind(BASE.parents[2]/'scripts/current_method/question_presentation_gate_v2.py'),bind(parent/'postweek_feedback_control_4root_v1/WEAKNESS_CARDS.json'),bind(parent/'postweek_question_firstpass_v2/FINAL_REPORT.json')])
 save(BASE/'PROTOCOL_FREEZE_V1.json',protocol)
 save(BASE/'TASK_STATUS.json',dict(schema_version='oet1_task_status_v1',overall_status='RUNNING',current_phase='freeze_rules_and_balanced_protocol',updated_at_utc=now(),phases={p:dict(status='IN_PROGRESS' if i==0 else 'PENDING') for i,p in enumerate(plan['phases'])},protocol=bind(BASE/'PROTOCOL_FREEZE_V1.json')))
 print(json.dumps(bind(BASE/'PROTOCOL_FREEZE_V1.json')))
