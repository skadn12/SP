"""Write prospective no-score common-evaluation policies; no Solver or training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

def bind(p):
    p=Path(p).resolve()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

def write(name,obj):
    p=HERE/'protocol'/name
    if p.exists():
        raise RuntimeError('NO_OVERWRITE:'+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    lock=WEEK/'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
    policy=WEEK/'day1/PROTOCOL_V1.json'
    model=WEEK/'day1/MODEL_BINDINGS.json'
    now=datetime.now(timezone.utc).isoformat()
    exclusions={
      'schema_version':'common_eval_exposure_interpretation_v1','created_at_utc':now,
      'status':'PROSPECTIVE_IMPLEMENTATION_CLARIFICATION_NOT_A_RELAXED_EXCLUSION_GATE',
      'source_lock':bind(lock),
      'rule':'Whole-root/source exposure and actual question fact-family exposure are separate records.',
      'whole_root_exposure':'All31 historically exposed heldout roots remain REGRESSION_ONLY. Source was exposed does not mean every fact was an examined question family.',
      'formal_exclusion_record':'Derive from actual historical question support, all-option source evidence union and every actual ordering event (historical item counts may differ; prospective new ordering has six). Never emit correctness, Gold labels, model predictions or scores.',
      'all_option_union':'Include supported and contradicted source propositions for all options, not just evidence for correct options; preserve evidence without option labels or truth status.',
      'whole_source_turn_library':'Optional lexical retrieval only. It cannot replace exact fact-family mapping, semantic equivalence review or become a blanket root exclusion.',
      'family_identity':'Use normalized necessary source-grounded fact tuples and source scopes. Root ID, SHA inequality, wording change, Gold subset and display permutations do not establish distinct family identity.',
      'unresolved_mapping':'Mark UNRESOLVED_FAMILY_MAPPING and retain a specific uncertainty scope for semantic review. Never silently map an unresolved question to a whole-root PASS family.',
      'missing_evidence':'PENDING/HOLD, not PASS; new-family selection waits for a complete, no-score indexed comparison.',
      'author_release_fields':['exclusion_record_id','root_id','original_split','source_sha256',
          'canonical_fact_family_id','semantic_fact_fingerprints','related_family_ids','evidence_turn_refs',
          'scope','exposure_role','exclusion_manifest_sha256'],
      'manifest_digest_convention':'Hash a standalone exclusion manifest containing input bindings/schema/coverage; rows reference that manifest digest. The manifest must not require a circular hash of itself or of rows containing its own digest.',
      'author_release_package':{'allowed_files': ['author_release/EXCLUSION_INDEX.jsonl', 'author_release/EXCLUSION_MANIFEST.json'], 'raw_boundary_manifest': 'INTERNAL_ONLY_NOT_AUTHOR_INPUT', 'raw_boundary_index': 'INTERNAL_ONLY_USE_REBOUND_SAFE_COPY', 'manifest_bindings': 'Opaque input IDs, SHA256, bytes and coarse roles only; no internal paths, outcome-coded locators, question IDs, scores, answers or feedback payloads.', 'status': 'BUILD_AND_INDEPENDENT_REVIEW_REQUIRED_BEFORE_AUTHOR_DELIVERY', 'supersedes': 'Earlier boundary report statement that raw boundary manifest is author-readable.'},
      'author_release_role_coarsening':{'rule': 'training_exposure iff original_split=train; otherwise historical_exposure', 'independent_of': 'scores, predictions, Gold, selection outcome and diagnostic identity', 'semantic_fields': 'All source facts, family IDs, related families, evidence refs and scopes remain byte-equivalent as JSON values. Only release metadata role is coarsened.', 'review_requirement': 'Coarsening never permits skipping any of the five semantic exclusion checks; all prior training exposures must be compared for both training and feedback-related exclusions.', 'original_role_mapping': 'INTERNAL_ONLY'},
      'semantic_clearance':'Five itemized checks per actually inventoried candidate, concrete compared record IDs, reviewer, evidence and rationale. Missing inventories remain PENDING.',
      'root_count_is_not_new_family_count':True,'new_solver_calls':0,'new_training_runs':0}
    write('EXPOSURE_AND_FAMILY_INTERPRETATION.json',exclusions)
    request={
      'schema_version':'common_eval_prospective_request_policy_v1','created_at_utc':now,
      'status':'POLICY_FROZEN_ACTUAL_CONTENT_REQUESTS_AND_FUTURE_MODEL_IDENTITIES_PENDING',
      'source_lock':bind(lock),'inherited_direct_protocol':bind(policy),'historical_model_bindings':bind(model),
      'scope':{'dev_roots':15,'test_roots':16,'abstract_slots':186,
          'cohort':'prospective_new_family_on_historically_exposed_heldout_root_only_after_semantic_clearance',
          'historical_role':'REGRESSION_ONLY','unseen_root_claim':False},
      'public_input':{'schema_keys':['candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'],
          'required_common_keys':['candidate_id','root_id','question_type','question','source_path','source_sha256'],
          'exactly_one_task_field':['options','items'],
          'source':'all admitted source turns in exact source order','context':'full_history',
          'forbidden':'Gold, private evidence, feedback cards, predicted answers, scores and per-item diagnosis',
          'public_question_schema':'Reuse bound dynamic4/5/6 adapter whitelist; exactly one of options or items.',
          'content_and_request_freeze_required':True},
      'prompt':{'adapter':bind(WEEK/'day2/solver/adapter.py'),
          'builder':'direct_prompt -> native contract.build_prompt(full_context)',
          'enable_thinking':False,'output':'one boxed direct answer',
          'extra_unverified_reasoning_supervision':False},
      'sampling':{'temperature':0.8,'top_p':0.9,'max_output_tokens':32,'batch_size':8,
          'multi_select_permutations':6,'multi_select_samples_per_permutation':3,
          'ordering_permutations':12,'ordering_samples_per_permutation':1,
          'finite_choice_constraint':'all nonempty label subsets for MC, all full permutations for ordering; never constrain using Gold cardinality',
          'stop':['</box>'],'include_stop_str_in_output':True,
          'seed_base':20261020,'display_seed':20261020,
          'sampling_seed_rule':'seed_base + global_slot_index_zero_based*18 + variant_index*sample_count_for_type + sample_index',
          'slot_index_rule':'all186 prelocked slots in DEV then TEST order, including shortfalls; never renumber after admission',
          'matched_model_rule':'same admitted questions, prompts, mappings, seeds and sampling parameters across every compared model',
          'automatic_content_retry':False},
      'display_mapping':{'implementation':bind(WEEK/'day2/solver/adapter.py'),
          'algorithm':'v15 independent seeded rebinding generalized to4/5/6 items; public-content IDs; no Gold reads',
          'unique_variants_per_question':True,'all_variants_one_family_and_split':True,
          'synthetic_roundtrip_preflight_required':True,'actual_question_roundtrip_preflight_required':True},
      'length':{'tokenizer_directory':'/root/autodl-tmp/Qwen3-4B',
          'max_model_len':16384,'reserved_answer_tokens':32,
          'zero_truncation':True,'check':'Tokenize every full chat-formatted actual variant and require prompt_tokens+32<=16384.',
          'training_length_separate':12288,'training_inputs_also_require_zero_truncation':True,
          'overflow_policy':'Record LENGTH_HOLD before scores; never truncate, replace source or hide required context.'},
      'model_identity':{'S0':'day1 binding S0_M_clean_merged; exact bytes must be reverified before execution',
          'comparators':['base','old_Round2_merged','A1','B1','A2','B2'],
          'future_checkpoint_identity_status':'PENDING_NOT_CREATED',
          'freeze_rule':'Bind exact configs, weights/adapters, tokenizer, lineage and runtime before each allowed evaluation. This policy is not an actual model/request manifest.',
          'merge_rule':'never add an adapter to an already merged model'},
      'selection':{'checkpoint_rule':'FIXED_PREDECLARED_FINAL_TRAINING_ENDPOINT',
          'dev':'aggregate regression reporting only; no outcome-based checkpoint selection in this execution policy',
          'test':'final-model regression reporting only after final model identity freeze; no interim TEST evaluation',
          'failure_rule':'numerical/infrastructure failure is a failed run, not permission to pick a better-scoring checkpoint',
          'challenger_release':'no selected evaluation facts, questions, Gold, item outcomes or source examples'},
      'metrics':{'primary':['question_equal_exact','root_equal_exact'],
          'secondary':['MC_FP_FN_Jaccard_overselection','ordering_pair_accuracy_within_across_scene',
              'display_permutation_consistency','production_and_source_coverage','actual_usage_and_time'],
          'invalid_or_missing':'retain request denominator; exact and ordering pair credit zero; unknown FP/FN remain null',
          'shortfall':'retain all186 in production coverage denominator; never fabricate solver failures for nonexistent questions',
          'scored_question_denominator':'fixed accepted frozen content, same for all compared models; type/split separately',
          'root_denominator':'report roots with accepted questions and planned source coverage separately',
          'family_dependence':'all permutations remain one question; report facts/root dependence and root-cluster uncertainty'},
      'actual_execution':{'generated_evaluation_questions':0,'frozen_evaluation_requests':0,
          'new_solver_calls':0,'new_training_runs':0,'full_evaluation_ready':False}}
    write('REQUEST_POLICY_V1.json',request)
    print(json.dumps({'status':request['status'],'written':2,'new_solver_calls':0,'new_training_runs':0}))

if __name__=='__main__': main()
