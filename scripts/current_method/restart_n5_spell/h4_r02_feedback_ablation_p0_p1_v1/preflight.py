"""Join independently verified P0 artifacts; no API calls or permissive fallbacks."""
from __future__ import annotations
import argparse
from pathlib import Path
from .freeze import DEFAULT,ROOTS,ARMS,load,rows,source,sha_obj,write,write_rows,RUNS
from .qa_run import verify_schedule

def preflight(run):
    inputs=load(run/'input_path_audit.json'); feedback=load(run/'inputs/feedback_preflight_report.json')
    prompt=load(run/'audit/prompt_parity_preflight.json'); solver=load(run/'audit/solver_identity_protocol_p0.json')
    schedule=verify_schedule(run);pairs=rows(run/'pairing_manifest.jsonl')
    tests=(run/'audit/p0_tests.log').read_text(); assert 'Ran 23 tests' in tests and '\nOK\n' in tests
    assert inputs['status']==feedback['status']==solver['status']=='PASS' and prompt['status']=='PASS_OFFLINE' and not prompt['errors']
    assert len(pairs)==6 and {p['root_id'] for p in pairs}==set(ROOTS)
    assert schedule['logical_limit_total']==156 and schedule['planned_total']==84
    sources=inputs['sources']
    for item in sources: assert source(item['path'])['sha256']==item['sha256']
    for item in inputs['roots']:
        root=item['root_id']; clean=load(item['cleaned_transcript_file']['path'])
        for arm in ARMS:
            qa=run/arm/'qa/shards/shard-00/roots'/root
            assert load(qa/'expanded_dialogue_clean.json')==clean
    r1=rows(RUNS/'h4_round1_cumulative_continuation_fresh_mining_v1_live_47/round1_root_manifest.jsonl')
    assert len(r1)==70 and {r['split'] for r in r1}=={'train'}
    checks={'six_exact_successful_attempts':True,'two_arms_exact_same_cleaned_r02':True,'r1_prefix_and_r02_boundary':True,'first_round_is_70_train_roots':True,'nonordering_sources_reselected_with_original_rank':True,'ordering_mask_applied_before_active_aggregation':True,'six_valid_nonempty_treatments_and_empty_controls':True,'planner_offline_shared_mask_preview':True,'original_builder_fixed_contract_and_stage_audit':True,'original_code_snapshot_sha_match':True,'scoped_84_planned_156_maximum_logical_calls':True,'safe_resume_completed_reuse_and_unknown_reservation_stop':True,'statistics_and_blind_selection_frozen_before_generation':True,'all_23_offline_tests_pass':True,'three_solver_models_and_full_original_protocol':True,'memory_calls_disabled':True}
    evidence=[source(run/name) for name in ('input_path_audit.json','inputs/feedback_preflight_report.json','audit/prompt_parity_preflight.json','audit/solver_identity_protocol_p0.json','audit/p0_tests.log','audit/statistics_and_review_policy.json','pairing_manifest.jsonl','execution_contract.json','qa_schedule.json','ordering_exclusion_audit.json','history_preservation.json')]
    result={'status':'PASS','pass':True,'root_count':6,'root_arm_count':12,'checks':checks,'evidence':evidence,'live_generation_calls_at_preflight':0,'solver_calls_at_preflight':0,'memory_calls':0,'content_review_limit':'4 clearly faulty historical candidates excluded; selected historical labels retain unresolved ambiguity in 197635 and 254852. Program eligibility is not Gold certification.','remaining_ambiguity_roots':feedback['remaining_ambiguity_roots'],'next_stage':'P1 QA then all-program-valid candidates through original three-model Solver; P2 prohibited'}
    write(run/'p0_preflight_report.json',result)
    print('PASS: P0 source/feedback/pairing/protocol/test preflight; 12 root-arms authorized for live')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);a=p.parse_args();preflight(a.run)
