"""Offline immutable source, pairing, analysis and blind-review freeze for P0/P1."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from ..core import canonical_json, sha256_bytes, sha256_file
from ..h4_batch30_memory_event_k3_probe_v1.plan import select_successful_expansion_source
from ..h4_qa_multiroot_k3_t08_probe_v1.root_inventory import clean_named_view

BASE = Path('/root/autodl-tmp/gxy-selfplay')
RUNS = BASE / 'restart-n5-runs'
REPO = BASE / 'current_method_github'
METHOD = REPO / 'scripts/current_method/restart_n5_spell'
DEFAULT = RUNS / 'h4_r02_feedback_ablation_p0_p1_v1_live_1'
SOURCE = RUNS / 'h4_round1_r03_r1_aligned_feedback_r02_qa_source_v1'
CONTROL = RUNS / 'h4_round1_r03_r1_aligned_feedback_r02_solver_eval_live_1'
ROOTS = tuple(f'cc-train-episode-{x}' for x in (122711,136202,197635,198480,254852,94289))
ATTEMPTS = dict(zip(ROOTS, ('attempt-02','attempt-01','attempt-03','attempt-02','attempt-01','attempt-04')))
ARMS = ('feedback_off','feedback_on')

def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha_obj(value): return sha256_bytes(canonical_json(value).encode())
def source(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':sha256_file(path)}
def frozen_bytes(path, content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content: raise RuntimeError(f'frozen_output_drift:{path}')
    else:
        with path.open('xb') as f: f.write(content)
    return source(path)
def write(path,value): return frozen_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
def write_rows(path,value): return frozen_bytes(path,(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in value)).encode())

def initial(run):
    bridge=load(SOURCE/'source_bridge_audit.json')
    summary_path=RUNS/'h4_round1_r03_r1_aligned_feedback_r02_corrected_batch_summary.json'
    summary=load(summary_path)
    assert bridge['summary_sha256']==sha256_file(summary_path)
    assert summary['status']=='R02_CORRECTED_BATCH_READY_FOR_QA'
    latest={r['root_id']:r for r in summary['latest_by_root']}
    r1path=RUNS/'h4_round1_cumulative_continuation_fresh_mining_v1_live_47/round1_root_manifest.jsonl'
    r1={r['root_id']:r for r in rows(r1path)}
    sources=[source(p) for p in (SOURCE/'batch_plan.json',SOURCE/'source_bridge_audit.json',summary_path,r1path)]
    results=[]
    for old in bridge['roots']:
        rid=old['root_id']; assert rid in ROOTS
        attempt=Path(summary['output_root'])/rid/ATTEMPTS[rid]
        assert latest[rid]['attempt']==ATTEMPTS[rid] and latest[rid]['status']=='completed' and latest[rid]['r02_memory_writes']==0
        freeze_path=attempt/'run_freeze_manifest.json'; freeze=load(freeze_path)
        assert sha256_file(freeze_path)==old['freeze_manifest_sha256']
        selected=select_successful_expansion_source(attempt,expected_root_id=rid,expected_input_sha256=freeze['input_sha256'],expected_parent_dialogue_sha256=freeze['parent_dialogue_sha256'])
        assert selected['eligible'] and selected==old['selection']
        sp=attempt/'promotion/promotable_expanded_dialogue.json'; expanded=load(sp)
        assert str(sp)==selected['selected_path'] and sha256_file(sp)==selected['selected_sha256']
        prior_path=Path(r1[rid]['canonical_dialogue_path']); assert sha256_file(prior_path)==r1[rid]['canonical_dialogue_sha256']
        prior=load(prior_path); n=len(prior['turns']); assert expanded['turns'][:n]==prior['turns']
        new=expanded['turns'][n:]; assert new and all(t['turn_ref'].startswith('R02:') for t in new)
        refs=[t['turn_ref'] for t in expanded['turns']]; assert len(refs)==len(set(refs))
        assert all(t['turn_ref'].startswith(('R01:',rid+':P')) for t in prior['turns'])
        clean,mapping=clean_named_view(expanded)
        clean_src=write(run/'inputs/shared'/rid/'expanded_dialogue_clean.json',clean)
        sources.extend([source(p) for p in (sp,freeze_path,prior_path,Path(selected['validation_path']))])
        results.append({'root_id':rid,'successful_attempt':ATTEMPTS[rid],'source':source(sp),'source_dialogue_sha256':selected['complete_dialogue_sha256'],'cleaned_transcript_canonical_sha256':sha_obj(clean),'cleaned_transcript_file':clean_src,'r1_source':source(prior_path),'r1_prefix_turn_count':n,'r02_new_turn_count':len(new),'total_turn_count':len(refs),'speaker_name_mapping':mapping,'r1_prefix_identical':True,'source_selection':selected})
    assert {r['root_id'] for r in results}==set(ROOTS)
    write(run/'input_path_audit.json',{'status':'PASS','sources':sources,'roots':results,'memory_calls':0})
    policy={
        'version':'r02_feedback_p1_statistics_v1','primary_scope':['single_choice','multi_select'],
        'primary_model':'M0','primary_context':'full_context','primary_metric':'equal_weight_six_root_mean_of_candidate_correct_scored_over_scored',
        'primary_difference':'feedback_on minus feedback_off','missing_group':'null; never zero; six-root macro null if any root missing',
        'metrics':['candidate_mean','request_weighted','root_macro_candidate_mean','question_type_equal_macro_candidate_mean'],
        'context_helpful_definition':'reuse h4_round0_b0_k3_difficulty_v1.difficulty.summarize_candidate_result: technically complete and full_correct*no_total > no_correct*full_total',
        'context_helpful_code':source(METHOD/'h4_round0_b0_k3_difficulty_v1/difficulty.py'),
        'inference':'exploratory six paired roots; one generation per arm; no request-level inferential sample size; no confidence interval planned',
        'ordering':'raw diagnostic scores only; excluded from main difficulty conclusion',
        'review':{'version':'symmetric_blind_content_review_v1','max_per_root_arm_type':2,'maximum_candidates':72,
            'hash_algorithm':'sha256(UTF-8 canonical_json of {root_id,candidate_id,question_type,question,options,items,gold,evidence_refs}); sort digest ascending within root/arm/type, tie canonical_json',
            'blind_id':'B + full sha256 of selected canonical payload; identical payloads share blind review',
            'blind_fields_removed':['arm','solver_scores','generation_source','intent_id','candidate_id','feedback_status','source_paths'],
            'review_inputs':'full frozen cleaned transcript, question/options/items/Gold/evidence refs, same root historical seed QA relation without model observations or treatment status',
            'checks':['each item/answer support','ambiguity','duplicate relation or stem','whether historical feedback relation occurs'],
            'uncertainty':'mark unknown or pending review; no new API, no Gold changes, no exclusion or admission',
            'candidate_selection_independent_of_solver_scores':True},
        'duplicate_diagnostics':'original generation stem/fingerprint functions only, no second admission',
    }
    write(run/'audit/statistics_and_review_policy.json',policy)
    # Historical immutable files are evidence, not active inputs.
    oldrun=RUNS/'h4_round1_r03_r1_aligned_feedback_r02_qa_generation_live_1'
    citations=[]
    for name in ('execution_contract.json','historical_baseline_program_audit.json','historical_baseline_prompt_audit.json','prompt_parity_audit.json'):
        p=oldrun/name; citations.append(source(p)); frozen_bytes(run/'audit/historical_alignment'/name,p.read_bytes())
    write(run/'audit/historical_alignment_sources.json',citations)
    print(json.dumps({'status':'PASS','input_roots':len(results),'r02_new_turns':{x['root_id']:x['r02_new_turn_count'] for x in results}}))

def pairing(run):
    inputs=load(run/'input_path_audit.json'); schedule=load(run/'qa_schedule.json')
    initial_identity=load(run/'audit/initial_code_identity.json')
    assert all(sha256_file(Path(x['path']))==x['current_sha256'] for x in initial_identity['sources'])
    # Freeze all new modules and relevant client/scoring implementation before live.
    paths=list((METHOD/'h4_r02_feedback_ablation_p0_p1_v1').glob('*.py'))+[METHOD/'live_backends.py',METHOD/'core.py',RUNS/'prepare_aligned_r02_solver_eval.py']
    paths += [Path(x['path']) for x in load(CONTROL/'solver_protocol.json')['contract_source_files']]
    code=[source(p) for p in sorted(set(paths))]
    for x in code:
        p=Path(x['path']); relative=p.relative_to(BASE)
        frozen_bytes(run/'audit/new_and_runtime_code_snapshot'/relative,p.read_bytes())
    template=source(METHOD/'h4_batch30_memory_event_k3_probe_v1/feedback_guidance_v1.md')
    model_identity=source(CONTROL/'model_identity_report.json'); protocol=source(CONTROL/'solver_protocol.json')
    generation={'model':'Qwen/Qwen3-14B','temperature':0.1,'thinking':False,'generation_seed_policy':'not_sent_by_original_client','intent_count':3,'intent_max_tokens':4096,'writer_max_tokens':8192,'guided_requested_counts':{'single_choice':2,'multi_select':1,'ordering':1},'free_requested_counts':{'single_choice':4,'multi_select':2,'ordering':2},'minimum_counts_each_mode':{'single_choice':2,'multi_select':1,'ordering':1},'guided_free_target_ratio':'1:2','repair_max_per_batch':1,'repair_policy':'original whole-batch replacement below minimum','calls_per_root_arm_planned':7,'calls_per_root_arm_max':13,'calls_all_planned':84,'calls_all_max':156}
    pairs=[]
    for rid in ROOTS:
        row=next(r for r in inputs['roots'] if r['root_id']==rid)
        arms={}
        for arm in ARMS:
            folder=run/'inputs'/arm/rid
            packet=load(folder/'feedback_packet.json')
            arms[arm]={'arm_id':arm,'qa_directory':str(run/arm/'qa'),'solver_directory':str(run/arm/'solver'),'packet':source(folder/'feedback_packet.json'),'context':source(folder/'feedback_context.json'),'seed_candidate_id':packet.get('source_candidate_id'),'historical_sources':packet.get('sources',{})}
        pairs.append({'root_id':rid,'arms':arms,'transcript':row,'template':template,'original_code_identity':source(run/'audit/initial_code_identity.json'),'runtime_code':code,'generation':generation,'solver_model_identity':model_identity,'solver_protocol':protocol,'call_schedule':source(run/'qa_schedule.json'),'statistics_policy':source(run/'audit/statistics_and_review_policy.json'),'r1_only_memory':'fixed historical R1-only start; no Memory/bootstrap/add calls in this task'})
    write_rows(run/'pairing_manifest.jsonl',pairs)
    write(run/'execution_contract.json',{'version':'r02_feedback_ablation_p0_p1_v1','generation':generation,'roots':list(ROOTS),'arms':list(ARMS),'original_generation_code':initial_identity['sources'],'runtime_code':code,'solver_protocol':protocol,'solver_model_identity':model_identity,'scope':'P0/P1 only','memory_calls_allowed':0,'p2_authorized':False,'schedule':source(run/'qa_schedule.json'),'statistics_policy':source(run/'audit/statistics_and_review_policy.json')})
    write(run/'manifest.json',{'version':'r02_feedback_ablation_p0_p1_v1','run_root':str(run),'source_run':str(SOURCE),'control_solver':str(CONTROL),'root_ids':list(ROOTS),'arms':list(ARMS),'pairing_manifest':source(run/'pairing_manifest.jsonl'),'execution_contract':source(run/'execution_contract.json')})
    print(json.dumps({'pairing_roots':len(pairs),'logical_call_budget':156,'status':'FROZEN'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,default=DEFAULT); p.add_argument('--phase',choices=['initial','pairing'],required=True); a=p.parse_args()
    {'initial':initial,'pairing':pairing}[a.phase](a.run.resolve())
