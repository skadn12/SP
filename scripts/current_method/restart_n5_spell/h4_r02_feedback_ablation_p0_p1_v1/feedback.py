"""Offline source mask and empirical feedback freezer; no generation or Memory calls.

Selection calls the actual first-round compiler's safe/rank functions. Rendering and
request validation reuse the existing empirical adapter rather than its old seeds.
All outputs use byte-identical, write-once resume semantics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path

from ..h4_round1_feedback_compiler_v1 import compiler as original

WORKSPACE = Path('/root/autodl-tmp/gxy-selfplay')
RUNS = WORKSPACE / 'restart-n5-runs'
DEFAULT_OUTPUT = RUNS / 'h4_r02_feedback_ablation_p0_p1_v1_live_1'
COMPILER = RUNS / 'h4_round1_feedback_compiler_v1_live_1'
EXCLUSION = RUNS / 'h4_round1_round0_ordering_feedback_exclusion_v1_offline_live_1/export'
ROOTS = tuple('cc-train-episode-' + str(x) for x in (122711, 136202, 197635, 198480, 254852, 94289))
ALLOWED = frozenset(('single_choice', 'multi_select'))
EXPECTED_OLD_ELIGIBLE = dict(zip(ROOTS, (3, 3, 2, 2, 4, 2)))
SPEC = importlib.util.spec_from_file_location('original_empirical_feedback_adapter', RUNS / 'prepare_r1_aligned_qa_feedback.py')
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def artifact_jsonl(path, values):
    return legacy.write_frozen(path, ''.join(legacy.canonical(x) + '\n' for x in values).encode())


def eligibility(row, confirmed_invalid):
    """Shared QA/Planner entry gate, before ranking or any derived projection."""
    if row.get('question_type') not in ALLOWED:
        return False, ['ordering_mask_before_eligibility_rank_aggregation_render']
    if row.get('split') != 'train' or row.get('root_id') not in ROOTS:
        return False, ['outside_frozen_train_root_scope']
    return original._safe_frontier(row, confirmed_invalid)


def selected_rows(transitions, confirmed_invalid):
    active = [x for x in transitions if eligibility(x, confirmed_invalid)[0]]
    return {root: sorted((x for x in active if x['root_id'] == root), key=original._frontier_rank)
            for root in ROOTS}


def review_record(cid, status, refs, judgment, risks=None):
    return {'candidate_id': cid, 'review_status': status,
            'reviewer': 'Codex direct full Parent+R01 transcript reading; no audit API',
            'historical_gold_changed': False, 'confirmed_source_defect': status == 'CONFIRMED_GOLD_DEFECT',
            'evidence_refs': refs, 'judgment': judgment, 'remaining_risks': risks or []}


def manual_reviews():
    """Content findings frozen before P1; only clear errors exclude, ambiguity is explicit."""
    return [
        review_record(ROOTS[0]+':I003-F-Q06', 'CONFIRMED_GOLD_DEFECT',
                      [ROOTS[0]+':P05:T010', ROOTS[0]+':P05:T011', 'R01:S03:T005', 'R01:S03:T009'],
                      'Gold B,C,E omits D, helping her neighbor with groceries: asked whether she helped lately, Emily explicitly answers Yes. The question has no R01-only restriction. Ground-school reading also includes future plans.'),
        review_record(ROOTS[0]+':I002-F-Q05', 'CONFIRMED_GOLD_DEFECT',
                      [ROOTS[0]+':P04:T003', ROOTS[0]+':P04:T013', ROOTS[0]+':P05:T011'],
                      'Gold A and D transfer reasons for practicing archery (relieve stress; improve herself) to reasons Emily gives for helping others. Those asserted causal relations are not in the dialogue.'),
        review_record(ROOTS[0]+':I002-F-Q06', 'REVIEWED_NO_CLEAR_GOLD_ERROR',
                      [ROOTS[0]+':P02:T007', ROOTS[0]+':P02:T008', ROOTS[0]+':P03:T008', ROOTS[0]+':P04:T010'],
                      'A study groups is acknowledged by James after Emily recalls the helpful science study group. B extra teacher help is explicit. D support from friends is supported by James crediting Emily advice and suggested techniques. C archery and E solo flying concern Emily, not James.',
                      ['Some original evidence refs are indirect; support from friends is a paraphrase, not an exact phrase.']),
        review_record(ROOTS[1]+':I003-F-Q06', 'REVIEWED_NO_CLEAR_GOLD_ERROR',
                      ['R01:S01:T003', 'R01:S02:T001', 'R01:S02:T007', 'R01:S03:T004', 'R01:S03:T005'],
                      'A counseling is paraphrased by booking therapy for frustration. B taking space is explicitly to cool off. C photography helps expression without becoming overwhelmed and staying grounded. D dishes is gratitude; E interruption is the behavior being reduced.',
                      ['The cited original refs are incomplete; photography as anger management is an inference from avoiding overwhelm and staying grounded.']),
        review_record(ROOTS[2]+':I003-C-Q02', 'CONFIRMED_GOLD_DEFECT',
                      ['R01:S01:T001', 'R01:S01:T008', 'R01:S01:T012', ROOTS[2]+':P01:T008', 'R01:S02:T011'],
                      'Gold C,E omits A despite the R01 discussion resolving their disagreement. E wrongly treats creating reunion memories as a way to move past interpersonal conflict: that reunion exchange concerns nostalgia, not conflict.'),
        review_record(ROOTS[2]+':I002-C-Q03', 'REVIEWED_WITH_UNRESOLVED_AMBIGUITY',
                      [ROOTS[2]+':P03:T001', ROOTS[2]+':P02:T003', ROOTS[2]+':P02:T009', ROOTS[2]+':P03:T004', 'R01:S02:T001', 'R01:S02:T007'],
                      'A is explicit preparation. C religious study and D volunteering are explicitly reported explorations and learning. Historical A,C,D can reflect broad learning/preparation; there is no uniquely specified definition of new experiences. B cramming and E reunion organization may also fit a broader reading. This ambiguity is recorded, not silently certified or converted into a new semantic gate.',
                      ['Question scope is broad; membership of B/E and the preparation-vs-participation distinction remain ambiguous. Historical gold is not certified.']),
        review_record(ROOTS[3]+':I003-C-Q02', 'CONFIRMED_GOLD_DEFECT',
                      ['R01:S01:T001', 'R01:S01:T003', 'R01:S03:T001', 'R01:S03:T007'],
                      'Gold D says finalized after lunch; transcript says ready to go before the planned after-lunch submission. Option C, revised after boss feedback, is explicitly true in R01:S03. D is not supported as the best timeline.'),
        review_record(ROOTS[3]+':I003-C-Q03', 'REVIEWED_NO_CLEAR_GOLD_ERROR',
                      [ROOTS[3]+':P01:T001', 'R01:S01:T003', 'R01:S03:T003', 'R01:S03:T007'],
                      'A morning draft; B timeline revision; D budget detail and E considering a backup are explicitly stated. C discussion during a trip is not stated; the proposal reminiscence occurs after David returned.',
                      ['Original evidence refs omit the clearest budget/timeline request at R01:S03:T003.']),
        review_record(ROOTS[4]+':I003-C-Q01', 'REVIEWED_WITH_UNRESOLVED_AMBIGUITY',
                      ['R01:S01:T001', 'R01:S01:T002', 'R01:S01:T003', 'R01:S01:T011', ROOTS[4]+':P01:T005'],
                      'B waiting for Jamie to return is supported by Robin preparing to call and check his return. A finished wall and D Robin planning Bali do not describe this local pre-building moment. C depends on whether project starts with material preparation or first brick.',
                      ['Before started working and wall project are underspecified; C can fit if started means laying bricks. No unique semantic certification.']),
        review_record(ROOTS[5]+':I001-C-Q03', 'REVIEWED_NO_CLEAR_GOLD_ERROR',
                      [ROOTS[5]+':P02:T009', ROOTS[5]+':P02:T010', ROOTS[5]+':P05:T009', 'R01:S01:T018', 'R01:S01:T020', 'R01:S03:T007', 'R01:S03:T009'],
                      'B cleaning is performed and calms her. C journaling and D exercising are contemplated healthier strategies; E inspirational reading is proposed and linked to coping. A destructive behavior is rejected.',
                      ['C/D are learned or contemplated strategies, not observed completed practices; question does not explicitly require completed practice.'])
    ]


def _planner_record(packet):
    qa = packet['historical_qa']
    return {'root_id': packet['root_id'], 'source_cluster_id': packet['source_candidate_id'],
            'source_candidate_id': packet['source_candidate_id'],
            'historical_qa': {'question_type': qa['question_type'], 'question': qa['question'],
                              'options_or_items': qa['options'] or qa['items'],
                              'canonical_gold': qa['gold'], 'evidence_refs': qa['evidence_refs'],
                              'source_data_status': {'canonical_gold': 'present', 'options_or_items': 'present'}},
            'performance': packet['solver_observations'], 'canonical_observation': {}}


def compile_feedback(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    preflight = legacy.load(COMPILER / 'preflight_report.json')
    # Pin exact historical versions; hashing Dev files is provenance only, not feedback aggregation.
    sources = {key: legacy.source(path, preflight['source_sha256'][key])
               for key, path in preflight['source_paths'].items()}
    frozen_input_sources = {'compiler_preflight': legacy.source(COMPILER/'preflight_report.json'), **sources}
    for name in ('candidate_primary_transition.jsonl', 'feedback_routing_manifest.jsonl',
                 'candidate_feedback_features.jsonl', 'feedback_seed_manifest.jsonl'):
        frozen_input_sources[name] = legacy.source(COMPILER/name)
    for name in ('policy.json', 'excluded_candidates.jsonl', 'history_preservation.json'):
        frozen_input_sources['ordering_'+name] = legacy.source(EXCLUSION/name)
    legacy.write_json(output/'inputs/feedback_source_versions.json', frozen_input_sources)
    source_snapshots = {}
    for name, path in {'old_compiler.py': Path(original.__file__),
                       'old_empirical_adapter.py': Path(legacy.__file__),
                       'feedback_guidance_v1.md': legacy.TEMPLATE}.items():
        source_snapshots[name] = legacy.write_frozen(output/'inputs/feedback_compiler_snapshots'/name, path.read_bytes())
    frozen_input_sources['code_snapshots'] = source_snapshots
    all_transitions = list(jsonl(COMPILER/'candidate_primary_transition.jsonl'))
    transitions = [x for x in all_transitions if x['root_id'] in ROOTS]
    difficulty = {x['candidate_id']: x for x in jsonl(sources['difficulty_manifest']['path']) if x['root_id'] in ROOTS}
    routes = {x['candidate_id']: x for x in jsonl(COMPILER/'feedback_routing_manifest.jsonl') if x['root_id'] in ROOTS}
    features = {x['candidate_id']: x for x in jsonl(COMPILER/'candidate_feedback_features.jsonl') if x['root_id'] in ROOTS}
    plans = {x['root_id']: x for x in jsonl(sources['round1_root_manifest']['path']) if x['root_id'] in ROOTS}
    exposure = {x['candidate_id']: x for x in jsonl(sources['candidate_training_exposure']['path']) if x['root_id'] in ROOTS}
    blacklist = {x['candidate_id'] for x in legacy.load(sources['confirmed_invalid_blacklist']['path'])['entries']}
    assert len({x['root_id'] for x in all_transitions}) == 70
    assert len(all_transitions) == 2294 and all(x['split'] == 'train' for x in all_transitions)
    old_eligible = {root: sum(x['root_id']==root and x['question_type'] in ALLOWED and
                             x['lifecycle_route']=='UNTOUCHED_PROBE_ELIGIBLE' for x in routes.values()) for root in ROOTS}
    assert old_eligible == EXPECTED_OLD_ELIGIBLE, old_eligible
    reviews = manual_reviews()
    review_by_id = {x['candidate_id']: x for x in reviews}
    new_defects = {x['candidate_id'] for x in reviews if x['confirmed_source_defect']}
    manual_artifact = artifact_jsonl(output/'inputs/seed_manual_review.jsonl', reviews)
    eligible = selected_rows(transitions, blacklist | new_defects)
    if any(not eligible[root] for root in ROOTS):
        raise ValueError('root_without_usable_feedback_source')
    chosen = {root: eligible[root][0] for root in ROOTS}
    chosen_ids = {x['candidate_id'] for x in chosen.values()}
    trained = {model: {x['candidate_id']: x for x in jsonl(sources[key]['path']) if x['candidate_id'] in chosen_ids}
               for model,key in (('M_clean_policy','clean_candidate_results'),('M_inclusive_policy','inclusive_candidate_results'))}
    initial_ranks = selected_rows(transitions, blacklist)
    audit=[]
    for row in transitions:
        cid = row['candidate_id']; source_row=difficulty[cid]
        assert source_row['split']=='train' and source_row['question_type']==row['question_type']
        assert source_row['root_id']==row['root_id'] and exposure[cid]['training_exposure']==row['training_exposure']
        allowed,reasons=eligibility(row,blacklist|new_defects)
        ranked = initial_ranks[row['root_id']]
        original_rank = next((i+1 for i,x in enumerate(ranked) if x['candidate_id']==cid),None)
        audit.append({'root_id':row['root_id'],'candidate_id':cid,'question_type':row['question_type'],
                      'candidate_sha256':legacy.digest_bytes(legacy.canonical(source_row['candidate']).encode()),
                      'old_lifecycle_route':routes[cid]['lifecycle_route'],
                      'active_feedback_eligible':allowed,'exclusion_reasons':reasons,
                      'rank_before_manual_confirmed_defects':original_rank,
                      'rank_key':list(original._frontier_rank(row)) if original_rank else None,
                      'selected':cid in chosen_ids,'manual_review':review_by_id.get(cid),
                      'historical_sources':{'difficulty_manifest':sources['difficulty_manifest'],
                                            'transition':frozen_input_sources['candidate_primary_transition.jsonl']}})
    selection_artifact=artifact_jsonl(output/'feedback_selection_audit.jsonl',audit)
    template = legacy.TEMPLATE.read_text()
    intent_template = legacy.template_block(template,'## Intent Miner 附加段（intent_suffix）')
    writer_template = legacy.template_block(template,'## QA Writer 附加段（slot_suffixes）')
    assert intent_template.count('{INTENT_FEEDBACK_JSON}')==1
    m0_contract=legacy.load(sources['solver_contract']['path'])
    identities={'M0':{'model_name':'M0','checkpoint':m0_contract['solver_checkpoint'],
                      'model_identity_source':sources['solver_contract'],
                      'checkpoint_identity_sha256':legacy.digest_bytes(legacy.canonical(m0_contract['checkpoint_identity']).encode())}}
    for model,folder in legacy.MODELS.items():
        if folder:
            identity_path=legacy.TRAINED_RUN/folder/'model_execution_report.json'; identity=legacy.load(identity_path)
            identities[model]={key:identity[key] for key in ('model_name','inference_model_path','model_fingerprint_sha256','solver_model_identifier')}
            identities[model]['model_identity_source']=legacy.source(identity_path)
    manifests={arm:{'roots':{}} for arm in ('feedback_off','feedback_on')}
    reports=[]
    old_planner_path=RUNS/'h4_round1_r03_r1_aligned_feedback_inputs_v1/manifest.json'
    old_planner=legacy.load(old_planner_path)
    old_seeds={x['root_id']:x for x in jsonl(COMPILER/'feedback_seed_manifest.jsonl') if x['root_id'] in ROOTS}
    planner_audit=[]
    from ..h4_round1_r03_feedback_view_v2.renderers import render_expansion_planner
    for root in ROOTS:
        row=chosen[root];cid=row['candidate_id']; candidate_row=difficulty[cid];candidate=candidate_row['candidate']
        assert cid in review_by_id and not review_by_id[cid]['confirmed_source_defect']
        assert legacy.digest_bytes(legacy.canonical(candidate).encode())==features[cid]['candidate_sha256']
        plan=plans[root]
        dialogue_source=legacy.source(plan['canonical_dialogue_path'],plan['canonical_dialogue_sha256'])
        dialogue=legacy.load(dialogue_source['path']);refs=[x['turn_ref'] for x in dialogue['turns']]
        assert dialogue['root_id']==root and len(refs)==len(set(refs))
        assert any(x.startswith('R01:') for x in refs) and all(x.startswith('R01:') or x.startswith(root+':P') for x in refs)
        assert candidate['evidence_refs'] and set(candidate['evidence_refs'])<=set(refs)
        assert set(review_by_id[cid]['evidence_refs'])<=set(refs)
        pool_source=legacy.source(candidate_row['candidate_pool_path'],candidate_row['candidate_pool_sha256'])
        raw_dialogue_source=legacy.source(candidate_row['dialogue_path'],candidate_row['dialogue_sha256'])
        raw_dialogue=legacy.load(raw_dialogue_source['path'])
        assert raw_dialogue['turns']==dialogue['turns'],('historical_qa_vs_r1_transcript_mismatch',root)
        pool=legacy.load(pool_source['path'])
        pool_candidates=pool if isinstance(pool,list) else pool.get('candidates',[])
        assert any(x==candidate for x in pool_candidates),('candidate_pool_record_mismatch',cid)
        observations={}; request_sets=[]
        for model,folder in legacy.MODELS.items():
            if folder is None:
                result_path=legacy.M0_RUN/'candidate_results'/(legacy.digest_bytes(cid.encode())+'.json')
                result=legacy.load(result_path);result_source=legacy.source(result_path)
            else:
                result=trained[model][cid];result_source={**sources['clean_candidate_results' if model=='M_clean_policy' else 'inclusive_candidate_results'], 'selector':{'candidate_id':cid}}
            assert all(result[field]==candidate_row[field] for field in ('candidate_id','root_id','candidate_pool_sha256','dialogue_sha256'))
            observed=legacy.observations(result,candidate['gold'],row['models'][model])
            assert all(v['unscored_or_unknown_count']==0 and v['unknown_correctness_count']==0 for v in observed.values())
            observations[model]={'model_identity':identities[model],'by_context':observed,'raw_result_source':result_source}
            request_sets.append({(x['request_id'],x['context_mode'],x['variant_id'],x['slot_id'],x['seed']) for x in result['requests']})
            legacy.write_json(output/'inputs/feedback_on'/root/'historical_solver_requests'/f'{model}.json',result)
        assert all(keys==request_sets[0] for keys in request_sets)
        citations={'seed_manifest':{**selection_artifact,'selector':{'root_id':root,'candidate_id':cid,'selected':True}},
                   'historical_qa':{**sources['difficulty_manifest'],'selector':{'candidate_id':cid}},
                   'historical_candidate_pool':pool_source,'canonical_r1_dialogue':dialogue_source}
        packet={'packet_version':'r1_aligned_empirical_qa_feedback_v1','root_id':root,'historical_fact_scope':'Parent+R01',
                'source_candidate_id':cid,'historical_qa':{key:candidate[key] for key in ('question_type','question','options','items','gold','evidence_refs')},
                'solver_observations':observations,'sources':citations}
        # Independent envelope does not enter the two-field context loader.
        packets={'feedback_on':packet,
                 'feedback_off':{'packet_version':'r1_aligned_empirical_qa_feedback_v1','root_id':root,
                                 'feedback_status':'NO_ACTIVE_FEEDBACK','historical_fact_scope':'Parent+R01',
                                 'source_candidate_id':None,'historical_qa':None,'solver_observations':{},'sources':{}}}
        contexts={'feedback_on':{'intent_suffix':intent_template.replace('{INTENT_FEEDBACK_JSON}',json.dumps(packet,ensure_ascii=False,indent=2)),
                                 'slot_suffixes':[writer_template]*3},
                  'feedback_off':{'intent_suffix':'','slot_suffixes':['','','']}}
        arm_report={}
        for arm in manifests:
            folder=output/'inputs'/arm/root
            pkt=legacy.write_json(folder/'feedback_packet.json',packets[arm]);ctx=legacy.write_json(folder/'feedback_context.json',contexts[arm])
            manifests[arm]['roots'][root]=ctx
            arm_report[arm]={'feedback_packet':pkt,'feedback_context':ctx}
        reports.append({'root_id':root,'seed_candidate_id':cid,'seed_question_type':candidate['question_type'],
                        'canonical_r1_dialogue':dialogue_source,'historical_qa_dialogue':raw_dialogue_source,
                        'historical_qa':citations['historical_qa'],'selected_manual_review':review_by_id[cid],
                        'solver_requests_per_model':len(request_sets[0]),'model_identities':identities,'arms':arm_report})
        # Offline original Planner renderer preview. Old derived card is rejected wholesale.
        old_item=next(x for x in old_planner['roots'] if x['root_id']==root)
        old_source=legacy.source(old_item['feedback_input'],old_item['feedback_input_sha256'])
        old_record=legacy.load(old_source['path'])
        old_cid=old_seeds[root]['candidate_id'];old_row=next(x for x in transitions if x['candidate_id']==old_cid)
        assert not eligibility(old_row,blacklist|new_defects)[0] and old_row['question_type']=='ordering'
        assert old_record['performance']['feedback_card']['target']['source_candidate_sha256'] in (
            difficulty[old_cid]['candidate_pool_sha256'],features[old_cid]['candidate_sha256'])
        assert eligibility(row,blacklist|new_defects)[0]
        preview_record=_planner_record(packet)
        preview=render_expansion_planner(record=preview_record,dialogue=dialogue,strict_speaker_refs=False)
        assert not any(key in legacy.canonical(preview['feedback_view']) for key in ('coverage_gaps','gap_type','global_priority','desired_event_operator'))
        record_artifact=legacy.write_json(output/'planner_feedback_preview'/root/'feedback_record.json',preview_record)
        preview_artifact=legacy.write_json(output/'planner_feedback_preview'/root/'rendered_preview.json',preview)
        planner_audit.append({'root_id':root,'old_input':old_source,'excluded_old_seed':old_cid,
                              'old_derived_performance_removed_entirely':True,
                              'new_source_candidate_id':cid,'same_shared_eligibility':True,
                              'feedback_record':record_artifact,'preview':preview_artifact,
                              'planner_live_calls':0,'memory_calls':0,
                              'scope':'OFFLINE_PREVIEW_ONLY; no existing Planner input/prompt rewritten; no global branch switch'})
    manifest_artifacts={arm:legacy.write_json(output/'inputs'/arm/'feedback_context_manifest.json',value) for arm,value in manifests.items()}
    ordering_ids={x['candidate_id'] for x in transitions if x['question_type']=='ordering'}
    excluded_records={x['candidate_id']:x for x in jsonl(EXCLUSION/'excluded_candidates.jsonl') if x['candidate_id'] in ordering_ids}
    assert ordering_ids==set(excluded_records)
    history=[]
    for cid in sorted(ordering_ids):
        src=difficulty[cid]; original_exclusion=excluded_records[cid]
        assert src['candidate_pool_sha256']==original_exclusion['source_snapshot']['candidate_pool_sha256']
        assert src['dialogue_sha256']==original_exclusion['source_snapshot']['dialogue_sha256']
        history.append({'root_id':src['root_id'],'candidate_id':cid,'question_type':'ordering','stem':src['candidate']['question'],
                        'candidate':src['candidate'],'candidate_sha256':legacy.digest_bytes(legacy.canonical(src['candidate']).encode()),
                        'solver_prompt_fingerprint':original_exclusion['solver_prompt_fingerprint'],
                        'source_snapshot':original_exclusion['source_snapshot'],
                        'active_feedback_eligible':False,'model_visible':False,'new_generation_dedup_gate':False})
    history_artifact=artifact_jsonl(output/'inputs/history/ordering_records.jsonl',history)
    history_report={'status':'PRESERVED_OFFLINE_ONLY','record_count':len(history),'history':history_artifact,
                    'source_policy':frozen_input_sources['ordering_policy.json'],
                    'source_history_preservation':frozen_input_sources['ordering_history_preservation.json'],
                    'historical_records_modified':False,'delivered_to_feedback_off':False,'delivered_to_feedback_on':False,
                    'second_generation_dedup_gate_added':False}
    legacy.write_json(output/'history_preservation.json',history_report)
    ordering_audit={'status':'PASS','scope':'six fixed Train roots; same QA/Planner source eligibility function',
                    'allowlist':sorted(ALLOWED),'excluded_ordering_count':len(ordering_ids),
                    'nonordering_old_untouched_eligible_per_root':old_eligible,
                    'selected_nonordering_seeds':{root:chosen[root]['candidate_id'] for root in ROOTS},
                    'all_old_six_ordering_seeds_replaced':True,'ordering_mask_before_rank_aggregate_render':True,
                    'ordering_performance_or_derived_guidance_in_active_packet':False,
                    'history_is_not_a_generation_admission_gate':True,
                    'planner_preview_manifest':legacy.write_json(output/'planner_feedback_preview/manifest.json',{
                        'status':'OFFLINE_PREVIEW_ONLY','old_manifest':legacy.source(old_planner_path),'roots':planner_audit,'api_calls':0,'memory_calls':0})}
    legacy.write_json(output/'ordering_exclusion_audit.json',ordering_audit)
    checks={'six_nonempty_on_and_empty_off_contexts':len(reports)==6,
            'source_versions_exact':True,'same_parent_plus_r01_scope':True,'candidate_pool_exact':True,
            'paired_three_model_full_no_raw_requests':True,'historical_gold_reviewed_not_certified':True,
            'old_safe_frontier_and_frontier_rank_reused':True,'original_empirical_template_and_writer_suffixes':True,
            'ordering_history_preserved_offline':True,'planner_shared_mask_preview':True,
            'no_dev_aggregate_or_gap_narrative_transmitted':True}
    report={'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,'roots':reports,
            'feedback_context_manifests':manifest_artifacts,'source_versions':frozen_input_sources,
            'template':source_snapshots['feedback_guidance_v1.md'],'compiler':legacy.source(Path(__file__)),
            'selection_audit':selection_artifact,'seed_manual_reviews':manual_artifact,
            'confirmed_new_source_defect_count':len(new_defects),
            'remaining_ambiguity_roots':[x['root_id'] for x in reports if x['selected_manual_review']['review_status']=='REVIEWED_WITH_UNRESOLVED_AMBIGUITY'],
            'content_limit':'Content reading is distinct from program eligibility; ambiguity is recorded and historical Gold not certified.',
            'api_calls':0,'memory_calls':0,'qa_generation_calls':0,'solver_calls':0}
    legacy.write_json(output/'inputs/feedback_preflight_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();result=compile_feedback(args.output)
    print(json.dumps({'status':result['status'],'roots':len(result['roots']),
                      'feedback_context_manifests':result['feedback_context_manifests'],
                      'confirmed_new_source_defect_count':result['confirmed_new_source_defect_count']}))

if __name__=='__main__':main()
