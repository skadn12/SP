"""Independent reconstruction of v89 shard admission and parent provenance.

This module reads saved records. It does not author semantic judgments, invoke
production admission validators, or interpret source READY as answer approval.
"""
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
GATE = PROJECT/'offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py'
GATE_SHA = '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'
CONSISTENCY = {'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition'}


def dt(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('timezone_required')
    return parsed


def source_contract(source, review, row, check):
    root=row['root_id']; suffix=':'+root
    turns=source['turns']; lookup={t['turn_ref']:t for t in turns}
    check(bool(turns) and len(lookup)==len(turns) and source['root_id']==root,
          'source_unique_turns_and_root'+suffix)
    check(review['schema_version']=='dialogue_consistency_v1' and review['root_id']==root
          and review['source']==row['source'] and review['review_scope']=='source_consistency_only'
          and bool(review['reviewer'].strip()) and bool(review['limitations'].strip()), 'source_review_identity'+suffix)
    disclosure={'full_source_read':True,'current_question_gold_read':False,'current_solver_scores_read':False,
                'historical_question_gold_known':True,'fresh_blind_review':False}
    check(all(review.get(k) is v for k,v in disclosure.items()), 'source_review_exposure_disclosure'+suffix)
    check(review['turn_refs_read']==[t['turn_ref'] for t in turns]
          and set(review['checks_completed'])==CONSISTENCY
          and all(v is True for v in review['checks_completed'].values()), 'source_review_all_turns_all_checks'+suffix)
    groups=OrderedDict()
    for turn in turns:
        check(turn['origin'] in ('source_grounded','generated_extension','provisional_candidate')
              and all(isinstance(turn[k],str) and turn[k].strip() for k in ('turn_ref','text','speaker')),
              'source_turn_schema'+suffix+':'+turn['turn_ref'])
        key=turn['session_ref'] if turn['origin']=='source_grounded' else turn['scene_ref']
        groups.setdefault(key,[]).append(turn)
    check([t['turn_ref'] for rows in groups.values() for t in rows]==list(lookup), 'source_contiguous_units'+suffix)
    units=review['units']; findings=review['findings']; fids=[f['finding_id'] for f in findings]
    check([u['unit_id'] for u in units]==list(groups) and len(set(fids))==len(fids), 'source_unit_and_finding_identity'+suffix)
    turn_units={t['turn_ref']:uid for uid,rows in groups.items() for t in rows}
    for finding in findings:
        key=suffix+':'+finding['finding_id']; cites=finding['evidence']; us=finding['unit_ids']
        check(finding['severity'] in ('BLOCKING','WARNING') and finding['kind'] in CONSISTENCY|{'naturalness'}
              and bool(finding['reason'].strip()) and bool(finding['proposed_action'].strip())
              and us and len(set(us))==len(us) and set(us)<=set(groups), 'source_finding_schema'+key)
        check(bool(cites) and all(set(e)=={'turn_ref','quote'} and e['quote']
              and e['turn_ref'] in lookup and e['quote'] in lookup[e['turn_ref']]['text'] for e in cites)
              and {turn_units[e['turn_ref']] for e in cites}==set(us), 'source_finding_exact_citations'+key)
    for unit in units:
        uid=unit['unit_id']; rows=groups[uid]; relevant=[f for f in findings if uid in f['unit_ids']]
        expected='HOLD' if any(f['severity']=='BLOCKING' for f in relevant) else 'WARN' if relevant else 'PASS'
        check(len({t['origin'] for t in rows})==1 and unit['origin']==rows[0]['origin']
              and unit['turn_refs']==[t['turn_ref'] for t in rows] and unit['reason'].strip()
              and unit['decision']==expected and unit['finding_ids']==[f['finding_id'] for f in relevant],
              'source_unit_decision_reconstructed'+suffix+':'+uid)
    check(review['decision']=='READY' and not any(f['severity']=='BLOCKING' for f in findings),
          'source_ready_reconstructed_without_blocking_findings'+suffix)


def validate_admission_contract(config, home, read_fn, bind_fn, check_fn):
    home=Path(home).resolve(); read=read_fn; bind=bind_fn; check=check_fn
    slots=config['multiselect_config']['slots']; roots=[s['root_id'] for s in slots]; n=len(roots)
    check(2<=n<=4 and len(set(roots))==n and roots==[s['root_id'] for s in config['ordering_slots']],
          'two_to_four_unique_roots_both_lanes')
    check(config==read(home/'CONFIG_TEMPLATE.json') and config['version']=='v89_full_turn_mixed_shard_v1', 'frozen_config_exact_template')
    multi=config['multiselect_config']
    check(config['max_generation_http']==3*n and config['max_candidates']==7*n and config['max_solver_calls_per_model']==114*n
          and multi['max_http']==2*n and multi['max_questions']==5*n and multi['max_calls_per_model']==90*n
          and multi['family_count']==n and multi['max_supplemental_requests_per_family']==1,
          'dynamic_shard_all_budgets')
    check(multi['new_child_target_distribution']=={str(g):n for g in range(2,6)}
          and multi['control_target_distribution']=={'6':n}
          and multi['selection_rule']=='all_answers_valid_and_0_lt_trained_partial_mean_lt_1', 'unchanged_child_targets_and_selection')
    installation=read(home/'SHARD_ENGINE_INSTALLATION.json')
    check(installation['version']=='v89_shard_engine_installation_v1'
          and installation['shard_home']==str(home) and installation['training_allowed'] is False,
          'separate_immutable_shard_engine')
    for row in installation['files']:
        check(bind(row['path'])==row,'installed_production_file:'+row['path'])
    check({Path(r['path']).resolve() for r in installation['files']}==set(home.glob('*.py')),
          'all_installed_production_python_bound')
    admission=read(home/'SOURCE_ADMISSION.COMPLETE.json'); summary=read(home/'SOURCE_ADMISSION_SUMMARY.json')
    inputs=read(home/'ADMISSION_INPUT.json'); frozen=read(home/'SHARD_INPUTS.COMPLETE.json')
    pilot=read(inputs['pilot_plan']['path'])
    check(bind(inputs['pilot_plan']['path'])==inputs['pilot_plan'] and pilot['version']=='v89_20_candidate_root_scale_pilot_v1'
          and pilot['candidate_root_budget']==20 and pilot['source_split']=='train'
          and pilot['training_allowed'] is False and pilot['formal_training_admission'] is False, 'pilot_scope_binding')
    check(inputs['source_gate']=={'path':str(GATE),'sha256':GATE_SHA}==bind(GATE), 'unchanged_v77_source_gate_binding')
    excluded={str(v) for v in pilot['exclude_repeated_roots']+pilot['exclude_known_hold_roots']}
    check(not {r.rsplit('-',1)[-1] for r in roots}&excluded, 'new_roots_exclude_prior_repeats_and_known_holds')
    check(inputs['version']=='v89_shard_admission_input_v1' and inputs['split']=='train'
          and [r['root_id'] for r in inputs['roots']]==roots==summary['ready_roots']==admission['ready_roots']==frozen['roots']
          and summary['held_roots']==admission['held_roots']==[] and summary['production_readiness']=='READY', 'admitted_train_root_coverage')
    for container,name in ((admission,'source_admission'),(frozen,'shard_inputs')):
        check(container['status']=='FROZEN' and container['training_allowed'] is False, name+'_frozen_no_training')
        paths=[r['path'] for r in container['files']]
        check(len(set(paths))==len(paths),name+'_unique_frozen_files')
        for row in container['files']:
            check(bind(row['path'])==row,name+'_binding:'+row['path'])
    check(config['source_consistency']['upstream_complete']==bind(home/'SOURCE_ADMISSION.COMPLETE.json')
          and config['source_consistency']['selected_roots']==roots
          and config['source_consistency']['source_review_is_question_answer_certification'] is False,
          'ready_is_not_answer_certification')
    prep=read(home/'BATCH_PREPARATION.STARTED.json')['started_at_utc']
    check(dt(admission['completed_at_utc'])<=dt(frozen['completed_at_utc'])<=dt(prep), 'source_admission_and_inputs_frozen_before_preparation')
    frozen_paths={r['path'] for r in frozen['files']}; expected_people=set(); sources={}
    required={home/name for name in ('CONFIG.json','CONFIG_TEMPLATE.json','ADMISSION_INPUT.json','SOURCE_ADMISSION.COMPLETE.json','SOURCE_ADMISSION_SUMMARY.json','SHARD_ENGINE_INSTALLATION.json')}
    required.update(home.glob('*.py')); required.update(home.glob('*.md')); required.add(Path(config['baseline_pool']))
    required.update(Path(config[k]['path']) for k in ('ordering_anchor_plan','character_registry','actor_reference_policy'))
    for slot in slots:
        required.update(Path(slot[k]) for k in ('source_path','history_path','parent_pool_path','parent_review_path','parent_provenance_path'))
        required.update(Path(p) for p in slot.get('prior_pool_paths',[]))
    for slot in config['ordering_slots']:
        required.add(Path(slot['prior_pool_path'])); required.update(Path(p) for p in slot.get('prior_pool_paths',[]))
    check(frozen_paths=={str(p.resolve()) for p in required},'shard_freeze_exact_input_file_coverage')
    admission_required={home/'ADMISSION_INPUT.json',home/'SOURCE_ADMISSION_SUMMARY.json',home/'source_admission.py',home/'SHARD_ENGINE_INSTALLATION.json',GATE,Path(inputs['pilot_plan']['path'])}
    for row in inputs['roots']:
        admission_required.update(Path(row[k]['path']) for k in ('source','review'))
    check({r['path'] for r in admission['files']}=={str(p.resolve()) for p in admission_required},'source_admission_exact_frozen_file_coverage')
    by_root={r['root_id']:r for r in summary['roots']}
    for slot,row in zip(slots,inputs['roots']):
        root=slot['root_id']; source=read(slot['source_path']); binding=bind(slot['source_path']); review=read(row['review']['path'])
        ordering=next(s for s in config['ordering_slots'] if s['root_id']==root)
        check(row['split']==slot['split']=='train' and row['source']==binding==by_root[root]['source']
              and binding=={'path':slot['source_path'],'sha256':slot['source_sha256']}
              and by_root[root]['decision']=='READY' and by_root[root]['blocking_findings']==[], 'complete_selected_source_binding:'+root)
        check(all(slot[k]==ordering[k] for k in ('source_path','source_sha256','history_path','history_sha256'))
              and Path(slot['source_path']).read_bytes()==Path(slot['history_path']).read_bytes()
              and slot['history_sha256']==slot['source_sha256'] and ordering['candidate_count']==2
              and slot['target_counts']==[2,3,4,5] and slot['control_count']==6, 'same_complete_source_both_lanes_and_targets:'+root)
        check(row['review']==bind(row['review']['path'])==by_root[root]['review'], 'source_review_binding:'+root)
        source_contract(source,review,row,check)
        check(dt(review['created_at_utc'])<=dt(admission['completed_at_utc']), 'source_review_saved_before_admission:'+root)
        judgment=read(slot['parent_review_path']); provenance=read(slot['parent_provenance_path'])
        check(bind(slot['parent_review_path'])['sha256']==slot['parent_review_sha256']
              and bind(slot['parent_provenance_path'])['sha256']==slot['parent_provenance_sha256'], 'parent_review_provenance_binding:'+root)
        public=read(provenance['public_parent']['path'])
        pool=read(slot['parent_pool_path'])['questions']; parents=[q for q in pool if q['candidate_id']==slot['parent_candidate_id']]
        check(len(parents)==1,'unique_parent_pool_member:'+root); parent=parents[0]
        check(judgment['version']=='v89_parent_source_judgments_v1'
              and judgment.get('content_status','CORE_USABLE')=='CORE_USABLE'
              and judgment['public_question']==public and judgment['public_question_path']==provenance['public_parent']['path']
              and judgment['public_question_sha256']==provenance['public_parent']['sha256']
              and judgment['source_path']==public['source_path']==slot['source_path']
              and judgment['source_sha256']==public['source_sha256']==slot['source_sha256']
              and judgment['root_id']==public['root_id']==parent['root_id']==root
              and judgment['candidate_id']==public['candidate_id']==parent['candidate_id']
              and judgment['full_source_read'] is True and judgment['full_source_turn_count']==len(source['turns']), 'saved_current_parent_full_source_review:'+root)
        check(judgment['known_old_parent_gold'] is True and judgment['fresh_blind_review'] is False
              and judgment['current_parent_private_gold_read_before_judgments_saved'] is False
              and judgment['current_parent_private_evidence_read_before_judgments_saved'] is False
              and judgment['current_solver_results_read'] is False
              and provenance['historical_gold_known'] is True and provenance['fresh_blind_review'] is False
              and provenance['current_solver_scores_read'] is False and provenance['training_allowed'] is False,
              'honest_historical_exposure_current_private_disclosure:'+root)
        for key in ('public_parent','source_judgments','source','historical_parent_pool','public_preparation'):
            check(bind(provenance[key]['path'])==provenance[key], 'parent_provenance_artifact:'+root+':'+key)
        check(provenance['source']==binding and provenance['source_judgments']==bind(slot['parent_review_path'])
              and provenance['source_judgments_saved_at_utc']==judgment['created_at_utc']
              and provenance['current_private_parent_created_after_saved_source_judgments'] is True
              and dt(judgment['created_at_utc'])<=dt(provenance['created_at_utc'])<=dt(frozen['completed_at_utc']),
              'saved_source_judgments_precede_private_parent_and_freeze:'+root)
        rows=judgment['judgments']; lookup={t['turn_ref']:t for t in source['turns']}
        check(len(rows)==6 and {j['option'] for j in rows}==set('ABCDEF')==set(public['options'])
              and parent['gold']==list('ABCDEF') and parent['options']==public['options']
              and parent['training_allowed'] is False and parent['formal_training_admission'] is False,
              'six_supported_parent_identity_and_no_training:'+root)
        for item in rows:
            label=item['option']; cites=item['required_evidence']
            check(item['proposition']==public['options'][label] and item['source_judgment']=='supported'
                  and item['entailed'] is True and item['reason'].strip() and cites, 'parent_supported_option:'+root+':'+label)
            for e in cites:
                check(e['turn_ref'] in lookup and e['speaker']==lookup[e['turn_ref']]['speaker']
                      and e['quote'] and e['quote'] in lookup[e['turn_ref']]['text'], 'parent_exact_evidence:'+root+':'+label)
            check(parent['option_evidence'][label]=={**cites[0],'relation':'supports','reason':item['reason'],'rationale':item['reason']},
                  'private_parent_exact_saved_judgment_projection:'+root+':'+label)
        for key in ('source_path','history_path','parent_pool_path','parent_review_path','parent_provenance_path'):
            check(str(Path(slot[key]).resolve()) in frozen_paths,'shard_freeze_required_input:'+root+':'+key)
        for p in slot.get('prior_pool_paths',[]):
            check(str(Path(p).resolve()) in frozen_paths,'shard_freeze_public_prior:'+root+':'+p)
        sources[root]=source; expected_people.update((root,t['speaker']) for t in source['turns'])
    registry=read(config['character_registry']['path']); policy=read(config['actor_reference_policy']['path'])
    check(bind(config['character_registry']['path'])==config['character_registry']
          and bind(config['actor_reference_policy']['path'])==config['actor_reference_policy']
          and policy['registry']==config['character_registry'] and policy['new_source_facts'] is False
          and policy['automatic_repair'] is False and policy['output_style']=='use_source_names_or_explicit_roles_for_people'
          and policy['forbidden_gendered_pronouns']==['he','him','his','himself','she','her','hers','herself'],
          'name_registry_and_actor_policy_frozen_no_new_facts')
    people=registry['characters']; actual=[(r['root_id'],r['canonical_name']) for r in people]
    check(len(set(actual))==len(actual) and set(actual)==expected_people
          and all(r[key]=={'value':None,'status':'not_inferred_or_used'} for r in people for key in ('gender','third_person_pronouns')),
          'registry_exact_source_names_no_inferred_gender')
    check(len(registry['source_bindings'])==n and {r['root_id'] for r in registry['source_bindings']}==set(roots),'registry_source_root_coverage')
    for row in registry['source_bindings']:
        root=row['root_id']; src=sources[root]
        check({'path':row['path'],'sha256':row['sha256']}==by_root[root]['source']
              and row['source_review']==by_root[root]['review'] and row['full_turn_count']==len(src['turns'])
              and row['all_turns_reviewed'] is True and row['speaker_names']==sorted({t['speaker'] for t in src['turns']}),
              'registry_complete_source_binding:'+root)
    return {'source_admission':bind(home/'SOURCE_ADMISSION.COMPLETE.json'),'shard_inputs':bind(home/'SHARD_INPUTS.COMPLETE.json'),
            'ready_roots':roots,'root_count':n,'fresh_current_parent_review_count':n,'fresh_blind_parent_review':False,
            'all_roots_first_round_train_sources':True,'independent_test_claimed':False,'source_answer_semantics_recertified':False}
