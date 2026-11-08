"""Freeze each shard's explicit inputs before preparation; no fixed old roots."""
from pathlib import Path
from common import HERE, V75, bind, need, now, read, require_bindings, require_shard, write
import source_admission
import fresh_parent
import batch_scope
import expansion_binding


def validate_unfrozen(value):
    require_shard();summary=source_admission.verify()
    need(value==read(HERE/'CONFIG_TEMPLATE.json'),'shard_config_differs_from_template')
    multi=value['multiselect_config'];slots=multi['slots'];roots=[s['root_id'] for s in slots];n=len(roots)
    batch_scope.validate_roots(roots)
    need(roots==summary['ready_roots']==[s['root_id'] for s in value['ordering_slots']], 'shard_admitted_root_coverage')
    need(value['source_consistency']['upstream_complete']==bind(HERE/'SOURCE_ADMISSION.COMPLETE.json')
         and value['source_consistency']['selected_roots']==roots
         and value['source_consistency']['source_review_is_question_answer_certification'] is False,
         'shard_admission_config_binding')
    need(value['version']=='v98_full_turn_mixed_shard_v1'
         and value['max_generation_http']==3*n and value['max_candidates']==7*n
         and value['max_solver_calls_per_model']==114*n
         and multi['max_http']==2*n and multi['max_questions']==5*n and multi['max_calls_per_model']==90*n
         and multi['family_count']==n,'shard_dynamic_budgets_changed')
    need(multi['new_child_target_distribution']=={str(g):n for g in range(2,6)}
         and multi['control_target_distribution']=={'6':n}
         and multi['max_supplemental_requests_per_family']==1
         and multi['selection_rule']=='all_answers_valid_and_0_lt_trained_partial_mean_lt_1',
         'shard_targets_or_selection_changed')
    by_root={row['root_id']:row for row in summary['roots']}
    for slot in slots:
        need(slot['split']=='train' and slot['target_counts']==[2,3,4,5] and slot['control_count']==6
             and by_root[slot['root_id']]['source']=={'path':slot['source_path'],'sha256':slot['source_sha256']}
             and slot['history_sha256']==slot['source_sha256']
             and Path(slot['history_path']).read_bytes()==Path(slot['source_path']).read_bytes(),
             'shard_slot_source_targets_changed')
        fresh_parent.validate_slot(slot)
        provenance=read(slot['parent_provenance_path'])
        expansion_binding.validate(slot['root_id'], provenance['source'],
                                   provenance['expanded_source_provenance']['path'])
    need(Path(value['multiselect_engine']).resolve()==V75,'v79_engine_changed')
    return summary


def input_paths(value):
    paths={HERE/'CONFIG.json',HERE/'CONFIG_TEMPLATE.json',HERE/'ADMISSION_INPUT.json',
           HERE/'SOURCE_ADMISSION.COMPLETE.json',HERE/'SOURCE_ADMISSION_SUMMARY.json',
           HERE/'SHARD_ENGINE_INSTALLATION.json',HERE/'evidence/INPUT_BUILD_RECEIPT.json',Path(value['baseline_pool'])}
    build_receipt=read(HERE/'evidence/INPUT_BUILD_RECEIPT.json')
    for key in ('spec','builder'):
        need(bind(build_receipt[key]['path'])==build_receipt[key], 'shard_build_input_changed:'+key)
        paths.add(Path(build_receipt[key]['path']))
    paths.update(HERE.glob('*.py'));paths.update(HERE.glob('*.md'))
    for key in ('ordering_anchor_plan','character_registry','actor_reference_policy'):
        need(bind(value[key]['path'])==value[key],'shard_input_binding_changed:'+key)
        paths.add(Path(value[key]['path']))
    for slot in value['multiselect_config']['slots']:
        for key in ('source_path','history_path','parent_pool_path','parent_review_path','parent_provenance_path'):
            paths.add(Path(slot[key]))
        for key in ('parent_review','parent_provenance'):
            need(bind(slot[key+'_path'])['sha256']==slot[key+'_sha256'],'shard_parent_binding_changed:'+key)
        provenance=read(slot['parent_provenance_path'])
        for key in ('public_parent','source_judgments','source','expanded_source_provenance'):
            need(bind(provenance[key]['path'])==provenance[key],'fresh_parent_provenance_input_changed:'+key)
            paths.add(Path(provenance[key]['path']))
        paths.update(Path(p) for p in slot.get('prior_pool_paths',[]))
    for slot in value['ordering_slots']:
        paths.add(Path(slot['prior_pool_path']));paths.update(Path(p) for p in slot.get('prior_pool_paths',[]))
    return paths


def finalize():
    require_shard();need(not (HERE/'SHARD_INPUTS.COMPLETE.json').exists(),'shard_inputs_no_replay')
    need(not (HERE/'BATCH_PREPARATION.STARTED.json').exists(),'cannot_change_inputs_after_preparation')
    value=read(HERE/'CONFIG.json');summary=validate_unfrozen(value)
    record={'version':'v98_shard_inputs_freeze_v1','status':'FROZEN','completed_at_utc':now(),
            'roots':summary['ready_roots'],'training_allowed':False,
            'files':[bind(p) for p in sorted(input_paths(value))]}
    write(HERE/'SHARD_INPUTS.COMPLETE.json',record)
    return record


def verified_summary():
    require_shard();record=read(HERE/'SHARD_INPUTS.COMPLETE.json');value=read(HERE/'CONFIG.json')
    need(record['version']=='v98_shard_inputs_freeze_v1' and record['status']=='FROZEN'
         and record['training_allowed'] is False,'shard_input_freeze_contract')
    require_bindings(record['files'],input_paths(value))
    summary=validate_unfrozen(value)
    need(record['roots']==summary['ready_roots'],'shard_input_roots_changed')
    return summary


def validate(value):
    verified_summary();need(value==read(HERE/'CONFIG_TEMPLATE.json'),'shard_frozen_config_changed')
    return value


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('finalize','verify'));args=parser.parse_args()
    print(json.dumps(finalize() if args.action=='finalize' else verified_summary(),ensure_ascii=False))
