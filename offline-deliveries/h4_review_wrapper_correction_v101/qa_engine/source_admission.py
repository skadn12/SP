"""Admit only independently reviewed full TRAIN sources in this installed shard."""
import importlib.util
from pathlib import Path
from common import HERE, DELIVERIES, bind, need, now, read, require_bindings, require_shard, sha, write
import fresh_parent
import batch_scope
import expansion_binding

GATE = DELIVERIES / 'h4_source_consistency_minfix_v77/source_gate.py'
GATE_SHA = '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'


def gate():
    need(sha(GATE) == GATE_SHA, 'source_gate_changed')
    spec=importlib.util.spec_from_file_location('_v98_unchanged_full_source_gate',GATE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def inputs():
    require_shard()
    value=read(HERE/'ADMISSION_INPUT.json')
    need(value['version']=='v98_shard_admission_input_v1' and value['split']=='train'
         and isinstance(value['shard_id'],str) and value['shard_id']
         and value['source_gate']==bind(GATE)
         and value['source_review_validator']==bind(HERE/'fresh_parent.py')
         and value['source_review_contract']=='v98_full_source_v77_checks_truthful_exposure',
         'shard_admission_input_contract')
    need(bind(value['pilot_plan']['path'])==value['pilot_plan'],'pilot_plan_changed')
    pilot=read(value['pilot_plan']['path'])
    batch_scope.validate_plan(pilot)
    rows=value['roots'];roots=[r['root_id'] for r in rows]
    batch_scope.validate_roots(roots)
    excluded={str(r) for r in pilot['exclude_repeated_roots']+pilot['exclude_known_hold_roots']}
    need(all(root.startswith('cc-train-episode-') and root.rsplit('-',1)[-1] not in excluded for root in roots),
         'shard_contains_excluded_root')
    for row in rows:
        need(row['split']=='train' and bind(row['source']['path'])==row['source']
             and bind(row['review']['path'])==row['review']
             and bind(row['expanded_source_provenance']['path'])==row['expanded_source_provenance'],
             'shard_source_or_review_binding_changed')
        expansion_binding.validate(row['root_id'], row['source'], row['expanded_source_provenance']['path'])
    return value


def reviews():
    value=inputs();gate();rows=[]
    for row in value['roots']:
        review=fresh_parent.validate_source_review({'root_id':row['root_id'],'source':row['source'],
                                         'review_path':row['review']['path']})
        need(review['decision']=='READY','selected_source_not_ready:'+row['root_id'])
        rows.append({'root_id':row['root_id'],'split':'train','decision':'READY',
                     'source':row['source'],'review':row['review'],
                     'turn_count':len(review['turn_refs_read']),'unit_count':len(review['units']),
                     'blocking_findings':[],
                     'warning_findings':[f['finding_id'] for f in review['findings'] if f['severity']=='WARNING']})
    return value,rows


def finalize():
    need(not (HERE/'SOURCE_ADMISSION.COMPLETE.json').exists(),'admission_no_replay')
    value,rows=reviews()
    summary={'version':'v98_shard_source_admission_v1','completed_at_utc':now(),
             'shard_id':value['shard_id'],'roots':rows,'ready_roots':[r['root_id'] for r in rows],
             'held_roots':[],'source_split':'train','source_edits':[],
             'full_source_policy':'hold_entire_full_source_if_any_blocking_unit',
             'question_answer_certification':False,'training_allowed':False,
             'production_readiness':'SOURCE_READY_NOT_PRODUCTION_CERTIFIED'}
    write(HERE/'SOURCE_ADMISSION_SUMMARY.json',summary)
    paths={HERE/'ADMISSION_INPUT.json',HERE/'SOURCE_ADMISSION_SUMMARY.json',HERE/'source_admission.py',
           HERE/'SHARD_ENGINE_INSTALLATION.json',GATE,Path(value['pilot_plan']['path'])}
    for row in value['roots']:paths.update(Path(row[key]['path']) for key in ('source','review','expanded_source_provenance'))
    record={'version':'v98_shard_source_admission_freeze_v1','status':'FROZEN','completed_at_utc':now(),
            'ready_roots':summary['ready_roots'],'held_roots':[],'training_allowed':False,
            'files':[bind(p) for p in sorted(paths)]}
    write(HERE/'SOURCE_ADMISSION.COMPLETE.json',record)
    return record


def verify():
    require_shard()
    record=read(HERE/'SOURCE_ADMISSION.COMPLETE.json')
    need(record['version']=='v98_shard_source_admission_freeze_v1' and record['status']=='FROZEN'
         and record['training_allowed'] is False,'source_admission_frozen_contract')
    value,rows=reviews();saved=read(HERE/'SOURCE_ADMISSION_SUMMARY.json')
    required={HERE/'ADMISSION_INPUT.json',HERE/'SOURCE_ADMISSION_SUMMARY.json',HERE/'source_admission.py',
              HERE/'SHARD_ENGINE_INSTALLATION.json',GATE,Path(value['pilot_plan']['path'])}
    for row in value['roots']:required.update(Path(row[key]['path']) for key in ('source','review','expanded_source_provenance'))
    require_bindings(record['files'],required)
    need(saved['roots']==rows and record['ready_roots']==saved['ready_roots']==[r['root_id'] for r in rows]
         and record['held_roots']==saved['held_roots']==[],'source_admission_decisions_changed')
    return saved


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('finalize','verify'));args=parser.parse_args()
    print(json.dumps(finalize() if args.action=='finalize' else verify(),ensure_ascii=False))
