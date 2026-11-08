#!/usr/bin/env python3
"""Versioned mechanical wrapper repair for not-yet-committed OET1 scenes only.

All turns, prompts, semantic attempts, independent review decisions, native
events and previous cumulative files stay unchanged. Existing cumulative files
are verified and reused byte for byte; FAIL/HOLD is never rescued here.
"""
import argparse
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('oet1_original_control',BASE/'scripts/control.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SCENES=old.SCENES;CHECKS=old.CHECKS;read=old.read;bind=old.bind;verify=old.verify;write=old.write;unit=old.unit;accept=old.accept;pending=old.pending

def commit(root,arm,scene):
    u=unit(root,arm);packet=read(u/'SOURCE_AUTHOR_INPUT.json');i=SCENES.index(scene)
    previous=u/SCENES[i-1].replace(':','_')/'CUMULATIVE_SOURCE.json' if i else Path(packet['source_binding']['path'])
    if i:
        receipt=read(previous.parent/'ACCEPTED.json')
        assert receipt['status']=='READY' and receipt['source']==bind(previous) and receipt['review']==bind(previous.parent/'REVIEW.json')
    else:verify(packet['source_binding'])
    source=read(previous);sd=u/scene.replace(':','_');draft=read(sd/'DRAFT.json');cp=sd/'CUMULATIVE_SOURCE.json'
    assert draft['root_id']==root and draft['scene_ref']==scene
    assert draft['input_source_binding']==bind(previous) and draft['author_input_binding']==bind(u/'SOURCE_AUTHOR_INPUT.json')
    if i>=3:assert draft['native_event_binding']==packet['native_events'][i-3]['native_event_binding']
    turns=draft['turns'];assert len(turns)==14
    for index,turn in enumerate(turns,1):
        assert turn['turn_ref']==f'{scene}:T{index:03}' and turn['scene_ref']==scene and turn['speaker'] in packet['speakers'] and turn['text'].strip()
    assert not {t['turn_ref'] for t in source['turns']}&{t['turn_ref'] for t in turns}
    all_turns=source['turns']+turns
    if cp.exists():
        existing=read(cp)
        assert existing['turns']==all_turns and existing['input_source_binding']==bind(previous) and existing['draft_binding']==bind(sd/'DRAFT.json')
        return {'source':bind(cp),'draft':bind(sd/'DRAFT.json'),'needs_independent_review':not (sd/'ACCEPTED.json').exists(),'existing_source_preserved_byte_for_byte':True,'metadata_rewritten':False}
    initial_count=len(packet['full_initial_source']['turns']);baseline=source.get('baseline_turn_count')
    full={**source,'schema_version':'oet1_cumulative_source_v3','scene_ref':scene,'condition':arm,'namespace':'OET1',
        'input_source_binding':bind(previous),'draft_binding':bind(sd/'DRAFT.json'),'turns':all_turns,
        'previous_turn_count':len(source['turns']),'new_turn_count':14,
        'oet1_initial_turn_count':initial_count,'oet1_added_turn_count':14*(i+1),'oet1_total_turn_count':len(all_turns),
        'oet1_initial_historical_extension_turn_count':packet['full_initial_source'].get('cumulative_new_turn_count'),
        'mechanical_metadata_version':'v3_exact_predecessor_and_namespace_counts'}
    if isinstance(baseline,int):full['cumulative_new_turn_count']=len(all_turns)-baseline
    write(cp,full)
    return {'source':bind(cp),'draft':bind(sd/'DRAFT.json'),'needs_independent_review':True,
        'metadata_rewritten':False,'new_scene_metadata_computed_before_first_commit':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['commit','accept','pending']);parser.add_argument('--root');parser.add_argument('--arm');parser.add_argument('--scene');args=parser.parse_args()
    print(json.dumps(pending() if args.action=='pending' else globals()[args.action](args.root,args.arm,args.scene),ensure_ascii=False))
