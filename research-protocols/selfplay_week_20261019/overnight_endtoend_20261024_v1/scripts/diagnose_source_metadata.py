#!/usr/bin/env python3
"""Read-only exact-prefix/stale-wrapper diagnosis; never rescues AI decisions."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p=Path(p).resolve();raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();cases=[]
    for cp in sorted((BASE/'sources').glob('*/*/OET1_*/CUMULATIVE_SOURCE.json')):
        source=read(cp);dp=cp.with_name('DRAFT.json');draft=read(dp);previous_path=Path(draft['input_source_binding']['path']);previous=read(previous_path)
        rp=cp.with_name('REVIEW.json');review=read(rp) if rp.exists() else {};codes=review.get('failure_codes',[])
        category='PACKAGING_METADATA_HOLD' if 'PREFIX_BOUNDARY_METADATA_MISMATCH' in codes else 'CONTENT_TEMPORAL_BINDING_FAIL' if 'UNRESOLVED_PROJECT_YEAR_BINDING' in codes else 'OTHER_NONREADY_REVIEW' if review.get('decision') in ['FAIL','HOLD'] else 'NO_NONREADY_REVIEW'
        prefix=source['turns'][:len(previous['turns'])]==previous['turns'];suffix=source['turns'][len(previous['turns']):]==draft['turns']
        cases.append({'root_id':draft['root_id'],'condition':cp.parents[1].name,'scene_ref':draft['scene_ref'],
            'previous_source_binding':bind(previous_path),'draft_binding':bind(dp),'cumulative_source_binding':bind(cp),'review_binding':bind(rp) if rp.exists() else None,
            'previous_binding_verified':draft['input_source_binding']==bind(previous_path),
            'actual_previous_turns':len(previous['turns']),'actual_appended_turns':len(draft['turns']),'actual_cumulative_turns':len(source['turns']),
            'exact_all_prior_turn_objects_preserved':prefix,'exact_draft_turn_objects_are_only_suffix':suffix,
            'exact_total_turn_count_matches_append':len(source['turns'])==len(previous['turns'])+len(draft['turns']),
            'wrapper_previous_turn_count':source.get('previous_turn_count'),'correct_new_stage_previous_turn_count':len(previous['turns']),
            'wrapper_previous_turn_count_stale':source.get('previous_turn_count')!=len(previous['turns']),
            'wrapper_count_inherited_unchanged_from_predecessor':source.get('previous_turn_count')==previous.get('previous_turn_count'),
            'wrapper_new_turn_count':source.get('new_turn_count'),'review_decision_unchanged':review.get('decision'),
            'original_source_or_gold_ambiguity_failure':review.get('source_or_gold_ambiguity_failure'),
            'original_failure_codes':codes,'failure_category':category,'accepted_receipt_present':cp.with_name('ACCEPTED.json').exists(),
            'admission_rescued':False,'semantic_or_metadata_bytes_changed':False})
    result={'schema_version':'oet1_exact_prefix_and_stale_wrapper_diagnostic_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'READ_ONLY_DIAGNOSTIC_COMPLETE','cases':cases,'failure_categories':dict(Counter(c['failure_category'] for c in cases)),
        'all_observed_prior_turn_objects_exactly_preserved':all(c['exact_all_prior_turn_objects_preserved'] for c in cases),
        'all_observed_suffixes_exactly_match_authored_draft':all(c['exact_draft_turn_objects_are_only_suffix'] for c in cases),
        'interpretation':'An exact preserved prefix can coexist with stale inherited wrapper counts. This diagnoses engineering metadata, does not change or retry the original independent FAIL/HOLD and does not admit any failed source.',
        'live_controller_driver_or_prompt_modified':False,'semantic_retries':0,'checker_binding':bind(Path(__file__))}
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':bind(p),'observed_scenes':len(cases),'failure_categories':result['failure_categories'],
        'exact_prefixes':result['all_observed_prior_turn_objects_exactly_preserved'],'admission_rescued':False},ensure_ascii=False))
if __name__=='__main__':main()
