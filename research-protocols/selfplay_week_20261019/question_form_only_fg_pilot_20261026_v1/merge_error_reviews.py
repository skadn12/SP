#!/usr/bin/env python3
"""Merge exhaustive, post-freeze semantic-error reviews with checked score IDs."""
import collections
import sys
sys.dont_write_bytecode=True
from assemble_review import HERE,read,save,binding

CLASSES={'OBSERVED_EXPLANATION_MATCH','OBSERVED_SELECTION_MATCH','ERROR_OTHER_RELATION',
         'NO_TARGET_ERROR_OBSERVED','INDETERMINATE','INVALID_RESPONSE'}
def main():
    scores={r['request_id']:r for r in read(HERE/'execution/SCORES.json')['records']
            if r['arm']=='F' and r['condition']=='FULL'}
    candidates={c['candidate_id']:c for c in read(HERE/'CANDIDATES_PRIVATE.json')}
    records=[];inputs=[]
    for group in (1,2):
        p=HERE/f'error_review/group{group}/FEEDBACK_ERROR_MATCHING_AI.json'
        packet_path=p.parent/'REVIEW_INPUT.json';packet=read(packet_path)
        expected={r['score']['request_id'] for r in packet['responses']}
        doc=read(p)
        assert len(doc['records'])==len(expected)==20
        assert {r['request_id'] for r in doc['records']}==expected
        assert doc['reviewer_id']
        for r in doc['records']:
            s=scores[r['request_id']];c=candidates[s['candidate_id']]
            for k in ('candidate_id','root_id','slot_id','variant_id','valid','exact','J',
                      'canonical_prediction','canonical_gold'):
                assert r[k]==s[k],(r['request_id'],k)
            assert r['predicted_observable_error']==c['feedback_target']['predicted_observable_error']
            assert r['match_class'] in CLASSES and r['rationale'].strip()
            assert 'observed_final_evidence' in r
            records.append(r|{'reviewer_id':doc['reviewer_id'],'review_binding':binding(p)})
        inputs += [binding(p),binding(packet_path)]
    assert len(records)==len({r['request_id'] for r in records})==len(scores)==40
    records.sort(key=lambda r:(r['root_id'],r['slot_id'],r['variant_id']))
    summary={}
    for scope in ('MC2_5_PRIMARY','MC6_CONTROL','ALL'):
        rows=[r for r in records if scope=='ALL' or ((r['slot_id']=='MC6')==(scope=='MC6_CONTROL'))]
        summary[scope]={'responses':len(rows),'questions':len({r['candidate_id'] for r in rows}),
                        'classes':dict(collections.Counter(r['match_class'] for r in rows))}
    groups=[]
    for dim in ('root_id','slot_id','candidate_id'):
        for val in sorted({r[dim] for r in records}):
            rows=[r for r in records if r[dim]==val]
            groups.append({'dimension':dim,'value':val,'responses':len(rows),
                 'classes':dict(collections.Counter(r['match_class'] for r in rows))})
    save(HERE/'FEEDBACK_ERROR_MATCHING_AI.json',{'reviewer_id':'root aggregation of two separate AI final-output reviews',
        'reviewer_is_human':False,'rules_binding':binding(HERE/'ERROR_MATCHING_RULES_ZH.md'),
        'content_freeze_binding':binding(HERE/'CONTENT_FREEZE.json'),
        'raw_freeze_binding':binding(HERE/'execution/RAW_RESPONSE_FREEZE.json'),
        'inputs':inputs,'records':records,'summary':summary,'strata':groups,
        'limits':'Observable final explanations and selections only; not hidden thinking, human adjudication, training or generalization evidence.'})
    print(summary)

if __name__=='__main__':main()
