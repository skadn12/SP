#!/usr/bin/env python3
"""Post-raw-freeze, exhaustive F FULL review packets; no score-based selection."""
import sys
sys.dont_write_bytecode=True
from assemble_review import HERE,read,save,binding,fingerprint

def main():
    raw_freeze=read(HERE/'execution/RAW_RESPONSE_FREEZE.json')
    scores=read(HERE/'execution/SCORES.json')
    assert scores['raw_freeze_binding']==binding(HERE/'execution/RAW_RESPONSE_FREEZE.json')
    candidates={c['candidate_id']:c for c in read(HERE/'CANDIDATES_PRIVATE.json') if c['arm']=='F'}
    plans={c['candidate_id']:c for c in read(HERE/'question_authoring/F/TASK_PLANS_FIXED.json')}
    public={r['request_id']:r for r in read(HERE/'execution/PUBLIC_REQUESTS.json')['requests']}
    private={r['request_id']:r for r in read(HERE/'execution/PRIVATE_SCORING.json')['requests']}
    records=[r for r in scores['records'] if r['arm']=='F' and r['condition']=='FULL']
    assert len(records)==40
    for group,tails in [(1,['218519','222937']),(2,['225575','229686'])]:
        subset=[r for r in records if r['root_id'].split('-')[-1] in tails]
        selected={r['candidate_id'] for r in subset}
        packet={'review_scope':'All F FULL responses for assigned roots, including correct answers and MC6. No selection by score.',
            'rules_binding':binding(HERE/'ERROR_MATCHING_RULES_ZH.md'),
            'content_freeze_binding':binding(HERE/'CONTENT_FREEZE.json'),
            'raw_freeze_binding':binding(HERE/'execution/RAW_RESPONSE_FREEZE.json'),
            'score_binding':binding(HERE/'execution/SCORES.json'),
            'source_bindings':[candidates[cid]['source_binding'] for cid in sorted(selected) if candidates[cid]['slot_id']=='MC2'],
            'fixed_plans':[plans[cid] for cid in sorted(selected)],
            'candidates':[candidates[cid] for cid in sorted(selected)],
            'responses':[{'score':r,'public_request':public[r['request_id']],
                'presentation':private[r['request_id']]['presentation']} for r in subset]}
        assert len(subset)==20 and len(selected)==10
        save(HERE/f'error_review/group{group}/REVIEW_INPUT.json',packet)
    print('Prepared two exhaustive F FULL error-review groups:20 responses each.')

if __name__=='__main__':main()
