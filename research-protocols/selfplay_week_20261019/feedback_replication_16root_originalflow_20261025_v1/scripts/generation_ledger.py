"""Record actual frozen production states and preserved draft receipts."""
import collections
import json
from pathlib import Path
from workflow_io import BASE, bind, load, now, save


def main():
    freeze = load(BASE/'QUESTION_FREEZE.json')
    final = {(r['root_id'],r['arm'],r['slot_id']):r for r in freeze['slots']}
    drafts = []
    for path in sorted((BASE/'questions').glob('*/*/CANDIDATES_DRAFT.json')):
        obj = load(path)
        candidates = obj if isinstance(obj,list) else obj['candidates']
        root, arm = path.parent.parent.name, path.parent.name
        review = BASE/'reviews'/f'new_{root.removeprefix("cc-train-episode-")}_{arm}'/'ADMISSION.json'
        outcomes = load(review)['slots'] if review.exists() else []
        for c in candidates:
            initial = [r for r in outcomes if r['candidate']==c]
            assert len(initial)<=1
            k=(root,arm,c['slot_id'])
            drafts.append({'root_id':root,'arm':arm,'slot_id':c['slot_id'],
                'candidate_id':c['candidate_id'],'draft_binding':bind(path),
                'initial_independent_review':initial[0]['status'] if initial else 'NOT_ADMITTED_OR_WITHDRAWN',
                'review_binding':bind(review) if review.exists() else None,
                'final_slot_status':final[k]['status']})
    all_attempt_artifacts = [bind(p) for p in sorted((BASE/'questions').glob('*/*/*.json'))]
    summary={a:{'planned_slots':128,
        'current_continuation_first_drafts_submitted':sum(r['arm']==a for r in drafts),
        'current_first_draft_independently_passed':sum(r['arm']==a and r['initial_independent_review']=='PASS' for r in drafts),
        'final_statuses':dict(collections.Counter(r['status'] for r in final.values() if r['arm']==a))}
        for a in ('F','G')}
    save(BASE/'GENERATION_LEDGER.json',{'at':now(),'adaptive_recovery':True,
        'first_draft_scope':'First drafts authored in the current continuation only; excludes reused successful slots and targeted recovery of already existing failed artifacts. Not an original-budget yield claim.',
        'summary':summary,'draft_events':drafts,
        'planned_slots':[{k:r[k] for k in ('root_id','arm','slot_id','status')} for r in final.values()],
        'all_current_author_attempt_artifacts':all_attempt_artifacts,
        'question_freeze_binding':bind(BASE/'QUESTION_FREEZE.json'),
        'prior_attempt_archives':[str(BASE.parent/n) for n in ('feedback_replication_20root_20261025_v1','feedback_replication_20root_resume_20261025_v1')],
        'prior_attempt_archives_immutable':True})
    print(json.dumps(summary))


if __name__=='__main__':
    main()
