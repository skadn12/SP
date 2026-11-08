"""Prepare completed QA-root review cases early, using the already frozen rule."""
import argparse
from pathlib import Path
from .analysis import qa_roots, TYPES, selected_payload, blinded_case
from .freeze import DEFAULT,ARMS,load,write,sha_obj
from ..core import canonical_json

def prepare_completed(run):
    policy=load(run/'audit/statistics_and_review_policy.json')['review'];complete=[];ids=set()
    for arm in ARMS:
        for root,rd in qa_roots(run,arm):
            if not (rd/'qa_completion_checkpoint.json').is_file():continue
            candidates=load(rd/'candidates_valid.json');transcript=load(rd/'expanded_dialogue_clean.json')
            assert transcript==load(run/'inputs/shared'/root/'expanded_dialogue_clean.json')
            history=load(run/'inputs/feedback_on'/root/'feedback_packet.json')['historical_qa']
            for qtype in TYPES:
                group=[c for c in candidates if c['question_type']==qtype]
                for c in sorted(group,key=lambda x:(sha_obj(selected_payload(x)),canonical_json(selected_payload(x))))[:2]:
                    case=blinded_case(c,history,transcript,policy['checks']);bid=case['blind_id']
                    write(run/'diagnostics/blind_review/cases'/(bid+'.json'),case);ids.add(bid)
            complete.append({'root_id':root,'arm':arm})
    print({'completed_root_arms':len(complete),'blind_cases_available':len(ids)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT);a=p.parse_args();prepare_completed(a.run)
