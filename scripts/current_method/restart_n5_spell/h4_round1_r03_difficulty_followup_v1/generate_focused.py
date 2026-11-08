"""Consume current measured feedback while preserving the first-round QA route."""
import argparse
from dataclasses import replace
from pathlib import Path

from .run import RUNS
from ..h4_round1_r03_feedback_batch_generation_v1 import API_CONFIG, MODEL, TEMPERATURE
from ..h4_round1_r03_feedback_batch_generation_v1.run import (prepare, read, save, sha,
    verify_frozen, generate_root, load_challenger_config, OpenAICompatibleChallengerClient)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feedback-run',type=Path,default=RUNS/'h4_round1_r03_current_solver_feedback_v1')
    parser.add_argument('--output',type=Path,default=RUNS/'h4_round1_r03_feedback_mixed_generation_aligned_v1_live_1')
    parser.add_argument('--type-focus-only',action='store_true',help='legacy divergent repair-only route; omitted for first-round parity')
    parser.add_argument('--live',action='store_true'); args=parser.parse_args()
    feedback=read(args.feedback_run/'manifest.json')
    roots=list(feedback['roots'])
    manifest=prepare(args.output,roots)
    for root,source in feedback['roots'].items():
        rd=args.output/root; supplied=args.feedback_run/root
        for filename,key in [('feedback_packet.json','packet_sha256'),('dialogue_source.json','dialogue_sha256'),('intent_prompt.txt','intent_prompt_sha256')]:
            if sha(supplied/filename)!=source[key]: raise ValueError('feedback_source_drift')
        if read(rd/'dialogue_source.json')['turns']!=read(supplied/'dialogue_source.json')['turns']:
            raise ValueError('current_feedback_dialogue_drift')
        save(rd/'feedback_packet.json',read(supplied/'feedback_packet.json'))
        (rd/'intent_prompt.txt').write_bytes((supplied/'intent_prompt.txt').read_bytes())
        entry=manifest['inputs'][root]
        entry['packet_sha256']=sha(rd/'feedback_packet.json')
        entry['intent_prompt_sha256']=sha(rd/'intent_prompt.txt')
        entry['feedback_count']=len(read(rd/'feedback_packet.json')['eligible_feedback'])
        entry['historical_question_count']=len(read(rd/'feedback_packet.json')['history'])
    policy=args.feedback_run/'difficulty_policy_v1.md'
    (args.output/'difficulty_policy_v1.md').write_bytes(policy.read_bytes())
    manifest.update({'current_feedback_manifest':str(args.feedback_run/'manifest.json'),
        'current_feedback_manifest_sha256':sha(args.feedback_run/'manifest.json'),
        'difficulty_policy_sha256':sha(policy),'type_focus_only':bool(args.type_focus_only),
        'generation_route':'first_round_mixed_primary_with_feedback' if not args.type_focus_only else 'legacy_type_focus_repair_only',
        'reused_intents':False})
    if args.type_focus_only:
        manifest['configuration']['sources']=['contract_guided']
        manifest['configuration']['target_counts']={'per_root':{'multi_select':2,'ordering':2}}
    save(args.output/'manifest.json',manifest)
    save(args.output/'configuration_comparison.json',manifest['configuration'])
    verify_frozen(args.output,manifest)
    if args.live:
        config=load_challenger_config(API_CONFIG)
        if config.model!=MODEL: raise ValueError('model_drift')
        client=OpenAICompatibleChallengerClient(replace(config,temperature=TEMPERATURE,
            enable_thinking=False,timeout_seconds=max(180,config.timeout_seconds)))
        reports=[]
        for root in roots:
            reports.append(generate_root(args.output,root,manifest,client,
                balanced_type_minimums=({'multi_select':2,'ordering':2} if args.type_focus_only else None),
                type_focus_only=args.type_focus_only))
            save(args.output/'progress.json',{'roots':reports})
        verify_frozen(args.output,manifest)
        save(args.output/'final_report.json',{'roots':reports,'current_feedback_consumed':True,
            'candidate_count':sum(r['candidate_count'] for r in reports),
            'structurally_accepted_count':sum(r['structurally_accepted_count'] for r in reports),
            'difficulty_status':'UNMEASURED','training_calls':0})

if __name__=='__main__': main()
