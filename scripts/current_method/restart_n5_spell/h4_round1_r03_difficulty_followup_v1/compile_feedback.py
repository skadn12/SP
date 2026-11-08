"""Compile measured behaviour into the existing fixed-slot miner input."""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from .run import RUNS, OUTPUT, MODELS
from ..h4_round1_r03_feedback_batch_generation_v1.run import read, save, sha, content_key
from ..h4_round1_r03_feedback_batch_generation_v1.intent_prompt_revision1 import render

BASELINE_EVAL=RUNS/'h4_round1_r03_feedback_solver_eval_base_and_trained_v1_live_1'
GENERATION_RUNS=[RUNS/name for name in (
    'h4_round1_r03_feedback_batch_generation_revision1_live_2',
    'h4_round1_r03_feedback_batch_generation_slot_recovery_live_1',
    'h4_round1_r03_feedback_balanced_type_repair_v1_live_1')]


def measurement_id(row):
    # Follow-up summaries preserve scope/source IDs, while the old runner emitted
    # candidate_id directly. Never join on source ID alone across cohorts.
    return row.get('candidate_id') or (
        f"{row['root_id']}:R03_DIFFICULTY:{row['evaluation_scope']}:{row['source_candidate_id']}")


def compile_packets(out, followup=OUTPUT):
    if out.exists(): raise RuntimeError('output_exists')
    quality_path=followup/'quality_review.json'
    quality={r['candidate_id']:r for r in read(quality_path)['reviews']}
    cohorts=[]
    for label,run in [('baseline16',BASELINE_EVAL),('difficulty_followup',followup)]:
        candidates=[json.loads(line) for line in (run/'candidate_manifest.jsonl').read_text().splitlines()]
        measurements={m:{measurement_id(r):r for r in map(json.loads,(run/m/'summary.jsonl').read_text().splitlines())} for m in MODELS}
        for candidate in candidates:
            values={m:measurements[m][candidate['candidate_id']] for m in MODELS}
            cohorts.append((label,run,candidate,values))
    by_root=defaultdict(list)
    for value in cohorts: by_root[value[2]['root_id']].append(value)
    out.mkdir(parents=True)
    policy=(Path(__file__).parent/'difficulty_policy_v1.md').read_text()
    (out/'difficulty_policy_v1.md').write_text(policy)
    manifest={'version':'current_solver_feedback_v1','roots':{},'new_api_calls':0,
              'quality_review_path':str(quality_path),'quality_review_sha256':sha(quality_path),
              'diagnoses_generated':False,'type_targets':{'multi_select':2,'ordering':2}}
    for root,values in sorted(by_root.items()):
        source=next(run/root for run in GENERATION_RUNS if (run/root/'feedback_packet.json').exists())
        packet=copy.deepcopy(read(source/'feedback_packet.json'))
        dialogue=read(source/'dialogue_source.json')
        current=[]; excluded=[]; seen=set(); qa_history=list(packet['history'])
        # Sorting prioritizes observed no-context solvability; it adds no diagnosis.
        values.sort(key=lambda v:(max(x['no_context_accuracy'] for x in v[3].values()),
                                  max(x['full_context_accuracy'] for x in v[3].values())),reverse=True)
        for label,run,candidate,values_by_model in values:
            review=quality.get(candidate['candidate_id'])
            if label=='difficulty_followup' and review is None:
                raise ValueError('missing_followup_quality_review')
            if review and not review['eligible_solver_feedback']:
                excluded.append(review)
                continue
            if content_key(candidate) in seen: continue
            seen.add(content_key(candidate))
            obs={}
            for model,row in values_by_model.items():
                obs[model]={k:row[k] for k in ('full_context_accuracy','no_context_accuracy',
                    'full_context_correct_count','full_context_total_count','no_context_correct_count',
                    'no_context_total_count','canonical_flip_rate')}
                obs[model]['context_gain']=row['full_context_accuracy']-row['no_context_accuracy']
            cid=candidate.get('source_candidate_id',candidate.get('original_candidate_id',candidate['candidate_id']))
            current.append({'feedback_id':f'F{101+len(current):03}',
                'source_identity':{'root_id':root,'candidate_id':cid,'evaluation_candidate_id':candidate['candidate_id'],'cohort':label},
                'source_qa':{'question':candidate['question'],'question_type':candidate['question_type'],
                    'options_or_items':candidate['items'] if candidate['question_type']=='ordering' else candidate['options'],
                    'canonical_gold':candidate['gold']},
                'evidence_refs':candidate['evidence_refs'],
                'solver_observations':{'M0':obs['M0'],'post_training':{m:obs[m] for m in MODELS[1:]}},
                'observed_all_models_no_context_perfect':all(x['no_context_accuracy']==1 for x in values_by_model.values()),
                'independent_question_count':1,
                'source_provenance':[{'model':m,'path':str(run/m/'summary.jsonl'),'sha256':sha(run/m/'summary.jsonl')} for m in MODELS]})
        if not current:
            save(out/'excluded_roots'/root/'qa_quality_feedback.json',excluded)
            continue
        for run in GENERATION_RUNS:
            pool=run/root/'candidate_pool.json'
            if pool.exists(): qa_history.extend(read(pool))
        packet['history']=qa_history
        cited=Counter(ref for q in qa_history for ref in (q.get('evidence_refs') or []))
        refreshed=[]
        for gap in packet['coverage_gaps']:
            refs=[t['turn_ref'] for t in dialogue['turns']
                  if t['turn_ref'].rsplit(':',1)[0]==gap['scene_ref'] and not cited[t['turn_ref']]]
            if refs:
                refreshed.append({**gap,'uncited_turn_refs':refs,
                    'measurement':'no_citation_in_active_bank_or_all_current_generation_attempts'})
        packet['coverage_gaps']=refreshed
        packet['historical_feedback_archive']=packet['eligible_feedback']
        packet['eligible_feedback']=current
        packet['version']='current_solver_feedback_v1'
        packet['current_observation_count']=len(current)
        packet['qa_quality_feedback']=excluded
        packet['difficulty_policy_sha256']=sha(out/'difficulty_policy_v1.md')
        rd=out/root
        save(rd/'feedback_packet.json',packet); save(rd/'dialogue_source.json',dialogue)
        (rd/'intent_prompt.txt').write_text(policy+'\n\n'+render(dialogue,packet))
        manifest['roots'][root]={'current_observation_count':len(current),'packet_sha256':sha(rd/'feedback_packet.json'),
            'excluded_quality_observation_count':len(excluded),
            'dialogue_sha256':sha(rd/'dialogue_source.json'),'intent_prompt_sha256':sha(rd/'intent_prompt.txt')}
    save(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=RUNS/'h4_round1_r03_current_solver_feedback_v1')
    args=parser.parse_args(); compile_packets(args.output)
