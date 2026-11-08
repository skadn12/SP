"""Scoped QPR1 wrapper around the immutable, reviewed FBC1 B1 runtime.

Same endpoint, source, labels, choices, schemas and parent sampling seeds.
Question/operation presentation is repaired before export, not at decoding.
"""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PARENT=BASE.parent/'postweek_feedback_control_4root_v1'
spec=importlib.util.spec_from_file_location('qpr1_frozen_fbc1_runtime',PARENT/'solver_runtime/run.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
read,save,bind=runtime.read,runtime.save,runtime.bind
original_code_paths=runtime.code_paths
runtime.HERE=HERE
runtime.EXPERIMENT=BASE

def sampling_seed(question,branch):
    return read(BASE/'inputs/PUBLIC_SAMPLING_SEEDS.json')['seeds'][question['candidate_id']][branch]

def code_paths():
    return original_code_paths()+[Path(__file__),BASE/'inputs/PUBLIC_SAMPLING_SEEDS.json',
        BASE.parents[2]/'scripts/current_method/question_presentation_gate_v2.py']

runtime.sampling_seed=sampling_seed
runtime.code_paths=code_paths

def prepare(args):
    runtime.prepare(args)
    rows=read(args.output/'PUBLIC_REQUESTS.json')['requests']
    flags=[]
    for row in rows:
        prompt=Path(row['prompt_path']).read_text()
        assert row['candidate_id'] not in prompt and row['request_id'] not in prompt
        assert 'MC_RECORD_AUDIT_CONTROL' not in prompt
        assert 'audit control' not in prompt.lower() and 'all-correct' not in prompt.lower()
        assert row['task_operation']!='MC_RECORD_AUDIT_CONTROL'
        flags.append({'request_id':row['request_id'],'opaque_ids_absent':True,'control_labels_absent':True,
            'prompt_binding':bind(row['prompt_path'])})
    save(args.output/'PUBLIC_PRESENTATION_CHECK.json',{'status':'PASS_RENDERED_PROMPTS_ONLY','requests':len(rows),
        'private_slot_or_gold_not_read':True,'checks':flags,'runtime_binding':bind(__file__)})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','verify','run','analyze'])
    ap.add_argument('--export-freeze',type=Path,default=BASE/'inputs/EXPORT_FREEZE.json')
    ap.add_argument('--source-root',type=Path,default=PARENT/'sources')
    ap.add_argument('--output',type=Path,default=HERE/'execution')
    ap.add_argument('--endpoint',type=Path,default=runtime.ENDPOINT)
    ap.add_argument('--preflight-review',type=Path,default=HERE/'PREFLIGHT_REVIEW.json')
    ap.add_argument('--execute-gpu',action='store_true')
    args=ap.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='verify':
        p,rows,t=runtime.verify(args);print({'status':'PUBLIC_INPUTS_VERIFIED','requests':len(rows)})
    else:getattr(runtime,args.action)(args)
