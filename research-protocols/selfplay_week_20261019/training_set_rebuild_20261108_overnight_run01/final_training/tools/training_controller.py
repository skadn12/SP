#!/usr/bin/env python3
"""Isolated final-data preflight, longest-record smoke and formal launch gates.

No action is performed by import. Each CLI stage records real evidence. This
controller never starts the cancelled overnight source-diagnostic task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from checkpoint_health import validate_checkpoint
from training_contract import (FINAL, INIT, MODEL, OUTPUT, RECIPE, TRAINER,
    bind, final_manifest, now, read, sha, verify_binding, write_once)

TOOLS=Path(__file__).resolve().parent
PYTHON=Path('/root/miniconda3/bin/python')


def environment(run_dir, formal=False, project='gxy-selfplay', entity=None):
    run_dir=Path(run_dir)
    result=dict(os.environ)
    result.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(run_dir/'tmp'), XDG_CACHE_HOME=str(run_dir/'cache'),
        HF_HOME=str(run_dir/'cache/huggingface'), TRANSFORMERS_CACHE=str(run_dir/'cache/huggingface/transformers'),
        TORCH_HOME=str(run_dir/'cache/torch'), WANDB_DIR=str(run_dir/'wandb'),
        WANDB_CACHE_DIR=str(run_dir/'cache/wandb'), WANDB_DATA_DIR=str(run_dir/'wandb_data'),
        WANDB_CONFIG_DIR=str(run_dir/'wandb_config'), WANDB_MODE='online' if formal else 'disabled',
        WANDB_PROJECT=project, WANDB_SILENT='true', WANDB_CONSOLE='off')
    if entity:
        result['WANDB_ENTITY']=entity
    return result


def make_run_dirs(path):
    path=Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for child in ('tmp','cache','adapter','wandb','wandb_data','wandb_config'):
        (path/child).mkdir()


def resources():
    disk=shutil.disk_usage('/root/autodl-tmp')
    assert disk.free >= 8*(1<<30), 'INSUFFICIENT_DATA_DISK_FOR_ISOLATED_TRAINING'
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], capture_output=True,text=True,check=True).stdout.strip()
    assert not gpu, 'GPU_ALREADY_OCCUPIED: '+gpu
    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    assert memory and int(memory.splitlines()[0]) >= 80000, 'INSUFFICIENT_GPU_MEMORY'
    return dict(data_disk_free_bytes=disk.free, gpu_memory_free_mib=int(memory.splitlines()[0]), gpu_compute_processes=[])


def native_command(data, run_dir, smoke=False):
    recipe=dict(RECIPE)
    if smoke:
        recipe.update(epochs=1.,grad_accum=1,eval_steps=1,save_steps=100000)
    result=['--model',str(MODEL),'--tokenizer',str(MODEL),'--train-file',str(data),
            '--output-dir',str(Path(run_dir)/'adapter'),'--init-adapter-dir',str(INIT)]
    for key,value in recipe.items():
        result.extend(['--'+key.replace('_','-'),str(value)])
    if smoke:
        result.extend(['--max-train-samples','1'])
    return result


def preflight():
    manifest=final_manifest()
    output=FINAL/'tokenization_preflight.json'
    assert not output.exists(), 'PREFLIGHT_ALREADY_RECORDED'
    env=environment(FINAL/'cpu_preflight')
    (FINAL/'cpu_preflight'/'tmp').mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON),str(TOOLS/'tokenization_preflight.py'),'--data',str(FINAL/'FINAL_TRAIN.jsonl'),
                    '--model',str(MODEL),'--max-length','12288','--out',str(output)],env=env,check=True)
    report=read(output)
    assert report['failure_count']==0 and report['record_count']==manifest['record_count']
    assert report['data_sha256']==sha(FINAL/'FINAL_TRAIN.jsonl')
    assert all(r['overflow']==0 and r['prefix_match'] and r['gold_tokens']>0 for r in report['rows'])
    longest=max(report['rows'],key=lambda r:(r['raw_full_tokens'],-r['index']))
    lines=(FINAL/'FINAL_TRAIN.jsonl').read_bytes().splitlines(keepends=True)
    chosen=lines[longest['index']]
    assert json.loads(chosen)['candidate_id']==longest['candidate_id']
    sample=FINAL/'LONGEST_SMOKE_TRAIN.jsonl'
    with sample.open('xb') as stream: stream.write(chosen if chosen.endswith(b'\n') else chosen+b'\n')
    selection=dict(status='LONGEST_ACTUAL_FINAL_RECORD_SELECTED', created_at_utc=now(),
        final_manifest=bind(FINAL/'FINAL_MANIFEST.json'), preflight=bind(output),
        smoke_data=bind(sample), longest_record=longest,
        note='One actual longest final-training record; not the historical first-record smoke.')
    write_once(FINAL/'LONGEST_SMOKE_SELECTION.json',selection)
    return selection


def smoke():
    manifest=final_manifest()
    selection=read(FINAL/'LONGEST_SMOKE_SELECTION.json')
    for key in ('final_manifest','preflight','smoke_data'): verify_binding(selection[key])
    snapshot=resources()
    run_dir=OUTPUT.with_name(OUTPUT.name+'_longest_smoke')
    assert not run_dir.exists(), 'LONGEST_SMOKE_OUTPUT_ALREADY_EXISTS'
    assert not (FINAL/'LONGEST_SMOKE_RESULT.json').exists()
    make_run_dirs(run_dir)
    command=[str(PYTHON),str(TOOLS/'audited_training_runner.py'),'--run-dir',str(run_dir),'--phase','smoke',
        '--expected-backward','1','--expected-updates','1','--']+native_command(FINAL/'LONGEST_SMOKE_TRAIN.jsonl',run_dir,True)
    write_once(run_dir/'SMOKE_SETUP.json',dict(status='STARTED',created_at_utc=now(),command=command,
                 selected_sample=selection,resource_snapshot=snapshot,formal_training=False))
    with (run_dir/'train.log').open('x') as log:
        result=subprocess.run(command,env=environment(run_dir),stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0, 'LONGEST_SMOKE_FAILED: '+str(run_dir/'train.log')
    audit=read(run_dir/'gradient_audit.json')
    assert audit['status']=='PASS' and audit['backward_calls']==1 and audit['optimizer_steps']==1
    assert audit['all_losses_finite'] and audit['all_gradients_finite_nonzero']
    checkpoint=validate_checkpoint(run_dir/'adapter',1)
    result=dict(status='PASS',completed_at_utc=now(),final_manifest=bind(FINAL/'FINAL_MANIFEST.json'),
        smoke_selection=bind(FINAL/'LONGEST_SMOKE_SELECTION.json'),
        audit=bind(run_dir/'gradient_audit.json'), checkpoint=checkpoint,
        original_mclean_start=True, formal_training=False, smoke_run_dir=str(run_dir))
    write_once(FINAL/'LONGEST_SMOKE_RESULT.json',result)
    return result


def freeze_launch(validation_manifest,validation_callback,project,entity):
    manifest=final_manifest()
    assert validation_manifest.is_file() and validation_callback.is_file()
    val=read(validation_manifest)
    smoke_report=read(FINAL/'LONGEST_SMOKE_RESULT.json')
    assert smoke_report['status']=='PASS'
    for key in ('final_manifest','smoke_selection','audit'): verify_binding(smoke_report[key])
    assert not (FINAL/'LAUNCH_CONTRACT.json').exists(), 'LAUNCH_CONTRACT_ALREADY_FROZEN'
    scripts=[TOOLS/name for name in ('training_contract.py','build_final_training.py',
         'audited_training_runner.py','checkpoint_health.py','training_controller.py','tokenization_preflight.py')]
    # Preserve every callback and validation input binding supplied by the owner.
    deps=[]
    for candidate in val.get('bindings',[]):
        verify_binding(candidate); deps.append(candidate)
    contract=dict(status='FORMAL_LAUNCH_CONTRACT_FROZEN_NO_TRAINING',frozen_at_utc=now(),
        final_manifest=bind(FINAL/'FINAL_MANIFEST.json'),longest_smoke_result=bind(FINAL/'LONGEST_SMOKE_RESULT.json'),
        validation_manifest=bind(validation_manifest),validation_callback=bind(validation_callback),
        script_bindings=[bind(path) for path in scripts],validation_dependency_bindings=deps,
        recipe=RECIPE,expected_backward_calls=manifest['expected_backward_calls'],
        expected_optimizer_updates=manifest['expected_optimizer_updates'],scheduler_horizon=manifest['scheduler_horizon'],
        evaluation_steps=[0]+list(range(20,manifest['expected_optimizer_updates']+1,20))+
            ([] if manifest['expected_optimizer_updates']%20==0 else [manifest['expected_optimizer_updates']]),
        eval_every_optimizer_steps=20, wandb=dict(mode='online',project=project,entity=entity),
        tensorboard_enabled=False,overnight_cancelled=True,output_dir=str(OUTPUT),
        checkpoint_policy='native every 50 updates + native final; extra first-update audit snapshot only',
        callback_interface='evaluate(*,model,tokenizer,step,output_dir,validation_manifest) -> metrics/artifacts/raw_cot_saved/evaluated_records')
    write_once(FINAL/'LAUNCH_CONTRACT.json',contract)
    return contract


def check_launch_contract():
    manifest=final_manifest()
    contract=read(FINAL/'LAUNCH_CONTRACT.json')
    assert contract['status']=='FORMAL_LAUNCH_CONTRACT_FROZEN_NO_TRAINING' and contract['recipe']==RECIPE
    assert contract['output_dir']==str(OUTPUT) and contract['tensorboard_enabled'] is False and contract['overnight_cancelled'] is True
    for key in ('final_manifest','longest_smoke_result','validation_manifest','validation_callback'):
        verify_binding(contract[key])
    for key in ('script_bindings','validation_dependency_bindings'):
        for binding in contract[key]: verify_binding(binding)
    smoke_report=read(contract['longest_smoke_result']['path'])
    assert smoke_report['status']=='PASS' and smoke_report['final_manifest']['sha256']==sha(FINAL/'FINAL_MANIFEST.json')
    for key in ('audit','smoke_selection'): verify_binding(smoke_report[key])
    assert contract['expected_optimizer_updates']==manifest['expected_optimizer_updates']
    return contract


def launch():
    contract=check_launch_contract()
    assert not OUTPUT.exists(), 'FORMAL_OUTPUT_ALREADY_EXISTS'
    assert not (FINAL/'FORMAL_LAUNCH.json').exists(), 'FORMAL_ALREADY_LAUNCHED'
    snapshot=resources()
    import wandb
    assert bool(wandb.api.api_key), 'WANDB_ONLINE_AUTHENTICATION_REQUIRED'
    lock=FINAL/'FORMAL_TRAINING.lock'
    with lock.open('x') as stream:
        stream.write(json.dumps(dict(locked_at_utc=now(),controller_pid=os.getpid(),launch_contract=bind(FINAL/'LAUNCH_CONTRACT.json')))+'\n')
    make_run_dirs(OUTPUT)
    command=[str(PYTHON),str(TOOLS/'audited_training_runner.py'),'--run-dir',str(OUTPUT),'--phase','formal',
        '--expected-backward',str(contract['expected_backward_calls']),
        '--expected-updates',str(contract['expected_optimizer_updates']),
        '--validation-manifest',contract['validation_manifest']['path'],
        '--validation-callback',contract['validation_callback']['path'],
        '--final-manifest',str(FINAL/'FINAL_MANIFEST.json'),'--']+native_command(FINAL/'FINAL_TRAIN.jsonl',OUTPUT)
    with (OUTPUT/'train.log').open('x') as log:
        proc=subprocess.Popen(command,env=environment(OUTPUT,True,contract['wandb']['project'],contract['wandb']['entity']),
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(status='LAUNCHED_NOT_YET_HEALTHY',launched_at_utc=now(),pid=proc.pid,
        launch_contract=bind(FINAL/'LAUNCH_CONTRACT.json'),command=command,resource_snapshot=snapshot,
        output_dir=str(OUTPUT),wandb_mode='online',tensorboard_enabled=False,overnight_cancelled=True)
    write_once(FINAL/'FORMAL_LAUNCH.json',record)
    write_once(OUTPUT/'LAUNCH.json',record)
    return record


def status():
    result=dict(checked_at_utc=now(),formal_output_exists=OUTPUT.exists(),overnight_cancelled=True)
    for name in ('LAUNCH.json','WANDB_RUN.json','TRAINING_HEALTHY.json','TRAINING_COMPLETED.json','gradient_audit.json'):
        if (OUTPUT/name).exists():
            data=read(OUTPUT/name)
            if name=='gradient_audit.json':
                data={k:data[k] for k in ('status','backward_calls','optimizer_steps','expected_backward_calls','expected_optimizer_steps','all_losses_finite','all_gradients_finite_nonzero','wandb_online')}
            result[name]=data
    if 'LAUNCH.json' in result:
        pid=result['LAUNCH.json']['pid']
        result['pid_exists']=Path(f'/proc/{pid}').exists()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','smoke','freeze-launch','launch','status'))
    parser.add_argument('--validation-manifest',type=Path)
    parser.add_argument('--validation-callback',type=Path)
    parser.add_argument('--wandb-project',default='gxy-selfplay')
    parser.add_argument('--wandb-entity',default='1946816004-tianjin-university')
    args=parser.parse_args()
    if args.mode=='freeze-launch':
        assert args.validation_manifest and args.validation_callback
        result=freeze_launch(args.validation_manifest,args.validation_callback,args.wandb_project,args.wandb_entity)
    else:
        result={'preflight':preflight,'smoke':smoke,'launch':launch,'status':status}[args.mode]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
