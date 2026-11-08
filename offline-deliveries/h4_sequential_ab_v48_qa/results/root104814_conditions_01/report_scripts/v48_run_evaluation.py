from pathlib import Path
import subprocess,json
b=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v48_qa');py='/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
run='runs/root104814_conditions_01_evaluation';gen='runs/root104814_conditions_01_generation'
for action in ['preflight','live','finalize']:
 args=[py,'-u','evaluate_and_pool.py',action,'--run',run]
 if action=='preflight':args+=['--generation',gen]
 with open('/tmp/v48_'+action+'.log','w') as log:
  subprocess.run(args,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(json.dumps({'stage_completed':action}),flush=True)
report=json.loads((b/run/'training_pool/REPORT.json').read_text())
print(json.dumps({'selected':report['new_unique_added'],'solver_full_calls':report['full_solver_calls']}),flush=True)
