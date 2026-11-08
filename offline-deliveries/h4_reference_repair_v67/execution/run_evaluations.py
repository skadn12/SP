from pathlib import Path
import subprocess,json,datetime,hashlib
R=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reference_repair_v67')
PY=Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
def write(name, data):
    with (R/'execution'/name).open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
write('evaluation_sequence.STARTED.json',{'time':now(),'training_allowed':False,'automatic_retry':False})
for stage in ('prepare','live-trained','live-base','verify'):
    with (R/'execution'/('evaluation_'+stage+'.log')).open('x') as log:
        result=subprocess.run([str(PY),'-B','-u',str(R/'evaluate.py'),stage],cwd=R,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'stage':stage,'returncode':result.returncode,'time':now()}),flush=True)
    if result.returncode:
        write('evaluation_sequence.ERROR.json',{'stage':stage,'returncode':result.returncode,'time':now(),'automatic_retry':False})
        raise SystemExit(result.returncode)
write('evaluation_sequence.COMPLETE.json',{'time':now(),'stages':['prepare','live-trained','live-base','verify'],'training_allowed':False})
