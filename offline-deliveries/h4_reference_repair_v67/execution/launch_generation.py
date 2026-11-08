from pathlib import Path
import json,hashlib,subprocess,datetime
R=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reference_repair_v67')
assert (R/'PREPARATION.COMPLETE.json').exists()
E=R/'execution';E.mkdir(exist_ok=True)
launch=E/'generation.LAUNCH.json';log=E/'generation.log'
assert not launch.exists() and not log.exists()
with log.open('x') as stream:
    child=subprocess.Popen(['/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python','-B','-u',str(R/'generation_driver.py'),'generate'],cwd=R,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':child.pid,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256((R/'generation_driver.py').read_bytes()).hexdigest(),'log':str(log),'automatic_retry':False,'training_allowed':False}
with launch.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
