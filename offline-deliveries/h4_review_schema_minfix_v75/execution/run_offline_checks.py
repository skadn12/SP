from pathlib import Path
import subprocess,datetime,hashlib,json
R=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (R/'execution/offline_tests.log').open('x') as log:
    result=subprocess.run([PY,'-B','-m','unittest','test_asset_generation','test_compose','test_evaluate','test_report','test_review_schema','test_review_lifecycle','-q'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
value={'status':'PASS' if result.returncode==0 else 'FAIL','started_at_utc':start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':result.returncode,'network_calls':0,'command':[PY,'-B','-m','unittest','test_asset_generation','test_compose','test_evaluate','test_report','test_review_schema','test_review_lifecycle','-q'],'log':{'path':str(R/'execution/offline_tests.log'),'sha256':hashlib.sha256((R/'execution/offline_tests.log').read_bytes()).hexdigest()}}
with (R/'execution/OFFLINE_TEST_RECEIPT.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value));raise SystemExit(result.returncode)
