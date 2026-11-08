from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
b=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_screened_ab_validation_v117')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
for name in ['DELIVERY_REVIEW.json','INDEPENDENT_SOLVER_AUDIT.json','SOURCE_PROVENANCE_AUDIT.json','EVIDENCE_EXPORT_AUDIT.json','evidence/FINAL_REPORT_LANGUAGE_REVIEW.json']:
 d=read(b/name);assert d['status']=='PASS',(name,d)
 for row in d.get('input_bindings',d.get('checked_input_files',[])):assert sha(row['path'])==row['sha256'],row['path']
for manifest in ['evidence/HISTORICAL_PROTECTION.json','EXECUTION_INPUTS_FROZEN.json','evidence/TOOLING_FREEZE.json','evidence/ANALYSIS_TOOL_BINDINGS.json','evidence/DELIVERY_REVIEW_TOOL_BINDING.json']:
 for row in read(b/manifest)['files']:assert sha(row['path'])==row['sha256'],row['path']
assert read(b/'FINAL_STATUS.json')['training_started'] is False
assert not (b/'COMPLETE_MANIFEST.json').exists()
files=[p for p in sorted(b.rglob('*')) if p.is_file()]
save(b/'SEAL_RECEIPT.json',{'status':'SEALED_COMPLETED_VALIDATION_NO_TRAINING','sealed_at_utc':datetime.now(timezone.utc).isoformat(),'historical_files_verified_unchanged':9624,'nonmanifest_files_before_receipt':len(files),'independent_delivery_review':{'path':str(b/'DELIVERY_REVIEW.json'),'sha256':sha(b/'DELIVERY_REVIEW.json')},'final_report_review':{'path':str(b/'evidence/FINAL_REPORT_LANGUAGE_REVIEW.json'),'sha256':sha(b/'evidence/FINAL_REPORT_LANGUAGE_REVIEW.json')},'training_started':False,'pool_merge_performed':False,'manifest_excludes_only_itself':True})
files=[p for p in sorted(b.rglob('*')) if p.is_file()]
save(b/'COMPLETE_MANIFEST.json',{'status':'COMPLETE_READ_ONLY_DELIVERY','created_at_utc':datetime.now(timezone.utc).isoformat(),'files':[{'path':str(p),'sha256':sha(p)} for p in files],'excluded':['COMPLETE_MANIFEST.json'],'training_started':False,'pool_merge_performed':False})
for row in read(b/'COMPLETE_MANIFEST.json')['files']:assert sha(row['path'])==row['sha256']
for p in sorted(b.rglob('*')):
 if p.is_file():p.chmod(0o444)
for p in sorted(b.rglob('*'),reverse=True):
 if p.is_dir():p.chmod(0o555)
b.chmod(0o555)
print(json.dumps({'status':'SEALED','files_in_manifest':len(files),'manifest_sha256':sha(b/'COMPLETE_MANIFEST.json')}))
