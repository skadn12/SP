import hashlib, json
from pathlib import Path
A=Path(__file__).parent
review=json.loads((A/'FINAL_REPORT_REVIEW.json').read_text())
assert review['status']=='PASS', 'Finish the final report review before sealing.'
validation=json.loads((A/'VALIDATION.json').read_text())
summary=json.loads((A/'SUMMARY.json').read_text())
assert validation['status']=='PASS'
assert summary['overall']['responses']==64 and summary['overall']['questions']==32
files=[]
for p in sorted(A.rglob('*')):
    if not p.is_file() or p.name in ('FINAL_MANIFEST.json','FINAL_MANIFEST.sha256') or '__pycache__' in p.parts:continue
    files.append(dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(schema_version='g_solver_error_analysis_manifest_v1',status='SEALED_POSTHOC_ANALYSIS_NO_NEW_SOLVER_OR_TRAINING',analysis_date='2026-10-25',base_directory=str(A),new_solver_calls=0,training=False,original_experiment_modified=False,original_experiment_manifest_sha256=summary['frozen_experiment_unchanged']['manifest_sha256'],files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files),excluded=['FINAL_MANIFEST.json','FINAL_MANIFEST.sha256','__pycache__'])
target=A/'FINAL_MANIFEST.json';target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
digest=hashlib.sha256(target.read_bytes()).hexdigest()
(A/'FINAL_MANIFEST.sha256').write_text(digest+'  FINAL_MANIFEST.json\n')
for f in files:assert hashlib.sha256((A/f['path']).read_bytes()).hexdigest()==f['sha256']
print(json.dumps(dict(status=manifest['status'],files=len(files),sha256=digest,validation_checks=validation['check_count'],questions=summary['overall']['questions'],responses=summary['overall']['responses']),ensure_ascii=False))
