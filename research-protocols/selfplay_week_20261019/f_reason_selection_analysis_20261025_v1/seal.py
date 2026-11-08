import json,hashlib
from pathlib import Path
A=Path(__file__).parent
audit=json.loads((A/'FINAL_REPORT_REVIEW.json').read_text())
assert audit['status']=='PASS'
assert json.loads((A/'VALIDATION.json').read_text())['status']=='PASS'
files=[]
for p in sorted(A.rglob('*')):
 if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_MANIFEST.sha256'] and '__pycache__' not in p.parts:
  files.append(dict(path=str(p.relative_to(A)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
m=dict(schema_version='f_reason_selection_manifest_v1',status='SEALED_POSTHOC_ANALYSIS',analysis_date='2026-10-25',new_solver_calls=0,training=False,original_experiment_modified=False,files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files),excluded=['FINAL_MANIFEST.json','FINAL_MANIFEST.sha256','__pycache__'])
p=A/'FINAL_MANIFEST.json';p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');digest=hashlib.sha256(p.read_bytes()).hexdigest();(A/'FINAL_MANIFEST.sha256').write_text(digest+'  FINAL_MANIFEST.json\n')
for f in files:assert hashlib.sha256((A/f['path']).read_bytes()).hexdigest()==f['sha256']
print(json.dumps(dict(status=m['status'],files=len(files),sha256=digest),ensure_ascii=False))
