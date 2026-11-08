"""Seal the completed Day 1 artifact set after audits finish."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import re

BASE=Path(__file__).resolve().parent
TARGET=BASE/'COMPLETE_MANIFEST.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not TARGET.exists(),'Never overwrite a sealed Day1 manifest'
    status=json.loads((BASE/'FINAL_STATUS.json').read_text())
    assert status['status']=='DAY1_COMPLETE' and all(status['goals'].values())
    for rel in ['VALIDATION.json','audit/INDEPENDENT_DAY1_AUDIT.json']:
        assert json.loads((BASE/rel).read_text())['status']=='PASS',rel
    assert status['legacy_review']['questions']==113
    assert sum(status['legacy_review'][k] for k in ['content_pass','hold','fail'])==113
    assert status['new_solver_calls']==status['new_training_runs']==0
    for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',(BASE/'REPORT_ZH.md').read_text()):
        if target.startswith('https://'):continue
        assert (BASE/re.sub(r':\d+$','',target)).exists(),target
    files=sorted(p for p in BASE.rglob('*') if p.is_file() and p!=TARGET and '__pycache__' not in p.parts and p.suffix!='.pyc')
    assert not any(p.is_symlink() for p in files)
    bindings=[{'path':str(p.relative_to(BASE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    now=datetime.now(timezone.utc)
    out={'schema_version':1,'status':'SEALED_DAY1_COMPLETE','sealed_at_utc':now.isoformat(),
         'sealed_at_beijing':now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
         'report':'REPORT_ZH.md','final_status':'FINAL_STATUS.json',
         'legacy_review':status['legacy_review'],'planned_future_question_slots':360,
         'actual_new_generated_questions':0,'actual_new_solver_calls':0,'actual_new_training_runs':0,
         'reference_models':'MODEL_BINDINGS.json; weights referenced with SHA256, not copied into this directory.',
         'audit_files':['VALIDATION.json','audit/INDEPENDENT_DAY1_AUDIT.json'],
         'binding_policy':{'path_base':'this directory','algorithm':'SHA256',
                           'excluded':['COMPLETE_MANIFEST.json','**/__pycache__/**','**/*.pyc']},
         'file_count':len(bindings),'total_bytes':sum(x['bytes'] for x in bindings),'files':bindings}
    with TARGET.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    loaded=json.loads(TARGET.read_text())
    for rec in loaded['files']:
        p=BASE/rec['path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],rec['path']
    assert len(files)==len(loaded['files'])
    print(json.dumps({'status':out['status'],'file_count':out['file_count'],'total_bytes':out['total_bytes'],
                      'manifest_sha256':sha(TARGET),'legacy_review':out['legacy_review']},ensure_ascii=False))


if __name__=='__main__':main()
