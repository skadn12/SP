#!/usr/bin/env python3
"""Seal this finite result after navigation publication and checks."""
from pathlib import Path
import json,hashlib,datetime
RUN=Path(__file__).resolve().parents[1]
def main():
    decision=json.loads((RUN/'READINESS_DECISION.json').read_text())
    assert not decision['training_start_go']
    for rel in ['review/TECHNICAL_CLOSEOUT_VERIFICATION.json','review/INDEPENDENT_CLOSEOUT_AUDIT.json']:
        assert json.loads((RUN/rel).read_text())['all_pass'] is True
    files=[]
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and p.name!='FINAL_MANIFEST.json':
            data=p.read_bytes();files.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    manifest=dict(schema_version='day3_mainline_readiness_final_manifest_v1',sealed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=decision['status'],files=files,new_solver_calls=0,new_model_inference=0,new_questions=0,new_dialogue_expansions=0,new_training_runs=0,training_start_go=False,full_day_complete=False,do_not_mutate_after_seal=True)
    target=RUN/'FINAL_MANIFEST.json';target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    check=json.loads(target.read_text())
    assert all(Path(r['path']).stat().st_size==r['bytes'] and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in check['files'])
    print(json.dumps(dict(sealed=True,files=len(files),manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),all_file_hashes_pass=True),ensure_ascii=False))
if __name__=='__main__':main()
