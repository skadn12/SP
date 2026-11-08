"""Seal or verify this completion directory without editing prior experiments."""
from pathlib import Path
import argparse,datetime,hashlib,json
W=Path(__file__).resolve().parent
MAN=W/'COMPLETE_MANIFEST.json'
def entries():
    return [{'path':str(p.relative_to(W)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        for p in sorted(W.rglob('*')) if p.is_file() and p!=MAN and '__pycache__' not in p.parts and p.suffix!='.pyc']
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['seal','verify']);a=parser.parse_args()
    if a.mode=='seal':
        assert not MAN.exists()
        status=json.loads((W/'FINAL_STATUS.json').read_text());assert status['day2_all_goals_complete'] is True
        audit=json.loads((W/'audit/RESULTS_AUDIT.json').read_text());assert audit['status']=='PASS'
        manifest={'schema_version':'day2_completion_seal_v1','status':'SEALED',
            'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'file_hash_algorithm':'SHA256','scope':'This completion directory only; old sealed directories untouched.',
            'excluded':['COMPLETE_MANIFEST.json (self-reference)','__pycache__ and .pyc (non-evidence interpreter caches)'],
            'files':entries()}
        with MAN.open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    manifest=json.loads(MAN.read_text());assert entries()==manifest['files'],'SEALED_FILESET_OR_BYTES_CHANGED'
    print(json.dumps({'status':'PASS','sealed_files':len(manifest['files']),'sealed_bytes':sum(x['bytes'] for x in manifest['files'])}))
if __name__=='__main__':main()
