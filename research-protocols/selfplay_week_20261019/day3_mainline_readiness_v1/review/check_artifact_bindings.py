#!/usr/bin/env python3
"""Recursively verify explicit SHA256 file bindings in final stream artifacts."""
from pathlib import Path
import json,hashlib,datetime
RUN=Path(__file__).resolve().parents[1]
def main():
    paths=[RUN/'READINESS_DECISION.json']
    paths+=sorted((RUN/'common_eval').glob('*.json'))
    paths+=sorted((RUN/'training').glob('*.json'))
    results=[]
    def walk(obj,owner):
        if isinstance(obj,dict):
            if isinstance(obj.get('path'),str) and isinstance(obj.get('sha256'),str):
                p=Path(obj['path'])
                if not p.is_absolute():p=owner.parent/p
                resolution=None
                if p==RUN.parent/'CURRENT_STATUS.json':
                    p=RUN/'navigation_history/CURRENT_STATUS.json';resolution='Immutable execution-start navigation snapshot.'
                actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
                results.append(dict(owner=str(owner),path=obj['path'],resolved_path=str(p),resolution=resolution,passed=actual==obj['sha256'] and ('bytes' not in obj or p.stat().st_size==obj['bytes']),expected=obj['sha256'],actual=actual))
            for v in obj.values():walk(v,owner)
        elif isinstance(obj,list):
            for v in obj:walk(v,owner)
    for p in paths:walk(json.loads(p.read_text()),p)
    out=dict(schema_version='day3_final_artifact_binding_check_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),binding_occurrences=len(results),passed=sum(r['passed'] for r in results),all_pass=all(r['passed'] for r in results),failures=[r for r in results if not r['passed']],results=results)
    (RUN/'review/FINAL_ARTIFACT_BINDING_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'},ensure_ascii=False))
    raise SystemExit(0 if out['all_pass'] else 1)
if __name__=='__main__':main()
