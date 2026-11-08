#!/usr/bin/env python3
"""Read-only, schema-agnostic audit for candidate source exposure.
Writes only into the caller-selected output directory.
"""
import argparse, hashlib, json, os, re, sys, urllib.request
from pathlib import Path

CANDIDATES = ["9386","51072","53623","98366","25286","85772","215656","133418","192156","106092"]
ALIAS_PATTERNS = {
    "root_token": lambda n: re.compile(rf"cc[-_]train[-_]episode[-_ ]?{re.escape(n)}(?!\d)", re.I),
    "episode_alias": lambda n: re.compile(rf"(?<!\d)episode[-_ ]?{re.escape(n)}(?!\d)", re.I),
    "data_id_alias": lambda n: re.compile(rf"(?<!\d)dataID[\"'=: ]+[-_ ]?{re.escape(n)}(?!\d)", re.I),
}
# Historical artifact scopes used by the prior exact-token scan. The source file and current
# admission/audit namespaces are intentionally excluded from exposure evidence.
DEFAULT_ROOTS = [
 "/root/autodl-tmp/gxy-selfplay/research-protocols",
 "/root/autodl-tmp/gxy-selfplay/offline-deliveries",
 "/root/autodl-tmp/gxy-selfplay/SP-820-publish",
 "/root/autodl-tmp/gxy-selfplay/restart-n5-runs",
 "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs",
 "/root/autodl-tmp/gxy-selfplay/restart-n5-derived",
 "/root/autodl-tmp/gxy-selfplay/restart-n5-exports",
 "/root/autodl-tmp/gxy-selfplay/restart-n5-freezes",
 "/root/autodl-tmp/gxy-selfplay/question_authoring",
]

def sha256_file(p, cap=64*1024*1024):
    h=hashlib.sha256(); total=0
    with open(p,'rb') as f:
        while total < cap:
            b=f.read(min(1024*1024,cap-total))
            if not b: break
            h.update(b); total+=len(b)
    return h.hexdigest(), total

def safe_json_metadata(p):
    try:
        if p.stat().st_size > 20*1024*1024: return None
        with open(p,'r',encoding='utf-8',errors='ignore') as f: obj=json.load(f)
        if isinstance(obj,dict):
            keys=sorted(obj.keys())[:80]
            vals={}
            for k,v in obj.items():
                if any(x in k.lower() for x in ('root','episode','dataid','parent','lineage','source','reservation','question','solver','author','family','sha')):
                    if isinstance(v,(str,int,float,bool)) or v is None: vals[k]=v
                    elif isinstance(v,list) and len(v)<=30: vals[k]=v
            return {'json_keys':keys,'identity_fields':vals}
    except Exception: return None
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--source-jsonl',required=True)
    ap.add_argument('--artifact-roots',nargs='*',default=DEFAULT_ROOTS)
    ap.add_argument('--max-files',type=int,default=300000)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    src=Path(args.source_jsonl)
    src_sha,src_hashed=sha256_file(src, cap=2*1024*1024*1024)
    hits=[]; scanned=0; skipped=0; unknown=[]
    pats={c:{name:fn(c) for name,fn in ALIAS_PATTERNS.items()} for c in CANDIDATES}
    excluded=('source_admission_cc_train_full_20261106_v1','source_history_audit_20261106_v2','source_reselection_metadata_20261107_v1','conversation_chronicles/train.jsonl')
    for root in args.artifact_roots:
        rp=Path(root)
        if not rp.exists(): unknown.append({'scope':root,'status':'MISSING'}); continue
        for dp,dns,fns in os.walk(rp):
            dns[:]=[d for d in dns if d not in ('.git','__pycache__','.venv','node_modules')]
            for fn in fns:
                p=Path(dp)/fn; s=str(p)
                if any(x in s for x in excluded): continue
                scanned+=1
                if scanned>args.max_files: break
                try:
                    st=p.stat()
                    if st.st_size>100*1024*1024: skipped+=1; continue
                    # Read bounded chunks; only text-like files are evidence-bearing.
                    with open(p,'rb') as f: b=f.read(8*1024*1024)
                    if b'\x00' in b[:4096]: skipped+=1; continue
                    t=b.decode('utf-8','ignore')
                except Exception: skipped+=1; continue
                for c in CANDIDATES:
                    for typ,pat in pats[c].items():
                        m=pat.search(t)
                        if m:
                            sh,_=sha256_file(p,cap=20*1024*1024)
                            hits.append({'candidate':c,'alias_type':typ,'path':s,'file_size':st.st_size,'file_sha256_prefix_scan':sh,'json_metadata':safe_json_metadata(p)})
            if scanned>args.max_files: break
        if scanned>args.max_files: break
    # Upstream provenance is recorded without requiring a network download.
    provenance={'local_path':str(src),'local_size':src.stat().st_size,'local_sha256':src_sha,'bytes_hashed':src_hashed,'expected_hf_sha256':'f0fccf01ec59eeeab68c0a300b0f1a24bacd8eea8719e77bdbd003c242f1407c','hf_repo':'jihyoung/ConversationChronicles','hf_commit':'9639b3552594a612ceafa76238a34f0f6eeaa207','sha_match_expected':src_sha=='f0fccf01ec59eeeab68c0a300b0f1a24bacd8eea8719e77bdbd003c242f1407c'}
    report={'schema_version':'cc_global_source_history_audit_v1','read_only':True,'candidate_ids':CANDIDATES,'scopes':args.artifact_roots,'scanned_files':scanned,'skipped_files':skipped,'unknown_scopes':unknown,'alias_hits':hits,'alias_hit_count':len(hits),'provenance':provenance,'status':'HOLD_ALIAS_OR_LINEAGE_IF_ANY' if hits else 'NO_ALIAS_HIT_IN_SCANNED_TEXT_SCOPES_BUT_LINEAGE_CONTENT_REMAINS_UNPROVEN','limitations':['bounded text scan; binary/oversize files skipped','parent-family lineage is not encoded reliably in all historical schemas','semantic/near-duplicate content requires separate source-wide fingerprints','no authoring, Solver, training or export was run']}
    (out/'SHORTLIST_GLOBAL_SOURCE_HISTORY_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # small machine-readable candidate statuses
    rows=[]
    for c in CANDIDATES:
        hs=[h for h in hits if h['candidate']==c]
        rows.append({'candidate':c,'alias_hit_count':len(hs),'status':'HOLD' if hs else 'UNKNOWN_NO_ALIAS_HIT','lineage_status':'UNPROVEN','content_duplicate_status':'UNPROVEN'})
    (out/'SHORTLIST_ALIAS_STATUS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'out':str(out),'scanned_files':scanned,'skipped_files':skipped,'alias_hit_count':len(hits),'local_sha256':src_sha},ensure_ascii=False))
if __name__=='__main__': main()
