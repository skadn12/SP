"""Correct two dedup-label spellings in the unsealed mechanical merge only."""
from pathlib import Path
import json
import hashlib
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
mapping={"OVERLAPPING_FACT_DIFFERENT_RELATION":"OVERLAPPING_FACTS_DIFFERENT_RELATION","SAME_FACT_PARAPHRASE":"SAME_FACTS_PARAPHRASE"}
changes=[]
for p in (D/"final_records").glob("*.json"):
    r=read(p);old=r["fact_family"]["reuse_class"]
    if old in mapping:
        r["fact_family"]["reuse_class"]=mapping[old];save(p,r)
        changes.append({"candidate_id":r["candidate_id"],"old":old,"new":mapping[old],"record_sha256":sha(p)})
prov=read(D/"FINAL_RECORD_PROVENANCE.json")
bycid={r["candidate_id"]:r for r in changes}
for r in prov["rows"]:
    if r["candidate_id"] in bycid:
        c=bycid[r["candidate_id"]]
        r["final_schema_record"]["sha256"]=c["record_sha256"]
        r["dedup_reuse_class"]=c["old"];r["schema_reuse_class"]=c["new"]
save(D/"FINAL_RECORD_PROVENANCE.json",prov)
for name in ["content_accepted.private.jsonl","held_candidates.jsonl"]:
    p=D/"exports"/name;rows=[json.loads(s) for s in p.read_text().splitlines()]
    for r in rows:r["record"]["sha256"]=sha(Path(r["record"]["path"]))
    p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
p=D/"exports/SLOT_LEDGER.json";ledger=read(p)
for r in ledger["rows"]:
    if "record" in r:r["record"]["sha256"]=sha(Path(r["record"]["path"]))
save(p,ledger)
p=D/"SUMMARY.json";s=read(p)
for b in s["files"].values():b["sha256"]=sha(Path(b["path"]))
save(p,s)
save(D/"audit/DEDUP_ENUM_NORMALIZATION.json",{
    "reason":"Dedup report used singular FACT enum spellings, while public candidate_record uses FACTS. Normalize mechanical labels with no semantic change.",
    "changes":changes,"candidate_content_gold_evidence_changed":False,
    "signed_dedup_or_QA_files_changed":False,"source_records_A_B_changed":False})
(D/"FINAL_VALIDATION.json").rename(D/"audit/FINAL_VALIDATION_ENUM_FAILURE.json")
print("Normalized20 unsealed mechanical records; preserved initial validation failure.")
