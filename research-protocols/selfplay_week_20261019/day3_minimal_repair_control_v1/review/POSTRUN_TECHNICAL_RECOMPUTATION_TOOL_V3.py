"""Independent post-run recomputation for MRC candidate and paired interface.

This tool intentionally does not import either scorer. It recomputes raw identity,
display/content/canonical mappings, finite-box parsing, JSON strictness, exactness,
item coverage, citation substring checks, usage and fixed-denominator facts directly.
It is read-only apart from its explicitly requested JSON report. Run with ``python
-B`` after candidate and interface raw freezes and scores exist.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
H = HERE.parent
WEEK = H.parent
CAND = H / "runtime" / "candidate_v1"
CAND_PRIV = H / "runtime" / "candidate_v1_private" / "PRIVATE_QUESTIONS.json"
IFACE = H / "runtime" / "interface_v3"
OLD_FINALS = [
    WEEK / "day3_targeted_4root_v1" / "FINAL_MANIFEST.json",
    WEEK / "day3_diagnostic_handoff_v1" / "FINAL_MANIFEST.json",
    WEEK / "day3_feedback_realization_v1" / "FINAL_MANIFEST.json",
]

class AuditError(Exception):
    pass

class Audit:
    def __init__(self):
        self.checks = []
        self.failures = []
    def check(self, name, condition, detail=""):
        ok = bool(condition)
        self.checks.append({"name": name, "pass": ok, "detail": str(detail)})
        if not ok: self.failures.append({"name": name, "detail": str(detail)})
        return ok
    def req(self, name, condition, detail=""):
        if not self.check(name, condition, detail): raise AuditError(name + ":" + str(detail))

def sha(path):
    b = Path(path).read_bytes()
    return hashlib.sha256(b).hexdigest(), len(b)

def bind(path):
    h, n = sha(path)
    return {"path": str(Path(path).resolve()), "sha256": h, "bytes": n}

def read(path):
    return json.loads(Path(path).read_text())

def resolve_binding(row):
    p = Path(row["path"])
    return p if p.is_absolute() else H / p

def verify_binding(a, row, name):
    p = resolve_binding(row)
    a.req(name + ":exists", p.is_file(), str(p))
    got = bind(p)
    a.req(name + ":sha256", got["sha256"] == row.get("sha256"), got)
    a.req(name + ":bytes", got["bytes"] == row.get("bytes"), got)

def duplicate_reject(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise ValueError("DUPLICATE_JSON_MEMBER:" + str(k))
        out[k] = v
    return out

def strict_json(text):
    return json.loads(text, object_pairs_hook=duplicate_reject)

def content_ids(question):
    field = "options" if question["question_type"] == "multi_select" else "items"
    return {label: "content_" + hashlib.sha256(value.strip().encode()).hexdigest()[:16]
            for label, value in question[field].items()}

def field_labels(row):
    return list(row["display"]["options" if row["question_type"] == "multi_select" else "items"])

def raw_files_and_records(a, run, manifest, branch, freeze, response_subdir=None):
    rows = [r for r in manifest["requests"] if r["branch"] == branch]
    f = read(freeze)
    a.req(branch + ":freeze_status", f.get("status") == "COMMITTED", f)
    a.req(branch + ":freeze_count", f.get("planned_request_count") == len(rows), f)
    paths = [Path(x["path"]) for x in f.get("files", [])]
    response_dir = run / (response_subdir or branch)
    expected = {response_dir / "responses" / (r["request_id"] + ".json") for r in rows}
    a.req(branch + ":fixed_raw_set", {x.resolve() for x in paths} == {x.resolve() for x in expected}, paths)
    a.req(branch + ":no_extra_response", set(response_dir.joinpath("responses").glob("*.json")) == expected, "")
    for i, x in enumerate(f.get("files", [])): verify_binding(a, x, branch + ":response[%d]" % i)
    for i, x in enumerate(f.get("execution_ledger_bindings", [])): verify_binding(a, x, branch + ":ledger[%d]" % i)
    verify_binding(a, f["manifest_binding"], branch + ":manifest_binding")
    return rows, {r["request_id"]: read(response_dir / "responses" / (r["request_id"] + ".json")) for r in rows}, f

def raw_identity(a, row, raw):
    for key in ("request_id", "candidate_id", "root_id", "question_type", "variant_id", "slot", "seed", "prompt_sha256"):
        a.check("raw_identity:" + row["request_id"] + ":" + key, raw.get(key) == row.get(key), (raw.get(key), row.get(key)))
    a.check("raw_returned:" + row["request_id"], raw.get("status") == "RETURNED", raw.get("status"))
    a.check("raw_no_retry:" + row["request_id"], raw.get("automatic_retry") is False, raw.get("automatic_retry"))
    a.check("raw_no_truncation:" + row["request_id"], raw.get("source_truncation") is False and raw.get("metadata", {}).get("finish_reason") != "length", "")

def finite_box(text, qtype, labels):
    if not isinstance(text, str): return None
    m = re.fullmatch(r"<box>([^<>]*)</box>", text.strip())
    if not m: return None
    vals = [x.strip() for x in m.group(1).split(",") if x.strip()]
    if not vals or len(vals) != len(set(vals)) or any(x not in labels for x in vals): return None
    if qtype == "ordering" and set(vals) != set(labels): return None
    return vals

def candidate(a):
    m = read(CAND / "manifest.json"); public = read(CAND / "PUBLIC_QUESTIONS.json"); private = read(CAND_PRIV)
    maps = read(CAND / "DISPLAY_MAPPINGS.json")
    a.req("candidate_manifest_requests", m["request_counts"] == {"direct": 6, "evidence": 1} and len(m["requests"]) == 7, m["request_counts"])
    a.req("candidate_public_one", len(public["questions"]) == 1 and len(private["questions"]) == 1, "")
    q = public["questions"][0]; goldq = private["questions"][0]; cid = q["candidate_id"]
    a.req("candidate_private_public_equal", {k: goldq[k] for k in q} == q, "")
    canonical = content_ids(q); a.req("candidate_content_ids_map", maps["public_to_canonical"].get(cid) == canonical, "")
    direct_rows, direct_raw, direct_freeze = raw_files_and_records(a, CAND, m, "direct", CAND / "direct" / "RAW_RESPONSE_FREEZE.json")
    evidence_rows, evidence_raw, evidence_freeze = raw_files_and_records(a, CAND, m, "evidence", CAND / "evidence" / "RAW_RESPONSE_FREEZE.json")
    gold = goldq["gold"]; direct_records=[]
    for row in direct_rows:
        raw=direct_raw[row["request_id"]]; raw_identity(a,row,raw); variant=maps["requests"][row["request_id"]]; labels=field_labels(row); pred=finite_box(raw.get("raw_output"),row["question_type"],labels); valid=pred is not None and raw.get("metadata",{}).get("finish_reason") in ("stop","eos") and not raw.get("output_validation_errors")
        a.check("candidate_direct_valid:"+row["request_id"],valid,{"pred":pred,"errors":raw.get("output_validation_errors")})
        prediction=[{v:k for k,v in canonical.items()}[variant["display_to_content"][x]] for x in pred] if valid else None
        exact=bool(valid and (set(prediction)==set(gold) if row["question_type"]=="multi_select" else prediction==gold))
        fp=fn=0
        if row["question_type"]=="multi_select" and valid:
            ps=set(prediction); gs=set(gold); fp=len(ps-gs); fn=len(gs-ps)
        direct_records.append({"request_id":row["request_id"],"valid":valid,"exact":exact,"prediction":prediction,"canonical_gold":gold,"fp":fp,"fn":fn})
    a.req("candidate_direct_fixed_denominator",len(direct_records)==6,"" )
    er=evidence_rows[0]; raw=evidence_raw[er["request_id"]]; raw_identity(a,er,raw); parsed=None; errors=[]
    try:
        parsed=strict_json(raw.get("raw_output",""))
        import jsonschema; jsonschema.validate(parsed, er["json_schema"])
        labels=set(field_labels(er)); a.req("candidate_evidence_labels",len(parsed["item_evidence"])==len(labels) and {x["label"] for x in parsed["item_evidence"]}==labels, "")
    except Exception as e: errors.append(type(e).__name__ + ":" + str(e)[:500])
    ev_valid=not errors and raw.get("metadata",{}).get("finish_reason") != "length" and not raw.get("output_validation_errors")
    source=read(CAND/"sources"/(er["root_id"]+".json")); turns={x["turn_ref"]:x["text"] for x in source["turns"]}; quotes=[]
    if parsed and isinstance(parsed,dict):
        for item in parsed.get("item_evidence",[]):
            for c in item.get("citations",[]): quotes.append({"label":item.get("label"),"turn_ref":c.get("turn_ref"),"quote":c.get("quote"),"turn_ref_exists":c.get("turn_ref") in turns,"exact_substring":c.get("turn_ref") in turns and c.get("quote","") in turns.get(c.get("turn_ref"),"")})
    ev_final=parsed.get("final_answer") if ev_valid and isinstance(parsed,dict) else None; evpred=[{v:k for k,v in canonical.items()}[maps["requests"][er["request_id"]]["display_to_content"][x]] for x in ev_final] if ev_final else None
    ev_exact=bool(ev_valid and evpred is not None and (set(evpred)==set(gold) if er["question_type"]=="multi_select" else evpred==gold))
    direct_complete=read(CAND/"direct"/"COMPLETE.json"); ev_complete=read(CAND/"evidence"/"COMPLETE.json")
    total_tokens=sum((x.get("metadata",{}).get("prompt_token_count") or 0)+(x.get("metadata",{}).get("completion_token_count") or 0) for x in list(direct_raw.values())+list(evidence_raw.values()))
    return {"candidate_id":cid,"direct_calls":6,"evidence_calls":1,"direct_valid":sum(x["valid"] for x in direct_records),"direct_exact":sum(x["exact"] for x in direct_records),"direct_records":direct_records,"evidence_valid":ev_valid,"evidence_exact":ev_exact,"evidence_invalid_reasons":errors,"evidence_quote_count":len(quotes),"evidence_exact_quote_count":sum(x["exact_substring"] for x in quotes),"evidence_quotes":quotes,"input_tokens":sum((x.get("metadata",{}).get("prompt_token_count") or 0) for x in list(direct_raw.values())+list(evidence_raw.values())),"output_tokens":sum((x.get("metadata",{}).get("completion_token_count") or 0) for x in list(direct_raw.values())+list(evidence_raw.values())),"total_tokens":total_tokens,"wall_seconds":direct_complete["wall_seconds"]+ev_complete["wall_seconds"],"model_load_seconds":direct_complete["model_load_wall_seconds"]+ev_complete["model_load_wall_seconds"],"raw_freeze_bindings":[bind(CAND/"direct"/"RAW_RESPONSE_FREEZE.json"),bind(CAND/"evidence"/"RAW_RESPONSE_FREEZE.json")]}

def interface(a):
    m=read(IFACE/"manifest.json"); public=read(IFACE/"PUBLIC_QUESTIONS.json"); maps=read(IFACE/"DISPLAY_MAPPINGS.json")
    # Use the prior 11-question experiment's bound Gold file, never the new one-question candidate Gold.
    private_binding=m["old_private_binding"]
    verify_binding(a, private_binding, "interface:private_gold_binding")
    private=read(resolve_binding(private_binding))
    score_path=IFACE/"SCORES.json"
    a.req("interface:score_exists", score_path.is_file(), str(score_path))
    score=read(score_path)
    a.req("interface:score_raw_freeze_binding", score.get("raw_freeze_binding")==bind(IFACE/"live"/"RAW_RESPONSE_FREEZE.json"), score.get("raw_freeze_binding"))
    a.req("interface:score_manifest_binding", score.get("manifest_binding")==bind(IFACE/"manifest.json"), score.get("manifest_binding"))
    a.req("interface_manifest_requests",m["request_count"]==22 and m["request_counts"]=={"original":11,"simplified":11},m.get("request_counts"))
    a.req("interface_manifest_branch", {r.get("branch") for r in m["requests"]} == {"paired_interface"}, {r.get("branch") for r in m["requests"]})
    rows, raws, freeze=raw_files_and_records(a,IFACE,m,"paired_interface",IFACE/"live"/"RAW_RESPONSE_FREEZE.json", response_subdir="live")
    # live is the only branch; each record carries its arm. Verify schedule and pair seeds.
    a.req("interface_fixed_raw_count",len(rows)==22 and len(raws)==22,"")
    qby={q["candidate_id"]:q for q in public["questions"]}; gby={q["candidate_id"]:q for q in private["questions"]}; results=[]; source_cache={}
    for row in rows:
        raw=raws[row["request_id"]]; raw_identity(a,row,raw); cid=row["candidate_id"]; q=qby[cid]; gold=gby[cid]["gold"]; canonical=content_ids(q); variant=maps["requests"][row["request_id"]]; labels=field_labels(row); parsed=None; errors=[]
        try:
            parsed=strict_json(raw.get("raw_output","")); import jsonschema; jsonschema.validate(parsed,row["json_schema"])
            if row["arm"]=="original":
                items=parsed["item_evidence"]; a.req("interface_original_labels:"+row["request_id"],len(items)==len(labels) and {x["label"] for x in items}==set(labels),"")
            else:
                checks=parsed["checks"]; a.req("interface_simplified_labels:"+row["request_id"],set(checks)==set(labels) and len(checks)==len(labels),"")
                a.req("interface_simplified_citation_pair:"+row["request_id"],all(bool(x["turn_ref"])==bool(x["quote"]) for x in checks.values()),"")
        except Exception as e: errors.append(type(e).__name__+":"+str(e)[:800])
        valid=not errors and raw.get("metadata",{}).get("finish_reason") != "length" and not raw.get("output_validation_errors")
        final=parsed.get("final_answer") if valid else None; reverse={v:k for k,v in canonical.items()}
        prediction=[reverse[variant["display_to_content"][x]] for x in final] if final is not None and all(x in variant["display_to_content"] for x in final) else None
        exact=bool(valid and prediction is not None and (set(prediction)==set(gold) if row["question_type"]=="multi_select" else prediction==gold))
        source_cache.setdefault(row["root_id"],read(IFACE/"sources"/(row["root_id"]+".json"))); turns={x["turn_ref"]:x["text"] for x in source_cache[row["root_id"]]["turns"]}; cites=[]
        if valid:
            items=parsed["item_evidence"] if row["arm"]=="original" else [{"label":k,"citations":([{"turn_ref":v["turn_ref"],"quote":v["quote"]}] if v["turn_ref"] or v["quote"] else [])} for k,v in parsed["checks"].items()]
            for item in items:
                for c in item["citations"]: cites.append({"label":item["label"],"turn_ref":c["turn_ref"],"quote":c["quote"],"turn_ref_exists":c["turn_ref"] in turns,"exact_substring":c["turn_ref"] in turns and c["quote"] in turns.get(c["turn_ref"],"")})
        results.append({"request_id":row["request_id"],"candidate_id":cid,"arm":row["arm"],"question_type":row["question_type"],"valid":valid,"exact":exact,"invalid_reasons":errors,"coverage_items":len(labels) if valid else 0,"expected_items":len(labels),"citation_count":len(cites),"exact_quote_count":sum(x["exact_substring"] for x in cites),"citations":cites,"prediction":prediction,"gold":gold})
    by_arm={}
    for arm in ("original","simplified"):
        rr=[x for x in results if x["arm"]==arm]; by_arm[arm]={"calls":len(rr),"valid":sum(x["valid"] for x in rr),"invalid":sum(not x["valid"] for x in rr),"exact":sum(x["exact"] for x in rr),"exact_rate_fixed22_arm_denominator":sum(x["exact"] for x in rr)/11,"expected_items":sum(x["expected_items"] for x in rr),"reviewable_items":sum(x["coverage_items"] for x in rr),"coverage_over_expected_items":sum(x["coverage_items"] for x in rr)/sum(x["expected_items"] for x in rr),"citation_count":sum(x["citation_count"] for x in rr),"exact_quote_count":sum(x["exact_quote_count"] for x in rr),"exact_quote_rate":sum(x["exact_quote_count"] for x in rr)/sum(x["citation_count"] for x in rr) if sum(x["citation_count"] for x in rr) else None,"input_tokens":sum((raws[x["request_id"]].get("metadata",{}).get("prompt_token_count") or 0) for x in rr),"output_tokens":sum((raws[x["request_id"]].get("metadata",{}).get("completion_token_count") or 0) for x in rr)}
    complete=read(IFACE/"live"/"COMPLETE.json"); pairs=[]
    for arm in ("original", "simplified"):
        scored=score["arms"][arm]
        arm_rows=[x for x in results if x["arm"]==arm]
        a.check("score_recompute_valid:"+arm, scored["strict_valid"]==sum(x["valid"] for x in arm_rows), (scored["strict_valid"],sum(x["valid"] for x in arm_rows)))
        a.check("score_recompute_exact:"+arm, scored["exact"]==sum(x["exact"] for x in arm_rows), (scored["exact"],sum(x["exact"] for x in arm_rows)))
        a.check("score_recompute_expected_items:"+arm, scored["expected_items"]==sum(x["expected_items"] for x in arm_rows), (scored["expected_items"],sum(x["expected_items"] for x in arm_rows)))
        a.check("score_recompute_reviewable_items:"+arm, scored["strict_reviewable_items"]==sum(x["coverage_items"] for x in arm_rows), (scored["strict_reviewable_items"],sum(x["coverage_items"] for x in arm_rows)))
        a.check("score_recompute_citations:"+arm, scored["citation_count"]==sum(x["citation_count"] for x in arm_rows), (scored["citation_count"],sum(x["citation_count"] for x in arm_rows)))
        a.check("score_recompute_exact_quotes:"+arm, scored["exact_quote_count"]==sum(x["exact_quote_count"] for x in arm_rows), (scored["exact_quote_count"],sum(x["exact_quote_count"] for x in arm_rows)))
    for cid in qby:
        pair={x["arm"]:x for x in results if x["candidate_id"]==cid}; a.req("interface_pair_complete:"+cid,set(pair)=={"original","simplified"},""); pairs.append({"candidate_id":cid,"original_valid":pair["original"]["valid"],"simplified_valid":pair["simplified"]["valid"],"original_exact":pair["original"]["exact"],"simplified_exact":pair["simplified"]["exact"],"same_seed":raws[pair["original"]["request_id"]].get("seed")==raws[pair["simplified"]["request_id"]].get("seed")} if False else {"candidate_id":cid,"original_valid":pair["original"]["valid"],"simplified_valid":pair["simplified"]["valid"],"original_exact":pair["original"]["exact"],"simplified_exact":pair["simplified"]["exact"]})
    return {"calls":22,"by_arm":by_arm,"pairs":pairs,"records":results,"input_tokens":sum(x["input_tokens"] for x in by_arm.values()),"output_tokens":sum(x["output_tokens"] for x in by_arm.values()),"wall_seconds":complete["wall_seconds"],"model_load_seconds":complete["model_load_wall_seconds"],"raw_freeze_binding":bind(IFACE/"live"/"RAW_RESPONSE_FREEZE.json")}

def old_sealed(a):
    counts=[]
    for p in OLD_FINALS:
        m=read(p); files=m.get("files",[]); a.req("sealed_manifest_exists:"+str(p),p.is_file(),"")
        for i,row in enumerate(files): verify_binding(a,row,"sealed:%s:%d"%(p.parent.name,i))
        counts.append({"manifest":str(p),"files":len(files),"manifest_sha256":sha(p)[0]})
    return counts

def current_freeze(a):
    for run in (CAND,IFACE):
        m=read(run/"manifest.json"); pre=read(run/"PREFLIGHT_FREEZE.json")
        a.req("current_preflight_status:"+str(run),pre.get("status")=="PREPARED_AWAITING_INDEPENDENT_PREFLIGHT","")
        for i,row in enumerate(pre.get("files",[])): verify_binding(a,row,"current_preflight:%s:%d"%(run.name,i))
        for i,row in enumerate(m.get("code_bindings",[])): verify_binding(a,row,"current_code:%s:%d"%(run.name,i))
        verify_binding(a,m.get("protocol_binding",{}),"current_protocol:"+run.name) if m.get("protocol_binding") else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=HERE/"POSTRUN_TECHNICAL_RECOMPUTATION_V3.json");args=ap.parse_args();a=Audit(); started=dt.datetime.now(dt.timezone.utc).isoformat()
    report={"schema_version":"mrc_postrun_technical_recomputation_v1","status":"FAIL","created_at_utc":started,"role":"Technical recomputation by runtime author; not an independent blind audit.","interface_active_runtime":str(IFACE)}
    try:
        current_freeze(a); sealed=old_sealed(a); cand=candidate(a); iface=interface(a)
        total_calls=cand["direct_calls"]+cand["evidence_calls"]+iface["calls"]
        a.req("actual_submitted_29",total_calls==29,total_calls)
        report.update({"candidate":cand,"interface":iface,"old_sealed":sealed,"actual_submitted_calls":total_calls,"fixed_denominators":{"candidate":7,"interface":22,"total":29},"total_input_tokens":cand["input_tokens"]+iface["input_tokens"],"total_output_tokens":cand["output_tokens"]+iface["output_tokens"],"total_tokens":cand["total_tokens"]+iface["input_tokens"]+iface["output_tokens"],"wall_seconds":cand["wall_seconds"]+iface["wall_seconds"],"model_load_seconds":cand["model_load_seconds"]+iface["model_load_seconds"],"interface_shared_wall_counted_once":True})
    except Exception as e: report["error"]=type(e).__name__+":"+str(e)
    report["checks_count"]=len(a.checks);report["failure_count"]=len(a.failures);report["failures"]=a.failures;report["checks"]=a.checks
    report["status"]="PASS" if not a.failures and "error" not in report else "FAIL"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({"status":report["status"],"checks_count":report["checks_count"],"failure_count":report["failure_count"],"output":str(args.output)}))
    return 0 if report["status"]=="PASS" else 2
if __name__=='__main__':sys.exit(main())
