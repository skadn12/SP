#!/usr/bin/env python3
"""Read-only mechanical preflight for the isolated-author pilot.

The checker never edits an existing execution artifact. It writes only new
exclusive-create diagnostics in the requested output directory.  It treats
malformed JSON, duplicate keys, trailing data, unexpected fields, hash/binding
mismatches, access-boundary violations, missing NEW selector packets, and
invalid candidate answer structures as hard failures.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, re, sys
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "preflight_diagnostic_20261107_v1"

class DuplicateKey(ValueError):
    pass

def no_constants(s: str) -> Any:
    raise ValueError(f"non-standard JSON constant {s!r}")

def pairs_no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise DuplicateKey(f"duplicate key {k!r}")
        out[k] = v
    return out

def read_json(path: Path):
    """Strict JSON parse: UTF-8, one object/value, no duplicate keys/trailing data."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"invalid UTF-8: {e}") from e
    # json.loads rejects trailing data and invokes our duplicate-key hook.
    val = json.loads(text, object_pairs_hook=pairs_no_duplicates,
                     parse_constant=no_constants)
    return val, raw

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def json_hash(path: Path):
    val, raw = read_json(path)
    return val, raw, sha(raw)

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)

def result(ok=True, **kwargs):
    return {"pass": bool(ok), **kwargs}

def add_issue(container, code, message, **detail):
    container.append({"code": code, "message": message, **detail})

def is_hex64(v):
    return isinstance(v, str) and bool(HEX64.fullmatch(v))

def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve()); return True
    except Exception: return False

def path_value(v, expected: Path) -> bool:
    return isinstance(v, str) and Path(v).resolve() == expected.resolve()

def schedule_root_id(root):
    """Return the restricted-schedule root spelling for a role-packet root."""
    if not isinstance(root, str):
        return root
    return root.replace("cc-train-episode-", "")

def schedule_workflow(workflow):
    """Map packet workflow enum to the restricted schedule's public label."""
    return {
        "OLD": "current_author_self_selection",
        "NEW": "semantic_slots_independent_coverage_selection_authoring",
    }.get(workflow, workflow)

def scan_keys_values(value, forbidden, path="$", hits=None):
    if hits is None: hits=[]
    if isinstance(value, dict):
        for k,v in value.items():
            kl = str(k).lower()
            if any(token in kl for token in forbidden):
                hits.append({"path": path+"."+str(k), "key": str(k)})
            scan_keys_values(v, forbidden, path+"."+str(k), hits)
    elif isinstance(value, list):
        for i,v in enumerate(value): scan_keys_values(v, forbidden, f"{path}[{i}]", hits)
    elif isinstance(value, str):
        low=value.lower()
        # Path/text leaks are checked separately; this catches obvious identity labels.
        for token in forbidden:
            if token in low and len(token) >= 5:
                hits.append({"path": path, "value_token": token})
    return hits

def source_turn_map(source):
    out={}; total=0
    if not isinstance(source, dict): return out, total
    for key,val in source.items():
        m=re.fullmatch(r"(.+)_dialogue", key)
        if not m or not isinstance(val,list): continue
        sess=m.group(1); speakers=source.get(sess+"_speakers", [])
        for i,text in enumerate(val,1):
            total+=1
            out[(sess,i)]={"text": text if isinstance(text,str) else str(text),
                            "speaker": speakers[i-1] if i-1<len(speakers) else None}
        # aliases used by generated artifacts
    aliases={}
    for sess,i in list(out):
        n=int(re.search(r"(\\d+)$", sess).group(1)) if False else None
    ordered=[]
    for key in source:
        m=re.fullmatch(r"(.+)_dialogue",key)
        if m and isinstance(source[key],list): ordered.append(m.group(1))
    # Source keys are in chronological order; map S01/S02/... aliases.
    for idx,sess in enumerate(ordered,1): aliases[f"S{idx:02d}"]=sess
    for (sess,i),v in list(out.items()):
        out[(sess,i)] = v
        for alias,real in aliases.items():
            if real==sess: out[(alias,i)] = v
    return out,total

def parse_turn_ref(ref):
    if not isinstance(ref,str): return None
    # Accept canonical session:Tn and Snn:Tn.  A range is handled as endpoints.
    m=re.fullmatch(r"([^:]+):T(\\d+)",ref)
    if m: return [(m.group(1),int(m.group(2)))]
    m=re.fullmatch(r"([^:]+)\\[(\\d+)\\]",ref)
    if m: return [(m.group(1),int(m.group(2)))]
    m=re.fullmatch(r"([^:]+):T(\\d+)-T(\\d+)",ref)
    if m: return [(m.group(1),int(m.group(2))),(m.group(1),int(m.group(3)))]
    return None

def evidence_checks(obj, turnmap):
    issues=[]; checked=0
    def walk(v):
        nonlocal checked
        if isinstance(v,dict):
            if "turn_ref" in v and "quote" in v:
                checked+=1; ref=v.get("turn_ref"); quote=v.get("quote")
                points=parse_turn_ref(ref)
                if points is None: add_issue(issues,"TURN_REF_FORMAT","unsupported turn_ref",turn_ref=ref)
                elif not isinstance(quote,str): add_issue(issues,"QUOTE_TYPE","quote is not string",turn_ref=ref)
                else:
                    # For range references, a quote may be a quote_pair list; a string
                    # is checked against either endpoint as legacy artifacts permit.
                    ok=False
                    for sess,n in points:
                        rec=turnmap.get((sess,n))
                        if rec and quote in rec["text"]: ok=True
                    if not ok:
                        add_issue(issues,"QUOTE_NOT_SUBSTRING","quote is not an exact substring at referenced turn",turn_ref=ref)
            for x in v.values(): walk(x)
        elif isinstance(v,list):
            for x in v: walk(x)
    walk(obj)
    return {"pass": not issues, "evidence_entries_checked": checked, "issues": issues}

def candidate_checks(wrapper, turnmap):
    issues=[]
    if not isinstance(wrapper,dict): return result(False, issues=[{"code":"CANDIDATE_WRAPPER_TYPE","message":"candidate artifact must be an object"}])
    cands=wrapper.get("candidates")
    if not isinstance(cands,list):
        add_issue(issues,"CANDIDATES_MISSING","candidates must be a list")
        return result(False, issues=issues)
    if len(cands) not in (0,1): add_issue(issues,"CANDIDATE_COUNT","candidates must contain 0 or 1",count=len(cands))
    evidence=evidence_checks(wrapper,turnmap)
    issues.extend(evidence["issues"])
    for c in cands:
        if not isinstance(c,dict): add_issue(issues,"CANDIDATE_TYPE","candidate must be object"); continue
        typ=c.get("question_type", c.get("type"))
        opts=c.get("options")
        if isinstance(opts,list):
            ids=[]
            for o in opts:
                if isinstance(o,dict): ids.append(o.get("option_id",o.get("id")))
                else: ids.append(None)
            if any(not isinstance(x,str) for x in ids) or len(set(ids))!=len(ids): add_issue(issues,"OPTION_IDS","option IDs must be unique strings")
            gold=c.get("gold_option_ids", c.get("correct_option_ids"))
            if typ in ("multi_select","multi-select"):
                if not isinstance(gold,list) or len(set(gold))!=len(gold) or any(x not in ids for x in gold): add_issue(issues,"MULTI_SELECT_GOLD","multi_select gold must be a unique subset of option IDs")
            elif typ=="ordering":
                order=c.get("correct_order",gold)
                if not isinstance(order,list) or len(order)!=len(ids) or len(set(order))!=len(order) or set(order)!=set(ids): add_issue(issues,"ORDERING_NOT_FULL","ordering answer must be a unique full order of every option")
            elif typ=="partial_order":
                qp=c.get("query_pairs",[]); gp=c.get("gold_pairs",[])
                def pairset(x):
                    out=[]
                    for p in x if isinstance(x,list) else []:
                        if isinstance(p,(list,tuple)) and len(p)==2: out.append(tuple(p))
                        elif isinstance(p,dict) and "from" in p and "to" in p: out.append((p["from"],p["to"]))
                    return out
                q=pairset(qp); g=pairset(gp)
                if len(q)!=len(set(q)): add_issue(issues,"PARTIAL_QUERY_DUP","query_pairs contain duplicates")
                if len(g)!=len(set(g)): add_issue(issues,"PARTIAL_GOLD_DUP","gold_pairs contain duplicates")
                qset=set(q); gset=set(g)
                if not gset <= qset: add_issue(issues,"PARTIAL_GOLD_OUTSIDE","gold_pairs must be subset of query_pairs")
                if any(a==b for a,b in qset): add_issue(issues,"PARTIAL_SELF_PAIR","partial-order query contains self pair")
                # Pair closure: if a->b and b->c are entailed, a->c must be declared.
                closure=set(gset); changed=True
                while changed:
                    changed=False
                    for a,b in list(closure):
                        for c0,d in list(closure):
                            if b==c0 and (a,d) not in closure:
                                closure.add((a,d)); changed=True
                if not closure <= gset: add_issue(issues,"PARTIAL_PAIR_CLOSURE","gold_pairs omit a transitive entailed pair",missing=sorted(closure-gset))
                if "query_scope" not in c or "pair_audit" not in c: add_issue(issues,"PARTIAL_AUDIT_MISSING","partial_order requires query_scope and pair_audit")
    return result(not issues, issues=issues, candidates_count=len(cands), evidence=evidence)

# Strict schemas are intentionally small: they describe the controller-to-role packet,
# not generated question files. Future packet builders should fail closed on additions.
ROLE_REQUIRED={"opportunity_id","root_id","workflow","group_id","source_path","source_sha256","source_turn_count","neutral_path","neutral_sha256","output_dir","candidate_id","contract_path","contract_sha256","opportunity_ceiling","slots_ceiling","generation_authorization"}
ROLE_OPTIONAL={"feedback_card_path","feedback_card_sha256"}
FORBIDDEN_ROLE_KEYS={"target","gold","solver","history","coverage","question","option","answer","selection","reviewer","adjudication"}
SELECTOR_FORBIDDEN={"root_id","group_id","workflow","arm","feedback","author_id","candidate_id","source_path","neutral_path"}

class Checker:
    def __init__(self, base:Path):
        self.base=base; self.issues=[]; self.rows=[]; self.counts={}
        self.manifest={}; self.input_manifest={}; self.schedule={}; self.auth={}; self.access={}
    def issue(self, code,msg,**d): add_issue(self.issues,code,msg,**d)
    def load_top(self, name):
        p=self.base/name
        try: v,raw,h=json_hash(p); return v,raw,h,None
        except Exception as e: return None,None,None,str(e)
    def run(self):
        # Controller authorization and immutable manifests are read-only reference inputs.
        for name,attr in [("PACKET_MANIFEST.json","manifest"),("inputs/INPUT_MANIFEST.json","input_manifest"),("SCHEDULE_RESTRICTED.json","schedule"),("AUTHOR_LAUNCH_AUTHORIZATION.json","auth"),("ACCESS_RULES.json","access")]:
            v,raw,h,e=self.load_top(name)
            setattr(self,attr,v if isinstance(v,dict) else {})
            if e: self.issue("REFERENCE_JSON_INVALID",f"{name}: {e}",path=rel(self.base/name))
        auth=self.auth
        scope=auth.get("scope",{}) if isinstance(auth,dict) else {}
        if scope and any(scope.get(k) for k in ("solver","training","export","old_execution_reuse","retry","fallback")):
            self.issue("AUTH_SCOPE_TOO_BROAD","authorization permits forbidden downstream action",scope=scope)
        # The authorization artifact records an explicit isolated-author
        # approval, while the immutable access/manifest/schedule controls are
        # still frozen at NONE.  These are separate controller states; do not
        # silently choose one and do not infer permission to launch from the
        # approval file alone.
        auth_states={
            "AUTHOR_LAUNCH_AUTHORIZATION.generation_authorization": auth.get("generation_authorization"),
            "PACKET_MANIFEST.generation_authorization": self.manifest.get("generation_authorization"),
            "SCHEDULE_RESTRICTED.generation_authorization": self.schedule.get("generation_authorization"),
            "ACCESS_RULES.current_generation_authorization": self.access.get("current_generation_authorization"),
        }
        present_states={k:v for k,v in auth_states.items() if v is not None}
        if len(set(present_states.values())) > 1:
            self.issue("AUTHORIZATION_STATE_CONFLICT",
                       "controller authorization artifacts disagree; no new execution may be launched from this diagnostic",
                       states=present_states,
                       approval_status=auth.get("status"),
                       packet_status=self.manifest.get("status"),
                       packet_authorized=self.manifest.get("author_launch_authorized"),
                       access_status=self.access.get("status"))
        # The next-execution gate and isolated selector contract require a
        # status/reason/hash-bearing selection record.  The current dedicated
        # selector-output gate intentionally accepts only selected_slot_id;
        # this incompatibility is surfaced explicitly rather than hidden in
        # per-file extra-field failures.
        gate_path=self.base/"NEXT_EXECUTION_GATE_20261107_ZH.md"
        selector_contract_path=self.base/"contracts"/"ISOLATED_SELECTOR_CONTRACT.txt"
        try:
            gate_text=gate_path.read_text(encoding="utf-8")
            selector_text=selector_contract_path.read_text(encoding="utf-8")
            required_tokens=["selected_slot_id", "理由", "输入哈希"]
            gate_has_status=("status" in gate_text or "选择状态" in gate_text or "状态" in gate_text)
            if all(t in gate_text for t in required_tokens) and gate_has_status and "status" in selector_text and "input hashes" in selector_text.lower():
                self.issue("CONTRACT_CONFLICT",
                           "selector gate requires status/reason/input hashes but the current anonymous output check allows only selected_slot_id",
                           gate_path=rel(gate_path), selector_contract_path=rel(selector_contract_path),
                           required_output_fields=["selected_slot_id","status","selection_reason","input_hashes"],
                           checker_allowed_output_fields=["selected_slot_id"],
                           consequence="prior full selection records remain quarantined until one selector contract is frozen")
        except Exception as e:
            self.issue("CONTRACT_REFERENCE_READ_FAILURE", "could not inspect selector contract references", error=str(e))
        # Do not infer authorization from generated artifacts: require packet-level NONE hold.
        role_dir=self.base/"role_packets"
        packet_rows=[]
        packets=self.manifest.get("packets",[]) if isinstance(self.manifest,dict) else []
        by_opp={p.get("opportunity_id"):p for p in packets if isinstance(p,dict)}
        input_records=self.input_manifest.get("records",{}) if isinstance(self.input_manifest,dict) else {}
        sched_rows=self.schedule.get("rows",self.schedule.get("opportunities",[])) if isinstance(self.schedule,dict) else []
        sched_by={r.get("opportunity_id"):r for r in sched_rows if isinstance(r,dict)}
        all_inputs=sorted(role_dir.glob("*/INPUT.json")) if role_dir.exists() else []
        for p in packets:
            opp=p.get("opportunity_id"); ip=role_dir/opp/"INPUT.json"
            row={"opportunity_id":opp,"path":rel(ip),"checks":{},"issues":[]}
            try: obj,raw,actual_h=json_hash(ip); row["json_parse"]=True
            except Exception as e:
                row["json_parse"]=False; row["issues"].append({"code":"INPUT_JSON_INVALID","message":str(e)}); packet_rows.append(row); continue
            if not isinstance(obj,dict): row["issues"].append({"code":"INPUT_TYPE","message":"INPUT.json must be an object"}); packet_rows.append(row); continue
            keys=set(obj); unknown=sorted(keys-(ROLE_REQUIRED|ROLE_OPTIONAL)); missing=sorted(ROLE_REQUIRED-keys)
            if unknown: row["issues"].append({"code":"INPUT_EXTRA_FIELDS","fields":unknown})
            if missing: row["issues"].append({"code":"INPUT_MISSING_FIELDS","fields":missing})
            if obj.get("workflow")=="OLD" and obj.get("slots_ceiling") is not None: row["issues"].append({"code":"OLD_SLOTS_CEILING","message":"OLD slots_ceiling must be null"})
            if obj.get("workflow")=="NEW" and obj.get("slots_ceiling") is None: row["issues"].append({"code":"NEW_SLOTS_CEILING","message":"NEW slots_ceiling required"})
            if obj.get("generation_authorization")!="NONE": row["issues"].append({"code":"INPUT_AUTHORIZATION","message":"role input generation_authorization must be NONE"})
            hits=scan_keys_values(obj,FORBIDDEN_ROLE_KEYS)
            # Remove benign key substrings in paths/field names already explicitly handled.
            hits=[h for h in hits if h.get("key") not in {"output_dir","candidate_id"}]
            if hits: row["issues"].append({"code":"ROLE_BOUNDARY_KEYS","hits":hits[:20]})
            root=obj.get("root_id"); rec=input_records.get(str(root).replace("cc-train-episode-","episode-"),{})
            # Input manifest keys are episode-N while packet root is cc-train-episode-N.
            if not rec and isinstance(root,str): rec=input_records.get(root.replace("cc-train-episode-","episode-"),{})
            src=self.base/"inputs"/str(root)/"SOURCE.json"; neu=self.base/"inputs"/str(root)/"NEUTRAL_SOURCE_INFO.json"
            for label,path,decl in (("source",src,obj.get("source_sha256")),("neutral",neu,obj.get("neutral_sha256"))):
                if not path.exists(): row["issues"].append({"code":f"{label.upper()}_MISSING","path":rel(path)}); continue
                try:
                    val,b,h=json_hash(path); row["checks"][f"{label}_hash_matches_packet"]=h==decl
                    if h!=decl: row["issues"].append({"code":f"{label.upper()}_HASH","declared":decl,"actual":h})
                except Exception as e: row["issues"].append({"code":f"{label.upper()}_JSON_INVALID","message":str(e)})
            cpath=self.base/str(obj.get("contract_path","").replace(str(self.base)+os.sep,"")) if not Path(str(obj.get("contract_path",""))).is_absolute() else Path(str(obj.get("contract_path")))
            # absolute packet paths are canonical; reject paths outside this execution.
            if not under(cpath,self.base): row["issues"].append({"code":"CONTRACT_OUTSIDE_EXECUTION","path":str(cpath)})
            elif not cpath.exists(): row["issues"].append({"code":"CONTRACT_MISSING","path":rel(cpath)})
            else:
                ch=sha(cpath.read_bytes()); row["checks"]["contract_hash_matches_packet"]=ch==obj.get("contract_sha256")
                if ch!=obj.get("contract_sha256"): row["issues"].append({"code":"CONTRACT_HASH","declared":obj.get("contract_sha256"),"actual":ch})
            # canonical path and source turn binding
            if not path_value(obj.get("source_path"),src): row["issues"].append({"code":"SOURCE_PATH_BINDING","expected":str(src),"declared":obj.get("source_path")})
            if not path_value(obj.get("neutral_path"),neu): row["issues"].append({"code":"NEUTRAL_PATH_BINDING","expected":str(neu),"declared":obj.get("neutral_path")})
            if src.exists():
                try:
                    sd,_=read_json(src); _,actual_turns=source_turn_map(sd)
                    row["checks"]["source_turn_count_matches_source"]=actual_turns==obj.get("source_turn_count")
                    if actual_turns!=obj.get("source_turn_count"): row["issues"].append({"code":"SOURCE_TURN_COUNT","declared":obj.get("source_turn_count"),"actual":actual_turns})
                except Exception: pass
            if neu.exists():
                try:
                    nd,_=read_json(neu); nt=nd.get("turn_count") if isinstance(nd,dict) else None
                    if nt!=obj.get("source_turn_count"): row["issues"].append({"code":"NEUTRAL_TURN_COUNT","declared":obj.get("source_turn_count"),"actual":nt})
                except Exception: pass
            if opp in sched_by:
                s=sched_by[opp]
                # Restricted schedule deliberately uses short root IDs and
                # descriptive workflow labels, while role packets use the
                # controller's canonical root and OLD/NEW enum. Compare the
                # semantic values after this protocol-only normalization.
                expected_schedule={
                    "root_id": schedule_root_id(obj.get("root_id")),
                    "workflow": schedule_workflow(obj.get("workflow")),
                    "group_id": obj.get("group_id"),
                }
                for fld,expected in expected_schedule.items():
                    if fld in s and s[fld]!=expected:
                        row["issues"].append({"code":"SCHEDULE_BINDING","field":fld,"packet":obj.get(fld),"normalized_packet":expected,"schedule":s[fld]})
            # Feedback cards are only accepted on F arms and require matching bytes hash.
            if "feedback_card_path" in obj:
                fp=Path(obj["feedback_card_path"])
                if not fp.exists(): row["issues"].append({"code":"FEEDBACK_MISSING","path":str(fp)})
                elif obj.get("group_id") not in ("OLD_F","NEW_F"): row["issues"].append({"code":"FEEDBACK_BOUNDARY","message":"feedback card on non-F role"})
                elif "feedback_card_sha256" in obj and sha(fp.read_bytes())!=obj["feedback_card_sha256"]: row["issues"].append({"code":"FEEDBACK_HASH"})
            # Author outputs: strict JSON, candidate count/provenance/evidence, NEW slot bytes freeze.
            ad=ip.parent/"author_output"; turnmap={};
            if src.exists():
                try: sd,_=read_json(src); turnmap,_=source_turn_map(sd)
                except Exception: pass
            candidates=sorted(ad.glob("CANDIDATES_FIRST_DRAFT.json")) if ad.exists() else []
            if candidates:
                try:
                    co,_,_=json_hash(candidates[0]); row["candidate_check"]=candidate_checks(co,turnmap)
                    if not row["candidate_check"]["pass"]: row["issues"].append({"code":"CANDIDATE_STRUCTURAL_FAILURE","detail":row["candidate_check"]})
                except Exception as e: row["issues"].append({"code":"CANDIDATE_JSON_INVALID","message":str(e)})
            if obj.get("workflow")=="NEW":
                slots=ad/"NEW_SLOTS.json"; freeze=ad/"SLOTS_FREEZE.json"
                row["new_slot_files"]={"slots_exists":slots.exists(),"freeze_exists":freeze.exists()}
                if slots.exists() and freeze.exists():
                    try:
                        sv,sraw,sh=json_hash(slots); fv,_,_=json_hash(freeze)
                        declared=fv.get("slots_sha256",fv.get("slots_file_sha256",fv.get("new_slots_sha256"))) if isinstance(fv,dict) else None
                        if declared!=sh: row["issues"].append({"code":"SLOT_BYTES_HASH","declared":declared,"actual":sh})
                        if isinstance(fv,dict) and "slots_bytes" in fv and fv["slots_bytes"]!=len(sraw): row["issues"].append({"code":"SLOT_BYTES_LENGTH","declared":fv["slots_bytes"],"actual":len(sraw)})
                    except Exception as e: row["issues"].append({"code":"SLOT_JSON_INVALID","message":str(e)})
                else: row["issues"].append({"code":"NEW_SLOT_ARTIFACT_MISSING","message":"NEW requires NEW_SLOTS.json and SLOTS_FREEZE.json before selector"})
            row["pass"]=not row["issues"]; packet_rows.append(row)
        # Any role packet directory not listed in manifest is an unauthorized surface.
        listed={p.get("opportunity_id") for p in packets if isinstance(p,dict)}
        extras=[rel(p.parent) for p in all_inputs if p.parent.name not in listed]
        if extras: self.issue("UNREGISTERED_ROLE_PACKET","role INPUT not in PACKET_MANIFEST",paths=extras)
        # NEW selector separation gate. A selector packet is a dedicated INPUT under
        # selector_packets; old SELECTION files are outputs and are never accepted as packets.
        new_opps=[p for p in packets if isinstance(p,dict) and p.get("workflow")=="NEW"]
        selector_rows=[]
        selector_packet_dir=self.base/"selector_packets"
        for p in new_opps:
            opp=p.get("opportunity_id"); sp=selector_packet_dir/opp/"INPUT.json"
            sr={"opportunity_id":opp,"selector_packet_path":rel(sp),"selector_packet_exists":sp.exists(),"issues":[]}
            if not sp.exists(): sr["issues"].append({"code":"SELECTOR_PACKET_MISSING","message":"NEW opportunity has no dedicated selector packet"})
            else:
                try:
                    so,_,_=json_hash(sp)
                    if not isinstance(so,dict): sr["issues"].append({"code":"SELECTOR_PACKET_TYPE"})
                    else:
                        allowed={"slot_ids","source","neutral","slots","history_projection"}
                        extra=sorted(set(so)-allowed)
                        if extra: sr["issues"].append({"code":"SELECTOR_PACKET_EXTRA_FIELDS","fields":extra})
                        hits=scan_keys_values(so,SELECTOR_FORBIDDEN)
                        if hits: sr["issues"].append({"code":"SELECTOR_PACKET_IDENTITY_LEAK","hits":hits[:20]})
                except Exception as e: sr["issues"].append({"code":"SELECTOR_PACKET_JSON_INVALID","message":str(e)})
            sr["pass"]=not sr["issues"]; selector_rows.append(sr)
        # Existing selector returns are diagnostic only; they must be anonymous and single-field.
        for p in sorted((self.base/"selector").glob("*/SELECTION.json")):
            sr={"path":rel(p),"issues":[]}
            try:
                so,_,_=json_hash(p)
                if not isinstance(so,dict): sr["issues"].append({"code":"SELECTOR_OUTPUT_TYPE"})
                else:
                    extra=sorted(set(so)-{"selected_slot_id"})
                    if extra: sr["issues"].append({"code":"SELECTOR_OUTPUT_EXTRA_FIELDS","fields":extra})
                    hits=scan_keys_values(so,SELECTOR_FORBIDDEN)
                    if hits: sr["issues"].append({"code":"SELECTOR_OUTPUT_IDENTITY_LEAK","hits":hits[:20]})
                    if "selected_slot_id" not in so: sr["issues"].append({"code":"SELECTOR_OUTPUT_MISSING_SLOT"})
            except Exception as e: sr["issues"].append({"code":"SELECTOR_OUTPUT_JSON_INVALID","message":str(e)})
            sr["pass"]=not sr["issues"]; selector_rows.append(sr)
        # Gate downstream candidate outputs: do not classify as quality pass here.
        self.rows=packet_rows
        self.selector_rows=selector_rows
        self.counts={"role_packets":len(packet_rows),"role_packets_pass":sum(r.get("pass",False) for r in packet_rows),"new_opportunities":len(new_opps),"selector_packet_rows":len(selector_rows),"selector_packet_pass":sum(r.get("pass",False) for r in selector_rows)}
        self.overall=not self.issues and all(r.get("pass") for r in packet_rows) and all(r.get("pass") for r in selector_rows)
        return self
    def report(self):
        return {
            "schema_version":"cc_isolated_role_selector_preflight_diagnostic_v1",
            "checked_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z"),
            "execution_directory":str(self.base),
            "read_only":True,
            "exclusive_create_output":True,
            "no_author_solver_training_export_started_by_checker":True,
            "checks":{
                "authorization": {
                    "scope":self.auth.get("scope",{}),
                    "generation_authorization_packets":"NONE",
                    "artifact_states":{
                        "author_launch":self.auth.get("generation_authorization"),
                        "packet_manifest":self.manifest.get("generation_authorization"),
                        "schedule_restricted":self.schedule.get("generation_authorization"),
                        "access_rules":self.access.get("current_generation_authorization"),
                    },
                },
                "role_packet_rows":self.rows,
                "selector_rows":self.selector_rows,
                "global_issues":self.issues,
            },
            "counts":self.counts,
            "all_pass":self.overall,
            "decision":"HOLD_AND_QUARANTINE" if not self.overall else "PREPARED_FOR_MANUAL_REVIEW",
            "limitations":[
                "This mechanical checker does not grant generation authorization.",
                "Selector output files from the prior run are audited as artifacts; they cannot substitute for a dedicated anonymous selector packet.",
                "Quality, structure-overlap, target-fidelity, Solver, training, and export remain out of scope."
            ]
        }

def atomic_exclusive_write(path:Path, data:bytes):
    # O_EXCL means reruns cannot overwrite diagnostic evidence.
    fd=os.open(str(path), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    try:
        os.write(fd,data)
    finally: os.close(fd)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base",type=Path,default=ROOT); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); args=ap.parse_args()
    base=args.base.resolve(); out=args.out.resolve()
    if out==base or not under(out,base): raise SystemExit("output must be a new directory under execution")
    # Script is normally placed inside the already-created diagnostic directory. Every
    # report artifact itself remains exclusive-create.
    out.mkdir(mode=0o700, parents=False, exist_ok=True)
    report=Checker(base).run().report()
    # Include this script's own bytes hash for reproducibility without modifying source.
    report["checker_sha256"]=sha(Path(__file__).read_bytes())
    raw=json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8")+b"\n"
    atomic_exclusive_write(out/"PREFLIGHT_DIAGNOSTIC_20261107_v1.json",raw)
    # Human-readable gate is also exclusive-created.
    lines=["# Isolated role/selector mechanical preflight", "", f"- all_pass: `{report['all_pass']}`", f"- decision: `{report['decision']}`", f"- role packets: {report['counts']['role_packets_pass']}/{report['counts']['role_packets']}", f"- NEW selector packets: {report['counts']['selector_packet_pass']}/{report['counts']['selector_packet_rows']}", "", "This report is mechanical only; no author, selector, Solver, training, or export was started by the checker.", ""]
    atomic_exclusive_write(out/"PREFLIGHT_DIAGNOSTIC_20261107_v1.md",("\n".join(lines)).encode())
    print(json.dumps({"report":str(out/"PREFLIGHT_DIAGNOSTIC_20261107_v1.json"),"all_pass":report["all_pass"],"decision":report["decision"],"counts":report["counts"]},ensure_ascii=False))
    return 0 if report["all_pass"] else 2

if __name__=="__main__":
    try: raise SystemExit(main())
    except FileExistsError as e:
        print(f"exclusive-create refusal: {e}",file=sys.stderr); raise SystemExit(3)
