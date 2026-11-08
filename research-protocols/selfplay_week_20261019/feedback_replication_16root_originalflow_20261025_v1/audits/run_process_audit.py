from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

D = Path(__file__).resolve().parents[1]
issues = []
checks = Counter()
binding_results = {}
historical_failed_receipts = []


def load(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def objhash(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def problem(kind, location, detail):
    issues.append({"kind": kind, "location": str(location), "detail": detail})


def binding(b, where):
    if not isinstance(b, dict) or not all(k in b for k in ("path", "sha256")):
        return
    key = (b["path"], b["sha256"], b.get("bytes"))
    if key in binding_results:
        return
    p = Path(b["path"])
    # Never read Solver outputs or scores in this pre-Solver process audit.
    if any(part.lower() in {"solver", "solvers", "solver_raw", "solver_outputs", "raw", "scores"} for part in p.parts):
        problem("excluded_binding", where, str(p))
        return
    ok = p.is_file() and digest(p) == b["sha256"] and ("bytes" not in b or p.stat().st_size == b["bytes"])
    binding_results[key] = ok
    if not ok:
        problem("binding_mismatch", where, b)


def all_bindings(o, where):
    if isinstance(o, dict):
        binding(o, where)
        for k, v in o.items():
            all_bindings(v, f"{where}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            all_bindings(v, f"{where}[{i}]")


def quotes(o, turns, where):
    if isinstance(o, dict):
        if isinstance(o.get("quote"), str) and isinstance(o.get("turn_ref"), str):
            checks["exact_quotes_checked"] += 1
            t = turns.get(o["turn_ref"])
            if t is None or o["quote"] not in t["text"]:
                problem("non_exact_quote", where, {"turn_ref": o["turn_ref"], "quote": o["quote"]})
            if t is not None and "speaker" in o and o["speaker"] != t["speaker"]:
                problem("quote_speaker_mismatch", where, o)
        for k, v in o.items():
            quotes(v, turns, f"{where}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            quotes(v, turns, f"{where}[{i}]")


def key(s):
    return s["root_id"], s["arm"], s["slot_id"]


def gold_equal(candidate, answer):
    if candidate["question_type"] == "multi_select":
        return set(candidate["gold"]) == set(answer)
    return candidate["gold"] == answer


selection = load(D / "SOURCE_SELECTION.json")
sf = load(D / "SOURCE_FREEZE.json")
reused = load(D / "REUSED_SLOTS.json")["slots"]
qf = load(D / "QUESTION_FREEZE.json")
sources = {}
source_bindings = {}
for b in selection["roots"]:
    binding(b, "SOURCE_SELECTION")
    root = b["root_id"]
    sources[root] = load(b["path"])
    source_bindings[root] = {k: b[k] for k in ("path", "sha256", "bytes")}
all_bindings(sf, "SOURCE_FREEZE")
all_bindings(qf["bindings"], "QUESTION_FREEZE.bindings")

slots = qf["slots"]
all_roots = {o["root_id"] for o in sf["outcomes"]}
slot_ids = {"MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3"}
expected = {(r, a, s) for r in all_roots for a in ("F", "G") for s in slot_ids}
actual = [key(s) for s in slots]
if len(set(actual)) != len(actual) or set(actual) != expected:
    problem("slot_universe_mismatch", "QUESTION_FREEZE", {"duplicates": len(actual)-len(set(actual)), "missing": list(expected-set(actual)), "extra": list(set(actual)-expected)})

passes = {key(s): s for s in slots if s["status"] == "PASS"}
admissions = defaultdict(list)
packet_details = []
identity_details = []
metadata_limits = []
for folder in sorted((D / "reviews").iterdir()):
    ap = folder / "ADMISSION.json"
    if not ap.exists():
        continue
    a = load(ap)
    public = load(folder / "PUBLIC_PACKET.json")
    mapping = load(folder / "PRIVATE_MAPPING.json")
    gold = load(folder / "INDEPENDENT_GOLD.json")
    content = load(folder / "CONTENT_REVIEW.json")
    root = public["root_id"]
    src = public["full_source"]
    turns = {t["turn_ref"]: t for t in src["turns"]}
    if src != sources[root]:
        problem("public_source_differs_from_selected_full_source", folder, root)
    checks["public_source_objects_checked"] += 1
    if "source_object_sha256" in gold and gold["source_object_sha256"] != objhash(src):
        problem("gold_source_object_hash_mismatch", folder, root)
    if "source_turns_read" in gold and gold["source_turns_read"] != len(turns):
        problem("gold_source_turn_count_mismatch", folder, root)
    if "source_object_sha256" not in gold or "source_turns_read" not in gold:
        metadata_limits.append({"packet": folder.name, "missing_gold_full_source_metadata": [k for k in ("source_object_sha256", "source_turns_read") if k not in gold]})
    fsr = content.get("full_source_read")
    if isinstance(fsr, dict) and (fsr.get("turn_count") != len(turns) or fsr.get("turn_refs") != list(turns)):
        problem("content_source_read_receipt_mismatch", folder, root)
    all_bindings(content, f"{folder.name}.content")
    pm = {x["review_id"]: x for x in public["questions"]}
    mm = {x["review_id"]: x["candidate"] for x in mapping["entries"]}
    gm = {x["review_id"]: x for x in gold["slots"]}
    cm = {x["review_id"]: x for x in content["slots"]}
    if not (set(pm) == set(mm) == set(gm) == set(cm)):
        problem("packet_review_id_mismatch", folder, [list(x) for x in (pm, mm, gm, cm)])
    for rid, c in mm.items():
        for field in ("question_type", "task_operation", "question", "options", "items"):
            if field in c and pm[rid].get(field) != c[field]:
                problem("public_private_question_mismatch", f"{folder.name}.{rid}", field)
        if c.get("source_binding") != source_bindings[root]:
            problem("candidate_source_binding_mismatch", f"{folder.name}.{rid}", c.get("source_binding"))
        quotes(c, turns, f"{folder.name}.{rid}.author")
        quotes(gm[rid], turns, f"{folder.name}.{rid}.gold")
        quotes(cm[rid], turns, f"{folder.name}.{rid}.content")
        checks["current_candidates_checked"] += 1
    for s in a["slots"]:
        admissions[key(s)].append((folder.name, s))
        match = [rid for rid, c in mm.items() if c["candidate_id"] == s["candidate_id"]]
        if len(match) != 1 or s["candidate"] != mm[match[0]]:
            problem("admission_candidate_mismatch", folder, s["candidate_id"])
            continue
        rid = match[0]
        c = s["candidate"]
        expected_gold = dict(gm[rid])
        expected_content = dict(cm[rid])
        expected_gold.setdefault("slot_id", c["slot_id"])
        expected_content.setdefault("slot_id", c["slot_id"])
        if s["review"].get("gold") != expected_gold or s["review"].get("review") != expected_content:
            problem("admission_review_object_mismatch", f"{folder.name}.{rid}", s["candidate_id"])
        if s["status"] == "PASS":
            if gm[rid]["decision"] != "PASS" or cm[rid]["decision"] != "PASS" or not gold_equal(c, gm[rid]["gold"]):
                problem("admission_unapproved_or_gold_disagrees", f"{folder.name}.{rid}", s["candidate_id"])
            independent_gold = cm[rid].get("independently_derived_gold", cm[rid].get("independent_gold"))
            if independent_gold is not None and not gold_equal(c, independent_gold):
                problem("content_independent_gold_disagrees", f"{folder.name}.{rid}", independent_gold)
            final = passes.get(key(s))
            if final is None or final["candidate"] != c:
                problem("admitted_pass_missing_or_different_at_freeze", folder, key(s))
            elif final["review"] != s["review"]:
                problem("frozen_review_differs_from_admission", folder, key(s))
            actual_author = "/root" if folder.name == "recovery_179186" else ("/root/g_author" if folder.name == "recovery_246750_G" else c["author_id"])
            actual_gold = "/root/stall_review" if str(gold["reviewer_id"]).startswith("/root/stall_review") else gold["reviewer_id"]
            actual_content = "/root/artifact_audit" if folder.name == "recovery_179186" else "/root"
            if len({actual_author, actual_gold, actual_content}) != 3:
                problem("actual_agent_not_distinct", folder, [actual_author, actual_gold, actual_content])
            identity_details.append({"packet": folder.name, "candidate_id": c["candidate_id"], "slot": key(s), "declared_author": c["author_id"], "actual_author": actual_author, "declared_gold": gold["reviewer_id"], "actual_gold": actual_gold, "declared_content": content["reviewer_id"], "actual_content": actual_content, "basis": "Parent agent clarified actual aliases at audit time; recovery179186 author/content receipts independently inspected. Historical symbolic authors here differ from both current reviewer agents."})
    packet_details.append({"packet": folder.name, "source_turns": len(turns), "questions": len(mm), "admission_counts": dict(Counter(s["status"] for s in a["slots"]))})

receipt_cache = {}


def receipt(p):
    p = Path(p)
    if str(p) not in receipt_cache:
        r = load(p)
        receipt_cache[str(p)] = r
        if not r.get("accepted") or r.get("returncode") != 0 or r.get("timed_out"):
            historical_failed_receipts.append({"path": str(p), "status": {k: r.get(k) for k in ("accepted", "returncode", "timed_out")}, "failure_reasons": r.get("failure_reasons")})
        for name, b in r.get("hashes", {}).items():
            # Reviewer/author protocol artifacts only; receipts are never Solver calls.
            binding(b, f"{p}.{name}")
    return receipt_cache[str(p)]


historical_identity = []
for s in reused:
    c = s["candidate"]
    root, arm, sid = key(s)
    turns = {t["turn_ref"]: t for t in sources[root]["turns"]}
    quotes(c, turns, f"reused.{c['candidate_id']}.author")
    quotes(s["review"]["gold"], turns, f"reused.{c['candidate_id']}.gold")
    quotes(s["review"]["review"], turns, f"reused.{c['candidate_id']}.content")
    if c["source_binding"] != source_bindings[root]:
        problem("reused_source_binding_mismatch", c["candidate_id"], c["source_binding"])
    if passes.get(key(s), {}).get("candidate") != c:
        problem("reused_candidate_changed_or_missing", c["candidate_id"], key(s))
    elif passes[key(s)]["review"] != s["review"]:
        problem("frozen_reused_review_differs", c["candidate_id"], key(s))
    gr = receipt(s["review"]["gold_receipt"])
    cr = receipt(s["review"]["review_receipt"])
    exp = Path(s["review"]["gold_receipt"]).parents[2]
    # Select the author call whose returned candidate exactly matches the reused candidate.
    authors = []
    for stage in ("question_draft", "question_revision_1"):
        p = exp / "cli_calls" / f"{root}_{arm}_{stage}" / "RECEIPT.json"
        if not p.exists():
            continue
        ar = receipt(p)
        response = ar.get("response", {})
        candidates = response.get("candidates", []) if isinstance(response, dict) else []
        if any(x == c for x in candidates):
            authors.append((p, ar))
    if len(authors) != 1:
        problem("historical_author_candidate_receipt_unresolved", c["candidate_id"], [str(p) for p, _ in authors])
        author_id = None
    else:
        author_id = authors[0][1].get("backend", {}).get("thread_id")
    gold_id = gr.get("backend", {}).get("thread_id")
    content_id = cr.get("backend", {}).get("thread_id")
    used = [(s["review"]["gold_receipt"], gr), (s["review"]["review_receipt"], cr)] + authors
    for p, r in used:
        if not r.get("accepted") or r.get("returncode") != 0 or r.get("timed_out"):
            problem("used_historical_receipt_not_successful", p, c["candidate_id"])
    if not all((author_id, gold_id, content_id)) or len({author_id, gold_id, content_id}) != 3:
        problem("historical_actual_session_independence_unresolved", c["candidate_id"], [author_id, gold_id, content_id])
    for r, field in ((gr, "gold"), (cr, "review")):
        got = [x for x in r["response"]["slots"] if x["slot_id"] == sid]
        if len(got) != 1 or got[0] != s["review"][field]:
            problem("historical_receipt_slot_differs", c["candidate_id"], field)
    historical_identity.append({"candidate_id": c["candidate_id"], "slot": key(s), "author_thread_id": author_id, "gold_thread_id": gold_id, "content_thread_id": content_id, "author_receipt": str(authors[0][0]) if len(authors) == 1 else None, "gold_receipt": s["review"]["gold_receipt"], "content_receipt": s["review"]["review_receipt"]})

# Current independent rechecks of old authored questions retain their original author session.
for identity in identity_details:
    if not identity["packet"].startswith("existing_"):
        continue
    c = passes[tuple(identity["slot"])]["candidate"]
    exp = Path(c["source_binding"]["path"]).parents[2]
    authors = []
    for stage in ("question_draft", "question_revision_1"):
        p = exp / "cli_calls" / f"{c['root_id']}_{c['arm']}_{stage}" / "RECEIPT.json"
        if not p.exists():
            continue
        ar = receipt(p)
        if any(x == c for x in ar.get("response", {}).get("candidates", [])):
            authors.append((p, ar))
    if len(authors) != 1:
        problem("current_existing_original_author_unresolved", c["candidate_id"], len(authors))
    else:
        identity["actual_author"] = "CLI-session:" + authors[0][1]["backend"]["thread_id"]
        identity["author_receipt"] = str(authors[0][0])
        identity["basis"] = "Candidate exactly matches this original author CLI receipt. Gold is current /root/stall_review and content is actual /root; symbolic author label is not used as identity proof."

plan_details = []
for folder in sorted((D / "questions").glob("*/*")):
    root, arm = folder.parent.name, folder.name
    if not folder.is_dir():
        continue
    fp = next(iter(sorted(folder.glob("TASK_PLAN*FREEZE.json"))), None)
    ap = folder / "AUTHOR_RECEIPT.json"
    if fp is None:
        continue
    f = load(fp)
    all_bindings(f, str(fp))
    a = load(ap)
    all_bindings(a, str(ap))
    refs = a.get("all_turn_refs_read", a.get("full_turn_refs", a.get("full_source_read_receipt", {}).get("turn_refs")))
    expected_refs = [t["turn_ref"] for t in sources[root]["turns"]]
    if refs != expected_refs:
        problem("author_complete_source_refs_mismatch", ap, {"actual_len": len(refs) if refs else None, "expected_len": len(expected_refs)})
    before = f.get("frozen_at_utc", f.get("created_at", f.get("at")))
    after = a.get("completed_at_utc", a.get("at"))
    if before and after and datetime.fromisoformat(before.replace("Z", "+00:00")) > datetime.fromisoformat(after.replace("Z", "+00:00")):
        problem("plan_frozen_after_author_receipt", fp, {"freeze": before, "author": after})
    for p in folder.glob("*REVISION*RECEIPT.json"):
        all_bindings(load(p), str(p))
    plan_details.append({"root_id": root, "arm": arm, "plan_freeze": str(fp), "plan_freeze_at": before, "author_receipt_at": after, "full_source_turn_refs_match": refs == expected_refs})

reused_keys = {key(s) for s in reused}
admitted_pass_keys = {k for k, records in admissions.items() for _, s in records if s["status"] == "PASS"}
if reused_keys & admitted_pass_keys:
    problem("reused_and_new_pass_duplicate", "QUESTION_FREEZE", list(reused_keys & admitted_pass_keys))
if reused_keys | admitted_pass_keys != set(passes):
    problem("final_pass_origins_do_not_cover", "QUESTION_FREEZE", list(set(passes) ^ (reused_keys | admitted_pass_keys)))
for k, records in admissions.items():
    if sum(s["status"] == "PASS" for _, s in records) > 1:
        problem("multiple_final_pass_admissions", "QUESTION_FREEZE", {"slot": k, "packets": [p for p, _ in records]})

revision_preservation = []
for root, arm in (("cc-train-episode-179186", "G"), ("cc-train-episode-68549", "F")):
    folder = D / "questions" / root / arm
    files = [folder / name for name in ("CANDIDATES_DRAFT.json", "CANDIDATES_REVISION.json", "REVISION_RECEIPT.json", "SLOT_FAILURES.json")]
    if not all(p.is_file() for p in files):
        problem("revision_history_artifact_missing", folder, [str(p) for p in files if not p.is_file()])
    revision_preservation.append({"root_id": root, "arm": arm, "artifacts": [{"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size} for p in files if p.is_file()]})
mc4_key = ("cc-train-episode-68549", "F", "MC4")
mc4_records = admissions[mc4_key]
if [(name, s["status"]) for name, s in mc4_records] != [("new_68549_F", "FAIL"), ("revision_68549_F_MC4", "PASS")]:
    problem("mc4_revision_failure_history_not_preserved", "admissions", [(name, s["status"]) for name, s in mc4_records])
for s in reused:
    if not all(k in s for k in ("initial_candidate", "initial_audit", "revision_used", "candidate", "review")):
        problem("reused_initial_history_missing", s["candidate_id"], list(s))

preservation = {
    "reused_initial_candidate_and_audit_preserved": all(all(k in s for k in ("initial_candidate", "initial_audit", "revision_used", "candidate", "review")) for s in reused),
    "reused_revision_used_count": sum(bool(s["revision_used"]) for s in reused),
    "reused_initial_audit_statuses": dict(Counter(s["initial_audit"].get("status") for s in reused)),
    "current_revision_artifacts": revision_preservation,
    "mc4_initial_fail_and_revised_pass_admissions_retained": [(name, s["status"]) for name, s in mc4_records],
    "source_failed_root": [o["root_id"] for o in sf["outcomes"] if o["status"] != "PASS"],
    "final_failure_slots_with_attempt_history": sum(s["status"] != "PASS" and bool(s.get("attempt_history")) for s in slots),
    "note": "PASS slots need not duplicate attempt histories inside the final slot: REUSED_SLOTS and original/revision admission files are preserved and bound by QUESTION_FREEZE. Same MC4 candidate_id is retained across revision; exact candidate objects and immutable artifact hashes disambiguate versions.",
}

summary = {
    "schema_version": "independent_process_audit_v1",
    "auditor_id": "/root/stall_review",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Static pre-Solver process audit. No new Solver raw outputs or scores read. No question, review, admission, source or plan file modified.",
    "question_freeze_binding": {"path": str(D / "QUESTION_FREEZE.json"), "sha256": digest(D / "QUESTION_FREEZE.json"), "bytes": (D / "QUESTION_FREEZE.json").stat().st_size},
    "counts": {"planned_slots": len(expected), "frozen_slots": len(slots), "frozen_unique_slots": len(set(actual)), "frozen_status": dict(Counter(s["status"] for s in slots)), "reused_pass_slots": len(reused), "new_admitted_pass_slots": len(admitted_pass_keys), "source_pass_roots": len(sources), "source_fail_roots": len(all_roots)-len(sources), "current_admission_packets": len(packet_details), "source_objects_checked": checks["public_source_objects_checked"], "exact_quotes_checked": checks["exact_quotes_checked"], "bindings_checked": len(binding_results), "bindings_matched": sum(binding_results.values()), "historical_protocol_receipts_checked": len(receipt_cache)},
    "mechanical_findings": issues,
    "status": "PASS_WITH_LIMITATIONS" if not issues else "FINDINGS_REQUIRE_REVIEW",
    "failure_and_revision_preservation": preservation,
    "preserved_unused_failed_historical_receipts": historical_failed_receipts,
    "packets": packet_details,
    "new_pass_actual_agent_independence": identity_details,
    "reused_pass_actual_cli_session_independence": historical_identity,
    "current_plans": plan_details,
    "metadata_limitations": metadata_limits,
    "not_certified_by_this_audit": ["Static receipts and parent identity clarification support author/Gold/content separation; they do not establish model-provider independence or absence of unrecorded exposure.", "No new Solver raw or score inspected, so absence of reruns is not inferred from storage inspection. The freeze preserves all73 reused candidate identities and classifies their origin; execution scheduling requires a separate permitted provenance check.", "This audit checks exact quotations and admitted Gold consistency, and does not replace semantic question review.", "Freeze timestamps attest declared order relative to author completion receipts. A static filesystem snapshot cannot independently prove the instant of first candidate drafting."],
}
(D / "audits" / "PROCESS_AUDIT_WORKING.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
(D / "audits" / "INDEPENDENT_PROCESS_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"counts": summary["counts"], "issues_by_kind": dict(Counter(i["kind"] for i in issues)), "issues": issues[:30]}, ensure_ascii=False, indent=2))
