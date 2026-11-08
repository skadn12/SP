#!/usr/bin/env python3
"""Single-controller deterministic reliability baseline for the feedback loop.

The compiler uses only immutable source statements as positive claims and
explicit denial-of-a-present-statement templates as negatives.  This makes the
Gold key program-verifiable and prevents the failures seen when an LLM authored
options, citations, and answer keys in one response.
"""
from __future__ import annotations

import argparse
import shutil
import datetime as dt
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
PROTOCOL = json.loads((BASE / "PROTOCOL_LOCK.json").read_text())
SOURCE_FREEZE = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/feedback_replication_20root_resume_20261025_v1/SOURCE_FREEZE.json")
MODEL = "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model"
ROOTS = tuple(PROTOCOL["new16_roots"])
PILOT = tuple(ROOTS[:4])
OPS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3")
ARMS = ("F", "G")
LABELS = tuple("ABCDEF")
GOLD_N = {"MC2": 2, "MC3": 3, "MC4": 4, "MC5": 5, "MC6": 6}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bind(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw)}


def atomic(path: Path, value: Any) -> dict[str, Any]:
    path = path.resolve()
    if BASE not in path.parents and path != BASE:
        raise ValueError("write outside reliability baseline")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)
    return bind(path)


def status(phase: str, **updates: Any) -> None:
    lock = BASE / ".status.lock"
    with lock.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        previous = json.loads((BASE / "RUN_STATUS.json").read_text()) if (BASE / "RUN_STATUS.json").exists() else {}
        previous.update(updates)
        previous.update({"phase": phase, "updated_at_utc": now(), "training": False, "semantic_calls": 0})
        atomic(BASE / "RUN_STATUS.json", previous)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    rows = []
    start = 0
    for match in re.finditer(r"[.!?](?:[\"'”’)]*)?(?=\s|$)", text):
        end = match.end()
        part = text[start:end]
        left = len(part) - len(part.lstrip())
        right = len(part.rstrip())
        if right > left:
            rows.append((start + left, start + right, text[start + left:start + right]))
        start = end
    if start < len(text):
        part = text[start:]
        left = len(part) - len(part.lstrip())
        right = len(part.rstrip())
        if right > left:
            rows.append((start + left, start + right, text[start + left:start + right]))
    return rows


def category(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(?:correct|correction|replace|replaces|instead|changed|switch)\b", low):
        return "correction"
    if re.search(r"\b(?:book|booking|booked|confirm|confirmed|complete|completed|happened|sent|received|planned|plan|propose|proposal|schedule|scheduled|arranged|unconfirmed|not reported)\b", low):
        return "plan_completion"
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}:\d{2}|\b20\d\d\b)\b", low):
        return "time"
    if re.search(r"\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|half|twice)\b", low):
        return "quantity"
    if re.search(r"\b(?:because|therefore|so that|need|want|reason)\b", low):
        return "causal"
    return "neutral"


def source_rows(root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(SOURCE_FREEZE.read_text())
    row = next(x for x in freeze["outcomes"] if x["root_id"] == root)
    b = row["fixed_initial_draft"]
    path = Path(b["path"])
    raw = path.read_bytes()
    if sha(raw) != b["sha256"] or len(raw) != b["bytes"]:
        raise ValueError("initial draft binding drift: " + root)
    draft = json.loads(raw)
    return draft, b


def ledger(root: str) -> dict[str, Any]:
    draft, source_binding = source_rows(root)
    facts = []
    source_turns = []
    order = 0
    for scene in draft["scenes"]:
        for turn in scene["turns"]:
            order += 1
            ref, speaker, text = turn["turn_ref"], turn["speaker"], turn["text"]
            source_turns.append({"order": order, "turn_ref": ref, "speaker": speaker, "text": text})
            for n, (start, end, sentence) in enumerate(sentence_spans(text), 1):
                facts.append({"fact_id": f"{ref}:F{n:02d}", "turn_ref": ref, "speaker": speaker,
                              "order": order, "start": start, "end": end,
                              "text": sentence, "category": category(sentence)})
    if len(source_turns) != 48 or not facts:
        raise ValueError("expected 48 R03 source turns: " + root)
    return {"schema_version": "feedback_reliability_v4_fact_ledger", "root_id": root,
            "source_draft": source_binding, "turns": source_turns, "facts": facts,
            "fact_count": len(facts), "source_mutation": False}


def selected_facts(data: dict[str, Any], arm: str, operation: str) -> list[dict[str, Any]]:
    facts = data["facts"]
    focus = {
        "MC2": ("quantity", "time"), "MC3": ("plan_completion", "correction"),
        "MC4": ("time", "quantity"), "MC5": ("causal", "plan_completion"),
        "MC6": ("plan_completion", "correction"), "ORDER1": ("time",),
        "ORDER2": ("plan_completion", "causal"), "ORDER3": ("correction", "plan_completion"),
    }[operation]
    if arm == "F":
        pool = [f for f in facts if f["category"] in focus]
    else:
        pool = [f for f in facts if f["category"] == "neutral"]
        if len(pool) < 6:
            pool = [f for f in facts if f["category"] not in focus]
    if len(pool) < 6:
        pool = facts
    seed = int.from_bytes(hashlib.sha256(f"{data['root_id']}:{arm}:{operation}".encode()).digest()[:8], "big")
    return [pool[(seed + i * 7) % len(pool)] for i in range(6)]


def false_option(fact: dict[str, Any], index: int) -> str:
    # This claim is false by construction: an exact sentence occurs in source,
    # yet this option says the source expressly denies that same sentence.
    return f"The source explicitly denies this exact statement: \"{fact['text']}\""


def permutation(seed: str, canonical: list[str], ordering: bool) -> tuple[dict[str, str], dict[str, str]]:
    order = list(canonical)
    rnd = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    rnd.shuffle(order)
    if order == canonical:
        order = order[1:] + order[:1]
    mapping = dict(zip(LABELS, order))
    inverse = {v: k for k, v in mapping.items()}
    if ordering and [inverse[x] for x in canonical] == list(LABELS):
        order = order[1:] + order[:1]
        mapping = dict(zip(LABELS, order)); inverse = {v: k for k, v in mapping.items()}
    return mapping, inverse


def compile_slot(data: dict[str, Any], arm: str, operation: str) -> dict[str, Any]:
    facts = selected_facts(data, arm, operation)
    root = data["root_id"]
    cid = f"REL4-{arm}-{root.split('-')[-1]}-{operation}"
    options: dict[str, dict[str, Any]] = {}
    if operation in GOLD_N:
        true_count = GOLD_N[operation]
        for i, fact in enumerate(facts):
            label = LABELS[i]
            truth = i < true_count
            options[label] = {"claim": fact["text"] if truth else false_option(fact, i), "truth": truth,
                              "fact_id": fact["fact_id"], "turn_ref": fact["turn_ref"],
                              "source_span": fact["text"], "category": fact["category"]}
        gold = list(LABELS[:true_count])
        qtype = "multi_select"
        question = f"For {operation}, select every claim explicitly supported by the dialogue source."
    else:
        ranked = sorted(facts, key=lambda f: (f["order"], f["start"], f["fact_id"]))
        canonical = list(LABELS)
        for label, fact in zip(canonical, ranked):
            options[label] = {"claim": fact["text"], "truth": True, "fact_id": fact["fact_id"],
                              "turn_ref": fact["turn_ref"], "source_span": fact["text"], "category": fact["category"]}
        gold = canonical
        qtype = "ordering"
        question = f"For {operation}, order the six statements by their appearance in the dialogue source."
    variants = []
    for v in range(2):
        d2c, c2d = permutation(f"{cid}:v{v}", list(LABELS), qtype == "ordering")
        display = {label: options[d2c[label]] for label in LABELS}
        display_gold = sorted(c2d[x] for x in gold) if qtype == "multi_select" else [c2d[x] for x in gold]
        variants.append({"variant_id": f"perm_{v:02d}", "display": display, "display_to_canonical": d2c,
                         "gold": display_gold, "nonidentity": list(d2c.values()) != list(LABELS)})
    slot = {"candidate_id": cid, "root_id": root, "arm": arm, "slot_id": operation,
            "question_type": qtype, "question": question, "options": options, "gold": gold,
            "source_binding": data["source_draft"], "feedback_category": [f["category"] for f in facts],
            "variants": variants}
    validate_slot(slot, data)
    return slot


def validate_slot(slot: dict[str, Any], data: dict[str, Any]) -> None:
    facts = {f["fact_id"]: f for f in data["facts"]}
    if list(slot["options"]) != list(LABELS) or len(slot["variants"]) != 2:
        raise ValueError("slot option or display cardinality")
    if slot["question_type"] == "multi_select":
        expected = GOLD_N[slot["slot_id"]]
        actual = [k for k in LABELS if slot["options"][k]["truth"]]
        if actual != slot["gold"] or len(actual) != expected:
            raise ValueError("gold quota")
    else:
        correct = sorted(LABELS, key=lambda x: (facts[slot["options"][x]["fact_id"]]["order"], facts[slot["options"][x]["fact_id"]]["start"]))
        if correct != slot["gold"]:
            raise ValueError("ordering gold")
    source_all = "\n".join(t["text"] for t in data["turns"])
    for o in slot["options"].values():
        fact = facts[o["fact_id"]]
        if o["source_span"] != fact["text"] or o["source_span"] not in source_all:
            raise ValueError("span evidence")
        if o["truth"] and o["claim"] != fact["text"]:
            raise ValueError("true claim must be literal source")
        if not o["truth"] and (not o["claim"].startswith("The source explicitly denies") or o["claim"] in source_all):
            raise ValueError("false mutation not provably absent")
    for v in slot["variants"]:
        if not v["nonidentity"] or set(v["display"]) != set(LABELS):
            raise ValueError("display permutation")


def compile_roots(roots: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ledgers, feedback, slots = [], [], []
    for root in roots:
        data = ledger(root); ledgers.append(data)
        selections = {arm: {op: [f["fact_id"] for f in selected_facts(data, arm, op)] for op in OPS} for arm in ARMS}
        feedback.append({"root_id": root, "F_policy": "feedback-relevant categories", "G_policy": "neutral/matched categories", "selections": selections})
        for arm in ARMS:
            for op in OPS:
                slots.append(compile_slot(data, arm, op))
    return ledgers, feedback, slots


def source_context(data: dict[str, Any]) -> str:
    return "\n".join(f"{t['turn_ref']} {t['speaker']}: {t['text']}" for t in data["turns"])


def requests_for(slots: list[dict[str, Any]], ledgers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger_map = {x["root_id"]: x for x in ledgers}
    rows = []
    for slot in slots:
        context = source_context(ledger_map[slot["root_id"]])
        for v in slot["variants"]:
            choices = "\n".join(f"{k}. {x['claim']}" for k, x in v["display"].items())
            instruction = "Return your answer labels only inside <box>...</box>."
            prompt = f"Dialogue source:\n{context}\n\nQuestion:\n{slot['question']}\n{choices}\n{instruction}"
            seed = int.from_bytes(hashlib.sha256(f"{slot['candidate_id']}:{v['variant_id']}".encode()).digest()[:4], "big") % 2147483647
            rows.append({"request_id": f"{slot['candidate_id']}__{v['variant_id']}__FULL__THINK_OFF__REL4", "root_id": slot["root_id"],
                         "arm": slot["arm"], "slot_id": slot["slot_id"], "question_type": slot["question_type"],
                         "gold": v["gold"], "prompt": prompt, "sampling_seed": seed})
    return rows


def run_b1(requests: list[dict[str, Any]], label: str) -> dict[str, Any]:
    exe = BASE / "b1_v4_worker.py"
    req = BASE / "execution" / label / "requests.jsonl"; out = BASE / "execution" / label / "RAW_RESPONSE_FREEZE.json"
    req.parent.mkdir(parents=True, exist_ok=True); req.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in requests) + "\n")
    # The first pilot initialization used the merged checkpoint directory as
    # its tokenizer.  That directory's tokenizer config is malformed under the
    # installed Transformers version, before any request was submitted.  Keep
    # that zero-output receipt and recover only initialization with the frozen
    # training tokenizer (Qwen3-4B) used by the established B1 runner.
    if out.exists():
        previous = json.loads(out.read_text())
        if previous.get("returned_count") == 0 and "AttributeError" in str(previous.get("backend", {}).get("error", "")):
            archive = out.parent / "INITIALIZATION_FAILURE_MERGED_TOKENIZER.json"
            if not archive.exists():
                shutil.copy2(out, archive)
            atomic(out.parent / "INITIALIZATION_RECOVERY.json", {
                "at_utc": now(), "reason": "Merged checkpoint tokenizer config incompatible with installed Transformers before any request submission.",
                "submitted_requests_before_recovery": 0,
                "returned_responses_before_recovery": 0,
                "archived_failure": bind(archive),
                "recovery_tokenizer": "/root/autodl-tmp/Qwen3-4B",
                "sampling_parameters_unchanged": True,
                "training": False,
            })
        elif previous.get("returned_count", 0) > 0:
            return previous
    proc = subprocess.run([sys.executable, str(exe), "--requests", str(req), "--out", str(out)], cwd=BASE, timeout=14400)
    data = json.loads(out.read_text()) if out.exists() else {"prepared_count": len(requests), "returned_count": 0, "records": [], "exit_code": proc.returncode}
    data["prepared_count"] = len(requests); data["returned_count"] = len(data.get("records", [])); data["missing_count"] = len(requests)-data["returned_count"]
    atomic(out, data); return data


def score(raw: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    gold = {r["request_id"]: r for r in requests}; records=[]
    for row in raw.get("records", []):
        req = gold[row["request_id"]]; boxes = re.findall(r"<box>\s*([^<]+?)\s*</box>", row.get("text", ""), flags=re.I)
        pred = boxes[-1].strip().upper().split() if boxes else []
        valid = bool(pred) and all(x in LABELS for x in pred) and len(set(pred)) == len(pred)
        exact = valid and (sorted(pred) == sorted(req["gold"]) if req["question_type"] == "multi_select" else pred == req["gold"])
        records.append({"request_id": req["request_id"], "root_id": req["root_id"], "arm": req["arm"], "slot_id": req["slot_id"], "returned": True, "contract_valid": valid, "exact": exact, "prediction": pred})
    return {"records": records, "returned_count": len(records), "contract_valid": sum(x["contract_valid"] for x in records), "exact": sum(x["exact"] for x in records), "missing_count": len(requests)-len(records)}


def report(phase: str, roots: list[str], ledgers: list[dict[str, Any]], slots: list[dict[str, Any]], raw: dict[str, Any], scores: dict[str, Any]) -> None:
    atomic(BASE / "ANALYSIS.json", {"phase": phase, "roots": roots, "planned_slots": len(roots)*16, "valid_slots": len(slots), "planned_presentations": len(roots)*32, "returned_presentations": raw["returned_count"], "missing_presentations": raw["missing_count"], "contract_valid": scores["contract_valid"], "exact": scores["exact"], "feedback_benefit_claim": False, "training": False})
    text = f"# Reliability v4 {phase}\n\nroots: {len(roots)}\nslots: {len(slots)}/{len(roots)*16}\npresentations: {raw['returned_count']}/{len(roots)*32}\nvalid contract: {scores['contract_valid']}\nexact: {scores['exact']}\n"
    (BASE / f"RESULTS_{phase}_ZH.md").write_text(text)


def check_inputs() -> None:
    rows=[]
    for b in json.loads((BASE/"INPUT_BINDINGS.json").read_text())["bindings"]:
        raw=Path(b["path"]).read_bytes(); ok=sha(raw)==b["sha256"] and len(raw)==b["bytes"]; rows.append({**b,"ok":ok})
    atomic(BASE / "INPUT_INTEGRITY.json", {"checks": rows, "all_ok": all(x["ok"] for x in rows)})
    if not all(x["ok"] for x in rows): raise RuntimeError("input integrity")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); args=parser.parse_args()
    lock=(BASE/".controller.lock").open("w")
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("controller already running")
    status("preflight", controller_pid=os.getpid(), single_controller=True, semantic_calls=0)
    check_inputs()
    # tests are embedded in test_v4.py and must run before the pilot.
    test=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py"],cwd=BASE)
    atomic(BASE/"UNIT_TEST_RECEIPT.json", {"at":now(),"returncode":test.returncode})
    if test.returncode: raise SystemExit("unit tests failed")
    pilot=list(PILOT); ledgers, feedback, slots=compile_roots(pilot)
    atomic(BASE/"FACT_LEDGER_FREEZE.json", {"phase":"pilot","ledgers":ledgers}); atomic(BASE/"FEEDBACK_SELECTION.json", {"phase":"pilot","roots":feedback}); atomic(BASE/"QUESTION_FREEZE.json", {"phase":"pilot","planned_slots":64,"valid_slots":len(slots),"slots":slots})
    status("pilot_compiled", pilot_slots=len(slots))
    if len(slots)!=64:
        atomic(BASE/"STABILITY_BLOCKER.json", {"reason":"pilot_compile_gate","slots":len(slots)}); return 1
    req=requests_for(slots,ledgers); raw=run_b1(req,"PILOT"); scores=score(raw,req); atomic(BASE/"execution/PILOT/SCORES.json",scores); report("PILOT",pilot,ledgers,slots,raw,scores)
    if raw["returned_count"]!=128:
        atomic(BASE/"STABILITY_BLOCKER.json", {"reason":"pilot_return_gate","returned":raw["returned_count"]}); return 1
    status("full_compile", pilot_passed=True, pilot_returned=128)
    ledgers,feedback,slots=compile_roots(list(ROOTS)); atomic(BASE/"FACT_LEDGER_FREEZE.json", {"phase":"full","ledgers":ledgers}); atomic(BASE/"FEEDBACK_SELECTION.json", {"phase":"full","roots":feedback}); atomic(BASE/"QUESTION_FREEZE.json", {"phase":"full","planned_slots":256,"valid_slots":len(slots),"slots":slots})
    if len(slots)!=256:
        atomic(BASE/"STABILITY_BLOCKER.json", {"reason":"full_compile_gate","slots":len(slots)}); return 1
    req=requests_for(slots,ledgers); raw=run_b1(req,"FULL"); scores=score(raw,req); atomic(BASE/"execution/FULL/SCORES.json",scores); report("FULL",list(ROOTS),ledgers,slots,raw,scores)
    state="COMPLETED" if raw["returned_count"]==512 else "PARTIAL"
    atomic(BASE/"FINAL_REPORT.json", {"status":state,"new_roots":16,"sealed_old_roots":4,"compiled_slots":len(slots),"returned_presentations":raw["returned_count"],"missing_presentations":raw["missing_count"],"scores":scores,"feedback_mode":"deterministic_relation_selection","training":False,"old4_rerun":False,"claim_limit":"reliable compiler-mediated feedback loop; not autonomous challenger causal proof"})
    status("complete", status=state, full_slots=256, full_returned=raw["returned_count"], full_missing=raw["missing_count"], semantic_calls=0)
    return 0 if state=="COMPLETED" else 1


if __name__=="__main__": raise SystemExit(main())
