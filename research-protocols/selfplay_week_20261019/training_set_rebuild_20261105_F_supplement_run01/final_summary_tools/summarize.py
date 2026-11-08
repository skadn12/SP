#!/usr/bin/env python3
"""Read-only readiness checks and a separate, immutable run ledger export."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
INCREMENT = "adjudicated_increment_01"
BATCHES = [f"batch_{n:02d}" for n in range(1, 10)]
SCOPE = {"batch_05", "batch_06", "batch_08"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.errors = []
        self.pending = []
        self.inputs = {}

    def read(self, path, required=True):
        path = Path(path)
        if not path.is_absolute():
            path = RUN / path
        if not path.is_file():
            if required:
                self.errors.append(f"missing input: {path}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.inputs[str(path)] = sha(path)
            return data
        except (OSError, ValueError) as exc:
            self.errors.append(f"invalid input {path}: {exc}")
            return None

    def verify(self, path, expected, label):
        path = Path(path)
        if not path.is_absolute():
            path = RUN / path
        if not path.is_file():
            self.errors.append(f"{label}: missing {path}")
        else:
            actual = self.inputs.get(str(path))
            if actual is None:
                actual = sha(path)
            self.inputs[str(path)] = actual
            if actual != expected:
                self.errors.append(f"{label}: hash mismatch {path}")

    def bindings(self, path):
        data = self.read(path, required=False)
        if data is not None:
            for row in data.get("files", []):
                self.verify(row["path"], row["sha256"], str(path))


def index(rows, key, audit, label):
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            audit.errors.append(f"{label}: missing or duplicate {key}: {value}")
        result[value] = row
    return result


def solver(audit, batch, items, quality, sources):
    d = RUN / "evaluation" / batch / "solver"
    req = audit.read(d / "REQUESTS.json") or []
    pre = audit.read(d / "PREFLIGHT.json") or {}
    audit.bindings(d / "INPUT_BINDINGS.json")
    requests = index(req, "request_id", audit, f"{batch} requests")
    eligible = {cid for cid in items if quality[cid].get("content_pass") is True
                and sources[cid].get("source_qualified") is True}
    if pre.get("status") != "PASS" or pre.get("truncation") is not False:
        audit.errors.append(f"{batch}: preflight not passing or truncated")
    if (d / "REQUESTS.json").is_file():
        audit.verify(d / "REQUESTS.json", pre.get("requests_sha256"), f"{batch} preflight")
    if pre.get("request_count") != len(req):
        audit.errors.append(f"{batch}: preflight request count mismatch")
    by_item = collections.defaultdict(list)
    for row in req:
        cid = row.get("candidate_id")
        by_item[cid].append(row)
        if cid not in eligible or row.get("root_id") != items.get(cid, {}).get("root_id"):
            audit.errors.append(f"{batch}: unexpected request {row.get('request_id')}")
        for kind in ("prompt", "schema"):
            if row.get(f"{kind}_path") and row.get(f"{kind}_sha256"):
                audit.verify(d / row[f"{kind}_path"], row[f"{kind}_sha256"], row["request_id"])
            else:
                audit.errors.append(f"{batch}: missing {kind} binding {row.get('request_id')}")
    for cid in eligible:
        expected = min(6, math.factorial(len(items[cid]["options"]))) * 3
        for mode in ("full_context", "no_context"):
            selected = [r for r in by_item[cid] if r.get("context_mode") == mode]
            pairs = {(r.get("variant_id"), r.get("repeat_index")) for r in selected}
            if len(selected) != expected or len(pairs) != expected:
                audit.errors.append(f"{batch}/{cid}: {mode} request matrix incomplete")
    for mode in ("full_context", "no_context"):
        if pre.get(mode + "_requests") != sum(r.get("context_mode") == mode for r in req):
            audit.errors.append(f"{batch}: {mode} preflight count mismatch")

    completed = audit.read(d / "COMPLETED.json", required=False)
    raw = audit.read(d / "RAW_ANSWERS.json", required=False)
    scores = audit.read(d / "SCORES_FINAL_ANSWER_ONLY.json", required=False)
    decisions = audit.read(d / "CANDIDATE_LAYER_DECISIONS.json", required=False)
    auxiliary = audit.read(d / "AUXILIARY_DIAGNOSTICS.json", required=False)
    complete = all(x is not None for x in (completed, raw, scores, decisions, auxiliary))
    results = {}
    if not complete:
        audit.pending.append(f"{batch}: solver output incomplete")
    else:
        raw_by_id = index(raw, "request_id", audit, f"{batch} raw answers")
        if set(raw_by_id) != set(requests) or completed.get("returned_requests") != len(req):
            audit.errors.append(f"{batch}: raw answer/request coverage mismatch")
        for rid, row in raw_by_id.items():
            if row.get("status") != "RETURNED" or not isinstance(row.get("raw_prediction"), str):
                audit.errors.append(f"{batch}: missing raw prediction {rid}")
            if rid in requests and row.get("prompt_sha256") != requests[rid].get("prompt_sha256"):
                audit.errors.append(f"{batch}: raw prompt hash mismatch {rid}")
        score_items = scores.get("items", {})
        decision_items = index(decisions.get("items", []), "candidate_id", audit, f"{batch} decisions")
        scored_requests = index(scores.get("requests", []), "request_id", audit, f"{batch} scored requests")
        if set(scored_requests) != set(requests):
            audit.errors.append(f"{batch}: scored request coverage mismatch")
        if set(score_items) != eligible or set(decision_items) != set(items):
            audit.errors.append(f"{batch}: score/decision candidate coverage mismatch")
        for cid in set(items) - eligible:
            status = decision_items.get(cid, {}).get("status", "")
            if status not in {"QUALITY_HOLD_NOT_EVALUATED", "SOURCE_HOLD_NOT_EVALUATED"}:
                audit.errors.append(f"{batch}/{cid}: excluded candidate has unexpected decision {status}")
        for cid in eligible & set(score_items) & set(decision_items):
            score = score_items[cid]
            dec = decision_items[cid]
            for mode in ("full_context", "no_context"):
                stat = score.get(mode, {})
                count = sum(r.get("context_mode") == mode for r in by_item[cid])
                if stat.get("planned_calls") != count or stat.get("valid_calls", -1) + stat.get("invalid_calls", -1) != count:
                    audit.errors.append(f"{batch}/{cid}: {mode} score count mismatch")
            mean = score.get("full_context", {}).get("mean_directed_pair_f1", score.get("full_context", {}).get("mean_J"))
            if not isinstance(mean, (int, float)) or not math.isfinite(mean) or dec.get("mean_task_score") != mean:
                audit.errors.append(f"{batch}/{cid}: full-context score mismatch")
            results[cid] = {"status": dec.get("status"), "full_context": score.get("full_context"),
                            "no_context": score.get("no_context"), "mean_task_score": mean,
                            "raw_answer_file": str(d / "RAW_ANSWERS.json"),
                            "raw_answer_sha256": audit.inputs[str(d / "RAW_ANSWERS.json")],
                            "request_ids": [r["request_id"] for r in by_item[cid]]}
    return {"batch": batch, "preflight": pre, "complete": complete, "requests": len(req),
            "eligible": sorted(eligible), "results": results, "by_item": by_item,
            "auxiliary": auxiliary}


def build():
    a = Audit()
    config = a.read("RUN_CONFIG.json") or {}
    freeze = a.read("INPUT_FREEZE.json") or {}
    roots = index(freeze.get("roots", []), "root_id", a, "freeze")
    if len(roots) != 68 or config.get("scope", {}).get("required_roots") != 68:
        a.errors.append("frozen root inventory is not 68")
    for rid, row in roots.items():
        a.verify(row["source_snapshot"], row["source_sha256"], rid + " source")
        a.verify(row["neutral_source_info_snapshot"], row["neutral_source_info_sha256"], rid + " neutral info")
        src = a.read(row["source_snapshot"]) or {}
        if len(src.get("turns", [])) != row["turn_count"]:
            a.errors.append(f"{rid}: source turn count mismatch")
    for name in ("RUN_PROTOCOL.md", "INPUT_CHECKSUMS.sha256", "EVIDENCE_METADATA_POLICY.md"):
        p = RUN / name
        if p.is_file():
            a.inputs[str(p)] = sha(p)
        else:
            a.errors.append(f"missing protocol input: {p}")
    reference = a.read("semantic_review/REFERENCE_FREEZE.json") or {}
    if reference.get("roots") != 68 or reference.get("score_free_projection") is not True or reference.get("damaged_selection_json_used") is not False:
        a.errors.append("semantic reference freeze is incomplete or not score-free")
    for binding in reference.get("input_bindings", []):
        a.verify(binding["path"], binding["sha256"], "semantic reference freeze")
    pre_adj = a.read("PRE_EVALUATION_QUALITY_ADJUDICATION.json") or {}
    pre_by_id = index(pre_adj.get("items", []), "candidate_id", a, "pre-evaluation adjudication")
    inc_record = a.read(f"evaluation/{INCREMENT}/ADJUDICATION.json") or {}
    inc_ids = set(inc_record.get("candidate_ids", []))
    if len(inc_ids) != 5:
        a.errors.append("increment must contain the five adjudicated candidates")
    for key in ("quality_adjudication", "source_adjudication"):
        ref = inc_record.get(key, {})
        if ref.get("path") and ref.get("sha256"):
            a.verify(ref["path"], ref["sha256"], "increment adjudication")
        else:
            a.errors.append(f"increment missing {key} binding")
    batches = {}
    all_items = {}
    for batch in BATCHES + [INCREMENT]:
        effective = batch + "_scope_adjudicated" if batch in SCOPE else batch
        d = RUN / "evaluation" / effective
        items = index(a.read(d / "ITEMS_PRIVATE.json") or [], "candidate_id", a, effective)
        qual = index((a.read(d / "QUALITY_FINAL.json") or {}).get("items", []), "candidate_id", a, effective + " quality")
        sources = index((a.read(d / "SOURCE_QUALIFICATION.json") or {}).get("items", []), "candidate_id", a, effective + " source")
        a.read(d / "SOURCE_FREEZE.json")
        a.bindings(d / "INPUT_BINDINGS.json")
        a.bindings(d / "ADJUDICATION_INPUT_BINDINGS.json")
        if set(items) != set(qual) or set(items) != set(sources):
            a.errors.append(f"{effective}: item/quality/source candidate sets differ")
            continue
        for cid, item in items.items():
            if item.get("root_id") not in roots or item.get("source_sha256") != roots.get(item.get("root_id"), {}).get("source_sha256"):
                a.errors.append(f"{effective}/{cid}: root or source binding mismatch")
            if qual[cid].get("source_sha256") != item.get("source_sha256") or sources[cid].get("source_sha256") != item.get("source_sha256"):
                a.errors.append(f"{effective}/{cid}: gate source mismatch")
        batches[effective] = {"items": items, "quality": qual, "sources": sources}
        if batch != INCREMENT:
            packet_binding = a.read(RUN / "semantic_review" / "packets" / batch / "PACKET_BINDINGS.json") or {}
            if set(packet_binding.get("new_candidate_ids", [])) != set(items):
                a.errors.append(f"{batch}: semantic packet candidate coverage mismatch")
            for binding in packet_binding.get("input_bindings", []):
                a.verify(binding["path"], binding["sha256"], f"{batch} semantic packet")
            for cid, item in items.items():
                if cid in all_items:
                    a.errors.append(f"duplicate candidate across base batches: {cid}")
                all_items[cid] = (effective, item)
    if INCREMENT in batches:
        if set(batches[INCREMENT]["items"]) != inc_ids:
            a.errors.append("increment candidate set differs from adjudication")
        for cid in inc_ids:
            if cid not in all_items or all_items[cid][0] not in {"batch_02", "batch_03"}:
                a.errors.append(f"{cid}: invalid increment parent")
            elif batches[INCREMENT]["items"][cid] != all_items[cid][1]:
                a.errors.append(f"{cid}: increment item differs from parent")
    root_candidates = collections.defaultdict(list)
    for cid, (_, item) in all_items.items():
        root_candidates[item.get("root_id")].append(cid)
    if set(root_candidates) != set(roots) or any(len(v) != 1 for v in root_candidates.values()):
        a.errors.append("candidate/root mapping is not one-to-one across all 68 roots")
    solver_batches = {}
    for batch, data in batches.items():
        solver_batches[batch] = solver(a, batch, data["items"], data["quality"], data["sources"])
    for cid in inc_ids:
        if cid in all_items:
            parent = all_items[cid][0]
            if solver_batches[parent]["by_item"].get(cid):
                a.errors.append(f"{cid}: increment candidate already had parent solver requests")
    ledger = []
    missing_second_pass = []
    second_pass_duplicates = []
    second_pass_uncertain = []
    for root_id, root in roots.items():
        candidates = root_candidates.get(root_id, [])
        if len(candidates) != 1:
            continue
        cid = candidates[0]
        parent, item = all_items[cid]
        effective = INCREMENT if cid in inc_ids else parent
        data, sol = batches[effective], solver_batches[effective]
        quality, source = data["quality"][cid], data["sources"][cid]
        semantic_path = RUN / "semantic_review" / "results" / f"{cid}.json"
        semantic = a.read(semantic_path) or {}
        if semantic.get("candidate_id") != cid or semantic.get("score_free") is not True:
            a.errors.append(f"{cid}: missing or invalid score-free semantic review")
        if semantic.get("judgment") not in {"fully_duplicate", "overlap_but_distinct"}:
            a.errors.append(f"{cid}: unrecognized semantic judgment")
        original_batch = parent.removesuffix("_scope_adjudicated")
        packet_path = RUN / "semantic_review" / "packets" / original_batch / f"{cid}.json"
        packet = a.read(packet_path) or {}
        references = packet.get("references", [])
        ref_ids = [r.get("candidate_id") for r in references]
        if (packet.get("candidate_id") != cid or packet.get("root_id") != root_id
                or packet.get("reference_count") != len(ref_ids) or not all(ref_ids)
                or len(ref_ids) != len(set(ref_ids))):
            a.errors.append(f"{cid}: invalid or duplicate semantic packet references")
        second_path = RUN / "semantic_review" / "second_pass" / f"{cid}.json"
        second = a.read(second_path, required=False)
        second_valid = False
        if second is None:
            missing_second_pass.append(cid)
        else:
            comparisons = second.get("comparisons")
            second_valid = (second.get("candidate_id") == cid and second.get("score_free") is True
                            and isinstance(comparisons, list)
                            and all(isinstance(c, dict) and c.get("reference_candidate_id")
                                    and c.get("judgment") in {"fully_duplicate", "overlap_but_distinct", "distinct", "uncertain"}
                                    and isinstance(c.get("reason"), str) and c["reason"].strip()
                                    and c.get("score_free") is True for c in comparisons))
            if second_valid:
                compared_ids = [c["reference_candidate_id"] for c in comparisons]
                if len(compared_ids) != len(set(compared_ids)) or set(compared_ids) != set(ref_ids):
                    a.errors.append(f"{cid}: second-pass reference coverage mismatch: missing={sorted(set(ref_ids) - set(compared_ids))}, extra={sorted(set(compared_ids) - set(ref_ids))}, duplicate_count={len(compared_ids) - len(set(compared_ids))}")
                    second_valid = False
            if not second_valid:
                a.errors.append(f"{cid}: invalid second-pass semantic review")
            elif any(c["judgment"] == "fully_duplicate" for c in comparisons):
                second_pass_duplicates.append(cid)
            if second_valid and any(c["judgment"] == "uncertain" for c in comparisons):
                second_pass_uncertain.append(cid)
        model = sol["results"].get(cid)
        if quality.get("content_pass") is not True:
            status = "QUALITY_HOLD"
        elif source.get("source_qualified") is not True:
            status = "SOURCE_HOLD"
        elif not sol["complete"]:
            status = "SOLVER_PENDING"
        elif model is None:
            status = "SOLVER_MISSING"
            a.errors.append(f"{cid}: eligible candidate missing solver result")
        elif not isinstance(model.get("mean_task_score"), (int, float)) or not math.isfinite(model["mean_task_score"]):
            status = "SOLVER_INVALID"
        elif not 0 < model["mean_task_score"] < 1:
            status = model["status"]
        elif not second_valid:
            status = "PENDING_SEMANTIC_SUPPLEMENT"
        elif semantic.get("judgment") == "fully_duplicate" or cid in second_pass_duplicates or cid in second_pass_uncertain:
            status = "PENDING_SEMANTIC_FAMILY_ADJUDICATION"
        else:
            status = "ELIGIBLE_PENDING_UNION_SELECTION"
        ledger.append({"root_id": root_id, "candidate_id": cid, "review_id": item.get("review_id"),
                       "question_type": item.get("question_type"), "original_batch": parent,
                       "effective_evaluation": effective, "source_sha256": root["source_sha256"],
                       "quality": quality, "source_qualification": source, "model": model,
                       "pre_evaluation_adjudication": pre_by_id.get(cid), "semantic": semantic,
                       "semantic_second_pass": second,
                       "semantic_second_pass_path": str(second_path) if second is not None else None,
                       "semantic_packet_path": str(packet_path), "semantic_packet_sha256": a.inputs.get(str(packet_path)),
                       "final_status": status})
    if missing_second_pass:
        a.pending.append(f"semantic second pass incomplete: {len(missing_second_pass)} candidates")
    total_requests = sum(x["requests"] for x in solver_batches.values())
    if total_requests > config.get("solver_budget", {}).get("max_calls_total", 0):
        a.errors.append("total request budget exceeded")
    report = {"root_count": len(roots), "ledger_count": len(ledger), "candidate_count": len(all_items),
              "question_type_counts": dict(collections.Counter(x["question_type"] for x in ledger)),
              "status_counts": dict(collections.Counter(x["final_status"] for x in ledger)),
              "requests_total": total_requests,
              "solver_batches": {k: {z: v for z, v in x.items() if z not in {"results", "by_item"}} for k, x in solver_batches.items()},
              "semantic_fully_duplicate_pending": [x["candidate_id"] for x in ledger if x["semantic"].get("judgment") == "fully_duplicate"],
              "semantic_second_pass_fully_duplicate": second_pass_duplicates,
              "semantic_second_pass_uncertain": second_pass_uncertain,
              "semantic_second_pass_missing": missing_second_pass,
              "integrity_errors": a.errors, "pending": a.pending,
              "training_export_created": False, "input_sha256": a.inputs}
    return ledger, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only readiness and integrity check")
    mode.add_argument("--export", type=Path, help="new directory for an immutable report, after all solvers complete")
    args = parser.parse_args()
    ledger, report = build()
    if args.check:
        print(json.dumps({k: v for k, v in report.items() if k not in {"input_sha256", "solver_batches"}}, ensure_ascii=False, indent=2))
        return 1 if report["integrity_errors"] else 0
    if report["integrity_errors"] or report["pending"]:
        print(json.dumps({"export_blocked": True, "integrity_errors": report["integrity_errors"], "pending": report["pending"]}, ensure_ascii=False, indent=2))
        return 1
    output = args.export.resolve()
    output.mkdir(parents=False, exist_ok=False)
    for name, value in (("ROOT_LEDGER.json", {"items": ledger}), ("RUN_SUMMARY.json", report)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"output": str(output), "roots": len(ledger), "semantic_supplement_pending": len(report["semantic_fully_duplicate_pending"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
