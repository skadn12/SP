"""Offline, fail-closed ordering review to Solver input bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LABELS = tuple(f"I{i}" for i in range(1, 7))
FIRST_STATUS = "AUTOMATIC_SEMANTIC_REVIEW_COMPLETED"
CHALLENGE_STATUS = "ADVERSARIAL_SEMANTIC_CHALLENGE_COMPLETED"
INDEPENDENT_STATUS = "DIFFERENT_CHECKPOINT_SEMANTIC_REVIEW_COMPLETED"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def keyed(rows: list[dict], key: str, label: str) -> dict:
    require(isinstance(rows, list), f"{label}:not_list")
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"{label}:duplicate_{key}")
    return result


def validate_question(question: dict, root: str, source: dict) -> None:
    qid = question["candidate_id"]
    require(question.get("root_id") == root and question.get("question_type") == "ordering"
            and question.get("chronology_kind") == "conversation_order", f"{qid}:identity")
    require(set(question["items"]) == set(question["item_evidence"]) == set(LABELS)
            and set(question["gold"]) == set(LABELS) and len(question["gold"]) == 6,
            f"{qid}:labels")
    turns = keyed(source["turns"], "turn_ref", f"{root}:turns")
    positions = {turn["turn_ref"]: index for index, turn in enumerate(source["turns"])}
    refs = []
    for label in LABELS:
        entry = question["item_evidence"][label]
        ref = entry["turn_ref"]
        require(ref in turns and entry["speaker"] == turns[ref]["speaker"]
                and isinstance(entry["quote"], str) and entry["quote"].strip()
                and entry["quote"] in turns[ref]["text"], f"{qid}:{label}:evidence")
        require(isinstance(question["items"][label], str) and question["items"][label].strip(),
                f"{qid}:{label}:item")
        refs.append(ref)
    require(len(set(refs)) == 6 and any(ref.startswith("R01:") for ref in refs),
            f"{qid}:anchors")
    expected_gold = sorted(LABELS, key=lambda label: positions[question["item_evidence"][label]["turn_ref"]])
    require(question["gold"] == expected_gold and question["gold"] != list(LABELS),
            f"{qid}:gold")


def validate_review(run: Path, summary: dict, root: str, candidates: list[dict],
                    source: dict, source_hash: str, candidate_hash: str,
                    prior_hash: str, prior: dict, generator_model: str,
                    first_hash: str | None = None) -> dict:
    group = keyed(summary["roots"], "root_id", "summary_roots")[root]
    directory = run / root
    index, request, response = (read(directory / name) for name in
                                ("INDEX.json", "REQUEST.json", "RESPONSE.json"))
    model = summary["reviewer_model_requested"]
    ids = [row["candidate_id"] for row in candidates]
    require(index["root_id"] == response["root_id"] == root
            and index["candidate_ids"] == ids and group["source_sha256"] == index["source_sha256"] == source_hash,
            f"{root}:{run.name}:source_or_candidates")
    require(request["model"] == model == response["envelope"]["model"] == group["reviewer_model_returned"]
            and response["http_status"] == 200 and response["status"] == "RESPONSE_RECEIVED"
            and response["payload_sha256"] == group["payload_sha256"] == index["payload_sha256"]
            == digest(canonical(request)) and group["response_file_sha256"] == sha(directory / "RESPONSE.json")
            and group["reviewer_response_id"] == response["envelope"]["id"],
            f"{root}:{run.name}:response_binding")
    prompt = request["messages"][0]["content"]
    require(len(request["messages"]) == 1 and request["messages"][0]["role"] == "user"
            and index["prompt_sha256"] == group["prompt_sha256"] == digest(prompt.encode("utf-8")),
            f"{root}:{run.name}:prompt_binding")
    require(not group["schema_or_binding_errors"] and group["candidate_count"] == len(ids),
            f"{root}:{run.name}:review_errors")
    if first_hash is None:
        require(index["label_candidate_file_sha256"] == candidate_hash
                and index["prior_summary_sha256"] == prior_hash
                and index["generator_model"] == generator_model
                and group.get("candidate_file_sha256", candidate_hash) == candidate_hash,
                f"{root}:{run.name}:input_hashes")
    else:
        require(index["prior_review_sha256"] == first_hash,
                f"{root}:{run.name}:prior_review")
    prompt_data = json.loads(prompt.split("\nDATA:\n", 1)[1])
    if first_hash is None:
        require(prompt_data["root_id"] == root, f"{root}:{run.name}:prompt_root")
        supplied_rows = prompt_data["current_candidates"]
        fields = ("candidate_id", "question_type", "topic", "question", "evidence_refs", "item_text")
        expected_history = [{key: row.get(key) for key in fields} for row in prior["candidates"]]
        require(prompt_data["historical_candidates"] == expected_history,
                f"{root}:{run.name}:history_binding")
    else:
        supplied_rows = prompt_data
    supplied = keyed(supplied_rows, "candidate_id", f"{root}:{run.name}:prompt_candidates")
    require(set(supplied) == set(ids), f"{root}:{run.name}:prompt_ids")
    by_turn = keyed(source["turns"], "turn_ref", f"{root}:turns")
    for row in candidates:
        qid = row["candidate_id"]
        if first_hash is None:
            require(all(supplied[qid][field] == row[field] for field in
                        ("topic", "question", "gold")), f"{qid}:{run.name}:prompt_question_binding")
        item_rows = keyed(supplied[qid]["items"], "label", f"{qid}:prompt_items")
        require(set(item_rows) == set(LABELS), f"{qid}:prompt_item_set")
        for label in LABELS:
            item = item_rows[label]
            evidence = row["item_evidence"][label]
            require(item.get("model_written_item", item.get("claim")) == row["items"][label]
                    and item.get("cited_quote", item.get("quote")) == evidence["quote"]
                    and item.get("cited_full_turn", item.get("source_turn", item.get("full_turn")))
                    == by_turn[evidence["turn_ref"]], f"{qid}:{label}:prompt_item_binding")
    envelope = response["envelope"]
    choices = envelope["choices"]
    require(len(choices) == 1 and choices[0]["finish_reason"] == "stop",
            f"{root}:{run.name}:completion")
    raw = keyed(json.loads(choices[0]["message"]["content"])["reviews"], "candidate_id",
                f"{root}:{run.name}:raw_reviews")
    checks = keyed(group["checks"], "candidate_id", f"{root}:{run.name}:checks")
    require(set(raw) == set(checks) == set(ids), f"{root}:{run.name}:review_ids")
    for qid in ids:
        check, verdict = checks[qid], raw[qid]
        require(check["review"] == verdict and not check["schema_or_binding_errors"],
                f"{qid}:{run.name}:review_changed")
        require(set(verdict["items"]) == set(LABELS), f"{qid}:{run.name}:review_items")
        if first_hash is None:
            require(all(type(verdict["items"][label]["supported"]) is bool for label in LABELS)
                    and all(isinstance(verdict["items"][label].get("reason"), str)
                            and verdict["items"][label]["reason"].strip() for label in LABELS)
                    and all(type(verdict[field]) is bool for field in
                            ("coherent", "novel_vs_history", "distinct_within_batch")),
                    f"{qid}:{run.name}:verdict_schema")
            for label in LABELS:
                item = verdict["items"][label]
                if "unsupported_spans" in item:
                    details = item["unsupported_spans"]
                    require(isinstance(details, list) and all(isinstance(x, str) and x.strip() for x in details)
                            and bool(details) is not item["supported"],
                            f"{qid}:{label}:{run.name}:unsupported_spans")
            passed = all(verdict["items"][label]["supported"] for label in LABELS) and all(
                verdict[field] for field in ("coherent", "novel_vs_history", "distinct_within_batch"))
            require(check["semantic_pass"] is passed, f"{qid}:{run.name}:pass_changed")
        else:
            require(all(type(verdict["items"][label]["entailed"]) is bool
                        and isinstance(verdict["items"][label].get("reason"), str)
                        and verdict["items"][label]["reason"].strip() for label in LABELS),
                    f"{qid}:{run.name}:verdict_schema")
            for label in LABELS:
                item = verdict["items"][label]
                details = item.get("unsupported_details")
                require(isinstance(details, list) and all(isinstance(x, str) and x.strip() for x in details)
                        and bool(details) is not item["entailed"],
                        f"{qid}:{label}:{run.name}:unsupported_details")
            passed = all(verdict["items"][label]["entailed"] for label in LABELS)
            require(check["challenge_pass"] is passed, f"{qid}:{run.name}:pass_changed")
    count_key = "challenge_pass_count" if first_hash else "semantic_pass_count"
    pass_key = "challenge_pass" if first_hash else "semantic_pass"
    require(group[count_key] == sum(row[pass_key] for row in checks.values()),
            f"{root}:{run.name}:count_changed")
    return checks


def build(batch: Path, pilot: Path, first_run: Path, challenge_run: Path,
          independent_run: Path) -> tuple[dict, dict]:
    manifest = read(batch / "BATCH_PREPARED.json")
    report = read(pilot / "REPORT.json")
    candidates_path = pilot / "LABEL_REBOUND_UNREVIEWED.json"
    candidate_hash = sha(candidates_path)
    candidates = read(candidates_path)
    first_path = first_run / "SEMANTIC_REVIEW.json"
    challenge_path = challenge_run / "CHALLENGE_REVIEW.json"
    independent_path = independent_run / "SEMANTIC_REVIEW.json"
    first, challenge, independent = (read(path) for path in (first_path, challenge_path, independent_path))
    require(report["source_batch_sha256"] == sha(batch / "BATCH_PREPARED.json")
            and report["source_batch"] == str(batch.resolve())
            and report["generator_model"] == manifest["requested_model"]
            and report["semantic_reviewer_used"] is False
            and report["solver_calls"] == 0, "batch_binding")
    require(first["status"] == FIRST_STATUS and challenge["status"] == CHALLENGE_STATUS
            and independent["status"] == INDEPENDENT_STATUS
            and all(review["no_codex_content_review"] is True for review in
                    (first, challenge, independent))
            and first["reviewer_model_requested"] == challenge["reviewer_model_requested"]
            and first["reviewer_model_requested"] != independent["reviewer_model_requested"]
            and challenge["first_pass_review_sha256"] == sha(first_path), "review_provenance")
    roots = keyed(manifest["roots"], "root_id", "batch_roots")
    report_roots = keyed(report["roots"], "root_id", "pilot_roots")
    first_roots = keyed(first["roots"], "root_id", "first_roots")
    independent_roots = keyed(independent["roots"], "root_id", "independent_roots")
    challenge_roots = keyed(challenge["roots"], "root_id", "challenge_roots")
    candidate_roots = {row["root_id"] for row in candidates}
    require(set(roots) == set(report_roots) and candidate_roots <= set(roots)
            and set(first_roots) == set(independent_roots) == candidate_roots,
            "root_sets")
    by_candidate = keyed(candidates, "candidate_id", "candidates")
    ready, audits = [], []
    expected_challenge_roots = set()
    for root in roots:
        source_path = batch / root / "source.json"
        prior_path = batch / root / "PRIOR_CANDIDATE_SUMMARY.json"
        source, source_hash = read(source_path), sha(source_path)
        require(source["root_id"] == root and roots[root]["source_sha256"] == source_hash
                and report_roots[root]["source_sha256"] == source_hash
                and roots[root]["prior_candidate_summary_sha256"] == digest(canonical(read(prior_path)["candidates"])),
                f"{root}:batch_source")
        rows = [row for row in candidates if row["root_id"] == root]
        approved_mechanical = {check["candidate_id"] for check in report_roots[root]["checks"]
                               if check["label_only_mechanical_candidate"] is True}
        require({row["candidate_id"] for row in rows} == approved_mechanical,
                f"{root}:mechanical_candidate_set")
        if not rows:
            continue
        for row in rows:
            validate_question(row, root, source)
        prior = read(prior_path)
        first_checks = validate_review(first_run, first, root, rows, source, source_hash,
                                       candidate_hash, sha(prior_path), prior, manifest["requested_model"])
        independent_checks = validate_review(independent_run, independent, root, rows, source,
                                             source_hash, candidate_hash, sha(prior_path), prior,
                                             manifest["requested_model"])
        first_passed = [row for row in rows if first_checks[row["candidate_id"]]["semantic_pass"]]
        challenge_checks = {}
        if first_passed:
            expected_challenge_roots.add(root)
            challenge_checks = validate_review(challenge_run, challenge, root, first_passed,
                                               source, source_hash, candidate_hash,
                                               sha(prior_path), prior, manifest["requested_model"],
                                               sha(first_path))
        for row in rows:
            qid = row["candidate_id"]
            accepted = (qid in challenge_checks and challenge_checks[qid]["challenge_pass"]
                        and independent_checks[qid]["semantic_pass"])
            audits.append({"candidate_id": qid, "root_id": root,
                           "first_pass": first_checks[qid]["semantic_pass"],
                           "challenge_pass": challenge_checks[qid]["challenge_pass"] if qid in challenge_checks else None,
                           "independent_pass": independent_checks[qid]["semantic_pass"],
                           "ready_for_solver": bool(accepted)})
            if accepted:
                ready.append({**row, "source_path": str(source_path.resolve()),
                              "source_sha256": source_hash})
    require(set(challenge_roots) == expected_challenge_roots, "challenge_root_set")
    require(len(by_candidate) == len(audits), "audit_count")
    pool = {"status": "AUTOMATIC_REVIEWED_ORDERING_READY_FOR_SOLVER" if ready else "NO_ELIGIBLE_ORDERING_CANDIDATES",
            "candidate_count": len(ready), "questions": ready,
            "provenance": {"generator_model": manifest["requested_model"],
                           "first_review_model": first["reviewer_model_requested"],
                           "challenge_review_model": challenge["reviewer_model_requested"],
                           "independent_review_model": independent["reviewer_model_requested"],
                           "batch_sha256": sha(batch / "BATCH_PREPARED.json"),
                           "candidate_sha256": candidate_hash,
                           "first_review_sha256": sha(first_path),
                           "challenge_review_sha256": sha(challenge_path),
                           "independent_review_sha256": sha(independent_path)},
            "solver_calls": 0, "training_started": False}
    audit = {"status": "AUTOMATIC_ORDERING_REVIEW_BRIDGE_V1",
             "candidate_count": len(audits), "ready_count": len(ready), "candidates": audits,
             "input_sha256": {str(path.resolve()): sha(path) for path in
                              (batch / "BATCH_PREPARED.json", pilot / "REPORT.json",
                               candidates_path, first_path, challenge_path, independent_path)}}
    return pool, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "pilot", "first_run", "challenge_run", "independent_run", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    pool, audit = build(args.batch, args.pilot, args.first_run,
                        args.challenge_run, args.independent_run)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("READY_FOR_EVALUATION.json", pool), ("AUTOMATIC_REVIEW_AUDIT.json", audit)):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"status": pool["status"], "candidate_count": pool["candidate_count"]}))


if __name__ == "__main__":
    main()
