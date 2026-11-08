"""Re-score frozen probe responses offline; never calls a model or training code."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from admission import EPS, VERSION, candidate_decision
from scoring import score_response

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "h4_259192_manual_qa_v2"
RUN = SOURCE / "probe_v3" / "run"
OUT = HERE / "results_revision_2.json"
REVISION = 2
CONFIG_KEYS = ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_sha256(question: dict) -> str:
    return hashlib.sha256(json.dumps(question, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def checked_audit(directory: Path, kind: str, question: dict, hashes: dict) -> dict:
    """Use question-bound review evidence; absence is pending, never a fabricated FAIL."""
    path = directory / f"{kind}_audit_revision_2.json"
    if not path.exists():
        return {"status": "EVIDENCE_MISSING", "issues": [str(path)]}
    hashes[str(path)] = sha256(path)
    document = load(path)
    row = document.get("questions", {}).get(question["candidate_id"])
    if not row:
        return {"status": "EVIDENCE_MISSING", "issues": ["QUESTION_NOT_REVIEWED"]}
    issues = []
    if row.get("question_sha256") != question_sha256(question):
        issues.append("QUESTION_REVIEW_VERSION_MISMATCH")
    if not row.get("evidence_files"):
        issues.append("REVIEW_EVIDENCE_FILES_MISSING")
    for evidence in row.get("evidence_files", []):
        evidence_path = Path(evidence["path"])
        if not evidence_path.is_file() or sha256(evidence_path) != evidence.get("sha256"):
            issues.append("REVIEW_EVIDENCE_CHANGED:" + str(evidence_path))
        else:
            hashes[str(evidence_path)] = evidence["sha256"]
    return {**row, "audit_path": str(path), "issues": issues,
            "status": "EVIDENCE_MISMATCH" if issues else row.get("status", "EVIDENCE_MISSING")}


def pair_key(request: dict) -> tuple:
    return request["candidate_id"], request["variant"]["variant_id"], request["slot"]


def verify_pair(no_request: dict, full_request: dict, no_response: dict, full_response: dict) -> list[str]:
    issues = []
    if pair_key(no_request) != pair_key(full_request):
        issues.append("QUESTION_VARIANT_SLOT_MISMATCH")
    if no_request["variant"] != full_request["variant"]:
        issues.append("DISPLAY_VARIANT_MISMATCH")
    if no_response.get("model_identifier") != full_response.get("model_identifier"):
        issues.append("MODEL_MISMATCH")
    for field in CONFIG_KEYS:
        if any(response.get("metadata", {}).get(field) is None for response in (no_response, full_response)):
            issues.append("DECODE_CONFIG_MISSING:" + field)
        elif no_response.get("metadata", {}).get(field) != full_response.get("metadata", {}).get(field):
            issues.append("DECODE_CONFIG_MISMATCH:" + field)
    for request, response in ((no_request, no_response), (full_request, full_response)):
        for field in ("request_id", "seed", "prompt_sha256"):
            if request.get(field) != response.get(field):
                issues.append("REQUEST_RESPONSE_MISMATCH:" + field)
        if response.get("variant_id") != request["variant"]["variant_id"] or response.get("mode") != request["mode"]:
            issues.append("REQUEST_RESPONSE_IDENTITY_MISMATCH")
    return sorted(set(issues))


def option_rates(scored: list[dict], question: dict) -> dict:
    valid = [row for row in scored if row["scoring_status"] in ("SCORED", "PARTIAL_NOT_APPLICABLE")]
    if not valid:
        return {"denominator": 0, "false_positive_rate": None, "false_negative_rate": None}
    gold = set(question["gold"])
    options = set(question["options"])
    return {
        "denominator": len(valid),
        "false_positive_rate": {key: sum(key in row["canonical_prediction"] for row in valid) / len(valid)
                                for key in sorted(options - gold)},
        "false_negative_rate": {key: sum(key not in row["canonical_prediction"] for row in valid) / len(valid)
                                for key in sorted(gold)},
    }


def aggregate(scored: list[dict], expected: int) -> dict:
    exact_valid = [row for row in scored if row["scoring_status"] in ("SCORED", "PARTIAL_NOT_APPLICABLE")]
    partial_valid = [row for row in scored if row["scoring_status"] == "SCORED"]
    return {
        "expected": expected,
        "exact_valid": len(exact_valid),
        "partial_valid": len(partial_valid),
        "parse_or_scoring_failure_count": len(scored) - len(exact_valid),
        "partial_not_applicable_count": len(exact_valid) - len(partial_valid),
        "exact_correct": sum(row["exact_match"] is True for row in exact_valid),
        "exact_rate": sum(row["exact_match"] is True for row in exact_valid) / len(exact_valid)
        if len(exact_valid) == expected and expected else None,
        "partial_mean": sum(row["partial_score"] for row in partial_valid) / len(partial_valid)
        if len(partial_valid) == expected and expected else None,
    }


def analyze(source: Path = SOURCE, run: Path = RUN, audit_dir: Path = HERE) -> dict:
    manifest_path = run / "manifest.json"
    question_path = source / "candidates.json"
    manifest = load(manifest_path)
    questions = {q["candidate_id"]: q for q in load(question_path)["candidates"] if q["question_type"] == "multi_select"}
    requests = [r for r in manifest["requests"] if r["question_type"] == "multi_select"]
    by_key: dict[tuple, dict] = defaultdict(dict)
    hashes = {str(manifest_path): sha256(manifest_path), str(question_path): sha256(question_path)}
    frozen_issues = ["FROZEN_INPUT_HASH_MISSING:" + name for name in
                     ("questions.json", "gold.json", "source.json") if name not in manifest.get("input_sha256", {})]
    for name, expected_hash in manifest.get("input_sha256", {}).items():
        path = run / name
        if not path.is_file() or sha256(path) != expected_hash:
            frozen_issues.append("FROZEN_INPUT_MISMATCH:" + name)
        else:
            hashes[str(path)] = expected_hash
    if (run / "questions.json").exists() and (run / "gold.json").exists():
        frozen_questions = {q["candidate_id"]: q for q in load(run / "questions.json")["candidates"]}
        frozen_gold = load(run / "gold.json")
        for qid, question in questions.items():
            frozen = {**frozen_questions.get(qid, {}), "gold": frozen_gold.get(qid)}
            if frozen != question:
                frozen_issues.append("FROZEN_QUESTION_VERSION_MISMATCH:" + qid)
    for request in requests:
        response_path = run / "responses" / (request["request_id"] + ".json")
        response = load(response_path) if response_path.exists() else None
        if response is not None:
            hashes[str(response_path)] = sha256(response_path)
        mode = request["mode"]
        key = pair_key(request)
        if mode in by_key[key]:
            raise ValueError(f"duplicate_mode:{key}:{mode}")
        by_key[key][mode] = (request, response)
    results = {}
    for question_id, question in questions.items():
        pair_rows = []
        mode_rows = {"no_context": [], "full_context": []}
        for key in sorted(k for k in by_key if k[0] == question_id):
            members = by_key[key]
            row = {"variant_id": key[1], "slot": key[2], "pair_issues": [], "modes": {}}
            for mode in ("no_context", "full_context"):
                if mode not in members or members[mode][1] is None:
                    row["pair_issues"].append("MISSING_" + mode.upper())
                    continue
                request, response = members[mode]
                try:
                    score = score_response(question, request, response)
                except (ValueError, KeyError, TypeError) as error:
                    score = {"request_id": request["request_id"],
                             "raw_output": response.get("raw_output") if isinstance(response, dict) else None,
                             "scoring_status": "METADATA_FAILURE", "parse_status": "not_scored",
                             "parse_reason": f"{type(error).__name__}:{error}",
                             "display_prediction": None, "canonical_prediction": None,
                             "mapping": request.get("variant", {}).get("display_to_canonical"),
                             "tp": None, "fp": None, "fn": None, "exact_match": None, "partial_score": None}
                row["modes"][mode] = {"request_id": request["request_id"], "seed": request["seed"],
                                       "prompt_sha256": request["prompt_sha256"], "score": score}
                mode_rows[mode].append(score)
                if score["scoring_status"] not in ("SCORED", "PARTIAL_NOT_APPLICABLE"):
                    row["pair_issues"].append("UNSCORED_" + mode.upper() + ":" + score["scoring_status"])
            if len(members) == 2 and all(members[m][1] is not None for m in ("no_context", "full_context")):
                nr, ns = members["no_context"]
                fr, fs = members["full_context"]
                row["pair_issues"].extend(verify_pair(nr, fr, ns, fs))
                if ns.get("model_identifier") != manifest["model"] or fs.get("model_identifier") != manifest["model"]:
                    row["pair_issues"].append("MANIFEST_MODEL_MISMATCH")
                for field in ("temperature", "top_p", "max_tokens"):
                    if manifest.get(field) is None or any(s.get("metadata", {}).get(field) != manifest[field] for s in (ns, fs)):
                        row["pair_issues"].append("MANIFEST_DECODE_CONFIG_MISMATCH:" + field)
                row["seed_equal"] = nr["seed"] == fr["seed"]
                if not row["pair_issues"] and all(
                    row["modes"][mode]["score"]["partial_score"] is not None
                    for mode in ("no_context", "full_context")
                ):
                    no_score = row["modes"]["no_context"]["score"]
                    full_score = row["modes"]["full_context"]["score"]
                    row["exact_context_gain"] = int(full_score["exact_match"]) - int(no_score["exact_match"])
                    row["partial_context_gain"] = full_score["partial_score"] - no_score["partial_score"]
            pair_rows.append(row)
        no = aggregate(mode_rows["no_context"], len(pair_rows))
        full = aggregate(mode_rows["full_context"], len(pair_rows))
        expected_variants = ({f"perm_{index:02d}" for index in range(2 * len(question["options"]))}
                             if manifest.get("display_protocol") == "balanced" else None)
        variant_ids = {row["variant_id"] for row in pair_rows}
        expected_slots = set(range(manifest.get("rollouts_per_position", 0)))
        pairing_complete = (bool(pair_rows) and not frozen_issues and all(not r["pair_issues"] for r in pair_rows)
                            and (expected_variants is None or variant_ids == expected_variants)
                            and all({row["slot"] for row in pair_rows if row["variant_id"] == variant} == expected_slots
                                    for variant in variant_ids))
        metrics = {
            "full_exact_rate": full["exact_rate"] if pairing_complete else None,
            "no_exact_rate": no["exact_rate"] if pairing_complete else None,
            "full_partial_mean": full["partial_mean"] if pairing_complete else None,
            "no_partial_mean": no["partial_mean"] if pairing_complete else None,
        }
        metrics["exact_context_gain"] = (metrics["full_exact_rate"] - metrics["no_exact_rate"]
                                         if None not in (metrics["full_exact_rate"], metrics["no_exact_rate"]) else None)
        metrics["partial_context_gain"] = (metrics["full_partial_mean"] - metrics["no_partial_mean"]
                                           if None not in (metrics["full_partial_mean"], metrics["no_partial_mean"]) else None)
        old_bucket = ("technical_excluded" if not pairing_complete else
                      "full_zero_review" if full["exact_correct"] == 0 else
                      "hard_candidate" if 0 < full["exact_correct"] < full["expected"] and
                      full["exact_rate"] is not None and no["exact_rate"] is not None and
                      full["exact_rate"] >= no["exact_rate"] else "OTHER")
        audits = {kind: checked_audit(audit_dir, kind, question, hashes)
                  for kind in ("quality", "split", "permutation")}
        checks = candidate_decision(
            metrics,
            quality_check_status=audits["quality"]["status"],
            permutation_check_status=audits["permutation"]["status"],
            pairing_check_status="PASS" if pairing_complete else "INCOMPLETE",
            split_check_status=audits["split"]["status"],
            confirmed_quarantine=any(a.get("confirmed_quarantine") is True for a in audits.values()),
        )
        gains = [r["partial_context_gain"] for r in pair_rows if "partial_context_gain" in r]
        results[question_id] = {
            "question_type": "multi_select", "root_id": question["root_id"],
            "gold": question["gold"], "option_count": len(question["options"]),
            "solver_version": manifest["model"], "observations_are_independent_training_questions": False,
            "legacy_eligibility": {"difficulty_bucket": old_bucket,
                                   "numeric_hard_candidate": old_bucket == "hard_candidate",
                                   "training_admitted": False},
            "review_evidence": audits,
            "pairing": {"complete": pairing_complete, "frozen_input_issues": frozen_issues,
                        "join_key": ["candidate_id", "variant_id", "slot"],
                        "seed_equality_required": False,
                        "scope": "Observed matched-display/slot comparison; no causal or significance claim."},
            "metrics": metrics, "modes": {
                mode: {**aggregate(mode_rows[mode], len(pair_rows)),
                       "option_rates": option_rates(mode_rows[mode], question)}
                for mode in ("no_context", "full_context")},
            "seed_pair_warning": "SEEDS_DIFFER_FOR_EVERY_PAIR" if pair_rows and all(not r.get("seed_equal", True) for r in pair_rows) else None,
            "partial_gain_variants": {"increased": sum(g > EPS for g in gains),
                                      "unchanged": sum(abs(g) <= EPS for g in gains),
                                      "decreased": sum(g < -EPS for g in gains)},
            "per_permutation": pair_rows, **checks,
        }
    return {"policy_version": VERSION, "implementation_revision": REVISION,
            "source_probe": str(run), "source_sha256": hashes,
            "new_policy_is_production_default": False, "training_promoted_count": 0,
            "training_promotion_executed": False,
            "counts": {"evaluated_questions": len(results),
                       "numeric_candidates": sum(r["numeric_candidate"] for r in results.values()),
                       "new_numeric_candidates": sum(r["numeric_candidate"] and not r["legacy_eligibility"]["numeric_hard_candidate"]
                                                     for r in results.values()),
                       "training_candidates": sum(r["new_policy_candidate_status"] == "TRAINING_CANDIDATE" for r in results.values())},
            "questions": results}


def main() -> None:
    result = analyze()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for question_id, row in result["questions"].items():
        print(question_id, row["metrics"], row["legacy_eligibility"]["difficulty_bucket"],
              row["new_policy_candidate_status"], row["new_policy_reason"])


if __name__ == "__main__":
    main()
