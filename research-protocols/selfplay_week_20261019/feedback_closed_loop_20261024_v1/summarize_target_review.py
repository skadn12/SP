"""Validate post-Solver reviews against frozen criteria and aggregate repeats."""
import collections
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    frozen = read(P / "TARGET_PRE_SOLVER_FREEZE.json")
    for binding in frozen["bindings"]:
        assert bind(binding["path"]) == binding
    targets = {r["candidate_id"]: r for r in frozen["records"]}
    scores = {r["request_id"]: r for r in read(P / "execution/SCORES.json")["records"]}
    candidates = {c["candidate_id"]: c for c in read(P / "CANDIDATES_PRIVATE.json")}
    records, bindings, seen = [], [], set()
    for group in ("review_a", "review_b"):
        path = P / "target_review" / group / "POST_SOLVER_TARGET_REVIEW.json"
        doc = read(path)
        bindings.append(bind(path))
        for record in doc["records"]:
            cid = record["candidate_id"]
            assert cid in targets and cid not in seen
            seen.add(cid)
            c, t = candidates[cid], targets[cid]
            assert record["target_realization"] == t["target_realization"]
            assert record["control"] == (c["slot_id"] == "MC6")
            assert record["card_id"] == c["feedback_target"]["card_id"]
            responses = record["responses"]
            wanted = {r["request_id"] for r in scores.values() if r["strict"]["candidate_id"] == cid}
            assert len(responses) == 2 and {r["request_id"] for r in responses} == wanted
            for response in responses:
                rid = response["request_id"]
                score = scores[rid]
                assert response["response_valid"] is score["strict"]["valid"]
                assert response["final_correct"] is score["strict"]["exact"]
                for field in ("response_valid", "final_correct", "final_mechanism_error_confirmed", "reason_only_target_error", "citation_defect"):
                    assert type(response[field]) is bool, (rid, field)
                for field in ("target_reason_error", "target_citation_error", "citation_only_target_error"):
                    if field in response:
                        assert type(response[field]) is bool, (rid, field)
                pattern = response["target_final_pattern_observed"]
                assert pattern is None or type(pattern) is bool
                status = response["visible_mechanism_status"]
                assert status in ("CONFIRMED", "NOT_OBSERVED", "UNKNOWN", "OFF_TARGET")
                if t["target_realization"] == "HOLD":
                    assert pattern is None and status == "OFF_TARGET"
                    assert not response["final_mechanism_error_confirmed"] and not response["reason_only_target_error"]
                    assert not any(response.get(field, False) for field in ("target_reason_error", "target_citation_error", "citation_only_target_error"))
                if response["final_mechanism_error_confirmed"]:
                    assert not response["final_correct"] and pattern is True and status == "CONFIRMED"
                if response["reason_only_target_error"]:
                    assert response["final_correct"] and status == "CONFIRMED"
                raw_path = Path(score["raw_binding"]["path"])
                assert bind(raw_path) == score["raw_binding"]
                raw = read(raw_path)["raw_output"]
                if status == "CONFIRMED":
                    assert response["evidence"], (rid, "no evidence")
                for item in response["evidence"]:
                    quote = item["raw_quote"]
                    assert isinstance(quote, str) and quote and quote in raw, (rid, quote)
            records.append(record)
    assert seen == set(targets) and len(records) == 32

    def summarize(rs):
        responses = [response for r in rs for response in r["responses"]]
        def count(field):
            return {"responses": sum(r.get(field) is True for r in responses),
                "semantic_questions_any_display": sum(any(x.get(field) is True for x in r["responses"]) for r in rs),
                "semantic_questions_both_displays": sum(all(x.get(field) is True for x in r["responses"]) for r in rs),
                "roots_with_any": sorted({candidates[r["candidate_id"]]["root_id"] for r in rs if any(x.get(field) is True for x in r["responses"])})}
        return {"semantic_questions": len(rs), "dependent_display_responses": len(responses),
            "target_final_pattern": count("target_final_pattern_observed"),
            "confirmed_final_mechanism_error": count("final_mechanism_error_confirmed"),
            "reason_only_target_error": count("reason_only_target_error"),
            "target_reason_error": count("target_reason_error"),
            "target_citation_error": count("target_citation_error"),
            "citation_only_target_error": count("citation_only_target_error"),
            "citation_defect": count("citation_defect"),
            "visible_mechanism_status_counts": dict(collections.Counter(r["visible_mechanism_status"] for r in responses))}

    scopes = {
        "ALL_F": records,
        "NON_MC6_ALL_ATTEMPTS": [r for r in records if not r["control"]],
        "REALIZED_NON_MC6": [r for r in records if not r["control"] and r["target_realization"] == "PASS"],
        "HOLD_NON_MC6": [r for r in records if not r["control"] and r["target_realization"] == "HOLD"],
        "MC6_CONTROL": [r for r in records if r["control"]],
    }
    summaries = {name: summarize(rs) for name, rs in scopes.items()}
    roots = sorted({candidates[r["candidate_id"]]["root_id"] for r in records})
    out = {"schema_version": "closed_loop_target_summary_v1", "status": "PASS_STRUCTURAL_AND_QUOTE_CHECKS",
        "bindings": bindings + [bind(P / "TARGET_PRE_SOLVER_FREEZE.json"), bind(P / "execution/SCORES.json")],
        "summaries": summaries,
        "realized_noncontrol_by_root": {root: summarize([r for r in scopes["REALIZED_NON_MC6"] if candidates[r["candidate_id"]]["root_id"] == root]) for root in roots},
        "confirmed_final_error_candidate_ids": [r["candidate_id"] for r in scopes["REALIZED_NON_MC6"] if any(x["final_mechanism_error_confirmed"] for x in r["responses"])],
        "records": records,
        "interpretation": "Visible explanation corroborates local target errors only; not faithful hidden CoT, stable cross-root reproduction, a feedback-versus-control mechanism comparison or training repair. HOLD targets are excluded from original-card hits. Two displays are dependent.",
        "training": False}
    with (P / "TARGET_ERROR_SUMMARY.json").open("x") as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
