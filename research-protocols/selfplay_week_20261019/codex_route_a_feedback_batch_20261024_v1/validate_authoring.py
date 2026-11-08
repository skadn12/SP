"""Mechanical validation only; semantic admission is performed separately."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    contract = read(P / "REVIEW_CONTRACT.json")
    selected = read(P / "SELECTION_FREEZE.json")
    expected_ids = {row["candidate_id"] for row in selected["records"]}
    all_cases, all_cards, all_errors = [], [], []
    checks, failures, core_bindings = [], [], []
    def check(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(name)
    def required(name, value, key):
        check(name, set(contract[key]).issubset(value), sorted(set(contract[key]) - set(value)))
    def source_quotes(name, rows, source):
        for i, row in enumerate(rows):
            ref, quote = row.get("turn_ref"), row.get("quote")
            check(f"{name}:{i}:source_quote", ref in source and isinstance(quote, str) and bool(quote) and quote in source.get(ref, ""))
    for root in sorted({row["root_id"] for row in selected["records"]}):
        short = root.split("-")[-1]
        directory = P / "authoring" / f"root{short}"
        source_packet = read(P / "inputs" / f"{root}_CONTENT_FIRST.json")
        source = {row["turn_ref"]: row["text"] for row in source_packet["source"]["turns"]}
        questions = {row["candidate_id"]: row for row in source_packet["questions"]}
        responses = {row["candidate_id"]: row for row in read(P / "inputs" / f"{root}_SCORING_AND_RESPONSES_PRIVATE.json")["records"]}
        cases = read(directory / "CASE_REVIEWS.json")["records"]
        cards = read(directory / "CANDIDATE_CARDS_PRIVATE.json")["cards"]
        public = read(directory / "CHALLENGER_PUBLIC_CARDS.json")["cards"]
        for name in ("CONTENT_FIRST_PASS.json", "CASE_REVIEWS.json", "CANDIDATE_CARDS_PRIVATE.json", "CHALLENGER_PUBLIC_CARDS.json"):
            core_bindings.append(bind(directory / name))
        check(f"{root}:case_coverage", {row["candidate_id"] for row in cases} == set(questions) and len(cases) == len(questions))
        for case in cases:
            cid = case["candidate_id"]
            row = responses[cid]
            q = questions[cid]
            required(f"{cid}:case_fields", case, "case_required")
            required(f"{cid}:content_fields", case["content_review"], "content_review_required")
            derived = case["content_review"]["independently_derived_display_gold"]
            check(f"{cid}:derived_gold_matches", derived == row["scoring"]["display_gold"] and case["content_review"]["frozen_gold_matches"] is True)
            check(f"{cid}:selected_request", case["request_id"] == row["request_id"])
            check(f"{cid}:all_labels", {item["label"] for item in case["item_checks"]} == set(q["display"]) and len(case["item_checks"]) == len(q["display"]))
            for item in case["item_checks"]:
                required(f"{cid}:{item['label']}:item_fields", item, "item_check_required")
                check(f"{cid}:{item['label']}:refs_exist", all(ref in source for ref in item["source_refs"]))
                source_quotes(f"{cid}:{item['label']}", item["source_quotes"], source)
            for error in case["observed_errors"]:
                eid = error["error_id"]
                required(f"{eid}:error_fields", error, "error_required")
                check(f"{eid}:solver_quote", bool(error["solver_quote"]) and error["solver_quote"] in row["final_text"])
                check(f"{eid}:labels", set(error["affected_labels"]).issubset(q["display"]))
                check(f"{eid}:status", error["status"] in ("OBSERVED_ERROR", "OBSERVED_REASON_ERROR"))
                source_quotes(eid, error["source_evidence"], source)
                all_errors.append(error)
            for finding in case["questioner_findings"]:
                required(f"{finding['issue_id']}:finding_fields", finding, "finding_required")
            all_cases.append(case)
        root_error_ids = {error["error_id"] for case in cases for error in case["observed_errors"]}
        for card in cards:
            fid = card["feedback_id"]
            required(f"{fid}:card_fields", card, "card_required")
            required(f"{fid}:solver_fields", card["solver_feedback"], "solver_feedback_required")
            required(f"{fid}:extension_fields", card["extension_task"], "extension_required")
            required(f"{fid}:task_fields", card["question_task"], "question_task_required")
            check(f"{fid}:candidate_only", card["status"] == "HYPOTHESIS")
            check(f"{fid}:support_error_ids", bool(card["supporting_error_ids"]) and set(card["supporting_error_ids"]).issubset(root_error_ids))
            check(f"{fid}:support_cases", set(card["supporting_candidate_ids"]).issubset(questions))
            ext = card["extension_task"]
            check(f"{fid}:no_unreviewed_extend", ext["needed"] is False and not ext["material_targets"] and not ext["source_shortfall_issue_ids"])
            all_cards.append(card)
        check(f"{root}:public_card_ids", {row["feedback_id"] for row in public} == {row["feedback_id"] for row in cards})
        for card in public:
            fid = card["feedback_id"]
            check(f"{fid}:public_whitelist", set(card).issubset(contract["public_fields"]))
            text = json.dumps(card, ensure_ascii=False)
            check(f"{fid}:no_old_question_ids_or_box", "PDF1-" not in text and "<box>" not in text and "display_gold" not in text and "final_answer" not in text)
    check("all_30_unique_cases", len(all_cases) == 30 and {row["candidate_id"] for row in all_cases} == expected_ids)
    check("unique_error_ids", len(all_errors) == len({row["error_id"] for row in all_errors}))
    check("unique_card_ids", len(all_cards) == len({row["feedback_id"] for row in all_cards}))
    for record in selected["records"]:
        for key in ("source_binding", "response_binding"):
            check(f"{record['candidate_id']}:{key}", bind(record[key]["path"]) == record[key])
    result = {"status": "PASS" if not failures else "FAIL", "checks": checks,
              "failures": failures, "cases": len(all_cases),
              "items": sum(len(row["item_checks"]) for row in all_cases),
              "errors": len(all_errors), "cards": len(all_cards),
              "core_bindings": core_bindings,
              "scope": "Mechanical consistency only. Does not imply semantic validity or automatic candidate admission."}
    (P / "AUTHORING_MECHANICAL_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "failures", "cases", "items", "errors", "cards")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
