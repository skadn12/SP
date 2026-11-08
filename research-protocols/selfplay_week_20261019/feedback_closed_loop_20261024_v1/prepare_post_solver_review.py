"""Create reproducible, arm-F-only semantic review packets after raw freeze."""
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    freeze = read(P / "TARGET_PRE_SOLVER_FREEZE.json")
    for binding in freeze["bindings"]:
        assert bind(binding["path"]) == binding
    assert (P / "execution/RAW_RESPONSE_FREEZE.json").exists()
    scores = read(P / "execution/SCORES.json")
    assert scores["raw_freeze_binding"] == bind(P / "execution/RAW_RESPONSE_FREEZE.json")
    candidates = {c["candidate_id"]: c for c in read(P / "CANDIDATES_PRIVATE.json")}
    public = {r["request_id"]: r for r in read(P / "execution/PUBLIC_REQUESTS.json")["requests"]}
    private = {r["request_id"]: r for r in read(P / "execution/PRIVATE_SCORING.json")["requests"]}
    all_ids = set()
    for group in ("review_a", "review_b"):
        target_path = P / "target_review" / group / "PRE_SOLVER_TARGET_REVIEW.json"
        targets = read(target_path)
        records = []
        for target in targets["records"]:
            cid = target["candidate_id"]
            c = candidates[cid]
            assert c["arm"] == "F" and cid not in all_ids
            all_ids.add(cid)
            responses = []
            rs = [r for r in scores["records"] if r["strict"]["candidate_id"] == cid]
            assert len(rs) == 2
            for score in rs:
                rid = score["request_id"]
                raw_path = Path(score["raw_binding"]["path"])
                assert bind(raw_path) == score["raw_binding"]
                raw = read(raw_path)
                responses.append({"request_id": rid, "public_request": public[rid],
                    "presentation": private[rid]["presentation"], "score": score,
                    "raw_output": raw["raw_output"], "raw_binding": bind(raw_path)})
            records.append({"candidate": c, "frozen_target_review": target, "responses": responses})
        packet = {"schema_version": "closed_loop_post_solver_target_review_v1", "group": group,
            "bindings": [bind(target_path), bind(P / "TARGET_PRE_SOLVER_FREEZE.json"),
                bind(P / "execution/SCORES.json"), bind(P / "execution/RAW_RESPONSE_FREEZE.json")],
            "rules": [
                "Use only frozen pre-Solver target definitions; do not change a target, candidate or Gold after seeing outputs.",
                "Inspect all 16 semantic questions and both displays, not just wrong answers. Two displays are dependent repeats.",
                "HOLD targets remain valid content but cannot count as original-card mechanism hits. MC6 controls remain separate.",
                "Distinguish a target-label/pair final error from a confirmed visible mechanism error. A wrong answer alone does not identify its mechanism.",
                "A confirmed mechanism needs verbatim visible explanation and source checks supporting the specific frozen trigger. Retrieval, binding, citation-only errors and unknown causes remain separate.",
                "Correct final answers with erroneous visible target reasoning are reason-only errors; they do not lower final-answer accuracy.",
                "Do not infer faithful hidden CoT or a stable weakness. Review uses same reviewer as preaudit, not a fresh independent sample.",
                "Every evidence quote must be an exact substring of this response raw_output. Use canonical labels with display mapping stated."
            ],
            "required_record_fields": ["candidate_id", "card_id", "target_realization", "control",
                "responses: [{request_id, response_valid, final_correct, target_final_pattern_observed: bool|null, visible_mechanism_status: CONFIRMED|NOT_OBSERVED|UNKNOWN|OFF_TARGET, final_mechanism_error_confirmed: bool, reason_only_target_error: bool, citation_defect: bool, evidence: [{canonical_label_or_pair, display_label_or_pair, raw_quote, source_refs, assessment}], competing_explanation, rationale}]"],
            "records": records, "training": False}
        save(P / "target_review" / group / "POST_SOLVER_REVIEW_PACKET.json", packet)
    assert len(all_ids) == 32
    selected = read(P / "EXAMPLE_SELECTION_FREEZE.json")
    example_records = []
    for cid in selected["candidate_ids"]:
        rs = [r for r in scores["records"] if r["strict"]["candidate_id"] == cid
            and r["strict"]["variant_id"] == selected["variant"]]
        assert len(rs) == 1
        r = rs[0]
        example_records.append({"candidate": candidates[cid], "public_request": public[r["request_id"]],
            "presentation": private[r["request_id"]]["presentation"], "score": r,
            "raw_output": read(r["raw_binding"]["path"])["raw_output"]})
    save(P / "EXAMPLES_PRIVATE.json", {"selection_binding": bind(P / "EXAMPLE_SELECTION_FREEZE.json"),
        "scores_binding": bind(P / "execution/SCORES.json"), "records": example_records})
    print(json.dumps({"review_questions": len(all_ids), "review_responses": len(all_ids) * 2,
        "preselected_examples": len(example_records)}))


if __name__ == "__main__":
    main()
