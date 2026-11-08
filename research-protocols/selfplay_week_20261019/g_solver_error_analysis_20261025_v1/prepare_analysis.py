"""Prepare read-only analyses of the frozen G-arm responses; no inference."""
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
E = P.parent / "feedback_closed_loop_20261024_v1"
ROOTS = ("218519", "222937", "225575", "229686")
TAXONOMY = ["QUANTITY_UNIT_OR_ARITHMETIC_ERROR", "VALUE_TO_JUDGMENT_MISMATCH",
    "ENTITY_ROLE_OR_OBJECT_BINDING_ERROR", "STATE_PLAN_CONFIRM_COMPLETE_ERROR",
    "CONSTRAINT_CONJUNCTION_ERROR", "CAUSAL_RELATION_ERROR",
    "EVIDENCE_SCOPE_OR_CITATION_BINDING_ERROR", "CLOCK_AXIS_OR_ORDERING_ERROR",
    "PROCEDURE_DEPENDENCY_ERROR", "DISCOURSE_ANCHOR_OR_ORDERING_ERROR",
    "ANSWER_ASSEMBLY_INCONSISTENCY", "ORDERING_OMISSION_OR_DUPLICATION", "UNKNOWN_CAUSE"]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    assert hashlib.sha256((E / "FINAL_MANIFEST.json").read_bytes()).hexdigest() == (E / "FINAL_MANIFEST.sha256").read_text().split()[0]
    candidates = read(E / "CANDIDATES_PRIVATE.json")
    gs = [c for c in candidates if c["arm"] == "G"]
    fs = {(c["root_id"].split("-")[-1], c["slot_id"]): c for c in candidates if c["arm"] == "F"}
    scores = read(E / "execution/SCORES.json")["records"]
    public = {r["request_id"]: r for r in read(E / "execution/PUBLIC_REQUESTS.json")["requests"]}
    private = {r["request_id"]: r for r in read(E / "execution/PRIVATE_SCORING.json")["requests"]}
    reviews = {r["candidate_id"]: r for r in read(E / "CONTENT_REVIEW_AI.json")["reviews"]}
    cards = {}
    for group in ("review_a", "review_b"):
        for c in read(E / "target_review" / group / "PRE_SOLVER_TARGET_PACKET.json")["public_cards"]:
            cards[c["feedback_id"]] = c
    bindings = [bind(E / name) for name in ("FINAL_MANIFEST.json", "CANDIDATES_PRIVATE.json", "execution/SCORES.json", "execution/RAW_RESPONSE_FREEZE.json", "execution/PREPARATION.json")]
    for root in ROOTS:
        records, card_records = [], []
        source = E / "sources" / f"cc-train-episode-{root}.json"
        for c in [c for c in gs if c["root_id"].endswith(root)]:
            assert bind(source) == c["source_binding"]
            responses = []
            for score in [r for r in scores if r["strict"]["candidate_id"] == c["candidate_id"]]:
                rid = score["request_id"]
                path = Path(score["raw_binding"]["path"])
                assert bind(path) == score["raw_binding"]
                raw = read(path)
                assert raw["status"] == "RETURNED" and raw["finish_reason"] == "stop"
                responses.append({"request_id": rid, "public_request": public[rid],
                    "presentation": private[rid]["presentation"], "score": score,
                    "raw_output": raw["raw_output"], "raw_binding": bind(path)})
            assert len(responses) == 2
            records.append({"candidate": c, "content_review": reviews[c["candidate_id"]], "responses": responses})
            fid = fs[(root, c["slot_id"])]["feedback_target"]["card_id"]
            card_records.append({"candidate_id": c["candidate_id"], "slot_id": c["slot_id"], "card_id": fid, "original_public_card": cards[fid]})
        assert len(records) == 8
        dest = P / "reviews" / root
        save(dest / "ERROR_REVIEW_PACKET.json", {"schema_version": "g_solver_error_review_packet_v1",
            "root_id": root, "source_binding": bind(source), "records": records, "taxonomy": TAXONOMY,
            "rules": ["Read the full frozen source and all 16 responses, including correct ones. This is a post-hoc G-arm analysis, not a blinded arm comparison.",
                "Keep canonical and displayed labels explicit; quotes must be exact raw_output substrings and source quotes exact turn text substrings.",
                "Distinguish a wrong final answer, wrong visible explanation, unsupported citation and contract failure; a wrong answer alone does not identify a mechanism.",
                "Do not infer hidden cognition. If the visible evidence cannot distinguish causes, use UNKNOWN_CAUSE.",
                "Do not read CARD_MATCH_PACKET.json until ERROR_REVIEW.json has been written and its hash recorded.",
                "Do not read F responses or overall F/G result tables. Never change a frozen question, source, Gold or official score."],
            "required_records": "8 candidate records: candidate_id, question_type, slot_id, gold_concern(bool), gold_concern_details, responses[2]",
            "required_response_fields": ["request_id", "response_valid(bool)", "strict_exact(bool)", "primary_failure_code(string|null)",
                "final_failure_codes(list)", "reasoning_error_codes(list)", "citation_defect(bool)",
                "evidence[{canonical_label_or_pair,display_label_or_pair,raw_quote,source_checks:[{turn_ref,source_quote}],assessment}]",
                "rationale", "unknown_or_competing_explanation"], "training": False, "new_solver_calls": 0})
        save(dest / "CARD_MATCH_PACKET.json", {"schema_version": "g_posthoc_original_card_match_v1", "records": card_records,
            "rules": ["Read this only after writing ERROR_REVIEW.json; bind that file before card matching.",
                "Match the existing public card intent to G content and observed errors without using F-specific option labels or loosening its trigger.",
                "Historical card scope saying not consumed is an old snapshot; it does not describe current execution status.",
                "This G target operationalization is after observing results and must be labelled exploratory, not pre-Solver registration or prospective replication.",
                "MC6 is a control. Report HOLD when the narrow trigger is not realized. Such errors cannot count as original-card mechanism hits."],
            "required_records": "8 records: candidate_id, card_id, target_realization(PASS|HOLD), control(bool), rationale, target_labels_or_pairs, responses[2]",
            "required_response_fields": ["request_id", "target_final_pattern_observed(bool|null)", "visible_mechanism_status(CONFIRMED|NOT_OBSERVED|UNKNOWN|OFF_TARGET)",
                "final_mechanism_error_confirmed(bool)", "reason_only_target_error(bool)", "citation_only_target_error(bool)",
                "evidence[{canonical_label_or_pair,display_label_or_pair,raw_quote,source_refs,assessment}]", "rationale"],
            "no_pre_solver_G_target_registration": True, "training": False})
    save(P / "INPUT_BINDINGS.json", {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bindings": bindings, "G_semantic_questions": 32, "G_response_repeats": 64,
        "frozen_experiment_modified": False, "training": False, "new_solver_calls": 0})
    protocol = """# G组完整上下文作答错误分析

本分析复用已封存闭环的32道G题、64次FULL/THINK_OFF/B1作答，不调用Solver，不训练，不改原题、Gold、源对话、正式评分或原实验manifest。完整对话始终在Solver输入中；严格exact同时检查答案和规定的证据输出合同。

阶段1：每root完整阅读来源和8题两展示，先定位具体最终错、可见解释错、引用问题和排序缺项；正确题也审，防止只看错误。保存ERROR_REVIEW.json后记录hash。

阶段2：才读对应旧公开弱点卡，将原卡窄触发映射到G题，另记目标实现、最终模式、可见机制、理由单独错和目标引用错。映射与审核均发生在分数已知后，是探索性后审，不是预注册的G目标验证。

审核者不读取F作答或F/G总体分数；角色来自同模型家族、共享文件系统，G组身份可知，不声称严格组别盲审或人类审核。两展示是同题重复测量，不作独立样本。MC6控制单列。总错答数、已确认机制数、未知原因及引用缺陷分别报告，类别可多选且不加成总错误数。

如果发现题意/Gold/来源冲突，必须保留为问题项并报告，不能改Gold或把它当Solver弱点。阶段2与既有F审核的比较仅作探索，保留原实验“未证明反馈总体更难”的结论。
"""
    with (P / "PROTOCOL_ZH.md").open("x") as stream:
        stream.write(protocol)
    print(json.dumps({"prepared_roots": list(ROOTS), "semantic_questions": 32, "responses": 64, "new_inference": False}))


if __name__ == "__main__":
    main()
