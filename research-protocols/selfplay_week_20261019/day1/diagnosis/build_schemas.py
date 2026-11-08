"""Build day-1 schema artifacts only; does not read model answers or Gold."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

tags = [
    ("obs.mc.false_positive", "observed_error", "多选误选", "预测集合中的非Gold项。"),
    ("obs.mc.false_negative", "observed_error", "多选漏选", "Gold集合中未预测的项。"),
    ("obs.mc.over_selection", "observed_error", "多选过选", "预测集合大小大于Gold集合；与误选并非同义。"),
    ("obs.mc.under_selection", "observed_error", "多选少选", "预测集合大小小于Gold集合；与漏选并非同义。"),
    ("obs.order.within_scene_pair", "observed_error", "同场景错序对", "审核确定的同场景两事项相对次序预测错误。"),
    ("obs.order.across_scene_pair", "observed_error", "跨场景错序对", "审核确定的不同场景两事项相对次序预测错误。"),
    ("obs.order.repeated_report_occurrence", "observed_error", "重复报告定位错误", "在题意已经唯一限定首次/修订/指定言语行为时选错出现位置。"),
    ("obs.display.position_sensitivity", "observed_error", "展示位置敏感", "同一题双射换位后还原到内容ID的答案不一致；记录采样不确定性。"),
    ("hyp.binding.role_action_object", "mechanism_hypothesis", "人物—行动—对象绑定", "证据中的人物、行动或对象被错配。"),
    ("hyp.binding.state_revision", "mechanism_hypothesis", "状态与修订绑定", "把旧计划、撤回项或条件性方案当作指定时点有效状态。"),
    ("hyp.binding.condition_conjunction", "mechanism_hypothesis", "条件联合与桥接", "未同时满足多个必要条件，或跨轮指代/归属桥接失败。"),
    ("hyp.evidence.retrieval", "mechanism_hypothesis", "证据定位", "在全文可见的前提下未找到相关证据；证据提示对照可以提供支持。"),
    ("hyp.reasoning.relation", "mechanism_hypothesis", "给定证据后的关系判断", "相关证据已明确提供，仍把人物、状态、条件或相对次序关系判断错。"),
    ("hyp.shortcut.lexical", "mechanism_hypothesis", "词面捷径依赖", "对同义改写、近邻用词或引用去除敏感；需最小对照而非仅看原题成绩。"),
]
excluded = [
    {"id": "excluded.invalid_gold", "label": "Gold错误", "disposition": "退回内容审核；保留原稿及改动链，不入弱点频次。"},
    {"id": "excluded.ambiguous_item", "label": "题意/排序锚点不唯一", "disposition": "HOLD或FAIL；不能按Solver预测选择Gold。"},
    {"id": "excluded.unsupported_item", "label": "证据不足", "disposition": "退回审核；未给定证据不是推理弱点。"},
    {"id": "excluded.format_defect", "label": "输出格式/映射缺陷", "disposition": "单列协议有效率；不计内容能力失败。"},
    {"id": "excluded.context_coverage_loss", "label": "memory或截断丢证据", "disposition": "单列证据覆盖损失；不计全文条件下能力弱点。"},
]
taxonomy = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:gxy-selfplay:day1:diagnosis:weakness-taxonomy:1.0",
    "title": "Diagnostic annotation taxonomy; catalog fields also document operational policy",
    "type": "object",
    "oneOf": [
        {"properties": {"kind": {"const": "observed_error"}, "tag_id": {"enum": [i for i,k,_,_ in tags if k == "observed_error"]}, "note": {"type": "string", "minLength": 1}}, "required": ["kind", "tag_id", "note"], "additionalProperties": False},
        {"properties": {"kind": {"const": "mechanism_hypothesis"}, "tag_id": {"enum": [i for i,k,_,_ in tags if k == "mechanism_hypothesis"]}, "note": {"type": "string", "minLength": 1}}, "required": ["kind", "tag_id", "note"], "additionalProperties": False},
        {"properties": {"kind": {"const": "excluded"}, "tag_id": {"enum": [x["id"] for x in excluded]}, "note": {"type": "string", "minLength": 1}}, "required": ["kind", "tag_id", "note"], "additionalProperties": False}
    ],
    "schema_version": "1.0", "artifact_type": "weakness_taxonomy_catalog", "status": "READY_FOR_TRAIN_SIDE_DIAGNOSIS_NOT_RESULTS",
    "model_baseline": "S0=M-clean; exact checkpoint binding belongs to the parent experiment protocol",
    "tags": [{"id": i, "kind": k, "label_zh": l, "definition_zh": d} for i, k, l, d in tags],
    "excluded_categories": excluded,
    "confidence_states": {
        "hypothesis": "一个或多个合格失败案例支持的机制假设，尚未达到跨root复现门槛。",
        "reproduced": "同一冻结模型上，至少3个不同Train root及3个题族的合格失败；至少一个与机制相关的受控探针提供支持。工作阈值，非统计显著性或内部因果证明。",
        "improving": "已有reproduced证据；新题族训练侧探针较前版改善并记录共同协议，仍不称Mastered。",
        "recurred": "先前改善后，新合格题族再次出现该机制；继续保留整个历史。",
        "retired": "不再用于本轮定向生成；说明原因，保留证据，不能表示能力永久掌握。"
    },
    "aggregation": {
        "frequency_unit": "unique(root_id, family_id)",
        "root_count": "distinct root_id", "family_count": "distinct(root_id, family_id)",
        "repeated_calls": "保留分母、采样率与排列一致性；不增加独立失败频次。",
        "failure_case_rule": "按预先冻结规则聚合同题的直接答案记录；不得从18次重复中只挑最差一次。",
        "root_equal_reporting": True
    },
    "known_limits": ["观察标签不等于机制原因。", "结构化解释不是忠实内部CoT的证明。", "同一root多个题族仍可能相关。", "schema约束声明；实际split、来源hash及读写权限仍需运行时审计。"]
}

def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required, "additionalProperties": False}
def arr(items, minimum=0, unique=False):
    d = {"type": "array", "items": items, "minItems": minimum}
    if unique: d["uniqueItems"] = True
    return d
def ref(name): return {"$ref": "#/$defs/" + name}
text = {"type": "string", "minLength": 1}
ident = {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"}
sha = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
timestamp = {"type": "string", "format": "date-time"}
enum = lambda xs: {"type": "string", "enum": list(xs)}
obs_ids = [i for i,k,_,_ in tags if k == "observed_error"]
hyp_ids = [i for i,k,_,_ in tags if k == "mechanism_hypothesis"]

defs = {}
defs["artifact_ref"] = obj({"path": text, "sha256": sha})
defs["train_provenance"] = obj({"split": {"const": "train"}, "namespace": {"const": "train.feedback"}, "root_id": ident, "family_id": ident, "source_id": ident, "source_sha256": sha})
defs["evidence_ref"] = obj({"source_id": ident, "source_sha256": sha, "scene_id": ident, "turn_start": {"type": "integer", "minimum": 0}, "turn_end": {"type": "integer", "minimum": 0}, "speaker": text, "quote": text})
defs["content_gate"] = obj({"decision": {"const": "PASS"}, "score_used": {"const": False}, "completed_before_solver_scoring": {"const": True}, "record": ref("artifact_ref")})
defs["diagnostic_item"] = obj({"content_id": ident, "judgment": enum(["supported", "refuted", "uncertain", "ordering_occurrence"]), "evidence_refs": arr(ref("evidence_ref")), "brief_basis": text})
defs["diagnostic_pair"] = obj({"first_content_id": ident, "second_content_id": ident, "basis": enum(["source_appearance_order", "uncertain"]), "evidence_refs": arr(ref("evidence_ref")), "brief_basis": text})
defs["solver_trace"] = obj({
    "document_type": {"const": "solver_diagnostic_trace"}, "schema_version": {"const": "1.0"},
    "trace_id": ident, "request_id": ident, "created_at": timestamp, "model_id": text,
    "provenance": ref("train_provenance"), "question_id": ident,
    "task_type": enum(["multi_select", "ordering"]), "public_question": ref("artifact_ref"),
    "source_input": ref("artifact_ref"), "display_mapping": ref("artifact_ref"),
    "protocol": ref("artifact_ref"), "gold_visible": {"const": False}, "author_evidence_visible": {"const": False},
    "output_budget_tokens": {"type": "integer", "minimum": 1}, "actual_output_tokens": {"type": "integer", "minimum": 0},
    "per_item": arr(ref("diagnostic_item"), 1), "order_pairs": arr(ref("diagnostic_pair")),
    "final_prediction_content_ids": arr(ident, 1, True), "format_valid": {"type": "boolean"},
    "raw_response": ref("artifact_ref")
})
defs["failure_case"] = obj({
    "failure_id": ident, "question_id": ident, "model_id": text, "provenance": ref("train_provenance"),
    "content_gate": ref("content_gate"), "format_valid": {"const": True}, "full_evidence_visible": {"const": True},
    "observed_errors": arr(enum(obs_ids), 1, True), "evidence_refs": arr(ref("evidence_ref"), 1),
    "direct_answer_record": ref("artifact_ref"), "diagnostic_trace": {"oneOf": [ref("artifact_ref"), {"type": "null"}]},
    "postgold_evaluation": ref("artifact_ref"),
    "repeat_aggregation": obj({"protocol": ref("artifact_ref"), "valid_calls": {"type": "integer", "minimum": 1}, "content_failures": {"type": "integer", "minimum": 1}, "failed_display_mappings": {"type": "integer", "minimum": 0}, "total_display_mappings": {"type": "integer", "minimum": 1}})
})
defs["competing_explanation"] = obj({"explanation": text, "status": enum(["not_tested", "supported", "weakened", "unresolved"]), "probe_ids": arr(ident), "interpretation_limit": text})
defs["probe"] = obj({"probe_id": ident, "type": enum(["option_permutation", "semantic_paraphrase", "evidence_hint", "minimal_relation_contrast", "new_root_transfer"]), "provenance": ref("train_provenance"), "compared_model_id": text, "frozen_protocol": ref("artifact_ref"), "record": ref("artifact_ref"), "outcome": enum(["supports_hypothesis", "weakens_hypothesis", "inconclusive"]), "confounds": arr(text)})
defs["target_design"] = obj({"design_id": ident, "target_relation": text, "needed_evidence": arr(text, 1), "distractor_or_order_mechanism": text, "new_fact_or_root_requirement": text, "ordering_definition": enum(["source_appearance_order", "not_applicable"]), "avoid_shortcuts": arr(text, 1), "content_gate_before_score": {"const": True}})
defs["attempt"] = obj({"attempt_id": ident, "design_id": ident, "model_id": text, "candidate_ledger": ref("artifact_ref"), "all_attempts_preserved": {"const": True}, "candidate_count": {"type": "integer", "minimum": 0}, "admitted_count": {"type": "integer", "minimum": 0}, "rejected_count": {"type": "integer", "minimum": 0}, "held_count": {"type": "integer", "minimum": 0}, "score_based_filtering": {"const": False}, "input_tokens": {"type": "integer", "minimum": 0}, "output_tokens": {"type": "integer", "minimum": 0}})
defs["new_question_result"] = obj({"result_id": ident, "provenance": ref("train_provenance"), "new_family_for_memory": {"const": True}, "before_model_id": text, "after_model_id": text, "same_protocol": {"const": True}, "before_exact_rate": {"type": "number", "minimum": 0, "maximum": 1}, "after_exact_rate": {"type": "number", "minimum": 0, "maximum": 1}, "record": ref("artifact_ref"), "interpretation": text})
defs["cause"] = obj({
    "cause_id": ident, "mechanism_tag": enum(hyp_ids), "mechanism": text, "confidence_status": enum(taxonomy["confidence_states"]),
    "model_version": text, "supporting_failure_ids": arr(ident, 1, True), "supporting_root_ids": arr(ident, 1, True),
    "supporting_family_keys": arr(text, 1, True), "evidence_refs": arr(ref("evidence_ref"), 1),
    "competing_explanations": arr(ref("competing_explanation"), 1), "probe_ids": arr(ident, 0, True),
    "targeted_designs": arr(ref("target_design"), 1), "attempts": arr(ref("attempt")),
    "new_question_results": arr(ref("new_question_result")),
    "state_history": arr(obj({"at": timestamp, "model_id": text, "from": enum(["none"] + list(taxonomy["confidence_states"])), "to": enum(taxonomy["confidence_states"]), "reason": text, "record": ref("artifact_ref")}), 1),
    "limitations": arr(text, 1)
})
defs["cause"]["allOf"] = [
    {"if": {"properties": {"confidence_status": {"enum": ["reproduced", "improving", "recurred"]}}},
     "then": {"properties": {"supporting_root_ids": {"minItems": 3}, "supporting_family_keys": {"minItems": 3}, "probe_ids": {"minItems": 1}}}},
    {"if": {"properties": {"confidence_status": {"const": "improving"}}},
     "then": {"properties": {"new_question_results": {"minItems": 1}}}}
]
defs["memory_snapshot"] = obj({
    "document_type": {"const": "challenger_weakness_memory"}, "schema_version": {"const": "1.0"},
    "snapshot_id": ident, "created_at": timestamp, "model_id": text,
    "namespace": {"const": "train.feedback"}, "audience": {"const": "challenger"},
    "taxonomy": ref("artifact_ref"), "allowed_train_roots_manifest": ref("artifact_ref"),
    "parent_snapshot": {"oneOf": [ref("artifact_ref"), {"type": "null"}]},
    "frequency_unit": {"const": "unique(root_id,family_id)"}, "reproduction_min_distinct_roots": {"const": 3},
    "failure_cases": arr(ref("failure_case")), "probes": arr(ref("probe")), "causes": arr(ref("cause")),
    "eval_error_feedback_included": {"const": False}, "known_limits": arr(text, 1)
})
defs["postgold_item"] = obj({"content_id": ident, "gold_selected": {"type": "boolean"}, "predicted_selected": {"type": "boolean"}, "classification": enum(["TP", "FP", "FN", "TN"]), "reference_evidence": arr(ref("evidence_ref"), 1)})
defs["postgold_pair"] = obj({"first_content_id": ident, "second_content_id": ident, "scene_relation": enum(["within_scene", "across_scene"]), "predicted_correct": {"type": "boolean"}, "reference_evidence": arr(ref("evidence_ref"), 1)})
defs["postgold_evaluation"] = obj({
    "document_type": {"const": "postgold_diagnostic_evaluation"}, "schema_version": {"const": "1.0"},
    "evaluation_id": ident, "created_at": timestamp, "evaluator_id": text, "model_id": text,
    "provenance": ref("train_provenance"), "question_id": ident, "task_type": enum(["multi_select", "ordering"]),
    "content_gate": ref("content_gate"), "frozen_solver_trace": ref("artifact_ref"), "solver_trace_committed_before_gold_read": {"const": True},
    "gold_reference": ref("artifact_ref"), "direct_answer_record": ref("artifact_ref"),
    "per_item": arr(ref("postgold_item")), "order_pairs": arr(ref("postgold_pair")),
    "observed_errors": arr(enum(obs_ids), 0, True), "cause_hypotheses": arr(enum(hyp_ids), 0, True),
    "competing_explanations": arr(ref("competing_explanation")), "explanation_faithfulness_assumed": {"const": False}, "limitations": arr(text, 1)
})

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:gxy-selfplay:day1:diagnosis:memory-schema:1.0",
    "title": "Train-only challenger weakness memory and diagnostic records",
    "description": "Schemas for strict train.feedback artifacts. Semantic/root/lineage checks additionally required; schema declarations alone cannot prove data isolation.",
    "oneOf": [ref("memory_snapshot"), ref("solver_trace"), ref("postgold_evaluation")], "$defs": defs
}
for filename, value in [("WEAKNESS_TAXONOMY.json", taxonomy), ("MEMORY_SCHEMA.json", schema)]:
    (BASE / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
fact_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:gxy-selfplay:day1:diagnosis:fact-memory-schema:1.0",
    "title": "Prefix-derived fact memory for conversation expansion; separate from weakness memory",
    **obj({
        "document_type": {"const": "prefix_fact_memory"}, "schema_version": {"const": "1.0"},
        "snapshot_id": ident, "created_at": timestamp, "root_id": ident,
        "namespace": {"const": "train.expansion.fact_memory"},
        "audience": {"const": "expansion_planner_realizer"},
        "source_id": ident, "source_sha256": sha, "accepted_prefix": ref("artifact_ref"),
        "prefix_last_turn": {"type": "integer", "minimum": 0},
        "prefix_content_gate": ref("artifact_ref"),
        "gold_or_question_conditioned": {"const": False}, "future_turns_read": {"const": False},
        "parent_snapshot": {"oneOf": [ref("artifact_ref"), {"type": "null"}]},
        "facts": arr(obj({
            "fact_id": ident, "fact_type": enum(["person", "relation", "event", "item", "state", "open_issue"]),
            "subject_ids": arr(ident, 1, True), "claim": text,
            "validity": enum(["active", "superseded", "cancelled", "uncertain"]),
            "valid_from_turn": {"type": "integer", "minimum": 0},
            "valid_until_turn": {"type": ["integer", "null"], "minimum": 0},
            "supersedes_fact_ids": arr(ident, 0, True),
            "evidence_refs": arr(ref("evidence_ref"), 1), "uncertainty": {"type": "string"}
        })),
        "known_limits": arr(text, 1)
    }),
    "$defs": {"artifact_ref": defs["artifact_ref"], "evidence_ref": defs["evidence_ref"]}
}
(BASE / "FACT_MEMORY_SCHEMA.json").write_text(json.dumps(fact_schema, ensure_ascii=False, indent=2) + "\n")
print("Wrote taxonomy catalog and draft-2020-12 schema; no model data read.")
