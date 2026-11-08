"""Assemble already authored/reviewed single-rollout feedback; no inference calls."""
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parent
ROOTS = ("218519", "222937", "225575", "229686")


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def decision(row):
    return row.get("verdict", row.get("decision", row.get("status")))


THEMES = {
    "ENTITY_ACTION_BINDING": {
        "name_zh": "人物—具体动作绑定",
        "card_ids": ["CAFB-7c31-person-binding"],
        "material_relations": ["同一场景的两位既有人物分别执行可区分的具体子动作；人物、对象与动作在发言中明确绑定。", "保留共享话题及一般活动，具体子动作不能仅靠一般活动推定。"],
        "paired_material_conditions": ["保留具体人物与动作的对应关系。", "若以后创建新的独立素材对照，只改变某一具体子动作的执行者，其余条件一致并重新审核。"],
    },
    "CAUSAL_ATTACHMENT": {
        "name_zh": "原因—决定—对象绑定",
        "card_ids": ["CAFB-7c31-cause-binding", "CA225575-03", "CFA-R229686-CAUSAL-v1"],
        "material_relations": ["为两个不同对象或决定分别提供明示原因，记录原因究竟解释哪个动作。", "至少一处真实决定与另一个真实原因同时存在，但来源不把二者连成同一因果关系。"],
        "paired_material_conditions": ["原因及决定分别有明确来源；保持原有真实因果边。", "未来对照可改变一条因果边，必须在新素材中自然、明确表达，不能改写旧事实。"],
    },
    "QUANTITY_UNIT_OPERAND": {
        "name_zh": "数量、单位及运算对象绑定",
        "card_ids": ["CF-218519-01", "CAFB-7c31-quantity-binding", "CA225575-04", "CFA-R229686-QUANTITY-v1"],
        "material_relations": ["数量分别绑定人物/对象、单位、时间端点与场景；十二/二十四小时表达若共存，含义明确。", "人数、时段及覆盖范围分别表达；若需要组合单位，明确同一批人覆盖哪些时段。", "直接数量、差值、比例及组合单位分别可以独立核验，不要求把多个表现视为同一机制。"],
        "paired_material_conditions": ["单一数量直接查找与联合两个数量的关系分别保留。", "未来按一个操作或单位设置对照，其余数量与实体绑定保持可核验。"],
    },
    "CLAIM_JUDGMENT_COHERENCE": {
        "name_zh": "完整命题与判断一致性",
        "card_ids": ["CF-218519-02", "CA225575-02"],
        "material_relations": ["人物、事件及约束分别明示，支持核验完整复合关系，不能仅由其中一个真事实决定整句。", "保留具有明确状态和角色边界的正反事实，避免语用歧义。"],
        "paired_material_conditions": ["完整关系有来源支持。", "未来对照只改变其中一个有明确源依据的约束，其余关系保持一致。"],
    },
    "TASK_TIME_AXIS": {
        "name_zh": "题设比较轴遵循",
        "card_ids": ["CFA-R229686-CLOCK_AXIS-v1"],
        "material_relations": ["相关事件同时具有明确钟面时刻和日期/星期信息；若有修正，明确旧计划与新计划。", "钟面大小与实际日历先后可以区分；既有来源前缀与事实保持不变。"],
        "paired_material_conditions": ["材料保留两种可核验的时间信息。", "后续题面分别声明比较轴；素材作者不写正确排列或答案提示。"],
    },
    "DISCOURSE_ORDER": {
        "name_zh": "话语贡献定位与排序一致性",
        "card_ids": ["CF-218519-04", "CA225575-06", "CFA-R229686-ORDER_AGGREGATION-v1"],
        "material_relations": ["实际发言先后可追溯，每个目标贡献在完整动作/宾语/后续请求组合上唯一可定位。", "允许自然的问答、拒绝或总结功能重复，但具体对象和贡献功能有清晰区别。"],
        "paired_material_conditions": ["不同贡献可共享动作词，而完整宾语及动作组合保持可区分。", "不能为了取消词面线索删去真实问答依赖；素材不附带待排序项或正确排列。"],
    },
    "EVIDENCE_STAGE_BINDING": {
        "name_zh": "证据位置、覆盖与阶段绑定",
        "card_ids": ["CF-218519-03", "CAFB-7c31-evidence-location", "CA225575-01", "CFA-R229686-REASON_EVIDENCE-v1"],
        "material_relations": ["问题、提议、同意、确认、完成等阶段若出现，应分别明确表达并标明对应人物/对象。", "复合事实的每个必要部分均有来源；相邻同话题发言不必承担相同证明作用。", "局部动作先后明确，避免仅靠常识猜测完成状态。"],
        "paired_material_conditions": ["单阶段直接事实与需要跨阶段联合的事实分别可核验。", "正常问答和计划完成顺序保持自然，不向素材作者提供Solver错答或预期错误。"],
    },
    "UNSUPPORTED_ELABORATION": {
        "name_zh": "辅助理由添加未述事实",
        "card_ids": ["CA225575-05"],
        "material_relations": ["区分将来意向与已经完成的动作，必要时提供明确的未完成限定。", "两位既有人物的相近物品或动作状态分别明示，不由一人的情况推导另一人。"],
        "paired_material_conditions": ["主要事实有窄而直接的证据。", "未来受控素材可只改变一个动作的计划/完成状态；日常赠礼交付语义不清时不作为检验目标。"],
    },
}
THEME_BY_CARD = {fid: key for key, value in THEMES.items() for fid in value["card_ids"]}


REVISION_JUDGMENTS = {
    "CF-218519-04": "已核对原ORDER3错误及来源：早先拒绝参考号不同于后面拒绝餐桌预订并索取联系方式。补丁只保留完整贡献错配及受其影响的相对顺序，删除不确定的MC3泛称reference支持及多选范围。",
    "CA225575-05": "已核对两个保留观察：I'll look被写成looked；另一人物已找到自己的物品却被说未找到。补丁排除赠礼交付语用UNKNOWN，仅保留两个明确的理由附加事实，未改最终答案评分。",
    "CFA-R229686-QUANTITY-v1": "两项比例/人夜错误原证据成立。补丁撤回确认题目缺陷，改成前瞻单操作控制；两类表现分开检验，不认定共同内部机制。",
    "CFA-R229686-REASON_EVIDENCE-v1": "保留原MC4局部先后反转和MC6约定/后续确认引用覆盖两项；隔离MC5空引文UNKNOWN，收窄为明确阶段关系及引用绑定，分开两种表现且不认定共享内部过程。",
}


def main():
    selected = read(P / "SELECTION_FREEZE.json")
    contract = read(P / "REVIEW_CONTRACT.json")
    all_records, all_cards, all_public, card_audit, root_stats = [], [], [], [], []
    all_errors, all_reviews, source_author_tasks = {}, {}, []
    reviewed_root_bindings = []
    for root in ROOTS:
        adir = P / "authoring" / ("root" + root)
        rdir = P / "review" / ("root" + root)
        revdir = P / "revisions" / ("root" + root)
        content_path = P / "inputs" / ("cc-train-episode-" + root + "_CONTENT_FIRST.json")
        private_path = P / "inputs" / ("cc-train-episode-" + root + "_SCORING_AND_RESPONSES_PRIVATE.json")
        content, private = read(content_path), read(private_path)
        cases = read(adir / "CASE_REVIEWS.json")["records"]
        review = read(rdir / "REVIEW.json")
        cards = read(adir / "CANDIDATE_CARDS_PRIVATE.json")["cards"]
        public = read(adir / "CHALLENGER_PUBLIC_CARDS.json")["cards"]
        qmap = {row["candidate_id"]: row for row in content["questions"]}
        pmap = {row["candidate_id"]: row for row in private["records"]}
        errors = {e["error_id"]: e for c in cases for e in c["observed_errors"]}
        ermap = {e["error_id"]: e for e in review["error_reviews"]}
        crmap = {e["feedback_id"]: e for e in review["card_reviews"]}
        case_rmap = {e["candidate_id"]: e for e in review["case_reviews"]}
        assert set(ermap) == set(errors)
        all_errors.update(errors)
        all_reviews.update(ermap)
        patches, ppatches = {}, {}
        if revdir.exists():
            patches = {c["feedback_id"]: c for c in read(revdir / "CANDIDATE_CARDS_PRIVATE.json")["cards"]}
            ppatches = {c["feedback_id"]: c for c in read(revdir / "CHALLENGER_PUBLIC_CARDS.json")["cards"]}
            assert set(patches) == set(ppatches)
            assert set(patches) == {fid for fid, row in crmap.items() if decision(row) == "REVISE"}
            rr = []
            for fid, card in patches.items():
                assert fid in REVISION_JUDGMENTS
                assert all(decision(ermap[eid]) == "PASS" for eid in card["supporting_error_ids"])
                pc = ppatches[fid]
                assert all(pc[k] == card["solver_feedback"][k] for k in card["solver_feedback"])
                assert all(pc[k] == card[k] for k in ("questioner_feedback", "extension_task", "question_task", "limitations"))
                assert set(pc).issubset(contract["public_fields"])
                rr.append({"feedback_id": fid, "decision": "ACCEPT", "reason": REVISION_JUDGMENTS[fid],
                           "supporting_error_ids": card["supporting_error_ids"], "candidate_maturity": "HYPOTHESIS",
                           "original_review_disposition_preserved": "REVISE", "public_projection_synchronized": True,
                           "independent_of_revision_author": True, "limitations": "主代理复核其他Codex代理的修订；共享模型家族及文件系统，不是外部或人审认证。"})
            rb = [binding(revdir / f) for f in ("CANDIDATE_CARDS_PRIVATE.json", "CHALLENGER_PUBLIC_CARDS.json", "REVISION_NOTES.json")]
            if (revdir / "CASE_AUDIT_OVERLAY.json").exists():
                rb.append(binding(revdir / "CASE_AUDIT_OVERLAY.json"))
            write(str((rdir / "REVISION_REVIEW.json").relative_to(P)), {"reviewer": "Codex root reviewing another agent's revision", "bindings": rb, "cards": rr, "status": "PASS_CANDIDATE_ONLY"})
        overlays = read(revdir / "CASE_AUDIT_OVERLAY.json")["records"] if (revdir / "CASE_AUDIT_OVERLAY.json").exists() else []
        for case in cases:
            cid = case["candidate_id"]
            raw = pmap[cid]
            observations = []
            for err in case["observed_errors"]:
                verdict = decision(ermap[err["error_id"]])
                observations.append({"observation": copy.deepcopy(err), "semantic_review": copy.deepcopy(ermap[err["error_id"]]),
                                     "review_status": verdict, "admitted_to_candidate_support": verdict == "PASS"})
            qfindings = copy.deepcopy(case["questioner_findings"])
            if cid == "PDF1-229686-MC3":
                for f in qfindings:
                    if f["issue_id"] == "QF-R229686-MC3-ISOLATION":
                        f["author_status_preserved"] = f["status"]
                        f["status"] = "PROSPECTIVE_CONTROL_SUGGESTION"
                        f["review_overlay"] = "撤回确认缺陷：复合题有效且已有逐项诊断；建议另设受控对照，未确认SOURCE_SHORTFALL。"
            all_records.append({"candidate_id": cid, "root_id": content["root_id"], "question": qmap[cid],
                "source_status": content["source_status"], "source_rounds": content["source_rounds"],
                "bindings": {"source": content["source_binding"], "raw_response": raw["response_binding"],
                             "prompt": raw["prompt_binding"], "schema": raw["schema_binding"],
                             "content_input": binding(content_path), "response_input": binding(private_path),
                             "final_text_sha256": hashlib.sha256(raw["final_text"].encode()).hexdigest()},
                "solver_selection": {"model": private["solver_model"], "thinking": private["thinking"],
                                     "condition": private["condition"], "display": private["display"], "request_id": raw["request_id"], "rollout_count": 1},
                "frozen_scoring": raw["scoring"], "original_final_text": raw["final_text"],
                "author_case": case, "exchange_case_review": case_rmap[cid],
                "effective_observations": observations, "effective_questioner_findings": qfindings,
                "item_dimension_overlays": [row for row in overlays if row["candidate_id"] == cid],
                "aggregation_rule": "原作者记录保留；有效语义结论采用审核及显式sidecar，未知不作高置信支持。"})
        for card, pc in zip(cards, public):
            fid = card["feedback_id"]
            assert fid == pc["feedback_id"]
            effective = copy.deepcopy(patches.get(fid, card))
            effective_public = copy.deepcopy(ppatches.get(fid, pc))
            assert decision(crmap[fid]) in ("ACCEPT", "REVISE")
            assert effective["status"] == "HYPOTHESIS"
            assert all(decision(ermap[eid]) == "PASS" for eid in effective["supporting_error_ids"])
            assert effective["extension_task"]["needed"] is False
            assert not effective["extension_task"]["material_targets"]
            all_cards.append(effective)
            all_public.append(effective_public)
            card_audit.append({"feedback_id": fid, "initial_decision": decision(crmap[fid]), "effective_decision": "ACCEPT",
                "effective_version": "review_revision_v2" if fid in patches else "author_v1",
                "effective_card_binding": binding((revdir if fid in patches else adir) / "CANDIDATE_CARDS_PRIVATE.json"),
                "effective_public_binding": binding((revdir if fid in patches else adir) / "CHALLENGER_PUBLIC_CARDS.json"),
                "review_binding": binding(rdir / ("REVISION_REVIEW.json" if fid in patches else "REVIEW.json")),
                "supporting_error_ids": effective["supporting_error_ids"], "maturity": "HYPOTHESIS",
                "formal_weakness_admitted": False, "theme": THEME_BY_CARD[fid]})
            theme = THEMES[THEME_BY_CARD[fid]]
            source_author_tasks.append({"task_id": fid + "-prospective-material", "purpose": "TEST_HYPOTHESIS", "split": "train",
                "neutral_material_objectives": theme["material_relations"], "contrast_conditions": theme["paired_material_conditions"],
                "scene_sequence": ["A1", "A2", "A3", "B1", "B2", "B3"],
                "continuity_constraints": ["保留所有既有前缀与父事实，明确人物、对象、时间、状态。", "如进入下一次扩充，沿用既有A1–A3后B1–B3的顺序累计协议；不是独立分支或重置旧轮。", "新人物的身份属性须有依据；没有必要不引入新人物。", "不得把合成编辑版本称为恢复原始真实事实。"],
                "material_acceptance_tests": ["完整来源独立审核每个所需关系、事实连续性及唯一可追溯位置。", "任一控制条件或实体绑定不清先返修，不把缺关系记为Solver错误。"]})
        root_stats.append({"root_id": content["root_id"], "source_status": content["source_status"], "questions": len(cases),
            "items": sum(len(c["item_checks"]) for c in cases), "final_correct": sum(pmap[c["candidate_id"]]["scoring"]["exact"] for c in cases),
            "initial_observations": len(errors), "observation_review": dict(Counter(decision(x) for x in ermap.values())),
            "initial_card_review": dict(Counter(decision(x) for x in crmap.values())), "effective_candidate_cards": len(cards)})
        reviewed_root_bindings.append(binding(rdir / "REVIEW.json"))
    assert len(all_records) == 30 and len(all_cards) == 19
    assert len(all_errors) == 56 and sum(decision(x) == "PASS" for x in all_reviews.values()) == 53
    assert sum(len(x["author_case"]["item_checks"]) for x in all_records) == 168
    assert {x["candidate_id"] for x in all_records} == {x["candidate_id"] for x in selected["records"]}
    summary = {"schema_version": "codex_route_a_feedback_summary_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": "Asia/Shanghai", "split": "train", "selection": {k: v for k, v in selected.items() if k not in ("records", "input_bindings")},
        "roots": root_stats, "questions": 30, "items_reviewed": 168,
        "final_answers": {"correct": 16, "incorrect": 14, "multi_select": {"correct": 10, "total": 20}, "ordering": {"correct": 6, "total": 10}},
        "observations": {"initial": 56, "reviewed_pass": 53, "unknown": 3, "fail": 0,
            "accepted_answer_error_units": sum(e["status"] == "OBSERVED_ERROR" and decision(all_reviews[eid]) == "PASS" for eid, e in all_errors.items()),
            "accepted_reason_or_citation_units": sum(e["status"] == "OBSERVED_REASON_ERROR" and decision(all_reviews[eid]) == "PASS" for eid, e in all_errors.items()),
            "unknown_error_ids": [eid for eid in all_errors if decision(all_reviews[eid]) == "UNKNOWN"],
            "denominator_note": "观察为逐项错误/理由/引用单元，可以同题多项；不是53道独立错题，也不改14道整题错误分母。"},
        "cards": {"initial": 19, "initial_accept": 15, "initial_revise": 4, "revisions_independently_reviewed": 4,
                  "effective_accepted_candidates": 19, "mechanism_themes": 8, "formal_reproduced_weaknesses": 0},
        "confirmed_question_or_gold_defects": 0, "confirmed_source_shortfalls": 0,
        "new_execution": {"solver_calls": 0, "extensions": 0, "question_generation": 0, "training_runs": 0, "challenger_consumption": False, "runtime_modified": False},
        "review_limits": ["同源Codex分上下文作者与交换审核；不是外部或人类独立认证，也非严格盲审。", "30题已暴露的训练来源、4 roots、单模式单排列单输出；不是总体正确率或未见root泛化结果。", "诊断基于可见理由和引文，不把报告理由当成忠实内部思维。", "19卡是按root保留的候选记录；8主题为描述性归类，不能视作19或8个正式复现弱点。"]}
    write("PRIVATE_AUDIT_RECORDS.json", {"schema_version": "codex_route_a_effective_private_audit_v1", "selection_binding": binding(P / "SELECTION_FREEZE.json"), "records": all_records})
    write("DUAL_FEEDBACK_CARDS_PRIVATE.json", {"schema_version": "codex_route_a_effective_feedback_cards_v1", "cards": all_cards, "card_audit": card_audit, "formal_weakness_admitted": False, "feedback_consumed": False})
    write("CHALLENGER_PUBLIC_CARDS.json", {"schema_version": "codex_route_a_challenger_projection_v1", "scope": "train_only_exploratory", "cards": all_public, "consumed": False})
    write("SOURCE_AUTHOR_EXTENSION_TASKS.json", {"schema_version": "prospective_source_author_projection_draft_v1", "scope": "DESIGN_ONLY_NOT_RUNTIME_CONNECTED", "tasks": source_author_tasks, "consumed": False,
        "execution_metadata_not_for_author": {"execution_status": "PROPOSED_NOT_EXECUTED", "current_source_shortfall_confirmed": False,
            "existing_runtime_action": "REUSE", "missing_execution_fields": ["new_root_and_source_binding", "controlled_variable_and_invariants_per_probe", "design_freeze_binding", "generation_budget", "consumption_receipt_plan"],
            "note": "19张材料关系草案，不是19个已就绪扩充请求；执行前须选择新root、实例化单一变量对照及冻结预算。"}})
    write("MECHANISM_THEME_INDEX.json", {"status": "DESCRIPTIVE_GROUPING_NOT_REPRODUCTION", "themes": THEMES})
    write("QUESTIONER_FEEDBACK.json", {"schema_version": "codex_route_a_questioner_feedback_v1", "confirmed_question_or_gold_defects": [], "confirmed_source_shortfalls": [],
        "review_overlay": {"issue_id": "QF-R229686-MC3-ISOLATION", "initial_status_preserved": "CONFIRMED", "effective_status": "PROSPECTIVE_CONTROL_SUGGESTION", "reason": "题目有效且已有逐项证据，原作者确认缺陷强于证据。"},
        "prospective_design_recommendations": [
            {"id": "PROSPECTIVE-MECHANISM-CONTROLS", "status": "DESIGN_RECOMMENDATION", "requirement": "每次明确目标关系，配保留/移除触发条件的受控项；逐项诊断，不只报告整题exact。", "acceptance": "独立来源审核确认目标关系及Gold，再看Solver是否发生预期可见错误。"},
            {"id": "PROSPECTIVE-DIVERSITY", "status": "DESIGN_RECOMMENDATION", "requirement": "保留7种操作，后续增加同操作下的关系实现；不把换皮、排列或同root改写当独立root。", "acceptance": "记录目标关系与表面形式；不自动把Gold6控制或必要程序词判成坏题。"},
            {"id": "PROSPECTIVE-ANSWER-REASON-CITATION", "status": "DESIGN_RECOMMENDATION", "requirement": "分别评分最终答案、理由事实、理由对命题的蕴含、引用覆盖及排序一致性。", "acceptance": "真理由/错判断和答对/错引文可独立标记；语用不清项留UNKNOWN。"}],
        "per_card_feedback": [{"feedback_id": c["feedback_id"], "questioner_feedback": c["questioner_feedback"], "question_task": c["question_task"]} for c in all_cards],
        "scope": "没有证据就不强凑缺陷；设计建议不代表既有题目错误或素材不足。"})
    write("SUMMARY.json", summary)
    make_reports(summary, all_records, all_cards)
    print(json.dumps({"status": "ASSEMBLED", "questions": len(all_records), "observations": summary["observations"], "cards": summary["cards"]}, ensure_ascii=False))


def make_reports(s, records, cards):
    header = """# Codex A路线反馈模板试填报告

本轮完成反馈模板设计和一批实际诊断。没有执行下一轮扩充、出题或训练；Challenger尚未消费这些卡，现有pipeline未改。

## 数据与方法

- 冻结4个Train roots的30道语义题，20多选、10排序、7种操作；多选Gold2/3/4/5/6各4题。
- 每题只读B1 / FULL / THINK_OFF / perm_00的一条已有回答；全部保留，包括答对题。没有新增Solver调用，也不读其他排列或thinking-on对照来诊断。
- 先保存基于完整来源的独立推导，再核对Gold、逐项理由和引文；作者与交换审核为不同Codex上下文。历史上下文存在曝光，因此不是严格盲审，也不是外部人审认证。
- 使用v125累计来源：R01三段、R02:A三段、R02:B三段，两个阶段累计9段新增对话。A、B顺序累计，不是独立分支。218519和229686为已声明的EDITED_SYNTHETIC，只认证提供版本中的关系。

## 结果

| root | 题数 | 逐项数 | 最终正确 | 最终错误 | 确认观察 | 不确定 | 候选卡 |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for x in s["roots"]:
        header += f"| {x['root_id'].split('-')[-1]} | {x['questions']} | {x['items']} | {x['final_correct']} | {x['questions']-x['final_correct']} | {x['observation_review'].get('PASS',0)} | {x['observation_review'].get('UNKNOWN',0)} | {x['effective_candidate_cards']} |\n"
    header += """
合计逐项核验168项；最终16/30正确，多选10/20，排序6/10。此为模板开发批，不是新评测或总体模型能力估计。

56条初始观察经交换审核保留53条（22个影响答案的错误单元、31个理由/引用单元），另3条UNKNOWN隔离。观察可以同题多条，不能说有53道错题；整题错误仍是14道。19张初卡中15张直接接受、4张要求收窄；4张修订经非修订作者复核后，当前19张均可作为探索候选。19张按root保留，不是19种独立弱点。

## 找到的候选机制

| 主题 | 已观察的行为 | 后续怎么检验 |
|---|---|---|
| 人物—动作绑定 | 理由说对执行者，却接受把具体动作换到另一人物的选项 | 共享话题、只替换执行者的配对项 |
| 原因—决定—对象 | 决定是真的，但把另一个对象的真实原因接到它上面 | 真实因果边和跨对象错接边分别检验 |
| 数量/单位/运算对象 | 用入住人数否定人夜总数；计算时选到别的实体或时间端点 | 直接检索、比例、差值、组合单位分开控制 |
| 命题—判断一致 | 理由中的事实为真，却没有落实为对完整命题的正确支持/排除 | 事实内容、命题蕴含、最终选择分开打分 |
| 题设比较轴 | 题目只按钟面排序，理由及序列混入星期或计划顺序 | 同素材分别问钟面轴与实际日历轴 |
| 话语定位/顺序一致 | 共享“拒绝”词但对象不同导致定位错；或引文坐标全对、最终顺序仍错 | 完整贡献定位与由坐标形成排列分别验收 |
| 证据位置/阶段覆盖 | 用问句证明答复，用较早约定证明后续确认，或只覆盖复合断言的一部分 | 明确阶段、最小充分证据及全部必要从句 |
| 理由补造事实 | 最终排除正确，却把“将看菜单”写成已看，或补造另一人物未找到物品 | 最终答案与理由附加断言分别审核 |

主题是描述性归类。正确反例、其他可能解释及否证条件保留在每张卡中；没有从单条回答推断隐藏思维或稳定能力缺陷。

## 反馈模板怎么用

1. 私有审计绑定题目、来源、模型设置、原始回答与哈希；完整核验题意/Gold后才诊断Solver。
2. 将“具体可观察错误”与“候选触发关系”分开；写预期可见错误、正确行为、边界/反例、竞争解释和否证条件。
3. 出题反馈只登记有证据的问题，允许为空。未确认素材缺口或Gold缺陷；原数量卡的“已确认机制隔离问题”已撤回为前瞻对照建议。
4. Challenger仅收抽象机制与出题验收，不收原题Gold、标签、正确顺序或原始回答。
5. 对话作者另收仅含材料关系和连续性要求的投影，不收Solver错答、预期错误或未来Gold配额。

## 扩充接口的关键区别

**当前旧流程：** 所有卡needed=false、material_targets=[]、REUSE_WITH_PROSPECTIVE_PROBE。原因是没有确认SOURCE_SHORTFALL，并非反馈不能用于下一轮扩充。

**后续前瞻路径：** 为检验候选假设，可在下一次冻结计划中选择TEST_HYPOTHESIS，按既有A1–A3再B1–B3协议增加自然关系及对照。它与修补来源缺口的REPAIR_SOURCE_SHORTFALL不同。接口已设计成独立草案，未修改或假装接入现有runtime。SOURCE_AUTHOR_EXTENSION_TASKS.json是前瞻任务草案，不是已执行扩充。

## 结论边界与下一步

本轮证明“可以从现有可见回答中，形成逐项有据、经复核、可操作的候选反馈”；尚未证明反馈指导Challenger能稳定生成新弱点题，更未证明训练修复或因果收益。正式复现弱点数仍为0。

下一步用少量独立Train roots完成反馈卡→前瞻扩充/复用→新题→同一冻结Solver复测；同时保留去除触发条件及通用/错配反馈对照。新题先审来源、Gold和目标实现，再看是否出现预期错误，不能只把准确率下降当“挖掘成功”。本轮止于模板与试填交付。
"""
    (P / "REPORT_ZH.md").write_text(header)
    examples = [("人物与具体动作", "PDF1-222937-MC2", "ERR-222937-MC2-02", "CAFB-7c31-person-binding"),
                ("拒绝原因替换", "PDF1-229686-MC2", "R229686-MC2-E01", "CFA-R229686-CAUSAL-v1"),
                ("组合单位退化为人数", "PDF1-229686-MC3", "R229686-MC3-E02", "CFA-R229686-QUANTITY-v1"),
                ("理由真实但判断错误", "PDF1-218519-MC6", "E218519-MC6-01", "CF-218519-02"),
                ("正确坐标与最终排序不一致", "PDF1-229686-ORDER3", "R229686-ORDER3-E01", "CFA-R229686-ORDER_AGGREGATION-v1"),
                ("局部判断正确仍补造完成动作", "PDF1-225575-MC3", "A225575-MC3-E03", "CA225575-05")]
    rmap = {r["candidate_id"]: r for r in records}
    cmap = {c["feedback_id"]: c for c in cards}
    text = "# 反馈试填的具体例子（私有审计版）\n\n以下题面、回答原句与来源来自固定单条回答。这里展示原题标签和事实用于人工追溯；不得把本文件直接放进Challenger提示。对外生成使用CHALLENGER_PUBLIC_CARDS.json。来源中的合成编辑声明沿用，不声称恢复原始事实。\n"
    for n, (title, cid, eid, fid) in enumerate(examples, 1):
        record, card = rmap[cid], cmap[fid]
        obs = next(e for e in record["effective_observations"] if e["observation"]["error_id"] == eid)
        err, q = obs["observation"], record["question"]
        text += f"\n## {n}. {title}\n\n题目：{cid}；反馈：{fid}；来源状态：{record['source_status']}。\n\n题干：{q['question']}\n\n"
        for label in err["affected_labels"]:
            if label in q["display"]:
                text += f"- 相关项 {label}：{q['display'][label]}\n"
        text += "\n模型原句：\n\n```text\n" + err["solver_quote"] + "\n```\n\n关键来源：\n\n"
        content = read(P / "inputs" / (record["root_id"] + "_CONTENT_FIRST.json"))
        tmap = {t["turn_ref"]: t for t in content["source"]["turns"]}
        for evidence in err["source_evidence"]:
            t = tmap[evidence["turn_ref"]]
            text += f"- `{evidence['turn_ref']}`（{t.get('speaker','')}）：{evidence['quote']}\n"
        text += f"\n可确认的问题：{err['interpretation']}\n\n应有行为：{err['correct_behavior']}\n\n审核：{obs['review_status']}；观察类型：{err['status']}。最终整题{'正确' if record['frozen_scoring']['exact'] else '错误'}，理由/引用不反向改变整题评分。\n\n"
        sf = card["solver_feedback"]
        for title2, key in [("候选关系", "candidate_hypothesis"), ("触发条件", "trigger_conditions"), ("预期可见错误", "expected_observable_error"), ("边界/反例", "boundary_conditions"), ("其他解释", "competing_explanations"), ("否证条件", "falsification_condition")]:
            value = sf[key]
            text += f"**{title2}：** " + ("；".join(value) if isinstance(value, list) else value) + "\n\n"
        text += "**出题任务：** " + card["question_task"]["target_relation"] + "\n\n**扩充处置：** 当前未确认素材缺口，先复用；若后续扩充，按TEST_HYPOTHESIS新增受控关系，而不是改写原事实。\n"
    text += "\n## 隔离的三条不确定观察\n\n- E218519-MC3-02：把列车标识泛称reference有语用歧义，不作为确定标识混淆错误。\n- A225575-MC6-E04：gave与got it for birthday存在日常赠礼交付概括歧义，不作为确定完成事件错误。\n- R229686-MC5-E01：否定项空引文，现有合同未证明必须非空；保留引用质量提醒，不认定语义弱点。\n\n初稿、修订要求、UNKNOWN原记录及修订补丁全部保留。\n"
    (P / "EXAMPLES_ZH.md").write_text(text)
    (P / "README_ZH.md").write_text("""# 反馈模板与30题试填交付

建议依次阅读：

1. REPORT_ZH.md：本轮方法、规模、结果与结论边界。
2. EXAMPLES_ZH.md：六个可追溯的具体弱点反馈例子（私有审计，不可直接给Challenger）。
3. FEEDBACK_TEMPLATE_ZH.md、FEEDBACK_TEMPLATE.json、FEEDBACK_TEMPLATE.schema.json：可复用模板、空白结构与机器校验规则。
4. DUAL_FEEDBACK_CARDS_PRIVATE.json：19张有效候选卡及原稿/修订/审核版本绑定。
5. CHALLENGER_PUBLIC_CARDS.json：去原题答案的公开投影；尚未实际消费。
6. SOURCE_AUTHOR_EXTENSION_TASKS.json、EXTENSION_TASK_INTERFACE_DRAFT.json：前瞻材料任务与接口草案；不是已接入runtime。
7. PRIVATE_AUDIT_RECORDS.json：30题原回答、题面、来源/提示/模型绑定、168项作者审计及交换审核，保留UNKNOWN和维度覆盖层。
8. QUESTIONER_FEEDBACK.json、MECHANISM_THEME_INDEX.json：出题建议及8主题描述性索引。
9. SUMMARY.json、DELIVERY_CHECK.json、FINAL_REPORT_REVIEW.json、FINAL_MANIFEST.json、FINAL_MANIFEST_CHECK.json：统计、验证、独立上下文复核及封存。

inputs保留冻结的完整来源及固定单条回答。authoring保存初稿，review保存交换审核及修订复核，revisions为显式补丁，不覆盖原记录。build_delivery.py可重建聚合；重建将更新聚合和其版本绑定，若已封存须重新验证并封存。validate_authoring.py仅检查初稿机械一致性，不能替代语义审核。validate_delivery.py检查有效交付的支持集合、投影、哈希及完整性。

不包含新扩充、新题或新训练。未修改旧实验、Gold或正式训练池。19卡均为HYPOTHESIS，正式复现弱点数0；“模板可用”不等于“反馈因果收益已经成立”。
""")


if __name__ == "__main__":
    main()
