#!/usr/bin/env python3
"""Controller-only draft of the 74 unresolved sealed-question query scopes."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
INPUT = HERE / "SEALED135_FIELD_AUDIT_CONTROLLER.json"
OPERATION_AUDIT = PLAN / "execution_20261106_v1/review/HISTORICAL_SEALED135_OPERATION_AUDIT.json"
OUTPUT = HERE / "SEALED135_SCOPE_REVIEW_DRAFT_CONTROLLER.json"

# Each entry is a reviewer-written description of the question domain, not an answer.
SCOPES = {
    "H0003": "两次列车咨询各自列出的时刻及时间界限，仅比较同一次咨询内指定的配对；到达前时刻视为界限。",
    "H0004": "联系方式尚未核实期间，道歉草稿的修改及保存、交接处理事项。",
    "H0005": "一次已安排的出租车交接记录中的所列字段，包括出发时间要求的含义。",
    "H0006": "筹款准备讨论后的任务安排、抽奖及邀请措辞清单。",
    "H0007": "明日查看账户警报前的待核对问题，依据双方最后同意的本人查看方案。",
    "H0008": "缺失回复率记录和未发联络草稿条件下，为次日会议与后续联络已商定的做法。",
    "H0009": "指定四人午餐的餐厅推荐、预约及更正通知过程，按实际发生时间排列所列步骤。",
    "H0014": "这次蔬菜汤制作中列出的步骤，依据双方安排及进度报告确定先后。",
    "H0015": "诊所内公司日常事务交接所列事项，按实际发生先后排列。",
    "H0016": "修改后已经发送给 IT 的跟进邮件所保留的内容。",
    "H0017": "删去重复品项后实际提交的三人午餐整单中，所列品项及数量。",
    "H0018": "该次出租车服务交流中从补充上车地点到保存联系方式并结束聊天的所列事项，按报告顺序排列。",
    "H0019": "最终收窄后的海洋宣传稿的目标、受众及开头形式。",
    "H0020": "新周一课程的数学夹整理和检查过程中列出的里程碑，按实际先后排列。",
    "H0021": "第一次周六问路练习中本人实际采取或明确请求的练习动作。",
    "H0022": "更新后的噪声提案中仍需团队讨论的事项，排除已完成的检查与行动。",
    "H0023": "从第一次安静约会到下次预留夫妻时段的所列家庭安排，按事件日历排列，包含尚在计划中的安排。",
    "H0024": "读书组建议准备工作流所列步骤的约定计划顺序，不预设步骤已完成。",
    "H0025": "练习搭帐篷结束时，至周六出发前或出发时仍待完成的所列行动。",
    "H0026": "咖喱场景中从发现胡萝卜偏硬到上菜的五项已完成步骤的实际执行次序。",
    "H0027": "一次晚餐咨询从首选不可用到预订确认的六个所列步骤的发生顺序。",
    "H0028": "新十月计划周的四项准备和活动按事件时间排序，包含已完成准备、未来活动和已约定替换。",
    "H0029": "双方为观察射箭稳定性而选择的记分本结构和记录规则。",
    "H0030": "一次住宿咨询从原定周一请求到预订成功并结束交流的六个步骤，按发生先后排列。",
    "H0031": "截至对话结束，为周四课程处理录音中不清楚段落而实际完成的准备行动。",
    "H0032": "该次餐馆及酒店咨询中所列请求和回复的发生顺序。",
    "H0033": "该次列车咨询从最初方案到最后旅行笔记澄清的所列步骤，按实际发生次序排列。",
    "H0034": "新十一月足球、午餐及登记复核计划中明确报告已完成的准备。",
    "H0035": "指定酒店预订的实际步骤顺序，不以回顾叙述中的首次提及顺序代替发生顺序。",
    "H0036": "修订后的周五朗读流程，从已完成准备到最后反馈的所列步骤。",
    "H0037": "报告从首次结构修改到发送更正稿的所列步骤，按实际发生顺序排列。",
    "H0038": "保存的视觉简报目前包含的内容及新周二预约仍保留的保护条件。",
    "H0039": "水边周末计划四项行动的全部六个无向配对，只报告有依据的实际或计划事件先后及传递关系；未定相对先后的配对不输出。",
    "H0040": "数学夹整理中 query_pairs 指定的四个行动配对，只报告明确指令要求的前置关系；讲话先后不构成行动前置。",
    "H0041": "本人明确表示已经为新周一数学课装好的所列物品。",
    "H0042": "该次独立列车行程的候选车次中，不满足周四或严格晚于 11:45 出发条件至少一项者。",
    "H0043": "安排并授权偶尔学校接人帮助之前仍需解决的所列事项。",
    "H0044": "同一对帮手覆盖周五两个时段的所列购物计划步骤；假定双方同意，排序的是未来计划。",
    "H0045": "下周各项提案中，两个指定时段始终由相同人员覆盖的情况；尚未接受的安排保持条件性。",
    "H0046": "短墙砌筑中从第二块砖试放至三块砖到位的所列实际操作先后。",
    "H0047": "新咖啡计划中四项未来行动的全部六个无向配对，仅输出有依据的计划先后及传递关系；未确定方向的配对不输出。",
    "H0048": "首次语言交换中三个指定阶段的全部三个无向配对；报告已约定计划要求的阶段先后及蕴含关系，不补无方向配对。",
    "H0049": "已约定的周五活动准备中，所列人员职责及现有物品的状态。",
    "H0050": "柠檬水试做及后续所列行动的实际先后。",
    "H0051": "三份独立存档列车咨询各自记录的订票状态。",
    "H0052": "单次存档餐厅咨询从初始请求到修正后成功预约的五个交流步骤的原始顺序。",
    "H0053": "双方为比较混合动力车报价而同意需要取得的所列成本信息。",
    "H0054": "截至指定场景结束时，新周二烹饪活动实际完成的准备；更新取代更早状态，排除旧活动。",
    "H0055": "飞船设计活动中所列阶段的顺序，至切割前约定的比较为止。",
    "H0056": "处理不完整回复率分析，避免客户联络把它呈现为最终结果的所列已选措施。",
    "H0057": "新十一月出游计划四项行动的全部六个无向配对，报告固定的实际或计划事件顺序及传递关系，区分暂定和最终确认。",
    "H0058": "耳环拍照测试及保存的所列步骤按发生先后排列。",
    "H0059": "一次预订失败后最终餐厅安排及其保留条件的所列描述。",
    "H0060": "已完成学习准备及接下来计划中的学习步骤的先后；不把未来步骤当成已完成。",
    "H0061": "发出图书馆消息后，图书捐赠活动当前准备状态的所列描述。",
    "H0062": "与医生核对笔记后，更正的预约文件所包含的所列内容。",
    "H0063": "准备并发送 pilot 讨论邮件的所列步骤按发生先后排列。",
    "H0064": "回应上司进一步问题的四项假设性备选行动的全部两两配对；只报告明确要求的前置顺序及其蕴含关系，不声称已发生。",
    "H0065": "所列事件的实际发生顺序，以对话中相对时间为依据，不按回忆时的叙述次序排列。",
    "H0066": "已交付审阅的完整吉他视频版本中所列标题和字幕变更。",
    "H0067": "同一新烹饪周期从原周一计划更改到最终周二计划时仍不变的所列承诺，排除旧周六活动。",
    "H0068": "周六烹饪中第一张煎饼的六个所列步骤，依据示范指令与执行报告排序。",
    "H0070": "新十一月出游候选日期中双方各自已查个人日历且均可用的日期，包含未被暂定选中的备选。",
    "H0072": "周五杂货复核四项任务的全部两两配对，按各任务被分配的时段判断严格先后；同一时段无方向，不推断完成。",
    "H0122": "传阅项目日程前四项任务的全部两两配对；仅报告计划规定及传递蕴含的前置关系，未规定的配对留空。",
    "H0124": "储藏室购物与晚宴座位准备的六项行动的全部不同项配对；仅报告明确计划前置及传递关系，不推断跨活动日历顺序。",
    "H0125": "预防会议安排的五项所列事项的全部两两配对；仅报告明确或必然蕴含的流程前置及传递关系，不用叙述或完成时间代替。",
    "H0126": "车夹准备四项行动的全部两两配对；只报告指令要求及传递蕴含的前置关系，忽略叙述顺序与完成状态。",
    "H0127": "同一次多次更改的列车咨询结束时，重新搜索仍须使用的所列条件。",
    "H0130": "婚礼备用空间的三个规划事项的全部两两配对；只报告对话规定的先后或前置及其蕴含关系，不强加完整排序。",
    "H0131": "准备外套草稿并交接精品店的所列里程碑的约定顺序，包含尚未执行的末步。",
    "H0132": "指定匿名服务记录中博物馆搜索从最初请求到详情交流的所列里程碑发生顺序。",
    "H0133": "指定周四计划中所列列车时刻、最晚到达界限及晚餐时刻按时钟排序，不推断旅行已完成。",
    "H0135": "一次宾馆咨询所列阶段的实际请求、回复与纠错顺序，不按事后摘要中的提及顺序排序。",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(INPUT) == "fc4614c594e33bd21ad39e1d6676224507070ba404c9028947133322dd93892c"
    field_audit = read(INPUT)
    operation_audit = read(OPERATION_AUDIT)
    expected = {r["history_item_id"] for r in field_audit["records"]
                if r["query_scope"]["status"] == "UNKNOWN_PENDING_SCOPE_SPLIT"}
    assert len(expected) == 74 and set(SCOPES) == expected
    results = []
    for item in field_audit["records"]:
        hid = item["history_item_id"]
        if hid not in SCOPES:
            continue
        index = item["audit_record_index_zero_based"]
        original_audit = operation_audit["records"][index]
        ref = item["reference"]
        assert sha(ref["path"]) == ref["sha256"]
        original_reference = read(ref["path"])[ref["record_index_zero_based"]]
        assert original_reference["candidate_id"] == item["candidate_id_controller_only"]
        question = original_audit["question_and_displayed_options"].split("\n\n")[0]
        assert original_reference.get("question") == question
        assert original_audit["evidence_quotations"] and item["evidence_turn_refs"]
        explicit = original_reference.get("query_scope")
        results.append({
            "history_item_id": hid,
            "candidate_id_controller_only": item["candidate_id_controller_only"],
            "root_id_controller_only": item["root_id_controller_only"],
            "query_scope": SCOPES[hid],
            "status": "REVIEWER_DRAFT_NOT_INDEPENDENTLY_VERIFIED",
            "basis": "Exact question and displayed items with operation-audit event/scope and same-ID evidence anchors.",
            "top_level_reference_query_scope_present": bool(explicit),
            "top_level_reference_query_scope_sha256": hashlib.sha256(
                json.dumps(explicit, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest() if explicit else None,
            "question_projection_index_zero_based": item["projection_record_index_zero_based"],
            "question_content_sha256": item["question_content_sha256"],
            "operation_audit_record_index_zero_based": index,
            "reference": ref,
            "evidence_turn_refs": item["evidence_turn_refs"],
            "selector_exposure": "FORBIDDEN_PENDING_SELECTOR_SAFE_NORMALIZATION_AND_REVIEW",
        })
    counts = Counter("top_level_scope" if r["top_level_reference_query_scope_present"] else "no_top_level_scope"
                     for r in results)
    assert len(results) == 74 and counts == {"top_level_scope": 13, "no_top_level_scope": 61}
    out = {
        "schema_version": "sealed135_scope_review_controller_draft_v1",
        "status": "DRAFT_NOT_COMPLETE_NOT_SELECTOR_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": "CONTROLLER_ONLY",
        "scope": "74 previously unseparated query-scope fields; does not review natural needs, Gold or quality.",
        "source_bindings": {
            "field_audit_path": str(INPUT), "field_audit_sha256": sha(INPUT),
            "operation_audit_path": str(OPERATION_AUDIT), "operation_audit_sha256": sha(OPERATION_AUDIT),
        },
        "counts": {"draft_rows": len(results), "reference_query_scope_present": counts["top_level_scope"],
                   "reference_query_scope_absent": counts["no_top_level_scope"], "independent_verification_complete": 0},
        "limitations": [
            "This is a reviewer draft requiring independent semantic verification and leakage-safe normalization.",
            "These scope descriptions may reveal historical question wording or answer-related context; never send directly to a selector.",
            "49 other sealed rows still have UNKNOWN natural information need. This draft does not alter that status.",
            "The 86 existing event/natural-need annotations have not been independently reviewed here.",
            "Evidence anchors refer to the earlier operation audit, which did not exhaustively inspect full narratives for alternative shortcuts.",
        ],
        "records": results,
    }
    assert not OUTPUT.exists(), "Refusing to overwrite an existing artifact."
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha(OUTPUT), "counts": out["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
