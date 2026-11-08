"""Render the corrected offline decision separately from historical reports."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    result = json.loads((HERE / "results_revision_2.json").read_text())
    lines = [
        "# 多选候选准入复核：实现修正 2", "",
        "本报告取代 REPORT.md 中关于 seed、split 和额外独立人工审核门槛的解释。旧报告和 results.json 原样保留，供追溯；本次结果使用 results_revision_2.json。策略仍为 multi_select_partial_admission_v1，本次修正实现，不另建补充训练池。", "",
        "## 修正结果", "",
        "| 题目 | no → full 整题 | no → full 平均 J | 数值候选 | 质量 | split | 配对 | 换位 | 最终候选状态 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for qid, row in result["questions"].items():
        metrics = row["metrics"]
        no, full = row["modes"]["no_context"], row["modes"]["full_context"]
        lines.append(f"| {qid} | {no['exact_correct']}/{no['expected']} → {full['exact_correct']}/{full['expected']} | "
                     f"{metrics['no_partial_mean']:.6f} → {metrics['full_partial_mean']:.6f} | "
                     f"{row['new_policy_reason']} | {row['quality_check_status']} | {row['split_check_status']} | "
                     f"{row['pairing_check_status']} | {row['permutation_check_status']} | {row['new_policy_candidate_status']} |")
    counts = result["counts"]
    lines += ["", f"数值候选 {counts['numeric_candidates']} 题，其中相对旧分桶新增 {counts['new_numeric_candidates']} 题；全部准入检查已通过的训练候选 {counts['training_candidates']} 题。实际入训 0 是本次仅做离线诊断、没有执行训练池写入的操作事实，不能用来判定题目有没有训练价值。", "",
              "## 取消的额外限制", "",
              "- full/no 按同题版本、同选项排列、同采样槽位和相同解码配置核对。原评测的 seed 按请求递增，不再要求两侧 seed 相等；差异仍作为警告保存。比较不据此声称控制了采样噪声或有统计显著性。",
              "- 不再把质量统一写为待独立 Gold 裁定，不增加人工逐题复核依赖。本次逐项对照冻结原文、已有 REVIEW 和上游内容审核，保存支持/排除证据及适用范围；不是凭单个 VALID 标签放行。",
              "- split 从正式 root_split_manifest 与源对话继承链验证，不把 training_promotion=false 当成非训练 split。",
              "- 换位状态从独立、带证据哈希的复核产物读取，不再按题目 ID 后缀硬编码。所有审核都绑定题目内容哈希，凭据缺失或漂移仍不能放行。", "",
              "## 仍保留的换位问题", "",
              "两题的循环/反向排列是真实换位证据，并非完全没有换位。当前每个排列只有一条回答；旧机制诊断依赖每排列三个采样槽位，现记录不足以分开位置效应和采样波动。旧代码没有可直接用于这批单槽位数据的内容稳定性通过标准，因此不能自行标为 PASS。位置敏感不等同于纯位置题，也不等同于 Gold 错误。", "",
              "第一题存在内容信号：canonical E 在所有位置均被选择，D 均未被选；错误 B 又有强位置差异。第二题显示字母 C 总被选，但 canonical 选择也随内容改变。这些是混合现象，需要解释，不能仅凭平均收益或固定字母的个别统计就完成准入。", "",
              "第一题各排列部分收益按 EPS=1e-9 计为 8 升、3 平、1 降；DATA_AUDIT.md 中旧 9/2/1 是浮点直接比较造成的误计，本次更正，均值收益仍为 11/36。", "",
              "## 来源与审核边界", "",
              "权威 split、逐选项质量和换位证据分别见 SPLIT_AUDIT_REVISION_2.md、QUALITY_AUDIT_REVISION_2.md、PERMUTATION_AUDIT_REVISION_2.md。原始 R03 的内容审核和历史晋升状态均保留。题目在冻结记录上的答案可判定性通过，不代表整份源对话获得历史晋升，也不会自动解除其他路线的隔离。", "",
              "## 后续最小评估范围（未执行）", "",
              "若后续授权 Solver 评估，可先按旧 k=3 合同复用每题六个正向循环排列（perm_00至perm_05），每模式补两个采样槽位，共 2题×6排列×2模式×2槽位=48 条回答。原反向排列作为补充证据保留，不混作同排列重复。需显式保留 perm 到旧 shift 标识的映射，以及相同题目/Gold/解码配置和可核查模型版本；若无法验证与原评测为同版本，不能把新旧结果混算。已有机制分类可用于诊断，不能承诺补足次数就必然通过。仅有固定字母驱动或仍无法解释时继续待诊断；不得临时发明通过阈值。", "",
              "具体48条请求、原prompt哈希、旧shift映射、采样槽位和运行前提已准备在 followup_48_requests_revision_2.json，状态为 PREPARED_NOT_EXECUTED；没有启动推理。", "",
              "本次没有生成新题、调用 Solver 或训练，不改变生产入口、单选/排序、Planner/Questioner、扩充、增难或回放规则。", "",
              "## 复算", "", "在本目录运行：", "", "```bash",
              "PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_scoring.py test_admission.py test_reanalyze.py",
              "PYTHONDONTWRITEBYTECODE=1 python reanalyze.py",
              "PYTHONDONTWRITEBYTECODE=1 python write_revision_report.py",
              "PYTHONDONTWRITEBYTECODE=1 python prepare_followup.py", "```", "",
              "本次22个测试通过；两题所有审核凭据的内容哈希校验通过。独立代码复核确认seed仅警告，候选/最终准入/实际入训计数分开，确认隔离仍阻断。", ""]
    (HERE / "REPORT_REVISION_2.md").write_text("\n".join(lines), encoding="utf-8")
    pool = {
        "policy_version": result["policy_version"], "implementation_revision": 2,
        "pool": "gold_sft_candidates", "production_connected": False,
        "note": "One offline candidate pool; final admission remains per-question. No SFT examples exported.",
        "counts": counts,
        "candidates": [{"candidate_id": qid, "root_id": row["root_id"],
                        "numeric_candidate": row["numeric_candidate"],
                        "gold": row["gold"], "reason": row["new_policy_reason"],
                        "status": row["new_policy_candidate_status"], "blockers": row["blockers"]}
                       for qid, row in result["questions"].items() if row["numeric_candidate"]],
    }
    (HERE / "candidate_pool_revision_2.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
