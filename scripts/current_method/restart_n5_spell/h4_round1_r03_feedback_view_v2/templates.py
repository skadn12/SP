"""Frozen v2 template blocks shared by expansion and QA renderers."""

from __future__ import annotations

from typing import Any, Mapping


SHARED_FEEDBACK_RULES_V2 = """【共享反馈使用规则——扩充与出题共用】

你会收到当前角色需要的历史 QA、精简的 Solver 行为观测，
以及权威的完整累计对话。未提供的反馈不需要补齐。

反馈可能包含上一轮及训练后的有／无上下文答题表现，
以及含义明确的错误输出或错误选项。

这些都是行为观测：
匹配历史 Gold，只表示输出匹配，不代表答案语义正确性已被验证；
没有匹配 Gold，或 full/no-context 结果不同，
不证明模型缺少某种能力。
不要根据正确率或错误选项编造认知诊断。
缺失信息不编造，不可靠的诊断使用 UNKNOWN。

历史 QA 和反馈用于参考，完整累计对话才是历史事实依据。
不要为了迁就旧题或历史 Gold，改写历史或补造过去的事实。

已有旧题证据可以作为定位提示；没有提供时直接阅读完整原文，
不需要额外补证据或生成摘要。
"""


EXPANSION_TASK_V2 = """【扩充专属任务——固定规则】

你的任务是在当前完整累计对话之后追加自然、有新信息的对话，
为后续 QA 提供更多可用事实和关系，
不是回答旧题，也不是直接生成 QA。

必须沿用完整 Parent 和当前分支所有已完成的扩充轮次。
不得退回原始 Parent，不得删除或修改历史，
不得整段复制历史来充当新增内容。

可以参考共享反馈，自主选择值得延展的人物、事件或关系。
不需要逐题处理反馈，不能把整轮扩充锁定在某一道历史 QA 上，
也不能只为了修补某个历史错误答案而安排情节。

可以继续发展原有事件，也可以引入相关的新人物、其他事件或支线。
可以形成状态变化、人物关联、跨事件联系或前后约束。
这些是可选方向，不要求全部使用，
也不根据成绩标签机械指定必须增加什么内容。

新增内容应带来实际的新事实、事件进展或关系变化，
不只是反复复述旧事实、重复表达感受，
或反复提出同一个尚未发展的未来计划。

允许出现相似、容易混淆但能够区分的事实。
复杂度应来自需要结合上下文才能判断的人物、时间、事件和关系，
而不是无法消解的矛盾、关键事实缺失或故意写不清楚。

发生状态更新时，应能区分旧状态与新状态；
提出计划、确认安排和实际完成事件，也应能够区分。
保持人物行为、对话语气和事件发展自然，
不为增加复杂度硬塞情节。

不要把题干、选项标签、Gold、Solver 成绩或能力诊断
写进正常对话。
"""


QA_TASK_V2 = """【出题专属任务——固定规则】

仅根据已经冻结的完整对话生成问题，不创造事实。
题目应有明确的答案或正确选项集合；
非 Gold 选项不能在题干规定的范围内同样成立。
这些是生成要求，不代表输出已经通过语义验证。

不得用旧题改写、选项洗牌或更换 ID 冒充新题。
改变证据编号或上下文 provenance 本身不证明新颖性。
Exact Replay 不属于新 QA。

按模式使用固定任务块：

- Targeted：参考选定来源题，改变答案目标、证据关系或推理结构，
  遵守已有冻结 strategy space。
  本次只同步模板／接口，不调度真实 Targeted 生成。

- Composition：参考已选定的多条历史 QA，
  尝试形成需要结合相关信息的新问题，
  不仅仅并排复述或拼接旧题。

- Fresh：从完整对话探索新问题，
  使用既有的有限历史／覆盖约束减少重复，
  不强制绑定某道旧题或展示其整套成绩。

仅在本轮新增内容真正有贡献时引用它，
不能为了满足引用要求机械加入本轮编号。
不为 quota 强行生成；沿用各模式已有的不出题结果及编码。

严格遵守现有 revision-1 输出 schema。
不新增或删除模型输出字段。
旧题输入证据可以缺省，新生成 QA 仍须至少一条真实 evidence_refs。
不调用 semantic auditor，不把语义判断加入正式硬门。
"""


def _with_placeholders(blocks: list[str], role_placeholder: str) -> str:
    return "\n\n".join(blocks + [
        "【当前角色可见的共享反馈】\n{ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW}",
        "【权威的完整累计对话】\n{AUTHORITATIVE_CUMULATIVE_DIALOGUE}",
        "【当前角色的既有执行要求及输出合同】\n{" + role_placeholder + "}",
    ])


EXPANSION_TEMPLATE_V2 = _with_placeholders(
    [SHARED_FEEDBACK_RULES_V2, EXPANSION_TASK_V2],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
)
QA_TEMPLATE_V2 = _with_placeholders(
    [SHARED_FEEDBACK_RULES_V2, QA_TASK_V2],
    "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
)

# v3 is the active route for the repaired R03 branch.  The v2 blocks above
# remain immutable so historical prompts can still be reproduced byte-for-byte.
SHARED_FEEDBACK_RULES_V3 = SHARED_FEEDBACK_RULES_V2

PLANNER_EXPANSION_TASK_V3 = """【Planner 扩充专属任务——固定规则】

你的职责是为新增对话制定计划，不是直接写 Child 对话，也不是回答历史 QA。
在完整累计对话之后规划自然、有新信息的场景，保留既有 State Transition
Contract：before_state -> required_transition -> after_state。

Memory 提供人物、事件等导航线索；事件桶是主要候选来源，人物桶是辅助背景。
Memory 摘要不是自动核实的最新状态，完整累计对话才是历史事实依据。
所选 verified_track_ref 必须对应本场目标、动作和已有状态；若要选择另一事件，
使用它自己的真实 ref。新发展走 new_development 分支，不伪造旧事件 ID 或证据。

interaction_beats 只能使用第三人称动作／互动摘要，说明谁提出、决定、执行、回应
以及产生什么变化。不要输出完整台词、speaker: "..." 形式或逐字复制历史 turn，
不要用旧事件换词复述冒充新进展。每个 beat 必须帮助实现本场目标，且不能写入题干、
选项、Gold、Solver 成绩或能力诊断。

仅输出既有 Planner JSON 合同要求的字段。候选池由本地程序按既有 Gate 与显式配置
选择；不要自行补足 quota，不要编造缺失事实。"""

PLANNER_DEVELOPMENT_GUIDANCE_V1 = """【Planner 选材与内容质量指导】

为后续 QA 增加可用事实和关系，规划自然、有新信息的新场景。
你的职责是制定计划，不是直接生成对白或 QA。

共享反馈用于参考，自主选择值得延展的人物、事件或关系。
不需要逐题处理反馈，不能把整轮扩充锁定在某一道历史 QA 上，
也不能只为了修补某个历史错误答案而安排情节，
不根据成绩标签机械指定必须增加什么内容。

可以继续发展原有事件，也可以引入相关的新人物、其他事件或支线。
可以形成状态变化、人物关联、跨事件联系或前后约束。
这些是可选方向，不要求全部使用；仍遵守当前事件来源、人物声明和预算规则。

允许出现相似、容易混淆但能够通过上下文区分的事实。
事实相似不代表允许候选重复，也不能把旧事实换词复述成新进展。

复杂度应来自需要结合上下文才能判断的人物、时间、事件和关系，
而不是无法消解的矛盾、关键事实缺失或故意写不清楚。
保持人物行为和事件发展自然，不为增加复杂度硬塞情节。"""

PLANNER_EXPANSION_TASK_V4 = PLANNER_EXPANSION_TASK_V3 + "\n\n" + PLANNER_DEVELOPMENT_GUIDANCE_V1

REALIZER_CONTENT_GUIDANCE_V1 = """【Realizer 内容质量指导】

只实现已经选定的 Planner 场景，追加自然、有新信息的对话，
为后续 QA 提供更多可用事实和关系。不得自行重选人物、事件或增加支线。
新人物遵守计划声明，辅助说话人遵守参与者规则。

新增内容应带来实际的新事实、事件进展或关系变化，
不只是反复复述旧事实、重复表达感受，
或反复提出同一个尚未发展的未来计划。

允许出现相似、容易混淆但能够区分的事实。
复杂度应来自需要结合上下文才能判断的人物、时间、事件和关系，
而不是无法消解的矛盾、关键事实缺失或故意写不清楚。

发生状态更新时，应能区分旧状态与新状态；
提出计划、确认安排和实际完成事件，也应能够区分。
保持人物行为、对话语气和事件发展自然，
不为增加复杂度硬塞情节。

不要把题干、选项标签、Gold、Solver 成绩或能力诊断
写进正常对话。不要输出 QA、事实摘要、bundle hash 或 evidence index。"""

REALIZER_EXPANSION_TASK_FULL_V4 = """【Realizer 扩充专属任务——full_parent 模式】

你的职责是实现已经冻结的 Planner 场景计划，只输出新增 Child 对话。
保留完整历史，不回退、不删除、不修改历史，不把计划字段或历史 QA 写入正常对话。
每个场景必须完成自己的 before_state -> required_transition -> after_state 小弧线，
并遵守既有 Realizer JSON 输出合同、参与者身份和 turn 数要求。

full_parent 模式会收到完整累计历史；历史只作为参考，不能复制或重放。
只输出冻结计划对应的新增 Child。bounded continuation 是独立目的，只能接收
本场已经生成的实际末尾与固定计划，不重新注入无关的整段历史。

只实现新增内容，不回答旧题，不输出 QA、evidence index、Solver 诊断或新的证据索引。"""


REALIZER_EXPANSION_TASK_TEXT_FREE_V4 = """【Realizer 扩充专属任务——text_free 模式】

你的职责是实现已经冻结的 Planner 场景计划，只输出新增 Child 对话。
保留完整历史，不回退、不删除、不修改历史，不把计划字段或历史 QA 写入正常对话。
每个场景必须完成自己的 before_state -> required_transition -> after_state 小弧线，
并遵守既有 Realizer JSON 输出合同、参与者身份和 turn 数要求。

text_free 模式不会提供历史 turn 原文；只能使用 continuity shape 保持形式和节奏，
不能猜测、引用、复制或近似改写被省略的旧文本。不要重建旧事件或旧情绪收束。
bounded continuation 是独立目的，只能接收本场已经生成的实际末尾与固定计划，
不重新注入无关的整段历史。

只实现新增内容，不回答旧题，不输出 QA、evidence index、Solver 诊断或新的证据索引。"""


# Kept as a compatibility alias for historical imports.  New renderers use
# the explicit mode-specific v4 blocks above.
REALIZER_EXPANSION_TASK_V3 = REALIZER_EXPANSION_TASK_FULL_V4

QA_TASK_V3 = QA_TASK_V2


def _with_placeholders_v3(blocks: list[str], role_placeholder: str,
                          *, include_dialogue: bool = True) -> str:
    dialogue_block = ("【权威的完整累计对话】\n{AUTHORITATIVE_CUMULATIVE_DIALOGUE}"
                      if include_dialogue else
                      "【权威累计对话输入】\n历史原文由当前 Realizer 输入模式控制；"
                      "本段不重复注入被省略的文本。")
    return "\n\n".join(blocks + [
        "【当前角色可见的共享反馈】\n{ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW}",
        dialogue_block,
        "【当前角色的既有执行要求及输出合同】\n{" + role_placeholder + "}",
    ])


EXPANSION_PLANNER_TEMPLATE_V3 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, PLANNER_EXPANSION_TASK_V3],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
)
EXPANSION_PLANNER_TEMPLATE_V4 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, PLANNER_EXPANSION_TASK_V4],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
)
EXPANSION_REALIZER_FULL_TEMPLATE_V3 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, REALIZER_EXPANSION_TASK_V3],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
    include_dialogue=False,
)
EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V3 = EXPANSION_REALIZER_FULL_TEMPLATE_V3
EXPANSION_REALIZER_FULL_TEMPLATE_V4 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, REALIZER_EXPANSION_TASK_FULL_V4],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
    include_dialogue=False,
)
EXPANSION_REALIZER_TEXT_FREE_TEMPLATE_V4 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, REALIZER_EXPANSION_TASK_TEXT_FREE_V4],
    "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
    include_dialogue=False,
)
QA_TEMPLATE_V3 = _with_placeholders_v3(
    [SHARED_FEEDBACK_RULES_V3, QA_TASK_V3],
    "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
)


def render(template: str, values: Mapping[str, Any]) -> str:
    """Replace only declared placeholders; braces in data are not interpreted."""
    output = template
    for key in (
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW",
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE",
        "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
        "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
        "EXISTING_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT",
    ):
        output = output.replace("{" + key + "}", str(values.get(key, "")))
    return output
