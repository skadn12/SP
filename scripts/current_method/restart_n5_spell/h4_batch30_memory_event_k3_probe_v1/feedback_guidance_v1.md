# 第一轮流程的附加反馈模板 v1

依据用户已确认的原则制定；只在第一轮原 prompt 末尾追加，不替换原角色、输出合同或数据。模板是固定文本，花括号内容由程序从有来源的观测中填充。模型不生成或修改模板。

## Planner 附加段

```text
【反馈指导：仅影响规划内容】
下面是历史问答与 Solver 的行为观测。它们供选材参考，不是已发生事件，
也不能推翻原有 Memory、人物、场景、预算或输出合同。
未提供的成绩保持未知；答错或无上下文答对不证明缺少某种能力。
参考这些观测选择值得延展的事实与关系，使后续问题能检验需要上下文的区分。
新增发展须自然、可辨明，不能靠矛盾、遗漏或把旧分支的事件写成历史来增加难度。
不逐题修补旧答案；不输出 QA，不把题干、Gold 或成绩写进场景。
保持原 Planner JSON schema 与选择流程。
【有来源的反馈数据】
{PLANNER_FEEDBACK_JSON}
```

## Intent Miner 附加段（intent_suffix）

```text
【反馈指导：仅影响 intent 选材】
下面的历史问答、Gold、证据定位和 Solver 行为观测用于探索值得进一步考察的关系。
它们不代替上述编号对话。只有当前冻结对话能支持本次 reasoning_chain。
比较可辨认的事实、状态、时间、人物关系与约束，优先寻找需要结合上下文的区分，
避免仅替换旧题措辞。难度来自实际证据关系，不来自歧义或不成立的 Gold。
成绩及错误输出只是行为观测；不编造认知诊断，不补全未提供的测量。
仍输出原来的三个 intent，使用原 intent_type enum、原 reasoning_chain 字段和
原三类候选题型要求；不增加新的策略字段、固定反馈槽位或跳过生成指令。
【有来源的反馈数据】
{INTENT_FEEDBACK_JSON}
```

## QA Writer 附加段（slot_suffixes）

```text
【反馈指导：仅影响问题内容】
结合当前模式允许使用的 intent 信息与完整对话，探索能区分相关事实、时间、状态、
人物关系或约束的问题。尽量让答案需要对话证据，而非题干选项的表面提示。
不要把旧题改写、换选项位置或更换 ID 当成新问题。
难度提升必须保留确定的 Gold、合理干扰项和完整可辨明的证据关系。
仍按原模式生成混合 batch，保持原配额、输出 schema 及事实来源。
contract_guided 的证据合同仍是指导；free_exploration 仍可自由探索对话中的其他关系，
不强制实现指定反馈槽位，不强制引用某个轮次。
不输出模型能力诊断，不把反馈 metadata 当成对话事实。
```

Writer 的具体方向来自原 intent/contract 接口；原 free 模式只消费 intent label。
`slot_suffixes` 仅是现有接口中的三个附加字符串，可填相同的 Writer 指导文本，
不重新定义三个 intent 的含义。Writer 不额外接收完整 Solver 成绩。

## 冻结输入格式

```json
{
  "roots": {
    "cc-train-episode-EXAMPLE": {
      "path": "/absolute/path/feedback_context.json",
      "sha256": "该文件实际字节的 SHA-256"
    }
  }
}
```

每个 context 文件为：

```json
{
  "intent_suffix": "填好动态反馈的 Intent Miner 附加段",
  "slot_suffixes": ["Writer 附加段", "Writer 附加段", "Writer 附加段"]
}
```

反馈原始数据另存 packet，记录 source 文件、来源轮次、模型、样本数和哈希。
启动前冻结该 root 的 packet、模板与渲染后的 context。无反馈组不传此参数。
程序验输入文件与哈希，不因此添加候选语义审核或新的难度门槛。
