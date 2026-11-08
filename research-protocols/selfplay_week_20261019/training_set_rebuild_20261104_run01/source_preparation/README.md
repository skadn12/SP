# 首批来源准备

本目录仅准备首批 8 个 root 的完整冻结来源和既有审核追溯。未生成题目，未调用 Solver，未训练。

作者和盲审只读取各 root 的 `author_materials/`：

- `FULL_SOURCE.json` / `FULL_SOURCE.txt`：全部原回合，保留原始 turn_ref、speaker、text，无删改。JSON 绑定 canonical 原文件 SHA256；自身是中性呈现副本，不能把副本 SHA256 当成 canonical SHA256。
- `NEUTRAL_SOURCE_INFO.json`：仅来源层、历史风险和使用边界，不含旧题、Gold、盲审答案或 Solver 原答。
- `TURN_PROVENANCE.json`：逐回合来源类型、文本 hash、历史审核索引及已存在的精确使用范围。

`audit/` 和顶层汇总由执行协调者核对。历史审核索引表示存在可追溯的审核记录，并非本次重新批准每个事实。103866/104814 有精确 R02 使用回执；它不把 parent/R01 扩大成获新授权，也不因上下文部分存在风险而阻止整个 root。其他 root 复用实际绑定的既有独立来源审核，不因名称不是 ADMISSION 而丢弃。

所有题均需按实际引用证据验证身份、引文、状态、歧义和使用条件；未引用的 HOLD 或某一来源事实的限制不得升级成整 root 门禁。完整原文仍须用于冲突与替代路径扫描。

`BATCH_01_SOURCE_PREPARATION.json` 给出 8 个 canonical 路径与 hash；`BATCH_01_VALIDATION.json` 记录对白一致、回合追溯和绑定完整性检查。全部 8 根均通过机械完整性核验。机械核验不替代题目实际证据审核或新的来源许可。
