# H4 TEMPORAL Writer 模型对照配置

此目录属于 `h4_writer_model_comparison_v1`。两个模型都复用正式 `contract_guided` mixed TEMPORAL 路径，每个原始 intent 目标为 2 单选、1 多选、1 排序，排序节点数保持 4。

- `qwen14b.json`：源 run 六份实际 Writer payload 的参数。不是 Realizer 配置。`temperature=0.1`、`max_tokens=8192`、`enable_thinking=false`、`response_format={"type":"json_object"}`；未发送 seed 和 stop。
- `strong_model.json`：TJU 服务的公开配置。实际 API model ID 保持 `tju-llm`；用户提供的底层模型名 `qwen3.6-35b-a3b` 仅记录为 `user_reported_backend_model`，不能据此推定服务限制或已固定版本。

当前用户已明确授权带服务未知项启动，记录在 `user_authorized_service_exception.json`。强臂固定请求 `temperature=0.1`、`max_tokens=8192`、`response_format={"type":"json_object"}`，reasoning 采用服务默认且保持 UNKNOWN。`null` 仍表示未核实；授权不表示服务已经支持或采用了这些参数。准备过程不会读 Key 值或发连通性测试。源 run 和历史成绩仅用作输入与诊断来源，本次两臂都新生成。

## 未核实服务信息与配置字段

| 字段 | 含义 |
| --- | --- |
| `api.timeout_seconds` | 此轮单次 HTTP 超时秒数；两臂均无自动重试 |
| `generation.temperature` | 本次固定请求 0.1；服务支持与实际采用状态仍 UNKNOWN |
| `generation.max_output_tokens` | 本次固定请求 8192；不据此宣称服务容量已验证 |
| `generation.output_parameter` | 本次固定 `max_tokens`，服务拒绝时停止，不改参数重发 |
| `generation.seed_policy` | `not_sent` 或 `fixed`；`fixed` 时另填 `generation.seed`，记录与 14B 未发送 seed 的差异 |
| `generation.reasoning_settings` | 本次 `null`：不发送显式 reasoning 参数，服务默认行为记录 UNKNOWN |
| `generation.response_format` | 本次固定请求 `{"type":"json_object"}`；实际支持和采用状态 UNKNOWN |
| `generation.stop` | 源 Qwen 未发送；本次 `null` 明确不发送 stop，与基线一致 |
| `parameter_support.supported` | 所有实际请求参数名的列表，包括 reasoning 参数；只能列文档或用户确认的参数 |
| `parameter_support.unsupported` | 明确不支持或本次有依据不发送的参数名；与 `supported` 不得重叠 |
| `service_limits.context_window_tokens` | 此 endpoint/model 的实际上下文上限，不从公开模型名推定 |
| `service_limits.max_output_tokens` | 此服务已确认支持的输出上限 |
| `service_limits.reasoning_budget_tokens` | reasoning 预留 token 数；明确无额外 reasoning 时为 0 |
| `service_limits.reasoning_in_output_budget` | boolean，reasoning 是否已包含在输出预算里，避免重复或漏算 |
| `token_accounting` | 下面的离线完整 messages 计数配置 |
| `service_evidence` | 本地文档或用户确认记录的路径数组；对象格式 `{"path":"/absolute/local/evidence.md","kind":"user_confirmed_service_settings"}` |
| `pricing` | 可核实价格；没有可靠信息保留 `null`，费用报告为 UNKNOWN |

额外 `request_parameters` 对象可用于服务明确支持的原始参数；不得包含 `model`、`messages`、stream、tools 或改变调用阶段的字段。`output_limit` 可为使用其他输出字段的服务声明本次数值预算。任何配置差异都写入 run 的 `audit/service_compatibility.json`，本轮称“模型/服务配置组合比较”。

离线 token 配置形状：

```json
{
  "token_accounting": {
    "method": "huggingface_chat_template",
    "local_tokenizer_path": "/absolute/path/to/verified/local/tokenizer",
    "model": "与服务对应的明确 tokenizer 标识",
    "revision": "可固定时填写版本",
    "chat_template_kwargs": {"add_generation_prompt": true},
    "evidence_path": "/absolute/path/to/local/chat-template-and-service-evidence.md"
  }
}
```

`chat_template_kwargs` 必须与服务实际封装相符。准备时仅复制本地 tokenizer 资源，并使用 `local_files_only=True`；不会下载。未知真实 tokenizer/封装或容量时维持 UNKNOWN。本次仅对授权记录绑定的 TJU endpoint、model 和固定参数允许发送完整输入，不截短 transcript；这不是容量检查 PASS。缺少这份明确授权时仍阻止 live。Qwen 复用源 run 已冻结 tokenizer 与计数身份，当前六个 primary 完整 messages 共 47,958 input tokens，最大 primary output 为 49,152。

## Key 注入

公开配置禁止 `api_key`、Authorization header 等值。正常方式是在运行命令所在进程环境设置 `H4_STRONG_MODEL_API_KEY`。已启动 Codex 不会继承其他终端之后执行的 export。

当前 `api.credential_file` 指向 `/root/.config/gxy-selfplay/h4_writer_model_comparison_v1.key`。在原来已 export Key 的终端执行以下命令，可把环境变量安全转存到本窗口之后能读取的位置（未设置变量时提示隐藏输入）：

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
python -B -m scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.run configure-key
```

该配置指向同一用户所有、权限 `0600` 的本地文件，其内容仅为 Key，`api.credential_format` 写 `raw`。该文件只在显式 live 时读取，内容不进入冻结产物。不要把它放在本目录、run 目录或仓库中。Qwen 使用 `H4_QWEN_API_KEY`，未设置时由运行入口读取现有 workspace `config.json`，不复制该凭证文件。

## 同一正式入口

在 `/root/autodl-tmp/gxy-selfplay/current_method_github/` 执行：

```bash
python -B -m scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.run prepare
python -B -m scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.run mock
python -B -m scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.run preflight
```

本次配置例外及 12/12/24 预算已获用户授权，完成冻结与 mock/回归验证后执行同一入口：

```bash
python -B -m scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.run live \
  --authorize-primary 12 --authorize-repair-max 12 --authorize-http-max 24
```

这些授权参数不会覆盖输入冻结、工程验证或预算检查。仅明确记录的强臂服务未知项可通过用户例外；Qwen token/容量检查不变。准备和 mock 均零外部调用。强配置补齐后可重新 `prepare`；一旦已有 live 请求或 HTTP 记录则禁止覆盖准备产物。

调用顺序在 `schedule.json` 固定交错。两臂各 6 primary，各最多 6 repair，总逻辑调用及 HTTP 尝试均最多 24。每次逻辑调用最多一次 HTTP，不存在 429、SSL、SDK、重定向自动重试。任何技术失败（参数拒绝、上下文超限、429、SSL 失败、超时、未知完成等）均停止所有后续未发请求，不重发、不进入内容 repair。只有正常取得明确完成响应后，无效候选 JSON/结构不足才执行既有一次整批 repair。repair 使用本臂自己的 primary 内容，所以 repair 正文不要求跨臂相同。

结果保留全部 primary/repair、程序接受/替换/拒绝、失败及成本；语义诊断使用现有离线口径并标为 Codex 离线诊断。生产结构准入规则不增加语义认证。不会自动启动 Solver、Memory 或其他实验。
