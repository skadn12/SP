from pathlib import Path
import json,hashlib,collections,datetime,shutil
D=Path(__file__).resolve().parent;W=D.parent

def read(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 s=read(D/'SUMMARY.json');packet=D/'next_generation/INPUT_PACKET.json';pp=D/'next_generation/agent_plan';plan=read(pp/'EXPANSION_QUESTION_PLAN.json');receipt=read(pp/'ACTUAL_AGENT_READ_MANIFEST.json')
 assert plan['actual_plan_consumption'] is True and plan['next_candidates_generated'] is False and plan['all_targets_loop_closed'] is False
 assert receipt.get('actual_plan_consumption',receipt.get('actual_agent_consumption',True)) is True
 hist=D/'review/history/before_next_consumption';hist.mkdir(exist_ok=True);shutil.copy2(D/'SUMMARY.json',hist/'SUMMARY_BEFORE_CONSUMPTION.json')
 s.update(status='BOUNDED_FEEDBACK_REALIZATION_ITERATION_COMPLETE_WITH_ONE_HELD_TARGET',actual_next_consumer_read=True,actual_next_plan_binding=bind(pp/'EXPANSION_QUESTION_PLAN.json'),actual_next_read_receipt_binding=bind(pp/'ACTUAL_AGENT_READ_MANIFEST.json'),feedback_to_candidate_to_review_to_next_plan_executed=True,all_five_targets_realized=False,feedback_loop_closed=False,all_targets_loop_closed=False,next_candidates_generated=False,updated_at_utc=now())
 (D/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 reviews=[q for a in ['a','b'] for q in read(D/('review/part_'+a+'_CONTENT_REVIEW.json'))['questions']];novelty=collections.Counter(q['novelty_class'] for q in reviews if q['disposition']=='ACCEPT')
 text=f'''# 反馈定向出题与诊断回流执行 v1

## 结果与解释

已按上次实际读取的五卡计划完成一次受限执行：冻结四个既有Train roots与12题位→一次生成12候选→交叉内容审核11 ACCEPT/1 HOLD→66次直接作答＋11次结构化依据诊断→原始响应冻结→统一标签域评分→11份语义审核→新版反馈→下一消费者实际读取并写具体计划。没有新增对话、root、训练或Dev/Test调用。

这证明反馈已实际影响候选与下一步计划；尚未证明反馈让题目更难、训练更有效或完成全部五卡闭环。五卡中四卡落到合格候选，来源事实拒绝卡的唯一候选被HOLD。全部目标完成=false、formal reproduced=false、matched A/B=false。

## 内容质量与多样性

12候选包含8多选、4排序；11内容通过包含7多选、4排序。多选Gold2/3/4分别2/3/2道，Gold5为0。唯一Gold5候选FBR-R4-MC2已生成但未准入：题干直接给noon，与排除缺cutoff的bundle F理由冲突，来源内容本身没有因此改动。遵守预先零修订预算，不重试、不改Gold。

355项结构检查通过，124次作者引文均精确源子串；内容审核另由角色逐项重算语义和唯一序。角色共享文件系统，也有历史作者/审核交叉，不能当技术盲法或人工独立标注。

新关系包括派生时长排序，以及状态、版本、报告序对的多选表达；按审核分类计数：{dict(novelty)}。这些是同源任务/关系变体，不作为独立新事实家族；R3-clock排序还是同题型重测，R4报告排序只改变了选定锚点。后续训练池须按family分组去重，不能把排列和复问都算独立样本。

## Solver主要指标

S0=M-clean原模型，全文输入，temperature0.8/top-p0.9，enable_thinking=false；直接答案沿原生有限合法答案约束，max32，每题3种展示×2次采样。全部66次有效；准确率保留固定调用分母。

| 题型 | 内容通过 | 完全正确/调用 | 题均exact | Root均exact | 部分正确指标 |
|---|---:|---:|---:|---:|---:|
| 多选 | 7 | 13/42 | 30.95% | 18.06% | 题均Jaccard 63.73% |
| 排序 | 4 | 2/24 | 8.33% | 8.33% | 题均pair accuracy 69.58% |

排序合并序对147/198=74.24%，与题均69.58%因不同arity权重而不同。多选平均选择3.738项，对应平均Gold3.0项；51次错选、20次漏选，42次中只有1次全选。过选存在，但不能笼统归结为每次蒙全选。

本批不是与上一23题的匹配对照，题目构成、样本数及定向选择不同。不能因本轮聚合准确率更低就断言反馈提高难度；不能据此认定模型能力下降。11题与4 roots均为探索性小批。

## 依据诊断与审核

每道内容准入题固定配对perm00/sample00一次，max3072，无JSON grammar或答案约束；全文预算预检，最小余量929 token。实际11次返回，8格式有效、3无效，0截断、0重试。无效项FBR-R2-ORD1、FBR-R3-MC1、FBR-R3-MC3保留NOT_REVIEWABLE，不提取final救分。

8份有效诊断都为FAIL，但这不是8道题目错误：其中7份有来源可核查的反馈目标错误，另1份条件判断与final正确但引用失真。7份目标错误来自数字卡4条、时钟状态卡1条、报告序卡2条；不授予正式reproduced资格。

- 数字：R4-MC1六个引文均精确，仍把12:00−10:08写为1h43，称42分钟为1小时；R2-MC1把12:00−11:20写20分钟而非40。不能仅用引用接口问题解释。
- 时长排序：R1-ORD1正确算出部分51/133分钟却在最终次序逆置；另把cutoff区间换为arrival区间。区分抽取、对象绑定、计算和输出对齐。
- 请求/确认：R3-ORD1把19:15失败请求误作18:15，并排在18:15确认项之前。对应状态多选直接6/6，但其诊断缺项，不能冒称过程可靠。
- 报告序：R4-MC3仍正确处理旧点名两对，却误选另外两个反向序对；R4-ORD1违反T005<T007但保留T011<T013。不能说全部旧错对都复现。
- 条件：R3-MC2 direct5/6、诊断final及六项判断/理由总体正确，但0/6引文在指定turn精确，含把draft2已批准说成未批准的虚构引用；target failure=false不表示整张卡已被修复。

有效诊断共64引用，仅24在指定turn逐字匹配；精确引文不等于语义充分。短解释属于可见可审核输出，不声称真实隐藏推理。诊断MC exact1/7、排序0/4保持独立描述，不能与不同提示、预算及约束的direct结果推断CoT因果收益。

## 实际反馈回流

FEEDBACK_SNAPSHOT.json保留五张任务卡＋一张诊断接口卡。来源事实拒绝卡记CANDIDATE_HOLD_NO_SOLVER_TEST；条件卡记NO_NEW_VALID_TARGET_FAILURE_THIS_BATCH；数字/时钟/报告序只授予SOURCE_VERIFIED_LOCAL_TARGET_FAILURE_CONTINUES。

下一消费者已实际读取绑定packet、反馈语义及相关证据，写next_generation/agent_plan/，未生成下一候选。工具可见读取不是API完整prompt trace；程序写包和LLM实际消费分别记录。四张候选实现卡形成反馈→计划→候选→评估→审核→新反馈→下一计划链路，第五卡仍停在内容HOLD，故全目标闭环仍false，更不能当作训练收益结论。

## 成本和验证

共77 Solver调用，已知输入760424 tokens、输出7850 tokens、合计768274。两分支推理墙时合计87.22秒，包含两次模型加载合计33.63秒；不含作者生成、内容审核、预检、语义审核和报告时间。本地GPU，无货币费用数据。没有重试或隐藏额外Solver调用。

公开preflight1882项通过；根agent独立重算1734项通过；旧封存971+122=1093文件逐项hash不变。预检角色也是adapter作者，已披露非独立代码实现审核。根agent的首次重算误把多选集合按列表顺序比较，机械修复为集合比较；原raw、评分和模型不变。初始检查及更正保存在review/history。另有候选slot Counter工具错误更正，未改候选内容。

## 下一步

按已消费的新计划处理证据题范围冲突和诊断接口小对照，分开冻结预算；这批HOLD不追认通过，旧raw不重写。数值/时钟/报告序后续做预声明、固定source/mapping的局部检验，需要新root素材时才按A旧事件3场＋B新事件3场扩充。进入重训前仍需同family去重、共同评估准入、匹配反馈/无反馈训练池和对称资源；训练启动仍false。

## 文件入口

- exports/QUESTIONS_WITH_GOLD_ZH.md：全部12候选、Gold与内容处置。
- exports/DIAGNOSTIC_OUTPUTS_ZH.md：全部11份完整可见依据返回。
- exports/per_question.csv：每题直接评估表现。
- SEMANTIC_REVIEWS.jsonl：11份逐项来源审核。
- FEEDBACK_ZH.md、FEEDBACK_SNAPSHOT.json：新版回流反馈。
- next_generation/agent_plan/PLAN_ZH.md：实际消费反馈后的后续计划。
- runtime/eval_v1/：公开prompt、模型绑定、映射、raw冻结与原始评分。
'''
 (D/'REPORT_ZH.md').write_text(text)
 (D/'README_ZH.md').write_text('''# 反馈定向出题执行 v1

已完成12候选→11内容准入/1HOLD→66直接+11依据诊断→来源审核→新版反馈→下一消费者实际计划。四个既有Train roots，未新增对话、训练或Dev/Test调用。

主要结果：多选13/42=30.95%，排序2/24=8.33%；诊断8有效/3无效，7条局部目标错+1条正确答案但引用错。四卡候选实现，第五卡HOLD，不能声称全目标闭环或反馈收益。Gold5候选被HOLD，不在准入池。

阅读REPORT_ZH.md和SUMMARY.json。题目见exports/QUESTIONS_WITH_GOLD_ZH.md；完整诊断见exports/DIAGNOSTIC_OUTPUTS_ZH.md；新反馈见FEEDBACK_ZH.md；下一计划见next_generation/agent_plan/PLAN_ZH.md。历史sealed目录只读；本目录FINAL_MANIFEST封存后也不可改。
''')
 print(json.dumps(dict(status=s['status'],actual_next_consumer_read=s['actual_next_consumer_read'],report=str(D/'REPORT_ZH.md'),all_targets_loop_closed=False),ensure_ascii=False))
if __name__=='__main__':main()
