"""Build the readable full-pipeline summary from final audited artifacts."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import json

HOME = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def percent(v): return f"{float(Fraction(v))*100:.1f}%" if v is not None else "n/a"
def is_control(q): return q['question_type'] == 'multi_select' and len(q['gold']) == 6

def main():
    assert not (HOME/'PILOT_COMPLETE.json').exists()
    new=read(HOME/'AUDITED_NEW_CANDIDATES.json')
    result=read(Path(new['result_audit']['path']))
    assert result['status']=='PASS'
    shard=Path(result['shards'][0])
    allocation=read(shard/'multiselect/ALLOCATION.json')
    pool=read(shard/'EVALUATION_POOL.json')['questions']
    content=[q for q in pool if not is_control(q)]
    gold_content={str(g):sum(q['question_type']=='multi_select' and len(q['gold'])==g for q in content) for g in range(2,6)}
    root_tokens={r['root_id']:r for r in result['expansion_metrics']['roots']}
    families={r['root_id']:r for r in allocation['families']}
    ordered_public=read(shard/'ordering/PUBLIC_QUESTIONS.json')['questions']
    accepted_roots=result['expansion_metrics']['accepted_new_scene_roots']
    root_count=len(result['per_root'])
    model_calls={side:sum(q['calls'] for q in result['replayed_models'][side]) for side in ('trained','base')}
    assert model_calls['trained']==model_calls['base']
    new_multi_calls=result['by_type']['multi_select']['trained']['calls']
    ordering_calls=result['by_type']['ordering']['trained']['calls']
    control_calls=model_calls['trained']-new_multi_calls-ordering_calls
    lines=['# v90 四 root 完整流程试跑', '',
        f'本轮尝试{root_count}个root，{accepted_roots}个完成扩充和后续流程，{root_count-accepted_roots}个停在planner。后续包括新父题、多选与排序出题、完整源审核、冻结、第一轮已训练模型与初始模型在本轮同题的配对评测、原始响应复算和导出。全部原稿及初判保留。', '',
        f"扩充成功 {accepted_roots}/{root_count} 个 root，新增 {result['expansion_metrics']['new_turns']} turns、{result['expansion_metrics']['new_body_tokens']} Qwen3-4B 正文 tokens。正文 tokens 按每轮 text 独立编码，不含 speaker/ref、问题或请求包装。", '',
        '| Root | 原有/新增 turns | 扩充及父题 | 新多选：组装/内容通过/入选 | 排序：生成/内容通过/入选 |',
        '|---|---:|---|---:|---:|']
    for r in result['per_root']:
        root=r['root_id'];t=root_tokens[root];f=families.get(root)
        generated_m=len(f['children']) if f else 0
        generated_o=sum(q['root_id']==root for q in ordered_public)
        stage='READY；新父题通过' if r['parent_prepared'] else 'planner 输出6项发展，违反2–5项契约'
        lines.append(f"| {root.rsplit('-',1)[-1]} | {t['historical_turns']}/{t['new_turns']} | {stage} | {generated_m}/{r['content_usable']['multi_select']}/{r['selected']['multi_select']} | {generated_o}/{r['content_usable']['ordering']}/{r['selected']['ordering']} |")
    lines += ['', '多选 Gold 数量在不同阶段的结果：', '', '| Gold | 组装 | 内容通过 | 最终入选 |','|---|---:|---:|---:|']
    for g in range(2,6):
        k=str(g);lines.append(f"| {g} | {allocation['assembled_distribution'][k]} | {gold_content[k]} | {new['gold_distribution'][k]} |")
    lines += ['', '103866 的 Gold2/3 缺口来自明确矛盾事实不足；135466 的 Gold2/3/5 已组装，但含 staying in/later 的歧义选项，整题复审排除。present progressive 可以表示未来安排；另一时间的安排可以与现在出门共存。初判与修订判均保留，生成题目没有修改。', '',
        f"排序共生成{len(ordered_public)}题，内容通过{sum(q['question_type']=='ordering' for q in content)}题。135466 的第2题将“告知礼物限制”“确认能否拜访”强化为“检查限制”“安排拜访”，因全命题/模态支持不足而排除。", '',
        '两道新父题 Gold6 是对照题，参与评测但不计入新训练候选；排序完整6项排列按排序题计算。', '',
        f"配对评测按完整对话、相同随机显示排列、seed及采样设置运行，每模型{model_calls['trained']}次（新多选{new_multi_calls}次、对照多选{control_calls}次、排序{ordering_calls}次）。下表排除 Gold6 多选对照：", '',
        '| 题型 | trained 完全正确/调用 | base 完全正确/调用 | trained/base 平均 partial |','|---|---|---|---|']
    for kind,label in [('multi_select','多选'),('ordering','排序')]:
        a=result['by_type'][kind]['trained'];b=result['by_type'][kind]['base']
        lines.append(f"| {label} | {a['exact_correct']}/{a['calls']} ({percent(a['exact_accuracy'])}) | {b['exact_correct']}/{b['calls']} ({percent(b['exact_accuracy'])}) | {float(Fraction(a['partial_mean'])):.4f} / {float(Fraction(b['partial_mean'])):.4f} |")
    lines += ['', f"本轮最终入选 {new['candidate_count']} 题：多选 {new['by_type']['multi_select']}、排序 {new['by_type']['ordering']}；累计候选 {result['cumulative_count']} 题。入选要求全部作答有效，且 trained 平均 partial 严格在(0,1)内；内容审核与难度入选分别计数。", '',
        '扩充质量为“可用，有警告”，信息密度和剧情多样性中等。103866 订位回述铺垫不足、后半客套较多；135466 面试申请者通过前台收留言不够自然，联系方式查询与历史模板重复。新增27轮没有明确阻断问题；历史缺陷独立列出。', '',
        '当前完整试跑仍不能证明量产稳定：四个历史训练 root 只有两个扩充成功，Gold2/3没有通过整题审核。数值审计验证保存的审核记录、调用、得分及导出，并不重新证明语义。兄弟题与多次 rollout 具有依赖性；本轮未训练、未正式入训练池。', '',
        '后续应优先修复 planner 项数契约，以及混淆项保留同一事实、时间和模态的要求；随后再做独立 root 的完整流程试跑。', '',
        '记录与失败定位：', '',
        '- [全部入选新题及答案](AUDITED_NEW_CANDIDATES.json)',
        '- [累计候选](CUMULATIVE_AUDITED_NEW_CANDIDATES.json)',
        '- [出题、语义审核及难度失败索引](FAILURE_INDEX.json)',
        '- [前两次离线准备失败和修复](evidence/PREPARATION_FAILURE_INDEX.json)',
        '- [整题接入前缺少机械审核记录的失败](shards/shard_01r2/evidence/FINAL_REVIEW_INTEGRATION_ATTEMPT1.json)',
        '- [初次输入审计失败及读取方式修复](evidence/audit_tool_revisions/BINDING_FIX_RECEIPT.json)',
        '- [时间歧义独立复核](evidence/TEMPORAL_OPTION_SPOTCHECK.json)',
        '- [初判保留及最终修订绑定](evidence/superseded_reviews/GROUP_B_REVISION_COMPLETE.json)',
        '- [独立扩充质量审核](evidence/EXPANSION_QUALITY_INDEPENDENT_REVIEW.json)',
        '- [生成的全部公开多选题](shards/shard_01r2/multiselect/PUBLIC_QUESTIONS.json)',
        '- [生成的全部公开排序题](shards/shard_01r2/ordering/PUBLIC_QUESTIONS.json)',
        '- [全部文件索引，封存时生成](ARTIFACT_INDEX.md)', '']
    with (HOME/'COMPLETE_FLOW_RESULTS_ZH.md').open('x') as f:f.write('\n'.join(lines))
    print({'report':str(HOME/'COMPLETE_FLOW_RESULTS_ZH.md'),'selected':new['candidate_count']})

if __name__=='__main__':main()
