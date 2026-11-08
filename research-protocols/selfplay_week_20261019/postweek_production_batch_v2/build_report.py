"""Export the second production batch, retaining shortages and all solver results."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write(path, text):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def fraction(correct, count):
    return f"{correct}/{count}（{correct/count:.1%}）" if count else "无样本"


def main():
    candidates = read(HERE / "CANDIDATES_PRIVATE.json")
    review_doc = read(HERE / "CONTENT_REVIEW_AI.json")
    reviews = {r['candidate_id']: r for r in review_doc['reviews']}
    slots = read(HERE / "SLOT_LEDGER.json")['slots']
    selection = read(HERE / 'SELECTION_FREEZE.json')
    collection = read(HERE / 'COLLECTION_AUDIT.json')
    summary = read(HERE / 'execution/ANALYSIS_SUMMARY.json')
    raw = read(HERE / 'execution/RAW_RESPONSE_FREEZE.json')
    scores = read(HERE / 'execution/SCORES.json')['records']
    rows = read(HERE / 'execution/PUBLIC_REQUESTS.json')['requests']
    source_inventory = read(HERE / 'SOURCE_INVENTORY.json')
    reasons = {r['request_id']: r for r in read(HERE / 'execution/REASON_REVIEW_PACKET.json')['requests']}
    admitted = [c for c in candidates if reviews[c['candidate_id']]['decision'] == 'PASS']
    exported = []
    for c in admitted:
        exported.append({**c, 'production_batch': HERE.name,
                         'content_review_binding': binding(HERE / 'CONTENT_REVIEW_AI.json'),
                         'pool_status': 'CONTENT_APPROVED_TRAIN_CANDIDATE',
                         'solver_score_admission_filter': False,
                         'solver_reasons_are_training_targets': False,
                         'formal_B_targeted_generation': False})
    with (HERE / 'TRAINING_POOL_CANDIDATES.jsonl').open('x', encoding='utf-8') as stream:
        for record in exported:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    roots = []
    for selected in selection['roots']:
        rid = selected['root_id']
        rs = [s for s in slots if s['root_id'] == rid]
        rc = [c for c in admitted if c['root_id'] == rid]
        gold_counts = Counter(str(len(c['gold'])) for c in rc if c['question_type'] == 'multi_select')
        roots.append({'root_id': rid, 'planned_slots': 7, 'slot_statuses': dict(Counter(s['status'] for s in rs)),
                      'admitted': len(rc), 'mc': sum(c['question_type'] == 'multi_select' for c in rc),
                      'ordering': sum(c['question_type'] == 'ordering' for c in rc),
                      'mc_gold_histogram': dict(gold_counts),
                      'mc_gold_23456_complete': all(gold_counts[str(k)] >= 1 for k in range(2, 7)),
                      'all_seven_planned_slots_pass': sum(s['status'] == 'PASS' for s in rs) == 7})
    reason_flags = []
    for rid, record in reasons.items():
        checks = record.get('mechanical_consistency', {})
        if checks.get('answer_matches_supported_labels') is False:
            reason_flags.append({'request_id': rid, 'code': 'ITEM_JUDGMENTS_FINAL_SET_MISMATCH', 'checks': checks})
    operation_cells = []
    for condition in ('FULL', 'QUESTION_ONLY'):
        for operation in sorted({c['task_operation'] for c in admitted}):
            selected = [s for s in scores if s['condition'] == condition and s['task_operation'] == operation]
            operation_cells.append({'condition': condition, 'task_operation': operation,
                                    'requests': len(selected), 'correct': sum(s['exact'] for s in selected)})
    pool = {'schema_version': 'production_training_candidate_pool_v1', 'status': 'FROZEN_CONTENT_APPROVED_CANDIDATES',
            'planned_slots': 70, 'exported_unique_questions': len(exported), 'roots': 10,
            'question_types': dict(Counter(c['question_type'] for c in admitted)),
            'task_operations': dict(Counter(c['task_operation'] for c in admitted)),
            'mc_gold_histogram': dict(Counter(str(len(c['gold'])) for c in admitted if c['question_type'] == 'multi_select')),
            'root_inventory': roots, 'candidate_file': binding(HERE / 'TRAINING_POOL_CANDIDATES.jsonl'),
            'content_approved_pool_only': True, 'training_mix_built': False, 'training_launched': False,
            'solver_scores_used_for_admission': False, 'formal_weakness_targeting': False,
            'source_versions_reused': True, 'new_dialogue_turns': 0,
            'reason_semantics_automatically_certified': False, 'feedback_causal_gain_proven': False}
    save(HERE / 'POOL_MANIFEST.json', pool)
    save(HERE / 'ROOT_VALIDATION.json', {'status': 'PRODUCTION_BATCH_COMPLETE_RETAINING_SHORTFALLS',
         'root_inventory': roots, 'collection': collection, 'solver_summary': summary,
         'invalid_response_reasons': dict(Counter(s.get('invalid_error') for s in scores if not s['valid'])),
         'mechanical_judgment_final_mismatches': reason_flags,
         'operation_cells': operation_cells,
         'reviewer_is_human': False, 'gold_blind_review': False,
         'new_dialogue_turns': 0, 'new_training_runs': 0,
         'semantic_shortcuts_globally_eliminated': False,
         'formal_reproduced_weakness_claim': False, 'feedback_causal_benefit_proven': False})
    mc_count = pool['question_types'].get('multi_select', 0)
    order_count = pool['question_types'].get('ordering', 0)
    shortfalls = [s for s in slots if s['status'] == 'MATERIAL_SHORTFALL']
    held = [s for s in slots if s['status'] in ('HOLD', 'FAIL')]
    lines = ['# 修复版第二批批量生产报告', '',
             f"10个Train roots、70个冻结槽位；提交{len(candidates)}道初稿，最终内容审核通过{len(admitted)}题（多选{mc_count}、排序{order_count}），素材缺额{len(shortfalls)}槽，审核HOLD/FAIL {len(held)}槽。", '',
             '本批使用历史已审核完整对话，没有新增对话扩充、DEV/TEST调用或训练。候选经另一AI逐项重建答案与来源依赖审核；不是人工或Gold盲审。合格题不按solver成绩筛选，原稿、一次修复与失败记录完整保留。', '',
             f"来源共{source_inventory['total_turns']}个turn；完整对话序列化平均{source_inventory['mean_source_tokens']:.1f} tokens，范围{source_inventory['min_source_tokens']}—{source_inventory['max_source_tokens']}（Qwen3-4B tokenizer，未含题目和系统提示）。逐root计数见SOURCE_INVENTORY.json。", '',
             '## 生产收尾和稳定性', '',
             '- 来源绑定由选择冻结自动生成绝对路径、SHA256和字节数；逐字引文、说话人及排序证据图由程序复核。',
             '- 排序按具体操作审核唯一全序、提及顺序、常识流程、编辑提示和来源必要性；合法话语重建不因顺序与原文重合而机械拒绝。',
             '- 排序与多选由程序打乱，两个展示排列不同，答案映射私有保存。Gold不参与排列选择。',
             f"- 初稿联合通过：{collection['initial_joint_pass']}/70={collection['initial_joint_pass_rate_planned']:.1%}；仅已提交题分母：{collection['initial_joint_pass_rate_submitted']:.1%}。冻结的90%放大门槛{'达到' if collection['initial_joint_90_percent_scaleup_threshold_met'] else '未达到'}；缺额不从计划分母删去。", '',
             f"- 初稿语义通过：{collection['initial_semantic_content_pass']}/{len(candidates)}个已提交题；计划70槽和提交题两个分母均保留。", '',
             '## 题库覆盖', '', '| 类型／Gold | 数量 |', '| --- | --- |',
             f'| 多选 | {mc_count} |', f'| 排序 | {order_count} |']
    lines.extend(f"| Gold {k} | {pool['mc_gold_histogram'].get(str(k), 0)} |" for k in range(2, 7))
    lines.extend(['', '6 Gold为合法全选控制，不代表干扰项设计质量；Gold桶、题目操作和root存在混杂，桶间准确率不能解释为Gold数量的因果作用。', '',
                  '| root | 多选通过 | 排序通过 | 完整7槽 |', '| --- | --- | --- | --- |'])
    lines.extend(f"| {r['root_id']} | {r['mc']} | {r['ordering']} | {'是' if r['all_seven_planned_slots_pass'] else '否'} |" for r in roots)
    lines.extend(['', '## 缺额及未准入槽', ''])
    for slot in shortfalls + held:
        lines.append(f"- {slot['candidate_id']}：{slot['status']}；{json.dumps({k:v for k,v in slot.items() if k not in ('candidate_id','root_id','slot')}, ensure_ascii=False)}")
    if not shortfalls and not held:
        lines.append('无。')
    lines.extend(['', '## Solver实测', '',
                  '| 条件 | 有效回答 | 多选完全正确 | 排序完全正确 | 合计完全正确 |',
                  '| --- | --- | --- | --- | --- |'])
    for cell in summary['conditions']:
        lines.append(f"| {cell['condition']} | {cell['valid_requests']}/{cell['frozen_requests']} | {fraction(cell['mc_correct'], cell['mc_requests'])} | {fraction(cell['ordering_correct'], cell['ordering_requests'])} | {fraction(cell['correct'], cell['frozen_requests'])} |")
    lines.extend(['', '各操作在FULL条件下的表现（每题两个展示；不同操作的素材不匹配，不能据此作因果难度比较）：', '',
                  '| 操作 | 完全正确 |', '| --- | --- |'])
    lines.extend(f"| {cell['task_operation']} | {fraction(cell['correct'], cell['requests'])} |"
                 for cell in operation_cells if cell['condition'] == 'FULL')
    full = next(c for c in summary['conditions'] if c['condition'] == 'FULL')
    lines.extend(['', f"多选固定全选策略基线：{fraction(full['mc_allselect_correct'], full['mc_requests'])}；排序成对顺序正确率：{fraction(full['ordering_pairwise_correct'], full['ordering_pairwise_total'])}。全选基线仅计最终集合，不证明逐项核实。"])
    lines.extend(['', f"真实返回{raw['returned_requests']}/{raw['requests']}；GPU耗时{raw['wall_seconds']:.1f}秒（含模型加载与关闭）；最长prompt {max(r['prompt_tokens'] for r in rows)} tokens。无输入截断、无自动重试。", '',
                  'FULL为完整来源阅读；QUESTION_ONLY只用于捷径诊断。每道题两个展示属于相关观测，不能把请求数当独立题数。无效响应留在冻结分母，最终答案正确不认证理由与引用。', '',
                  f"程序发现多选逐项判断与最终选项集合不一致的响应{len(reason_flags)}条；全部可见理由保留在REASON_REVIEW_PACKET及原始返回中。没有独立完成语义审核的理由不自动升级为正式反馈。", '',
                  '## 交付与边界', '',
                  '- TRAINING_POOL_CANDIDATES.jsonl：最终内容合格的独立题，待后续训练混合与输入冻结；不包含solver理由训练标签。',
                  '- CANDIDATES_PRIVATE.json / CONTENT_REVIEW_AI.json：全部已提交候选及最终逐题审核。',
                  '- author_a、author_b、review_a、review_b：原稿、读取回执、初审、一次修复和复审。',
                  '- SLOT_LEDGER.json：全部70槽，包括缺额与HOLD，不能只展示成功题。',
                  '- execution：公开请求、私有映射、全部原始响应、评分和理由。',
                  '- NEW_QUESTIONS_AND_REASONS_ZH.md：逐题题干、选项、Gold和真实回答。', '',
                  '本批为普通生产，作者B此前参与首批理由审核；两作者组不构成反馈/无反馈对照，输入暴露见AUTHOR_EXPOSURE_DISCLOSURE.json。', '',
                  '本批验证当前规范下的生产产物与审核记录；不等于任意root稳定满额、不等于自动Qwen生成质量、未见root泛化、正式弱点复现或反馈训练因果收益。对话扩充当前版本的真实端到端验证另行进行。', ''])
    write(HERE / 'REPORT_ZH.md', '\n'.join(lines))
    details = ['# 第二批题目、来源与真实回答', '', '来源/Gold/作者分析为私有审核评分资料，没有进入solver公开请求。', '']
    by_score = {s['request_id']: s for s in scores}
    for candidate in candidates:
        cid = candidate['candidate_id']
        details.extend([f'## {cid}', '', f"审核：{reviews[cid]['decision']}；Gold：{' '.join(candidate['gold'])}", '', candidate['question'], ''])
        field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
        details.extend(f'- {label}. {text}' for label, text in candidate[field].items())
        details.extend(['', f"完整来源：[对话]({candidate['source_binding']['path']})", ''])
        for row in (r for r in rows if r['candidate_id'] == cid):
            rid = row['request_id']; score = by_score[rid]; record = reasons[rid]
            details.extend([f"### {row['variant_id']} / {row['condition']}", ''])
            details.extend(f'- {label}. {text}' for label, text in row['display'].items())
            details.extend(['', f"显示Gold：{' '.join(score['display_gold'])}；有效：{score['valid']}；完全正确：{score['exact']}；无效原因：{score.get('invalid_error') or '无'}。", '',
                            f"[原始响应]({record['raw_binding']['path']}) · [实际prompt]({row['prompt_path']})", ''])
            parsed = record.get('parsed')
            if parsed:
                details.append('```json\n' + json.dumps(parsed, ensure_ascii=False, indent=2) + '\n```\n')
    write(HERE / 'NEW_QUESTIONS_AND_REASONS_ZH.md', '\n'.join(details))
    write(HERE / 'README_ZH.md', '# 第二批批量生产交接\n\n先读REPORT_ZH.md；逐题浏览NEW_QUESTIONS_AND_REASONS_ZH.md。\n\nTRAINING_POOL_CANDIDATES.jsonl是内容合格训练候选，完整计数见POOL_MANIFEST.json；70个计划槽的失败/缺额见SLOT_LEDGER.json。对话以SELECTION_FREEZE.json绑定的既有完整来源为准。\n\n本批未新增对话、未训练。solver理由是模型可见输出，不作正确推理标签；原始返回在execution/raw，评分见execution/SCORES.json。初稿审核、修复和最终审核分别保留。复现入口是assemble.py、run.py prepare/run --execute-gpu/analyze和build_report.py，均禁止覆盖或重跑既有结果。最终hash见FINAL_MANIFEST.json。\n')
    print(json.dumps({'admitted': len(admitted), 'mc': mc_count, 'ordering': order_count,
                      'material_shortfalls': len(shortfalls), 'held': len(held)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
