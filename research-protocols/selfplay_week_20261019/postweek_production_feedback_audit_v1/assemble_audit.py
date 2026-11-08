"""Validate full-response semantic review coverage and export exploratory cards."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare import HERE, BATCH, binding, read, save

VALID = {'PASS', 'FAIL', 'UNKNOWN', 'NA'}


def decision(value):
    return value['decision'] if isinstance(value, dict) else value


def main():
    freeze = read(HERE / 'REVIEW_FREEZE.json')
    for b in freeze['bindings']:
        assert binding(b['path']) == b, b['path']
    all_reviews, bindings = [], [binding(HERE / 'REVIEW_FREEZE.json')]
    source_map = {s['root_id']: s for s in freeze['sources']}
    response_inputs, candidates = {}, {}
    reviewers = []
    for group in ('a', 'b'):
        packet = read(HERE / f'REVIEW_PACKET_{group.upper()}.json')
        p = HERE / f'review_{group}/SEMANTIC_REVIEW_AI.json'
        review = read(p)
        assert review['gold_visible'] is True and review['human'] is False
        reviewer = review['reviewer_id']
        reviewers.append(reviewer)
        receipts = {s['root_id']: s for s in review['source_read_receipt']}
        assert set(receipts) == set(packet['roots'])
        for root, receipt in receipts.items():
            source = source_map[root]
            assert receipt['source_binding'] == source['source_binding']
            assert receipt['transcript_binding'] == source['transcript_binding']
            assert receipt['turns_read'] == source['turns'] and receipt['complete'] is True
            assert binding(source['source_binding']['path']) == source['source_binding']
            assert binding(source['transcript_binding']['path']) == source['transcript_binding']
        expected = {q['candidate']['candidate_id']: q for q in packet['questions']}
        reviews = review['reviews']
        assert len(reviews) == len(expected) and {r['candidate_id'] for r in reviews} == set(expected)
        for r in reviews:
            q = expected[r['candidate_id']]
            c = q['candidate']
            assert reviewer != c['author_id']
            assert r['root_id'] == c['root_id']
            assert r['content_decision'] in {'PASS', 'HOLD', 'FAIL'}
            assert r['content_notes'] and r['reconstructed_gold']
            if r['content_decision'] == 'PASS':
                if c['question_type'] == 'multi_select':
                    assert set(r['reconstructed_gold']) == set(c['gold'])
                else:
                    assert r['reconstructed_gold'] == c['gold']
            sources = read(c['source_binding']['path'])
            refs = {t['turn_ref'] for t in sources['turns']}
            expected_responses = {x['request_id']: x for x in q['responses']}
            assert len(r['responses']) == 2 and {x['request_id'] for x in r['responses']} == set(expected_responses)
            for response in r['responses']:
                rid = response['request_id']
                inp = expected_responses[rid]
                assert binding(inp['raw_binding']['path']) == inp['raw_binding']
                assert set(response['items']) == set(inp['display'])
                for item in response['items'].values():
                    for key in ('judgment_correct', 'citation_support', 'reason_support'):
                        assert decision(item[key]) in VALID, (rid, key)
                    assert item['note'] and set(item['reviewer_turn_refs']) <= refs
                for key in ('answer_reason_support', 'summary_final_consistency', 'item_final_consistency'):
                    assert decision(response[key]) in VALID, (rid, key)
                    if isinstance(response[key], dict):
                        assert response[key].get('note') or response[key].get('rationale')
                assert set(response['error_dimensions']) <= set(freeze['review_contract']['dimensions'])
                assert response['notes']
                response_inputs[rid] = inp
                response['reviewer_id'] = reviewer
            r['reviewer_id'] = reviewer
            candidates[c['candidate_id']] = c
        all_reviews.extend(reviews)
        bindings.append(binding(p))
    assert len(set(reviewers)) == 2
    assert len(all_reviews) == 66 and len(response_inputs) == 132
    item_counts = {k: Counter() for k in ('judgment_correct', 'citation_support', 'reason_support')}
    response_counts = {k: Counter() for k in ('answer_reason_support', 'summary_final_consistency', 'item_final_consistency')}
    evidence = defaultdict(list)
    clean, exact_clean = [], []
    concerns = []
    flat = []
    for q in all_reviews:
        cid = q['candidate_id']
        if q['content_decision'] != 'PASS':
            concerns.append({'candidate_id': cid, 'root_id': q['root_id'],
                             'decision': q['content_decision'], 'notes': q['content_notes'],
                             'reconstructed_gold': q['reconstructed_gold'],
                             'frozen_original_gold': candidates[cid]['gold'],
                             'original_batch_unchanged': True})
        for r in q['responses']:
            inp = response_inputs[r['request_id']]
            for item in r['items'].values():
                for k in item_counts:
                    item_counts[k][decision(item[k])] += 1
            for k in response_counts:
                response_counts[k][decision(r[k])] += 1
            all_supported = inp['score']['valid'] and all(
                decision(item[k]) == 'PASS' for item in r['items'].values()
                for k in ('judgment_correct', 'citation_support', 'reason_support')) and all(
                decision(r[k]) in {'PASS', 'NA'} for k in response_counts)
            if all_supported:
                clean.append(r['request_id'])
                if inp['score']['exact']:
                    exact_clean.append(r['request_id'])
            record = {**r, 'candidate_id': cid, 'root_id': q['root_id'],
                      'question_type': candidates[cid]['question_type'],
                      'task_operation': candidates[cid]['task_operation'],
                      'content_decision': q['content_decision'], 'score': inp['score'],
                      'raw_binding': inp['raw_binding'], 'all_review_dimensions_pass': all_supported}
            flat.append(record)
            for dim in r['error_dimensions']:
                evidence[dim].append({'request_id': r['request_id'], 'candidate_id': cid,
                                      'root_id': q['root_id'], 'exact': inp['score']['exact'],
                                      'valid': inp['score']['valid'], 'content_decision': q['content_decision'],
                                      'notes': r['notes'], 'reviewer_id': r['reviewer_id']})
    cards = []
    designs = {
        'QUANTITY_OR_TIME': ('数量与时间关系核对', 'MC_QUANTITATIVE_RELATION',
                            '从来源分别绑定基准值、比较值和单位；在选项中只局部交换差值、比例、时间长度或比较方向，保留独立可核查的正项。',
                            '即使提取出部分正确数字，仍可能误算差值、比较方向或单位；也可能只是检索、标签关联或引用失败。'),
        'SOURCE_STATE': ('提议、确认与完成状态区分', 'MC_EVIDENCE_SELECTION',
                         '用同一事件的提议、接受、记录及执行状态构造引用—主张配对；只改变一项状态或归属，不凭空新增已完成事件。',
                         '可能把意向、安排或信息询问当成确认或已完成；也可能只是引用不充分或选项标签错位。'),
        'ENTITY_BINDING': ('人物与属性归属核对', 'MC_CONSTRAINT_CONJUNCTION',
                           '在同场或跨场的真实人物、地点、数量之间做最小属性错绑，确保全文可唯一判断且正项有独立事实支持。',
                           '可能把真实属性绑定给错误人物或对象；也可能是显示标签或长合取遗漏。'),
        'ORDER_RELATION': ('来源支持的顺序关系', 'ORDER_EXPLICIT_PROCEDURE',
                           '先独立审核明确流程、话语演变或时间关系的唯一全序，再打乱显示；保留与提及顺序不同的可用证据，不虚构依赖。',
                           '可能在识别项目后仍违反来源顺序关系；也可能是引用、操作范围或不完整输出。')
    }
    for dim, (title, operation, construction, prediction) in designs.items():
        support = [e for e in evidence[dim] if e['valid'] and e['content_decision'] == 'PASS']
        if not support:
            continue
        cards.append({'card_id': 'PROD1-HYP-' + dim, 'status': 'EXPLORATORY_OBSERVATION_NOT_REPRODUCED',
                      'dimension': dim, 'title': title, 'solver_role': 'B1',
                      'supporting_requests': support,
                      'supporting_unique_questions': len({e['candidate_id'] for e in support}),
                      'supporting_roots': sorted({e['root_id'] for e in support}),
                      'target_operation_example': operation, 'construction_constraints': construction,
                      'falsifiable_expected_error_and_alternatives': prediction,
                      'required_next_validation': 'Prospectively freeze matched Train-only tasks and controlled probes across roots and fact families, including counterexamples; two displays alone do not meet formal weakness admission.',
                      'challenger_consumed': False, 'formal_weakness_admission': False,
                      'training_target': False, 'internal_cause_claim': False})
    summary = {
        'schema_version': 'production_full_semantic_audit_summary_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'ALL_FULL_RESPONSES_SEMANTICALLY_REVIEWED_WITH_RETAINED_FAILURES',
        'questions': len(all_reviews), 'responses': len(flat), 'roots': 10,
        'content_decisions': dict(Counter(q['content_decision'] for q in all_reviews)),
        'item_counts': {k: dict(v) for k, v in item_counts.items()},
        'response_counts': {k: dict(v) for k, v in response_counts.items()},
        'exact_responses': sum(x['score']['exact'] for x in flat),
        'invalid_responses': sum(not x['score']['valid'] for x in flat),
        'all_review_dimensions_pass': len(clean), 'exact_and_all_review_dimensions_pass': len(exact_clean),
        'fully_reviewed_response_ids': clean,
        'error_dimension_counts': {k: {'responses': len(v), 'questions': len({x['candidate_id'] for x in v}),
                                       'roots': len({x['root_id'] for x in v})} for k, v in evidence.items()},
        'exploratory_cards': len(cards), 'new_solver_calls': 0, 'new_training_runs': 0,
        'question_rewrites': 0, 'formal_weakness_admission': False, 'feedback_causal_gain_proven': False,
        'reviewer_is_human': False, 'gold_blind': False, 'faithful_internal_cot_claim': False,
        'bindings': bindings,
    }
    save(HERE / 'SEMANTIC_REVIEW_AI.json', {'reviews': all_reviews, 'bindings': bindings})
    save(HERE / 'FULL_RESPONSE_REVIEWS.json', {'records': flat})
    save(HERE / 'TRAINING_CONTENT_CONCERNS.json', {'concerns': concerns,
         'policy': 'Do not alter the sealed original batch. Resolve any listed concern explicitly before a new training freeze.',
         'original_batch': binding(BATCH / 'FINAL_MANIFEST.json')})
    save(HERE / 'EXPLORATORY_WEAKNESS_CARDS.json', {'cards': cards,
         'semantic_review_binding': binding(HERE / 'SEMANTIC_REVIEW_AI.json'),
         'formal_feedback_release': False, 'challenger_consumed': False})
    save(HERE / 'SUMMARY.json', summary)
    lines = ['# 首批生产题的完整对话回答语义审核', '',
             f"已审核66题、10个Train roots的132条FULL真实响应，包含答对与答错；未重新调用solver，未改原题、Gold或响应。内容结论：{summary['content_decisions']}。", '',
             f"最终完全正确{summary['exact_responses']}/132；所有逐项判断、引用、理由及适用一致性检查均通过{len(clean)}/132，其中答案也正确{len(exact_clean)}条。无效回答{summary['invalid_responses']}条保留。", '',
             '以上为两位独立于原作者的AI审核判断，审核可见Gold，不是人工盲审；模型的可见理由不等于内部思维链。', '',
             '| 审核项 | PASS | FAIL | UNKNOWN | NA |', '| --- | ---: | ---: | ---: | ---: |']
    for k, counts in {**item_counts, **response_counts}.items():
        lines.append('| ' + k + ' | ' + ' | '.join(str(counts[s]) for s in ['PASS', 'FAIL', 'UNKNOWN', 'NA']) + ' |')
    lines += ['', '错误维度可重叠；同题双排列不是两道独立题。判断类别不精确与最终误选分别记录，排序逐项支持不直接编码排列。', '',
              '| 观察维度 | 响应 | 独立题 | root |', '| --- | ---: | ---: | ---: |']
    for dim, values in summary['error_dimension_counts'].items():
        lines.append(f"| {dim} | {values['responses']} | {values['questions']} | {values['roots']} |")
    lines += ['', '已导出探索性候选卡，包含真实错误、来源绑定、任务构造限制、预期错误和竞争解释。尚未做新的受控复现，不升级为正式弱点，不声称反馈带来训练收益。', '',
              '逐项审核见SEMANTIC_REVIEW_AI.json；可复核真实响应索引见FULL_RESPONSE_REVIEWS.json；候选卡见EXPLORATORY_WEAKNESS_CARDS.json。', '',
              '训练内容疑点单独记录于TRAINING_CONTENT_CONCERNS.json。原封存题库保持历史记录；如有疑点，后续训练准入须显式处置，不能静默覆盖。', '']
    with (HERE / 'REPORT_ZH.md').open('x') as stream:
        stream.write('\n'.join(lines))
    examples = ['# 真实输出与审核示例', '',
                '示例来自事前纳入的全部FULL审核记录；用于阅读具体错误，不估计总体频率。原始输出未修改。', '']
    for card in cards:
        support = card['supporting_requests']
        example = next((s for s in support if not s['exact']), support[0])
        rid = example['request_id']
        inp = response_inputs[rid]
        review = next(r for r in flat if r['request_id'] == rid)
        candidate = candidates[example['candidate_id']]
        examples += [f"## {card['title']}：{rid}", '', candidate['question'], '']
        examples += [f'- {label}. {text}' for label, text in inp['display'].items()]
        examples += ['', f"显示Gold：{' '.join(inp['score']['display_gold'])}；实际框内答案见下方模型原文。", '',
                     f"[完整来源]({candidate['source_binding']['path']}) · [原始返回]({inp['raw_binding']['path']})", '',
                     '```json\n' + json.dumps(inp['parsed'], ensure_ascii=False, indent=2) + '\n```', '',
                     '审核观察：' + review['notes'], '']
        for label, item in review['items'].items():
            if any(decision(item[k]) == 'FAIL' for k in ('judgment_correct', 'citation_support', 'reason_support')):
                examples.append(f"- {label}：{item['note']}；复核来源：{'、'.join(item['reviewer_turn_refs'])}")
        examples += ['', '预期错误及竞争解释：' + card['falsifiable_expected_error_and_alternatives'], '']
    with (HERE / 'EXAMPLES_ZH.md').open('x') as stream:
        stream.write('\n'.join(examples))
    print(json.dumps({k: summary[k] for k in ['status', 'questions', 'responses', 'content_decisions',
                     'all_review_dimensions_pass', 'exact_and_all_review_dimensions_pass', 'exploratory_cards']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
