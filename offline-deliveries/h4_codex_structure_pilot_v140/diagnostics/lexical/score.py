#!/usr/bin/env python3
"""Score committed predictions; Gold is never imported by predict.py."""
import collections
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def strata(q):
    age = q['new_or_old']
    author = q['author_group_for_reporting_only']
    quote = 'quoted' if q['has_double_quoted_item'] else 'unquoted'
    return [('all', 'all'), ('new_or_old', age), ('root', q['root_id']),
            ('author', author), ('has_double_quoted_item', quote),
            ('author_x_new_or_old', author + '_' + age),
            ('quote_x_new_or_old', quote + '_' + age)]


def metrics(rows):
    n = len(rows)
    pairs = sum(r['pairs'] for r in rows)
    correct = sum(r['correct_pairs'] for r in rows)
    resolved = sum(r['resolved_pairs'] for r in rows)
    items = sum(r['items'] for r in rows)
    return {
        'questions': n,
        'whole_correct': sum(r['whole_correct'] for r in rows),
        'whole_accuracy': sum(r['whole_correct'] for r in rows) / n if n else None,
        'whole_unresolved': sum(r['predicted_order'] is None for r in rows),
        'pairs': pairs, 'correct_pairs': correct, 'resolved_pairs': resolved,
        'unknown_pairs': pairs - resolved,
        'pair_accuracy': correct / pairs if pairs else None,
        'pair_accuracy_resolved': correct / resolved if resolved else None,
        'pair_coverage': resolved / pairs if pairs else None,
        'items': items,
        'unique_top_turn_correct': sum(r['unique_top_turn_correct'] for r in rows),
        'top_turn_contains_gold': sum(r['top_turn_contains_gold'] for r in rows),
        'tied_top_items': sum(r['tied_top_items'] for r in rows),
        'unmatched_items': sum(r['unmatched_items'] for r in rows),
    }


def main():
    config = json.loads((OUT / 'CONFIG.json').read_text())
    precommit = json.loads((OUT / 'PRECOMMIT.json').read_text())
    commit = json.loads((OUT / 'PREDICTIONS_COMMIT.json').read_text())
    for filename, expected in [('PREDICTIONS.json', commit['predictions_sha256']),
                               ('predict.py', precommit['predict_py_sha256']),
                               ('CONFIG.json', precommit['config_sha256']),
                               ('FEATURES.json', commit['features_sha256'])]:
        assert digest(OUT / filename) == expected, filename
    predictions = json.loads((OUT / 'PREDICTIONS.json').read_text())
    assert predictions['gold_read'] is False
    gold_path = BASE / 'private/ADMITTED_QUESTIONS.json'
    admitted = json.loads(gold_path.read_text())['questions']
    admitted = {q['candidate_id']: q for q in admitted if q['question_type'] == 'ordering'}
    assert len(admitted) == len(predictions['questions']) == 16
    sources = {}
    gold_checks = []
    detail = []
    groups = collections.defaultdict(list)
    source_groups = collections.defaultdict(list)
    for pred in predictions['questions']:
        gold = admitted[pred['candidate_id']]
        assert list(pred['items']) == list(gold['items'])
        assert all(pred['items'][i]['text'] == gold['items'][i] for i in gold['items'])
        root = pred['root_id']
        if root not in sources:
            source_path = BASE / 'sources' / (root + '.json')
            assert digest(source_path) == predictions['input_sha256'][str(source_path.relative_to(BASE))]
            sources[root] = json.loads(source_path.read_text())['turns']
        turns = sources[root]
        positions = {}
        item_checks = {}
        for item, span in gold['order_anchor_spans'].items():
            hits = [(i, t) for i, t in enumerate(turns) if t['turn_ref'] == span['turn_ref']]
            assert len(hits) == 1, (pred['candidate_id'], item, 'Gold turn reference not unique')
            index, turn = hits[0]
            assert turn['speaker'] == span['speaker']
            quote_hits = [[m.start(), m.end()] for m in re.finditer(re.escape(span['quote']), turn['text'])]
            assert len(quote_hits) == 1, (pred['candidate_id'], item, 'Gold quote not unique within turn', quote_hits)
            start, end = quote_hits[0]
            positions[item] = [index, start]
            item_checks[item] = {
                'turn_ref': span['turn_ref'], 'actual_source_index_0based': index,
                'literal_quote_start': start, 'literal_quote_end': end,
                'quote_exact_match': True,
                'declared_source_index': span.get('source_index'),
                'declared_source_index_matches': span.get('source_index', index) == index,
                'declared_start': span.get('start'), 'declared_end': span.get('end'),
                'declared_start_matches': span.get('start', start) == start,
                'declared_end_matches': span.get('end', end) == end,
            }
        source_order = sorted(positions, key=lambda item: positions[item])
        unique_gold_positions = len(set(tuple(pos) for pos in positions.values())) == len(positions)
        check = {'candidate_id': pred['candidate_id'], 'supplied_gold': gold['gold'],
                 'source_position_gold': source_order, 'unique_gold_positions': unique_gold_positions,
                 'source_position_gold_matches_supplied': source_order == gold['gold'], 'items': item_checks}
        gold_checks.append(check)
        for method in config['methods']:
            m = pred['methods'][method]
            per_item = {i: data['methods'][method] for i, data in pred['items'].items()}
            item_scoring = {item: {
                'gold_source_index_0based': pos[0],
                'predicted_positions_0based': per_item[item]['positions_0based'],
                'status': per_item[item]['status'],
                'unique_top_turn_correct': per_item[item]['positions_0based'] == [pos[0]],
                'top_turn_contains_gold': pos[0] in per_item[item]['positions_0based'],
            } for item, pos in positions.items()}
            question_details = {'candidate_id': pred['candidate_id'], 'method': method,
                                'strata': dict(strata(pred)), 'items': item_scoring, 'vs_gold': {}}
            for target_name, target in [('admitted_gold', gold['gold']), ('source_position_gold', source_order)]:
                target_rank = {item: index for index, item in enumerate(target)}
                pair_checks = []
                for p in m['pair_relations']:
                    true_relation = 'before' if target_rank[p['left']] < target_rank[p['right']] else 'after'
                    pair_checks.append({**p, 'gold_relation': true_relation, 'correct': p['relation'] == true_relation})
                row = {
                    'candidate_id': pred['candidate_id'], 'predicted_order': m['total_order'],
                    'gold_order': target, 'whole_correct': m['total_order'] == target,
                    'pairs': len(pair_checks), 'correct_pairs': sum(p['correct'] for p in pair_checks),
                    'resolved_pairs': sum(p['relation'] != 'unknown' for p in pair_checks),
                    'items': len(per_item),
                    'unique_top_turn_correct': sum(x['unique_top_turn_correct'] for x in item_scoring.values()),
                    'top_turn_contains_gold': sum(x['top_turn_contains_gold'] for x in item_scoring.values()),
                    'tied_top_items': sum(x['status'] == 'tied_top_turns' for x in item_scoring.values()),
                    'unmatched_items': sum(x['status'] == 'unknown' for x in item_scoring.values()),
                }
                question_details['vs_gold'][target_name] = {**row, 'pair_checks': pair_checks}
                target_groups = groups if target_name == 'admitted_gold' else source_groups
                for stratum, name in strata(pred):
                    target_groups[(method, stratum, name)].append(row)
            detail.append(question_details)
    summaries = [{'method': method, 'stratum': stratum, 'group': group, **metrics(rows)}
                 for (method, stratum, group), rows in sorted(groups.items())]
    source_summaries = [{'method': method, 'stratum': stratum, 'group': group, **metrics(rows)}
                        for (method, stratum, group), rows in sorted(source_groups.items())]
    save('SOURCE_POSITION_GOLD_CHECK.json', {
        'gold_file_sha256': digest(gold_path), 'question_count': len(gold_checks),
        'all_source_orders_match_supplied_gold': all(x['source_position_gold_matches_supplied'] for x in gold_checks),
        'all_declared_span_coordinates_match': all(all(item[k] for k in ('declared_source_index_matches', 'declared_start_matches', 'declared_end_matches')) for q in gold_checks for item in q['items'].values()),
        'questions': gold_checks,
    })
    save('SCORES.json', {
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'predictions_sha256': digest(OUT / 'PREDICTIONS.json'), 'gold_file_sha256': digest(gold_path),
        'score_py_sha256': digest(Path(__file__)),
        'metrics': config['metrics_precommit'], 'summaries_vs_admitted_gold': summaries,
        'summaries_vs_source_position_gold': source_summaries, 'question_details': detail,
    })
    lines = [
        '# v140 冻结排序题：一次性 CPU 词面检索诊断', '',
        '16 道排序题（12 新、4 旧），每道 6 项。本目录独立保存全部诊断产物；原稿、planned、最终 records 及主流程均未修改。', '',
        '预测阶段仅读取 public/QUESTIONS.json 与四个完整 sources/*.json。依据题面明确给出的 scene_ref 过滤候选 turns；scene 只用于公开范围过滤，检索打分完全由 item 与 speaker+turn text 的词面特征构成。候选出现位置使用完整 source 数组的原始下标。未使用私有 anchor/scene 到答案的映射。', '',
        '固定参数：BM25 k1=1.5、b=0.75；IDF=ln(1+(N-df+0.5)/(df+0.5))；[a-zA-Z0-9]+ 分词并转小写、查询词去重；无停用词、词干、训练或调参。第二版本在 item 双引号片段逐字连续命中 turn text 时优先该候选池（仅忽略大小写），然后用同一 BM25 排分；无引语命中即回退同一 BM25。未使用 LLM、GPU、API 或新依赖。', '',
        '所有 top 分数差不超过 1e-12 的 turns 保留为 tie；零分为 unknown。不同 item 同 turn 时保留未知先后，不以标签、scene 或作者打破 tie。整题仅完整严格序列完全相同才算正确；成对正确率分母含 unknown，另报已解析对上的准确率及覆盖率。', '',
        'CONFIG.json 与 predict.py 哈希首先写入 PRECOMMIT.json，随后生成无 Gold 的 PREDICTIONS.json 与 PREDICTIONS_COMMIT.json，最后才读取 private/ADMITTED_QUESTIONS.json 评分。检索参数及代码未随 Gold 修改。GOLD_READ_AUDIT.json 记录读取顺序。', '',
        '| 方法 | 分组 | 题数 | 整题正确 | 成对正确 | 成对准确率 | 未知对 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    wanted = {'all', 'new_or_old', 'root', 'author', 'has_double_quoted_item', 'author_x_new_or_old', 'quote_x_new_or_old'}
    for s in summaries:
        if s['stratum'] not in wanted:
            continue
        label = 'BM25' if s['method'] == 'bm25' else 'BM25+quote'
        group_label = s['stratum'] + ':' + s['group']
        lines.append(f"| {label} | {group_label} | {s['questions']} | {s['whole_correct']}/{s['questions']} | {s['correct_pairs']}/{s['pairs']} | {s['pair_accuracy']:.2%} | {s['unknown_pairs']} |")
    lines.extend(['', 'A 作者组为 root129761/198141，B 作者组为 root203634/214456；仅用于报告分组。有引语指至少一个 item 含双引号片段；旧题中的数字、专名不自动算作精确引语。', '',
                  f"Gold 校验：{sum(x['source_position_gold_matches_supplied'] for x in gold_checks)}/{len(gold_checks)} 道题的 admitted Gold 与独立逐字定位 Gold 引文、按实际 source 数组位置及引文 offset 排序得到的 source-position Gold 一致。两套 Gold 的全部分组评分均保存在 SCORES.json。", '',
                  '逐题误配及每个 pair 的判定见 SCORES.json；每个 item 对每个候选 turn 的分数、每个词的 tf/df/IDF/贡献、字符位置及全部精确引语位置见 FEATURES.json；所有 top 候选、ties 与 unknowns 见 PREDICTIONS.json。', '',
                  '本结果只表明这套冻结的词面检索方法足以作答其中一部分题，不能据此推断 Solver 的实际策略；未答对的部分也不证明需要复杂推理。本诊断不提出主流程修题建议。', '',
                  '预测 SHA256：`' + digest(OUT / 'PREDICTIONS.json') + '`。',
                  '冻结检索代码 SHA256：`' + digest(OUT / 'predict.py') + '`。',
                  '冻结配置 SHA256：`' + digest(OUT / 'CONFIG.json') + '`。', ''])
    with (OUT / 'REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({'overall': [s for s in summaries if s['stratum'] == 'all'],
                      'all_source_orders_match': all(x['source_position_gold_matches_supplied'] for x in gold_checks),
                      'report': str(OUT / 'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
