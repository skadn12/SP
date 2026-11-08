#!/usr/bin/env python3
"""Check the completed selected/exported data and seal a factual final report.

No authoring, scoring, inference, token truncation, or training. Run once after
the actual independent collection review and one-shot export have finished.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

R = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def content_binding(path, raw):
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def unique(rows):
    result = {row['candidate_id']: row for row in rows}
    assert len(result) == len(rows), 'DUPLICATE_CANDIDATE'
    return result


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def check_original_item(item, decision, original, digest):
    """Selection metadata must preserve every frozen private payload field."""
    assert digest(original) == decision['original_item_sha256'] == item['selection_original_item_sha256']
    assert all(item.get(key) == value for key, value in original.items() if key != 'origin'), 'ORIGINAL_ITEM_FIELD_CHANGED'
    assert item['selection_original_origin'] == original.get('origin')
    assert item['selection_original_items_binding'] == decision['original_items_binding']
    assert item['origin'] == decision['origin']
    assert item['family_id'] == decision['family_id']
    assert item['selection_reason'] == decision['collection']['reason']
    expected_tie = hashlib.sha256((item['family_id'] + '\0' + item['candidate_id']).encode()).hexdigest()
    assert item['selection_tie_value_sha256'] == decision['tie_value_sha256'] == expected_tie
    assert item['selection_eligible_family_peers'] == decision['eligible_family_peer_candidate_ids']


def check_lineage_identity(item, lin, source, source_binding, fits):
    for key in ('candidate_id', 'root_id', 'question_type', 'origin', 'family_id', 'selection_reason'):
        assert lin[key] == item[key], ('LINEAGE_IDENTITY_DRIFT', item['candidate_id'], key)
    assert lin['split'] == item.get('split', 'train') == 'train'
    assert lin['source_binding'] == source_binding
    refs = [turn['turn_ref'] for turn in source['turns']]
    assert source['root_id'] == item['root_id'] and len(refs) == len(set(refs))
    assert lin['source_turn_count'] == item['source_turn_count'] == len(refs)
    rounds = Counter(re.match(r'R\d{2}(?=:)', ref).group(0) if re.match(r'R\d{2}(?=:)', ref) else 'parent' for ref in refs)
    assert lin['source_turn_counts_by_round'] == dict(rounds)
    numbered = sorted(key for key in rounds if key != 'parent')
    assert lin['actual_source_latest_round'] == (numbered[-1] if numbered else 'parent')
    assert lin['export_status'] == ('EXPORTED' if fits else 'EXPORT_PENDING_OVER_MAX_LENGTH')


def main():
    outputs = [R / name for name in ('FINAL_REPORT.md', 'FINAL_CHECKSUMS.json', 'DATA_SEAL.json')]
    assert not any(p.exists() for p in outputs), 'SEAL_OUTPUT_ALREADY_EXISTS'
    execution_path = R / 'FINAL_EXECUTION_AUDIT.json'
    selected_path = R / 'union_selection/FINAL_SELECTED_ITEMS.json'
    export = R / 'training_export_v1'
    entry_bytes = {path: path.read_bytes() for path in (execution_path, selected_path, export / 'SUMMARY.json')}
    execution, selected, summary = (json.loads(entry_bytes[path]) for path in (execution_path, selected_path, export / 'SUMMARY.json'))
    assert execution['status'] == 'PASS' and execution['duplicate_requests'] == 0
    assert execution['completed_author_opportunities'] == 136 and execution['required_roots'] == 68
    assert execution['rescoring'] is False and execution['training_started'] is False
    assert selected['schema_version'] == 'final_union_selection_v1'
    assert selected['status'] == 'FINAL_SELECTED_FROZEN_NO_EXPORT_OR_TRAINING'
    assert selected['training_started'] is False and summary['training_started'] is False
    assert summary['status'] in ('SEALED', 'SEALED_WITH_EXPORT_PENDING')
    assert summary['truncate'] is False and summary['full_source'] is True
    assert summary['assistant_only_loss'] is True and summary['training_display_count_per_question'] == 1
    assert summary['max_length'] == 12288
    assert summary['selection_binding']['sha256'] == binding(selected_path)['sha256']
    checked = {}

    def check(record):
        path = Path(record['path'])
        actual = binding(path)
        assert actual['sha256'] == record['sha256'], ('BINDING_CHANGED', str(path))
        if 'size_bytes' in record:
            assert actual['size_bytes'] == record['size_bytes'], ('BINDING_SIZE_CHANGED', str(path))
        previous = checked.get(str(path.resolve()))
        assert previous is None or previous == actual, ('BINDING_CHANGED_DURING_SEAL', str(path))
        checked[str(path.resolve())] = actual

    for path, raw in entry_bytes.items():
        check(content_binding(path, raw))
    check(execution['coverage_binding'])
    for row in execution['batch_closure_bindings']:
        check(row)
    for comp in execution['components']:
        for row in comp['bindings']:
            check(row)
    for row in execution['audit_input_bindings']:
        check(row)
    for row in selected['input_bindings']:
        check(row)
    check(selected['global_score_free_review_binding'])
    for row in summary['dependencies'] + summary['outputs']:
        check(row)
    items = unique(selected['items'])
    decisions = unique(selected['candidate_decisions'])
    lineage = unique(jsonl(export / 'QUESTION_LINEAGE.jsonl'))
    train = unique(jsonl(export / 'TRAIN.jsonl'))
    pending = unique(read(export / 'EXPORT_PENDING.json')['items'])
    assert set(items) == set(lineage) == set(train) | set(pending)
    assert not set(train) & set(pending)
    assert len({i['family_id'] for i in items.values()}) == len(items)
    assert {cid for cid, row in decisions.items() if row['selected']} == set(items)
    assert len(items) == summary['selected_count'] == selected['counts']['selected_candidates']
    assert len(train) == summary['exported_count'] and len(pending) == summary['export_pending_count']
    assert summary['status'] == ('SEALED_WITH_EXPORT_PENDING' if pending else 'SEALED')
    assert len(decisions) == selected['counts']['all_candidates']
    assert sum(row['eligible'] is True for row in decisions.values()) == selected['counts']['eligible_candidates']
    assert summary['question_types'] == dict(Counter(row['question_type'] for row in train.values()))
    assert summary['origins'] == dict(Counter(str(item['origin']) for item in items.values()))
    assert summary['source_versions'] == dict(Counter(row['actual_source_latest_round'] for row in lineage.values()))
    assert {str(export / name) for name in ('TRAIN.jsonl', 'QUESTION_LINEAGE.jsonl', 'EXPORT_PENDING.json')} == {row['path'] for row in summary['outputs']}

    # Reconstruct the public display and answer from the unchanged selected item.
    # This loads only the local tokenizer; no model weights, scoring, or training.
    spec = importlib.util.spec_from_file_location('sealed_export_contract', R / 'packing/export_training.py')
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    deps = pack.dependencies()
    assert {row['path']: row for row in summary['dependencies']} == {row['path']: row for row in deps[-1]}, 'EXPORT_DEPENDENCY_SET_DRIFT'
    assert summary['display_seed'] == deps[1].h.DISPLAY_SEED
    inventory = {row['root_id']: row for row in read(pack.INVENTORY)['rows'] if row['split'] == 'train'}
    original_files = {}
    seen_prompts = set()
    for cid, item in items.items():
        decision, lin = decisions[cid], lineage[cid]
        assert decision['eligible'] is True and decision['selected'] is True
        assert decision['quality']['status'] == 'PASS'
        assert decision['training_value']['all_full_calls_strictly_valid'] is True
        assert 0 < decision['training_value']['mean_full_score'] < 1
        assert decision['source_qualification']['status'] in ('PASS_EXISTING_CITED_USE', 'LEGACY_FROZEN_QUALIFIED')
        original_binding = decision['original_items_binding']
        check(original_binding)
        original_path = original_binding['path']
        if original_path not in original_files:
            original_files[original_path] = unique(read(Path(original_path)))
        check_original_item(item, decision, original_files[original_path][cid], pack.digest)
        source_path = Path(item['source_path'])
        source_binding = binding(source_path)
        assert source_binding['sha256'] == item['source_sha256']
        check(source_binding)
        source = read(source_path)
        messages, variant, gold, displayed, pairs = pack.public_and_gold(item, source, deps)
        counts, fits = pack.checked_features(messages, deps)
        assert pack.digest(item) == lin['selected_item_sha256']
        assert pack.digest(messages) == lin['messages_sha256']
        assert pack.digest(messages[:-1]) == lin['prompt_sha256']
        assert lin['prompt_sha256'] not in seen_prompts, 'DUPLICATE_SELECTED_PUBLIC_PROMPT'
        seen_prompts.add(lin['prompt_sha256'])
        assert lin['canonical_gold'] == gold and lin['displayed_gold'] == displayed
        assert lin['display_variant'] == variant and lin['public_query_pairs'] == pairs
        assert lin['display_to_option_id'] == variant['display_to_option_id']
        assert lin['source_binding']['sha256'] == item['source_sha256']
        check_lineage_identity(item, lin, source, source_binding, fits)
        assert item['root_id'] in inventory
        assert lin['matches_inventory_latest_source'] == (item['source_sha256'] == inventory[item['root_id']]['source_sha256'])
        assert all(lin[key] == value for key, value in counts.items())
        assert fits == (cid in train)
        if fits:
            row = train[cid]
            assert row['messages'] == messages and row['root_id'] == item['root_id']
            assert row['question_type'] == item['question_type']
            assert row['gold_cardinality'] == len(gold['pairs' if item['question_type'] == 'partial_order' else 'option_ids'])
            assert row['split'] == 'train' and row['selection_role'] == item['origin']
            assert row['dependence_group_id'] == item['family_id']
            assert len(messages) == 3 and messages[-1]['role'] == 'assistant'
            target = messages[-1]['content']
            assert target.startswith('<box>') and target.endswith('</box>')
            assert counts['input_token_count'] <= 12288 and counts['supervised_token_count'] > 0
        else:
            assert pending[cid]['reason'] == 'OVER_MAX_LENGTH_NO_TRUNCATION'
            assert pending[cid]['max_length'] == 12288
            assert pending[cid]['selected_item_sha256'] == lin['selected_item_sha256']
            assert pending[cid]['source_binding'] == source_binding
            assert all(pending[cid][key] == value for key, value in counts.items())
            assert counts['input_token_count'] > 12288

    concentration = selected['global_concentration_review']
    assert concentration['fine_families_independent'] is False
    quality_counts = Counter(row['quality']['status'] for row in decisions.values())
    value_counts = Counter(row['training_value']['status'] for row in decisions.values())
    source_counts = Counter(row['source_qualification']['status'] for row in decisions.values())
    isolation_incidents = sorted((R / 'isolation_incidents').glob('*.json'))
    now = datetime.now(timezone.utc).isoformat()
    # Tokenizer reconstruction can take time: seal the exact bytes checked at
    # entry, and fail before creating a report if anything changed meanwhile.
    for record in list(checked.values()):
        check(record)
    report = [
        '# 训练集重建封存报告', '',
        f'封存时间：{now}。本轮只完成数据准备与导出，未启动训练。', '',
        f'覆盖全部 68 个 Train roots、136 个 F/G 作者机会；新首稿 {execution["actual_new_first_drafts"]} 道，'
        '全部首稿和首次审核保留。8 roots 仅为执行分批，没有按批次产出停止后续覆盖。', '',
        f'唯一评估调用 {execution["unique_requests"]} 次：{json.dumps(execution["requests_by_group"], ensure_ascii=False)}；'
        '全局重复请求为 0。Solver 为固定 Qwen3-4B + M-clean 合并权重，未重复叠加 adapter；'
        'Full/No-context 各 min(6,n!) 个唯一排列 ×3，使用冻结严格协议评分，未按结果重跑或改题。', '',
        f'联合候选 {len(decisions)} 道；质量状态：{json.dumps(quality_counts, ensure_ascii=False)}。'
        f'来源资格：{json.dumps(source_counts, ensure_ascii=False)}。', '',
        f'训练价值分层：{json.dumps(value_counts, ensure_ascii=False)}。'
        '质量、来源资格、训练价值与集合选择分开保存；满分、低分、重复和协议待处理不改写内容质量。', '',
        f'最终入选 {len(items)} 道、{len(items)} 个语义细族；导出 {len(train)} 道，'
        f'超长待导出 {len(pending)} 道。导出覆盖 {len({x["root_id"] for x in train.values()})} 个 roots；'
        '执行覆盖全部 roots 不意味着每个 root 都必须贡献训练题。', '',
        f'导出题型：{json.dumps(Counter(x["question_type"] for x in train.values()), ensure_ascii=False)}；'
        f'来源角色：{json.dumps(Counter(x["origin"] for x in items.values()), ensure_ascii=False)}。', '',
        f'入选题的全局结构类别：{json.dumps(selected["selected_statistics"]["structural_clusters_global_reporting"], ensure_ascii=False)}。'
        f'来源最新层：{json.dumps(selected["selected_statistics"]["source_latest_rounds"], ensure_ascii=False)}；'
        f'实际引用层组合：{json.dumps(selected["selected_statistics"]["cited_layer_combinations"], ensure_ascii=False)}。', '',
        f'入选题的每 root 贡献数量分布（题数→root 数）：'
        f'{json.dumps(Counter(row["selected_count"] for row in selected["selected_statistics"]["per_root"]), ensure_ascii=False)}；'
        f'每个精确来源哈希的入选题数分布（题数→来源数）：'
        f'{json.dumps(Counter(selected["selected_statistics"]["exact_source_hash_clusters"].values()), ensure_ascii=False)}。'
        '逐 root 的继承来源关系、共享引用及全部结构原标签保存在最终选择统计中。', '',
        f'入选题的选项数分布：{json.dumps(selected["selected_statistics"]["option_counts"], ensure_ascii=False)}；'
        f'分题型 Gold 基数分布：{json.dumps(selected["selected_statistics"]["gold_cardinalities_by_question_type"], ensure_ascii=False)}。'
        '部分排序的 Gold 基数是完整传递闭包的有向关系对数。', '',
        '同族仅选一个合格代表。先由独立无分数审计核语义覆盖/等价边界，等价者使用预先冻结的 family_id+NUL+candidate_id SHA-256 最小值；'
        '不按分数最小、F/G 比例、语言或 Gold 数量取舍。完整 peers、tie 值和未选原因见 FINAL_SELECTED_ITEMS.json。', '',
        '每题仅一个 Gold-blind 展示，保留该题的完整冻结来源与原始/合成 provenance、warnings；'
        'target 只有冻结 Gold 的 <box> 答案，assistant-only 监督，12288 tokens，不截断。'
        '作者证书、Reviewer 答案和 Solver 理由均不进入训练 target。', '',
        '## 解释边界', '',
        '来源包含历史训练暴露和同模型合成扩展，审核是同模型独立上下文审核，并非外部事实认证。'
        'root、来源、语义细族和重复采样彼此相关；细族数量不代表独立推理结构。', '',
        'F/G 还与作者、语言、措辞和任务结构混杂，统计只能描述本轮候选，不能解释为反馈的因果效果。'
        '部分题在 No-context 也容易，No-context 仅作诊断。严格格式有效或最终答案正确不保证逐项理由/引用忠实；'
        '这些观察不改写冻结评分，也不据此构造训练 CoT。', '',
        f'隔离事件事实记录 {len(isolation_incidents)} 份，已纳入校验和。'
        '第8批盲审协调者经状态工具接触作者元数据后，在尚未解题或写首答时停止参与答题；'
        '实际首次盲答交由不继承历史的独立子上下文仅按公开材料完成，原事件记录保留。', '',
        *['- ' + text for text in concentration['limitations']], '',
        '## 核验入口', '',
        '- `FINAL_EXECUTION_AUDIT.json`：全覆盖、唯一请求、预算和固定推理配置。',
        '- `union_selection/CROSS_BATCH_FAMILY_REVIEW.json`：独立无分数全局审查。',
        '- `union_selection/FINAL_SELECTED_ITEMS.json`：逐候选四层决策、分族、tie 与完整统计。',
        '- `training_export_v1/SUMMARY.json`：导出、来源版本和依赖绑定。',
        '- `training_export_v1/QUESTION_LINEAGE.jsonl`：展示映射、Gold、来源及 token 数。',
        '- `FINAL_CHECKSUMS.json`、`DATA_SEAL.json`：封存检查及文件哈希。', '',
        '数据封存后停止；实际训练留待单独安排。', ''
    ]
    for path in [Path(__file__), execution_path, selected_path, export / 'SUMMARY.json',
                 R / 'model_baseline/MODEL_IDENTITY_AUDIT.json']:
        check(binding(path))
    for path in (R / 'feedback_cards').glob('*.json'):
        check(binding(path))
    for path in isolation_incidents:
        check(binding(path))
    for record in list(checked.values()):
        check(record)
    report_raw = '\n'.join(report).encode('utf-8')
    report_binding = content_binding(outputs[0], report_raw)
    checked[str(outputs[0])] = report_binding
    checksums = {'created_at_utc': now, 'files': [checked[key] for key in sorted(checked)],
                 'scope': 'Frozen inputs and outputs bound by final execution, selection, export and feedback records; no self-hash.'}
    checksums_raw = (json.dumps(checksums, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    seal = {'created_at_utc': now, 'status': summary['status'], 'training_started': False,
            'model_calls': 0, 'rescoring': False, 'full_source_and_answer_only_verified': True,
            'assistant_only_mask_and_no_truncation_verified': True, 'selected_count': len(items),
            'exported_count': len(train), 'export_pending_count': len(pending),
            'final_execution': binding(execution_path), 'selection': binding(selected_path),
            'export_summary': binding(export / 'SUMMARY.json'), 'report': report_binding,
            'checksums': content_binding(outputs[1], checksums_raw)}
    seal_raw = (json.dumps(seal, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    # Resolve/serialize every prerequisite before the first exclusive creation.
    for path, raw in zip(outputs, (report_raw, checksums_raw, seal_raw)):
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps(seal, ensure_ascii=False))


if __name__ == '__main__':
    main()
