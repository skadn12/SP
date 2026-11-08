#!/usr/bin/env python3
"""Read-only historical Writer replay; no normalizer writes or model calls.

Run from the isolated v4 checkout with PYTHONPATH set to that checkout.
The only writes are artifacts beside this script. Review labels are not inputs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from scripts.current_method.restart_n5_spell.h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.prepare import _capacity
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.evidence_repair_feedback import build_evidence_repair_feedback
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.repair_feedback import build_structural_repair_feedback

SOURCE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_writer_selection_repair_v3_prepare_1')
EXECUTION = SOURCE / 'executions/writer_repair_comparison'
OUT = Path(__file__).resolve().parent
ORDERING = {'version': 'ordering_grounded_options_v2', 'node_count': 4}
READ_FILES: dict[str, str] = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_text(path):
    path = Path(path)
    data = path.read_bytes()
    READ_FILES[str(path)] = digest(data)
    return data.decode('utf-8')


def read(path):
    return json.loads(read_text(path))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def summary(localizations):
    status = Counter(x['status'] for x in localizations)
    bad = [x for x in localizations if not x.get('submitted_ref_contains_quote', False)]
    failure_status = Counter(x['status'] for x in bad)
    # The locator makes no semantic decision; wrong location means only that
    # the submitted exact quote is not contained in the submitted turn.
    return {'span_occurrences': len(localizations),
            'status_counts': {key: status[key] for key in ('UNIQUE','MULTIPLE','NOT_FOUND','INVALID')},
            'submitted_quote_location_failures': len(bad),
            'failure_status_counts': {key: failure_status[key] for key in ('UNIQUE','MULTIPLE','NOT_FOUND','INVALID')},
            'unique_alternative_location_occurrences': sum(x['status'] == 'UNIQUE' for x in bad)}


def main():
    profile = read(SOURCE / 'profile.json')
    source_frozen = read(SOURCE / 'audit/frozen_manifest.json')
    closed = read(EXECUTION / 'reports/closed_execution_manifest.json')
    assert closed, 'A closed source execution is required.'
    schedule = read(SOURCE / 'schedule.json')
    slots = {(x['root_id'], x['arm_id'], x['slot_id']): x for x in schedule}
    outcomes = {(x['root_id'], x['arm_id'], x['slot_id']): x
                for x in read(EXECUTION / 'reports/slot_outcomes.json')}
    expanded_by_root = {root: read(SOURCE / 'inputs/TRANSCRIPT_ONLY' / root / 'expanded.json')
                        for root in sorted({x['root_id'] for x in schedule})}
    contracts_by_root = {root: read(SOURCE / 'inputs/TRANSCRIPT_ONLY' / root / 'contracts.json')
                         for root in expanded_by_root}
    entries = []
    for folder in sorted((EXECUTION / 'arms').glob('*/roots/*')):
        validations = {x['request_id']: x['validation']
                       for x in read(folder / 'candidate_generation.json')}
        for path in sorted(folder.glob('stage_calls/*/*.json')):
            stage = read(path)
            stage['validation'] = validations[stage['request_id']]
            stage['source_stage_path'] = str(path)
            raw_text = read_text(folder / 'raw/responses' / (stage['request_id'] + '.json.txt'))
            assert digest(raw_text.encode()) == stage['raw_text_sha256']
            assert json.loads(raw_text) == stage['raw']
            stage['_raw_text'] = raw_text
            entries.append(stage)
    entries.sort(key=lambda x: (x['order'], x['stage'] != 'primary'))
    assert len(entries) == 18
    assert sum(x['stage'] == 'primary' for x in entries) == 12
    assert sum(len(x['raw']['candidates']) for x in entries) == 58
    primary_entries = {(x['root_id'], x['arm_id'], x['slot_id']): x for x in entries if x['stage'] == 'primary'}
    original_prompt_parity = []
    for entry in entries:
        key = (entry['root_id'], entry['arm_id'], entry['slot_id'])
        expanded, contract = expanded_by_root[entry['root_id']], contracts_by_root[entry['root_id']][slots[key]['raw_index']]
        policy = 'selection_repair_v3' if entry['arm_id'] == 'GROUNDED_V2_REFINED' else 'legacy'
        if entry['stage'] == 'primary':
            rebuilt = build_prompt(expanded, contract, ordering_policy=ORDERING, writer_prompt_policy=policy)
        else:
            primary = primary_entries[key]
            normalized = primary['validation']
            kwargs = {}
            if policy != 'legacy':
                kwargs = {'repair_feedback': build_structural_repair_feedback(primary['raw'], normalized,
                    normalized['minimum_acceptance'], expanded=expanded, parse_error=primary['parse_error']),
                    'primary_raw_text': primary['_raw_text']}
            rebuilt = build_repair_prompt(expanded, contract, primary['raw'], 'contract_guided',
                ordering_policy=ORDERING, writer_prompt_policy=policy,
                primary_abstentions=normalized.get('effective_abstentions') or [], **kwargs)
        old_path = EXECUTION / 'arms' / entry['arm_id'] / 'roots' / entry['root_id'] / 'raw/prompts' / (entry['request_id'] + '.txt')
        old_prompt = read_text(old_path)
        same = old_prompt == rebuilt['prompt']
        old_request_path = old_path.parent.parent / 'requests' / (entry['request_id'] + '.json')
        old_payload = read(old_request_path)['payload']
        rebuilt_payload = build_payload(profile, rebuilt['prompt'])
        original_prompt_parity.append({'request_id': entry['request_id'], 'arm_id': entry['arm_id'],
            'root_id': entry['root_id'], 'slot_id': entry['slot_id'], 'stage': entry['stage'],
            'writer_prompt_policy': policy, 'source_path': str(old_path),
            'original_sha256': digest(old_prompt.encode()), 'rebuilt_sha256': rebuilt['prompt_sha256'],
            'byte_equal': same, 'original_request_path': str(old_request_path),
            'messages_identical': old_payload['messages'] == rebuilt_payload['messages'],
            'payload_identical': old_payload == rebuilt_payload})
    write('original_prompt_parity.json', original_prompt_parity)
    assert all(x['byte_equal'] for x in original_prompt_parity), 'Old prompt behavior changed.'
    assert all(x['payload_identical'] for x in original_prompt_parity), 'Old request payload changed.'
    primary_feedback, diagnostics, prompt_cases = [], [], []
    for entry in entries:
        key = (entry['root_id'], entry['arm_id'], entry['slot_id'])
        expanded = expanded_by_root[entry['root_id']]
        normalized = entry['validation']
        acceptance = normalized['minimum_acceptance']
        originals = deepcopy((entry['raw'], normalized, acceptance, expanded))
        feedback = build_evidence_repair_feedback(entry['raw'], normalized, acceptance,
            expanded=expanded, parse_error=entry['parse_error'])
        assert originals == (entry['raw'], normalized, acceptance, expanded)
        catalog = {x['turn_ref']: x['text'] for x in expanded['turns']}
        localizations = deepcopy(feedback['evidence_localization'])
        for item in localizations:
            quote, ref = item.get('quote'), item.get('submitted_turn_ref')
            item['submitted_ref_contains_quote'] = (isinstance(quote, str) and bool(quote)
                                                     and quote in catalog.get(ref, ''))
        meta = {k: entry[k] for k in ('root_id', 'arm_id', 'slot_id', 'order', 'stage', 'request_id')}
        diagnostics.append({**meta, 'scope': 'OFFLINE_DIAGNOSTIC_NOT_A_NEW_REPAIR',
                            'summary': summary(localizations), 'localizations': localizations})
        if entry['stage'] != 'primary':
            continue
        original_trigger = outcomes[key]['repair_requested']
        assert original_trigger == (not acceptance['accepted'])
        assert original_trigger == any(x['stage'] == 'repair' and
            (x['root_id'], x['arm_id'], x['slot_id']) == key for x in entries)
        is_refined = entry['arm_id'] == 'GROUNDED_V2_REFINED'
        primary_feedback.append({**meta, 'source_stage_path': entry['source_stage_path'],
            'original_repair_trigger': original_trigger,
            'same_v3_primary_direct_applicability': is_refined and original_trigger,
            'comparison_qualification': ('V3_PRIMARY_REPLAY' if is_refined else
                                         'LEGACY_PRIMARY_DIAGNOSTIC_NOT_V4_MATCHED_PRIMARY'),
            'feedback': feedback})
        if original_trigger:
            slot = slots[key]
            prompt = build_repair_prompt(expanded, contracts_by_root[entry['root_id']][slot['raw_index']],
                entry['raw'], 'contract_guided', ordering_policy=ORDERING,
                primary_abstentions=normalized.get('effective_abstentions') or [],
                writer_prompt_policy='evidence_repair_v4', repair_feedback=feedback,
                primary_raw_text=entry['_raw_text'])
            payload = build_payload(profile, prompt['prompt'])
            capacity = _capacity(profile, payload['messages'], profile['output_limit'], entry['request_id'])
            artifact = OUT / 'historical_repair_prompts' / entry['arm_id'] / entry['root_id'] / entry['slot_id']
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / 'prompt.txt').write_text(prompt['prompt'])
            (artifact / 'messages.json').write_text(json.dumps(payload['messages'], ensure_ascii=False, indent=2) + '\n')
            (artifact / 'payload.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
            prompt_cases.append({**meta, 'source_response_sha256': entry['raw_text_sha256'],
                'same_v3_primary_direct_applicability': is_refined,
                'qualification': 'HISTORICAL_OUTPUT_DEPENDENT_INPUT_EXAMPLE_NOT_FUTURE_REPAIR_BUDGET',
                'capacity': capacity, 'prompt_path': str(artifact / 'prompt.txt'),
                'prompt_sha256': prompt['prompt_sha256'],
                'remaining_counts': feedback['repair_plan']['remaining_counts']})
    assert len(primary_feedback) == 12 and len(prompt_cases) == 6
    assert sum(x['same_v3_primary_direct_applicability'] for x in primary_feedback) == 4
    groups = []
    for arm in sorted({x['arm_id'] for x in primary_feedback}):
        rows = [x for x in primary_feedback if x['arm_id'] == arm]
        triggered = [x for x in rows if x['original_repair_trigger']]
        eligible_localizations = [item for row in triggered for item in row['feedback']['evidence_localization']
                                  if not item.get('immutable_retained')]
        groups.append({'historical_arm_id': arm, 'primary_count': len(rows),
            'existing_repair_trigger_count': len(triggered),
            'same_v3_primary': arm == 'GROUNDED_V2_REFINED',
            'triggered_rejected_span_status_counts': dict(Counter(x['status'] for x in eligible_localizations)),
            'source_context_turns_across_triggered_prompts': sum(len(x['feedback']['source_context']['turns']) for x in triggered),
            'remaining_counts_across_triggered_prompts': dict(sum((Counter(x['feedback']['repair_plan']['remaining_counts']) for x in triggered), Counter()))})
    math = next(x for x in primary_feedback if x['order'] == 10)
    math_turns = expanded_by_root['cc-train-episode-122711']['turns']
    index = next(i for i, turn in enumerate(math_turns) if turn['turn_ref'] == 'cc-train-episode-122711:P04:T010')
    math_diagnostic = {'order': 10, 'root_id': 'cc-train-episode-122711', 'slot_id': 'intent_2',
        'qualification': 'CONTEXT_AVAILABILITY_EXAMPLE_ONLY_NOT_RUNTIME_REPAIR_INPUT',
        'neighboring_turns': math_turns[index-1:index+2],
        'remaining_counts': math['feedback']['repair_plan']['remaining_counts'],
        'immutable_retained_rows': math['feedback']['repair_plan']['immutable_retained_rows'],
        'runtime_source_context_turn_count': len(math['feedback']['source_context']['turns']),
        'ordering_repair_authorized_by_original_trigger': False,
        'interpretation': 'P04:T009 is available before the submitted P04:T010 quote containing them. The ordering candidate was retained; v4 preserves it and repairs only the missing SC. This replay does not fix the submitted evidence or establish semantic support.'}
    assert math_diagnostic['remaining_counts'] == {'single_choice': 1}
    assert math_diagnostic['runtime_source_context_turn_count'] == 0
    for path, sha in READ_FILES.items():
        assert digest(Path(path).read_bytes()) == sha, 'Historical input changed: ' + path
    aggregate = {'status': 'PASS', 'scope': 'OFFLINE_LOCATION_CONTEXT_AND_REPAIR_PLAN_REPLAY',
        'source_run': str(SOURCE), 'external_calls': 0, 'new_model_outputs': 0,
        'primary_responses': 12, 'all_responses': 18, 'raw_candidate_occurrences': 58,
        'existing_repair_triggers': 6, 'same_v3_primary_existing_repair_triggers': 4,
        'legacy_primary_existing_repair_triggers': 2, 'new_repair_triggers_added': 0,
        'all_responses_quote_diagnostic': summary([x for row in diagnostics for x in row['localizations']]),
        'groups': groups,
        'historical_repair_prompt_capacity': prompt_cases,
        'historical_inputs_checked_unchanged': len(READ_FILES),
        'original_primary_and_repair_prompt_parity': {'checked': len(original_prompt_parity), 'all_byte_equal': True},
        'semantic_quality_or_effective_yield_change': 'NOT_MEASURED_NO_NEW_MODEL_OUTPUT',
        'guards': {'no_normalizer_called': True, 'no_review_labels_used_as_feedback': True,
                   'raw_candidates_and_inputs_not_mutated': True, 'retained_rows_not_replaced': True,
                   'repair_trigger_not_changed': True, 'source_file_hashes_unchanged': True}}
    write('primary_feedback_replay.json', primary_feedback)
    write('all_response_quote_localization.json', diagnostics)
    write('math_context_boundary.json', math_diagnostic)
    write('replay_summary.json', aggregate)
    write('replay_source_manifest.json', {'files': [{'path': k, 'sha256': v} for k,v in sorted(READ_FILES.items())],
                                       'source_frozen_manifest_version': source_frozen.get('version')})
    code_paths = {Path(fn.__code__.co_filename).resolve() for fn in
                  (build_evidence_repair_feedback, build_structural_repair_feedback, build_repair_prompt, _capacity, build_payload)}
    write('replay_code_inputs.json', {'files': [{'path': str(p), 'sha256': digest(p.read_bytes())}
                                              for p in sorted(code_paths)]})
    failures = [(row, loc) for row in diagnostics for loc in row['localizations'] if not loc['submitted_ref_contains_quote']]
    unique_wrong = [(row, loc) for row, loc in failures if loc['status'] == 'UNIQUE']
    archery = [(row, loc) for row, loc in failures if row['order'] == 12]
    cases = {'unique_alternative_location_examples': [{**{k:r[k] for k in ('order','stage','root_id','arm_id','slot_id')}, **l} for r,l in unique_wrong],
             'archery_quote_failures': [{**{k:r[k] for k in ('order','stage','root_id','arm_id','slot_id')}, **l} for r,l in archery]}
    write('replay_cases.json', cases)
    capacities = [x['capacity']['input_tokens'] for x in prompt_cases]
    lines = ['# v4 证据定位修复：历史输出离线回放', '',
        '本报告只确认定位、上下文回取和缺题计划是否可用。未发送模型请求，未改候选、引用、答案、旧冻结输入或封存报告；没有新输出，因此不能推断有效题增加。', '',
        '读取原 12 次 primary 和 6 次 repair（58 条原始候选），使用当时已保存的程序 validation 与 acceptance。Codex 内容诊断、人工修正案例和 Gold 均没有进入反馈。', '',
        '|历史 arm|primary|原有 repair 触发|本轮 primary 可比性|', '|---|---:|---:|---|']
    for group in groups:
        lines.append(f"|{group['historical_arm_id']}|{group['primary_count']}|{group['existing_repair_trigger_count']}|{'同 v3 primary，可直接检查修复反馈' if group['same_v3_primary'] else '仅定位诊断，不属于 v4 配对基线'}|")
    lines += ['', '正式 v4 对比固定同一 v3 primary，只改变 repair。不能把旧版 2 个槽位也计成这项修复在匹配输入上覆盖了 6 个槽位。原有触发条件、保留题和合法弃答不变。', '',
        f"18 份响应包含 {aggregate['all_responses_quote_diagnostic']['span_occurrences']} 处 span；定位状态：`{json.dumps(aggregate['all_responses_quote_diagnostic']['status_counts'])}`。提交 ref 与 quote 不符 {len(failures)} 处，其中 {len(unique_wrong)} 处可唯一定位到原文。此处按提交次数计数，不是去重错误数，也不是语义正确数。", '',
        '定位结果：UNIQUE 仅建议核对唯一原文位置；MULTIPLE 不替模型任选；NOT_FOUND 不改写原文补造引文。定位真实不代表事件、人物或时间方向成立。', '',
        '|具体案例|离线反馈可提供什么|不能据此声称什么|', '|---|---|---|',
        '|198480 intent_1：P01:S01 / P01:S03 错误编号|已提交 quote 唯一位于 R01:S01:T003 / R01:S03:T007，可返回位置及同段前后文|不能证明提交、修订等事件状态和选项已正确|',
        '|122711 intent_3：射箭句错挂 R02:S01:T003|原 quote 唯一位于 P04:T005，可让模型核对；现有 repair 本来就因 quote 错误触发|不是将这句话自动改成原题所需事实|',
        '|198480 intent_2：行李回顾句|完整 quote 在冻结原文找不到，明确 NOT_FOUND；返回提交 ref 对应原文供模型核对|不拼凑、改写一句相似文本来使验收通过|', '',
        '数学题的 P04:T010 使用 “them”，其前句 P04:T009 可在完整原文找到。但该排序已被程序保留，原 repair 只缺 1 道单选。v4 将它列为 immutable，实际 source_context 不回取该题；不能将此离线展示称为修好了数学题。风暴题已保留时同样不因离线语义意见新增 repair。', '',
        f'按历史响应构成的 6 份完整 repair messages 输入为 {min(capacities):,}–{max(capacities):,} tokens（本地冻结 tokenizer），均预留 8,192 输出和服务余量。它们只是历史响应实例，其中仅 4 份来自 v3 primary；未来 repair 输入取决于新 primary，不能当作固定总预算。', '',
        '旧两臂 12 份 primary 与 6 份 repair 在新代码用原 policy 重建，18 份 prompt 逐字一致、完整请求 payload 一致；明细见 original_prompt_parity.json。', '',
        f"读取的 {len(READ_FILES)} 份历史文件逐一复核 SHA 未变。新外部调用 0；新 repair 触发 0；新质量结果未测。", '',
        '产物：primary_feedback_replay.json（12 份反馈）；all_response_quote_localization.json（18 份只读定位）；replay_cases.json（错引文案例）；math_context_boundary.json（保留题边界）；replay_summary.json（分组与 token 实例）；historical_repair_prompts/（未发送消息）。']
    (OUT / 'REPLAY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: aggregate[k] for k in ('status','external_calls','primary_responses','all_responses',
                     'same_v3_primary_existing_repair_triggers','all_responses_quote_diagnostic','groups')}, ensure_ascii=False))


if __name__ == '__main__':
    with forbid_network():
        main()
