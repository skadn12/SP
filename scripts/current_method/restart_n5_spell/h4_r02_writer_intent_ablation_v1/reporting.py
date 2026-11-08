"""No-service preparation and layered offline diagnostics for the Writer ablation.

This module never generates, semantically grades, or clusters questions.  Human-
authored Codex offline judgments are validated and aggregated separately from
unchanged production admission.  Empty preparations contain no synthetic QA.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ARMS = ('WITH_CURRENT_INTENT', 'WITHOUT_INTENT')
ROOTS = ('cc-train-episode-198480', 'cc-train-episode-122711')
REVIEWER = 'Codex offline diagnostic'
DIMENSIONS = (
    'QUESTION_SCOPE_ANSWERABILITY', 'GOLD_ITEM_SUPPORT',
    'SINGLE_CHOICE_UNIQUENESS', 'MULTI_SELECT_COMPLETENESS',
    'ANSWER_UNAMBIGUITY', 'PERSON_TIME_STATE_ACCURACY',
    'UNSUPPORTED_INFERENCE', 'ORDERING_RELATION_SUPPORT',
)
TYPE_SPECIFIC = {
    'SINGLE_CHOICE_UNIQUENESS': 'single_choice',
    'MULTI_SELECT_COMPLETENESS': 'multi_select',
    'ORDERING_RELATION_SUPPORT': 'ordering',
}
JUDGMENTS = ('PASS', 'FAIL', 'UNDETERMINED', 'NOT_EVALUATED_STRUCTURE')
VERSION = 'h4_r02_writer_intent_ablation_v1_review_v1'
POLICY = {
    'version': VERSION,
    'reviewer': REVIEWER,
    'independent_blind_review': False,
    'human_ground_truth_certification': False,
    'review_api_calls': 0,
    'production_admission_changed': False,
    'generation_source_both_arms': 'contract_guided',
    'all_original_v3_intents_retained': True,
    'quality_sequence': [
        'Expose only candidate-owned question/answer/evidence fields and the complete frozen transcript; hide arm, original intent, Miner diagnosis and contract alignment metadata.',
        'Review every parsed primary and repair question, including extra and program-rejected rows. Keep unreadable requests/rows as NOT_EVALUATED_STRUCTURE; no missing denominator.',
        'After all planned primary requests have stage records and no unknown service state remains, validate all judgments/quotes and freeze quality-review bytes and view/input identities.',
        'Only after this freeze expose original A contracts/Miner findings for coverage, duplicate and error-propagation analysis. B intent alignment and contract_group_alignment are NOT_APPLICABLE.',
    ],
    'dimensions': list(DIMENSIONS),
    'type_specific_dimensions': TYPE_SPECIFIC,
    'dimension_rubric': {
        'QUESTION_SCOPE_ANSWERABILITY': 'The object/person/event and question scope are clear and answerable using the full dialogue.',
        'GOLD_ITEM_SUPPORT': 'Inspect every selected answer/item against exact dialogue refs and short quotations; a deterministic label-rebinding check does not establish semantic Gold correctness.',
        'SINGLE_CHOICE_UNIQUENESS': 'For single_choice, exactly the marked option is supported as the answer; inspect all other options, including plausible alternative readings.',
        'MULTI_SELECT_COMPLETENESS': 'For multi_select, inspect every option and identify supported but unmarked options as well as unsupported marked options.',
        'ANSWER_UNAMBIGUITY': 'Check ambiguity among options or ordering relations under the stated scope, rather than accepting the supplied Gold by default.',
        'PERSON_TIME_STATE_ACCURACY': 'Do not exchange people/events, remembered/current/future times, recommendation/intention/confirmation/action/outcome.',
        'UNSUPPORTED_INFERENCE': 'Check implied relations and causal claims against transcript support; lack of support is not proof of the contrary.',
        'ORDERING_RELATION_SUPPORT': 'Ordering only: inspect all items and the claimed order; keep diagnosis separate from the principal SC/MS quality conclusion.',
    },
    'aggregation': 'Any applicable FAIL -> FAIL; else any UNDETERMINED -> UNDETERMINED; all applicable PASS -> PASS. Inapplicable type-specific dimensions must be NOT_APPLICABLE with reason. Non-reviewable rows remain NOT_EVALUATED_STRUCTURE.',
    'program_vs_content': 'Normalizer/minima/final validator acceptance and offline semantic quality are separate. No quality or dedup judgment feeds production, repair or resampling.',
    'layers': {
        'primary': 'All primary parsed candidates, including later replaced or rejected rows; retain parse failures and call-level errors.',
        'repair': 'All repair parsed candidates and failures. A repair replaces the entire primary even if worse; no primary fallback or merge.',
        'final': 'Only the final minima-accepted batch per scheduled slot and original root-arm final-validator pass rows. Replaced primary never contributes to final yield.',
        'ordering_indirect_effect': 'Report missing ordering in primary minima, ordering-only repair triggers, repair failures and any whole-batch rejection; display primary SC/MS quality alongside final SC/MS.',
    },
    'denominators': {
        'statistical_unit': 'root; two roots equal weight',
        'planned_slots_per_root_arm': 3, 'planned_slots_per_arm': 6,
        'primary_calls': 12, 'maximum_repair_calls': 12, 'maximum_total_calls': 24,
        'target_candidates_per_arm': 24, 'target_sc_ms_per_arm': 18, 'target_ordering_per_arm': 6,
        'target_is_not_output_cap': True,
        'effective_yield': 'count(final validator accepted AND quality PASS SC/MS) / 3 scheduled slots per root-arm, including failed slots',
        'quality_pass_fraction': 'PASS / all final structure-accepted SC/MS; separately list FAIL, UNDETERMINED, NOT_EVALUATED_STRUCTURE, PENDING_REVIEW',
        'paired_difference': 'WITH_CURRENT_INTENT minus WITHOUT_INTENT at each root; average the two root differences with equal weights. No candidate-ID pairing, accuracy renaming, significance or causal-stability claims.',
    },
    'dedup_rubric': {
        'judge': 'Codex offline semantic judgment after quality freeze; exact text/fingerprint are hints only and never automatic clusters.',
        'same_question': 'Same answer relation, object/person/event and evidence scope despite wording or option-label permutation is a duplicate, including SC versus MS representations of the same complete answer relation.',
        'not_sufficient': 'Shared topic, entity, overlapping refs, or repeated generic stem alone is insufficient. Distinct temporal phase, relation, object, or materially different answer content can be distinct; explain the boundary.',
        'main_population': 'Final quality-PASS SC/MS within each root-arm, covering same-batch and cross-slot duplicates; clusters must partition the population exactly once, including explicit singleton judgments.',
        'dedup_yield': 'Manually supported cluster count / 3 scheduled slots per root-arm; missing/incomplete clustering is NOT_REVIEWED, never implicit singleton counting.',
        'other_layers': 'Primary and repair duplicates and costs are separate; their counts never add to final coverage. Ordering clusters are separate.',
        'cross_arm': 'Overlap is descriptive, never remove A questions from B as historical duplicates.',
    },
    'history_policy': {
        'pool': 'Only each frozen inputs/shared/<root>/history_only.json; retain path, SHA256, record count and available fields.',
        'limitations': 'Not all historical QA. Where options/Gold are absent, report justified stem/fingerprint overlap hints or UNDETERMINED; do not infer Gold error or claim exhaustive historical semantic deduplication.',
        'no_prompt_injection_or_pool_expansion': True,
    },
    'post_quality': {
        'A_alignment_values': ['FOLLOWS_TARGET', 'CORRECTS_TARGET_OR_EVIDENCE', 'OTHER_RELATION', 'UNDETERMINED'],
        'B_alignment': 'NOT_APPLICABLE',
        'error_propagation': 'For each known source error, compare the actual QA relation/Gold and exact refs. Record SAME_ERROR_PRESENT, SAME_ERROR_ABSENT or UNDETERMINED with reasons; lexical similarity or ref overlap does not establish propagation or model-internal causality.',
        'difficulty_description': 'Describe DIRECT_FACT_LOOKUP, MULTIFACT_RELATION or MIXED_OR_UNDETERMINED from the question. No Solver means no verified equal difficulty or training benefit.',
    },
    'cost_policy': {
        'include': 'All actual logical requests in the arm: failed attempts, replaced primary, repair and extra accepted rows. Separate primary/repair input/output/total tokens, logical requests, HTTP attempts, 429 retries and missing/unknown usage/completion/payment.',
        'effective_cost': 'All available actual arm generation tokens / final quality-PASS SC/MS count, and / manually deduplicated count. Never use only successful/final-stage costs.',
        'zero_yield': 'UNDEFINED_ZERO_YIELD',
        'missing_usage': 'INCOMPLETE_USAGE: known token subtotal is not total cost and effective cost is null.',
        'not_run': 'NOT_RUN: no observed quality, yield or zero-cost efficiency claim.',
        'money': 'UNKNOWN_NO_BILLING_OR_PRICING_EVIDENCE unless reliable frozen pricing/bill evidence is separately supplied.',
    },
    'interpretation': {
        'A_better': 'Supports usefulness of the current complete intent package in these two cases only.',
        'B_better': 'Supports investigating misleading/restrictive guidance; B may select easier targets, so it does not prove all intent is useless.',
        'similar': 'Insufficient evidence of difference.',
        'both_poor': 'Investigate common Writer/Gold issues first.',
        'excluded_claims': ['all intents unnecessary', 'feedback ineffective', 'universal model capability', 'training benefit', 'significance', 'stable causal effect'],
    },
}


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def root_path(run, arm, root):
    return Path(run) / 'arms' / arm / 'qa/shards/shard-00/roots' / root


def _schemas():
    quote = {'type': 'object', 'required': ['turn_ref', 'quote'], 'properties': {'turn_ref': {'type': 'string'}, 'quote': {'type': 'string', 'minLength': 1}}, 'additionalProperties': False}
    assessment = {'type': 'object', 'required': ['judgment', 'reason', 'quotes'], 'properties': {'judgment': {'enum': ['PASS', 'FAIL', 'UNDETERMINED', 'NOT_APPLICABLE']}, 'reason': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote}}, 'additionalProperties': False}
    quality = {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'Empty format only; one future authored row per candidate view; no QA is fabricated',
        'type': 'object', 'required': ['view_id', 'reviewer', 'judgment', 'reason', 'dimensions', 'option_checks', 'quotes'],
        'properties': {
            'view_id': {'type': 'string'}, 'reviewer': {'const': REVIEWER}, 'judgment': {'enum': list(JUDGMENTS)},
            'reason': {'type': 'string', 'minLength': 1},
            'dimensions': {'type': 'object', 'properties': {d: assessment for d in DIMENSIONS}, 'additionalProperties': False},
            'option_checks': {'type': 'array', 'items': {'type': 'object', 'required': ['label', 'selected_as_gold', 'support_status', 'reason', 'quotes'], 'properties': {'label': {'type': 'string'}, 'selected_as_gold': {'type': 'boolean'}, 'support_status': {'enum': ['SUPPORTED', 'CONTRADICTED', 'NOT_ESTABLISHED', 'UNDETERMINED']}, 'reason': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote}}, 'additionalProperties': False}},
            'quotes': {'type': 'array', 'items': quote},
        }, 'additionalProperties': False,
    }
    cluster = {'$schema': quality['$schema'], 'type': 'object', 'required': ['arm_id', 'root_id', 'scope', 'cluster_id', 'members', 'reason', 'reviewer'], 'properties': {
        'arm_id': {'enum': list(ARMS)}, 'root_id': {'enum': list(ROOTS)},
        'scope': {'enum': ['final_sc_ms', 'primary_sc_ms', 'repair_sc_ms', 'final_ordering', 'primary_ordering', 'repair_ordering']},
        'cluster_id': {'type': 'string'}, 'members': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string'}},
        'reason': {'type': 'string', 'minLength': 1}, 'reviewer': {'const': REVIEWER},
    }, 'additionalProperties': False}
    post = {'$schema': quality['$schema'], 'type': 'object', 'required': ['view_id', 'reviewer', 'intent_alignment', 'reason', 'error_propagation', 'question_relation_kind', 'coverage_description', 'quotes'], 'properties': {
        'view_id': {'type': 'string'}, 'reviewer': {'const': REVIEWER}, 'intent_alignment': {'enum': POLICY['post_quality']['A_alignment_values'] + ['NOT_APPLICABLE']},
        'reason': {'type': 'string', 'minLength': 1}, 'error_propagation': {'type': 'array', 'items': {'type': 'object', 'required': ['source_error', 'judgment', 'reason', 'quotes'], 'properties': {'source_error': {'type': 'string'}, 'judgment': {'enum': ['SAME_ERROR_PRESENT', 'SAME_ERROR_ABSENT', 'UNDETERMINED', 'NOT_APPLICABLE']}, 'reason': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote}}, 'additionalProperties': False}},
        'question_relation_kind': {'enum': ['DIRECT_FACT_LOOKUP', 'MULTIFACT_RELATION', 'MIXED_OR_UNDETERMINED']}, 'coverage_description': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote},
    }, 'additionalProperties': False}
    overlap = {'$schema': quality['$schema'], 'type': 'object', 'required': ['view_id', 'comparison_kind', 'comparison_identity', 'judgment', 'reason', 'reviewer'], 'properties': {
        'view_id': {'type': 'string'}, 'comparison_kind': {'enum': ['FROZEN_HISTORY', 'CROSS_ARM']}, 'comparison_identity': {'type': 'string'}, 'judgment': {'enum': ['SAME_QUESTION', 'DISTINCT_RELATION', 'STEM_OR_FINGERPRINT_HINT_ONLY', 'UNDETERMINED']}, 'reason': {'type': 'string', 'minLength': 1}, 'reviewer': {'const': REVIEWER},
    }, 'additionalProperties': False}
    return {'quality_review.schema.json': quality, 'dedup_clusters.schema.json': cluster, 'post_quality_review.schema.json': post, 'overlap_review.schema.json': overlap}


def _schema_errors(value, schema, location='row'):
    """Check the limited, self-contained vocabulary used by the four formats."""
    errors = []
    kind = schema.get('type')
    valid_type = {'object': isinstance(value, dict), 'array': isinstance(value, list),
                  'string': isinstance(value, str), 'boolean': isinstance(value, bool)}
    if kind is not None and not valid_type.get(kind, False):
        return ['format_type:' + location]
    if 'enum' in schema and value not in schema['enum']:
        errors.append('format_enum:' + location)
    if 'const' in schema and value != schema['const']:
        errors.append('format_const:' + location)
    if isinstance(value, str) and len(value) < schema.get('minLength', 0):
        errors.append('format_empty_string:' + location)
    if kind == 'object':
        properties = schema.get('properties', {})
        if set(schema.get('required', [])) - set(value):
            errors.append('format_required:' + location)
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            errors.append('format_extra_fields:' + location)
        for key in set(properties) & set(value):
            errors.extend(_schema_errors(value[key], properties[key], location + '.' + key))
    if kind == 'array':
        if len(value) < schema.get('minItems', 0):
            errors.append('format_empty_array:' + location)
        if schema.get('uniqueItems') and len({_digest(v) for v in value}) != len(value):
            errors.append('format_duplicate_items:' + location)
        for index, row in enumerate(value):
            errors.extend(_schema_errors(row, schema.get('items', {}), f'{location}[{index}]'))
    return errors


def prepare_review_assets(run_dir):
    """Freeze the future rubric and genuinely empty review formats, idempotently."""
    run = Path(run_dir)
    path = run / 'review_policy.json'
    if path.exists() and _read(path) != POLICY:
        raise ValueError('existing_review_policy_differs_do_not_overwrite')
    _write(path, POLICY)
    for name, schema in _schemas().items():
        target = run / 'policy_snapshot/review_schemas' / name
        if target.exists() and _read(target) != schema:
            raise ValueError('existing_review_schema_differs_do_not_overwrite:' + name)
        _write(target, schema)
    for name in ('quality_review', 'dedup_clusters', 'post_quality_review', 'overlap_review'):
        target = run / 'reports' / (name + '.jsonl')
        if not target.exists():
            _write_jsonl(target, [])
    plan = '''# 当前 intent 对 Writer 的下游作用：两 root 配对准备

本轮只实现和离线准备，PREPARED_NOT_RUN；外部调用为零。未来仍需明确授权，固定身份不换目录、不创建授权标记或后台任务。

比较当前完整 V3_OFF intent 包对 Guided Writer 的作用，包含目标、类型和证据链。两 root 198480、122711 的原三个槽全部保留，包括已知错误；不改 Miner NOT_READY 或旧评分。A 使用原 contract，B 只移除指导语义并补一条中性选材说明。B 的 OUTPUT SHAPE、repair 及 normalization 必须无原 intent 隐藏依赖；候选自己的变量和证据仍必填。两臂 generation_source 都是 contract_guided。

两臂通过同一原 Writer 阶段、2 SC + 1 MS + 1 ordering minima、一次整批替换 repair、原 normalizer、root-arm 最终 validator 和 Gold 确定性置换检查。失败占预算；不补采样、不合并 primary/repair、不跨臂复用缓存。12 primary 的输入/身份离线冻结；真实 repair 内容尚未知，未来按同臂 builder 继承并逐次精确 token 检查。

未来顺序固定：198480 slot 1 A→B、slot 2 B→A、slot 3 A→B；122711 slot 1 B→A、slot 2 A→B、slot 3 B→A。每格 repair 紧随 primary。12 primary + 最多 12 repair = 最多 24 次逻辑调用；每 root-arm 3 + 最多 3。原 HTTP 429 重试单列，不改变逻辑上限。未知服务完成/付费状态立即停后续请求。

预冻结诊断见 review_policy.json 和 policy_snapshot/review_schemas/。准备时所有审阅 JSONL 为空，不伪造 QA。report 无服务调用：首先生成仅含候选与完整 transcript 的质量视图，arm/原 intent 映射单独保存；Codex 已知背景，不称独立盲审。完整质量判断经精确引用核查后冻结，再揭示 A 原合同用于覆盖、重复及已知错误传播诊断。B 的合同对齐为 NOT_APPLICABLE，不作质量失败。所有解析题都保留在 primary/repair 层，无法解析保留结构未评与请求分母。

主统计是每 root-arm 最终结构接受且质量 PASS 的 SC/MS 数及 /3 预定槽有效产量；两个 root 等权配对。ordering 单列，但其触发整批 repair/未接受的间接影响与 primary SC/MS 同时报告。原始有效量、人工语义重复簇去重产量均报告；缺完整簇判断不默认互不重复。历史比较只读已冻结 history_only.json，缺 options/Gold 时只作可靠 stem/fingerprint 提示，不称全历史去重。

成本包含全部失败、替换的 primary 和 repair；input/output/total、HTTP、逻辑调用和未知 usage 分开。零产量为 UNDEFINED_ZERO_YIELD，未知 usage 不把子集合当总成本，金额无可靠证据则未知。结果只适用于两个已知问题 root；B 自选更简单考点的可能性需保留，不推断所有 intent 无用、普遍能力或训练收益。

完成准备即停止。本轮无真实 Writer、Miner、Free、Solver、Planner、Realizer、Memory、训练或审核 API；不自动恢复生成、不实施 v4。准确未来执行及恢复命令由 EXECUTE_AFTER_AUTHORIZATION.md 给出。
'''
    target = run / 'EXPERIMENT_PLAN.md'
    if target.exists() and target.read_text() != plan:
        raise ValueError('existing_experiment_plan_differs_do_not_overwrite')
    target.write_text(plan)
    return {'review_policy': str(path), 'empty_review_rows': {n: len(_jsonl(run / 'reports' / (n + '.jsonl'))) for n in ('quality_review', 'dedup_clusters', 'post_quality_review', 'overlap_review')}, 'service_calls': 0}


def aggregate_quality(dimensions, question_type):
    if set(dimensions) != set(DIMENSIONS):
        raise ValueError('all_quality_dimensions_required')
    judgments = []
    for dimension, assessment in dimensions.items():
        judgment = assessment.get('judgment')
        if judgment not in ('PASS', 'FAIL', 'UNDETERMINED', 'NOT_APPLICABLE'):
            raise ValueError('invalid_quality_judgment:' + dimension)
        applicable = dimension not in TYPE_SPECIFIC or TYPE_SPECIFIC[dimension] == question_type
        if applicable == (judgment == 'NOT_APPLICABLE'):
            raise ValueError('quality_dimension_applicability:' + dimension)
        if not assessment.get('reason'):
            raise ValueError('quality_dimension_reason_missing:' + dimension)
        judgments.append(judgment)
    return 'FAIL' if 'FAIL' in judgments else 'UNDETERMINED' if 'UNDETERMINED' in judgments else 'PASS'


def _identity(row):
    identity = row.get('analysis_identity')
    if isinstance(identity, list) and len(identity) == 5:
        return tuple(identity)
    candidate = row.get('candidate', row)
    return (row['arm_id'], row['root_id'], row['slot_id'], row['stage'], candidate['candidate_id'])


def _candidate_visible(candidate):
    return {key: candidate[key] for key in (
        'question_type', 'question', 'options', 'items', 'correct_option', 'gold',
        'evidence_refs', 'evidence_groups', 'discriminative_variable',
    ) if key in candidate}


def _reviewable(candidate):
    qtype = candidate.get('question_type')
    container = candidate.get('items') if qtype == 'ordering' else candidate.get('options')
    gold = candidate.get('correct_option') if qtype == 'single_choice' else candidate.get('gold')
    return (qtype in ('single_choice', 'multi_select', 'ordering') and isinstance(candidate.get('question'), str)
            and bool(candidate['question'].strip()) and isinstance(container, Mapping) and bool(container)
            and all(isinstance(v, str) and v.strip() for v in container.values()) and bool(gold))


def collect_candidates(run_dir):
    """Retain stage rows including invalid/extras; absent JSON remains a request row."""
    run = Path(run_dir)
    stages = _jsonl(run / 'reports/runtime_stage_records.jsonl')
    completed_stage_keys = {(r['arm_id'], r['root_id'], r['slot_id'], r.get('stage', 'repair' if r.get('repair_of') else 'primary')) for r in stages}
    for path in sorted(run.glob('arms/*/qa/shards/shard-00/roots/*/stage_calls/*/*.json')):
        partial = _read(path, {})
        key = (partial.get('arm_id'), partial.get('root_id'), partial.get('slot_id'), partial.get('stage'))
        if key not in completed_stage_keys:
            stages.append({**partial, 'validation': {}, 'normalization_status': 'NOT_COMPLETED_NOT_INFERRED'})
            completed_stage_keys.add(key)
    finals = _jsonl(run / 'reports/runtime_final_candidates.jsonl')
    final_keys = {_identity(row) for row in finals}
    items, requests, errors, seen = [], [], [], set()
    for record in stages:
        arm, root, slot = record['arm_id'], record['root_id'], record['slot_id']
        stage = record.get('stage', 'repair' if record.get('repair_of') else 'primary')
        request_key = (arm, root, slot, stage)
        if request_key in seen:
            errors.append('duplicate_stage_record:' + repr(request_key))
            continue
        seen.add(request_key)
        normalized = record.get('validation') or {}
        rows = normalized.get('candidates') or []
        raw = record.get('raw_parsed')
        raw_rows = raw.get('candidates') if isinstance(raw, Mapping) else None
        if not isinstance(raw_rows, list):
            raw_rows = []
        acceptance = normalized.get('minimum_acceptance') or {}
        missing = acceptance.get('missing_minimums') or {}
        has_repair = any(r.get('arm_id') == arm and r.get('root_id') == root and r.get('slot_id') == slot and (r.get('stage') == 'repair' or r.get('repair_of')) for r in stages)
        requests.append({
            'arm_id': arm, 'root_id': root, 'slot_id': slot, 'stage': stage, 'request_id': record.get('request_id'),
            'raw_response_path': record.get('raw_response_path'), 'raw_parsed_candidate_rows': len(raw_rows),
            'normalized_candidate_rows': len(rows), 'parsed_candidate_rows': max(len(rows), len(raw_rows)),
            'parsed_question_rows': sum(_reviewable(r) for r in (rows or raw_rows) if isinstance(r, Mapping)),
            'structure_status': 'HAS_PARSED_ROWS' if rows or raw_rows else 'NOT_EVALUATED_STRUCTURE',
            'normalization_status': record.get('normalization_status', 'RECORDED_BY_ORIGINAL_STAGE'),
            'parse_error': record.get('parse_error'), 'normalization_errors': normalized.get('errors') or [],
            'minimum_acceptance': acceptance, 'replaced_by_repair': stage == 'primary' and has_repair,
            'ordering_shortfall': bool(missing.get('ordering')), 'ordering_only_shortfall': set(missing) == {'ordering'},
            'ordering_triggered_repair': stage == 'primary' and has_repair and bool(missing.get('ordering')),
            'whole_batch_not_accepted': not acceptance['accepted'] if 'accepted' in acceptance else None,
        })
        for index in range(max(len(rows), len(raw_rows))):
            candidate = rows[index] if index < len(rows) and isinstance(rows[index], Mapping) else {}
            original = raw_rows[index] if index < len(raw_rows) else None
            source_id = str(candidate.get('candidate_id') or (original.get('candidate_id') if isinstance(original, Mapping) else None) or f'__unnormalized_row_{index + 1}')
            candidate_id = source_id if source_id.startswith(root + ':') else root + ':' + source_id
            identity = (arm, root, slot, stage, candidate_id)
            visible = _candidate_visible(candidate if candidate else (original if isinstance(original, Mapping) else {}))
            items.append({
                'analysis_identity': list(identity), 'view_id': 'Q-' + _digest(identity)[:20],
                'arm_id': arm, 'root_id': root, 'slot_id': slot, 'stage': stage,
                'candidate_id': candidate_id, 'source_candidate_id': candidate.get('candidate_id'),
                'raw_declared_candidate_id': original.get('candidate_id') if isinstance(original, Mapping) else None,
                'source_row_index': index + 1, 'raw_candidate': original, 'normalized_candidate': candidate,
                'candidate': visible, 'generation_source': 'contract_guided',
                'program_schema_valid': bool(candidate.get('schema_valid')),
                'minima_batch_accepted': bool(acceptance.get('accepted')),
                'final_validator_accepted': identity in final_keys,
                'replaced_primary': stage == 'primary' and has_repair,
                'content_reviewable': _reviewable(visible),
                'contract_group_alignment_analysis': 'NOT_APPLICABLE' if arm == 'WITHOUT_INTENT' else candidate.get('contract_group_alignment'),
            })
    item_keys = {_identity(item) for item in items}
    if len(item_keys) != len(items):
        errors.append('duplicate_analysis_candidate_identity')
    for key in final_keys - item_keys:
        errors.append('final_candidate_has_no_stage_row:' + repr(key))
    for item in items:
        if item['replaced_primary'] and item['final_validator_accepted']:
            errors.append('replaced_primary_in_final_pool:' + repr(_identity(item)))
    return items, requests, errors


def _check_quotes(quotes, turns, errors, location):
    if not isinstance(quotes, list):
        errors.append('quotes_not_array:' + location)
        return
    for quote in quotes:
        if not isinstance(quote, Mapping) or quote.get('turn_ref') not in turns or not isinstance(quote.get('quote'), str) or not quote['quote'] or quote['quote'] not in turns[quote['turn_ref']]:
            errors.append('nonexact_quote:' + location)


def validate_quality_reviews(run_dir, items, authored):
    run = Path(run_dir)
    expected = {i['view_id']: i for i in items}
    by_view, errors, judgments = {}, [], {}
    for row in authored:
        format_errors = _schema_errors(row, _schemas()['quality_review.schema.json'], 'quality_review')
        errors.extend(format_errors)
        if format_errors:
            continue
        view = row.get('view_id')
        if view in by_view:
            errors.append('duplicate_quality_review:' + str(view))
        if view not in expected:
            errors.append('unknown_quality_view:' + str(view))
        by_view[view] = row
    transcripts = {}
    for item in items:
        view = item['view_id']
        row = by_view.get(view)
        if row is None:
            judgments[view] = 'PENDING_REVIEW' if item['content_reviewable'] else 'NOT_EVALUATED_STRUCTURE'
            continue
        if row.get('reviewer') != REVIEWER or not row.get('reason'):
            errors.append('quality_attribution_or_reason_missing:' + view)
        root = item['root_id']
        if root not in transcripts:
            transcript = _read(run / 'inputs/shared' / root / 'expanded_dialogue_clean.json', {})
            transcripts[root] = {t['turn_ref']: t['text'] for t in transcript.get('turns', [])}
        turns = transcripts[root]
        _check_quotes(row.get('quotes'), turns, errors, view)
        if not item['content_reviewable']:
            if row.get('judgment') != 'NOT_EVALUATED_STRUCTURE':
                errors.append('quality_pass_on_unreviewable_structure:' + view)
            judgments[view] = 'NOT_EVALUATED_STRUCTURE'
            continue
        try:
            judgment = aggregate_quality(row.get('dimensions', {}), item['candidate'].get('question_type'))
        except ValueError as error:
            errors.append(str(error) + ':' + view)
            judgments[view] = 'PENDING_REVIEW'
            continue
        if row.get('judgment') != judgment:
            errors.append('quality_aggregate_conflict:' + view)
        if not row.get('quotes'):
            errors.append('quality_exact_evidence_required:' + view)
        for name, assessment in row['dimensions'].items():
            _check_quotes(assessment.get('quotes'), turns, errors, view + ':' + name)
        candidate = item['candidate']
        options = candidate.get('items') if candidate.get('question_type') == 'ordering' else candidate.get('options')
        checks = row.get('option_checks') or []
        if len(checks) != len(options) or {c.get('label') for c in checks} != set(options):
            errors.append('each_option_must_be_reviewed_once:' + view)
        gold = [candidate.get('correct_option')] if candidate.get('question_type') == 'single_choice' else candidate.get('gold', [])
        if isinstance(gold, str):
            gold = [s.strip() for s in gold.split(',') if s.strip()]
        for check in checks:
            if check.get('selected_as_gold') != (check.get('label') in gold) or check.get('support_status') not in ('SUPPORTED', 'CONTRADICTED', 'NOT_ESTABLISHED', 'UNDETERMINED') or not check.get('reason'):
                errors.append('invalid_option_check:' + view)
            _check_quotes(check.get('quotes'), turns, errors, view + ':' + str(check.get('label')))
        judgments[view] = judgment
    return judgments, errors


def history_inventory(run_dir):
    rows = []
    for root in ROOTS:
        path = Path(run_dir) / 'inputs/shared' / root / 'history_only.json'
        data = _read(path, {})
        records = data.get('records', []) if isinstance(data, Mapping) else []
        rows.append({'root_id': root, 'path': str(path), 'sha256': _sha(path) if path.is_file() else None,
                     'size': path.stat().st_size if path.is_file() else None, 'record_count': len(records),
                     'available_fields': sorted({k for record in records if isinstance(record, Mapping) for k in record}),
                     'records_with_options': sum(bool(r.get('options')) for r in records),
                     'records_with_gold': sum(bool(r.get('gold') or r.get('correct_option')) for r in records),
                     'scope': 'ONLY_FROZEN_HISTORY_ONLY_NOT_FULL_HISTORY',
                     'gold_assessment_from_stem_only': 'NOT_PERMITTED'})
    return rows


def call_accounting(run_dir):
    run = Path(run_dir)
    groups, known_paths = [], set()
    for arm in ARMS:
        for root in ROOTS:
            base = root_path(run, arm, root)
            requests = []
            for path in sorted((base / 'ledger/requests').glob('*.json')):
                known_paths.add(path)
                state = _read(path, {})
                request_id = state.get('request_id', path.stem)
                stage = 'repair' if '-repair-' in request_id else 'primary'
                outcome = _read(base / 'ledger/transport_outcomes' / (request_id + '.json'), {})
                metadata = state.get('response_metadata') or {}
                usage = metadata.get('usage') if metadata.get('usage') is not None else outcome.get('usage')
                known_failure = outcome.get('service_completion_status') == 'KNOWN_HTTP_REJECTION'
                complete = state.get('status') == 'completed'
                # Completed errors without usage and known HTTP failures remain missing-cost evidence.
                token_fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
                usage_complete = isinstance(usage, Mapping) and all(isinstance(usage.get(k), (int, float)) and not isinstance(usage.get(k), bool) for k in token_fields)
                requests.append({'request_id': request_id, 'stage': stage, 'ledger_status': state.get('status'),
                    'backend_mode': state.get('backend_mode'),
                    'service_completion_status': outcome.get('service_completion_status', 'COMPLETED' if complete else 'UNKNOWN'),
                    'completed': complete, 'known_failure': known_failure, 'unknown': not complete and not known_failure,
                    'http_attempts': metadata.get('transport_attempt_count', outcome.get('transport_attempt_count')),
                    'http_429_retries': metadata.get('rate_limit_retry_count', outcome.get('rate_limit_retry_count')),
                    'usage': usage, 'usage_complete': usage_complete, 'payment_status': 'UNKNOWN_NO_BILLING_EVIDENCE'})
            events = [e for e in _jsonl(run / 'audit/http_events.jsonl') if e.get('arm_id') == arm and e.get('root_id') == root]
            attempt_count = sum(e.get('event') == 'http_attempt_started' for e in events)
            groups.append({'arm_id': arm, 'root_id': root, 'requests': requests, 'logical_calls': len(requests),
                'primary_calls': sum(r['stage'] == 'primary' for r in requests), 'repair_calls': sum(r['stage'] == 'repair' for r in requests),
                'completed_calls': sum(r['completed'] for r in requests), 'known_failed_calls': sum(r['known_failure'] for r in requests),
                'unknown_calls': sum(r['unknown'] for r in requests), 'usage_missing_calls': sum(not r['usage_complete'] for r in requests),
                'http_attempts': attempt_count if events else (sum(r['http_attempts'] for r in requests) if all(isinstance(r['http_attempts'], int) for r in requests) else None),
                'http_429_retries': sum(r['http_429_retries'] for r in requests) if all(isinstance(r['http_429_retries'], int) for r in requests) else None,
                'token_usage_known_subtotal': {key: sum((r['usage'] or {}).get(key, 0) for r in requests if isinstance(r['usage'], Mapping) and isinstance(r['usage'].get(key), (int, float))) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                'token_usage_status': 'NOT_RUN' if not requests else 'COMPLETE' if all(r['usage_complete'] for r in requests) else 'INCOMPLETE_USAGE'})
    unexpected = sorted(str(p) for p in set(run.glob('arms/**/ledger/requests/*.json')) - known_paths)
    return {'planned_primary_calls': 12, 'maximum_repair_calls': 12, 'maximum_logical_calls': 24,
        'groups': groups, **{k: sum(g[k] for g in groups) for k in ('logical_calls', 'primary_calls', 'repair_calls', 'completed_calls', 'known_failed_calls', 'unknown_calls', 'usage_missing_calls')},
        'nonlive_ledger_records': [r['request_id'] for g in groups for r in g['requests'] if r['backend_mode'] != 'live'],
        'unexpected_ledger_paths': unexpected, 'monetary_cost': None, 'monetary_cost_status': 'UNKNOWN_NO_BILLING_OR_PRICING_EVIDENCE'}


def dedup_count(items, judgments, clusters, arm, root, scope):
    stage, category = scope.split('_', 1)
    population = {i['view_id'] for i in items if i['arm_id'] == arm and i['root_id'] == root
        and (i['final_validator_accepted'] if stage == 'final' else i['stage'] == stage)
        and (i['candidate'].get('question_type') in ('single_choice', 'multi_select') if category == 'sc_ms' else i['candidate'].get('question_type') == 'ordering')
        and judgments.get(i['view_id']) == 'PASS'}
    chosen = [c for c in clusters if c.get('arm_id') == arm and c.get('root_id') == root and c.get('scope') == scope]
    members = [v for c in chosen for v in c.get('members', [])]
    if not population:
        return {'status': 'EMPTY_PASS_POPULATION', 'count': 0, 'population': 0, 'clusters': []}
    valid = (set(members) == population and len(members) == len(population)
             and len({c.get('cluster_id') for c in chosen}) == len(chosen)
             and all(c.get('reviewer') == REVIEWER and c.get('reason') and c.get('members') for c in chosen))
    return {'status': 'COMPLETE' if valid else 'NOT_REVIEWED_OR_INVALID_PARTITION',
            'count': len(chosen) if valid else None, 'population': len(population), 'clusters': chosen}


def cost_per_yield(accounting_group, count):
    if not accounting_group['logical_calls']:
        return {'status': 'NOT_RUN', 'tokens_per_effective_question': None}
    if count is None:
        return {'status': 'YIELD_NOT_ESTABLISHED', 'tokens_per_effective_question': None}
    if count == 0:
        return {'status': 'UNDEFINED_ZERO_YIELD', 'tokens_per_effective_question': None,
                'usage_status': accounting_group['token_usage_status']}
    if accounting_group['token_usage_status'] != 'COMPLETE':
        return {'status': 'INCOMPLETE_USAGE', 'tokens_per_effective_question': None}
    return {'status': 'COMPLETE', 'tokens_per_effective_question': accounting_group['token_usage_known_subtotal']['total_tokens'] / count}


def _combine_accounting(groups):
    numeric = ('logical_calls', 'primary_calls', 'repair_calls', 'completed_calls', 'known_failed_calls', 'unknown_calls', 'usage_missing_calls')
    requests = [r for g in groups for r in g['requests']]
    result = {k: sum(g[k] for g in groups) for k in numeric}
    result.update(requests=requests,
                  token_usage_known_subtotal={k: sum(g['token_usage_known_subtotal'][k] for g in groups) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                  token_usage_status='NOT_RUN' if not requests else 'COMPLETE' if all(r['usage_complete'] for r in requests) else 'INCOMPLETE_USAGE')
    for field in ('http_attempts', 'http_429_retries'):
        result[field] = sum(g[field] for g in groups) if all(g[field] is not None for g in groups) else None
    result['per_stage'] = {}
    for stage in ('primary', 'repair'):
        own = [r for r in requests if r['stage'] == stage]
        result['per_stage'][stage] = {'logical_calls': len(own),
            'known_failed_calls': sum(r['known_failure'] for r in own), 'unknown_calls': sum(r['unknown'] for r in own),
            'usage_missing_calls': sum(not r['usage_complete'] for r in own),
            'token_usage_known_subtotal': {k: sum((r['usage'] or {}).get(k, 0) for r in own if isinstance(r['usage'], Mapping) and isinstance(r['usage'].get(k), (int, float))) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'token_usage_status': 'NOT_RUN' if not own else 'COMPLETE' if all(r['usage_complete'] for r in own) else 'INCOMPLETE_USAGE'}
    return result


def summarize(items, requests, judgments, clusters, accounting, quality_frozen=False):
    groups = []
    for arm in ARMS:
        for root in ROOTS:
            own = [i for i in items if i['arm_id'] == arm and i['root_id'] == root]
            calls = [r for r in requests if r['arm_id'] == arm and r['root_id'] == root]
            count = next(g for g in accounting['groups'] if g['arm_id'] == arm and g['root_id'] == root)
            layers = {}
            for stage in ('primary', 'repair', 'final'):
                selected = [i for i in own if i['final_validator_accepted']] if stage == 'final' else [i for i in own if i['stage'] == stage]
                layers[stage] = {}
                for category in ('sc_ms', 'ordering'):
                    subset = [i for i in selected if i['candidate'].get('question_type') in (('single_choice', 'multi_select') if category == 'sc_ms' else ('ordering',))]
                    labels = Counter(judgments.get(i['view_id'], 'PENDING_REVIEW') for i in subset)
                    denominator = len(subset)
                    layers[stage][category] = {'candidate_rows': denominator,
                        'normalizer_structural_valid_rows': sum(i['program_schema_valid'] for i in subset),
                        'minima_batch_accepted_rows': sum(i['program_schema_valid'] and i['minima_batch_accepted'] for i in subset),
                        'quality': {j: labels[j] for j in (*JUDGMENTS, 'PENDING_REVIEW')},
                        'quality_pass_fraction': labels['PASS'] / denominator if denominator else None,
                        'fraction_denominator': 'all final validator accepted rows' if stage == 'final' else 'all parsed rows of this stage and type',
                    }
            final = layers['final']['sc_ms']
            pass_count = final['quality']['PASS']
            eligible = bool(count['logical_calls']) and quality_frozen
            dedup = dedup_count(items, judgments, clusters, arm, root, 'final_sc_ms') if quality_frozen else {'status': 'WAITING_QUALITY_FREEZE', 'count': None}
            groups.append({'arm_id': arm, 'root_id': root, 'planned_slots': 3,
                'primary_raw_generated_rows': sum(r['parsed_candidate_rows'] for r in calls if r['stage'] == 'primary'),
                'stage_requests': calls, 'layers': layers,
                'final_sc_ms_pass_count': pass_count if eligible else None,
                'effective_sc_ms_yield_per_planned_slot': pass_count / 3 if eligible else None,
                'yield_status': 'QUALITY_FROZEN' if eligible else 'NOT_RUN' if not count['logical_calls'] else 'PENDING_QUALITY_FREEZE',
                'dedup_final_sc_ms': dedup,
                'dedup_effective_sc_ms_yield_per_planned_slot': dedup['count'] / 3 if dedup.get('count') is not None and eligible else None,
                'primary_and_repair_dedup': {s: dedup_count(items, judgments, clusters, arm, root, s + '_sc_ms') if quality_frozen else {'status': 'WAITING_QUALITY_FREEZE', 'count': None} for s in ('primary', 'repair')},
                'ordering_dedup': {s: dedup_count(items, judgments, clusters, arm, root, s + '_ordering') if quality_frozen else {'status': 'WAITING_QUALITY_FREEZE', 'count': None} for s in ('primary', 'repair', 'final')},
                'ordering_triggered_repair_count': sum(r['ordering_triggered_repair'] for r in calls),
                'ordering_only_repair_count': sum(r['stage'] == 'primary' and r['replaced_by_repair'] and r['ordering_only_shortfall'] for r in calls),
                'unparsed_or_empty_request_count': sum(r['structure_status'] == 'NOT_EVALUATED_STRUCTURE' for r in calls),
                'accounting': {**count, 'per_stage': _combine_accounting([count])['per_stage']},
                'token_cost_per_final_pass': cost_per_yield(count, pass_count if eligible else None),
                'token_cost_per_dedup_final_pass': cost_per_yield(count, dedup.get('count') if eligible else None),
            })
    arms = []
    for arm in ARMS:
        selected = [g for g in groups if g['arm_id'] == arm]
        count = _combine_accounting([g['accounting'] for g in selected])
        passes = [g['final_sc_ms_pass_count'] for g in selected]
        dedups = [g['dedup_final_sc_ms']['count'] for g in selected]
        total_pass = sum(passes) if all(p is not None for p in passes) else None
        total_dedup = sum(dedups) if all(p is not None for p in dedups) and quality_frozen else None
        denominator = sum(g['layers']['final']['sc_ms']['candidate_rows'] for g in selected)
        arms.append({'arm_id': arm, 'planned_slots': 6, 'final_sc_ms_pass_count': total_pass,
            'effective_yield_per_planned_slot': total_pass / 6 if total_pass is not None else None,
            'final_structure_accepted_sc_ms': denominator,
            'final_sc_ms_quality_counts': {j: sum(g['layers']['final']['sc_ms']['quality'][j] for g in selected) for j in (*JUDGMENTS, 'PENDING_REVIEW')},
            'quality_pass_fraction': total_pass / denominator if total_pass is not None and denominator else None,
            'dedup_final_sc_ms_count': total_dedup,
            'dedup_effective_yield_per_planned_slot': total_dedup / 6 if total_dedup is not None else None,
            'accounting': count, 'token_cost_per_final_pass': cost_per_yield(count, total_pass),
            'token_cost_per_dedup_final_pass': cost_per_yield(count, total_dedup)})
    pairs = []
    for root in ROOTS:
        a = next(g for g in groups if g['root_id'] == root and g['arm_id'] == ARMS[0])
        b = next(g for g in groups if g['root_id'] == root and g['arm_id'] == ARMS[1])
        av, bv = a['effective_sc_ms_yield_per_planned_slot'], b['effective_sc_ms_yield_per_planned_slot']
        ad, bd = a['dedup_effective_sc_ms_yield_per_planned_slot'], b['dedup_effective_sc_ms_yield_per_planned_slot']
        pairs.append({'root_id': root, 'A_effective_yield': av, 'B_effective_yield': bv,
                      'A_minus_B': av - bv if av is not None and bv is not None else None,
                      'A_minus_B_dedup': ad - bd if ad is not None and bd is not None else None})
    def mean(key):
        values = [p[key] for p in pairs]
        return sum(values) / 2 if all(v is not None for v in values) else None
    return {'groups': groups, 'arms': arms, 'root_paired': pairs, 'equal_root_weight_mean_delta': mean('A_minus_B'),
            'equal_root_weight_mean_dedup_delta': mean('A_minus_B_dedup'),
            'statistical_unit': 'root; equal weights; slot is call tracking only', 'planned_slots_per_arm': 6,
            'semantic_clustering_is_authored_only': True}


def report(run_dir):
    """Prepare empty schemas or report real runtime files, without any service call."""
    run = Path(run_dir)
    prepare_review_assets(run)
    items, requests, errors = collect_candidates(run)
    transcripts = {root: _read(run / 'inputs/shared' / root / 'expanded_dialogue_clean.json', {}) for root in ROOTS}
    views = sorted([{'view_id': i['view_id'], 'candidate': i['candidate'], 'complete_transcript': transcripts[i['root_id']],
              'content_reviewable': i['content_reviewable'], 'reviewer': REVIEWER} for i in items]
              , key=lambda row: row['view_id'])
    mapping = [{k: i[k] for k in ('view_id', 'analysis_identity', 'arm_id', 'root_id', 'slot_id', 'stage', 'candidate_id', 'source_row_index')} for i in items]
    _write_jsonl(run / 'reports/quality_review_views.jsonl', views)
    _write_jsonl(run / 'audit/quality_view_mapping.jsonl', mapping)
    _write_jsonl(run / 'reports/candidate_stage_inventory.jsonl', items)
    _write_jsonl(run / 'reports/request_structure_inventory.jsonl', requests)
    authored = _jsonl(run / 'reports/quality_review.jsonl')
    judgments, review_errors = validate_quality_reviews(run, items, authored)
    errors.extend(review_errors)
    accounting = call_accounting(run)
    if accounting['unexpected_ledger_paths']:
        errors.append('unexpected_request_ledger_paths')
    if accounting['nonlive_ledger_records']:
        errors.append('nonlive_or_synthetic_ledger_cannot_be_reported_as_real_QA')
    if requests and not accounting['logical_calls']:
        errors.append('runtime_QA_without_live_request_ledger')
    if accounting['unknown_calls']:
        errors.append('unknown_service_completion_pause_required')
    primary_keys = {(r['arm_id'], r['root_id'], r['slot_id']) for r in requests if r['stage'] == 'primary'}
    expected_keys = {(a, r, 'intent_' + str(s)) for a in ARMS for r in ROOTS for s in (1, 2, 3)}
    freeze_path = run / 'audit/quality_review_freeze.json'
    freeze_evidence = {
        'quality_review_sha256': _sha(run / 'reports/quality_review.jsonl'),
        'view_digest': _digest(views), 'mapping_digest': _digest(mapping),
        'runtime_stage_records_sha256': _sha(run / 'reports/runtime_stage_records.jsonl') if (run / 'reports/runtime_stage_records.jsonl').is_file() else None,
        'runtime_final_candidates_sha256': _sha(run / 'reports/runtime_final_candidates.jsonl') if (run / 'reports/runtime_final_candidates.jsonl').is_file() else None,
    }
    freeze = _read(freeze_path)
    if freeze is not None and freeze.get('evidence') != freeze_evidence:
        errors.append('frozen_quality_inputs_or_judgments_changed')
    all_reviewed = all(judgments.get(i['view_id']) != 'PENDING_REVIEW' for i in items)
    runtime = _read(run / 'reports/runtime_summary.json', {})
    runtime_complete = (runtime.get('status') == 'COMPLETED' and runtime.get('backend_mode') == 'live'
                        and (run / 'reports/runtime_final_candidates.jsonl').is_file())
    can_freeze = runtime_complete and bool(accounting['logical_calls']) and primary_keys == expected_keys and not accounting['unknown_calls'] and all_reviewed and not errors
    if freeze is None and can_freeze:
        freeze = {'reviewer': REVIEWER, 'status': 'QUALITY_FROZEN', 'evidence': freeze_evidence,
                  'primary_planned_slots': 12, 'candidate_rows': len(items), 'reviewed_rows': len(authored),
                  'not_evaluated_structure_rows': sum(j == 'NOT_EVALUATED_STRUCTURE' for j in judgments.values())}
        _write(freeze_path, freeze)
    quality_frozen = freeze is not None and freeze.get('evidence') == freeze_evidence and not errors
    clusters = _jsonl(run / 'reports/dedup_clusters.jsonl')
    post = _jsonl(run / 'reports/post_quality_review.jsonl')
    overlap = _jsonl(run / 'reports/overlap_review.jsonl')
    valid_clusters = []
    for name, rows in (('dedup_clusters.schema.json', clusters), ('post_quality_review.schema.json', post), ('overlap_review.schema.json', overlap)):
        for row in rows:
            format_errors = _schema_errors(row, _schemas()[name], name)
            errors.extend(format_errors)
            if name == 'dedup_clusters.schema.json' and not format_errors:
                valid_clusters.append(row)
    if (clusters or post or overlap) and not quality_frozen:
        errors.append('post_quality_judgments_before_valid_quality_freeze')
    if quality_frozen:
        post_views = []
        for item in items:
            contracts = _read(run / 'inputs/shared' / item['root_id'] / 'reasoning_contracts.json', [])
            if isinstance(contracts, dict):
                contracts = contracts.get('contracts', [])
            contract = next((c for c in contracts if c.get('intent_id') == item['slot_id']), None) if item['arm_id'] == ARMS[0] else None
            source_findings = _read(run / 'inputs/shared' / item['root_id'] / 'source_intent_diagnostics.json', [])
            if isinstance(source_findings, Mapping):
                source_findings = source_findings.get('records', [])
            diagnostic = next((d for d in source_findings if d.get('intent_id') == item['slot_id'] or d.get('declared_intent_id') == item['slot_id']), None) if item['arm_id'] == ARMS[0] else None
            post_views.append({'view_id': item['view_id'], 'analysis_identity': item['analysis_identity'],
                'quality_judgment': judgments[item['view_id']], 'original_contract': contract,
                'original_miner_diagnostic': diagnostic,
                'miner_error_provenance_status': 'AVAILABLE' if diagnostic is not None else 'NOT_APPLICABLE' if item['arm_id'] == ARMS[1] else 'SOURCE_DIAGNOSTIC_SIDECAR_NOT_AVAILABLE',
                'intent_alignment': 'TO_REVIEW' if item['arm_id'] == ARMS[0] else 'NOT_APPLICABLE',
                'contract_group_alignment_analysis': item['contract_group_alignment_analysis'],
                'candidate': item['candidate'], 'reviewer': REVIEWER})
        _write_jsonl(run / 'reports/post_quality_views.jsonl', post_views)
        by_view = {i['view_id']: i for i in items}
        seen_post = set()
        for row in post:
            if _schema_errors(row, _schemas()['post_quality_review.schema.json']):
                continue
            item = by_view.get(row.get('view_id'))
            if item is None or row.get('reviewer') != REVIEWER:
                errors.append('unknown_or_invalid_post_quality_review')
                continue
            if item['view_id'] in seen_post:
                errors.append('duplicate_post_quality_review:' + item['view_id'])
            seen_post.add(item['view_id'])
            allowed = ['NOT_APPLICABLE'] if item['arm_id'] == 'WITHOUT_INTENT' else POLICY['post_quality']['A_alignment_values']
            if row.get('intent_alignment') not in allowed:
                errors.append('post_quality_intent_alignment_invalid:' + item['view_id'])
            turns = {t['turn_ref']: t['text'] for t in transcripts[item['root_id']].get('turns', [])}
            _check_quotes(row.get('quotes'), turns, errors, 'post_quality:' + item['view_id'])
            for finding in row.get('error_propagation', []):
                _check_quotes(finding.get('quotes'), turns, errors, 'error_propagation:' + item['view_id'])
    summary = summarize(items, requests, judgments, valid_clusters if quality_frozen else [], accounting, quality_frozen)
    status = 'BLOCKED' if errors else 'PREPARED_NOT_RUN' if not accounting['logical_calls'] else 'QUALITY_FROZEN_DIAGNOSTICS_PENDING' if quality_frozen else 'QUALITY_REVIEW_PENDING'
    result = {'status': status, 'reviewer': REVIEWER, 'service_calls_by_report': 0,
        'LIVE_STATUS': 'NOT_RUN' if not accounting['logical_calls'] else 'RUNTIME_ARTIFACTS_PRESENT',
        'ACTUAL_EXTERNAL_LOGICAL_CALLS': accounting['logical_calls'],
        'quality_frozen': quality_frozen, 'quality_judgment_rows': len(authored), 'candidate_rows': len(items),
        'post_quality_review_rows': len(post), 'dedup_cluster_rows': len(clusters), 'overlap_review_rows': len(overlap),
        'errors': errors, 'accounting': accounting, 'summary': summary, 'history_pools': history_inventory(run),
        'synthetic_as_actual_results': False, 'production_admission_changed': False,
        'interpretation_limits': POLICY['interpretation'],
    }
    _write(run / 'audit/report_accounting.json', accounting)
    _write(run / 'reports/history_pool_inventory.json', result['history_pools'])
    _write(run / 'reports/paired_comparison.json', summary)
    _write(run / 'reports/offline_diagnostic_status.json', result)
    lines = ['# Writer intent ablation: offline diagnostic status', '',
        'Status: ' + status + '. Reporter service calls: 0.', '',
        'No actual QA exists yet; empty templates are not generated outcomes.' if not accounting['logical_calls'] else 'Quality and clustering are authored Codex offline diagnostics, separate from original program acceptance.', '',
        'Every root-arm keeps three scheduled slots. SC/MS is primary; ordering and its batch-repair effects are reported separately.',
        'Quality views hide arm and original intent. This is not an independent blind review or human truth certification.',
        'Complete quality judgments are frozen before original A guidance, coverage and semantic duplicate/propagation analysis are exposed.', '',
        f"Candidate rows: {len(items)}; authored quality rows: {len(authored)}; quality frozen: {quality_frozen}.",
        'See paired_comparison.json for stage-specific denominators, root-paired yield and complete-arm token costs; no pending result is represented as measured zero accuracy.',
        'See history_pool_inventory.json for the finite frozen history fields and Gold/options limitations.', '',
        'Errors: ' + json.dumps(errors, ensure_ascii=False), '',
        'This entry never sends model calls, changes historical scores, modifies production acceptance, or starts additional experiments.', '']
    (run / 'reports/offline_diagnostic_report.md').write_text('\n'.join(lines))
    return result
