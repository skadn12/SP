"""No-service preparation and layered diagnostics for SC/MS quality guidance.

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

ARMS = ('BASE_WRITER', 'SCMS_QUALITY_WRITER')
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
VERSION = 'h4_r02_writer_scms_quality_ablation_v1_review_v1'
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
        'Only after this freeze expose the same original contracts/Miner findings for BOTH intent-guided arms, then perform coverage, duplicate and error-propagation analysis.',
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
        'paired_difference': 'SCMS_QUALITY_WRITER minus BASE_WRITER at each root; average the two root differences with equal weights. No candidate-ID pairing, accuracy renaming, significance or causal-stability claims.',
    },
    'dedup_rubric': {
        'judge': 'Codex offline semantic judgment after quality freeze; exact text/fingerprint are hints only and never automatic clusters.',
        'same_question': 'Same answer relation, object/person/event and evidence scope despite wording or option-label permutation is a duplicate, including SC versus MS representations of the same complete answer relation.',
        'not_sufficient': 'Shared topic, entity, overlapping refs, or repeated generic stem alone is insufficient. Distinct temporal phase, relation, object, or materially different answer content can be distinct; explain the boundary.',
        'main_population': 'Final quality-PASS SC/MS within each root-arm, covering same-batch and cross-slot duplicates; clusters must partition the population exactly once, including explicit singleton judgments.',
        'dedup_yield': 'Manually supported cluster count / 3 scheduled slots per root-arm; missing/incomplete clustering is NOT_REVIEWED, never implicit singleton counting.',
        'other_layers': 'Primary and repair duplicates and costs are separate; their counts never add to final coverage. Ordering clusters are separate.',
        'cross_arm': 'Overlap is descriptive, never remove BASE questions from QUALITY as historical duplicates.',
    },
    'history_policy': {
        'pool': 'Only each frozen inputs/shared/<root>/history_only.json; retain path, SHA256, record count and available fields.',
        'limitations': 'Not all historical QA. Where options/Gold are absent, report justified stem/fingerprint overlap hints or UNDETERMINED; do not infer Gold error or claim exhaustive historical semantic deduplication.',
        'no_prompt_injection_or_pool_expansion': True,
    },
    'post_quality': {
        'alignment_values_both_arms': ['FOLLOWS_TARGET', 'CORRECTS_TARGET_OR_EVIDENCE', 'OTHER_RELATION', 'UNDETERMINED'],
        'intent_guidance_both_arms': True,
        'error_propagation': 'For each known source error, compare the actual QA relation/Gold and exact refs. Record SAME_ERROR_PRESENT, SAME_ERROR_ABSENT or UNDETERMINED with reasons; A deliberately incorrect distractor alone is not error propagation; lexical similarity or ref overlap does not establish propagation or model-internal causality. Correlated error dimensions are not independent samples.',
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
        'QUALITY_better': 'Supports a local benefit of this SC/MS guidance in these two known cases only; assess error changes, dedup yield, shared facts, relation selection and complete token costs together.',
        'BASE_better': 'The added guidance may introduce a tradeoff or fail to help these two cases; do not generalize beyond this experiment.',
        'similar': 'Insufficient evidence of difference.',
        'both_poor': 'Investigate common Writer/Gold issues first.',
        'excluded_claims': ['all intents unnecessary', 'feedback ineffective', 'universal model capability', 'training benefit', 'significance', 'stable causal effect', 'observed model internal self-check', 'unchanged actual ordering outputs', 'production validation'],
    },
    'shared_fact_coverage': {
        'format': 'One authored row per root and primary/repair/final layer. List every quality-PASS SC/MS view for each arm and cover every listed view in factual groups; a question may span multiple groups.',
        'judge': 'Describe person/event/object/time, answer relation and exact transcript evidence. Different question clusters can share facts. Facts/groups are descriptive, not independent knowledge-point or difficulty counts.',
        'selection': 'Explain scope/time shifts and DIRECT_FACT_LOOKUP versus MULTIFACT_RELATION selection, including whether quality trades away complex relations or coverage. Never infer internal self-check execution.',
        'empty': 'Explicitly describe empty arm populations or both-empty layers; empty preparation has no authored rows.',
    },
    'completion': {
        'requirements': ['live runtime complete and 12 primary slots recorded', 'valid frozen quality for all parsed rows; unreviewable structure preserved', 'post review of every view and every known source FAIL dimension for both arms', 'complete explicit PASS dedup partitions for all six layer/type scopes', 'every view compared with its frozen finite history pool', 'all final PASS SC/MS cross-arm pairs and discarded-to-retained same-arm pairs', 'root/layer shared-fact coverage and relation-selection descriptions'],
        'invalid': 'BLOCKED for invalid/contradictory rows; valid missing work is listed precisely as pending. Complete valid diagnostics are COMPLETED, never permanently pending.',
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
        'view_id': {'type': 'string'}, 'reviewer': {'const': REVIEWER}, 'intent_alignment': {'enum': POLICY['post_quality']['alignment_values_both_arms']},
        'reason': {'type': 'string', 'minLength': 1}, 'error_propagation': {'type': 'array', 'items': {'type': 'object', 'required': ['source_error', 'judgment', 'reason', 'quotes'], 'properties': {'source_error': {'type': 'string'}, 'judgment': {'enum': ['SAME_ERROR_PRESENT', 'SAME_ERROR_ABSENT', 'UNDETERMINED']}, 'reason': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote}}, 'additionalProperties': False}},
        'question_relation_kind': {'enum': ['DIRECT_FACT_LOOKUP', 'MULTIFACT_RELATION', 'MIXED_OR_UNDETERMINED']}, 'coverage_description': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'items': quote},
    }, 'additionalProperties': False}
    overlap = {'$schema': quality['$schema'], 'type': 'object', 'required': ['view_id', 'comparison_kind', 'comparison_identity', 'judgment', 'reason', 'reviewer'], 'properties': {
        'view_id': {'type': 'string'}, 'comparison_kind': {'enum': ['FROZEN_HISTORY', 'CROSS_ARM', 'DISCARDED_TO_FINAL']}, 'comparison_identity': {'type': 'string'}, 'judgment': {'enum': ['SAME_QUESTION', 'DISTINCT_RELATION', 'STEM_OR_FINGERPRINT_HINT_ONLY', 'UNDETERMINED']}, 'reason': {'type': 'string', 'minLength': 1}, 'reviewer': {'const': REVIEWER},
    }, 'additionalProperties': False}
    coverage = {'$schema': quality['$schema'], 'type': 'object',
        'required': ['root_id', 'scope', 'reviewer', 'arm_populations', 'fact_groups', 'coverage_comparison', 'relation_selection_tradeoff', 'limitations'],
        'properties': {
            'root_id': {'enum': list(ROOTS)}, 'scope': {'enum': ['primary', 'repair', 'final']}, 'reviewer': {'const': REVIEWER},
            'arm_populations': {'type': 'object', 'required': list(ARMS), 'properties': {arm: {'type': 'array', 'uniqueItems': True, 'items': {'type': 'string'}} for arm in ARMS}, 'additionalProperties': False},
            'fact_groups': {'type': 'array', 'items': {'type': 'object', 'required': ['fact_id', 'members', 'fact_description', 'person_event_time_scope', 'answer_relation_boundaries', 'quotes'], 'properties': {
                'fact_id': {'type': 'string', 'minLength': 1}, 'members': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string'}},
                'fact_description': {'type': 'string', 'minLength': 1}, 'person_event_time_scope': {'type': 'string', 'minLength': 1},
                'answer_relation_boundaries': {'type': 'string', 'minLength': 1}, 'quotes': {'type': 'array', 'minItems': 1, 'items': quote},
            }, 'additionalProperties': False}},
            'coverage_comparison': {'type': 'string', 'minLength': 1}, 'relation_selection_tradeoff': {'type': 'string', 'minLength': 1},
            'limitations': {'type': 'string', 'minLength': 1},
        }, 'additionalProperties': False}
    return {'quality_review.schema.json': quality, 'dedup_clusters.schema.json': cluster, 'post_quality_review.schema.json': post, 'overlap_review.schema.json': overlap, 'shared_fact_coverage.schema.json': coverage}


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
    for name in ('quality_review', 'dedup_clusters', 'post_quality_review', 'overlap_review', 'shared_fact_coverage'):
        target = run / 'reports' / (name + '.jsonl')
        if not target.exists():
            _write_jsonl(target, [])
    plan = """# Writer SC/MS quality guidance paired preparation

PREPARED_NOT_RUN. All service calls are zero. Both BASE_WRITER and SCMS_QUALITY_WRITER retain the same complete original six V3_OFF intents and full transcripts for roots 198480 and 122711. Source Miner judgments do not change. BASE primary must equal both historical frozen and actual WITH_CURRENT_INTENT requests. QUALITY differs only by one newline plus the frozen SC/MS guidance block; both use intent_guidance=True and contract_guided.

Both arms reuse execute_writer_batch and finalize_writer_candidates, original 2 SC + 1 MS + 1 ordering minima, one whole-batch replacing repair, original normalization, root-arm validator and Gold permutation. Existing explicit ordering rules are unchanged; mixed output length, ordering content and repair probability may be indirectly affected. No new semantic admission, Gold editing, partial batch salvage, or extra calls.

Future primary order: 198480 slot 1 BASE then QUALITY, slot 2 QUALITY then BASE, slot 3 BASE then QUALITY; 122711 slot 1 QUALITY then BASE, slot 2 BASE then QUALITY, slot 3 QUALITY then BASE. Repair immediately follows its primary when the original minima rule triggers. Twelve primary plus at most twelve repair, twenty-four logical calls total; failures count. Known completed identity may resume; unknown completion/payment, identity drift or capacity failure stops without resending or changing directories.

All parseable primary and repair rows remain in offline diagnosis, including extras, rejected batches and replaced primary. First expose candidate-owned content with complete transcript, hide arms/intent/Miner diagnosis; the reviewer is Codex offline diagnostic, not independent blind review or human certification. Freeze complete validated quality judgments and input views before exposing either arm's original contract. Then assess both arms against every source FAIL dimension, descriptive relation choice, authored semantic dedup, finite history overlap, cross-arm overlap, discarded-to-retained relations, and shared factual coverage. Quality never triggers generation or changes production admission.

SC and MS and their combination are reported separately for primary/repair/final. Ordering is separate, including whole-batch effects. Principal yield is final accepted quality-PASS SC/MS / three scheduled slots per root-arm, retaining failed slots. Compare QUALITY minus BASE by root, then average two roots equally. Dedup requires explicit complete semantic partitions including singletons. Zero structure denominator is undefined, empty preparation is NOT_RUN. Full arm tokens include all calls, discarded primary, repairs, failed/rejected batches and ordering; no invented SC/MS token fraction. Missing usage means incomplete cost. Monetary cost remains unknown without evidence.

Shared-fact coverage describes overlapping facts, event/person/time/answer relations and shifts toward direct lookup; question clusters are not independent knowledge points. No Solver-based difficulty or training claims. Original history_only remains a finite stem/fingerprint pool, never expanded with prior Writer QA or placed in Writer prompts. Incomplete judgments remain precise pending items; COMPLETED requires validated quality, post, all dedup populations, all finite history and required pairs, and shared-fact coverage. This preparation fabricates no QA or future result.

Future live requires subsequent explicit authorization. Stop after preparation; do not call Writer, Miner, Free, Solver, Planner, Realizer, Memory, training or review APIs, implement v4/P2-2, fix ordering or expand roots.
"""
    plan_path = run / 'EXPERIMENT_PLAN.md'
    if not plan_path.exists() or plan_path.read_text() != plan:
        plan_path.write_text(plan)
    _write(run / 'policy_snapshot/review_policy_diff.json', {
        'source_version': 'h4_r02_writer_intent_ablation_v1_review_v1', 'version': VERSION,
        'unchanged': ['eight dimensions and applicability', 'all-option exact transcript review', 'quality aggregation', 'production admission separation', 'semantic dedup standard', 'finite history', 'complete-call token costs'],
        'changed': ['BASE_WRITER and SCMS_QUALITY_WRITER both retain intent', 'QUALITY minus BASE, roots equal-weight', 'SC and MS separate alongside combined at every stage', 'both-arm intent alignment and source-error checks after quality freeze', 'explicit shared-fact/selection and discarded-to-retained review formats', 'validated dynamic COMPLETED status with exact pending requirements'],
        'new_quality_criteria': False, 'new_production_admission': False,
    })


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
                'contract_group_alignment_analysis': candidate.get('contract_group_alignment'),
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
    if not population and not chosen:
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
                for category, types in (('single_choice', ('single_choice',)), ('multi_select', ('multi_select',)), ('sc_ms', ('single_choice', 'multi_select')), ('ordering', ('ordering',))):
                    subset = [i for i in selected if i['candidate'].get('question_type') in types]
                    labels = Counter(judgments.get(i['view_id'], 'PENDING_REVIEW') for i in subset)
                    denominator = len(subset)
                    layers[stage][category] = {'candidate_rows': denominator,
                        'normalizer_structural_valid_rows': sum(i['program_schema_valid'] for i in subset),
                        'minima_batch_accepted_rows': sum(i['program_schema_valid'] and i['minima_batch_accepted'] for i in subset),
                        'quality': {j: labels[j] for j in (*JUDGMENTS, 'PENDING_REVIEW')},
                        'quality_pass_fraction': labels['PASS'] / denominator if denominator else None,
                        'fraction_status': 'NOT_RUN' if not count['logical_calls'] else 'UNDEFINED_ZERO_DENOMINATOR' if not denominator else 'PENDING_REVIEW' if labels['PENDING_REVIEW'] else 'OBSERVED',
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
                'discarded_sc_ms': [{'view_id': i['view_id'], 'stage': i['stage'], 'quality': judgments.get(i['view_id'], 'PENDING_REVIEW'),
                    'replaced_primary': i['replaced_primary'], 'batch_accepted': i['minima_batch_accepted'],
                    'reason_to_check': next((r['minimum_acceptance'] for r in calls if r['slot_id'] == i['slot_id'] and r['stage'] == i['stage']), {})}
                    for i in own if i['candidate'].get('question_type') in ('single_choice', 'multi_select') and not i['final_validator_accepted']],
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
            'quality_pass_fraction_weighting': 'pooled final candidate ratio, not the equal-root macro rate',
            'root_macro_quality_pass_fraction': sum(g['layers']['final']['sc_ms']['quality_pass_fraction'] for g in selected) / 2
                if quality_frozen and all(g['layers']['final']['sc_ms']['quality_pass_fraction'] is not None for g in selected) else None,
            'layers': {stage: {category: {
                'candidate_rows': sum(g['layers'][stage][category]['candidate_rows'] for g in selected),
                'quality': {j: sum(g['layers'][stage][category]['quality'][j] for g in selected) for j in (*JUDGMENTS, 'PENDING_REVIEW')},
                'pooled_quality_pass_fraction': sum(g['layers'][stage][category]['quality']['PASS'] for g in selected) / sum(g['layers'][stage][category]['candidate_rows'] for g in selected)
                    if sum(g['layers'][stage][category]['candidate_rows'] for g in selected) else None,
                'root_macro_quality_pass_fraction': sum(g['layers'][stage][category]['quality_pass_fraction'] for g in selected) / 2
                    if all(g['layers'][stage][category]['quality_pass_fraction'] is not None for g in selected) else None,
            } for category in ('single_choice', 'multi_select', 'sc_ms', 'ordering')} for stage in ('primary', 'repair', 'final')},
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
        by_type = {}
        for category in ('single_choice', 'multi_select', 'sc_ms'):
            base_count = a['layers']['final'][category]['quality']['PASS'] if a['yield_status'] == 'QUALITY_FROZEN' else None
            quality_count = b['layers']['final'][category]['quality']['PASS'] if b['yield_status'] == 'QUALITY_FROZEN' else None
            by_type[category] = {'BASE_count': base_count, 'QUALITY_count': quality_count,
                'QUALITY_minus_BASE_count': quality_count - base_count if base_count is not None and quality_count is not None else None,
                'QUALITY_minus_BASE_per_slot': (quality_count - base_count) / 3 if base_count is not None and quality_count is not None else None}
        pairs.append({'root_id': root, 'BASE_effective_yield': av, 'QUALITY_effective_yield': bv,
                      'QUALITY_minus_BASE': bv - av if av is not None and bv is not None else None,
                      'QUALITY_minus_BASE_dedup': bd - ad if ad is not None and bd is not None else None,
                      'by_type': by_type})
    def mean(key):
        values = [p[key] for p in pairs]
        return sum(values) / 2 if all(v is not None for v in values) else None
    return {'groups': groups, 'arms': arms, 'root_paired': pairs, 'equal_root_weight_mean_delta': mean('QUALITY_minus_BASE'),
            'equal_root_weight_mean_dedup_delta': mean('QUALITY_minus_BASE_dedup'),
            'difference_direction': 'SCMS_QUALITY_WRITER minus BASE_WRITER',
            'equal_root_weight_by_type': {category: {
                metric: sum(p['by_type'][category][metric] for p in pairs) / 2 if all(p['by_type'][category][metric] is not None for p in pairs) else None
                for metric in ('QUALITY_minus_BASE_count', 'QUALITY_minus_BASE_per_slot')}
                for category in ('single_choice', 'multi_select', 'sc_ms')},
            'statistical_unit': 'root; equal weights; slot is call tracking only', 'planned_slots_per_arm': 6,
            'semantic_clustering_is_authored_only': True}


def _source_context(run, item):
    base = run / 'inputs/shared' / item['root_id']
    contracts = _read(base / 'reasoning_contracts.json', [])
    if isinstance(contracts, Mapping):
        contracts = contracts.get('contracts', [])
    contract = next((c for c in contracts if c.get('intent_id') == item['slot_id']), None)
    findings = _read(base / 'source_intent_diagnostics.json', [])
    if isinstance(findings, Mapping):
        findings = findings.get('records', [])
    diagnostic = next((d for d in findings if d.get('intent_id') == item['slot_id'] or d.get('declared_intent_id') == item['slot_id']), None)
    source_errors = sorted(name for name, finding in (diagnostic or {}).get('dimensions', {}).items() if finding.get('judgment') == 'FAIL')
    return contract, diagnostic, source_errors


def _pass_population(items, judgments, root, scope, arm=None):
    return {i['view_id'] for i in items if i['root_id'] == root and (arm is None or i['arm_id'] == arm)
        and (i['final_validator_accepted'] if scope == 'final' else i['stage'] == scope)
        and i['candidate'].get('question_type') in ('single_choice', 'multi_select') and judgments.get(i['view_id']) == 'PASS'}


def diagnostic_requirements(run, items, judgments, histories):
    """Enumerate identities, never fabricate diagnostic judgments."""
    by_view = {i['view_id']: i for i in items}
    history_keys = {row['root_id']: 'history:' + row['root_id'] + ':' + str(row['sha256']) for row in histories}
    overlap = [('FROZEN_HISTORY', i['view_id'], history_keys[i['root_id']]) for i in items]
    for root in ROOTS:
        base = _pass_population(items, judgments, root, 'final', ARMS[0])
        quality = _pass_population(items, judgments, root, 'final', ARMS[1])
        overlap.extend(('CROSS_ARM', a, b) for a in sorted(base) for b in sorted(quality))
        for arm in ARMS:
            final = _pass_population(items, judgments, root, 'final', arm)
            discarded = {v for v, i in by_view.items() if i['root_id'] == root and i['arm_id'] == arm
                and not i['final_validator_accepted'] and i['candidate'].get('question_type') in ('single_choice', 'multi_select') and judgments.get(v) == 'PASS'}
            overlap.extend(('DISCARDED_TO_FINAL', a, b) for a in sorted(discarded) for b in sorted(final))
    return {'post_view_ids': sorted(by_view), 'overlap_keys': [list(key) for key in overlap],
        'coverage': [{'root_id': root, 'scope': scope, 'arm_populations': {arm: sorted(_pass_population(items, judgments, root, scope, arm)) for arm in ARMS}}
                     for root in ROOTS for scope in ('primary', 'repair', 'final')],
        'dedup_scopes': ['primary_sc_ms', 'repair_sc_ms', 'final_sc_ms', 'primary_ordering', 'repair_ordering', 'final_ordering'],
        'history_identity_rule': 'Each FROZEN_HISTORY row covers the complete finite pool identified by history:<root>:<sha256>; absent options/Gold allow only an overlap hint or UNDETERMINED.'}


def validate_post_diagnostics(run, items, judgments, clusters, post, overlap, coverage, transcripts, histories):
    """Return exact invalid and pending diagnostics independently of QA admission."""
    errors, pending, post_views = [], [], []
    by_view = {i['view_id']: i for i in items}
    requirements = diagnostic_requirements(run, items, judgments, histories)
    expected_source = {}
    for item in items:
        contract, diagnostic, source_errors = _source_context(run, item)
        expected_source[item['view_id']] = source_errors
        if contract is None:
            pending.append('source_contract_missing:' + item['view_id'])
        if diagnostic is None:
            pending.append('source_diagnostic_missing:' + item['view_id'])
        post_views.append({'view_id': item['view_id'], 'analysis_identity': item['analysis_identity'],
            'quality_judgment': judgments[item['view_id']], 'original_contract': contract, 'original_miner_diagnostic': diagnostic,
            'miner_error_provenance_status': 'AVAILABLE' if diagnostic is not None else 'SOURCE_DIAGNOSTIC_SIDECAR_NOT_AVAILABLE',
            'intent_guidance': True, 'intent_alignment': 'TO_REVIEW', 'required_source_error_keys': source_errors,
            'contract_group_alignment_analysis': item['contract_group_alignment_analysis'], 'candidate': item['candidate'], 'reviewer': REVIEWER})
    _write_jsonl(run / 'reports/post_quality_views.jsonl', post_views)
    _write(run / 'reports/diagnostic_requirements.json', requirements)
    valid = {}
    for name, rows in (('dedup_clusters', clusters), ('post_quality_review', post), ('overlap_review', overlap), ('shared_fact_coverage', coverage)):
        valid[name] = []
        for row in rows:
            row_errors = _schema_errors(row, _schemas()[name + '.schema.json'], name)
            errors.extend(row_errors)
            if not row_errors:
                valid[name].append(row)
    seen_post = set()
    for row in valid['post_quality_review']:
        view = row['view_id']
        if view not in by_view:
            errors.append('unknown_post_quality_view:' + view)
            continue
        if view in seen_post:
            errors.append('duplicate_post_quality_review:' + view)
        seen_post.add(view)
        item = by_view[view]
        turns = {t['turn_ref']: t['text'] for t in transcripts[item['root_id']].get('turns', [])}
        _check_quotes(row['quotes'], turns, errors, 'post_quality:' + view)
        if item['content_reviewable'] and not row['quotes']:
            errors.append('post_quality_exact_evidence_required:' + view)
        keys = [f['source_error'] for f in row['error_propagation']]
        if len(keys) != len(set(keys)) or set(keys) - set(expected_source[view]):
            errors.append('unknown_or_duplicate_source_error:' + view)
        for key in set(expected_source[view]) - set(keys):
            pending.append('source_error_unreviewed:' + view + ':' + key)
        for finding in row['error_propagation']:
            _check_quotes(finding['quotes'], turns, errors, 'source_error:' + view + ':' + finding['source_error'])
            if item['content_reviewable'] and not finding['quotes']:
                errors.append('source_error_exact_evidence_required:' + view + ':' + finding['source_error'])
    pending.extend('post_quality_unreviewed:' + view for view in set(by_view) - seen_post)

    for arm in ARMS:
        for root in ROOTS:
            for scope in requirements['dedup_scopes']:
                population = dedup_count(items, judgments, valid['dedup_clusters'], arm, root, scope)
                if population['count'] is None:
                    pending.append('dedup_incomplete:' + ':'.join((arm, root, scope)))

    expected_overlap = {tuple(k) for k in requirements['overlap_keys']}
    seen_overlap = set()
    history_by_root = {h['root_id']: h for h in histories}
    for row in valid['overlap_review']:
        key = (row['comparison_kind'], row['view_id'], row['comparison_identity'])
        if key[0] == 'CROSS_ARM' and key not in expected_overlap:
            key = (key[0], key[2], key[1])
        if key not in expected_overlap:
            errors.append('unexpected_overlap_identity:' + repr(key))
            continue
        if key in seen_overlap:
            errors.append('duplicate_overlap_identity:' + repr(key))
        seen_overlap.add(key)
        if key[0] == 'FROZEN_HISTORY':
            history = history_by_root[by_view[key[1]]['root_id']]
            if history['record_count'] and (history['records_with_options'] < history['record_count'] or history['records_with_gold'] < history['record_count']):
                if row['judgment'] not in ('STEM_OR_FINGERPRINT_HINT_ONLY', 'UNDETERMINED'):
                    errors.append('history_without_options_gold_overclaim:' + key[1])
    pending.extend('overlap_unreviewed:' + ':'.join(key) for key in expected_overlap - seen_overlap)

    expected_coverage = {(row['root_id'], row['scope']): row for row in requirements['coverage']}
    seen_coverage = set()
    for row in valid['shared_fact_coverage']:
        key = (row['root_id'], row['scope'])
        if key in seen_coverage:
            errors.append('duplicate_shared_fact_coverage:' + ':'.join(key))
        seen_coverage.add(key)
        expected = expected_coverage[key]
        if any(set(row['arm_populations'][arm]) != set(expected['arm_populations'][arm]) for arm in ARMS):
            pending.append('shared_fact_population_incomplete:' + ':'.join(key))
        population = {v for arm in ARMS for v in expected['arm_populations'][arm]}
        members = {v for group in row['fact_groups'] for v in group['members']}
        if members != population:
            pending.append('shared_fact_members_incomplete:' + ':'.join(key))
        if len({g['fact_id'] for g in row['fact_groups']}) != len(row['fact_groups']):
            errors.append('duplicate_fact_id:' + ':'.join(key))
        turns = {t['turn_ref']: t['text'] for t in transcripts[row['root_id']].get('turns', [])}
        for group in row['fact_groups']:
            _check_quotes(group['quotes'], turns, errors, 'shared_fact:' + ':'.join(key) + ':' + group['fact_id'])
    pending.extend('shared_fact_unreviewed:' + ':'.join(key) for key in set(expected_coverage) - seen_coverage)
    return valid['dedup_clusters'], sorted(set(errors)), sorted(set(pending))


def report(run_dir):
    """Prepare empty formats or diagnose local runtime artifacts; never call a service."""
    run = Path(run_dir)
    prepare_review_assets(run)
    items, requests, errors = collect_candidates(run)
    transcripts = {root: _read(run / 'inputs/shared' / root / 'expanded_dialogue_clean.json', {}) for root in ROOTS}
    views = sorted([{'view_id': i['view_id'], 'candidate': i['candidate'], 'complete_transcript': transcripts[i['root_id']],
        'content_reviewable': i['content_reviewable'], 'reviewer': REVIEWER} for i in items], key=lambda row: row['view_id'])
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
    runtime = _read(run / 'reports/runtime_summary.json', {})
    runtime_complete = (runtime.get('status') == 'COMPLETED' and runtime.get('backend_mode') == 'live'
        and (run / 'reports/runtime_final_candidates.jsonl').is_file() and primary_keys == expected_keys)
    ledger_keys = {(g['arm_id'], g['root_id'], r['stage'], r['request_id']) for g in accounting['groups'] for r in g['requests']}
    recorded_keys = {(r['arm_id'], r['root_id'], r['stage'], r['request_id']) for r in requests}
    if runtime_complete and ledger_keys != recorded_keys:
        errors.append('completed_runtime_stage_and_ledger_identity_mismatch')
    if runtime_complete and (accounting['primary_calls'] != 12 or accounting['repair_calls'] > 12 or accounting['logical_calls'] > 24):
        errors.append('completed_runtime_call_budget_or_primary_coverage_invalid')
    freeze_path = run / 'audit/quality_review_freeze.json'
    freeze_evidence = {'quality_review_sha256': _sha(run / 'reports/quality_review.jsonl'),
        'view_digest': _digest(views), 'mapping_digest': _digest(mapping),
        'runtime_stage_records_sha256': _sha(run / 'reports/runtime_stage_records.jsonl') if (run / 'reports/runtime_stage_records.jsonl').is_file() else None,
        'runtime_final_candidates_sha256': _sha(run / 'reports/runtime_final_candidates.jsonl') if (run / 'reports/runtime_final_candidates.jsonl').is_file() else None}
    freeze = _read(freeze_path)
    clusters = _jsonl(run / 'reports/dedup_clusters.jsonl')
    post = _jsonl(run / 'reports/post_quality_review.jsonl')
    overlap = _jsonl(run / 'reports/overlap_review.jsonl')
    coverage = _jsonl(run / 'reports/shared_fact_coverage.jsonl')
    if freeze is None and (clusters or post or overlap or coverage):
        errors.append('post_quality_judgments_before_valid_quality_freeze')
    if freeze is not None and freeze.get('evidence') != freeze_evidence:
        errors.append('frozen_quality_inputs_or_judgments_changed')
    all_reviewed = all(judgments.get(i['view_id']) != 'PENDING_REVIEW' for i in items)
    if freeze is None and runtime_complete and accounting['logical_calls'] and all_reviewed and not errors:
        freeze = {'reviewer': REVIEWER, 'status': 'QUALITY_FROZEN', 'evidence': freeze_evidence,
            'primary_planned_slots': 12, 'candidate_rows': len(items), 'reviewed_rows': len(authored),
            'not_evaluated_structure_rows': sum(j == 'NOT_EVALUATED_STRUCTURE' for j in judgments.values())}
        _write(freeze_path, freeze)
    quality_frozen = freeze is not None and freeze.get('evidence') == freeze_evidence and not errors
    histories = history_inventory(run)
    pending, valid_clusters = [], []
    if (clusters or post or overlap or coverage) and not quality_frozen:
        errors.append('post_quality_judgments_before_valid_quality_freeze')
    if quality_frozen:
        valid_clusters, post_errors, pending = validate_post_diagnostics(run, items, judgments, clusters, post, overlap, coverage, transcripts, histories)
        errors.extend(post_errors)
    elif accounting['logical_calls']:
        if not runtime_complete:
            pending.append('runtime_not_complete')
        pending.extend('primary_slot_unrecorded:' + ':'.join(key) for key in expected_keys - primary_keys)
        pending.extend('quality_unreviewed:' + view for view, judgment in judgments.items() if judgment == 'PENDING_REVIEW')
        pending.append('quality_not_frozen')
    summary = summarize(items, requests, judgments, valid_clusters, accounting, quality_frozen)
    status = ('BLOCKED' if errors else 'PREPARED_NOT_RUN' if not accounting['logical_calls'] else
        'COMPLETED' if quality_frozen and not pending else
        'QUALITY_FROZEN_DIAGNOSTICS_PENDING' if quality_frozen else 'QUALITY_REVIEW_PENDING')
    result = {'status': status, 'reviewer': REVIEWER, 'service_calls_by_report': 0,
        'LIVE_STATUS': 'NOT_RUN' if not accounting['logical_calls'] else 'COMPLETED' if runtime_complete else 'PARTIAL_OR_STOPPED',
        'ACTUAL_EXTERNAL_LOGICAL_CALLS': accounting['logical_calls'], 'quality_frozen': quality_frozen,
        'quality_judgment_rows': len(authored), 'candidate_rows': len(items),
        'post_quality_review_rows': len(post), 'dedup_cluster_rows': len(clusters), 'overlap_review_rows': len(overlap), 'shared_fact_coverage_rows': len(coverage),
        'pending_requirements': sorted(set(pending)), 'errors': sorted(set(errors)), 'accounting': accounting, 'summary': summary, 'history_pools': histories,
        'synthetic_as_actual_results': False, 'production_admission_changed': False, 'intent_guidance_both_arms': True,
        'interpretation_limits': POLICY['interpretation']}
    for name, value in (('audit/report_accounting.json', accounting), ('reports/history_pool_inventory.json', histories),
                        ('reports/paired_comparison.json', summary), ('reports/offline_diagnostic_status.json', result)):
        _write(run / name, value)
    _write(run / 'reports/diagnostic_completion.json', {'status': status, 'quality_frozen': quality_frozen,
        'pending_requirements': result['pending_requirements'], 'errors': result['errors'],
        'review_artifact_sha256': {name: _sha(run / 'reports' / (name + '.jsonl')) for name in ('quality_review', 'post_quality_review', 'dedup_clusters', 'overlap_review', 'shared_fact_coverage')},
        'not_a_claim_of_independent_review_or_model_internal_self_checks': True})
    lines = ['# Writer SC/MS quality guidance: offline diagnostic status', '',
        'Status: ' + status + '. Reporter service calls: 0.', '',
        'NOT_RUN: no generated QA or measured treatment effect.' if not accounting['logical_calls'] else 'Codex authored offline diagnosis is distinct from original production admission.', '',
        'Both arms retain the same complete intent. Only the frozen SC/MS guidance block differs in generation.',
        'First freeze arm-hidden candidate quality; then reveal both contracts and analyze source errors, semantic duplicates, finite history, shared facts and relation selection.',
        'SC, MS and SC/MS combined are separate in primary/repair/final layers. Ordering and entire-batch losses are separate.',
        'QUALITY minus BASE is paired by root; two roots have equal weights. Every root-arm retains three planned slots.',
        'Entire actual arm token cost includes failures, replaced primary, repairs and ordering. Missing usage and zero yield remain explicitly undefined.', '',
        f"Candidate rows: {len(items)}; quality rows: {len(authored)}; quality frozen: {quality_frozen}.",
        'Pending requirements: ' + json.dumps(result['pending_requirements'], ensure_ascii=False),
        'Errors: ' + json.dumps(result['errors'], ensure_ascii=False), '',
        'See paired_comparison.json, diagnostic_requirements.json (after quality freeze), and history_pool_inventory.json. No automatic model call, resampling, Miner score change or further experiment.', '']
    (run / 'reports/offline_diagnostic_report.md').write_text('\n'.join(lines))
    return result
