#!/usr/bin/env python3
"""Validate frozen inputs and select one representative per reviewed family.

No authoring, inference, rescoring, source upgrade, export, or training occurs.
Default execution exclusive-creates FINAL_SELECTED_ITEMS.json after every check.
--dry-run performs the same validation/selection in memory and writes no files.

CROSS_BATCH_FAMILY_REVIEW.json has schema cross_batch_family_review_v1:
  status: COMPLETE_INDEPENDENT_SCORE_FREE_COLLECTION_REVIEW
  scope: {all_candidate_ids: [...], solver_scores_read_or_used: false,
          model_calls_executed: 0, training_executed: false}
  input_bindings: [{path: absolute_path, sha256: file_sha256}, ...]
    Bind all nine final FAMILY_AUDITs, FINAL_SELECTION_RULES.json, the run-root
    FINALIZATION_REQUIREMENTS.json, and every original ITEMS_PRIVATE.json.
    The independent reviewer must not read solver, quality, layer-decision, or
    feedback files. Selection validates those separate execution inputs itself.
  candidate_family_map: [{candidate_id, family_id, reporting_structural_cluster}]
  duplicate_family_resolutions: [{family_id, candidate_ids, score_free: true,
    reason: nonempty_text, disposition: equivalent_variants|score_free_resolution,
    information_need_equivalent: true, operation_equivalent: true,
    preferred_candidate_ids: [[first_priority_equivalent_ids], [next_ids], ...]}]
    Every family with multiple distinct candidates needs one explicit review,
    including families whose members later prove ineligible. Equivalence requires
    both equivalence attestations and an explanatory reason. A score-free
    resolution supplies ordered groups partitioning ALL members, or [] to retain
    none. Select from the first group containing eligible candidates, with the
    frozen hash tie within that group. A same-family label alone is insufficient.
  concentration_review: {operation_clusters_reviewed: true,
    question_types_reviewed: true, source_and_root_correlation_reviewed: true,
    inherited_source_correlation_reviewed: true,
    fine_families_independent: false, limitations: [nonempty_text, ...]}

Different local structural_operation labels remain in the decision record;
reporting_structural_cluster supplies the independent global reporting label.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
OUTPUT = HERE / 'FINAL_SELECTED_ITEMS.json'
RULES = HERE / 'FINAL_SELECTION_RULES.json'
REQUIREMENTS = RUN / 'FINALIZATION_REQUIREMENTS.json'
GLOBAL_REVIEW = HERE / 'CROSS_BATCH_FAMILY_REVIEW.json'
FAMILY_AUDITS = [HERE / f'BATCH{i:02d}_FAMILY_AUDIT.json' for i in range(1, 10)]
TIE_RULE = 'sha256(UTF-8(family_id + U+0000 + candidate_id)), ascending'


class SelectionError(Exception):
    """An incomplete, changed, or inconsistent frozen input prevents selection."""


def require(condition, message):
    if not condition:
        raise SelectionError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def object_sha256(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def tie_value(family_id, candidate_id):
    return hashlib.sha256((family_id + '\0' + candidate_id).encode('utf-8')).hexdigest()


def unique_rows(rows, field, label):
    require(isinstance(rows, list), label + ': expected list')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get(field), str) and row[field],
                label + ': missing ' + field)
        require(row[field] not in result, label + ': duplicate ' + row[field])
        result[row[field]] = row
    return result


class FrozenInputs:
    """Read once, validate every declared binding, and rehash before publication."""

    def __init__(self):
        self.bindings = {}
        self.values = {}
        self.raw = {}

    def binding(self, path, refresh=False):
        path = Path(path).resolve()
        if str(path) in self.bindings and not refresh:
            return self.bindings[str(path)]
        require(path.is_file(), 'MISSING_FROZEN_INPUT: ' + str(path))
        raw = path.read_bytes()
        value = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
        old = self.bindings.get(str(path))
        require(old is None or old == value, 'INPUT_CHANGED_WHILE_READING: ' + str(path))
        self.bindings[str(path)] = value
        if old is None:
            self.raw[str(path)] = raw
        return value

    def check_binding(self, value):
        path = value.get('path')
        require(isinstance(path, str) and Path(path).is_absolute(), 'BINDING_PATH_NOT_ABSOLUTE: ' + str(path))
        actual = self.binding(path)
        require(isinstance(value.get('sha256'), str) and actual['sha256'] == value['sha256'],
                'FROZEN_HASH_MISMATCH: ' + path)
        if value.get('expected_sha256') is not None:
            require(actual['sha256'] == value['expected_sha256'], 'EXPECTED_HASH_MISMATCH: ' + path)
        for key in ('size_bytes', 'bytes', 'byte_count'):
            if key in value:
                require(actual['size_bytes'] == value[key], 'FROZEN_SIZE_MISMATCH: ' + path)
        return actual

    def check_embedded_bindings(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value:
                self.check_binding(value)
            for child in value.values():
                self.check_embedded_bindings(child)
        elif isinstance(value, list):
            for child in value:
                self.check_embedded_bindings(child)

    def read(self, path, embedded=True):
        path = Path(path).resolve()
        key = str(path)
        self.binding(path)
        if key not in self.values:
            try:
                self.values[key] = json.loads(self.raw[key].decode('utf-8'))
            except (ValueError, UnicodeError) as exc:
                raise SelectionError('INVALID_FROZEN_JSON: ' + key) from exc
        value = self.values[key]
        if embedded:
            self.check_embedded_bindings(value)
        return value

    def unchanged(self):
        for key, binding in list(self.bindings.items()):
            self.binding(key, refresh=True)


def prerequisite_paths():
    return [RUN / 'ROOT_COVERAGE.json', RUN / 'FINAL_EXECUTION_AUDIT.json',
            RULES, REQUIREMENTS, GLOBAL_REVIEW,
            *[RUN / f'batch_{i:02d}/COMPLETED_BATCH.json' for i in range(1, 10)],
            *[RUN / f'batch_{i:02d}/CANDIDATE_LAYER_DECISIONS.json' for i in range(1, 10)],
            *FAMILY_AUDITS]


def validate_gates(inputs):
    missing = [str(path.relative_to(RUN)) for path in prerequisite_paths() if not path.is_file()]
    coverage = inputs.read(RUN / 'ROOT_COVERAGE.json') if (RUN / 'ROOT_COVERAGE.json').is_file() else {}
    problems = []
    if coverage.get('author_opportunities_completed') != 136 or coverage.get('completed_both_arms') != 68:
        problems.append('ALL_136_AUTHOR_OPPORTUNITIES_NOT_FROZEN: '
                        + str(coverage.get('author_opportunities_completed')) + '/136')
    if missing:
        problems.append('MISSING_FINAL_PREREQUISITES: ' + ', '.join(missing))
    require(not problems, '\n'.join(problems))
    rules = inputs.read(RULES)
    inputs.read(REQUIREMENTS)
    require(rules.get('same_family_maximum') == 1 and rules.get('training_started') is False
            and rules.get('final_review_required_before_export') is True,
            'FINAL_SELECTION_RULES_CONTRACT_CHANGED')
    require('SHA-256' in rules.get('same_family_tie_rule', '')
            and 'U+0000' in rules['same_family_tie_rule'], 'FROZEN_HASH_TIE_RULE_MISSING')
    require(coverage.get('required_roots') == 68 and coverage.get('required_author_opportunities') == 136,
            'COVERAGE_REQUIREMENTS_CHANGED')
    roots = unique_rows(coverage.get('roots'), 'root_id', 'ROOT_COVERAGE')
    require(len(roots) == 68, 'EXPECTED_68_UNIQUE_ROOTS')
    source_rows = unique_rows(inputs.read(RUN / 'SOURCE_FREEZE_ALL68.json')['sources'], 'root_id', 'SOURCE_FREEZE_ALL68')
    require(set(source_rows) == set(roots), 'FROZEN_TRAIN_ROOT_COVERAGE_DRIFT')
    for index in range(1, 10):
        freeze = RUN / ('SOURCE_FREEZE.json' if index == 1 else f'batch_{index:02d}/SOURCE_FREEZE.json')
        rows = unique_rows(inputs.read(freeze)['sources'], 'root_id', str(freeze))
        require(set(rows) == {root for root, row in roots.items() if row['batch_index'] == index},
                'BATCH_FROZEN_ROOT_COVERAGE_DRIFT: ' + str(index))
        require(all(row['sha256'] == source_rows[root]['sha256'] for root, row in rows.items()),
                'BATCH_ALL68_SOURCE_FREEZE_HASH_DRIFT: ' + str(index))
    first_drafts = 0
    for root, row in roots.items():
        require(row.get('both_author_opportunities_completed') is True and set(row.get('arms', {})) == {'F', 'G'},
                'INCOMPLETE_AUTHOR_COVERAGE: ' + root)
        require(type(row.get('batch_index')) is int and 1 <= row['batch_index'] <= 9, 'BAD_BATCH_INDEX: ' + root)
        for arm, opportunity in row['arms'].items():
            require(opportunity.get('state') in ('DRAFTS_FROZEN', 'NO_SAFE_QA_FROZEN'),
                    'AUTHOR_OPPORTUNITY_NOT_FROZEN: ' + root + ':' + arm)
            expected = RUN / f'batch_{row["batch_index"]:02d}' / root / arm / 'author/CANDIDATES_FIRST_DRAFT.json'
            require(Path(opportunity['path']).resolve() == expected, 'AUTHOR_FREEZE_PATH_DRIFT: ' + root + ':' + arm)
            inputs.check_binding(opportunity)
            draft = inputs.read(expected)
            author_input = inputs.read(expected.parent / 'AUTHOR_INPUT.json')
            candidates = draft.get('candidates')
            require(author_input.get('root_id') == root and author_input.get('arm') == arm
                    and Path(author_input.get('output_directory', '')).resolve() == expected.parent
                    and draft.get('root_id') == root and draft.get('arm', author_input['arm']) == arm
                    and isinstance(candidates, list) and len(candidates) <= 2,
                    'AUTHOR_FIRST_DRAFT_CONTRACT: ' + str(expected))
            require(opportunity.get('first_drafts') == len(candidates), 'COVERAGE_DRAFT_COUNT_DRIFT: ' + str(expected))
            require((opportunity['state'] == 'NO_SAFE_QA_FROZEN') == (len(candidates) == 0),
                    'NO_SAFE_QA_FREEZE_DRIFT: ' + str(expected))
            first_drafts += len(candidates)
    require(first_drafts <= 272, 'FIRST_DRAFT_CAP_EXCEEDED')
    closed = []
    for index in range(1, 10):
        value = inputs.read(RUN / f'batch_{index:02d}/COMPLETED_BATCH.json')
        require(value.get('status') == 'COMPLETE' and value.get('batch_index') == index
                and value.get('both_arms_frozen_and_first_review_complete') is True
                and value.get('duplicate_requests') == 0 and value.get('training_started') is False,
                'BATCH_NOT_CLOSED: ' + str(index))
        expected_roots = sum(row['batch_index'] == index for row in roots.values())
        require(value.get('root_count') == expected_roots, 'BATCH_ROOT_COUNT_DRIFT: ' + str(index))
        closed.append(value)
    audit = inputs.read(RUN / 'FINAL_EXECUTION_AUDIT.json')
    require(audit.get('status') == 'PASS' and audit.get('required_roots') == 68
            and audit.get('completed_author_opportunities') == 136
            and audit.get('actual_new_first_drafts') == first_drafts
            and audit.get('duplicate_requests') == 0 and audit.get('rescoring') is False
            and audit.get('training_started') is False, 'FINAL_EXECUTION_AUDIT_NOT_PASS')
    require(audit.get('coverage_binding', {}).get('path') == str(RUN / 'ROOT_COVERAGE.json'),
            'EXECUTION_AUDIT_COVERAGE_BINDING_MISSING')
    actual_closed = {str(RUN / f'batch_{i:02d}/COMPLETED_BATCH.json') for i in range(1, 10)}
    require({x['path'] for x in audit.get('batch_closure_bindings', [])} == actual_closed,
            'EXECUTION_AUDIT_DOES_NOT_BIND_ALL_NINE_CLOSURES')
    groups = audit.get('requests_by_group', {})
    require(groups.get('calibration') == 144 and groups.get('legacy') == 2448
            and groups.get('new') == sum(x['unique_completed_requests'] for x in closed)
            and groups['new'] <= 9792 and sum(groups.values()) <= 23004
            and audit.get('unique_requests') == sum(groups.values()), 'EXECUTION_BUDGET_OR_TOTAL_DRIFT')
    require(sum(x['first_drafts'] for x in closed) == first_drafts, 'CLOSED_FIRST_DRAFT_TOTAL_DRIFT')
    audited_component_bindings = {}
    for component in audit.get('components', []):
        for binding in component['bindings']:
            previous = audited_component_bindings.get(binding['path'])
            require(previous is None, 'REPEATED_EXECUTION_COMPONENT_BINDING: ' + binding['path'])
            audited_component_bindings[binding['path']] = binding['sha256']
    for batch in closed:
        for component in batch['components']:
            for binding in component['bindings']:
                require(audited_component_bindings.get(binding['path']) == binding['sha256'],
                        'CLOSED_COMPONENT_MISSING_FROM_FINAL_EXECUTION_AUDIT: ' + binding['path'])
    return roots, closed, audit


def quality_path(directory):
    path = directory / 'QUALITY_FINAL.json'
    for view in ('review_enum_interface_view', 'target_hash_interface_view',
                 'review_citation_interface_view', 'warning_policy_view'):
        candidate = directory / view / 'QUALITY_FINAL.json'
        if candidate.is_file():
            path = candidate
    return path


def item_source(inputs, item):
    path = item.get('source_path')
    require(isinstance(path, str) and Path(path).is_absolute(), 'ITEM_SOURCE_PATH_INVALID: ' + item['candidate_id'])
    inputs.check_binding({'path': path, 'sha256': item.get('source_sha256')})
    source = inputs.read(path)
    require(source.get('root_id') == item['root_id'] and isinstance(source.get('turns'), list)
            and source['turns'], 'SOURCE_ROOT_OR_TURNS_INVALID: ' + item['candidate_id'])
    turns = unique_rows(source['turns'], 'turn_ref', 'SOURCE:' + path)
    require(all(all(isinstance(t.get(key), str) and t[key].strip() for key in ('speaker', 'text'))
                for t in turns.values()), 'INVALID_SOURCE_TURN: ' + path)
    require(item.get('source_turn_count') == len(turns), 'SOURCE_TURN_COUNT_DRIFT: ' + item['candidate_id'])
    return source


def load_candidates(inputs, roots, closed, execution):
    items, provenance, qualities, layers, scores = {}, {}, {}, {}, {}
    original_item_paths = set()
    frozen_sources = {row['root_id']: row for row in inputs.read(RUN / 'SOURCE_FREEZE_ALL68.json')['sources']}
    for root, root_row in roots.items():
        index = root_row['batch_index']
        bound_paths = {x['path'] for x in closed[index - 1]['quality_and_item_bindings']}
        for arm in ('F', 'G'):
            base = RUN / f'batch_{index:02d}' / root / arm
            path = base / 'batch/ITEMS_PRIVATE.json'
            require(str(path) in bound_paths, 'CLOSED_BATCH_DOES_NOT_BIND_ORIGINAL_ITEMS: ' + str(path))
            original_item_paths.add(str(path))
            rows = unique_rows(inputs.read(path), 'candidate_id', str(path))
            draft = inputs.read(base / 'author/CANDIDATES_FIRST_DRAFT.json')
            require(set(rows) == {x['candidate_id'] for x in draft['candidates']}, 'DRAFT_ITEMS_ID_MISMATCH: ' + str(path))
            qp = quality_path(base / 'batch')
            require(str(qp) in bound_paths, 'CLOSED_BATCH_DOES_NOT_BIND_QUALITY: ' + str(qp))
            qs = unique_rows(inputs.read(qp)['items'], 'candidate_id', str(qp))
            require(set(qs) == set(rows), 'QUALITY_ITEMS_ID_MISMATCH: ' + str(qp))
            for cid, item in rows.items():
                require(cid not in items and item.get('root_id') == root and item.get('arm') == arm,
                        'DUPLICATE_OR_WRONG_ROOT_ARM_ITEM: ' + cid)
                item_author_path = Path(item.get('author_path', ''))
                if not item_author_path.is_absolute():
                    item_author_path = RUN / item_author_path
                require(item_author_path.resolve() == Path(root_row['arms'][arm]['path']).resolve()
                        and item.get('author_sha256') == root_row['arms'][arm]['sha256'], 'ITEM_AUTHOR_FREEZE_DRIFT: ' + cid)
                require(item.get('source_sha256') == frozen_sources[root]['sha256'], 'ITEM_ALL68_SOURCE_FREEZE_DRIFT: ' + cid)
                items[cid] = item
                provenance[cid] = {'origin': 'new_' + arm, 'batch_index': index,
                                   'original_items_binding': inputs.binding(path), 'original_item_sha256': object_sha256(item)}
                qualities[cid] = qs[cid]
                item_source(inputs, item)
    # The frozen B1 public-scope HOLD is applied separately by the closure tool.
    for index in range(1, 10):
        hold_path = RUN / f'batch_{index:02d}/PUBLIC_QUERY_SCOPE_HOLD_APPLICATION.json'
        if hold_path.is_file():
            require(str(hold_path) in {b['path'] for b in closed[index - 1]['quality_and_item_bindings']},
                    'UNBOUND_CONTENT_HOLD_APPLICATION: ' + str(hold_path))
            addendum = inputs.read(hold_path)
            originals = unique_rows(inputs.read(addendum['original_quality_gate']['path'])['items'], 'candidate_id', 'HOLD_ORIGINAL_GATE')
            for row in addendum['items']:
                require(row == originals[row['candidate_id']] and row.get('content_pass') is False,
                        'CONTENT_HOLD_PROJECTION_DRIFT')
                qualities[row['candidate_id']] = row
        lp = RUN / f'batch_{index:02d}/CANDIDATE_LAYER_DECISIONS.json'
        layer = inputs.read(lp)
        require(layer.get('batch_index') == index
                and layer.get('quality_training_value_source_qualification_collection_separated') is True
                and layer.get('all_original_items_answers_and_scores_preserved') is True
                and layer.get('no_rescoring_or_new_calls') is True, 'LAYER_CONTRACT_DRIFT: ' + str(lp))
        rows = unique_rows(layer['items'], 'candidate_id', str(lp))
        expected = {cid for cid, p in provenance.items() if p['batch_index'] == index}
        require(set(rows) == expected, 'BATCH_LAYER_ID_COVERAGE_DRIFT: ' + str(index))
        layers.update(rows)
    legacy_freeze = inputs.read(RUN / 'LEGACY_REPRESENTATIVES_FREEZE.json')
    require(legacy_freeze.get('candidate_count') == 68 and legacy_freeze.get('selected_before_current_calls') is True
            and legacy_freeze.get('selection_uses_current_or_historical_scores') is False, 'LEGACY_PREFREEZE_CONTRACT_DRIFT')
    legacy_ids = legacy_freeze['candidate_ids_numeric_sorted']
    require(len(legacy_ids) == len(set(legacy_ids)) == 68, 'LEGACY_REPRESENTATIVE_ID_COUNT')
    pool = inputs.read(legacy_freeze['input_pool']['path'])
    pool_items = unique_rows(pool['questions'], 'candidate_id', 'LEGACY_PREFROZEN_POOL')
    require(set(pool_items) == set(legacy_ids), 'LEGACY_PREFROZEN_POOL_DRIFT')
    triage_path = Path(legacy_freeze['input_pool']['path']).parent / 'CANDIDATE_TRIAGE.json'
    triage = unique_rows(inputs.read(triage_path)['candidates'], 'candidate_id', 'LEGACY_FROZEN_TRIAGE')
    legacy_layers = inputs.read(RUN / 'legacy_evaluation/CANDIDATE_LAYER_DECISIONS.json')
    require(legacy_layers.get('no_context_or_answer_only_confer_eligibility') is False, 'LEGACY_DIAGNOSTIC_ADMISSION_DRIFT')
    legacy_layer_map = unique_rows(legacy_layers['items'], 'candidate_id', 'LEGACY_LAYERS')
    require(set(legacy_layer_map) == set(legacy_ids), 'LEGACY_LAYER_ID_COVERAGE')
    for part in sorted((RUN / 'legacy_evaluation').glob('part_*')):
        path = part / 'ITEMS_PRIVATE.json'
        original_item_paths.add(str(path))
        rows = unique_rows(inputs.read(path), 'candidate_id', str(path))
        qs = unique_rows(inputs.read(part / 'QUALITY_FINAL.json')['items'], 'candidate_id', 'LEGACY_QUALITY')
        require(set(rows) == set(qs), 'LEGACY_QUALITY_ID_DRIFT')
        for cid, item in rows.items():
            require(cid in legacy_ids and cid not in items, 'LEGACY_ITEM_ID_DRIFT: ' + cid)
            require(item.get('original_question_record') == pool_items[cid], 'LEGACY_ORIGINAL_RECORD_CHANGED: ' + cid)
            tr = triage[cid]
            # The frozen v125 effective record can retain an earlier historical
            # original-record hash. Current reuse binds the already frozen
            # effective payload; it never applies another historical repair.
            require(object_sha256(pool_items[cid]) == tr.get('effective_record_sha256')
                    and tr.get('source_upgraded') is False and tr.get('source_round') == 'R02'
                    and tr['source_binding']['sha256'] == item['source_sha256']
                    and item.get('quality_reuse') == tr.get('quality'), 'LEGACY_FROZEN_QUALIFICATION_DRIFT: ' + cid)
            inputs.check_embedded_bindings(tr)
            require(item.get('new_content_rewrite') is False and tr['quality'].get('quality_status') == 'PASS'
                    and tr['quality'].get('mode') == 'reused_exact_source_first_review'
                    and qs[cid].get('content_pass') is True and qs[cid].get('status') == 'PASS',
                    'LEGACY_NOT_FROZEN_QUALIFIED: ' + cid)
            item_source(inputs, item)
            items[cid] = item
            qualities[cid] = qs[cid]
            layers[cid] = legacy_layer_map[cid]
            provenance[cid] = {'origin': 'legacy_round2_reused', 'batch_index': roots[item['root_id']]['batch_index'],
                               'original_items_binding': inputs.binding(path), 'original_item_sha256': object_sha256(item),
                               'legacy_frozen_qualification_binding': inputs.binding(triage_path),
                               'legacy_historical_original_record_sha256': tr.get('original_record_sha256'),
                               'legacy_frozen_effective_record_sha256': tr['effective_record_sha256']}
    require({cid for cid, p in provenance.items() if p['origin'] == 'legacy_round2_reused'} == set(legacy_ids),
            'LEGACY_PRIVATE_ITEM_COVERAGE')
    # Use already frozen score summaries; this performs no answer parsing/scoring.
    for component in execution['components']:
        if component['group'] not in ('new', 'legacy'):
            continue
        by_name = {Path(b['path']).name: b for b in component['bindings']}
        score_path = Path(by_name['SCORES.json']['path'])
        data = inputs.read(score_path)
        execution_items = unique_rows(inputs.read(score_path.parent / 'ITEMS_PRIVATE.json'), 'candidate_id', 'SOLVER_ITEMS')
        component_items = unique_rows(inputs.read(score_path.parent.parent / 'ITEMS_PRIVATE.json'), 'candidate_id', 'COMPONENT_ORIGINAL_ITEMS')
        for cid, item in component_items.items():
            require(cid in items and items[cid] == item, 'COMPONENT_ORIGINAL_ITEM_CHANGED: ' + cid)
        for cid, modes in data['items'].items():
            require(cid in items and cid not in scores and component_items.get(cid) == items[cid],
                    'REPEATED_OR_CHANGED_EVALUATED_ITEM: ' + cid)
            # The fixed adapter adds only these execution-budget fields. The
            # selected payload always comes from the untouched original list.
            normalized = {**items[cid], 'permutations': min(6, math.factorial(len(items[cid]['options']))),
                          'repeats_per_permutation': 3, 'turn_count': items[cid]['source_turn_count']}
            require(execution_items.get(cid) == normalized, 'SOLVER_ITEM_NORMALIZATION_DRIFT: ' + cid)
            scores[cid] = {'modes': modes, 'binding': inputs.binding(score_path)}
        # Components retain the original intake slots even when only a disjoint
        # quality-PASS supplement is evaluated. Frozen execution audit bindings
        # prove the actual subset; every PASS candidate needs exactly one score
        # below, while global scores[cid] rejects repeated evaluations.
        require(set(data['items']) == set(execution_items) and set(execution_items) <= set(component_items),
                'SOLVER_ITEM_SCORE_SUBSET_COVERAGE_DRIFT')
    return items, provenance, qualities, layers, scores, original_item_paths


def validate_source_qualifications(inputs, items, provenance, layers, closed):
    qualifications = {}
    for index in range(1, 10):
        if index == 1:
            paths = [RUN / 'source_preparation/BATCH01_EVIDENCE_QUALIFICATION_AUDIT.json']
        elif index == 2:
            paths = [RUN / 'source_preparation/BATCH02_EVIDENCE_QUALIFICATION_INDEPENDENT_AUDIT.json']
        else:
            paths = [RUN / f'source_preparation/BATCH{index:02d}_{arm}_EVIDENCE_QUALIFICATION_AUDIT.json' for arm in ('F', 'G')]
        expected = {cid for cid, p in provenance.items() if p['batch_index'] == index and p['origin'] != 'legacy_round2_reused'}
        batch_qualified = {}
        for path in paths:
            audit = inputs.read(path)
            rows = unique_rows(audit.get('candidates', audit.get('candidate_rows')), 'candidate_id', str(path))
            for cid, row in rows.items():
                require(cid in expected and cid not in batch_qualified, 'SOURCE_AUDIT_SCOPE_DRIFT: ' + cid)
                item = items[cid]
                source_binding = row.get('source_version', row.get('source', row.get('source_binding', {})))
                require(source_binding.get('path') == item['source_path'] and source_binding.get('sha256') == item['source_sha256'],
                        'SOURCE_QUALIFICATION_VERSION_DRIFT: ' + cid)
                if index == 1:
                    passed = row.get('conclusion') == 'QUALIFIED_BY_EXISTING_REVIEW_FOR_CITED_USE' and row.get('actual_source_qualification_blocker') is False
                elif index == 2:
                    passed = row.get('source_qualification_status') == 'EXISTING_REVIEW_SCOPE_MATCHED_NO_CITED_WARNING_CONFLICT_FOUND'
                else:
                    require(row.get('quotes_and_origins_match') is True, 'CITED_QUOTE_ORIGIN_NOT_BOUND: ' + cid)
                    passed = row.get('source_blocker') is False
                status = 'PASS_EXISTING_CITED_USE' if passed else 'HOLD'
                require(layers[cid].get('source_qualification_status') == status, 'SOURCE_LAYER_AUDIT_DISAGREEMENT: ' + cid)
                batch_qualified[cid] = {'status': status, 'audit_binding': inputs.binding(path),
                                        'audit_candidate_id': cid, 'new_source_qualification_granted': False}
        require(set(batch_qualified) == expected, 'SOURCE_AUDIT_MISSING_CANDIDATES: ' + str(index))
        holds = {cid for cid, row in batch_qualified.items() if row['status'] == 'HOLD'}
        require(holds == set(closed[index - 1]['source_hold_ids']), 'SOURCE_HOLD_CLOSURE_DISAGREEMENT: ' + str(index))
        hold_application = RUN / f'source_preparation/BATCH{index:02d}_SOURCE_HOLD_APPLICATION.json'
        if holds:
            require(hold_application.is_file(), 'MISSING_FROZEN_SOURCE_HOLD_APPLICATION')
            require(set(inputs.read(hold_application)['source_HOLD_candidate_ids']) == holds, 'SOURCE_HOLD_APPLICATION_DRIFT')
        qualifications.update(batch_qualified)
    for cid, p in provenance.items():
        if p['origin'] == 'legacy_round2_reused':
            qualifications[cid] = {'status': 'LEGACY_FROZEN_QUALIFIED',
                                   'audit_binding': p['legacy_frozen_qualification_binding'],
                                   'new_source_qualification_granted': False}
    return qualifications


def validate_layers(items, provenance, qualities, layers, scores, qualifications):
    decisions = {}
    for cid, item in items.items():
        layer, quality = layers[cid], qualities[cid]
        require(layer.get('root_id') == item['root_id'] and layer.get('question_type') == item['question_type'],
                'LAYER_ITEM_IDENTITY_DRIFT: ' + cid)
        require(layer.get('quality_status') == quality.get('status'), 'QUALITY_LAYER_DISAGREEMENT: ' + cid)
        passed = quality.get('status') == 'PASS' and quality.get('content_pass') is True
        legacy = provenance[cid]['origin'] == 'legacy_round2_reused'
        if not legacy:
            require(layer.get('quality_projection') == quality, 'QUALITY_PROJECTION_DRIFT: ' + cid)
            require(layer.get('final_answer_only_or_no_context_confers_eligibility') is False,
                    'DIAGNOSTIC_ADMISSION_DRIFT: ' + cid)
        metric = 'directed_pair_F1' if item['question_type'] == 'partial_order' else 'signed_J'
        planned = min(6, math.factorial(len(item['options']))) * 3
        score = scores.get(cid)
        mean, full, valid = None, {}, False
        if not passed:
            require(score is None, 'QUALITY_HOLD_WAS_EVALUATED: ' + cid)
            value_state = 'QUALITY_HOLD_NOT_EVALUATED'
        else:
            require(score is not None, 'PASS_CANDIDATE_MISSING_FROZEN_EVALUATION: ' + cid)
            full = score['modes']['full_context']
            mean = full.get('mean_directed_pair_f1' if metric == 'directed_pair_F1' else 'mean_J')
            require(isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(mean),
                    'FULL_METRIC_MISSING_OR_NONFINITE: ' + cid)
            require(full.get('planned_calls') == planned and 0 <= full.get('valid_calls', -1) <= planned
                    and full.get('invalid_calls') == planned - full['valid_calls'], 'FULL_CARDINALITY_DRIFT: ' + cid)
            valid = full['valid_calls'] == planned and full['invalid_calls'] == 0
            value_state = ('OUTPUT_PROTOCOL_PENDING' if not valid else 'MASTERED_RESERVE' if mean == 1
                           else 'HARD_EXAMPLE_DIAGNOSTIC' if mean <= 0 else 'MAIN_POOL_CANDIDATE')
            require(mean <= 1, 'FULL_METRIC_OUTSIDE_CONTRACT: ' + cid)
            require(layer.get('full_statistics') == full, 'LAYER_FULL_STATISTICS_DRIFT: ' + cid)
            if legacy:
                require(layer.get('score_path') == score['binding']['path'], 'LEGACY_SCORE_PATH_DRIFT: ' + cid)
            else:
                require(layer.get('all_mode_statistics') == score['modes']
                        and layer.get('score_binding', {}).get('sha256') == score['binding']['sha256']
                        and layer.get('mean_full_score') == mean and layer.get('metric') == metric,
                        'FROZEN_SCORE_LAYER_DRIFT: ' + cid)
        require(layer.get('training_value_status') == value_state, 'TRAINING_VALUE_LAYER_DRIFT: ' + cid)
        source_pass = qualifications[cid]['status'] in ('PASS_EXISTING_CITED_USE', 'LEGACY_FROZEN_QUALIFIED')
        eligible = passed and source_pass and valid and mean is not None and 0 < mean < 1 and value_state == 'MAIN_POOL_CANDIDATE'
        exclusions = []
        if not passed:
            exclusions.append('QUALITY_NOT_PASS')
        if not source_pass:
            exclusions.append('SOURCE_QUALIFICATION_HOLD')
        if value_state != 'MAIN_POOL_CANDIDATE':
            exclusions.append(value_state)
        decisions[cid] = {'candidate_id': cid, 'root_id': item['root_id'], 'arm': item.get('arm'),
                          'origin': provenance[cid]['origin'], 'question_type': item['question_type'],
                          'quality': {'status': quality['status'], 'content_pass': passed},
                          'source_qualification': qualifications[cid],
                          'training_value': {'status': value_state, 'metric': metric,
                                             'planned_full_calls': planned, 'valid_full_calls': full.get('valid_calls', 0),
                                             'all_full_calls_strictly_valid': valid, 'mean_full_score': mean,
                                             'score_binding': score['binding'] if score else None,
                                             'no_context_and_answer_only_are_diagnostic': True},
                          'eligible': eligible, 'ineligibility_reasons': exclusions,
                          'original_items_binding': provenance[cid]['original_items_binding'],
                          'original_item_sha256': provenance[cid]['original_item_sha256'],
                          'original_layer_decision': copy.deepcopy(layer)}
    return decisions


def validate_family_audits(inputs, items, provenance):
    mappings, audit_occurrences = {}, defaultdict(list)
    family_members = defaultdict(set)
    for index, path in enumerate(FAMILY_AUDITS, 1):
        audit = inputs.read(path)
        scope = audit.get('scope', {})
        require(scope.get('solver_scores_read_or_used') is False and scope.get('model_calls_executed') == 0
                and scope.get('training_executed') is False and scope.get('candidate_content_rewritten') is False,
                'FAMILY_AUDIT_NOT_SCORE_FREE: ' + str(path))
        require('PREAUDIT' not in audit.get('status', '').upper(), 'G_ONLY_PREAUDIT_IS_NOT_FINAL: ' + str(path))
        rows = unique_rows(audit.get('candidate_family_map'), 'candidate_id', str(path))
        new_expected = {cid for cid, p in provenance.items() if p['batch_index'] == index and p['origin'] != 'legacy_round2_reused'}
        legacy_expected = {cid for cid, p in provenance.items() if p['origin'] == 'legacy_round2_reused'
                           and (index == 1 or p['batch_index'] == index)}
        require(set(rows) == new_expected | legacy_expected, 'FAMILY_AUDIT_MUST_INCLUDE_ALL_F_G_AND_SCOPED_LEGACY: ' + str(index))
        families = unique_rows(audit.get('families'), 'family_id', str(path) + ':families')
        represented = defaultdict(set)
        for cid, row in rows.items():
            require(cid in items and row.get('root_id') == items[cid]['root_id']
                    and row.get('question_type') == items[cid]['question_type']
                    and row.get('source_sha256') == items[cid]['source_sha256'], 'FAMILY_ITEM_IDENTITY_DRIFT: ' + cid)
            require(isinstance(row.get('family_id'), str) and row['family_id']
                    and isinstance(row.get('structural_operation'), str) and row['structural_operation'], 'FAMILY_MAPPING_INCOMPLETE: ' + cid)
            if row.get('candidate_object_sha256'):
                # Legacy family maps may hash the pre-adapter original question.
                permitted = {object_sha256(items[cid])}
                if provenance[cid]['origin'] == 'legacy_round2_reused':
                    permitted.add(object_sha256(items[cid]['original_question_record']))
                require(row['candidate_object_sha256'] in permitted, 'FAMILY_ORIGINAL_ITEM_OBJECT_HASH_DRIFT: ' + cid)
            if row.get('original_item_path'):
                require(row['original_item_path'] == provenance[cid]['original_items_binding']['path']
                        and row.get('original_item_file_sha256') == provenance[cid]['original_items_binding']['sha256'],
                        'FAMILY_ORIGINAL_ITEMS_FILE_HASH_DRIFT: ' + cid)
            if cid in mappings:
                require(mappings[cid]['family_id'] == row['family_id'], 'CROSS_BATCH_FAMILY_ID_DISAGREEMENT: ' + cid)
            else:
                mappings[cid] = row
            represented[row['family_id']].add(cid)
            family_members[row['family_id']].add(cid)
            audit_occurrences[cid].append({'audit_binding': inputs.binding(path), 'family_id': row['family_id'],
                                           'structural_operation': row['structural_operation']})
        require(set(represented) == set(families), 'FAMILY_TABLE_MAPPING_MISMATCH: ' + str(path))
        for fid, members in represented.items():
            require(set(families[fid]['candidate_ids']) == members, 'FAMILY_MEMBERSHIP_DRIFT: ' + fid)
    require(set(mappings) == set(items), 'GLOBAL_FAMILY_MAPPING_INCOMPLETE')
    return mappings, family_members, audit_occurrences


def validate_global_review(inputs, items, mappings, members, original_paths):
    review = inputs.read(GLOBAL_REVIEW)
    require(review.get('schema_version') == 'cross_batch_family_review_v1'
            and review.get('status') == 'COMPLETE_INDEPENDENT_SCORE_FREE_COLLECTION_REVIEW', 'GLOBAL_SCORE_FREE_REVIEW_NOT_COMPLETE')
    scope = review.get('scope', {})
    all_ids = scope.get('all_candidate_ids', [])
    require(isinstance(all_ids, list) and len(all_ids) == len(set(all_ids))
            and set(all_ids) == set(items) and scope.get('solver_scores_read_or_used') is False
            and scope.get('model_calls_executed') == 0 and scope.get('training_executed') is False,
            'GLOBAL_REVIEW_SCORE_FREE_SCOPE_OR_COVERAGE_DRIFT')
    bindings = review.get('input_bindings')
    require(isinstance(bindings, list), 'GLOBAL_REVIEW_INPUT_BINDINGS_MISSING')
    bound_paths = {b.get('path') for b in bindings}
    required_paths = {str(p) for p in FAMILY_AUDITS} | {str(RULES), str(REQUIREMENTS)} | original_paths
    require(required_paths <= bound_paths, 'GLOBAL_REVIEW_DOES_NOT_BIND_REQUIRED_SCORE_FREE_INPUTS: '
            + ', '.join(sorted(required_paths - bound_paths)))
    for binding in bindings:
        path = binding.get('path', '').lower()
        require(not any(word in path for word in ('/solver/', '/feedback', 'scores.json', 'quality_final', 'candidate_layer_decisions')),
                'GLOBAL_REVIEW_HAS_FORBIDDEN_SCORING_QUALITY_OR_FEEDBACK_INPUT: ' + path)
        inputs.check_binding(binding)
    global_map = unique_rows(review.get('candidate_family_map'), 'candidate_id', 'GLOBAL_CANDIDATE_FAMILY_MAP')
    require(set(global_map) == set(items), 'GLOBAL_REVIEW_MAP_COVERAGE_DRIFT')
    for cid, row in global_map.items():
        require(row.get('family_id') == mappings[cid]['family_id'], 'GLOBAL_REVIEW_CANNOT_SILENTLY_CHANGE_FROZEN_FAMILY: ' + cid)
        require(isinstance(row.get('reporting_structural_cluster'), str) and row['reporting_structural_cluster'].strip(),
                'GLOBAL_REPORTING_STRUCTURAL_CLUSTER_MISSING: ' + cid)
    resolutions = unique_rows(review.get('duplicate_family_resolutions'), 'family_id', 'GLOBAL_DUPLICATE_RESOLUTIONS')
    duplicate_ids = {fid for fid, group in members.items() if len(group) > 1}
    require(set(resolutions) == duplicate_ids, 'GLOBAL_REVIEW_MUST_EXPLICITLY_RESOLVE_EVERY_DUPLICATE_FAMILY')
    for fid, row in resolutions.items():
        candidate_ids = row.get('candidate_ids', [])
        require(isinstance(candidate_ids, list) and len(candidate_ids) == len(set(candidate_ids))
                and set(candidate_ids) == members[fid] and row.get('score_free') is True
                and isinstance(row.get('reason'), str) and row['reason'].strip(), 'GLOBAL_FAMILY_RESOLUTION_INCOMPLETE: ' + fid)
        if row.get('disposition') == 'equivalent_variants':
            require(row.get('information_need_equivalent') is True and row.get('operation_equivalent') is True,
                    'EQUIVALENCE_NOT_EXPLICITLY_REVIEWED: ' + fid)
        else:
            require(row.get('disposition') == 'score_free_resolution', 'UNRESOLVED_NON_EQUIVALENT_FAMILY_TRADEOFF: ' + fid)
            groups = row.get('preferred_candidate_ids')
            require(isinstance(groups, list) and all(isinstance(g, list) and g for g in groups),
                    'SCORE_FREE_RESOLUTION_NEEDS_ORDERED_NONEMPTY_GROUPS_OR_EMPTY_LIST: ' + fid)
            flattened = [cid for group in groups for cid in group]
            require(not groups or (len(flattened) == len(set(flattened)) and set(flattened) == members[fid]),
                    'SCORE_FREE_PRIORITY_GROUPS_MUST_PARTITION_ALL_FAMILY_MEMBERS: ' + fid)
    concentration = review.get('concentration_review', {})
    require(all(concentration.get(key) is True for key in ('operation_clusters_reviewed', 'question_types_reviewed',
                'source_and_root_correlation_reviewed', 'inherited_source_correlation_reviewed'))
            and concentration.get('fine_families_independent') is False
            and isinstance(concentration.get('limitations'), list) and concentration['limitations']
            and all(isinstance(x, str) and x.strip() for x in concentration['limitations']),
            'GLOBAL_CONCENTRATION_AND_DEPENDENCE_REVIEW_INCOMPLETE')
    return review, global_map, resolutions


def choose_families(decisions, mappings, members, occurrences, global_map, resolutions):
    """Only eligibility booleans, reviewed priority groups, and fixed hashes select."""
    family_decisions, selected = [], set()
    for fid in sorted(members):
        peers = sorted(members[fid])
        eligible = [cid for cid in peers if decisions[cid]['eligible']]
        ties = {cid: tie_value(fid, cid) for cid in peers}
        resolution = resolutions.get(fid)
        if resolution and resolution['disposition'] == 'score_free_resolution':
            groups = resolution['preferred_candidate_ids']
            pool = next(([cid for cid in group if cid in eligible] for group in groups
                         if any(cid in eligible for cid in group)), [])
            reason = 'SCORE_FREE_REVIEW_PRIORITY_THEN_FROZEN_HASH' if pool else 'SCORE_FREE_REVIEW_RETAINS_NONE' if not groups else 'NO_ELIGIBLE_MEMBER'
        else:
            pool = eligible
            reason = ('ONLY_ELIGIBLE_DISTINCT_FAMILY_MEMBER' if len(peers) == 1
                      else 'SCORE_FREE_EQUIVALENCE_THEN_FROZEN_HASH') if pool else 'NO_ELIGIBLE_MEMBER'
        winner = min(pool, key=ties.__getitem__) if pool else None
        if winner:
            selected.add(winner)
        record = {'family_id': fid, 'candidate_ids': peers, 'eligible_candidate_ids': eligible,
                  'eligible_selection_priority_group': sorted(pool), 'tie_values_sha256': ties,
                  'tie_rule': TIE_RULE, 'selected_candidate_id': winner, 'selection_reason': reason,
                  'score_free_review_resolution': copy.deepcopy(resolution),
                  'fine_family_is_an_independent_structural_or_statistical_unit': False}
        family_decisions.append(record)
        for cid in peers:
            row = decisions[cid]
            row.update(family_id=fid, structural_operation=mappings[cid]['structural_operation'],
                       reporting_structural_cluster=global_map[cid]['reporting_structural_cluster'],
                       family_audit_occurrences=occurrences[cid], family_peer_candidate_ids=[p for p in peers if p != cid],
                       eligible_family_peer_candidate_ids=[p for p in eligible if p != cid],
                       tie_value_sha256=ties[cid], family_tie_values_sha256=ties,
                       selected=cid == winner, selected_family_representative=winner)
            row['collection'] = {'status': 'SELECTED' if cid == winner else 'NOT_ELIGIBLE_RETAIN_ORIGINAL_STATUS' if not row['eligible']
                                 else 'ELIGIBLE_NOT_SELECTED_SCORE_FREE_REVIEW' if not pool or cid not in pool
                                 else 'ELIGIBLE_NOT_SELECTED_FROZEN_HASH_TIE',
                                 'reason': reason, 'original_quality_source_and_training_value_preserved': True}
    return selected, family_decisions


def evidence_refs(value):
    refs = set()
    if isinstance(value, dict):
        if isinstance(value.get('turn_ref'), str):
            refs.add(value['turn_ref'])
        for child in value.values():
            refs.update(evidence_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(evidence_refs(child))
    return refs


def round_of(ref):
    found = re.match(r'R\d{2}(?=:)', ref)
    return found.group(0) if found else 'parent'


def gold_count(item):
    gold = item['gold_option_ids']
    if item['question_type'] != 'partial_order':
        return len(gold)
    # Same deterministic transitive closure used by the typed export interface;
    # this normalizes the stored Gold representation, never evaluates a model.
    ids = set(item['label_to_option_id'].values())
    reachable = {key: set() for key in ids}
    for pair in gold:
        require(isinstance(pair, list) and len(pair) == 2 and set(pair) <= ids and pair[0] != pair[1],
                'INVALID_STORED_PARTIAL_ORDER_GOLD: ' + item['candidate_id'])
        reachable[pair[0]].add(pair[1])
    for via in sorted(ids):
        for start in sorted(ids):
            if via in reachable[start]:
                reachable[start].update(reachable[via])
    require(all(key not in values for key, values in reachable.items()), 'CYCLIC_STORED_PARTIAL_ORDER_GOLD')
    return sum(len(values) for values in reachable.values())


def statistics(inputs, ids, items, decisions, all_items):
    def counts(values):
        return dict(sorted(Counter(map(str, values)).items()))
    chosen = [items[cid] for cid in sorted(ids)]
    sources = {q['source_path']: inputs.read(q['source_path']) for q in chosen}
    citations = {q['candidate_id']: evidence_refs(q) for q in chosen}
    ref_counts = Counter((q['root_id'], ref) for q in chosen for ref in citations[q['candidate_id']])
    root_rows = []
    for root in sorted({q['root_id'] for q in all_items.values()}):
        selected_here = [q for q in chosen if q['root_id'] == root]
        legacy = next(q for q in all_items.values() if q['root_id'] == root and decisions[q['candidate_id']]['origin'] == 'legacy_round2_reused')
        legacy_source = inputs.read(legacy['source_path'])
        old = {t['turn_ref']: t for t in legacy_source['turns']}
        versions = []
        for path in sorted({q['source_path'] for q in selected_here}):
            new = {t['turn_ref']: t for t in sources[path]['turns']}
            shared = old.keys() & new.keys()
            unchanged = sum(all(old[ref].get(k) == new[ref].get(k) for k in ('speaker', 'text')) for ref in shared)
            versions.append({'source_binding': inputs.binding(path), 'selected_count': sum(q['source_path'] == path for q in selected_here),
                             'same_bytes_as_frozen_legacy_source': inputs.binding(path)['sha256'] == legacy['source_sha256'],
                             'legacy_turn_count': len(old), 'shared_turn_ref_count': len(shared),
                             'shared_dialogue_fields_unchanged_count': unchanged,
                             'all_legacy_turn_dialogue_fields_preserved': len(shared) == unchanged == len(old)})
        root_rows.append({'root_id': root, 'selected_count': len(selected_here),
                          'selected_by_origin': counts(decisions[q['candidate_id']]['origin'] for q in selected_here),
                          'selected_by_arm': counts(q.get('arm') for q in selected_here),
                          'selected_semantic_family_count': len({decisions[q['candidate_id']]['family_id'] for q in selected_here}),
                          'inherited_source_versions': versions, 'independent_sample_claim': False})
    return {'candidate_count': len(chosen),
            'question_types': counts(q['question_type'] for q in chosen),
            'origins': counts(decisions[q['candidate_id']]['origin'] for q in chosen),
            'arms': counts(q.get('arm') for q in chosen),
            'structural_operations_original_labels': counts(decisions[q['candidate_id']]['structural_operation'] for q in chosen),
            'structural_clusters_global_reporting': counts(decisions[q['candidate_id']]['reporting_structural_cluster'] for q in chosen),
            'option_counts': counts(len(q['options']) for q in chosen),
            'gold_cardinalities_by_question_type': {qtype: counts(gold_count(q) for q in chosen if q['question_type'] == qtype)
                                                    for qtype in sorted({q['question_type'] for q in chosen})},
            'gold_count_partial_order_policy': 'Transitive closure of the stored Gold relation; no model output rescoring.',
            'source_latest_rounds': counts(max((round_of(t['turn_ref']) for t in sources[q['source_path']]['turns']),
                                              key=lambda x: (-1 if x == 'parent' else int(x[1:]))) for q in chosen),
            'cited_layer_combinations': counts('+'.join(sorted({round_of(ref) for ref in citations[q['candidate_id']]})) for q in chosen),
            'cited_layer_candidate_counts': counts(layer for q in chosen for layer in {round_of(ref) for ref in citations[q['candidate_id']]}),
            'exact_source_hash_clusters': counts(q['source_sha256'] for q in chosen),
            'per_root': root_rows,
            'shared_cited_refs': [{'root_id': root, 'turn_ref': ref, 'candidate_count': count}
                                 for (root, ref), count in sorted(ref_counts.items()) if count > 1],
            'semantic_family_count': len({decisions[q['candidate_id']]['family_id'] for q in chosen}),
            'fine_family_count_is_independent_structure_count': False,
            'root_split_policy': 'Existing 68 Train roots; shared and inherited sources remain correlated; no unseen-root claim.',
            'concentration_interpretation': 'Limited candidate set. Fine family counts do not establish independent structural diversity or independent statistical samples.'}


def build_selection(inputs):
    inputs.binding(Path(__file__))
    roots, closed, execution = validate_gates(inputs)
    items, provenance, qualities, layers, scores, originals = load_candidates(inputs, roots, closed, execution)
    qualifications = validate_source_qualifications(inputs, items, provenance, layers, closed)
    decisions = validate_layers(items, provenance, qualities, layers, scores, qualifications)
    mappings, members, occurrences = validate_family_audits(inputs, items, provenance)
    review, global_map, resolutions = validate_global_review(inputs, items, mappings, members, originals)
    selected, family_decisions = choose_families(decisions, mappings, members, occurrences, global_map, resolutions)
    selected_items = []
    for cid in sorted(selected):
        original = items[cid]
        item = copy.deepcopy(original)
        metadata = {'origin': provenance[cid]['origin'], 'family_id': decisions[cid]['family_id'],
                    'selection_reason': decisions[cid]['collection']['reason'],
                    'selection_tie_value_sha256': decisions[cid]['tie_value_sha256'],
                    'selection_original_item_sha256': provenance[cid]['original_item_sha256'],
                    'selection_original_items_binding': provenance[cid]['original_items_binding'],
                    'selection_eligible_family_peers': decisions[cid]['eligible_family_peer_candidate_ids'],
                    'selection_original_origin': original.get('origin')}
        for key, value in metadata.items():
            require(key == 'origin' or key not in item or item[key] == value, 'SELECTION_METADATA_COLLISION: ' + cid + ':' + key)
            item[key] = value
        restored = {key: copy.deepcopy(value) for key, value in item.items() if key not in metadata}
        for key in metadata:
            if key in original:
                restored[key] = copy.deepcopy(original[key])
        require(restored == original and object_sha256(restored) == provenance[cid]['original_item_sha256'],
                'SELECTED_ORIGINAL_ITEM_CONTENT_CHANGED: ' + cid)
        selected_items.append(item)
    eligible = {cid for cid, row in decisions.items() if row['eligible']}
    report = {'schema_version': 'final_union_selection_v1', 'status': 'FINAL_SELECTED_FROZEN_NO_EXPORT_OR_TRAINING',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'items': selected_items,
              'candidate_decisions': [decisions[cid] for cid in sorted(decisions)], 'family_decisions': family_decisions,
              'counts': {'all_candidates': len(items), 'eligible_candidates': len(eligible),
                         'selected_candidates': len(selected), 'semantic_families': len(members)},
              'selected_statistics': statistics(inputs, selected, items, decisions, items),
              'eligible_statistics': statistics(inputs, eligible, items, decisions, items),
              'global_score_free_review_binding': inputs.binding(GLOBAL_REVIEW),
              'global_concentration_review': review['concentration_review'],
              'tie_rule': TIE_RULE, 'same_family_maximum': 1,
              'quality_source_training_value_and_collection_separated': True,
              'original_items_preserved_except_export_origin_metadata': True,
              'original_origin_metadata_retained': True, 'scores_used_to_rank_candidates': False,
              'fine_families_claimed_independent': False, 'model_calls': 0, 'rescoring': False,
              'training_started': False, 'training_export_created': False}
    # Include source reads performed for concentration reporting, then check all
    # frozen bytes again immediately before returning to the publication step.
    inputs.unchanged()
    report['input_bindings'] = [inputs.bindings[key] for key in sorted(inputs.bindings)]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='Validate and select in memory; write no files.')
    args = parser.parse_args()
    try:
        require(not OUTPUT.exists(), 'FINAL_SELECTION_ALREADY_EXISTS: ' + str(OUTPUT))
        inputs = FrozenInputs()
        result = build_selection(inputs)
        if not args.dry_run:
            payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            inputs.unchanged()
            # Exclusive creation is the only mutation in this tool. No output
            # exists when prerequisites, bindings, or review checks fail.
            with OUTPUT.open('x', encoding='utf-8') as handle:
                handle.write(payload)
        print(json.dumps({'status': 'DRY_RUN_PASS_NO_FILES_WRITTEN' if args.dry_run else result['status'],
                          'counts': result['counts'], 'output_written': not args.dry_run,
                          'output_path': str(OUTPUT), 'model_calls': 0, 'rescoring': False,
                          'training_started': False, 'training_export_created': False}, ensure_ascii=False, indent=2))
        return 0
    except (SelectionError, FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({'status': 'BLOCKED_NO_SELECTION_WRITTEN', 'reason': str(exc),
                          'output_written': False, 'model_calls': 0, 'rescoring': False,
                          'training_started': False, 'training_export_created': False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
