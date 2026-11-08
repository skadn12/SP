#!/usr/bin/env python3
"""Independent CPU-only mechanical audit for the frozen stability experiment.

This module audits the gate as the system under test. It never imports the
experiment runner, never calls runner validation/scoring functions, never runs
inference, never changes candidates, and creates only new audit output files.
Use --stage initial first; candidates/prepared/execution require their inputs.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
HERE = AUDIT.parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
GATE = REPO / 'scripts/current_method/question_presentation_gate.py'
BASELINE = AUDIT / 'HISTORICAL_BASELINE.json'
FIXED_SEEDS = tuple(range(2026102300, 2026102400))
SCHEMA_VERSION = 'question_presentation_gate_v1'
PUBLIC_BASE_KEYS = {'schema_version', 'candidate_id', 'root_id', 'question_type',
                    'task_operation', 'question', 'source_binding'}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data)}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def literal_constants(path):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


def import_gate():
    spec = importlib.util.spec_from_file_location('stability_gate_system_under_test', GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_mapping(candidate, seed):
    """Independent implementation of the documented Gold-independent shuffle."""
    field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
    original = list(candidate[field])
    material = json.dumps([SCHEMA_VERSION, seed, candidate['candidate_id'], original],
                          ensure_ascii=False, separators=(',', ':')).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    permutation = original[:]
    for _ in range(128):
        rng.shuffle(permutation)
        if permutation != original:
            break
    else:
        permutation = original[1:] + original[:1]
    return {chr(65 + index): key for index, key in enumerate(permutation)}


def quote_nodes(value):
    if isinstance(value, dict):
        if 'quote' in value:
            yield value
        for child in value.values():
            yield from quote_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from quote_nodes(child)


def changed_leaf_paths(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            path = prefix + ('.' if prefix else '') + key
            if key not in before or key not in after:
                result.append(path)
            else:
                result.extend(changed_leaf_paths(before[key], after[key], path))
        return result
    return [] if before == after else [prefix]


def item_metadata(candidate, item_ids):
    for key in ('item_analysis', 'evidence_and_author_notes', 'item_evidence', 'evidence_by_item', 'evidence'):
        value = candidate.get(key)
        if isinstance(value, dict) and set(value) == set(item_ids):
            return value
    for value in candidate.values():
        if isinstance(value, dict) and set(value) == set(item_ids) and all(list(quote_nodes(item)) for item in value.values()):
            return value
    raise AssertionError('No complete per-item evidence mapping: ' + candidate['candidate_id'])


def independent_aggregate(records):
    n = len(records)
    mc = [r for r in records if r['question_type'] == 'multi_select']
    ordering = [r for r in records if r['question_type'] == 'ordering']
    correct = sum(r['exact'] for r in records)
    pair_total = sum(r['pairwise_total'] for r in ordering)
    pair_correct = sum(r['pairwise_correct'] for r in ordering)
    return {'frozen_requests': n, 'valid_requests': sum(r['valid'] for r in records),
            'invalid_requests': sum(not r['valid'] for r in records), 'correct': correct,
            'exact_accuracy': correct / n if n else None,
            'mc_requests': len(mc), 'mc_correct': sum(r['exact'] for r in mc),
            'mc_false_positive': sum(r['false_positive'] for r in mc),
            'mc_false_negative': sum(r['false_negative'] for r in mc),
            'mc_allselect_correct': sum(r['allselect_exact'] for r in mc),
            'mc_allselect_accuracy': sum(r['allselect_exact'] for r in mc) / len(mc) if mc else None,
            'ordering_requests': len(ordering), 'ordering_correct': sum(r['exact'] for r in ordering),
            'ordering_pairwise_correct': pair_correct, 'ordering_pairwise_total': pair_total,
            'ordering_pairwise_accuracy': pair_correct / pair_total if pair_total else None}


def independent_summary(scores, private):
    cells = []
    conditions = []
    buckets = []
    root_cells = []
    for condition in ('FULL', 'QUESTION_ONLY'):
        for variant in ('perm_00', 'perm_01'):
            rows = [r for r in scores if r['condition'] == condition and r['variant_id'] == variant]
            cells.append({'condition': condition, 'variant_id': variant, **independent_aggregate(rows)})
        conditions.append({'condition': condition, **independent_aggregate([r for r in scores if r['condition'] == condition])})
        for count in range(2, 7):
            rows = [r for r in scores if r['condition'] == condition and r['gold_count'] == count]
            buckets.append({'condition': condition, 'gold_count': count, **independent_aggregate(rows)})
    for root in sorted({r['root_id'] for r in scores}):
        for condition in ('FULL', 'QUESTION_ONLY'):
            rows = [r for r in scores if r['root_id'] == root and r['condition'] == condition]
            root_cells.append({'root_id': root, 'condition': condition, **independent_aggregate(rows)})
    variant_pairs = []
    for cid in sorted({r['candidate_id'] for r in scores}):
        for condition in ('FULL', 'QUESTION_ONLY'):
            rows = sorted([r for r in scores if r['candidate_id'] == cid and r['condition'] == condition], key=lambda r: r['variant_id'])
            assert len(rows) == 2
            a, b = rows
            both_valid = a['valid'] and b['valid']
            same = bool(both_valid and (set(a['prediction']) == set(b['prediction']) if a['question_type'] == 'multi_select' else a['prediction'] == b['prediction']))
            variant_pairs.append({'candidate_id': cid, 'condition': condition, 'question_type': a['question_type'],
                'both_valid': both_valid, 'canonical_prediction_same': same,
                'both_exact': a['exact'] and b['exact'], 'neither_exact': not a['exact'] and not b['exact'],
                'accuracy_changed': a['exact'] != b['exact'], 'first_prediction': a['prediction'], 'second_prediction': b['prediction']})
    initial_pass = sum(r['initial_decision'] == 'PASS' for r in private['new_admission'])
    semantic_pass = sum(r['initial_content_decision'] == 'PASS' for r in private['new_admission'])
    return {'requests': len(scores), 'valid': sum(r['valid'] for r in scores), 'correct': sum(r['exact'] for r in scores),
        'planned_questions': 28, 'admitted_questions': sum(r['decision'] == 'PASS' for r in private['new_admission']),
        'initial_content_pass': initial_pass, 'initial_content_pass_rate': initial_pass / 28,
        'initial_semantic_content_pass': semantic_pass, 'initial_semantic_content_pass_rate': semantic_pass / 28,
        'initial_content_stability_threshold': .9, 'initial_content_stability_threshold_met': initial_pass / 28 >= .9,
        'coverage_by_root': private['coverage_by_root'], 'new_mc_coverage': private['new_mc_coverage'],
        'cells': cells, 'conditions': conditions, 'new_mc_gold_buckets': buckets, 'root_cells': root_cells,
        'variant_pairs': variant_pairs, 'new_admission': private['new_admission'],
        'question_only_both_variants_correct': [p['candidate_id'] for p in variant_pairs if p['condition'] == 'QUESTION_ONLY' and p['both_exact']],
        'all_frozen_requests_in_exact_denominator': True, 'invalid_mc_prediction_for_descriptive_fp_fn': 'empty_set',
        'semantic_reason_review': 'NOT_AUTOMATICALLY_CERTIFIED', 'training_admission': False, 'feedback_eligible': False,
        'historical_scores_reused': False, 'new_dialogue_turns': 0,
        'inference_limits': 'Four selected Train roots, two stochastic permutations each. Separately assess mechanical stability, content yield and possible shortcuts; no population or causal feedback claim.'}


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.details = {}
        self.bindings = []

    def need(self, value, message, group='other'):
        self.counts[group] += 1
        if not value:
            raise AssertionError(message)

    def check_binding(self, bound, group='binding'):
        observed = binding(bound['path'])
        self.need(all(observed[k] == bound[k] for k in ('path', 'sha256', 'bytes')),
                  'Binding drift: ' + bound['path'], group)

    def sources(self):
        selection = read(HERE / 'SELECTION_FREEZE.json')
        roots = selection['roots']
        expected = ['122711', '130221', '135466', '136202']
        self.need([r['root_id'].split('-')[-1] for r in roots] == expected,
                  'Unexpected frozen root order', 'selection')
        files = sorted((WEEK / 'day3_production_v1/sources').glob('*.json'))
        excluded = set(selection['excluded_previous_pilot_roots'])
        chosen = [p for p in files if p.stem.split('-')[-1] not in excluded][:4]
        self.need([str(p.resolve()) for p in chosen] == [r['path'] for r in roots],
                  'Selection is not lexicographic unexposed roots', 'selection')
        self.need(selection['planned_candidates'] == 28 and selection['maximum_solver_requests'] == 112,
                  'Planned inventory drift', 'selection')
        sources = {}
        for root in roots:
            self.check_binding(root, 'source_binding')
            source = read(root['path'])
            refs = [t['turn_ref'] for t in source['turns']]
            self.need(source['root_id'] == root['root_id'] and len(refs) == root['turns'],
                      'Source root/turn mismatch', 'source_binding')
            self.need(len(refs) == len(set(refs)) and all(t.get('speaker') and isinstance(t.get('text'), str) for t in source['turns']),
                      'Invalid reference/speaker/text inventory', 'source_binding')
            sources[root['root_id']] = source
        self.details['selection'] = {'roots': [r['root_id'] for r in roots],
                                     'source_turns': {k: len(v['turns']) for k, v in sources.items()},
                                     'rule_verified': 'filename_lexicographic_only'}
        return selection, sources

    def baseline(self, create=False):
        if create and not BASELINE.exists():
            old = read(WEEK / 'postweek_question_quality_pilot_v1/HISTORICAL_FREEZE_BEFORE.json')
            final_manifests = sorted(p for p in WEEK.rglob('FINAL_MANIFEST.json') if HERE not in p.parents)
            inner = []
            for bound in old['bindings'][:2]:
                self.check_binding(bound, 'historical_original_chain')
                inner.extend(read(bound['path'])['bindings'])
            self.need(len(inner) == 61, 'Original historical61 set changed', 'historical_original_chain')
            pilot = WEEK / 'postweek_question_quality_pilot_v1/FINAL_MANIFEST.json'
            value = {'schema_version': 'stability_independent_historical_baseline_v1',
                     'created_at_utc': datetime.now(timezone.utc).isoformat(),
                     'all_existing_final_manifests': [binding(p) for p in final_manifests],
                     'historical_outer_bindings': old['bindings'],
                     'historical_61_bindings': inner,
                     'previous_pilot_manifest': binding(pilot),
                     'previous_pilot_manifest_bindings': read(pilot)['bindings'],
                     'frozen_current_gate': binding(GATE),
                     'selection_binding': binding(HERE / 'SELECTION_FREEZE.json'),
                     'protocol_binding': binding(HERE / 'PROTOCOL_ZH.md')}
            save_new(BASELINE, value)
        base = read(BASELINE)
        for group, key in [('existing_final_manifest', 'all_existing_final_manifests'),
                           ('historical_outer_binding', 'historical_outer_bindings'),
                           ('historical_61_binding', 'historical_61_bindings'),
                           ('previous_pilot_bound_file', 'previous_pilot_manifest_bindings')]:
            for bound in base[key]:
                self.check_binding(bound, group)
        for key in ('previous_pilot_manifest', 'frozen_current_gate', 'selection_binding', 'protocol_binding'):
            self.check_binding(base[key], 'audit_baseline')
        self.details['historical_preservation'] = {
            'all_existing_final_manifests_unchanged': len(base['all_existing_final_manifests']),
            'historical_61_entries_unchanged': len(base['historical_61_bindings']),
            'previous_pilot_manifest_entries_unchanged': len(base['previous_pilot_manifest_bindings']),
            'gate_unchanged': True}
        self.bindings.append(binding(BASELINE))

    def gate_compatibility(self):
        gate = import_gate()
        synthetic = {'candidate_id': 'AUDIT-SYNTHETIC-ORDER', 'root_id': 'synthetic',
                     'question_type': 'ordering', 'task_operation': 'ORDER_EXPLICIT_TIME',
                     'question': 'Order the recorded events.',
                     'items': {'I1': 'One event', 'I2': 'Another event', 'I3': 'A third event', 'I4': 'A fourth event'},
                     'gold': ['I3', 'I2', 'I4', 'I1'],
                     'source_binding': {'sha256': '0' * 64}}
        before = fingerprint(synthetic)
        public, private = gate.prepare_display(synthetic, FIXED_SEEDS[0])
        self.need(fingerprint(synthetic) == before, 'Gate mutates input', 'gate_compatibility')
        self.need(private['display_to_canonical'] == own_mapping(synthetic, FIXED_SEEDS[0]),
                  'Synthetic shuffle differs from independent reconstruction', 'gate_compatibility')
        partial = private['display_gold'][:-1]
        # This is intentionally a compatibility observation, never a batch score.
        accepts_partial_mapping = isinstance(gate.restore_prediction(private, partial), list)
        self.need(gate.is_correct_display_answer(private, partial) is False,
                  'Incomplete ordering must not receive exact credit', 'gate_compatibility')
        c2 = copy.deepcopy(synthetic)
        c2['gold'] = list(reversed(synthetic['gold']))
        public2, private2 = gate.prepare_display(c2, FIXED_SEEDS[0])
        self.need(public2 == public and private2['display_to_canonical'] == private['display_to_canonical'],
                  'Shuffle depends on Gold', 'gate_compatibility')
        rejection_checks = []
        for prediction in (['A', 'A'], ['Z'], 'A B'):
            try:
                gate.restore_prediction(private, prediction)
            except ValueError:
                rejection_checks.append(True)
            else:
                rejection_checks.append(False)
        self.need(all(rejection_checks), 'Malformed prediction accepted', 'gate_compatibility')
        try:
            gate.prepare_display(synthetic, True)
        except ValueError:
            pass
        else:
            raise AssertionError('Boolean seed accepted')
        self.details['gate_compatibility'] = {
            'gate_binding': binding(GATE), 'current_batch_4_to_6_items_supported': True,
            'gate_general_ordering_support': '2 to 26 items; batch schema supports only A to F, so batch must separately cap at 6',
            'restore_prediction_accepts_incomplete_ordering': accepts_partial_mapping,
            'incomplete_ordering_gets_exact_credit': False,
            'required_runner_validation': 'Require every displayed ordering label exactly once before scoring; gate.restore_prediction is a mapping primitive.',
            'gold_independent_shuffle_on_synthetic_case': True,
            'nonidentity_definition': 'Nonidentity to original candidate key order; can coincidentally equal Gold answer order. Never redraw by Gold.',
            'lint_limitations_zh': ['题面提示正则只给机械标记，不能替代语义审核，也不能保证检测全部答案数量提示。',
                                  'gate不验证来源文件真实性、逐字证据、语义总序、内容准入或多样性；本审计及独立内容审核须补足。',
                                  '排序不完整预测可经restore_prediction映射，但is_correct_display_answer不会授予exact；runner仍须标INVALID而不是当有效错误回答。',
                                  '随机排列可能恰好与Gold顺序相同，这不违反相对作者输入非原排列要求，必须保留并披露。'],
            'historical_gate_modified': False}

    def initial_review_history(self, candidates, reviews, sources, combined_doc):
        """Bind actual initial drafts/reviews; never infer first-pass from final PASS."""
        by_id = {c['candidate_id']: c for c in candidates}
        expected_bindings = []
        results = []
        allowed = ('PASS', 'HOLD', 'FAIL')
        for arm in ('a', 'b'):
            draft_path = HERE / f'author_{arm}/CANDIDATES_PRIVATE_DRAFT.json'
            final_path = HERE / f'author_{arm}/CANDIDATES_PRIVATE.json'
            initial_path = HERE / f'review_{arm}/INITIAL_CONTENT_REVIEW_AI.json'
            final_review_path = HERE / f'review_{arm}/CONTENT_REVIEW_AI.json'
            repair_path = HERE / f'author_{arm}/REPAIR_LOG.json'
            drafts, finals = read(draft_path), read(final_path)
            initial_doc, final_doc = read(initial_path), read(final_review_path)
            initial_rows = initial_doc if isinstance(initial_doc, list) else initial_doc['reviews']
            final_rows = final_doc if isinstance(final_doc, list) else final_doc['reviews']
            initial = {r['candidate_id']: r for r in initial_rows}
            final_reviews = {r['candidate_id']: r for r in final_rows}
            self.need(len(drafts) == len(finals) == len(initial_rows) == len(initial) == len(final_reviews) == 14,
                      'Each independent arm requires14 original/final/review slots', 'initial_history')
            self.need({c['candidate_id'] for c in drafts} == {c['candidate_id'] for c in finals} == set(initial) == set(final_reviews),
                      'Arm draft/final/review identity drift', 'initial_history')
            final_by = {c['candidate_id']: c for c in finals}
            repair = read(repair_path)
            self.need(repair.get('repair_pass', repair.get('repair_round')) == 1 and
                      repair.get('max_repair_passes', repair.get('maximum_repair_rounds')) == 1,
                      'Exactly one documented pre-execution repair round allowed', 'initial_history')
            self.need(repair.get('solver_outputs_read') is False or repair.get('no_solver_outputs_or_scores_read') is True,
                      'Repair Solver-blindness disclosure missing', 'initial_history')
            raw_initial_path = Path(repair['independent_initial_review_path'])
            self.need((raw_initial_path if raw_initial_path.is_absolute() else HERE / raw_initial_path).resolve() == initial_path.resolve(),
                      'Repair points to a different initial review', 'initial_history')
            for key in ('initial_draft', 'final_file'):
                if key in repair:
                    self.check_binding(repair[key], 'repair_file_binding')
                    self.need(repair[key] == binding(draft_path if key == 'initial_draft' else final_path),
                              'Repair file identity drift', 'repair_file_binding')
            changed_ids = set()
            semantic_pass = joint_pass = literal_path_mismatches = 0
            for draft in drafts:
                cid = draft['candidate_id']
                final, rv, assembled = final_by[cid], initial[cid], reviews[cid]
                final_rv = final_reviews[cid]
                self.need(final == by_id[cid], 'Assembled final candidate differs from author final', 'initial_history')
                self.need(rv['candidate_sha256'] == fingerprint(draft), 'Initial review is not bound to original draft ' + cid, 'initial_history')
                self.need(rv['author_id'] == draft['author_id'] == final['author_id'] and rv['reviewer_id'] != draft['author_id'],
                          'Initial author/reviewer separation drift ' + cid, 'initial_history')
                semantic, joint = rv['initial_content_decision'], rv['decision']
                self.need(semantic in allowed and joint in allowed and rv['initial_decision'] == joint,
                          'Initial semantic/joint review decision malformed', 'initial_history')
                source_file = Path(draft['source_binding']['path'])
                resolved_source = (source_file if source_file.is_absolute() else REPO / source_file).resolve()
                source = next(r for r in read(HERE / 'SELECTION_FREEZE.json')['roots'] if r['root_id'] == draft['root_id'])
                expected_source = {k: source[k] for k in ('path', 'sha256', 'bytes')}
                self.need(str(resolved_source) == source['path'] and all(draft['source_binding'][k] == source[k] for k in ('sha256', 'bytes')),
                          'Initial draft source does not resolve to frozen content', 'initial_history')
                literal_ok = draft['source_binding'] == expected_source
                literal_path_mismatches += not literal_ok
                self.need((joint == 'PASS') == (semantic == 'PASS' and literal_ok),
                          'Initial joint PASS was not content AND exact source binding ' + cid, 'initial_history')
                source_read = rv['source_read']
                self.need(all(source_read[k] == expected_source[k] for k in ('path', 'sha256', 'bytes')) and source_read['full_read'] is True,
                          'Initial source-read record mismatch', 'initial_history')
                self.need(source_read.get('turns', source_read.get('turn_count')) == len(sources[draft['root_id']]['turns']),
                          'Initial full-source-read turn count mismatch', 'initial_history')
                self.need(assembled['initial_decision'] == joint and assembled['initial_content_decision'] == semantic and
                          assembled['initial_candidate_sha256'] == fingerprint(draft) and assembled['initial_review_binding'] == binding(initial_path),
                          'Assembled initial accounting/provenance drift', 'initial_history')
                self.need(all(assembled[k] == v for k, v in final_rv.items() if k not in ('initial_decision', 'initial_content_decision')),
                          'Final review changed during assembly', 'initial_history')
                fields = changed_leaf_paths(draft, final)
                changed = bool(fields)
                self.need(assembled['repaired_after_initial_review'] == changed,
                          'Assembled repair flag mismatch', 'initial_history')
                if changed:
                    changed_ids.add(cid)
                    if 'repairs' in repair:
                        entries = [r for r in repair['repairs'] if r['candidate_id'] == cid]
                        self.need(len(entries) == 1, 'Changed candidate absent/duplicated in repair log', 'repair_diff')
                        entry = entries[0]
                        self.need(entry['before_candidate_sha256'] == fingerprint(draft) and entry['after_candidate_sha256'] == fingerprint(final),
                                  'Repair candidate fingerprint mismatch', 'repair_diff')
                        self.need(set(entry['changed_fields']) == set(fields), 'Repair changed-fields list mismatch', 'repair_diff')
                    else:
                        entries = [r for r in repair['changes'] if r['candidate_id'] == cid]
                        self.need({r['field'] for r in entries} == set(fields), 'Repair changed-fields list mismatch', 'repair_diff')
                        for entry in entries:
                            old, new = draft, final
                            for key in entry['field'].split('.'):
                                old, new = old[key], new[key]
                            self.need(old == entry['before'] and new == entry['after'], 'Logged repair before/after mismatch', 'repair_diff')
                semantic_pass += semantic == 'PASS'
                joint_pass += joint == 'PASS'
            logged_ids = {r['candidate_id'] for r in repair.get('repairs', repair.get('changes', []))}
            self.need(logged_ids == changed_ids, 'Repair log contains unobserved or missing changes', 'repair_diff')
            normalization = initial_doc.get('initial_review_accounting_normalization')
            if normalization:
                self.need(normalization['execution_started'] is False, 'Initial accounting normalized after execution', 'initial_history')
                self.check_binding(normalization['original_semantic_review'], 'original_semantic_review')
                original_path = Path(normalization['original_semantic_review']['path'])
                original = read(original_path)
                original_by = {r['candidate_id']: r for r in original['reviews']}
                self.need(set(original_by) == set(initial), 'Original semantic review identity drift', 'original_semantic_review')
                mutable = {'decision', 'initial_decision', 'decision_scope', 'initial_joint_hold_reasons'}
                for cid, rv in initial.items():
                    old = original_by[cid]
                    self.need({k: v for k, v in old.items() if k not in mutable} == {k: v for k, v in rv.items() if k not in mutable},
                              'Normalization changed semantic judgments/evidence', 'original_semantic_review')
                    self.need(old['decision'] == rv['initial_content_decision'], 'Original semantic decision not preserved', 'original_semantic_review')
                self.bindings.append(binding(original_path))
            results.append({'arm': arm, 'planned': 14, 'initial_semantic_content_pass': semantic_pass,
                            'initial_joint_content_and_binding_pass': joint_pass,
                            'initial_literal_source_binding_mismatches': literal_path_mismatches,
                            'single_repair_changed_candidate_ids': sorted(changed_ids)})
            paths = (draft_path, final_path, initial_path, final_review_path)
            expected_bindings.extend(binding(p) for p in paths)
            self.bindings.extend(binding(p) for p in (*paths, repair_path))
        self.need({b['path']: b for b in combined_doc['bindings']} == {b['path']: b for b in expected_bindings},
                  'Combined review provenance binding set differs', 'initial_history')
        self.details['initial_review_history'] = {'planned_denominator': 28, 'arms': results,
            'initial_semantic_content_pass': sum(r['initial_semantic_content_pass'] for r in results),
            'initial_joint_content_and_binding_pass': sum(r['initial_joint_content_and_binding_pass'] for r in results),
            'frozen_primary_threshold': .9, 'threshold_applied_to': 'initial_joint_content_and_binding_pass / 28',
            'semantic_only_rate_cannot_replace_frozen_primary_joint_rate': True,
            'original_drafts_and_initial_decisions_bound_not_inferred_from_final_pass': True,
            'one_documented_repair_round_each_arm': True,
            'semantic_truth_remains_ai_review_not_mechanical_certification': True}

    def candidates(self, sources):
        candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
        if isinstance(candidates, dict):
            candidates = candidates['candidates']
        doc = read(HERE / 'CONTENT_REVIEW_AI.json')
        reviews = {r['candidate_id']: r for r in doc['reviews']}
        ids = [c['candidate_id'] for c in candidates]
        self.need(len(ids) == len(set(ids)) == 28 and set(ids) == set(reviews),
                  'Expected28 unique candidates, including failed slots, and all reviews', 'inventory')
        self.need(set(c['root_id'] for c in candidates) == set(sources), 'Root coverage drift', 'inventory')
        selection = read(HERE / 'SELECTION_FREEZE.json')
        expected_ids = {f"QQS1-{root.split('-')[-1]}-{slot}" for root in sources for slot in selection['slots_per_root']}
        self.need(set(ids) == expected_ids, 'Frozen root-slot ID set drift', 'inventory')
        self.initial_review_history(candidates, reviews, sources, doc)
        signatures = []
        for candidate in candidates:
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            signatures.append(fingerprint({'question_type': candidate['question_type'],
                'question': candidate['question'], 'choices': sorted(candidate[field].values())}))
        self.need(len(signatures) == len(set(signatures)), 'Whole-question exact duplicate admitted', 'inventory')
        quotes = 0
        admitted = []
        per_candidate = []
        for candidate in candidates:
            cid = candidate['candidate_id']
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            choices = candidate[field]
            if '-MC' in cid:
                self.need(candidate['question_type'] == 'multi_select' and len(candidate['gold']) == int(cid.rsplit('-MC', 1)[1]), 'Slot Gold-count/type drift', 'inventory')
            else:
                self.need(candidate['question_type'] == 'ordering' and cid.rsplit('-', 1)[1] in ('ORDER1', 'ORDER2'), 'Ordering slot/type drift', 'inventory')
            source = sources[candidate['root_id']]
            refs = {t['turn_ref']: t for t in source['turns']}
            self.check_binding(candidate['source_binding'], 'candidate_source_binding')
            selection_root = next(r for r in read(HERE / 'SELECTION_FREEZE.json')['roots'] if r['root_id'] == candidate['root_id'])
            self.need(candidate['source_binding']['path'] == selection_root['path'] and candidate['source_binding']['sha256'] == selection_root['sha256'],
                      'Candidate outside selected source', 'candidate_source_binding')
            self.need((len(choices) == 6 and len(candidate['gold']) in (2, 3, 4, 5, 6)) if field == 'options' else 4 <= len(choices) <= 6 and set(candidate['gold']) == set(choices),
                      'Invalid Gold/item shape', 'inventory')
            self.need(len(candidate['gold']) == len(set(candidate['gold'])) and set(candidate['gold']) <= set(choices),
                      'Duplicate/unknown Gold', 'inventory')
            review = reviews[cid]
            self.need(review['candidate_sha256'] == fingerprint(candidate), 'Review fingerprint drift ' + cid, 'review')
            self.need(review['author_id'] == candidate['author_id'] and review['reviewer_id'] != candidate['author_id'],
                      'Author/reviewer separation ' + cid, 'review')
            self.need(review.get('gold_blind', doc.get('gold_blind')) is False,
                      'Non-blind review must be disclosed', 'review')
            decision = review.get('decision', review.get('status'))
            self.need(decision in ('PASS', 'HOLD', 'FAIL'), 'Unknown admission decision', 'review')
            if decision == 'PASS':
                admitted.append(candidate)
                self.need(review['independent_reconstructed_gold'] == candidate['gold'], 'Independent review Gold mismatch', 'review')
            item_notes = item_metadata(candidate, choices)
            for item_id in choices:
                self.need(bool(list(quote_nodes(item_notes[item_id]))), 'Item has no exact quote: ' + cid + '/' + item_id, 'quote_inventory')
            for origin, value in [('author', candidate), ('review', review)]:
                for quote in quote_nodes(value):
                    self.need(isinstance(quote['quote'], str) and bool(quote['quote']), 'Empty quote', 'exact_quote')
                    self.need(quote.get('turn_ref') in refs, 'Quote reference missing from source: ' + cid, 'exact_quote')
                    turn = refs[quote['turn_ref']]
                    self.need(quote.get('speaker') == turn['speaker'], 'Quote speaker mismatch: ' + cid, 'exact_quote')
                    self.need(quote['quote'] in turn['text'], 'Quote is not an exact substring: ' + cid, 'exact_quote')
                    quotes += 1
            per_candidate.append({'candidate_id': cid, 'decision': decision, 'gold_count': len(candidate['gold']),
                                  'question_type': candidate['question_type'], 'candidate_sha256': fingerprint(candidate)})
        root_coverage = []
        for root in sources:
            all_root = [c for c in candidates if c['root_id'] == root]
            root_pass = [c for c in admitted if c['root_id'] == root]
            hist = Counter(str(len(c['gold'])) for c in root_pass if c['question_type'] == 'multi_select')
            self.need(len(all_root) == 7 and sum(c['question_type'] == 'multi_select' for c in all_root) == 5,
                      'Root slot inventory is not 5MC+2order', 'inventory')
            root_coverage.append({'root_id': root, 'all_slots': len(all_root), 'admitted': len(root_pass),
                                  'admitted_gold_histogram': dict(sorted(hist.items())),
                                  'five_gold_buckets_complete': hist == Counter({'2': 1, '3': 1, '4': 1, '5': 1, '6': 1}),
                                  'admitted_ordering': sum(c['question_type'] == 'ordering' for c in root_pass)})
        global_hist = Counter(str(len(c['gold'])) for c in admitted if c['question_type'] == 'multi_select')
        self.details['candidate_audit'] = {'candidates': 28, 'admitted': len(admitted), 'failed_or_held': 28 - len(admitted),
                                           'global_admitted_gold_histogram': dict(sorted(global_hist.items())),
                                           'all_gold_buckets_have_four': global_hist == Counter({str(k): 4 for k in range(2, 7)}),
                                           'per_root': root_coverage, 'exact_quote_ref_speaker_checks': quotes,
                                           'all_slots_preserved': True, 'reviews': per_candidate,
                                           'semantic_truth_not_replaced_by_quote_match': True}
        for path in [HERE / 'CANDIDATES_PRIVATE.json', HERE / 'CONTENT_REVIEW_AI.json']:
            self.bindings.append(binding(path))
        return candidates, reviews, admitted

    def exhaustive_gate(self, candidates):
        gate = import_gate()
        rows = []
        for candidate in candidates:
            before = fingerprint(candidate)
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            canonical_ids = list(candidate[field])
            n = len(canonical_ids)
            labels = [chr(65 + index) for index in range(n)]
            # MC is set-scored, but every ordering of every nonempty subset is
            # legal output syntax. Include empty MC as a defensive extra case.
            if field == 'options':
                predictions = [list(p) for size in range(n + 1) for p in itertools.permutations(labels, size)]
                alternative_golds = [list(p) for size in range(2, 7) for p in itertools.combinations(canonical_ids, size)]
            else:
                predictions = [list(p) for p in itertools.permutations(labels)]
                alternative_golds = [list(p) for p in itertools.permutations(canonical_ids)]
            predictions_checked = 0
            gold_variations_checked = 0
            for seed in FIXED_SEEDS:
                public, private = gate.prepare_display(candidate, seed)
                mapping = private['display_to_canonical']
                inverse = private['canonical_to_display']
                self.need(mapping == own_mapping(candidate, seed), 'Independent RNG mismatch', 'exhaustive_mapping')
                self.need(set(mapping) == set(labels) and set(mapping.values()) == set(canonical_ids) and inverse == {v: k for k, v in mapping.items()},
                          'Mapping not bijective', 'exhaustive_mapping')
                self.need(list(mapping.values()) != canonical_ids and private['nonidentity_permutation'] is True,
                          'Identity display', 'exhaustive_mapping')
                self.need(public[field] == {label: candidate[field][key] for label, key in mapping.items()},
                          'Public display text drift', 'exhaustive_mapping')
                expected_gold = [inverse[g] for g in candidate['gold']]
                if field == 'options': expected_gold.sort()
                self.need(private['canonical_gold'] == candidate['gold'] and private['display_gold'] == expected_gold,
                          'Gold mapping drift', 'exhaustive_mapping')
                for prediction in predictions:
                    expected = [mapping[label] for label in prediction]
                    if field == 'options': expected.sort()
                    restored = gate.restore_prediction(private, prediction)
                    self.need(restored == expected, 'Prediction restoration mismatch', 'exhaustive_prediction')
                    roundtrip = [inverse[item] for item in restored]
                    self.need(set(roundtrip) == set(prediction) if field == 'options' else roundtrip == prediction,
                              'Prediction roundtrip mismatch', 'exhaustive_prediction')
                    correct = set(expected) == set(candidate['gold']) if field == 'options' else expected == candidate['gold']
                    self.need(gate.is_correct_display_answer(private, prediction) == correct,
                              'Gate exact scoring mismatch', 'exhaustive_prediction')
                    predictions_checked += 1
                # Strong Gold-independence check: all legal Gold sets/orders.
                # Gate serializes full candidate fingerprints; checking all
                # alternate Golds at all seeds is CPU-only and deterministic.
                for alt_gold in alternative_golds:
                    if alt_gold == candidate['gold']: continue
                    changed = copy.deepcopy(candidate)
                    changed['gold'] = alt_gold
                    public2, private2 = gate.prepare_display(changed, seed)
                    self.need(public2 == public and private2['display_to_canonical'] == mapping,
                              'Permutation/public view depends on Gold', 'exhaustive_gold_independence')
                    gold_variations_checked += 1
            self.need(fingerprint(candidate) == before, 'Candidate mutated by gate audit', 'candidate_immutability')
            rows.append({'candidate_id': candidate['candidate_id'], 'seeds_checked': len(FIXED_SEEDS),
                         'predictions_per_seed': len(predictions), 'predictions_checked': predictions_checked,
                         'alternate_legal_golds_per_seed': len(alternative_golds) - 1,
                         'alternate_gold_checks': gold_variations_checked,
                         'all_prediction_roundtrips_and_exact_scores_match': True,
                         'gold_changes_do_not_change_public_display': True,
                         'candidate_unchanged': True})
        self.details['exhaustive_gate'] = {'fixed_seeds': list(FIXED_SEEDS), 'candidate_count': len(candidates),
                                           'total_seed_cases': len(candidates) * len(FIXED_SEEDS),
                                           'all_legal_ordered_mc_subsets_including_empty_defensive_case': True,
                                           'all_complete_ordering_permutations': True,
                                           'predictions_checked': sum(r['predictions_checked'] for r in rows),
                                           'alternate_gold_checks': sum(r['alternate_gold_checks'] for r in rows),
                                           'candidates': rows}

    def prepared(self, candidates, reviews, admitted):
        from transformers import AutoTokenizer
        prep = read(HERE / 'execution/PREPARATION.json')
        public_path = HERE / 'execution/PUBLIC_REQUESTS.json'
        private_path = HERE / 'execution/PRIVATE_SCORING.json'
        tokens_path = HERE / 'execution/PROMPT_TOKEN_IDS.json'
        public = read(public_path)['requests']
        private = read(private_path)
        tokens = read(tokens_path)
        private_paths = {str(p.resolve()) for p in (HERE / 'CANDIDATES_PRIVATE.json', HERE / 'CONTENT_REVIEW_AI.json', private_path)}
        for bound in prep['bindings']:
            self.need(bound['path'] not in private_paths, 'Private file in generator bindings', 'public_private_isolation')
            self.check_binding(bound, 'preparation_binding')
        self.check_binding(prep['private_scoring_binding'], 'private_binding')
        for bound in private['bindings']:
            self.check_binding(bound, 'private_binding')
        self.need({b['path'] for b in private['bindings']} == private_paths - {str(private_path.resolve())},
                  'Private candidate/review binding set differs', 'private_binding')
        self.need(private['public_requests_binding'] == binding(public_path) and private['prompt_token_ids_binding'] == binding(tokens_path),
                  'Private/public/token chain mismatch', 'private_binding')
        ids = [r['request_id'] for r in public]
        self.need(len(ids) == len(set(ids)) == len(admitted) * 4 and set(ids) == set(tokens),
                  'Expected2 variants×2conditions per admitted item', 'prepared_inventory')
        self.need(prep['requests'] == len(ids) and prep['fixed_order'] == ids,
                  'Prepared order/count mismatch', 'prepared_inventory')
        self.need(prep['display_seed'] == 2026102302 and prep['variants'] == ['perm_00', 'perm_01'] and prep['conditions'] == ['FULL', 'QUESTION_ONLY'],
                  'Preparation seed/variant/condition drift', 'prepared_configuration')
        self.need(prep['max_output_tokens'] == 3072 and prep['max_model_len'] == 16384 and prep['temperature'] == .8 and prep['top_p'] == .9,
                  'Preparation inference budget drift', 'prepared_configuration')
        self.need(prep['model_role'] == 'B1' and prep['automatic_retry'] is False and prep['source_truncation'] is False and prep['native_hidden_thinking'] is False,
                  'Preparation protocol flags drift', 'prepared_configuration')
        self.need(prep['generation_reads_private_scoring'] is False and prep['independent_requests_no_shared_history_or_memory'] is True and prep['historical_scores_reusable'] is False,
                  'Preparation isolation/history flags drift', 'prepared_configuration')
        private_by = {r['request_id']: r for r in private['requests']}
        self.need([r['request_id'] for r in private['requests']] == ids, 'Private request order/set drift', 'private_binding')
        by_id = {c['candidate_id']: c for c in candidates}
        self.need(set(r['candidate_id'] for r in public) == {c['candidate_id'] for c in admitted},
                  'Request admission set mismatch', 'prepared_inventory')
        ledger = private['new_admission']
        self.need(len(ledger) == 28 and {r['candidate_id'] for r in ledger} == set(by_id),
                  'All28 planned slots must survive ledger', 'prepared_inventory')
        for item in ledger:
            c = by_id[item['candidate_id']]
            rv = reviews[item['candidate_id']]
            self.need(item['candidate_sha256'] == fingerprint(c) and item['decision'] == rv['decision'] and
                      item['initial_decision'] == rv['initial_decision'] and item['initial_content_decision'] == rv['initial_content_decision'],
                      'Admission/first-review ledger drift', 'review')
            self.need(item['root_id'] == c['root_id'] and item['author_id'] == c['author_id'] and item['reviewer_id'] == rv['reviewer_id'],
                      'Admission identity drift', 'review')
        self.need(private['submitted_new_drafts'] == 28 and private['planned_questions'] == {'new': 28},
                  'Planned candidate denominator drift', 'prepared_inventory')
        histogram = Counter(str(len(c['gold'])) for c in admitted if c['question_type'] == 'multi_select')
        self.need(private['new_mc_coverage']['gold_count_histogram'] == {str(k): histogram[str(k)] for k in range(2, 7)},
                  'Gold coverage histogram drift', 'coverage')
        self.need(private['new_mc_coverage']['minimum_per_count'] == 4 and private['new_mc_coverage']['coverage_complete'] == all(histogram[str(k)] >= 4 for k in range(2, 7)),
                  'Gold coverage verdict drift', 'coverage')
        for root, coverage in private['coverage_by_root'].items():
            rh = Counter(str(len(c['gold'])) for c in admitted if c['root_id'] == root and c['question_type'] == 'multi_select')
            self.need(coverage['gold_count_histogram'] == {str(k): rh[str(k)] for k in range(2, 7)} and coverage['coverage_complete'] == all(rh[str(k)] >= 1 for k in range(2, 7)),
                      'Root coverage drift', 'coverage')
        source_cache = {}
        expected_displays = {}
        variant_audit = []
        for candidate in admitted:
            cid = candidate['candidate_id']
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            seen = set()
            variants = {}
            for index, variant in enumerate(('perm_00', 'perm_01')):
                seed = 2026102302 + index
                skipped = []
                while True:
                    mapping = own_mapping(candidate, seed)
                    permutation = tuple(mapping.values())
                    if permutation not in seen:
                        break
                    skipped.append(seed)
                    seed += 2
                    self.need(len(skipped) < 10000, 'Unexpected collision loop', 'display_variant')
                seen.add(permutation)
                variants[variant] = {'mapping': mapping, 'seed': seed, 'skipped': skipped}
            self.need(len(seen) == 2, 'Two actual displays are identical', 'display_variant')
            expected_displays[cid] = variants
            rrs = [r for r in public if r['candidate_id'] == cid]
            self.need(len(rrs) == 4 and {(r['variant_id'], r['condition']) for r in rrs} == {(v, c) for v in ('perm_00', 'perm_01') for c in ('FULL', 'QUESTION_ONLY')},
                      'Missing candidate variant/condition', 'prepared_inventory')
            variant_audit.append({'candidate_id': cid, 'variants': variants,
                                  'distinct': True, 'selection_uses_gold': False})
        constants = literal_constants(WEEK / 'postweek_diversity_reason_v1/run.py')
        operations = dict(constants['OPERATIONS'])
        operations.update(literal_constants(WEEK / 'postweek_question_quality_pilot_v1/run.py')['OPERATION_OVERRIDES'])
        common = constants['COMMON']
        qonly_task = None
        for node in ast.parse((WEEK / 'postweek_question_quality_pilot_v1/run.py').read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name == 'question_only_prompt':
                for item in node.body:
                    if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'task' for t in item.targets):
                        qonly_task = ast.literal_eval(item.value)
        self.need(isinstance(qonly_task, str), 'QUESTION_ONLY template not found', 'prompt')
        endpoint = read(WEEK / 'day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json')
        tokenizer = AutoTokenizer.from_pretrained(endpoint['tokenizer_directory'], local_files_only=True)
        public_keys = PUBLIC_BASE_KEYS | {'request_id', 'cohort', 'condition', 'display', 'labels',
                      'sampling_seed', 'prompt_path', 'schema_path', 'prompt_tokens', 'variant_id'}
        private_gold_leaks = []
        chance_answer_order = []
        prompt_counts = []
        for row in public:
            rid = row['request_id']
            candidate = by_id[row['candidate_id']]
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            pres = private_by[rid]['presentation']
            expected = expected_displays[candidate['candidate_id']][row['variant_id']]
            mapping = expected['mapping']
            inverse = {key: label for label, key in mapping.items()}
            self.need(set(row) == public_keys and row['cohort'] == 'new', 'Public field allowlist/cohort drift', 'public_private_isolation')
            self.need(rid == f"{candidate['candidate_id']}__{row['variant_id']}__{row['condition']}__B1_stability_v1", 'Request ID drift', 'prepared_inventory')
            self.need(all(row[k] == candidate[k] for k in ('candidate_id', 'root_id', 'question_type', 'task_operation', 'question', 'source_binding')),
                      'Public question metadata drift', 'public_private_isolation')
            self.need(pres['display_to_canonical'] == mapping and pres['canonical_to_display'] == inverse and pres['seed'] == expected['seed'] and pres['duplicate_display_seed_skips'] == expected['skipped'],
                      'Variant permutation/collision protocol drift', 'display_variant')
            self.need(list(mapping.values()) != list(candidate[field]) and pres['nonidentity_permutation'] is True,
                      'Actual display equals original author order', 'display_variant')
            self.need(row['display'] == {label: candidate[field][key] for label, key in mapping.items()} and row['labels'] == list(mapping),
                      'Displayed text/labels drift', 'display_variant')
            self.need(pres['candidate_sha256'] == fingerprint(candidate) and pres['canonical_gold'] == candidate['gold'],
                      'Private candidate/Gold drift', 'private_binding')
            display_gold = [inverse[g] for g in candidate['gold']]
            if field == 'options': display_gold.sort()
            self.need(pres['display_gold'] == display_gold, 'Private displayed Gold drift', 'private_binding')
            public_view = {k: row[k] for k in PUBLIC_BASE_KEYS}
            public_view[field] = row['display']
            self.need(pres['public_sha256'] == fingerprint(public_view), 'Public fingerprint drift', 'private_binding')
            seed = int.from_bytes(hashlib.sha256(('quality_stability_v1:' + candidate['candidate_id']).encode()).digest()[:4], 'big') % 2147483647
            self.need(row['sampling_seed'] == seed, 'Sampling seed drift', 'display_variant')
            self.need(private_by[rid]['variant_id'] == row['variant_id'] and private_by[rid]['parent_candidate_id'] is None and private_by[rid]['strictly_paired'] is False,
                      'Private variant identity drift', 'private_binding')
            if field == 'items' and display_gold == row['labels']:
                chance_answer_order.append({'request_id': rid, 'display_gold': display_gold,
                                            'note': 'Retained Gold-independent shuffle may coincide with answer order; no Gold-conditioned redraw.'})
            source_path = row['source_binding']['path']
            if source_path not in source_cache:
                self.check_binding(row['source_binding'], 'prompt_source_binding')
                source_cache[source_path] = read(source_path)
            source = source_cache[source_path]
            refs = [t['turn_ref'] for t in source['turns']]
            instruction = operations[row['task_operation']]
            suffix = (f"Question ({row['question_type']}):\n{row['question']}\n\nDisplayed items:\n" +
                      '\n'.join(f'{label}. {text}' for label, text in row['display'].items()) +
                      f"\n\nRequested operation: {row['task_operation']}\n{instruction}\n\n")
            if row['condition'] == 'FULL':
                dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
                user = 'Dialogue:\n' + dialogue + '\n\n' + suffix + common
                system = 'Use only the supplied dialogue. Follow the question\'s requested operation. Give concise, verifiable explanations before the final boxed answer.'
                refs_schema = {'type': 'array', 'maxItems': 3, 'items': {'type': 'string', 'enum': refs}}
            else:
                self.need(row['condition'] == 'QUESTION_ONLY', 'Unknown input condition', 'prompt')
                user = suffix + qonly_task
                system = 'Make a best prediction from the question and displayed items. No dialogue is supplied. Never fabricate source references. Give a brief explanation and final boxed answer.'
                refs_schema = {'type': 'array', 'maxItems': 0, 'items': {'type': 'string'}}
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.need(prompt == Path(row['prompt_path']).read_text(), 'Exact public prompt reconstruction failed: ' + rid, 'prompt_reconstruction')
            recomputed_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            self.need(recomputed_tokens == tokens[rid] and len(recomputed_tokens) == row['prompt_tokens'], 'Exact token reencoding failed', 'token_reencoding')
            self.need(row['prompt_tokens'] + 3072 <= 16384, 'Context overflow', 'token_reencoding')
            prompt_counts.append(row['prompt_tokens'])
            item_schema = {'type': 'object', 'additionalProperties': False,
                'required': ['judgment', 'turn_refs', 'reason'], 'properties': {
                'judgment': {'type': 'string', 'enum': ['supported', 'contradicted', 'not_supported', 'ambiguous']},
                'turn_refs': refs_schema, 'reason': {'type': 'string', 'minLength': 1, 'maxLength': 320}}}
            schema = {'type': 'object', 'additionalProperties': False,
                'required': ['item_evidence', 'answer_reason', 'final_answer'], 'properties': {
                'item_evidence': {'type': 'object', 'additionalProperties': False, 'required': row['labels'],
                    'properties': {label: copy.deepcopy(item_schema) for label in row['labels']}},
                'answer_reason': {'type': 'string', 'minLength': 1, 'maxLength': 800},
                'final_answer': {'type': 'string', 'pattern': r'^<box>[A-F]( [A-F]){0,5}</box>$'}}}
            self.need(schema == read(row['schema_path']), 'Schema reconstruction failed', 'schema_reconstruction')
            for path in (row['prompt_path'], row['schema_path']):
                self.need(binding(path) in prep['bindings'], 'Unbound prompt/schema', 'preparation_binding')
        for path in (public_path, tokens_path, HERE / 'run.py', WEEK / 'postweek_question_quality_pilot_v1/run.py'):
            self.need(binding(path) in prep['bindings'], 'Required file not bound', 'preparation_binding')
        self.details['prepared_input_audit'] = {
            'frozen_requests': len(public), 'planned_candidates': 28, 'admitted_candidates': len(admitted),
            'all_28_ledger_entries_retained': True, 'per_variant_admitted_questions': len(admitted),
            'per_condition_requests': len(admitted) * 2,
            'all_prompts_schemas_and_token_ids_independently_reconstructed': True,
            'private_gold_author_explanations_not_in_model_prompt_or_schema': True,
            'full_all_source_turns_included': True, 'question_only_no_hidden_dialogue_or_reference_enum': True,
            'question_only_all_turn_refs_must_be_empty': True,
            'max_prompt_tokens': max(prompt_counts), 'min_prompt_tokens': min(prompt_counts),
            'max_output_tokens': 3072, 'context_limit': 16384,
            'max_combined_tokens': max(prompt_counts) + 3072,
            'sampling_seed_same_for_all_four_requests_per_candidate': True,
            'two_displays_distinct_relative_to_each_other_and_original_author_order': True,
            'seed_collision_skips_inspect_only_permutation_not_gold': True,
            'display_variants': variant_audit, 'chance_gold_equals_display_order': chance_answer_order,
            'private_scoring_mode': oct(private_path.stat().st_mode & 0o777),
            'generation_worker_isolation': {'historical_base_module_instance_adapted_only_in_memory': True,
                'base_out_redirected_to_new_execution': True,
                'base_public_preparation_redirected_to_new_public_allowlist': True,
                'score_function_original_captured_before_instance_override': True,
                'base_run_and_analyze_write_only_new_out': True,
                'base_new_review_historical_globals_not_read_on_run_analyze_paths': True,
                'generation_gold_not_read_checked_by_code_and_input_reconstruction': True,
                'os_level_file_access_trace_available': False,
                'disclosure_zh': '共享文件系统不构成技术盲法；结论基于冻结代码、public/private白名单与精确输入重构，未声称操作系统级读文件追踪。'},
            'all_28_evaluation_requests_only_if_all_28_admitted': len(admitted) == 28,
            'held_slots_in_content_denominator': 28,
            'no_semantic_truth_or_difficulty_guarantee_from_this_mechanical_pass': True}
        for path in (HERE / 'run.py', public_path, private_path, tokens_path, HERE / 'execution/PREPARATION.json', HERE / 'build_report.py'):
            self.bindings.append(binding(path))
        return public, private, tokens

    def execution(self, rows, private, tokens):
        from jsonschema import validate as json_validate
        out = HERE / 'execution'
        prep = read(out / 'PREPARATION.json')
        start = read(out / 'STARTED.json')
        frozen = read(out / 'RAW_RESPONSE_FREEZE.json')
        ids = [r['request_id'] for r in rows]
        self.need(frozen['requests'] == start['requests'] == len(rows) and frozen['request_ids'] == start['request_ids'] == ids,
                  'Raw/STARTED count/order drift', 'raw_seal')
        self.need(start['preparation'] == binding(out / 'PREPARATION.json') and start['private_scoring_binding'] == prep['private_scoring_binding'] and frozen['started_binding'] == binding(out / 'STARTED.json'),
                  'Raw seal chain drift', 'raw_seal')
        self.need(start['gold_read'] is False and start['automatic_retry'] is False and frozen['automatic_retry'] is False,
                  'Runtime Gold/retry flags drift', 'raw_seal')
        expected_paths = {str((out / 'raw' / (rid + '.json')).resolve()) for rid in ids}
        self.need({b['path'] for b in frozen['bindings']} == expected_paths and len(frozen['bindings']) == len(ids),
                  'Raw seal file set drift', 'raw_seal')
        self.need({str(p.resolve()) for p in (out / 'raw').glob('*.json')} == expected_paths,
                  'Raw directory set drift', 'raw_seal')
        for bound in frozen['bindings'] + frozen['validation_bindings']:
            self.check_binding(bound, 'raw_sealed_binding')
        private_by = {r['request_id']: r for r in private['requests']}
        raw_records = {rid: read(out / 'raw' / (rid + '.json')) for rid in ids}
        self.need(frozen['returned_requests'] == sum(r['status'] == 'RETURNED' for r in raw_records.values()),
                  'Returned-count mismatch', 'raw_seal')
        records = []
        invalid_details = []
        backend_errors = 0
        truncations = 0
        for row in rows:
            rid = row['request_id']
            raw = raw_records[rid]
            scoring = private_by[rid]
            pres = scoring['presentation']
            results = raw.get('returned_results', [])
            capture_errors = []
            if raw.get('returned_result_count') != 1 or len(results) != 1:
                capture_errors.append('RETURNED_RESULT_COUNT_MISMATCH')
            for result in results:
                if result.get('prompt') != Path(row['prompt_path']).read_text(): capture_errors.append('RETURNED_PROMPT_DRIFT')
                if result.get('prompt_token_ids') != tokens[rid]: capture_errors.append('PROMPT_TOKEN_IDS_DRIFT')
                if len(result.get('outputs', [])) != 1: capture_errors.append('COMPLETION_COUNT_MISMATCH')
            if results and results[0].get('outputs'):
                comp = results[0]['outputs'][0]
                for key in ('raw_output', 'finish_reason', 'stop_reason', 'output_token_ids'):
                    if raw.get(key) != comp.get(key): capture_errors.append('RAW_COMPLETION_FIELD_DRIFT:' + key)
                if raw.get('prompt_token_ids') != results[0].get('prompt_token_ids'): capture_errors.append('RAW_PROMPT_TOKEN_FIELD_DRIFT')
            validation_path = out / 'raw_validation' / (rid + '.json')
            if validation_path.exists():
                validation = read(validation_path)
                self.need(validation['errors'] == capture_errors and validation['raw_binding'] == binding(out / 'raw' / (rid + '.json')),
                          'Capture validator result mismatch', 'capture_validation')
            parsed, answer, valid, error = None, [], True, None
            try:
                assert raw['request_id'] == rid, 'RAW_REQUEST_ID_DRIFT'
                assert raw['status'] == 'RETURNED', 'BACKEND_OR_SUBMISSION_FAILURE:' + raw['status']
                assert not capture_errors, 'RETURNED_RESPONSE_DRIFT'
                assert raw['finish_reason'] == 'stop', 'TRUNCATED_OR_NONSTOP_COMPLETION'
                parsed = json.loads(raw['raw_output'])
                json_validate(parsed, read(row['schema_path']))
                assert list(parsed) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
                assert parsed['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
                for item in parsed['item_evidence'].values():
                    assert item['reason'].strip(), 'EMPTY_ITEM_REASON'
                    if row['condition'] == 'FULL':
                        assert len(item['turn_refs']) == len(set(item['turn_refs'])), 'DUPLICATE_REFERENCE'
                        assert item['judgment'] != 'supported' or item['turn_refs'], 'SUPPORTED_WITHOUT_CITATION'
                    else:
                        assert item['turn_refs'] == [], 'QUESTION_ONLY_REFERENCE_FORBIDDEN'
                match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>', parsed['final_answer'])
                assert match, 'ANSWER_BOX_REQUIRED'
                answer = match.group(1).split()
                assert len(answer) == len(set(answer)) and set(answer) <= set(row['labels']), 'INVALID_LABEL_SET'
                if row['question_type'] == 'ordering':
                    assert set(answer) == set(row['labels']), 'ORDERING_INCOMPLETE'
                    if row['condition'] == 'FULL':
                        assert all(v['turn_refs'] for v in parsed['item_evidence'].values()), 'NO_ANCHOR_FOR_ORDERING'
            except Exception as exc:
                valid, error = False, str(exc)
            mapping = pres['display_to_canonical']
            gold = pres['canonical_gold']
            prediction = [mapping[a] for a in answer] if valid else []
            if row['question_type'] == 'multi_select': prediction.sort()
            exact = bool(valid and (set(prediction) == set(gold) if row['question_type'] == 'multi_select' else prediction == gold))
            record = {key: row[key] for key in ('request_id', 'candidate_id', 'root_id', 'cohort', 'condition', 'question_type', 'task_operation')}
            record.update(valid=valid, exact=exact, prediction=prediction, gold=gold,
                display_prediction=answer if valid else [], display_gold=pres['display_gold'],
                gold_count=len(gold) if row['question_type'] == 'multi_select' else None,
                false_positive=None, false_negative=None, pairwise_correct=None, pairwise_total=None, allselect_exact=None)
            if row['question_type'] == 'multi_select':
                record.update(false_positive=len(set(prediction) - set(gold)), false_negative=len(set(gold) - set(prediction)),
                              allselect_exact=len(gold) == len(mapping))
            else:
                positions = {v: index for index, v in enumerate(prediction)}
                pairs = [(g, h) for index, g in enumerate(gold) for h in gold[index + 1:]]
                record.update(pairwise_correct=sum(positions[g] < positions[h] for g, h in pairs) if valid else 0,
                              pairwise_total=len(pairs))
            record.update(variant_id=row['variant_id'], parent_candidate_id=None, strictly_paired=False,
                          raw_binding=binding(out / 'raw' / (rid + '.json')), invalid_error=error)
            records.append(record)
            if not valid:
                backend_errors += raw['status'] != 'RETURNED'
                truncations += raw.get('finish_reason') == 'length'
                final_answer = parsed.get('final_answer') if isinstance(parsed, dict) else None
                box = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>', final_answer or '')
                partial = box.group(1).split() if box else []
                invalid_details.append({'request_id': rid, 'error': error,
                    'condition': row['condition'], 'variant_id': row['variant_id'], 'question_type': row['question_type'],
                    'raw_status': raw['status'], 'finish_reason': raw.get('finish_reason'),
                    'output_tokens': len(raw.get('output_token_ids', [])), 'parseable_json': isinstance(parsed, dict),
                    'raw_boxed_final_answer': final_answer, 'has_parseable_boxed_answer': bool(box),
                    'raw_labels': partial, 'missing_display_labels': [a for a in row['labels'] if a not in partial],
                    'all_required_labels_present': set(partial) == set(row['labels']),
                    'not_semantic_difficulty_evidence': True, 'zero_score_and_denominator_retained': True})
        runner_scores = read(out / 'SCORES.json')
        self.need(records == runner_scores['records'], 'Independent per-request score disagreement', 'independent_score')
        self.need(runner_scores['raw_freeze_binding'] == binding(out / 'RAW_RESPONSE_FREEZE.json') and runner_scores['private_scoring_binding'] == prep['private_scoring_binding'] and runner_scores['gold_read_after_all_raw_sealed'] is True,
                  'Scoring seal provenance drift', 'independent_score')
        own = independent_summary(records, private)
        runner_summary = read(out / 'ANALYSIS_SUMMARY.json')
        self.need(own == runner_summary, 'Independent full summary disagreement', 'independent_summary')
        self.need(sum(c['frozen_requests'] for c in own['cells']) == len(records),
                  'Some frozen failures excluded from summary', 'independent_summary')
        pair_stats = {}
        for condition in ('FULL', 'QUESTION_ONLY'):
            pairs = [p for p in own['variant_pairs'] if p['condition'] == condition]
            pair_stats[condition] = {'candidate_denominator': len(pairs),
                'both_valid': sum(p['both_valid'] for p in pairs),
                'canonical_prediction_same': sum(p['canonical_prediction_same'] for p in pairs),
                'both_exact': sum(p['both_exact'] for p in pairs),
                'accuracy_changed': sum(p['accuracy_changed'] for p in pairs),
                'exact_unchanged_but_prediction_changed_among_both_valid': sum(p['both_valid'] and not p['canonical_prediction_same'] and not p['accuracy_changed'] for p in pairs)}
        mc6 = {r['candidate_id'] for r in records if r['gold_count'] == 6}
        qonly_both = own['question_only_both_variants_correct']
        self.details['independent_execution_audit'] = {'frozen_requests': len(records),
            'returned_requests': frozen['returned_requests'], 'submitted_requests': frozen['submitted_requests'],
            'valid': sum(r['valid'] for r in records), 'invalid': len(invalid_details), 'correct': sum(r['exact'] for r in records),
            'all_per_request_predictions_exact_fp_fn_pairwise_equal_runner': True,
            'full_analysis_summary_equal_runner': True, 'planned_candidate_denominator': 28,
            'initial_content_pass_denominator': 28, 'per_variant_denominator': own['admitted_questions'],
            'initial_joint_content_and_binding_pass': own['initial_content_pass'],
            'initial_semantic_content_pass': own['initial_semantic_content_pass'],
            'initial_primary_threshold_uses_joint_not_semantic_only_rate': True,
            'per_condition_request_denominator': own['admitted_questions'] * 2,
            'all_frozen_failures_retained_in_exact_and_pairwise_denominators': True,
            'invalid_mc_prediction_for_descriptive_fp_fn': 'empty_set', 'invalid_ordering_pairwise': 0,
            'automatic_retry': False, 'backend_error_count': backend_errors, 'truncation_count': truncations,
            'invalid_details': invalid_details, 'variant_pair_statistics': pair_stats,
            'question_only_both_variants_correct_non_six_gold': [cid for cid in qonly_both if cid not in mc6],
            'question_only_both_variants_correct_six_gold_structural_baseline': [cid for cid in qonly_both if cid in mc6],
            'six_gold_allselect_is_structural_baseline_not_distractor_or_extra_shortcut_evidence': True,
            'no_inference_of_internal_model_strategy_from_visible_reasons': True,
            'no_causal_position_effect_or_formal_weakness_claim': True,
            'independently_reconstructed_summary': own, 'independently_reconstructed_scores': records}
        for path in (out / 'STARTED.json', out / 'RAW_RESPONSE_FREEZE.json', out / 'SCORES.json', out / 'ANALYSIS_SUMMARY.json'):
            self.bindings.append(binding(path))

    def report(self, stage, status):
        return {'schema_version': 'question_quality_stability_mechanical_audit_v1',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'reviewer_id': '/root/quality_sort_review', 'reviewer_type': 'codex_ai_agent',
                'stage': stage, 'status': status,
                'cpu_only': True, 'new_model_calls': 0, 'new_training_runs': 0,
                'runner_modules_imported': False, 'runner_validation_or_scoring_functions_called': False,
                'gate_is_system_under_test': True,
                'candidate_semantic_truth_or_difficulty_certified': False,
                'historical_files_or_candidates_modified': False,
                'checks_by_group': dict(self.counts), 'checks': sum(self.counts.values()),
                **self.details, 'bindings': self.bindings + [binding(GATE), binding(Path(__file__)),
                    binding(HERE / 'SELECTION_FREEZE.json'), binding(HERE / 'PROTOCOL_ZH.md')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['initial', 'candidates', 'prepared', 'execution'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parent-raw-seal-notified', action='store_true')
    args = parser.parse_args()
    audit = Audit()
    selection, sources = audit.sources()
    audit.baseline(create=args.stage == 'initial')
    audit.gate_compatibility()
    if args.stage != 'initial':
        candidates, reviews, admitted = audit.candidates(sources)
        if args.stage == 'execution':
            audit.need(args.parent_raw_seal_notified, 'Parent seal notification flag required', 'raw_access_authorization')
            mechanical_path = AUDIT / 'MECHANICAL_AUDIT.json'
            mechanical = read(mechanical_path)
            audit.need(mechanical['status'].startswith('PASS'), 'Mechanical audit must pass before execution audit', 'mechanical_preflight')
            for bound in mechanical['bindings']:
                audit.check_binding(bound, 'mechanical_preflight_binding')
            audit.details['exhaustive_gate'] = mechanical['exhaustive_gate']
            audit.details['exhaustive_gate']['reused_bound_pre_execution_mechanical_proof'] = True
            audit.bindings.append(binding(mechanical_path))
        else:
            audit.exhaustive_gate(candidates)
        if args.stage in ('prepared', 'execution'):
            rows, private, tokens = audit.prepared(candidates, reviews, admitted)
        if args.stage == 'execution':
            audit.execution(rows, private, tokens)
    audit.baseline()
    statuses = {'initial': 'PASS_INITIAL_INTERFACE_AND_HISTORICAL_FREEZE',
                'candidates': 'PASS_CANDIDATE_MECHANICAL_AUDIT',
                'prepared': 'PASS_MECHANICAL_INPUT_AND_EXHAUSTIVE_GATE_AUDIT',
                'execution': 'PASS_INDEPENDENT_EXECUTION_AND_SCORING_AUDIT'}
    report = audit.report(args.stage, statuses[args.stage])
    defaults = {'initial': 'INITIAL_AUDIT_AI.json', 'candidates': 'CANDIDATE_AUDIT.json',
                'prepared': 'MECHANICAL_AUDIT.json', 'execution': 'EXECUTION_AUDIT.json'}
    output = args.output or AUDIT / defaults[args.stage]
    save_new(output, report)
    print(json.dumps({'report': binding(output), 'stage': args.stage, 'checks': report['checks'], 'status': statuses[args.stage]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
