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


def item_metadata(candidate, item_ids):
    for key in ('evidence_and_author_notes', 'item_evidence', 'evidence_by_item', 'evidence'):
        value = candidate.get(key)
        if isinstance(value, dict) and set(value) == set(item_ids):
            return value
    for value in candidate.values():
        if isinstance(value, dict) and set(value) == set(item_ids) and all(list(quote_nodes(item)) for item in value.values()):
            return value
    raise AssertionError('No complete per-item evidence mapping: ' + candidate['candidate_id'])


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
        quotes = 0
        admitted = []
        per_candidate = []
        for candidate in candidates:
            cid = candidate['candidate_id']
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            choices = candidate[field]
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
        # Filled against the frozen runner contract once the preparation exists;
        # no runner functions are imported or invoked by this module.
        prep = read(HERE / 'execution/PREPARATION.json')
        public_path = HERE / 'execution/PUBLIC_REQUESTS.json'
        private_path = HERE / 'execution/PRIVATE_SCORING.json'
        tokens_path = HERE / 'execution/PROMPT_TOKEN_IDS.json'
        public = read(public_path)['requests']
        private = read(private_path)
        tokens = read(tokens_path)
        for bound in prep['bindings']:
            self.check_binding(bound, 'preparation_binding')
        self.check_binding(prep['private_scoring_binding'], 'private_binding')
        for bound in private['bindings']:
            self.check_binding(bound, 'private_binding')
        self.need(private['public_requests_binding'] == binding(public_path) and private['prompt_token_ids_binding'] == binding(tokens_path),
                  'Private/public/token chain mismatch', 'private_binding')
        ids = [r['request_id'] for r in public]
        self.need(len(ids) == len(set(ids)) == len(admitted) * 4 and set(ids) == set(tokens),
                  'Frozen requests do not cover2 variants×2conditions for every admitted item', 'prepared_inventory')
        self.need(prep['requests'] == len(ids) and prep['fixed_order'] == ids,
                  'Prepared order/count mismatch', 'prepared_inventory')
        # Detailed interface verification is deliberately fail-closed until the
        # stability runner's actual variant and seed contract has been reviewed.
        raise NotImplementedError('Prepared interface requires frozen stability runner contract review; do not treat bindings-only as PASS')

    def execution(self):
        raise NotImplementedError('Execution audit requires parent notification that all raw are sealed and scoring complete')

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
        audit.exhaustive_gate(candidates)
        if args.stage in ('prepared', 'execution'):
            audit.prepared(candidates, reviews, admitted)
        if args.stage == 'execution':
            audit.need(args.parent_raw_seal_notified, 'Parent seal notification flag required', 'raw_access_authorization')
            audit.execution()
    audit.baseline()
    status = 'PASS_INITIAL_INTERFACE_AND_HISTORICAL_FREEZE' if args.stage == 'initial' else 'PASS_MECHANICAL_STAGE'
    report = audit.report(args.stage, status)
    output = args.output or AUDIT / (args.stage.upper() + '_AUDIT_AI.json')
    save_new(output, report)
    print(json.dumps({'report': binding(output), 'stage': args.stage, 'checks': report['checks'], 'status': status}, ensure_ascii=False))


if __name__ == '__main__':
    main()
