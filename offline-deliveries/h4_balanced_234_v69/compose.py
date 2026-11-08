"""Compose v69 questions offline from immutable independently reviewed assets."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random

VERSION = 'v69_verified_immutable_composition_v1'
MODEL = 'Qwen/Qwen3.6-35B-A3B'
COUNTS = tuple(range(2, 7))
TARGET = {str(k): 2 for k in COUNTS}
OUTPUTS = ('POOL.json', 'ALLOCATION.json', 'PUBLIC_QUESTIONS.json')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def norm(text):
    return ' '.join(text.casefold().split())


def statement_key(root, statements, kind='multi_select'):
    return (root, kind, tuple(sorted(norm(s) for s in statements)))


def _rows(obj):
    if isinstance(obj, list):
        return obj
    require(isinstance(obj, dict), 'prior_pool_invalid')
    return obj.get('questions', obj.get('candidates', []))


def prior_keys(obj, root):
    result = set()
    for original in _rows(obj):
        row = original.get('original_selected_question', original)
        if row.get('root_id', root) != root or row.get('question_type') != 'multi_select':
            continue
        if 'hard_dedup' in row and row['hard_dedup'] is not True:
            continue
        statements = row.get('options', row.get('facts'))
        if isinstance(statements, dict):
            statements = list(statements.values())
        if isinstance(statements, list) and len(statements) == 6 and all(isinstance(s, str) for s in statements):
            result.add(statement_key(root, statements))
    return result


def _unique(rows, name):
    require(isinstance(rows, list), name + '_not_list')
    require(all(isinstance(r, dict) and isinstance(r.get('version_id'), str) for r in rows), name + '_bad_version')
    result = {r['version_id']: r for r in rows}
    require(len(result) == len(rows), name + '_duplicate_version')
    return result


def _bound(inputs, path):
    path = Path(path).resolve()
    require(path.is_file(), 'missing_input:' + str(path))
    value = sha(path)
    require(str(path) not in inputs or inputs[str(path)] == value, 'input_changed_while_reading:' + str(path))
    inputs[str(path)] = value
    return read(path)


def _check_binding(obj, field, path):
    require(obj.get(field) == sha(path), 'binding_mismatch:' + field)


def load_slot(report, slot, inputs):
    """Require public-first judgments; never repair a writer's incorrect relation."""
    run = report / 'assets' / slot['slot_id']
    manifest = _bound(inputs, run / 'MANIFEST.json')
    complete = _bound(inputs, run / 'GENERATION.COMPLETE.json')
    source = _bound(inputs, run / 'SOURCE.json')
    for value in (manifest, complete):
        require(value.get('root_id') == slot['root_id'] and value.get('slot_id') == slot['slot_id'], 'mixed_root_or_slot')
        require(value.get('training_allowed') is False, 'training_must_be_disabled')
    require(manifest.get('slot') == slot, 'manifest_slot_changed')
    require(manifest.get('config') == {'path': str(report / 'CONFIG.json'), 'sha256': sha(report / 'CONFIG.json')}, 'manifest_config_changed')
    require(manifest.get('source_path') == str(run / 'SOURCE.json'), 'manifest_source_path_changed')
    _check_binding(manifest, 'source_sha256', run / 'SOURCE.json')
    require(source.get('root_id') == slot['root_id'], 'mixed_source_root')
    require(complete.get('semantic_retries') == 0 and complete.get('status') in ('ASSETS_READY', 'INSUFFICIENT_FACTS', 'NO_VALID_ASSETS'), 'generation_completion_invalid')
    # Bind every frozen generation artifact/dependency, including requests and response.
    for obj in (manifest, complete):
        require(isinstance(obj.get('files'), list), 'generation_file_bindings_missing')
        for binding in obj['files']:
            path = Path(binding['path']).resolve()
            require(path.is_file() and sha(path) == binding['sha256'], 'generation_bound_file_changed:' + str(path))
            inputs[str(path)] = binding['sha256']
    prior = set()
    for path in slot.get('prior_pool_paths', []):
        prior |= prior_keys(_bound(inputs, path), slot['root_id'])
    if (run / 'PRIOR.json').exists():
        prior |= prior_keys(_bound(inputs, run / 'PRIOR.json'), slot['root_id'])
    if complete['status'] == 'NO_VALID_ASSETS':
        require(not (run / 'ASSETS.json').exists() and complete.get('fact_count') == 0, 'failed_generation_has_assets')
        return {'slot': slot, 'versions': [], 'source_path': str(run / 'SOURCE.json'), 'source_sha256': sha(run / 'SOURCE.json'),
                'stem': None, 'prior': prior, 'status': 'NO_VALID_ASSETS', 'input_dir': str(run)}
    assets = _bound(inputs, run / 'ASSETS.json')
    public = _bound(inputs, run / 'PUBLIC_ASSETS.json')
    mapping = _bound(inputs, run / 'VERSION_MAP.json')
    judgment = _bound(inputs, run / 'SOURCE_JUDGMENTS.json')
    review = _bound(inputs, run / 'ASSET_REVIEW.json')
    for value in (assets, public):
        require(value.get('root_id') == slot['root_id'] and value.get('slot_id') == slot['slot_id'], 'mixed_asset_root_or_slot')
    require(isinstance(assets.get('public_stem'), str) and assets['public_stem'].strip()
            and public.get('public_stem') == assets['public_stem'], 'public_stem_changed')
    public_keys = {'root_id', 'slot_id', 'source_path', 'source_sha256', 'assets_sha256', 'public_stem', 'versions',
                   'contains_relations', 'contains_solver_scores'}
    require(set(public) == public_keys and public['contains_relations'] is False and public['contains_solver_scores'] is False,
            'public_asset_private_field_leak')
    require(public['source_path'] == str(run / 'SOURCE.json'), 'public_source_path_changed')
    for obj in (public, mapping, judgment, review):
        _check_binding(obj, 'source_sha256', run / 'SOURCE.json')
    for obj in (public, mapping, review):
        _check_binding(obj, 'assets_sha256', run / 'ASSETS.json')
    for obj in (mapping, judgment, review):
        _check_binding(obj, 'public_assets_sha256', run / 'PUBLIC_ASSETS.json')
    _check_binding(review, 'source_judgments_sha256', run / 'SOURCE_JUDGMENTS.json')
    require(judgment.get('gold_read_before_save') is False and judgment.get('solver_scores_used') is False,
            'judgment_not_source_first')
    require(review.get('source_first') is True and review.get('solver_scores_used') is False, 'review_not_source_first')
    require(type(judgment.get('stem_valid')) is bool and type(review.get('stem_valid')) is bool, 'stem_judgment_missing')
    pubs = _unique(public.get('versions'), 'public')
    maps = _unique(mapping.get('versions'), 'mapping')
    judges = _unique(judgment.get('versions'), 'judgment')
    reviews = _unique(review.get('versions'), 'review')
    require(set(pubs) == set(maps) == set(judges) == set(reviews), 'review_version_coverage_mismatch')
    require(all(set(p) == {'version_id', 'fact_id', 'proposition'} for p in pubs.values()), 'public_version_private_field_leak')
    turns = {t['turn_ref']: t for t in source.get('turns', [])}
    facts = assets.get('facts')
    require(isinstance(facts, list) and len(facts) <= 8, 'fact_count_invalid')
    require(complete.get('fact_count') == len(facts)
            and complete['status'] == ('ASSETS_READY' if len(facts) >= 6 else 'INSUFFICIENT_FACTS'), 'generation_fact_count_changed')
    expected, fact_ids = {}, set()
    for fact in facts:
        fid = fact.get('fact_id')
        require(isinstance(fid, str) and fid not in fact_ids, 'duplicate_or_missing_fact_id')
        fact_ids.add(fid)
        require(set(fact.get('scope', {})) == {'actor', 'event', 'time', 'condition', 'action_state'}, 'fact_scope_missing')
        anchor = fact.get('anchor', {})
        require(set(anchor) == {'turn_ref', 'speaker', 'quote'}, 'fact_anchor_invalid')
        turn = turns.get(anchor['turn_ref'])
        require(turn is not None and anchor['speaker'] == turn['speaker'] and isinstance(anchor['quote'], str)
                and bool(anchor['quote'].strip()) and anchor['quote'] in turn['text'], 'fact_anchor_not_source_exact')
        for variant, code, relation in (('supported', 's', 'supports'), ('contradicted', 'c', 'contradicts')):
            value = fact.get(variant)
            if variant == 'contradicted' and value is None:
                continue
            require(isinstance(value, dict) and set(value) == {'proposition', 'relation', 'rationale'}
                    and value['relation'] == relation and isinstance(value['proposition'], str) and value['proposition'].strip()
                    and isinstance(value['rationale'], str) and value['rationale'].strip(), 'writer_version_schema_invalid')
            expected[fid + ':' + code] = {'fact_id': fid, 'variant': variant, **deepcopy(value), 'anchor': deepcopy(anchor)}
    require(len(maps) == len(expected) and {m.get('internal_version_id') for m in maps.values()} == set(expected), 'mapping_not_bijective')
    usable = []
    for vid, m in maps.items():
        value = expected[m['internal_version_id']]
        require(all(m.get(k) == value[k] for k in ('fact_id', 'variant', 'relation')), 'mapping_writer_label_changed')
        require(pubs[vid]['fact_id'] == value['fact_id'] and pubs[vid]['proposition'] == value['proposition'], 'public_writer_text_changed')
        j, r = judges[vid], reviews[vid]
        require(j.get('relation') in ('supports', 'contradicts', 'unknown') and r.get('relation') == j['relation'], 'review_judgment_disagreement')
        require(type(r.get('usable')) is bool and type(j.get('scope_valid')) is bool and type(j.get('fact_distinct')) is bool,
                'review_boolean_missing')
        require(isinstance(j.get('reason'), str) and isinstance(j.get('citations'), list)
                and isinstance(r.get('reason'), str), 'review_reason_missing')
        if r['usable']:
            require(judgment['stem_valid'] is True and review['stem_valid'] is True, 'usable_invalid_stem')
            require(r['relation'] == value['relation'] and j['scope_valid'] is True and j['fact_distinct'] is True,
                    'usable_unverified_or_false_label')
            usable.append({**value, 'version_id': vid, 'internal_version_id': m['internal_version_id']})
    return {'slot': slot, 'versions': usable, 'source_path': str(run / 'SOURCE.json'), 'source_sha256': sha(run / 'SOURCE.json'),
            'stem': assets['public_stem'], 'prior': prior, 'status': complete['status'], 'input_dir': str(run)}


def enumerate_candidates(asset, extra_prior=()):
    by_fact = defaultdict(list)
    for version in asset['versions']:
        by_fact[version['fact_id']].append(version)
    prior = asset['prior'] | set(extra_prior)
    candidates = {k: [] for k in COUNTS}
    seen = set()
    for fact_ids in itertools.combinations(sorted(by_fact), 6):
        for versions in itertools.product(*(sorted(by_fact[f], key=lambda x: x['version_id']) for f in fact_ids)):
            k = sum(v['relation'] == 'supports' for v in versions)
            if k not in candidates:
                continue
            key = statement_key(asset['slot']['root_id'], [v['proposition'] for v in versions])
            if len(set(key[2])) != 6 or key in prior or key in seen:
                continue
            seen.add(key)
            candidates[k].append({'versions': versions, 'key': key, 'id': digest(key)})
    for values in candidates.values():
        values.sort(key=lambda c: (c['id'], tuple(v['version_id'] for v in c['versions'])))
    return candidates


def _distinct_assignment(candidates):
    """Find first deterministic matching of slots to whole-statement sets."""
    def walk(index, used, chosen):
        if index == len(candidates):
            return chosen
        for candidate in candidates[index]:
            if candidate is None:
                found = walk(index + 1, used, chosen + [None])
            elif candidate['key'] not in used:
                found = walk(index + 1, used | {candidate['key']}, chosen + [candidate])
            else:
                continue
            if found is not None:
                return found
        return None
    return walk(0, set(), [])


def allocate(assets, available, target=None, priority_counts=(), required_counts=()):
    """Cover required counts jointly, then balance their capped target fractions.

    With no required counts, preserve the legacy priority-count objective. The
    DP state retains all information used by either objective: capped count
    coverage and capped distinct-root coverage. Roots are processed as complete
    groups, so two assignments with the same state have identical future
    opportunities; retaining only the best (retained, preferred) pair is exact.
    """
    target = TARGET if target is None else target
    caps = tuple(target[str(k)] for k in COUNTS)
    require(isinstance(required_counts, (list, tuple))
            and all(type(k) is int and k in COUNTS for k in required_counts)
            and len(required_counts) == len(set(required_counts)), 'invalid_allocation_required_counts')
    required_counts = tuple(required_counts)
    # A zero target must not erase presence for a required count. Such a count
    # participates in required coverage, but has no target ratio/denominator.
    state_caps = tuple(max(cap, int(k in required_counts)) for k, cap in zip(COUNTS, caps))
    balance_counts = tuple(k for k in required_counts if target[str(k)] > 0)

    def fractions(hist):
        return {str(k): Fraction(min(hist[COUNTS.index(k)], target[str(k)]), target[str(k)])
                for k in balance_counts}

    def objective_score(item):
        (hist, roots), (retained, preferred, _) = item
        if required_counts:
            prefix = (sum(hist[COUNTS.index(k)] > 0 for k in required_counts),
                      tuple(sorted(fractions(hist).values())))
        else:
            prefix = tuple(hist[COUNTS.index(k)] for k in priority_counts)
        return (*prefix, sum(min(n, cap) for n, cap in zip(hist, caps)),
                sum(roots), retained, preferred)

    root_count = len({asset['slot']['root_id'] for asset in assets})
    root_caps = tuple(min(cap, root_count) for cap in caps)
    groups = defaultdict(list)
    for i, asset in enumerate(assets):
        groups[asset['slot']['root_id']].append(i)
    zero = (0,) * 5
    states = {(zero, zero): (0, 0, ())}  # retained, initial preferences, [(slot,candidate)]
    for root in sorted(groups):
        indices = groups[root]
        local = []
        choices = [[k for k in COUNTS if available[i][k]] + [None] for i in indices]
        for counts in itertools.product(*choices):
            options = [available[i][k] if k is not None else [None] for i, k in zip(indices, counts)]
            selected = _distinct_assignment(options)
            if selected is None:
                continue
            freq = Counter(k for k in counts if k is not None)
            hist = tuple(freq[k] for k in COUNTS)
            roots = tuple(int(freq[k] > 0) for k in COUNTS)
            retained = sum(k is not None for k in counts)
            preferred = sum(k == assets[i]['slot']['initial_target_count'] for i, k in zip(indices, counts))
            assignment = tuple((i, c) for i, c in zip(indices, selected))
            local.append((hist, roots, retained, preferred, assignment))
        next_states = {}
        for (hist, roots), (retained, preferred, assignment) in states.items():
            for lh, lr, ln, lp, la in local:
                state = (tuple(min(cap, a + b) for cap, a, b in zip(state_caps, hist, lh)), tuple(min(cap, a + b) for cap, a, b in zip(root_caps, roots, lr)))
                score = (retained + ln, preferred + lp)
                old = next_states.get(state)
                if old is None or score > old[:2]:
                    next_states[state] = (*score, assignment + la)
        states = next_states
    # Valid extra questions always retained; no count quota rejects valid deviations.
    best_state, best = max(states.items(), key=objective_score)
    ratios = fractions(best_state[0])
    exact = lambda value: f'{value.numerator}/{value.denominator}'
    return dict(best[2]), {
        'required_counts_covered': sum(best_state[0][COUNTS.index(k)] > 0 for k in required_counts),
        'balance_counts': list(balance_counts),
        'balance_target_fractions': {k: exact(value) for k, value in ratios.items()},
        'balance_sorted_target_fractions': [exact(value) for value in sorted(ratios.values())],
        'balance_excluded_zero_target_counts': [k for k in required_counts if target[str(k)] == 0],
        'priority_count_coverage': {str(k): min(best_state[0][COUNTS.index(k)], target[str(k)]) for k in priority_counts},
        'target_coverage': sum(min(n, cap) for n, cap in zip(best_state[0], caps)),
        'distinct_root_target_coverage': sum(best_state[1]),
        'retained_questions': best[0], 'initial_preferences_matched': best[1]}


def assemble_question(asset, candidate):
    slot = asset['slot']
    versions = list(candidate['versions'])
    # Seed depends on public content and slot only. No truth-label sorting.
    seed = int(digest({'slot_id': slot['slot_id'], 'statements': candidate['key'][2]})[:16], 16)
    versions.sort(key=lambda v: v['version_id'])
    random.Random(seed).shuffle(versions)
    options, evidence, gold, receipts = {}, {}, [], []
    for label, value in zip('ABCDEF', versions):
        options[label] = value['proposition']
        evidence[label] = {**deepcopy(value['anchor']), 'relation': value['relation'],
                           'rationale': value['rationale'], 'reason': value['rationale']}
        if value['relation'] == 'supports':
            gold.append(label)
        receipts.append({'label': label, 'version_id': value['version_id'], 'fact_id': value['fact_id'],
                         'internal_version_id': value['internal_version_id']})
    candidate_id = slot['root_id'] + '-v69-' + slot['slot_id'] + '-' + candidate['id'][:12]
    question = {'candidate_id': candidate_id, 'root_id': slot['root_id'], 'question_type': 'multi_select',
        'topic': slot['focus']['description'], 'question': asset['stem'], 'options': options, 'gold': gold,
        'option_evidence': evidence, 'answer_reason': 'Gold is the set of independently source-verified supported statements.',
        'source_path': asset['source_path'], 'source_sha256': asset['source_sha256'],
        'mechanical_gate_version': 'multiselect_scoring_contract_v15', 'citation_gate': 'DIAGNOSTIC_ONLY',
        'generator_model': MODEL, 'raw_question_content_preserved': True, 'semantic_review_performed': True,
        'training_allowed': False, 'source_training_pool_eligible': False, 'source_semantic_certified': False,
        'source_admission_status': 'FROZEN_FORMAL_ORIGINAL_PLUS_R01', 'upstream_scene_promotion': False,
        'training_target': 'complete_canonical_gold', 'loss_weight_from_partial_score': False}
    receipt = {'candidate_id': candidate_id, 'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
        'initial_target_count': slot['initial_target_count'], 'actual_gold_count': len(gold),
        'permutation_seed': seed, 'selected_versions': receipts,
        'unchanged_propositions_and_evidence': True, 'unchanged_writer_relations': True}
    return question, receipt


def build(report):
    report = Path(report).resolve()
    inputs = {}
    config = _bound(inputs, report / 'CONFIG.json')
    require(config.get('training_allowed') is False, 'training_must_be_disabled')
    slots = config.get('slots')
    require(isinstance(slots, list) and 1 <= len(slots) <= 10 and len({s['slot_id'] for s in slots}) == len(slots), 'slots_invalid')
    require(all(type(s.get('initial_target_count')) is int and s['initial_target_count'] in COUNTS for s in slots), 'initial_target_invalid')
    assets = [load_slot(report, slot, inputs) for slot in slots]
    root_sources = defaultdict(set)
    for asset in assets:
        root_sources[asset['slot']['root_id']].add(asset['source_sha256'])
    require(all(len(s) == 1 for s in root_sources.values()), 'same_root_source_mismatch')
    baseline = _bound(inputs, config['baseline_pool'])
    if config.get('baseline_result_bindings'):
        _bound(inputs, config['baseline_result_bindings'])
    available = [enumerate_candidates(a, prior_keys(baseline, a['slot']['root_id'])) for a in assets]
    target = config.get('target_distribution', TARGET)
    require(isinstance(target, dict) and set(target) == {str(k) for k in COUNTS}
            and all(type(v) is int and 0 <= v <= 10 for v in target.values()), 'invalid_target_distribution')
    if 'target_distribution' in config:
        require(sum(target.values()) == len(slots), 'target_distribution_slot_total_mismatch')
    priority_counts = config.get('allocation_priority_counts', [])
    require(isinstance(priority_counts, list) and all(type(k) is int and k in COUNTS for k in priority_counts)
            and len(priority_counts) == len(set(priority_counts)), 'invalid_allocation_priority_counts')
    required_counts = config.get('allocation_required_counts', [])
    require(isinstance(required_counts, list), 'invalid_allocation_required_counts')
    selected, objective = allocate(assets, available, target, priority_counts, required_counts)
    root_targets = {str(k): min(target[str(k)], len(root_sources)) for k in COUNTS}
    questions, receipts, slot_results = [], [], []
    for i, asset in enumerate(assets):
        candidate = selected.get(i)
        result = {'slot_id': asset['slot']['slot_id'], 'root_id': asset['slot']['root_id'],
            'initial_target_count': asset['slot']['initial_target_count'], 'generation_status': asset['status'],
            'feasible_counts': [k for k in COUNTS if available[i][k]],
            'distinct_candidate_counts': {str(k): len(available[i][k]) for k in COUNTS},
            'usable_version_count': len(asset['versions']), 'assembled': candidate is not None}
        if candidate is not None:
            question, receipt = assemble_question(asset, candidate)
            questions.append(question); receipts.append(receipt)
            result.update(candidate_id=question['candidate_id'], actual_gold_count=len(question['gold']),
                          deviates_from_initial_target=len(question['gold']) != asset['slot']['initial_target_count'])
        else:
            result['reason'] = 'NO_SIX_FACT_VERIFIED_NOVEL_COMBINATION' if not result['feasible_counts'] else 'SAME_BATCH_EXACT_SET_CONFLICT'
        slot_results.append(result)
    require(len(questions) <= config.get('max_questions', 10), 'question_budget_exceeded')
    coverage = Counter(len(q['gold']) for q in questions)
    roots = {str(k): sorted({q['root_id'] for q in questions if len(q['gold']) == k}) for k in COUNTS}
    pool = {'version': VERSION, 'questions': questions, 'candidate_count': len(questions), 'training_allowed': False,
            'provenance': {'requested_model': MODEL, 'composition': 'verified_frozen_versions_only',
                           'config_sha256': sha(report / 'CONFIG.json')}}
    public = {'questions': [{k: deepcopy(q[k]) for k in ('candidate_id', 'root_id', 'question_type', 'topic', 'question',
                                                       'options', 'source_path', 'source_sha256')} for q in questions],
              'candidate_count': len(questions), 'training_allowed': False}
    allocation = {'version': VERSION, 'target_distribution': target, 'allocation_priority_counts': priority_counts,
        'allocation_required_counts': required_counts,
        'required_counts_covered': [k for k in required_counts if coverage[k] > 0],
        'missing_required_counts': [k for k in required_counts if coverage[k] == 0],
        'distinct_root_targets': root_targets, 'assembled_distribution': {str(k): coverage[k] for k in COUNTS},
        'missing_target_counts': {str(k): max(0, target[str(k)] - coverage[k]) for k in COUNTS},
        'distinct_roots_by_count': roots, 'missing_distinct_root_coverage': {str(k): max(0, root_targets[str(k)] - len(roots[str(k)])) for k in COUNTS},
        'allocation_objective': objective, 'slots': slot_results,
        'deviations': [s for s in slot_results if s.get('deviates_from_initial_target')],
        'immutability': {'propositions_copied_verbatim': True, 'evidence_copied_verbatim': True,
                         'writer_relations_unchanged': True, 'gold_from_verified_supports': True,
                         'one_version_per_fact': True, 'no_mixed_roots': True},
        'extra_model_calls': 0, 'training_allowed': False,
        'selection_policy': ('maximize required count presence, then leximin sorted capped target fractions for positive-target required counts, then all target coverage, distinct root coverage, retained questions, initial preferences; deterministic ties'
                             if required_counts else 'maximize declared priority count coverage, then all target coverage, distinct root coverage, retained questions, initial preferences; deterministic ties'),
        'exact_dedup': 'same root/type/sorted whole statement set; casefold and whitespace only',
        'distribution_stage': 'assembled_before_final_content_review_and_difficulty_selection'}
    return {'POOL.json': pool, 'PUBLIC_QUESTIONS.json': public, 'ALLOCATION.json': allocation}, inputs, receipts


def compose(report_dir):
    report = Path(report_dir).resolve()
    require(not any((report / name).exists() for name in (*OUTPUTS, 'ASSEMBLY_RECEIPTS.json')), 'assembly_already_exists_use_verify')
    outputs, inputs, receipts = build(report)
    for name in OUTPUTS:
        write(report / name, outputs[name])
    write(report / 'ASSEMBLY_RECEIPTS.json', {'version': VERSION, 'config_sha256': sha(report / 'CONFIG.json'),
        'inputs': [{'path': path, 'sha256': value} for path, value in sorted(inputs.items())],
        'outputs': [{'path': str(report / name), 'sha256': sha(report / name)} for name in OUTPUTS],
        'selections': receipts, 'training_allowed': False})
    return verify_assembly(report)


def verify_assembly(report_dir):
    report = Path(report_dir).resolve()
    receipt = read(report / 'ASSEMBLY_RECEIPTS.json')
    require(receipt.get('version') == VERSION and receipt.get('training_allowed') is False, 'assembly_receipt_contract_changed')
    _check_binding(receipt, 'config_sha256', report / 'CONFIG.json')
    for group in ('inputs', 'outputs'):
        require(isinstance(receipt.get(group), list), 'assembly_bindings_missing')
        for item in receipt[group]:
            require(Path(item['path']).is_file() and sha(item['path']) == item['sha256'], 'assembly_bound_file_changed:' + item['path'])
    outputs, inputs, selections = build(report)
    require(receipt['inputs'] == [{'path': p, 'sha256': v} for p, v in sorted(inputs.items())], 'assembly_input_coverage_changed')
    require(receipt['outputs'] == [{'path': str(report / name), 'sha256': sha(report / name)} for name in OUTPUTS], 'assembly_output_coverage_changed')
    require(receipt.get('selections') == selections, 'assembly_selection_receipt_changed')
    for name, value in outputs.items():
        require(read(report / name) == value, 'assembly_replay_changed:' + name)
    return outputs['POOL.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify_assembly(args.report) if args.verify else compose(args.report)
    print(json.dumps({'status': 'PASS', 'candidate_count': result['candidate_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
