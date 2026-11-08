"""Independent standard-library-only audit. Never imports production evaluators/scorers.

Run only after RESULT_SUMMARY, both native COMPLETE/RESULTS and every response exist.
Reads frozen data only after finalization; creates two new audit artifacts in --output-dir.
The script uses only the Python standard library and imports no production modules.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_more_roots_composition_v65')
SEED = 2026100800
POLICY = 'r02_groundless_full_only_random_position_v2'
LABELS = tuple('ABCDEF')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def cid(text):
    return 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def seed_for(qid, index):
    return int.from_bytes(hashlib.sha256(f'{SEED}|{qid}|{index}'.encode()).digest()[:8], 'big')


def variants(q):
    """Reconstruct the gold-independent content-ID shuffle using only public data."""
    order = [cid(q['options'][l]) for l in LABELS]
    texts = {cid(q['options'][l]): q['options'][l] for l in LABELS}
    random.Random(seed_for(q['candidate_id'], -1)).shuffle(order)
    used, result = set(), []
    for n in range(6):
        rng = random.Random(seed_for(q['candidate_id'], n))
        while True:
            display = list(order)
            rng.shuffle(display)
            if tuple(display) not in used:
                break
        used.add(tuple(display))
        result.append({'variant_id': f'perm_{n:02d}', 'canonical_order_content_ids': list(order),
            'display_order_content_ids': display, 'options': dict(zip(LABELS, [texts[x] for x in display])),
            'display_to_content': dict(zip(LABELS, display)), 'content_to_text': texts})
    return result


def rescore(raw, mapping, gold):
    """Parse/map/score from first principles, including invalid/empty answer cases."""
    reason, labels = None, None
    if not isinstance(raw, str):
        reason = 'raw_output_missing_or_not_text'
    else:
        match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S)
        if match is None:
            reason = 'exactly_one_box_required'
        else:
            text = match[1].strip()
            labels = [] if not text else [x.strip().upper() for x in text.split(',')]
            if any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', x) is None for x in labels):
                labels, reason = None, 'malformed_id_list'
            elif len(labels) != len(set(labels)):
                labels, reason = None, 'duplicate_id'
    result = {'parse_status': 'success' if reason is None else 'failure', 'parse_reason': reason,
        'scored': False, 'display_prediction': labels, 'content_prediction': None,
        'content_gold': list(gold), 'tp': None, 'fp': None, 'fn': None,
        'exact_match': None, 'partial_score': None}
    if labels is None:
        result['scoring_status'] = 'PARSE_FAILURE'
        return result
    invalid = [x for x in labels if x not in mapping]
    if invalid:
        result.update(scoring_status='INVALID_ID', invalid_ids=invalid)
        return result
    selected = [mapping[x] for x in labels]
    if len(selected) != len(set(selected)):
        result.update(scoring_status='DUPLICATE_CONTENT_ID', invalid_ids=[])
        return result
    pred, truth = set(selected), set(gold)
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    part = Fraction(tp, len(truth))
    if len(truth) < len(mapping):
        part -= Fraction(fp, len(mapping) - len(truth))
    result.update(scoring_status='SCORED', scored=True, content_prediction=sorted(pred),
        tp=tp, fp=fp, fn=fn, exact_match=pred == truth, partial_score=str(part), partial_score_fraction=str(part))
    return result


def aggregate(rows):
    n = len(rows)
    exact = sum(r['exact'] for r in rows)
    part = sum((Fraction(r['partial']) for r in rows), Fraction())
    return {'questions': len({r['candidate_id'] for r in rows}), 'calls': n, 'exact_correct': exact,
        'exact_rate': str(Fraction(exact, n)) if n else None,
        'exact_rate_percent': round(100 * exact / n, 2) if n else None,
        'partial_mean': str(part / n) if n else None,
        'partial_mean_percent': round(100 * float(part / n), 2) if n else None}


def compare(pairs):
    b, t = (aggregate([{'candidate_id': p['candidate_id'], **p[s]} for p in pairs]) for s in ['base', 'trained'])
    return {'base': b, 'trained': t,
        'trained_minus_base_exact_percentage_points': round(100 * (float(Fraction(t['exact_rate'])) - float(Fraction(b['exact_rate']))), 2) if pairs else None,
        'trained_minus_base_partial_points': str(Fraction(t['partial_mean']) - Fraction(b['partial_mean'])) if pairs else None,
        'paired_exact': {'both_correct': sum(p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'trained_only_correct': sum(not p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'base_only_correct': sum(p['base']['exact'] and not p['trained']['exact'] for p in pairs),
            'both_wrong': sum(not p['base']['exact'] and not p['trained']['exact'] for p in pairs)}}


def distribution(qs):
    multi = [q for q in qs if q['question_type'] == 'multi_select']
    hist = Counter(len(q['gold']) for q in multi)
    roots = sorted({q['root_id'] for q in qs})
    return {'questions': len(qs), 'by_type': dict(Counter(q['question_type'] for q in qs)),
        'multiselect_gold_count_histogram': {str(k): hist[k] for k in range(2, 7)},
        'missing_multiselect_gold_counts': [k for k in range(2, 7) if not hist[k]],
        'five_or_six_fraction': str(Fraction(hist[5] + hist[6], len(multi))) if multi else None,
        'by_root': {root: {'questions': sum(q['root_id'] == root for q in qs),
            'by_type': dict(Counter(q['question_type'] for q in qs if q['root_id'] == root)),
            'multiselect_gold_count_histogram': {str(k): sum(q['root_id'] == root and q['question_type'] == 'multi_select' and len(q['gold']) == k for q in qs) for k in range(2, 7)}} for root in roots}}


def select_question(q, rows):
    means = {f'perm_{i:02d}': sum((Fraction(r['partial']) for r in rows if r['variant_id'] == f'perm_{i:02d}'), Fraction()) / 3 for i in range(6)}
    mean = sum(means.values(), Fraction()) / 6
    return {'policy_version': POLICY, 'ground_used': False, 'selected_by_policy': 0 < mean < 1,
        'selection_reason': 'FULL_PARTIAL_IN_OPEN_INTERVAL' if 0 < mean < 1 else ('FULL_PARTIAL_NONPOSITIVE' if mean <= 0 else 'FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM'),
        'full_partial_mean': str(mean), 'permutation_means': {k: str(v) for k, v in means.items()},
        'permutation_score_min': str(min(means.values())), 'permutation_score_max': str(max(means.values())),
        'permutation_score_range': str(max(means.values()) - min(means.values())),
        'full_exact_rate': str(Fraction(sum(r['exact'] for r in rows), len(rows))), 'context_gain': None, 'data_errors': [],
        'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
        'expected_full_calls': 18, 'actual_full_calls': len(rows)}


class Audit:
    def __init__(self, root, expected_questions=8):
        self.root, self.failures, self.checks, self.bound = root, [], 0, {}
        self.expected_questions = expected_questions

    def eq(self, actual, expected, label):
        self.checks += 1
        if actual != expected:
            self.failures.append({'check': label, 'actual': actual, 'expected': expected})

    def yes(self, value, label):
        self.eq(bool(value), True, label)

    def binding(self, path, expected, label):
        path = Path(path)
        actual = sha(path)
        self.eq(actual, expected, label + ':' + str(path))
        self.bound[str(path)] = actual

    def side(self, side, qs, freeze):
        run = self.root / 'evaluation' / side
        m = read(run / 'manifest.json')
        model = {'trained': 'M_clean_policy', 'base': 'M0'}[side]
        self.eq(m['solver_model'], model, side + ':model')
        self.eq(m['request_count'], len(qs) * 18, side + ':request_count')
        self.eq(m['candidate_count'], len(qs), side + ':candidate_count')
        self.eq(m['random_display'], {'version': 'independent_seeded_permutation_v2', 'seed': SEED, 'gold_blind': True, 'cyclic_rebinding': False, 'shape_filter': False}, side + ':random_display')
        self.eq(m['policy_version'], 'v15_full_only_random_mcq6x3_order12x1_v1', side + ':policy')
        self.eq(m['execution'], 'full_only_no_ground_v1', side + ':full_only')
        self.eq(m['training_started'], False, side + ':training_started')
        self.eq(m['training_promotion'], False, side + ':promotion')
        self.eq(m['new_generation_calls'], 0, side + ':generation_calls')
        for key, expected in {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False, 'structured_output_version': 'finite_choice_set_v1'}.items():
            self.eq(m['decoder_contract'][key], expected, side + ':decoder:' + key)
        for path, value in m['code_sha256'].items():
            self.binding(path, value, side + ':code')
        for path, value in m['input_sha256'].items():
            self.binding(m['pool_path'] if path == 'pool' else run / path, value, side + ':input')
        start = read(run / 'STARTED.json')
        self.binding(run / 'manifest.json', start['manifest_sha256'], side + ':start_manifest')
        self.yes(datetime.fromisoformat(start['started_at_utc']) > datetime.fromisoformat(freeze['completed_at_utc']), side + ':freeze_before_start')
        self.yes(not (run / 'FAILED.json').exists(), side + ':no_failed_marker')
        identity = read(run / 'solver_identity.json')
        self.eq(identity, m['solver_identity'], side + ':identity')
        self.eq(identity['model_path'], m['model'], side + ':identity_path')
        self.binding(identity['audit_path'], identity['audit_sha256'], side + ':model_identity_audit')
        expected_path = '/root/autodl-tmp/Qwen3-4B' if side == 'base' else '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
        self.eq(m['model'], expected_path, side + ':fixed_model')
        gold = read(run / 'gold.json')
        canonical = read(run / 'canonical_bindings.json')['candidates']
        self.eq([q['candidate_id'] for q in read(run / 'questions.json')['candidates']], [q['candidate_id'] for q in qs], side + ':question_order')
        for q in qs:
            qid = q['candidate_id']
            self.eq(gold[qid], [cid(q['options'][l]) for l in q['gold']], side + ':gold_mapping:' + qid)
            b = next(b for b in canonical if b['candidate_id'] == qid)
            self.eq(b['content_id_by_label'], {l: cid(q['options'][l]) for l in LABELS}, side + ':canonical_binding:' + qid)
            self.eq(b['gold_content_ids'], gold[qid], side + ':canonical_gold:' + qid)
        plans = {r['request_id']: r for r in m['requests']}
        self.eq(len(plans), len(m['requests']), side + ':unique_planned_ids')
        self.eq(len({(r['candidate_id'], r['variant']['variant_id'], r['slot']) for r in m['requests']}), len(plans), side + ':unique_slots')
        qmap = {q['candidate_id']: q for q in qs}
        variant_cache = {qid: variants(q) for qid, q in qmap.items()}
        for i, req in enumerate(m['requests']):
            rid = req['request_id']; q = qmap[req['candidate_id']]
            self.eq(req['seed'], SEED + i, side + ':planned_seed:' + rid)
            self.eq(req['mode'], 'full_context', side + ':mode:' + rid)
            self.eq(req['question_type'], 'multi_select', side + ':type:' + rid)
            vi = int(req['variant']['variant_id'].split('_')[-1])
            self.eq(req['variant'], variant_cache[req['candidate_id']][vi], side + ':permutation:' + rid)
            prompt = run / 'prompts' / (rid + '.txt')
            self.binding(prompt, req['prompt_sha256'], side + ':prompt')
            self.yes(req['prompt_tokens'] + 32 <= 16384, side + ':context_limit:' + rid)
        files = sorted((run / 'responses').glob('*.json'))
        records = [read(p) for p in files]
        self.eq(Counter(r['request_id'] for r in records), Counter(plans.keys()), side + ':complete_request_set')
        self.eq(len(records), read(run / 'COMPLETE.json')['actual_call_count'], side + ':complete_count')
        self.eq(len(records), len(qs) * 18, side + ':actual_count')
        scores = []
        for f, rec in zip(files, records):
            rid = rec['request_id']; req = plans[rid]; qid = req['candidate_id']
            self.eq(f.stem, rid, side + ':response_filename:' + rid)
            expected = rescore(rec['raw_output'], req['variant']['display_to_content'], gold[qid])
            self.eq(rec['score'], expected, side + ':native_score:' + rid)
            metadata = rec['metadata']; errors = []
            for k in ['temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version']:
                if metadata.get(k) != m['decoder_contract'][k]: errors.append(k)
            valid = expected['scored'] and metadata.get('finish_reason') in ['stop', 'eos'] and not errors
            self.eq(rec['evaluation_valid'], valid, side + ':computed_validity:' + rid)
            self.yes(valid, side + ':response_valid:' + rid)
            self.eq(rec['data_errors'], [], side + ':data_errors:' + rid)
            for key in ['candidate_id', 'root_id', 'question_type', 'mode', 'slot', 'seed']:
                self.eq(rec[key], req[key], side + ':record_' + key + ':' + rid)
            self.eq(rec['variant_id'], req['variant']['variant_id'], side + ':variant_identity:' + rid)
            self.eq(rec['solver_model'], model, side + ':response_model:' + rid)
            self.eq(rec['model_path'], m['model'], side + ':response_model_path:' + rid)
            self.eq(metadata['sampling_seed'], req['seed'], side + ':sampling_seed:' + rid)
            self.eq(metadata['prompt_token_count'], req['prompt_tokens'], side + ':prompt_tokens:' + rid)
            if expected['scored']:
                scores.append({'request_id': rid, 'candidate_id': qid, 'root_id': req['root_id'], 'variant_id': rec['variant_id'],
                    'exact': expected['exact_match'], 'partial': expected['partial_score_fraction'],
                    'tp': expected['tp'], 'fp': expected['fp'], 'fn': expected['fn'],
                    'selected_count': len(expected['content_prediction']), 'gold_count': len(gold[qid]),
                    'display_prediction': expected['display_prediction'], 'content_prediction': expected['content_prediction'],
                    'response_sha256': sha(f)})
        results = read(run / 'RESULTS.json')
        decisions = []
        for q in qs:
            rows = [r for r in scores if r['candidate_id'] == q['candidate_id']]
            self.eq(Counter(r['variant_id'] for r in rows), Counter({f'perm_{i:02d}': 3 for i in range(6)}), side + ':six_by_three:' + q['candidate_id'])
            decisions.append(select_question(q, rows))
        self.eq(results['questions'], decisions, side + ':selection_results')
        self.eq(results['selected_questions'], [q['candidate_id'] for q in decisions if q['selected_by_policy']], side + ':selected_ids')
        self.eq(results['call_set_errors'], [], side + ':call_errors')
        self.eq(results['ground_calls'], 0, side + ':ground_calls')
        return m, scores, decisions

    def run(self):
        root = self.root
        summary = read(root / 'RESULT_SUMMARY.json')
        cfg = read(root / 'CONFIG.json')
        pool = read(root / 'POOL.json')['questions']
        usable = read(root / 'EVALUATION_POOL.json')['questions']
        review = read(root / 'FINAL_CONTENT_REVIEW.json')
        freeze = read(root / 'FINAL_CONTENT_REVIEW.COMPLETE.json')
        self.eq(freeze['before_any_solver'], True, 'freeze:before_any_solver')
        self.eq(freeze['solver_scores_used'], False, 'freeze:solver_scores_not_used')
        self.binding(root / 'POOL.json', review['pool_sha256'], 'review:pool')
        reviewed_ids = [q['candidate_id'] for q in pool if next(x for x in review['questions'] if x['candidate_id'] == q['candidate_id'])['core_status'] == 'CORE_USABLE']
        self.eq([q['candidate_id'] for q in usable], reviewed_ids, 'evaluation:content_usable_set')
        self.eq(len(usable), self.expected_questions, 'authorized:expected_questions')
        self.eq(len(usable) * 18, self.expected_questions * 18, 'authorized:expected_calls_per_model')
        self.yes(len(pool) <= cfg['max_questions'] <= 10, 'authorized:question_budget')
        self.yes(len(usable) * 18 <= cfg['max_calls_per_model'] <= 180, 'authorized:solver_call_budget')
        manifests, sides, decisions = {}, {}, {}
        for side in ['trained', 'base']:
            manifests[side], sides[side], decisions[side] = self.side(side, usable, freeze)
        permitted = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
        self.eq({k: v for k, v in manifests['trained'].items() if k not in permitted}, {k: v for k, v in manifests['base'].items() if k not in permitted}, 'paired:all_non_model_manifest_fields')
        for req in manifests['base']['requests']:
            rid = req['request_id']
            self.eq((root / 'evaluation/base/prompts' / (rid + '.txt')).read_bytes(), (root / 'evaluation/trained/prompts' / (rid + '.txt')).read_bytes(), 'paired:byte_identical_prompt:' + rid)
        selected_ids = {q['candidate_id'] for q in decisions['trained'] if q['selected_by_policy']}
        selected = [q for q in usable if q['candidate_id'] in selected_ids]
        byside = {s: {r['request_id']: r for r in rows} for s, rows in sides.items()}
        self.eq(set(byside['base']), set(byside['trained']), 'paired:response_ids')
        qmap = {q['candidate_id']: q for q in usable}
        pairs = [{'request_id': rid, 'candidate_id': b['candidate_id'], 'root_id': b['root_id'],
                  'gold_count': len(qmap[b['candidate_id']]['gold']), 'base': b, 'trained': byside['trained'][rid],
                  'trained_selected': b['candidate_id'] in selected_ids} for rid, b in byside['base'].items()]
        comparison = {'primary_scope': 'all_content_usable_questions_frozen_before_solver', 'primary': compare(pairs),
            'by_root': {x: compare([p for p in pairs if p['root_id'] == x]) for x in sorted({q['root_id'] for q in usable})},
            'by_correct_count': {str(n): compare([p for p in pairs if p['gold_count'] == n]) for n in range(2, 7)},
            'selected_secondary': compare([p for p in pairs if p['trained_selected']]),
            'questions': [{'candidate_id': qid, 'root_id': q['root_id'], 'gold_count': len(q['gold']),
                           'trained_selected': qid in selected_ids, **compare([p for p in pairs if p['candidate_id'] == qid])} for qid, q in qmap.items()]}
        for target in [summary['base_trained_comparison'], read(root / 'BASE_TRAINED_COMPARISON.json')]:
            for k, v in comparison.items(): self.eq(target[k], v, 'comparison:' + k)
        emitted_pairs = read(root / 'PAIRED_RAW_SCORES.json')['pairs']
        self.eq(len(emitted_pairs), len(pairs), 'paired_export:count')
        ep = {p['request_id']: p for p in emitted_pairs}
        for pair in pairs:
            e = ep[pair['request_id']]
            for key in ['candidate_id', 'root_id', 'gold_count', 'trained_selected']:
                self.eq(e[key], pair[key], 'paired_export:' + pair['request_id'] + ':' + key)
            for side in ['base', 'trained']:
                for key in ['exact', 'partial', 'tp', 'fp', 'fn', 'selected_count', 'gold_count']:
                    self.eq(e[side][key], pair[side][key], 'paired_export:' + pair['request_id'] + ':' + side + ':' + key)
        baseline = read(cfg['baseline_pool'])['questions']
        self.eq(len(baseline), 60, 'baseline:60')
        self.eq(len({q['candidate_id'] for q in baseline}), 60, 'baseline:unique_60')
        normalize = lambda s: ' '.join(s.casefold().split())
        merged, seen, duplicates = [], {}, []
        for q in baseline + selected:
            f = 'options' if q['question_type'] == 'multi_select' else 'items'
            key = (q['root_id'], q['question_type'], tuple(sorted(normalize(x) for x in q[f].values())))
            if key in seen:
                original = seen[key]
                og = [normalize(original[f][k]) for k in original['gold']]
                qg = [normalize(q[f][k]) for k in q['gold']]
                if f == 'options': og.sort(); qg.sort()
                duplicates.append({'candidate_id': q['candidate_id'], 'retained_candidate_id': original['candidate_id'],
                    'root_id': q['root_id'], 'reason': 'same_root_type_normalized_full_six_statement_set', 'gold_content_disagreement': og != qg})
            else:
                merged.append(q); seen[key] = q
        self.eq(summary['exact_merge_duplicates'], duplicates, 'merge:exact_duplicates')
        self.yes(not any(d['gold_content_disagreement'] for d in duplicates), 'merge:no_gold_disagreement')
        planned = [{'candidate_id': s['slot_id'], 'root_id': s['root_id'], 'question_type': 'multi_select', 'gold': list(range(s['initial_target_count']))} for s in cfg['slots']]
        groups = {'planned': planned, 'assembled': pool, 'content_usable': usable, 'selected': selected, 'baseline': baseline, 'merged': merged}
        distributions = {k: distribution(v) for k, v in groups.items()}
        self.eq(summary['distributions'], distributions, 'distribution:all_groups_roots_types_counts')
        self.eq(read(root / 'COVERAGE.json')['distributions'], distributions, 'distribution:coverage_export')
        for stem, qs in [('ASSEMBLED', pool), ('CONTENT_USABLE', usable), ('SELECTED', selected), ('MERGED_USABLE', merged)]:
            self.eq(read(root / (stem + '_CANDIDATES.json'))['questions'], qs, 'export:questions_gold_unchanged:' + stem)
        counts = {'planned_slots': len(cfg['slots']), 'assembled_count': len(pool), 'content_usable_count': len(usable),
            'known_issue_count': len(pool) - len(usable), 'selected_count': len(selected), 'merged_count': len(merged), 'calls_per_model': len(sides['base'])}
        for k, v in counts.items(): self.eq(summary[k], v, 'summary:' + k)
        blind = read(root / 'BLIND_ALL_SIX_BASELINE.json')
        blind_recomputed = {}
        for name in ['baseline', 'content_usable', 'selected', 'merged']:
            rows = []
            for q in groups[name]:
                if q['question_type'] != 'multi_select': continue
                k = len(q['gold']); truth = set(q['gold']); pred = set(q['options'])
                tp, fp, fn = len(truth & pred), len(pred - truth), len(truth - pred)
                partial = Fraction(tp, k) - (Fraction(fp, 6 - k) if k < 6 else 0)
                rows.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'exact': pred == truth,
                    'partial': str(partial), 'tp': tp, 'fp': fp, 'fn': fn, 'selected_count': 6, 'gold_count': k})
            expected = {'scope': 'multiselect_only_native_scorer_equal_question_weights', 'aggregate': aggregate(rows),
                'by_correct_count': {str(k): aggregate([r for r in rows if r['gold_count'] == k]) for k in range(2, 7)},
                'questions': rows, 'formula': 'TP/k - FP/(6-k); k=6 has no false-positive term',
                'all_six_partial_is_zero_for_gold_two_to_five': True}
            self.eq(blind[name], expected, 'blind_all_six:' + name)
            blind_recomputed[name] = expected['aggregate']
        old = read(cfg['baseline_result_bindings'])['files']
        for b in old: self.binding(b['path'], b['sha256'], 'previous_bound_file')
        self.yes({'path': str(Path(cfg['baseline_pool']).resolve()), 'sha256': sha(cfg['baseline_pool'])} in old, 'baseline:previously_bound')
        self.eq(summary['previous_bound_files_unchanged'], len(old), 'baseline:binding_count')
        for name in ['EVALUATION_PREPARATION.COMPLETE.json', 'RESULT_BINDINGS.json']:
            for b in read(root / name)['files']: self.binding(b['path'], b['sha256'], 'frozen:' + name)
        for q in pool:
            self.binding(q['source_path'], q['source_sha256'], 'source:unchanged')
        self.eq(summary['training_allowed'], False, 'report:training_allowed')
        self.eq(summary['training_started'], False, 'report:training_started')
        self.eq(summary['automatic_followup'], False, 'report:automatic_followup')
        self.eq(summary['new_expansion_tokens'], 0, 'report:no_new_source_expansion')
        self.eq(summary['source_scope'], 'full_original_plus_R01_no_new_AB', 'report:source_scope')
        self.eq(summary['root_only_causal_comparison'], False, 'report:no_root_only_causal_claim')
        self.eq(summary['new_root_ids'], sorted({s['root_id'] for s in cfg['slots']}), 'report:new_root_scope')
        self.eq(len(summary['new_root_ids']), 5, 'report:five_new_roots')
        self.eq(read(root / 'RESULT_BINDINGS.json')['scope'], 'all_v65_final_artifacts_and_direct_frozen_inputs_except_mutable_process_logs', 'report:result_binding_scope')
        expected_limitation = 'five_new_roots_relative_to_v64_with_full_original_plus_R01; not_pure_root_only_comparison_to_AB_sources; not_second_round_training_effect'
        self.eq(summary['base_trained_comparison']['limitation'], expected_limitation, 'comparison:scope_limitation')
        for slot in cfg['slots']:
            source_path = root / 'assets' / slot['slot_id'] / 'SOURCE.json'
            self.binding(slot['history_path'], slot['history_sha256'], 'source:complete_original_history')
            self.eq(read(source_path)['turns'], read(slot['history_path'])['turns'], 'source:complete_original_plus_R01:' + slot['slot_id'])
            corrected = source_path.parent / 'SOURCE_JUDGMENTS.METADATA_CORRECTION.json'
            if corrected.exists():
                receipt = read(corrected)
                initial = read(source_path.parent / 'SOURCE_JUDGMENTS.INITIAL.json')
                canonical = read(source_path.parent / 'SOURCE_JUDGMENTS.json')
                self.eq({k:v for k,v in initial.items() if k != 'public_assets_sha256'}, {k:v for k,v in canonical.items() if k != 'public_assets_sha256'}, 'metadata_correction:judgments_unchanged:' + slot['slot_id'])
                self.binding(source_path.parent / 'SOURCE_JUDGMENTS.INITIAL.json', receipt['original_source_first_file']['sha256'], 'metadata_correction:original_saved_bytes')
                self.binding(source_path.parent / 'SOURCE_JUDGMENTS.json', receipt['corrected_source_judgments']['sha256'], 'metadata_correction:corrected_saved_bytes')
                self.binding(source_path.parent / 'PUBLIC_ASSETS.json', canonical['public_assets_sha256'], 'metadata_correction:public_binding')
                self.eq(receipt['changed_fields'], ['public_assets_sha256'], 'metadata_correction:only_binding_field')
                self.eq(receipt['private_labels_read_before_metadata_correction'], True, 'metadata_correction:timing_disclosed')
                self.eq(initial['gold_read_before_save'], False, 'metadata_correction:original_source_first_flag')
                self.eq(receipt['solver_scores_used'], False, 'metadata_correction:no_solver_scores')
        return {'counts': counts, 'selected_candidate_ids': [q['candidate_id'] for q in selected],
            'selection': decisions, 'comparison': comparison, 'distributions': distributions,
            'blind_all_six_aggregates': blind_recomputed, 'merge_duplicates': duplicates,
            'previous_bound_files_checked': len(old), 'unique_file_hashes_checked': len(self.bound),
            'per_response_recalculation': sides}


def self_test():
    mapping = dict(zip(LABELS, ['x' + x for x in LABELS]))
    assert rescore('<box>B,A</box>', mapping, ['xA', 'xB'])['exact_match']
    assert rescore('<box>A,C</box>', mapping, ['xA', 'xB'])['partial_score_fraction'] == '1/4'
    assert rescore('<box>C,D,E,F</box>', mapping, ['xA', 'xB'])['partial_score_fraction'] == '-1'
    assert rescore('<box>A,B,C,D,E,F</box>', mapping, ['xA', 'xB'])['partial_score_fraction'] == '0'
    assert rescore('<box>A,B,C,D,E,F</box>', mapping, list(mapping.values()))['partial_score_fraction'] == '1'
    assert rescore('<box></box>', mapping, ['xA', 'xB'])['partial_score_fraction'] == '0'
    assert rescore('<box>A,A</box>', mapping, ['xA', 'xB'])['parse_reason'] == 'duplicate_id'
    assert rescore('<box>G</box>', mapping, ['xA', 'xB'])['scoring_status'] == 'INVALID_ID'
    assert rescore('explanation <box>A</box>', mapping, ['xA', 'xB'])['parse_status'] == 'failure'
    assert rescore('<box>A</box><box>B</box>', mapping, ['xA', 'xB'])['parse_status'] == 'failure'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--expected-questions', type=int, default=8)
    args = p.parse_args()
    self_test()
    if args.self_test:
        print('READY: independent parser/scorer self-tests passed; no evaluation data read.')
        return
    for required in ['RESULT_SUMMARY.json', 'RESULT_BINDINGS.json', 'evaluation/trained/COMPLETE.json', 'evaluation/base/COMPLETE.json', 'evaluation/trained/RESULTS.json', 'evaluation/base/RESULTS.json']:
        if not (args.root / required).exists(): raise SystemExit('NOT_READY: ' + required)
    output_dir = args.output_dir or (args.root / 'execution/independent_audit')
    output_dir.mkdir(parents=True, exist_ok=True)
    a = Audit(args.root, args.expected_questions)
    report = {'audit_kind': 'independent_standard_library_raw_recalculation', 'production_audit_functions_called': False,
        'model_calls': 0, 'frozen_files_modified': 0, 'scoring_content_review_reopened': False,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__)}
    try:
        report['recomputed'] = a.run()
    except Exception:
        a.failures.append({'check': 'audit_exception', 'traceback': traceback.format_exc()})
    report.update(status='PASS' if not a.failures else 'FAIL', checks_completed=a.checks, failures=a.failures,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), result_summary_sha256=sha(args.root / 'RESULT_SUMMARY.json'))
    target = output_dir / 'INDEPENDENT_RESULT_AUDIT.json'
    with target.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda x: x.decode(errors='replace') if isinstance(x, bytes) else sorted(x) if isinstance(x, (set, frozenset)) else str(x))
        f.write('\n')
    counts = report.get('recomputed', {}).get('counts', {})
    lines = ['# v65 独立结果复算', '', '状态：' + report['status'], '',
        '仅使用 Python 标准库从冻结输入和原始回答复算；未调用 evaluate.verify、原评分器或筛选器，未修改冻结文件或运行模型。',
        '', f"完成 {a.checks} 项检查；发现 {len(a.failures)} 个差异。", '',
        '检查覆盖：回答格式与合法性、显示字母到内容映射、exact/partial、6×3 调用完整性和逐题筛选；Base 配对提示词/置换/seed/解码参数；各层汇总、精确去重合并、分布、盲选全部六项、历史和当前绑定文件 hash。',
        '', '复算计数：' + json.dumps(counts, ensure_ascii=False)]
    if a.failures: lines += ['', '差异详情见 JSON 中的 failures；本审计未修补任何评分或报告。']
    with (output_dir / 'INDEPENDENT_RESULT_AUDIT.md').open('x') as f: f.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'checks': a.checks, 'failures': len(a.failures), 'path': str(target)}))
    if a.failures: raise SystemExit(1)

if __name__ == '__main__':
    main()
