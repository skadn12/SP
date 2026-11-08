"""Independent stdlib audit: frozen v71 raw scoring, reviewed asset rounds and merge.

No production imports or network. --self-test reads synthetic data only. Real
execution requires completed finalization and an explicit usable question count;
only new audit artifacts are written, never frozen results or content reviews.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import hmac
import json
from pathlib import Path
import random
import re
import tempfile
import traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEED=2026100800
POLICY='r02_groundless_full_only_random_position_v2'
LABELS=tuple('ABCDEF')
COUNTS=(2,3,4,5)
FACTS=[f'F{i:02d}' for i in range(1,7)]
FORBIDDEN=re.compile(r"(?<![\w])(?:he|she|him|her|his|hers|himself|herself)(?![\w])",re.I)

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


def normalized_proposition(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value).casefold() if c.isalnum())


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def accepted_raw_rows(raw, manifest, parent, targets):
    """Independently apply the narrow response contract, retaining row failures."""
    try:
        obj = strict_json(raw)
        if not isinstance(obj, dict) or set(obj) != {'root_id', 'family_id', 'replacements'}:
            raise ValueError('response_schema')
        if (obj['root_id'], obj['family_id']) != (manifest['root_id'], manifest['family_id']):
            raise ValueError('response_identity')
        if not isinstance(obj['replacements'], list) or [r.get('fact_id') for r in obj['replacements']] != targets:
            raise ValueError('target_fact_order')
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return None, [{'fact_id': None, 'error': str(error)}]
    positives = {normalized_proposition(t) for t in parent['options'].values()}
    errors, result = [], []
    for fid, row in zip(targets, obj['replacements']):
        error = None
        if set(row) != {'fact_id', 'contradicted', 'contradiction_unavailable_reason'}:
            error = 'replacement_schema'
        else:
            c, reason = row['contradicted'], row['contradiction_unavailable_reason']
            if c is None:
                if not isinstance(reason, str) or not reason.strip(): error = 'null_alternative_reason'
            elif (not isinstance(c, dict) or set(c) != {'proposition', 'relation', 'rationale'}
                    or c['relation'] != 'contradicts' or any(not isinstance(c[k], str) or not c[k].strip() for k in ('proposition', 'rationale'))):
                error = 'contradicted_schema'
            elif reason is not None: error = 'nonnull_alternative_reason'
            elif FORBIDDEN.search(c['proposition']): error = 'gendered_public_pronoun'
            elif normalized_proposition(c['proposition']) in positives: error = 'copied_supported_parent_proposition'
        if error:
            errors.append({'fact_id': fid, 'error': error})
            result.append({'fact_id': fid, 'contradicted': None,
                'contradiction_unavailable_reason': 'Mechanically rejected raw row: ' + error})
        else: result.append(deepcopy(row))
    return {'root_id': manifest['root_id'], 'family_id': manifest['family_id'], 'replacements': result}, errors


def fixed_assets(manifest, parent, data):
    rows = {r['fact_id']: r for r in data['replacements']} if data else {}
    facts = []
    for fid, label in zip(FACTS, LABELS):
        evidence = parent['option_evidence'][label]
        anchor = {k: evidence[k] for k in ('turn_ref', 'speaker', 'quote')}
        replacement = rows.get(fid, {'contradicted': None, 'contradiction_unavailable_reason': 'Not supplied in this generation round.'})
        counter = deepcopy(replacement['contradicted'])
        if counter is not None: counter['anchor'] = deepcopy(anchor)
        facts.append({'fact_id': fid, 'parent_label': label, 'supported': {'proposition': parent['options'][label],
            'relation': 'supports', 'rationale': evidence.get('rationale', evidence.get('reason')), 'anchor': anchor},
            'contradicted': counter, 'anchor': deepcopy(anchor) if counter else None,
            'contradiction_unavailable_reason': replacement['contradiction_unavailable_reason']})
    return {'root_id': manifest['root_id'], 'slot_id': manifest['slot_id'], 'family_id': manifest['family_id'],
        'parent_candidate_id': parent['candidate_id'], 'parent_sha256': manifest['parent_sha256'],
        'source_sha256': manifest['source_sha256'], 'public_stem': parent['question'], 'facts': facts}


def projected_versions(manifest, assets):
    public, private = [], []
    rid = manifest.get('round_id', 'final')
    for fact in assets['facts']:
        if fact['contradicted'] is None: continue
        internal = fact['fact_id'] + ':c:' + rid
        vid = fact.get('origin', {}).get('version_id') or 'v_' + hmac.new(bytes.fromhex(manifest['version_id_salt']), internal.encode(), hashlib.sha256).hexdigest()[:24]
        public.append({'version_id': vid, 'fact_id': fact['fact_id'], 'proposition': fact['contradicted']['proposition']})
        private.append({'version_id': vid, 'fact_id': fact['fact_id'], 'internal_version_id': internal,
            'variant': 'contradicted', 'relation': 'contradicts'})
    return sorted(public, key=lambda r: r['version_id']), sorted(private, key=lambda r: r['version_id'])


def supplement_targets(initial):
    if not all(p['usable'] for p in initial['review']['parents']): raise ValueError('parent_review_failed_no_supplement')
    approved = set(initial['approved_fact_ids'])
    if len(approved) >= 4: raise ValueError('four_distinct_false_facts_already_approved')
    return [fid for fid in FACTS if fid not in approved]


def merge_approved(manifest, parent, rounds):
    """Merge saved accepted round alternatives once, retaining their original IDs."""
    assets = fixed_assets(manifest, parent, None)
    facts = {f['fact_id']: f for f in assets['facts']}
    accepted, judgments, reviews = [], [], []
    for r in rounds:
        byid = {v['version_id']: v for v in r['mapping']['versions']}
        for vid in sorted(r['approved']):
            fid = byid[vid]['fact_id']
            if facts[fid]['contradicted'] is not None: raise ValueError('accepted_initial_version_overwritten')
            facts[fid] = deepcopy(next(f for f in r['assets']['facts'] if f['fact_id'] == fid))
            facts[fid]['origin'] = {'round_id': r['round_id'], 'version_id': vid}
            accepted.append({'fact_id': fid, 'version_id': vid, 'round_id': r['round_id']})
            judgments.append(deepcopy(next(v for v in r['judgment']['versions'] if v['version_id'] == vid)))
            reviews.append(deepcopy(next(v for v in r['review']['versions'] if v['version_id'] == vid)))
    assets['facts'] = [facts[fid] for fid in FACTS]
    return assets, sorted(accepted, key=lambda r: r['fact_id']), sorted(judgments, key=lambda r: r['version_id']), sorted(reviews, key=lambda r: r['version_id'])


class Audit:
    def __init__(self,root,expected_questions):
        self.root=Path(root).resolve();self.expected_questions=expected_questions
        self.failures=[];self.checks=0;self.bound={};self.walked=set();self.hash_cache={};self.ledger_rows=0
    def eq(self,actual,expected,label):
        self.checks+=1
        if actual!=expected:self.failures.append({'check':label,'actual':actual,'expected':expected})
    def yes(self,value,label):self.eq(bool(value),True,label)
    def binding(self,path,expected,label):
        path=Path(path).resolve()
        signature=(path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)
        key=(str(path),expected)
        if key not in self.hash_cache:
            actual=sha(path)
            after=(path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)
            self.eq(after,signature,label+':stable_during_hash:'+str(path))
            self.hash_cache[key]=(actual,after)
        actual,previous=self.hash_cache[key]
        self.eq(signature,previous,label+':unchanged_since_cached_hash:'+str(path))
        self.eq(actual,expected,label+':'+str(path));self.bound[str(path)]=actual
    def recursive_bindings(self,path):
        """Verify declared rows; recurse only explicit RESULT_BINDINGS ancestry.
        Current v71 JSON may declare files; external JSON stays a hash-bound endpoint.
        Current evaluation checkpoint_files declare explicit model inputs.
        Relative entries resolve against their originating JSON directory.
        """
        path=Path(path).resolve()
        if path in self.walked:return
        self.walked.add(path);obj=read(path)
        if not isinstance(obj,dict):return
        rows=obj.get('files',[])
        if not isinstance(rows,list):rows=[]
        if path.name=='solver_identity.json' and path.parent in (self.root/'evaluation/base',self.root/'evaluation/trained'):
            rows=list(rows)+list(obj.get('checkpoint_files',[]))
        for index,row in enumerate(rows):
            if not (isinstance(row,dict) and isinstance(row.get('path'),str) and isinstance(row.get('sha256'),str)
                    and re.fullmatch('[0-9a-f]{64}',row['sha256'])):continue
            target=Path(row['path'])
            if not target.is_absolute():target=path.parent/target
            self.binding(target,row['sha256'],'explicit_file_binding:'+str(path));self.ledger_rows+=1
            if target.suffix=='.json' and (target.name=='RESULT_BINDINGS.json' or target.resolve().is_relative_to(self.root)):
                self.recursive_bindings(target)
    def citations(self,anchor,source,label):
        self.eq(set(anchor),{'turn_ref','speaker','quote'},label+':fields')
        matches=[t for t in source['turns'] if t['turn_ref']==anchor.get('turn_ref')]
        self.eq(len(matches),1,label+':unique_ref')
        if matches:
            self.eq(anchor.get('speaker'),matches[0]['speaker'],label+':speaker')
            self.yes(isinstance(anchor.get('quote'),str) and bool(anchor['quote'].strip()) and anchor['quote'] in matches[0]['text'],label+':quote')
    def side(self, side, qs, freeze):
        run = self.root / 'evaluation' / side
        if not qs:
            skipped = read(run / 'SKIPPED.json')
            self.eq(skipped['status'], 'EMPTY_CONTENT_USABLE_POOL', side + ':explicit_empty_skip')
            self.eq(skipped['actual_call_count'], 0, side + ':zero_calls')
            self.yes(not (run / 'STARTED.json').exists() and not list((run / 'responses').glob('*.json')), side + ':no_solver_for_empty_pool')
            self.eq(read(self.root / (side.upper() + '_EVALUATION.COMPLETE.json'))['calls'], 0, side + ':empty_stage_complete')
            return None, [], []
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


    def declared_files(self, rows, label):
        for row in rows:
            self.binding(row['path'], row['sha256'], label)

    def projection(self, folder, manifest, assets, parent, source):
        folder = Path(folder); label = str(folder.relative_to(self.root))
        self.eq(read(folder/'ASSETS.json'), assets, 'reconstructed_assets:' + label)
        public_versions, private_versions = projected_versions(manifest, assets)
        public = {'root_id': manifest['root_id'], 'slot_id': manifest['slot_id'], 'family_id': manifest['family_id'],
            'source_path': manifest['source_path'], 'source_sha256': manifest['source_sha256'],
            'assets_sha256': sha(folder/'ASSETS.json'), 'public_stem': parent['question'],
            'parent_facts': [{'fact_id': f, 'parent_label': l, 'proposition': parent['options'][l]} for f,l in zip(FACTS,LABELS)],
            'versions': public_versions, 'contains_relations': False, 'contains_solver_scores': False,
            'parent_path': str(Path(manifest['source_path']).parent/'PARENT.json'), 'parent_sha256': manifest['parent_sha256'],
            'round_id': manifest.get('round_id', 'final')}
        self.eq(read(folder/'PUBLIC_ASSETS.json'), public, 'public_projection:' + label)
        mapping = {'source_sha256': manifest['source_sha256'], 'assets_sha256': sha(folder/'ASSETS.json'),
            'versions': private_versions, 'public_assets_sha256': sha(folder/'PUBLIC_ASSETS.json')}
        self.eq(read(folder/'VERSION_MAP.json'), mapping, 'private_projection:' + label)
        self.eq(len({v['version_id'] for v in public_versions}), len(public_versions), 'unique_anonymous_ids:' + label)
        positives = {normalized_proposition(v) for v in parent['options'].values()}
        for fact in assets['facts']:
            anchor = {k: parent['option_evidence'][fact['parent_label']][k] for k in ('turn_ref','speaker','quote')}
            self.eq(fact['supported']['anchor'], anchor, 'fixed_parent_anchor:' + label + fact['fact_id'])
            self.citations(anchor, source, 'parent_exact_anchor:' + label + fact['fact_id'])
            if fact['contradicted'] is not None:
                self.eq(fact['anchor'], anchor, 'fixed_fact_anchor:' + label + fact['fact_id'])
                self.eq(fact['contradicted']['anchor'], anchor, 'fixed_counter_anchor:' + label + fact['fact_id'])
                text = fact['contradicted']['proposition']
                self.yes(normalized_proposition(text) not in positives, 'normalized_parent_copyguard:' + label + fact['fact_id'])
                self.yes(not FORBIDDEN.search(text), 'public_gender_word_gate:' + label + fact['fact_id'])
        return public, mapping

    def source_review(self, run, folder, source, assets, public, mapping):
        rid = folder.name; label = run.name + ':' + rid
        j = read(folder/'SOURCE_JUDGMENTS.json'); r = read(folder/'ASSET_REVIEW.json'); receipt = read(folder/'REVIEW_RECEIPT.json')
        for k,v in {'source_sha256':sha(run/'SOURCE.json'), 'public_assets_sha256':sha(folder/'PUBLIC_ASSETS.json'),
                'complete_source_turns_read':len(source['turns']), 'new_variant_private_artifacts_read_before_save':False,
                'solver_scores_used':False, 'current_solver_scores_read':False,
                'parent_all_true_provenance_known':True, 'parent_gold_known_before_review':True}.items():
            self.eq(j.get(k), v, 'source_judgment:' + label + ':' + k)
        self.eq(type(j['stem_valid']), bool, 'stem_boolean:' + label)
        self.eq([p['fact_id'] for p in j['parents']], FACTS, 'source_parent_coverage:' + label)
        self.eq(sorted(v['version_id'] for v in j['versions']), sorted(v['version_id'] for v in mapping['versions']), 'source_version_coverage:' + label)
        facts = {f['fact_id']: f for f in assets['facts']}; maps = {v['version_id']:v for v in mapping['versions']}
        expected_rows = {}
        for key,idkey in [('parents','fact_id'),('versions','version_id')]:
            expected_rows[key] = []
            for row in j[key]:
                identity = row[idkey]
                fact = facts[identity if key == 'parents' else maps[identity]['fact_id']]
                writer = fact['supported' if key == 'parents' else 'contradicted']
                self.yes(row['relation'] in ('supports','contradicts','unknown'), 'judged_relation:' + label + identity)
                for k in ('scope_valid','fact_distinct'): self.eq(type(row[k]), bool, 'judgment_boolean:' + label + identity + k)
                self.yes(isinstance(row['reason'],str) and row['reason'].strip(), 'reason_required:' + label + identity)
                if 'proposition' in row: self.eq(row['proposition'], writer['proposition'], 'judged_public_text:' + label + identity)
                if 'fact_id' in row: self.eq(row['fact_id'], fact['fact_id'], 'judged_fact_identity:' + label + identity)
                self.yes(isinstance(row.get('citations'),list) and row['citations'], 'source_citations_required:' + label + identity)
                for citation in row['citations']: self.citations(citation, source, 'review_exact_citation:' + label + identity)
                agrees = row['relation'] == writer['relation']
                expected_rows[key].append({idkey:identity, 'usable':bool(j['stem_valid'] and row['scope_valid'] and row['fact_distinct'] and agrees),
                    'relation':row['relation'], 'reason':row['reason'] + ' Writer relation ' + ('agrees.' if agrees else 'differs; retain original label and exclude.')})
        expected = {'version':'v71_asset_private_crosscheck_v1','round_id':rid,'source_sha256':sha(run/'SOURCE.json'),
            'assets_sha256':sha(folder/'ASSETS.json'),'public_assets_sha256':sha(folder/'PUBLIC_ASSETS.json'),
            'source_judgments_sha256':sha(folder/'SOURCE_JUDGMENTS.json'),'solver_scores_used':False,'source_first':True,
            'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,
            'stem_valid':j['stem_valid'], **expected_rows}
        self.eq(r, expected, 'independent_private_crosscheck:' + label)
        for k,v in {'version':'v71_asset_source_first_receipt_v1','family_id':run.name,'round_id':rid,'source_first':True,
                'parent_all_true_provenance_known':True,'new_variant_private_artifacts_read_before_save':False,'solver_scores_used':False,
                'source_judgments_saved_at_utc':j['created_at_utc'],'usable_parents':sum(p['usable'] for p in expected['parents']),
                'usable_contradicts':sum(v['usable'] for v in expected['versions'])}.items():
            self.eq(receipt[k], v, 'review_receipt:' + label + ':' + k)
        self.yes(datetime.fromisoformat(j['created_at_utc']) <= datetime.fromisoformat(receipt['private_crosscheck_started_at_utc'])
            <= datetime.fromisoformat(receipt['review_completed_at_utc']), 'source_first_timestamp_order:' + label)
        self.declared_files(receipt['files'], 'round_review_files:' + label)
        approved = {v['version_id'] for v in expected['versions'] if v['usable']}
        return {'round_id':rid,'folder':folder,'judgment':j,'review':expected,'public':public,'assets':assets,'mapping':mapping,
            'approved':approved,'approved_fact_ids':sorted(maps[v]['fact_id'] for v in approved)}

    def generation_round(self, run, rid, manifest, parent, source, previous=None):
        folder = run/'rounds'/rid; label = run.name + ':' + rid
        rm = read(folder/'MANIFEST.json'); complete = read(folder/'GENERATION.COMPLETE.json')
        expected_rm = {**manifest, 'round_id':rid, 'budget':{'http_attempts_max':1},
            'root_manifest':{'path':str(run/'MANIFEST.json'),'sha256':sha(run/'MANIFEST.json')}}
        for k,v in expected_rm.items(): self.eq(rm[k], v, 'round_manifest:' + label + ':' + k)
        targets, feedback, feedback_paths = FACTS, None, []
        if rid == 'supplement':
            targets = supplement_targets(previous); feedback = []
            self.yes(len(set(previous['approved_fact_ids'])) < 4, 'supplement_shortage_only:' + label)
            maps = {v['version_id']:v for v in previous['mapping']['versions']}
            reviewed = {maps[v['version_id']]['fact_id']:v for v in previous['judgment']['versions']}
            for fid in targets:
                row = reviewed.get(fid)
                feedback.append({'fact_id':fid,'independent_relation':row['relation'] if row else None,
                    'reason':row['reason'] if row else 'No mechanically accepted alternative was available.',
                    'scope_valid':row['scope_valid'] if row else None,'fact_distinct':row['fact_distinct'] if row else None,
                    'mechanical_errors':[e['error'] for e in previous['mechanical_errors'] if e['fact_id'] in (None,fid)]})
            feedback_paths = [previous['folder']/n for n in ('PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json','REVIEW_RECEIPT.json','GENERATION.COMPLETE.json')]
        self.eq(rm['target_fact_ids'], targets, 'only_unapproved_targets:' + label)
        self.eq(rm['content_feedback'], feedback, 'saved_source_feedback_only:' + label)
        self.eq(sorted(rm['review_feedback_files'],key=lambda r:r['path']), sorted([{'path':str(p),'sha256':sha(p)} for p in feedback_paths],key=lambda r:r['path']), 'feedback_binding_set:' + label)
        self.declared_files(rm['review_feedback_files'], 'feedback_files:' + label)
        request = read(folder/'INITIAL_REQUEST.json'); proof = read(folder/'INITIAL_PROOF.json')
        self.eq(request['max_tokens'],6144,'generation_output_cap:' + label)
        context = json.JSONDecoder().raw_decode(request['messages'][0]['content'].split('\nINPUT_JSON\n',1)[1])[0]
        for k,v in {'source_turns':source['turns'],'source_metadata':{k:v for k,v in source.items() if k!='turns'},
                'parent_facts':[{'fact_id':f,'parent_label':l,'proposition':parent['options'][l]} for f,l in zip(FACTS,LABELS)],
                'required_public_stem':parent['question'],'target_fact_ids':targets,'generation_round':rid,
                'source_mode':'parent_exact_full_source','new_expansion_tokens':0,'formal_training_admission':False}.items():
            self.eq(context[k],v,'writer_complete_scope:' + label + ':' + k)
        self.eq(context.get('independent_content_feedback'),feedback,'writer_feedback:' + label)
        self.yes(not any(k in context for k in ('gold','target_count','target_counts','parent_candidate_id','mechanical_repair')), 'no_writer_gold_or_retry:' + label)
        for k,v in {'attempt':1 if rid=='initial' else 2,'round_id':rid,'target_fact_ids':targets,
                'payload_sha256':object_digest(request),'current_solver_feedback':False,
                'source_sha256':sha(run/'SOURCE.json'),'parent_sha256':sha(run/'PARENT.json')}.items():
            self.eq(proof[k],v,'round_frozen_proof:' + label + ':' + k)
        requests = sorted((folder/'requests').glob('*/REQUEST.json')); http = list((folder/'requests').glob('*/HTTP_ATTEMPT.json'))
        self.eq(len(requests),1,'one_request_per_round:' + label); self.yes(len(http)<=1,'no_http_retry:' + label)
        self.yes((folder/'GENERATION.STARTED.json').exists(),'round_started:' + label)
        data, errors = None, []
        for path in requests:
            actual_proof = read(path.parent/'PROOF.json')
            self.eq(read(path),request,'frozen_actual_request:' + label)
            self.yes(set(actual_proof) in (set(proof),set(proof)|{'capacity'}),'actual_proof_keys:' + label)
            self.eq({k:actual_proof.get(k) for k in proof},proof,'actual_proof_values:' + label)
            if (path.parent/'HTTP_ATTEMPT.json').exists():
                attempt = read(path.parent/'HTTP_ATTEMPT.json')
                for k,v in {'transport_retries':0,'http_attempt_number':1,'actual_payload_equals_frozen':True,'payload_sha256':object_digest(request)}.items():
                    self.eq(attempt[k],v,'http_attempt:' + label + ':' + k)
                self.eq(read(path.parent/'THROTTLE.json')['minimum_gap_after_previous_response_seconds'],35.0,'global_completion_gap:' + label)
            if not (path.parent/'ERROR.json').exists():
                self.eq(len(http),1,'response_requires_http:' + label)
                response=read(path.parent/'RESPONSE.json');metadata=response['metadata']
                self.eq(metadata['returned_model'],manifest['generator_model'],'writer_model:' + label)
                self.eq(metadata['finish_reason'],'stop','writer_finish:' + label)
                self.eq(metadata.get('http_status',200),200,'writer_http_status:' + label)
                data,errors=accepted_raw_rows(response['text'],rm,parent,targets)
                validation={'mechanical_errors':errors,'fact_count':len(data['replacements']) if data else 0,
                    'counterversion_count':sum(r['contradicted'] is not None for r in data['replacements']) if data else 0,
                    'semantic_review_performed':False,'shortage_triggers_retry':False,'repair_allowed':False,
                    'program_fixed_citations':True,'rejected_rows_retained_in_raw_response':True}
                self.eq(read(path.parent/'VALIDATION.json'),validation,'independent_raw_validation:' + label)
        assets=fixed_assets(rm,parent,data); public,mapping=self.projection(folder,rm,assets,parent,source)
        for k,v in {'round_id':rid,'target_fact_ids':targets,'http_attempts':len(http),'requests_prepared':1,
                'fact_count':6,'counterversion_count':len(public['versions']),'status':'ASSETS_READY' if data else 'NO_VALID_ASSETS',
                'semantic_retries':0,'mechanical_repairs':0,'transport_retries':0,'training_allowed':False,
                'bounded_supplement':rid=='supplement'}.items():self.eq(complete[k],v,'round_completion:' + label + ':' + k)
        self.declared_files(complete['files'],'round_completion_files:' + label)
        result=self.source_review(run,folder,source,assets,public,mapping)
        result.update(http_attempts=len(http),mechanical_errors=errors,complete=complete,manifest=rm)
        return result

    def families(self,cfg,pool,members):
        receipts=read(self.root/'ASSEMBLY_RECEIPTS.json')['selections'];http_count=0;supplements=0;initial_http=0;observed=[];salts=[]
        for key,value in {'max_http':4,'max_supplemental_requests_per_family':1,'bounded_supplements_per_family':1,
                'program_fixed_citations':True,'reject_copied_parent_propositions':True,'max_questions':10,'max_calls_per_model':180}.items():
            self.eq(cfg[key],value,'bounded_config:'+key)
        for slot in cfg['slots']:
            fid=slot.get('family_id',slot['slot_id']);run=self.root/'assets'/fid
            parent=read(run/'PARENT.json');source=read(run/'SOURCE.json');manifest=read(run/'MANIFEST.json');complete=read(run/'GENERATION.COMPLETE.json')
            original=read(slot['parent_pool_path']);original=original if isinstance(original,list) else original['questions']
            self.eq([q for q in original if q['candidate_id']==slot['parent_candidate_id']],[parent],'parent_exact_pool:'+fid)
            self.eq(source['turns'],read(slot['history_path'])['turns'],'source_complete_turns:'+fid)
            self.eq((run/'SOURCE.json').read_bytes(),Path(parent['source_path']).read_bytes(),'source_exact_bytes:'+fid)
            self.eq(parent['source_sha256'],sha(run/'SOURCE.json'),'source_parent_binding:'+fid)
            self.eq(manifest['source_sha256'],sha(run/'SOURCE.json'),'source_manifest_binding:'+fid)
            self.eq(manifest['parent_sha256'],sha(run/'PARENT.json'),'parent_manifest_binding:'+fid)
            self.eq(manifest['source_mode'],'parent_exact_full_source','source_mode:'+fid)
            self.eq(manifest['budget'],{'http_attempts_max':2},'family_generation_budget:'+fid)
            self.yes(re.fullmatch('[0-9a-f]{64}',manifest['version_id_salt']),'salt_32_bytes:'+fid)
            salts.append(manifest['version_id_salt'])
            for k,v in {'version':'v71_program_bound_parent_replacements_v1','training_allowed':False,
                    'formal_training_admission':False,'current_solver_feedback':False,'new_expansion_tokens':0,
                    'source_turn_count':len(source['turns']),'source_metadata_preserved':True}.items():
                self.eq(manifest[k],v,'frozen_family_contract:'+fid+':'+k)
            for k,v in {'fixed_parent_facts':6,'max_alternatives_per_fact':1,'http_attempts_max_per_family':2,
                    'bounded_supplements_per_family':1,'program_fixed_citations':True,'reject_copied_parent_propositions':True,
                    'mechanical_repair_max':0,'semantic_retries':0,'transport_retries':0,'current_solver_feedback':False,
                    'target_count_in_writer_input':False,'supported_stem_source_gold_immutable':True,'training_allowed':False}.items():
                self.eq(manifest['contract'][k],v,'generation_contract:'+fid+':'+k)
            self.declared_files(manifest['files'],'family_frozen_files:'+fid)
            round_names=sorted(p.name for p in (run/'rounds').iterdir())
            self.yes(round_names in (['initial'],['initial','supplement']),'initial_plus_at_most_one_supplement:'+fid)
            rounds=[self.generation_round(run,'initial',manifest,parent,source)]
            initial_http+=rounds[0]['http_attempts']
            if 'supplement' in round_names: rounds.append(self.generation_round(run,'supplement',manifest,parent,source,rounds[0]))
            supplements+=len(rounds)-1;family_http=sum(r['http_attempts'] for r in rounds);http_count+=family_http
            self.yes(family_http<=2,'family_http_bound:'+fid)
            assets,accepted,jversions,rversions=merge_approved(manifest,parent,rounds)
            public,mapping=self.projection(run,manifest,assets,parent,source)
            provenance_rows=[]
            for rr in rounds:
                folder=rr['folder']
                row={'round_id':rr['round_id'],'target_fact_ids':rr['manifest']['target_fact_ids'],
                    'status':rr['complete']['status'],'http_attempts':rr['http_attempts'],'approved_distinct_false_facts':len(set(rr['approved_fact_ids']))}
                for key,name in [('public_assets','PUBLIC_ASSETS.json'),('source_judgments','SOURCE_JUDGMENTS.json'),('asset_review','ASSET_REVIEW.json'),('review_receipt','REVIEW_RECEIPT.json')]:
                    row[key]={'path':str(folder/name),'sha256':sha(folder/name)}
                provenance_rows.append(row)
            provenance={'version':'v71_round_review_provenance_v1','family_id':fid,'rounds':provenance_rows,
                'accepted_versions':accepted,'source_first_per_round':True,'solver_scores_used':False,'aggregate_is_new_independent_blind_review':False}
            self.eq(read(run/'ROUND_PROVENANCE.json'),provenance,'round_provenance:'+fid)
            expected_j=deepcopy(rounds[0]['judgment']);expected_j.update(version='v71_derived_round_source_judgments_v1',
                review_mode='derived_from_saved_round_source_judgments',derived_from_rounds=True,source_first_per_round=True,
                new_variant_private_artifacts_read_before_save=True,created_at_utc=None,versions=jversions,provenance_rounds=provenance_rows,
                public_assets_sha256=sha(run/'PUBLIC_ASSETS.json'))
            expected_r=deepcopy(rounds[0]['review']);expected_r.update(version='v71_derived_round_private_crosschecks_v1',
                review_mode='derived_round_private_crosschecks',source_first_per_round=True,new_variant_private_artifacts_read_before_save=True,
                versions=rversions,provenance_rounds=provenance_rows)
            for index,row in enumerate(expected_r['parents']):row['usable']=all(rr['review']['parents'][index]['usable'] for rr in rounds)
            expected_r['stem_valid']=all(rr['review']['stem_valid'] for rr in rounds);expected_j['stem_valid']=expected_r['stem_valid']
            self.eq(read(run/'SOURCE_JUDGMENTS.json'),expected_j,'honest_derived_judgments:'+fid)
            expected_r.update(source_sha256=sha(run/'SOURCE.json'),assets_sha256=sha(run/'ASSETS.json'),
                public_assets_sha256=sha(run/'PUBLIC_ASSETS.json'),source_judgments_sha256=sha(run/'SOURCE_JUDGMENTS.json'))
            self.eq(read(run/'ASSET_REVIEW.json'),expected_r,'honest_derived_private_review:'+fid)
            receipt=read(run/'REVIEW_RECEIPT.json')
            for k,v in {'version':'v71_derived_round_review_receipt_v1','derived_from_saved_round_reviews':True,
                    'source_first_per_round':True,'aggregate_is_new_independent_blind_review':False,'provenance_rounds':provenance_rows}.items():
                self.eq(receipt[k],v,'derived_review_receipt:'+fid+':'+k)
            self.declared_files(receipt['files'],'aggregate_review_files:'+fid)
            for k,v in {'status':'ASSETS_READY','fact_count':6,'counterversion_count':len(accepted),'requests_prepared':len(rounds),
                    'http_attempts':family_http,'bounded_supplements':len(rounds)-1,'rounds':provenance_rows,'training_allowed':False,
                    'formal_training_admission':False,'semantic_retries':0,'mechanical_repairs':0,'transport_retries':0}.items():
                self.eq(complete[k],v,'family_completion:'+fid+':'+k)
            self.declared_files(complete['files'],'family_completion_files:'+fid)
            family=[q for q in pool if members[q['candidate_id']]['family_id']==fid]
            self.eq([q for q in family if members[q['candidate_id']]['role']=='control6'],[parent],'control_immutable:'+fid)
            child_counts=Counter(len(q['gold']) for q in family if members[q['candidate_id']]['role']=='child')
            self.yes(set(child_counts)<=set(COUNTS) and all(n<=1 for n in child_counts.values()),'one_child_per_requested_count:'+fid)
            factmap={f['fact_id']:f for f in assets['facts']};mapped={v['version_id']:v for v in mapping['versions']}
            for q in family:
                member=members[q['candidate_id']]
                self.eq(member['fixed_fact_ids'],FACTS,'fixed_family_ids:'+q['candidate_id'])
                self.eq(member['split_group'],q['root_id'],'same_root_split:'+q['candidate_id'])
                self.eq(member['new_candidate'],member['role']=='child','role_new_candidate:'+q['candidate_id'])
                self.eq(q['question'],parent['question'],'child_stem_immutable:'+q['candidate_id'])
                if member['role']=='control6':continue
                self.yes(all(p['usable'] for p in expected_r['parents']),'children_require_all_valid_parents:'+q['candidate_id'])
                child_receipt=next(r for r in receipts if r['candidate_id']==q['candidate_id'])
                selected=child_receipt['selected_versions'];self.eq(sorted(r['fact_id'] for r in selected),FACTS,'child_each_fact_once:'+q['candidate_id'])
                truth=[]
                for row in selected:
                    fact=factmap[row['fact_id']];version=fact[row['variant']];label=row['label']
                    self.eq(q['options'][label],version['proposition'],'child_version_text:'+q['candidate_id']+label)
                    self.eq(row['verified_relation'],version['relation'],'child_reviewed_relation:'+q['candidate_id']+label)
                    self.eq(q['option_evidence'][label],{**version['anchor'],'relation':version['relation'],'rationale':version['rationale'],'reason':version['rationale']},'child_exact_citation:'+q['candidate_id']+label)
                    if row['variant']=='supported':truth.append(label)
                    else:self.eq(mapped[row['version_id']]['fact_id'],row['fact_id'],'child_approved_version_map:'+q['candidate_id']+label)
                self.eq(q['gold'],sorted(truth),'child_gold_from_reviewed_versions:'+q['candidate_id'])
            observed.append({'family_id':fid,'control_id':parent['candidate_id'],'fixed_fact_count':6,'counterversions':len(accepted),
                'rounds':len(rounds),'http_attempts':family_http,'bounded_supplements':len(rounds)-1})
        self.yes(http_count<=4,'global_generation_http_bound');self.yes(supplements<=2,'global_supplement_bound')
        self.eq(len(set(salts)),len(salts),'independent_family_salts')
        global_complete=read(self.root/'GENERATION.COMPLETE.json')
        self.eq(global_complete['http_attempts'],http_count,'aggregate_generation_http')
        self.eq(global_complete['bounded_supplements'],supplements,'aggregate_supplements')
        self.eq(read(self.root/'INITIAL_GENERATION.COMPLETE.json')['http_attempts'],initial_http,'aggregate_initial_http')
        self.eq(len(global_complete['slots']),2,'global_completion_two_families')
        for k in ('training_allowed','automatic_followup'):self.eq(global_complete[k],False,'global_generation_flag:'+k)
        for k in ('semantic_retries','mechanical_repairs','transport_retries'):self.eq(global_complete[k],0,'global_generation_retry:'+k)
        return observed

    def run(self):
        for name in ('RESULT_BINDINGS.json','EXECUTION_PLAN.json','EVALUATION_PREPARATION.COMPLETE.json'):
            self.recursive_bindings(self.root/name)
        cfg=read(self.root/'CONFIG.json');summary=read(self.root/'RESULT_SUMMARY.json')
        self.recursive_bindings(cfg['baseline_result_bindings'])
        self.eq(cfg['training_allowed'],False,'config_training');self.eq(summary['training_started'],False,'result_no_training')
        pool=read(self.root/'POOL.json')['questions'];members={m['candidate_id']:m for m in read(self.root/'FAMILY_SIDECAR.json')['members']}
        self.yes(len(pool)<=cfg['max_questions'],'assembled_question_budget')
        self.eq(set(members),{q['candidate_id'] for q in pool},'family_members_coverage')
        self.eq(len(cfg['slots']),2,'two_families');self.eq(len({s['root_id'] for s in cfg['slots']}),2,'two_roots')
        families=self.families(cfg,pool,members)
        review=read(self.root/'FINAL_CONTENT_REVIEW.json');freeze=read(self.root/'FINAL_CONTENT_REVIEW.COMPLETE.json')
        self.eq(review['solver_scores_used'],False,'final_content_no_solver')
        self.eq(review['pool_sha256'],sha(self.root/'POOL.json'),'final_content_pool_binding')
        statuses={r['candidate_id']:r['core_status'] for r in review['questions']}
        usable=[q for q in pool if statuses[q['candidate_id']]=='CORE_USABLE']
        self.eq(len(usable),self.expected_questions,'explicit_expected_count')
        self.eq(read(self.root/'EVALUATION_POOL.json')['questions'],usable,'evaluation_pool_exact_content_usable')
        self.eq(freeze['usable_candidate_ids'],[q['candidate_id'] for q in usable],'frozen_usable_ids')
        for q in usable:
            self.eq(sha(q['source_path']),q['source_sha256'],'evaluation_fullsource_hash:'+q['candidate_id'])
            source=read(q['source_path']);slot=next(s for s in cfg['slots'] if s['root_id']==q['root_id'])
            self.eq(source['turns'],read(slot['history_path'])['turns'],'evaluation_fullsource_turns:'+q['candidate_id'])
        manifests={};scores={};decisions={}
        for side in ('base','trained'):manifests[side],scores[side],decisions[side]=self.side(side,usable,freeze)
        if usable:
            keep=lambda m:{k:v for k,v in m.items() if k not in {'prepared_at_utc','model','solver_model','solver_identity'}}
            self.eq(keep(manifests['base']),keep(manifests['trained']),'paired_manifests_whitelist')
            for left,right in zip(manifests['base']['requests'],manifests['trained']['requests']):
                rid=left['request_id'];self.eq((self.root/'evaluation/base/prompts'/(rid+'.txt')).read_bytes(),(self.root/'evaluation/trained/prompts'/(rid+'.txt')).read_bytes(),'paired_prompt_bytes:'+rid)
                # Every complete source turn must appear verbatim in the full-context prompt.
                source=read(next(q['source_path'] for q in usable if q['candidate_id']==left['candidate_id']))
                prompt=(self.root/'evaluation/base/prompts'/(rid+'.txt')).read_text()
                for t in source['turns']:self.yes(t['text'] in prompt,'full_source_prompt_turn:'+rid+':'+t['turn_ref'])
        selected_ids={d['candidate_id'] for d in decisions['trained'] if d['selected_by_policy']}
        selected_all=[q for q in usable if q['candidate_id'] in selected_ids]
        selected=[q for q in selected_all if members[q['candidate_id']]['role']=='child']
        baseline=read(cfg['baseline_pool'])['questions'];merged=list(baseline);seen={statement_key(q) for q in baseline};added=[]
        for q in selected:
            key=statement_key(q)
            if key not in seen:merged.append(q);added.append(q);seen.add(key)
            else:
                original=next(x for x in merged if statement_key(x)==key)
                content_gold=lambda x:sorted(' '.join(x['options'][l].casefold().split()) for l in x['gold'])
                self.eq(content_gold(q),content_gold(original),'merge_duplicate_gold_content:'+q['candidate_id'])
        exports={'ASSEMBLED':pool,'CONTENT_USABLE':usable,'SELECTED':selected,'ADDED':added,'CONTROLS':[q for q in pool if members[q['candidate_id']]['role']=='control6'],'MERGED_USABLE':merged}
        for name,qs in exports.items():self.eq(read(self.root/(name+'_CANDIDATES.json'))['questions'],qs,'exact_export:'+name)
        self.eq(merged[:len(baseline)],baseline,'baseline_verbatim_prefix')
        self.eq(len({q['candidate_id'] for q in merged}),len(merged),'merged_unique_ids')
        self.yes(all(members[q['candidate_id']]['role']=='child' for q in added),'controls_never_new_admission')
        counts={'baseline_count':len(baseline),'assembled_count':len(pool),'assembled_child_count':len(pool)-2,'control_count':2,
            'content_usable_count':len(usable),'content_usable_child_count':sum(members[q['candidate_id']]['role']=='child' for q in usable),
            'selected_count':len(selected),'selected_control_count':len(selected_all)-len(selected),'added_count':len(added),'merged_count':len(merged),'calls_per_model':18*len(usable)}
        for k,v in counts.items():self.eq(summary[k],v,'summary_count:'+k)
        stageqs={'assembled':pool,'content_usable':usable,'evaluated':usable,'selected':selected_all,'merged':added};stages=read(self.root/'FAMILY_STAGES.json')
        for stage,qs in stageqs.items():
            children=[q for q in qs if members[q['candidate_id']]['role']=='child'];self.eq(stages['children_stages'][stage],distribution(children),'children_distribution:'+stage)
            hist=Counter(len(q['gold']) for q in children);targets={str(k):cfg['target_distribution'][str(k)] for k in COUNTS}
            gaps={str(k):max(0,targets[str(k)]-hist[k]) for k in COUNTS}
            coverage={'required_counts':list(COUNTS),'targets':targets,'actual_counts':{str(k):hist[k] for k in COUNTS},
                'shortfalls':gaps,'missing_counts':[k for k in COUNTS if not hist[k]],
                'all_required_counts_present':all(hist[k]>0 for k in COUNTS),'all_target_counts_met':not any(gaps.values()),
                'scope':'current_batch_children_only','controls_counted_as_new':False,'target_is_not_actual_coverage':True}
            self.eq(stages['required_count_coverage'][stage],coverage,'stage_required_2345_coverage:'+stage)
            if stage!='merged':self.eq(stages['control_stages'][stage],distribution([q for q in qs if members[q['candidate_id']]['role']=='control6']),'control_distribution:'+stage)
            for family in stages['families']:
                fid=family['family_id']
                for k in COUNTS:self.eq(family['children_by_count'][str(k)][stage],sum(members[q['candidate_id']]['family_id']==fid and members[q['candidate_id']]['role']=='child' and len(q['gold'])==k for q in qs),'family_stage:'+fid+':'+stage+':'+str(k))
                self.eq(family['parent_control']['newly_merged'],False,'control_never_new:'+fid)
                self.eq(family['parent_control']['retained_in_baseline'],family['parent_candidate_id'] in {q['candidate_id'] for q in baseline},'control_old_admission_unaffected:'+fid)
        for name,qs in {'assembled':pool,'content_usable':usable,'evaluated':usable,'selected':selected,'added':added,'controls':exports['CONTROLS'],'baseline':baseline,'merged':merged}.items():self.eq(summary['distributions'][name],distribution(qs),'summary_distribution:'+name)
        byside={s:{r['request_id']:r for r in scores[s]} for s in scores};self.eq(set(byside['base']),set(byside['trained']),'paired_response_ids')
        pairs=[]
        for rid,b in byside['base'].items():
            t=byside['trained'][rid];self.eq(b['candidate_id'],t['candidate_id'],'paired_candidate:'+rid)
            pairs.append({'candidate_id':b['candidate_id'],'base':b,'trained':t})
        comparison=read(self.root/'BASE_TRAINED_COMPARISON.json');childrenpairs=[p for p in pairs if members[p['candidate_id']]['role']=='child'];controlpairs=[p for p in pairs if members[p['candidate_id']]['role']=='control6']
        for name,ps in [('primary',childrenpairs),('control6',controlpairs),('all_evaluated_descriptive',pairs),('selected_children_secondary',[p for p in childrenpairs if p['candidate_id'] in selected_ids])]:self.eq(comparison[name],compare(ps),'paired_comparison:'+name)
        for k in COUNTS:self.eq(comparison['by_correct_count'][str(k)],compare([p for p in childrenpairs if members[p['candidate_id']]['gold_count']==k]),'comparison_gold_count:'+str(k))
        for fid,entry in comparison['by_family'].items():
            for name,ps in [('children',childrenpairs),('control6',controlpairs)]:self.eq(entry[name],compare([p for p in ps if members[p['candidate_id']]['family_id']==fid]),'comparison_family:'+fid+name)
            for k in COUNTS:self.eq(entry['children_by_correct_count'][str(k)],compare([p for p in childrenpairs if members[p['candidate_id']]['family_id']==fid and members[p['candidate_id']]['gold_count']==k]),'comparison_family_count:'+fid+str(k))
        for side in ('base','trained'):
            groups=[compare([p for p in childrenpairs if members[p['candidate_id']]['family_id']==fid])[side] for fid in {m['family_id'] for m in members.values()}]
            groups=[g for g in groups if g['calls']]
            self.eq(comparison['equal_family_weight_children'][side],{'contributing_families':len(groups),'equal_family_weight_partial_mean':str(sum((Fraction(g['partial_mean']) for g in groups),Fraction())/len(groups)) if groups else None,'equal_family_weight_exact_rate':str(sum((Fraction(g['exact_rate']) for g in groups),Fraction())/len(groups)) if groups else None},'equal_family_weight:'+side)
        rawpairs={p['request_id']:p for p in read(self.root/'PAIRED_RAW_SCORES.json')['pairs']}
        self.eq(set(rawpairs),set(byside['base']),'raw_pair_export_coverage')
        for rid in byside['base']:
            for side in ('base','trained'):
                for key in ('exact','partial','tp','fp','fn','selected_count','gold_count'):self.eq(rawpairs[rid][side][key],byside[side][rid][key],'raw_score_export:'+rid+side+key)
        blind=read(self.root/'BLIND_ALL_SIX_BASELINE.json')
        for name,qs in [('controls',exports['CONTROLS']),('content_usable_children',[q for q in usable if members[q['candidate_id']]['role']=='child']),('selected_children',selected),('merged',merged)]:
            rows=[]
            for q in qs:
                if q['question_type']!='multi_select':continue
                v=variants(q)[0];gold=[cid(q['options'][l]) for l in q['gold']]
                score=rescore('<box>A,B,C,D,E,F</box>',v['display_to_content'],gold)
                rows.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'exact':score['exact_match'],'partial':score['partial_score_fraction'],
                    'tp':score['tp'],'fp':score['fp'],'fn':score['fn'],'selected_count':6,'gold_count':len(q['gold'])})
            self.eq(blind[name]['questions'],rows,'blind_all_six_rows:'+name)
            self.eq(blind[name]['aggregate'],aggregate(rows),'blind_all_six_aggregate:'+name)
            for k in range(2,7):self.eq(blind[name]['by_correct_count'][str(k)],aggregate([r for r in rows if r['gold_count']==k]),'blind_by_gold_count:'+name+str(k))
        return {'counts':counts,'families':families,'children_stage_distributions':stages['children_stages'],'selected_child_ids':[q['candidate_id'] for q in selected],'added_child_ids':[q['candidate_id'] for q in added],'unique_hash_bound_files':len(self.bound)}


def statement_key(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    return q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values()))

def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate_json_key')
            result[k]=v
        return result
    def constant(x):raise ValueError('nonfinite_json_number')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def self_test():
    # These fixtures probe contract violations without reading real generation.
    parent={'candidate_id':'parent','question':'Which facts are supported?',
        'options':dict(zip(LABELS,['Fact '+l for l in LABELS])),
        'option_evidence':{l:{'turn_ref':'T'+l,'speaker':'Sam','quote':'Fact '+l,'rationale':'Exact source.'} for l in LABELS}}
    manifest={'root_id':'root','slot_id':'family','family_id':'family','parent_sha256':'a'*64,
        'source_sha256':'b'*64,'version_id_salt':'01'*32,'round_id':'initial'}
    raw={'root_id':'root','family_id':'family','replacements':[{'fact_id':f,'contradicted':None,
        'contradiction_unavailable_reason':'No contradiction.'} for f in FACTS]}
    raw['replacements'][0].update(contradicted={'proposition':'Different timing.','relation':'contradicts','rationale':'Explicit timing differs.'},contradiction_unavailable_reason=None)
    for copied in ('FACT A',' f-a_c.t A! ','ＦＡＣＴ Ａ'):
        candidate=deepcopy(raw);candidate['replacements'][0]['contradicted']['proposition']=copied
        parsed,errors=accepted_raw_rows(json.dumps(candidate),manifest,parent,FACTS)
        assert parsed['replacements'][0]['contradicted'] is None
        assert errors==[{'fact_id':'F01','error':'copied_supported_parent_proposition'}]
    candidate=deepcopy(raw);candidate['replacements'][0]['anchor']={'turn_ref':'invented'}
    parsed,errors=accepted_raw_rows(json.dumps(candidate),manifest,parent,FACTS)
    assert errors==[{'fact_id':'F01','error':'replacement_schema'}] and parsed['replacements'][0]['contradicted'] is None
    candidate=deepcopy(raw);candidate['replacements'].reverse()
    assert accepted_raw_rows(json.dumps(candidate),manifest,parent,FACTS)[0] is None
    parsed,errors=accepted_raw_rows(json.dumps(raw),manifest,parent,FACTS);assert not errors
    assets=fixed_assets(manifest,parent,parsed)
    assert assets['facts'][0]['contradicted']['anchor']=={k:parent['option_evidence']['A'][k] for k in ('turn_ref','speaker','quote')}
    assert all(f['supported']['proposition']==parent['options'][f['parent_label']] for f in assets['facts'])
    pub,private=projected_versions(manifest,assets)
    supplement_pub,_=projected_versions({**manifest,'round_id':'supplement'},assets)
    assert pub[0]['version_id']!=supplement_pub[0]['version_id']
    assert private[0]['internal_version_id']=='F01:c:initial'
    first={'round_id':'initial','assets':assets,'mapping':{'versions':private},'approved':{pub[0]['version_id']},
        'approved_fact_ids':['F01'],'judgment':{'versions':[pub[0]]},'review':{'parents':[{'usable':True} for f in FACTS],'versions':[{'version_id':pub[0]['version_id'],'usable':True}]}}
    assert supplement_targets(first)==FACTS[1:]
    for invalid in ({**first,'approved_fact_ids':FACTS[:4]}, {**first,'review':{'parents':[{'usable':False}]}}):
        try:supplement_targets(invalid)
        except ValueError:pass
        else:raise AssertionError('ineligible supplement accepted')
    merged,accepted,_,_=merge_approved(manifest,parent,[first])
    final_pub,_=projected_versions({k:v for k,v in manifest.items() if k!='round_id'},merged)
    assert final_pub[0]['version_id']==pub[0]['version_id']
    assert merged['facts'][0]['origin']=={'round_id':'initial','version_id':pub[0]['version_id']}
    assert accepted==[{'fact_id':'F01','version_id':pub[0]['version_id'],'round_id':'initial'}]
    try:merge_approved(manifest,parent,[first,{**first,'round_id':'supplement'}])
    except ValueError as error:assert str(error)=='accepted_initial_version_overwritten'
    else:raise AssertionError('approved initial overwritten')
    mapping=dict(zip(LABELS,['x'+l for l in LABELS]))
    assert rescore('<box>B,A</box>',mapping,['xA','xB'])['exact_match']
    assert rescore('<box>A,C</box>',mapping,['xA','xB'])['partial_score_fraction']=='1/4'
    assert rescore('<box>C,D,E,F</box>',mapping,['xA','xB'])['partial_score_fraction']=='-1'
    for k in range(2,6):assert rescore('<box>A,B,C,D,E,F</box>',mapping,list(mapping.values())[:k])['partial_score_fraction']=='0'
    assert rescore('<box>A,B,C,D,E,F</box>',mapping,list(mapping.values()))['partial_score_fraction']=='1'
    for raw in ('<box>A,A</box>','<box>G</box>','text <box>A</box>','<box>A</box><box>B</box>'):assert not rescore(raw,mapping,['xA','xB'])['scored']
    assert FORBIDDEN.search("She's proposed Tuesday") and not FORBIDDEN.search('Alex proposed Tuesday')
    q={'candidate_id':'q','root_id':'r','question_type':'multi_select','options':dict(zip(LABELS,['fact '+l for l in LABELS]))}
    vv=variants(q);assert len({tuple(v['display_order_content_ids']) for v in vv})==6
    rows=[{'candidate_id':'q','variant_id':f'perm_{i:02d}','partial':'1/4','exact':False} for i in range(6) for j in range(3)]
    assert select_question(q,rows)['selected_by_policy']
    for val in ('0','1','-1'):
        assert not select_question(q,[{**r,'partial':val} for r in rows])['selected_by_policy']
    for raw in ('{"a":1,"a":2}','{"a":NaN}'):
        try:strict_json(raw)
        except ValueError:pass
        else:raise AssertionError('ambiguous JSON accepted')
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);child=root/'child.json';child.write_text('{}\n');outer=root/'outer.json';outer.write_text(json.dumps({'files':[{'path':str(child),'sha256':sha(child)}]}))
        audit=Audit(root,0);audit.recursive_bindings(outer);assert not audit.failures
        child.write_text('{"changed":true}\n');audit=Audit(root,0);audit.recursive_bindings(outer);assert audit.failures

        # Relative ledgers, historical metadata boundaries, duplicate cache and
        # conflicting hash expectations must each retain their intended behavior.
        child.write_text('{}\n')
        outer.write_text(json.dumps({'files':[{'path':'child.json','sha256':sha(child)},
            {'path':'child.json','sha256':sha(child)}],
            'records':[{'path':'absent_historical.txt','sha256':'0'*64}]}))
        audit=Audit(root,0);audit.recursive_bindings(outer)
        assert not audit.failures and len(audit.hash_cache)==1 and audit.ledger_rows==2
        audit.binding(child,'0'*64,'conflicting_expected_hash');assert audit.failures

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--expected-questions',type=int);p.add_argument('--output-dir',type=Path);args=p.parse_args();self_test()
    if args.self_test:print('READY: stdlib synthetic self-tests PASS; no real results read.');return
    if args.expected_questions is None or not 0<=args.expected_questions<=10:raise SystemExit('--expected-questions must be actual frozen usable count 0–10')
    for name in ('RESULT_SUMMARY.json','RESULT_BINDINGS.json','TRAINED_EVALUATION.COMPLETE.json','BASE_EVALUATION.COMPLETE.json'):
        if not (args.root/name).exists():raise SystemExit('NOT_READY: '+name)
    audit=Audit(args.root,args.expected_questions);report={'audit_kind':'independent_standard_library_raw_recalculation','production_audit_functions_called':False,
        'model_calls':0,'frozen_files_modified':0,'content_review_reopened':False,'script_sha256':sha(__file__),'started_at_utc':datetime.now(timezone.utc).isoformat()}
    try:report['recomputed']=audit.run()
    except Exception:audit.failures.append({'check':'audit_exception','traceback':traceback.format_exc()})
    report.update(binding_scope='recursive_RESULT_BINDINGS_ancestry_current_v71_files_and_current_checkpoint_files',historical_environment_reproducibility_claimed=False,explicit_binding_rows=audit.ledger_rows,cached_path_expected_hash_pairs=len(audit.hash_cache),status='PASS' if not audit.failures else 'FAIL',checks_completed=audit.checks,failures=audit.failures,completed_at_utc=datetime.now(timezone.utc).isoformat(),result_summary_sha256=sha(args.root/'RESULT_SUMMARY.json'))
    output=args.output_dir or args.root/'execution/independent_audit';output.mkdir(parents=True,exist_ok=True)
    with (output/'INDEPENDENT_RESULT_AUDIT.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=str);f.write('\n')
    with (output/'INDEPENDENT_RESULT_AUDIT.md').open('x') as f:f.write('# v71 independent audit\n\nStatus: '+report['status']+'\n\n'+str(audit.checks)+' checks; '+str(len(audit.failures))+' differences.\n\nStdlib raw scoring, 6×3 call sets, original selection policy, paired inputs, immutable parent families, source citations, retained positive text, control exclusion, stage counts, baseline preservation, dedup merge, and explicit frozen file-ledger hashes. Per-round source-first review receipts, fixed parent anchors, copy rejection, bounded supplements and derived final review provenance are checked. Historical environment reproducibility is not claimed. No model calls or frozen-file edits.\n')
    print(json.dumps({'status':report['status'],'checks':audit.checks,'failures':len(audit.failures),'output':str(output)}))
    if audit.failures:raise SystemExit(1)

if __name__=='__main__':main()
