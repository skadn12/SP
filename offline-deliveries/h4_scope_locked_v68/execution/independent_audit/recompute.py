"""Independent standard-library-only audit. Never imports production evaluators/scorers.

Run only after finalization and both evaluation stages complete (or explicitly skip an empty pool).
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
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
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


def field_counts(facts):
    counts = {key: 0 for key in ['object_facts', 'supported_versions', 'contradicted_versions',
        'contradicted_explicit_null', 'contradicted_field_missing', 'contradicted_invalid_nonnull',
        'reason_field_missing_all_facts', 'reason_field_invalid_all_facts', 'reason_condition_unknown_all_facts',
        'null_reason_nonempty', 'null_reason_field_missing', 'null_reason_empty_null_or_invalid']}
    reasons = Counter()
    for item in facts:
        if not isinstance(item, dict):
            continue
        counts['object_facts'] += 1
        reason_key = 'contradiction_unavailable_reason'
        condition_known = 'contradicted' in item and (item['contradicted'] is None or isinstance(item['contradicted'], dict))
        counts['reason_condition_unknown_all_facts'] += not condition_known
        if reason_key not in item:
            counts['reason_field_missing_all_facts'] += 1
        else:
            reason = item[reason_key]
            valid_text = isinstance(reason, str) and bool(reason.strip())
            valid_reason = (valid_text if item['contradicted'] is None else reason is None) if condition_known else (reason is None or valid_text)
            counts['reason_field_invalid_all_facts'] += not valid_reason
        if isinstance(item.get('supported'), dict):
            counts['supported_versions'] += 1
        if 'contradicted' not in item:
            counts['contradicted_field_missing'] += 1
        elif isinstance(item['contradicted'], dict):
            counts['contradicted_versions'] += 1
        elif item['contradicted'] is not None:
            counts['contradicted_invalid_nonnull'] += 1
        else:
            counts['contradicted_explicit_null'] += 1
            key = 'contradiction_unavailable_reason'
            if key not in item:
                counts['null_reason_field_missing'] += 1
            elif not isinstance(item[key], str) or not item[key].strip():
                counts['null_reason_empty_null_or_invalid'] += 1
            else:
                counts['null_reason_nonempty'] += 1
                reasons[item[key]] += 1
    return {'fact_entries': len(facts), **counts, 'null_reason_verbatim_histogram': dict(sorted(reasons.items())),
        'reason_all_facts_definition': 'Missing counts absent fields on all object facts. Invalid counts present values violating null/nonempty-string type or the condition of a valid contradicted field. Unknown condition is recorded separately when contradicted is missing or invalid. Counts are mechanical, not semantic.',
        'null_reasons_semantically_verified': False, 'unknown_or_missing_reason_is_not_counterevidence': True}


def review_counts(rows):
    counts = dict(usable_supported_versions=0, usable_contradicted_versions=0,
                  reviewed_unknown_versions=0, reviewed_unusable_versions=0)
    for row in rows:
        relation, usable = row.get('relation'), row.get('usable') is True
        if usable and relation == 'supports': counts['usable_supported_versions'] += 1
        if usable and relation == 'contradicts': counts['usable_contradicted_versions'] += 1
        if relation == 'unknown': counts['reviewed_unknown_versions'] += 1
        if not usable: counts['reviewed_unusable_versions'] += 1
    return counts


def raw_evidence(home, config):
    """Independent JSON traversal; never loads production generation/report modules."""
    requests, slots, all_raw, all_accepted, all_reviews = [], [], [], [], []
    for spec in config['slots']:
        run = Path(home) / 'assets' / spec['slot_id']
        raws = []
        for path in sorted((run / 'requests').glob('*/RESPONSE.json')):
            status, facts = 'UNPARSEABLE_JSON', []
            try:
                obj = json.loads(read(path).get('text', ''))
            except (TypeError, ValueError):
                obj = None
            else:
                status = 'FACT_LIST_MISSING_OR_INVALID'
                if isinstance(obj, dict) and isinstance(obj.get('facts'), list):
                    facts, status = obj['facts'], 'FACT_LIST_PRESENT'
            requests.append({'slot_id': spec['slot_id'], 'root_id': spec['root_id'],
                'response': {'path': str(path.resolve()), 'sha256': sha(path)},
                'parse_status': status, 'field_statistics': field_counts(facts)})
            raws.extend(facts)
        asset_path, review_path = run / 'ASSETS.json', run / 'ASSET_REVIEW.json'
        assets = read(asset_path)['facts'] if asset_path.exists() else []
        review = read(review_path)['versions'] if review_path.exists() else []
        slots.append({'slot_id': spec['slot_id'], 'root_id': spec['root_id'],
            'accepted_assets_present': asset_path.exists(), 'raw_attempt_fields': field_counts(raws),
            'accepted_asset_fields': field_counts(assets), **review_counts(review)})
        all_raw.extend(raws); all_accepted.extend(assets); all_reviews.extend(review)
    return {'scope': 'raw_attempts_include_retries_and_rejected_responses; accepted_assets_and_reviews_count_each_slot_once',
        'raw_response_count': len(requests),
        'raw_response_parse_statuses': dict(Counter(r['parse_status'] for r in requests)),
        'raw_attempt_fields': field_counts(all_raw), 'accepted_asset_fields': field_counts(all_accepted),
        **review_counts(all_reviews),
        'null_reason_interpretation': 'Verbatim writer explanations only; nonempty, unknown or missing reasons do not establish contradiction. Usability comes only from source-first content review.',
        'requests': requests, 'slots': slots}


def repair_observations(root, config):
    def binding(path):
        return {'path': str(path.resolve()), 'sha256': sha(path)}
    request_rows, pairs, duplicate_texts = [], [], {}
    for spec in config['slots']:
        previous_text = previous_errors = previous_hash = previous_folder = None
        for request_file in sorted((Path(root) / 'assets' / spec['slot_id'] / 'requests').glob('*/REQUEST.json')):
            folder = request_file.parent
            context, status = None, 'INPUT_JSON_UNAVAILABLE'
            try:
                content = read(request_file)['messages'][0]['content']
                input_tail = content.split('\nINPUT_JSON\n', 1)[1]
                context = json.JSONDecoder().raw_decode(input_tail)[0]
                status = 'PARSED' if isinstance(context, dict) else 'INPUT_JSON_NOT_OBJECT'
            except (KeyError, IndexError, ValueError, TypeError):
                pass
            repair = context.get('mechanical_repair') if isinstance(context, dict) else None
            has_repair = isinstance(repair, dict) if status == 'PARSED' else None
            hint_rows = repair.get('reference_hints') if has_repair else None
            hint_count = len(hint_rows) if isinstance(hint_rows, list) else (0 if has_repair and 'reference_hints' not in repair else None)
            response_file, validation_file = folder / 'RESPONSE.json', folder / 'VALIDATION.json'
            raw = read(response_file).get('text') if response_file.exists() else None
            validation = read(validation_file) if validation_file.exists() else None
            errs = validation.get('mechanical_errors') if isinstance(validation, dict) else None
            if not isinstance(errs, list): errs = None
            outcome = 'UNKNOWN_NO_VALIDATION' if errs is None else 'FAILED' if len(errs) else 'PASSED'
            raw_hash = hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None
            request_rows.append({'slot_id': spec['slot_id'], 'root_id': spec['root_id'], 'attempt_folder': folder.name,
                'request': binding(request_file), 'context_status': status,
                'remote_http_attempt': (folder / 'HTTP_ATTEMPT.json').exists(),
                'repair_prompt_present': has_repair, 'repair_errors': repair.get('errors') if has_repair else None,
                'reference_hint_count': hint_count, 'reference_hints': hint_rows if isinstance(hint_rows, list) else None,
                'previous_response_matches_recorded_raw': repair.get('previous_response') == previous_text
                    if has_repair and isinstance(previous_text, str) else None,
                'prior_mechanical_errors': previous_errors,
                'repair_errors_match_prior_validation': repair.get('errors') == previous_errors
                    if has_repair and previous_errors is not None else None,
                'validation_outcome': outcome, 'mechanical_errors': errs,
                'response': binding(response_file) if response_file.exists() else None,
                'validation': binding(validation_file) if validation_file.exists() else None,
                'raw_text_sha256': raw_hash, 'transport_error_recorded': (folder / 'ERROR.json').exists()})
            if isinstance(raw, str):
                duplicate_texts.setdefault(raw_hash, []).append({'slot_id': spec['slot_id'],
                    'attempt_folder': folder.name, 'response': binding(response_file)})
                if isinstance(previous_text, str):
                    pairs.append({'slot_id': spec['slot_id'], 'first_attempt': previous_folder,
                        'second_attempt': folder.name, 'raw_text_exactly_equal': previous_text == raw,
                        'first_raw_text_sha256': previous_hash, 'second_raw_text_sha256': raw_hash})
            previous_text, previous_errors, previous_hash, previous_folder = raw, errs, raw_hash, folder.name
    repairs = [row for row in request_rows if row['repair_prompt_present'] is True]
    remote = [row for row in repairs if row['remote_http_attempt']]
    states = Counter(row['validation_outcome'] for row in remote)
    return {'scope': 'recorded_INPUT_JSON.mechanical_repair_and_validation; exact_raw_text_comparison_only',
        'causal_repair_effect_claimed': False, 'prepared_requests': len(request_rows),
        'repair_prompt_requests': len(repairs), 'repair_http_attempts': len(remote),
        'repair_passed': states['PASSED'], 'repair_failed': states['FAILED'], 'repair_outcome_unknown': states['UNKNOWN_NO_VALIDATION'],
        'repair_requests_with_reference_hints': sum(bool(row['reference_hint_count']) for row in repairs),
        'reference_hint_count': sum(row['reference_hint_count'] or 0 for row in repairs),
        'hinted_repair_passed': sum(bool(row['reference_hint_count']) and row['validation_outcome'] == 'PASSED' for row in remote),
        'comparable_within_slot_response_pairs': len(pairs),
        'exact_repeated_within_slot_response_pairs': sum(row['raw_text_exactly_equal'] for row in pairs),
        'exact_repeated_response_groups': [{'raw_text_sha256': key, 'responses': rows}
            for key, rows in sorted(duplicate_texts.items()) if len(rows) > 1],
        'requests': request_rows, 'within_slot_response_pairs': pairs,
        'interpretation': 'Pass/fail follows recorded mechanical validation after a repair request. Missing prompt, response or validation evidence is not counted as a successful repair. Exact repetition does not diagnose provider caching. These observations do not establish causal repair effectiveness.'}


def locked_semantics(raw):
    """Independently recover immutable meaning-bearing fields for a repair baseline.

    Missing non-null-counterstatement reasons may be filled with JSON null; an
    unavailable counterstatement still requires its existing nonempty reason.
    Anchor corrections and removal of unrecognized schema keys are outside this
    projection and must be checked separately against the mechanical errors.
    """
    try:
        value = json.loads(raw)
        if not isinstance(value, dict): return None
        result = {k:value[k] for k in ['root_id','slot_id','public_stem']}
        if any(not isinstance(v,str) or not v.strip() for v in result.values()): return None
        facts = value['facts']
        if not isinstance(facts,list) or len(facts) > 8: return None
        result['facts'] = []
        ids = set()
        for fact in facts:
            if not isinstance(fact,dict): return None
            fid = fact['fact_id']
            if not isinstance(fid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',fid) or fid in ids: return None
            ids.add(fid)
            scope = {k:fact['scope'][k] for k in ['actor','event','time','condition','action_state']}
            if any(not isinstance(v,str) or not v.strip() for v in scope.values()): return None
            item = {'fact_id':fid,'scope':scope}
            for key, relation in [('supported','supports'),('contradicted','contradicts')]:
                branch = fact[key]
                if key == 'contradicted' and branch is None:
                    item[key] = None
                    continue
                if not isinstance(branch,dict): return None
                branch = {k:branch[k] for k in ['proposition','relation','rationale']}
                if branch['relation'] != relation: return None
                if any(not isinstance(branch[k],str) or not branch[k].strip() for k in ['proposition','rationale']): return None
                item[key] = branch
            reason = fact.get('contradiction_unavailable_reason')
            if item['contradicted'] is None:
                if not isinstance(reason,str) or not reason.strip(): return None
            elif reason is not None:
                return None
            item['contradiction_unavailable_reason'] = reason
            result['facts'].append(item)
        return result
    except (KeyError,TypeError,ValueError):
        return None


def guard_observations(root, cfg):
    records=[]
    counts=Counter()
    for slot in cfg['slots']:
        run=Path(root)/'assets'/slot['slot_id']
        for request in sorted((run/'requests').glob('*/REQUEST.json')):
            directory=request.parent
            receipt=directory/'REPAIR_GUARD.json'
            data=read(receipt) if receipt.exists() else None
            if data is not None: counts[data['status']]+=1
            records.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],'attempt_folder':directory.name,
                'request':{'path':str(request.resolve()),'sha256':sha(request)},
                'remote_http_attempt':(directory/'HTTP_ATTEMPT.json').exists(),
                'response_present':(directory/'RESPONSE.json').exists(),
                'guard_receipt':{'path':str(receipt.resolve()),'sha256':sha(receipt)} if data is not None else None,
                'guard':data,'accepted_assets_present':(run/'ASSETS.json').exists()})
    blocked={r['slot_id'] for r in records if r['guard'] is not None and
        r['guard']['status'] in ['BASELINE_BLOCKED','REPAIR_BLOCKED']}
    return {'scope':'persisted_mechanical_repair_semantic_lock_receipts',
        'source_semantic_review_still_required':True,'prepared_requests':len(records),
        'response_requests':sum(r['response_present'] for r in records),
        'guard_receipts':sum(r['guard'] is not None for r in records),
        'missing_guard_for_responses':sum(r['response_present'] and r['guard'] is None for r in records),
        'statuses':dict(sorted(counts.items())),
        'initial_mechanically_valid_count':counts['NOT_NEEDED'],
        'initial_repair_allowed_count':counts['REPAIR_ALLOWED'],
        'initial_repair_blocked_count':counts['BASELINE_BLOCKED'],
        'repair_response_accepted_count':counts['REPAIR_ACCEPTED'],
        'repair_response_blocked_count':counts['REPAIR_BLOCKED'],
        'http_requests_without_response_or_guard':sum(r['remote_http_attempt'] and not r['response_present']
            and r['guard'] is None for r in records),
        'blocked_slots':sorted(blocked),
        'blocked_slots_with_assets':sorted({r['slot_id'] for r in records if r['slot_id'] in blocked and r['accepted_assets_present']}),
        'requests':records,
        'interpretation':'A mechanical validation pass is not a semantic-lock pass or content endorsement. Guard decisions preserve existing fields across a repair; source-first independent review still determines truth and whole-question usability. Missing response/guard evidence is not a repair success.'}


def strict_json(raw):
    def unique(items):
        keys=[k for k,v in items]
        if len(keys) != len(set(keys)): raise ValueError('duplicate key')
        return dict(items)
    def nonfinite(value): raise ValueError('not a JSON number')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)


def content_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def allowed_guard_edits(data, root_id, source):
    """Derive fixed edits solely from raw schema and exact current-source turns."""
    edits=[]
    def extras(obj, names, at):
        if isinstance(obj,dict):
            edits.extend({'path':at+[key],'operation':'delete'} for key in sorted(set(obj)-set(names)))
    def set_value(at,value): edits.append({'path':at,'operation':'set','value':value})
    extras(data,['root_id','slot_id','public_stem','facts'],[])
    if not isinstance(data,dict) or not isinstance(data.get('facts'),list): return edits
    for index,fact in enumerate(data['facts']):
        if not isinstance(fact,dict): continue
        at=['facts',index]
        extras(fact,['fact_id','scope','anchor','supported','contradicted','contradiction_unavailable_reason'],at)
        extras(fact.get('scope'),['actor','event','time','condition','action_state'],at+['scope'])
        for name in ['supported','contradicted']:
            extras(fact.get(name),['proposition','relation','rationale'],at+[name])
        if 'contradiction_unavailable_reason' not in fact and isinstance(fact.get('contradicted'),dict):
            set_value(at+['contradiction_unavailable_reason'],None)
        anchor=fact.get('anchor')
        extras(anchor,['turn_ref','speaker','quote'],at+['anchor'])
        if not isinstance(anchor,dict): continue
        ref=anchor.get('turn_ref');quote=anchor.get('quote');speaker=anchor.get('speaker')
        direct=[t for t in source['turns'] if t['turn_ref']==ref] if isinstance(ref,str) else []
        if not direct and isinstance(ref,str) and ref.startswith(root_id+':') and source.get('root_id')==root_id:
            stripped=ref[len(root_id)+1:]
            matches=[t for t in source['turns'] if t['turn_ref']==stripped]
            if len(matches)==1 and speaker==matches[0]['speaker'] and isinstance(quote,str) and quote.strip() and quote in matches[0]['text']:
                set_value(at+['anchor','turn_ref'],stripped);direct=matches
        if len(direct)==1:
            turn=direct[0]
            if speaker != turn['speaker']: set_value(at+['anchor','speaker'],turn['speaker'])
            if not (isinstance(quote,str) and quote.strip() and quote in turn['text']):
                set_value(at+['anchor','quote'],turn['text'])
        elif source.get('root_id')==root_id and isinstance(speaker,str) and speaker.strip() and isinstance(quote,str) and quote.strip():
            matches=[t for t in source['turns'] if t['speaker']==speaker and quote in t['text']]
            if len(matches)==1: set_value(at+['anchor','turn_ref'],matches[0]['turn_ref'])
    return edits


def edit_tree(value, edits, mask=False):
    result=json.loads(json.dumps(value))
    for edit in edits:
        obj=result
        for key in edit['path'][:-1]: obj=obj[key]
        key=edit['path'][-1]
        if mask or edit['operation']=='delete': obj.pop(key,None)
        else: obj[key]=edit['value']
    return result


def tree_differences(left,right,at=None):
    at=[] if at is None else at
    if type(left) is not type(right): return [at]
    if isinstance(left,dict):
        out=[]
        for key in sorted(set(left)|set(right)):
            out += [at+[key]] if key not in left or key not in right else tree_differences(left[key],right[key],at+[key])
        return out
    if isinstance(left,list):
        if len(left)!=len(right): return [at]
        return [path for i in range(len(left)) for path in tree_differences(left[i],right[i],at+[i])]
    return [] if left==right else [at]


def mechanically_valid_asset(value, spec, source):
    def keys(obj,names): return isinstance(obj,dict) and set(obj)==set(names)
    def text(value): return isinstance(value,str) and bool(value.strip())
    if not keys(value,['root_id','slot_id','public_stem','facts']): return False
    if value['root_id']!=spec['root_id'] or value['slot_id']!=spec['slot_id'] or not text(value['public_stem']): return False
    if not isinstance(value['facts'],list) or len(value['facts'])>8: return False
    ids=[]
    for fact in value['facts']:
        if not keys(fact,['fact_id','scope','anchor','supported','contradicted','contradiction_unavailable_reason']): return False
        fid=fact['fact_id']
        if not isinstance(fid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',fid) or fid in ids: return False
        ids.append(fid)
        if not keys(fact['scope'],['actor','event','time','condition','action_state']) or not all(text(v) for v in fact['scope'].values()): return False
        anchor=fact['anchor']
        if not keys(anchor,['turn_ref','speaker','quote']) or not all(text(v) for v in anchor.values()): return False
        matches=[t for t in source['turns'] if t['turn_ref']==anchor['turn_ref']]
        prefix=spec['root_id']+':'
        if not matches and source.get('root_id')==spec['root_id'] and anchor['turn_ref'].startswith(prefix):
            matches=[t for t in source['turns'] if t['turn_ref']==anchor['turn_ref'][len(prefix):]]
        if len(matches)!=1 or anchor['speaker']!=matches[0]['speaker'] or anchor['quote'] not in matches[0]['text']: return False
        for name,relation in [('supported','supports'),('contradicted','contradicts')]:
            branch=fact[name]
            if name=='contradicted' and branch is None: continue
            if not keys(branch,['proposition','relation','rationale']) or branch['relation']!=relation or not text(branch['proposition']) or not text(branch['rationale']): return False
        reason=fact['contradiction_unavailable_reason']
        if (not text(reason)) if fact['contradicted'] is None else (reason is not None): return False
    return True


class Audit:
    def __init__(self, root, expected_questions):
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

    def generation_usage(self, cfg, summary, usable_count):
        """Count HTTP receipts; missing provider usage is never estimated or zeroed."""
        rows = []
        for slot in cfg['slots']:
            run = self.root / 'assets' / slot['slot_id']
            complete = read(run / 'GENERATION.COMPLETE.json')
            local = []
            for request in sorted((run / 'requests').glob('*/REQUEST.json')):
                folder = request.parent
                attempted = (folder / 'HTTP_ATTEMPT.json').exists()
                response = read(folder / 'RESPONSE.json') if (folder / 'RESPONSE.json').exists() else {}
                metadata = response.get('metadata', {})
                usage = metadata.get('usage') or {}
                inp = metadata.get('input_tokens', usage.get('prompt_tokens'))
                out = metadata.get('output_tokens', usage.get('completion_tokens'))
                status = ('known' if type(inp) is int and type(out) is int else 'unknown') if attempted else 'not_called'
                local.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],
                    'request':{'path':str(request.resolve()),'sha256':sha(request)},
                    'remote_http_attempt':attempted,'reported_input_tokens':inp,'reported_output_tokens':out,
                    'token_usage_status':status,'token_usage_source':'provider_response_metadata_no_estimation',
                    'error':read(folder / 'ERROR.json') if (folder / 'ERROR.json').exists() else None})
            self.eq(complete['requests_prepared'], len(local), 'generation:prepared:' + slot['slot_id'])
            self.eq(complete['http_attempts'], sum(row['remote_http_attempt'] for row in local), 'generation:http:' + slot['slot_id'])
            self.yes(len(local) <= read(run / 'MANIFEST.json')['budget']['http_attempts_max'], 'generation:slot_budget:' + slot['slot_id'])
            rows.extend(local)
        attempted = [r for r in rows if r['remote_http_attempt']]
        known = [r for r in attempted if r['token_usage_status'] == 'known']
        all_known = len(known) == len(attempted)
        inputs = sum(r['reported_input_tokens'] for r in known)
        outputs = sum(r['reported_output_tokens'] for r in known)
        expected = {'http_attempts':len(attempted),'prepared_requests':len(rows),'known_usage_http_attempts':len(known),
            'unknown_usage_http_attempts':len(attempted)-len(known),'usage_complete':all_known,
            'input_tokens':inputs if all_known else 'unknown','output_tokens':outputs if all_known else 'unknown',
            'known_input_token_subtotal':inputs,'known_output_token_subtotal':outputs,
            'money_cost':'unknown_no_frozen_provider_pricing','standalone_planner_calls':0,
            'content_usable_questions':usable_count,
            'http_per_content_usable_question':str(Fraction(len(attempted),usable_count)) if usable_count else None}
        self.eq(summary['generation_cost'], expected, 'generation:cost_unknown_usage_and_subtotals')
        self.eq(read(self.root / 'GENERATION_COST.json'), {'summary':expected,'requests':rows}, 'generation:cost_export')
        self.eq(read(self.root / 'GENERATION.COMPLETE.json')['http_attempts'], len(attempted), 'generation:aggregate_http')
        self.yes(len(attempted) <= cfg['max_http'], 'generation:configured_http_budget')
        return expected

    def repair_guards(self,cfg,summary,pool,usable,selected):
        observed=guard_observations(self.root,cfg)
        self.eq(summary['repair_guard_statistics'],observed,'guard:summary_from_raw_receipts')
        self.eq(read(self.root/'REPAIR_SEMANTIC_GUARD_STATISTICS.json'),observed,'guard:export')
        self.eq(observed['missing_guard_for_responses'],0,'guard:all_responses_have_receipts')
        self.eq(observed['blocked_slots_with_assets'],[],'guard:blocked_have_no_assets')
        assembly=read(self.root/'ASSEMBLY_RECEIPTS.json')['selections']
        blocked=set(observed['blocked_slots'])
        forbidden={row['candidate_id'] for row in assembly if row['slot_id'] in blocked}
        for label,questions in [('assembled',pool),('usable',usable),('selected',selected)]:
            self.eq(sorted(forbidden & {q['candidate_id'] for q in questions}),[],'guard:blocked_absent_'+label)
        for spec in cfg['slots']:
            run=self.root/'assets'/spec['slot_id']
            source=read(run/'SOURCE.json')
            requests=sorted((run/'requests').glob('*/REQUEST.json'))
            baseline_raw=None;baseline_errors=None
            for index,request in enumerate(requests,1):
                folder=request.parent
                response_path=folder/'RESPONSE.json'
                if not response_path.exists():
                    self.yes(not (folder/'REPAIR_GUARD.json').exists(),'guard:no_receipt_without_response:'+spec['slot_id'])
                    continue
                raw=read(response_path)['text']
                errors=read(folder/'VALIDATION.json')['mechanical_errors']
                try: raw_schema_valid=mechanically_valid_asset(json.loads(raw),spec,source)
                except (TypeError,ValueError): raw_schema_valid=False
                self.eq(not bool(errors),raw_schema_valid,'guard:'+spec['slot_id']+':'+str(index)+':mechanical_schema_outcome')
                if index==1: baseline_raw,baseline_errors=raw,errors
                receipt=read(folder/'REPAIR_GUARD.json')
                label='guard:'+spec['slot_id']+':'+str(index)+':'
                self.eq(receipt['version'],'v68_mechanical_repair_semantic_lock_v1',label+'version')
                self.eq(receipt['attempt'],index,label+'attempt')
                for key in ['root_id','slot_id']: self.eq(receipt[key],spec[key],label+key)
                self.eq(receipt['source_sha256'],sha(run/'SOURCE.json'),label+'source_bytes')
                self.eq(receipt['source_content_sha256'],content_digest(source),label+'source_content')
                self.eq(receipt['baseline_raw_text_sha256'],hashlib.sha256(baseline_raw.encode()).hexdigest(),label+'baseline_raw')
                self.eq(receipt['candidate_raw_text_sha256'],hashlib.sha256(raw.encode()).hexdigest() if index==2 else None,label+'candidate_raw')
                state='BASELINE_BLOCKED';repair_allowed=False;accepted=False
                edits=[];changed=[];unauthorized=[];base_hash=None;candidate_hash=None
                before=after=None
                try:
                    before=strict_json(baseline_raw)
                except (TypeError,ValueError):
                    pass
                else:
                    edits=allowed_guard_edits(before,spec['root_id'],source)
                    try:
                        expected=edit_tree(before,edits)
                        base_hash=content_digest(edit_tree(before,edits,mask=True))
                    except (TypeError,ValueError,KeyError,IndexError):
                        pass
                    else:
                        if mechanically_valid_asset(expected,spec,source):
                            if not baseline_errors:
                                state='NOT_NEEDED' if index==1 else 'REPAIR_BLOCKED';accepted=index==1
                            else:
                                state='REPAIR_ALLOWED';repair_allowed=True
                                if index==2:
                                    state='REPAIR_BLOCKED'
                                    try: after=strict_json(raw)
                                    except (TypeError,ValueError): pass
                                    else:
                                        changed=tree_differences(before,after)
                                        by_path={tuple(row['path']):row for row in edits}
                                        for path in changed:
                                            edit=by_path.get(tuple(path));valid=edit is not None
                                            try:
                                                obj=after
                                                for part in path[:-1]: obj=obj[part]
                                                if valid:
                                                    valid=(path[-1] not in obj) if edit['operation']=='delete' else (path[-1] in obj and type(obj[path[-1]]) is type(edit['value']) and obj[path[-1]]==edit['value'])
                                            except (KeyError,IndexError,TypeError): valid=False
                                            if not valid: unauthorized.append(path)
                                        try: candidate_hash=content_digest(edit_tree(after,edits,mask=True))
                                        except (KeyError,IndexError,TypeError): pass
                                        else:
                                            if not unauthorized and base_hash==candidate_hash and mechanically_valid_asset(after,spec,source):
                                                state='REPAIR_ACCEPTED';accepted=True
                self.eq(receipt['allowed_changes'],edits,label+'allowed_from_schema_and_source')
                self.eq(receipt['changed_paths'],changed,label+'all_raw_changes')
                self.eq(receipt['unauthorized_changes'],unauthorized,label+'unauthorized_changes')
                self.eq(receipt['baseline_projection_sha256'],base_hash,label+'baseline_projection')
                self.eq(receipt['candidate_projection_sha256'],candidate_hash,label+'candidate_projection')
                self.eq(receipt['status'],state,label+'decision')
                self.eq(receipt['repair_allowed'],repair_allowed,label+'repair_authorization')
                self.eq(receipt['accepted'],accepted,label+'accepted')
                self.eq(bool(receipt['lock_errors']),state in ['BASELINE_BLOCKED','REPAIR_BLOCKED'],label+'errors_match_decision')
                if state=='BASELINE_BLOCKED': self.eq(len(requests),1,label+'no_retry_when_baseline_unavailable')
                if state=='REPAIR_ACCEPTED':
                    self.eq(locked_semantics(baseline_raw),locked_semantics(raw),label+'immutable_meaning_fields')
                    self.yes(locked_semantics(raw) is not None,label+'meaning_fields_available')
                if index==2:
                    self.eq(read(requests[0].parent/'REPAIR_GUARD.json')['status'],'REPAIR_ALLOWED',label+'prior_guard_authorized')
            if spec['slot_id'] in blocked:
                self.eq(read(run/'GENERATION.COMPLETE.json')['status'],'NO_VALID_ASSETS','guard:blocked_slot_status:'+spec['slot_id'])
        return observed

    def protocol_exclusions(self, pool, usable, selected, baseline, merged, review, freeze):
        path = self.root / 'execution' / 'PROTOCOL_EXCLUSIONS.json'
        if not path.exists():
            return {'present': False, 'excluded_slots': [], 'excluded_candidate_ids': []}
        protocol = read(path)
        self.eq(protocol.get('before_assembly'), True, 'protocol:declared_before_assembly')
        self.eq(protocol.get('before_solver'), True, 'protocol:declared_before_solver')
        self.eq(protocol.get('solver_scores_used'), False, 'protocol:no_solver_scores')
        self.eq(protocol.get('followup_http_attempts'), 0, 'protocol:no_replacement_calls')
        self.eq(protocol.get('frozen_files_modified'), 0, 'protocol:no_frozen_edits')
        declared = datetime.fromisoformat(protocol['recorded_at_utc'])
        self.yes(declared < datetime.fromisoformat(freeze['completed_at_utc']), 'protocol:declared_before_final_content_freeze')
        for side in ['base', 'trained']:
            start_path = self.root / 'evaluation' / side / 'STARTED.json'
            if start_path.exists():
                self.yes(declared < datetime.fromisoformat(read(start_path)['started_at_utc']), 'protocol:declared_before_' + side)
        exclusions = protocol['exclusions']
        slot_ids = [row['slot_id'] for row in exclusions]
        self.eq(len(set(slot_ids)), len(slot_ids), 'protocol:unique_excluded_slots')
        assembly = read(self.root / 'ASSEMBLY_RECEIPTS.json')['selections']
        candidate_ids = [row['candidate_id'] for row in assembly if row['slot_id'] in slot_ids]
        pool_ids = {q['candidate_id'] for q in pool}
        status = {row['candidate_id']: row['core_status'] for row in review['questions']}
        for qid in candidate_ids:
            self.yes(qid in pool_ids, 'protocol:assembled_candidate_exists:' + qid)
            self.eq(status.get(qid), 'KNOWN_SUBSTANTIVE_ISSUE', 'protocol:entire_question_excluded:' + qid)
        forbidden = set(candidate_ids)
        for name, questions in [('evaluation_pool', usable), ('selected_pool', selected)]:
            self.eq(sorted(forbidden & {q['candidate_id'] for q in questions}), [], 'protocol:absent_from_' + name)
        new_merged = {q['candidate_id'] for q in merged} - {q['candidate_id'] for q in baseline}
        self.eq(sorted(forbidden & new_merged), [], 'protocol:absent_from_new_merged')
        for row in exclusions:
            evidence_rows = row['response_bindings'] + row['request_bindings'] + [row['semantic_review'], row['source_judgments']]
            for evidence in evidence_rows:
                self.binding(evidence['path'], evidence['sha256'], 'protocol:exclusion_evidence:' + row['slot_id'])
            self.yes(isinstance(row.get('reason'), str) and row['reason'].strip(), 'protocol:reason_present:' + row['slot_id'])
            self.eq(set(row['excluded_stages']), {'content_usable', 'evaluation', 'difficulty_selected', 'merged_new'}, 'protocol:excluded_stages:' + row['slot_id'])
        return {'present': True, 'protocol_sha256': sha(path), 'excluded_slots': slot_ids,
                'excluded_candidate_ids': candidate_ids, 'content_review_unchanged_by_audit': True}

    def evidence_and_history(self, cfg, summary):
        current = raw_evidence(self.root, cfg)
        self.eq(summary['evidence_statistics'], current, 'evidence:summary_all_raw_accepted_reviewed_fields')
        self.eq(read(self.root / 'EVIDENCE_STATISTICS.json'), current, 'evidence:export_all_fields')
        observed_repairs = repair_observations(self.root, cfg)
        self.eq(summary['mechanical_repair_statistics'], observed_repairs, 'repair:all_recorded_requests_outcomes_repeats')
        self.eq(read(self.root / 'MECHANICAL_REPAIR_STATISTICS.json'), observed_repairs, 'repair:export')
        for row in observed_repairs['requests']:
            if row['repair_prompt_present'] is not True: continue
            self.eq(row['previous_response_matches_recorded_raw'], True, 'repair:previous_raw_bound:' + row['slot_id'])
            self.eq(row['repair_errors_match_prior_validation'], True, 'repair:previous_errors_bound:' + row['slot_id'])
            source = read(self.root / 'assets' / row['slot_id'] / 'SOURCE.json')
            self.eq(source['root_id'], row['root_id'], 'repair:same_source_root:' + row['slot_id'])
            for hint in row['reference_hints'] or []:
                matches = [turn['turn_ref'] for turn in source['turns'] if turn['speaker'] == hint['speaker']
                    and isinstance(hint['quote'], str) and hint['quote'] and hint['quote'] in turn['text']]
                self.eq(matches, [hint['candidate_turn_ref']], 'repair:unique_exact_source_hint:' + row['slot_id'] + ':' + str(hint['fact_index']))
        comparison = read(self.root / 'DESCRIPTIVE_V67_COMPARISON.json')
        self.eq(summary['descriptive_v67_comparison'], comparison, 'history:comparison_export')
        previous_home = Path(cfg['baseline_pool']).parent
        previous = read(previous_home / 'RESULT_SUMMARY.json')
        previous_cfg = read(previous_home / 'CONFIG.json')
        previous_stats = raw_evidence(previous_home, previous_cfg)
        self.eq(comparison['previous_evidence_statistics'], previous_stats, 'history:previous_raw_evidence_fields')
        self.eq(comparison['previous_mechanical_repair_statistics'], repair_observations(previous_home, previous_cfg), 'history:previous_repair_records')
        self.eq(comparison['allocation_config'], {'previous_target_distribution': previous_cfg.get('target_distribution'),
            'current_target_distribution': cfg.get('target_distribution'),
            'previous_priority_counts': previous_cfg.get('allocation_priority_counts'),
            'current_priority_counts': cfg.get('allocation_priority_counts')}, 'history:allocation_changes')
        self.binding(previous_home / 'RESULT_SUMMARY.json', comparison['previous_summary']['sha256'], 'history:summary')
        self.binding(previous_home / 'CONFIG.json', comparison['previous_config']['sha256'], 'history:config')
        self.eq(comparison['comparison_kind'], 'descriptive_sequential_batches', 'history:comparison_type')
        self.eq(comparison['strict_randomized_control'], False, 'history:not_rct')
        self.eq(comparison['causal_prompt_effect_claimed'], False, 'history:no_causal_prompt_claim')
        old_by_slot = {s['slot_id']: s for s in previous_cfg['slots']}
        self.eq(comparison['same_slot_set'], set(old_by_slot) == {s['slot_id'] for s in cfg['slots']}, 'history:same_slot_set')
        old_roots, new_roots = {s['root_id'] for s in previous_cfg['slots']}, {s['root_id'] for s in cfg['slots']}
        self.eq(comparison['previous_root_ids'], sorted(old_roots), 'history:previous_roots')
        self.eq(comparison['current_root_ids'], sorted(new_roots), 'history:current_roots')
        self.eq(comparison['overlapping_root_ids'], sorted(old_roots & new_roots), 'history:overlapping_roots')
        actual_checks, actual_priors = [], []
        for spec in cfg['slots']:
            old = old_by_slot.get(spec['slot_id'], {})
            actual_checks.append({'slot_id': spec['slot_id'],
                'same_root_and_source_sha256': spec['root_id'] == old.get('root_id') and spec['history_sha256'] == old.get('history_sha256'),
                'same_focus': spec['focus'] == old.get('focus'),
                'same_initial_target_count': spec['initial_target_count'] == old.get('initial_target_count')})
            paths = [old.get('prior_pool_paths', []), spec['prior_pool_paths']]
            ids = [{q['candidate_id'] for path in side for q in read(path)['questions']} for side in paths]
            actual_priors.append({'slot_id': spec['slot_id'], 'previous_count': len(ids[0]), 'current_count': len(ids[1]),
                'added_candidate_ids': sorted(ids[1] - ids[0]), 'removed_candidate_ids': sorted(ids[0] - ids[1]),
                'previous_files': [{'path': str(Path(path).resolve()), 'sha256': sha(path)} for path in paths[0]],
                'current_files': [{'path': str(Path(path).resolve()), 'sha256': sha(path)} for path in paths[1]]})
            run = self.root / 'assets' / spec['slot_id']
            if (run / 'ASSETS.json').exists():
                for index, fact in enumerate(read(run / 'ASSETS.json')['facts']):
                    if fact['contradicted'] is None:
                        reason = fact.get('contradiction_unavailable_reason')
                        self.yes(isinstance(reason, str) and reason.strip(), 'evidence:accepted_null_reason_nonempty:' + spec['slot_id'] + ':' + str(index))
                    else:
                        self.eq(fact.get('contradiction_unavailable_reason'), None, 'evidence:accepted_counterstatement_reason_null:' + spec['slot_id'] + ':' + str(index))
                judgment = read(run / 'SOURCE_JUDGMENTS.json')
                review = read(run / 'ASSET_REVIEW.json')
                writer = {v['version_id']: v for v in read(run / 'VERSION_MAP.json')['versions']}
                by_id = {v['version_id']: v for v in judgment['versions']}
                for reviewed in review['versions']:
                    vid = reviewed['version_id']; j = by_id[vid]
                    expected_usable = (j['relation'] != 'unknown' and j['relation'] == writer[vid]['relation']
                        and j['scope_valid'] and j['fact_distinct'] and judgment['stem_valid'] and review['stem_valid'])
                    self.eq(reviewed['usable'], expected_usable, 'evidence:usable_from_source_judgments:' + vid)
                    self.eq(reviewed['relation'], j['relation'], 'evidence:independent_relation:' + vid)
        self.eq(comparison['slot_checks'], actual_checks, 'history:per_slot_source_focus_target')
        self.eq(comparison['historical_prior_changes'], actual_priors, 'history:actual_prior_added_removed_ids')
        self.eq(comparison['same_sources_and_focus'], comparison['same_slot_set'] and all(r['same_root_and_source_sha256'] and r['same_focus'] for r in actual_checks), 'history:source_focus_claim')
        for key, result, evidence in [('v67', previous, previous_stats), ('v68', summary, current)]:
            metrics = comparison[key]
            for field in ['assembled_count', 'content_usable_count', 'known_issue_count', 'selected_count', 'merged_count', 'calls_per_model', 'distributions']:
                self.eq(metrics[field], result[field], 'history:' + key + ':' + field)
            self.eq(metrics['http_attempts'], result['generation_cost']['http_attempts'], 'history:' + key + ':http')
            self.eq(metrics['base_trained_primary'], result['base_trained_comparison']['primary'], 'history:' + key + ':model_primary')
            for target, container, field in [('raw_contradicted_versions', 'raw_attempt_fields', 'contradicted_versions'),
                ('accepted_contradicted_versions', 'accepted_asset_fields', 'contradicted_versions'),
                ('accepted_contradicted_explicit_null', 'accepted_asset_fields', 'contradicted_explicit_null'),
                ('accepted_null_reason_nonempty', 'accepted_asset_fields', 'null_reason_nonempty'),
                ('accepted_null_reason_field_missing', 'accepted_asset_fields', 'null_reason_field_missing'),
                ('raw_reason_field_missing_all_facts', 'raw_attempt_fields', 'reason_field_missing_all_facts'),
                ('raw_reason_field_invalid_all_facts', 'raw_attempt_fields', 'reason_field_invalid_all_facts'),
                ('accepted_reason_field_missing_all_facts', 'accepted_asset_fields', 'reason_field_missing_all_facts'),
                ('accepted_reason_field_invalid_all_facts', 'accepted_asset_fields', 'reason_field_invalid_all_facts')]:
                self.eq(metrics[target], evidence[container][field], 'history:' + key + ':' + target)
            self.eq(metrics['usable_contradicted_versions'], evidence['usable_contradicted_versions'], 'history:' + key + ':usable_counterstatements')
        self.eq(comparison['v68_minus_v67_counts'], {k: comparison['v68'][k] - v for k, v in comparison['v67'].items() if type(v) is int}, 'history:all_count_deltas')
        return current

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
        self.yes(len(pool) <= cfg['max_questions'] <= len(cfg['slots']), 'authorized:question_budget')
        self.yes(len(usable) * 18 <= cfg['max_calls_per_model'] <= cfg['max_questions'] * 18, 'authorized:solver_call_budget')
        manifests, sides, decisions = {}, {}, {}
        for side in ['trained', 'base']:
            manifests[side], sides[side], decisions[side] = self.side(side, usable, freeze)
        permitted = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
        self.eq({k: v for k, v in (manifests['trained'] or {}).items() if k not in permitted}, {k: v for k, v in (manifests['base'] or {}).items() if k not in permitted}, 'paired:all_non_model_manifest_fields')
        self.eq(summary['evaluation_status'], 'COMPLETED' if usable else 'SKIPPED_EMPTY_CONTENT_USABLE_POOL', 'evaluation:explicit_status')
        for req in (manifests['base'] or {}).get('requests', []):
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
        self.eq(len(baseline), 68, 'baseline:68')
        self.eq(len({q['candidate_id'] for q in baseline}), 68, 'baseline:unique_68')
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
        target = {str(k): v for k, v in cfg['target_distribution'].items()}
        self.eq(distributions['planned']['multiselect_gold_count_histogram'], target, 'allocation:configured_planned_distribution')
        self.eq(summary['target_distribution'], cfg['target_distribution'], 'summary:configured_targets')
        self.eq(summary['allocation_priority_counts'], cfg.get('allocation_priority_counts', []), 'summary:allocation_priority')
        self.eq(summary['baseline_count'], len(baseline), 'summary:actual_baseline_count')
        allocation = read(root / 'ALLOCATION.json')
        self.eq(summary['allocation'], allocation, 'allocation:summary_export')
        self.eq(allocation['target_distribution'], target, 'allocation:target_distribution')
        priority = cfg.get('allocation_priority_counts', [])
        self.eq(allocation['allocation_priority_counts'], priority, 'allocation:priority_counts')
        hist = distributions['assembled']['multiselect_gold_count_histogram']
        root_targets = {str(k): min(target[str(k)], len({slot['root_id'] for slot in cfg['slots']})) for k in range(2, 7)}
        roots_by_count = {str(k): sorted({q['root_id'] for q in pool if len(q['gold']) == k}) for k in range(2, 7)}
        self.eq(allocation['distinct_root_targets'], root_targets, 'allocation:dynamic_root_targets')
        self.eq(allocation['assembled_distribution'], hist, 'allocation:actual_distribution')
        self.eq(allocation['missing_target_counts'], {str(k): max(0, target[str(k)] - hist[str(k)]) for k in range(2, 7)}, 'allocation:missing_counts')
        self.eq(allocation['distinct_roots_by_count'], roots_by_count, 'allocation:actual_roots')
        self.eq(allocation['missing_distinct_root_coverage'], {str(k): max(0, root_targets[str(k)] - len(roots_by_count[str(k)])) for k in range(2, 7)}, 'allocation:missing_roots')
        self.eq(allocation['allocation_objective'], {
            'priority_count_coverage': {str(k): min(target[str(k)], hist[str(k)]) for k in priority},
            'target_coverage': sum(min(target[str(k)], hist[str(k)]) for k in range(2, 7)),
            'distinct_root_target_coverage': sum(min(root_targets[str(k)], len(roots_by_count[str(k)])) for k in range(2, 7)),
            'retained_questions': len(pool),
            'initial_preferences_matched': sum(row.get('assembled') is True and row['initial_target_count'] == row['actual_gold_count'] for row in allocation['slots'])}, 'allocation:observed_objective_components')
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
        self.eq(summary['new_root_ids'], sorted({s['root_id'] for s in cfg['slots']}), 'report:current_root_scope')
        self.eq(len(summary['new_root_ids']), len({s['root_id'] for s in cfg['slots']}), 'report:configured_root_count')
        self.eq(read(root / 'RESULT_BINDINGS.json')['scope'], 'all_v68_final_artifacts_and_direct_frozen_inputs_except_mutable_process_logs', 'report:result_binding_scope')
        expected_limitation = 'new_roots_and_focus_relative_to_v67; scope_instructions_repair_guard_and_historical_priors_changed; descriptive_not_randomized_control; not_second_round_training_effect'
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
        evidence = self.evidence_and_history(cfg, summary)
        usage = self.generation_usage(cfg, summary, len(usable))
        guards = self.repair_guards(cfg,summary,pool,usable,selected)
        excluded = self.protocol_exclusions(pool, usable, selected, baseline, merged, review, freeze)
        return {'counts': counts, 'selected_candidate_ids': [q['candidate_id'] for q in selected],
            'selection': decisions, 'comparison': comparison, 'distributions': distributions,
            'blind_all_six_aggregates': blind_recomputed, 'merge_duplicates': duplicates,
            'previous_bound_files_checked': len(old), 'unique_file_hashes_checked': len(self.bound),
            'per_response_recalculation': sides, 'evidence_statistics': evidence, 'generation_usage': usage,
            'repair_guards': guards, 'protocol_exclusions': excluded}


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
    fields = field_counts([{'contradicted': None, 'contradiction_unavailable_reason': 'Unknown completion.'},
        {'contradicted': None}, {'contradicted': None, 'contradiction_unavailable_reason': None},
        {'contradicted': {}}, {}, {'contradicted': False}])
    assert fields['contradicted_explicit_null'] == 3
    assert fields['contradicted_versions'] == fields['contradicted_field_missing'] == fields['contradicted_invalid_nonnull'] == 1
    assert fields['null_reason_nonempty'] == fields['null_reason_field_missing'] == fields['null_reason_empty_null_or_invalid'] == 1
    assert fields['null_reason_verbatim_histogram'] == {'Unknown completion.': 1}
    assert fields['null_reasons_semantically_verified'] is False
    assert review_counts([{'relation': 'unknown', 'usable': False}])['usable_contradicted_versions'] == 0
    assert fields['reason_field_missing_all_facts'] == 4
    assert fields['reason_field_invalid_all_facts'] == 1
    assert fields['reason_condition_unknown_all_facts'] == 2
    baseline = {'root_id':'r','slot_id':'s','public_stem':'During this event, which statements apply?',
        'facts':[{'fact_id':'F1','scope':dict(actor='A',event='E',time='T',condition='C',action_state='planned'),
            'anchor':{'turn_ref':'invalid','speaker':'A','quote':'exact quote'},
            'supported':{'proposition':'A planned E.','relation':'supports','rationale':'The speaker described a plan.'},
            'contradicted':{'proposition':'A completed E.','relation':'contradicts','rationale':'The event was explicitly pending.'}}]}
    original = locked_semantics(json.dumps(baseline))
    assert original is not None
    permitted = json.loads(json.dumps(baseline))
    permitted['facts'][0]['anchor']['turn_ref'] = 'P:T1'
    permitted['facts'][0]['contradiction_unavailable_reason'] = None
    assert locked_semantics(json.dumps(permitted)) == original
    drift = json.loads(json.dumps(permitted))
    drift['facts'][0]['supported']['proposition'] = 'A completed E.'
    assert locked_semantics(json.dumps(drift)) != original
    drift = json.loads(json.dumps(permitted))
    drift['facts'][0]['scope']['time'] = 'A different event'
    assert locked_semantics(json.dumps(drift)) != original
    incomplete = json.loads(json.dumps(baseline))
    incomplete['facts'][0]['contradicted'] = None
    assert locked_semantics(json.dumps(incomplete)) is None
    with tempfile.TemporaryDirectory(prefix='v68-independent-self-test-') as td:
        fixture = Path(td)
        errors = ['unknown_turn_ref']
        repair = {'errors': errors, 'previous_response': 'identical text',
                  'reference_hints': [{'candidate_turn_ref': 'P:T1'}]}
        for name, context in [('01', {}), ('02', {'mechanical_repair': repair})]:
            folder = fixture / 'assets' / 's' / 'requests' / name
            folder.mkdir(parents=True)
            documents = {'REQUEST.json': {'messages': [{'content': 'X\nINPUT_JSON\n' + json.dumps(context) + '\nTASK'}]},
                'RESPONSE.json': {'text': 'identical text'}, 'HTTP_ATTEMPT.json': {},
                'VALIDATION.json': {'mechanical_errors': errors if name == '01' else []}}
            for filename, value in documents.items():
                (folder / filename).write_text(json.dumps(value))
        stats = repair_observations(fixture, {'slots': [{'slot_id': 's', 'root_id': 'r'}]})
        assert stats['repair_prompt_requests'] == stats['repair_http_attempts'] == stats['repair_passed'] == 1
        assert stats['reference_hint_count'] == stats['hinted_repair_passed'] == 1
        assert stats['exact_repeated_within_slot_response_pairs'] == 1
        assert stats['requests'][1]['previous_response_matches_recorded_raw'] is True
        assert stats['requests'][1]['repair_errors_match_prior_validation'] is True
        (folder / 'VALIDATION.json').unlink()
        stats = repair_observations(fixture, {'slots': [{'slot_id': 's', 'root_id': 'r'}]})
        assert stats['repair_passed'] == 0 and stats['repair_outcome_unknown'] == 1
        execution = fixture / 'execution'
        execution.mkdir()
        proof = execution / 'proof.json'
        proof.write_text('{}')
        proof_binding = {'path': str(proof), 'sha256': sha(proof)}
        protocol = {'recorded_at_utc': '2026-01-01T00:00:00+00:00', 'before_assembly': True,
            'before_solver': True, 'solver_scores_used': False, 'followup_http_attempts': 0, 'frozen_files_modified': 0,
            'exclusions': [{'slot_id': 's', 'reason': 'SEMANTIC_DRIFT', 'response_bindings': [], 'request_bindings': [],
                'semantic_review': proof_binding, 'source_judgments': proof_binding,
                'excluded_stages': ['content_usable', 'evaluation', 'difficulty_selected', 'merged_new']}]}
        (execution / 'PROTOCOL_EXCLUSIONS.json').write_text(json.dumps(protocol))
        (fixture / 'ASSEMBLY_RECEIPTS.json').write_text(json.dumps({'selections': [{'slot_id': 's', 'candidate_id': 'q'}]}))
        auditor = Audit(fixture, 0)
        result = auditor.protocol_exclusions([{'candidate_id': 'q'}], [], [], [], [],
            {'questions': [{'candidate_id': 'q', 'core_status': 'KNOWN_SUBSTANTIVE_ISSUE'}]},
            {'completed_at_utc': '2026-01-02T00:00:00+00:00'})
        assert not auditor.failures and result['excluded_candidate_ids'] == ['q']
        auditor.protocol_exclusions([{'candidate_id': 'q'}], [{'candidate_id': 'q'}], [], [], [],
            {'questions': [{'candidate_id': 'q', 'core_status': 'CORE_USABLE'}]},
            {'completed_at_utc': '2026-01-02T00:00:00+00:00'})
        assert len(auditor.failures) == 2
    with tempfile.TemporaryDirectory(prefix='v68-usage-self-test-') as td:
        fixture = Path(td)
        run = fixture / 'assets' / 's'
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        request_rows = []
        for index in [1, 2]:
            folder = run / 'requests' / str(index)
            put(folder / 'REQUEST.json', {})
            put(folder / 'HTTP_ATTEMPT.json', {})
            if index == 1:
                put(folder / 'RESPONSE.json', {'metadata':{'usage':{'prompt_tokens':100,'completion_tokens':20}}})
            else:
                put(folder / 'ERROR.json', {'kind':'timeout'})
            request_rows.append({'slot_id':'s','root_id':'r',
                'request':{'path':str((folder / 'REQUEST.json').resolve()),'sha256':sha(folder / 'REQUEST.json')},
                'remote_http_attempt':True,'reported_input_tokens':100 if index == 1 else None,
                'reported_output_tokens':20 if index == 1 else None,'token_usage_status':'known' if index == 1 else 'unknown',
                'token_usage_source':'provider_response_metadata_no_estimation','error':None if index == 1 else {'kind':'timeout'}})
        put(run / 'GENERATION.COMPLETE.json', {'requests_prepared':2,'http_attempts':2})
        put(run / 'MANIFEST.json', {'budget':{'http_attempts_max':2}})
        put(fixture / 'GENERATION.COMPLETE.json', {'http_attempts':2})
        expected = {'http_attempts':2,'prepared_requests':2,'known_usage_http_attempts':1,
            'unknown_usage_http_attempts':1,'usage_complete':False,'input_tokens':'unknown','output_tokens':'unknown',
            'known_input_token_subtotal':100,'known_output_token_subtotal':20,
            'money_cost':'unknown_no_frozen_provider_pricing','standalone_planner_calls':0,
            'content_usable_questions':1,'http_per_content_usable_question':'2'}
        put(fixture / 'GENERATION_COST.json', {'summary':expected,'requests':request_rows})
        auditor = Audit(fixture, 1)
        config = {'slots':[{'slot_id':'s','root_id':'r'}],'max_http':2}
        auditor.generation_usage(config, {'generation_cost':expected}, 1)
        assert not auditor.failures
        auditor.generation_usage(config, {'generation_cost':{**expected,'input_tokens':100}}, 1)
        assert len(auditor.failures) == 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--expected-questions', type=int, default=None, help='Required for the final audit; actual frozen usable count, possibly zero.')
    args = p.parse_args()
    self_test()
    if args.self_test:
        print('READY: independent parser/scorer self-tests passed; no evaluation data read.')
        return
    max_questions = read(args.root / 'CONFIG.json')['max_questions']
    if args.expected_questions is None or not 0 <= args.expected_questions <= max_questions:
        raise SystemExit('--expected-questions must name the actual frozen usable count (0 through configured max_questions).')
    for required in ['RESULT_SUMMARY.json', 'RESULT_BINDINGS.json', 'EVALUATION_PREPARATION.COMPLETE.json', 'TRAINED_EVALUATION.COMPLETE.json', 'BASE_EVALUATION.COMPLETE.json']:
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
    lines = ['# v68 独立结果复算', '', '状态：' + report['status'], '',
        '仅使用 Python 标准库从冻结输入和原始回答复算；未调用 evaluate.verify、原评分器或筛选器，未修改冻结文件或运行模型。',
        '', f"完成 {a.checks} 项检查；发现 {len(a.failures)} 个差异。", '',
        '检查覆盖：回答格式与合法性、显示字母到内容映射、exact/partial、6×3 调用完整性和逐题筛选；Base 配对提示词/置换/seed/解码参数；各层汇总、精确去重合并、分布、盲选全部六项、历史和当前绑定文件 hash；原始/接受/可用反述与实际 null 理由字段统计，以及 v67 描述性比较和 prior 变化。',
        '', '复算计数：' + json.dumps(counts, ensure_ascii=False)]
    if a.failures: lines += ['', '差异详情见 JSON 中的 failures；本审计未修补任何评分或报告。']
    with (output_dir / 'INDEPENDENT_RESULT_AUDIT.md').open('x') as f: f.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'checks': a.checks, 'failures': len(a.failures), 'path': str(target)}))
    if a.failures: raise SystemExit(1)

if __name__ == '__main__':
    main()
