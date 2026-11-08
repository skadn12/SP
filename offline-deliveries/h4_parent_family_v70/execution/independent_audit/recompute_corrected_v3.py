"""Append-only corrected independent stdlib audit: frozen v70 raw scoring, parent families and merge.

No production imports or network. --self-test reads synthetic data only. Real
execution requires completed finalization and an explicit usable question count;
only new audit artifacts are written, never frozen results or content reviews.
"""
from __future__ import annotations
import argparse
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
        Current v70 JSON may declare files; external JSON stays a hash-bound endpoint.
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


    def families(self,cfg,pool,members):
        receipts=read(self.root/'ASSEMBLY_RECEIPTS.json')['selections'];http_count=0;observed=[]
        for slot in cfg['slots']:
            fid=slot.get('family_id',slot['slot_id']);run=self.root/'assets'/fid
            parent=read(run/'PARENT.json');source=read(run/'SOURCE.json');manifest=read(run/'MANIFEST.json');complete=read(run/'GENERATION.COMPLETE.json')
            original=read(slot['parent_pool_path']);original=original if isinstance(original,list) else original['questions']
            self.eq([q for q in original if q['candidate_id']==slot['parent_candidate_id']],[parent],'parent_exact_pool:'+fid)
            self.eq(source['turns'],read(slot['history_path'])['turns'],'source_complete_turns:'+fid)
            self.eq((run/'SOURCE.json').read_bytes(),Path(parent['source_path']).read_bytes(),'source_exact_bytes:'+fid)
            self.eq(parent['source_sha256'],sha(run/'SOURCE.json'),'source_parent_binding:'+fid)
            self.eq(manifest['parent_sha256'],sha(run/'PARENT.json'),'parent_manifest_binding:'+fid)
            self.eq(manifest['source_mode'],'parent_exact_full_source','source_mode:'+fid)
            self.eq(manifest['budget'],{'http_attempts_max':1},'generation_budget:'+fid)
            for obj in (manifest,complete):self.eq(obj['training_allowed'],False,'generation_training:'+fid)
            for key in ('semantic_retries','mechanical_repairs','transport_retries'):self.eq(complete[key],0,'generation_'+key+':'+fid)
            requests=sorted((run/'requests').glob('*/REQUEST.json'));self.eq(len(requests),1,'generation_single_request:'+fid)
            http=list((run/'requests').glob('*/HTTP_ATTEMPT.json'));http_count+=len(http)
            self.eq(complete['http_attempts'],len(http),'generation_http:'+fid)
            self.yes(len(http)<=1,'generation_no_retry:'+fid)
            family=[q for q in pool if members[q['candidate_id']]['family_id']==fid]
            self.eq([q for q in family if members[q['candidate_id']]['role']=='control6'],[parent],'control_immutable:'+fid)
            for q in family:
                m=members[q['candidate_id']]
                self.eq(m['fixed_fact_ids'],FACTS,'fixed_family_fact_ids:'+q['candidate_id'])
                self.eq(m['new_candidate'],m['role']=='child','role_new_candidate:'+q['candidate_id'])
            for path in requests:
                req=read(path);folder=path.parent
                self.eq(req,read(run/'INITIAL_REQUEST.json'),'exact_single_request:'+fid)
                self.eq(req['max_tokens'],6144,'output_limit:'+fid)
                ctx=json.JSONDecoder().raw_decode(req['messages'][0]['content'].split('\nINPUT_JSON\n',1)[1])[0]
                self.eq(ctx['source_turns'],source['turns'],'complete_writer_source:'+fid)
                self.eq(ctx['source_metadata'],{k:v for k,v in source.items() if k!='turns'},'writer_metadata:'+fid)
                self.yes(not any(k in ctx for k in ('gold','target_count','target_counts','mechanical_repair')),'no_writer_gold_retry:'+fid)
                self.eq(read(folder/'PROOF.json')['attempt'],1,'one_attempt:'+fid)
                if (folder/'HTTP_ATTEMPT.json').exists():
                    self.eq(read(folder/'HTTP_ATTEMPT.json')['transport_retries'],0,'no_transport_retry:'+fid)
                    self.eq(read(folder/'THROTTLE.json')['minimum_gap_after_previous_response_seconds'],35.0,'completion_gap:'+fid)
            if complete['status']=='NO_VALID_ASSETS':
                self.yes(not any((run/n).exists() for n in ('ASSETS.json','PUBLIC_ASSETS.json','VERSION_MAP.json')),'failed_family_no_exports:'+fid)
                self.eq(complete['fact_count'],0,'failed_family_fact_count:'+fid)
                self.eq(complete['counterversion_count'],0,'failed_family_counterversion_count:'+fid)
                response_files=sorted((run/'requests').glob('*/RESPONSE.json'))
                self.eq(len(response_files),1,'failed_family_raw_response_saved:'+fid)
                if response_files:
                    raw_obj=read(response_files[0]);self.yes(isinstance(raw_obj.get('text'),str),'failed_family_raw_text_saved:'+fid)
                    raw=strict_json(raw_obj['text'])
                    refs={t['turn_ref'] for t in source['turns']}
                    invalid_refs=[r['anchor']['turn_ref'] for r in raw['replacements'] if isinstance(r.get('anchor'),dict) and r['anchor']['turn_ref'] not in refs]
                    self.yes(bool(invalid_refs),'failed_family_independent_exact_ref_rejection:'+fid)
                    self.eq(read(response_files[0].parent/'VALIDATION.json')['repair_allowed'],False,'failed_family_no_repair:'+fid)
                self.yes(all(members[q['candidate_id']]['role']=='control6' for q in family),'failed_family_no_children:'+fid)
                observed.append({'family_id':fid,'control_id':parent['candidate_id'],'fixed_fact_count':0,'counterversions':0,'status':'NO_VALID_ASSETS'})
                continue
            assets=read(run/'ASSETS.json');public=read(run/'PUBLIC_ASSETS.json');mapping=read(run/'VERSION_MAP.json')
            self.eq(assets['public_stem'],parent['question'],'immutable_stem:'+fid)
            self.eq([f['fact_id'] for f in assets['facts']],FACTS,'fixed_parent_facts:'+fid)
            self.eq(assets['parent_sha256'],sha(run/'PARENT.json'),'asset_parent_binding:'+fid)
            self.eq(public['assets_sha256'],sha(run/'ASSETS.json'),'public_assets_binding:'+fid)
            self.eq(mapping['public_assets_sha256'],sha(run/'PUBLIC_ASSETS.json'),'map_public_binding:'+fid)
            expected_public=[];expected_map=[]
            for label,fact in zip(LABELS,assets['facts']):
                ev=parent['option_evidence'][label]
                self.eq(fact['parent_label'],label,'parent_label:'+fid+label)
                self.eq(fact['supported'],{'proposition':parent['options'][label],'relation':'supports','rationale':ev.get('rationale',ev.get('reason')),
                    'anchor':{k:ev[k] for k in ('turn_ref','speaker','quote')}},'immutable_positive:'+fid+label)
                self.citations(fact['supported']['anchor'],source,'positive_citation:'+fid+label)
                c=fact['contradicted']
                if c is None:
                    self.yes(isinstance(fact.get('contradiction_unavailable_reason'),str) and fact['contradiction_unavailable_reason'].strip(),'null_reason:'+fid+label)
                    self.eq(fact['anchor'],None,'null_anchor:'+fid+label);continue
                self.eq(c['relation'],'contradicts','counter_relation:'+fid+label)
                self.eq(fact['contradiction_unavailable_reason'],None,'nonnull_reason:'+fid+label)
                self.yes(not FORBIDDEN.search(c['proposition']),'new_public_pronoun_gate:'+fid+label)
                self.eq(c['anchor'],fact['anchor'],'counter_anchor_projection:'+fid+label);self.citations(c['anchor'],source,'counter_citation:'+fid+label)
                internal=fact['fact_id']+':c';vid='v_'+hmac.new(bytes.fromhex(manifest['version_id_salt']),internal.encode(),hashlib.sha256).hexdigest()[:24]
                expected_public.append({'version_id':vid,'fact_id':fact['fact_id'],'proposition':c['proposition']})
                expected_map.append({'version_id':vid,'fact_id':fact['fact_id'],'internal_version_id':internal,'variant':'contradicted','relation':'contradicts'})
            self.eq(public['parent_facts'],[{'fact_id':f,'parent_label':l,'proposition':parent['options'][l]} for f,l in zip(FACTS,LABELS)],'public_parent_projection:'+fid)
            self.eq(public['versions'],sorted(expected_public,key=lambda v:v['version_id']),'public_version_projection:'+fid)
            self.eq(mapping['versions'],sorted(expected_map,key=lambda v:v['version_id']),'private_version_projection:'+fid)
            for path in requests:
                request=read(path);folder=path.parent
                self.eq(request,read(run/'INITIAL_REQUEST.json'),'generation_frozen_payload:'+fid)
                self.eq(request['max_tokens'],6144,'generation_output_cap:'+fid)
                proof=read(folder/'PROOF.json');self.eq(proof['attempt'],1,'generation_attempt:'+fid)
                text=request['messages'][0]['content'];context=json.JSONDecoder().raw_decode(text.split('\nINPUT_JSON\n',1)[1])[0]
                self.eq(context['source_turns'],source['turns'],'writer_complete_source:'+fid)
                self.eq(context['source_metadata'],{k:v for k,v in source.items() if k!='turns'},'writer_source_metadata:'+fid)
                self.eq(context['parent_facts'],public['parent_facts'],'writer_fixed_positive_text:'+fid)
                self.yes(not any(k in context for k in ('gold','target_count','target_counts','parent_candidate_id','mechanical_repair')),'writer_no_target_or_repair:'+fid)
                self.eq(context['required_public_stem'],parent['question'],'writer_stem:'+fid)
                if (folder/'HTTP_ATTEMPT.json').exists():
                    attempt=read(folder/'HTTP_ATTEMPT.json');self.eq(attempt['transport_retries'],0,'transport_no_retry:'+fid)
                    self.eq(read(folder/'THROTTLE.json')['minimum_gap_after_previous_response_seconds'],35.0,'global_completion_gap:'+fid)
                raw=read(folder/'RESPONSE.json');self.eq(raw['metadata']['finish_reason'],'stop','writer_complete_response:'+fid)
                data=strict_json(raw['text']);self.eq(set(data),{'root_id','family_id','replacements'},'writer_schema:'+fid)
                self.eq([r['fact_id'] for r in data['replacements']],FACTS,'writer_fixed_order:'+fid)
                for f,row in zip(assets['facts'],data['replacements']):
                    self.eq(set(row),{'fact_id','anchor','contradicted','contradiction_unavailable_reason'},'writer_row_schema:'+fid+f['fact_id'])
                    self.eq(row['anchor'],f['anchor'],'raw_counter_anchor:'+fid+f['fact_id'])
                    self.eq(row['contradicted'],{k:v for k,v in f['contradicted'].items() if k!='anchor'} if f['contradicted'] else None,'raw_counter_text:'+fid+f['fact_id'])
                    self.eq(row['contradiction_unavailable_reason'],f['contradiction_unavailable_reason'],'raw_reason:'+fid+f['fact_id'])
            judgment=read(run/'SOURCE_JUDGMENTS.json');review=read(run/'ASSET_REVIEW.json')
            self.eq(judgment.get('new_variant_private_artifacts_read_before_save',judgment.get('gold_read_before_save')),False,'source_first_blind_new_variants:'+fid)
            self.eq(judgment['solver_scores_used'],False,'source_review_no_solver:'+fid)
            self.eq(review['source_first'],True,'asset_review_source_first:'+fid)
            self.eq(review['source_judgments_sha256'],sha(run/'SOURCE_JUDGMENTS.json'),'review_judgment_binding:'+fid)
            for key in ('parents','versions'):
                idkey='fact_id' if key=='parents' else 'version_id';js={r[idkey]:r for r in judgment[key]}
                for row in review[key]:
                    j=js[row[idkey]];self.eq(row['relation'],j['relation'],'review_relation:'+row[idkey])
                    expected_relation='supports' if key=='parents' else 'contradicts'
                    usable=j['relation']==expected_relation and j['scope_valid'] and j['fact_distinct'] and judgment['stem_valid'] and review['stem_valid']
                    self.eq(row['usable'],usable,'review_usable:'+row[idkey])
                    for citation in j.get('citations',[]):self.citations(citation,source,'review_citation:'+row[idkey])
            family=[q for q in pool if members[q['candidate_id']]['family_id']==fid]
            controls=[q for q in family if members[q['candidate_id']]['role']=='control6']
            self.eq(controls,[parent],'control_immutable:'+fid)
            factmap={f['fact_id']:f for f in assets['facts']};mapped={v['version_id']:v for v in mapping['versions']}
            for q in family:
                member=members[q['candidate_id']]
                self.eq(member['fixed_fact_ids'],FACTS,'fixed_family_ids:'+q['candidate_id'])
                self.eq(member['split_group'],q['root_id'],'same_root_split:'+q['candidate_id'])
                self.eq(member['new_candidate'],member['role']=='child','role_new_candidate:'+q['candidate_id'])
                self.eq(q['question'],parent['question'],'child_stem_immutable:'+q['candidate_id'])
                if member['role']=='control6':continue
                receipt=next(r for r in receipts if r['candidate_id']==q['candidate_id'])
                selected=receipt['selected_versions'];self.eq(sorted(r['fact_id'] for r in selected),FACTS,'child_each_fact_once:'+q['candidate_id'])
                truth=[]
                for r in selected:
                    f=factmap[r['fact_id']];v=f[r['variant']];label=r['label']
                    self.eq(q['options'][label],v['proposition'],'child_version_text:'+q['candidate_id']+label)
                    self.eq(r['verified_relation'],v['relation'],'child_reviewed_relation:'+q['candidate_id']+label)
                    self.eq(q['option_evidence'][label],{**v['anchor'],'relation':v['relation'],'rationale':v['rationale'],'reason':v['rationale']},'child_exact_citation:'+q['candidate_id']+label)
                    if r['variant']=='supported':
                        truth.append(label);self.eq(v['proposition'],parent['options'][f['parent_label']],'retained_positive_unchanged:'+q['candidate_id']+label)
                    else:self.eq(mapped[r['version_id']]['fact_id'],r['fact_id'],'child_version_map:'+q['candidate_id']+label)
                self.eq(q['gold'],sorted(truth),'child_gold_from_reviewed_versions:'+q['candidate_id'])
            observed.append({'family_id':fid,'control_id':parent['candidate_id'],'fixed_fact_count':6,'counterversions':len(expected_public)})
        self.eq(http_count,2,'exact_two_generation_http');self.eq(read(self.root/'GENERATION.COMPLETE.json')['http_attempts'],http_count,'aggregate_generation_http')
        return observed

    def run(self):
        for name in ('RESULT_BINDINGS.json','EXECUTION_PLAN.json','EVALUATION_PREPARATION.COMPLETE.json'):
            self.recursive_bindings(self.root/name)
        cfg=read(self.root/'CONFIG.json');summary=read(self.root/'RESULT_SUMMARY.json')
        self.recursive_bindings(cfg['baseline_result_bindings'])
        self.eq(cfg['training_allowed'],False,'config_training');self.eq(summary['training_started'],False,'result_no_training')
        pool=read(self.root/'POOL.json')['questions'];members={m['candidate_id']:m for m in read(self.root/'FAMILY_SIDECAR.json')['members']}
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
            if k in result:raise ValueError('duplicate_key')
            result[k]=v
        return result
    def constant(x):raise ValueError('nonfinite_number')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def self_test():
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
    report.update(binding_scope='recursive_RESULT_BINDINGS_ancestry_current_v70_files_and_current_checkpoint_files',historical_environment_reproducibility_claimed=False,initial_audit_status='FAIL',initial_audit_findings_retained=7,explicit_binding_rows=audit.ledger_rows,cached_path_expected_hash_pairs=len(audit.hash_cache),status='PASS' if not audit.failures else 'FAIL',checks_completed=audit.checks,failures=audit.failures,completed_at_utc=datetime.now(timezone.utc).isoformat(),result_summary_sha256=sha(args.root/'RESULT_SUMMARY.json'))
    output=args.output_dir or args.root/'execution/independent_audit/corrected/final_v3';output.mkdir(parents=True,exist_ok=True)
    with (output/'INDEPENDENT_RESULT_AUDIT.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=str);f.write('\n')
    with (output/'INDEPENDENT_RESULT_AUDIT.md').open('x') as f:f.write('# v70 independent audit\n\nStatus: '+report['status']+'\n\n'+str(audit.checks)+' checks; '+str(len(audit.failures))+' differences.\n\nStdlib raw scoring, 6×3 call sets, original selection policy, paired inputs, immutable parent families, source citations, retained positive text, control exclusion, stage counts, baseline preservation, dedup merge, and explicit frozen file-ledger hashes. Historical environment reproducibility is not claimed; original FAIL and seven findings remain retained. No model calls or frozen-file edits.\n')
    print(json.dumps({'status':report['status'],'checks':audit.checks,'failures':len(audit.failures),'output':str(output)}))
    if audit.failures:raise SystemExit(1)

if __name__=='__main__':main()
