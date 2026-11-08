"""Read frozen score shards; retain the prespecified paired root denominator."""
import collections
import csv
import json
import random
import re
import statistics
from pathlib import Path
from workflow_io import BASE, SLOTS, bind, load, now, save

PRIMARY = [s for s in SLOTS if s != 'MC6']
WEEK = BASE.parent
OLD = WEEK / 'feedback_closed_loop_20261024_v1'
PRIOR = WEEK / 'feedback_replication_20root_resume_20261025_v1'
INITIAL = WEEK / 'feedback_replication_20root_20261025_v1'


def checked(binding):
    assert bind(binding['path']) == binding, ('Binding drift', binding['path'])


def answer_only(raw, presentation, kind):
    """Independent diagnostic: final box only, no evidence-contract penalty."""
    text = raw.get('raw_output', raw.get('text', ''))
    text = text.replace('<|im_end|>', '').strip()
    text = re.sub(r'^<think>.*?</think>\s*', '', text, count=1, flags=re.S)
    try:
        obj = json.loads(text)
        match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>', obj['final_answer'])
        assert match
        labels = match[1].split()
        assert len(set(labels)) == len(labels)
        if kind == 'ordering':
            assert set(labels) == set('ABCDEF')
        pred = [presentation['display_to_canonical'][s] for s in labels]
        gold = presentation['canonical_gold']
        exact = pred == gold if kind == 'ordering' else set(pred) == set(gold)
        return {'valid': True, 'exact': exact, 'prediction': pred}
    except (ValueError, KeyError, AssertionError, TypeError):
        return {'valid': False, 'exact': False, 'prediction': []}


def shard(folder, cohort):
    scores_path = folder / 'execution/SCORES.json'
    scores = load(scores_path)
    mapping = {r['request_id']: r['presentation'] for r in load(folder / 'execution/PRIVATE_SCORING.json')['requests']}
    normalized = []
    for row in scores['records']:
        strict = row['strict']
        raw_binding = row.get('raw_binding', row.get('raw'))
        checked(raw_binding)
        raw = load(raw_binding['path'])
        if 'status' in raw:
            returned = raw['status'] == 'RETURNED'
        else:
            # The prior recovery captured direct vLLM responses as `text`.
            # Its score row binds that actual response and records returned=True.
            returned = row.get('returned') is True and isinstance(raw.get('text'), str)
        diagnostic = answer_only(raw, mapping[row['request_id']], strict['question_type'])
        if 'answer_only' in row:
            assert bool(row['answer_only']['exact']) == diagnostic['exact'], ('Answer diagnostic drift', row['request_id'])
        normalized.append({'cohort': cohort, 'request_id': row['request_id'],
            'root_id': strict['root_id'], 'candidate_id': strict['candidate_id'],
            'arm': row['arm'], 'slot_id': row['slot_id'],
            'variant_id': row.get('variant_id', strict.get('variant_id', re.search(r'__perm_\d\d', row['request_id'])[0][2:])),
            'returned': returned, 'valid': bool(strict['valid']), 'exact': bool(strict['exact']),
            'answer_only': diagnostic, 'prediction': strict['prediction'], 'gold': strict['gold'],
            'gold_count': strict['gold_count'],
            'question_type': strict['question_type'], 'partial': row.get('partial', {}),
            'false_positive': strict.get('false_positive'), 'false_negative': strict.get('false_negative'),
            'pairwise_correct': strict.get('pairwise_correct'), 'pairwise_total': strict.get('pairwise_total'),
            'raw_binding': raw_binding, 'errors': row.get('errors', []),
            'finish_reason': raw.get('finish_reason')})
    return normalized


def percentile(xs, p):
    xs = sorted(xs)
    index = (len(xs) - 1) * p
    low = int(index)
    return xs[low] + (xs[min(low + 1, len(xs) - 1)] - xs[low]) * (index - low)


def bootstrap(differences):
    if not differences:
        return None
    rng = random.Random(20261025)
    draws = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000)]
    return {'resamples': 10000, 'seed': 20261025, 'unit': 'root',
            'percentile95': [percentile(draws, .025), percentile(draws, .975)]}


def paired(records, planned_roots, slots, with_bootstrap=True):
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[(r['root_id'], r['slot_id'], r['arm'])].append(r)
    results, pooled = [], []
    for root in planned_roots:
        common = []
        for slot in slots:
            arms = {a: grouped[(root, slot, a)] for a in ('F','G')}
            if all(len(rs) == 2 and {r['variant_id'] for r in rs} == {'perm_00','perm_01'}
                   and all(r['returned'] for r in rs) for rs in arms.values()):
                assert all(len({r['candidate_id'] for r in rs}) == 1 for rs in arms.values())
                means = {a: statistics.mean(int(r['exact']) for r in rs) for a, rs in arms.items()}
                answers = {a: statistics.mean(int(r['answer_only']['exact']) for r in rs) for a, rs in arms.items()}
                common.append({'slot_id': slot, 'F': means['F'], 'G': means['G'], 'difference': means['F']-means['G'],
                               'answer_F': answers['F'], 'answer_G': answers['G'],
                               'answer_difference': answers['F']-answers['G']})
        pooled.extend(common)
        results.append({'root_id': root, 'common_slot_count': len(common), 'common_slots': common,
            'F': statistics.mean(s['F'] for s in common) if common else None,
            'G': statistics.mean(s['G'] for s in common) if common else None,
            'difference': statistics.mean(s['difference'] for s in common) if common else None,
            'answer_difference': statistics.mean(s['answer_difference'] for s in common) if common else None})
    eligible = [r for r in results if r['common_slot_count']]
    differences = [r['difference'] for r in eligible]
    complete = [r for r in eligible if r['common_slot_count'] == len(slots)]
    return {'slots': slots, 'planned_roots': len(planned_roots), 'eligible_roots': len(eligible),
        'common_root_slots': len(pooled), 'paired_returned_responses': len(pooled) * 4,
        'F': statistics.mean(r['F'] for r in eligible) if eligible else None,
        'G': statistics.mean(r['G'] for r in eligible) if eligible else None,
        'difference': statistics.mean(differences) if differences else None,
        'bootstrap': bootstrap(differences) if with_bootstrap else None,
        'answer_only_difference': statistics.mean(r['answer_difference'] for r in eligible) if eligible else None,
        'complete_slot_roots': len(complete),
        'complete_slot_root_difference': statistics.mean(r['difference'] for r in complete) if complete else None,
        'pooled_common_slot_difference_descriptive': statistics.mean(s['difference'] for s in pooled) if pooled else None,
        'roots': results}


def descriptive(records):
    rows = []
    groups = {'ALL': list(SLOTS), 'NON_MC6': PRIMARY, 'MC2_5': ['MC2','MC3','MC4','MC5'],
              'ORDERING': ['ORDER1','ORDER2','ORDER3'], 'MC6_CONTROL': ['MC6'], **{s:[s] for s in SLOTS},
              'PRESENTATION_perm_00':list(SLOTS), 'PRESENTATION_perm_01':list(SLOTS)}
    for arm in ('F','G'):
        for group, slots in groups.items():
            all_rows = [r for r in records if r['arm'] == arm and r['slot_id'] in slots]
            if group.startswith('PRESENTATION_'):
                all_rows = [r for r in all_rows if r['variant_id'] == group.removeprefix('PRESENTATION_')]
            rs = [r for r in all_rows if r['returned']]
            count = len(rs)
            ordering = [r for r in rs if r['question_type'] == 'ordering']
            valid_order = [r for r in ordering if r['valid']]
            mc = [r for r in rs if r['question_type'] == 'multi_select']
            semantic = collections.defaultdict(list)
            for r in rs:
                semantic[r['candidate_id']].append(r)
            pairs = [v for v in semantic.values() if len(v) == 2]
            mc_partials = []
            for r in mc:
                tp = len(set(r['prediction']) & set(r['gold']))
                fp, fn = r['false_positive'], r['false_negative']
                mc_partials.append({'precision':tp/(tp+fp) if tp+fp else 0,
                    'recall':tp/(tp+fn), 'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
            rows.append({'arm': arm, 'group': group, 'prepared':len(all_rows), 'returned': count,
                'contract_valid': sum(r['valid'] for r in rs), 'strict_exact_count':sum(r['exact'] for r in rs),
                'strict_exact': statistics.mean(r['exact'] for r in rs) if count else None,
                'answer_only_exact':statistics.mean(r['answer_only']['exact'] for r in rs) if count else None,
                'mc_false_positive_labels': sum(r['false_positive'] for r in mc),
                'mc_false_negative_labels': sum(r['false_negative'] for r in mc),
                'mc_overselection_response_rate':statistics.mean(len(r['prediction']) > len(r['gold']) for r in mc) if mc else None,
                'mc_underselection_response_rate':statistics.mean(len(r['prediction']) < len(r['gold']) for r in mc) if mc else None,
                'mc_contains_false_positive_rate':statistics.mean(r['false_positive'] > 0 for r in mc) if mc else None,
                'mc_contains_false_negative_rate':statistics.mean(r['false_negative'] > 0 for r in mc) if mc else None,
                'mc_label_count_greater_than_gold_rate':statistics.mean(len(r['prediction']) > len(r['gold']) for r in mc) if mc else None,
                'mc_label_count_less_than_gold_rate':statistics.mean(len(r['prediction']) < len(r['gold']) for r in mc) if mc else None,
                'mc_precision':statistics.mean(r['precision'] for r in mc_partials) if mc else None,
                'mc_recall':statistics.mean(r['recall'] for r in mc_partials) if mc else None,
                'mc_f1':statistics.mean(r['f1'] for r in mc_partials) if mc else None,
                'order_pairwise_strict': sum(r['pairwise_correct'] for r in ordering)/(15*len(ordering)) if ordering else None,
                'order_pairwise_contract_valid_only':sum(r['pairwise_correct'] for r in valid_order)/(15*len(valid_order)) if valid_order else None,
                'semantic_pairs':len(pairs), 'both_exact':sum(all(r['exact'] for r in v) for v in pairs),
                'at_least_one_exact':sum(any(r['exact'] for r in v) for v in pairs),
                'both_valid_canonical_predictions_identical':sum(all(r['valid'] for r in v) and
                    (set(v[0]['prediction']) == set(v[1]['prediction']) if v[0]['question_type']=='multi_select'
                     else v[0]['prediction'] == v[1]['prediction']) for v in pairs),
                'canonical_predictions_identical':sum(all(r['valid'] for r in v) and
                    (set(v[0]['prediction']) == set(v[1]['prediction']) if v[0]['question_type']=='multi_select'
                     else v[0]['prediction'] == v[1]['prediction']) for v in pairs)})
    return rows


def main():
    manifest = load(BASE/'CONTINUATION_MANIFEST.json')
    roots16 = manifest['roots']
    old = shard(OLD, 'original4')
    reused = shard(PRIOR, 'prior_recovery')
    new = shard(BASE, 'current_continuation')
    admitted = {r['candidate_id'] for r in load(BASE/'QUESTION_FREEZE.json')['slots'] if r['status']=='PASS'}
    merged = reused + new
    assert len({r['request_id'] for r in merged}) == len(merged)
    assert {r['candidate_id'] for r in merged} == admitted
    assert len({(r['root_id'],r['arm'],r['slot_id'],r['variant_id']) for r in merged}) == len(merged)
    old_roots = sorted({r['root_id'] for r in old})
    analysis = {'at':now(), 'adaptive_recovery':True, 'positive_difference_means':'F easier; negative means F harder',
        'two_displays_are_dependent':True,
        'new16_primary':paired(merged, roots16, PRIMARY),
        'new16_MC2_5_secondary':paired(merged, roots16, ['MC2','MC3','MC4','MC5']),
        'new16_ordering_secondary':paired(merged, roots16, ['ORDER1','ORDER2','ORDER3']),
        'new16_MC6_control':paired(merged, roots16, ['MC6'], False),
        'original4_frozen':paired(old, old_roots, PRIMARY, False),
        'combined20_secondary':paired(old+merged, old_roots+roots16, PRIMARY),
        'new16_per_operation':{s:paired(merged,roots16,[s],False) for s in SLOTS},
        'descriptive_new16':descriptive(merged), 'descriptive_original4':descriptive(old),
        'initial16_reported_run':{'status':'PARTIAL', 'source_pass':5, 'content_candidates':63,
             'content_admitted':49, 'prepared_requests':98, 'returned':0,
             'basis':[bind(INITIAL/p) for p in ('FINAL_REPORT.json','SOURCE_FREEZE.json','CONTENT_GATE_FINAL.json','execution/RAW_RESPONSE_FREEZE.json')],
             'limitation':'Initial run itself included source recovery; this is its recorded closeout, not a claim that all work stayed inside an unchanged initial budget.'},
        'shard_counts':{'original4':len(old),'prior_recovery':len(reused),'current_continuation':len(new)},
        'score_bindings':[bind(p/'execution/SCORES.json') for p in (OLD,PRIOR,BASE)]}
    save(BASE/'COMBINED_ANALYSIS.json',analysis)
    save(BASE/'NORMALIZED_SCORES.json',{'records':old+merged})
    with (BASE/'ROOT_DIFFERENCES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['root_id','common_slot_count','F','G','difference','answer_difference'])
        writer.writeheader()
        for row in analysis['combined20_secondary']['roots']:
            writer.writerow({k:row[k] for k in writer.fieldnames})
    print(json.dumps({k:{n:v for n,v in analysis[k].items() if n in ('eligible_roots','common_root_slots','F','G','difference','bootstrap')}
                      for k in ('new16_primary','original4_frozen','combined20_secondary')}))


if __name__ == '__main__':
    main()
