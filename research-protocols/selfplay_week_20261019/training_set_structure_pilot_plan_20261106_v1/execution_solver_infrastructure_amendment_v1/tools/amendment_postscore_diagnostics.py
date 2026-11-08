#!/usr/bin/env python3
"""Read only independently frozen amended scores; exclusively create diagnostics.

No adapter, model, tokenizer or API imports. Never opens RAW_ANSWERS or raw files.
Run only after controller announces completion and freezes EXECUTION_RESULT_FREEZE.
The original execution directory is read-only. No generation, repair or selection.
No-context triage (declared before results): all18 returned, exact accuracy >=0.8.
This is a bounded wording-audit trigger, not a primary endpoint or proof of shortcut.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

A = Path(__file__).resolve().parents[1]
E = A.parent / 'execution_20261106_v1'
GROUPS = ('OLD_F', 'OLD_G', 'NEW_F', 'NEW_G')
MODES = ('full_context', 'no_context')
NAMESPACE = 'structure_pilot_20261106_solver_infrastructure_amendment_v1'
ORIGINAL_REQUESTS_SHA256 = 'a229fa52121d8254c9eda42b6b66fd658e91a60ed774aecb18be4e635794c026'
ORIGINAL_FINAL_SHA256 = '364bfd6e3d8a8ab3a47340ff1e316ce8814043449aa629ea84a986140ecb6936'
ORIGINAL_CONSTRUCTION_SHA256 = '16a70c3bf6acdcf8c21e863879fc5fdccd41bd968b2d5d4996bb4ff69b2b76a1'
ORIGINAL_SEAL_SHA256 = '6b4e34d39e8e1ff0b67f3d5ea5bc52187036c90e611017e8d999829ec41047bb'
NO_CONTEXT_HIGH_EXACT_THRESHOLD = 0.8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def alternate_path(edges, a, b):
    graph = defaultdict(set)
    for x, y in edges:
        if (x, y) != (a, b):
            graph[x].add(y)
    seen, pending = set(), list(graph[a])
    while pending:
        x = pending.pop()
        if x == b:
            return True
        if x not in seen:
            seen.add(x)
            pending.extend(graph[x] - seen)
    return False


def error_patterns(r, c):
    """Same semantics as original postsolver_diagnostics; missing != invalid."""
    result = {'patterns': []}
    if r['execution_status'] != 'RETURNED':
        result['patterns'] = ['INFRASTRUCTURE_MISSING_NOT_MODEL_ERROR']
    elif not r['valid']:
        result['patterns'] = ['INVALID_FINAL_ANSWER_SEPARATE_FROM_SEMANTIC_PATTERNS']
    elif c['question_type'] == 'multi_select':
        gold, pred = set(c.get('gold_option_ids', c.get('gold'))), set(r['predicted_option_ids'])
        result.update(missing_ids=sorted(gold-pred), extra_ids=sorted(pred-gold))
        if gold-pred:
            result['patterns'].append('SUPPORTED_SET_OMISSION')
        if pred-gold:
            result['patterns'].append('UNSUPPORTED_EXTRA_SELECTION')
    elif c['question_type'] == 'ordering':
        gold = c.get('gold_option_ids', c.get('gold'))
        pos = {x: i for i, x in enumerate(r['predicted_option_ids'])}
        assert len(pos) == len(gold) and set(pos) == set(gold)
        pairs = [[a, b] for i, a in enumerate(gold) for b in gold[i+1:] if pos[a] > pos[b]]
        result['reversed_gold_pairs'] = pairs
        if pairs:
            result['patterns'].append('FULL_ORDER_RELATION_REVERSAL')
    else:
        gold, pred = set(map(tuple, c['gold_pairs'])), set(map(tuple, r['predicted_option_ids']))
        missing, extra = gold-pred, pred-gold
        reversed_pairs = {x for x in extra if tuple(reversed(x)) in gold}
        indirect_missing = {x for x in missing if alternate_path(gold, *x)}
        result.update(missing_pairs=sorted(missing), extra_pairs=sorted(extra),
                      reversed_pairs=sorted(reversed_pairs), graph_transitive_missing_pairs=sorted(indirect_missing))
        for condition, name in ((missing, 'SUPPORTED_RELATION_OMISSION'),
                                (reversed_pairs, 'PARTIAL_RELATION_REVERSAL'),
                                (extra-reversed_pairs, 'UNSUPPORTED_EXTRA_RELATION'),
                                (indirect_missing, 'GRAPH_TRANSITIVE_RELATION_OMISSION')):
            if condition:
                result['patterns'].append(name)
    return result


def summarize(rows):
    """Exact denominators match frozen adapter; metrics always split by type."""
    returned = [r for r in rows if r['execution_status'] == 'RETURNED']
    valid = [r for r in returned if r['valid']]
    exact = sum(bool(r['exact_correct']) for r in returned)
    result = {'planned': len(rows), 'returned': len(returned),
              'infrastructure_missing': len(rows)-len(returned),
              'execution_status_counts': dict(Counter(r['execution_status'] for r in rows)),
              'valid_returned': len(valid), 'invalid_returned': len(returned)-len(valid),
              'exact_correct_returned': exact,
              'exact_accuracy_returned': exact/len(returned) if returned else None,
              'exact_accuracy_planned': exact/len(rows) if rows and len(returned) == len(rows) else None,
              'semantic_pattern_denominator_valid_returned': len(valid),
              'pattern_counts_nonexclusive': dict(Counter(p for r in rows for p in r['patterns'])),
              'metrics_by_type': {}}
    for kind in sorted({r['question_type'] for r in rows}):
        sub = [r for r in rows if r['question_type'] == kind]
        ret = [r for r in sub if r['execution_status'] == 'RETURNED']
        fields = ('precision', 'recall', 'partial_score') if kind == 'partial_order' else ('J',)
        metrics = {'metric': 'directed_pair_precision_recall_F1' if kind == 'partial_order' else 'J',
                   'planned': len(sub), 'returned': len(ret),
                   'valid_returned': sum(r['valid'] for r in ret),
                   'invalid_returned': sum(not r['valid'] for r in ret),
                   'exact_correct_returned': sum(bool(r['exact_correct']) for r in ret),
                   'exact_accuracy_returned': sum(bool(r['exact_correct']) for r in ret)/len(ret) if ret else None,
                   'exact_accuracy_planned': sum(bool(r['exact_correct']) for r in ret)/len(sub) if sub and len(ret) == len(sub) else None}
        for field in fields:
            # Preserve scorer values, including inherited negative J and invalid zero.
            name = 'directed_pair_F1' if field == 'partial_score' else field
            metrics['mean_'+name+'_returned'] = sum(r.get(field, 0.0) for r in ret)/len(ret) if ret else None
        result['metrics_by_type'][kind] = metrics
    return result


def grouped(rows, fields):
    keys = sorted({tuple(r[k] for k in fields) for r in rows})
    return [{**dict(zip(fields, key)), **summarize([r for r in rows if tuple(r[k] for k in fields) == key])} for key in keys]


def candidate(document, qid):
    if isinstance(document, list):
        rows = document
    elif 'candidates' in document:
        rows = document['candidates']
    else:
        rows = [document]
    found = [r for r in rows if r.get('candidate_id') == qid]
    assert len(found) == 1, ('EXACTLY_ONE_FROZEN_CANDIDATE', qid)
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    score_path = A/'solver/evaluation/SCORES.json'
    request_path = A/'solver/evaluation/REQUESTS.json'
    freeze_path = A/'solver/evaluation/EXECUTION_RESULT_FREEZE.json'
    # This gate precedes reading any actual answer or score content.
    assert freeze_path.exists(), 'CONTROLLER_EXECUTION_RESULT_FREEZE_REQUIRED'
    freeze = read(freeze_path)
    assert freeze.get('frozen') is True and freeze['schema_version'] == 'amended_solver_execution_result_freeze_v1'
    bindings = {str(Path(x['path']).resolve()): x['sha256'] for x in freeze['files']}
    assert str(score_path.resolve()) in bindings, 'EXACT_SCORES_BINDING_REQUIRED'
    assert digest(score_path) == bindings[str(score_path.resolve())], 'SCORES_BINDING_CHANGED'
    assert digest(request_path) == ORIGINAL_REQUESTS_SHA256, 'ORIGINAL_REQUESTS_BYTES_CHANGED'
    if str(request_path.resolve()) in bindings:
        assert bindings[str(request_path.resolve())] == ORIGINAL_REQUESTS_SHA256
    final_path = E/'results/FINAL_RESULTS.json'
    construction_path = E/'results/CONSTRUCTION_STRUCTURE_TABLES.json'
    seal_path = E/'solver/INPUT_REVIEW_SEAL.json'
    assert digest(final_path) == ORIGINAL_FINAL_SHA256
    assert digest(construction_path) == ORIGINAL_CONSTRUCTION_SHA256
    assert digest(seal_path) == ORIGINAL_SEAL_SHA256
    scores, requests = read(score_path), read(request_path)
    original, construction, seal = read(final_path), read(construction_path), read(seal_path)
    assert scores['namespace'] == NAMESPACE and scores['primary_scoring'] == 'final_answer_only'
    assert len(requests) == len(scores['requests']) == 1008
    reqmap = {r['request_id']: r for r in requests}
    assert len(reqmap) == 1008 and len({r['request_id'] for r in scores['requests']}) == 1008
    assert set(reqmap) == {r['request_id'] for r in scores['requests']}, 'SCORE_REQUEST_ID_MISMATCH'
    outdir = A/'results'
    destinations = [outdir/'BEHAVIOR_SUMMARY.json', outdir/'FEEDBACK_PATTERN_DIAGNOSTICS.json']
    assert not any(p.exists() for p in destinations), 'NO_RESULT_OVERWRITE'
    source_registry = {}

    def source(path, pointer=None):
        path = Path(path).resolve()
        h = digest(path)
        source_registry[str(path)] = {'path': str(path), 'sha256': h}
        return {**source_registry[str(path)], **({'json_pointer': pointer} if pointer is not None else {})}

    for path in (Path(__file__), freeze_path, score_path, request_path, final_path, construction_path, seal_path, E/'tools/postsolver_diagnostics.py'):
        source(path)
    sealmap = {r['candidate_id']: (i, r) for i, r in enumerate(seal['items'])}
    cmap = {r['candidate_id']: r for r in construction['opportunities'] if r['candidate_id']}
    fmap = {r['candidate_id']: r for r in original['opportunities'] if r['candidate_id']}
    assert len(sealmap) == 30
    candidates = {}
    for qid in sorted({r['candidate_id'] for r in requests}):
        si, item = sealmap[qid]
        path = Path(item['raw_draft']['path'])
        assert digest(path) == item['raw_draft']['sha256']
        candidates[qid] = candidate(read(path), qid)
        source(path)
        assert qid not in ('Q_24', 'Q_29')
    assert len(candidates) == 28
    rows = []
    metadata = ('candidate_id', 'root_id', 'group', 'question_type', 'context_mode', 'variant_id', 'repeat_index', 'sampling_seed', 'protocol_markers', 'clean_contrast_eligible')
    for i, r in enumerate(scores['requests']):
        req = reqmap[r['request_id']]
        assert all(r[k] == req[k] for k in metadata), ('SCORED_METADATA_CHANGED', r['request_id'])
        assert r['protocol_markers'] == fmap[r['candidate_id']]['final_protocol_markers']
        assert r['clean_contrast_eligible'] == (not r['protocol_markers'])
        assert r['execution_status'] in ('RETURNED', 'MISSING', 'BACKEND_INVALID')
        assert isinstance(r['valid'], bool) and isinstance(r['exact_correct'], bool)
        if r['execution_status'] != 'RETURNED':
            assert not r['valid'] and not r['exact_correct']
        if not r['valid']:
            assert not r['exact_correct']
        row = {k: r[k] for k in ('request_id',)+metadata}
        row.update(execution_status=r['execution_status'], valid=r['valid'], exact_correct=r['exact_correct'],
                   score_source=source(score_path, '/requests/'+str(i)))
        for key in ('J', 'precision', 'recall', 'partial_score'):
            if key in r:
                assert isinstance(r[key], (int, float)) and math.isfinite(r[key])
                row[key] = r[key]
        if r['valid']:
            row['predicted_option_ids'] = r['predicted_option_ids']
        row.update(error_patterns(r, candidates[r['candidate_id']]))
        rows.append(row)
    assert Counter(r['context_mode'] for r in rows) == {'full_context': 504, 'no_context': 504}
    assert Counter(r['question_type'] for r in rows) == {'multi_select': 720, 'ordering': 288}
    assert all(r['protocol_markers'] for r in rows if r['group'].startswith('NEW'))
    overall = summarize(rows)
    assert overall['planned'] == freeze['counts']['planned'] == 1008
    assert overall['returned'] == freeze['counts']['returned']
    assert overall['infrastructure_missing'] == freeze['counts']['missing']
    # Compare own aggregate with adapter summaries, without recomputing any score.
    for mode in MODES:
        own = summarize([r for r in rows if r['context_mode'] == mode])
        frozen = scores['by_context'][mode]
        assert own['returned'] == frozen['executed_returned_calls']
        assert own['invalid_returned'] == frozen['invalid_executed_calls']
        assert own['exact_correct_returned'] == frozen['exact_correct_calls']
        assert own['exact_accuracy_returned'] == frozen['exact_accuracy_executed']
        assert own['exact_accuracy_planned'] == frozen['planned_accuracy']
        typed = frozen.get('metrics_by_type', {})
        for kind, x in own['metrics_by_type'].items():
            y = typed[kind] if typed else frozen
            if kind != 'partial_order':
                assert x['mean_J_returned'] == y['mean_J_executed']
    dimensions = {'by_group': ('group',), 'by_root': ('root_id',), 'by_item': ('candidate_id',),
                  'by_type': ('question_type',), 'by_context': ('context_mode',),
                  'by_group_type_context': ('group', 'question_type', 'context_mode'),
                  'by_root_type_context': ('root_id', 'question_type', 'context_mode'),
                  'by_item_context': ('candidate_id', 'context_mode'),
                  'by_protocol_stratum_type_context': ('clean_contrast_eligible', 'question_type', 'context_mode')}
    grouped_summaries = {name: grouped(rows, fields) for name, fields in dimensions.items()}
    triage = []
    for qid in sorted(candidates):
        sub = [r for r in rows if r['candidate_id'] == qid and r['context_mode'] == 'no_context']
        stat = summarize(sub)
        if stat['planned'] == stat['returned'] == 18 and stat['exact_accuracy_returned'] >= NO_CONTEXT_HIGH_EXACT_THRESHOLD:
            c, frozen_c = candidates[qid], cmap[qid]
            triage.append({'candidate_id': qid, 'root_id': sub[0]['root_id'], 'group': sub[0]['group'],
                           'question_type': sub[0]['question_type'], 'no_context': stat,
                           'frozen_draft_source': source(sealmap[qid][1]['raw_draft']['path']),
                           'visible_question': c['question'],
                           'visible_options': [{k: option[k] for k in ('option_id', 'id', 'label', 'text') if k in option} for option in c['options']],
                           'previously_frozen_structure_evidence': frozen_c['independent_structure'],
                           'wording_shortcut_audit_status': 'PENDING_BOUNDED_EVIDENCE_REVIEW_NOT_AUTOMATIC_PROOF'})
    limits = ['All requests are correlated repeated observations of fixed exposed training roots, not independent generalization trials.',
              'Protocol flags and original N/S/quality/overlap/fidelity remain unchanged. No clean NEW contrast exists; descriptive scores do not establish causal F/G or workflow effects.',
              'J is reported separately for multi_select and ordering; directed-pair precision/recall/F1 is never averaged with J. Both partial_order items remain held, with no behavioral estimate.',
              'Missing/backend requests are infrastructure missing, excluded from returned-model denominators. Returned invalid answers count as zero under the frozen scorer; semantic patterns apply only to valid returned answers.',
              'Pattern counts are nonexclusive, compared to independently frozen source-reviewed Gold; they are not prospective author-hypothesis hits or training benefit.',
              'A Gold alternate graph path is a diagnostic only; it does not prove source primitive dependency or internal reasoning.',
              'Full/No-context remain separate. High No-context is wording-audit triage, not automatic quality rejection; low No-context never proves novelty.',
              'No score changes a candidate, selection, target, Gold, prompt, source, structural endpoint or expansion decision. S=0 remains the no-expansion stop.',
              'Original failed execution and original final report remain immutable. Only missing behavioral observations receive new amended evidence; comparable incremental role costs remain null.']
    common = {'schema_version': 'amended_postscore_diagnostics_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'namespace': NAMESPACE, 'execution_result_freeze': source(freeze_path), 'scores': source(score_path),
              'original_results': source(final_path), 'original_construction': source(construction_path),
              'original_failed_attempt_preserved': True, 'original_execution_writes': 0, 'raw_answer_files_read': 0,
              'new_model_API_tokenizer_calls': 0, 'training': False, 'export': False,
              'primary_scoring': 'final_answer_only', 'limits': limits}
    diagnostics = {**common, 'artifact_kind': 'FEEDBACK_PATTERN_DIAGNOSTICS', 'requests': rows,
                   'overall': overall, **grouped_summaries,
                   'explicit_prospective_author_hypotheses_available': False, 'hypothesis_hit_rate': None,
                   'no_context_high_exact_trigger': {'all18_returned_required': True, 'exact_accuracy_threshold': NO_CONTEXT_HIGH_EXACT_THRESHOLD, 'status': 'PREFROZEN_DIAGNOSTIC_TRIAGE_ONLY'},
                   'no_context_wording_audit_queue': triage}
    summary = {**common, 'artifact_kind': 'DESCRIPTIVE_BEHAVIOR_SUMMARY', 'overall': overall, **grouped_summaries,
               'original_descriptive_N': {'OLD_F': 2, 'OLD_G': 3, 'NEW_F': 6, 'NEW_G': 6},
               'original_S': dict.fromkeys(GROUPS, 0), 'expansion_decision': 'DO_NOT_EXPAND_S_ZERO',
               'both_partial_order_holds_retained': True, 'partial_order_behavior_metrics': None,
               'clean_NEW_contrast_available': False, 'feedback_causal_effect': None,
               'incremental_role_tokens': None, 'total_monetary_cost': None, 'cost_per_effective_task': None,
               'no_context_wording_audit_queue_candidate_ids': [x['candidate_id'] for x in triage],
               'no_context_wording_audit_status': 'PENDING_BOUNDED_REVIEW' if triage else 'NO_HIGH_EXACT_TRIGGERED_ITEMS'}
    for result in (summary, diagnostics):
        result['source_artifacts'] = sorted(source_registry.values(), key=lambda x: x['path'])
    for item in source_registry.values():
        assert digest(item['path']) == item['sha256'], 'SOURCE_CHANGED_DURING_DIAGNOSTICS'
    outdir.mkdir(parents=True, exist_ok=True)
    for path, result in zip(destinations, (summary, diagnostics)):
        assert path.is_relative_to(A) and not path.is_relative_to(E)
        with path.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({'outputs': [{'path': str(p), 'sha256': digest(p)} for p in destinations],
                      'overall': overall, 'no_context_high_queue': [x['candidate_id'] for x in triage]}, ensure_ascii=False))


def self_test():
    # Denominator boundaries, frozen-pattern semantics and type isolation only.
    c = {'question_type': 'multi_select', 'gold_option_ids': ['a', 'b']}
    valid = {'execution_status': 'RETURNED', 'valid': True, 'exact_correct': False,
             'question_type': 'multi_select', 'predicted_option_ids': ['a', 'x'], 'J': -0.25}
    assert error_patterns(valid, c) == {'patterns': ['SUPPORTED_SET_OMISSION', 'UNSUPPORTED_EXTRA_SELECTION'], 'missing_ids': ['b'], 'extra_ids': ['x']}
    invalid = {'execution_status': 'RETURNED', 'valid': False, 'exact_correct': False, 'question_type': 'multi_select', 'J': 0.0}
    missing = {'execution_status': 'MISSING', 'valid': False, 'exact_correct': False, 'question_type': 'ordering', 'J': 0.0}
    rows = [{**r, **error_patterns(r, c)} for r in (valid, invalid, missing)]
    s = summarize(rows)
    assert (s['planned'], s['returned'], s['infrastructure_missing'], s['valid_returned'], s['invalid_returned']) == (3, 2, 1, 1, 1)
    assert s['exact_accuracy_planned'] is None and s['exact_accuracy_returned'] == 0.0
    assert s['metrics_by_type']['multi_select']['mean_J_returned'] == -0.125
    assert s['metrics_by_type']['ordering']['mean_J_returned'] is None
    order = {'question_type': 'ordering', 'gold_option_ids': ['a', 'b', 'c']}
    assert error_patterns({**valid, 'predicted_option_ids': ['c', 'a', 'b']}, order)['reversed_gold_pairs'] == [['a', 'c'], ['b', 'c']]
    assert alternate_path({('a', 'b'), ('b', 'c'), ('a', 'c')}, 'a', 'c')
    assert not alternate_path({('a', 'b'), ('a', 'c')}, 'a', 'c')
    print(json.dumps({'status': 'PASS_PURE_SYNTHETIC_SELF_TEST', 'actual_score_or_answer_reads': 0, 'model_API_calls': 0, 'files_written': 0}))


if __name__ == '__main__':
    main()
