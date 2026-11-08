"""Freeze full-dialogue consistency reviews; block unresolved sources before generation.

Semantic findings are explicitly authored judgments, never inferred from model
scores. This validator establishes complete records and exact citations, not
the truth of every statement. Historical question pools are retained unchanged.
"""
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = {'time', 'location', 'actor_identity', 'plan_vs_completion', 'cross_scene_continuity', 'semantic_repetition'}
KINDS = CHECKS | {'naturalness'}


def need(value, message):
    if not value:
        raise ValueError('source_consistency:' + message)


def read(path):
    def strict_pairs(pairs):
        result = {}
        for k, v in pairs:
            need(k not in result, 'duplicate_json_key:' + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: need(False, 'invalid_json_constant:' + value))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError('source_consistency:timestamp_required')
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'timezone_required')
    return parsed


def text(value):
    return isinstance(value, str) and bool(value.strip())


def verify_binding(row, path=None):
    need(isinstance(row, dict) and set(row) == {'path', 'sha256'}
         and text(row['path']) and Path(row['path']).is_absolute() and text(row['sha256']), 'binding_schema')
    need(path is None or Path(row['path']).resolve() == Path(path).resolve(), 'binding_path_changed')
    need(bind(row['path']) == row, 'binding_changed:' + row['path'])


def index(rows, key):
    need(isinstance(rows, list) and all(isinstance(r, dict) and text(r.get(key)) for r in rows), 'row_schema:' + key)
    result = {r[key]: r for r in rows}
    need(len(result) == len(rows), 'duplicate_identity:' + key)
    return result


def units_for(source):
    turns = source.get('turns'); by_ref = index(turns, 'turn_ref'); groups = OrderedDict()
    need(bool(turns), 'empty_source')
    for row in turns:
        need(all(text(row.get(k)) for k in ('speaker', 'text', 'origin')), 'source_turn_schema')
        need(row['origin'] in ('source_grounded', 'generated_extension', 'provisional_candidate'), 'unknown_origin')
        unit = row.get('session_ref') if row['origin'] == 'source_grounded' else row.get('scene_ref')
        need(text(unit), 'source_unit_missing')
        groups.setdefault(unit, []).append(row)
        need(len({r['origin'] for r in groups[unit]}) == 1, 'mixed_unit_origins')
    need([t['turn_ref'] for rows in groups.values() for t in rows] == list(by_ref), 'noncontiguous_units')
    return groups


def config(home=HERE):
    home = Path(home).resolve(); value = read(home / 'CONFIG.json')
    need(value['schema_version'] == 'source_consistency_gate_v1'
         and value['root_policy'] == 'hold_entire_full_source_if_any_blocking_unit'
         and value['training_allowed'] is False and value['automatic_retry'] is False
         and value['source_edits_allowed'] is False, 'config_policy')
    verify_binding(value['source_config']); upstream = read(value['source_config']['path'])
    need(Path(value['upstream_engine']).resolve() == Path(value['source_config']['path']).parent, 'upstream_engine_mismatch')
    roots = index(value['roots'], 'root_id'); slots = index(upstream['ordering_slots'], 'root_id')
    need(set(roots) == set(slots) == {s['root_id'] for s in upstream['multiselect_config']['slots']}, 'root_coverage')
    need(value['baseline_pool']['path'] == upstream['baseline_pool'], 'baseline_pool_changed')
    for root, row in roots.items():
        need(row['source'] == {'path': slots[root]['source_path'], 'sha256': slots[root]['source_sha256']}, 'configured_source_changed')
        need(Path(row['review_path']).resolve().parent == home / 'reviews', 'review_path_outside_home')
    need({p.resolve() for p in (home / 'reviews').glob('*.json')} == {Path(r['review_path']).resolve() for r in roots.values()},
         'missing_or_unregistered_review')
    return value


def validate_review(row):
    verify_binding(row['source']); source = read(row['source']['path'])
    need(source['root_id'] == row['root_id'], 'source_root_changed')
    path = Path(row['review_path']); original = bind(path); review = read(path)
    need(review.get('schema_version') == 'dialogue_consistency_v1'
         and review.get('root_id') == row['root_id'] and review.get('source') == row['source']
         and review.get('review_scope') == 'source_consistency_only'
         and text(review.get('reviewer')) and text(review.get('limitations')), 'review_identity_or_scope')
    declarations = {'full_source_read': True, 'current_question_gold_read': False, 'current_solver_scores_read': False,
                    'historical_question_gold_known': True, 'fresh_blind_review': False}
    need(all(type(review.get(k)) is bool and review[k] is v for k,v in declarations.items()), 'review_disclosure')
    need(timestamp(review['created_at_utc']) <= timestamp(now()), 'future_review')
    need(review.get('turn_refs_read') == [t['turn_ref'] for t in source['turns']], 'incomplete_full_source_read')
    need(isinstance(review.get('checks_completed'), dict) and set(review['checks_completed']) == CHECKS
         and all(v is True for v in review['checks_completed'].values()), 'incomplete_consistency_checks')
    groups = units_for(source); units = index(review.get('units'), 'unit_id'); findings = index(review.get('findings'), 'finding_id')
    need(list(units) == list(groups), 'incomplete_unit_coverage')
    turns = {t['turn_ref']:t for t in source['turns']}
    unit_by_turn = {t['turn_ref']:u for u, rows in groups.items() for t in rows}
    for finding in findings.values():
        need(finding.get('severity') in ('BLOCKING', 'WARNING') and finding.get('kind') in KINDS
             and text(finding.get('reason')) and text(finding.get('proposed_action')), 'finding_schema')
        us = finding.get('unit_ids'); es = finding.get('evidence')
        need(isinstance(us, list) and us and all(text(u) for u in us)
             and len(set(us)) == len(us) and set(us) <= set(groups), 'finding_unit_coverage')
        need(isinstance(es, list) and es and all(isinstance(e, dict) and set(e) == {'turn_ref', 'quote'} for e in es), 'finding_evidence_required')
        for e in es:
            need(text(e['turn_ref']) and e['turn_ref'] in turns and text(e['quote'])
                 and e['quote'] in turns[e['turn_ref']]['text'], 'finding_quote_not_exact')
        need({unit_by_turn[e['turn_ref']] for e in es} == set(us), 'finding_evidence_unit_mismatch')
    for uid, rows in groups.items():
        unit = units[uid]; relevant = [f for f in findings.values() if uid in f['unit_ids']]
        expected = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in relevant) else 'WARN' if relevant else 'PASS'
        need(unit.get('origin') == rows[0]['origin'] and unit.get('turn_refs') == [t['turn_ref'] for t in rows]
             and text(unit.get('reason')) and unit.get('decision') == expected
             and unit.get('finding_ids') == [f['finding_id'] for f in relevant], 'unit_decision_or_coverage')
    decision = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in findings.values()) else 'READY'
    need(review.get('decision') == decision, 'root_decision_disagrees_with_findings')
    need(bind(path) == original and bind(row['source']['path']) == row['source'], 'inputs_changed_during_review')
    return review


def verify_reviews(home=HERE):
    home = Path(home).resolve(); value = config(home)
    # All source-only records validate before opening the baseline question pool.
    reviews = [validate_review(row) for row in value['roots']]
    roots = []
    for review in reviews:
        roots.append({'root_id': review['root_id'], 'decision': review['decision'], 'source': review['source'],
                      'turn_count': len(review['turn_refs_read']), 'unit_count': len(review['units']),
                      'unit_decisions': dict(Counter(u['decision'] for u in review['units'])),
                      'blocking_findings': [f['finding_id'] for f in review['findings'] if f['severity'] == 'BLOCKING'],
                      'warning_findings': [f['finding_id'] for f in review['findings'] if f['severity'] == 'WARNING'],
                      'held_units': [u['unit_id'] for u in review['units'] if u['decision'] == 'HOLD']})
    return {'schema_version': 'source_consistency_summary_v1', 'roots': roots,
            'ready_roots': [r['root_id'] for r in roots if r['decision'] == 'READY'],
            'held_roots': [r['root_id'] for r in roots if r['decision'] == 'HOLD'],
            'review_files': [bind(row['review_path']) for row in value['roots']],
            'latest_review_at_utc': max(timestamp(r['created_at_utc']) for r in reviews).isoformat(),
            'full_source_policy': value['root_policy'], 'ready_is_question_answer_certification': False,
            'source_edits': [], 'question_or_gold_edits': [], 'training_allowed': False}


def cited_refs(obj):
    if isinstance(obj, dict):
        found = {obj['turn_ref']} if text(obj.get('turn_ref')) else set()
        for value in obj.values():
            found |= cited_refs(value)
        return found
    if isinstance(obj, list):
        return set().union(*(cited_refs(v) for v in obj)) if obj else set()
    if isinstance(obj, str):
        # Includes native support_refs and alternative anchor reference arrays.
        return {obj}
    return set()


def question_impact(home, summary):
    """Citations only: affected means review needed, never automatically wrong."""
    value = config(home); verify_binding(value['baseline_pool'])
    questions = read(value['baseline_pool']['path'])['questions']; index(questions, 'candidate_id')
    roots = {r['root_id']:r for r in summary['roots']}; entries = []
    for question in questions:
        root = question['root_id']
        if root not in roots:
            continue
        reviewed = roots[root]; source = read(reviewed['source']['path'])
        groups = units_for(source); held = set(reviewed['held_units'])
        held_refs = {t['turn_ref'] for unit, rows in groups.items() if unit in held for t in rows}
        evidence_fields = ('option_evidence', 'item_evidence', 'order_anchor_spans', 'generated_order_anchor_spans',
                           'support_refs', 'alternative_matching_anchors', 'allowed_matching_anchor_spans')
        direct = sorted(set().union(*(cited_refs(question.get(k)) for k in evidence_fields)) & held_refs)
        same_version = question['source_sha256'] == reviewed['source']['sha256']
        classification = ('SOURCE_VERSION_RECHECK_REQUIRED' if not same_version else
                          'DIRECT_HELD_UNIT_CITATION' if direct else
                          'FULL_SOURCE_HOLD_NO_DIRECT_CITATION' if reviewed['decision'] == 'HOLD' else
                          'NO_SOURCE_GATE_HOLD')
        entries.append({'candidate_id': question['candidate_id'], 'root_id': root,
                        'question_type': question['question_type'], 'classification': classification,
                        'matching_held_turn_refs': direct, 'question_source_sha256': question['source_sha256'],
                        'question_answer_recertified': False, 'existing_pool_modified': False})
    return {'schema_version': 'source_consistency_question_impact_v1', 'baseline_pool': value['baseline_pool'],
            'questions': entries, 'classification_counts': dict(Counter(e['classification'] for e in entries)),
            'scope': 'Existing question dependency inventory only; citations are an approximate locator. No automatic answer verdict or deletion.',
            'question_or_gold_edits': [], 'existing_pool_modified': False}


def frozen_inputs(home, value):
    home = Path(home).resolve(); paths = [home / 'CONFIG.json', Path(value['source_config']['path']), Path(value['baseline_pool']['path'])]
    for row in value['roots']:
        paths += [Path(row['source']['path']), Path(row['review_path'])]
    paths += [HERE / 'source_gate.py', HERE / 'guarded_batch.py']
    engine = Path(value['upstream_engine'])
    paths += list(engine.glob('*.py')) + list(engine.glob('*.md'))
    return sorted(set(paths))


def finalize(home=HERE):
    home = Path(home).resolve()
    names = ('CONSISTENCY_SUMMARY.json', 'QUESTION_IMPACT.json', 'CONSISTENCY_REVIEW.COMPLETE.json')
    need(not any((home / n).exists() for n in names), 'review_already_finalized')
    summary = verify_reviews(home); value = config(home)
    impact_replay_started = now()
    need(timestamp(summary['latest_review_at_utc']) <= timestamp(impact_replay_started), 'impact_read_before_review_saved')
    impact = question_impact(home, summary)
    write(home / names[0], summary); write(home / names[1], impact)
    paths = frozen_inputs(home, value) + [home / names[0], home / names[1]]
    result = {'schema_version': 'source_consistency_freeze_v1', 'status': 'FROZEN', 'completed_at_utc': now(),
              'impact_replay_started_at_utc': impact_replay_started, 'files': [bind(p) for p in sorted(set(paths))],
              'all_sources_ready': not summary['held_roots'], 'training_allowed': False}
    write(home / names[2], result); return summary


def verify_frozen(home=HERE):
    home = Path(home).resolve(); summary = verify_reviews(home); value = config(home)
    frozen = read(home / 'CONSISTENCY_REVIEW.COMPLETE.json')
    need(frozen.get('schema_version') == 'source_consistency_freeze_v1' and frozen.get('status') == 'FROZEN'
         and frozen.get('training_allowed') is False
         and frozen.get('all_sources_ready') is (not summary['held_roots']), 'frozen_policy')
    need(timestamp(summary['latest_review_at_utc']) <= timestamp(frozen['impact_replay_started_at_utc'])
         <= timestamp(frozen['completed_at_utc']) <= timestamp(now()), 'review_impact_freeze_time')
    paths = frozen_inputs(home, value) + [home / 'CONSISTENCY_SUMMARY.json', home / 'QUESTION_IMPACT.json']
    records = index(frozen.get('files'), 'path')
    need(set(records) == {str(p.resolve()) for p in paths}, 'frozen_binding_coverage')
    # Public source/review hashes first; private pool is not opened on stale judgments.
    for path in [Path(r['review_path']) for r in value['roots']]:
        verify_binding(records[str(path.resolve())], path)
    for record in records.values():
        verify_binding(record)
    need(read(home / 'CONSISTENCY_SUMMARY.json') == summary, 'frozen_summary_changed')
    need(read(home / 'QUESTION_IMPACT.json') == question_impact(home, summary), 'frozen_impact_changed')
    return summary


def require_ready(home=HERE):
    summary = verify_frozen(home)
    need(not summary['held_roots'], 'held_sources:' + ','.join(summary['held_roots']))
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'finalize', 'verify'))
    parser.add_argument('--home', type=Path, default=HERE); args = parser.parse_args()
    print(json.dumps({'check': verify_reviews, 'finalize': finalize, 'verify': verify_frozen}[args.action](args.home), ensure_ascii=False, indent=2))
