"""Admit saved fresh six-fact judgments without fabricating historical exposure.

This module validates records and projects private parent evidence. It does not
infer semantic support, author judgments, invoke a model, or certify READY as a
question answer. The public facts and source judgments must exist before private
Gold is constructed. Historical v89 delivery files remain read-only.
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PARENT_REVIEW_VERSION = 'v97_fresh_parent_source_judgments_v1'
PROVENANCE_VERSION = 'v97_fresh_parent_private_construction_v1'
PARENT_ORIGIN = 'v97_fresh_six_fact_parent_saved_full_source_review'
CHECKS = {'time', 'location', 'actor_identity', 'plan_vs_completion',
          'cross_scene_continuity', 'semantic_repetition'}
PUBLIC_FIELDS = {'candidate_id', 'root_id', 'question_type', 'question',
                 'options', 'source_path', 'source_sha256'}


def need(value, reason):
    if not value:
        raise ValueError('fresh_parent:' + reason)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def read(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, 'duplicate_json_key:' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: need(False, 'invalid_constant:' + value))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError('fresh_parent:timestamp_required')
    need(result.utcoffset() is not None, 'timezone_required')
    return result


def verify_binding(row):
    need(isinstance(row, dict) and set(row) == {'path', 'sha256'}
         and text(row['path']) and Path(row['path']).is_absolute()
         and isinstance(row['sha256'], str)
         and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'binding_schema')
    need(bind(row['path']) == row, 'binding_changed:' + row['path'])


def exact_evidence(evidence, turns):
    need(isinstance(evidence, list) and bool(evidence), 'evidence_required')
    for row in evidence:
        need(isinstance(row, dict) and set(row) == {'turn_ref', 'speaker', 'quote'}, 'evidence_schema')
        turn = turns.get(row['turn_ref'])
        need(turn is not None and row['speaker'] == turn['speaker'] and text(row['quote'])
             and row['quote'] in turn['text'], 'evidence_not_exact')


def validate_source_review(row):
    """The v77 six consistency checks with truthful exposure instead of fixed old-Gold exposure."""
    verify_binding(row['source']); source = read(row['source']['path'])
    need(source['root_id'] == row['root_id'], 'source_root_changed')
    path = Path(row['review_path']); binding = bind(path); review = read(path)
    need(review.get('schema_version') == 'dialogue_consistency_v1'
         and review.get('root_id') == row['root_id'] and review.get('source') == row['source']
         and review.get('review_scope') == 'source_consistency_only'
         and text(review.get('reviewer')) and text(review.get('limitations')), 'source_review_identity')
    for key, expected in {'full_source_read': True, 'current_question_gold_read': False,
                          'current_solver_scores_read': False}.items():
        need(type(review.get(key)) is bool and review[key] is expected, 'source_review_disclosure:' + key)
    for key in ('historical_question_gold_known', 'fresh_blind_review'):
        need(type(review.get(key)) is bool, 'source_review_disclosure_boolean:' + key)
    need(not review['fresh_blind_review'] or not review['historical_question_gold_known'],
         'source_fresh_blind_incompatible_with_known_historical_gold')
    need(timestamp(review['created_at_utc']) <= timestamp(now()), 'source_review_future')
    turns = source.get('turns')
    need(isinstance(turns, list) and bool(turns), 'source_turns_required')
    lookup = {}; groups = OrderedDict()
    for turn in turns:
        need(isinstance(turn, dict) and all(text(turn.get(k)) for k in ('turn_ref', 'speaker', 'text', 'origin')),
             'source_turn_schema')
        need(turn['turn_ref'] not in lookup, 'source_duplicate_ref'); lookup[turn['turn_ref']] = turn
        need(turn['origin'] in ('source_grounded', 'generated_extension', 'provisional_candidate'), 'source_origin')
        uid = turn.get('session_ref') if turn['origin'] == 'source_grounded' else turn.get('scene_ref')
        need(text(uid), 'source_unit_missing'); groups.setdefault(uid, []).append(turn)
        need(len({t['origin'] for t in groups[uid]}) == 1, 'source_unit_mixed_origin')
    need([t['turn_ref'] for values in groups.values() for t in values] == list(lookup), 'source_unit_noncontiguous')
    need(review.get('turn_refs_read') == list(lookup), 'source_full_turn_coverage')
    need(isinstance(review.get('checks_completed'), dict) and set(review['checks_completed']) == CHECKS
         and all(v is True for v in review['checks_completed'].values()), 'source_all_six_checks')
    units = review.get('units'); findings = review.get('findings')
    need(isinstance(units, list) and all(isinstance(u, dict) for u in units)
         and [u.get('unit_id') for u in units] == list(groups), 'source_unit_coverage')
    need(isinstance(findings, list) and all(isinstance(f, dict) and text(f.get('finding_id')) for f in findings)
         and len({f['finding_id'] for f in findings}) == len(findings), 'source_finding_identity')
    unit_by_ref = {t['turn_ref']: uid for uid, rows in groups.items() for t in rows}
    for finding in findings:
        need(finding.get('severity') in ('BLOCKING', 'WARNING') and finding.get('kind') in CHECKS | {'naturalness'}
             and text(finding.get('reason')) and text(finding.get('proposed_action')), 'source_finding_schema')
        ids = finding.get('unit_ids'); evidence = finding.get('evidence')
        need(isinstance(ids, list) and ids and all(text(uid) for uid in ids)
             and len(set(ids)) == len(ids) and set(ids) <= set(groups), 'source_finding_units')
        need(isinstance(evidence, list) and evidence and all(isinstance(e, dict) and set(e) == {'turn_ref', 'quote'}
             and e['turn_ref'] in lookup and text(e['quote']) and e['quote'] in lookup[e['turn_ref']]['text']
             for e in evidence), 'source_finding_exact_quotes')
        need({unit_by_ref[e['turn_ref']] for e in evidence} == set(ids), 'source_finding_quote_units')
    for unit in units:
        uid = unit['unit_id']; rows = groups[uid]; relevant = [f for f in findings if uid in f['unit_ids']]
        decision = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in relevant) else 'WARN' if relevant else 'PASS'
        need(unit.get('origin') == rows[0]['origin'] and unit.get('turn_refs') == [t['turn_ref'] for t in rows]
             and text(unit.get('reason')) and unit.get('decision') == decision
             and unit.get('finding_ids') == [f['finding_id'] for f in relevant], 'source_unit_decision')
    decision = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in findings) else 'READY'
    need(review.get('decision') == decision, 'source_root_decision')
    need(bind(path) == binding and bind(row['source']['path']) == row['source'], 'source_review_inputs_changed')
    return review


def validate_judgments(public_path, judgment_path, expansion_path):
    public_path = Path(public_path).resolve(); judgment_path = Path(judgment_path).resolve()
    public = read(public_path); judgment = read(judgment_path)
    need(set(public) == PUBLIC_FIELDS, 'public_parent_exact_fields_no_private_gold_or_evidence')
    need(all(text(public[k]) for k in ('candidate_id', 'root_id', 'question', 'source_path', 'source_sha256'))
         and public['question_type'] == 'multi_select', 'public_parent_schema')
    need(isinstance(public['options'], dict) and set(public['options']) == set('ABCDEF')
         and all(text(v) for v in public['options'].values()), 'public_parent_six_options')
    normalized = [''.join(c for c in v.casefold() if c.isalnum()) for v in public['options'].values()]
    need(len(set(normalized)) == 6, 'public_parent_duplicate_fact')
    source_path = Path(public['source_path']).resolve(); source = read(source_path)
    need(source['root_id'] == public['root_id'] and bind(source_path)['sha256'] == public['source_sha256'], 'parent_source_binding')
    lookup = {t['turn_ref']: t for t in source['turns']}
    need(lookup and len(lookup) == len(source['turns']), 'parent_source_unique_turns')
    need(judgment.get('version') == PARENT_REVIEW_VERSION
         and judgment.get('root_id') == public['root_id'] and judgment.get('candidate_id') == public['candidate_id']
         and judgment.get('public_question') == public
         and {'path': judgment.get('public_question_path'), 'sha256': judgment.get('public_question_sha256')} == bind(public_path)
         and judgment.get('source_path') == str(source_path) and judgment.get('source_sha256') == public['source_sha256'],
         'parent_review_identity_and_public_binding')
    need(text(judgment.get('reviewer')) and text(judgment.get('review_mode')) and text(judgment.get('limitations')),
         'parent_reviewer_and_limitations')
    need(judgment.get('full_source_read') is True and type(judgment.get('full_source_turn_count')) is int
         and judgment['full_source_turn_count'] == len(source['turns'])
         and judgment.get('turn_refs_read') == list(lookup) and judgment.get('stem_valid') is True,
         'parent_full_source_read_and_stem')
    for key, expected in {'known_old_parent_gold': False,
                          'current_parent_private_gold_read_before_judgments_saved': False,
                          'current_parent_private_evidence_read_before_judgments_saved': False,
                          'current_solver_results_read': False}.items():
        need(type(judgment.get(key)) is bool and judgment[key] is expected, 'parent_disclosure:' + key)
    for key in ('fresh_blind_review', 'public_parent_authored_by_reviewer',
                'proposed_all_supported_intent_known', 'historical_question_gold_known'):
        need(type(judgment.get(key)) is bool, 'parent_disclosure_boolean:' + key)
    need(not judgment['fresh_blind_review'] or (not judgment['public_parent_authored_by_reviewer']
         and not judgment['historical_question_gold_known']), 'parent_fresh_blind_exposure_incompatible')
    need(judgment.get('content_status') == 'CORE_USABLE', 'parent_not_core_usable')
    need(timestamp(judgment['created_at_utc']) <= timestamp(now()), 'parent_judgment_future')
    rows = judgment.get('judgments')
    need(isinstance(rows, list) and len(rows) == 6 and [j.get('option') for j in rows] == list('ABCDEF'), 'parent_judgment_coverage')
    for item in rows:
        need(item.get('proposition') == public['options'][item['option']]
             and item.get('source_judgment') == 'supported' and item.get('relation') == 'supports'
             and item.get('entailed') is True and item.get('scope_valid') is True
             and item.get('fact_distinct') is True and text(item.get('reason')), 'parent_saved_supported_fact')
        exact_evidence(item.get('required_evidence'), lookup)
    expansion = read(expansion_path)
    need(expansion.get('root_id') == public['root_id'] and expansion.get('output_source') == bind(source_path),
         'expanded_source_provenance_identity')
    return public, judgment, source


def construct_parent(public_path, judgment_path, expansion_path, provenance_path):
    """Only this saved-review path creates private Gold; authoring a public draft is separate."""
    public, judgment, source = validate_judgments(public_path, judgment_path, expansion_path)
    created = now()
    need(timestamp(judgment['created_at_utc']) <= timestamp(created), 'private_created_before_saved_judgments')
    parent = deepcopy(public); parent['gold'] = list('ABCDEF'); parent['option_evidence'] = {}
    for item in judgment['judgments']:
        parent['option_evidence'][item['option']] = {**deepcopy(item['required_evidence'][0]),
            'relation': 'supports', 'reason': item['reason'], 'rationale': item['reason']}
    parent.update(parent_origin=PARENT_ORIGIN, historical_parent_gold_known=False,
                  current_parent_review=bind(judgment_path), training_allowed=False, formal_training_admission=False)
    provenance = {'version': PROVENANCE_VERSION, 'created_at_utc': created,
        'source_judgments_saved_at_utc': judgment['created_at_utc'], 'new_parent_candidate_id': public['candidate_id'],
        'public_parent': bind(public_path), 'source_judgments': bind(judgment_path),
        'source': bind(public['source_path']), 'expanded_source_provenance': bind(expansion_path),
        'historical_parent_reused': False, 'historical_gold_known': False,
        'historical_question_gold_known': judgment['historical_question_gold_known'],
        'fresh_blind_review': judgment['fresh_blind_review'],
        'public_parent_authored_by_reviewer': judgment['public_parent_authored_by_reviewer'],
        'proposed_all_supported_intent_known': judgment['proposed_all_supported_intent_known'],
        'current_private_parent_created_after_saved_source_judgments': True,
        'current_solver_scores_read': False, 'training_allowed': False,
        'bindings': {'source_judgments': bind(judgment_path)}}
    write(provenance_path, provenance)
    return parent, provenance


def validate_slot(slot):
    """Reconstruct the full fresh-parent proof before each shard preparation/verification."""
    need(slot.get('parent_origin') == PARENT_ORIGIN and slot.get('parent_review_mode') == 'saved_fresh_public_parent_full_source_review',
         'fresh_parent_slot_mode')
    for prefix in ('parent_review', 'parent_provenance'):
        need(bind(slot[prefix + '_path'])['sha256'] == slot[prefix + '_sha256'], 'slot_binding:' + prefix)
    provenance = read(slot['parent_provenance_path'])
    need(provenance.get('version') == PROVENANCE_VERSION, 'fresh_provenance_version')
    for key in ('public_parent', 'source_judgments', 'source', 'expanded_source_provenance'):
        verify_binding(provenance[key])
    need(provenance['source_judgments'] == bind(slot['parent_review_path'])
         and provenance['bindings']['source_judgments'] == provenance['source_judgments']
         and provenance['source'] == {'path': slot['source_path'], 'sha256': slot['source_sha256']}, 'slot_provenance_source_and_review')
    public, judgment, source = validate_judgments(provenance['public_parent']['path'], slot['parent_review_path'],
                                                 provenance['expanded_source_provenance']['path'])
    candidates = read(slot['parent_pool_path'])['questions']
    parents = [p for p in candidates if p['candidate_id'] == slot['parent_candidate_id']]
    need(len(parents) == 1, 'slot_unique_parent'); parent = parents[0]
    need(parent['candidate_id'] == public['candidate_id'] == provenance['new_parent_candidate_id']
         and parent['root_id'] == public['root_id'] == slot['root_id']
         and parent['options'] == public['options'] and parent['question'] == public['question']
         and parent['source_path'] == public['source_path'] and parent['source_sha256'] == public['source_sha256']
         and parent['gold'] == list('ABCDEF') and parent['parent_origin'] == PARENT_ORIGIN
         and parent['historical_parent_gold_known'] is False and parent['current_parent_review'] == bind(slot['parent_review_path'])
         and parent['training_allowed'] is False and parent['formal_training_admission'] is False, 'private_parent_identity')
    for item in judgment['judgments']:
        expected = {**item['required_evidence'][0], 'relation': 'supports', 'reason': item['reason'], 'rationale': item['reason']}
        need(parent['option_evidence'][item['option']] == expected, 'private_parent_saved_judgment_projection')
    for key in ('fresh_blind_review', 'public_parent_authored_by_reviewer',
                'proposed_all_supported_intent_known', 'historical_question_gold_known'):
        need(provenance[key] == judgment[key], 'provenance_exposure_changed:' + key)
    need(provenance['historical_parent_reused'] is False and provenance['historical_gold_known'] is False
         and provenance['current_private_parent_created_after_saved_source_judgments'] is True
         and provenance['current_solver_scores_read'] is False and provenance['training_allowed'] is False
         and provenance['source_judgments_saved_at_utc'] == judgment['created_at_utc']
         and timestamp(judgment['created_at_utc']) <= timestamp(provenance['created_at_utc']) <= timestamp(now()),
         'fresh_parent_construction_time_and_disclosure')
    need(slot['parent_gold_known_before_review'] is False and slot['parent_gold_known_before_current_review'] is False
         and slot['fresh_blind_parent_review'] == judgment['fresh_blind_review']
         and slot['current_parent_source_reread_claimed'] is True, 'fresh_slot_exposure')
    return {'root_id': slot['root_id'], 'parent_candidate_id': parent['candidate_id'],
            'fresh_blind_parent_review': judgment['fresh_blind_review'], 'full_source_turn_count': len(source['turns'])}
