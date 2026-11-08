"""Required review disclosures and bindings; validation never fills declarations.

These checks establish saved records, coverage and ordering. They do not establish
that a reviewer actually read the source or that a semantic judgment is correct.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 'source_judgment_v1'
COMMON_DISCLOSURES = (
    'parent_all_true_provenance_known', 'parent_gold_known_before_review',
    'fresh_blind_parent_review', 'current_solver_scores_read', 'solver_scores_used',
)
DISCLOSURE_FIELDS = COMMON_DISCLOSURES


def fail(path, stage, field, why):
    raise ValueError(f'review_schema:{path}:{stage}:{field}:{why}')


def need(value, path, stage, field, why):
    if not value:
        fail(path, stage, field, why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def boolean(record, key, path, stage, expected=None):
    need(key in record, path, stage, key, 'required')
    value = record[key]
    need(type(value) is bool, path, stage, key, 'explicit_bool_required')
    if expected is not None:
        need(value is expected, path, stage, key, f'expected_{str(expected).lower()}')
    return value


def timestamp(value, path, stage, field):
    try:
        result = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        fail(path, stage, field, 'timestamp_required')
    need(result.tzinfo is not None and result.utcoffset() is not None,
         path, stage, field, 'timezone_required')
    return result


def validate_disclosures(record, path, stage='round'):
    need(isinstance(record, dict), path, stage, 'record', 'object_required')
    need(record.get('review_schema_version') == SCHEMA_VERSION,
         path, stage, 'review_schema_version', 'unsupported_or_missing')
    for key in COMMON_DISCLOSURES:
        boolean(record, key, path, stage)
    known = record['parent_gold_known_before_review']
    need(not (known and record['fresh_blind_parent_review']), path, stage,
         'fresh_blind_parent_review', 'known_parent_cannot_be_fresh_blind')
    # This route explicitly supplies all-true parents to the writer and reviewer.
    boolean(record, 'parent_all_true_provenance_known', path, stage, True)
    boolean(record, 'parent_gold_known_before_review', path, stage, True)
    boolean(record, 'fresh_blind_parent_review', path, stage, False)
    boolean(record, 'current_solver_scores_read', path, stage, False)
    boolean(record, 'solver_scores_used', path, stage, False)
    if stage == 'round':
        boolean(record, 'new_variant_private_artifacts_read_before_save', path, stage, False)
        boolean(record, 'full_source_read', path, stage, True)
        if 'derived_from_rounds' in record:
            boolean(record, 'derived_from_rounds', path, stage, False)
    elif stage == 'final':
        boolean(record, 'child_gold_read_before_save', path, stage, False)
        boolean(record, 'current_pool_read_before_save', path, stage, False)
        for key in ('new_child_private_artifacts_read', 'asset_reviews_read', 'writer_raw_responses_read'):
            if key in record:
                boolean(record, key, path, stage, False)
    elif stage == 'derived':
        boolean(record, 'derived_from_rounds', path, stage, True)
        boolean(record, 'source_first_per_round', path, stage, True)
        boolean(record, 'new_variant_private_artifacts_read_before_save', path, stage, True)
        need(record.get('review_mode') == 'derived_from_saved_round_source_judgments'
             and 'created_at_utc' in record and record['created_at_utc'] is None,
             path, stage, 'review_mode', 'derived_provenance_required')
        boolean(record, 'aggregate_is_new_independent_blind_review', path, stage, False)
    else:
        fail(path, stage, 'stage', 'unsupported')
    return record


def verify_binding(row, path, stage, field, expected_path=None):
    need(isinstance(row, dict) and isinstance(row.get('path'), str)
         and isinstance(row.get('sha256'), str), path, stage, field, 'binding_required')
    if expected_path is not None:
        need(Path(row['path']).resolve() == Path(expected_path).resolve(),
             path, stage, field, 'path_changed')
    need(sha(row['path']) == row['sha256'], path, stage, field, 'sha_changed')


def rows(record, path, stage):
    keys = [key for key in ('questions', 'rows') if key in record]
    need(len(keys) == 1 and isinstance(record[keys[0]], list),
         path, stage, 'questions', 'one_row_list_required')
    return record[keys[0]]


def index(items, key, path, stage, field):
    need(isinstance(items, list) and all(isinstance(v, dict) and key in v for v in items),
         path, stage, field, 'row_list_required')
    result = {v[key]: v for v in items}
    need(len(result) == len(items), path, stage, field, 'duplicate_rows')
    return result


def complete_count(value, count, path, stage, field='complete_source_turns_read'):
    need(type(value) is int and value == count, path, stage, field, 'complete_source_required')


def validate_option_judgment(row, public_text, turns, path, stage):
    need(row.get('proposition') == public_text, path, stage, 'proposition', 'public_text_changed')
    need(row.get('relation') in ('supports', 'contradicts', 'unknown'),
         path, stage, 'relation', 'invalid')
    boolean(row, 'scope_valid', path, stage)
    boolean(row, 'fact_distinct', path, stage)
    need(isinstance(row.get('reason'), str) and row['reason'].strip(), path, stage, 'reason', 'required')
    citations = row.get('citations')
    need(isinstance(citations, list) and bool(citations), path, stage, 'citations', 'required')
    for c in citations:
        need(isinstance(c, dict), path, stage, 'citations', 'object_required')
        turn = turns.get(c.get('turn_ref'))
        need(turn is not None and c.get('speaker') == turn['speaker']
             and isinstance(c.get('quote'), str) and bool(c['quote'])
             and c['quote'] in turn['text'], path, stage, 'citations', 'exact_source_required')


def validate_parent_reuse(slot, public, current_at, path):
    """Legacy parent records remain legacy, bound as historical evidence only."""
    stage = 'historical_parent_reuse'
    fields = ('parent_review_path', 'parent_review_sha256',
              'parent_provenance_path', 'parent_provenance_sha256')
    if not any(k in slot for k in fields):
        return []  # A synthetic/new parent has no historical reuse claim.
    need(all(k in slot for k in fields), path, stage, 'bindings', 'incomplete_reuse')
    bindings = []; provenance = None
    for prefix in ('parent_review', 'parent_provenance'):
        binding = {'path': slot[prefix + '_path'], 'sha256': slot[prefix + '_sha256']}
        verify_binding(binding, path, stage, prefix)
        legacy = read(binding['path'])
        saved = timestamp(legacy.get('created_at_utc'), binding['path'], stage, 'created_at_utc')
        need(saved <= current_at, binding['path'], stage, 'created_at_utc', 'future_reuse')
        bindings.append(binding)
        if prefix == 'parent_provenance':
            provenance = legacy
    old = read(slot['parent_review_path'])
    # Frozen legacy records have two documented shapes. This reads their public
    # content for reuse bindings; it never translates their disclosures to v1.
    if 'public_question' in old:
        row = old['public_question']
        verify_binding({'path': old.get('public_question_path'), 'sha256': old.get('public_question_sha256')},
                       path, stage, 'legacy_public_question')
        need(read(old['public_question_path']) == row, path, stage, 'legacy_public_question', 'content_changed')
        need(old.get('candidate_id') == row.get('candidate_id') and old.get('root_id') == row.get('root_id')
             and old.get('source_sha256') == row.get('source_sha256'), path, stage, 'legacy_identity', 'changed')
        old_rows = [row]
    else:
        old_rows = rows(old, slot['parent_review_path'], stage)
    matching = [r for r in old_rows if r.get('candidate_id') == slot['parent_candidate_id']]
    need(len(matching) == 1, path, stage, 'candidate_id', 'parent_review_identity')
    row = matching[0]
    need(row.get('root_id') == public['root_id'] and row.get('source_sha256') == public['source_sha256']
         and row.get('question') == public['public_stem'], path, stage, 'source', 'parent_binding_changed')
    options = row.get('options', {})
    for fact in public['parent_facts']:
        prior = options.get(fact['parent_label'], {})
        prior_text = prior if isinstance(prior, str) else prior.get('proposition')
        need(prior_text == fact['proposition'], path, stage,
             'proposition', 'reused_parent_text_changed')
    need(provenance.get('new_parent_candidate_id', provenance.get('parent_candidate_id')) == slot['parent_candidate_id'],
         path, stage, 'parent_provenance', 'wrong_parent')
    source_binding = provenance.get('source', provenance.get('bindings', {}).get('source'))
    if source_binding is None and 'source_path' in provenance:
        source_binding = {'path': provenance['source_path'], 'sha256': provenance.get('source_sha256')}
    verify_binding(source_binding, path, stage, 'parent_provenance_source')
    need(source_binding['sha256'] == public['source_sha256']
         and read(source_binding['path']).get('root_id') == public['root_id'],
         path, stage, 'parent_provenance_source', 'wrong_source_or_root')
    if 'source_judgments' in provenance.get('bindings', {}):
        verify_binding(provenance['bindings']['source_judgments'], path, stage, 'parent_provenance_judgment', slot['parent_review_path'])
    if 'parent_review_path' in provenance:
        verify_binding({'path': provenance['parent_review_path'], 'sha256': provenance.get('parent_review_sha256')},
                       path, stage, 'parent_provenance_judgment', slot['parent_review_path'])
    return bindings


def validate_round(run, rid, judgment=None, before_private_at=None):
    """Read only source, public projection, manifest and saved judgment first."""
    run = Path(run); folder = run / 'rounds' / rid
    path = folder / 'SOURCE_JUDGMENTS.json'; stage = 'round:' + rid
    saved = read(path)
    need(judgment is None or judgment == saved, path, stage, 'record', 'not_saved_or_changed')
    j = validate_disclosures(saved, path, 'round')
    created = timestamp(j.get('created_at_utc'), path, stage, 'created_at_utc')
    limit = timestamp(before_private_at, path, stage, 'private_started_at') if before_private_at is not None else datetime.now(timezone.utc)
    need(created <= limit, path, stage, 'created_at_utc', 'source_judgment_future_timestamp')
    need(isinstance(j.get('reviewer'), str) and j['reviewer'].strip(), path, stage, 'reviewer', 'required')
    source = read(run / 'SOURCE.json'); public = read(folder / 'PUBLIC_ASSETS.json')
    need(j.get('round_id') == rid == public.get('round_id') and j.get('root_id') == source['root_id'] == public['root_id'],
         path, stage, 'identity', 'root_or_round_changed')
    need(j.get('source_sha256') == sha(run / 'SOURCE.json') == public.get('source_sha256')
         and j.get('public_assets_sha256') == sha(folder / 'PUBLIC_ASSETS.json'),
         path, stage, 'binding', 'source_or_public_changed')
    complete_count(j.get('complete_source_turns_read'), len(source['turns']), path, stage)
    boolean(j, 'stem_valid', path, stage)
    turns = {t['turn_ref']: t for t in source['turns']}
    for key, id_key, public_key in (('parents', 'fact_id', 'parent_facts'), ('versions', 'version_id', 'versions')):
        reviewed = index(j.get(key), id_key, path, stage, key)
        expected = index(public.get(public_key), id_key, path, stage, public_key)
        need(set(reviewed) == set(expected), path, stage, key, 'review_coverage')
        for rid_key, row in reviewed.items():
            validate_option_judgment(row, expected[rid_key]['proposition'], turns, path, stage)
    manifest = read(run / 'MANIFEST.json')
    need(manifest.get('root_id') == source['root_id'], path, stage, 'root_id', 'manifest_changed')
    validate_parent_reuse(manifest.get('slot', {}), public, created, path)
    return j


def validate_final_sources(home):
    """All configured groups save their current public judgments before Gold access."""
    home = Path(home); config = read(home / 'CONFIG.json'); public_path = home / 'PUBLIC_QUESTIONS.json'
    need(config.get('review_schema_version') == SCHEMA_VERSION, home / 'CONFIG.json', 'final',
         'review_schema_version', 'unsupported_or_missing')
    groups = config.get('final_review_groups')
    need(isinstance(groups, list) and bool(groups) and len(set(groups)) == len(groups)
         and all(isinstance(g, str) and g.isalnum() for g in groups),
         home / 'CONFIG.json', 'final', 'final_review_groups', 'unique_groups_required')
    expected_paths = {home / 'final_review' / f'group_{g}_SOURCE_JUDGMENTS.json' for g in groups}
    need(set((home / 'final_review').glob('*_SOURCE_JUDGMENTS.json')) == expected_paths,
         home / 'final_review', 'final', 'groups', 'missing_or_unregistered_group')
    public = read(public_path)
    questions = index(public.get('questions'), 'candidate_id', public_path, 'final', 'questions')
    result = {'groups': [], 'questions': questions, 'public_binding': bind(public_path)}
    covered = set()
    slots = config['slots']; roots = {s['root_id']: s for s in slots}
    need(len(roots) == len(slots), home / 'CONFIG.json', 'final', 'slots', 'duplicate_root')
    for group in groups:
        path = home / 'final_review' / f'group_{group}_SOURCE_JUDGMENTS.json'
        original_binding = bind(path)
        j = validate_disclosures(read(path), path, 'final')
        need(j.get('group') == group and isinstance(j.get('reviewer'), str) and j['reviewer'].strip(),
             path, 'final', 'identity', 'group_or_reviewer_required')
        created = timestamp(j.get('created_at_utc'), path, 'final', 'created_at_utc')
        need(created <= datetime.now(timezone.utc), path, 'final', 'created_at_utc', 'future')
        verify_binding(j.get('public_questions'), path, 'final', 'public_questions', public_path)
        judged = index(rows(j, path, 'final'), 'candidate_id', path, 'final', 'questions')
        need(set(judged) <= set(questions) and not (set(judged) & covered),
             path, 'final', 'questions', 'duplicate_or_unregistered_question')
        covered.update(judged)
        group_roots = {q['root_id'] for q in judged.values()}
        need(group_roots <= set(roots), path, 'final', 'root_id', 'unknown_root')
        sources = index(j.get('source'), 'root_id', path, 'final', 'source')
        readings = j.get('current_complete_source_read')
        need(set(sources) == group_roots and isinstance(readings, dict) and set(readings) == group_roots,
             path, 'final', 'source', 'complete_source_coverage')
        lookup = {}
        for root in group_roots:
            sb = sources[root]; verify_binding(sb, path, 'final', 'source')
            source = read(sb['path']); history = read(roots[root]['history_path'])
            need(source['root_id'] == root and source['turns'] == history['turns'],
                 path, 'final', 'source', 'complete_frozen_history_required')
            count = len(source['turns']); complete_count(sb.get('complete_turn_count'), count, path, 'final', 'complete_turn_count')
            boolean(readings[root], 'complete', path, 'final', True)
            need(readings[root].get('turn_positions_inclusive') == [1, count], path, 'final', 'source', 'full_turn_range_required')
            lookup[root] = {t['turn_ref']: t for t in source['turns']}
        for qid, row in judged.items():
            q = questions[qid]; root = q['root_id']
            need(row.get('root_id') == root and row.get('source_sha256') == q['source_sha256'] == sources[root]['sha256']
                 and row.get('question') == q['question'], path, 'final', 'source', 'question_or_source_binding')
            need(sha(q['source_path']) == q['source_sha256'], path, 'final', 'source', 'public_source_changed')
            complete_count(row.get('complete_source_turns_read'), len(lookup[root]), path, 'final')
            boolean(row, 'stem_valid', path, 'final')
            need(q.get('question_type') == 'multi_select' and set(q.get('options', {})) == set('ABCDEF')
                 and isinstance(row.get('options'), dict) and set(row['options']) == set(q['options']),
                 path, 'final', 'options', 'whole_question_coverage')
            for label, option in row['options'].items():
                validate_option_judgment(option, q['options'][label], lookup[root], path, 'final')
        need(bind(path) == original_binding, path, 'final', 'source_judgments', 'bytes_changed_during_validation')
        result['groups'].append({'group': group, 'path': path, 'judgment': j,
                                 'binding': original_binding, 'created': created, 'rows': judged})
    need(covered == set(questions), public_path, 'final', 'questions', 'final_review_must_cover_every_assembled_question')
    return result


def validate_final_reviews(home, pool=None, summary=None):
    """Recheck source records, private crosschecks and any merged summary together."""
    home = Path(home); context = validate_final_sources(home)
    pool_path = home / 'POOL.json'; pool_sha = sha(pool_path)
    if pool is None:
        pool = read(pool_path)
    need(pool == read(pool_path), pool_path, 'final', 'pool', 'pool_not_saved_or_changed')
    private = index(pool.get('questions'), 'candidate_id', pool_path, 'final', 'questions')
    need(set(private) == set(context['questions']), pool_path, 'final', 'questions', 'public_pool_coverage')
    records = [context['public_binding']]; merged = []; audit_times = []
    expected_saved = [{'group': g['group'], **g['binding'],
                       'source_judgments_saved_at_utc': g['judgment']['created_at_utc']} for g in context['groups']]
    expected_audits = {home / 'final_review' / f"group_{g['group']}_AUDIT.json" for g in context['groups']}
    need(set((home / 'final_review').glob('*_AUDIT.json')) == expected_audits,
         home / 'final_review', 'final_audit', 'groups', 'missing_or_unregistered_audit')
    review_files = []
    for g in context['groups']:
        path = home / 'final_review' / f"group_{g['group']}_AUDIT.json"
        original_audit = bind(path); a = read(path); stage = 'final_audit'
        need(a.get('group') == g['group'] and a.get('pool_sha256') == pool_sha
             and a.get('source_judgments_sha256') == g['binding']['sha256'],
             path, stage, 'binding', 'audit_binding_changed')
        for key, expected in (('solver_scores_used', False), ('current_solver_scores_read', False),
                              ('source_first_for_current_children', True), ('child_gold_read_before_save', False),
                              ('current_pool_read_after_both_group_source_judgments_save', True),
                              ('source_judgments_bytes_changed_after_private_read', False), ('training_allowed', False)):
            boolean(a, key, path, stage, expected)
        need(a.get('question_or_gold_edits') == [], path, stage, 'question_or_gold_edits', 'edits_forbidden')
        need(a.get('both_source_judgments_saved_bindings') == expected_saved,
             path, stage, 'both_source_judgments_saved_bindings', 'saved_source_binding_changed')
        private_at = timestamp(a.get('first_private_pool_read_at_utc'), path, stage, 'first_private_pool_read_at_utc')
        completed = timestamp(a.get('created_at_utc'), path, stage, 'created_at_utc')
        need(max(x['created'] for x in context['groups']) <= private_at <= completed <= datetime.now(timezone.utc),
             path, stage, 'created_at_utc', 'source_before_private_required')
        audit_times.append(completed)
        audited = index(a.get('questions'), 'candidate_id', path, stage, 'questions')
        need(set(audited) == set(g['rows']), path, stage, 'questions', 'audit_coverage')
        for qid, row in audited.items():
            q = private[qid]; pub = context['questions'][qid]; j = g['rows'][qid]
            need(all(q.get(k) == pub.get(k) for k in ('root_id', 'question_type', 'question', 'options', 'source_sha256', 'source_path')),
                 path, stage, 'public_questions', 'public_private_changed')
            need(row.get('root_id') == q['root_id'] and row.get('source_sha256') == q['source_sha256'],
                 path, stage, 'source', 'audit_row_binding')
            need(row.get('core_status') in ('CORE_USABLE', 'KNOWN_SUBSTANTIVE_ISSUE'), path, stage, 'core_status', 'invalid')
            if row['core_status'] == 'CORE_USABLE':
                options = j['options']; supported = {k for k, o in options.items() if o['relation'] == 'supports'}
                need(j['stem_valid'] and all(o['relation'] != 'unknown' and o['scope_valid'] and o['fact_distinct'] for o in options.values())
                     and isinstance(q.get('gold'), list) and len(q['gold']) == len(set(q['gold'])) and supported == set(q['gold']),
                     path, stage, 'gold', 'usable_must_match_saved_judgment')
                evidence = q.get('option_evidence', {})
                need(set(evidence) == set(options), path, stage, 'option_evidence', 'coverage')
                source = read(q['source_path']); turns = {t['turn_ref']: t for t in source['turns']}
                for label, e in evidence.items():
                    turn = turns.get(e.get('turn_ref'))
                    need(e.get('relation') == options[label]['relation'] and turn is not None
                         and e.get('speaker') == turn['speaker'] and isinstance(e.get('quote'), str)
                         and bool(e['quote']) and e['quote'] in turn['text'], path, stage, 'option_evidence', 'exact_relation_and_source_required')
            merged.append(row)
        need(bind(path) == original_audit, path, stage, 'audit', 'bytes_changed_during_validation')
        review_files.extend([g['binding'], original_audit])
    if summary is not None:
        path = home / 'FINAL_CONTENT_REVIEW.json'
        need(summary.get('review_schema_version') == SCHEMA_VERSION, path, 'summary',
             'review_schema_version', 'unsupported_or_missing')
        boolean(summary, 'derived_from_saved_final_reviews', path, 'summary', True)
        for key in COMMON_DISCLOSURES + ('child_gold_read_before_save', 'current_pool_read_before_save'):
            value = boolean(summary, key, path, 'summary')
            need(all(g['judgment'][key] is value for g in context['groups']),
                 path, 'summary', key, 'derived_disclosure_changed')
        need(summary.get('pool_sha256') == pool_sha and summary.get('independent_review_files') == review_files
             and summary.get('questions') == merged, path, 'summary', 'bindings', 'merged_review_changed')
        boolean(summary, 'solver_scores_used', path, 'summary', False)
        merged_at = timestamp(summary.get('reviewed_at_utc'), path, 'summary', 'reviewed_at_utc')
        need(max(audit_times) <= merged_at <= datetime.now(timezone.utc),
             path, 'summary', 'reviewed_at_utc', 'audit_before_merge_required')
        need(summary == read(path), path, 'summary', 'record', 'not_saved_or_changed')
    need(sha(pool_path) == pool_sha, pool_path, 'final', 'pool', 'bytes_changed_during_validation')
    for record_binding in records + review_files:
        verify_binding(record_binding, home, 'final', 'review_records')
    records.extend(review_files)
    return {'review_schema_version': SCHEMA_VERSION, 'review_records': records, 'questions': merged}
