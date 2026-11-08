"""Saved source-only ordering judgments and immutable admission artifacts.

This module never asks a model for review and never fills reviewer declarations.
Save ordering/SOURCE_JUDGMENTS.json before finish_review. Its record contains:
review_schema_version='source_judgment_v1', reviewer, created_at_utc, every
DECLARATIONS boolean, public_questions={path,sha256}, sources=[{root_id,path,
sha256,complete_turn_count}], and questions=[...]. Each question repeats the
public identity, source_path/source_sha256, question_type, chronology_kind and
question, plus complete_source_turns_read and the native coherent, unique_order,
neutral_stem_ok, alternative_orders fields. Each I1..I6 item contains proposition
(exact public text), the five native booleans, order_anchor, support_refs and
alternative_matching_anchors. Optional reason fields are retained. Anchors use
the native {turn_ref,quote[,reason]} shape. See the synthetic test fixture.

validate_sources reads only public metadata, complete sources and saved review.
Only after that succeeds may finish_review or verify_review open private files.
The complete Cartesian product of matching occurrences must preserve one strict
order. A private Gold disagreement excludes a question; Gold is never repaired.
This independent review is not represented as Qwen review metadata.
"""
from copy import deepcopy
import json
from pathlib import Path

from common import aware, bind, load_ordering_gate, need, now, read, sha, write


SCHEMA_VERSION = 'source_judgment_v1'
LABELS = tuple('I' + str(n) for n in range(1, 7))
DECLARATIONS = {
    'parent_all_true_provenance_known': False,
    'parent_gold_known_before_review': False,
    'fresh_blind_parent_review': False,
    'current_solver_scores_read': False,
    'solver_scores_used': False,
    'full_source_read': True,
    'new_variant_private_artifacts_read_before_save': False,
    'child_gold_read_before_save': False,
    'current_pool_read_before_save': False,
}
ITEM_BOOLEANS = ('supported', 'actor_ok', 'modality_ok', 'state_ok', 'neutral_stem_ok')
QUESTION_BOOLEANS = ('coherent', 'unique_order', 'neutral_stem_ok')
PRIVATE_KEYS = {'gold', 'original_gold', 'item_evidence', 'order_anchor',
                'order_anchor_spans', 'generated_order_anchor_spans',
                'allowed_matching_anchor_spans', 'alternative_matching_anchors',
                'qwen_review'}
IDENTITY_FIELDS = ('candidate_id', 'root_id', 'question_type', 'chronology_kind',
                   'question', 'source_path', 'source_sha256')


def _need(condition, reason):
    need(condition, 'ordering_review:' + reason)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _boolean(record, key, expected=None):
    _need(key in record and type(record[key]) is bool, key + ':explicit_bool_required')
    if expected is not None:
        _need(record[key] is expected, key + ':wrong_declaration')


def _index(rows, key, description):
    _need(isinstance(rows, list) and all(isinstance(r, dict) and _text(r.get(key)) for r in rows),
          description + ':row_list_required')
    result = {row[key]: row for row in rows}
    _need(len(result) == len(rows), description + ':duplicate_rows')
    return result


def _binding(record, expected_path, description, check_hash=True):
    _need(isinstance(record, dict) and _text(record.get('path')) and _text(record.get('sha256')),
          description + ':binding_required')
    _need(Path(record['path']).resolve() == Path(expected_path).resolve(), description + ':path_changed')
    _need(len(record['sha256']) == 64 and all(c in '0123456789abcdef' for c in record['sha256']),
          description + ':invalid_sha256')
    if check_hash:
        _need(sha(expected_path) == record['sha256'], description + ':sha_changed')
    return {'path': str(Path(expected_path).resolve()), 'sha256': record['sha256']}


def _no_private_fields(value):
    if isinstance(value, dict):
        _need(not (set(value) & PRIVATE_KEYS), 'public_contains_private_fields')
        for nested in value.values():
            _no_private_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _no_private_fields(nested)


def _generation_files(generation, folder):
    files = _index(generation.get('files'), 'path', 'generation.files')
    result = {}
    for row in files.values():
        path = Path(row['path']).resolve()
        _need(path.is_relative_to(folder.resolve()), 'generation_file_outside_ordering')
        key = str(path)
        _need(key not in result, 'generation_duplicate_resolved_path')
        result[key] = _binding(row, path, 'generation_file', check_hash=False)
    return result


def _anchor_schema(anchor):
    _need(isinstance(anchor, dict) and {'turn_ref', 'quote'} <= set(anchor)
          <= {'turn_ref', 'quote', 'reason'}
          and _text(anchor.get('turn_ref')) and _text(anchor.get('quote')), 'anchor_schema')
    if 'reason' in anchor:
        _need(_text(anchor['reason']), 'anchor_reason_required')


def _review_row(row, public, source, gate):
    """Validate native shape, then compute substantive exclusion findings."""
    fields = {*IDENTITY_FIELDS, 'complete_source_turns_read', 'items',
              'alternative_orders', *QUESTION_BOOLEANS}
    _need(fields <= set(row) <= fields | {'reason'}, 'native_question_schema')
    for key in IDENTITY_FIELDS:
        _need(row.get(key) == public.get(key), 'question_public_binding:' + key)
    _need(type(row.get('complete_source_turns_read')) is int
          and row['complete_source_turns_read'] == len(source['turns']), 'complete_source_turns_read')
    for key in QUESTION_BOOLEANS:
        _boolean(row, key)
    _need(isinstance(row.get('items'), dict) and set(row['items']) == set(LABELS), 'review_items_coverage')
    turns = {t['turn_ref']: t for t in source['turns']}
    positions = {t['turn_ref']: n for n, t in enumerate(source['turns'])}
    for label in LABELS:
        item = row['items'][label]
        _need(isinstance(item, dict), label + ':item_object_required')
        fields = {'proposition', 'order_anchor', 'support_refs', 'alternative_matching_anchors', *ITEM_BOOLEANS}
        _need(fields <= set(item) <= fields | {'reason'}, label + ':native_item_schema')
        _need(item['proposition'] == public['items'][label], label + ':public_text_changed')
        for key in ITEM_BOOLEANS:
            _boolean(item, key)
        _anchor_schema(item['order_anchor'])
        refs = item['support_refs']
        _need(isinstance(refs, list) and bool(refs) and all(_text(ref) for ref in refs)
              and len(set(refs)) == len(refs) and set(refs) <= set(turns), label + ':support_refs_source_required')
        _need(isinstance(item['alternative_matching_anchors'], list), label + ':alternatives_list_required')
        for anchor in item['alternative_matching_anchors']:
            _anchor_schema(anchor)
    orders = row.get('alternative_orders')
    _need(isinstance(orders, list), 'alternative_orders_list_required')
    for order in orders:
        _need(isinstance(order, list) and len(order) == 6 and all(isinstance(x, str) for x in order)
              and set(order) == set(LABELS), 'alternative_order_permutation_required')
    reasons, anchors, computed = gate.resolve_ordering_anchors(
        {label: row['items'][label]['order_anchor'] for label in LABELS}, source)
    scope = gate.ordering_match_scope(public, source)
    matching = set(scope['matching_turn_refs'])
    alternatives, outside = {}, {}
    for label in LABELS:
        item = row['items'][label]
        alternatives[label] = [anchors[label]] if label in anchors else []
        outside[label] = []
        if label in anchors and anchors[label]['turn_ref'] not in matching:
            reasons.append(label + ':primary_anchor_outside_public_stem_scope')
        for number, entry in enumerate(item['alternative_matching_anchors']):
            errors, resolved = gate._resolve_anchor(entry, turns, positions)
            reasons.extend(label + ':alternative_' + str(number) + ':' + error for error in errors)
            if resolved is None:
                continue
            if resolved['turn_ref'] not in matching:
                outside[label].append(resolved)
            elif resolved not in alternatives[label]:
                alternatives[label].append(resolved)
        alternatives[label].sort(key=lambda a: (a['turn_index'], a['char_start'], a['char_end']))
        for key in ITEM_BOOLEANS:
            if not item[key]:
                reasons.append(label + ':' + key + '_failed')
    for key in QUESTION_BOOLEANS:
        if not row[key]:
            reasons.append(key + '_failed')
    if computed:
        if any(order != computed for order in orders):
            reasons.append('alternative_order_changes_gold')
        # Adjacent pairs over every combination imply all 6-way Cartesian
        # products retain the same strict, nonoverlapping permutation.
        for before, after in zip(computed, computed[1:]):
            if any(not gate._strictly_before(left, right)
                   for left in alternatives[before] for right in alternatives[after]):
                reasons.append('alternative_anchor_changes_or_ties_order:' + before + ':' + after)
    return {'candidate_id': public['candidate_id'], 'root_id': public['root_id'],
            'reasons': sorted(set(reasons)), 'computed_order': computed,
            'order_anchor_spans': anchors, 'allowed_matching_anchor_spans': alternatives,
            'ordering_match_scope': scope, 'out_of_scope_alternative_anchor_spans': outside}


def validate_sources(home):
    """Fail invalid source records without opening or hashing any private file."""
    home = Path(home).resolve(); folder = home / 'ordering'
    config_path = home / 'CONFIG.json'; config = read(config_path)
    generation_path = folder / 'GENERATION.COMPLETE.json'; generation = read(generation_path)
    generation_binding = bind(generation_path)
    _need(generation.get('status') == 'GENERATED', 'generation_not_complete')
    generated_at = aware(generation.get('completed_at_utc'))
    _need(generated_at <= aware(now()), 'generation_future_timestamp')
    generated_files = _generation_files(generation, folder)
    public_path = folder / 'PUBLIC_QUESTIONS.json'; public = read(public_path)
    _no_private_fields(public)
    questions = _index(public.get('questions'), 'candidate_id', 'public.questions')
    for q in questions.values():
        _need(q.get('question_type') == 'ordering' and q.get('chronology_kind') == 'conversation_order'
              and _text(q.get('question')) and _text(q.get('root_id'))
              and isinstance(q.get('items'), dict) and set(q['items']) == set(LABELS)
              and all(_text(x) for x in q['items'].values()), 'public_ordering_schema')
    public_binding = bind(public_path)
    _need(str(public_path) in generated_files and generated_files[str(public_path)] == public_binding,
          'generation_public_binding')
    slots = _index(config.get('ordering_slots'), 'slot_id', 'config.ordering_slots')
    roots = {s.get('root_id'): s for s in slots.values()}
    _need(len(roots) == len(slots) and all(_text(r) for r in roots), 'config_duplicate_or_missing_root')
    sources, source_bindings = {}, {}
    for root, slot in roots.items():
        _need(slot['slot_id'].isalnum() or all(c.isalnum() or c == '_' for c in slot['slot_id']), 'invalid_slot_id')
        source_path = folder / slot['slot_id'] / 'SOURCE.json'
        _need(str(source_path) in generated_files, 'generation_source_binding_missing')
        binding = _binding(generated_files[str(source_path)], source_path, 'generation_source')
        source = read(source_path)
        errors, _, _ = load_ordering_gate()._source(source)
        _need(not errors and source.get('root_id') == root, 'source_schema_or_root')
        _need(binding['sha256'] == slot.get('source_sha256') == sha(slot['source_path']), 'complete_frozen_source_sha')
        original = read(slot['source_path'])
        _need(source == original, 'complete_frozen_source_changed')
        if 'history_path' in slot:
            history = read(slot['history_path'])
            _need(source['turns'] == history.get('turns'), 'complete_frozen_history_changed')
            if 'history_sha256' in slot:
                _need(sha(slot['history_path']) == slot['history_sha256'], 'history_sha_changed')
        sources[root] = source; source_bindings[root] = binding
    for q in questions.values():
        _need(q['root_id'] in roots, 'public_unknown_root')
        sb = source_bindings[q['root_id']]
        _need(q.get('source_path') == sb['path'] and q.get('source_sha256') == sb['sha256'], 'public_source_binding')
    _need(type(generation.get('candidate_count')) is int and generation['candidate_count'] == len(questions),
          'generation_candidate_count')
    path = folder / 'SOURCE_JUDGMENTS.json'; judgment_binding = bind(path); judgment = read(path)
    _need(isinstance(judgment, dict) and judgment.get('review_schema_version') == SCHEMA_VERSION,
          'review_schema_version')
    for key, expected in DECLARATIONS.items():
        _boolean(judgment, key, expected)
    _need(_text(judgment.get('reviewer')), 'reviewer_required')
    created = aware(judgment.get('created_at_utc'))
    _need(generated_at <= created <= aware(now()), 'judgment_timestamp_before_generation_or_future')
    _binding(judgment.get('public_questions'), public_path, 'judgment_public')
    declared_sources = _index(judgment.get('sources'), 'root_id', 'judgment.sources')
    _need(set(declared_sources) == set(sources), 'judgment_source_coverage')
    for root, entry in declared_sources.items():
        _binding(entry, source_bindings[root]['path'], 'judgment_source')
        _need(type(entry.get('complete_turn_count')) is int
              and entry['complete_turn_count'] == len(sources[root]['turns']), 'complete_turn_count')
    reviewed = _index(judgment.get('questions'), 'candidate_id', 'judgment.questions')
    _need(set(reviewed) == set(questions), 'review_must_cover_every_mechanical_question')
    gate = load_ordering_gate()
    checks = [_review_row(reviewed[qid], q, sources[q['root_id']], gate) for qid, q in questions.items()]
    bindings = [bind(config_path), generation_binding, public_binding,
                *source_bindings.values(), judgment_binding]
    _need(bind(path) == judgment_binding, 'judgment_bytes_changed_during_validation')
    return {'questions': questions, 'sources': sources, 'judgment': judgment,
            'judgment_binding': judgment_binding, 'generation': generation,
            'generation_files': generated_files, 'checks': checks, 'bindings': bindings,
            'created_at_utc': judgment['created_at_utc']}


def _input_bindings(context):
    return list({r['path']: r for r in [*context['bindings'], *context['generation_files'].values()]}.values())


def _private_review(home, context):
    folder = Path(home).resolve() / 'ordering'; pool_path = folder / 'POOL.json'
    # The source validation must finish before even hashing this file.
    private_started = now()
    _need(aware(context['created_at_utc']) <= aware(private_started), 'private_read_before_saved_source_judgment')
    expected = context['generation_files'].get(str(pool_path))
    _binding(expected, pool_path, 'generation_private_pool')
    pool = read(pool_path)
    private = _index(pool.get('questions'), 'candidate_id', 'private.questions')
    _need(set(private) == set(context['questions']), 'public_private_coverage')
    for record in context['generation_files'].values():
        _binding(record, record['path'], 'generation_artifact')
    checks, accepted = [], []
    import ordering_generation
    import ordering_plan
    plans_by_root, answers_by_root, plan_bindings = {}, {}, {}
    for root, source in context['sources'].items():
        run = Path(next(q['source_path'] for q in context['questions'].values() if q['root_id'] == root)).parent if any(q['root_id'] == root for q in context['questions'].values()) else None
        if run is None:
            continue
        manifest = read(run / 'MANIFEST.json')
        answers_by_root[root] = ordering_generation.verify_gold_plan(run, manifest, source)
        plan_bindings[root] = manifest['gold_plan']
        plans_by_root[root] = {q['candidate_id']: q for q in read(run / 'ANCHOR_PLAN.json')['questions']}
    for source_check in context['checks']:
        qid = source_check['candidate_id']; q = private[qid]; public = context['questions'][qid]
        _need(all(q.get(k) == public.get(k) for k in IDENTITY_FIELDS) and q.get('items') == public['items'],
              'private_public_text_or_binding_changed')
        check = deepcopy(source_check)
        gold = q.get('gold')
        plan = plans_by_root[q['root_id']][qid]
        program = answers_by_root[q['root_id']][qid]
        _need(gold == program['gold'] and q.get('gold_origin') == ordering_plan.GOLD_ORIGIN
              and q.get('program_gold_plan') == plan_bindings[q['root_id']],
              'private_program_gold_changed')
        fixed_positions = program['anchor_positions']
        for label in LABELS:
            _need(all(q['item_evidence'][label].get(k) == plan['anchors'][label][k]
                      for k in ('turn_ref', 'speaker', 'quote')), 'private_fixed_anchor_changed')
            independent = check['order_anchor_spans'].get(label)
            fixed = fixed_positions[label]
            if independent is not None and not (
                    independent['turn_ref'] == fixed['turn_ref']
                    and fixed['quote_start'] <= independent['char_start']
                    and independent['char_end'] <= fixed['quote_end']):
                check['reasons'].append(label + ':fixed_anchor_disagrees_with_independent_semantic_anchor')
        if not (isinstance(gold, list) and len(gold) == 6 and all(isinstance(x, str) for x in gold)
                and set(gold) == set(LABELS)):
            check['reasons'].append('private_gold_not_complete_permutation')
        elif check['computed_order'] != gold:
            check['reasons'].append('private_gold_disagrees_with_independent_source_order')
        check['reasons'] = sorted(set(check['reasons']))
        check['status'] = 'EXCLUDED' if check['reasons'] else 'USABLE'
        checks.append(check)
        if check['reasons']:
            continue
        row = deepcopy(q)
        _need('qwen_review' not in row, 'private_qwen_review_metadata_forbidden')
        row['order_anchor_spans'] = deepcopy(check['order_anchor_spans'])
        row['allowed_matching_anchor_spans'] = deepcopy(check['allowed_matching_anchor_spans'])
        row['ordering_match_scope'] = deepcopy(check['ordering_match_scope'])
        row['ordering_anchor_contract'] = 'exact_clause_span_v1'
        row['independent_source_review'] = {
            'status': 'PASS', 'review_schema_version': SCHEMA_VERSION,
            'reviewer': context['judgment']['reviewer'], 'candidate_id': qid,
            'source_judgments': context['judgment_binding'],
            'source_judgments_saved_at_utc': context['created_at_utc'],
            'source_sha256': public['source_sha256'],
            'public_questions_sha256': context['judgment']['public_questions']['sha256'],
        }
        row['status'] = 'INDEPENDENT_SOURCE_REVIEWED_READY_FOR_SOLVER'
        accepted.append(row)
    # Freeze the exact files used, including private generation artifacts.
    files = _input_bindings(context)
    for record in files:
        _binding(record, record['path'], 'review_input')
    return {'checks': checks, 'accepted': accepted, 'files': files,
            'private_started_at_utc': private_started, 'private_pool': bind(pool_path)}


def finish_review(home):
    """Create an append-only content review and pool after source-only validation."""
    folder = Path(home).resolve() / 'ordering'
    output_paths = [folder / name for name in ('CONTENT_REVIEW.json', 'EVALUATION_POOL.json', 'CONTENT_REVIEW.COMPLETE.json')]
    _need(not any(p.exists() for p in output_paths), 'review_outputs_already_exist')
    context = validate_sources(home)
    result = _private_review(home, context)
    completed = now()
    report = {
        'version': 'v76_ordering_content_review_v1', 'status': 'CONTENT_REVIEWED',
        'review_schema_version': SCHEMA_VERSION, 'created_at_utc': completed,
        'source_judgments_saved_at_utc': context['created_at_utc'],
        'first_private_pool_read_at_utc': result['private_started_at_utc'],
        'source_judgments': context['judgment_binding'], 'private_pool': result['private_pool'],
        'files': result['files'], 'questions': result['checks'],
        'accepted_count': len(result['accepted']),
        'rejected_count': len(result['checks']) - len(result['accepted']),
        'question_or_gold_edits': [],
    }
    evaluation = {'version': 'v76_ordering_evaluation_pool_v1', 'questions': result['accepted']}
    write(output_paths[0], report); write(output_paths[1], evaluation)
    write(output_paths[2], {'version': 'v76_ordering_content_review_complete_v1',
          'status': 'CONTENT_REVIEW_FROZEN', 'completed_at_utc': now(),
          'files': [bind(output_paths[0]), bind(output_paths[1])],
          'inputs': result['files']})
    return report


def verify_review(home):
    """Replay every saved source/private judgment and all hashes before reuse."""
    folder = Path(home).resolve() / 'ordering'
    context = validate_sources(home)
    report_path = folder / 'CONTENT_REVIEW.json'; evaluation_path = folder / 'EVALUATION_POOL.json'
    complete_path = folder / 'CONTENT_REVIEW.COMPLETE.json'
    complete = read(complete_path); report = read(report_path)
    _need(complete.get('status') == 'CONTENT_REVIEW_FROZEN', 'content_review_not_frozen')
    frozen_outputs = _index(complete.get('files'), 'path', 'complete.files')
    _need(set(frozen_outputs) == {str(report_path), str(evaluation_path)}, 'complete_output_coverage')
    # Compare saved source/input identities and time ordering before any frozen
    # private file is read or hashed. Private hashes themselves are checked below.
    _binding(frozen_outputs[str(report_path)], report_path, 'frozen_review_output')
    _need(complete.get('inputs') == _input_bindings(context) == report.get('files'), 'frozen_input_bindings_changed')
    _need(report.get('status') == 'CONTENT_REVIEWED' and report.get('review_schema_version') == SCHEMA_VERSION,
          'content_review_schema')
    _need(report.get('source_judgments') == context['judgment_binding']
          and report.get('private_pool') == context['generation_files'].get(str(folder / 'POOL.json')),
          'report_input_binding')
    _need(report.get('source_judgments_saved_at_utc') == context['created_at_utc'], 'report_judgment_time_changed')
    saved = aware(context['created_at_utc']); private_at = aware(report.get('first_private_pool_read_at_utc'))
    completed = aware(report.get('created_at_utc')); frozen = aware(complete.get('completed_at_utc'))
    _need(saved <= private_at <= completed <= frozen <= aware(now()), 'review_time_order')
    result = _private_review(home, context)
    _binding(frozen_outputs[str(evaluation_path)], evaluation_path, 'frozen_review_output')
    evaluation = read(evaluation_path)
    _need(report.get('questions') == result['checks']
          and report.get('accepted_count') == len(result['accepted'])
          and report.get('rejected_count') == len(result['checks']) - len(result['accepted'])
          and report.get('question_or_gold_edits') == [], 'content_review_replay_mismatch')
    _need(evaluation == {'version': 'v76_ordering_evaluation_pool_v1', 'questions': result['accepted']},
          'evaluation_pool_replay_mismatch')
    return report
