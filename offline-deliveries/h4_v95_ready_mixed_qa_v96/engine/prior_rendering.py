"""Lossless grouping of public prior metadata; every original index is retained.

The frozen v79 engine, full source, previous-round feedback and on-disk PRIOR
remain unchanged. This adapter only renders repeated question_type/dedup_action
once per group in multiselect writer contexts. Ordering payloads are untouched.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

VERSION = 'v96_lossless_public_prior_grouping_v1'
FIELDS = {'question_type', 'facts', 'dedup_action'}
EXPLANATION = (
    'For every group, question_type and dedup_action apply to EVERY corresponding '
    'statement_sets entry. Read an entry by its original prior_index from '
    'prior_indices at the same position. used_six_statement_sets.prior_indices '
    'refer to those original indices, never to group positions. Every original '
    'public statement and dedup action is retained; compare across BOTH question '
    'types under the existing dedup rules. The inventory is data, not evidence.'
)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def bind_self():
    path = Path(__file__).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def encode_prior(rows):
    need(isinstance(rows, list), 'prior_rows_not_list')
    groups = []
    positions = {}
    for index, row in enumerate(rows):
        need(isinstance(row, dict) and set(row) == FIELDS,
             'unsupported_prior_projection_fields')
        need(isinstance(row['question_type'], str) and isinstance(row['dedup_action'], str)
             and isinstance(row['facts'], list)
             and all(isinstance(text, str) for text in row['facts']),
             'unsupported_prior_projection_values')
        key = (row['question_type'], row['dedup_action'])
        if key not in positions:
            positions[key] = len(groups)
            groups.append({'question_type': key[0], 'dedup_action': key[1],
                           'prior_indices': [], 'statement_sets': []})
        group = groups[positions[key]]
        group['prior_indices'].append(index)
        group['statement_sets'].append(deepcopy(row['facts']))
    encoded = {'version': VERSION, 'read_by_original_prior_index': EXPLANATION,
               'row_count': len(rows), 'groups': groups}
    need(decode_prior(encoded) == rows, 'public_prior_roundtrip_failed')
    return encoded


def decode_prior(encoded):
    need(isinstance(encoded, dict)
         and set(encoded) == {'version', 'read_by_original_prior_index', 'row_count', 'groups'}
         and encoded['version'] == VERSION
         and encoded['read_by_original_prior_index'] == EXPLANATION,
         'prior_encoding_schema')
    count = encoded['row_count']
    need(type(count) is int and count >= 0 and isinstance(encoded['groups'], list),
         'prior_encoding_count')
    rows = [None] * count
    for group in encoded['groups']:
        need(isinstance(group, dict)
             and set(group) == {'question_type', 'dedup_action', 'prior_indices', 'statement_sets'}
             and isinstance(group['question_type'], str)
             and isinstance(group['dedup_action'], str)
             and isinstance(group['prior_indices'], list)
             and isinstance(group['statement_sets'], list)
             and len(group['prior_indices']) == len(group['statement_sets']),
             'prior_group_schema')
        for index, facts in zip(group['prior_indices'], group['statement_sets']):
            need(type(index) is int and 0 <= index < count and rows[index] is None,
                 'prior_index_duplicate_or_invalid')
            need(isinstance(facts, list) and all(isinstance(text, str) for text in facts),
                 'prior_statement_schema')
            rows[index] = {'question_type': group['question_type'], 'facts': deepcopy(facts),
                           'dedup_action': group['dedup_action']}
    need(all(row is not None for row in rows), 'prior_index_missing')
    return rows


def grouped_context(context):
    need(isinstance(context, dict), 'writer_context_not_dict')
    rows = context.get('previous_candidates_dedup_only')
    if not isinstance(rows, list):
        return context, None
    result = deepcopy(context)
    encoded = encode_prior(rows)
    result['previous_candidates_dedup_only'] = encoded
    need(result.get('source_turns') == context.get('source_turns')
         and result.get('source_metadata') == context.get('source_metadata'),
         'source_changed_by_prior_rendering')
    need(result.get('used_six_statement_sets') == context.get('used_six_statement_sets'),
         'used_prior_indices_changed')
    return result, {'version': VERSION, 'adapter': bind_self(),
                    'original_prior_rows': len(rows), 'rendered_groups': len(encoded['groups']),
                    'decoded_equals_original': True, 'all_original_prior_indices_preserved': True,
                    'cross_type_inventory_preserved': True, 'source_changed': False,
                    'frozen_feedback_changed': False, 'on_disk_prior_changed': False}


def install(assets):
    """Hook the shared payload helper once; contexts without the flat field pass through.

    The dependency hook binds this installed adapter into the asset records.
    No existing feedback receipt field is replaced; the adapter adds one field.
    """
    target = assets.legacy.legacy
    original = target.payload_with_feedback
    installed = getattr(original, '_v96_prior_rendering_version', None)
    if installed is not None:
        need(installed == VERSION and getattr(original, '_v96_prior_rendering_binding', None) == bind_self(),
             'prior_rendering_hook_binding_changed')
        return

    def wrapped(run, root, instructions, context, max_tokens):
        transformed, adapter_receipt = grouped_context(context)
        payload, receipt = original(run, root, instructions, transformed, max_tokens)
        if adapter_receipt is not None:
            need('public_prior_rendering_receipt' not in receipt, 'prior_rendering_receipt_collision')
            receipt = {**receipt, 'public_prior_rendering_receipt': adapter_receipt}
        return payload, receipt

    wrapped._v96_prior_rendering_version = VERSION
    wrapped._v96_prior_rendering_binding = bind_self()
    target.payload_with_feedback = wrapped
    original_dependencies = assets._dependencies

    def bound_dependencies():
        provider, paths = original_dependencies()
        return provider, set(paths) | {Path(__file__).resolve()}

    assets._dependencies = bound_dependencies
