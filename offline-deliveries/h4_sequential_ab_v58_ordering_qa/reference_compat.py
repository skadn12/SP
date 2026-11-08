"""Narrow, replayable reference spelling compatibility; never repair evidence."""
from copy import deepcopy


def normalize_entry(entry, root, source):
    original = deepcopy(entry)
    if not isinstance(entry, dict):
        return original, []
    ref, quote = entry.get('turn_ref'), entry.get('quote')
    turns = source.get('turns', [])
    if not isinstance(ref, str) or any(t.get('turn_ref') == ref for t in turns):
        return original, []
    prefix = root + ':'
    if not ref.startswith(prefix):
        return original, []
    target = ref[len(prefix):]
    matches = [t for t in turns if t.get('turn_ref') == target]
    if len(matches) != 1 or not isinstance(quote, str) or not quote:
        return original, []
    turn = matches[0]
    if 'speaker' in entry and entry['speaker'] != turn.get('speaker'):
        return original, []
    text = turn.get('text', '')
    start = text.find(quote)
    if start < 0 or text.find(quote, start + 1) >= 0:
        return original, []
    original['turn_ref'] = target
    return original, [{'kind': 'CURRENT_ROOT_PREFIX_REMOVED', 'original_turn_ref': ref,
        'resolved_turn_ref': target, 'exact_unique_quote': True,
        'semantic_fields_changed': False, 'raw_response_unchanged': True}]


def normalize_question(row, root, source):
    result, records = deepcopy(row), []
    if not isinstance(row, dict):
        return result, records
    if isinstance(row.get('options'), list):
        for index, entry in enumerate(row['options']):
            fixed, changes = normalize_entry(entry, root, source)
            result['options'][index] = fixed
            records += [{**change, 'field': f'options[{index}].turn_ref'} for change in changes]
    if isinstance(row.get('item_evidence'), dict):
        for label, entry in row['item_evidence'].items():
            fixed, changes = normalize_entry(entry, root, source)
            result['item_evidence'][label] = fixed
            records += [{**change, 'field': f'item_evidence.{label}.turn_ref'} for change in changes]
    return result, records
