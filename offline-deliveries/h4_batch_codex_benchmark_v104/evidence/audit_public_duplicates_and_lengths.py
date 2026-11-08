#!/usr/bin/env python3
"""Read-only historical public-question scan and offline artifact measurements."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

DELIVERIES = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
BATCH = DELIVERIES / 'h4_batch_codex_benchmark_v104'
EVIDENCE = BATCH / 'evidence'
ROOTS = ['14304', '20432', '24719', '38601']
PRIVATE_KEYS = {'gold', 'answer', 'answers', 'correct_answer', 'correct_answers',
                'correct_option', 'correct_options', 'correct_order', 'is_correct',
                'option_evidence', 'item_evidence', 'answer_reason', 'order_reason',
                'private_gold', 'private_design'}
PATH_EXCLUDE = re.compile(r'(?:private|gold|secret|credential|api[_-]?key|tokenizer)', re.I)
GENERIC_EXCLUDE = re.compile(r'(?:history|feedback|event|memory|ledger|reservation|plan|report|result|audit|summary|receipt|design|review|validation|status|source|generation)', re.I)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def normalize_root(value, path):
    if value is not None:
        match = re.fullmatch(r'(?:cc-train-episode-|root)?(\d+)', str(value).strip())
        if match:
            return 'cc-train-episode-' + match.group(1)
    match = re.search(r'(?:cc-train-episode-|root)(\d+)', str(path), re.I)
    return 'cc-train-episode-' + match.group(1) if match else None


def normalize_type(value):
    value = str(value or '').lower().replace('-', '_').replace(' ', '_')
    if value in {'multi_select', 'multiselect', 'multiple_select', 'multiple_selection'}:
        return 'multi_select'
    if value in {'ordering', 'order', 'order_items', 'sequence', 'ranking'}:
        return 'ordering'
    return None


def normalize_sentence(text):
    return re.sub(r'\s+', '', text.casefold())


def has_private_fields(value):
    if isinstance(value, dict):
        if any(str(key).lower() in PRIVATE_KEYS for key in value):
            return True
        return any(has_private_fields(v) for v in value.values())
    if isinstance(value, list):
        return any(has_private_fields(v) for v in value)
    return False


def extract_questions(data, path):
    """Inspect public projections only; never use any answer-side field."""
    found, excluded = [], collections.Counter()

    def visit(value, pointer='$', inherited_root=None, inherited_type=None):
        if isinstance(value, list):
            for index, row in enumerate(value):
                visit(row, f'{pointer}[{index}]', inherited_root, inherited_type)
            return
        if not isinstance(value, dict):
            return
        root_raw = value.get('root_id', value.get('root', inherited_root))
        type_raw = value.get('question_type', value.get('qtype', inherited_type))
        qtype = normalize_type(type_raw)
        question = value.get('question', value.get('stem', value.get('prompt')))
        fields_present = 'options' in value or 'items' in value
        if fields_present and isinstance(question, str):
            if has_private_fields(value):
                excluded['answer_bearing_question_record'] += 1
                return
            if qtype is None:
                excluded['unsupported_or_missing_question_type'] += 1
                return
            field = 'options' if qtype == 'multi_select' else 'items'
            sentences = value.get(field)
            if isinstance(sentences, dict):
                sentences = list(sentences.values())
            if not isinstance(sentences, list) or len(sentences) != 6:
                excluded['not_exactly_six_sentences'] += 1
                return
            if not all(isinstance(t, str) and t.strip() for t in sentences):
                excluded['non_string_or_empty_sentence'] += 1
                return
            root = normalize_root(root_raw, path)
            if root is None:
                excluded['missing_root_id'] += 1
                return
            norm = sorted(set(normalize_sentence(t) for t in sentences))
            if len(norm) != 6:
                excluded['duplicate_sentence_within_question'] += 1
                return
            key = (root, qtype, tuple(norm))
            found.append({'root_id': root, 'question_type': qtype,
                          'candidate_id': value.get('candidate_id', value.get('question_id')),
                          'file': str(path), 'json_pointer': pointer,
                          'key': key})
            return
        for key, child in value.items():
            if str(key).lower() in PRIVATE_KEYS:
                continue
            if isinstance(child, (dict, list)):
                visit(child, f'{pointer}.{key}', root_raw, type_raw)

    visit(data)
    return found, excluded


def load_objects(path):
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in {'.jsonl', '.ndjson'}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def public_candidate(path):
    rel = path.relative_to(DELIVERIES)
    if BATCH.name in rel.parts or path.suffix.lower() not in {'.json', '.jsonl', '.ndjson'}:
        return False
    if PATH_EXCLUDE.search(str(rel)):
        return False
    if 'public' in str(rel).lower():
        return True
    return bool(re.search(r'(?:question|candidate|pool)', path.stem, re.I)) and not GENERIC_EXCLUDE.search(path.stem)


def descriptor(record):
    return {k: v for k, v in record.items() if k != 'key'}


def duplicate_audit():
    started = utc()
    files = sorted(p for p in DELIVERIES.rglob('*') if p.is_file() and public_candidate(p))
    files.sort(key=lambda p: (p.name.upper() != 'PUBLIC_QUESTIONS.JSON', str(p)))
    historical, file_rows, failures = [], [], []
    all_excluded = collections.Counter()
    for path in files:
        try:
            data = load_objects(path)
            records, excluded = extract_questions(data, path)
            historical.extend(records)
            all_excluded.update(excluded)
            file_rows.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'public_question_occurrences': len(records),
                              'excluded_records': dict(excluded)})
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            failures.append({'file': str(path), 'error_type': type(exc).__name__, 'error': str(exc)[:400]})
    current = []
    current_files = []
    for root in ROOTS:
        path = BATCH / 'roots' / ('root' + root) / 'PUBLIC_QUESTIONS.json'
        records, excluded = extract_questions(load_objects(path), path)
        if excluded:
            raise ValueError(f'Current public extraction rejected records in {path}: {excluded}')
        current.extend(records)
        current_files.append({'file': str(path), 'question_count': len(records),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(current) != 28:
        raise ValueError(f'Expected 28 current public questions, found {len(current)}')
    hist_index, current_index = collections.defaultdict(list), collections.defaultdict(list)
    for row in historical:
        hist_index[row['key']].append(row)
    for row in current:
        current_index[row['key']].append(row)
    cross_matches = [{'current': descriptor(row), 'historical_matches': [descriptor(r) for r in hist_index[row['key']]]}
                     for row in current if row['key'] in hist_index]
    internal = [{'members': [descriptor(row) for row in rows]}
                for rows in current_index.values() if len(rows) > 1]
    result = {
        'schema_version': 'codex_v104_public_exact_duplicate_audit_v1',
        'started_at_utc': started, 'finished_at_utc': utc(),
        'historical_scope': str(DELIVERIES), 'excluded_current_batch': str(BATCH),
        'normalization': {'group_key': ['canonical_root_id', 'question_type', 'set_of_six_sentences'],
                          'sentence': 'Unicode casefold then remove all whitespace; preserve punctuation and wording',
                          'ignored': ['option/item display labels', 'option/item display order', 'question stem'],
                          'root_canonicalization': 'cc-train-episode-N; rootN or bare N accepted'},
        'discovery': {'priority': 'PUBLIC_QUESTIONS.json first',
                      'candidate_rule': 'JSON/JSONL/NDJSON with public anywhere in relative path, or question/candidate/pool filename without history/feedback/event/memory/ledger/reservation/plan/report/result/audit/summary/receipt/design/review/validation/status/source/generation in basename',
                      'excluded_path_regex': PATH_EXCLUDE.pattern,
                      'answer_bearing_records': 'excluded if any nested answer/Gold/evidence key from private_keys is present; answer values are never used',
                      'private_keys': sorted(PRIVATE_KEYS)},
        'historical_candidate_file_count': len(files),
        'historical_successfully_parsed_file_count': len(file_rows),
        'historical_files_with_public_questions': sum(x['public_question_occurrences'] > 0 for x in file_rows),
        'historical_public_question_occurrences': len(historical),
        'historical_unique_normalized_questions': len(hist_index),
        'historical_occurrences_by_type': dict(collections.Counter(r['question_type'] for r in historical)),
        'historical_occurrences_for_current_roots': dict(collections.Counter(r['root_id'] for r in historical if r['root_id'] in {x['root_id'] for x in current})),
        'historical_excluded_record_counts': dict(all_excluded),
        'parse_failures': failures, 'parse_failure_count': len(failures),
        'current_files': current_files, 'current_question_count': len(current),
        'current_by_type': dict(collections.Counter(r['question_type'] for r in current)),
        'current_unique_normalized_questions': len(current_index),
        'current_vs_historical_exact_duplicate_count': len(cross_matches),
        'current_vs_historical_exact_duplicates': cross_matches,
        'current_internal_duplicate_group_count': len(internal),
        'current_internal_duplicate_groups': internal,
        'scanned_files': file_rows,
        'semantic_deduplication_performed': False,
        'limitations': [
            'This is a mechanical exact six-sentence-set comparison, not semantic paraphrase detection. No model or network call was made.',
            'Historical copies in multiple deliveries are counted as occurrences; unique normalized question count removes such repeats.',
            'Only parseable public-shaped six-option/item records with a resolvable root and supported question type are compared. Older ordering questions with fewer than six items, single-choice records, private/answer-bearing pools, malformed records and undiscovered nonstandard filenames are excluded and cannot be covered by this exact check.',
            'No historical semantic uniqueness or full historical public-ordering manual review is claimed. A zero exact-match count does not prove conceptual novelty.',
            'Current family variants are correlated even if their six-sentence sets differ and therefore are not exact duplicates.'
        ]
    }
    (EVIDENCE / 'HISTORICAL_DUPLICATE_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return {key: result[key] for key in ['historical_candidate_file_count', 'historical_successfully_parsed_file_count', 'historical_files_with_public_questions', 'historical_public_question_occurrences', 'historical_unique_normalized_questions', 'parse_failure_count', 'current_question_count', 'current_vs_historical_exact_duplicate_count', 'current_internal_duplicate_group_count']}


def local_tokenizer():
    try:
        import os
        import tiktoken
        import tiktoken.load
        os.environ['TIKTOKEN_CACHE_DIR'] = '/tmp/data-gym-cache'
        def no_download(*args, **kwargs):
            raise RuntimeError('Network/download disabled for artifact measurement')
        tiktoken.load.read_file = no_download
        encoding = tiktoken.get_encoding('o200k_base')
        return encoding, {'available': True, 'library': 'tiktoken', 'encoding': encoding.name,
                          'cache_directory': '/tmp/data-gym-cache', 'download_allowed': False,
                          'network_fallback': 'tiktoken.load.read_file replaced with fail-closed function before loading encoding'}
    except Exception as exc:
        return None, {'available': False, 'download_allowed': False,
                      'failure': type(exc).__name__ + ': ' + str(exc)[:300]}


def length_audit():
    started = utc()
    encoding, tokenizer = local_tokenizer()
    def metrics(text):
        return {'characters': len(text), 'utf8_bytes': len(text.encode('utf-8')),
                'whitespace_words': len(text.split()),
                'tokens': len(encoding.encode(text, disallowed_special=())) if encoding else None}
    rows = []
    for root in ROOTS:
        files = {'baseline': BATCH / 'inputs' / ('root' + root + '_BASELINE.json'),
                 'new': BATCH / 'roots' / ('root' + root) / 'EXPANSION_DRAFT.json',
                 'source': BATCH / 'roots' / ('root' + root) / 'CUMULATIVE_SOURCE.json'}
        parts = {}
        for kind, path in files.items():
            raw = path.read_text(); data = json.loads(raw); turns = data['turns']
            dialogue_text = '\n'.join(t['speaker'] + ': ' + t['text'] for t in turns)
            parts[kind] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'turn_count': len(turns), 'whole_json_artifact': metrics(raw),
                           'dialogue_text': metrics(dialogue_text)}
        rows.append({'root_id': 'cc-train-episode-' + root, 'parts': parts,
                     'turn_count_additive_check': parts['baseline']['turn_count'] + parts['new']['turn_count'] == parts['source']['turn_count']})
    totals = {}
    for kind in ['baseline', 'new', 'source']:
        totals[kind] = {'turn_count': sum(r['parts'][kind]['turn_count'] for r in rows)}
        for mode in ['whole_json_artifact', 'dialogue_text']:
            totals[kind][mode] = {metric: sum(r['parts'][kind][mode][metric] for r in rows)
                                 if all(r['parts'][kind][mode][metric] is not None for r in rows) else None
                                 for metric in ['characters', 'utf8_bytes', 'whitespace_words', 'tokens']}
    result = {'schema_version': 'codex_v104_artifact_lengths_v1',
              'started_at_utc': started, 'finished_at_utc': utc(), 'tokenizer': tokenizer,
              'api_usage': False,
              'definitions': {'baseline': 'The unchanged input baseline JSON and its turns.',
                              'new': 'The single saved expansion-draft JSON, including its metadata; dialogue_text measures its suffix turns only.',
                              'source': 'The full cumulative-source JSON and all baseline-plus-new turns.',
                              'whole_json_artifact': 'Exactly the saved UTF-8 JSON file text including whitespace, metadata, references and origin fields.',
                              'dialogue_text': 'All artifact turns joined by newline as speaker + colon + space + original text; no IDs or JSON metadata.',
                              'whitespace_words': 'Python str.split count; descriptive only, not a linguistic token count.',
                              'totals': 'Sum of independently measured root artifacts, not retokenization of a combined prompt.'},
              'roots': rows, 'totals': totals,
              'limitations': ['These are local content/JSON token measurements, not API input/output/total usage, billable usage or raw model compute.',
                              'The encoding is a locally cached reference encoding and is not claimed to be the hidden actual Codex model tokenizer.',
                              'Author instructions, repeated context reads, all model reasoning, review prompts, tool outputs, rejected candidates and orchestration are excluded from dialogue counts.',
                              'Standalone baseline/new/source text token counts need not be algebraically additive because token boundaries change when content is joined.',
                              'No model call, network access, cache download or key file was used.']}
    (EVIDENCE / 'ARTIFACT_LENGTHS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return {'tokenizer': tokenizer, 'totals': totals}


if __name__ == '__main__':
    for output in ['HISTORICAL_DUPLICATE_AUDIT.json', 'ARTIFACT_LENGTHS.json']:
        if (EVIDENCE / output).exists():
            raise FileExistsError('Refusing to overwrite existing evidence: ' + output)
    print(json.dumps({'duplicates': duplicate_audit(), 'lengths': length_audit()}, ensure_ascii=False))
