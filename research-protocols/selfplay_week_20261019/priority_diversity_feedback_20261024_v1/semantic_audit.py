#!/usr/bin/env python3
"""CPU-only preparation/validation of independent FULL-response semantic reviews.

No Solver calls, admission changes, question rewrites, or semantic auto-labels.
Run prepare after solver_runner.py analyze; fill the two review templates through
independent source/response reading; run assemble only after both are complete.
Every output uses exclusive creation. Historical or current files are never edited.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STATUS = {'PASS', 'FAIL', 'UNKNOWN', 'NA'}
PREDICTED = {'OBSERVED', 'NOT_OBSERVED', 'UNASSESSABLE'}
DIMENSIONS = {'CONTENT_OR_GOLD', 'SOURCE_STATE', 'ENTITY_BINDING', 'QUANTITY_OR_TIME',
 'RELATION_OR_CONSTRAINT', 'CITATION_UNSUPPORTED', 'REASON_UNSUPPORTED',
 'ITEM_FINAL_MISMATCH', 'SUMMARY_FINAL_MISMATCH', 'ORDER_RELATION',
 'INCOMPLETE_OUTPUT', 'FORMAT_OR_LIMIT', 'SELECTION_OR_OMISSION'}
ITEM_FIELDS = ('judgment_correct', 'citation_support', 'reason_support')
RESPONSE_FIELDS = ('answer_reason_support', 'summary_final_consistency', 'item_final_consistency')
DEFAULT_REVIEWERS = {'a': '/root', 'b': '/root/followup_eval_runner_audit'}


def need(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def check(bound):
    need(isinstance(bound, dict) and binding(bound['path']) == bound,
         'BOUND_FILE_DRIFT:' + str(bound.get('path')))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def by_id(rows, key, label):
    result = {r[key]: r for r in rows}
    need(len(result) == len(rows), 'DUPLICATE_' + label)
    return result


def decision(value):
    return value.get('decision') if isinstance(value, dict) else value


def status_note(value, label):
    need(isinstance(value, dict) and decision(value) in STATUS
         and nonempty(value.get('note', value.get('rationale'))),
         'EXPLICIT_STATUS_AND_NOTE_REQUIRED:' + label)


def source_data(bound):
    check(bound)
    source = read(bound['path'])
    turns = source['turns']
    refs = [t['turn_ref'] for t in turns]
    need(len(refs) == len(set(refs)) and all(nonempty(r) for r in refs), 'SOURCE_REF_DRIFT')
    return source, {t['turn_ref']: t for t in turns}


def actual_item(parsed, label):
    if not isinstance(parsed, dict):
        return None
    return parsed.get('item_evidence', {}).get(label)


def review_contract():
    return {
      'item_fields': list(ITEM_FIELDS), 'response_fields': list(RESPONSE_FIELDS),
      'statuses': sorted(STATUS), 'error_dimensions': sorted(DIMENSIONS),
      'predicted_error_statuses': sorted(PREDICTED),
      'rules': [
        'Read each complete bound source and actual raw response; no semantic inference from mechanical scores alone.',
        'Judge displayed letters against displayed statements and their explicit canonical mapping.',
        'Citations must support the submitted judgment and reason using the actual cited turns, not merely nearby or topic-related turns.',
        'Assess source truth of a reason separately from whether its submitted citations are adequate.',
        'For ordering, item-content support and source-required pair/order support are distinct.',
        'Answer correctness does not imply supported reasons; wrong answers do not automatically diagnose the targeted card.',
        'Correct arithmetic plus a missing final option is selection/consistency failure, not an observed arithmetic defect.',
        'Final ordering inversions may establish observable order failure, never an internal cause.',
        'Invalid output and post-freeze content concerns require predicted_error_assessment=UNASSESSABLE.',
        'Retain post-freeze content concerns and original scores; do not rewrite questions, alter admission or silently drop requests.',
        'Visible reasons are outputs, not faithful internal chain-of-thought. No formal weakness or causal training-gain claim.',
        'Both displays are reviewed but are dependent observations of one question; count questions and roots separately.',
      ]}


def prepare(args):
    out = args.output_dir.resolve()
    need(not out.exists(), 'NO_PREPARATION_OVERWRITE')
    execution = args.execution_dir.resolve()
    protocol = read(HERE / 'PROTOCOL_FREEZE.json')
    rules = read(HERE / 'FEEDBACK_OUTCOME_RULES.json')
    selection = read(HERE / 'SELECTION_FREEZE.json')
    content_freeze = read(HERE / 'CONTENT_FREEZE.json')
    need(rules['frozen_before_current_solver_calls'] is True, 'FEEDBACK_RULES_NOT_PROSPECTIVE')
    need(set(rules['predicted_error_statuses']) == PREDICTED, 'PREDICTION_RULE_DRIFT')
    for bound in content_freeze['bindings']:
        check(bound)
    selected = by_id(selection['roots'], 'root_id', 'SELECTED_ROOT')
    need(all(r['split'] == 'train' for r in selected.values()), 'TRAIN_ONLY_REQUIRED')
    all_candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
    content_reviews = by_id(read(HERE / 'CONTENT_REVIEW_AI.json')['reviews'], 'candidate_id', 'CONTENT_REVIEW')
    admitted = [c for c in all_candidates if content_reviews[c['candidate_id']]['decision'] == 'PASS']
    candidates = by_id(admitted, 'candidate_id', 'CANDIDATE')
    need(len(candidates) == content_freeze['admitted'] and 0 < len(candidates) <= protocol['planned_slots'], 'ADMISSION_COUNT_DRIFT')
    scores_doc = read(execution / 'SCORES.json')
    packet_doc = read(execution / 'REASON_REVIEW_PACKET.json')
    raw_freeze = read(execution / 'RAW_RESPONSE_FREEZE.json')
    raw_bound = binding(execution / 'RAW_RESPONSE_FREEZE.json')
    need(scores_doc['raw_freeze_binding'] == packet_doc['raw_freeze_binding'] == raw_bound, 'RAW_FREEZE_BINDING_DRIFT')
    for bound in raw_freeze['bindings'] + raw_freeze['validation_bindings'] + raw_freeze.get('submission_bindings', []):
        check(bound)
    check(scores_doc['private_scoring_binding'])
    private_doc = read(scores_doc['private_scoring_binding']['path'])
    private = by_id(private_doc['requests'], 'request_id', 'PRIVATE_REQUEST')
    scores = by_id(scores_doc['records'], 'request_id', 'SCORE')
    records = by_id(packet_doc['requests'], 'request_id', 'RESPONSE_RECORD')
    expected_ids = set(raw_freeze['request_ids'])
    need(set(scores) == set(records) == set(private) == expected_ids, 'ALL_REQUEST_COVERAGE_DRIFT')
    need(len(records) == 4 * len(candidates) <= protocol['maximum_solver_requests'], 'ALL_REQUEST_FACTOR_DRIFT')
    need({r['candidate_id'] for r in records.values()} == set(candidates), 'REQUEST_CANDIDATE_DRIFT')
    groups = {g: sorted(r for r, s in selected.items() if s['author_group'] == g) for g in ('a', 'b')}
    reviewers = {'a': args.reviewer_a, 'b': args.reviewer_b}
    need(all(nonempty(v) for v in reviewers.values()) and len(set(reviewers.values())) == 2, 'TWO_INDEPENDENT_REVIEWER_IDENTITIES_REQUIRED')
    sources, refs_by_root, transcripts = [], {}, {}
    for root in sorted({c['root_id'] for c in candidates.values()}):
        entry = selected[root]
        bound = {k: entry[k] for k in ('path', 'sha256', 'bytes')}
        source, refs = source_data(bound)
        need(all(c['source_binding'] == bound for c in candidates.values() if c['root_id'] == root), 'CANDIDATE_SOURCE_BINDING_DRIFT')
        need(source['root_id'] == root, 'SOURCE_ROOT_DRIFT')
        refs_by_root[root] = refs
        text = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns']) + '\n'
        transcript_path = out / 'sources' / (root + '.txt')
        raw = text.encode()
        transcript_bound = {'path': str(transcript_path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        sources.append({'root_id': root, 'source_binding': bound, 'transcript_binding': transcript_bound,
                        'turns': len(source['turns']), 'turn_refs': list(refs)})
        transcripts[transcript_path] = text
    prepared = {}
    for group in ('a', 'b'):
        questions, reviews = [], []
        root_group = [r for r in groups[group] if r in refs_by_root]
        for cid, c in candidates.items():
            if c['root_id'] not in root_group:
                continue
            need(reviewers[group] != c['author_id'], 'AUTHOR_CANNOT_RESPONSE_REVIEW_SELF')
            responses, review_responses = [], []
            all_c = [r for r in records.values() if r['candidate_id'] == cid]
            need(len(all_c) == 4 and {(r['condition'], r['request']['variant_id']) for r in all_c}
                 == {(co, va) for co in ('FULL', 'QUESTION_ONLY') for va in ('perm_00', 'perm_01')}, 'DISPLAY_CONDITION_COVERAGE_DRIFT:' + cid)
            for record in sorted((r for r in all_c if r['condition'] == 'FULL'), key=lambda r: r['request']['variant_id']):
                rid = record['request_id']; raw_b = record['raw_binding']; check(raw_b)
                need(raw_b in raw_freeze['bindings'], 'RESPONSE_NOT_IN_RAW_FREEZE:' + rid)
                raw = read(raw_b['path'])
                need(raw['request_id'] == rid and scores[rid]['raw_binding'] == raw_b, 'RAW_RESPONSE_ID_OR_SCORE_DRIFT')
                display = record['request']['display']; mapping = private[rid]['presentation']['display_to_canonical']
                choices = c['items' if c['question_type'] == 'ordering' else 'options']
                need(set(mapping) == set(display) and set(mapping.values()) == set(choices), 'DISPLAY_BIJECTION_DRIFT')
                need(all(display[d] == choices[k] for d, k in mapping.items()), 'DISPLAY_TEXT_DRIFT')
                need(private[rid]['presentation']['canonical_gold'] == c['gold'], 'PRIVATE_GOLD_DRIFT')
                parsed = record.get('parsed')
                response = {'request_id': rid, 'variant_id': record['request']['variant_id'],
                  'display': display, 'display_to_canonical': mapping, 'raw_binding': raw_b,
                  'raw_output': raw.get('raw_output'), 'raw_finish_reason': raw.get('finish_reason'),
                  'status': record['status'], 'parsed': parsed,
                  'invalid_error': record.get('error', record.get('invalid_error')),
                  'mechanical_consistency': record.get('mechanical_consistency'), 'score': scores[rid]}
                responses.append(response)
                items = {}
                for label, statement in display.items():
                    actual = actual_item(parsed, label)
                    actual_refs = actual.get('turn_refs', []) if isinstance(actual, dict) else []
                    items[label] = {'canonical_label': mapping[label], 'display_statement': statement,
                      'actual_item_output': actual,
                      'actual_citation_text': [refs_by_root[c['root_id']][ref] for ref in actual_refs if ref in refs_by_root[c['root_id']]],
                      'unknown_actual_turn_refs': [ref for ref in actual_refs if ref not in refs_by_root[c['root_id']]],
                      **{field: {'decision': 'UNKNOWN', 'note': ''} for field in ITEM_FIELDS},
                      'reviewer_turn_refs': [], 'note': ''}
                review_responses.append({'request_id': rid, 'raw_binding': raw_b,
                  'review_completed': False, 'items': items,
                  **{field: {'decision': 'UNKNOWN', 'note': ''} for field in RESPONSE_FIELDS},
                  'predicted_error_assessment': {'card_id': c['feedback_target']['card_id'],
                    'decision': 'UNASSESSABLE', 'affected_display_items': [], 'reviewer_turn_refs': [],
                    'note': '', 'competing_explanation_review': ''},
                  'error_dimensions': [], 'notes': ''})
            questions.append({'candidate': c, 'frozen_content_review': content_reviews[cid], 'responses': responses})
            reviews.append({'candidate_id': cid, 'candidate_sha256': fingerprint(c), 'root_id': c['root_id'],
              'question_type': c['question_type'], 'task_operation': c['task_operation'],
              'content_decision': 'HOLD', 'content_notes': '', 'reconstructed_gold': [],
              'gold_reconstruction_evidence': [], 'responses': review_responses})
        packet = {'schema_version': 'priority_full_review_packet_v1', 'group': group, 'roots': root_group,
          'reviewer_id': reviewers[group], 'sources': [s for s in sources if s['root_id'] in root_group],
          'review_contract': review_contract(), 'feedback_outcome_rules': rules, 'questions': questions}
        template = {'schema_version': 'priority_full_source_visible_reason_semantic_review_v1',
          'reviewer_id': reviewers[group], 'gold_visible': True, 'human': False,
          'reviewer_type': 'independent_codex_ai_agent_gold_visible_not_human',
          'source_read_receipt': [{**s, 'turns_read': 0, 'complete': False, 'turn_refs_read': []}
                                  for s in sources if s['root_id'] in root_group],
          'new_solver_calls': 0, 'reviews': reviews}
        prepared[group] = packet, template
    out.mkdir()
    for path, text in transcripts.items():
        path.parent.mkdir(exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    for group, (packet, template) in prepared.items():
        save(out / f'REVIEW_PACKET_{group.upper()}.json', packet)
        save(out / f'review_{group}/SEMANTIC_REVIEW_TEMPLATE.json', template)
    paths = [HERE / n for n in ('PROTOCOL_FREEZE.json', 'SELECTION_FREEZE.json', 'FEEDBACK_INPUT_FREEZE.json',
             'FEEDBACK_OUTCOME_RULES.json', 'CONTENT_FREEZE.json', 'CANDIDATES_PRIVATE.json',
             'CONTENT_REVIEW_AI.json', 'SLOT_LEDGER.json')]
    paths += [execution / n for n in ('SCORES.json', 'REASON_REVIEW_PACKET.json', 'RAW_RESPONSE_FREEZE.json', 'PRIVATE_SCORING.json')]
    paths += [Path(__file__)] + [out / f'REVIEW_PACKET_{g.upper()}.json' for g in ('a', 'b')]
    initial_paths = [HERE / f'review_{g}/INITIAL_CONTENT_REVIEW_AI.json' for g in ('a', 'b')]
    paths += [p for p in initial_paths if p.exists()]
    save(out / 'REVIEW_FREEZE.json', {'schema_version': 'priority_full_semantic_review_freeze_v1',
      'created_at_utc': datetime.now(timezone.utc).isoformat(), 'execution_dir': str(execution),
      'questions': len(candidates), 'responses': 2 * len(candidates), 'roots': len(sources),
      'groups': groups, 'reviewers': reviewers, 'sources': sources,
      'selection': 'Every admitted question and both FULL displays, including invalid and exact responses.',
      'selection_uses_answer_correctness': False, 'bindings': [binding(p) for p in paths],
      'review_contract': review_contract(), 'new_solver_calls': 0,
      'semantic_approval_is_not_mechanical_proof': True, 'formal_weakness_admission': False,
      'causal_feedback_gain_proven': False, 'training_admission': False})
    print(json.dumps({'prepared_questions': len(candidates), 'full_responses': 2 * len(candidates), 'output_dir': str(out)}))


def validate_group(packet, review, freeze):
    group = packet['group']; reviewer = review['reviewer_id']
    need(reviewer == freeze['reviewers'][group] == packet['reviewer_id'], 'REVIEWER_ID_DRIFT')
    need(review.get('gold_visible') is True and review.get('human') is False and review.get('new_solver_calls') == 0,
         'REVIEW_SCOPE_DISCLOSURE_REQUIRED')
    receipts = by_id(review['source_read_receipt'], 'root_id', 'SOURCE_RECEIPT')
    sources = {s['root_id']: s for s in packet['sources']}
    need(set(receipts) == set(sources), 'SOURCE_RECEIPT_COVERAGE')
    source_refs = {}
    for root, receipt in receipts.items():
        source = sources[root]
        for key in ('source_binding', 'transcript_binding'):
            check(source[key]); need(receipt[key] == source[key], 'SOURCE_RECEIPT_BINDING_DRIFT')
        need(receipt.get('complete') is True and receipt.get('turns_read') == source['turns']
             and receipt.get('turn_refs_read') == source['turn_refs'], 'FULL_SOURCE_READ_RECEIPT_REQUIRED:' + root)
        source_refs[root] = set(source['turn_refs'])
    expected = {q['candidate']['candidate_id']: q for q in packet['questions']}
    reviews = by_id(review['reviews'], 'candidate_id', 'QUESTION_REVIEW')
    need(set(reviews) == set(expected), 'QUESTION_REVIEW_COVERAGE_DRIFT')
    flat, concerns = [], []
    for cid, r in reviews.items():
        q = expected[cid]; c = q['candidate']; refs = source_refs[c['root_id']]
        need(c['author_id'] != reviewer and r.get('candidate_sha256') == fingerprint(c)
             and r['root_id'] == c['root_id'], 'REVIEW_AUTHOR_HASH_OR_ROOT_DRIFT:' + cid)
        need(r['content_decision'] in {'PASS', 'HOLD', 'FAIL'} and nonempty(r['content_notes']), 'CONTENT_REVIEW_REQUIRED:' + cid)
        reconstructed = r.get('reconstructed_gold')
        need(isinstance(reconstructed, list) and len(set(reconstructed)) == len(reconstructed), 'GOLD_RECONSTRUCTION_REQUIRED')
        if r['content_decision'] == 'PASS':
            need((set(reconstructed) == set(c['gold']) if c['question_type'] == 'multi_select'
                  else reconstructed == c['gold']), 'RECONSTRUCTED_GOLD_MISMATCH:' + cid)
        evidence = r.get('gold_reconstruction_evidence')
        need(isinstance(evidence, list) and evidence and all(isinstance(e, dict) and e.get('refs')
             and set(e['refs']) <= refs and nonempty(e.get('note')) for e in evidence), 'GOLD_EVIDENCE_REQUIRED:' + cid)
        if r['content_decision'] != 'PASS':
            concerns.append({'candidate_id': cid, 'root_id': c['root_id'], 'decision': r['content_decision'],
              'notes': r['content_notes'], 'reconstructed_gold': reconstructed,
              'frozen_original_gold': c['gold'], 'original_admission_unchanged': True,
              'original_scores_retained': True, 'training_pool_hold': True})
        responses = by_id(r['responses'], 'request_id', 'RESPONSE_REVIEW')
        expected_responses = {x['request_id']: x for x in q['responses']}
        need(set(responses) == set(expected_responses) and len(responses) == 2, 'FULL_PAIR_REVIEW_COVERAGE:' + cid)
        for rid, response in responses.items():
            inp = expected_responses[rid]; check(inp['raw_binding'])
            need(response.get('review_completed') is True and response.get('raw_binding') == inp['raw_binding'], 'EXPLICIT_COMPLETE_RAW_BOUND_REVIEW_REQUIRED:' + rid)
            need(set(response['items']) == set(inp['display']), 'DISPLAY_ITEM_COVERAGE:' + rid)
            for label, item in response['items'].items():
                need(item.get('canonical_label') == inp['display_to_canonical'][label]
                     and item.get('display_statement') == inp['display'][label], 'ITEM_LABEL_MAPPING_DRIFT:' + rid)
                need(item.get('actual_item_output') == actual_item(inp.get('parsed'), label), 'ACTUAL_ITEM_OUTPUT_EDITED:' + rid)
                for field in ITEM_FIELDS:
                    status_note(item[field], rid + ':' + label + ':' + field)
                if inp['score']['valid']:
                    need(item.get('reviewer_turn_refs'), 'VALID_ITEM_NEEDS_REVIEWER_SOURCE_EVIDENCE:' + rid)
                need(nonempty(item.get('note')) and isinstance(item.get('reviewer_turn_refs'), list)
                     and set(item['reviewer_turn_refs']) <= refs, 'ITEM_REVIEW_NOTE_OR_REFS_REQUIRED:' + rid)
            for field in RESPONSE_FIELDS:
                status_note(response[field], rid + ':' + field)
            need(nonempty(response.get('notes')) and isinstance(response.get('error_dimensions'), list)
                 and set(response['error_dimensions']) <= DIMENSIONS, 'RESPONSE_NOTES_OR_DIMENSIONS_REQUIRED:' + rid)
            pa = response.get('predicted_error_assessment', {})
            need(pa.get('card_id') == c['feedback_target']['card_id'] and pa.get('decision') in PREDICTED,
                 'TARGET_CARD_AND_PREDICTED_ERROR_STATUS_REQUIRED:' + rid)
            need(nonempty(pa.get('note')) and nonempty(pa.get('competing_explanation_review')),
                 'PREDICTION_AND_COMPETING_EXPLANATION_NOTES_REQUIRED:' + rid)
            need(isinstance(pa.get('affected_display_items'), list)
                 and set(pa['affected_display_items']) <= set(inp['display'])
                 and isinstance(pa.get('reviewer_turn_refs'), list) and set(pa['reviewer_turn_refs']) <= refs,
                 'PREDICTION_EVIDENCE_REFERENCES_INVALID:' + rid)
            assessable = inp['score']['valid'] and r['content_decision'] == 'PASS'
            need(assessable or pa['decision'] == 'UNASSESSABLE', 'INVALID_OR_CONTENT_CONCERN_MUST_BE_UNASSESSABLE:' + rid)
            if pa['decision'] == 'OBSERVED':
                need(pa['affected_display_items'] and pa['reviewer_turn_refs'], 'OBSERVED_TARGET_ERROR_REQUIRES_SOURCE_AND_DISPLAY_EVIDENCE:' + rid)
            if not inp['score']['valid']:
                need(all(decision(item[f]) in {'UNKNOWN', 'NA'} for item in response['items'].values() for f in ITEM_FIELDS),
                     'INVALID_RESPONSE_MUST_NOT_RECEIVE_ITEM_CERTIFICATION:' + rid)
            record = copy.deepcopy(response)
            record.update(candidate_id=cid, root_id=c['root_id'], question_type=c['question_type'],
              task_operation=c['task_operation'], feedback_target=c['feedback_target'],
              content_decision=r['content_decision'], score=inp['score'], variant_id=inp['variant_id'],
              reviewer_id=reviewer, source_binding=c['source_binding'])
            record['all_semantic_dimensions_pass'] = bool(assessable
              and all(decision(item[f]) == 'PASS' for item in response['items'].values() for f in ITEM_FIELDS)
              and all(decision(response[f]) in {'PASS', 'NA'} for f in RESPONSE_FIELDS))
            flat.append(record)
    return list(reviews.values()), flat, concerns


def accuracy(records):
    n = len(records)
    return {'requests': n, 'valid': sum(bool(r['valid']) for r in records),
            'exact': sum(bool(r['exact']) for r in records),
            'exact_accuracy': sum(bool(r['exact']) for r in records) / n if n else None}


def assemble(args):
    out = args.output_dir.resolve(); freeze = read(out / 'REVIEW_FREEZE.json')
    for bound in freeze['bindings']:
        check(bound)
    destination = out / 'assembled'
    need(not destination.exists(), 'NO_ASSEMBLY_OVERWRITE')
    all_reviews, flat, concerns, review_bindings = [], [], [], []
    for group in ('a', 'b'):
        packet_path = out / f'REVIEW_PACKET_{group.upper()}.json'
        review_path = out / f'review_{group}/SEMANTIC_REVIEW_AI.json'
        packet, review = read(packet_path), read(review_path)
        reviews, records, found = validate_group(packet, review, freeze)
        all_reviews.extend(reviews); flat.extend(records); concerns.extend(found)
        review_bindings.append(binding(review_path))
    need(len(all_reviews) == freeze['questions'] and len(flat) == freeze['responses'], 'ASSEMBLED_REVIEW_COUNT_DRIFT')
    need(len({r['request_id'] for r in flat}) == len(flat), 'DUPLICATE_ASSEMBLED_RESPONSE')
    by_card, by_question = defaultdict(list), defaultdict(list)
    for response in flat:
        by_card[response['feedback_target']['card_id']].append(response)
        by_question[response['candidate_id']].append(response)
    card_summary = []
    for card, records in sorted(by_card.items()):
        observed = [r for r in records if r['predicted_error_assessment']['decision'] == 'OBSERVED']
        pairs = defaultdict(list)
        for r in records:
            pairs[r['candidate_id']].append(r)
        both = [cid for cid, rows in pairs.items() if len(rows) == 2 and all(
            r['predicted_error_assessment']['decision'] == 'OBSERVED' for r in rows)]
        card_summary.append({'card_id': card, 'responses': len(records),
          'predicted_error_statuses': dict(Counter(r['predicted_error_assessment']['decision'] for r in records)),
          'responses_observed': len(observed), 'unique_questions_observed': len({r['candidate_id'] for r in observed}),
          'roots_observed': sorted({r['root_id'] for r in observed}),
          'roots_observed_count': len({r['root_id'] for r in observed}),
          'questions_observed_in_both_displays': sorted(both),
          'questions_observed_in_both_displays_count': len(both),
          'competing_errors': dict(Counter(d for r in records for d in r['error_dimensions'])),
          'formal_weakness_admission': False})
    execution = Path(freeze['execution_dir'])
    scores = read(execution / 'SCORES.json')['records']
    paired = []
    for cid, rows in sorted(by_question.items()):
        qonly = [s for s in scores if s['candidate_id'] == cid and s['condition'] == 'QUESTION_ONLY']
        paired.append({'candidate_id': cid, 'root_id': rows[0]['root_id'], 'question_type': rows[0]['question_type'],
          'task_operation': rows[0]['task_operation'], 'FULL': accuracy([r['score'] for r in rows]),
          'QUESTION_ONLY': accuracy(qonly), 'descriptive_only': True})
    item_counts = {f: dict(Counter(decision(i[f]) for r in flat for i in r['items'].values())) for f in ITEM_FIELDS}
    response_counts = {f: dict(Counter(decision(r[f]) for r in flat)) for f in RESPONSE_FIELDS}
    slot_doc = read(HERE / 'SLOT_LEDGER.json')
    admission_rows = slot_doc['slots']
    current_by_condition = {condition: accuracy([s for s in scores if s['condition'] == condition])
                            for condition in ('FULL', 'QUESTION_ONLY')}
    initial = []
    initial_bindings = []
    for group in ('a', 'b'):
        path = HERE / f'review_{group}/INITIAL_CONTENT_REVIEW_AI.json'
        if path.exists():
            doc = read(path); initial.extend(doc if isinstance(doc, list) else doc['reviews']); initial_bindings.append(binding(path))
    expected_initial_ids = {s['candidate_id'] for s in admission_rows if s['status'] != 'MATERIAL_SHORTFALL'}
    initial_complete = {r['candidate_id'] for r in initial} == expected_initial_ids
    content_summary = {'planned_slots': len(admission_rows),
      'final_slot_decisions': dict(Counter(s['status'] for s in admission_rows)),
      'initial_review_records_complete': initial_complete,
      'initial_content_decisions': dict(Counter(r['decision'] for r in initial)) if initial_complete else None,
      'final_operation_coverage': dict(Counter(s['task_operation'] for s in admission_rows if s['status'] == 'PASS')),
      'material_shortfall_slots': [s for s in admission_rows if s['status'] == 'MATERIAL_SHORTFALL'],
      'per_root_slot_yield': [{'root_id': root, 'planned_slots': sum(s['root_id'] == root for s in admission_rows),
        'final_decisions': dict(Counter(s['status'] for s in admission_rows if s['root_id'] == root))}
        for root in sorted({s['root_id'] for s in admission_rows})],
      'source_population_stability_established': False}
    summary = {'schema_version': 'priority_full_semantic_audit_summary_v1',
      'created_at_utc': datetime.now(timezone.utc).isoformat(), 'questions': len(all_reviews),
      'full_responses': len(flat), 'roots': len(freeze['sources']),
      'content_postfreeze_decisions': dict(Counter(r['content_decision'] for r in all_reviews)),
      'item_status_counts': item_counts, 'response_status_counts': response_counts,
      'full_response_scores': accuracy([r['score'] for r in flat]),
      'all_semantic_dimensions_pass': sum(r['all_semantic_dimensions_pass'] for r in flat),
      'exact_and_all_semantic_dimensions_pass': sum(r['all_semantic_dimensions_pass'] and r['score']['exact'] for r in flat),
      'card_outcomes': card_summary, 'accuracy_by_condition': current_by_condition,
      'operation_scores': [{'condition': condition, 'operation': operation,
          **accuracy([s for s in scores if s['condition'] == condition and s['task_operation'] == operation])}
          for condition in ('FULL', 'QUESTION_ONLY') for operation in sorted({s['task_operation'] for s in scores})],
      'root_scores': [{'condition': condition, 'root_id': root,
          **accuracy([s for s in scores if s['condition'] == condition and s['root_id'] == root])}
          for condition in ('FULL', 'QUESTION_ONLY') for root in sorted({s['root_id'] for s in scores})],
      'gold_bucket_scores': [{'condition': condition, 'gold_count': gold_count,
          **accuracy([s for s in scores if s['condition'] == condition and s.get('gold_count') == gold_count])}
          for condition in ('FULL', 'QUESTION_ONLY') for gold_count in (2, 3, 4, 5, 6)],
      'primary_control_scores': [{'condition': condition, 'group': group,
          **accuracy([s for s in scores if s['condition'] == condition and
            ((group == 'MC2_5_PRIMARY' and s.get('gold_count') in (2, 3, 4, 5))
             or (group == 'MC6_CONTROL' and s.get('gold_count') == 6)
             or (group == 'ORDERING' and s['question_type'] == 'ordering'))])}
          for condition in ('FULL', 'QUESTION_ONLY') for group in ('MC2_5_PRIMARY', 'MC6_CONTROL', 'ORDERING')],
      'content_production': content_summary, 'postfreeze_content_concerns': len(concerns),
      'training_pool_hold': bool(concerns), 'original_scores_and_admission_unchanged': True,
      'new_solver_calls': 0, 'new_training_runs': 0, 'formal_weakness_admission': False,
      'causal_feedback_gain_proven': False, 'faithful_internal_cot_claim': False,
      'unseen_root_generalization_claim': False, 'reviewer_is_human': False, 'gold_blind': False,
      'paired_displays_are_independent_questions': False,
      'bindings': [binding(out / 'REVIEW_FREEZE.json')] + review_bindings + initial_bindings}
    original_cards = read(HERE / 'FEEDBACK_INPUT_FREEZE.json')['cards']
    revised = []
    for original in original_cards:
        old_id = original['card_id']; records = by_card.get(old_id, [])
        card = copy.deepcopy(original)
        card.update(card_id='PDF1-HYP-' + original['dimension'], parent_card_id=old_id,
          source_card_status=original['status'], status='EXPLORATORY_OBSERVATION_NOT_REPRODUCED',
          supporting_requests=[{'request_id': r['request_id'], 'candidate_id': r['candidate_id'],
            'root_id': r['root_id'], 'raw_binding': r['raw_binding'], 'reviewer_id': r['reviewer_id'],
            'exact': r['score']['exact'], 'valid': r['score']['valid'], 'content_decision': r['content_decision'],
            'notes': r['predicted_error_assessment']['note'],
            'assessment': r['predicted_error_assessment']} for r in records
            if r['predicted_error_assessment']['decision'] == 'OBSERVED'],
          current_batch_outcome=next((s for s in card_summary if s['card_id'] == old_id), None),
          counterobservations=[{'request_id': r['request_id'], 'candidate_id': r['candidate_id'],
            'root_id': r['root_id'], 'assessment': r['predicted_error_assessment']} for r in records
            if r['predicted_error_assessment']['decision'] != 'OBSERVED'],
          parent_card_consumed_in_this_generation=True, challenger_consumed=False,
          next_generation_status='PENDING_ACTUAL_READ_AND_TASK_PLANNING', formal_weakness_admission=False,
          training_target=False, internal_cause_claim=False, causal_feedback_gain_proven=False)
        support = card['supporting_requests']
        card['supporting_unique_questions'] = len({r['candidate_id'] for r in support})
        card['supporting_roots'] = sorted({r['root_id'] for r in support})
        card['historical_support_not_pooled_as_new_observation'] = True
        revised.append(card)
    save(destination / 'SEMANTIC_REVIEW_AI.json', {'reviews': all_reviews, 'bindings': review_bindings})
    save(destination / 'FULL_RESPONSE_REVIEWS.json', {'records': flat})
    save(destination / 'POSTFREEZE_CONTENT_CONCERNS.json', {'concerns': concerns,
      'training_pool_hold': bool(concerns), 'original_questions_gold_scores_admission_unchanged': True})
    save(destination / 'PAIRED_FULL_QUESTION_ONLY.json', {'questions': paired,
      'paired_displays_are_dependent': True, 'causal_feedback_gain_proven': False})
    save(destination / 'SUMMARY.json', summary)
    save(destination / 'EXPLORATORY_WEAKNESS_CARDS.json', {'schema_version': 'priority_revised_exploratory_cards_v1',
      'cards': revised, 'semantic_review_binding': binding(destination / 'SEMANTIC_REVIEW_AI.json'),
      'outcome_rules_binding': binding(HERE / 'FEEDBACK_OUTCOME_RULES.json'),
      'formal_feedback_release': False, 'challenger_consumed': False})
    report = ['# 本批 FULL 回答独立语义审核', '',
      f"已逐项审核 {len(all_reviews)} 道题的 {len(flat)} 个 FULL 显示响应，来源为 {len(freeze['sources'])} 个既有 Train roots。",
      f"后审内容决策：{summary['content_postfreeze_decisions']}；内容疑点 {len(concerns)} 道，原始准入与分数均保留。",
      f"FULL 分数：{summary['accuracy_by_condition']['FULL']}；QUESTION_ONLY：{summary['accuracy_by_condition']['QUESTION_ONLY']}。", '',
      '## 预先预测错误的观察', '',
      '| 卡片 | 观察显示数 | 独立题数 | Root 数 | 两个显示均观察到的题数 |',
      '|---|---:|---:|---:|---:|']
    for s in card_summary:
        report.append(f"| {s['card_id']} | {s['responses_observed']} | {s['unique_questions_observed']} | {len(s['roots_observed'])} | {len(s['questions_observed_in_both_displays'])} |")
    report += ['', '显示数不能作为独立题数；四个 root 的有界试验不能证明总体稳定产率。理由是可见输出，不能证明内部因果过程。',
      '本批没有训练、没有随机反馈/无反馈对照，不能证明因果训练收益或未见 root 泛化。',
      '新版卡片仍是探索观察，下一代是否实际读取和出题须另存消费回执；这里只写下一代待消费状态。', '']
    with (destination / 'REPORT_ZH.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(report))
    save(destination / 'MANIFEST.json', {'schema_version': 'priority_semantic_audit_manifest_v1',
      'bindings': [binding(p) for p in sorted(destination.iterdir()) if p.is_file()],
      'review_freeze_binding': binding(out / 'REVIEW_FREEZE.json'), 'new_solver_calls': 0,
      'formal_weakness_admission': False, 'training_admission': False})
    print(json.dumps({'reviewed_questions': len(all_reviews), 'full_responses': len(flat),
                      'content_concerns': len(concerns), 'output': str(destination)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'assemble'))
    parser.add_argument('--execution-dir', type=Path, default=HERE / 'execution')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'semantic_audit')
    parser.add_argument('--reviewer-a', default=DEFAULT_REVIEWERS['a'])
    parser.add_argument('--reviewer-b', default=DEFAULT_REVIEWERS['b'])
    args = parser.parse_args()
    {'prepare': prepare, 'assemble': assemble}[args.action](args)


if __name__ == '__main__':
    main()
