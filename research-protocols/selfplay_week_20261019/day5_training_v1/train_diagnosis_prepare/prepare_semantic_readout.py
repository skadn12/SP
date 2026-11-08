"""After raw freezes and direct scoring, prepare strict evidence review rows.

Mechanical quote matches and answer scores do not establish semantic support.
No Challenger feedback is released by this CPU postprocessor.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_prepare as b
import run_v2 as r


def main(role):
    r.verify()
    run = HERE / 'models' / role
    direct_scores = b.read(run / 'private/DIRECT_SCORES.json')
    dfreeze = b.read(run / 'direct/RAW_RESPONSE_FREEZE.json')
    efreeze = b.read(run / 'brief_evidence/RAW_RESPONSE_FREEZE.json')
    r.check(dfreeze['files'] + efreeze['files'] + [dfreeze['execution_manifest'], efreeze['execution_manifest']])
    r.require(dfreeze['planned_request_count'] == 360 and efreeze['planned_request_count'] == 24, 'INCOMPLETE_RAW')
    baseline = b.read(HERE / 'private/S0_BASELINE_BINDINGS.json')
    r.check([baseline['gold_binding']])
    gold = {q['candidate_id']: q for q in b.read(baseline['gold_binding']['path'])['questions']}
    public = {q['candidate_id']: q for q in b.read(HERE / 'public/PUBLIC_QUESTIONS.json')['questions']}
    rows = b.read(HERE / 'public/BRIEF_EVIDENCE_REQUESTS.json')
    canonical = b.read(HERE / 'public/DIRECT_DISPLAY_MAPS.json')['public_to_canonical']
    annotations = {q['candidate_id']: q for q in b.read(HERE / 'SELECTION.json')['question_annotations']}
    scores_by_id = {x['request_id']: x for x in direct_scores['models'][role]['requests']}
    import jsonschema
    reviews = []
    for row in rows['requests']:
        raw_path = run / 'brief_evidence/responses' / (row['request_id'] + '.json')
        raw = b.read(raw_path)
        errors, parsed = [], None
        if raw['status'] != 'RETURNED':
            errors.append('NOT_RETURNED:' + raw['status'])
        if raw.get('metadata', {}).get('finish_reason') not in ('stop', 'eos'):
            errors.append('NONTERMINAL_OR_LENGTH_FINISH')
        errors.extend(raw.get('output_validation_errors', []))
        try:
            parsed = json.loads(raw['raw_output'])
            jsonschema.validate(parsed, row['json_schema'])
            labels = list(row['display']['options' if row['question_type'] == 'multi_select' else 'items'])
            supplied = [item['label'] for item in parsed['item_evidence']]
            r.require(len(supplied) == len(set(supplied)) and set(supplied) == set(labels), 'DUPLICATE_OR_MISSING_EVIDENCE_LABEL')
        except Exception as error:
            errors.append(type(error).__name__ + ':' + str(error)[:800])
        q = public[row['candidate_id']]
        source = b.read(q['source_path'])
        r.require(b.sha(q['source_path']) == q['source_sha256'], 'SOURCE_CHANGED')
        turns = {t['turn_ref']: (i, t) for i, t in enumerate(source['turns'])}
        turn_order = [t['turn_ref'] for t in source['turns']]
        mapping = rows['maps'][row['request_id']]['display_to_content']
        reverse = {v: k for k, v in canonical[row['candidate_id']].items()}
        gold_content = [canonical[row['candidate_id']][x] for x in gold[row['candidate_id']]['gold']]
        items, prediction, exact, pair_accuracy, consistency = [], None, False, 0.0, None
        if not errors:
            prediction = [mapping[x] for x in parsed['final_answer']]
            exact = set(prediction) == set(gold_content) if row['question_type'] == 'multi_select' else prediction == gold_content
            if row['question_type'] == 'ordering':
                ranks = {x: i for i, x in enumerate(prediction)}
                pairs = [(a, z) for i, a in enumerate(gold_content) for z in gold_content[i + 1:]]
                pair_accuracy = sum(ranks[a] < ranks[z] for a, z in pairs) / len(pairs)
            positions = {}
            for item in parsed['item_evidence']:
                label = item['label']
                citations = []
                for cite in item['citations']:
                    matched = cite['turn_ref'] in turns
                    tpos, turn = turns[cite['turn_ref']] if matched else (None, None)
                    count = turn['text'].count(cite['quote']) if turn and cite['quote'] else 0
                    excerpts = source['turns'][max(0, tpos - 1):tpos + 2] if matched else []
                    pos = (tpos, turn['text'].find(cite['quote'])) if count == 1 else None
                    all_occurrences = []
                    for si, st in enumerate(source['turns']):
                        start = 0
                        while cite['quote'] and (offset := st['text'].find(cite['quote'], start)) >= 0:
                            all_occurrences.append({'source_turn_index': si, 'turn_ref': st['turn_ref'], 'quote_char_offset': offset, 'speaker': st['speaker']})
                            start = offset + 1
                    citations.append({**cite, 'turn_ref_exists': matched, 'exact_quote_matches_in_referenced_turn': count,
                        'mechanically_unique_position': pos, 'source_turn': turn, 'neighboring_source_turns': excerpts,
                        'all_exact_quote_occurrences_in_full_source': all_occurrences,
                        'semantic_support': 'REVIEW_REQUIRED_MATCH_IS_NOT_SUPPORT'})
                unique = [x['mechanically_unique_position'] for x in citations if x['mechanically_unique_position'] is not None]
                if len(citations) == 1 and len(unique) == 1:
                    positions[label] = unique[0]
                content = mapping[label]
                items.append({'label': label, 'canonical_label': reverse[content], 'displayed_text': row['display']['options' if row['question_type'] == 'multi_select' else 'items'][label],
                    'model_judgment': item['judgment'], 'model_reason': item['reason'], 'model_final_selected_or_rank': parsed['final_answer'].index(label) if label in parsed['final_answer'] else None,
                    'gold_selected_or_rank': (content in gold_content) if row['question_type'] == 'multi_select' else gold_content.index(content),
                    'citations': citations, 'semantic_review': {'status': 'PENDING', 'support_or_refutation': None, 'judgment_source_alignment': None, 'reason': None}})
            if row['question_type'] == 'multi_select':
                supported = {x['label'] for x in parsed['item_evidence'] if x['judgment'] == 'supported'}
                consistency = {'supported_judgments_equal_final_answer': supported == set(parsed['final_answer']),
                    'not_a_truth_claim': True}
            elif len(positions) == len(items) and len(set(positions.values())) == len(items):
                checked = sorted(positions, key=positions.get)
                consistency = {'own_unique_citation_order': checked, 'own_unique_citation_order_equal_final_answer': checked == parsed['final_answer'],
                    'not_semantic_first_occurrence_admission': True}
            else:
                consistency = {'own_unique_citation_order': None, 'reason': 'Missing/multiple/nonverbatim/unresolvable citation anchors; do not impute positions.'}
        reviews.append({'model_role': role, **annotations[row['candidate_id']], 'question': q['question'], 'request_id': row['request_id'],
            'raw_binding': b.binding(raw_path), 'source_binding': b.binding(q['source_path']), 'display_mapping': rows['maps'][row['request_id']],
            'schema_valid': not errors, 'strict_validation_errors': errors, 'original_parsed_output_if_json': parsed,
            'diagnostic_final_exact_fixed_denominator': bool(not errors and exact), 'diagnostic_ordering_pair_accuracy_fixed_denominator': pair_accuracy,
            'evidence_answer_consistency_mechanical_only': consistency, 'direct_same_mapping_seed_record': scores_by_id[row['direct_request_id']],
            'items_for_semantic_review': items, 'question_review_status': 'PENDING_SOURCE_SEMANTIC_REVIEW' if not errors else 'INVALID_DIAGNOSTIC_DO_NOT_RESCUE_ANSWER',
            'review_scope': 'Audited visible citations/judgments and answer against actual full source. No faithful-internal-COT claim; existing admitted Gold is reused, not relabeled.'})
    out = run / 'private/SEMANTIC_READOUT.json'
    b.save(out, {'status': 'MECHANICAL_READOUT_NOT_SEMANTIC_ADMISSION', 'model_role': role,
        'endpoint_manifest': b.binding(run / 'EXECUTION_MANIFEST.json'), 'direct_scores': b.binding(run / 'private/DIRECT_SCORES.json'),
        'raw_evidence_freeze': b.binding(run / 'brief_evidence/RAW_RESPONSE_FREEZE.json'), 'postprocessor': b.binding(Path(__file__)),
        'questions': reviews, 'summary': {'questions': 24, 'strict_valid': sum(x['schema_valid'] for x in reviews),
            'invalid': sum(not x['schema_valid'] for x in reviews), 'exact_diagnostic_answers_fixed24': sum(x['diagnostic_final_exact_fixed_denominator'] for x in reviews),
            'source_semantic_reviews_completed': 0, 'B1_formal_reproduced_inferred': False}, 'challenger_release': False,
        'next_feedback_snapshot_required_fields': ['B1 model endpoint/hash', 'supporting source-reviewed Train cases and response hashes',
            'root/family deduplicated frequency', 'counterexamples and improving cases', 'citation/judgment/final-answer error dimension',
            'controlled support or not executed', 'hypothesis/reproduced/improving/recurred with reason', 'second-wave eligibility with reason',
            'no S0 reproduced inheritance', 'A1 not delivered to A challenger', 'no DEV/TEST facts or failures']})
    print(json.dumps({'status': 'SEMANTIC_REVIEW_READY', 'role': role, 'file': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=('A1', 'B1'), required=True)
    main(p.parse_args().role)
