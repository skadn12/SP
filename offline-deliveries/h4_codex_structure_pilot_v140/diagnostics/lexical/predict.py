#!/usr/bin/env python3
"""Frozen, standard-library-only lexical retrieval. This file never reads Gold."""
import collections
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    config = json.loads((OUT / 'CONFIG.json').read_text())
    commit = json.loads((OUT / 'PRECOMMIT.json').read_text())
    assert digest(Path(__file__)) == commit['predict_py_sha256']
    assert digest(OUT / 'CONFIG.json') == commit['config_sha256']
    token_re = re.compile(config['token_regex'])
    quote_re = re.compile(config['quote_regex'])
    questions_path = BASE / 'public/QUESTIONS.json'
    questions = json.loads(questions_path.read_text())['questions']
    questions = [q for q in questions if q['question_type'] == 'ordering']
    assert len(questions) == config['expected_question_count']
    predictions = []
    all_features = []
    input_hashes = {'public/QUESTIONS.json': digest(questions_path)}
    source_cache = {}
    for q in questions:
        root = q['root_id']
        source_path = BASE / 'sources' / (root + '.json')
        if root not in source_cache:
            source_cache[root] = json.loads(source_path.read_text())
            input_hashes[str(source_path.relative_to(BASE))] = digest(source_path)
        assert digest(source_path) == q['source_sha256']
        source = source_cache[root]
        scene_scope = sorted(set(re.findall(config['public_scope_regex'], q['question'])))
        assert scene_scope
        documents = []
        for index, turn in enumerate(source['turns']):
            if turn.get('scene_ref') not in scene_scope:
                continue
            doc_text = turn['speaker'] + ' ' + turn['text']
            tokens = [m.group().lower() for m in token_re.finditer(doc_text)]
            occurrences = collections.defaultdict(list)
            for match in token_re.finditer(doc_text):
                occurrences[match.group().lower()].append([match.start(), match.end()])
            documents.append({
                'source_index_0based': index,
                'source_position_1based': index + 1,
                'turn_ref': turn['turn_ref'],
                'speaker': turn['speaker'],
                'text': turn['text'],
                'doc_text': doc_text,
                'tf': collections.Counter(tokens),
                'length': len(tokens),
                'token_occurrences_in_speaker_plus_text': dict(occurrences),
            })
        n = len(documents)
        mean_length = sum(d['length'] for d in documents) / n
        df = collections.Counter(term for d in documents for term in d['tf'])
        item_predictions = {}
        item_features = {}
        for item_id, item_text in q['items'].items():
            terms = sorted(set(m.group().lower() for m in token_re.finditer(item_text)))
            quotes = [a or b for a, b in quote_re.findall(item_text)]
            records = []
            for doc in documents:
                contributions = []
                for term in terms:
                    freq = doc['tf'][term]
                    if not freq:
                        continue
                    idf = math.log1p((n - df[term] + 0.5) / (df[term] + 0.5))
                    denom = freq + config['bm25']['k1'] * (
                        1 - config['bm25']['b'] + config['bm25']['b'] * doc['length'] / mean_length)
                    score = idf * freq * (config['bm25']['k1'] + 1) / denom
                    contributions.append({
                        'term': term, 'tf': freq, 'df': df[term], 'idf': idf,
                        'contribution': score,
                        'offsets_in_speaker_plus_text': doc['token_occurrences_in_speaker_plus_text'][term],
                    })
                quote_hits = []
                for quote in quotes:
                    for match in re.finditer(re.escape(quote), doc['text'], flags=re.IGNORECASE):
                        quote_hits.append({'quote': quote, 'char_start': match.start(), 'char_end': match.end()})
                records.append({
                    'source_index_0based': doc['source_index_0based'],
                    'source_position_1based': doc['source_position_1based'],
                    'turn_ref': doc['turn_ref'],
                    'bm25_score': sum(c['contribution'] for c in contributions),
                    'exact_quote_hits': quote_hits,
                    'term_features': contributions,
                })
            methods = {}
            for method in config['methods']:
                pool = records
                quote_override = method == 'bm25_exact_quote_priority' and any(r['exact_quote_hits'] for r in records)
                if quote_override:
                    pool = [r for r in records if r['exact_quote_hits']]
                max_score = max(r['bm25_score'] for r in pool)
                if max_score <= 0:
                    top = []
                else:
                    top = [r for r in pool if abs(r['bm25_score'] - max_score) <= config['tie_absolute_tolerance']]
                methods[method] = {
                    'status': 'unknown' if not top else ('unique' if len(top) == 1 else 'tied_top_turns'),
                    'quote_priority_applied': quote_override,
                    'max_bm25_score': max_score,
                    'positions_0based': [r['source_index_0based'] for r in top],
                    'turn_refs': [r['turn_ref'] for r in top],
                }
            item_predictions[item_id] = {'text': item_text, 'has_double_quoted_span': bool(quotes), 'methods': methods}
            item_features[item_id] = {'text': item_text, 'query_unique_tokens': terms, 'quotes': quotes, 'all_turn_scores': records}
        method_predictions = {}
        for method in config['methods']:
            per_item = {key: value['methods'][method] for key, value in item_predictions.items()}
            relations = []
            for left, right in itertools.combinations(q['items'], 2):
                lp, rp = per_item[left]['positions_0based'], per_item[right]['positions_0based']
                if not lp or not rp:
                    relation, reason = 'unknown', 'unmatched_item'
                elif max(lp) < min(rp):
                    relation, reason = 'before', 'all_top_positions_before'
                elif min(lp) > max(rp):
                    relation, reason = 'after', 'all_top_positions_after'
                else:
                    relation, reason = 'unknown', 'shared_or_overlapping_top_positions'
                relations.append({'left': left, 'right': right, 'relation': relation, 'reason': reason})
            unique_items = all(len(x['positions_0based']) == 1 for x in per_item.values())
            ordered_groups = None
            total_order = None
            if unique_items:
                by_position = collections.defaultdict(list)
                for item, value in per_item.items():
                    by_position[value['positions_0based'][0]].append(item)
                ordered_groups = [{'source_index_0based': pos, 'items': members} for pos, members in sorted(by_position.items())]
                if all(len(group['items']) == 1 for group in ordered_groups):
                    total_order = [group['items'][0] for group in ordered_groups]
            method_predictions[method] = {
                'total_order': total_order,
                'ordered_groups_with_ties': ordered_groups,
                'pair_relations': relations,
                'resolved_pairs': sum(p['relation'] != 'unknown' for p in relations),
                'unknown_pairs': sum(p['relation'] == 'unknown' for p in relations),
            }
        predictions.append({
            'candidate_id': q['candidate_id'], 'root_id': root,
            'new_or_old': 'old' if '-old-' in q['candidate_id'] else 'new',
            'author_group_for_reporting_only': 'A' if root.rsplit('-', 1)[-1] in ('129761', '198141') else 'B',
            'has_double_quoted_item': any(v['has_double_quoted_span'] for v in item_predictions.values()),
            'public_scene_scope': scene_scope, 'eligible_turn_count': n,
            'items': item_predictions, 'methods': method_predictions,
        })
        all_features.append({'candidate_id': q['candidate_id'], 'root_id': root, 'corpus_size': n,
                             'mean_document_token_length': mean_length, 'items': item_features})
    document_inventory = {}
    for root, source in source_cache.items():
        document_inventory[root] = [{'source_index_0based': i, **t} for i, t in enumerate(source['turns'])]
    save('FEATURES.json', {'schema': 'lexical_turn_features_v1', 'features': all_features})
    save('SOURCE_TURN_INDEX.json', document_inventory)
    save('PREDICTIONS.json', {
        'schema': 'frozen_gold_blind_lexical_predictions_v1',
        'created_at_utc': now(), 'config_sha256': digest(OUT / 'CONFIG.json'),
        'predict_py_sha256': digest(Path(__file__)), 'input_sha256': input_hashes,
        'gold_read': False, 'questions': predictions,
    })
    save('PREDICTIONS_COMMIT.json', {
        'created_at_utc': now(), 'gold_read': False,
        'predictions_sha256': digest(OUT / 'PREDICTIONS.json'),
        'features_sha256': digest(OUT / 'FEATURES.json'),
        'source_turn_index_sha256': digest(OUT / 'SOURCE_TURN_INDEX.json'),
        'predict_py_sha256': digest(Path(__file__)), 'config_sha256': digest(OUT / 'CONFIG.json'),
    })
    print(json.dumps({'questions': len(predictions), 'new': sum(p['new_or_old'] == 'new' for p in predictions),
                      'old': sum(p['new_or_old'] == 'old' for p in predictions),
                      'prediction_sha256': digest(OUT / 'PREDICTIONS.json'),
                      'committed_at_utc': now()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
