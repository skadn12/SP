#!/usr/bin/env python3
"""Fixed standard-library option-only lexical-correlation diagnostic.

This is Multinomial Naive Bayes, not the unexecuted sklearn TFIDF/LR probe.
All files are exclusively created. Main model/data/old packages are read only.
"""
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json')
OUT = Path(__file__).resolve().parent
SEED = 20261018
TOKENIZER = re.compile(r"[a-z]+(?:'[a-z]+)?")
CONFIG = {
    'method': 'MultinomialNaiveBayes_standard_library',
    'folds': 5,
    'group': 'root_id',
    'splitter': 'sort roots; Random(seed).shuffle(roots); test roots = roots[fold::5]',
    'seed': SEED,
    'features': 'option_text_only; lowercase English word unigrams and adjacent bigram counts',
    'token_regex': TOKENIZER.pattern,
    'vocabulary': 'built from each training fold only; test-only tokens ignored',
    'smoothing_alpha': 1.0,
    'class_prior': 'empirical training-option-label prior',
    'prediction_rule': 'log_posterior_positive >= log_posterior_negative',
    'question_exact': 'predicted_set == gold_set',
    'signed_partial': 'TP/g - FP/(6-g)',
    'auc': 'Mann-Whitney U using average ranks for exact score ties',
    'hyperparameter_search': False,
    'trained_weights_saved': False,
}


def save(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def features(text):
    words = TOKENIZER.findall(text.lower())
    return Counter(['u:' + w for w in words] +
                   ['b:' + a + ' ' + b for a, b in zip(words, words[1:])])


def auc(rows):
    ordered = sorted(rows, key=lambda r: r['log_odds_positive'])
    positive = sum(r['target'] for r in ordered)
    negative = len(ordered) - positive
    if positive == 0 or negative == 0:
        return None
    sum_rank_positive = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j]['log_odds_positive'] == ordered[i]['log_odds_positive']:
            j += 1
        mean_rank = ((i + 1) + j) / 2.0
        sum_rank_positive += mean_rank * sum(r['target'] for r in ordered[i:j])
        i = j
    return (sum_rank_positive - positive * (positive + 1) / 2) / (positive * negative)


def metrics(option_rows, question_rows):
    tp = sum(r['target'] == 1 and r['prediction'] == 1 for r in option_rows)
    tn = sum(r['target'] == 0 and r['prediction'] == 0 for r in option_rows)
    fp = sum(r['target'] == 0 and r['prediction'] == 1 for r in option_rows)
    fn = sum(r['target'] == 1 and r['prediction'] == 0 for r in option_rows)
    return {
        'questions': len(question_rows), 'roots': len({r['root_id'] for r in question_rows}),
        'options': len(option_rows),
        'question_equal_exact': sum(r['exact'] for r in question_rows) / len(question_rows),
        'question_equal_signed_partial': sum(r['signed_partial'] for r in question_rows) / len(question_rows),
        'option_accuracy': (tp + tn) / len(option_rows),
        'option_balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2,
        'option_auc': auc(option_rows),
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }


def main():
    raw = INPUT.read_bytes()
    questions = [q for q in json.loads(raw)['questions'] if q['question_type'] == 'multi_select']
    assert len(questions) == 238
    question_by_id = {q['candidate_id']: q for q in questions}
    assert len(question_by_id) == len(questions)
    roots = sorted({q['root_id'] for q in questions})
    random.Random(SEED).shuffle(roots)
    fold_for_root = {root: fold for fold in range(5) for root in roots[fold::5]}
    rows = []
    for q in questions:
        assert set(q['options']) == set('ABCDEF')
        assert 2 <= len(q['gold']) <= 5
        for label in 'ABCDEF':
            rows.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                         'label_for_scoring_only': label, 'text': q['options'][label],
                         'target': int(label in q['gold']),
                         'fold': fold_for_root[q['root_id']]})
    caches = [features(r['text']) for r in rows]
    fold_evidence = []
    for fold in range(5):
        train_ids = [i for i, row in enumerate(rows) if row['fold'] != fold]
        test_ids = [i for i, row in enumerate(rows) if row['fold'] == fold]
        train_roots = {rows[i]['root_id'] for i in train_ids}
        test_roots = {rows[i]['root_id'] for i in test_ids}
        assert not train_roots.intersection(test_roots)
        word_counts = [Counter(), Counter()]
        class_counts = [0, 0]
        for i in train_ids:
            target = rows[i]['target']
            class_counts[target] += 1
            word_counts[target].update(caches[i])
        vocab = set(word_counts[0]) | set(word_counts[1])
        token_totals = [sum(x.values()) for x in word_counts]
        denominators = [total + len(vocab) for total in token_totals]
        log_prior = [math.log(n / len(train_ids)) for n in class_counts]
        for i in test_ids:
            logs = list(log_prior)
            for token, count in caches[i].items():
                if token not in vocab:
                    continue
                for label in (0, 1):
                    logs[label] += count * math.log((word_counts[label][token] + 1) / denominators[label])
            margin = logs[1] - logs[0]
            prob = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
            rows[i].update({'prediction': int(logs[1] >= logs[0]),
                            'log_odds_positive': margin, 'probability_positive': prob})
        fold_evidence.append({'fold': fold, 'train_roots': sorted(train_roots),
                              'test_roots': sorted(test_roots),
                              'train_options': len(train_ids), 'test_options': len(test_ids),
                              'class_counts': {'negative': class_counts[0], 'positive': class_counts[1]},
                              'vocabulary_size': len(vocab)})
    assert all('prediction' in r for r in rows)
    by_id = defaultdict(list)
    for row in rows:
        by_id[row['candidate_id']].append(row)
    scored = []
    for q in questions:
        pred = {r['label_for_scoring_only'] for r in by_id[q['candidate_id']] if r['prediction']}
        gold = set(q['gold']); g = len(gold)
        assert len({r['fold'] for r in by_id[q['candidate_id']]}) == 1
        scored.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                       'gold_count': g, 'fold': fold_for_root[q['root_id']],
                       'predicted_labels': sorted(pred), 'gold_labels': sorted(gold),
                       'exact': int(pred == gold),
                       'signed_partial': len(pred & gold) / g - len(pred - gold) / (6 - g)})
    layers = {}
    for layer in ['all', 2, 3, 4, 5]:
        qs = [q for q in scored if layer == 'all' or q['gold_count'] == layer]
        ids = {q['candidate_id'] for q in qs}
        opts = [row for row in rows if row['candidate_id'] in ids]
        layers[str(layer)] = metrics(opts, qs)
    fold_metrics = {str(fold): metrics([r for r in rows if r['fold'] == fold],
                                      [q for q in scored if q['fold'] == fold]) for fold in range(5)}
    text_roots = defaultdict(set)
    for row in rows:
        text_roots[row['text']].add(row['root_id'])
    repeated_cross_root_text = {text: sorted(rts) for text, rts in text_roots.items() if len(rts) > 1}
    save('INPUT_AND_PARAMETERS.json', {
        'input_path': str(INPUT), 'input_sha256': hashlib.sha256(raw).hexdigest(),
        'questions': len(questions), 'options': len(rows), 'roots': len(roots),
        'gold_counts': dict(sorted(Counter(len(q['gold']) for q in questions).items())),
        'python': sys.executable, 'python_version': platform.python_version(), 'config': CONFIG,
    })
    save('FOLDS.json', fold_evidence)
    save('OOF_OPTION_PREDICTIONS.json', rows)
    save('OOF_QUESTION_SCORES.json', scored)
    save('REPEATED_TEXT_ACROSS_ROOTS.json', repeated_cross_root_text)
    save('RESULTS.json', {
        'status': 'COMPLETE', 'method': CONFIG['method'], 'by_gold_count': layers,
        'per_fold': fold_metrics,
        'exact_text_shared_across_roots_count': len(repeated_cross_root_text),
        'interpretation': 'Option-text label lexical-correlation diagnostic only.',
        'limitations': [
            'Gold labels supervise the temporary classifier; Gold count, stem, dialogue and option labels are never features.',
            'The same root is always in the same fold. Different roots can still share templates or vocabulary.',
            'This is not a new independent generalization evaluation.',
            'It does not establish whether the LLM uses this shortcut; neither high nor low scores establish causal claims.',
            'Scores from this Naive Bayes method must not be called TFIDF LogisticRegression results.',
            'Question variants from the same root are correlated; option count is not independent sample size.',
        ],
    })


if __name__ == '__main__':
    main()
