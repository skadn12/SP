#!/usr/bin/env python3
"""Predeclared option-text-only grouped probe. Never writes model weights.

Outputs use exclusive create; the input and all prior deliveries are read only.
No question, dialogue, option label or Gold cardinality is a feature.
Gold is used only as an option-level supervised target and for evaluation.
"""
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json')
OUT = Path(__file__).resolve().parent
SEED = 2026101801
CONFIG = {
    'folds': 5,
    'group': 'root_id',
    'splitter': 'sklearn.model_selection.GroupKFold(n_splits=5)',
    'features': 'option_text_only',
    'tfidf': {'analyzer': 'word', 'ngram_range': [1, 2], 'min_df': 2},
    'classifier': {'name': 'LogisticRegression', 'C': 1.0, 'max_iter': 1000,
                   'random_state': SEED},
    'positive_threshold': 0.5,
    'question_exact': 'predicted_set == gold_set',
    'signed_partial': 'TP/g - FP/(6-g)',
    'question_weight': 'equal',
    'option_weight': 'equal',
    'hyperparameter_search': False,
    'trained_weight_saved': False,
}


def save(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    raw = INPUT.read_bytes()
    pool = json.loads(raw)['questions']
    questions = [q for q in pool if q['question_type'] == 'multi_select']
    assert len(questions) == 238
    rows = []
    for q in questions:
        assert set(q['options']) == set('ABCDEF')
        assert 2 <= len(q['gold']) <= 5
        for label in 'ABCDEF':
            rows.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                         'label': label, 'text': q['options'][label],
                         'target': int(label in q['gold'])})
    evidence = {
        'input_path': str(INPUT),
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'questions': len(questions), 'options': len(rows),
        'roots': len({q['root_id'] for q in questions}),
        'gold_counts': dict(sorted(Counter(len(q['gold']) for q in questions).items())),
        'python': sys.executable, 'python_version': platform.python_version(),
        'config': CONFIG,
        'scope': 'Label lexical-correlation diagnostic only; not a new independent generalization evaluation.',
    }
    save('INPUT_AND_PARAMETERS.json', evidence)
    try:
        import numpy as np
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
    except ImportError as error:
        save('RESULTS.json', {
            'status': 'NOT_RUN_DEPENDENCY_MISSING',
            'error': f'{type(error).__name__}: {error}',
            'sklearn_installed_by_this_probe': False,
            'oof_predictions_created': False,
            'scores': None,
            'interpretation': 'No classification was performed; no shortcut strength estimate is available.',
        })
        return

    text = np.asarray([r['text'] for r in rows], dtype=object)
    targets = np.asarray([r['target'] for r in rows])
    groups = np.asarray([r['root_id'] for r in rows], dtype=object)
    folds = np.full(len(rows), -1, dtype=int)
    probs = np.full(len(rows), np.nan, dtype=float)
    fold_evidence = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(text, targets, groups)):
        train_roots = set(groups[train]); test_roots = set(groups[test])
        assert not (train_roots & test_roots)
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
        train_x = vectorizer.fit_transform(text[train])
        test_x = vectorizer.transform(text[test])
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        classifier.fit(train_x, targets[train])
        probs[test] = classifier.predict_proba(test_x)[:, 1]
        folds[test] = fold
        fold_evidence.append({'fold': fold, 'train_roots': sorted(train_roots),
                              'test_roots': sorted(test_roots),
                              'train_options': len(train), 'test_options': len(test),
                              'vocabulary_size': len(vectorizer.vocabulary_)})
    assert np.all(folds >= 0) and np.all(np.isfinite(probs))
    by_id = defaultdict(list)
    for i, row in enumerate(rows):
        row.update({'fold': int(folds[i]), 'oof_probability': float(probs[i]),
                    'oof_prediction': int(probs[i] >= 0.5)})
        by_id[row['candidate_id']].append(row)
    scored = []
    for q in questions:
        predictions = {r['label'] for r in by_id[q['candidate_id']] if r['oof_prediction']}
        gold = set(q['gold']); g = len(gold)
        scored.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                       'gold_count': g, 'fold': by_id[q['candidate_id']][0]['fold'],
                       'predicted_labels': sorted(predictions), 'gold_labels': sorted(gold),
                       'exact': int(predictions == gold),
                       'signed_partial': len(predictions & gold) / g - len(predictions - gold) / (6 - g)})
    metrics = {}
    for layer in ['all', 2, 3, 4, 5]:
        selected = [s for s in scored if layer == 'all' or s['gold_count'] == layer]
        ids = {s['candidate_id'] for s in selected}
        sel = [r for r in rows if r['candidate_id'] in ids]
        metrics[str(layer)] = {
            'questions': len(selected),
            'question_equal_exact': sum(s['exact'] for s in selected) / len(selected),
            'question_equal_signed_partial': sum(s['signed_partial'] for s in selected) / len(selected),
            'option_balanced_accuracy': float(balanced_accuracy_score([r['target'] for r in sel], [r['oof_prediction'] for r in sel])),
            'option_auc': float(roc_auc_score([r['target'] for r in sel], [r['oof_probability'] for r in sel])),
        }
    save('OOF_OPTION_PREDICTIONS.json', rows)
    save('OOF_QUESTION_SCORES.json', scored)
    save('FOLDS.json', fold_evidence)
    save('RESULTS.json', {'status': 'COMPLETE', 'sklearn_version': sklearn.__version__,
                          'metrics': metrics,
                          'limitations': ['This estimates lexical correlation with Gold labels.',
                                          'It does not establish whether the LLM uses the same shortcut.',
                                          'Neither high nor low scores establish causal generalization claims.',
                                          'This is not a new independent generalization evaluation.']})


if __name__ == '__main__':
    main()
