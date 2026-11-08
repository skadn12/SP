"""CPU structural statistics only; never loads or changes a solver model."""
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BATCH = Path(__file__).resolve().parent
DELIVERIES = BATCH.parent
SOURCES = {
    'train': DELIVERIES / 'h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json',
    'old_dev': DELIVERIES / 'h4_codex_dev_eval_v127/EVALUATION_POOL.json',
    'new_heldout': DELIVERIES / 'h4_codex_held_roots_eval_v134/evaluation/EVALUATION_POOL.json',
}


def scene(ref):
    match = re.match(r'(R02:[AB]:S0[123]):T\d+', ref)
    assert match, ref
    return match.group(1)


def main():
    output = BATCH / 'STRUCTURAL_AUDIT.json'
    assert not output.exists(), 'REFUSE_OVERWRITE'
    pools, bindings = {}, []
    for key, path in SOURCES.items():
        raw = path.read_bytes()
        pools[key] = json.loads(raw)['questions']
        bindings.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    multi = [q for q in pools['train'] if q['question_type'] == 'multi_select']
    roots = defaultdict(list)
    stems = Counter(q['question'] for q in multi)
    texts = defaultdict(set)
    for q in multi:
        roots[q['root_id']].append(q)
        for label, text in q['options'].items():
            texts[text].add(label in q['gold'])
    anchor_set_counts, shared = Counter(), []
    for questions in roots.values():
        sets = {tuple(sorted(e['turn_ref'] for e in q['option_evidence'].values())) for q in questions}
        anchor_set_counts[len(sets)] += 1
        shared.extend(len(set(a['options'].values()) & set(b['options'].values()))
                      for a, b in itertools.combinations(questions, 2))
    marker_counts = {}
    for word in ['already', 'still']:
        appearance = Counter()
        unique = Counter()
        for q in multi:
            for label, text in q['options'].items():
                if re.search(r'\b' + word + r'\b', text, re.I):
                    appearance['correct' if label in q['gold'] else 'incorrect'] += 1
        for text, labels in texts.items():
            if re.search(r'\b' + word + r'\b', text, re.I):
                for gold in labels:
                    unique['correct' if gold else 'incorrect'] += 1
        marker_counts[word] = {'appearances': dict(appearance), 'unique_texts': dict(unique)}
    ms = {
        'questions': len(multi), 'roots': len(roots), 'unique_exact_stems': len(stems),
        'most_frequent_stem_question_count': max(stems.values()),
        'stem_frequencies': [{'question': k, 'count': v} for k, v in stems.most_common()],
        'roots_with_complete_gold2345': sum({len(q['gold']) for q in rows} == {2, 3, 4, 5}
                                          for rows in roots.values()),
        'distinct_primary_evidence_sets_per_root_histogram': dict(anchor_set_counts),
        'option_appearances': sum(len(q['options']) for q in multi),
        'unique_option_texts': len(texts),
        'unique_option_texts_with_conflicting_labels': sum(len(labels) != 1 for labels in texts.values()),
        'within_root_question_pairs': len(shared),
        'mean_exact_shared_options_per_within_root_pair': sum(shared) / len(shared),
        'marker_correlations': marker_counts,
    }
    order = {}
    groups = {
        'train': pools['train'], 'old_dev': pools['old_dev'],
        'new_dev': [q for q in pools['new_heldout'] if q['split'] == 'dev'],
        'new_test': [q for q in pools['new_heldout'] if q['split'] == 'test'],
    }
    all_stems = set()
    for group, questions in groups.items():
        questions = [q for q in questions if q['question_type'] == 'ordering']
        signatures = Counter()
        repeated_scenes = 0
        for q in questions:
            ev = q['item_evidence']
            signature = tuple(scene(ev[item]['turn_ref']) for item in q['gold'])
            signatures[signature] += 1
            repeated_scenes += len(set(signature)) != len(signature)
            all_stems.add(q['question'])
        order[group] = {
            'questions': len(questions), 'roots': len({q['root_id'] for q in questions}),
            'unique_exact_stems': len({q['question'] for q in questions}),
            'questions_with_multiple_items_in_same_scene': repeated_scenes,
            'gold_scene_signatures': [{'scenes': list(s), 'count': n} for s, n in signatures.items()],
        }
    result = {
        'status': 'COMPLETE', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_bindings': bindings, 'solver_inference_calls': 0,
        'solver_training_or_weight_changes': False,
        'multi_select_training': ms, 'ordering': order,
        'unique_exact_ordering_stems_across_all_four_groups': len(all_stems),
        'limitations': [
            'Structural concentration is not by itself proof of invalid Gold or direct answer leakage.',
            'Keyword correlations do not establish that the solver used those keywords.',
            'New option permutations do not create unseen question semantics.',
            'Question and option variants within each root are correlated.',
        ],
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'COMPLETE', 'output': str(output),
                      'multi_select': {k: v for k, v in ms.items() if k != 'stem_frequencies'},
                      'ordering': order}, ensure_ascii=False))


if __name__ == '__main__':
    main()
