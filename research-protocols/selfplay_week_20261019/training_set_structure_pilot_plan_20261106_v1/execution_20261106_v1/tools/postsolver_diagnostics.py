#!/usr/bin/env python3
"""Descriptive frozen-answer error counts after Solver, never selection/repair."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import pilot_artifacts as pa

E = Path(__file__).resolve().parents[1]


def alternate_path(edges, a, b):
    graph = defaultdict(set)
    for x, y in edges:
        if (x, y) != (a, b):
            graph[x].add(y)
    seen, pending = set(), list(graph[a])
    while pending:
        x = pending.pop()
        if x == b:
            return True
        if x not in seen:
            seen.add(x)
            pending.extend(graph[x] - seen)
    return False


def main():
    score_path = E / 'solver/evaluation/SCORES.json'
    seal = pa.read(E / 'solver/INPUT_REVIEW_SEAL.json')
    items = {x['candidate_id']: pa.candidate(x['raw_draft']['path']) for x in seal['items']}
    scores = pa.read(score_path)
    rows = []
    for r in scores['requests']:
        c = items[r['candidate_id']]
        row = {k: r[k] for k in ('request_id', 'candidate_id', 'group', 'root_id', 'question_type', 'context_mode', 'protocol_markers', 'clean_contrast_eligible')}
        row['patterns'] = []
        row['execution_status'] = r['execution_status']
        row['valid_final_answer'] = r.get('valid', False)
        if r['execution_status'] != 'RETURNED':
            row['patterns'].append('INFRASTRUCTURE_MISSING_NOT_MODEL_ERROR')
        elif not r.get('valid'):
            row['patterns'].append('INVALID_FINAL_ANSWER_SEPARATE_FROM_SEMANTIC_PATTERNS')
        else:
            pred = r['predicted_option_ids']
            kind = c['question_type']
            if kind == 'multi_select':
                gold = set(c.get('gold_option_ids', c.get('gold')))
                p = set(pred)
                row.update(missing_ids=sorted(gold - p), extra_ids=sorted(p - gold))
                if gold - p:
                    row['patterns'].append('SUPPORTED_SET_OMISSION')
                if p - gold:
                    row['patterns'].append('UNSUPPORTED_EXTRA_SELECTION')
            elif kind == 'ordering':
                gold = c.get('gold_option_ids', c.get('gold'))
                pos = {x: i for i, x in enumerate(pred)}
                reversed_pairs = [[a, b] for i, a in enumerate(gold) for b in gold[i+1:] if pos[a] > pos[b]]
                row['reversed_gold_pairs'] = reversed_pairs
                if reversed_pairs:
                    row['patterns'].append('FULL_ORDER_RELATION_REVERSAL')
            else:
                gold = set(map(tuple, c['gold_pairs']))
                p = set(map(tuple, pred))
                missing, extra = gold - p, p - gold
                reversed_pairs = {x for x in extra if tuple(reversed(x)) in gold}
                transitive_missing = {x for x in missing if alternate_path(gold, *x)}
                row.update(missing_pairs=sorted(missing), extra_pairs=sorted(extra),
                           reversed_pairs=sorted(reversed_pairs),
                           graph_transitive_missing_pairs=sorted(transitive_missing))
                if missing:
                    row['patterns'].append('SUPPORTED_RELATION_OMISSION')
                if reversed_pairs:
                    row['patterns'].append('PARTIAL_RELATION_REVERSAL')
                if extra - reversed_pairs:
                    row['patterns'].append('UNSUPPORTED_EXTRA_RELATION')
                if transitive_missing:
                    row['patterns'].append('GRAPH_TRANSITIVE_RELATION_OMISSION')
        rows.append(row)
    counts = {}
    for group in ('OLD_F', 'OLD_G', 'NEW_F', 'NEW_G'):
        counts[group] = {}
        for mode in ('full_context', 'no_context'):
            subset = [r for r in rows if r['group'] == group and r['context_mode'] == mode]
            counts[group][mode] = {'planned': len(subset),
                'returned': sum(r['execution_status'] == 'RETURNED' for r in subset),
                'valid_answers': sum(r['valid_final_answer'] for r in subset),
                'pattern_counts_nonexclusive': dict(Counter(p for r in subset for p in r['patterns']))}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scores_sha256': pa.digest(score_path), 'requests': rows, 'by_group': counts,
              'limits': ['Errors are compared with independently frozen source-reviewed Gold, not author hypotheses or rewritten targets.',
                         'Graph-transitive means the Gold graph has an alternate path of length at least two; this does not prove a primitive source dependency or internal reasoning mechanism.',
                         'Counts are correlated repeated observations and nonexclusive; they do not establish feedback training benefit or a causal F/G effect.',
                         'No-context errors and scores remain separate from source-supported Full-context behavior; no score affects selection.']}
    pa.write_new(E / 'solver/evaluation/FEEDBACK_PATTERN_DIAGNOSTICS.json', result)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
