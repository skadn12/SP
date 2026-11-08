"""Recompute every raw answer's canonical score independently of native scoring."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import itertools
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    errors = []; totals = {}; prompt_bindings = {}; input_bindings = []
    for side in ('base', 'trained'):
        run = HERE / 'evaluation/run' / side
        manifest = read(run / 'manifest.json')
        requests = {r['request_id']: r for r in manifest['requests']}
        assert len(requests) == len(manifest['requests']) == manifest['request_count']
        candidate_ids = {r['candidate_id'] for r in manifest['requests']}
        assert len(candidate_ids) == manifest['candidate_count']
        identity_by_candidate = {r['candidate_id']: (r['root_id'], r['question_type']) for r in manifest['requests']}
        gold = read(run / 'gold.json'); scores = defaultdict(lambda: defaultdict(list)); exacts = defaultdict(list)
        assert set(gold) == candidate_ids
        seen = set()
        for f in (run / 'responses').glob('*.json'):
            row = read(f); rid = row['request_id']
            assert rid not in seen and rid in requests; seen.add(rid)
            request = requests[rid]; cid = request['candidate_id']
            assert all(row[k] == request[k] for k in ('candidate_id','root_id','question_type','seed','slot'))
            assert row['variant_id'] == request['variant']['variant_id']
            assert row['evaluation_valid'] is True
            match = re.fullmatch(r'\s*<box>\s*([^<>]*)\s*</box>\s*', row['raw_output'])
            assert match
            prediction = [s.strip().upper() for s in match.group(1).split(',') if s.strip()]
            assert len(prediction) == len(set(prediction))
            mapping = request['variant']['display_to_content']
            assert all(p in mapping for p in prediction)
            canonical = [mapping[p] for p in prediction]; expected = gold[cid]
            if row['question_type'] == 'multi_select':
                tp = len(set(canonical) & set(expected)); fp = len(set(canonical) - set(expected))
                score = Fraction(tp, len(expected))
                if len(expected) != len(mapping): score -= Fraction(fp, len(mapping) - len(expected))
                exact = set(canonical) == set(expected)
            else:
                assert len(canonical) == 6 and set(canonical) == set(expected)
                rank = {k: i for i,k in enumerate(canonical)}
                wrong = sum(rank[a] > rank[b] for a,b in itertools.combinations(expected, 2))
                score = Fraction(15 - 2 * wrong, 15); exact = canonical == expected
            assert score == Fraction(row['score']['partial_score_fraction'])
            assert exact == row['score']['exact_match']
            scores[cid][row['variant_id']].append(score); exacts[cid].append(exact)
            prompt = run / 'prompts' / (rid + '.txt')
            assert sha(prompt) == request['prompt_sha256']
            if side == 'base': prompt_bindings[rid] = (sha(prompt), request['seed'], request['variant'])
            else: assert prompt_bindings[rid] == (sha(prompt), request['seed'], request['variant'])
        assert seen == set(requests)
        result = read(run / 'RESULTS.json')
        assert result['call_set_errors'] == []
        result_ids = [r['candidate_id'] for r in result['questions']]
        assert len(result_ids) == len(set(result_ids)) and set(result_ids) == candidate_ids
        for r in result['questions']:
            cid = r['candidate_id']; groups = scores[cid]
            assert (r['root_id'], r['question_type']) == identity_by_candidate[cid]
            means = {k: sum(v, Fraction()) / len(v) for k,v in groups.items()}
            assert len(means) == (6 if r['question_type'] == 'multi_select' else 12)
            assert all(len(v) == (3 if r['question_type'] == 'multi_select' else 1) for v in groups.values())
            mean = sum(means.values(), Fraction()) / len(means)
            assert mean == Fraction(r['full_partial_mean'])
            assert Fraction(sum(exacts[cid]), len(exacts[cid])) == Fraction(r['full_exact_rate'])
            assert r['selected_by_policy'] == (0 < mean < 1)
        selected_ids = result['selected_questions']
        assert len(selected_ids) == len(set(selected_ids))
        assert set(selected_ids) == {r['candidate_id'] for r in result['questions'] if r['selected_by_policy'] is True}
        totals[side] = {'calls': len(seen), 'candidates': len(scores), 'errors': 0}
        input_bindings.extend({'path': str(path.resolve()), 'sha256': sha(path)} for path in
                              [run / 'manifest.json', run / 'gold.json', run / 'RESULTS.json',
                               *sorted((run / 'responses').glob('*.json')),
                               *sorted((run / 'prompts').glob('*.txt'))])
    doc = {'status': 'PASS', 'checks': 'Every raw answer, canonical mapping, exact score, partial score, complete call set, paired prompt/permutation/seed, and admission decision recomputed.',
           'results': totals, 'errors': errors, 'same_inherited_model_external_certification': False,
           'input_bindings': input_bindings, 'training_started': False}
    with (HERE / 'INDEPENDENT_SOLVER_AUDIT.json').open('x') as f: json.dump(doc, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(doc, ensure_ascii=False))

if __name__ == '__main__': main()
