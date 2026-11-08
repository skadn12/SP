"""Check historical quarantine, exact-source evidence and raw metric accounting.

Read-only historical fixtures: no inference, training or production mutation.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v118_delivery_test_module', HERE / 'deliver.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def main():
    manifest = d.read(d.B / 'MANIFEST.json')
    overlay = d.read(d.B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json')
    qs, selected, batches = [], [], set()
    checks = []
    for row in manifest['roots']:
        if row['action'] != 'REUSE': continue
        q, chosen, folder, provenance = d.historical(row, overlay)
        d.verify_candidates(q); d.verify_candidates(chosen)
        assert len(q) == 7 and all(c['candidate_id'] in {v['candidate_id'] for v in q} for c in chosen)
        source = d.read(folder / 'CUMULATIVE_SOURCE.json'); baseline = d.read(row['baseline_path'])
        assert source['turns'][:len(baseline['turns'])] == baseline['turns']
        checks.append('historical_root_and_full_baseline_prefix:' + row['root_id'])
        qs.extend(q); selected.extend(chosen); batches.add(Path(row['reuse_batch']))
    assert len(qs) == 91 and len(selected) == 64
    assert not set(overlay['quarantined_original_ids']) & {q['candidate_id'] for q in qs + selected}
    assert {q['candidate_id'] for q in overlay['questions']} <= {q['candidate_id'] for q in selected}
    checks.append('mandatory_v114_quarantine_and_full_overlay_applied')
    for batch in sorted(batches):
        pool = d.read(batch / 'evaluation/EVALUATION_POOL.json')['questions']; qmap = {q['candidate_id']: q for q in pool}
        for side in ['base', 'trained']:
            run = batch / 'evaluation/run' / side
            metrics = d.summarize_calls(qmap, run)
            result = d.read(run / 'RESULTS.json')
            for rid in {q['root_id'] for q in pool}:
                rows = [r for r in result['questions'] if r['root_id'] == rid]
                calls = sum(r['actual_full_calls'] for r in rows)
                correct = sum(Fraction(r['full_exact_rate']) * r['actual_full_calls'] for r in rows)
                mean = sum(Fraction(r['full_partial_mean']) * r['actual_full_calls'] for r in rows) / calls
                m = metrics['root:' + rid]
                assert m['calls'] == calls and m['correct'] == correct and Fraction(m['partial_mean']) == mean
            checks.append('native_results_vs_independent_raw_metrics:' + batch.name + ':' + side)
    # Source-evidence failures must fail closed, rather than repair the candidate.
    original = qs[0]
    for label, mutation in [('source_hash', lambda q: q.update(source_sha256='0' * 64)),
                            ('wrong_quote', lambda q: next(iter(q['option_evidence'].values())).update(quote='THIS QUOTE IS NOT IN THE SOURCE')),
                            ('wrong_speaker', lambda q: next(iter(q['option_evidence'].values())).update(speaker='NOT_THE_SOURCE_SPEAKER'))]:
        bad = deepcopy(original); mutation(bad)
        try: d.verify_candidates([bad])
        except AssertionError: pass
        else: raise AssertionError('Accepted bad delivery evidence: ' + label)
        checks.append('rejects:' + label)
    try: d.verify_candidates([original, original])
    except AssertionError: pass
    else: raise AssertionError('Accepted duplicate candidate ID')
    checks.append('rejects:duplicate_candidate_id')
    print(json.dumps({'status': 'PASS', 'check_count': len(checks), 'checks': checks,
                      'inference_calls': 0, 'production_files_changed': False}, ensure_ascii=False))


if __name__ == '__main__': main()
