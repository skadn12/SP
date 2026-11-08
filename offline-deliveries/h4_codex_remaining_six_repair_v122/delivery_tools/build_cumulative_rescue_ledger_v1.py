"""Build a reference-only rescue ledger; never merge candidates or train.

Read formal final exports and provenance metadata. Solver RESULTS files and
private evidence are only hashed, never decoded. Old batches are never written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
B120 = P / 'h4_codex_scene_retry_v120'
B121 = P / 'h4_codex_all_failed_repair_v121'
B122 = P / 'h4_codex_remaining_six_repair_v122'
OUT = B122 / 'delivery/CUMULATIVE_RESCUE_LEDGER_V1.json'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def verify(b):
    need(Path(b['path']).is_file() and sha(b['path']) == b['sha256'],
         'BINDING_CHANGED:' + b['path'])


def counter(values):
    return dict(sorted(Counter(values).items()))


def row_sha(q):
    raw = json.dumps(q, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def inspect_export(batch, export, expected_roots, expected_candidates=None):
    export = Path(export).resolve()
    need(export == batch / 'delivery/TRAINING_CANDIDATES_V1.json',
         'UNEXPECTED_FORMAL_EXPORT_PATH')
    d = read(export)
    need(d['status'] == 'CANDIDATES_ONLY_NOT_MERGED_OR_TRAINED', 'EXPORT_STATUS')
    for key in ('training_started', 'training_pool_merge', 'training_or_pool_merge'):
        need(d.get(key, False) is False, 'TRAINING_OR_POOL_MERGE:' + key)
    need(d['selection_model'] == 'M_clean_policy', 'SELECTION_MODEL')
    need(d['native_policy'] == 'v15_full_only_random_mcq6x3_order12x1_v1',
         'NATIVE_POLICY')
    header_bindings = {}
    for key in ('source_pool_binding', 'selection_results_binding', 'summary_binding'):
        if key in d:
            verify(d[key])  # Opaque hash only, no solver result decode.
            header_bindings[key] = d[key]
    disposition_path = batch / 'delivery/QUESTION_DISPOSITION_V1.json'
    disposition = read(disposition_path)
    need(disposition.get('training_or_pool_merge') is False, 'DISPOSITION_MERGED')
    content = disposition['questions']
    content_ids = [q['candidate_id'] for q in content]
    need(len(content_ids) == len(set(content_ids)), 'CONTENT_ID_DUPLICATE')
    content_roots = {q['root_id'] for q in content}
    need(len(content_roots) == expected_roots, 'CONTENT_ROOT_COUNT:' + batch.name)
    manifest_path = batch / 'COMPLETE_MANIFEST.json'
    manifest_binding = None
    if manifest_path.exists():
        manifest = read(manifest_path)
        index = {r['path']: r['sha256'] for r in manifest['files']}
        for p in (export, disposition_path):
            need(index.get(str(p)) == sha(p), 'HISTORICAL_EXPORT_NOT_SEALED:' + str(p))
        manifest_binding = bind(manifest_path)
    else:
        need(batch == B122, 'OLD_BATCH_MISSING_COMPLETE_MANIFEST')
    qs = d['questions']
    ids = [q['candidate_id'] for q in qs]
    need(len(ids) == len(set(ids)), 'CANDIDATE_ID_DUPLICATE:' + batch.name)
    need(set(ids) <= set(content_ids), 'CANDIDATE_OUTSIDE_CONTENT_SET')
    need(len(qs) == d['candidate_count'], 'CANDIDATE_HEADER_COUNT')
    if expected_candidates is not None:
        need(len(qs) == expected_candidates, 'HISTORICAL_CANDIDATE_COUNT')
    types = counter(q['question_type'] for q in qs)
    golds = counter(str(len(q['gold'])) for q in qs if q['question_type'] == 'multi_select')
    need(types == d['question_type_counts'], 'CANDIDATE_TYPE_HEADER')
    need(golds == d['multi_select_gold_counts'], 'CANDIDATE_GOLD_HEADER')
    need(d['root_count'] == len({q['root_id'] for q in qs}), 'CANDIDATE_ROOT_HEADER')
    need(set(golds) <= {'2', '3', '4', '5'}, 'CONTROL_OR_BAD_GOLD_IN_CANDIDATES')
    by_root = defaultdict(list)
    for q in qs:
        by_root[q['root_id']].append(q)
    roots = []
    for root in sorted(content_roots):
        rows = by_root[root]
        need(rows, 'RECOVERED_ROOT_HAS_NO_CANDIDATE:' + root)
        sources = {(q['source_path'], q['source_sha256']) for q in rows}
        need(len(sources) == 1, 'ROOT_MULTIPLE_SOURCE_VERSIONS:' + root)
        source_path, source_hash = next(iter(sources))
        source_binding = {'path': source_path, 'sha256': source_hash}
        verify(source_binding)
        source_folder = Path(source_path).parent
        evidence_bindings = {}
        candidate_refs = []
        for q in rows:
            evidence = q['evidence_export']
            need(evidence['all_reviewed_citations_preserved'] is True,
                 'LOSSLESS_EVIDENCE_FLAG:' + q['candidate_id'])
            for name, digest in evidence['input_sha256'].items():
                b = {'path': str(source_folder / name), 'sha256': digest}
                verify(b)  # Private files are opaque hashes, never decoded.
                need(name not in evidence_bindings or evidence_bindings[name] == b,
                     'ROOT_EVIDENCE_VERSION_CONFLICT:' + root)
                evidence_bindings[name] = b
            candidate_refs.append({
                'candidate_id': q['candidate_id'],
                'question_type': q['question_type'],
                'gold_count': len(q['gold']) if q['question_type'] == 'multi_select' else None,
                'export_row_sha256_canonical_json': row_sha(q),
            })
        provenance = d.get('root_source_provenance', {}).get(root)
        if provenance:
            status = provenance['source_status']
            for value in provenance.values():
                if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                    verify(value)
            provenance_basis = 'Final export root_source_provenance, with bound source-finalize receipt.'
        else:
            need(batch == B120, 'MISSING_SOURCE_PROVENANCE')
            slug = 'root' + root.rsplit('-', 1)[1]
            baseline = batch / 'units' / slug / 'inputs/BASELINE.json'
            old_baseline = P / 'h4_codex_fullscale_ab_v118/units' / slug / 'inputs/BASELINE.json'
            need(sha(baseline) == sha(old_baseline), 'V120_BASELINE_EDITED:' + root)
            status = 'UNCHANGED'
            provenance = {'source_status': status, 'baseline_binding': bind(baseline),
                          'historical_baseline_binding': bind(old_baseline)}
            provenance_basis = 'v120 baseline byte-identical to v118 baseline; UNCHANGED describes rescue-source edits, not factual certification.'
        content_rows = [q for q in content if q['root_id'] == root]
        roots.append({
            'root_id': root, 'successful_rescue_batch': batch.name,
            'content_pass_question_count': len(content_rows),
            'content_pass_question_type_counts': counter(q['question_type'] for q in content_rows),
            'candidate_count': len(rows),
            'candidate_question_type_counts': counter(q['question_type'] for q in rows),
            'candidate_multi_select_gold_counts': counter(str(len(q['gold'])) for q in rows if q['question_type'] == 'multi_select'),
            'source_status': status, 'source_provenance': provenance,
            'source_status_basis': provenance_basis,
            'complete_dialogue_binding': source_binding,
            'candidate_export_binding': bind(export),
            'source_and_review_evidence_bindings': evidence_bindings,
            'candidate_references': candidate_refs,
        })
    provenance_counts = counter(r['source_status'] for r in roots for _ in r['candidate_references'])
    if 'source_provenance_counts' in d:
        need(provenance_counts == d['source_provenance_counts'], 'PROVENANCE_COUNT_HEADER')
    return {
        'batch_id': batch.name, 'candidate_export_binding': bind(export),
        'question_disposition_binding': bind(disposition_path),
        'complete_manifest_binding': manifest_binding,
        'final_export_provenance_bindings': header_bindings,
        'content_pass_root_count': len(roots), 'content_pass_question_count': len(content),
        'candidate_root_count': len(by_root), 'candidate_count': len(qs),
        'candidate_question_type_counts': types, 'candidate_multi_select_gold_counts': golds,
        'candidate_source_status_counts': provenance_counts, 'roots': roots,
    }


def build(v122_export=None):
    batches = [inspect_export(B120, B120 / 'delivery/TRAINING_CANDIDATES_V1.json', 4, 19),
               inspect_export(B121, B121 / 'delivery/TRAINING_CANDIDATES_V1.json', 18, 100)]
    if v122_export is not None:
        batches.append(inspect_export(B122, Path(v122_export), 6))
    roots = [r for b in batches for r in b['roots']]
    root_ids = [r['root_id'] for r in roots]
    ids = [q['candidate_id'] for r in roots for q in r['candidate_references']]
    need(len(root_ids) == len(set(root_ids)), 'CROSS_BATCH_ROOT_DUPLICATE')
    need(len(ids) == len(set(ids)), 'CROSS_BATCH_CANDIDATE_DUPLICATE')
    scope121 = read(B121 / 'REPAIR_SCOPE.json')
    expected = set(scope121['already_recovered_v120']) | {r['root_id'] for r in scope121['roots']}
    need(len(expected) == 28, 'HISTORICAL_FAILED_COHORT_COUNT')
    need(set(root_ids) <= expected, 'ROOT_OUTSIDE_FAILED_COHORT')
    missing = sorted(expected - set(root_ids))
    scope122 = read(B122 / 'REPAIR_SCOPE.json')
    if v122_export is None:
        need(set(missing) == {r['root_id'] for r in scope122['roots']}, 'REMAINING_SIX_SCOPE')
    else:
        need(not missing and len(root_ids) == 28, 'CUMULATIVE_COHORT_INCOMPLETE')
    allrefs = [q for r in roots for q in r['candidate_references']]
    return {
        'schema_version': 'v122_cumulative_rescue_ledger_v1',
        'status': 'COMPLETE_REFERENCE_LEDGER_NOT_MERGED_OR_TRAINED' if not missing else 'OLD_BATCH_CHECK_ONLY_WAITING_V122_FINAL_EXPORT',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'actual_executor': '/root/repair_v122_engineering',
        'scope': 'The original 28 failed roots only, using their successful v120/v121/v122 rescue exports. Original failed attempts remain in immutable batches.',
        'historical_failed_root_count': 28, 'recovered_content_root_count': len(roots),
        'candidate_root_count': len(roots), 'candidate_count': len(ids),
        'content_pass_question_count': sum(b['content_pass_question_count'] for b in batches),
        'candidate_question_type_counts': counter(q['question_type'] for q in allrefs),
        'candidate_multi_select_gold_counts': counter(str(q['gold_count']) for q in allrefs if q['question_type'] == 'multi_select'),
        'candidate_source_status_counts': counter(r['source_status'] for r in roots for _ in r['candidate_references']),
        'root_source_status_counts': counter(r['source_status'] for r in roots),
        'remaining_failed_roots': missing,
        'duplicate_root_ids': [], 'duplicate_candidate_ids': [],
        'frozen_cohort_bindings': [bind(B121 / 'REPAIR_SCOPE.json'), bind(B122 / 'REPAIR_SCOPE.json')],
        'batches': [{k: v for k, v in b.items() if k != 'roots'} for b in batches],
        'roots': sorted(roots, key=lambda r: r['root_id']),
        'candidate_data_copied': False, 'training_or_pool_merge': False,
        'split_policy': 'All original, synthetic edited, expanded and related questions from one root must remain in one split. Gold-family children share source facts and are not independent samples.',
        'verification': {'formal_export_status_checked': True, 'export_counts_recomputed': True,
                         'old_export_manifest_bindings_checked': True,
                         'source_and_all_export_evidence_hashes_checked': True,
                         'solver_result_files_decoded': False, 'private_evidence_files_decoded': False,
                         'question_or_answer_content_rewritten': False,
                         'old_batches_written': False,
                         'canonical_row_sha256': 'UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators=(comma,colon).'},
        'generator_binding': bind(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-only', action='store_true')
    parser.add_argument('--v122-export', type=Path)
    args = parser.parse_args()
    need(args.old_only != (args.v122_export is not None), 'CHOOSE_OLD_ONLY_OR_FINAL_EXPORT')
    report = build(args.v122_export)
    if args.old_only:
        print(json.dumps({k: v for k, v in report.items() if k != 'roots'}, ensure_ascii=False, indent=2))
        return
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'path': str(OUT), 'sha256': sha(OUT),
                      'root_count': report['recovered_content_root_count'],
                      'candidate_count': report['candidate_count'],
                      'training_or_pool_merge': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
