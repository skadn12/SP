"""Offline alternate finalization for the capacity supplement.

The stock legacy delivery failed while formatting an empty-by-policy answer
set (KeyError for a missing ordering count). This script replays the completed
Solver scores, proves that no question was selected, and writes an explicitly
alternate empty provisional pool. It never calls a model, erases the original
ERROR, or creates evaluation.COMPLETE.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent
CODE = BASE / 'h4_sequential_ab_v55_capacity_qa'
RUN = CODE / 'runs/root135466_c05_capacity4096_01_evaluation'
GEN = CODE / 'runs/root135466_c05_capacity4096_01_generation'
ALT = RUN / 'alternate_finalization_empty_pool'
MARKER = HERE / 'FINALIZATION_RECOVERY.COMPLETE.json'
ORIGINAL_ERROR = HERE / 'evaluation.ERROR.json'
sys.path.insert(0, str(CODE))
import evaluate_and_pool as pipeline

VERSION = 'v55_alternate_empty_finalization_after_legacy_delivery_error_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def files(root):
    return sorted(path for path in Path(root).rglob('*') if path.is_file())


def verify_bindings(rows):
    for row in rows:
        require(binding(row['path']) == row, 'alternate_bound_file_changed:' + row['path'])


def run():
    require(ORIGINAL_ERROR.is_file(), 'stock_evaluation_error_missing')
    require(read(ORIGINAL_ERROR)['error_type'] == 'CalledProcessError', 'unexpected_stock_error_type')
    require(not (HERE / 'evaluation.COMPLETE.json').exists(), 'must_not_fake_evaluation_complete')
    require(not ALT.exists() and not MARKER.exists(), 'alternate_finalization_already_exists_no_replay')
    original_bindings = [binding(path) for path in files(RUN)]
    original_bindings += [binding(path) for path in
                          (ORIGINAL_ERROR, HERE / 'FINALIZE.log', Path(__file__))]
    # Reuse the exact offline pipeline verification and score replay. This reads
    # existing artifacts only; it does not make a Solver call.
    outer = pipeline.verify(RUN)
    manifest, questions, provenance, inspector = pipeline.verify_generation(GEN)
    require(len(questions) == 1, 'supplement_solver_question_count_changed')
    solver = RUN / 'solver'
    solver_manifest, records, _ = inspector.solver(solver)
    results = read(solver / 'RESULTS.json')
    selected = pipeline.verify_scores(questions, solver_manifest, records, results)
    require(selected == [], 'alternate_empty_pool_requires_zero_selected_questions')
    # Keep the old stock delivery error and partial legacy delivery intact. The
    # alternate output is namespaced so it cannot be mistaken for stock output.
    ALT.mkdir(parents=True, exist_ok=False)
    (ALT / 'TRAINING_POOL.jsonl').write_text('', encoding='utf-8')
    (ALT / 'DISPLAY_CANDIDATES.jsonl').write_text('', encoding='utf-8')
    save(ALT / 'SELECTED_QUESTIONS.json', {'questions': [], 'training_allowed': False,
         'training_started': False, 'source_admission_status': 'PROVISIONAL_CANDIDATE_HISTORY',
         'pool_kind': 'provisional_qa_selected_candidates', 'selection_status': 'EMPTY_BY_POLICY'})
    save(ALT / 'REPORT.json', {'version': VERSION,
         'status': 'ALTERNATE_IMMUTABLE_PROVISIONAL_QA_EMPTY_POOL',
         'selection_status': 'EMPTY_BY_POLICY', 'reason': 'FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM',
         'evaluated': len(questions), 'selected_before_exact_dedup': 0, 'new_unique_added': 0,
         'duplicates': [], 'full_solver_calls': len(records), 'ground_calls': 0, 'training_calls': 0,
         'training_started': False, 'training_allowed': False, 'semantic_gold_certified': False,
         'stock_delivery_error_preserved': str(ORIGINAL_ERROR),
         'stock_delivery_partial_artifacts_preserved': str(RUN / 'delivery'),
         'solver_scores_replayed': True, 'solver_complete': True,
         'canonical_question_ids': [q['candidate_id'] for q in questions],
         'selected_question_ids': [], 'alternate_finalization_only': True})
    pool_files = files(ALT)
    save(ALT / 'MANIFEST.json', {'version': VERSION, 'status': 'ALTERNATE_EMPTY_PROVISIONAL_QA_POOL',
         'pipeline_manifest': binding(RUN / 'MANIFEST.json'),
         'generation_manifest': binding(GEN / 'MANIFEST.json'),
         'solver_manifest': binding(solver / 'manifest.json'),
         'solver_results': binding(solver / 'RESULTS.json'),
         'stock_evaluation_error': binding(ORIGINAL_ERROR),
         'files': [binding(path) for path in pool_files],
         'selected_question_count': 0, 'evaluated_question_count': len(questions),
         'full_solver_calls': len(records), 'training_started': False,
         'training_allowed': False, 'selection_status': 'EMPTY_BY_POLICY',
         'semantic_correctness_certified': False,
         'alternate_finalization_does_not_replace_stock_error': True})
    # Bind the completed alternate artifacts in an immutable success marker.
    verify_bindings(original_bindings)
    marker = {'version': VERSION, 'status': 'COMPLETE',
        'recovery_kind': 'offline_export_only',
        'new_solver_calls': 0, 'new_http_calls': 0,
        'original_failure_preserved': True,
        'evaluation_run': str(RUN), 'alternate_pool_directory': str(ALT),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_error_preserved': binding(ORIGINAL_ERROR),
        'alternate_pool': binding(ALT / 'MANIFEST.json'),
        'selected': 0, 'evaluated': len(questions), 'full_solver_calls': len(records),
        'models_called_by_recovery': 0, 'training_started': False,
        'inputs': original_bindings, 'outputs': [binding(path) for path in files(ALT)],
        'stock_evaluation_complete_marker_created': False,
        'note': 'Offline score replay passed; stock legacy delivery remained failed due to missing ordering key.'}
    save(MARKER, marker)
    # Final read-only binding replay catches accidental writes immediately.
    verify_bindings(read(ALT / 'MANIFEST.json')['files'])
    verify_bindings(read(MARKER)['inputs'])
    verify_bindings(read(MARKER)['outputs'])
    return {'status': marker['status'], 'selected': 0, 'evaluated': len(questions),
            'full_solver_calls': len(records), 'alternate_pool': str(ALT),
            'marker': str(MARKER), 'stock_error_preserved': str(ORIGINAL_ERROR)}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
