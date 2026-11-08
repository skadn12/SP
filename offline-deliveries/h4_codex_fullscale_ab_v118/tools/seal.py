"""Seal a fully processed, independently reviewed delivery; never generate or train."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'tools'))
from status import population


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(row): assert sha(row['path']) == row['sha256'], 'BOUND_FILE_CHANGED:' + row['path']
def save(p, d):
    with p.open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    assert not (B / 'COMPLETE_MANIFEST.json').exists()
    assert not (B / 'SEAL_RECEIPT.json').exists()
    state = population()
    assert state['population_denominator'] == state['terminal_roots'] == 68, 'POPULATION_NOT_COMPLETED'
    manifest = read(B / 'MANIFEST.json')
    assert len(manifest['roots']) == 68
    for item in read(B / 'MASTER_FREEZE.json')['files']: verify(item)
    historical = read(B / 'evidence/HISTORICAL_PROTECTION.json')['files']
    for item in historical: verify(item)
    audit = read(B / 'evidence/ALL_CONTENT_AUDIT.json')
    assert audit['status'] == 'PASS' and not audit['errors'] and audit['root_denominator'] == 53
    for item in audit['input_bindings']: verify(item)
    final = read(B / 'delivery/all/FINAL_STATUS.json')
    assert final['status'] == 'FULL_FIXED_POPULATION_PROCESSED'
    assert final['population_denominator'] == final['terminal_roots'] == 68
    assert final['training_started'] is False and final['training_pool_merged'] is False
    review_path = B / 'evidence/FINAL_DELIVERY_REVIEW.json'
    review = read(review_path)
    assert review['status'] == 'PASS' and not review['errors']
    assert review['reviewer'].startswith('/root/') and review['reviewer'] != '/root'
    assert review.get('external_semantic_certification') is False
    assert review['input_bindings']
    for item in review['input_bindings']: verify(item)
    required = {str((B / p).resolve()) for p in ['MANIFEST.json', 'delivery/all/FINAL_STATUS.json',
                    'delivery/all/CONTENT_QUESTIONS.json', 'delivery/all/SELECTED_CANDIDATES.json',
                    'delivery/all/EXCLUDED_CANDIDATES.json', 'delivery/all/REPORT_ZH.md', 'evidence/ALL_CONTENT_AUDIT.json']}
    assert required <= {row['path'] for row in review['input_bindings']}, 'FINAL_REVIEW_MISSING_DELIVERY_COVERAGE'
    for wave in ['validation', 'production']:
        scope = B / 'evaluations' / wave
        a = read(scope / 'ADAPTER_BINDINGS.json')
        if a['candidate_count']:
            raw_audit = read(scope / 'INDEPENDENT_SOLVER_AUDIT.json')
            assert raw_audit['status'] == 'PASS' and not raw_audit['errors']
            for item in raw_audit['input_bindings']: verify(item)
            for side in ['base', 'trained']:
                assert read(scope / (side.upper() + '_COMPLETE.json'))['actual_calls'] == a['calls_per_model']
    save(B / 'SEAL_RECEIPT.json', {'status': 'SEALED_FULL_FIXED_POPULATION_PROCESSED_NO_TRAINING',
         'sealed_at_utc': datetime.now(timezone.utc).isoformat(), 'population_denominator': 68,
         'complete_dialogues': final['complete_dialogues'], 'selected_candidates': final['selected_candidates'],
         'historical_files_verified_unchanged': len(historical), 'independent_delivery_review': bind(review_path),
         'external_semantic_certification': False, 'training_started': False, 'training_pool_merged': False,
         'manifest_excludes_only_itself': True, 'file_mode_after_sealing': '0444',
         'limitation': 'A reproducible hash inventory and read-only file modes, not cryptographic caller authentication or prevention of privileged filesystem changes.'})
    paths = [p for p in sorted(B.rglob('*')) if p.is_file() and p.name != 'COMPLETE_MANIFEST.json']
    save(B / 'COMPLETE_MANIFEST.json', {'status': 'SEALED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'files': [bind(p) for p in paths], 'excludes_only': str(B / 'COMPLETE_MANIFEST.json')})
    for p in [*paths, B / 'COMPLETE_MANIFEST.json']: p.chmod(0o444)
    for item in read(B / 'COMPLETE_MANIFEST.json')['files']: verify(item)
    print(json.dumps({'status': 'SEALED', 'files': len(paths), 'population_denominator': 68,
                      'complete_dialogues': final['complete_dialogues'], 'selected_candidates': final['selected_candidates']}))


if __name__ == '__main__': main()
