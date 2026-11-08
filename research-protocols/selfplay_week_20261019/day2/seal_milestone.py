"""Seal the Solver diagnosis milestone while retaining unfinished Day 2 goals."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import ast
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
TARGET = BASE / 'COMPLETE_MANIFEST.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def included_files():
    nodes = sorted(BASE.rglob('*'))
    assert not any(p.is_symlink() for p in nodes), 'Unexpected symlink in milestone tree'
    return [p for p in nodes if p.is_file() and p != TARGET
            and '__pycache__' not in p.parts and p.suffix != '.pyc']


def main():
    assert not TARGET.exists(), 'Never overwrite a sealed milestone'
    status = read(BASE / 'FINAL_STATUS.json')
    assert status['status'] == 'SOLVER_DIAGNOSIS_MILESTONE_COMPLETE'
    assert status['day2_all_goals_complete'] is False
    assert status['new_generated_training_questions'] == status['new_training_runs'] == 0
    assert status['fact_memory_pilot']['actual_scenes'] == 0
    assert status['fact_memory_pilot']['semantic_pass'] == 0
    assert status['weakness_memory']['causes'] == status['weakness_memory']['reproduced_mechanisms'] == 0
    assert status['cost']['returned_outputs'] == 711
    assert status['cost']['submitted_request_attempts'] == 719
    assert status['cost']['planned_record_denominator'] == 750
    audit_files = [
        'VALIDATION.json', 'INDEPENDENT_SOLVER_PREFLIGHT.json',
        'INDEPENDENT_EVIDENCE_V2_PREFLIGHT.json',
        'EVIDENCE_RUNTIME_V2_GRAMMAR_PREFLIGHT.json',
        'audit/RESULTS_AUDIT_DIRECT.json', 'audit/RESULTS_AUDIT_EVIDENCE_V1.json',
        'audit/RESULTS_AUDIT_EVIDENCE_V2.json', 'audit/RESULTS_AUDIT_HINT.json',
        'audit/RESULTS_AUDIT_HINT_PAIRED.json',
        'audit/RESULTS_AUDIT_OBSERVATIONS_02.json',
        'audit/RESULTS_AUDIT_OBSERVATIONS_BINDING_REPAIR.json',
        'audit/RESULTS_AUDIT_RUNTIME_COSTS.json',
        'audit/PARENT_OBSERVED_MEMORY_RECHECK.json',
        'analysis/order_trace_consistency/VALIDATION.json',
        'fact_memory/VALIDATION_SEALED.json',
        'diagnosis_exports/observed_six_v002/CONTROLLER_VALIDATION.json',
    ]
    for rel in audit_files:
        assert read(BASE / rel)['status'].startswith('PASS'), rel
    for rel in status['preserved_failures']:
        assert (BASE / rel).is_dir() and any((BASE / rel).iterdir()), rel
    for rel in [status['canonical_observations'], status['canonical_memory']]:
        assert (BASE / rel).is_file(), rel
    for _, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', (BASE / 'REPORT_ZH.md').read_text()):
        if target.startswith(('https://', 'http://', '#')):
            continue
        local = re.sub(r':\d+$', '', target.strip('<>')).split('#', 1)[0]
        assert (BASE / local).exists(), target
    files = included_files()
    for path in files:
        if path.suffix == '.py':
            ast.parse(path.read_text())
    bindings = [{'path': str(path.relative_to(BASE)), 'bytes': path.stat().st_size,
                 'sha256': sha(path)} for path in files]
    now = datetime.now(timezone.utc)
    out = {
        'schema_version': 1, 'status': status['status'], 'sealed': True,
        'day2_all_goals_complete': False,
        'sealed_at_utc': now.isoformat(),
        'sealed_at_beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Solver diagnosis milestone, not all Day 2 goals.',
        'report': 'REPORT_ZH.md', 'final_status': 'FINAL_STATUS.json',
        'prior_day1_manifest': {
            'path': '../day1/COMPLETE_MANIFEST.json',
            'sha256': sha(BASE.parent / 'day1/COMPLETE_MANIFEST.json'),
        },
        'cost': status['cost'],
        'fact_memory_pilot': status['fact_memory_pilot'],
        'weakness_memory': status['weakness_memory'],
        'new_generated_training_questions': 0, 'new_training_runs': 0,
        'remaining_day2': status['remaining_day2'],
        'reference_models': 'Original model paths and hashes bound in run manifests; no weights copied here.',
        'preserved_failures': status['preserved_failures'],
        'canonical_observations': status['canonical_observations'],
        'canonical_memory': status['canonical_memory'],
        'audit_files': audit_files,
        'binding_policy': {
            'path_base': 'this directory', 'algorithm': 'SHA256',
            'excluded': ['COMPLETE_MANIFEST.json', '**/__pycache__/**', '**/*.pyc'],
        },
        'file_count': len(bindings), 'total_bytes': sum(row['bytes'] for row in bindings),
        'files': bindings,
    }
    assert included_files() == files, 'Tree changed during binding'
    for row in bindings:
        path = BASE / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    with TARGET.open('x', encoding='utf-8') as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    sealed = read(TARGET)
    assert len(included_files()) == sealed['file_count']
    for row in sealed['files']:
        path = BASE / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    print(json.dumps({
        'status': sealed['status'], 'sealed': True, 'day2_all_goals_complete': False,
        'file_count': sealed['file_count'], 'total_bytes': sealed['total_bytes'],
        'manifest_sha256': sha(TARGET),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
