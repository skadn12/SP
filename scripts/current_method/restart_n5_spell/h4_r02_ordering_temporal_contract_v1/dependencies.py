"""Complete repository closure and immutable input verification."""
from pathlib import Path
import json
import sys

from ..core import atomic_json, sha256_file
from ..h4_r02_basic_miner_v3_off_paired_v1.dependencies import (
    record, static_closure, loaded_repository_files, environment_identity,
    forbid_network, repository_execution_guard, check_loaded_coverage,
)
from .constants import CODE, METHOD, VERSION, SCHEDULE, CONFIG


def collect_coverage(run, dry):
    observed = loaded_repository_files() | {Path(p) for p in dry['repository_modules']}
    traced, edges, dynamic = static_closure(set(CODE.glob('*.py')) | observed)
    required = {METHOD / p for p in (
        'backends.py', 'live_backends.py', 'simple_append_v1/live_runtime.py', 'ledger.py',
        'h4_qa_multiroot_k3_t08_probe_v1/generation.py',
        'h4_qa_multiroot_k3_t08_probe_v1/solver_adapter.py',
        'h4_qa_multiroot_k3_t08_probe_v1/validator.py',
        'h4_qa_multiroot_k3_t08_probe_v1/permutation.py',
        'h4_qa_mining_hybrid_generation_v1/candidate_contract.py',
        'h4_qa_mining_hybrid_generation_v1/contract_guided_generator.py',
        'h4_qa_mining_hybrid_generation_v1/free_exploration_generator.py')}
    checks = {'offline_scheduler_pass': dry['status'] == 'PASS',
              'observed_covered': observed <= traced, 'required_covered': required <= traced,
              'bytecode_disabled': sys.dont_write_bytecode}
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'repository_files': [record(p) for p in sorted(traced)],
              'observed_repository_modules': [record(p) for p in sorted(observed)],
              'static_import_edges': edges, 'nonliteral_dynamic_import_sites': dynamic,
              'method': 'Recursive AST import closure plus observed injected-client scheduler, '
                        'Guided/Free stage and Solver renderer. Real client constructors forbidden '
                        'during fixtures; transport implementation covered statically. Live profile '
                        'guard rejects unfrozen repository code before execution.',
              'environment': environment_identity()}
    atomic_json(Path(run) / 'audit/dependency_coverage_report.json', result)
    return result


def verify_frozen(run, config_path=None):
    from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import GuardrailStop
    run = Path(run)
    path = run / 'audit/frozen_manifest.json'
    if not path.is_file():
        raise GuardrailStop('PREPARATION_NOT_FROZEN')
    frozen = json.loads(path.read_text())
    if frozen.get('version') != VERSION or frozen.get('schedule') != SCHEDULE:
        raise GuardrailStop('FROZEN_SCOPE_DRIFT')
    if Path(config_path or CONFIG).resolve() != Path(frozen['config']['path']).resolve():
        raise GuardrailStop('CONFIG_PATH_DRIFT')
    for row in frozen['code'] + frozen['inputs'] + frozen['sources'] + [frozen['config'], frozen['preflight']]:
        p = Path(row['path'])
        if not p.is_file() or p.stat().st_size != row['size'] or sha256_file(p) != row['sha256']:
            raise GuardrailStop('FROZEN_IDENTITY_DRIFT:' + str(p))
    current = {str(p.resolve()) for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected = {r['path'] for r in frozen['code'] if Path(r['path']).is_relative_to(CODE)}
    if current != expected:
        raise GuardrailStop('NEW_CODE_INVENTORY_DRIFT')
    for folder in ('inputs', 'policy_snapshot', 'audit/service_evidence'):
        directory = run / folder
        current = {str(p.resolve()) for p in directory.rglob('*') if p.is_file()}
        expected = {r['path'] for r in frozen['inputs'] if Path(r['path']).is_relative_to(directory)}
        if current != expected:
            raise GuardrailStop('FROZEN_INPUT_INVENTORY_DRIFT:' + folder)
    check_loaded_coverage(r['path'] for r in frozen['code'])
    if environment_identity() != frozen['environment']:
        raise GuardrailStop('ENVIRONMENT_DRIFT')
    if json.loads((run / 'audit/preflight_report.json').read_text()).get('status') != 'PASS':
        raise GuardrailStop('PREFLIGHT_NOT_PASS')
    return frozen
