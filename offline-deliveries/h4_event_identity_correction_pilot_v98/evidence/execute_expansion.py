"""Sequentially prepare, capacity-check, execute and audit the frozen batch once."""
from pathlib import Path
import json
import subprocess
import sys

HOME = Path(__file__).resolve().parents[1]
PY = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
ENGINE = HOME / 'expansion_engine'
RUN = ENGINE / 'runs/four_roots_B_scene01'

def step(name, script, *args):
    path = HOME / 'evidence' / (name + '.log')
    with path.open('x') as stream:
        result = subprocess.run([PY, '-B', str(script), *map(str, args)],
                                cwd=HOME, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({'phase': name, 'returncode': result.returncode, 'log': str(path)}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

def main():
    plan = json.loads((HOME / 'PRECALL_PLAN.json').read_text())
    assert not RUN.exists(), 'no replay'
    step('EXPANSION_PREPARE', ENGINE / 'runner.py', 'prepare', '--run', RUN,
         '--scenes', '1', '--routes', 'B', '--roots', *plan['candidate_roots'])
    step('EXPANSION_VERIFY', ENGINE / 'runner.py', 'verify', '--run', RUN)
    step('EXPANSION_CAPACITY', ENGINE / 'capacity_probe.py', '--run', RUN)
    cap = json.loads((RUN / 'OFFLINE_CAPACITY_CHECKS.json').read_text())
    assert cap['status'] == 'PASS', 'baseline capacity estimate failed'
    step('ORDERING_PRE_EXPANSION_CAPACITY', HOME / 'evidence/ordering_pre_expansion_capacity.py')
    ordering = json.loads((HOME / 'evidence/ORDERING_PRE_EXPANSION_CAPACITY.json').read_text())
    assert ordering['status'] == 'PASS_ESTIMATE_ONLY', 'ordering capacity estimate failed'
    step('EXPANSION_LIVE', ENGINE / 'runner.py', 'live', '--run', RUN)
    step('EXPANSION_FINAL_STRUCTURAL_AUDIT', ENGINE / 'audit_run.py', '--run', RUN)
    step('EXPANSION_BIND_ACTUAL', HOME / 'evidence/bind_expansion.py')
    step('EXPANSION_MATERIALIZE', HOME / 'evidence/materialize_expansion.py')
    step('EXPANSION_COLLECT', HOME / 'evidence/collect_expansion_results.py')
    step('TRANSPORT_FEEDBACK_AUDIT', HOME / 'evidence/audit_actual_expansion.py', '--run', RUN)

if __name__ == '__main__': main()
