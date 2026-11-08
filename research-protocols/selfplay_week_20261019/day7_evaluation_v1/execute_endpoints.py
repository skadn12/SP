"""Run reviewed fixed TEST endpoints sequentially; preserve logs and failures.

Only invoke after INDEPENDENT_PREFLIGHT.json is PASS. This launcher never
scores results or reads private answer values and never retries a started role.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROLES = ('S0', 'A1', 'B1')


def main():
    audit = HERE / 'INDEPENDENT_PREFLIGHT.json'
    review = json.loads(audit.read_text())
    if review.get('status') != 'PASS':
        raise RuntimeError('INDEPENDENT_PREFLIGHT_REQUIRED')
    launcher_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not any(Path(row['path']).resolve() == Path(__file__).resolve()
               and row['sha256'] == launcher_sha for row in review.get('audited_files', [])):
        raise RuntimeError('AUDIT_DOES_NOT_BIND_LAUNCHER_BYTES')
    if (HERE / 'ENDPOINT_LAUNCH.json').exists():
        raise RuntimeError('LAUNCH_ALREADY_RECORDED_NO_AUTOMATIC_RETRY')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(n.strip()) > 1000 for n in gpu.splitlines()):
        raise RuntimeError('GPU_BUSY_BEFORE_TEST')
    launch = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'roles': ROLES, 'sequential': True, 'automatic_retry': False,
              'python': sys.executable, 'audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest(),
              'launcher_sha256': launcher_sha,
              'gpu_memory_used_mib_before_start': gpu.splitlines(), 'private_gold_values_read': False}
    with (HERE / 'ENDPOINT_LAUNCH.json').open('x') as f:
        json.dump(launch, f, indent=2)
        f.write('\n')
    exits = {}
    for role in ROLES:
        if (HERE / 'models' / role / 'STARTED.json').exists():
            raise RuntimeError('ROLE_ALREADY_STARTED:' + role)
        print(json.dumps({'status': 'START_ROLE', 'role': role}), flush=True)
        with (HERE / (role + '_LIVE.log')).open('x') as log:
            proc = subprocess.run([sys.executable, '-u', str(HERE / 'runtime.py'), 'live', '--role', role,
                                   '--audit', str(audit)], stdout=log, stderr=subprocess.STDOUT)
        exits[role] = proc.returncode
        print(json.dumps({'status': 'ROLE_TERMINAL', 'role': role, 'exit_code': proc.returncode}), flush=True)
        if not (HERE / 'models' / role / 'RAW_RESPONSE_FREEZE.json').is_file():
            raise RuntimeError('ROLE_NOT_SEALED_STOP_NO_RETRY:' + role)
    with (HERE / 'ENDPOINT_TERMINALS.json').open('x') as f:
        json.dump({'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_codes': exits,
                   'all_roles_raw_sealed': True, 'automatic_retry': False}, f, indent=2)
        f.write('\n')
    if any(exits.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
