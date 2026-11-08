"""Record live base process identity and PYTHONPATH only; never reconstruct past runs."""
from pathlib import Path
import json, os, datetime
HOME = Path(__file__).resolve().parents[1]
observed = []
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        command = [c.decode() for c in (process / 'cmdline').read_bytes().split(b'\0') if c]
        cwd = (process / 'cwd').resolve()
        parent = 'live-base' in command and any(Path(c).name == 'batch.py' for c in command)
        native = 'live' in command and any(Path(c).name == 'run_solver.py' for c in command)
        if cwd != HOME or not (parent or native):
            continue
        entries = (process / 'environ').read_bytes().split(b'\0')
        values = [x.split(b'=', 1)[1].decode() for x in entries if x.startswith(b'PYTHONPATH=')]
        observed.append({'pid': int(process.name), 'command': command,
                         'process_role': 'batch_live_base' if parent else 'native_solver_live',
                         'pythonpath_present': bool(values), 'pythonpath_value': values[0] if values else None})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert any(p['process_role'] == 'batch_live_base' for p in observed), 'live base parent must actually be observed'
assert any(p['process_role'] == 'native_solver_live' for p in observed), 'native live Solver must actually be observed'
record = {'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'shell_pythonpath_present': 'PYTHONPATH' in os.environ, 'shell_pythonpath_value': os.environ.get('PYTHONPATH'),
          'live_processes': observed,
          'limitation': 'Observed during the live base parent stage; native process may still be starting. No continuous observation or earlier trained process environment is reconstructed. No other environment values retained.'}
with (HOME / 'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json').open('x') as stream:
    json.dump(record, stream, indent=2); stream.write('\n')
print({'processes_observed': len(observed), 'PYTHONPATH_present': any(p['pythonpath_present'] for p in observed)})
