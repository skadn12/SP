"""Bind actual process observations and completed stages, including startup timing."""
from pathlib import Path
from datetime import datetime
import sys
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import read, write, bind, now
python = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
actions = [('freeze', 'FREEZE'), ('prepare-evaluation', 'PREPARE_EVALUATION'), ('live-trained', 'LIVE_TRAINED'), ('live-base', 'LIVE_BASE'), ('report', 'REPORT')]
observation = read(HOME / 'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json')
assert observation['shell_pythonpath_present'] is False
assert observation['live_processes'] and all(not p['pythonpath_present'] for p in observation['live_processes'])
commands = [p['command'] for p in observation['live_processes']]
assert any('live-base' in c and any(Path(x).name == 'batch.py' for x in c) for c in commands)
assert any('live' in c and any(Path(x).name == 'run_solver.py' for x in c) for c in commands)
parent_start = read(HOME / 'BASE_EVALUATION.STARTED.json')['time']
parent_complete = read(HOME / 'BASE_EVALUATION.COMPLETE.json')['time']
native_start = read(HOME / 'solver/base/STARTED.json')['started_at_utc']
observed = observation['observed_at_utc']
parse = datetime.fromisoformat
assert parse(parent_start) <= parse(observed) <= parse(parent_complete)
assert parse(parent_start) <= parse(native_start) <= parse(parent_complete)
before_native_record = parse(observed) < parse(native_start)
write(HOME / 'evidence/EVALUATION_RUNTIME_RECEIPT.json', {
    'created_at_utc': now(), 'working_directory': str(HOME), 'python': python, 'python_flags': ['-B'],
    'additional_environment': {}, 'removed_environment': [],
    'scope': 'Direct batch.py invocations; frozen common.py handles imports internally. No external PYTHONPATH was added or removed.',
    'commands': [{'action': action, 'command': python + ' -B batch.py ' + action,
                  'stdout_stderr_log': bind(HOME / 'evidence' / (name + '.log'))} for action, name in actions],
    'preparation': bind(HOME / 'EVALUATION_PREPARATION.COMPLETE.json'),
    'trained_complete': bind(HOME / 'TRAINED_EVALUATION.COMPLETE.json'),
    'base_complete': bind(HOME / 'BASE_EVALUATION.COMPLETE.json'),
    'report': bind(HOME / 'MIXED_EVALUATION_REPORT.json'),
    'environment_observation': bind(HOME / 'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json'),
    'environment_observation_limitation': observation['limitation'],
    'environment_observation_stage': {
        'stage': 'base native process startup before native STARTED record' if before_native_record else 'base native process live after native STARTED record',
        'parent_started_at_utc': parent_start, 'parent_completed_at_utc': parent_complete,
        'observed_at_utc': observed, 'native_started_at_utc': native_start,
        'seconds_after_parent_start': (parse(observed) - parse(parent_start)).total_seconds(),
        'seconds_before_native_start_record': (parse(native_start) - parse(observed)).total_seconds(),
        'basis': 'Real parent and native commands and environments were observed during the parent base evaluation interval; original observation retained.',
        'limitation': 'One base process observation only; no continuous environment observation or earlier trained process environment is reconstructed.'},
    'recoveries': [], 'solver_automatic_replays': 0, 'frozen_code_edits': [], 'training_performed': False})
