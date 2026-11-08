"""One-shot sequential model execution after both QA and independent preflight.

Run only when explicitly starting the frozen evaluation; merely importing this
module never starts inference. Each subprocess exits and releases its GPU model
before the next one starts. No automatic retry or partial-run resumption.
"""
from pathlib import Path
import subprocess
import sys
import evaluate as ev


def main():
    ev.frozen(require_independent=True)
    status = ev.HERE / 'RUN_STARTED.json'
    ev.require(not status.exists(), 'ALREADY_STARTED_NO_AUTOMATIC_RERUN')
    ev.require(all(not (ev.HERE / 'run' / side / 'STARTED.json').exists() for side in ev.MODELS), 'MODEL_SIDE_ALREADY_STARTED')
    ev.save(status, {'started_at_utc': ev.now(), 'sides': ev.MODELS, 'automatic_retry': False})
    for side in ev.MODELS:
        with (ev.HERE / (side + '.log')).open('x') as output:
            subprocess.run([sys.executable, '-u', str(ev.HERE / 'evaluate.py'), 'live', '--side', side],
                           stdout=output, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ev.HERE / 'report.py')], check=True)
    ev.save(ev.HERE / 'RUN_COMPLETE.json', {'status': 'COMPLETE', 'finished_at_utc': ev.now(),
        'automatic_retry': False, 'summary': str(ev.HERE / 'reports/SUMMARY.json')})


if __name__ == '__main__':
    main()
