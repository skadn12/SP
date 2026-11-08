"""Run each paired stage only after the frozen review passes validation."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate

PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')
STAGES = ('prepare', 'live-trained', 'live-base', 'verify')


def main(home=ROOT):
    home = Path(home).resolve()
    evaluate.verify_review(home)
    evaluate.no_started_solver(home)
    execution = home / 'execution'
    evaluate.write(execution / 'evaluation_sequence.STARTED.json',
                   {'time': evaluate.now(), 'training_allowed': False, 'automatic_retry': False})
    for stage in STAGES:
        with (execution / ('evaluation_' + stage + '.log')).open('x') as log:
            result = subprocess.run([str(PYTHON), '-B', '-u', str(home / 'evaluate.py'), stage],
                                    cwd=home, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'stage': stage, 'returncode': result.returncode, 'time': evaluate.now()}), flush=True)
        if result.returncode:
            evaluate.write(execution / 'evaluation_sequence.ERROR.json',
                           {'stage': stage, 'returncode': result.returncode,
                            'time': evaluate.now(), 'automatic_retry': False})
            raise SystemExit(result.returncode)
    evaluate.write(execution / 'evaluation_sequence.COMPLETE.json',
                   {'time': evaluate.now(), 'stages': list(STAGES), 'training_allowed': False})


if __name__ == '__main__':
    main()
