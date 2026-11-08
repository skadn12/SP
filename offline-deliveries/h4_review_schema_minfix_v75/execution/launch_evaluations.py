"""Verify the frozen review before launching the paired evaluation runner."""
from pathlib import Path
import datetime
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate


def main(home=ROOT):
    home = Path(home).resolve()
    execution = home / 'execution'
    review = evaluate.verify_review(home)
    evaluate.require(review['status'] == 'FROZEN', 'review_not_frozen')
    evaluate.no_started_solver(home)
    evaluate.require(not (execution / 'evaluation_sequence.LAUNCH.json').exists()
                     and not (execution / 'evaluation_sequence.log').exists(), 'evaluation_already_launched')
    runner = execution / 'run_evaluations.py'
    with (execution / 'evaluation_sequence.log').open('x') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(runner)], cwd=home,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'pid': child.pid, 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'runner': evaluate.bind(runner), 'review': evaluate.bind(home / 'FINAL_CONTENT_REVIEW.COMPLETE.json'),
              'usable_questions': review['usable_count'], 'calls_per_model': 18 * review['usable_count'],
              'automatic_retry': False, 'training_allowed': False}
    evaluate.write(execution / 'evaluation_sequence.LAUNCH.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    main()
