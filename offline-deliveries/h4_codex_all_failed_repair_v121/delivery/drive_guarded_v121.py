"""Parent-only dispatch of completed review chains; creates no semantic content."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def run(dispatch=False):
    output = []
    roles = {r['root_id']: r for r in read(B / 'ROLE_ASSIGNMENTS.json')['roots']}
    for row in read(B / 'MANIFEST.json')['roots']:
        if not row['prefix_reuse_approved']:
            continue
        root = row['root_id']
        unit = Path(row['unit_path'])
        folder = unit / 'roots' / unit.name
        executor_hash = hashlib.sha256(roles[root]['reviewer'].encode()).hexdigest()[:16]
        for action, phase, artifact, result in (
            ('compose', 'asset', 'ASSET_SOURCE_JUDGMENTS.json', 'PUBLIC_QUESTIONS.json'),
            ('finalize', 'public', 'PUBLIC_SOURCE_JUDGMENTS.json', 'FINAL_CROSSCHECK.json'),
        ):
            receipt = unit / 'evidence' / ('GUARDED_' + action.upper() + '_RECEIPT.json')
            review_receipt = unit / 'evidence/reads' / f'{phase}_root_{executor_hash}' / 'SAVE_RECEIPT.json'
            if receipt.exists():
                status = read(receipt)['status']
                if status != 'GUARDED_STAGE_COMPLETED':
                    output.append({'root_id': root, 'stage': action, 'status': status, 'receipt': str(receipt)})
                continue
            if (folder / result).exists() or not (folder / artifact).exists() or not review_receipt.exists():
                continue
            item = {'root_id': root, 'stage': action, 'reviewer': roles[root]['reviewer'], 'status': 'READY_FOR_PARENT_DISPATCH'}
            if dispatch:
                completed = subprocess.run([sys.executable, '-B', str(B / 'tools/run_guarded_questions_v2.py'),
                    action, '--batch', str(B), '--root', root], capture_output=True, text=True)
                value = json.loads(completed.stdout)
                with receipt.open('x') as stream:
                    json.dump(value, stream, ensure_ascii=False, indent=2)
                    stream.write('\n')
                item.update(status=value['status'], returncode=completed.returncode, receipt=str(receipt))
            output.append(item)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.dispatch), ensure_ascii=False))
